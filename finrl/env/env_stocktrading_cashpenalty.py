import numpy as np
import pandas as pd
import random
from copy import deepcopy
import gym
import time
from gym.utils import seeding
from gym import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common import logger
from finrl.env.accounting.ledger import Ledger

"""
flowchart

compute transactions
perform transactions
update scalars
compute reward 
repeat
"""


class StockTradingEnvCashpenalty(gym.Env):
    """
    A stock trading environment for OpenAI gym
    This environment penalizes the model for not maintaining a reserve of cash.
    This enables the model to manage cash reserves in addition to performing trading procedures.
    Reward at any step is given as follows
        r_i = (sum(cash, asset_value) - initial_cash - max(0, sum(cash, asset_value)*cash_penalty_proportion-cash))/(days_elapsed)
        This reward function takes into account a liquidity requirement, as well as long-term accrued rewards.
    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        buy_cost_pct (float): cost for buying shares
        sell_cost_pct (float): cost for selling shares
        hmax (int, array): maximum cash to be traded in each trade per asset. If an array is provided, then each index correspond to each asset
        discrete_actions (bool): option to choose whether perform dicretization on actions space or not
        shares_increment (int): multiples number of shares can be bought in each trade. Only applicable if discrete_actions=True
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        initial_amount: (int, float): Amount of cash initially available
        daily_information_columns (list(str)): Columns to use when building state space from the dataframe. It could be OHLC columns or any other variables such as technical indicators and turbulence index
        cash_penalty_proportion (int, float): Penalty to apply if the algorithm runs out of cash
        patient (bool): option to choose whether end the cycle when we're running out of cash or just don't buy anything until we got additional cash

    RL Inputs and Outputs
        action space: [<n_assets>,] in range {-1, 1}
        state space: {start_cash, [shares_i for in in assets], [[indicator_j for j in indicators] for i in assets]]}
    TODO:
        Organize functions
        Write README
        Document tests
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        long_term_tax_rate=0.15,
        short_term_tax_rate=0.35,
        tax_horizon_days=365,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        turbulence_threshold=None,
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.long_term_tax_rate = long_term_tax_rate
        self.short_term_tax_rate = short_term_tax_rate
        self.tax_horizon_days = tax_horizon_days
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        if self.cache_indicator_data:
            print("caching data")
            self.cached_data = [
                self.get_date_vector(i) for i, _ in enumerate(self.dates)
            ]
            print("data cached!")

    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    @property
    def current_date(self):
        return self.dates[self.date_index]

    @property
    def cash_on_hand(self):
        # amount of cash held at current timestep
        return self.ledger.cash

    @property
    def holdings(self):
        # Quantity of shares held at current timestep
        return self.ledger.holdings

    @property
    def turbulence(self):
        return self.get_date_vector(self.date_index, cols=["turbulence"])[0]

    @property
    def closings(self):
        return np.array(self.get_date_vector(self.date_index, cols=["close"]))

    def reset(self):
        self.seed()
        self.ledger = Ledger(
            assets=self.assets, tax_threshold_days=self.tax_horizon_days
        )
        self.sum_trades = 0
        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.reward_memory = []
        _ = self.ledger.log_transactions(
            self.current_date, [0 for _ in self.assets], self.closings
        )
        self.ledger.log_scalars(
            self.current_date,
            {
                "cash": self.initial_amount,
                "long_term_tax_paid": 0,
                "short_term_tax_paid": 0,
                "long_term_profits": 0,
                "short_term_profits": 0,
                "asset_value": self.initial_amount,
            },
        )
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )

        self.state_memory.append(init_state)
        return init_state

    def get_date_vector(self, date, cols=None):
        if (cols is None) and (self.cached_data is not None):
            return self.cached_data[date]
        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[date]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v

    def log_terminal(self):
        gl_pct = (self.ledger.total_value) / self.initial_amount
        logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
        logger.record(
            "environment/total_assets",
            int(self.ledger.total_value),
        )
        logger.record("environment/total_trades", self.sum_trades)
        logger.record(
            "environment/avg_daily_trades",
            self.sum_trades / (self.current_step),
        )
        logger.record(
            "environment/avg_daily_trades_per_asset",
            self.sum_trades / (self.current_step) / len(self.assets),
        )
        logger.record("environment/completed_steps", self.current_step)
        logger.record("environment/sum_rewards", np.sum(self.reward_memory))
        logger.record(
            "environment/cash_proportion",
            self.ledger.cash / self.ledger.total_value,
        )

    def return_terminal(self, reason="Last Date"):
        self.log_terminal()
        state = self.state_memory[-1]
        self.log_step(reason=reason)

        return state, reward, True, {}

    def log_step(self, reason):

        cash_pct = self.ledger.cash / self.ledger.total_value
        gl_pct = self.ledger.total_value / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.ledger.cash))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.ledger.total_value))}",
            f"{self.get_reward()*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def log_header(self):
        if self.printed_header is False:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "LAST_REWARD",
                    "GAINLOSS_PCT",
                    "CASH_PROPORTION",
                )
            )
            self.printed_header = True

    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            assets = self.ledger.total_value
            cash = self.ledger.cash
            cash_penalty = max(0, (assets * self.cash_penalty_proportion - cash))
            assets -= cash_penalty
            reward = (assets / self.initial_amount) - 1
            reward /= self.current_step
            return reward

    def get_transactions(self, actions):
        """
        This function takes in a raw 'action' from the model and makes it into realistic transactions
        This function includes logic for discretizing
        It also includes turbulence logic.
        """
        # record actions of the model
        self.actions_memory.append(actions)

        # multiply actions by the hmax value
        actions = actions * self.hmax

        # Do nothing for shares with zero value
        actions = np.where(self.closings > 0, actions, 0)

        # discretize optionally
        if self.discrete_actions:
            actions = actions // self.closings
            actions = actions.astype(int)
            actions = np.where(
                actions >= 0,
                (actions // self.shares_increment) * self.shares_increment,
                ((actions + self.shares_increment) // self.shares_increment)
                * self.shares_increment,
            )
        else:
            actions = actions / self.closings

        # can't sell more than we have
        actions = np.maximum(actions, -np.array(self.holdings))

        # deal with turbulence
        if self.turbulence_threshold is not None:
            # if turbulence goes over threshold, just clear out all positions
            if self.turbulence >= self.turbulence_threshold:
                actions = -(np.array(self.holdings))
                self.log_step(reason="TURBULENCE")

        self.sum_trades += np.sum(np.abs(actions))
        return actions

    def step(self, actions):
        self.date_index += 1
        # let's just log what we're doing in terms of max actions at each step.
        self.log_header()
        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())
        else:
            """
            Now, let's get down to business at hand.
            """
            transactions = self.get_transactions(actions)
            # last, let's deal with taxes.
            begin_cash = self.ledger.cash
            tax_implications = self.ledger.log_transactions(
                self.current_date, transactions, self.closings
            )
            # TODO: carry forward losses
            short_profits, long_profits = 0, 0
            for a, d in tax_implications["assets"].items():
                short_profits += d["short_profit"]
                long_profits += d["long_profit"]
            short_tax, long_tax = max(0, short_profits * self.short_term_tax_rate), max(
                0, long_profits * self.long_term_tax_rate
            )
            # TODO: record taxes
            taxes = short_tax + long_tax

            # compute our proceeds from sells, and add to cash
            spend = tax_implications["spend"]
            proceeds = tax_implications["proceeds"]
            costs = 0
            costs += proceeds * self.sell_cost_pct
            costs += spend * self.buy_cost_pct

            # if we run out of cash...
            if (begin_cash + proceeds - spend - costs - taxes) < 0:
                # ... end the cycle and penalize
                return self.return_terminal(
                    reason="CASH SHORTAGE", reward=self.get_reward()
                )
            self.transaction_memory.append(
                transactions
            )  # capture what the model's could do
            # update our holdings
            coh = begin_cash + proceeds - spend - costs - taxes
            asset_value = np.dot(self.holdings, self.closings)

            # let's log scalars on our ledger
            self.ledger.log_scalars(
                self.current_date,
                {
                    "cash": coh,
                    "long_term_tax_paid": long_tax,
                    "short_term_tax_paid": short_tax,
                    "transaction_costs": costs,
                    "long_term_profits": long_profits,
                    "short_term_profits": short_profits,
                    "asset_value": asset_value,
                },
            )
            # compute reward once we've computed the value of things!
            reward = self.get_reward()
            self.reward_memory.append(reward)
            state = [coh] + list(self.holdings) + self.get_date_vector(self.date_index)
            self.state_memory.append(state)
            return state, reward, False, {}

    def get_sb_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        if self.current_step == 0:
            return None
        else:
            self.account_information["date"] = self.dates[
                -len(self.account_information["cash"]) :
            ]
            return pd.DataFrame(self.account_information)

    def save_action_memory(self):
        if self.current_step == 0:
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[-len(self.account_information["cash"]) :],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,
                }
            )
