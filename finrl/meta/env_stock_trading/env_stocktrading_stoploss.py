from __future__ import annotations

import random
import time
from copy import deepcopy

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

matplotlib.use("Agg")


class StockTradingEnvStopLoss(gym.Env):
    """
    A stock trading environment for OpenAI gym
    This environment penalizes the model if excedeed the stop-loss threshold, selling assets with under expectation %profit, and also
    for not maintaining a reserve of cash.
    This enables the model to do trading with high confidence and manage cash reserves in addition to performing trading procedures.

    Reward at any step is given as follows
        r_i = (sum(cash, asset_value) + additional_reward - total_penalty - initial_cash) / initial_cash / days_elapsed
        , where total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty
                cash_penalty = max(0, sum(cash, asset_value)*cash_penalty_proportion-cash)
                stop_loss_penalty = -1 * dot(holdings,negative_closing_diff_avg_buy)
                low_profit_penalty = -1 * dot(holdings,negative_profit_sell_diff_avg_buy)
                additional_reward = dot(holdings,positive_profit_sell_diff_avg_buy)

        This reward function takes into account a profit/loss ratio constraint, liquidity requirement, as well as long-term accrued rewards.
        This reward function also forces the model to trade only when it's really confident to do so.

    Parameters:
    state space: {start_cash, <owned_shares>, for s in stocks{<stock.values>}, }
        df (pandas.DataFrame): Dataframe containing data
        buy_cost_pct (float): cost for buying shares
        sell_cost_pct (float): cost for selling shares
        hmax (int): max number of share purchases allowed per asset
        discrete_actions (bool): option to choose whether perform dicretization on actions space or not
        shares_increment (int): multiples number of shares can be bought in each trade.
        stoploss_penalty (float): Maximum loss we can tolerate. Valid value range is between 0 and 1. If x is specified, then agent will force sell all holdings for a particular asset if current price < x * avg_buy_price
        profit_loss_ratio (int, float): Expected profit/loss ratio. Only applicable when stoploss_penalty < 1.
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
        initial_amount: (int, float): Amount of cash initially available
        daily_information_columns (list(str)): Columns to use when building state space from the dataframe. It could be OHLC columns or any other variables such as technical indicators and turbulence index
        cash_penalty_proportion (int, float): Penalty to apply if the algorithm runs out of cash
        patient (bool): option to choose whether end the cycle when we're running out of cash or just don't buy anything until we got additional cash
    action space: <share_dollar_purchases>
    TODO:
        add holdings to memory
        move transactions to after the clip step.
    tests:
        after reset, static strategy should result in same metrics
        given no change in prices, no change in asset values
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
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
        self.stoploss_penalty = stoploss_penalty
        self.min_profit_penalty = 1 + profit_loss_ratio * (1 - self.stoploss_penalty)
        self.turbulence_threshold = turbulence_threshold
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.turbulence = 0
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

    def reset(self):
        self.seed()
        self.sum_trades = 0
        self.actual_num_trades = 0
        self.closing_diff_avg_buy = np.zeros(len(self.assets))
        self.profit_sell_diff_avg_buy = np.zeros(len(self.assets))
        self.n_buys = np.zeros(len(self.assets))
        self.avg_buy_price = np.zeros(len(self.assets))
        if self.random_start:
            starting_point = random.choice(range(int(len(self.dates) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
        self.date_index = self.starting_point
        self.turbulence = 0
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }
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
            trunc_df = self.df.loc[[date]]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        # Add outputs to logger interface
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
        logger.record(
            "environment/total_assets",
            int(self.account_information["total_assets"][-1]),
        )
        reward_pct = self.account_information["total_assets"][-1] / self.initial_amount
        logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
        logger.record("environment/total_trades", self.sum_trades)
        logger.record(
            "environment/actual_num_trades",
            self.actual_num_trades,
        )
        logger.record(
            "environment/avg_daily_trades",
            self.sum_trades / (self.current_step),
        )
        logger.record(
            "environment/avg_daily_trades_per_asset",
            self.sum_trades / (self.current_step) / len(self.assets),
        )
        logger.record("environment/completed_steps", self.current_step)
        logger.record(
            "environment/sum_rewards", np.sum(self.account_information["reward"])
        )
        logger.record(
            "environment/cash_proportion",
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1],
        )
        return state, reward, True, {}

    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD_unsc",
                "GAINLOSS_PCT",
                "CASH_PROPORTION",
            )
        )
        self.printed_header = True

    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            total_assets = self.account_information["total_assets"][-1]
            cash = self.account_information["cash"][-1]
            holdings = self.state_memory[-1][1 : len(self.assets) + 1]
            neg_closing_diff_avg_buy = np.clip(self.closing_diff_avg_buy, -np.inf, 0)
            neg_profit_sell_diff_avg_buy = np.clip(
                self.profit_sell_diff_avg_buy, -np.inf, 0
            )
            pos_profit_sell_diff_avg_buy = np.clip(
                self.profit_sell_diff_avg_buy, 0, np.inf
            )

            cash_penalty = max(0, (total_assets * self.cash_penalty_proportion - cash))
            if self.current_step > 1:
                prev_holdings = self.state_memory[-2][1 : len(self.assets) + 1]
                stop_loss_penalty = -1 * np.dot(
                    np.array(prev_holdings), neg_closing_diff_avg_buy
                )
            else:
                stop_loss_penalty = 0
            low_profit_penalty = -1 * np.dot(
                np.array(holdings), neg_profit_sell_diff_avg_buy
            )
            total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty

            additional_reward = np.dot(np.array(holdings), pos_profit_sell_diff_avg_buy)

            reward = (
                (total_assets - total_penalty + additional_reward) / self.initial_amount
            ) - 1
            reward /= self.current_step

            return reward

    def step(self, actions):
        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))
        # print header only first time
        if self.printed_header is False:
            self.log_header()
        # print if it's time.
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # if we're at the end
        if self.date_index == len(self.dates) - 1:
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())
        else:
            # compute value of cash + assets
            begin_cash = self.state_memory[-1][0]
            holdings = self.state_memory[-1][1 : len(self.assets) + 1]
            assert min(holdings) >= 0
            closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))
            asset_value = np.dot(holdings, closings)
            # reward is (cash + assets) - (cash_last_step + assets_last_step)
            reward = self.get_reward()
            # log the values of cash, assets, and total assets
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(asset_value)
            self.account_information["total_assets"].append(begin_cash + asset_value)
            self.account_information["reward"].append(reward)

            # multiply action values by our scalar multiplier and save
            actions = actions * self.hmax
            self.actions_memory.append(
                actions * closings
            )  # capture what the model's trying to do
            # buy/sell only if the price is > 0 (no missing data in this particular date)
            actions = np.where(closings > 0, actions, 0)
            if self.turbulence_threshold is not None:
                # if turbulence goes over threshold, just clear out all positions
                if self.turbulence >= self.turbulence_threshold:
                    actions = -(np.array(holdings) * closings)
                    self.log_step(reason="TURBULENCE")
            # scale cash purchases to asset
            if self.discrete_actions:
                # convert into integer because we can't buy fraction of shares
                actions = np.where(closings > 0, actions // closings, 0)
                actions = actions.astype(int)
                # round down actions to the nearest multiplies of shares_increment
                actions = np.where(
                    actions >= 0,
                    (actions // self.shares_increment) * self.shares_increment,
                    ((actions + self.shares_increment) // self.shares_increment)
                    * self.shares_increment,
                )
            else:
                actions = np.where(closings > 0, actions / closings, 0)

            # clip actions so we can't sell more assets than we hold
            actions = np.maximum(actions, -np.array(holdings))

            self.closing_diff_avg_buy = closings - (
                self.stoploss_penalty * self.avg_buy_price
            )
            if begin_cash >= self.stoploss_penalty * self.initial_amount:
                # clear out position if stop-loss criteria is met
                actions = np.where(
                    self.closing_diff_avg_buy < 0, -np.array(holdings), actions
                )

                if any(np.clip(self.closing_diff_avg_buy, -np.inf, 0) < 0):
                    self.log_step(reason="STOP LOSS")

            # compute our proceeds from sells, and add to cash
            sells = -np.clip(actions, -np.inf, 0)
            proceeds = np.dot(sells, closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds
            # compute the cost of our buys
            buys = np.clip(actions, 0, np.inf)
            spend = np.dot(buys, closings)
            costs += spend * self.buy_cost_pct
            # if we run out of cash...
            if (spend + costs) > coh:
                if self.patient:
                    # ... just don't buy anything until we got additional cash
                    self.log_step(reason="CASH SHORTAGE")
                    actions = np.where(actions > 0, 0, actions)
                    spend = 0
                    costs = 0
                else:
                    # ... end the cycle and penalize
                    return self.return_terminal(
                        reason="CASH SHORTAGE", reward=self.get_reward()
                    )

            self.transaction_memory.append(actions)  # capture what the model's could do

            # get profitable sell actions
            sell_closing_price = np.where(
                sells > 0, closings, 0
            )  # get closing price of assets that we sold
            profit_sell = np.where(
                sell_closing_price - self.avg_buy_price > 0, 1, 0
            )  # mark the one which is profitable

            self.profit_sell_diff_avg_buy = np.where(
                profit_sell == 1,
                closings - (self.min_profit_penalty * self.avg_buy_price),
                0,
            )

            if any(np.clip(self.profit_sell_diff_avg_buy, -np.inf, 0) < 0):
                self.log_step(reason="LOW PROFIT")
            else:
                if any(np.clip(self.profit_sell_diff_avg_buy, 0, np.inf) > 0):
                    self.log_step(reason="HIGH PROFIT")

            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh

            # log actual total trades we did up to current step
            self.actual_num_trades = np.sum(np.abs(np.sign(actions)))

            # update our holdings
            coh = coh - spend - costs
            holdings_updated = holdings + actions

            # Update average buy price
            buys = np.sign(buys)
            self.n_buys += buys
            self.avg_buy_price = np.where(
                buys > 0,
                self.avg_buy_price + ((closings - self.avg_buy_price) / self.n_buys),
                self.avg_buy_price,
            )  # incremental average

            # set as zero when we don't have any holdings anymore
            self.n_buys = np.where(holdings_updated > 0, self.n_buys, 0)
            self.avg_buy_price = np.where(holdings_updated > 0, self.avg_buy_price, 0)

            self.date_index += 1
            if self.turbulence_threshold is not None:
                self.turbulence = self.get_date_vector(
                    self.date_index, cols=["turbulence"]
                )[0]

            # Update State
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
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
