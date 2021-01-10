import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
from copy import deepcopy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    """
    state space: {start_cash, <owned_shares>, for s in stocks{<stock.values>}, }
    transaction_cost (float): cost for buying or selling shares
    hmax (int): max number of share purchases allowed per asset
    turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
    print_verbosity(int): When iterating (step), how often to print stats about state of env
    reward_scaling (float): Scaling value to multiply reward by at each step. 
    initial_amount: (int, float): Amount of cash initially available
    daily_information_columns (list(str)): Columns to use when building state space from the dataframe. 
    out_of_cash_penalty (int, float): Penalty to apply if the algorithm runs out of cash
    


    tests:
        after reset, static strategy should result in same metrics

        buy zero should result in no costs, no assets purchased
        given no change in prices, no change in asset values
    """

    def __init__(
        self,
        df,
        transaction_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        turbulence_threshold=None,
        print_verbosity=10,
        reward_scaling=1e-4,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        out_of_cash_penalty=100000,
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = df[self.stock_col].unique()
        self.dates = df[date_col_name].sort_values().unique()
        self.df = self.df.set_index(date_col_name)
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.out_of_cash_penalty = out_of_cash_penalty
        self.print_verbosity = print_verbosity
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.daily_information_cols = daily_information_cols
        self.close_index = self.daily_information_cols.index("close")
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.episode = -1  # initialize so we can call reset
        self._seed()

    def _seed(self):
        self.reward = 0
        self.cumulative_reward = 0
        self.date_index = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode += 1
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [self.initial_amount],
            "asset_value": [0],
            "total_assets": [self.initial_amount],
        }
        self.state_memory.append(
            np.array(
                [self.initial_amount]
                + [0] * len(self.assets)
                + self.get_date_vector(self.date_index)
            )
        )

    def reset(self):
        self._seed()
        return [0 for _ in range(self.state_space)]

    def get_date_vector(self, date, cols=None):
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

    def step(self, actions):
        # multiply action values by our scalar multiplier and save
        actions = actions * self.hmax
        self.actions_memory.append(actions)

        # define terminal function in scope so we can do something about the cycle being over
        def return_terminal(reason=None, penalty=0):

            state = self.state_memory[-1]
            reward = (
                self.account_information["total_assets"][-1]
                - self.account_information["total_assets"][0]
            )
            reward += penalty
            reward = reward*self.reward_scaling
            self.cumulative_reward+=reward
            if reason is None:
                reason = f"end of dates! end assets: {self.account_information['total_assets'][-1]:0.2f} at date {self.dates[self.date_index]}"
            reason += f" cumul reward : {self.cumulative_reward:0.2f}"
            print(reason)   
            return state, reward * self.reward_scaling, True, {}

        # print if it's time.
        if (self.date_index + 1) % self.print_verbosity == 0:
            print(f" date index: {self.date_index} of {len(self.dates)}")
            total_assets, cash_assets = (
                self.account_information["total_assets"][-1],
                self.account_information["cash"][-1],
            )
            print(
                f"assets: {total_assets:0.2f}, cash proportion: {(cash_assets/total_assets)*100:0.2f}%"
            )

        #if we're at the end
        if self.date_index == len(self.dates) - 1:
            return return_terminal()
        else:
            begin_cash = self.state_memory[-1][0]
            holdings = self.state_memory[-1][1 : len(self.assets) + 1]
            closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))

            # compute current value of holdings
            asset_value = np.dot(holdings, closings)

            # reward is (cash + assets) - (cash_last_step + assets_last_step)
            reward = (
                begin_cash + asset_value - self.account_information["total_assets"][-1]
            )

            # log the values of cash, assets, and total assets
            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(asset_value)
            self.account_information["total_assets"].append(begin_cash + asset_value)

            # clip actions so we can't sell more assets than we hold
            actions = np.maximum(actions, -np.array(holdings))

            # compute our proceeds from sales, and add to cash
            sells = -np.clip(actions, -np.inf, 0)
            proceeds = np.dot(sells, closings)
            costs = proceeds * self.transaction_cost_pct
            coh = begin_cash + proceeds

            # compute the cost of our buys
            buys = np.clip(actions, 0, np.inf)
            spend = np.dot(buys, closings)
            costs += spend * self.transaction_cost_pct

            # if we run out of cash, end the cycle and penalize
            if (spend + costs) > coh:
                return return_terminal(
                    f"ran out of cash at step: {self.date_index}, spend: {spend:0.2f}, costs: {costs:0.2f}, coh: {coh:0.2f}",
                    penalty=self.out_of_cash_penalty,
                )

            # verify we didn't do anything impossible here
            assert (spend + costs) <= coh

            # update our holdings
            coh = coh - spend - costs
            holdings_updated = holdings + actions
            self.date_index += 1
            state = (
                [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
            )
            self.state_memory.append(state)

            reward = reward * self.reward_scaling

            self.cumulative_reward+= reward

            return state, reward * self.reward_scaling, False, {}

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def get_multiproc_env(self, n = 10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method = 'fork')
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        self.account_information["date"] = self.dates[: self.date_index + 1]
        return pd.DataFrame(self.account_information)

    def save_action_memory(self):
        return pd.DataFrame(
            {"date": self.dates[: self.date_index], "actions": self.actions_memory}
        )

