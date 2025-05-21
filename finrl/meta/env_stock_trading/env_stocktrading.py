from __future__ import annotations

from typing import List

import gymnasium as gym  # Use gymnasium for newer SB3 compatibility
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces  # Use gymnasium for newer SB3 compatibility
from gymnasium.utils import seeding  # Use gymnasium for newer SB3 compatibility
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):  # Inherit from gym.Env
    """A stock trading environment for OpenAI gym"""

    metadata = {
        "render_modes": ["human"],
        "render_fps": 1,
    }  # render_modes and render_fps for gymnasium

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],  # This is initial_stocks in some versions
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,  # This is usually self.stock_dim
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",  # Unused in this version if turbulence_threshold is None
        make_plots: bool = False,
        print_verbosity=10,
        day=0,  # This will track the index into unique_dates
        initial=True,  # This flag is used in _initiate_state
        previous_state=[],  # For re-initializing from a previous state
        model_name="",  # For saving results
        mode="",  # For saving results (train, trade, etc.)
        iteration="",  # For saving results
    ):
        self.day = day  # Index for unique_dates
        self.df = df.copy()  # Work with a copy to avoid modifying original df
        self.stock_dim = stock_dim
        self.hmax = hmax
        # self.num_stock_shares = num_stock_shares # Original attribute name from args
        self.initial_stocks = (
            num_stock_shares  # More common name for initial holdings, used later
        )
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space_dim = state_space  # Store the passed state_space dimension
        self.action_space_dim = action_space  # Store the passed action_space dimension
        self.tech_indicator_list = tech_indicator_list
        # action_space & observation_space
        # The passed action_space arg is the dimension, not the gym.Space object itself
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_dim,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space_dim,)
        )

        # --- FIX for self.data initialization ---
        self.unique_dates = self.df.date.unique()
        if not len(self.unique_dates) > 0:
            raise ValueError("DataFrame has no unique dates or is empty.")
        if self.day >= len(self.unique_dates):
            raise ValueError(
                f"Initial day {self.day} is out of bounds for unique dates {len(self.unique_dates)}."
            )

        current_date_for_init = self.unique_dates[self.day]
        self.data = self.df[self.df.date == current_date_for_init].reset_index(
            drop=True
        )
        if self.data.empty:
            raise ValueError(
                f"No data found for the initial date: {current_date_for_init}. Check DataFrame."
            )
        # --- End FIX ---

        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = (
            turbulence_threshold  # if not None, turbulence logic is active
        )
        self.risk_indicator_col = (
            risk_indicator_col  # Name of the turbulence column in df
        )

        self.initial = initial  # Flag used in _initiate_state
        self.previous_state = previous_state  # For non-initial starts
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # Initialize state
        self.state = (
            self._initiate_state()
        )  # This will now use the correctly shaped self.data

        # Initialize reward and other tracking variables
        self.reward = 0
        self.turbulence = 0  # This should be read from self.data if threshold is active
        self.cost = 0
        self.trades = 0
        self.episode = 0  # To be incremented in reset

        # Memorize all the total balance change
        # FIX: Use self.initial_stocks and ensure self.data.close is used correctly
        if (
            self.data.empty
            or not isinstance(self.data.close, pd.Series)
            or self.data.close.empty
        ):
            raise ValueError(
                "self.data or self.data.close is not correctly populated for asset_memory calculation."
            )

        initial_portfolio_value = np.sum(
            np.array(self.initial_stocks)
            * self.data.close.values  # self.data.close is now a Series
        )
        self.asset_memory = [self.initial_amount + initial_portfolio_value]

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]  # Store the actual date string

        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:  # Price of stock i
                sell_num_shares = min(
                    abs(action), self.state[index + self.stock_dim + 1]
                )  # Current shares of stock i
                sell_amount = (
                    self.state[index + 1]
                    * sell_num_shares
                    * (1 - self.sell_cost_pct[index])
                )
                self.state[0] += sell_amount  # Add to cash
                self.state[
                    index + self.stock_dim + 1
                ] -= sell_num_shares  # Reduce shares
                self.cost += (
                    self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                )
                self.trades += 1
            else:
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is None:
            sell_num_shares = _do_sell_normal()
        else:
            if self.turbulence < self.turbulence_threshold:
                sell_num_shares = _do_sell_normal()
            else:
                sell_num_shares = 0
        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:  # Price of stock i
                # Calculate available amount based on current cash and price of stock_i
                # Ensure price is not zero to avoid division by zero
                price_plus_cost = self.state[index + 1] * (1 + self.buy_cost_pct[index])
                if price_plus_cost > 0:
                    available_amount_shares = self.state[0] // price_plus_cost
                else:
                    available_amount_shares = 0

                buy_num_shares = min(available_amount_shares, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount  # Reduce cash
                self.state[
                    index + self.stock_dim + 1
                ] += buy_num_shares  # Increase shares
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
        return buy_num_shares

    def _make_plot(self):
        # Ensure results directory exists
        import os

        if not os.path.exists("results"):
            os.makedirs("results")
        plt.figure()  # Create a new figure to avoid overplotting
        plt.plot(self.asset_memory, "r")
        plt.title(f"Account Value Over Time (Episode {self.episode})")
        plt.xlabel("Time Step")
        plt.ylabel("Account Value")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):  # actions from agent, shape (self.action_space_dim,)
        self.terminal = self.day >= len(self.unique_dates) - 1  # Use unique_dates

        if self.terminal:
            if self.make_plots:
                self._make_plot()

            # Calculate final portfolio value
            final_portfolio_value = self.state[0] + np.sum(
                np.array(self.state[1 : 1 + self.stock_dim])
                * np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
            )

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            if len(self.date_memory) == len(df_total_value):
                df_total_value["date"] = self.date_memory
            else:
                df_total_value["date"] = pd.to_datetime(
                    self.unique_dates[: len(df_total_value)]
                )

            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            sharpe = 0.0
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print(f"Episode: {self.episode}")
                print(
                    f"day: {self.day}, episode: {self.episode}"
                )  # self.day is final day index
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {final_portfolio_value:0.2f}")
                print(
                    f"total_accumulated_rewards_memory: {np.sum(self.rewards_memory):0.2f}"
                )
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if sharpe != 0.0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            # Saving logic (ensure 'results' directory exists)
            import os

            if not os.path.exists("results"):
                os.makedirs("results")

            if (self.model_name != "") and (self.mode != ""):
                df_actions_mem = self.save_action_memory()  # Call instance method
                df_actions_mem.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                # df_rewards was not defined, let's use self.rewards_memory
                if self.rewards_memory:
                    df_rewards_mem = pd.DataFrame(
                        self.rewards_memory, columns=["reward"]
                    )
                    # Ensure date_memory for rewards is appropriate (usually one less than asset_memory)
                    if (
                        len(self.date_memory) > 1
                        and len(df_rewards_mem) == len(self.date_memory) - 1
                    ):
                        df_rewards_mem["date"] = self.date_memory[
                            1:
                        ]  # Rewards correspond to t+1 state
                    elif len(df_rewards_mem) == len(self.unique_dates) - 1:
                        df_rewards_mem["date"] = self.unique_dates[
                            1 : len(df_rewards_mem) + 1
                        ]
                    df_rewards_mem.to_csv(
                        f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                        index=False,
                    )

            return self.state, self.reward, True, False, {}

        else:
            # Ensure actions is a numpy array
            actions = np.array(actions, dtype=np.float32)

            # Scale actions to hmax range
            actions = actions * self.hmax
            actions = actions.astype(int)

            # Update turbulence value for the current day based on self.data
            if (
                self.turbulence_threshold is not None
                and self.risk_indicator_col in self.data.columns
            ):
                if len(self.data.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col].iloc[0]
                elif not self.data.empty:
                    self.turbulence = self.data[self.risk_indicator_col].iloc[0]
                else:
                    self.turbulence = 0
            else:
                self.turbulence = 0

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    # Sell all if turbulence is high
                    actions = np.array([-self.hmax] * self.stock_dim)

            # Calculate portfolio value before actions
            begin_total_asset = self.state[0] + np.sum(
                np.array(self.state[1 : 1 + self.stock_dim])
                * np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
            )

            # Create a copy of actions to store the actual shares traded
            actual_actions = np.zeros_like(actions)

            # Sort actions to process sells first, then buys
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            # Process sell actions
            for index in sell_index:
                actual_actions[index] = self._sell_stock(index, actions[index]) * (-1)

            # Process buy actions
            for index in buy_index:
                actual_actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actual_actions)

            # Transition to next day
            self.day += 1
            current_date_for_step = self.unique_dates[self.day]
            self.data = self.df[self.df.date == current_date_for_step].reset_index(
                drop=True
            )
            if self.data.empty:
                raise ValueError(
                    f"No data found for date {current_date_for_step} in step. Day index {self.day}"
                )

            self.state = self._update_state()

            # Calculate new portfolio value
            end_total_asset = self.state[0] + np.sum(
                np.array(self.state[1 : 1 + self.stock_dim])
                * np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
            )

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = (end_total_asset - begin_total_asset) * self.reward_scaling
            self.rewards_memory.append(self.reward)
            self.state_memory.append(self.state)

            return self.state, self.reward, False, False, {}

    def reset(
        self,
        *,  # Gymnasium requires keyword-only arguments after this
        seed: int | None = None,  # Use seed from gymnasium.utils.seeding
        options: dict | None = None,  # options for gymnasium
    ):
        super().reset(
            seed=seed
        )  # Call super for gym.Env compatibility if needed for seeding

        self.day = 0  # Reset day to the start
        current_date_for_reset = self.unique_dates[self.day]
        self.data = self.df[self.df.date == current_date_for_reset].reset_index(
            drop=True
        )
        if self.data.empty:
            raise ValueError(
                f"No data found for initial date {current_date_for_reset} in reset."
            )

        self.state = self._initiate_state()  # Re-initialize state for the first day

        if self.initial:  # self.initial is a flag passed during __init__
            # Calculate initial portfolio value based on initial cash and initial stocks at current day's prices
            initial_portfolio_value = np.sum(
                np.array(self.initial_stocks)
                * self.data.close.values  # self.data is for day 0
            )
            self.asset_memory = [self.initial_amount + initial_portfolio_value]
        else:
            # This case is for starting from a previous_state, ensure previous_state is properly structured
            if not self.previous_state:
                raise ValueError("previous_state is empty but initial=False.")
            # Assuming previous_state[0] is cash, and shares are in previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
            # Prices for previous_state value calculation should ideally be from *that* previous day, not current day.
            # This logic might need refinement if previous_state refers to a state from a *different* data point.
            # For simplicity, if previous_state is just a prior step's state, its value was already calculated.
            # Let's assume previous_total_asset needs to be calculated based on prices in self.data (current day 0 prices)
            # and shares from previous_state. This is a bit ambiguous.
            # A clearer way is to store the portfolio value with previous_state.
            # For now, let's recalculate with current prices, similar to initial state.
            previous_shares = np.array(
                self.previous_state[1 + self.stock_dim : 1 + 2 * self.stock_dim]
            )
            previous_portfolio_value = np.sum(previous_shares * self.data.close.values)
            self.asset_memory = [self.previous_state[0] + previous_portfolio_value]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]  # Date of the first day

        self.episode += 1  # Increment episode count

        # Gymnasium reset returns (observation, info)
        return self.state, {}

    def render(self, mode="human"):  # mode is for gymnasium compatibility
        # This environment doesn't have a sophisticated visual rendering.
        # Returning the state is a common placeholder.
        # If 'human' mode, could print state info.
        if mode == "human":
            print(f"Day: {self.day}, Date: {self._get_date()}")
            print(f"State: {self.state}")
            print(
                f"Portfolio Value: {self.asset_memory[-1] if self.asset_memory else 'N/A'}"
            )
        return self.state  # Or a more visual representation if implemented

    def _initiate_state(self):
        # self.data is already set to the DataFrame for the current day (all tickers)
        if (
            self.data.empty
            or not isinstance(self.data.close, pd.Series)
            or len(self.data.close) != self.stock_dim
        ):
            raise ValueError(
                f"self.data.close is not a Series of expected length {self.stock_dim} for state initiation. "
                f"Current self.data.close: {self.data.close if not self.data.empty else 'empty df'}"
            )

        if self.initial:
            # For Initial State
            if self.stock_dim > 1:  # Multiple stocks
                state = (
                    [self.initial_amount]  # Cash
                    + self.data.close.values.tolist()  # Prices for all stocks on current day
                    + self.initial_stocks  # Initial shares owned for each stock
                    + sum(
                        (
                            self.data[
                                tech
                            ].values.tolist()  # Tech indicators for all stocks
                            for tech in self.tech_indicator_list
                        ),
                        [],  # Start with an empty list for sum() to concatenate to
                    )
                )
            else:  # Single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close.iloc[0]]  # Price of the single stock
                    + self.initial_stocks  # Should be a list of 1 element
                    + sum(
                        (
                            [self.data[tech].iloc[0]]
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
        else:
            # Using Previous State to initialize current state (e.g., for continuous evaluation)
            # previous_state: [cash, p0_prev, p1_prev, ..., s0_prev, s1_prev, ..., t00_prev, t01_prev, ...]
            # New state: [prev_cash, p0_curr, p1_curr, ..., s0_prev, s1_prev, ..., t00_curr, t01_curr, ...]
            if (
                not self.previous_state
                or len(self.previous_state) != self.state_space_dim
            ):
                raise ValueError(
                    "previous_state is invalid or has incorrect dimension for re-initialization."
                )

            if self.stock_dim > 1:
                state = (
                    [self.previous_state[0]]  # Previous cash
                    + self.data.close.values.tolist()  # Current prices
                    + self.previous_state[
                        1 + self.stock_dim : 1 + 2 * self.stock_dim
                    ]  # Previous shares
                    + sum(
                        (
                            self.data[tech].values.tolist()  # Current tech indicators
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:  # Single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close.iloc[0]]
                    + self.previous_state[1 + self.stock_dim : 1 + 2 * self.stock_dim]
                    + sum(
                        (
                            [self.data[tech].iloc[0]]
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )

        # Ensure state has the expected dimension
        if len(state) != self.state_space_dim:
            raise ValueError(
                f"Constructed state length {len(state)} does not match expected state_space dimension {self.state_space_dim}."
            )
        return np.array(state, dtype=np.float32)

    def _update_state(self):
        # self.data is already updated to the next day's data (all tickers)
        if (
            self.data.empty
            or not isinstance(self.data.close, pd.Series)
            or len(self.data.close) != self.stock_dim
        ):
            raise ValueError(
                f"self.data.close is not a Series of expected length {self.stock_dim} for state update."
            )

        if self.stock_dim > 1:
            state = (
                [self.state[0]]  # Current cash
                + self.data.close.values.tolist()  # New prices for all stocks
                + list(
                    self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim]
                )  # Current shares owned
                + sum(
                    (
                        self.data[
                            tech
                        ].values.tolist()  # New tech indicators for all stocks
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )
        else:  # Single stock
            state = (
                [self.state[0]]
                + [self.data.close.iloc[0]]  # New price
                + list(
                    self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim]
                )  # Current shares
                + sum(
                    ([self.data[tech].iloc[0]] for tech in self.tech_indicator_list), []
                )  # New tech indicators
            )

        if len(state) != self.state_space_dim:
            raise ValueError(
                f"Updated state length {len(state)} does not match expected state_space dimension {self.state_space_dim}."
            )
        return np.array(state, dtype=np.float32)

    def _get_date(self):
        # self.data is a DataFrame for the current day, for all tickers.
        # All rows in self.data should have the same date.
        if not self.data.empty:
            return self.data.date.iloc[
                0
            ]  # Get date from the first row (should be same for all)
        elif (
            self.unique_dates
        ):  # Fallback if self.data somehow became empty but we have dates
            return self.unique_dates[
                self.day if self.day < len(self.unique_dates) else -1
            ]
        return "N/A"  # Should not happen

    def save_state_memory(
        self,
    ):  # This method seems specific to a fixed number of assets (Bitcoin, Gold)
        # It needs to be generalized or adapted if self.stock_dim can vary.
        # For now, assuming it's a placeholder or for a specific use case.
        # If used generally, column names for prices/nums need to be dynamic.
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[
                :-1
            ]  # Exclude current day's date as state is for t
            if not date_list:
                return pd.DataFrame()  # No memory to save

            df_date = pd.DataFrame(date_list, columns=["date"])

            state_list = self.state_memory
            if not state_list:
                return pd.DataFrame()

            # Dynamic column naming based on stock_dim and tech_indicator_list
            price_cols = [
                f"{self.data.tic.iloc[i]}_price" for i in range(self.stock_dim)
            ]
            num_cols = [f"{self.data.tic.iloc[i]}_num" for i in range(self.stock_dim)]
            # tech_indicator columns are complex to name here as they are flattened.
            # The state includes all tech indicators for all stocks sequentially.
            # For simplicity, let's just number them or use a generic prefix.
            num_tech_features_per_stock = len(self.tech_indicator_list)
            tech_cols = [
                f"tech_{j}_{i}"
                for j in range(self.stock_dim)
                for i in range(num_tech_features_per_stock)
            ]

            state_df_columns = ["cash"] + price_cols + num_cols + tech_cols
            # Check if state_list items have correct length
            if state_list and len(state_list[0]) != len(state_df_columns):
                print(
                    f"Warning: Mismatch in state memory columns. Expected {len(state_df_columns)}, got {len(state_list[0]) if state_list else 'N/A'}"
                )
                # Fallback to generic numbered columns if mismatch
                state_df_columns = [f"state_{i}" for i in range(len(state_list[0]))]

            df_states = pd.DataFrame(
                state_list, columns=state_df_columns[: len(state_list[0])]
            )  # Slice columns if mismatch
            df_states.index = df_date.date
        else:  # Single stock case
            date_list = self.date_memory[:-1]
            if not date_list:
                return pd.DataFrame()
            state_list = self.state_memory
            if not state_list:
                return pd.DataFrame()
            # Simpler naming for single stock
            # Columns might be: cash, price, num_shares, tech1, tech2, ...
            # state_df_columns = ["cash", "price", "num_shares"] + self.tech_indicator_list
            # df_states = pd.DataFrame(state_list, columns=state_df_columns)
            df_states = pd.DataFrame(
                {"date": date_list, "states": state_list}
            )  # Original simpler version
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        if not date_list or not asset_list:
            return pd.DataFrame()
        # Ensure lengths match, truncate if necessary (should not happen with correct logic)
        min_len = min(len(date_list), len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list[:min_len], "account_value": asset_list[:min_len]}
        )
        return df_account_value

    def save_action_memory(self):
        if not self.actions_memory:
            return pd.DataFrame()

        date_list = self.date_memory[
            :-1
        ]  # Actions correspond to state at t, leading to state t+1
        if not date_list:
            return pd.DataFrame()

        # Ensure date_list and actions_memory align
        min_len = min(len(date_list), len(self.actions_memory))
        date_list_aligned = date_list[:min_len]
        action_list_aligned = self.actions_memory[:min_len]

        if not action_list_aligned:
            return pd.DataFrame()

        if self.stock_dim > 1:
            df_date = pd.DataFrame(date_list_aligned, columns=["date"])
            df_actions = pd.DataFrame(action_list_aligned)
            # Get actual ticker names for columns
            # self.data might be for the *next* day when this is called at episode end.
            # Better to get tickers from self.df once.
            if not self.df.empty and "tic" in self.df.columns:
                tic_names = self.df.tic.unique()[
                    : self.stock_dim
                ]  # Get unique tickers up to stock_dim
                if len(tic_names) == df_actions.shape[1]:
                    df_actions.columns = tic_names
                else:  # Fallback if tic names don't match action dimensions
                    df_actions.columns = [
                        f"action_stock_{i}" for i in range(df_actions.shape[1])
                    ]
            else:  # Fallback
                df_actions.columns = [
                    f"action_stock_{i}" for i in range(df_actions.shape[1])
                ]

            df_actions.index = df_date.date
        else:  # Single stock
            df_actions = pd.DataFrame(
                {
                    "date": date_list_aligned,
                    "actions": [a[0] for a in action_list_aligned],
                }
            )  # Assuming actions is list of lists/arrays
        return df_actions

    def _seed(self, seed=None):  # Conform to gymnasium seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        # This is a utility method, often used for quickly wrapping the env for SB3
        # if it's not already wrapped (e.g., for some testing).
        # In the main script, DummyVecEnv is usually applied directly.
        e = DummyVecEnv([lambda: self])
        # obs = e.reset() # SB3 DummyVecEnv reset returns a tuple if using new gymnasium (obs, info)
        # For now, just return the env. The user can call reset on 'e'.
        return e, e.reset()[0]  # Return env and initial observation
