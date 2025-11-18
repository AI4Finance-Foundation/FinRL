"""SPY Options Trading Environment with Greeks and Real-time Learning.

This environment supports options trading with:
- Real-time strike selection based on Greeks
- Call and Put option trading
- Price target prediction
- Continuous learning from trades
- Multi-timeframe analysis
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import warnings
warnings.filterwarnings('ignore')


class SPYOptionsEnv(gym.Env):
    """A SPY options trading environment for RL agents with Greeks.

    Features:
    - Options trading (calls and puts)
    - Greeks-based strike selection
    - Price target prediction
    - Trade learning and feedback
    - Multi-indicator state space

    Attributes
    ----------
    df : pd.DataFrame
        Preprocessed data with features and Greeks
    initial_amount : float
        Initial capital
    transaction_cost : float
        Transaction cost percentage
    state_space : int
        Dimension of observation space
    action_space_dim : int
        Dimension of action space
    max_options : int
        Maximum number of option contracts
    reward_scaling : float
        Scaling factor for rewards
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 10000,
        transaction_cost: float = 0.001,
        state_space: int = 50,
        max_options: int = 10,
        reward_scaling: float = 1e-4,
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        tech_indicator_list: List[str] = None,
        Greeks_list: List[str] = None,
    ):
        """Initialize the SPY options trading environment."""
        self.day = day
        self.df = df
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.max_options = max_options
        self.reward_scaling = reward_scaling
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity

        # Feature lists
        self.tech_indicator_list = tech_indicator_list or []
        self.greeks_list = greeks_list or [
            'call_delta', 'call_gamma', 'call_theta', 'call_vega',
            'put_delta', 'put_gamma', 'put_theta', 'put_vega'
        ]

        # Calculate state space size
        # State: [cash, options_positions, current_price, features, greeks]
        self.state_dim = (
            1 +  # cash
            2 +  # call_position, put_position
            1 +  # current_price
            len(self.tech_indicator_list) +  # technical indicators
            len(self.greeks_list)  # greeks
        )

        # Action space: [action_type, position_size]
        # action_type: -1 (sell/close), 0 (hold), 1 (buy call), 2 (buy put)
        # position_size: 0 to 1 (percentage of max_options)
        self.action_space = spaces.Box(
            low=np.array([-1, 0]),
            high=np.array([2, 1]),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.data = self.df.loc[self.day, :] if not self.df.empty else pd.Series()
        self.terminal = False
        self.state = self._initiate_state()

        # Trading memory
        self.cash = initial_amount
        self.call_position = 0  # Number of call contracts
        self.put_position = 0   # Number of put contracts
        self.call_entry_price = 0
        self.put_entry_price = 0
        self.call_strike = 0
        self.put_strike = 0

        # Performance tracking
        self.asset_memory = [initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.trades_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_value_memory = [initial_amount]

        # Trade learning
        self.trade_history = []
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0

        # Metrics
        self.cost = 0
        self.trades = 0
        self.episode = 0

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _initiate_state(self) -> np.ndarray:
        """Initialize the state vector.

        Returns
        -------
        np.ndarray
            Initial state vector
        """
        if self.data.empty:
            return np.zeros(self.state_dim)

        state = []

        # Cash
        state.append(self.initial_amount)

        # Positions
        state.extend([0, 0])  # call_position, put_position

        # Current price
        state.append(self.data.get('close', 0))

        # Technical indicators
        for indicator in self.tech_indicator_list:
            state.append(self.data.get(indicator, 0))

        # Greeks
        for greek in self.greeks_list:
            state.append(self.data.get(greek, 0))

        return np.array(state, dtype=np.float32)

    def _update_state(self):
        """Update state with current market data and positions."""
        if self.terminal or self.day >= len(self.df.index.unique()) - 1:
            return

        self.data = self.df.loc[self.day, :]

        state = []

        # Cash
        state.append(self.cash)

        # Positions
        state.extend([self.call_position, self.put_position])

        # Current price
        current_price = self.data.get('close', 0)
        state.append(current_price)

        # Technical indicators
        for indicator in self.tech_indicator_list:
            state.append(self.data.get(indicator, 0))

        # Greeks
        for greek in self.greeks_list:
            state.append(self.data.get(greek, 0))

        self.state = np.array(state, dtype=np.float32)

    def _get_date(self):
        """Get current date."""
        if self.data.empty:
            return None
        return self.data.get('date', self.data.get('timestamp', None))

    def _calculate_option_price(self, option_type: str) -> float:
        """Calculate current option price based on Greeks.

        Parameters
        ----------
        option_type : str
            'call' or 'put'

        Returns
        -------
        float
            Estimated option price
        """
        # In real trading, this would fetch actual option prices
        # For simulation, we'll estimate based on intrinsic value + time value

        current_price = self.data.get('close', 0)

        if option_type == 'call':
            strike = self.data.get('call_strike', current_price)
            intrinsic = max(0, current_price - strike)
            delta = abs(self.data.get('call_delta', 0.5))
            gamma = self.data.get('call_gamma', 0.01)
            vega = self.data.get('call_vega', 0.1)
            iv = self.data.get('call_iv', 0.3)
        else:
            strike = self.data.get('put_strike', current_price)
            intrinsic = max(0, strike - current_price)
            delta = abs(self.data.get('put_delta', 0.5))
            gamma = self.data.get('put_gamma', 0.01)
            vega = self.data.get('put_vega', 0.1)
            iv = self.data.get('put_iv', 0.3)

        # Estimate time value
        time_value = vega * iv + gamma * (current_price ** 2) * 0.01

        # Total option price
        option_price = intrinsic + time_value

        # Minimum price (options have some value)
        option_price = max(option_price, 0.01)

        return option_price

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment.

        Parameters
        ----------
        actions : np.ndarray
            Action from the agent [action_type, position_size]

        Returns
        -------
        observation : np.ndarray
            Current state
        reward : float
            Reward from the action
        terminated : bool
            Whether episode is terminated
        truncated : bool
            Whether episode is truncated
        info : dict
            Additional information
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # End of episode
            final_value = self._calculate_portfolio_value()
            total_return = (final_value - self.initial_amount) / self.initial_amount
            sharpe = self._calculate_sharpe_ratio()

            info = {
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'num_trades': self.trades,
                'win_rate': self.win_rate,
            }

            return self.state, 0, True, False, info

        # Parse actions
        action_type = actions[0]  # -1, 0, 1, 2
        position_size = np.clip(actions[1], 0, 1)  # 0 to 1

        # Execute action
        begin_value = self._calculate_portfolio_value()

        # Action type:
        # -1: Close positions
        # 0: Hold
        # 1: Buy call
        # 2: Buy put
        if action_type < -0.5:  # Close positions
            self._close_positions()
        elif action_type > 0.5 and action_type < 1.5:  # Buy call
            self._buy_call(position_size)
        elif action_type >= 1.5:  # Buy put
            self._buy_put(position_size)
        # else: hold (do nothing)

        # Move to next day
        self.day += 1
        self._update_state()

        # Calculate reward
        end_value = self._calculate_portfolio_value()
        reward = (end_value - begin_value) * self.reward_scaling

        # Additional reward shaping
        # Penalize excessive trading
        if self.trades > 0:
            reward -= 0.0001

        # Bonus for profitable trades
        if end_value > begin_value:
            reward += 0.001

        self.rewards_memory.append(reward)
        self.actions_memory.append(actions)
        self.portfolio_value_memory.append(end_value)
        self.date_memory.append(self._get_date())

        info = {
            'portfolio_value': end_value,
            'cash': self.cash,
            'call_position': self.call_position,
            'put_position': self.put_position,
        }

        return self.state, reward, False, False, info

    def _buy_call(self, position_size: float):
        """Buy call options.

        Parameters
        ----------
        position_size : float
            Fraction of max_options to buy (0 to 1)
        """
        if position_size <= 0:
            return

        num_contracts = int(position_size * self.max_options)
        if num_contracts <= 0:
            return

        option_price = self._calculate_option_price('call')
        cost = num_contracts * option_price * 100  # 100 shares per contract
        total_cost = cost * (1 + self.transaction_cost)

        if total_cost <= self.cash:
            self.cash -= total_cost
            self.call_position += num_contracts
            self.call_entry_price = option_price
            self.call_strike = self.data.get('call_strike', self.data.get('close', 0))
            self.cost += cost * self.transaction_cost
            self.trades += 1

            # Record trade
            self.trades_memory.append({
                'date': self._get_date(),
                'type': 'BUY_CALL',
                'contracts': num_contracts,
                'price': option_price,
                'strike': self.call_strike,
                'cost': total_cost,
            })

    def _buy_put(self, position_size: float):
        """Buy put options.

        Parameters
        ----------
        position_size : float
            Fraction of max_options to buy (0 to 1)
        """
        if position_size <= 0:
            return

        num_contracts = int(position_size * self.max_options)
        if num_contracts <= 0:
            return

        option_price = self._calculate_option_price('put')
        cost = num_contracts * option_price * 100  # 100 shares per contract
        total_cost = cost * (1 + self.transaction_cost)

        if total_cost <= self.cash:
            self.cash -= total_cost
            self.put_position += num_contracts
            self.put_entry_price = option_price
            self.put_strike = self.data.get('put_strike', self.data.get('close', 0))
            self.cost += cost * self.transaction_cost
            self.trades += 1

            # Record trade
            self.trades_memory.append({
                'date': self._get_date(),
                'type': 'BUY_PUT',
                'contracts': num_contracts,
                'price': option_price,
                'strike': self.put_strike,
                'cost': total_cost,
            })

    def _close_positions(self):
        """Close all open positions."""
        total_pnl = 0

        # Close calls
        if self.call_position > 0:
            current_price = self._calculate_option_price('call')
            proceeds = self.call_position * current_price * 100
            proceeds *= (1 - self.transaction_cost)
            self.cash += proceeds

            # Calculate PnL
            entry_value = self.call_position * self.call_entry_price * 100
            pnl = proceeds - entry_value
            total_pnl += pnl

            # Record trade
            self.trades_memory.append({
                'date': self._get_date(),
                'type': 'CLOSE_CALL',
                'contracts': self.call_position,
                'price': current_price,
                'pnl': pnl,
            })

            self.call_position = 0
            self.call_entry_price = 0
            self.trades += 1

        # Close puts
        if self.put_position > 0:
            current_price = self._calculate_option_price('put')
            proceeds = self.put_position * current_price * 100
            proceeds *= (1 - self.transaction_cost)
            self.cash += proceeds

            # Calculate PnL
            entry_value = self.put_position * self.put_entry_price * 100
            pnl = proceeds - entry_value
            total_pnl += pnl

            # Record trade
            self.trades_memory.append({
                'date': self._get_date(),
                'type': 'CLOSE_PUT',
                'contracts': self.put_position,
                'price': current_price,
                'pnl': pnl,
            })

            self.put_position = 0
            self.put_entry_price = 0
            self.trades += 1

        # Update trade learning
        if total_pnl != 0:
            self._update_trade_learning(total_pnl)

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value.

        Returns
        -------
        float
            Total portfolio value (cash + positions)
        """
        portfolio_value = self.cash

        # Add call position value
        if self.call_position > 0:
            call_price = self._calculate_option_price('call')
            portfolio_value += self.call_position * call_price * 100

        # Add put position value
        if self.put_position > 0:
            put_price = self._calculate_option_price('put')
            portfolio_value += self.put_position * put_price * 100

        return portfolio_value

    def _update_trade_learning(self, pnl: float):
        """Update learning metrics from completed trade.

        Parameters
        ----------
        pnl : float
            Profit/loss from the trade
        """
        self.trade_history.append(pnl)

        # Calculate win rate
        wins = sum(1 for p in self.trade_history if p > 0)
        total_trades = len(self.trade_history)
        self.win_rate = wins / total_trades if total_trades > 0 else 0

        # Calculate average win/loss
        winning_trades = [p for p in self.trade_history if p > 0]
        losing_trades = [p for p in self.trade_history if p < 0]

        self.avg_win = np.mean(winning_trades) if winning_trades else 0
        self.avg_loss = np.mean(losing_trades) if losing_trades else 0

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of the strategy.

        Returns
        -------
        float
            Sharpe ratio
        """
        if len(self.portfolio_value_memory) < 2:
            return 0

        returns = np.diff(self.portfolio_value_memory) / self.portfolio_value_memory[:-1]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0

        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        return sharpe

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.

        Returns
        -------
        observation : np.ndarray
            Initial state
        info : dict
            Additional information
        """
        super().reset(seed=seed)

        self.day = 0
        self.data = self.df.loc[self.day, :] if not self.df.empty else pd.Series()
        self.cash = self.initial_amount
        self.call_position = 0
        self.put_position = 0
        self.call_entry_price = 0
        self.put_entry_price = 0
        self.call_strike = 0
        self.put_strike = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False

        self.state = self._initiate_state()

        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.trades_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_value_memory = [self.initial_amount]

        return self.state, {}

    def render(self, mode='human'):
        """Render the environment."""
        return self.state

    def get_trade_stats(self) -> Dict:
        """Get trading statistics.

        Returns
        -------
        Dict
            Trading statistics
        """
        return {
            'total_trades': self.trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_pnl': self.cash + self._calculate_portfolio_value() - self.initial_amount,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
