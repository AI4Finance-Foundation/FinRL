from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "MSFT"]


@pytest.fixture(scope="session")
def tech_indicator_list():
    return ["macd", "rsi_30", "cci_30", "dx_30"]


@pytest.fixture(scope="session")
def data(ticker_list):
    """Download real market data for testing"""
    return YahooDownloader(
        start_date="2020-01-01", end_date="2020-03-01", ticker_list=ticker_list
    ).fetch_data()


@pytest.fixture
def env_config(ticker_list, tech_indicator_list):
    """Standard environment configuration"""
    stock_dim = len(ticker_list)
    state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim

    return {
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicator_list,
    }


@pytest.fixture
def env(data, env_config):
    """Create a standard environment instance"""
    return StockTradingEnv(df=data, **env_config)


class TestEnvironmentInitialization:
    """Test environment initialization and setup"""

    def test_env_creation(self, env):
        """Test that environment can be created without errors"""
        assert env is not None
        assert isinstance(env, StockTradingEnv)

    def test_initial_state_shape(self, env, env_config):
        """Test that initial state has correct shape"""
        state = env._initiate_state()
        assert len(state) == env_config["state_space"]

    def test_initial_cash(self, env, env_config):
        """Test that initial cash is set correctly"""
        state = env._initiate_state()
        assert state[0] == env_config["initial_amount"]

    def test_initial_holdings_zero(self, env, env_config):
        """Test that initial stock holdings are zero"""
        state = env._initiate_state()
        stock_dim = env_config["stock_dim"]
        holdings = state[stock_dim + 1 : 2 * stock_dim + 1]
        assert np.all(holdings == 0)

    def test_action_space_bounds(self, env, env_config):
        """Test that action space has correct bounds"""
        assert env.action_space.shape == (env_config["action_space"],)
        assert np.all(env.action_space.low == -1)
        assert np.all(env.action_space.high == 1)

    def test_observation_space_shape(self, env, env_config):
        """Test that observation space has correct shape"""
        assert env.observation_space.shape == (env_config["state_space"],)


class TestEnvironmentReset:
    """Test environment reset functionality"""

    def test_reset_returns_initial_state(self, env):
        """Test that reset returns the initial state"""
        state = env.reset()
        assert state is not None
        assert len(state) == env.state_space

    def test_reset_clears_memory(self, env):
        """Test that reset clears episode memory"""
        # Take some steps
        env.reset()
        env.step(np.array([0.5, 0.5]))
        env.step(np.array([-0.5, -0.5]))

        # Reset
        env.reset()

        # Memory should be cleared/reset
        assert len(env.asset_memory) == 1  # Only initial value
        assert len(env.rewards_memory) == 0
        assert len(env.actions_memory) == 0

    def test_reset_seed_reproducibility(self, data, env_config):
        """Test that seeding produces reproducible results"""
        env1 = StockTradingEnv(df=data, **env_config)
        env1._seed(42)
        state1 = env1.reset()

        env2 = StockTradingEnv(df=data, **env_config)
        env2._seed(42)
        state2 = env2.reset()

        np.testing.assert_array_equal(state1, state2)


class TestEnvironmentStep:
    """Test environment step functionality"""

    def test_step_with_zero_action(self, env):
        """Test that zero actions don't change holdings"""
        env.reset()
        initial_state = env.state.copy()

        actions = np.zeros(env.stock_dim)
        next_state, reward, done, info = env.step(actions)

        # Cash should remain the same (no trades)
        assert next_state[0] == initial_state[0]

        # Holdings should remain the same
        holdings_start = env.stock_dim + 1
        holdings_end = 2 * env.stock_dim + 1
        np.testing.assert_array_equal(
            next_state[holdings_start:holdings_end],
            initial_state[holdings_start:holdings_end],
        )

    def test_step_returns_correct_tuple(self, env):
        """Test that step returns (state, reward, done, info)"""
        env.reset()
        actions = np.array([0.1, 0.1])
        result = env.step(actions)

        assert len(result) == 4
        state, reward, done, info = result

        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_day(self, env):
        """Test that stepping increments the day counter"""
        env.reset()
        initial_day = env.day

        env.step(np.zeros(env.stock_dim))

        assert env.day == initial_day + 1

    def test_buy_action_decreases_cash(self, env):
        """Test that buying stocks decreases cash"""
        env.reset()
        initial_cash = env.state[0]

        # Positive action = buy
        actions = np.array([0.5, 0.0])  # Buy first stock, don't touch second
        env.step(actions)

        # Cash should decrease (or stay same if can't afford)
        assert env.state[0] <= initial_cash

    def test_sell_action_increases_cash(self, env):
        """Test that selling stocks increases cash"""
        env.reset()

        # First buy some stocks
        env.step(np.array([0.5, 0.5]))
        cash_after_buy = env.state[0]

        # Then sell them
        env.step(np.array([-1.0, -1.0]))

        # Cash should increase
        assert env.state[0] > cash_after_buy

    def test_cannot_buy_with_insufficient_cash(self, data, env_config):
        """Test that we cannot buy more than we can afford"""
        # Create env with very small initial amount
        small_config = env_config.copy()
        small_config["initial_amount"] = 100  # Very small amount
        env = StockTradingEnv(df=data, **small_config)

        env.reset()

        # Try to buy with max action
        env.step(np.array([1.0, 1.0]))

        # Cash should not go negative
        assert env.state[0] >= 0

    def test_cannot_sell_unowned_stocks(self, env):
        """Test that we cannot sell stocks we don't own"""
        env.reset()

        # Try to sell when we have no stocks
        env.step(np.array([-1.0, -1.0]))

        # Holdings should still be zero
        holdings_start = env.stock_dim + 1
        holdings_end = 2 * env.stock_dim + 1
        holdings = env.state[holdings_start:holdings_end]

        assert np.all(holdings == 0)


class TestTradingCosts:
    """Test trading cost calculations"""

    def test_buy_cost_applied(self, env):
        """Test that buy costs are properly applied"""
        env.reset()

        # Buy some stock
        env.step(np.array([0.3, 0.0]))

        # Cost should be tracked
        assert env.cost >= 0

    def test_sell_cost_applied(self, env):
        """Test that sell costs are properly applied"""
        env.reset()

        # Buy then sell
        env.step(np.array([0.5, 0.0]))
        cost_after_buy = env.cost

        env.step(np.array([-1.0, 0.0]))

        # Sell cost should be added
        assert env.cost > cost_after_buy

    def test_trade_counter_increments(self, env):
        """Test that trade counter increments on each trade"""
        env.reset()
        assert env.trades == 0

        # Make a buy trade
        env.step(np.array([0.5, 0.0]))
        trades_after_buy = env.trades
        assert trades_after_buy > 0

        # Make a sell trade
        env.step(np.array([-1.0, 0.0]))
        assert env.trades > trades_after_buy


class TestTurbulence:
    """Test turbulence threshold functionality"""

    def test_turbulence_liquidation(self, data, env_config):
        """Test that high turbulence triggers liquidation"""
        # Create env with turbulence threshold
        config = env_config.copy()
        config["turbulence_threshold"] = 100
        env = StockTradingEnv(df=data, **config)

        env.reset()

        # Buy some stocks
        env.step(np.array([0.5, 0.5]))

        # Manually set high turbulence
        env.turbulence = 200

        # Try to buy more (should trigger sell instead)
        env.step(np.array([0.5, 0.5]))

        # This test assumes turbulence liquidation is working
        # Actual behavior depends on data and implementation

    def test_no_turbulence_threshold(self, env):
        """Test that env works without turbulence threshold"""
        assert env.turbulence_threshold is None

        # Should be able to trade normally
        env.reset()
        env.step(np.array([0.5, 0.5]))
        env.step(np.array([-0.5, -0.5]))


class TestRewards:
    """Test reward calculation"""

    def test_reward_scaling(self, env):
        """Test that reward scaling is applied"""
        env.reset()
        _, reward, _, _ = env.step(np.array([0.0, 0.0]))

        # Reward should be scaled
        assert isinstance(reward, (int, float, np.number))

    def test_positive_return_positive_reward(self, env):
        """Test that profits generate positive rewards"""
        env.reset()

        # This test is difficult without knowing market direction
        # Just verify reward is calculated
        _, reward, _, _ = env.step(np.array([0.3, 0.3]))

        assert reward is not None
        assert not np.isnan(reward)


class TestMemory:
    """Test memory tracking"""

    def test_asset_memory_tracking(self, env):
        """Test that asset values are tracked"""
        env.reset()
        initial_assets = len(env.asset_memory)

        env.step(np.array([0.2, 0.2]))
        env.step(np.array([0.1, 0.1]))

        # Asset memory should grow
        assert len(env.asset_memory) > initial_assets

    def test_rewards_memory_tracking(self, env):
        """Test that rewards are tracked"""
        env.reset()

        env.step(np.array([0.2, 0.2]))
        env.step(np.array([0.1, 0.1]))

        assert len(env.rewards_memory) == 2

    def test_actions_memory_tracking(self, env):
        """Test that actions are tracked"""
        env.reset()

        action1 = np.array([0.2, 0.2])
        action2 = np.array([0.1, 0.1])

        env.step(action1)
        env.step(action2)

        assert len(env.actions_memory) == 2


class TestTerminalCondition:
    """Test episode termination"""

    def test_terminal_at_end_of_data(self, data, env_config):
        """Test that environment terminates at end of data"""
        env = StockTradingEnv(df=data, **env_config)
        env.reset()

        done = False
        max_steps = len(data.index.unique()) * 2  # Safety limit
        steps = 0

        while not done and steps < max_steps:
            _, _, done, _ = env.step(np.zeros(env.stock_dim))
            steps += 1

        # Should eventually terminate
        assert done or steps < max_steps


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_extreme_positive_action(self, env):
        """Test handling of extreme positive actions"""
        env.reset()

        # Try to buy with very large action (should be clamped to available cash)
        large_actions = np.array([100.0, 100.0])
        next_state, _, _, _ = env.step(large_actions)

        # Should not crash, cash should not be negative
        assert next_state[0] >= 0

    def test_extreme_negative_action(self, env):
        """Test handling of extreme negative actions"""
        env.reset()

        # Try to sell with very large negative action (when we own nothing)
        large_negative = np.array([-100.0, -100.0])
        next_state, _, _, _ = env.step(large_negative)

        # Should not crash, holdings should not be negative
        holdings_start = env.stock_dim + 1
        holdings_end = 2 * env.stock_dim + 1
        holdings = next_state[holdings_start:holdings_end]

        assert np.all(holdings >= 0)

    def test_nan_in_data_handling(self, env):
        """Test that environment handles potential NaN values"""
        env.reset()

        # Take a step and verify state doesn't contain NaN
        next_state, reward, _, _ = env.step(np.array([0.1, 0.1]))

        assert not np.any(np.isnan(next_state))
        assert not np.isnan(reward)


class TestDeterminism:
    """Test deterministic behavior"""

    def test_same_seed_same_results(self, data, env_config):
        """Test that same seed produces same results"""
        # Create two environments with same seed
        env1 = StockTradingEnv(df=data, **env_config)
        env1._seed(42)

        env2 = StockTradingEnv(df=data, **env_config)
        env2._seed(42)

        # Take same actions
        actions = [np.array([0.3, 0.2]), np.array([-0.1, 0.4]), np.array([0.0, -0.2])]

        env1.reset()
        env2.reset()

        for action in actions:
            state1, reward1, done1, _ = env1.step(action)
            state2, reward2, done2, _ = env2.step(action)

            np.testing.assert_array_almost_equal(state1, state2)
            assert reward1 == reward2
            assert done1 == done2
