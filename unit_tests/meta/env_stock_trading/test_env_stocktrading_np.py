import pytest
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import numpy as np

@pytest.fixture
def env():
    config = {
        "price_array": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "tech_array": np.array([[5, 6], [7, 8]], dtype=np.float32),
        "turbulence_array": np.array([0, 1], dtype=np.float32),
        "if_train": True
    }
    return StockTradingEnv(config)

def test_reset(env):
    state = env.reset()
    assert env.day == 0
    assert isinstance(state, np.ndarray)
    assert env.total_asset > 0

def test_step(env):
    env.reset()
    actions = np.array([0.5, -0.5])
    state, reward, done, _ = env.step(actions)
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert not done

    # Test step function at the end of the episode
    env.day = env.max_step - 1  # Simulate last step
    _, _, done, _ = env.step(actions)
    assert done