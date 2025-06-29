import pytest
import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

def test_no_lookahead_bias_np():
    # Create dummy data
    price_array = np.ones((10, 2)) * 100
    tech_array = np.ones((10, 2)) * 10
    turbulence_array = np.zeros(10)
    config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env = StockTradingEnv(config)
    state, _ = env.reset()
    for _ in range(5):
        actions = np.zeros(env.action_dim)
        state, _, done, *_ = env.step(actions)
        # The state should only use data up to the current step
        day = env.day
        for i in range(env.action_dim):
            price_in_state = state[3 + i] / (2**-6)  # undo scaling
            price_in_array = price_array[day, i]
            assert np.isclose(price_in_state, price_in_array, atol=1e-2)
        if done:
            break
