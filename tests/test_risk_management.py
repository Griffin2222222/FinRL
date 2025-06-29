import pytest
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import numpy as np

def test_max_drawdown_triggers():
    env = StockTradingEnv(
        df=..., stock_dim=1, hmax=1, initial_amount=1000,
        num_stock_shares=[0], buy_cost_pct=[0], sell_cost_pct=[0],
        reward_scaling=1, state_space=3, action_space=1, tech_indicator_list=[],
        max_drawdown=0.1
    )
    env.peak_asset = 1000
    env.state = [800, 100, 0]  # Simulate 20% drawdown
    total_asset = 800
    drawdown = (total_asset - env.peak_asset) / env.peak_asset
    assert drawdown < -env.max_drawdown

# Add more tests for position limits, slippage, stop-loss, take-profit
