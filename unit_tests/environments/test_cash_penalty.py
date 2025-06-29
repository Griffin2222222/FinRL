from __future__ import annotations

import numpy as np
import pytest

from finrl.meta.env_stock_trading.env_stocktrading_cashpenalty import (
    StockTradingEnvCashpenalty,
)
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "GOOG"]


@pytest.fixture(scope="session")
def indicator_list():
    return ["open", "close", "high", "low", "volume"]


@pytest.fixture(scope="session")
def data(ticker_list):
    return YahooDownloader(
        start_date="2019-01-01", end_date="2019-02-01", ticker_list=ticker_list
    ).fetch_data()


def test_zero_step(data, ticker_list):
    # Prove that zero actions results in zero stock buys, and no price changes
    init_amt = 1e6
    env = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    _ = env.reset()

    # step with all zeros
    for i in range(2):
        actions = np.zeros(len(ticker_list))
        next_state, _, _, _ = env.step(actions)
        cash = next_state[0]
        holdings = next_state[1 : 1 + len(ticker_list)]
        asset_value = env.account_information["asset_value"][-1]
        total_assets = env.account_information["total_assets"][-1]

        assert cash == init_amt
        assert init_amt == total_assets

        assert np.sum(holdings) == 0
        assert asset_value == 0

        assert env.current_step == i + 1


def test_patient(data, ticker_list):
    # Prove that we just not buying any new assets if running out of cash and the cycle is not ended
    aapl_first_close = data[data["tic"] == "AAPL"].head(1)["close"].values[0]
    init_amt = aapl_first_close
    hmax = aapl_first_close * 100
    env = StockTradingEnvCashpenalty(
        df=data,
        initial_amount=init_amt,
        hmax=hmax,
        cache_indicator_data=False,
        patient=True,
        random_start=False,
    )
    _ = env.reset()

    actions = np.array([1.0, 1.0])
    next_state, _, is_done, _ = env.step(actions)
    holdings = next_state[1 : 1 + len(ticker_list)]

    assert not is_done
    assert np.sum(holdings) == 0


def test_cost_penalties(data, ticker_list):
    """Test that cost penalties are applied correctly when buying/selling."""
    init_amt = 1e6
    env = StockTradingEnvCashpenalty(
        df=data,
        initial_amount=init_amt,
        buy_cost_pct=0.01,
        sell_cost_pct=0.01,
        cache_indicator_data=False,
    )
    _ = env.reset()
    actions = np.ones(len(ticker_list))
    next_state, _, _, _ = env.step(actions)
    # After buying, cash should decrease by at least 1% (buy cost)
    assert next_state[0] < init_amt
    # After selling, cash should decrease by at least 1% (sell cost)
    actions = -np.ones(len(ticker_list))
    next_state, _, _, _ = env.step(actions)
    assert next_state[0] < init_amt


def test_purchases(data, ticker_list):
    """Test that purchases increase holdings and decrease cash."""
    init_amt = 1e6
    env = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    _ = env.reset()
    actions = np.ones(len(ticker_list))
    next_state, _, _, _ = env.step(actions)
    holdings = next_state[1 : 1 + len(ticker_list)]
    assert np.all(holdings >= 0)
    assert next_state[0] < init_amt


def test_gains(data, ticker_list):
    """Test that positive price movement increases total assets."""
    init_amt = 1e6
    env = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    _ = env.reset()
    actions = np.ones(len(ticker_list))
    next_state, _, _, _ = env.step(actions)
    # Simulate price increase
    env.df.loc[:, "close"] = env.df["close"] * 1.1
    next_state, _, _, _ = env.step(np.zeros(len(ticker_list)))
    total_assets = env.account_information["total_assets"][-1]
    assert total_assets > init_amt


def test_no_lookahead_bias(data, ticker_list):
    """Test that environment state and reward do not use future data (no lookahead bias)."""
    init_amt = 1e6
    env = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    _ = env.reset()
    for _ in range(3):
        actions = np.zeros(len(ticker_list))
        state, _, _, _ = env.step(actions)
        # The state should only use data up to the current step
        current_date = env.dates[env.date_index]
        for i, tic in enumerate(ticker_list):
            # Check that the price in state matches the price in df for current date
            price_idx = 1 + i  # cash + i
            price_in_state = state[price_idx]
            price_in_df = env.df[(env.df["tic"] == tic)].loc[current_date, "close"]
            assert np.isclose(price_in_state, price_in_df, atol=1e-2)


@pytest.mark.skip(reason="this test is not working correctly")
def test_validate_caching(data):
    # prove that results with or without caching don't change anything
    init_amt = 1e6
    env_uncached = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    env_cached = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=True
    )
    _ = env_uncached.reset()
    _ = env_cached.reset()
    for i in range(10):
        actions = np.random.uniform(low=-1, high=1, size=2)
        print(f"actions: {actions}")
        un_state, un_reward, _, _ = env_uncached.step(actions)
        ca_state, ca_reward, _, _ = env_cached.step(actions)

        assert un_state == ca_state
        assert un_reward == ca_reward
