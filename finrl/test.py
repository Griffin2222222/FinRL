from __future__ import annotations

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.feature_engineering.logging_explain import get_tensorboard_writer, log_metrics_tensorboard
from finrl.meta.market_event_manager import MarketEventManager, MarketEvent, MarketStructure
import os
import pickle
import datetime


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    max_failures=5,  # NEW: pass through
    num_episodes=1000,  # For overnight training
    checkpoint_every=10,  # Save model every N episodes
    checkpoint_dir="checkpoints",
    **kwargs,
):
    # import data processor
    from finrl.meta.data_processor import DataProcessor

    # fetch data
    dp = DataProcessor(data_source, **kwargs)
    event_manager = MarketEventManager()
    data = dp.download_data(ticker_list, start_date, end_date, time_interval, max_failures=max_failures)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    # Add multitimeframe features (fractal/EDA)
    from finrl.feature_engineering.pipeline import add_multitimeframe_features
    data = add_multitimeframe_features(data, timeframes=["1d", "1h", "5min"], indicators=["rsi", "macd", "ema"])

    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    # Build environment instance with correct arguments
    # env_config: price_array, tech_array, turbulence_array, if_train
    # Use the first date's data for initial state
    stock_dim = price_array.shape[1] if len(price_array.shape) > 1 else 1
    hmax = 100
    initial_amount = 10_000  # $10,000 starting balance for overnight run
    num_stock_shares = [0] * stock_dim
    buy_cost_pct = [0.001] * stock_dim
    sell_cost_pct = [0.001] * stock_dim
    reward_scaling = 1e-4
    state_space = price_array.shape[1] * 2 + len(technical_indicator_list) * price_array.shape[1] + 1
    action_space = stock_dim
    tech_indicator_list = technical_indicator_list
    turbulence_threshold = None
    risk_indicator_col = "turbulence"
    make_plots = False
    print_verbosity = 10

    env_instance = env(
        df=data,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=num_stock_shares,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=reward_scaling,
        state_space=state_space,
        action_space=action_space,
        tech_indicator_list=tech_indicator_list,
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col=risk_indicator_col,
        make_plots=make_plots,
        print_verbosity=print_verbosity,
        max_drawdown=0.01,  # 1% max drawdown
    )

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    writer = get_tensorboard_writer(log_dir=kwargs.get("tensorboard_log_dir", "runs/finrl_test"))

    os.makedirs(checkpoint_dir, exist_ok=True)
    training_log = []
    for episode in range(num_episodes):
        if drl_lib == "elegantrl":
            from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

            episode_total_assets = DRLAgent_erl.DRL_prediction(
                model_name=model_name,
                cwd=cwd,
                net_dimension=net_dimension,
                environment=env_instance,
            )
            # Log episode metrics to TensorBoard
            if isinstance(episode_total_assets, (list, tuple)):
                for step, asset in enumerate(episode_total_assets):
                    log_metrics_tensorboard(writer, step, {"total_assets": asset})
        elif drl_lib == "rllib":
            from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

            episode_total_assets = DRLAgent_rllib.DRL_prediction(
                model_name=model_name,
                env=env,
                price_array=price_array,
                tech_array=tech_array,
                turbulence_array=turbulence_array,
                agent_path=cwd,
            )
            if isinstance(episode_total_assets, (list, tuple)):
                for step, asset in enumerate(episode_total_assets):
                    log_metrics_tensorboard(writer, step, {"total_assets": asset})
        elif drl_lib == "stable_baselines3":
            from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

            episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
                model_name=model_name, environment=env_instance, cwd=cwd
            )
            if isinstance(episode_total_assets, (list, tuple)):
                for step, asset in enumerate(episode_total_assets):
                    log_metrics_tensorboard(writer, step, {"total_assets": asset})
        else:
            writer.close()
            raise ValueError("DRL library input is NOT supported. Please check.")

        # Example: check for structure invalidation or stop-loss
        if hasattr(env_instance, 'structure_invalidated') and env_instance.structure_invalidated:
            event_manager.on_event(MarketEvent.STRUCTURE_INVALIDATION, info={"episode": episode, "new_structure": MarketStructure.UNKNOWN})
        if hasattr(env_instance, 'stop_loss_triggered') and env_instance.stop_loss_triggered:
            event_manager.on_event(MarketEvent.STOP_LOSS_TRIGGERED, info={"episode": episode})

        # After each episode, log results
        episode_result = {
            'episode': episode,
            'timestamp': str(datetime.datetime.now()),
            # Add more metrics as needed (PnL, drawdown, actions, etc.)
            'event_log': event_manager.get_event_log(),
        }
        training_log.append(episode_result)
        # Save checkpoint
        if episode % checkpoint_every == 0:
            model_path = os.path.join(checkpoint_dir, f"model_ep{episode}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(env_instance, f)  # or your RL model object
            log_path = os.path.join(checkpoint_dir, "training_log.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(training_log, f)
            print(f"[CHECKPOINT] Saved at episode {episode}")

    writer.close()
    # Save event log to CSV after all episodes
    event_manager.save_event_log_to_csv(filepath=os.path.join(checkpoint_dir, "event_log.csv"))
    return episode_total_assets


if __name__ == "__main__":
    from finrl.config import TEST_START_DATE, TEST_END_DATE, INDICATORS
    from finrl.config_tickers import DOW_30_TICKER
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    import time
    max_drawdown = 0.01
    initial_amount = 10_000
    best_return = -float('inf')
    best_episode = -1
    episode = 0
    while True:
        print(f"\n=== Starting episode {episode} ===")
        results = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1d",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=StockTradingEnv,
            model_name="ppo",
            if_vix=True,
            num_episodes=1,  # One episode per loop
            checkpoint_every=1,
            checkpoint_dir="checkpoints",
            max_drawdown=max_drawdown,
            initial_amount=initial_amount
        )
        # Evaluate performance
        # Assume results is a list of total assets per step
        if isinstance(results, (list, tuple)) and len(results) > 0:
            final_asset = results[-1]
            gain = (final_asset - initial_amount) / initial_amount
            print(f"Episode {episode} final asset: {final_asset:.2f} | Gain: {gain*100:.2f}%")
            if gain > best_return:
                best_return = gain
                best_episode = episode
                print(f"[IMPROVEMENT] New best gain: {best_return*100:.2f}% at episode {best_episode}")
            # Enforce max drawdown
            min_asset = min(results)
            drawdown = (min_asset - initial_amount) / initial_amount
            if drawdown < -max_drawdown:
                print(f"[STOP] Max drawdown exceeded: {drawdown*100:.2f}%. Halting training.")
                break
        else:
            print("[WARN] No results for this episode.")
        episode += 1
        time.sleep(1)  # Optional: avoid hammering APIs
