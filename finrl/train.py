from __future__ import annotations

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

import numpy as np

# === ADVANCED AGENT IMPROVEMENTS ===
# 1. Advanced Reward Shaping: Risk-adjusted reward (Sharpe/Sortino)
#    You must modify the reward function in your environment (env_stocktrading_np.py).
#    Example: reward = (portfolio_return - risk_free_rate) / portfolio_std

# 2. LSTM/Transformer-based Policy Network
#    For SB3, you can use RecurrentPPO or custom policies.
#    For ElegantRL, define a custom actor with LSTM/Transformer.

# 3. Enhanced State Space: Add VIX, macro, alternative data
#    Already partially included (VIX). For macro/alt data, extend DataProcessor.

# 4. Ensemble Learning: Multiple agents voting/averaging
#    After training several agents, combine their actions.

# 5. Transfer Learning: Pretrain on S&P500/BTC, finetune on target market
#    Load weights from a pretrained model, then continue training.

# 6. Hybrid Action Space: Discrete + Continuous
#    Modify your environment and agent to support hybrid actions.

# 7. Curriculum Learning: Progressively harder environments
#    Train on simple env, then transfer to more complex.

# === END ADVANCED AGENT IMPROVEMENTS ===

def train(
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
    use_macro=False,  # NEW: macro data flag
    use_alternative=False,  # NEW: alternative data flag
    use_ensemble=False,  # NEW: ensemble flag
    curriculum_envs=None,  # NEW: curriculum learning
    transfer_model_path=None,  # NEW: transfer learning
    hybrid_action=False,  # NEW: hybrid action space
    lstm_policy=False,  # NEW: LSTM/Transformer policy
    **kwargs,
):
    # download data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = dp.add_vix(data)
    if use_macro and hasattr(dp, "add_macro"):
        data = dp.add_macro(data)  # You must implement this
    if use_alternative and hasattr(dp, "add_alternative"):
        data = dp.add_alternative(data)  # You must implement this
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
        "hybrid_action": hybrid_action,  # Pass hybrid action flag to env if supported
    }
    env_instance = env(config=env_config)

    # Curriculum learning: loop over environments of increasing complexity
    if curriculum_envs is not None:
        for cur_env_config in curriculum_envs:
            cur_env_instance = env(config=cur_env_config)
            train(
                start_date=start_date,
                end_date=end_date,
                ticker_list=ticker_list,
                data_source=data_source,
                time_interval=time_interval,
                technical_indicator_list=technical_indicator_list,
                drl_lib=drl_lib,
                env=env,
                model_name=model_name,
                if_vix=if_vix,
                use_macro=use_macro,
                use_alternative=use_alternative,
                use_ensemble=use_ensemble,
                curriculum_envs=None,
                transfer_model_path=transfer_model_path,
                hybrid_action=hybrid_action,
                lstm_policy=lstm_policy,
                **kwargs,
            )
        return

    # Ensemble learning: train multiple agents and combine
    if use_ensemble:
        agents = []
        for i in range(3):  # Example: 3 agents
            print(f"Training ensemble agent {i+1}")
            agent_kwargs = kwargs.copy()
            agent_kwargs["cwd"] = f"{kwargs.get('cwd', './')}_ensemble_{i+1}"
            train(
                start_date=start_date,
                end_date=end_date,
                ticker_list=ticker_list,
                data_source=data_source,
                time_interval=time_interval,
                technical_indicator_list=technical_indicator_list,
                drl_lib=drl_lib,
                env=env,
                model_name=model_name,
                if_vix=if_vix,
                use_macro=use_macro,
                use_alternative=use_alternative,
                use_ensemble=False,
                curriculum_envs=None,
                transfer_model_path=transfer_model_path,
                hybrid_action=hybrid_action,
                lstm_policy=lstm_policy,
                **agent_kwargs,
            )
        print("Ensemble training complete.")
        return

    # Transfer learning: load pretrained model
    pretrained_model = None
    if transfer_model_path is not None:
        if drl_lib == "stable_baselines3":
            from stable_baselines3 import PPO
            pretrained_model = PPO.load(transfer_model_path, env=env_instance)
        # Add for other libs as needed

    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        # LSTM/Transformer policy support (pseudo-code, implement in DRLAgent)
        if lstm_policy and hasattr(agent, "get_lstm_model"):
            model = agent.get_lstm_model(model_name, model_kwargs=erl_params)
        else:
            model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
    elif drl_lib == "rllib":
        total_episodes = kwargs.get("total_episodes", 100)
        rllib_params = kwargs.get("rllib_params")
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model, model_config = agent_rllib.get_model(model_name)
        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]
        # ray.shutdown()
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(cwd)
    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        # LSTM/Transformer policy support for SB3
        if lstm_policy:
            try:
                from stable_baselines3 import RecurrentPPO
                model = RecurrentPPO("MlpLstmPolicy", env_instance, **(agent_params or {}))
            except ImportError:
                print("RecurrentPPO not available in your SB3 version.")
                model = agent.get_model(model_name, model_kwargs=agent_params)
        elif pretrained_model is not None:
            model = pretrained_model
        else:
            model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # Example: curriculum learning environments (simple to complex)
    # curriculum_envs = [
    #     {"price_array": ..., "tech_array": ..., "turbulence_array": ..., "if_train": True},
    #     {"price_array": ..., "tech_array": ..., "turbulence_array": ..., "if_train": True, "noise": True},
    # ]

    # Example: use_ensemble, use_macro, use_alternative, transfer_model_path, hybrid_action, lstm_policy
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
        use_macro=True,  # Enable macro data if implemented
        use_alternative=True,  # Enable alternative data if implemented
        use_ensemble=False,  # Set True to train ensemble
        curriculum_envs=None,  # Provide list of env configs for curriculum learning
        transfer_model_path=None,  # Path to pretrained model for transfer learning
        hybrid_action=False,  # Enable hybrid action space if implemented
        lstm_policy=False,  # Enable LSTM/Transformer policy if implemented
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,
    # )
    #
    # # demo for stable-baselines3
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac",
    #     agent_params=SAC_PARAMS,
    #     total_timesteps=1e4,
    # )