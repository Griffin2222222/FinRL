import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG
import logging
from finrl.feature_engineering.logging_explain import get_tensorboard_writer, log_metrics_tensorboard


def ensemble_predict(env, models, log_to_csv=None, explain_log=None, tensorboard_logdir=None):
    """
    Run an ensemble of RL agents and combine their actions by majority vote (discrete) or average (continuous).
    Logs actions, rewards, and states for monitoring and explainability.
    Optionally saves logs to CSV, explain_log (list of dicts for SHAP/surrogate analysis), and TensorBoard.
    """
    obs = env.reset()
    done = False
    actions_list = []
    rewards_list = []
    states_list = []
    explain_records = []
    total_reward = 0
    writer = get_tensorboard_writer(tensorboard_logdir) if tensorboard_logdir else None
    step = 0
    while not done:
        try:
            model_actions = [model.predict(obs, deterministic=True)[0] for model in models]
            # For continuous actions, use average; for discrete, use majority vote
            if isinstance(model_actions[0], np.ndarray):
                action = np.mean(model_actions, axis=0)
            else:
                action = max(set(model_actions), key=model_actions.count)
            next_obs, reward, done, *rest = env.step(action)
            actions_list.append(action)
            rewards_list.append(reward)
            states_list.append(obs)
            explain_records.append({'state': obs, 'action': action, 'reward': reward})
            obs = next_obs
            total_reward += reward
            # TensorBoard logging
            if writer:
                log_metrics_tensorboard(writer, step, {'reward': reward, 'total_reward': total_reward})
            step += 1
        except Exception as e:
            logging.error(f"Ensemble step error: {e}")
            break
    # Monitoring: compute PnL, Sharpe, drawdown
    pnl = np.sum(rewards_list)
    returns = np.array(rewards_list)
    sharpe = returns.mean() / (returns.std() + 1e-9) if len(returns) > 1 else 0
    cum_rewards = np.cumsum(returns)
    peak = np.maximum.accumulate(cum_rewards)
    drawdown = np.max(peak - cum_rewards) if len(cum_rewards) > 0 else 0
    logging.info(f"Ensemble run: PnL={pnl}, Sharpe={sharpe}, MaxDrawdown={drawdown}")
    if writer:
        log_metrics_tensorboard(writer, step, {'PnL': pnl, 'Sharpe': sharpe, 'MaxDrawdown': drawdown})
        writer.close()
    # Save logs if requested
    if log_to_csv:
        df = pd.DataFrame({'state': states_list, 'action': actions_list, 'reward': rewards_list})
        df.to_csv(log_to_csv, index=False)
    if explain_log is not None:
        explain_log.extend(explain_records)
    return actions_list, rewards_list, states_list, {'PnL': pnl, 'Sharpe': sharpe, 'MaxDrawdown': drawdown}

# Example usage (requires trained models and env):
# models = [PPO.load('ppo.zip'), A2C.load('a2c.zip'), DDPG.load('ddpg.zip')]
# actions, rewards, states, metrics = ensemble_predict(env, models, log_to_csv='ensemble_log.csv')
