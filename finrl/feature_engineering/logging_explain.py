import pandas as pd
import shap
import xgboost as xgb
from torch.utils.tensorboard import SummaryWriter
import os


def log_trades(trades, filename="trades_log.csv"):
    pd.DataFrame(trades).to_csv(filename, index=False)


def shap_explain(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)


def get_tensorboard_writer(log_dir="runs/finrl_monitor"):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def log_metrics_tensorboard(writer, step, metrics: dict):
    """
    Log a dictionary of metrics to TensorBoard at a given step.
    """
    for key, value in metrics.items():
        writer.add_scalar(key, value, step)

# Example usage:
# trades = [{"date": ..., "action": ..., "reward": ...}, ...]
# log_trades(trades)
# model = xgb.XGBClassifier().fit(X_train, y_train)
# shap_explain(model, X_test)
# writer = get_tensorboard_writer()
# for step, metrics in enumerate(metrics_list):
#     log_metrics_tensorboard(writer, step, metrics)
# writer.close()
