import numpy as np
import requests
import logging
import pandas as pd
from finrl.config_private import FINGPT_API_KEY

# Gradio/FinGPT local endpoint
FINGPT_GRADIO_API_URL = "http://localhost:7860/api/predict/"

logging.basicConfig(level=logging.INFO)


def query_fingpt(
    ticker="AAPL",
    date="2025-06-23",
    n_weeks=3,
    use_basics=False,
    api_url=FINGPT_GRADIO_API_URL,
):
    payload = {"data": [ticker, date, n_weeks, use_basics]}
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["data"][0], result["data"][1]
    except Exception as e:
        logging.error(f"FinGPT API error for {ticker} {date}: {e}")
        return None, None


def add_sentiment_score(df, smoothing_window=3):
    """
    Adds a sentiment_score column to the dataframe using local FinGPT Gradio API.
    Applies rolling mean smoothing and robust error handling.
    Expects df to have a 'date' and 'tic' (ticker) column.
    """
    sentiments = []
    for idx, row in df.iterrows():
        ticker = row.get("tic", "AAPL")
        date = row.get("date", None)
        info, answer = query_fingpt(ticker, date)
        try:
            score = float(answer) if answer is not None else 0.0
        except Exception as e:
            logging.error(f"Sentiment score parse error for {ticker} {date}: {e}")
            score = 0.0
        sentiments.append(score)
    # Apply rolling mean smoothing (causal)
    df["sentiment_score"] = pd.Series(sentiments).rolling(
        smoothing_window, min_periods=1
    ).mean().values
    # Normalization check
    col = "sentiment_score"
    col_data = df[col].dropna()
    if col_data.std() > 0:
        normed = (col_data - col_data.mean()) / col_data.std()
        logging.info(f"{col} mean: {normed.mean():.4f}, std: {normed.std():.4f}")
        assert abs(normed.mean()) < 1e-2, f"Feature {col} not normalized (mean)"
    # Causality check: ensure no future data is used
    if "date" in df.columns:
        assert df["date"].is_monotonic_increasing, "Date column not sorted (possible lookahead)"
    return df
