import pandas as pd
import numpy as np
import logging
from .indicators import add_talib_indicators
from .sentiment import add_sentiment_score
from .normalization import normalize_features
from .position_sizing import calc_dynamic_position
from .rolling_stats import add_rolling_stats
from .fred_macro import add_fred_macro_features


try:
    from .earnings_finnhub import get_earnings_surprises, merge_earnings_to_price
except ImportError:
    get_earnings_surprises = None
    merge_earnings_to_price = None

try:
    from .nasdaq_data_link import get_us_stock_price, get_macro_data
except ImportError:
    get_us_stock_price = None
    get_macro_data = None


def feature_engineering_pipeline(df, fred_api_key=None, feature_selection=None):
    df = add_talib_indicators(df)
    df = add_sentiment_score(df)
    norm_cols = [
        "close",
        "volume",
        "MOM",
        "MACD_hist",
        "MACD_hist_div",
        "sentiment_score",
    ]
    df = normalize_features(df, [col for col in norm_cols if col in df.columns])
    # Normalization check
    for col in norm_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            if col_data.std() > 0:
                normed = (col_data - col_data.mean()) / col_data.std()
                logging.info(f"{col} mean: {normed.mean():.4f}, std: {normed.std():.4f}")
                assert abs(normed.mean()) < 1e-2, f"Feature {col} not normalized (mean)"
    # Causality check: ensure no future data is used
    if "date" in df.columns:
        assert df["date"].is_monotonic_increasing, "Date column not sorted (possible lookahead)"
    df = calc_dynamic_position(df)
    df = add_rolling_stats(df)
    df = add_advanced_features(df)
    if fred_api_key is not None:
        df = add_fred_macro_features(df, fred_api_key)
    # Feature selection: keep only selected features if provided
    if feature_selection is not None:
        keep_cols = [col for col in feature_selection if col in df.columns]
        df = df[keep_cols]
    df = add_earnings_features_all(df)
    return df


# Example: Add rolling mean, std, z-score, momentum, RSI, MACD, day-of-week, and month features
def add_advanced_features(df):
    if "close" in df.columns:
        df["close_rolling_mean_5"] = df["close"].rolling(5).mean()
        df["close_rolling_std_5"] = df["close"].rolling(5).std()
        df["close_zscore_5"] = (df["close"] - df["close_rolling_mean_5"]) / df[
            "close_rolling_std_5"
        ]
        df["momentum_5"] = df["close"] - df["close"].shift(5)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
    # Add more as needed
    # Add more TA-Lib indicators for free
    if "close" in df.columns and "high" in df.columns and "low" in df.columns:
        import talib

        df["ADX"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        df["CCI"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)
        df["OBV"] = (
            talib.OBV(df["close"], df["volume"]) if "volume" in df.columns else None
        )
    return df


# Example usage in your pipeline (pseudo-code):
def add_earnings_features(price_df, symbol, api_key=None):
    if get_earnings_surprises is None or merge_earnings_to_price is None:
        return price_df
    earnings_df = get_earnings_surprises(symbol, api_key=api_key)
    return merge_earnings_to_price(price_df, earnings_df)


def add_earnings_features_all(df, api_key=None):
    """
    For a DataFrame with a 'tic' column, fetch and merge Finnhub earnings data for each ticker.
    """
    if get_earnings_surprises is None or merge_earnings_to_price is None:
        return df
    tickers = df['tic'].unique()
    merged_frames = []
    for tic in tickers:
        price_df = df[df['tic'] == tic].copy()
        earnings_df = get_earnings_surprises(tic, api_key=api_key)
        merged = merge_earnings_to_price(price_df, earnings_df)
        merged_frames.append(merged)
    return pd.concat(merged_frames, ignore_index=True)


# Example usage:
def add_nasdaq_stock_data(symbol, start_date, end_date, api_key=None):
    if get_us_stock_price is None:
        return pd.DataFrame()
    return get_us_stock_price(symbol, start_date, end_date, api_key)

def add_nasdaq_macro_data(code, start_date, end_date, api_key=None):
    if get_macro_data is None:
        return pd.DataFrame()
    return get_macro_data(code, start_date, end_date, api_key)


def add_multitimeframe_features(df, timeframes=["1d", "1h", "5min"], indicators=["rsi", "macd", "ema"]):
    """
    Adds multitimeframe technical indicators to the dataframe.
    For each timeframe, resample and compute indicators, then merge as new columns.
    Ensures 'date' column exists and is datetime.
    """
    import pandas as pd
    import ta
    df_out = df.copy()
    # Ensure 'date' column exists and is datetime
    if 'date' not in df_out.columns:
        if df_out.index.name == 'date':
            df_out = df_out.reset_index()
        else:
            raise KeyError("Input DataFrame must have a 'date' column or datetime index named 'date'.")
    df_out['date'] = pd.to_datetime(df_out['date'])
    for tf in timeframes:
        if tf == "1d":
            df_tf = df_out.copy()
        else:
            rule = tf.replace("min", "T")
            df_tf = df_out.resample(rule, on="date").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna().reset_index()
        for ind in indicators:
            if ind == "rsi":
                df_tf[f"rsi_{tf}"] = ta.momentum.RSIIndicator(df_tf["close"]).rsi()
            elif ind == "macd":
                df_tf[f"macd_{tf}"] = ta.trend.MACD(df_tf["close"]).macd()
            elif ind == "ema":
                df_tf[f"ema_{tf}"] = ta.trend.EMAIndicator(df_tf["close"]).ema_indicator()
        # Merge back to main df
        for col in df_tf.columns:
            if any(ind in col for ind in indicators):
                df_out = pd.merge_asof(df_out.sort_values("date"), df_tf[["date", col]].sort_values("date"), on="date", direction="backward")
    return df_out


if __name__ == "__main__":
    n = 100
    df = pd.DataFrame(
        {
            "close": np.random.uniform(100, 200, n),
            "high": np.random.uniform(100, 210, n),
            "low": np.random.uniform(90, 195, n),
            "volume": np.random.uniform(1e5, 1e6, n),
        }
    )
    df = feature_engineering_pipeline(df)
    print(df.tail())
