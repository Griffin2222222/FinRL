import pandas as pd
from .indicators import add_talib_indicators
from .sentiment import add_sentiment_score
from .normalization import normalize_features
from .position_sizing import calc_dynamic_position
from .rolling_stats import add_rolling_stats

def feature_engineering_pipeline(df):
    df = add_talib_indicators(df)
    df = add_sentiment_score(df)
    norm_cols = ['close', 'volume', 'MOM', 'MACD_hist', 'MACD_hist_div', 'sentiment_score']
    df = normalize_features(df, [col for col in norm_cols if col in df.columns])
    df = calc_dynamic_position(df)
    df = add_rolling_stats(df)
    return df

if __name__ == "__main__":
    n = 100
    df = pd.DataFrame({
        'close': np.random.uniform(100, 200, n),
        'high': np.random.uniform(100, 210, n),
        'low': np.random.uniform(90, 195, n),
        'volume': np.random.uniform(1e5, 1e6, n),
    })
    df = feature_engineering_pipeline(df)
    print(df.tail())