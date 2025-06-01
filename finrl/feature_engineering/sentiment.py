import numpy as np

def add_sentiment_score(df):
    # Placeholder: Replace with real sentiment from FinGPT or Twitter API
    np.random.seed(42)
    df['sentiment_score'] = np.random.uniform(-1, 1, size=len(df))
    return df