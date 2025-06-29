import pandas as pd
from fredapi import Fred
import logging


def add_fred_macro_features(df, api_key, indicators=["VIXCLS", "FEDFUNDS", "UNRATE"]):
    """
    Adds FRED macroeconomic indicators to the dataframe by date.
    indicators: list of FRED series IDs (e.g., VIXCLS=VIX, FEDFUNDS=Fed Funds Rate, UNRATE=Unemployment)
    """
    fred = Fred(api_key=api_key)
    for ind in indicators:
        try:
            series = fred.get_series(ind)
            series = series.rename(ind)
            # Merge on date
            df = df.merge(series, left_on="date", right_index=True, how="left")
            logging.info(f"Fetched FRED indicator {ind}")
        except Exception as e:
            logging.error(f"FRED API fetch failed for {ind}: {e}")
            df[ind] = None
    return df


if __name__ == "__main__":
    # Example usage
    api_key = "YOUR_FRED_API_KEY"  # Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
    df = add_fred_macro_features(df, api_key)
    print(df.head())
