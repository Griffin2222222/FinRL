"""
Feature engineering: Finnhub earnings surprise and calendar integration
"""
import finnhub
import pandas as pd
import os
from typing import List, Optional

def get_finnhub_client(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        raise ValueError("Finnhub API key not set. Set FINNHUB_API_KEY env variable or pass as argument.")
    return finnhub.Client(api_key=api_key)

def get_earnings_surprises(symbol: str, limit: int = 20, api_key: Optional[str] = None) -> pd.DataFrame:
    client = get_finnhub_client(api_key)
    data = client.company_earnings(symbol, limit=limit)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

def get_earnings_calendar(symbol: str = None, from_date: str = None, to_date: str = None, api_key: Optional[str] = None) -> pd.DataFrame:
    client = get_finnhub_client(api_key)
    params = {}
    if symbol:
        params["symbol"] = symbol
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    data = client.earnings_calendar(**params)
    if not data or "earningsCalendar" not in data:
        return pd.DataFrame()
    return pd.DataFrame(data["earningsCalendar"])

def merge_earnings_to_price(price_df: pd.DataFrame, earnings_df: pd.DataFrame, on_col: str = "timestamp", symbol_col: str = "tic") -> pd.DataFrame:
    """
    Merge earnings data (surprise, etc.) into price dataframe by date and symbol.
    """
    if earnings_df.empty:
        for col in ["actual", "estimate", "surprise", "surprisePercent"]:
            price_df[col] = float('nan')
        return price_df
    # Convert period to timestamp if needed
    if "period" in earnings_df.columns:
        earnings_df[on_col] = pd.to_datetime(earnings_df["period"])
    if symbol_col in earnings_df.columns:
        earnings_df[symbol_col] = earnings_df[symbol_col].astype(str)
    # Merge on date and symbol
    merged = pd.merge(
        price_df,
        earnings_df[[on_col, symbol_col, "actual", "estimate", "surprise", "surprisePercent"]],
        how="left",
        left_on=[on_col, symbol_col],
        right_on=[on_col, symbol_col],
    )
    return merged
