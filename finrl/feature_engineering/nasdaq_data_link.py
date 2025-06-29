"""
Sample loader for US stock prices and macro data from Nasdaq Data Link (Quandl)
"""
import os
import quandl
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

# Load all API keys from .env if present
load_dotenv()

def set_quandl_api_key(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.environ.get("NASDAQ_API_KEY", "")
    if not api_key:
        raise ValueError("Nasdaq Data Link API key not set. Set NASDAQ_API_KEY env variable or pass as argument.")
    quandl.ApiConfig.api_key = api_key

def get_api_key(key_name: str, default: str = "") -> str:
    """Helper to get any API key with fallback"""
    return os.environ.get(key_name, default)

def get_us_stock_price(symbol: str, start_date: str = None, end_date: str = None, api_key: Optional[str] = None) -> pd.DataFrame:
    set_quandl_api_key(api_key)
    # WIKI/ is deprecated, use EOD/ for end-of-day US stocks (paid), or try 'NASDAQOMX/COMP' for Nasdaq Composite Index
    # Example: 'EOD/AAPL' for Apple
    dataset_code = f"EOD/{symbol.upper()}"
    try:
        df = quandl.get(dataset_code, start_date=start_date, end_date=end_date)
        df["tic"] = symbol.upper()
        return df.reset_index()
    except Exception as e:
        print(f"[ERROR] Could not fetch {symbol} from Nasdaq Data Link: {e}")
        return pd.DataFrame()

def get_macro_data(code: str, start_date: str = None, end_date: str = None, api_key: Optional[str] = None) -> pd.DataFrame:
    set_quandl_api_key(api_key)
    # Example macro code: 'FRED/GDP', 'FRED/UNRATE', 'FRED/CPALTT01USM657N'
    try:
        df = quandl.get(code, start_date=start_date, end_date=end_date)
        df["macro_code"] = code
        return df.reset_index()
    except Exception as e:
        print(f"[ERROR] Could not fetch macro data {code}: {e}")
        return pd.DataFrame()

# Example usage for Nasdaq Data Link
# set_quandl_api_key(get_api_key("NASDAQ_API_KEY"))
# Example usage for Finnhub
# finnhub_api_key = get_api_key("FINNHUB_API_KEY")
# Example usage for Alpha Vantage
# alpha_vantage_api_key = get_api_key("ALPHA_VANTAGE_API_KEY")
