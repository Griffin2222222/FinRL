import yfinance as yf
from finrl.config_tickers import DOW_30_TICKER

def check_yf_ticker_2023(ticker):
    try:
        data = yf.download(ticker, start="2023-01-01", end="2025-01-01", progress=False)
        return not data.empty
    except Exception:
        return False

def main():
    valid = []
    for t in DOW_30_TICKER:
        if check_yf_ticker_2023(t):
            print(f"OK: {t}")
            valid.append(t)
        else:
            print(f"FAIL: {t}")
    print("\nValid tickers for 2023-2025:")
    print(valid)

if __name__ == "__main__":
    main()
