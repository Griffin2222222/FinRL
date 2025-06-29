import yfinance as yf
from finrl.config_tickers import DOW_30_TICKER

def check_yf_ticker(ticker):
    try:
        data = yf.download(ticker, period="1y", progress=False)
        return not data.empty
    except Exception:
        return False

def main():
    valid = []
    for t in DOW_30_TICKER:
        if check_yf_ticker(t):
            print(f"OK: {t}")
            valid.append(t)
        else:
            print(f"FAIL: {t}")
    print("\nValid tickers:")
    print(valid)

if __name__ == "__main__":
    main()
