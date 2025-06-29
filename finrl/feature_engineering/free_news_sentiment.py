import requests
import datetime
import yfinance as yf
from pytrends.request import TrendReq
import praw
import logging


# --- Free Sentiment/News Functions ---
def get_yahoo_headlines(ticker):
    # Scrape Yahoo Finance news headlines for a ticker
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    try:
        resp = requests.get(url, timeout=5)
        headlines = []
        if resp.status_code == 200:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "html.parser")
            for h in soup.find_all("h3"):
                headlines.append(h.text.strip())
        return headlines
    except Exception as e:
        logging.error(f"Yahoo headlines fetch failed for {ticker}: {e}")
        return []


def get_google_trends(keyword, timeframe="now 7-d"):  # last 7 days
    try:
        pytrends = TrendReq()
        pytrends.build_payload([keyword], timeframe=timeframe)
        data = pytrends.interest_over_time()
        if not data.empty:
            return data[keyword].tolist()
        return []
    except Exception as e:
        logging.error(f"Google Trends fetch failed for {keyword}: {e}")
        return []


def get_reddit_sentiment(ticker, client_id, client_secret, user_agent):
    try:
        reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )
        headlines = []
        for submission in reddit.subreddit("stocks").search(ticker, limit=10):
            headlines.append(submission.title)
        return headlines
    except Exception as e:
        logging.error(f"Reddit sentiment fetch failed for {ticker}: {e}")
        return []


# --- Example Usage ---
if __name__ == "__main__":
    print("Yahoo headlines for AAPL:", get_yahoo_headlines("AAPL"))
    print("Google Trends for AAPL:", get_google_trends("AAPL"))
    # For Reddit, you need to set up a free Reddit app and fill in credentials:
    # print(get_reddit_sentiment("AAPL", client_id, client_secret, user_agent))
