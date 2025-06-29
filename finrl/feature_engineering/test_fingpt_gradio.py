import requests
import logging

logging.basicConfig(level=logging.INFO)


def query_fingpt(
    ticker="AAPL",
    date="2025-06-23",
    n_weeks=3,
    use_basics=False,
    api_url="http://localhost:8001/predict",
):
    """
    Query the FinGPT FastAPI endpoint for sentiment/news prediction.

    Args:
        ticker (str): Stock ticker symbol.
        date (str): Date in 'YYYY-MM-DD' format.
        n_weeks (int): Number of weeks for prediction context.
        use_basics (bool): Whether to use basic features.
        api_url (str): FinGPT API endpoint URL.

    Returns:
        tuple: (info, answer) from FinGPT API.
    """
    payload = {
        "ticker": ticker,
        "date": date,
        "n_weeks": n_weeks,
        "use_basics": use_basics,
    }
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logging.info(f"FinGPT API response: {result}")
        return result.get("info", None), result.get("answer", None)
    except requests.RequestException as e:
        logging.error(f"FinGPT API request failed: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None, None


def example_usage():
    """
    Example usage of the query_fingpt function.
    """
    info, answer = query_fingpt("AAPL", "2025-06-23", 3, False)
    print("Info:", info)
    print("Answer:", answer)


if __name__ == "__main__":
    example_usage()
