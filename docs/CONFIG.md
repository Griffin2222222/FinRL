# FinRL Configuration Guide

## API Keys
- Set in `.env` file: `NASDAQ_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`

## Environment Parameters
- `max_drawdown`: Maximum allowed drawdown before halting or reducing risk
- `position_limit`: Maximum position size per asset
- `slippage_pct`: Simulated slippage percent
- `stop_loss_pct`: Stop-loss trigger percent
- `take_profit_pct`: Take-profit trigger percent

## Feature Engineering
- Technical indicators, macro, sentiment, earnings, VIX, etc.

## Data Sources
- Yahoo Finance, Finnhub, Nasdaq Data Link, Alpha Vantage

## Logging
- Logs written to `finrl.log`

## Testing
- Run `pytest` to execute all tests in `tests/`

## CI/CD
- GitHub Actions workflow in `.github/workflows/ci.yml`

---

For more details, see the main README and code docstrings.
