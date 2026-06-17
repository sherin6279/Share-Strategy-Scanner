# NIFTY 500 Momentum & Leadership Screener

Enterprise-grade local desktop screener for NIFTY 500 stocks using Zerodha Kite Connect, DuckDB, and Streamlit.

## Features

- **7 institutional strategies** with backtest-ready `evaluate(symbol, candles, evaluation_index)` interface
- **Confluence engine** — stocks appearing in 2+ strategies
- **Local DuckDB storage** with UPSERT (no duplicate candles)
- **Rate-limited Kite API** refresh (350ms delay, 3 retries)
- **Vectorized indicators**: RSI, SMA, EMA, ATR, relative strength
- **Export** to CSV and Excel

## Requirements

- Python 3.12+
- Zerodha Kite Connect API key and secret
- Daily access token (generated via OAuth)

## Setup

```bash
cd ~/Projects/nifty500-screener
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Kite credentials
```

### Optional: TA-Lib

```bash
brew install ta-lib   # macOS
pip install TA-Lib
```

If TA-Lib is unavailable, the app falls back to `pandas-ta`.

## Kite Authentication

1. Set `KITE_API_KEY` and `KITE_API_SECRET` in `.env`
2. Run the app and open the Kite login link in the sidebar
3. Log in with your Zerodha account
4. Paste the `request_token` from the redirect URL
5. Copy the generated access token to `.env` as `KITE_ACCESS_TOKEN`

Access tokens expire daily — regenerate each trading session.

## Run

```bash
streamlit run app.py
```

## Usage

1. **Refresh Market Data** — Downloads 260 days of NIFTY 50 + NIFTY 500 daily candles
2. **Run Scan** — Evaluates all 7 strategies using local data only (no API calls)

## Architecture

```
config/         Settings from environment
database/       DuckDB manager + schema
data/           Kite fetcher, NIFTY 500 loader, refresh service
indicators/     RSI, MA, ATR, relative strength
strategies/     7 strategies + strategy engine
scanners/       Scan orchestration + confluence
exports/        CSV/Excel export
utils/          Logging and helpers
app.py          Streamlit UI
```

## Performance Targets

- Refresh: under 5 minutes (501 symbols × 350ms ≈ 3 min)
- Scan: under 10 seconds (multi-threaded evaluation)

## Future Backtesting

Each strategy exposes:

```python
def evaluate(symbol, candles, evaluation_index, context=None) -> StrategySignal | None
```

The strategy engine accepts an explicit `evaluation_index` for historical replay without modifying strategy logic.
