"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"
DB_PATH = PROJECT_ROOT / "data" / "market.duckdb"
NIFTY500_CACHE = DATA_DIR / "nifty500_symbols.csv"
LOG_DIR = PROJECT_ROOT / "logs"

# Kite Connect credentials (never hardcode secrets)
KITE_API_KEY: str = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET: str = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

# Rate limiting
REQUEST_DELAY_SEC: float = 0.35  # 350ms between requests
MAX_REQUESTS_PER_SEC: int = 3
RETRY_COUNT: int = 3

# Data parameters
CANDLE_DAYS: int = 260
NIFTY50_SYMBOL: str = "NIFTY 50"

# Strategy thresholds
MIN_TRADED_VALUE_CR_S1: float = 10.0  # ₹10 crore (Strategy 3)
MIN_TRADED_VALUE_CR_S6: float = 20.0  # ₹20 crore (Strategy 6)

# Performance
SCAN_THREAD_WORKERS: int = 8

# Backtest defaults
BACKTEST_HOLD_DAYS: int = 5
BACKTEST_STEP_DAYS: int = 5
BACKTEST_COST_BPS_EQUITY: float = 15.0
BACKTEST_COST_BPS_FNO: float = 5.0

# F&O intraday
FNO_INTERVAL: str = "5minute"
FNO_INTRADAY_DAYS: int = 30
FNO_INDEX_UNDERLYINGS: list[str] = ["NIFTY", "BANKNIFTY"]
FNO_STOCK_COUNT: int = 50
FNO_STOCK_VOLUME_LOOKBACK: int = 20

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
