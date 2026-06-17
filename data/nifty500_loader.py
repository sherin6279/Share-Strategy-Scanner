"""NIFTY 500 constituent loader."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from config.settings import NIFTY500_CACHE
from utils.logger import get_logger

logger = get_logger(__name__)

NSE_NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
)


def _fetch_from_nse() -> pd.DataFrame:
    """Download NIFTY 500 list from NSE indices website."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/json,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    resp = requests.get(NSE_NIFTY500_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    return df


def load_nifty500_symbols(force_refresh: bool = False) -> list[str]:
    """
    Load NIFTY 500 trading symbols (without exchange prefix).

    Caches locally for offline use.
    """
    if NIFTY500_CACHE.exists() and not force_refresh:
        df = pd.read_csv(NIFTY500_CACHE)
        return df["symbol"].tolist()

    try:
        df = _fetch_from_nse()
        # NSE CSV typically has 'Symbol' column
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = df[col].dropna().astype(str).str.strip().tolist()
        cache_df = pd.DataFrame({"symbol": symbols})
        cache_df.to_csv(NIFTY500_CACHE, index=False)
        logger.info("Loaded %d NIFTY 500 symbols from NSE", len(symbols))
        return symbols
    except Exception as exc:
        logger.warning("Failed to fetch NIFTY 500 from NSE: %s", exc)
        if NIFTY500_CACHE.exists():
            df = pd.read_csv(NIFTY500_CACHE)
            return df["symbol"].tolist()
        raise RuntimeError(
            "Cannot load NIFTY 500 symbols. Check network or provide cached file."
        ) from exc
