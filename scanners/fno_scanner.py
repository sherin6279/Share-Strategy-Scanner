"""F&O intraday scan orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

from database.duckdb_manager import DuckDBManager
from fno.strategy_engine import FNO_STRATEGIES, FnoStrategyEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class FnoScanner:
    """Run F&O strategies on latest intraday session."""

    def __init__(
        self,
        db: DuckDBManager | None = None,
        engine: FnoStrategyEngine | None = None,
        interval: str = "5minute",
    ) -> None:
        self.db = db or DuckDBManager()
        self.engine = engine or FnoStrategyEngine()
        self.interval = interval

    def run_scan(self) -> dict:
        intraday_map = self.db.load_intraday_candles(self.interval)
        if not intraday_map:
            raise RuntimeError("No F&O intraday data. Refresh F&O data first.")

        scan_time = datetime.now()
        run_id = uuid.uuid4().hex[:16]
        records: list[dict] = []

        for symbol, df in intraday_map.items():
            df = df.sort_values("trade_datetime").reset_index(drop=True)
            trade_date = pd.to_datetime(df.iloc[-1]["trade_datetime"]).date()
            day_df = df[pd.to_datetime(df["trade_datetime"]).dt.date == trade_date]
            day_df = day_df.reset_index(drop=True)

            if len(day_df) < 10:
                continue

            prior_dates = pd.to_datetime(df["trade_datetime"]).dt.date.unique()
            prior_dates = sorted([d for d in prior_dates if d < trade_date])
            prior_high = None
            if prior_dates:
                prev = df[pd.to_datetime(df["trade_datetime"]).dt.date == prior_dates[-1]]
                prior_high = float(prev["high"].max())

            signals = self.engine.evaluate_day(symbol, day_df, prior_session_high=prior_high)
            seen: set[int] = set()
            for bar_idx, sid, sig in signals:
                if sid in seen:
                    continue
                seen.add(sid)
                records.append(
                    {
                        "scan_run_id": run_id,
                        "scan_timestamp": scan_time,
                        "strategy_id": sid,
                        "symbol": symbol,
                        "signal_datetime": sig.signal_date,
                        "score": sig.score,
                        "trigger_price": sig.trigger_price,
                        "metrics": sig.metrics,
                    }
                )

        self.db.insert_fno_scan_results(records)
        self.db.insert_fno_scan_run(run_id, scan_time, len(records))
        self.db.set_metadata("last_fno_scan_run_id", run_id)
        self.db.set_metadata("last_fno_scan_timestamp", scan_time.isoformat())
        logger.info(
            "Stored %d F&O scan results (run %s) at %s",
            len(records),
            run_id,
            scan_time.isoformat(),
        )

        return {
            "scan_run_id": run_id,
            "scan_timestamp": scan_time.isoformat(),
            "signal_count": len(records),
        }
