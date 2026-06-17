"""DuckDB database manager with upsert support."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config.settings import DB_PATH
from utils.logger import get_logger

logger = get_logger(__name__)
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DuckDBManager:
    """Manages local DuckDB storage for candles and scan results."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._initialize_schema()
        return self._conn

    def _initialize_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.execute(schema_sql)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def upsert_candles(self, df: pd.DataFrame) -> int:
        """Insert or replace candle rows. Returns rows affected."""
        if df.empty:
            return 0

        required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing candle columns: {missing}")

        clean = df[list(required)].copy()
        clean["trade_date"] = pd.to_datetime(clean["trade_date"]).dt.date
        self.conn.register("_candles_tmp", clean)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO candles
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM _candles_tmp
            """
        )
        self.conn.unregister("_candles_tmp")
        return len(clean)

    def get_all_symbols(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM candles ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]

    def get_candles(self, symbol: str) -> pd.DataFrame:
        df = self.conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM candles WHERE symbol = ?
            ORDER BY trade_date
            """,
            [symbol],
        ).fetchdf()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        return df

    def load_all_candles(self) -> dict[str, pd.DataFrame]:
        """Batch load all candles grouped by symbol."""
        df = self.conn.execute(
            """
            SELECT symbol, trade_date, open, high, low, close, volume
            FROM candles ORDER BY symbol, trade_date
            """
        ).fetchdf()
        if df.empty:
            return {}

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        return {sym: grp.reset_index(drop=True) for sym, grp in df.groupby("symbol")}

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO scan_metadata (key, value) VALUES (?, ?)
            """,
            [key, value],
        )

    def get_metadata(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM scan_metadata WHERE key = ?", [key]
        ).fetchone()
        return row[0] if row else default

    def clear_scan_results(self, scan_timestamp: datetime | None = None) -> None:
        if scan_timestamp:
            self.conn.execute(
                "DELETE FROM scan_results WHERE scan_timestamp = ?",
                [scan_timestamp],
            )
        else:
            self.conn.execute("DELETE FROM scan_results")

    def insert_scan_results(self, results: list[dict[str, Any]]) -> int:
        if not results:
            return 0

        rows = []
        for r in results:
            metrics = r.get("metrics", {})
            rows.append(
                (
                    r["scan_timestamp"],
                    r["strategy_id"],
                    r["symbol"],
                    r["signal_date"],
                    r["score"],
                    r["trigger_price"],
                    json.dumps(metrics),
                )
            )

        self.conn.executemany(
            """
            INSERT INTO scan_results
            (scan_timestamp, strategy_id, symbol, signal_date, score, trigger_price, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def get_latest_scan_results(self) -> pd.DataFrame:
        ts = self.conn.execute(
            "SELECT MAX(scan_timestamp) FROM scan_results"
        ).fetchone()[0]
        if ts is None:
            return pd.DataFrame()

        df = self.conn.execute(
            """
            SELECT scan_timestamp, strategy_id, symbol, signal_date,
                   score, trigger_price, metrics
            FROM scan_results WHERE scan_timestamp = ?
            ORDER BY strategy_id, score DESC
            """,
            [ts],
        ).fetchdf()
        return df

    def get_last_refresh_timestamp(self) -> str | None:
        return self.get_metadata("last_refresh_timestamp")
