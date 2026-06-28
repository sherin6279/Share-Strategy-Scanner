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

    def __init__(
        self,
        db_path: Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.db_path = db_path or DB_PATH
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _open_connection(self) -> duckdb.DuckDBPyConnection:
        try:
            if self.read_only:
                return duckdb.connect(str(self.db_path), read_only=True)
            return duckdb.connect(str(self.db_path))
        except duckdb.IOException as exc:
            if "Conflicting lock" in str(exc):
                raise RuntimeError(
                    "Database is locked by another process (usually a running "
                    f"Streamlit app). Stop it first, then retry.\n"
                    f"  DB: {self.db_path}\n"
                    f"  Check: lsof {self.db_path}"
                ) from exc
            raise

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = self._open_connection()
            if not self.read_only:
                self._initialize_schema()
        return self._conn

    def _initialize_schema(self) -> None:
        if self._conn is None:
            return
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self._conn.execute(schema_sql)
        if not self.read_only:
            self._backfill_scan_run_ids()

    def _backfill_scan_run_ids(self) -> None:
        """Link legacy scan rows to scan_runs when run_id was not stored."""
        try:
            self.conn.execute(
                """
                UPDATE scan_results AS sr
                SET scan_run_id = r.run_id
                FROM scan_runs AS r
                WHERE sr.scan_run_id IS NULL
                  AND sr.scan_timestamp = r.scan_timestamp
                  AND r.segment = 'equity'
                """
            )
            self.conn.execute(
                """
                UPDATE fno_scan_results AS fr
                SET scan_run_id = r.run_id
                FROM scan_runs AS r
                WHERE fr.scan_run_id IS NULL
                  AND fr.scan_timestamp = r.scan_timestamp
                  AND r.segment = 'fno'
                """
            )
        except duckdb.Error as exc:
            logger.debug("scan_run_id backfill skipped: %s", exc)

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
                    r.get("scan_run_id"),
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
            (scan_run_id, scan_timestamp, strategy_id, symbol, signal_date,
             score, trigger_price, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def insert_scan_run(
        self,
        run_id: str,
        scan_timestamp: datetime,
        segment: str,
        signal_count: int,
        market_uptrend: bool,
        strategy_counts: dict[int, int],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO scan_runs
            (run_id, scan_timestamp, segment, signal_count, market_uptrend, strategy_counts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                scan_timestamp,
                segment,
                signal_count,
                market_uptrend,
                json.dumps({str(k): v for k, v in strategy_counts.items()}),
            ],
        )

    def list_scan_runs(self, segment: str = "equity", limit: int = 50) -> list[dict]:
        rows: list[tuple] = []
        try:
            rows = self.conn.execute(
                """
                SELECT run_id, scan_timestamp, signal_count, market_uptrend, strategy_counts
                FROM scan_runs
                WHERE segment = ?
                ORDER BY scan_timestamp DESC
                LIMIT ?
                """,
                [segment, limit],
            ).fetchall()
        except duckdb.Error:
            rows = []

        runs: list[dict] = []
        seen_timestamps: set[datetime] = set()
        for r in rows:
            counts_raw = r[4] or "{}"
            try:
                strategy_counts = {int(k): v for k, v in json.loads(counts_raw).items()}
            except (json.JSONDecodeError, ValueError, TypeError):
                strategy_counts = {}
            seen_timestamps.add(r[1])
            runs.append(
                {
                    "run_id": r[0],
                    "scan_timestamp": r[1],
                    "signal_count": r[2],
                    "market_uptrend": bool(r[3]) if r[3] is not None else True,
                    "strategy_counts": strategy_counts,
                }
            )

        table = "scan_results" if segment == "equity" else "fno_scan_results"
        legacy = self.conn.execute(
            f"""
            SELECT scan_timestamp, COUNT(*) AS cnt
            FROM {table}
            GROUP BY scan_timestamp
            ORDER BY scan_timestamp DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        for ts, cnt in legacy:
            if ts in seen_timestamps:
                continue
            runs.append(
                {
                    "run_id": None,
                    "scan_timestamp": ts,
                    "signal_count": cnt,
                    "market_uptrend": True,
                    "strategy_counts": {},
                }
            )

        runs.sort(key=lambda r: r["scan_timestamp"], reverse=True)
        return runs[:limit]

    def list_scan_timestamps(self, segment: str = "equity", limit: int = 50) -> list[datetime]:
        return [r["scan_timestamp"] for r in self.list_scan_runs(segment, limit)]

    def get_scan_results(
        self,
        scan_run_id: str | None = None,
        scan_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        if scan_run_id:
            df = self.conn.execute(
                """
                SELECT scan_run_id, scan_timestamp, strategy_id, symbol, signal_date,
                       score, trigger_price, metrics
                FROM scan_results WHERE scan_run_id = ?
                ORDER BY strategy_id, score DESC
                """,
                [scan_run_id],
            ).fetchdf()
            if not df.empty:
                return df

            if scan_timestamp is None:
                row = self.conn.execute(
                    "SELECT scan_timestamp FROM scan_runs WHERE run_id = ?",
                    [scan_run_id],
                ).fetchone()
                if row:
                    scan_timestamp = row[0]

            if scan_timestamp is not None:
                return self.conn.execute(
                    """
                    SELECT scan_run_id, scan_timestamp, strategy_id, symbol, signal_date,
                           score, trigger_price, metrics
                    FROM scan_results WHERE scan_timestamp = ?
                    ORDER BY strategy_id, score DESC
                    """,
                    [scan_timestamp],
                ).fetchdf()
            return df

        if scan_timestamp is not None:
            return self.conn.execute(
                """
                SELECT scan_run_id, scan_timestamp, strategy_id, symbol, signal_date,
                       score, trigger_price, metrics
                FROM scan_results WHERE scan_timestamp = ?
                ORDER BY strategy_id, score DESC
                """,
                [scan_timestamp],
            ).fetchdf()

        return self.get_latest_scan_results()

    def get_latest_scan_results(self) -> pd.DataFrame:
        latest_run = self.conn.execute(
            """
            SELECT run_id FROM scan_runs
            WHERE segment = 'equity'
            ORDER BY scan_timestamp DESC
            LIMIT 1
            """
        ).fetchone()
        if latest_run:
            return self.get_scan_results(scan_run_id=latest_run[0])

        ts = self.conn.execute(
            "SELECT MAX(scan_timestamp) FROM scan_results"
        ).fetchone()[0]
        if ts is None:
            return pd.DataFrame()
        return self.get_scan_results(scan_timestamp=ts)

    def get_last_refresh_timestamp(self) -> str | None:
        return self.get_metadata("last_refresh_timestamp")

    def upsert_intraday_candles(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        required = {
            "symbol", "interval", "trade_datetime",
            "open", "high", "low", "close", "volume",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing intraday columns: {missing}")

        clean = df[list(required)].copy()
        clean["trade_datetime"] = pd.to_datetime(clean["trade_datetime"])
        self.conn.register("_intraday_tmp", clean)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO candles_intraday
            SELECT symbol, interval, trade_datetime, open, high, low, close, volume
            FROM _intraday_tmp
            """
        )
        self.conn.unregister("_intraday_tmp")
        return len(clean)

    def load_intraday_candles(
        self,
        interval: str = "5minute",
        symbols: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            query = f"""
                SELECT symbol, interval, trade_datetime, open, high, low, close, volume
                FROM candles_intraday
                WHERE interval = ? AND symbol IN ({placeholders})
                ORDER BY symbol, trade_datetime
            """
            params: list = [interval, *symbols]
        else:
            query = """
                SELECT symbol, interval, trade_datetime, open, high, low, close, volume
                FROM candles_intraday WHERE interval = ?
                ORDER BY symbol, trade_datetime
            """
            params = [interval]

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return {}
        return {
            sym: grp.reset_index(drop=True)
            for sym, grp in df.groupby("symbol")
        }

    def get_intraday_symbols(self, interval: str = "5minute") -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM candles_intraday WHERE interval = ? ORDER BY symbol",
            [interval],
        ).fetchall()
        return [r[0] for r in rows]

    def insert_fno_scan_results(self, results: list[dict[str, Any]]) -> int:
        if not results:
            return 0
        rows = []
        for r in results:
            rows.append(
                (
                    r.get("scan_run_id"),
                    r["scan_timestamp"],
                    r["strategy_id"],
                    r["symbol"],
                    r["signal_datetime"],
                    r["score"],
                    r["trigger_price"],
                    json.dumps(r.get("metrics", {})),
                )
            )
        self.conn.executemany(
            """
            INSERT INTO fno_scan_results
            (scan_run_id, scan_timestamp, strategy_id, symbol, signal_datetime,
             score, trigger_price, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def insert_fno_scan_run(
        self,
        run_id: str,
        scan_timestamp: datetime,
        signal_count: int,
    ) -> None:
        self.insert_scan_run(run_id, scan_timestamp, "fno", signal_count, False, {})

    def list_fno_scan_runs(self, limit: int = 50) -> list[dict]:
        return self.list_scan_runs(segment="fno", limit=limit)

    def list_fno_scan_timestamps(self, limit: int = 50) -> list[datetime]:
        return self.list_scan_timestamps(segment="fno", limit=limit)

    def get_fno_scan_results(
        self,
        scan_run_id: str | None = None,
        scan_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        if scan_run_id:
            return self.conn.execute(
                """
                SELECT scan_run_id, scan_timestamp, strategy_id, symbol, signal_datetime,
                       score, trigger_price, metrics
                FROM fno_scan_results WHERE scan_run_id = ?
                ORDER BY strategy_id, score DESC
                """,
                [scan_run_id],
            ).fetchdf()

        if scan_timestamp is not None:
            return self.conn.execute(
                """
                SELECT scan_run_id, scan_timestamp, strategy_id, symbol, signal_datetime,
                       score, trigger_price, metrics
                FROM fno_scan_results WHERE scan_timestamp = ?
                ORDER BY strategy_id, score DESC
                """,
                [scan_timestamp],
            ).fetchdf()

        return self.get_latest_fno_scan_results()

    def get_latest_fno_scan_results(self) -> pd.DataFrame:
        latest_run = self.conn.execute(
            """
            SELECT run_id FROM scan_runs
            WHERE segment = 'fno'
            ORDER BY scan_timestamp DESC
            LIMIT 1
            """
        ).fetchone()
        if latest_run:
            return self.get_fno_scan_results(scan_run_id=latest_run[0])

        ts = self.conn.execute(
            "SELECT MAX(scan_timestamp) FROM fno_scan_results"
        ).fetchone()[0]
        if ts is None:
            return pd.DataFrame()
        return self.get_fno_scan_results(scan_timestamp=ts)

    def insert_portfolio_holdings(self, holdings: list[dict[str, Any]]) -> int:
        if not holdings:
            return 0
        rows = [
            (
                h["holding_id"],
                h["scan_run_id"],
                h["symbol"],
                h["source_type"],
                h["source_label"],
                json.dumps(h.get("strategy_ids", [])),
                h["purchase_date"],
                h["purchase_price"],
                h.get("quantity", 1),
                h.get("score"),
                h["created_at"],
            )
            for h in holdings
        ]
        self.conn.executemany(
            """
            INSERT INTO portfolio_holdings
            (holding_id, scan_run_id, symbol, source_type, source_label,
             strategy_ids, purchase_date, purchase_price, quantity, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def portfolio_exists_for_scan(self, scan_run_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM portfolio_holdings WHERE scan_run_id = ? LIMIT 1",
            [scan_run_id],
        ).fetchone()
        return row is not None

    def list_portfolio_holdings(self) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT holding_id, scan_run_id, symbol, source_type, source_label,
                   strategy_ids, purchase_date, purchase_price, quantity, score, created_at
            FROM portfolio_holdings
            ORDER BY purchase_date DESC, symbol
            """
        ).fetchdf()

    def count_portfolio_holdings(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM portfolio_holdings").fetchone()
        return int(row[0]) if row else 0

    def insert_paper_trade_batch(
        self,
        batch_id: str,
        scan_run_id: str | None,
        created_at: datetime,
        entry_date: date,
        position_count: int,
        notes: str = "",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO paper_trade_batches
            (batch_id, scan_run_id, created_at, entry_date, position_count, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [batch_id, scan_run_id, created_at, entry_date, position_count, notes],
        )

    def insert_paper_trade_positions(self, positions: list[dict[str, Any]]) -> int:
        if not positions:
            return 0
        rows = [
            (
                p["position_id"],
                p["batch_id"],
                p["symbol"],
                p["strategy_id"],
                p["entry_date"],
                p["entry_price"],
                p["score"],
            )
            for p in positions
        ]
        self.conn.executemany(
            """
            INSERT INTO paper_trade_positions
            (position_id, batch_id, symbol, strategy_id, entry_date, entry_price, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def get_paper_trade_batch(self, batch_id: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT batch_id, scan_run_id, created_at, entry_date, position_count, notes
            FROM paper_trade_batches WHERE batch_id = ?
            """,
            [batch_id],
        ).fetchone()
        if not row:
            return None
        return {
            "batch_id": row[0],
            "scan_run_id": row[1],
            "created_at": row[2],
            "entry_date": row[3],
            "position_count": row[4],
            "notes": row[5] or "",
        }

    def get_paper_trade_batch_by_scan(self, scan_run_id: str) -> dict | None:
        row = self.conn.execute(
            """
            SELECT batch_id, scan_run_id, created_at, entry_date, position_count, notes
            FROM paper_trade_batches WHERE scan_run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [scan_run_id],
        ).fetchone()
        if not row:
            return None
        return {
            "batch_id": row[0],
            "scan_run_id": row[1],
            "created_at": row[2],
            "entry_date": row[3],
            "position_count": row[4],
            "notes": row[5] or "",
        }

    def list_paper_trade_batches(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT batch_id, scan_run_id, created_at, entry_date, position_count, notes
            FROM paper_trade_batches
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [
            {
                "batch_id": r[0],
                "scan_run_id": r[1],
                "created_at": r[2],
                "entry_date": r[3],
                "position_count": r[4],
                "notes": r[5] or "",
            }
            for r in rows
        ]

    def get_paper_trade_positions(self, batch_id: str) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT position_id, batch_id, symbol, strategy_id, entry_date,
                   entry_price, score
            FROM paper_trade_positions
            WHERE batch_id = ?
            ORDER BY symbol, strategy_id
            """,
            [batch_id],
        ).fetchdf()

    def get_latest_closes(self, symbols: list[str]) -> dict[str, tuple[date, float]]:
        """Return latest (trade_date, close) per symbol from stored candles."""
        if not symbols:
            return {}
        placeholders = ", ".join(["?"] * len(symbols))
        df = self.conn.execute(
            f"""
            SELECT symbol, trade_date, close
            FROM candles
            WHERE symbol IN ({placeholders})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) = 1
            """,
            symbols,
        ).fetchdf()
        result: dict[str, tuple[date, float]] = {}
        for _, row in df.iterrows():
            td = row["trade_date"]
            if hasattr(td, "date"):
                td = td.date()
            result[row["symbol"]] = (td, float(row["close"]))
        return result

    def save_backtest_run(
        self,
        run_id: str,
        segment: str,
        start_date: date,
        end_date: date,
        config: dict,
        summary: list[dict],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO backtest_runs
            (run_id, segment, start_date, end_date, config, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                segment,
                start_date,
                end_date,
                json.dumps(config),
                json.dumps(summary),
                datetime.now(),
            ],
        )
