"""Strategy engine – orchestrates evaluation across all strategies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Callable

import pandas as pd

from config.settings import NIFTY50_SYMBOL, SCAN_THREAD_WORKERS
from indicators import enrich_candles
from indicators.relative_strength import compute_cross_sectional_returns
from strategies import (
    strategy_1_basic_breakout as s1,
    strategy_2_refined_breakout as s2,
    strategy_3_enterprise_breakout as s3,
    strategy_4_volatility_compression as s4,
    strategy_5_trend_pullback as s5,
    strategy_6_relative_strength_leaders as s6,
    strategy_7_stage1_base_breakout as s7,
)
from strategies.base import StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)

STRATEGIES: list[tuple[int, str, Callable]] = [
    (s1.STRATEGY_ID, s1.STRATEGY_NAME, s1.evaluate),
    (s2.STRATEGY_ID, s2.STRATEGY_NAME, s2.evaluate),
    (s3.STRATEGY_ID, s3.STRATEGY_NAME, s3.evaluate),
    (s4.STRATEGY_ID, s4.STRATEGY_NAME, s4.evaluate),
    (s5.STRATEGY_ID, s5.STRATEGY_NAME, s5.evaluate),
    (s6.STRATEGY_ID, s6.STRATEGY_NAME, s6.evaluate),
    (s7.STRATEGY_ID, s7.STRATEGY_NAME, s7.evaluate),
]


class StrategyEngine:
    """Runs all strategies against local candle data at a given evaluation index."""

    def __init__(self) -> None:
        self.market_uptrend: bool = True
        self.strategy_3_paused: bool = False

    def _build_context(
        self,
        enriched_map: dict[str, pd.DataFrame],
        idx_map: dict[str, int],
        nifty50_candles: pd.DataFrame | None,
        nifty_eval_idx: int | None = None,
    ) -> dict[str, Any]:
        """Build shared context including RS rankings for strategy 6."""
        context: dict[str, Any] = {"nifty50_candles": nifty50_candles}

        # Market regime — use aligned NIFTY index for historical replay
        if nifty50_candles is not None and len(nifty50_candles) > 0:
            nifty_idx = nifty_eval_idx
            if nifty_idx is None:
                nifty_idx = idx_map.get(NIFTY50_SYMBOL, len(nifty50_candles) - 1)
            if nifty_idx < 0 or nifty_idx >= len(nifty50_candles):
                nifty_idx = len(nifty50_candles) - 1
            row = nifty50_candles.iloc[nifty_idx]
            sma200 = row.get("sma200")
            self.market_uptrend = (
                not pd.isna(sma200) and row["close"] > sma200
            )
        else:
            self.market_uptrend = False

        self.strategy_3_paused = not self.market_uptrend
        context["market_uptrend"] = self.market_uptrend

        # Cross-sectional return rankings for strategy 6 (point-in-time via idx_map)
        ret_90 = compute_cross_sectional_returns(enriched_map, idx_map, 90)
        ret_180 = compute_cross_sectional_returns(enriched_map, idx_map, 180)

        ret_90_pctile = ret_90.rank(pct=True) * 100.0 if not ret_90.empty else pd.Series(dtype=float)
        ret_180_pctile = ret_180.rank(pct=True) * 100.0 if not ret_180.empty else pd.Series(dtype=float)

        context["rs_rankings"] = {
            "ret_90_pctile": ret_90_pctile.to_dict(),
            "ret_180_pctile": ret_180_pctile.to_dict(),
        }
        return context

    def _evaluate_symbol(
        self,
        symbol: str,
        candles: pd.DataFrame,
        evaluation_index: int,
        context: dict[str, Any],
    ) -> list[StrategySignal]:
        """Evaluate all strategies for one symbol."""
        signals: list[StrategySignal] = []
        for strategy_id, _name, evaluate_fn in STRATEGIES:
            if strategy_id == 3 and not context.get("market_uptrend", False):
                continue
            try:
                signal = evaluate_fn(symbol, candles, evaluation_index, context)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("Strategy %d failed for %s: %s", strategy_id, symbol, exc)
        return signals

    def run(
        self,
        candles_map: dict[str, pd.DataFrame],
        evaluation_index: int | None = None,
    ) -> tuple[list[StrategySignal], datetime]:
        """
        Run all strategies on all symbols using local data only.

        Uses latest completed candle if evaluation_index is None.
        """
        scan_time = datetime.now()

        # Enrich all candles with indicators
        enriched_map: dict[str, pd.DataFrame] = {}
        idx_map: dict[str, int] = {}

        for symbol, df in candles_map.items():
            if df.empty:
                continue
            enriched = enrich_candles(df)
            idx = evaluation_index if evaluation_index is not None else len(enriched) - 1
            if idx < 0 or idx >= len(enriched):
                continue
            enriched_map[symbol] = enriched
            idx_map[symbol] = idx

        nifty50 = enriched_map.get(NIFTY50_SYMBOL)
        context = self._build_context(enriched_map, idx_map, nifty50)

        # Exclude index from stock scans
        stock_symbols = [s for s in enriched_map if s != NIFTY50_SYMBOL]

        all_signals: list[StrategySignal] = []

        def _worker(sym: str) -> list[StrategySignal]:
            idx = idx_map[sym]
            return self._evaluate_symbol(sym, enriched_map[sym], idx, context)

        with ThreadPoolExecutor(max_workers=SCAN_THREAD_WORKERS) as executor:
            futures = {executor.submit(_worker, sym): sym for sym in stock_symbols}
            for future in as_completed(futures):
                try:
                    all_signals.extend(future.result())
                except Exception as exc:
                    sym = futures[future]
                    logger.error("Evaluation failed for %s: %s", sym, exc)

        logger.info(
            "Scan complete: %d signals across %d symbols",
            len(all_signals),
            len(stock_symbols),
        )
        return all_signals, scan_time

    def run_on_date(
        self,
        signal_date: date,
        enriched_map: dict[str, pd.DataFrame],
        idx_map: dict[str, int],
    ) -> list[StrategySignal]:
        """
        Run all strategies as-of a historical signal date.

        idx_map must map each symbol to its row index on signal_date.
        """
        if not idx_map:
            return []

        nifty50 = enriched_map.get(NIFTY50_SYMBOL)
        nifty_eval_idx = idx_map.get(NIFTY50_SYMBOL)
        context = self._build_context(
            enriched_map, idx_map, nifty50, nifty_eval_idx=nifty_eval_idx
        )

        stock_symbols = [
            s for s in idx_map if s != NIFTY50_SYMBOL and s in enriched_map
        ]
        all_signals: list[StrategySignal] = []

        for sym in stock_symbols:
            idx = idx_map[sym]
            try:
                all_signals.extend(
                    self._evaluate_symbol(sym, enriched_map[sym], idx, context)
                )
            except Exception as exc:
                logger.error("Evaluation failed for %s on %s: %s", sym, signal_date, exc)

        return all_signals

    @staticmethod
    def enrich_all(candles_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Enrich all symbol candle series once for backtesting."""
        return {
            symbol: enrich_candles(df)
            for symbol, df in candles_map.items()
            if not df.empty
        }

    @staticmethod
    def signals_to_records(
        signals: list[StrategySignal],
        scan_timestamp: datetime,
        scan_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert signals to database-ready records."""
        records = []
        for sig in signals:
            metrics = dict(sig.metrics)
            if sig.suggested_action:
                metrics["suggested_action"] = sig.suggested_action
            records.append(
                {
                    "scan_run_id": scan_run_id,
                    "scan_timestamp": scan_timestamp,
                    "strategy_id": sig.strategy_id,
                    "symbol": sig.symbol,
                    "signal_date": sig.signal_date,
                    "score": sig.score,
                    "trigger_price": sig.trigger_price,
                    "metrics": metrics,
                }
            )
        return records
