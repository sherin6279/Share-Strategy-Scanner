"""Equity swing backtest engine."""

from __future__ import annotations

from datetime import date

from config.settings import NIFTY50_SYMBOL
from backtest.date_index import DateIndex
from backtest.forward_returns import alpha_vs_benchmark, forward_return_pct
from backtest.metrics import summarize_all_strategies
from backtest.models import BacktestResult, TradeLog
from backtest.simulate_trade import simulate_equity_trade
from database.duckdb_manager import DuckDBManager
from strategies.strategy_engine import STRATEGIES, StrategyEngine
from utils.logger import get_logger

logger = get_logger(__name__)

_STRATEGY_NAMES = {sid: name for sid, name, _ in STRATEGIES}


class EquityBacktester:
    """Historical replay with calendar-based forward returns."""

    def __init__(
        self,
        db: DuckDBManager | None = None,
        engine: StrategyEngine | None = None,
    ) -> None:
        self.db = db or DuckDBManager()
        self.engine = engine or StrategyEngine()

    def run(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        hold_days: int = 5,
        step_days: int = 5,
        strategy_ids: list[int] | None = None,
        cost_bps: float = 15.0,
        simulate: bool = False,
        stop_pct: float = 5.0,
        target_pct: float = 10.0,
    ) -> BacktestResult:
        candles_map = self.db.load_all_candles()
        if not candles_map:
            raise RuntimeError("No candle data. Run Refresh Market Data first.")

        enriched_map = StrategyEngine.enrich_all(candles_map)
        date_index = DateIndex.from_enriched(enriched_map)

        if start_date is None:
            start_date = date_index.calendar[max(0, len(date_index.calendar) // 4)]
        if end_date is None:
            end_date = date_index.calendar[-hold_days - 1]

        allowed = set(strategy_ids) if strategy_ids else None
        trades: list[TradeLog] = []
        signal_dates = list(
            date_index.iter_signal_dates(start_date, end_date, hold_days, step_days)
        )

        logger.info(
            "Equity backtest: %d signal dates from %s to %s (step=%d)",
            len(signal_dates),
            start_date,
            end_date,
            step_days,
        )

        for signal_date in signal_dates:
            idx_map = date_index.idx_map_for_date(signal_date)
            if len(idx_map) < 10:
                continue

            signals = self.engine.run_on_date(signal_date, enriched_map, idx_map)
            for sig in signals:
                if allowed and sig.strategy_id not in allowed:
                    continue

                entry = float(sig.trigger_price)
                if simulate:
                    ret, exit_date, reason, exit_price = simulate_equity_trade(
                        enriched_map[sig.symbol],
                        date_index,
                        sig.symbol,
                        signal_date,
                        entry,
                        stop_pct=stop_pct,
                        target_pct=target_pct,
                        max_hold_days=hold_days,
                    )
                else:
                    ret, exit_date, entry, exit_price = forward_return_pct(
                        date_index, sig.symbol, signal_date, hold_days
                    )
                    reason = "forward_hold"

                alpha = alpha_vs_benchmark(
                    date_index, sig.symbol, NIFTY50_SYMBOL, signal_date, hold_days
                )

                if ret is None:
                    continue

                trades.append(
                    TradeLog(
                        signal_date=signal_date,
                        symbol=sig.symbol,
                        strategy_id=sig.strategy_id,
                        strategy_name=sig.strategy_name,
                        entry_price=entry,
                        exit_price=exit_price,
                        exit_date=exit_date,
                        hold_days=hold_days,
                        return_pct=ret,
                        alpha_pct=alpha,
                        score=sig.score,
                        exit_reason=reason,
                        cost_bps=cost_bps,
                        simulated=simulate,
                        metrics=dict(sig.metrics),
                    )
                )

        summaries = summarize_all_strategies(trades)
        return BacktestResult(
            segment="equity",
            start_date=start_date,
            end_date=end_date,
            hold_days=hold_days,
            trades=trades,
            summaries=summaries,
            cost_bps=cost_bps,
            mode="simulated" if simulate else "forward_return",
        )
