"""F&O intraday backtest engine."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from backtest.metrics import summarize_all_strategies
from backtest.models import BacktestResult, TradeLog
from database.duckdb_manager import DuckDBManager
from fno.strategy_engine import FNO_STRATEGIES, FnoStrategyEngine
from utils.logger import get_logger

logger = get_logger(__name__)

_STRATEGY_NAMES = {sid: name for sid, name, _ in FNO_STRATEGIES}


class FnoBacktester:
    """Intraday F&O backtest with bar-forward trade simulation."""

    def __init__(
        self,
        db: DuckDBManager | None = None,
        engine: FnoStrategyEngine | None = None,
        interval: str = "5minute",
    ) -> None:
        self.db = db or DuckDBManager()
        self.engine = engine or FnoStrategyEngine()
        self.interval = interval

    def run(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        strategy_ids: list[int] | None = None,
        cost_bps: float = 5.0,
        stop_pct: float = 0.5,
        target_pct: float = 1.0,
        symbols: list[str] | None = None,
    ) -> BacktestResult:
        intraday_map = self.db.load_intraday_candles(self.interval, symbols)
        if not intraday_map:
            raise RuntimeError(
                "No F&O intraday data. Run F&O data refresh from the app or CLI."
            )

        all_dates: set[date] = set()
        for df in intraday_map.values():
            dts = pd.to_datetime(df["trade_datetime"]).dt.date
            all_dates.update(dts.tolist())

        sorted_dates = sorted(all_dates)
        if start_date:
            sorted_dates = [d for d in sorted_dates if d >= start_date]
        if end_date:
            sorted_dates = [d for d in sorted_dates if d <= end_date]
        if not sorted_dates:
            raise RuntimeError("No trading dates in range.")

        trades: list[TradeLog] = []
        allowed = set(strategy_ids) if strategy_ids else None

        for symbol, full_df in intraday_map.items():
            full_df = full_df.sort_values("trade_datetime").reset_index(drop=True)
            full_df["_date"] = pd.to_datetime(full_df["trade_datetime"]).dt.date
            prior_high: float | None = None

            for trade_date in sorted_dates:
                day_df = full_df[full_df["_date"] == trade_date].drop(columns=["_date"])
                if len(day_df) < 10:
                    continue

                day_df = day_df.reset_index(drop=True)
                signals = self.engine.evaluate_day(
                    symbol, day_df, prior_session_high=prior_high, strategy_ids=strategy_ids
                )

                seen: set[tuple[int, int]] = set()
                for bar_idx, sid, sig in signals:
                    key = (sid, bar_idx)
                    if key in seen:
                        continue
                    seen.add(key)

                    ret, reason, exit_price = self.engine.simulate_intraday_trade(
                        day_df,
                        bar_idx,
                        float(sig.trigger_price),
                        stop_pct=stop_pct,
                        target_pct=target_pct,
                    )

                    signal_dt = sig.signal_date
                    if isinstance(signal_dt, pd.Timestamp):
                        signal_dt = signal_dt.to_pydatetime()

                    trades.append(
                        TradeLog(
                            signal_date=trade_date,
                            symbol=symbol,
                            strategy_id=sid,
                            strategy_name=sig.strategy_name,
                            entry_price=float(sig.trigger_price),
                            exit_price=exit_price,
                            exit_date=trade_date,
                            hold_days=0,
                            return_pct=ret,
                            alpha_pct=None,
                            score=sig.score,
                            exit_reason=reason,
                            cost_bps=cost_bps,
                            simulated=True,
                            metrics=dict(sig.metrics),
                        )
                    )

                prior_high = float(day_df["high"].max())

        summaries = summarize_all_strategies(trades)
        return BacktestResult(
            segment="fno",
            start_date=sorted_dates[0],
            end_date=sorted_dates[-1],
            hold_days=0,
            trades=trades,
            summaries=summaries,
            cost_bps=cost_bps,
            mode="intraday_simulated",
        )
