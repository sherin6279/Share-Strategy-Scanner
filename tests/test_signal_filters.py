"""Tests for RS Leaders top-N cap."""

from datetime import date

import pandas as pd

from scanners.signal_filters import (
    apply_scan_df_filters,
    apply_signal_filters,
    exclude_scan_df_above_max_price,
    exclude_signals_above_max_price,
    limit_rs_leader_scan_df,
    limit_rs_leader_signals,
)
from strategies.base import StrategySignal


def _rs_signal(symbol: str, score: float) -> StrategySignal:
    return StrategySignal(
        strategy_id=6,
        strategy_name="RS Leaders",
        symbol=symbol,
        signal_date=date(2025, 2, 28),
        score=score,
        trigger_price=100.0,
    )


def test_limit_rs_leader_signals_keeps_top_five():
    signals = [_rs_signal(f"S{i}", float(i)) for i in range(10)]
    signals.append(
        StrategySignal(
            strategy_id=1,
            strategy_name="Basic Breakout",
            symbol="OTHER",
            signal_date=date(2025, 2, 28),
            score=5.0,
            trigger_price=50.0,
        )
    )
    out = limit_rs_leader_signals(signals, max_picks=5)
    rs = [s for s in out if s.strategy_id == 6]
    assert len(rs) == 5
    assert {s.symbol for s in rs} == {"S9", "S8", "S7", "S6", "S5"}
    assert any(s.symbol == "OTHER" for s in out)


def test_limit_rs_leader_scan_df_keeps_top_five():
    rows = [
        {"strategy_id": 6, "symbol": f"S{i}", "score": float(i)}
        for i in range(8)
    ]
    rows.append({"strategy_id": 1, "symbol": "X", "score": 1.0})
    df = pd.DataFrame(rows)
    out = limit_rs_leader_scan_df(df, max_picks=5)
    assert len(out[out["strategy_id"] == 6]) == 5
    assert len(out[out["strategy_id"] == 1]) == 1


def test_exclude_signals_above_max_price():
    signals = [
        StrategySignal(
            strategy_id=1,
            strategy_name="Basic Breakout",
            symbol="CHEAP",
            signal_date=date(2025, 2, 28),
            score=5.0,
            trigger_price=5000.0,
        ),
        StrategySignal(
            strategy_id=1,
            strategy_name="Basic Breakout",
            symbol="DEAR",
            signal_date=date(2025, 2, 28),
            score=5.0,
            trigger_price=15000.0,
        ),
    ]
    out = exclude_signals_above_max_price(signals, max_price=10_000.0)
    assert {s.symbol for s in out} == {"CHEAP"}


def test_exclude_scan_df_above_max_price():
    df = pd.DataFrame(
        [
            {"strategy_id": 1, "symbol": "CHEAP", "score": 1.0, "trigger_price": 9000.0},
            {"strategy_id": 1, "symbol": "DEAR", "score": 2.0, "trigger_price": 12000.0},
        ]
    )
    out = exclude_scan_df_above_max_price(df, max_price=10_000.0)
    assert list(out["symbol"]) == ["CHEAP"]


def test_apply_signal_filters_caps_rs_and_price():
    signals = [_rs_signal(f"S{i}", float(i)) for i in range(8)]
    signals.append(
        StrategySignal(
            strategy_id=1,
            strategy_name="Basic Breakout",
            symbol="DEAR",
            signal_date=date(2025, 2, 28),
            score=9.0,
            trigger_price=20_000.0,
        )
    )
    out = apply_signal_filters(signals, max_rs_picks=3, max_price=10_000.0)
    assert len([s for s in out if s.strategy_id == 6]) == 3
    assert all(s.trigger_price <= 10_000.0 for s in out)
    assert not any(s.symbol == "DEAR" for s in out)
