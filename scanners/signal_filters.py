"""Post-scan signal limits applied before storage and portfolio."""

from __future__ import annotations

import pandas as pd

from config.settings import MAX_SHARE_PRICE_INR, RS_LEADERS_MAX_PICKS
from strategies.base import StrategySignal
from strategies.strategy_6_relative_strength_leaders import STRATEGY_ID as RS_LEADERS_ID


def exclude_signals_above_max_price(
    signals: list[StrategySignal],
    max_price: float | None = None,
) -> list[StrategySignal]:
    """Drop signals whose entry price exceeds the paper-trading share cap."""
    cap = max_price if max_price is not None else MAX_SHARE_PRICE_INR
    return [s for s in signals if s.trigger_price <= cap]


def exclude_scan_df_above_max_price(
    df: pd.DataFrame,
    max_price: float | None = None,
) -> pd.DataFrame:
    """Drop scan rows whose trigger price exceeds the paper-trading share cap."""
    if df.empty or "trigger_price" not in df.columns:
        return df
    cap = max_price if max_price is not None else MAX_SHARE_PRICE_INR
    return df[df["trigger_price"] <= cap].reset_index(drop=True)


def limit_rs_leader_signals(
    signals: list[StrategySignal],
    max_picks: int | None = None,
) -> list[StrategySignal]:
    """Keep only the top N RS Leaders signals by score."""
    cap = max_picks if max_picks is not None else RS_LEADERS_MAX_PICKS
    rs = [s for s in signals if s.strategy_id == RS_LEADERS_ID]
    other = [s for s in signals if s.strategy_id != RS_LEADERS_ID]
    rs_top = sorted(rs, key=lambda s: s.score, reverse=True)[:cap]
    return other + rs_top


def limit_rs_leader_scan_df(
    df: pd.DataFrame,
    max_picks: int | None = None,
) -> pd.DataFrame:
    """Keep only the top N RS Leaders rows by score in a scan results frame."""
    if df.empty:
        return df
    cap = max_picks if max_picks is not None else RS_LEADERS_MAX_PICKS
    rs = df[df["strategy_id"] == RS_LEADERS_ID].sort_values("score", ascending=False)
    other = df[df["strategy_id"] != RS_LEADERS_ID]
    return pd.concat([other, rs.head(cap)], ignore_index=True)


def apply_signal_filters(
    signals: list[StrategySignal],
    max_rs_picks: int | None = None,
    max_price: float | None = None,
) -> list[StrategySignal]:
    """Apply all post-scan caps before persisting or displaying signals."""
    signals = limit_rs_leader_signals(signals, max_picks=max_rs_picks)
    return exclude_signals_above_max_price(signals, max_price=max_price)


def apply_scan_df_filters(
    df: pd.DataFrame,
    max_rs_picks: int | None = None,
    max_price: float | None = None,
) -> pd.DataFrame:
    """Apply all post-scan caps to stored scan result rows."""
    df = limit_rs_leader_scan_df(df, max_picks=max_rs_picks)
    return exclude_scan_df_above_max_price(df, max_price=max_price)
