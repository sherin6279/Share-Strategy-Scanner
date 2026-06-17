"""Strategy 5 – Strong Trend Pullback.

Conditions:
    - SMA50 > SMA200                      (golden cross — long-term uptrend confirmed)
    - Close > SMA200                      (price above long-term trend)
    - Close > EMA20                       (price above short-term trend)
    - RS_60 > 10                          (outperforming NIFTY 50 by 10%+ over 60 days)
    - Drawdown from 50-day high 5–12%     (healthy pullback, not a breakdown)
    - 50 < RSI(14) < 70                   (momentum confirmed recovering, not overbought)
    - Volume > 20-day avg volume          (buying interest on the bounce)
    - Bullish candle (Close > Open)       (current candle confirms upward intent)
    - NIFTY 50 regime gate                (market itself must be above its 200 SMA)
    - Avg traded value > ₹10 Cr           (liquidity floor)

RSI Design Decision:
    RSI(14) must be strictly between 50 and 70.
    - Above 50  → momentum has crossed back to the bullish side (confirmed recovery)
    - Below 70  → stock has not yet become overbought (room left to run)
    This band captures the sweet spot: recovery is real but entry is not too late.
    RSI peaking near 60 is treated as the ideal entry point in the scoring model.

Scoring (all components normalised to 0–10 before weighting):
    Trend Strength    25%  — % gap between SMA50 and SMA200
    Relative Strength 30%  — how much stock beats NIFTY 50 over 60 days
    Pullback Quality  20%  — shallower drawdown scores higher
    Volume Surge      15%  — stronger volume scores higher
    RSI Momentum      10%  — peaks at RSI 60, tapers toward 50 and 70

Score range: 0–10. Higher = stronger signal.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from indicators.relative_strength import relative_strength_vs_benchmark
from strategies.base import StrategySignal
from strategies.regime import get_market_regime
from utils.helpers import (
    avg_traded_value_cr,
    drawdown_from_high,
    metrics_to_dict,
)

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

STRATEGY_ID = 5
STRATEGY_NAME = "Strong Trend Pullback"

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

# Minimum candle history before evaluation is meaningful.
# 200 for SMA200 + buffer for stability.
MIN_CANDLES_REQUIRED = 210

# ---------------------------------------------------------------------------
# Condition thresholds
# ---------------------------------------------------------------------------

RS_60_MIN           = 10.0   # stock must outperform NIFTY 50 by at least 10% over 60 days
DRAWDOWN_MIN        = 5.0    # pullback must be at least 5% from the 50-day high
DRAWDOWN_MAX        = 12.0   # pullback must not exceed 12% (deeper = possible breakdown)
RSI_MIN             = 50.0   # RSI must have crossed back above 50 (recovery confirmed)
RSI_MAX             = 70.0   # RSI must not be overbought yet (room left to run)
RSI_IDEAL           = 60.0   # RSI closest to this value scores highest in RSI component
AVG_TRADED_VALUE_MIN = 10.0  # ₹10 crore minimum average daily traded value (liquidity)

# ---------------------------------------------------------------------------
# Normalisation reference ranges
# These define the realistic min→max for each scoring factor across NIFTY 500.
# Values at or above the "max" receive a perfect 10; at or below "min" receive 0.
# ---------------------------------------------------------------------------

TREND_STRENGTH_MIN  = 0.0    # SMA50 and SMA200 at same level
TREND_STRENGTH_MAX  = 20.0   # 20%+ gap between SMA50 and SMA200

RS_60_SCORE_MIN     = RS_60_MIN          # 10% outperformance → score 0
RS_60_SCORE_MAX     = RS_60_MIN + 40.0   # 50%+ outperformance → score 10

PULLBACK_SCORE_MIN  = 0.0                            # deepest pullback (12%) → score 0
PULLBACK_SCORE_MAX  = DRAWDOWN_MAX - DRAWDOWN_MIN    # shallowest pullback (5%) → score 10

VOLUME_RATIO_MIN    = 1.0    # barely above average → score 0
VOLUME_RATIO_MAX    = 4.0    # 4× average or more → score 10

RSI_SCORE_RANGE     = RSI_IDEAL - RSI_MIN  # distance from 50 to the ideal 60

# ---------------------------------------------------------------------------
# Score weights — must sum to 1.0
# ---------------------------------------------------------------------------

WEIGHT_TREND    = 0.25
WEIGHT_RS       = 0.30
WEIGHT_PULLBACK = 0.20
WEIGHT_VOLUME   = 0.15
WEIGHT_RSI      = 0.10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise(value: float, low: float, high: float) -> float:
    """
    Clamp and linearly scale `value` to the range 0–10.

    Returns 0.0 if value <= low, 10.0 if value >= high,
    and a proportional value in between.
    """
    if high <= low:
        return 0.0
    return max(0.0, min(10.0, (value - low) / (high - low) * 10.0))


def _rsi_score(rsi: float) -> float:
    """
    Score RSI on a 0–10 tent (triangle) function peaking at RSI_IDEAL (60).

    RSI = 50  → score 0   (just crossed the threshold, weakest)
    RSI = 60  → score 10  (ideal momentum recovery point)
    RSI = 70  → score 0   (approaching overbought, weakest within range)

    Values outside [RSI_MIN, RSI_MAX] are not passed here (already filtered).
    """
    if rsi <= RSI_IDEAL:
        # Rising side: 50 → 60
        return _normalise(rsi, RSI_MIN, RSI_IDEAL)
    else:
        # Falling side: 60 → 70 (inverted so 70 scores 0)
        return _normalise(RSI_IDEAL - (rsi - RSI_IDEAL), RSI_MIN, RSI_IDEAL)


def _compute_score(
    trend_strength: float,
    rs_60: float,
    drawdown: float,
    volume_ratio: float,
    rsi_val: float,
) -> float:
    """
    Compute a normalised 0–10 composite signal score.

    Every component is first normalised to 0–10 on its own scale,
    then the weighted sum is taken. This ensures the declared weights
    (25 / 30 / 20 / 15 / 10) are meaningful regardless of each
    factor's raw numeric range.

    Parameters
    ----------
    trend_strength : float
        Percentage gap between SMA50 and SMA200, e.g. 8.5 means SMA50
        is 8.5% above SMA200.
    rs_60 : float
        Relative strength of the stock vs NIFTY 50 over 60 trading days,
        in percentage-point outperformance.
    drawdown : float
        Current drawdown from the 50-day rolling high, as a positive %.
    volume_ratio : float
        Today's volume divided by the 20-day average volume.
    rsi_val : float
        RSI(14) value, expected in (RSI_MIN, RSI_MAX) after filtering.

    Returns
    -------
    float
        Composite score in the range [0.0, 10.0], rounded to 2 dp.
    """
    # ── Trend strength ──────────────────────────────────────────────────────
    # Higher % gap between SMA50 and SMA200 = stronger uptrend = higher score
    trend_score = _normalise(trend_strength, TREND_STRENGTH_MIN, TREND_STRENGTH_MAX)

    # ── Relative strength ───────────────────────────────────────────────────
    # Higher outperformance vs NIFTY 50 = stronger stock = higher score
    rs_score = _normalise(rs_60, RS_60_SCORE_MIN, RS_60_SCORE_MAX)

    # ── Pullback quality ────────────────────────────────────────────────────
    # Shallower pullback = healthier trend = higher score.
    # We invert drawdown: use (MAX - drawdown) so small dd → large value → high score.
    pullback_score = _normalise(
        DRAWDOWN_MAX - drawdown,
        PULLBACK_SCORE_MIN,
        PULLBACK_SCORE_MAX,
    )

    # ── Volume surge ────────────────────────────────────────────────────────
    # Higher volume relative to average = stronger buying interest = higher score
    volume_score = _normalise(volume_ratio, VOLUME_RATIO_MIN, VOLUME_RATIO_MAX)

    # ── RSI momentum ────────────────────────────────────────────────────────
    # Tent function peaking at RSI 60: confirmed recovery but not yet overbought
    rsi_score = _rsi_score(rsi_val)

    # ── Weighted composite ──────────────────────────────────────────────────
    score = (
        trend_score    * WEIGHT_TREND
        + rs_score     * WEIGHT_RS
        + pullback_score * WEIGHT_PULLBACK
        + volume_score * WEIGHT_VOLUME
        + rsi_score    * WEIGHT_RSI
    )

    return round(score, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    symbol: str,
    candles: pd.DataFrame,
    evaluation_index: int,
    context: dict[str, Any] | None = None,
) -> StrategySignal | None:
    """
    Evaluate Strategy 5 – Strong Trend Pullback for one stock at one candle.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "RELIANCE".
    candles : pd.DataFrame
        Daily OHLCV data with pre-computed indicator columns:
            trade_date, open, high, low, close, volume,
            sma50, sma200, ema20, rsi14, avg_vol_20.
        Rows must be sorted chronologically (oldest first).
    evaluation_index : int
        iloc position of the candle to evaluate.
        Pass -1 to evaluate the most recent candle.
    context : dict, optional
        Must contain:
            "nifty50_candles" (pd.DataFrame) — NIFTY 50 daily candles
            aligned to the same date range as `candles`, with columns
            including close and sma200.

    Returns
    -------
    StrategySignal
        Populated signal object if all conditions pass.
    None
        If any condition fails, any required value is missing,
        or the regime gate blocks evaluation.
    """
    context = context or {}

    # ── Resolve negative index ───────────────────────────────────────────────
    if evaluation_index < 0:
        evaluation_index = len(candles) + evaluation_index

    # ── Guard: sufficient history ────────────────────────────────────────────
    if evaluation_index < MIN_CANDLES_REQUIRED or len(candles) <= evaluation_index:
        return None

    row = candles.iloc[evaluation_index]

    # ── Guard: all required indicator columns must be present and non-NaN ───
    sma50      = row.get("sma50")
    sma200     = row.get("sma200")
    ema20      = row.get("ema20")
    rsi_val    = row.get("rsi14")
    avg_vol_20 = row.get("avg_vol_20")

    if any(pd.isna(v) for v in [sma50, sma200, ema20, rsi_val, avg_vol_20]):
        return None

    # =========================================================================
    # MARKET REGIME (aligned to stock evaluation date — does not block signals)
    # =========================================================================
    market_uptrend, nifty_sma200 = get_market_regime(
        candles, context.get("nifty50_candles"), evaluation_index
    )
    if nifty_sma200 is None:
        return None

    # =========================================================================
    # CONDITION 1 — Full Trend Alignment
    # Golden cross confirmed and price respects both moving averages.
    # =========================================================================
    if not (
        sma50 > sma200              # golden cross: medium-term above long-term
        and row["close"] > sma200   # price above long-term trend
        and row["close"] > ema20    # price above short-term trend
    ):
        return None

    # =========================================================================
    # CONDITION 2 — Relative Strength vs NIFTY 50 (60-day)
    # Stock must be a market leader, not just moving with the index.
    # =========================================================================
    rs_60 = relative_strength_vs_benchmark(
        candles,
        context.get("nifty50_candles"),
        evaluation_index,
        60,
    )
    if pd.isna(rs_60) or rs_60 <= RS_60_MIN:
        return None

    # =========================================================================
    # CONDITION 3 — Healthy Pullback Depth (5%–12% from 50-day high)
    # Too shallow (<5%) = not a real pullback entry opportunity.
    # Too deep (>12%)   = potential trend breakdown, not a dip.
    # =========================================================================
    drawdown = drawdown_from_high(candles, 50, evaluation_index)
    if pd.isna(drawdown) or not (DRAWDOWN_MIN <= drawdown <= DRAWDOWN_MAX):
        return None

    # =========================================================================
    # CONDITION 4 — RSI Recovery Band: 50 < RSI(14) < 70
    # Above 50 → buyers have retaken momentum (confirmed recovery, not a guess).
    # Below 70 → stock is not yet overbought (room left to run higher).
    # RSI closest to 60 scores best in the RSI scoring component.
    # =========================================================================
    if not (RSI_MIN < rsi_val < RSI_MAX):
        return None

    # =========================================================================
    # CONDITION 5 — Volume Confirmation
    # Today's volume must exceed the 20-day average — buying interest is real.
    # =========================================================================
    volume_ratio = row["volume"] / avg_vol_20
    if volume_ratio <= 1.0:
        return None

    # =========================================================================
    # CONDITION 6 — Bullish Candle Body
    # Close must be above Open — the day itself is net positive.
    # =========================================================================
    if row["close"] <= row["open"]:
        return None

    # =========================================================================
    # CONDITION 7 — Liquidity Floor
    # Average daily traded value (Close × Volume) over last 20 days > ₹10 Cr.
    # Prevents signals on thinly traded stocks that are hard to enter/exit.
    # =========================================================================
    atv = avg_traded_value_cr(candles, 20, evaluation_index)
    if pd.isna(atv) or atv < AVG_TRADED_VALUE_MIN:
        return None

    # =========================================================================
    # SCORING
    # All conditions passed. Compute a 0–10 composite score.
    # =========================================================================
    trend_strength = ((sma50 / sma200) - 1.0) * 100.0  # % gap: SMA50 vs SMA200

    score = _compute_score(
        trend_strength=trend_strength,
        rs_60=rs_60,
        drawdown=drawdown,
        volume_ratio=volume_ratio,
        rsi_val=rsi_val,
    )

    # =========================================================================
    # SIGNAL
    # =========================================================================
    return StrategySignal(
        strategy_id=STRATEGY_ID,
        strategy_name=STRATEGY_NAME,
        symbol=symbol,
        signal_date=row["trade_date"],
        score=score,
        trigger_price=float(row["close"]),
        suggested_action="" if market_uptrend else "Caution — market below SMA200",
        metrics=metrics_to_dict(
            {
                "rsi14":               round(float(rsi_val), 2),
                "drawdown_pct":        round(float(drawdown), 2),
                "rs_60":               round(float(rs_60), 2),
                "volume_ratio":        round(float(volume_ratio), 2),
                "trend_strength_pct":  round(float(trend_strength), 2),
                "avg_traded_value_cr": round(float(atv), 2),
                "sma50":               round(float(sma50), 2),
                "sma200":              round(float(sma200), 2),
                "ema20":               round(float(ema20), 2),
                "nifty_sma200":        round(float(nifty_sma200), 2),
                "market_uptrend":      market_uptrend,
            }
        ),
    )
