"""Strategy 4 – Volatility Compression Near Highs.

What this strategy looks for:
    A stock that is coiling quietly near its 52-week high, with volatility
    and volume both contracting. This is the classic "spring loaded" setup —
    energy is building for an expansion move, and being near the highs means
    the path of least resistance is upward when the move comes.

Conditions:
    - NIFTY 50 regime gate         (market must be above its 200-day SMA)
    - Close > SMA200               (stock in long-term uptrend)
    - ATR(20) < ATR(60)            (short-term volatility contracting vs recent avg)
    - ATR percentile ≤ 20%         (current ATR in lowest quintile of last 250 days)
    - avg_vol_10 < avg_vol_30      (volume contracting — quiet accumulation)
    - Price within 5% of 250-day high  (coiling near the top, not mid-range)
    - 50 < RSI(14) < 70            (momentum is positive but not overbought)
    - Avg traded value > ₹10 Cr    (liquidity floor)

Scoring (all components normalised to 0–10 before weighting):
    ATR Compression   40%  — lower ATR percentile = more compressed = higher score
    Proximity to High 30%  — closer to 250-day high = higher score
    Volume Contraction 20% — larger gap between vol_10 and vol_30 = higher score
    RSI Momentum      10%  — peaks at RSI 60, tapers toward 50 and 70

Score range: 0–10. Higher = tighter compression = stronger pre-breakout setup.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import atr_percentile, previous_high
from strategies.base import StrategySignal
from strategies.regime import get_market_regime
from utils.helpers import avg_traded_value_cr, metrics_to_dict

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

STRATEGY_ID = 4
STRATEGY_NAME = "Volatility Compression"

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

MIN_CANDLES_REQUIRED = 250

# ---------------------------------------------------------------------------
# Condition thresholds
# ---------------------------------------------------------------------------

ATR_PERCENTILE_MAX      = 20.0   # ATR must be in the lowest 20% of last 250 days
DISTANCE_FROM_HIGH_MAX  = 5.0    # price must be within 5% of the 250-day high
RSI_MIN                 = 50.0   # RSI must be above 50 (momentum positive)
RSI_MAX                 = 70.0   # RSI must be below 70 (not overbought)
RSI_IDEAL               = 60.0   # RSI closest to 60 scores highest
AVG_TRADED_VALUE_MIN    = 10.0   # ₹10 crore minimum average daily traded value

# ---------------------------------------------------------------------------
# Normalisation reference ranges
# ---------------------------------------------------------------------------

# ATR compression: lower percentile = more compressed = higher score
# Percentile 0 → score 10 (maximally compressed)
# Percentile 20 → score 0 (just at the threshold)
ATR_SCORE_LOW  = 0.0
ATR_SCORE_HIGH = ATR_PERCENTILE_MAX   # 20.0

# Proximity: closer to high = higher score
# Distance 0% → score 10; Distance 5% → score 0
PROXIMITY_SCORE_LOW  = 0.0
PROXIMITY_SCORE_HIGH = DISTANCE_FROM_HIGH_MAX   # 5.0

# Volume contraction: how much 10-day avg has fallen below 30-day avg
# Ratio = avg_vol_10 / avg_vol_30; lower ratio = stronger contraction
# Ratio 1.0 (no contraction) → score 0
# Ratio 0.4 (60% contraction) → score 10
VOL_RATIO_SCORE_LOW  = 0.4
VOL_RATIO_SCORE_HIGH = 1.0

# Score weights — must sum to 1.0
WEIGHT_ATR_COMPRESSION  = 0.40
WEIGHT_PROXIMITY        = 0.30
WEIGHT_VOLUME           = 0.20
WEIGHT_RSI              = 0.10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalise(value: float, low: float, high: float) -> float:
    """Clamp and linearly scale `value` to 0–10 between low and high."""
    if high <= low:
        return 0.0
    return max(0.0, min(10.0, (value - low) / (high - low) * 10.0))


def _rsi_score(rsi: float) -> float:
    """
    Tent function peaking at RSI_IDEAL (60).

    RSI 50 → score 0  (just above threshold)
    RSI 60 → score 10 (ideal momentum recovery)
    RSI 70 → score 0  (approaching overbought)
    """
    if rsi <= RSI_IDEAL:
        return _normalise(rsi, RSI_MIN, RSI_IDEAL)
    else:
        return _normalise(RSI_IDEAL - (rsi - RSI_IDEAL), RSI_MIN, RSI_IDEAL)


def _compute_score(
    atr_percentile_val: float,
    distance_pct: float,
    vol_ratio: float,
    rsi_val: float,
) -> float:
    """
    Compute a normalised 0–10 composite signal score.

    All components are individually normalised to 0–10 before weighting
    so no single factor can dominate due to its raw numeric scale.

    Parameters
    ----------
    atr_percentile_val : float
        ATR percentile rank over last 250 days (0–20 after filtering).
        Lower = more compressed = better.
    distance_pct : float
        % distance of close from 250-day high (0–5 after filtering).
        Lower = closer to high = better.
    vol_ratio : float
        avg_vol_10 / avg_vol_30. Lower = more volume contraction = better.
    rsi_val : float
        RSI(14). Tent function peaks at 60.
    """
    # ATR compression: invert so lower percentile → higher score
    atr_score = _normalise(
        ATR_PERCENTILE_MAX - atr_percentile_val,
        ATR_SCORE_LOW,
        ATR_SCORE_HIGH,
    )

    # Proximity to high: invert so smaller distance → higher score
    proximity_score = _normalise(
        PROXIMITY_SCORE_HIGH - distance_pct,
        PROXIMITY_SCORE_LOW,
        PROXIMITY_SCORE_HIGH,
    )

    # Volume contraction: invert so lower ratio → higher score
    volume_score = _normalise(
        VOL_RATIO_SCORE_HIGH - vol_ratio,
        VOL_RATIO_SCORE_LOW,
        VOL_RATIO_SCORE_HIGH,
    )

    # RSI: tent function peaking at 60
    rsi_score = _rsi_score(rsi_val)

    score = (
        atr_score      * WEIGHT_ATR_COMPRESSION
        + proximity_score * WEIGHT_PROXIMITY
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
    Evaluate Strategy 4 – Volatility Compression Near Highs.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "INFY".
    candles : pd.DataFrame
        Daily OHLCV data with pre-computed indicator columns:
            trade_date, open, high, low, close, volume,
            sma200, rsi14, atr_pct, avg_vol_10, avg_vol_30.
        Rows must be sorted chronologically (oldest first).
    evaluation_index : int
        iloc position of the candle to evaluate.
        Pass -1 to evaluate the most recent candle.
    context : dict, optional
        Must contain:
            "nifty50_candles" (pd.DataFrame) — NIFTY 50 daily candles
            aligned to the same date range, with columns: close, sma200.

    Returns
    -------
    StrategySignal if all conditions pass, None otherwise.
    """
    context = context or {}

    # ── Resolve negative index ───────────────────────────────────────────────
    if evaluation_index < 0:
        evaluation_index = len(candles) + evaluation_index

    # ── Guard: sufficient history ────────────────────────────────────────────
    if evaluation_index < MIN_CANDLES_REQUIRED or len(candles) <= evaluation_index:
        return None

    row = candles.iloc[evaluation_index]

    # =========================================================================
    # MARKET REGIME (aligned to stock evaluation date — does not block signals)
    # =========================================================================
    market_uptrend, nifty_sma200 = get_market_regime(
        candles, context.get("nifty50_candles"), evaluation_index
    )
    if nifty_sma200 is None:
        return None

    # =========================================================================
    # CONDITION 1 — Long-Term Trend Filter
    # Close must be above the 200-day SMA — stock is in a long-term uptrend.
    # =========================================================================
    sma200 = row.get("sma200")
    if pd.isna(sma200) or row["close"] <= sma200:
        return None

    # =========================================================================
    # CONDITION 2 — ATR Compression (Short-Term vs Medium-Term)
    # 20-day average ATR% must be below 60-day average ATR%.
    # This confirms volatility is actively contracting, not just low.
    # =========================================================================
    if "atr_pct" not in candles.columns:
        return None

    atr_pct_series = candles["atr_pct"]

    # Slice with bounds checking
    atr_slice_20 = atr_pct_series.iloc[
        max(0, evaluation_index - 19) : evaluation_index + 1
    ]
    atr_slice_60 = atr_pct_series.iloc[
        max(0, evaluation_index - 59) : evaluation_index + 1
    ]

    atr_20 = atr_slice_20.mean()
    atr_60 = atr_slice_60.mean()

    if pd.isna(atr_20) or pd.isna(atr_60) or atr_20 >= atr_60:
        return None

    # =========================================================================
    # CONDITION 3 — ATR Percentile in Lowest 20%
    # Current ATR must be historically compressed relative to the last 250 days.
    # =========================================================================
    percentile = atr_percentile(candles, evaluation_index, 250)
    if pd.isna(percentile) or percentile > ATR_PERCENTILE_MAX:
        return None

    # =========================================================================
    # CONDITION 4 — Volume Contraction
    # 10-day average volume must be below 30-day average volume.
    # Confirms quiet accumulation — low participation = coiling, not selling.
    # =========================================================================
    avg_vol_10 = row.get("avg_vol_10")
    avg_vol_30 = row.get("avg_vol_30")

    if pd.isna(avg_vol_10) or pd.isna(avg_vol_30) or avg_vol_10 >= avg_vol_30:
        return None

    vol_ratio = avg_vol_10 / avg_vol_30   # < 1.0 after the filter above

    # =========================================================================
    # CONDITION 5 — Price Proximity to 250-Day High
    # Stock must be within 5% of its 52-week high.
    # Compression near lows is a distribution setup; near highs is accumulation.
    # =========================================================================
    prev_high_250 = previous_high(candles, evaluation_index, 250)
    if pd.isna(prev_high_250) or prev_high_250 <= 0:
        return None

    distance_pct = abs(row["close"] / prev_high_250 - 1.0) * 100.0
    if distance_pct > DISTANCE_FROM_HIGH_MAX:
        return None

    # =========================================================================
    # CONDITION 6 — RSI Band: 50 < RSI(14) < 70
    # Positive momentum (above 50) but not yet overbought (below 70).
    # Prevents entering a compressing stock that is quietly breaking down.
    # =========================================================================
    rsi_val = row.get("rsi14")
    if pd.isna(rsi_val) or not (RSI_MIN < rsi_val < RSI_MAX):
        return None

    # =========================================================================
    # CONDITION 7 — Liquidity Floor
    # Average daily traded value (Close × Volume) over 20 days > ₹10 Cr.
    # =========================================================================
    atv = avg_traded_value_cr(candles, 20, evaluation_index)
    if pd.isna(atv) or atv < AVG_TRADED_VALUE_MIN:
        return None

    # =========================================================================
    # SCORING — all components normalised to 0–10 before weighting
    # =========================================================================
    score = _compute_score(
        atr_percentile_val=float(percentile),
        distance_pct=float(distance_pct),
        vol_ratio=float(vol_ratio),
        rsi_val=float(rsi_val),
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
                "atr_percentile":          round(float(percentile), 2),
                "atr_20_pct":              round(float(atr_20), 4),
                "atr_60_pct":              round(float(atr_60), 4),
                "distance_from_high_pct":  round(float(distance_pct), 2),
                "vol_ratio_10_30":         round(float(vol_ratio), 3),
                "rsi14":                   round(float(rsi_val), 2),
                "avg_traded_value_cr":     round(float(atv), 2),
                "sma200":                  round(float(sma200), 2),
                "nifty_sma200":            round(float(nifty_sma200), 2),
                "market_uptrend":          market_uptrend,
            }
        ),
    )
