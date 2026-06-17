"""Strategy 7 – Stage 2 Breakout from Stage 1 Base.

Naming clarification (Weinstein Stage Analysis):
    Stage 1 = Basing / Accumulation — stock near its lows, moving sideways,
              volume low and contracting. This is the PRE-breakout phase.
    Stage 2 = Markup / Breakout — stock breaks above the Stage 1 base,
              volume surges, SMA50 turns up and price clears near-term highs.

    This strategy detects the TRANSITION from Stage 1 to Stage 2:
    the stock recently consolidated near its 250-day low (Stage 1 evidence),
    and is now breaking out above its 30-day high with rising volume and a
    turning SMA50 (Stage 2 confirmation). The correct name is Stage 2 Breakout.

What this strategy looks for:
    Early-stage breakouts where:
        1. The stock spent time near its 52-week low (base formation)
        2. SMA50 has turned upward (trend changing from down to up)
        3. Price has broken above the 30-day high (near-term resistance cleared)
        4. Volume has surged to confirm conviction
        5. RSI has recovered into bullish territory
    This is a high-reward setup when caught early in the new uptrend.

Conditions:
    - NIFTY 50 regime gate              (market must be above its 200-day SMA)
    - Within last 90 days, at least one candle traded within 5% of 250-day low
      (evidence of Stage 1 base formation)
    - Close > SMA50                     (price has crossed above medium-term average)
    - SMA50 slope is positive            (average itself is turning upward)
    - Close > 30-day high (prior day)   (near-term resistance cleared)
    - Volume > 1.5× 20-day avg volume   (breakout confirmed by participation)
    - 50 < RSI(14) < 70                 (momentum turned bullish, not overbought)
    - Avg traded value > ₹10 Cr         (liquidity floor)

Scoring (all components normalised to 0–10 before weighting):
    Breakout Strength  40%  — how far above the 30-day high the close is
    Volume Surge       30%  — how much volume exceeds the 20-day average
    RSI Momentum       20%  — tent function peaking at RSI 60
    Base Depth         10%  — how far below the 250-day low the base was
                              (deeper base = more significant breakout)

Score range: 0–10. Higher = stronger and more convincing Stage 2 breakout.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import previous_high, previous_low
from indicators.moving_averages import sma_slope_positive
from strategies.base import StrategySignal
from strategies.regime import get_market_regime
from utils.helpers import avg_traded_value_cr, metrics_to_dict

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

STRATEGY_ID   = 7
STRATEGY_NAME = "Stage 2 Breakout"

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

MIN_CANDLES_REQUIRED = 255   # 250 for prev_low + buffer

# ---------------------------------------------------------------------------
# Condition thresholds
# ---------------------------------------------------------------------------

BASE_LOOKBACK_DAYS    = 90    # how far back to look for Stage 1 base evidence
BASE_PROXIMITY_PCT    = 5.0   # how close to 250-day low counts as "base" (%)
BREAKOUT_HIGH_DAYS    = 30    # break above the 30-day prior high
VOLUME_MIN_RATIO      = 1.5   # volume must be at least 1.5× the 20-day average
RSI_MIN               = 50.0  # RSI must be above 50 (bullish momentum confirmed)
RSI_MAX               = 70.0  # RSI must be below 70 (not overbought)
RSI_IDEAL             = 60.0  # RSI closest to 60 scores highest
AVG_TRADED_VALUE_MIN  = 10.0  # ₹10 crore minimum average daily traded value

# ---------------------------------------------------------------------------
# Normalisation reference ranges
# ---------------------------------------------------------------------------

# Breakout strength: % above 30-day high
# 0% (just at the high) → score 0; 5%+ above high → score 10
BREAKOUT_SCORE_LOW  = 0.0
BREAKOUT_SCORE_HIGH = 5.0

# Volume ratio: today's volume / 20-day avg
# 1.5× (minimum) → score 0; 5×+ → score 10
VOLUME_SCORE_LOW  = VOLUME_MIN_RATIO   # 1.5
VOLUME_SCORE_HIGH = 5.0

# Base depth: how deep below 250-day low the base went
# Stored as the % below low that the base candle traded
# 0% → score 0 (barely at the low); 10%+ → score 10 (deep base)
BASE_DEPTH_SCORE_LOW  = 0.0
BASE_DEPTH_SCORE_HIGH = 10.0

# Score weights — must sum to 1.0
WEIGHT_BREAKOUT = 0.40
WEIGHT_VOLUME   = 0.30
WEIGHT_RSI      = 0.20
WEIGHT_BASE     = 0.10


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

    RSI 50 → score 0  (just crossed bullish threshold)
    RSI 60 → score 10 (ideal momentum for early-stage breakout)
    RSI 70 → score 0  (approaching overbought territory)
    """
    if rsi <= RSI_IDEAL:
        return _normalise(rsi, RSI_MIN, RSI_IDEAL)
    else:
        return _normalise(RSI_IDEAL - (rsi - RSI_IDEAL), RSI_MIN, RSI_IDEAL)


def _base_depth_score(
    candles: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    ref_low: float,
) -> float:
    """
    Compute how deep into the base the stock traded during the base period.

    Finds the minimum low in the base window and computes how far below
    the 250-day low it traded (as a %). A deeper base = more significant
    accumulation = higher base depth score.

    Returns a normalised 0–10 score.
    """
    if pd.isna(ref_low) or ref_low <= 0:
        return 0.0

    window_lows = candles["low"].iloc[start_idx : end_idx + 1]
    if window_lows.empty:
        return 0.0

    min_low_in_base = float(window_lows.min())
    # How far below ref_low did it go? (positive number if it went below)
    depth_pct = max(0.0, (ref_low - min_low_in_base) / ref_low * 100.0)

    return _normalise(depth_pct, BASE_DEPTH_SCORE_LOW, BASE_DEPTH_SCORE_HIGH)


def _traded_near_low_in_window(
    candles: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    ref_low: float,
    pct: float = BASE_PROXIMITY_PCT,
) -> bool:
    """
    Return True if any candle in [start_idx, end_idx) traded within `pct`%
    above the reference low, confirming Stage 1 base formation.

    Note: end_idx is exclusive (does not include the current evaluation candle).
    """
    if pd.isna(ref_low) or ref_low <= 0:
        return False

    # Threshold: ref_low + pct% (anything at or below this counts as "near low")
    threshold = ref_low * (1.0 + pct / 100.0)
    window    = candles.iloc[start_idx:end_idx]   # exclusive end

    return bool((window["low"] <= threshold).any())


def _compute_score(
    breakout_pct: float,
    volume_ratio: float,
    rsi_val: float,
    base_depth_score_val: float,
) -> float:
    """
    Compute a normalised 0–10 composite signal score.

    All components are individually normalised to 0–10 before weighting
    so the declared weights are meaningful regardless of raw numeric scale.

    Parameters
    ----------
    breakout_pct : float
        % above the 30-day prior high (0%+ after filter).
    volume_ratio : float
        Today's volume / 20-day avg volume (≥1.5 after filter).
    rsi_val : float
        RSI(14) in (RSI_MIN, RSI_MAX) after filter.
    base_depth_score_val : float
        Already-normalised 0–10 base depth score from _base_depth_score().
    """
    breakout_score = _normalise(breakout_pct, BREAKOUT_SCORE_LOW, BREAKOUT_SCORE_HIGH)
    volume_score   = _normalise(volume_ratio, VOLUME_SCORE_LOW, VOLUME_SCORE_HIGH)
    rsi_score      = _rsi_score(rsi_val)

    # base_depth_score_val is already 0–10
    score = (
        breakout_score        * WEIGHT_BREAKOUT
        + volume_score        * WEIGHT_VOLUME
        + rsi_score           * WEIGHT_RSI
        + base_depth_score_val * WEIGHT_BASE
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
    Evaluate Strategy 7 – Stage 2 Breakout from Stage 1 Base.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "TATAMOTORS".
    candles : pd.DataFrame
        Daily OHLCV data with pre-computed indicator columns:
            trade_date, open, high, low, close, volume,
            sma50, rsi14, avg_vol_20.
        Rows must be sorted chronologically (oldest first).
    evaluation_index : int
        iloc position of the candle to evaluate.
        Pass -1 to evaluate the most recent candle.
    context : dict, optional
        Must contain:
            "nifty50_candles" (pd.DataFrame) — NIFTY 50 daily candles
                with columns: close, sma200.

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

    # ── Guard: required indicator columns must be present and non-NaN ────────
    sma50      = row.get("sma50")
    rsi_val    = row.get("rsi14")
    avg_vol_20 = row.get("avg_vol_20")

    if any(pd.isna(v) for v in [sma50, rsi_val, avg_vol_20]):
        return None

    market_uptrend, nifty_sma200 = get_market_regime(
        candles, context.get("nifty50_candles"), evaluation_index
    )
    if nifty_sma200 is None:
        return None

    # =========================================================================
    # CONDITION 1 — Stage 1 Base Evidence
    # At some point in the last 90 trading days (excluding today),
    # the stock must have traded within 5% of its 250-day low.
    # This confirms a genuine basing period, not a mid-range breakout.
    #
    # NOTE: previous_low uses candles up to but not including evaluation_index
    # to avoid look-ahead bias (today's low cannot define yesterday's 250-day low).
    # =========================================================================
    prev_low_250 = previous_low(candles, evaluation_index, 250)
    if pd.isna(prev_low_250) or prev_low_250 <= 0:
        return None

    base_start = max(0, evaluation_index - BASE_LOOKBACK_DAYS)
    base_end   = evaluation_index   # exclusive — does not include today

    if not _traded_near_low_in_window(
        candles, base_start, base_end, prev_low_250
    ):
        return None

    # Pre-compute base depth score while we have the base window
    base_d_score = _base_depth_score(candles, base_start, base_end, prev_low_250)

    # =========================================================================
    # CONDITION 2 — Price Above SMA50
    # The stock has crossed above its medium-term moving average —
    # a key indicator that the downtrend / base is ending.
    # =========================================================================
    if row["close"] <= sma50:
        return None

    # =========================================================================
    # CONDITION 3 — SMA50 Slope is Positive (Turning Upward)
    # The average itself must be rising, not just flat.
    # A rising SMA50 = the trend is changing, not just a one-day spike.
    # =========================================================================
    if not sma_slope_positive(candles, 50, evaluation_index):
        return None

    # =========================================================================
    # CONDITION 4 — Breakout Above 30-Day High
    # Close must exceed the highest High of the previous 30 trading days
    # (excluding today, to avoid look-ahead bias).
    # This is the near-term resistance being cleared.
    # =========================================================================
    prev_high_30 = previous_high(candles, evaluation_index, BREAKOUT_HIGH_DAYS)
    if pd.isna(prev_high_30) or prev_high_30 <= 0:
        return None

    if row["close"] <= prev_high_30:
        return None

    breakout_pct = (row["close"] / prev_high_30 - 1.0) * 100.0

    # =========================================================================
    # CONDITION 5 — Volume Surge
    # Today's volume must be at least 1.5× the 20-day average.
    # A breakout without volume is a false breakout.
    # =========================================================================
    volume_ratio = row["volume"] / avg_vol_20
    if volume_ratio < VOLUME_MIN_RATIO:
        return None

    # =========================================================================
    # CONDITION 6 — RSI Band: 50 < RSI(14) < 70
    # RSI above 50 confirms the momentum shift from bearish to bullish.
    # RSI below 70 ensures we are not entering an already-extended move.
    # =========================================================================
    if not (RSI_MIN < rsi_val < RSI_MAX):
        return None

    # =========================================================================
    # CONDITION 7 — Liquidity Floor
    # Average daily traded value (Close × Volume) over 20 days > ₹10 Cr.
    # Early-stage breakouts in illiquid stocks are very hard to exit.
    # =========================================================================
    atv = avg_traded_value_cr(candles, 20, evaluation_index)
    if pd.isna(atv) or atv < AVG_TRADED_VALUE_MIN:
        return None

    # =========================================================================
    # SCORING — all components normalised to 0–10 before weighting
    # =========================================================================
    score = _compute_score(
        breakout_pct=float(breakout_pct),
        volume_ratio=float(volume_ratio),
        rsi_val=float(rsi_val),
        base_depth_score_val=float(base_d_score),
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
                "breakout_pct_30d":    round(float(breakout_pct), 2),
                "volume_ratio":        round(float(volume_ratio), 2),
                "rsi14":               round(float(rsi_val), 2),
                "base_depth_score":    round(float(base_d_score), 2),
                "prev_low_250":        round(float(prev_low_250), 2),
                "prev_high_30":        round(float(prev_high_30), 2),
                "avg_traded_value_cr": round(float(atv), 2),
                "sma50":               round(float(sma50), 2),
                "nifty_sma200":        round(float(nifty_sma200), 2),
                "market_uptrend":      market_uptrend,
            }
        ),
    )
