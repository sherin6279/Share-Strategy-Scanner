"""Strategy 6 – Relative Strength Leaders.

What this strategy looks for:
    Stocks that are genuine market leaders — consistently outperforming
    the NIFTY 500 universe over both medium (90-day) and long (180-day)
    timeframes, with trend alignment and healthy momentum. These are the
    stocks institutions are accumulating.

Conditions:
    - NIFTY 50 regime gate             (market must be above its 200-day SMA)
    - Top 10% by 90-day return rank    (strong recent momentum)
    - Top 20% by 180-day return rank   (sustained leadership, not a spike)
    - Close > SMA50 > SMA200           (full trend alignment)
    - Close > EMA20                    (short-term trend intact, not in pullback)
    - 55 ≤ RSI(14) ≤ 75               (momentum healthy, not overbought)
    - ATR(14) / Close < 6%             (excludes excessively volatile "leaders")
    - Avg traded value > ₹20 Cr        (institutional-grade liquidity)

Scoring (all components normalised to 0–10 before weighting):
    90-day RS Rank   40%  — higher percentile rank = stronger recent leader
    180-day RS Rank  35%  — higher percentile rank = more sustained leadership
    RSI Momentum     15%  — peaks at RSI 65, tapers toward 55 and 75
    Liquidity        10%  — higher traded value = more institutionally accessible

Score range: 0–10. Higher = stronger and more sustained market leader.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import MIN_TRADED_VALUE_CR_S6
from strategies.base import StrategySignal
from strategies.regime import get_market_regime
from utils.helpers import avg_traded_value_cr, metrics_to_dict

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

STRATEGY_ID = 6
STRATEGY_NAME = "RS Leaders"

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

MIN_CANDLES_REQUIRED = 185   # 180-day RS + small buffer

# ---------------------------------------------------------------------------
# Condition thresholds
# ---------------------------------------------------------------------------

RS_90_MIN_PERCENTILE  = 90.0   # must be in top 10% by 90-day return
RS_180_MIN_PERCENTILE = 80.0   # must be in top 20% by 180-day return
RSI_MIN               = 55.0   # RSI must be above 55 (healthy upward momentum)
RSI_MAX               = 75.0   # RSI must be below 75 (not overbought)
RSI_IDEAL             = 65.0   # RSI closest to 65 scores highest
ATR_PCT_MAX           = 6.0    # ATR% / Close must be below 6% (volatility cap)

# Liquidity threshold — pulled from config (same as original)
# MIN_TRADED_VALUE_CR_S6 is defined in config.settings

# ---------------------------------------------------------------------------
# Normalisation reference ranges
# ---------------------------------------------------------------------------

# RS rank percentiles: all qualifying stocks are in [90–100] for 90d
# and [80–100] for 180d. Normalise within those ranges so top stocks
# score 10 and boundary stocks score 0.
RS_90_SCORE_LOW  = RS_90_MIN_PERCENTILE    # 90.0 → score 0
RS_90_SCORE_HIGH = 100.0                   # 100.0 → score 10

RS_180_SCORE_LOW  = RS_180_MIN_PERCENTILE  # 80.0 → score 0
RS_180_SCORE_HIGH = 100.0                  # 100.0 → score 10

# Liquidity: normalise between ₹20 Cr (minimum) and ₹200 Cr (institutional)
LIQUIDITY_SCORE_LOW  = float(MIN_TRADED_VALUE_CR_S6)   # threshold → score 0
LIQUIDITY_SCORE_HIGH = 200.0                            # ₹200 Cr+ → score 10

# Score weights — must sum to 1.0
WEIGHT_RS_90    = 0.40
WEIGHT_RS_180   = 0.35
WEIGHT_RSI      = 0.15
WEIGHT_LIQUIDITY = 0.10


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
    Tent function peaking at RSI_IDEAL (65).

    RSI 55 → score 0  (just above threshold, weakest)
    RSI 65 → score 10 (ideal momentum for a leader)
    RSI 75 → score 0  (approaching overbought, weakest within range)
    """
    if rsi <= RSI_IDEAL:
        # Rising side: 55 → 65
        return _normalise(rsi, RSI_MIN, RSI_IDEAL)
    else:
        # Falling side: 65 → 75 (inverted so 75 scores 0)
        return _normalise(RSI_IDEAL - (rsi - RSI_IDEAL), RSI_MIN, RSI_IDEAL)


def _compute_score(
    ret_90_pct: float,
    ret_180_pct: float,
    rsi_val: float,
    traded_value_cr: float,
) -> float:
    """
    Compute a normalised 0–10 composite signal score.

    Every component is normalised to 0–10 on its own scale before weighting,
    so the declared weights (40 / 35 / 15 / 10) are meaningful regardless of
    each factor's raw numeric range.

    Parameters
    ----------
    ret_90_pct : float
        Percentile rank by 90-day return across NIFTY 500 (90–100 after filter).
    ret_180_pct : float
        Percentile rank by 180-day return across NIFTY 500 (80–100 after filter).
    rsi_val : float
        RSI(14). Tent function peaks at RSI_IDEAL (65).
    traded_value_cr : float
        Average daily traded value in ₹ crore over last 20 days.
    """
    rs_90_score    = _normalise(ret_90_pct, RS_90_SCORE_LOW, RS_90_SCORE_HIGH)
    rs_180_score   = _normalise(ret_180_pct, RS_180_SCORE_LOW, RS_180_SCORE_HIGH)
    rsi_score      = _rsi_score(rsi_val)
    liquidity_score = _normalise(traded_value_cr, LIQUIDITY_SCORE_LOW, LIQUIDITY_SCORE_HIGH)

    score = (
        rs_90_score     * WEIGHT_RS_90
        + rs_180_score  * WEIGHT_RS_180
        + rsi_score     * WEIGHT_RSI
        + liquidity_score * WEIGHT_LIQUIDITY
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
    Evaluate Strategy 6 – Relative Strength Leaders.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "HDFCBANK".
    candles : pd.DataFrame
        Daily OHLCV data with pre-computed indicator columns:
            trade_date, open, high, low, close, volume,
            sma50, sma200, ema20, rsi14, atr_pct.
        Rows must be sorted chronologically (oldest first).
    evaluation_index : int
        iloc position of the candle to evaluate.
        Pass -1 to evaluate the most recent candle.
    context : dict, optional
        Must contain:
            "nifty50_candles" (pd.DataFrame) — NIFTY 50 daily candles
                with columns: close, sma200.
            "rs_rankings" (dict) — pre-computed percentile rankings:
                {
                    "ret_90_pctile":  {symbol: percentile, ...},
                    "ret_180_pctile": {symbol: percentile, ...},
                }
                Percentile 99 = top 1%, 90 = top 10%, etc.

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

    market_uptrend, nifty_sma200 = get_market_regime(
        candles, context.get("nifty50_candles"), evaluation_index
    )
    if nifty_sma200 is None:
        return None

    # =========================================================================
    # CONDITION 1 — Relative Strength Rankings
    # Must be a true market leader on both medium and long timeframes.
    # Top 10% by 90-day return = strong recent momentum.
    # Top 20% by 180-day return = sustained leadership over 9 months.
    # The combination eliminates one-month wonders and slow grinders alike.
    # =========================================================================
    rankings     = context.get("rs_rankings", {})
    ret_90_pct   = rankings.get("ret_90_pctile", {}).get(symbol)
    ret_180_pct  = rankings.get("ret_180_pctile", {}).get(symbol)

    if ret_90_pct is None or ret_180_pct is None:
        return None

    if ret_90_pct < RS_90_MIN_PERCENTILE or ret_180_pct < RS_180_MIN_PERCENTILE:
        return None

    # =========================================================================
    # Read the current candle row
    # =========================================================================
    row    = candles.iloc[evaluation_index]
    sma50  = row.get("sma50")
    sma200 = row.get("sma200")
    ema20  = row.get("ema20")
    rsi_val = row.get("rsi14")

    if any(pd.isna(v) for v in [sma50, sma200, ema20, rsi_val]):
        return None

    # =========================================================================
    # CONDITION 2 — Full Trend Alignment
    # Close > SMA50 > SMA200: all three timeframes in bullish order.
    # This ensures we are not buying a leader that is in a phase transition.
    # =========================================================================
    if not (row["close"] > sma50 > sma200):
        return None

    # =========================================================================
    # CONDITION 3 — Short-Term Trend Intact
    # Close > EMA20: stock is not in a short-term pullback.
    # A leader below its EMA20 may be about to lose leadership status.
    # =========================================================================
    if row["close"] <= ema20:
        return None

    # =========================================================================
    # CONDITION 4 — RSI Band: 55 ≤ RSI(14) ≤ 75
    # Healthy upward momentum (≥55) but not yet overbought (≤75).
    # RSI closest to 65 scores highest in the RSI scoring component.
    # =========================================================================
    if not (RSI_MIN <= rsi_val <= RSI_MAX):
        return None

    # =========================================================================
    # CONDITION 5 — Volatility Cap
    # ATR% must be below 6%. Very high-RS stocks are sometimes high-RS
    # because they are extremely volatile — that inflates returns without
    # representing genuine institutional accumulation.
    # =========================================================================
    atr_pct = row.get("atr_pct")
    if pd.isna(atr_pct) or atr_pct >= ATR_PCT_MAX:
        return None

    # =========================================================================
    # CONDITION 6 — Institutional Liquidity Floor
    # Average daily traded value (Close × Volume) over 20 days > ₹20 Cr.
    # Leaders that cannot be traded at scale are not true institutional leaders.
    # =========================================================================
    traded_value_cr = avg_traded_value_cr(candles, 20, evaluation_index)
    if pd.isna(traded_value_cr) or traded_value_cr <= MIN_TRADED_VALUE_CR_S6:
        return None

    # =========================================================================
    # SCORING — all components normalised to 0–10 before weighting
    # =========================================================================
    score = _compute_score(
        ret_90_pct=float(ret_90_pct),
        ret_180_pct=float(ret_180_pct),
        rsi_val=float(rsi_val),
        traded_value_cr=float(traded_value_cr),
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
                "ret_90_percentile":   round(float(ret_90_pct), 2),
                "ret_180_percentile":  round(float(ret_180_pct), 2),
                "rsi14":               round(float(rsi_val), 2),
                "atr_pct":             round(float(atr_pct), 4),
                "traded_value_cr":     round(float(traded_value_cr), 2),
                "sma50":               round(float(sma50), 2),
                "sma200":              round(float(sma200), 2),
                "ema20":               round(float(ema20), 2),
                "nifty_sma200":        round(float(nifty_sma200), 2),
                "market_uptrend":      market_uptrend,
            }
        ),
    )
