"""Tests for bar-safe intraday indicators."""

from datetime import date

from fno.intraday_context import IntradayDayContext
from tests.fixtures import make_intraday_day


def test_vwap_does_not_use_future_bars():
    day = make_intraday_day("NIFTY", date(2025, 6, 1), bars=20)
    ctx = IntradayDayContext(day)

    ctx_early = ctx.build_bar_context(5)
    ctx_late = ctx.build_bar_context(15)

    assert ctx_early["vwap"] != ctx_late["vwap"]
    assert ctx_early["vwap"] > 0
