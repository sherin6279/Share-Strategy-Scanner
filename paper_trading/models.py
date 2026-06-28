"""Paper trading data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class PaperTradeBatch:
    batch_id: str
    scan_run_id: str | None
    created_at: datetime
    entry_date: date
    position_count: int
    notes: str = ""


@dataclass
class PaperTradePosition:
    position_id: str
    batch_id: str
    symbol: str
    strategy_id: int
    entry_date: date
    entry_price: float
    score: float


@dataclass
class PaperTradePL:
    position_id: str
    symbol: str
    strategy_id: int
    entry_date: date
    entry_price: float
    current_date: date | None
    current_price: float | None
    days_held: int
    pl_amount: float | None
    pl_pct: float | None
    score: float
