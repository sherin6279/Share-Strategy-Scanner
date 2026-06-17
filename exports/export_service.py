"""Export scan results to CSV and Excel."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd


def _flatten_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Expand metrics JSON column into separate columns."""
    if df.empty or "metrics" not in df.columns:
        return df

    out = df.copy()
    if out["metrics"].dtype == object:
        parsed = out["metrics"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else (x or {})
        )
        metrics_df = pd.json_normalize(parsed)
        out = pd.concat([out.drop(columns=["metrics"]), metrics_df], axis=1)
    return out


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for UI display."""
    if df.empty:
        return df
    display = _flatten_metrics(df)
    rename_map = {
        "symbol": "Symbol",
        "signal_date": "Signal Date",
        "score": "Score",
        "trigger_price": "Current Price",
        "scan_timestamp": "Scan Time",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})
    return display


def export_csv(df: pd.DataFrame) -> bytes:
    """Export dataframe to CSV bytes."""
    flat = _flatten_metrics(df)
    return flat.to_csv(index=False).encode("utf-8")


def export_excel(df: pd.DataFrame) -> bytes:
    """Export dataframe to Excel bytes."""
    flat = _flatten_metrics(df)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        flat.to_excel(writer, index=False, sheet_name="Scan Results")
    return buffer.getvalue()


def save_to_file(df: pd.DataFrame, path: Path, fmt: str = "csv") -> Path:
    """Save export to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "excel":
        path.write_bytes(export_excel(df))
    else:
        path.write_bytes(export_csv(df))
    return path
