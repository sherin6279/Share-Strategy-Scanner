"""CLI entry point for backtests."""

from __future__ import annotations

import argparse
from datetime import date

from backtest.equity_backtest import EquityBacktester
from backtest.fno_backtest import FnoBacktester
from backtest.metrics import summaries_to_dataframe
from database.duckdb_manager import DuckDBManager


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def run_equity_cli(args: argparse.Namespace) -> None:
    bt = EquityBacktester()
    result = bt.run(
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        hold_days=args.hold_days,
        step_days=args.step_days,
        strategy_ids=args.strategies,
        cost_bps=args.cost_bps,
        simulate=args.simulate,
        stop_pct=args.stop_pct,
        target_pct=args.target_pct,
    )

    print(f"\n=== Equity Backtest ({result.mode}) ===")
    print(f"Period: {result.start_date} → {result.end_date}")
    print(f"Hold: {result.hold_days} trading days | Cost: {result.cost_bps} bps")
    print(f"Total trades: {len(result.trades)}\n")

    summary_df = summaries_to_dataframe(result.summaries)
    if summary_df.empty:
        print("No trades generated.")
        return

    print(summary_df.to_string(index=False))

    if args.export:
        summary_df.to_csv(args.export, index=False)
        print(f"\nSummary exported to {args.export}")


def run_fno_cli(args: argparse.Namespace) -> None:
    bt = FnoBacktester()
    result = bt.run(
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        strategy_ids=args.strategies,
        cost_bps=args.cost_bps,
        stop_pct=args.stop_pct,
        target_pct=args.target_pct,
    )

    print(f"\n=== F&O Intraday Backtest ===")
    print(f"Period: {result.start_date} → {result.end_date}")
    print(f"Total trades: {len(result.trades)}\n")

    summary_df = summaries_to_dataframe(result.summaries)
    if summary_df.empty:
        print("No trades generated. Refresh F&O intraday data first.")
        return

    print(summary_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY 500 Screener Backtest")
    sub = parser.add_subparsers(dest="segment", required=True)

    eq = sub.add_parser("equity", help="Equity swing backtest")
    eq.add_argument("--start", help="Start date YYYY-MM-DD")
    eq.add_argument("--end", help="End date YYYY-MM-DD")
    eq.add_argument("--hold-days", type=int, default=5)
    eq.add_argument("--step-days", type=int, default=5, help="Days between signal dates")
    eq.add_argument("--strategies", type=int, nargs="*", help="Strategy IDs e.g. 1 4 6")
    eq.add_argument("--cost-bps", type=float, default=15.0)
    eq.add_argument("--simulate", action="store_true", help="Use stop/target simulation")
    eq.add_argument("--stop-pct", type=float, default=5.0)
    eq.add_argument("--target-pct", type=float, default=10.0)
    eq.add_argument("--export", help="Export summary CSV path")
    eq.set_defaults(func=run_equity_cli)

    fno = sub.add_parser("fno", help="F&O intraday backtest")
    fno.add_argument("--start", help="Start date YYYY-MM-DD")
    fno.add_argument("--end", help="End date YYYY-MM-DD")
    fno.add_argument("--strategies", type=int, nargs="*", help="F&O strategy IDs 101-103")
    fno.add_argument("--cost-bps", type=float, default=5.0)
    fno.add_argument("--stop-pct", type=float, default=0.5)
    fno.add_argument("--target-pct", type=float, default=1.0)
    fno.set_defaults(func=run_fno_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
