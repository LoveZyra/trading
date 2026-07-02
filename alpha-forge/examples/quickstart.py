#!/usr/bin/env python3
"""Quickstart: the whole loop on one asset, end to end.

Run from the skill root:  python examples/quickstart.py
Uses free yfinance data if available; falls back to a synthetic series so it always
runs (e.g. offline). Demonstrates: load -> strategy -> backtest -> compare ->
walk-forward out-of-sample validation -> report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.core import backtest as bt, optimize as opt, metrics as M

from scripts.reporting import report as rpt
from scripts.data.base import validate_ohlcv
from scripts.strategies import MACrossover, ZScoreReversion


def get_data():
    """Try real data; fall back to a reproducible synthetic trend+noise series."""
    try:
        from scripts.data.loader import load
        df = load("AAPL", source="yfinance", period="5y", use_cache=False)
        print(f"Loaded real AAPL data: {len(df)} bars")
        return df, "AAPL"
    except Exception as e:  # noqa: BLE001
        print(f"(yfinance unavailable: {e}) -> using synthetic data")
        np.random.seed(42)
        n = 1000
        idx = pd.bdate_range("2021-01-01", periods=n)
        ret = np.random.normal(0.0004, 0.013, n) + 0.001 * np.sin(np.arange(n) / 50)
        close = 100 * np.exp(np.cumsum(ret))
        df = pd.DataFrame({"open": close * 0.999, "high": close * 1.012,
                           "low": close * 0.988, "close": close,
                           "volume": np.random.randint(1_000_000, 5_000_000, n)}, index=idx)
        return validate_ohlcv(df, name="synthetic"), "SYNTH"


def main():
    df, name = get_data()

    # --- 1. one in-sample backtest, compared to buy & hold ---
    strat = MACrossover(fast=20, slow=50)
    res = bt.backtest(df, strat.generate_signal(df), commission_bps=1, slippage_bps=1)
    bench = bt.buy_and_hold(df)

    print("\n=== In-sample MA(20,50) vs Buy & Hold ===")
    print(M.format_summary(res.stats))
    print(f"\nBuy & Hold Sharpe: {bench.stats['sharpe']:.2f}, "
          f"return: {bench.stats['total_return']:+.1%}")

    # --- 2. honest out-of-sample walk-forward ---
    print("\n=== Walk-forward (OUT-OF-SAMPLE) ===")
    wf = opt.walk_forward(MACrossover, df,
                          grid={"fast": [10, 20, 30], "slow": [50, 100, 150]},
                          n_splits=5, metric="sharpe")
    print(M.format_summary(wf.oos_stats))
    print("\nPer-fold (note train vs test gap = overfitting check):")
    print(wf.folds.to_string(index=False))

    # --- 3. save a chart + markdown report ---
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    rpt.plot_result(res, benchmark=bench, title=f"MA(20,50) — {name}",
                    path=out / "equity.png")
    (out / "report.md").write_text(
        rpt.markdown_report(res, name=f"MA crossover — {name}", benchmark=bench,
                            params={"fast": 20, "slow": 50}))
    print(f"\nSaved chart + report to {out}/")


if __name__ == "__main__":
    main()
