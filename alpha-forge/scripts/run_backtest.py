#!/usr/bin/env python3
"""CLI: load data, run one strategy, print metrics, optionally chart + report.

Examples
--------
  # MA crossover on Apple, free yfinance data
  python run_backtest.py --symbol AAPL --strategy ma_crossover --fast 20 --slow 50

  # Mean reversion on a Shanghai A-share via akshare
  python run_backtest.py --symbol 600519 --source akshare --market cn \
      --strategy bollinger_reversion --start 2020-01-01

  # Walk-forward validated MA crossover (the honest, anti-overfit test)
  python run_backtest.py --symbol AAPL --strategy ma_crossover --walk-forward

  # Backtest from broker data Claude dumped to a JSON file
  python run_backtest.py --symbol broker_aapl.json --source ibkr --strategy ts_momentum

Run from the scripts/ directory (or with it on PYTHONPATH).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python run_backtest.py` from inside scripts/ as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import backtest as bt
from scripts import optimize as opt
from scripts import report as rpt
from scripts import metrics as M
from scripts.data.loader import load
from scripts.strategies import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quant backtest runner")
    p.add_argument("--symbol", required=True, help="ticker, A-share code, or broker JSON path (source=ibkr)")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "akshare", "pykrx", "ibkr"])
    p.add_argument("--market", default="cn", help="akshare market: cn|hk")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--period", default="2y", help="yfinance period if no --start")
    p.add_argument("--interval", default="1d")
    p.add_argument("--strategy", required=True, choices=sorted(REGISTRY))
    # generic strategy params (only the relevant ones are used per strategy)
    for name in ("fast", "slow", "lookback", "entry", "exit", "n", "signal"):
        p.add_argument(f"--{name}", type=float, default=None)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--commission-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--walk-forward", action="store_true", help="run out-of-sample walk-forward instead")
    p.add_argument("--metric", default="sharpe")
    p.add_argument("--out", default=None, help="directory for chart + markdown report")
    return p


def collect_params(args, StrategyCls) -> dict:
    """Pull only the constructor params the chosen strategy actually accepts."""
    import inspect
    valid = set(inspect.signature(StrategyCls.__init__).parameters) - {"self"}
    out = {}
    for k in ("fast", "slow", "lookback", "entry", "exit", "n", "signal"):
        v = getattr(args, k)
        if v is not None and k in valid:
            out[k] = int(v) if float(v).is_integer() and k not in ("entry", "exit") else v
    if "allow_short" in valid and args.allow_short:
        out["allow_short"] = True
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)

    load_kw = {}
    if args.source == "yfinance":
        load_kw = dict(start=args.start, end=args.end, period=args.period, interval=args.interval)
    elif args.source == "akshare":
        load_kw = dict(start=args.start, end=args.end, market=args.market)
    elif args.source == "pykrx":
        load_kw = dict(start=args.start, end=args.end)

    df = load(args.symbol, source=args.source, **load_kw)
    print(f"Loaded {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}\n")

    StrategyCls = REGISTRY[args.strategy]
    params = collect_params(args, StrategyCls)

    if args.walk_forward:
        # Sensible default grid around the chosen strategy's main knobs.
        grids = {
            "ma_crossover": {"fast": [10, 20, 30], "slow": [50, 100, 150]},
            "breakout": {"entry": [20, 40, 55], "exit": [10, 20]},
            "ts_momentum": {"lookback": [60, 90, 120, 180]},
            "macd_trend": {"fast": [8, 12], "slow": [21, 26], "signal": [9]},
            "zscore_reversion": {"lookback": [10, 20, 30], "entry": [1.0, 1.5, 2.0]},
            "bollinger_reversion": {"n": [10, 20, 30], "k": [1.5, 2.0, 2.5]},
            "rsi_reversion": {"n": [7, 14, 21], "oversold": [20, 30]},
        }
        grid = grids[args.strategy]
        res = opt.walk_forward(StrategyCls, df, grid, metric=args.metric,
                               commission_bps=args.commission_bps,
                               slippage_bps=args.slippage_bps)
        print("Walk-forward OUT-OF-SAMPLE performance (the number that matters):\n")
        print(M.format_summary(res.oos_stats))
        print("\nPer-fold chosen params:\n", res.folds.to_string(index=False))
        return

    strat = StrategyCls(**params)
    sig = strat.generate_signal(df)
    result = bt.backtest(df, sig, commission_bps=args.commission_bps,
                         slippage_bps=args.slippage_bps)
    bench = bt.buy_and_hold(df)

    print(f"Strategy: {strat}\n")
    print(rpt.markdown_report(result, name=strat.name, benchmark=bench, params=params))

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.symbol).stem  # safe for absolute paths / JSON files
        png = rpt.plot_result(result, benchmark=bench, title=f"{strat.name} — {stem}",
                              path=outdir / f"{stem}_{strat.name}.png")
        md = outdir / f"{stem}_{strat.name}.md"
        md.write_text(rpt.markdown_report(result, name=strat.name, benchmark=bench, params=params))
        print("Saved chart -> "+str(png)+"  | report -> "+str(md))


if __name__ == "__main__":
    main()
