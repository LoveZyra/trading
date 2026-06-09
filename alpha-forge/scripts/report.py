"""Turn a BacktestResult into a chart + markdown report.

Produces an equity-curve / drawdown PNG and a metrics table. Matplotlib only; no
seaborn. Designed so Claude can drop the outputs straight into a deliverable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import metrics as M


def plot_result(result, benchmark=None, *, title: str = "Backtest",
                path: str | Path = "backtest.png"):
    """Two-panel chart: equity curves (strategy vs benchmark) + drawdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1],
                                   sharex=True)
    ax1.plot(result.equity.index, result.equity.values, label="strategy", lw=1.6)
    if benchmark is not None:
        ax1.plot(benchmark.equity.index, benchmark.equity.values,
                 label="buy & hold", lw=1.2, alpha=0.7)
    ax1.set_title(title)
    ax1.set_ylabel("growth of $1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    dd = M.drawdown_series(result.equity)
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def markdown_report(result, *, name: str = "Strategy", benchmark=None,
                    params: dict | None = None) -> str:
    """Markdown block: params, metrics, and a strategy-vs-benchmark comparison."""
    lines = [f"# {name} — Backtest Report", ""]
    if params:
        lines.append("**Parameters:** " + ", ".join(f"`{k}={v}`" for k, v in params.items()))
        lines.append("")

    s = result.stats
    rows = [
        ("Total return", f"{s['total_return']:+.2%}"),
        ("CAGR", f"{s['cagr']:+.2%}"),
        ("Ann. volatility", f"{s['ann_volatility']:.2%}"),
        ("Sharpe", f"{s['sharpe']:.2f}"),
        ("Sortino", f"{s['sortino']:.2f}"),
        ("Max drawdown", f"{s['max_drawdown']:.2%}"),
        ("Calmar", f"{s['calmar']:.2f}"),
        ("Win rate", f"{s['win_rate']:.2%}"),
        ("Profit factor", f"{s['profit_factor']:.2f}"),
        ("Trades", s.get("n_trades", "—")),
        ("Exposure", f"{s.get('exposure', float('nan')):.2%}"),
    ]
    lines.append("| Metric | Strategy |" + (" Buy & Hold |" if benchmark else ""))
    lines.append("|---|---|" + ("---|" if benchmark else ""))
    bench = benchmark.stats if benchmark else {}
    bmap = {
        "Total return": f"{bench.get('total_return', 0):+.2%}",
        "CAGR": f"{bench.get('cagr', 0):+.2%}",
        "Ann. volatility": f"{bench.get('ann_volatility', 0):.2%}",
        "Sharpe": f"{bench.get('sharpe', 0):.2f}",
        "Sortino": f"{bench.get('sortino', 0):.2f}",
        "Max drawdown": f"{bench.get('max_drawdown', 0):.2%}",
        "Calmar": f"{bench.get('calmar', 0):.2f}",
    }
    for label, val in rows:
        bcell = f" {bmap.get(label, '—')} |" if benchmark else ""
        lines.append(f"| {label} | {val} |" + bcell)
    return "\n".join(lines)
