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

    def cell(d, key, spec):
        """Format d[key] with `spec`, or '—' if the metric is missing/NaN. Keeps the
        report honest about absent metrics instead of crashing (KeyError) or printing
        'nan%' — matching the skill's render-if-present contract."""
        v = d.get(key)
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)

    rows = [
        ("Total return", cell(s, "total_return", "+.2%")),
        ("CAGR", cell(s, "cagr", "+.2%")),
        ("Ann. volatility", cell(s, "ann_volatility", ".2%")),
        ("Sharpe", cell(s, "sharpe", ".2f")),
        ("Sortino", cell(s, "sortino", ".2f")),
        ("Max drawdown", cell(s, "max_drawdown", ".2%")),
        ("Calmar", cell(s, "calmar", ".2f")),
        ("Win rate", cell(s, "win_rate", ".2%")),
        ("Profit factor", cell(s, "profit_factor", ".2f")),
        ("Trades", cell(s, "n_trades", ",d")),
        ("Exposure", cell(s, "exposure", ".2%")),
    ]
    lines.append("| Metric | Strategy |" + (" Buy & Hold |" if benchmark else ""))
    lines.append("|---|---|" + ("---|" if benchmark else ""))
    bench = benchmark.stats if benchmark else {}
    bmap = {
        "Total return": cell(bench, "total_return", "+.2%"),
        "CAGR": cell(bench, "cagr", "+.2%"),
        "Ann. volatility": cell(bench, "ann_volatility", ".2%"),
        "Sharpe": cell(bench, "sharpe", ".2f"),
        "Sortino": cell(bench, "sortino", ".2f"),
        "Max drawdown": cell(bench, "max_drawdown", ".2%"),
        "Calmar": cell(bench, "calmar", ".2f"),
    }
    for label, val in rows:
        bcell = f" {bmap.get(label, '—')} |" if benchmark else ""
        lines.append(f"| {label} | {val} |" + bcell)
    return "\n".join(lines)
