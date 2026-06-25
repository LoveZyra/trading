"""Parameter search and OUT-OF-SAMPLE validation.

The whole point of this module is to fight overfitting -- the disease that makes a
backtest look brilliant and then lose money live. Two tools:

1. `grid_search` -- brute-force a parameter grid on ONE dataset. Useful, but on its
   own it is exactly how people overfit: pick the best of 500 combos and you've
   curve-fit to noise. Treat its winner with suspicion.

2. `walk_forward` -- the honest test. Repeatedly optimize on an in-sample window,
   then measure performance on the *next, unseen* window, and stitch those
   out-of-sample pieces into one equity curve. If walk-forward results collapse
   versus the in-sample grid search, the strategy is overfit. Trust walk-forward.

Both work with any Strategy subclass and a dict of param ranges.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import backtest
from . import metrics as M


# Metrics where SMALLER is better. Everything else is larger-is-better -- including
# max_drawdown, which is stored negative (so -0.05 already sorts above -0.30). Without
# this, passing metric="ann_volatility" would silently select the MOST volatile params.
_LOWER_IS_BETTER = {"ann_volatility"}


def _param_combos(grid: dict[str, list]):
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        yield dict(zip(keys, vals))


def grid_search(StrategyCls, df: pd.DataFrame, grid: dict[str, list],
                *, metric: str = "sharpe", lag: int = 1,
                commission_bps: float = 1.0, slippage_bps: float = 1.0,
                **bt_kwargs) -> pd.DataFrame:
    """Evaluate every parameter combination; return a table sorted by `metric`.

    metric: any key from metrics.summary -- 'sharpe', 'sortino', 'calmar', 'cagr'...
    Higher is better for almost everything (drawdowns are negative, so -5% sorts above
    -30%); the rare smaller-is-better metric (e.g. 'ann_volatility') is handled via
    _LOWER_IS_BETTER so the table still puts the best config first.
    """
    rows = []
    for params in _param_combos(grid):
        try:
            strat = StrategyCls(**params)
            sig = strat.generate_signal(df)
            res = backtest(df, sig, lag=lag, commission_bps=commission_bps,
                           slippage_bps=slippage_bps, **bt_kwargs)
            row = {**params, **res.stats}
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            rows.append({**params, "error": str(e)})
    table = pd.DataFrame(rows)
    if metric in table.columns:
        table = table.sort_values(metric, ascending=(metric in _LOWER_IS_BETTER),
                                  na_position="last").reset_index(drop=True)
    return table


@dataclass
class WalkForwardResult:
    oos_equity: pd.Series          # stitched out-of-sample equity curve
    oos_stats: dict                # metrics on the OOS curve -- the number that matters
    folds: pd.DataFrame            # chosen params + IS/OOS metric per fold
    is_stats: dict                 # metrics if you'd traded the IS-best in-sample

    def __repr__(self):
        return ("WalkForwardResult(OOS:\n" + M.format_summary(self.oos_stats) + "\n)")


def walk_forward(StrategyCls, df: pd.DataFrame, grid: dict[str, list],
                 *, n_splits: int = 5, train_frac: float = 0.6,
                 metric: str = "sharpe", lag: int = 1,
                 commission_bps: float = 1.0, slippage_bps: float = 1.0,
                 anchored: bool = False) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward optimization.

    The series is cut into `n_splits` sequential test windows. Before each test
    window, the best params are chosen by `metric` on the preceding train window
    only, then applied to the (unseen) test window. The test-window returns are
    concatenated into one out-of-sample curve.

    anchored=False : rolling window (train window slides).
    anchored=True  : expanding window (train always starts at the beginning).
    """
    n = len(df)
    fold_size = n // (n_splits + 1)
    if fold_size < 30:
        raise ValueError("not enough data for this many splits")

    oos_returns = []
    fold_rows = []

    for k in range(1, n_splits + 1):
        test_start = fold_size * k
        test_end = min(fold_size * (k + 1), n)
        # train window sized so train/(train+test) == train_frac (the old
        # fold_size/(1-train_frac) made train_frac=0.6 behave like 71%).
        train_len = int(fold_size * train_frac / max(1e-9, 1 - train_frac))
        train_start = 0 if anchored else max(0, test_start - train_len)
        train = df.iloc[train_start:test_start]
        test = df.iloc[test_start:test_end]
        if len(train) < 30 or len(test) < 5:
            continue

        # Optimize on train only. Compare on a signed score so smaller-is-better
        # metrics (ann_volatility) select correctly; keep the raw value for display.
        flip = -1.0 if metric in _LOWER_IS_BETTER else 1.0
        best_params, best_cmp, best_raw = None, -np.inf, np.nan
        for params in _param_combos(grid):
            try:
                sig = StrategyCls(**params).generate_signal(train)
                stats = backtest(train, sig, lag=lag, commission_bps=commission_bps,
                                 slippage_bps=slippage_bps).stats
                raw = stats.get(metric, np.nan)
                if not np.isfinite(raw):
                    continue
                cmp = flip * raw
                if cmp > best_cmp:
                    best_cmp, best_raw, best_params = cmp, raw, params
            except Exception:  # noqa: BLE001
                continue
        if best_params is None:
            continue
        best_score = best_raw

        # Apply to the unseen test window. Generate the signal on train+test so
        # indicators have warm-up history, then slice to the test window.
        ctx = df.iloc[train_start:test_end]
        sig_full = StrategyCls(**best_params).generate_signal(ctx)
        res = backtest(ctx, sig_full, lag=lag, commission_bps=commission_bps,
                       slippage_bps=slippage_bps)
        oos_seg = res.returns.reindex(test.index).fillna(0.0)
        oos_returns.append(oos_seg)

        fold_rows.append({
            "fold": k, **best_params,
            f"train_{metric}": round(best_score, 3),
            f"test_{metric}": round(res.stats.get(metric, np.nan), 3),
            "test_return": round(M.total_return(oos_seg), 4),
        })

    if not oos_returns:
        raise ValueError("walk-forward produced no out-of-sample segments")

    oos = pd.concat(oos_returns).sort_index()
    oos = oos[~oos.index.duplicated(keep="last")]
    oos_equity = (1 + oos).cumprod()
    return WalkForwardResult(
        oos_equity=oos_equity,
        oos_stats=M.summary(oos, equity=oos_equity),
        folds=pd.DataFrame(fold_rows),
        is_stats={},
    )
