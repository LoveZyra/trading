"""Factor lab — implement, VALIDATE, and backtest a custom factor.

This is the landing pad for RD-Agent's "extract a factor from a research report /
financial report / paper" idea (their fin_factor_report). The extraction itself is
LLM work — YOU (Claude) read the document and translate the described factor into a
function. This module gives you the safety rails and the test harness so the factor
you implement is causal and actually has predictive content before it ever enters a
strategy. See references/factor_extraction.md for the end-to-end workflow.

A "factor function" has signature:  f(df: OHLCV DataFrame) -> pd.Series
aligned to df.index. It must use ONLY past/current bars (causal). The validator below
mechanically checks that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest as bt
from . import indicators as ind  # noqa: F401  (handy for factor authors)

CUSTOM_FACTORS: dict = {}


def register_custom_factor(name: str, func):
    """Add a factor function to the registry so the CLI / autoresearch can use it."""
    CUSTOM_FACTORS[name] = func
    return func


@dataclass
class FactorCheck:
    causal: bool
    nan_ratio: float
    coverage: int
    messages: list

    @property
    def ok(self) -> bool:
        return self.causal and self.nan_ratio < 0.9

    def __repr__(self):
        tag = "OK" if self.ok else "PROBLEM"
        return (f"FactorCheck[{tag}] causal={self.causal} nan_ratio={self.nan_ratio:.2%} "
                f"coverage={self.coverage}\n  " + "\n  ".join(self.messages))


def validate_factor(func, df: pd.DataFrame, *, tol: float = 1e-8) -> FactorCheck:
    """Mechanically check a factor is causal (no look-ahead) and well-formed.

    Causality test: compute the factor on the full history, and again on the history
    truncated to the first K bars. A causal factor at position i<K uses only bars
    <=i, all present in the truncated series, so trunc[i] MUST equal full[i] wherever
    full[i] is non-NaN -- including being non-NaN there. If full has a value but trunc
    is NaN (it needed a bar >=K) or the values differ, the factor peeked at the
    future. (We must NOT dropna, since look-ahead shows up exactly at the boundary
    positions a dropna would discard.)

    We probe several cut points (50%, 70%, 90% of length), not one. A factor that only
    peeks a few bars ahead, or that conditionally looks ahead late in the series, can
    slip past a single cut at 60% but gets caught at one of the others — the boundary
    is where look-ahead shows up, so testing several boundaries is much harder to fool.
    """
    msgs = []
    full = func(df)
    if not isinstance(full, pd.Series):
        return FactorCheck(False, 1.0, 0, ["factor must return a pd.Series"])
    full = full.reindex(df.index)

    n = len(df)
    cut_points = sorted({max(30, int(n * f)) for f in (0.5, 0.7, 0.9)})
    causal = True
    checked_any = False
    worst_leak, worst_diff, worst_K = 0, 0.0, None
    for K in cut_points:
        if K >= n:        # need at least one future bar removed for the test to bite
            continue
        trunc = func(df.iloc[:K]).reindex(df.index[:K])
        overlap = full.iloc[:K]
        mask = overlap.notna()
        if mask.sum() == 0:
            continue
        checked_any = True
        leaked_to_nan = int((mask & trunc.reindex(overlap.index).isna()).sum())
        diffs = (overlap[mask] - trunc.reindex(overlap.index)[mask]).abs()
        max_diff = float(diffs.max()) if len(diffs.dropna()) else 0.0
        if leaked_to_nan or max_diff > tol:
            causal = False
            if leaked_to_nan >= worst_leak and max_diff >= worst_diff:
                worst_leak, worst_diff, worst_K = leaked_to_nan, max_diff, K
    if not checked_any:
        msgs.append("could not establish overlap for causality check (too few values)")
    elif not causal:
        why = []
        if worst_leak:
            why.append(f"{worst_leak} past values became undefined when future bars "
                       f"were removed (factor needs future data)")
        if worst_diff > tol:
            why.append(f"past values shifted by up to {worst_diff:.2e}")
        msgs.append(f"NOT CAUSAL (cut@{worst_K}): " + "; ".join(why) +
                    " -> the factor is looking ahead. Fix it.")
    else:
        msgs.append(f"causality check passed at {len(cut_points)} cut points "
                    "(past values stable when future appended)")

    nan_ratio = float(full.isna().mean())
    coverage = int(full.notna().sum())
    if nan_ratio > 0.5:
        msgs.append(f"high NaN ratio {nan_ratio:.0%} — factor is sparse; check warm-up/inputs")
    return FactorCheck(causal=causal, nan_ratio=nan_ratio, coverage=coverage, messages=msgs)


def factor_to_signal(factor: pd.Series, *, lookback: int = 60, mode: str = "momentum",
                     clip: float = 2.0) -> pd.Series:
    """Standardize a raw factor into a target position in [-1, 1].

    mode='momentum'  -> high factor = long  (trend/quality/positive-alpha factors)
    mode='reversion' -> high factor = short (overbought/expensive factors)
    Uses a rolling z-score so the factor's own scale/drift doesn't matter.
    """
    z = (factor - factor.rolling(lookback).mean()) / factor.rolling(lookback).std(ddof=0)
    z = z.clip(-clip, clip) / clip
    return (-z if mode == "reversion" else z).fillna(0.0)


def backtest_custom_factor(func, df: pd.DataFrame, *, mode: str = "momentum",
                           lookback: int = 60, commission_bps: float = 1.0,
                           slippage_bps: float = 1.0, validate: bool = True):
    """Validate (optional) then backtest a single-asset factor as a continuous signal.

    Returns (BacktestResult, FactorCheck|None). Refuses to backtest a non-causal
    factor — a look-ahead factor's backtest is meaningless and dangerous.
    """
    check = validate_factor(func, df) if validate else None
    if check is not None and not check.causal:
        raise ValueError(f"factor failed causality check — refusing to backtest.\n{check}")
    factor = func(df)
    sig = factor_to_signal(factor, lookback=lookback, mode=mode)
    res = bt.backtest(df, sig, commission_bps=commission_bps, slippage_bps=slippage_bps)
    return res, check


def factor_ic(func, df: pd.DataFrame, horizon: int = 21) -> float:
    """Single-asset information coefficient: correlation of the factor with the
    forward `horizon`-bar return. A quick read on whether the factor has any edge
    before you bother backtesting. |IC| > ~0.03 is already interesting on real data.
    """
    factor = func(df)
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    both = pd.concat([factor, fwd], axis=1).dropna()
    if len(both) < 10:
        return float("nan")
    return float(both.corr().iloc[0, 1])


def factor_scorecard(func, df: pd.DataFrame, *, horizon: int = 21,
                     existing: dict | None = None) -> dict:
    """Score a factor on the dimensions that matter — not just one backtest Sharpe.

    Inspired by AlphaEval (2508.13174) and AlphaAgent's decay/diversity controls
    (2502.16789). A factor can have a great backtest and still be useless: unstable,
    redundant with what you already have, or so high-turnover it can't be traded.

      ic            : mean information coefficient vs forward return (predictive power)
      ic_ir         : IC information ratio = mean(rolling IC)/std(rolling IC) — STABILITY
      autocorr      : lag-1 autocorrelation of the factor — smoothness / turnover proxy
      coverage      : fraction of bars with a value
      max_corr_existing : largest |correlation| to factors in `existing` (DIVERSITY)
    `existing`: {name: factor_func} of factors you already use, for the diversity check.
    Returns a dict plus a short 'verdict'.
    """
    f = func(df)
    ic = factor_ic(func, df, horizon)
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    win = max(63, horizon * 3)
    roll = f.rolling(win).corr(fwd)
    ic_ir = float(roll.mean() / roll.std(ddof=0)) if roll.std(ddof=0) else float("nan")
    autocorr = float(f.autocorr(lag=1)) if f.notna().sum() > 3 else float("nan")
    coverage = float(f.notna().mean())
    max_corr = 0.0
    if existing:
        for nm, g in existing.items():
            try:
                both = pd.concat([f, g(df)], axis=1).dropna()
                if len(both) > 10:
                    max_corr = max(max_corr, abs(both.corr().iloc[0, 1]))
            except Exception:  # noqa: BLE001
                continue
    verdict = []
    if abs(ic) < 0.02:
        verdict.append("weak predictive power")
    if np.isfinite(ic_ir) and abs(ic_ir) < 0.3:
        verdict.append("unstable IC")
    if np.isfinite(autocorr) and autocorr < 0.5:
        verdict.append("high turnover")
    if max_corr > 0.7:
        verdict.append(f"redundant (corr {max_corr:.2f} with existing)")
    return {"ic": round(ic, 4), "ic_ir": round(ic_ir, 3) if np.isfinite(ic_ir) else None,
            "autocorr": round(autocorr, 3) if np.isfinite(autocorr) else None,
            "coverage": round(coverage, 3), "max_corr_existing": round(max_corr, 3),
            "verdict": "; ".join(verdict) if verdict else "looks promising"}
