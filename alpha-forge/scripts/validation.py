"""Backtest-overfitting statistics — deflate the Sharpe you found by searching.

The single biggest lie in quant research: you try N strategies/parameters, report the
best Sharpe, and forget that the *maximum* of N noisy Sharpes is inflated even if none
has real edge. This module quantifies and removes that selection bias, following
Bailey & López de Prado:

  * probabilistic_sharpe_ratio (PSR): P(true Sharpe > benchmark) given track length,
    skew and kurtosis — fat tails and negative skew make a Sharpe less trustworthy.
  * deflated_sharpe_ratio (DSR): PSR with the benchmark raised to the level you'd
    expect from the BEST of N independent trials. If your grid had 200 configs, DSR
    asks "is the winner better than the best of 200 coin-flips?".
  * pbo_cscv: Probability of Backtest Overfitting via Combinatorially-Symmetric
    Cross-Validation — across many train/test splits, how often does the in-sample
    best strategy underperform the median out-of-sample? >0.5 means your selection
    process is overfitting.

These turn the auto-research leaderboard from "trust the top Sharpe" into "trust the
top Sharpe *after* accounting for how hard you looked". See references/pitfalls.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # scipy is optional; fall back to a numpy normal CDF
    from scipy.stats import norm
    _ncdf = norm.cdf
except Exception:  # noqa: BLE001
    def _ncdf(x):
        return 0.5 * (1.0 + np.vectorize(_erf)(np.asarray(x, float) / np.sqrt(2.0)))

    def _erf(x):
        # Abramowitz-Stegun 7.1.26 approximation
        t = 1.0 / (1.0 + 0.3275911 * abs(x))
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
        return np.sign(x) * y


def _moments(returns: pd.Series):
    r = np.asarray(returns.dropna(), float)
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else 0.0
    if sd == 0 or n < 3:
        return n, 0.0, 0.0, 3.0
    sr = r.mean() / sd                      # per-period Sharpe (not annualized)
    skew = ((r - r.mean()) ** 3).mean() / sd ** 3
    kurt = ((r - r.mean()) ** 4).mean() / sd ** 4
    return n, sr, skew, kurt


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark). Accounts for sample length and the
    non-normality (skew/kurtosis) of returns. Closer to 1 = more confident the Sharpe
    is real. `sr_benchmark` is a *per-period* Sharpe (use 0 for 'better than nothing').
    """
    n, sr, skew, kurt = _moments(returns)
    if n < 3:
        return float("nan")
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    if denom == 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(_ncdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """Expected maximum of N independent N(0, sr_std^2) Sharpes — the benchmark a
    winning strategy must clear to be more than luck. Uses the standard extreme-value
    approximation (Euler-Mascheroni constant)."""
    if n_trials < 2:
        return 0.0
    e = 0.5772156649
    z1 = _quantile(1 - 1.0 / n_trials)
    z2 = _quantile(1 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1 - e) * z1 + e * z2))


def _quantile(p):
    """Inverse normal CDF (Acklam's rational approximation) — no scipy needed."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          sr_std: float | None = None,
                          all_trial_returns: pd.DataFrame | None = None) -> float:
    """Deflated Sharpe Ratio: PSR against the benchmark you'd expect from the best of
    `n_trials`. A DSR near 1 means the strategy survives the multiple-testing haircut;
    near 0.5 or below means the 'winner' is likely a fluke of searching.

    sr_std: spread of per-period Sharpes across the trials. If you pass
    `all_trial_returns` (a DataFrame, one column of returns per trial), it's estimated
    for you; otherwise defaults to 1.0 (conservative).
    """
    if sr_std is None:
        if all_trial_returns is not None and all_trial_returns.shape[1] > 1:
            srs = []
            for c in all_trial_returns.columns:
                _, sr, _, _ = _moments(all_trial_returns[c])
                srs.append(sr)
            sr_std = float(np.std(srs, ddof=1)) or 1.0
        else:
            sr_std = 1.0
    bench = expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=bench)


def pbo_cscv(trial_returns: pd.DataFrame, n_splits: int = 10) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    trial_returns: DataFrame, one column per strategy/config, rows = aligned per-period
    returns. Splits time into `n_splits` blocks, forms train/test as complementary
    halves of the blocks (all balanced combinations), picks the in-sample best by
    Sharpe, and records its out-of-sample *rank*. PBO = fraction of splits where the
    IS-best lands in the bottom half OOS. PBO > 0.5 == your selection overfits.
    """
    import itertools

    M = trial_returns.dropna(how="any")
    if M.shape[1] < 2 or len(M) < n_splits * 2:
        return {"pbo": float("nan"), "n_combos": 0, "note": "insufficient data/strategies"}
    blocks = np.array_split(np.arange(len(M)), n_splits)
    half = n_splits // 2
    logits = []
    for combo in itertools.combinations(range(n_splits), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_sr = M.iloc[is_idx].mean() / M.iloc[is_idx].std(ddof=0).replace(0, np.nan)
        oos_sr = M.iloc[oos_idx].mean() / M.iloc[oos_idx].std(ddof=0).replace(0, np.nan)
        best = is_sr.idxmax()
        # rank of the IS-best among OOS performances (1 = best)
        rank = oos_sr.rank(ascending=False)[best]
        w = rank / (len(oos_sr) + 1)          # relative rank in (0,1)
        lam = np.log(w / (1 - w)) if 0 < w < 1 else 0.0
        logits.append(lam)
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())          # <=0 => IS-best below OOS median
    return {"pbo": pbo, "n_combos": len(logits),
            "interpretation": ("overfitting likely" if pbo > 0.5 else "selection looks robust")}
