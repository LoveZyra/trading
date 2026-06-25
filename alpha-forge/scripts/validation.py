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

TRADING_DAYS = 252            # default annualization for per-period (daily) Sharpe

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
    from them; otherwise it falls back to the analytic standard error of THIS track's own
    per-period Sharpe, sqrt((1 + sr^2/2)/(n-1)) (Lo 2002). A flat 1.0 is the wrong scale
    (a per-period Sharpe spread of 1.0 ~= 16 annualized) and pins DSR at ~0 for every
    realistic strategy -- pass the trial returns for the rigorous multiple-testing haircut.
    """
    if sr_std is None:
        if all_trial_returns is not None and all_trial_returns.shape[1] > 1:
            srs = [_moments(all_trial_returns[c])[1] for c in all_trial_returns.columns]
            sr_std = float(np.std(srs, ddof=1)) or None
        if sr_std is None:
            # No usable trial dispersion: use the sampling SE of a single per-period Sharpe
            # estimate as the per-trial spread proxy, instead of an arbitrary (too-large) 1.0.
            n, sr, _, _ = _moments(returns)
            sr_std = float(np.sqrt((1.0 + 0.5 * sr ** 2) / max(n - 1, 1))) if n > 2 else 1.0
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
        # OOS rank of the IS-best. ascending => rank 1 = WORST, n = BEST, matching
        # Lopez de Prado's relative rank omega so that a HIGH w means GOOD out-of-sample.
        rank = oos_sr.rank(ascending=True)[best]
        w = rank / (len(oos_sr) + 1)          # relative OOS rank in (0,1); higher = better OOS
        lam = np.log(w / (1 - w)) if 0 < w < 1 else 0.0
        logits.append(lam)
    logits = np.array(logits)
    # PBO = P(IS-best lands BELOW the OOS median) = P(logit < 0). A genuinely robust
    # selection (the IS-best also wins out-of-sample) has w -> 1, logit > 0, hence a LOW pbo.
    pbo = float((logits < 0).mean())
    return {"pbo": pbo, "n_combos": len(logits),
            "interpretation": ("overfitting likely" if pbo > 0.5 else "selection looks robust")}


# ----------------------------------------------------------------------------
# Combinatorial Purged CV — the OOS Sharpe DISTRIBUTION (not just one number)
# ----------------------------------------------------------------------------
def cpcv(returns: pd.Series, n_groups: int = 8, k_test: int = 2, *, embargo: int = 1,
         periods_per_year: int = TRADING_DAYS) -> dict:
    """Combinatorial Purged CV — distribution of OOS Sharpe for ONE strategy's returns.

    Split the per-period return series into `n_groups` contiguous blocks; for every way of
    choosing `k_test` blocks as the test set (C(n_groups, k_test) combinations), annualize
    the Sharpe on the concatenated test blocks. An `embargo` of a few bars is dropped at
    each block's leading edge to blunt autocorrelation leakage. A single OOS number hides
    whether an edge is consistent or driven by one lucky window — this shows the spread
    (median + 5/95% band + fraction of paths that are positive). López de Prado's CPCV;
    2024-25 evidence shows it gives the lowest PBO / best DSR of the CV schemes.
    """
    import itertools
    r = np.asarray(pd.Series(returns).dropna(), float)
    n = len(r)
    if n < n_groups * 3 or not (1 <= k_test < n_groups):
        return {"n_paths": 0, "median": float("nan"), "note": "insufficient data"}
    groups = np.array_split(np.arange(n), n_groups)
    ann = np.sqrt(periods_per_year)
    paths = []
    for combo in itertools.combinations(range(n_groups), k_test):
        idx = np.concatenate([groups[g][embargo:] if embargo and len(groups[g]) > embargo
                              else groups[g] for g in combo])
        seg = r[idx]
        sd = seg.std(ddof=1)
        if len(seg) >= 5 and sd > 0:
            paths.append(float(seg.mean() / sd * ann))
    if not paths:
        return {"n_paths": 0, "median": float("nan"), "note": "no valid paths"}
    arr = np.array(paths)
    return {"n_paths": len(arr),
            "median": round(float(np.median(arr)), 3), "mean": round(float(arr.mean()), 3),
            "q05": round(float(np.percentile(arr, 5)), 3),
            "q95": round(float(np.percentile(arr, 95)), 3),
            "frac_positive": round(float((arr > 0).mean()), 3),
            "sharpe_paths": [round(float(x), 3) for x in arr]}


# ----------------------------------------------------------------------------
# White's Reality Check / Hansen's SPA — data-snooping control for SELECTION
# ----------------------------------------------------------------------------
def _stationary_bootstrap_idx(n: int, block: int, rng) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index sequence (mean block length `block`)."""
    p = 1.0 / max(block, 1)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        idx[t] = rng.integers(0, n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def spa_test(trial_returns: pd.DataFrame, benchmark: float = 0.0, *, n_boot: int = 1000,
             block: int = 10, seed: int = 0) -> dict:
    """White's Reality Check & Hansen's SPA: is the BEST of many searched strategies truly
    better than `benchmark` (per-period return), or just the luckiest draw of the search?

    `trial_returns`: DataFrame, one column of aligned per-period returns per config. Uses
    a stationary bootstrap on the (studentized) performance differentials. Returns the best
    column, the test stat, and two p-values: `rc_p` (White, recenter-all = conservative)
    and `spa_p` (Hansen consistent recentering = more powerful). p < ~0.05 ⇒ the winner
    survives the data-snooping correction; a large p ⇒ the 'edge' is likely search luck.

    Deflated Sharpe haircuts ONE strategy for the number of trials; SPA is the matching
    test for 'I picked the best of K' — the right guard for strategy SELECTION.
    """
    M = trial_returns.dropna(how="any")
    cols = list(M.columns)
    n = len(M)
    if len(cols) < 1 or n < 20:
        return {"spa_p": float("nan"), "rc_p": float("nan"), "note": "insufficient data"}
    d = M.values - benchmark                                   # (n,k); higher = better
    wbar = d.mean(axis=0)
    sd = d.std(axis=0, ddof=1)
    sd = np.where(sd <= 0, np.nan, sd)
    z = np.sqrt(n) * wbar / sd
    T = float(np.nanmax(z))
    best = cols[int(np.nanargmax(z))]
    thr = -(sd / np.sqrt(n)) * np.sqrt(2 * np.log(np.log(n))) if n > 3 else np.full_like(sd, -np.inf)
    g = np.where(wbar >= thr, wbar, 0.0)                       # Hansen consistent recentering
    rng = np.random.default_rng(seed)
    cnt_rc = cnt_spa = 0
    for _ in range(n_boot):
        bi = _stationary_bootstrap_idx(n, block, rng)
        wb = d[bi].mean(axis=0)
        if np.nanmax(np.sqrt(n) * (wb - wbar) / sd) >= T:
            cnt_rc += 1
        if np.nanmax(np.sqrt(n) * (wb - g) / sd) >= T:
            cnt_spa += 1
    spa_p = cnt_spa / n_boot
    return {"best": best, "stat": round(T, 3), "n": n, "n_strategies": len(cols),
            "rc_p": round(cnt_rc / n_boot, 4), "spa_p": round(spa_p, 4),
            "interpretation": ("winner survives data-snooping (p<0.05)" if spa_p < 0.05
                               else "best may be search luck — not significant after SPA")}


def selection_robustness(trial_returns: pd.DataFrame, *, winner: str | None = None,
                         periods_per_year: int = TRADING_DAYS) -> dict:
    """One-stop "is the selected strategy real?" bundle over a searched set of configs.

    `trial_returns`: DataFrame, one column of per-period returns per strategy/config.
    Combines, in one dict for a report: the winner's DEFLATED SHARPE (n_trials=#configs),
    PBO (CSCV), SPA/Reality-Check p-values, and a CPCV OOS-Sharpe DISTRIBUTION for the
    winner. This is the honest verdict for the report's "策略测试选择 / 稳健性体检" block.
    """
    cols = list(trial_returns.columns)
    if not cols:
        return {"note": "no trials"}
    if winner is None:
        srs = {c: (trial_returns[c].mean() / s)
               for c in cols if (s := trial_returns[c].std(ddof=1)) and s > 0}
        winner = max(srs, key=srs.get) if srs else cols[0]
    out: dict = {"winner": winner, "n_trials": len(cols)}
    try:
        out["deflated_sharpe"] = round(deflated_sharpe_ratio(
            trial_returns[winner], n_trials=len(cols), all_trial_returns=trial_returns), 3)
    except Exception:  # noqa: BLE001
        out["deflated_sharpe"] = float("nan")
    try:
        out["pbo"] = round(pbo_cscv(trial_returns).get("pbo", float("nan")), 3)
    except Exception:  # noqa: BLE001
        out["pbo"] = float("nan")
    out["spa"] = spa_test(trial_returns)
    out["cpcv"] = cpcv(trial_returns[winner], periods_per_year=periods_per_year)
    return out
