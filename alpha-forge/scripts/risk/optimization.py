"""Portfolio construction & optimization — analytic + iterative solvers, pure numpy.

Why no cvxpy/scipy: this skill must run in a bare pandas/numpy environment, so
everything here is either a closed-form solution (min-variance, tangency portfolio,
unconstrained frontier, Black-Litterman posterior) or a projection iteration
(active-set for long-only, projected gradient for capped mean-variance) that
converges in tens of steps for the <=100-asset universes this skill handles.

Numerical hygiene: every covariance is symmetrized and ridge-conditioned
(cov + eye*1e-8) before inversion — trailing sample covariances of correlated
equities are routinely near-singular, and np.linalg.solve on such matrices returns
garbage weights that *look* plausible. Output invariants (sum(w)=1, no shorts when
long_only) are asserted, not assumed, so a solver bug fails loudly instead of
leaking a broken book into a backtest.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_RIDGE = 1e-8


# ---------------------------------------------------------------- plumbing

def _as_cov(cov, ridge: float = _RIDGE):
    """Symmetrize + ridge-condition a covariance. Returns (ndarray, names)."""
    if isinstance(cov, pd.DataFrame):
        names = list(cov.columns)
        c = cov.values.astype(float)
    else:
        c = np.asarray(cov, dtype=float)
        names = list(range(c.shape[0]))
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be a square matrix")
    c = 0.5 * (c + c.T) + np.eye(c.shape[0]) * ridge
    return c, names


def _align(vec, names) -> np.ndarray:
    if isinstance(vec, pd.Series):
        return vec.reindex(names).fillna(0.0).values.astype(float)
    v = np.asarray(vec, dtype=float).ravel()
    if len(v) != len(names):
        raise ValueError("vector length does not match covariance size")
    return v


def _check(w: pd.Series, long_only: bool) -> pd.Series:
    assert abs(float(w.sum()) - 1.0) < 1e-6, "optimizer invariant violated: sum(w) != 1"
    if long_only:
        assert float(w.min()) > -1e-9, "optimizer invariant violated: negative weight under long_only"
    return w


def _proj_box_simplex(v, lo: float, hi: float, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {lo <= w <= hi, sum(w) = total} via bisection on the
    simplex Lagrange multiplier. Exact (to bisection tol) and allocation-free — the
    workhorse behind every capped/long-only projection below."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if lo * n > total + 1e-12 or hi * n < total - 1e-12:
        raise ValueError(f"infeasible constraint set: n={n}, box=[{lo},{hi}], sum={total}")
    a = float(v.min()) - hi - 1.0     # f(a) = n*hi >= total
    b = float(v.max()) - lo + 1.0     # f(b) = n*lo <= total
    for _ in range(100):
        mid = 0.5 * (a + b)
        if np.clip(v - mid, lo, hi).sum() > total:
            a = mid
        else:
            b = mid
    return np.clip(v - 0.5 * (a + b), lo, hi)


def _apply_cap(w: pd.Series, cap: float, long_only: bool) -> pd.Series:
    """Redistribute weight above `cap`. Long-only: fix violators at cap, renormalize
    the rest (iterated — redistribution can push new names over). Long-short: exact
    box-simplex projection onto [-cap, cap]."""
    n = len(w)
    if cap * n < 1 - 1e-9:
        raise ValueError(f"weight_cap={cap} infeasible for {n} assets (n*cap < 1)")
    x = w.values.astype(float).copy()
    if not long_only:
        return pd.Series(_proj_box_simplex(x, -cap, cap), index=w.index)
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n):
        over = (x > cap + 1e-12) & ~fixed
        if not over.any():
            break
        fixed |= over
        x[fixed] = cap
        free = ~fixed
        budget = 1.0 - cap * fixed.sum()
        s = x[free].sum()
        x[free] = x[free] * budget / s if s > 0 else budget / max(free.sum(), 1)
    return pd.Series(x, index=w.index)


def _active_set(solve, n: int, long_only: bool) -> np.ndarray:
    """Long-only via active-set iteration: solve analytically, drop the assets the
    solution wants to short, re-solve on the survivors, repeat until no negatives.
    This is the classic projection heuristic — each pass strictly shrinks the set, so
    it terminates in <= n solves; the result is feasible by construction."""
    idx = list(range(n))
    w_sub = solve(idx)
    if long_only:
        for _ in range(n):
            neg = w_sub < -1e-12
            if not neg.any():
                break
            idx = [i for i, bad in zip(idx, neg) if not bad]
            if len(idx) == 1:
                w_sub = np.array([1.0])
                break
            w_sub = solve(idx)
        w_sub = np.clip(w_sub, 0.0, None)
        w_sub = w_sub / w_sub.sum()
    full = np.zeros(n)
    full[np.asarray(idx, dtype=int)] = w_sub
    return full


# ---------------------------------------------------------------- closed forms

def min_variance_weights(cov, *, long_only: bool = True, weight_cap: float | None = None) -> pd.Series:
    """Global minimum-variance portfolio: w = Σ⁻¹1 / (1'Σ⁻¹1).

    GMV is the one mean-variance portfolio that needs NO expected-return estimate —
    the input everyone gets wrong — which is why it out-of-samples so well. Long-only
    handled by active-set re-solving (see _active_set); weight_cap by capped
    redistribution.
    """
    c, names = _as_cov(cov)

    def solve(idx):
        sub = c[np.ix_(idx, idx)]
        x = np.linalg.solve(sub, np.ones(len(idx)))
        return x / x.sum()

    w = pd.Series(_active_set(solve, len(names), long_only), index=names)
    if weight_cap is not None:
        w = _apply_cap(w, float(weight_cap), long_only)
    return _check(w, long_only)


def max_sharpe_weights(cov, mu, rf: float = 0.0, *, long_only: bool = True,
                       weight_cap: float | None = None) -> pd.Series:
    """Tangency (max-Sharpe) portfolio: w ∝ Σ⁻¹(μ − rf), normalized to sum 1.

    Warning baked into the design: the tangency portfolio is hypersensitive to μ —
    tiny estimation noise flips weights wildly (Michaud's "error maximizer"). If all
    excess returns are <= 0, or the analytic solution nets to non-positive exposure,
    we degrade to min-variance with a warning instead of returning a leveraged
    nonsense book.
    """
    c, names = _as_cov(cov)
    ex = _align(mu, names) - float(rf)
    if (ex <= 0).all():
        warnings.warn("max_sharpe: no asset has positive excess return; "
                      "falling back to min-variance.", stacklevel=2)
        return min_variance_weights(cov, long_only=long_only, weight_cap=weight_cap)

    fallback = {"hit": False}

    def solve(idx):
        sub = c[np.ix_(idx, idx)]
        x = np.linalg.solve(sub, ex[np.asarray(idx, dtype=int)])
        s = x.sum()
        if s <= 1e-12:
            fallback["hit"] = True
            x = np.linalg.solve(sub, np.ones(len(idx)))
            s = x.sum()
        return x / s

    w = pd.Series(_active_set(solve, len(names), long_only), index=names)
    if fallback["hit"]:
        warnings.warn("max_sharpe: tangency net exposure <= 0 (dominant shorts); "
                      "used min-variance on the surviving set.", stacklevel=2)
    if weight_cap is not None:
        w = _apply_cap(w, float(weight_cap), long_only)
    return _check(w, long_only)


def efficient_frontier(cov, mu, n_points: int = 50) -> pd.DataFrame:
    """Analytic unconstrained frontier (fully invested, shorts allowed) via the
    two-fund theorem: w(r) = λΣ⁻¹1 + γΣ⁻¹μ with the classic A,B,C,D scalars.

    We sweep target returns from the GMV return UPWARD only — the lower branch is
    dominated and plotting it misleads. Columns: ret, vol, sharpe (rf=0), weights
    (dict per row). For a long-only frontier, call mean_variance_constrained over a
    risk_aversion sweep instead.
    """
    c, names = _as_cov(cov)
    m = _align(mu, names)
    ci1 = np.linalg.solve(c, np.ones(len(m)))
    cim = np.linalg.solve(c, m)
    A = float(ci1.sum())
    B = float(m @ ci1)
    C = float(m @ cim)
    D = A * C - B * B
    r_gmv = B / A
    if D <= 1e-16:      # degenerate (all mu equal): frontier collapses to GMV point
        w = ci1 / A
        vol = float(np.sqrt(w @ c @ w))
        return pd.DataFrame([{"ret": r_gmv, "vol": vol,
                              "sharpe": r_gmv / vol if vol > 0 else np.nan,
                              "weights": dict(zip(names, np.round(w, 6)))}])
    r_hi = max(float(m.max()), r_gmv + 1e-8)
    rows = []
    for r in np.linspace(r_gmv, r_hi, n_points):
        lam = (C - B * r) / D
        gam = (A * r - B) / D
        w = lam * ci1 + gam * cim
        vol = float(np.sqrt(max(w @ c @ w, 0.0)))
        rows.append({"ret": float(r), "vol": vol,
                     "sharpe": float(r / vol) if vol > 0 else np.nan,
                     "weights": dict(zip(names, np.round(w, 6)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Black-Litterman

def black_litterman(cov, market_weights, views_P=None, views_Q=None, *,
                    tau: float = 0.025, omega=None, risk_aversion: float = 2.5) -> dict:
    """Black-Litterman posterior: blend market-implied returns with subjective views.

    Why BL instead of raw mean-variance on predicted returns: the market portfolio
    anchors the prior (π = δΣw_mkt), so views only TILT the book instead of letting a
    noisy μ estimate dictate extreme weights. With no views this returns exactly the
    market-implied returns and market weights — the correct "I know nothing" answer.

        posterior_mu  = π + τΣP'(PτΣP' + Ω)⁻¹(Q − Pπ)
        posterior_cov = Σ + τΣ − τΣP'(PτΣP' + Ω)⁻¹PτΣ
        weights       = (δ·posterior_cov)⁻¹ posterior_mu, renormalized to sum 1

    omega defaults to diag(P τΣ P') (He-Litterman): view uncertainty proportional to
    the prior uncertainty of the view portfolio itself.

    views_P: (k x n) view-loading matrix (DataFrame aligned to cov columns, or array);
    views_Q: k expected view returns. Build them from ML predictions with
    views_from_predictions().
    """
    c, names = _as_cov(cov)
    w_mkt = _align(market_weights, names)
    s = w_mkt.sum()
    if abs(s) > 1e-12:
        w_mkt = w_mkt / s
    pi = risk_aversion * c @ w_mkt

    no_views = views_P is None or (hasattr(views_P, "__len__") and len(views_P) == 0)
    if no_views:
        return {"posterior_mu": pd.Series(pi, index=names),
                "posterior_cov": pd.DataFrame(c, index=names, columns=names),
                "weights": pd.Series(w_mkt, index=names)}

    if isinstance(views_P, pd.DataFrame):
        P = views_P.reindex(columns=names).fillna(0.0).values.astype(float)
    else:
        P = np.atleast_2d(np.asarray(views_P, dtype=float))
    Q = np.asarray(views_Q, dtype=float).ravel()
    if P.shape != (len(Q), len(names)):
        raise ValueError(f"views_P shape {P.shape} inconsistent with {len(Q)} views x {len(names)} assets")

    ts = tau * c
    if omega is None:
        omega_m = np.diag(np.diag(P @ ts @ P.T))
    else:
        om = np.asarray(omega, dtype=float)
        omega_m = np.diag(om) if om.ndim == 1 else om
    M = P @ ts @ P.T + omega_m + np.eye(len(Q)) * _RIDGE   # condition the k x k solve too
    resid = Q - P @ pi
    post_mu = pi + ts @ P.T @ np.linalg.solve(M, resid)
    post_cov = c + ts - ts @ P.T @ np.linalg.solve(M, P @ ts)
    post_cov = 0.5 * (post_cov + post_cov.T) + np.eye(len(names)) * _RIDGE

    w_raw = np.linalg.solve(risk_aversion * post_cov, post_mu)
    sw = w_raw.sum()
    if abs(sw) > 1e-12:
        w = w_raw / sw
    else:
        warnings.warn("black_litterman: posterior weights net to ~0; returning market weights.",
                      stacklevel=2)
        w = w_mkt
    return {"posterior_mu": pd.Series(post_mu, index=names),
            "posterior_cov": pd.DataFrame(post_cov, index=names, columns=names),
            "weights": pd.Series(w, index=names)}


def views_from_predictions(pred: pd.Series, conf=0.5):
    """Turn per-asset ML return predictions into BL absolute views: (P, Q, omega).

    One view per predicted asset: P is a k x n identity selection (DataFrame), Q the
    predicted returns, omega a diagonal uncertainty matrix. `conf` in (0, 1] — scalar
    or per-asset Series — maps to view variance via (1-c)/c scaled by the
    cross-sectional variance of the predictions themselves, so "confidence" is
    calibrated to how dispersed the model's own forecasts are rather than to an
    arbitrary absolute number. conf→1 means near-certain (omega→0, view dominates);
    conf→0 means the view barely moves the prior.
    """
    pred = pd.Series(pred).dropna()
    names = list(pred.index)
    k = len(names)
    if k == 0:
        raise ValueError("views_from_predictions: no finite predictions")
    P = pd.DataFrame(np.eye(k), index=[f"view_{n}" for n in names], columns=names)
    Q = pred.values.astype(float)
    if np.isscalar(conf):
        c = np.full(k, float(conf))
    else:
        c = pd.Series(conf).reindex(names).fillna(0.5).values.astype(float)
    c = np.clip(c, 1e-4, 1 - 1e-6)
    base = float(np.var(Q))
    if not np.isfinite(base) or base <= 0:
        base = 1e-4
    omega = np.diag(base * (1.0 - c) / c)
    return P, Q, omega


def views_prompt(predictions, context: str = "") -> str:
    """Prompt template for having an LLM (the agent itself) turn model predictions +
    qualitative context into structured BL views.

    Based on arXiv:2504.14345 (LLM-generated Black-Litterman views): LLMs can
    translate narratives into (P, Q, confidence) triples that BL digests cleanly.
    CAVEAT the paper and practice both flag: letting an LLM pick views is roughly
    equivalent to picking a STYLE (it systematically leans momentum/quality/narrative
    -consistent), not adding new information — treat the output as a style tilt to be
    sized small (low confidence / large omega), never as alpha to be levered.

    The LLM call itself happens agent-side; this function only builds the prompt.
    """
    pred_txt = pd.Series(predictions).round(4).to_string()
    return (
        "你是组合构建助手。基于以下模型预测(未来一期收益)与背景信息,提出不超过 5 条 "
        "Black-Litterman 观点。\n\n"
        f"模型预测(per-asset expected return):\n{pred_txt}\n\n"
        f"背景信息:\n{context or '(无)'}\n\n"
        "输出 JSON 列表,每条观点:{\"assets\": {symbol: 权重, ...}(相对观点权重和为 0,"
        "绝对观点单资产权重 1), \"expected_return\": 数值(期收益), \"confidence\": 0~1}。\n"
        "要求:只在预测与背景相互印证时给高 confidence;相互矛盾时 confidence <= 0.3;"
        "不要臆造预测之外的标的。注意:你的选择本质上是在选风格(动量/质量/叙事一致性),"
        "请保持观点数量少、confidence 保守。"
    )


# ---------------------------------------------------------------- constrained MV

def mean_variance_constrained(cov, mu, *, long_only: bool = True, weight_cap: float = 0.1,
                              sector_caps: dict | None = None, sector_map: dict | None = None,
                              risk_aversion: float = 2.5) -> pd.Series:
    """Mean-variance with box + sector caps via projected gradient ascent on
    f(w) = μ'w − (δ/2)·w'Σw.

    Why projected gradient: the constraint set (simplex ∩ box ∩ sector half-spaces)
    has no closed-form solution, but its projection is cheap (exact box-simplex
    bisection + POCS rounds for sector caps), and with step = 1/(δ·λmax(Σ)) the
    ascent is a contraction — a few hundred iterations is overkill-safe for the
    universes this skill handles. This is the practical replacement for cvxpy here.

    sector_caps: {sector: max_total_weight}; sector_map: {symbol: sector}. Sector
    caps assume long_only (scaling shorts is not meaningful).
    """
    c, names = _as_cov(cov)
    m = _align(mu, names)
    n = len(names)
    cap = float(weight_cap) if weight_cap is not None else 1.0
    lo = 0.0 if long_only else -cap
    if cap * n < 1 - 1e-9:
        raise ValueError(f"weight_cap={cap} infeasible for {n} assets (n*cap < 1)")
    sectors = None
    if sector_caps:
        if not sector_map:
            raise ValueError("sector_caps given without sector_map")
        sectors = np.array([sector_map.get(nm, "__none__") for nm in names])
        room = sum(min(float(sector_caps.get(s, 1.0)), cap * (sectors == s).sum())
                   for s in np.unique(sectors))
        if room < 1 - 1e-9:
            raise ValueError("sector_caps + weight_cap leave less than 100% investable")

    def project(v):
        w = _proj_box_simplex(v, lo, cap)
        if not sector_caps:
            return w
        for _ in range(200):        # POCS: alternate simplex/box and sector shrink
            viol = 0.0
            for s, cs in sector_caps.items():
                mask = sectors == s
                tot = w[mask].sum()
                if tot > cs + 1e-12:
                    viol = max(viol, tot - cs)
                    w[mask] *= cs / tot
            if viol < 1e-11:
                return w
            w = _proj_box_simplex(w, lo, cap)
        return w

    eig_max = float(np.linalg.eigvalsh(c)[-1])
    step = 1.0 / (risk_aversion * max(eig_max, 1e-12))
    w = project(np.full(n, 1.0 / n))
    for _ in range(300):
        grad = m - risk_aversion * (c @ w)
        w_new = project(w + step * grad)
        if np.abs(w_new - w).max() < 1e-11:
            w = w_new
            break
        w = w_new
    out = pd.Series(w, index=names)
    assert float(out.max()) <= cap + 1e-6, "weight_cap violated after projection"
    if sector_caps:
        for s, cs in sector_caps.items():
            assert float(out[sectors == s].sum()) <= cs + 1e-6, f"sector cap {s} violated"
    return _check(out, long_only)
