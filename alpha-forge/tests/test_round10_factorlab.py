"""Round-10B 因子质量套件回归:正交化/增量 IC(§2.2)、拥挤度(§2.3)、
衰减监控(§2.4)、AST 复杂度/新颖度(§2.14)、五维 alpha_eval(§2.16)。
全部用可控合成数据:因子→收益的因果链自己造,期望值可以手推。"""
import numpy as np
import pandas as pd
import pytest

from scripts.research import factor_lab as FL
from scripts.research import crowding as CW
from scripts.research import decay_monitor as DM


def _idx(n):
    return pd.bdate_range("2022-01-03", periods=n)


def _cols(k):
    return [f"S{i:02d}" for i in range(k)]


def _const_panel(k, n, seed):
    """每只票一个常数因子值(时间上不变)——预测力干净、秩序恒定,便于手推。"""
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(k)
    return pd.DataFrame(np.tile(vals, (n, 1)), index=_idx(n), columns=_cols(k))


def _close_from(panels, betas, n, k, seed, noise=0.01):
    """日收益 = Σ beta·panel + 噪声 → close 宽表。panel 时间上常数时无因果问题。"""
    rng = np.random.default_rng(seed)
    r = noise * rng.standard_normal((n, k))
    for P, b in zip(panels, betas):
        r = r + b * P.values
    return pd.DataFrame(100.0 * np.exp(np.cumsum(r, axis=0)),
                        index=_idx(n), columns=_cols(k))


# ---- §2.2 正交化 / 增量 IC ---------------------------------------------------

def test_orthogonalize_residual_uncorrelated_with_reference():
    k, n = 30, 120
    ref = _const_panel(k, n, seed=1)
    noise = _const_panel(k, n, seed=2)
    fac = ref + 0.5 * noise
    orth = FL.orthogonalize(fac, {"ref": ref})
    assert orth.shape == fac.shape
    # 逐日截面 Pearson:OLS 残差对回归子必须精确正交
    for i in [0, n // 2, n - 1]:
        y, x = orth.iloc[i].values, ref.iloc[i].values
        m = np.isfinite(y) & np.isfinite(x)
        c = np.corrcoef(y[m], x[m])[0, 1]
        assert abs(c) < 1e-8


def test_incremental_ic_separates_rehash_from_independent_signal():
    k, n, h = 30, 280, 5
    ref = _const_panel(k, n, seed=3)
    ind = _const_panel(k, n, seed=4)
    close = _close_from([ref, ind], [0.002, 0.002], n, k, seed=5)
    rehash = ref + 0.15 * _const_panel(k, n, seed=6)     # 参照因子换皮
    out_re = FL.incremental_ic(rehash, {"ref": ref}, close, horizon=h)
    out_in = FL.incremental_ic(ind, {"ref": ref}, close, horizon=h)
    assert out_re["raw_ic"] > 0.1                        # 换皮因子原始 IC 挺好看
    assert out_re["incremental_ratio"] < 0.4             # 但正交化后基本归零
    assert out_in["incremental_ratio"] > 0.6             # 独立信号扛得住正交化
    assert set(out_in) == {"raw_ic", "orthogonal_ic", "incremental_ratio"}


def test_factor_correlation_matrix_symmetric_diag_one():
    k, n = 25, 90
    A = _const_panel(k, n, seed=7)
    B = _const_panel(k, n, seed=8)
    M = FL.factor_correlation_matrix({"a": A, "neg_a": -A, "b": B})
    assert np.allclose(np.diag(M.values), 1.0)
    assert np.allclose(M.values, M.values.T, atol=1e-12)
    assert M.loc["a", "neg_a"] == pytest.approx(-1.0, abs=1e-9)
    assert abs(M.loc["a", "b"]) < 0.5


# ---- §2.14 AST 复杂度 / 新颖度 ------------------------------------------------

def test_complexity_control_gates_depth_and_params():
    ok = FL.complexity_control("ts_rank(close, 10) - ts_rank(volume, 10)")
    assert ok["ok"] and ok["depth"] <= 5 and ok["n_params"] == 2
    deep = FL.complexity_control("f(g(h(p(q(r(s(x)))))))")
    assert not deep["ok"] and deep["depth"] > 5
    fat = FL.complexity_control("f(x, 1.5, 2.0, 3.0, 4.0)")
    assert not fat["ok"] and fat["n_params"] == 4
    bad = FL.complexity_control("close +* 3")
    assert not bad["ok"] and "error" in bad


def test_novelty_check_self_zero_and_new_expr_novel():
    lib = {"mom": "ts_rank(close, 10)", "vol": "std(returns, 20)"}
    same = FL.novelty_check("ts_rank(close, 10)", lib)
    assert same["min_distance"] == 0.0 and same["nearest"] == "mom" and not same["novel"]
    tweak = FL.novelty_check("ts_rank(close, 15)", lib)   # 只改常数 ≠ 新因子
    assert not tweak["novel"]
    new = FL.novelty_check("corr(delta(open, 3), log(volume), 30)", lib)
    assert new["novel"] and new["min_distance"] > 0.25
    empty = FL.novelty_check("ts_rank(close, 10)", {})
    assert empty["novel"] and empty["nearest"] is None


# ---- §2.16 五维 alpha_eval ----------------------------------------------------

def test_alpha_eval_panel_five_dims_in_unit_range():
    k, n, h = 30, 280, 5
    fac = _const_panel(k, n, seed=10)
    close = _close_from([fac], [0.002], n, k, seed=11)
    out = FL.alpha_eval(fac, close, existing_panels={"self_copy": fac}, horizon=h)
    for key in ("predictive", "robustness", "diversity", "stability", "interpretability"):
        assert key in out
    for key in ("predictive", "robustness", "diversity", "stability", "composite"):
        v = out[key]
        assert np.isfinite(v) and 0.0 <= v <= 1.0, (key, v)
    assert out["interpretability"] is None
    assert isinstance(out["interpretability_prompt"], str) and out["interpretability_prompt"]
    assert out["predictive"] > 0.5                      # 构造出来的强信号
    assert out["diversity"] < 0.1                       # 和自身拷贝零多样性


def test_alpha_eval_accepts_callable_with_ohlcv_dict():
    rng = np.random.default_rng(12)
    idx = _idx(250)
    data = {}
    for i in range(12):
        r = 4e-4 + 0.015 * rng.standard_normal(250)
        c = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        data[f"S{i:02d}"] = pd.DataFrame({"open": c, "high": c * 1.01,
                                          "low": c * 0.99, "close": c,
                                          "volume": 1e6}, index=idx)
    out = FL.alpha_eval(lambda df: df["close"].pct_change(10), data, horizon=10)
    assert len(out["detail"]["robustness_ics"]) == 3   # horizon×{0.7,1,1.3} 三档
    for key in ("predictive", "robustness", "diversity", "stability", "composite"):
        assert 0.0 <= out[key] <= 1.0
    assert out["diversity"] == 1.0                      # 没给 existing → 满分


# ---- §2.3 拥挤度 --------------------------------------------------------------

def test_holdings_overlap_identity_one_antifactor_zero():
    k, n = 20, 60
    A = _const_panel(k, n, seed=13)
    same = CW.holdings_overlap(A, A + 1.0)              # 平移不改秩 → 完全重叠
    anti = CW.holdings_overlap(A, -A)                   # 反因子 → 多头桶不相交
    assert same.dropna().eq(1.0).all()
    assert anti.dropna().eq(0.0).all()


def test_crowding_score_flags_clone_not_independent():
    k, n = 30, 300
    A = _const_panel(k, n, seed=14)
    B = _const_panel(k, n, seed=15)
    close = _close_from([A], [0.002], n, k, seed=16)
    clone = A + 0.05 * _const_panel(k, n, seed=17)
    hot = CW.crowding_score(A, {"clone": clone}, close)
    cold = CW.crowding_score(A, {"indep": B}, close)
    assert hot["score"] > 0.7 and hot["warning"]
    assert cold["score"] < 0.5 and not cold["warning"]
    assert set(hot["components"]) == {"holdings_overlap", "return_correlation"}
    # value_panel 缺省被优雅跳过;给了就多一个分项
    withv = CW.crowding_score(A, {"clone": clone}, close, value_panel=B)
    assert "valuation_spread" in withv["components"]


def test_fit_hyperbolic_decay_recovers_known_params():
    K, lam = 0.08, 0.05
    t = np.arange(200, dtype=float)
    y = pd.Series(K / (1.0 + lam * t))
    fit = CW.fit_hyperbolic_decay(y)
    assert fit["K"] == pytest.approx(K, rel=1e-6)
    assert fit["lam"] == pytest.approx(lam, rel=1e-6)
    assert fit["half_life"] == pytest.approx(1.0 / lam, rel=1e-6)
    assert fit["r2"] > 0.999
    # 太少/非正观测 → NaN 而不是硬给数
    assert np.isnan(CW.fit_hyperbolic_decay(pd.Series([0.1, -0.1]))["K"])


# ---- §2.4 衰减监控 ------------------------------------------------------------

def test_ic_decay_declines_for_transient_signal():
    """因子每天换一张脸(iid)→ 只预测近端收益,累计 IC 随 h 被稀释而单调走低。"""
    k, n = 30, 400
    rng = np.random.default_rng(18)
    F = pd.DataFrame(rng.standard_normal((n, k)), index=_idx(n), columns=_cols(k))
    r = 0.01 * rng.standard_normal((n, k))
    r[1:] += 0.004 * F.values[:-1]                      # F(T) 只驱动 T+1 的收益
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=_idx(n), columns=_cols(k))
    dec = DM.ic_decay(F, close, horizons=(1, 5, 21, 42))
    assert dec[1] > 0.15
    assert dec[1] > dec[5] > dec[42]


def test_half_life_three_methods_agree_on_ar1_factor():
    """AR(1) 持续性 ρ=0.9 的因子,三种口径的半衰期都应落在 ln2/ln(1/ρ)≈6.6 天附近。"""
    k, n, rho = 30, 500, 0.9
    rng = np.random.default_rng(19)
    F = np.zeros((n, k))
    for t in range(1, n):
        F[t] = rho * F[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(k)
    Fp = pd.DataFrame(F, index=_idx(n), columns=_cols(k))
    r = 0.008 * rng.standard_normal((n, k))
    r[1:] += 0.004 * F[:-1]
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=_idx(n), columns=_cols(k))
    hl = DM.half_life(Fp, close)
    assert set(hl) == {"ar1", "ic_decay", "quantile_spread", "median"}
    true_hl = np.log(2) / -np.log(rho)                  # ≈ 6.58
    assert hl["ar1"] == pytest.approx(true_hl, rel=0.35)
    for key in ("ic_decay", "quantile_spread", "median"):
        assert np.isfinite(hl[key]) and 2.0 < hl[key] < 30.0
    with pytest.raises(ValueError):
        DM.half_life(Fp, close, method="voodoo")


def test_decay_warning_fires_on_dying_factor_only():
    k, n = 30, 500
    rng = np.random.default_rng(20)
    F = _const_panel(k, n, seed=21)
    r = 0.01 * rng.standard_normal((n, k))
    r[: n // 2] += 0.003 * F.values[: n // 2]           # 前半段有效,后半段猝死
    dying = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=_idx(n), columns=_cols(k))
    out = DM.decay_warning(F, dying, rebalance_days=21, horizon=5)
    assert out["warning"] and out["reasons"]
    r2 = 0.01 * rng.standard_normal((n, k)) + 0.003 * F.values   # 全程有效
    alive = pd.DataFrame(100 * np.exp(np.cumsum(r2, axis=0)), index=_idx(n), columns=_cols(k))
    out2 = DM.decay_warning(F, alive, rebalance_days=21, horizon=5)
    assert not out2["warning"]


def test_rolling_ic_shape_and_positive_for_live_factor():
    k, n = 30, 300
    F = _const_panel(k, n, seed=22)
    close = _close_from([F], [0.003], n, k, seed=23)
    roll = DM.rolling_ic(F, close, horizon=5, window=63)
    assert isinstance(roll, pd.Series) and len(roll) == n
    assert roll.dropna().mean() > 0.1


def test_mrp_takes_worst_regime():
    r = pd.Series([0.001] * 100 + [-0.002] * 50, index=_idx(150))
    g = pd.Series(["bull"] * 100 + ["bear"] * 50, index=_idx(150))
    assert DM.mrp(r, g) == pytest.approx(-0.002 * 252)
    assert np.isnan(DM.mrp(r, pd.Series(dtype=object)))  # 对不齐 → NaN
