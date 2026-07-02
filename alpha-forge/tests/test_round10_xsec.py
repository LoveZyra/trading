"""Round10 §2.7/§2.8/§2.13:横截面中性化评估 + 动态选池 + drift regime gate。
只用 numpy/pandas,离线、确定性(固定 seed)。"""
import numpy as np, pandas as pd, pytest
from scripts.xsec import panel as PN, xsec_eval, universe as U
from scripts.risk import regime as RG


def _idx(T=280):
    return pd.date_range("2022-01-03", periods=T, freq="B")


def _rand_panel(n=20, T=280, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(T, n)), index=_idx(T),
                        columns=[f"S{i:02d}" for i in range(n)])


def _ohlcv(n_sym=20, T=280, seed=0):
    rng = np.random.default_rng(seed)
    idx = _idx(T)
    mu = rng.normal(0, 0.0015, n_sym)
    data = {}
    for i in range(n_sym):
        c = 100 * np.exp(np.cumsum(mu[i] + rng.normal(0, 0.01, T)))
        data[f"S{i:02d}"] = pd.DataFrame(
            {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx)
    return data


# ---------- §2.7 cs_zscore / neutralize_panel ----------

def test_cs_zscore_row_mean0_std1():
    z = PN.cs_zscore(_rand_panel(seed=1))
    row = z.iloc[7]
    assert abs(float(row.mean())) < 1e-10
    assert abs(float(row.std(ddof=0)) - 1.0) < 1e-10


def test_cs_zscore_grouped_within_group():
    p = _rand_panel(seed=2)
    gb = {c: ("A" if i < 10 else "B") for i, c in enumerate(p.columns)}
    z = PN.cs_zscore(p, groupby=gb)
    for cols in (p.columns[:10], p.columns[10:]):
        row = z[list(cols)].iloc[5]
        assert abs(float(row.mean())) < 1e-10
        assert abs(float(row.std(ddof=0)) - 1.0) < 1e-10


def test_industry_neutral_group_mean_zero():
    p = _rand_panel(seed=3)
    sm = {c: ("A" if i % 2 == 0 else "B") for i, c in enumerate(p.columns)}
    p[[c for c in p.columns if sm[c] == "A"]] += 5.0        # 植入行业水平差
    n = PN.neutralize_panel(p, sector_map=sm)
    for g in ("A", "B"):
        cols = [c for c in p.columns if sm[c] == g]
        assert n[cols].mean(axis=1).abs().max() < 1e-10


def test_style_neutral_orthogonal_to_style():
    style = _rand_panel(seed=4)
    f = 3.0 * style + 0.1 * _rand_panel(seed=5)             # 因子里塞满风格暴露
    n = PN.neutralize_panel(f, style_panels={"sty": style})
    for i in (0, 100, -1):                                   # 残差与 style 截面相关 ≈ 0
        c = np.corrcoef(n.iloc[i].values, style.iloc[i].values)[0, 1]
        assert abs(c) < 1e-8


# ---------- §2.7 evaluate_cross_section 口径 ----------

def test_eval_default_matches_neutralize_none():
    data = _ohlcv(seed=6)
    r1 = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME", min_names=10)
    r2 = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME", min_names=10,
                                          neutralize=None)
    assert r1["scorecard"] == r2["scorecard"]               # 默认口径回归:逐字段一致
    assert r1["scorecard"]["neutralize"] is None
    pd.testing.assert_frame_equal(r1["preds"], r2["preds"])


def test_eval_industry_neutral_runs_and_records():
    data = _ohlcv(seed=7)
    sm = {s: ("A" if i % 2 == 0 else "B") for i, s in enumerate(data)}
    res = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME", min_names=10,
                                           neutralize="industry", sector_map=sm)
    assert res["scorecard"]["neutralize"] == "industry"
    assert res["scorecard"]["n_dates"] > 0
    with pytest.raises(ValueError):                         # industry 缺 sector_map 要报错
        xsec_eval.evaluate_cross_section(data, neutralize="industry")


# ---------- §2.8 动态选池 ----------

def _mk_px(idx, seed=0, volume=1e6):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"close": c, "volume": volume}, index=idx)


def test_dynamic_universe_filters():
    idx = _idx(200)
    prices = {"A": _mk_px(idx, 1), "B": _mk_px(idx[-30:], 2),          # B 历史不足 60 根
              "C": _mk_px(idx, 3, volume=1.0),                         # C 流动性极低
              "D": _mk_px(idx, 4), "E": _mk_px(idx, 5)}
    meta = {"A": {"mktcap": 5e9}, "C": {"mktcap": 5e9},
            "D": {"mktcap": 1e8},                                      # D 市值不达标
            "E": {}}                                                   # E 缺市值 -> 跳过该滤条
    out = U.dynamic_universe(meta, prices, date=idx[-1], cap_min=1e9, adv_min=1e5)
    assert set(out) == {"A", "E"}
    # 早期 as-of:D 的市值滤条仍生效,B 依旧欠历史
    out_early = U.dynamic_universe(meta, prices, date=idx[80], cap_min=1e9, adv_min=1e5)
    assert "B" not in out_early and "D" not in out_early and "A" in out_early


def test_anti_survivorship_excludes_future_listings():
    meta = {"OLD":  {"list_date": "2015-01-01", "delist_date": None},
            "NEW":  {"list_date": "2024-06-01", "delist_date": None},  # asof 后上市
            "DEAD": {"list_date": "2010-01-01", "delist_date": "2020-01-01"},
            "NOINFO": {}}                                              # 字段缺失 -> 保守保留
    syms, warns = U.anti_survivorship_pool(meta, asof_date="2023-01-01")
    assert "OLD" in syms and "NOINFO" in syms
    assert "NEW" not in syms and "DEAD" not in syms
    assert any("NOINFO" in w for w in warns)


def test_rolling_universe_evolves():
    idx = _idx(280)
    prices = {"A": _mk_px(idx, 1), "B": _mk_px(idx[180:], 2)}          # B 中途才上市
    rd = [idx[100], idx[-1]]
    pools = U.rolling_universe({}, prices, rebalance_dates=rd)
    assert pools[idx[100]] == ["A"]                                    # 早期 B 无历史
    assert set(pools[idx[-1]]) == {"A", "B"}                           # 后期 B 进池


# ---------- §2.13 drift regime gate ----------

def test_stock_drift_regime_causal():
    rng = np.random.default_rng(8)
    idx = _idx(250)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 250))), index=idx)
    st1 = RG.stock_drift_regime(close)
    tampered = close.copy()
    tampered.iloc[160:] = 1.0                                          # 篡改未来:崩盘
    st2 = RG.stock_drift_regime(tampered)
    # state 经 shift(1),第 160 根之前(含)只依赖 <=159 的数据,必须逐点不变
    assert (st1.iloc[:161].values == st2.iloc[:161].values).all()
    assert set(st1.dropna().unique()) <= {0.0, 1.0}


def test_drift_gate_nan_behavior():
    T, idx = 200, _idx(200)
    up = pd.DataFrame({"close": 100.0 + np.arange(T)}, index=idx)      # 纯漂移:全上涨日
    dn = pd.DataFrame({"close": 300.0 - np.arange(T)}, index=idx)      # 纯下跌:无漂移
    fp = pd.DataFrame(1.0, index=idx, columns=["UP", "DOWN", "GHOST"])
    g = RG.drift_regime_gate(fp, {"UP": up, "DOWN": dn}, window=63, activate_in="drift")
    assert g["UP"].iloc[100:].notna().all()                            # drift 状态保留因子值
    assert g["DOWN"].isna().all()                                      # 非 drift 全程置 NaN
    assert g["UP"].iloc[:60].isna().all()                              # warmup 保守视为非 drift
    assert g["GHOST"].isna().all()                                     # 无价格数据 -> 未知 -> NaN
    g2 = RG.drift_regime_gate(fp, {"UP": up, "DOWN": dn}, window=63, activate_in="non_drift")
    assert g2["DOWN"].notna().all() and g2["UP"].iloc[100:].isna().all()
    with pytest.raises(ValueError):
        RG.drift_regime_gate(fp, {}, activate_in="bogus")
