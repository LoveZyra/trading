"""Round-9 regression tests: A2 cost-stress selection, A3 ensembles/stacking,
A5 conformal gating, A6 regime-conditional weights, B2-B4 report blocks."""
import numpy as np
import pandas as pd
import pytest

from scripts.research import autoresearch as AR

from scripts.core import backtest as bt, metrics as M

from scripts.risk import conformal as C
from scripts.research import models as Mdl
from scripts.core import optimize as opt
from scripts.risk import regime as RG
from scripts.strategies import multi_factor as mf


def _ohlcv(n=700, seed=0, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    r = drift + 0.015 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(r))
    c = pd.Series(close, index=idx)
    return pd.DataFrame({"open": c.shift(1).fillna(c), "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": 1e6 + rng.integers(0, 1e5, n)}, index=idx)


def _universe(k=12, n=600, seed=1):
    rng = np.random.default_rng(seed)
    return {f"S{i:02d}": _ohlcv(n, seed=seed * 100 + i, drift=rng.normal(4e-4, 2e-4))
            for i in range(k)}


# ---- A2 ---------------------------------------------------------------------
def test_cost_stressed_sharpe_penalizes_turnover():
    hi = {"sharpe": 1.0, "turnover_annual": 50.0, "ann_volatility": 0.2}
    lo = {"sharpe": 1.0, "turnover_annual": 2.0, "ann_volatility": 0.2}
    s_hi, s_lo = M.cost_stressed_sharpe(hi, 10), M.cost_stressed_sharpe(lo, 10)
    assert s_lo > s_hi                        # churn is penalized
    assert s_lo == pytest.approx(1.0 - (2.0 * 10 / 1e4) / 0.2)
    assert M.cost_stressed_sharpe({"sharpe": 1.0}, 10) == 1.0   # graceful w/o turnover


def test_walk_forward_reports_oos_turnover():
    from scripts.strategies import MACrossover
    wf = opt.walk_forward(MACrossover, _ohlcv(), {"fast": [10, 20], "slow": [50]},
                          n_splits=3, cost_stress_bps=10)
    assert "turnover_annual" in wf.oos_stats and wf.oos_stats["turnover_annual"] >= 0


def test_research_single_selects_by_stressed_score():
    rep = AR.research_single(_ohlcv(500, seed=3), iterations=30, cost_stress_bps=10, seed=1)
    assert "sel_score" in rep.leaderboard.columns
    sel = rep.leaderboard["sel_score"].dropna().values
    assert (np.diff(sel) <= 1e-9).all()       # sorted by the stressed selection score


def test_multi_factor_weight_smoothing_cuts_turnover():
    data = _universe()
    w0 = mf.multi_factor_signal(data, {"momentum": 1.0}, rebalance="ME")
    w1 = mf.multi_factor_signal(data, {"momentum": 1.0}, rebalance="ME", weight_smoothing=0.5)
    t0 = w0.diff().abs().sum().sum()
    t1 = w1.diff().abs().sum().sum()
    assert t1 < t0                            # smoothing reduces total turnover
    g0 = w0.abs().sum(axis=1).max()
    g1 = w1.abs().sum(axis=1).max()
    assert g1 == pytest.approx(g0, rel=0.05)  # same gross exposure


# ---- A3 ---------------------------------------------------------------------
def test_ensemble_weighting_modes_and_dedupe():
    df = _ohlcv(500, seed=5)
    rep = AR.research_single(df, iterations=30, seed=2)
    for mode in ("equal", "ewma", "regime"):
        res, members = AR.ensemble_top_k(rep, df, k=3, weighting=mode)
        assert np.isfinite(res.stats["sharpe"])
        fams = [m[0] for m in members]
        assert len(fams) == len(set(fams))    # deduped by family


def test_stacking_model_beats_nothing_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = X @ np.array([0.5, -0.2, 0.1, 0.0]) + 0.1 * rng.standard_normal(300)
    m = Mdl.StackingModel(members=[Mdl.RidgeModel(0.3), Mdl.RidgeModel(3.0)]).fit(X, y)
    p = m.predict(X)
    assert p.shape == (300,) and np.isfinite(p).all()
    assert np.corrcoef(p, y)[0, 1] > 0.5


def test_xsec_rank_average_ensemble():
    from scripts.xsec import xsec_autoresearch as XAR
    data = _universe(14, 500, seed=7)
    lb = XAR.search(data, horizon=21, rebalance="ME", top_n=4, min_names=10)
    out = XAR.ensemble_top(data, lb, k=2, horizon=21, rebalance="ME", min_names=10)
    assert out["members"] and "RankIC" in out["scorecard"]
    p = out["preds"]
    assert ((p["pred"] >= 0) & (p["pred"] <= 1)).all()   # rank percentiles in [0,1]


# ---- A6 ---------------------------------------------------------------------
def test_regime_conditional_weights_causal_and_normalized():
    df = _ohlcv(600, seed=11)
    rets = pd.DataFrame({"a": df["close"].pct_change().fillna(0),
                         "b": -df["close"].pct_change().fillna(0) * 0.5})
    w = RG.regime_conditional_weights(rets, df["close"])
    assert w.shape == rets.shape
    rs = w.sum(axis=1)
    assert np.allclose(rs, 1.0, atol=1e-6)    # rows sum to 1
    assert (w.values >= 0).all()


# ---- A5 ---------------------------------------------------------------------
def test_conformal_qhat_and_scaling():
    rng = np.random.default_rng(0)
    q = C.split_conformal_qhat(rng.standard_normal(200) * 0.02, alpha=0.2)
    assert 0.015 < q < 0.05                   # ~80% quantile of |N(0,0.02)|
    pred = pd.Series([0.001, q, 3 * q])
    s = C.conviction_scale(pred, q)
    assert s.iloc[0] < 0.2 and s.iloc[1] == pytest.approx(1.0) and s.iloc[2] == 1.0
    assert C.conviction_scale(pred, float("nan")).eq(1.0).all()  # no fake precision
    assert np.isnan(C.split_conformal_qhat([0.1] * 3, 0.2))      # too little data


def test_ml_backtest_conformal_gate_reduces_gross():
    data = _universe(10, 550, seed=13)
    r0 = Mdl.ml_factor_backtest(data, model=Mdl.RidgeModel(1.0))
    r1 = Mdl.ml_factor_backtest(data, model=Mdl.RidgeModel(1.0), conformal_alpha=0.2)
    assert r1.extra.get("n_calibrations", 0) > 0
    g0 = r0.weights.abs().sum(axis=1).mean()
    g1 = r1.weights.abs().sum(axis=1).mean()
    assert g1 <= g0 + 1e-9                    # the gate never ADDS exposure


# ---- B2/B3/B4 ---------------------------------------------------------------
def test_downturn_slices_shape():
    df = _ohlcv(700, seed=17)
    res = bt.backtest(df, pd.Series(1.0, index=df.index) * 0.5)
    rows = M.downturn_slices(res.returns, bt.buy_and_hold(df).returns)
    assert rows and all({"label", "period", "strategy", "benchmark", "excess"} <= set(r) for r in rows)


def test_cost_sensitivity_monotone():
    df = _ohlcv(500, seed=19)
    from scripts.strategies import MACrossover
    sig = MACrossover(10, 50).generate_signal(df)
    rows = bt.cost_sensitivity(df, sig, levels=(0, 10, 30))
    assert [r["bps"] for r in rows] == [0, 10, 30]
    tr = [r["total_return"] for r in rows]
    assert tr[0] >= tr[1] >= tr[2]            # more cost, less return


def test_build_research_carries_new_blocks_and_renders():
    from scripts.reporting import build_research as BR, html_report as H
    df = _ohlcv(520, seed=23)
    r = BR.build_research(df, "TEST", iterations=10)
    assert r.get("robustness") and r.get("cost_curve") and r["cost_curve"]["rows"]
    html = H.render({"meta": {"title": "t", "date": "2026-07-02"}, "research": r})
    for needle in ("稳健性体检", "下跌切片", "成本敏感度", "robustnessSection"):
        assert needle in html


# ---- Bug-sweep fixes (2026-07-02 巡检) ---------------------------------------
def test_ml_weights_hold_between_rebalances():
    """P0: NaN-init + ffill — the book must stay invested between monthly rebalances,
    not hold for a single bar per month."""
    data = _universe(8, 500, seed=29)
    r = Mdl.ml_factor_backtest(data, model=Mdl.RidgeModel(1.0))
    in_market = (r.weights.abs().sum(axis=1) > 1e-9).mean()
    assert in_market > 0.5                     # was ~4% before the fix
    assert r.stats["turnover_annual"] < 10     # monthly rebalance ≈ 2-6, was ~18


def test_ml_smoothing_does_not_undo_conformal_gate():
    """P1: smoothing rescales to the post-gate gross, never renormalizes to 1.0."""
    data = _universe(10, 550, seed=13)
    g_gate = Mdl.ml_factor_backtest(data, model=Mdl.RidgeModel(1.0),
                                    conformal_alpha=0.2).weights.abs().sum(axis=1)
    g_both = Mdl.ml_factor_backtest(data, model=Mdl.RidgeModel(1.0), conformal_alpha=0.2,
                                    weight_smoothing=0.3).weights.abs().sum(axis=1)
    # smoothing may not blow the gated gross back up to (near) full exposure
    assert g_both.max() <= g_gate.max() + 0.15
    assert g_both.mean() <= g_gate.mean() * 1.5 + 1e-9


def test_ensemble_weights_no_hidden_leverage():
    """P1: a flat member (NaN sharpe) must get neutral weight — rows sum to 1, and the
    blended signal stays within [-1, 1]."""
    df = _ohlcv(500, seed=31)
    rets = pd.DataFrame({"a": df["close"].pct_change().fillna(0),
                         "b": 0.0,                                  # flat member
                         "c": -df["close"].pct_change().fillna(0)})
    from scripts.research.autoresearch import _ewma_softmax_weights
    for w in (_ewma_softmax_weights(rets), RG.regime_conditional_weights(rets, df["close"])):
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
        assert w.values.max() <= 1.0 + 1e-9


def test_downturn_slices_robust_indexes():
    """P2: non-DatetimeIndex and duplicate dates must not crash."""
    rng = np.random.default_rng(3)
    r = pd.Series(0.001 + 0.01 * rng.standard_normal(300))          # RangeIndex
    assert M.downturn_slices(r, r) == [] or isinstance(M.downturn_slices(r, r), list)
    idx = pd.DatetimeIndex(list(pd.bdate_range("2024-01-01", periods=150)) * 2)  # dup dates
    r2 = pd.Series(0.001 + 0.01 * rng.standard_normal(300), index=idx)
    assert isinstance(M.downturn_slices(r2, r2), list)               # no crash


def test_stacking_short_data_raises_value_error():
    """P2: too little data is a ValueError, not a fake 'lib missing' ImportError."""
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((12, 3)), rng.standard_normal(12)
    with pytest.raises(ValueError):
        Mdl.StackingModel(members=[Mdl.RidgeModel(1.0)]).fit(X, y)
