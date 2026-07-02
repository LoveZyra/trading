"""AI 选股横截面能力的离线测试:植入信号可被检出、噪声≈0、无前视、组池护栏、报告渲染。
只用 numpy/pandas,无网络/无 torch。"""
import warnings
import numpy as np, pandas as pd, pytest
from scripts.xsec import universe, panel as PN, xsec_eval, xsec_autoresearch, xsec_report

def _make_panel(n_sym=40, T=420, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=T, freq="B")
    mu = rng.normal(0, 0.0015, n_sym) if signal else np.zeros(n_sym)   # 持续漂移=横截面动量信号
    data = {}
    for i in range(n_sym):
        r = mu[i] + rng.normal(0, 0.01, T)
        c = 100 * np.exp(np.cumsum(r))
        data[f"S{i:02d}"] = pd.DataFrame({"open": c, "high": c*1.01, "low": c*0.99, "close": c, "volume": 1e6}, index=idx)
    return data

def test_universe_list_dedupe():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert universe.from_list([" aaa", "AAA", "bbb"]) == ["AAA", "BBB"]

def test_universe_breadth_warnings():
    with pytest.warns(UserWarning):
        universe.from_list(["X", "Y"])                                  # <30 只
    with pytest.warns(UserWarning):
        universe.from_list(["A", "B", "C"], sectors={"A": "t", "B": "t", "C": "t"})  # 单板块

def test_panel_factors_are_cross_section_z():
    P = PN.price_factor_panels(_make_panel(n_sym=30, T=200))
    assert "mom20" in P and isinstance(P["mom20"], pd.DataFrame)
    row = P["mom20"].dropna(how="all").iloc[-1].dropna()
    assert abs(float(row.mean())) < 1e-6 + 0.2                          # 截面 z-score,均值≈0

def test_signal_detected_and_beats_noise():
    sig = xsec_eval.evaluate_cross_section(_make_panel(signal=True, seed=1), horizon=21, rebalance="ME", min_names=10)
    noi = xsec_eval.evaluate_cross_section(_make_panel(signal=False, seed=2), horizon=21, rebalance="ME", min_names=10)
    assert sig["scorecard"]["n_dates"] > 0
    assert sig["scorecard"]["RankIC"] > 0.05                            # 植入动量被检出
    assert sig["scorecard"]["RankIC"] > noi["scorecard"]["RankIC"]      # 信号 > 噪声

def test_no_lookahead_all_fwd_realized():
    res = xsec_eval.evaluate_cross_section(_make_panel(seed=3), horizon=21, rebalance="ME", min_names=10)
    assert res["preds"]["fwd"].notna().all()                           # 进评测的样本前向收益都已实现

def test_autoresearch_leaderboard_sorted():
    lb = xsec_autoresearch.search(_make_panel(signal=True, seed=4), horizon=21, rebalance="ME", top_n=6, min_names=10)
    assert len(lb) > 0 and "RankICIR" in lb.columns
    vals = lb["RankICIR"].dropna().values
    assert (np.diff(vals) <= 1e-9).all()                               # 降序

def test_report_renders():
    res = xsec_eval.evaluate_cross_section(_make_panel(seed=5), horizon=21, rebalance="ME", min_names=10)
    assert "RankIC" in xsec_report.scorecard_markdown(res)
    assert "横截面" in xsec_report.scorecard_html(res)

def test_render_html_report():
    from scripts.xsec import xsec_report
    res = xsec_eval.evaluate_cross_section(_make_panel(signal=True, seed=9), horizon=21, rebalance="ME", min_names=10)
    cr = [{"symbol": f"S{i:02d}", "sector": "x", "score": 0.2 - 0.01*i} for i in range(40)]
    h = xsec_report.render_html_report({"H=21": res, "H=5": res}, cr, title="T", subtitle="s", out_path=None)
    assert "<html" in h and "最终排名" in h and "S00" in h and "诚实红线" in h
