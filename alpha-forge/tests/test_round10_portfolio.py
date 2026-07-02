"""Round-10 D: portfolio optimization family, factor tearsheet, and metric add-ons.

Covers scripts/risk/optimization.py, scripts/reporting/factor_tearsheet.py, and the
appended kelly_* (sizing), mae_mfe/capm_decompose (metrics), brinson (attribution).
Analytic cases are checked against hand-computed / closed-form answers; constraint
solvers are checked for feasibility invariants rather than exact weights.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.core.metrics import capm_decompose, mae_mfe
from scripts.reporting.attribution import brinson
from scripts.reporting.factor_tearsheet import factor_tearsheet, tearsheet_data
from scripts.risk.optimization import (
    black_litterman,
    efficient_frontier,
    max_sharpe_weights,
    mean_variance_constrained,
    min_variance_weights,
    views_from_predictions,
)
from scripts.risk.sizing import kelly_fraction, kelly_weights


def _rand_cov(n=6, seed=42):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.02, size=(300, n)) @ np.diag(np.linspace(0.5, 1.5, n))
    r[:, 1] = 0.9 * r[:, 0] + 0.1 * r[:, 1]      # a correlated pair -> shorts unconstrained
    names = [f"A{i}" for i in range(n)]
    return pd.DataFrame(np.cov(r, rowvar=False), index=names, columns=names)


# ---------------------------------------------------------------- optimization

def test_min_variance_diagonal_matches_inverse_variance():
    # For diagonal Σ the GMV solution is w_i ∝ 1/σ_i², in closed form.
    var = np.array([0.04, 0.01, 0.0225])
    cov = pd.DataFrame(np.diag(var), index=list("ABC"), columns=list("ABC"))
    w = min_variance_weights(cov)
    expected = (1 / var) / (1 / var).sum()
    np.testing.assert_allclose(w.values, expected, rtol=1e-5)
    assert abs(w.sum() - 1) < 1e-9


def test_min_variance_long_only_and_cap():
    cov = _rand_cov()
    w = min_variance_weights(cov, long_only=True, weight_cap=0.35)
    assert float(w.min()) >= -1e-9
    assert float(w.max()) <= 0.35 + 1e-9
    assert abs(float(w.sum()) - 1) < 1e-6
    # unconstrained solution on this cov really does want shorts (test is meaningful)
    w_free = min_variance_weights(cov, long_only=False)
    assert float(w_free.min()) < 0


def test_max_sharpe_matches_analytic_tangency():
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=list("AB"), columns=list("AB"))
    mu = pd.Series([0.08, 0.12], index=list("AB"))
    w = max_sharpe_weights(cov, mu, long_only=False)
    x = np.linalg.solve(cov.values + np.eye(2) * 1e-8, mu.values)
    np.testing.assert_allclose(w.values, x / x.sum(), rtol=1e-6)


def test_efficient_frontier_monotone():
    cov = _rand_cov(5, seed=7)
    mu = pd.Series(np.linspace(0.04, 0.12, 5), index=cov.columns)
    ef = efficient_frontier(cov, mu, n_points=30)
    assert list(ef.columns) == ["ret", "vol", "sharpe", "weights"]
    ret, vol = ef["ret"].values, ef["vol"].values
    assert (np.diff(ret) > 0).all()                    # sweeping targets upward
    assert (np.diff(vol) > -1e-10).all()               # upper branch: vol rises with ret
    assert np.isfinite(ef["sharpe"]).all()
    w0 = pd.Series(ef["weights"].iloc[0])
    assert abs(w0.sum() - 1) < 1e-6                    # every point fully invested


def test_black_litterman_no_views_recovers_market():
    cov = _rand_cov(4, seed=3)
    w_mkt = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.columns)
    out = black_litterman(cov, w_mkt, risk_aversion=3.0)
    pi = 3.0 * (cov.values + np.eye(4) * 1e-8) @ w_mkt.values
    np.testing.assert_allclose(out["posterior_mu"].values, pi, rtol=1e-6)
    np.testing.assert_allclose(out["weights"].values, w_mkt.values, atol=1e-9)


def test_black_litterman_single_view_pulls_asset():
    cov = pd.DataFrame(np.diag([0.04, 0.03, 0.05]), index=list("ABC"), columns=list("ABC"))
    w_mkt = pd.Series([1 / 3] * 3, index=list("ABC"))
    base = black_litterman(cov, w_mkt)
    pi_a = float(base["posterior_mu"]["A"])
    P, Q, om = views_from_predictions(pd.Series({"A": pi_a + 0.05}), conf=0.9)
    out = black_litterman(cov, w_mkt, P, Q, omega=om)
    assert float(out["posterior_mu"]["A"]) > pi_a          # bullish view lifts A's mu
    assert float(out["weights"]["A"]) > float(base["weights"]["A"])  # and its weight
    # diagonal cov: untouched assets keep their implied returns
    np.testing.assert_allclose(out["posterior_mu"]["B"], base["posterior_mu"]["B"], rtol=1e-9)


def test_mean_variance_constrained_respects_caps():
    cov = _rand_cov(8, seed=11)
    mu = pd.Series(np.linspace(-0.02, 0.15, 8), index=cov.columns)
    sector_map = {n: ("T" if i < 4 else "F") for i, n in enumerate(cov.columns)}
    w = mean_variance_constrained(cov, mu, weight_cap=0.2,
                                  sector_caps={"F": 0.45}, sector_map=sector_map)
    assert abs(float(w.sum()) - 1) < 1e-6
    assert float(w.min()) >= -1e-9
    assert float(w.max()) <= 0.2 + 1e-6
    f_tot = float(w[[n for n in w.index if sector_map[n] == "F"]].sum())
    assert f_tot <= 0.45 + 1e-6
    # infeasible cap must raise, not return a silently-broken book
    with pytest.raises(ValueError):
        mean_variance_constrained(cov, mu, weight_cap=0.05)


# ---------------------------------------------------------------- kelly

def test_kelly_fraction_hand_computed():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)      # 0.6 - 0.4/2
    assert kelly_fraction(0.9, 5.0) == pytest.approx(0.5)      # 0.88 hits the cap
    assert kelly_fraction(0.9, 5.0, cap=0.25) == pytest.approx(0.25)
    assert kelly_fraction(0.3, 1.0) == 0.0                     # negative edge -> no bet
    assert kelly_fraction(0.5, 0.0) == 0.0                     # degenerate payoff


def test_kelly_weights_gross_one():
    rng = np.random.default_rng(5)
    r = pd.DataFrame(rng.normal(0.001, 0.02, (400, 4)), columns=list("ABCD"))
    w = kelly_weights(r, cap=0.5)
    assert w.abs().sum() == pytest.approx(1.0)
    assert list(w.index) == list("ABCD")
    # zero-edge book -> all zeros, not a fabricated allocation
    flat = pd.DataFrame(0.0, index=range(50), columns=list("AB"))
    assert kelly_weights(flat).abs().sum() == 0.0


# ---------------------------------------------------------------- metrics

def test_mae_mfe_hand_computed():
    idx = pd.date_range("2024-01-01", periods=4)
    px = pd.Series([100.0, 110.0, 90.0, 105.0], index=idx)
    trades = pd.DataFrame({"date": [idx[0], idx[3]],
                           "from_pos": [0.0, 1.0], "to_pos": [1.0, 0.0]})
    seg = mae_mfe(trades, px)
    assert len(seg) == 1
    row = seg.iloc[0]
    assert row["mfe"] == pytest.approx(0.10)     # peak 110 vs entry 100
    assert row["mae"] == pytest.approx(-0.10)    # trough 90
    assert row["ret"] == pytest.approx(0.05)     # exit 105
    # short segment: excursions flip sign
    trades_s = pd.DataFrame({"date": [idx[0], idx[3]],
                             "from_pos": [0.0, -1.0], "to_pos": [-1.0, 0.0]})
    seg_s = mae_mfe(trades_s, px)
    assert seg_s.iloc[0]["mfe"] == pytest.approx(0.10)   # drop to 90 favors the short
    assert seg_s.iloc[0]["ret"] == pytest.approx(-0.05)


def test_capm_beta_one_portfolio_has_zero_alpha():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    mkt = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)
    out = capm_decompose(mkt, mkt)
    assert out["beta"] == pytest.approx(1.0, abs=1e-9)
    assert out["alpha_ann"] == pytest.approx(0.0, abs=1e-9)
    assert out["corr"] == pytest.approx(1.0)
    assert out["r2"] == pytest.approx(1.0)
    # constant positive tilt shows up as alpha with beta still 1
    out2 = capm_decompose(mkt + 0.0002, mkt)
    assert out2["beta"] == pytest.approx(1.0, abs=1e-6)
    assert out2["alpha_ann"] == pytest.approx(0.0002 * 252, rel=1e-6)
    assert out2["information_ratio"] > 0


# ---------------------------------------------------------------- attribution

def test_brinson_effects_sum_to_excess():
    syms = list("ABCDEF")
    sector_map = dict(zip(syms, ["T", "T", "T", "F", "F", "E"]))
    rng = np.random.default_rng(21)
    wp = pd.Series(rng.random(6), index=syms); wp /= wp.sum()
    wb = pd.Series(rng.random(6), index=syms); wb /= wb.sum()
    rp = pd.Series(rng.normal(0.01, 0.05, 6), index=syms)
    rb = pd.Series(rng.normal(0.01, 0.05, 6), index=syms)
    out = brinson(wp, wb, rp, rb, sector_map=sector_map)
    excess = float((wp * rp).sum() - (wb * rb).sum())
    assert out["allocation"] + out["selection"] + out["interaction"] == pytest.approx(excess)
    assert out["total"] == pytest.approx(excess)
    assert out["by_sector"]["total"].sum() == pytest.approx(excess)
    assert set(out["by_sector"].index) == {"T", "F", "E"}


# ---------------------------------------------------------------- tearsheet

@pytest.fixture
def factor_and_close():
    """12 symbols x 180 days; factor = leaked-then-noised next-day return, so the
    tearsheet has a real signal to detect (positive RankIC at h=1)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=180, freq="B")
    syms = [f"S{i}" for i in range(12)]
    close = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(idx)))) for s in syms},
        index=idx)
    fwd1 = close.shift(-1) / close - 1
    factor = fwd1 * 0.5 + pd.DataFrame(rng.normal(0, 0.02, close.shape),
                                       index=idx, columns=syms)
    return factor, close


def test_tearsheet_data_shapes_and_signal(factor_and_close):
    factor, close = factor_and_close
    d = tearsheet_data(factor, close, quantiles=5, horizons=(1, 5, 21))
    assert d["quantile_cum"].shape[1] == 5
    assert set(d["ic_table"].index) == {1, 5, 21}
    assert d["ic_table"].loc[1, "RankIC"] > 0.1            # embedded signal detected
    assert 0.0 <= d["turnover_monthly"] <= 1.0
    # top quantile of a positive-IC factor should out-compound the bottom
    assert d["quantile_cum"].iloc[-1, 4] > d["quantile_cum"].iloc[-1, 0]
    assert float(d["ls_cum"].iloc[-1]) > 1.0
    assert d["meta"]["curve_horizon"] == 1


def test_tearsheet_html_written_with_key_sections(tmp_path, factor_and_close):
    factor, close = factor_and_close
    out = tmp_path / "ts.html"
    html = factor_tearsheet(factor, close, name="unit", out_path=str(out))
    assert out.exists() and out.stat().st_size > 5000
    for token in ["分位累计收益", "RankIC", "多空价差累计", "月均换手", "分位统计",
                  "不扣交易成本", "<svg", "<polyline"]:
        assert token in html, f"missing tearsheet section: {token}"
    assert out.read_text(encoding="utf-8") == html
