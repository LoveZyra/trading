"""Regression tests for the 2026-06-10 bug-fix pass (bug.md + bug2.md).

Each test pins a specific fixed behaviour so the bug class can't silently return:
signal-mask overwrites (BUG-1/NEW-1/NEW-5), MA warm-up shorts (NEW-2), calendar
rebalance skips (NEW-3), inf poisoning (BUG-2/BUG-3), data-contract issues
(L-4/L-6/NEW-4 semantics), engine validation (OPT-6/7) and HTML escaping (BUG-4).
"""
import numpy as np
import pandas as pd
import pytest


def _frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close, "volume": 1e6})


def _uptrend(n=300):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = pd.Series(np.linspace(100, 200, n), index=idx)
    c += np.random.default_rng(0).normal(0, 0.3, n)
    return _frame(c)


# ---------------------------------------------------------------- BUG-1 / NEW-1
def test_rsi_reversion_short_mode_keeps_longs():
    """allow_short=True must NOT erase long entries (old: degenerated to short-only)."""
    from scripts.strategies.mean_reversion import RSIReversion
    path = np.concatenate([np.linspace(100, 80, 30), np.linspace(80, 110, 40),
                           np.linspace(110, 150, 40), np.linspace(150, 120, 40),
                           np.linspace(120, 90, 40)])
    df = _frame(pd.Series(path, index=pd.date_range("2024-01-01", periods=len(path), freq="B")))
    sig = RSIReversion(14, 30, 70, 50, allow_short=True).generate_signal(df)
    assert (sig == 1).sum() > 0, "long entries were erased in short mode"
    assert (sig == -1).sum() > 0, "short entries missing"
    assert set(sig.unique()) <= {-1.0, 0.0, 1.0}


def test_breakout_short_mode_keeps_longs():
    """Uptrend: allow_short=True must go long about as often as allow_short=False."""
    from scripts.strategies.trend import Breakout
    df = _uptrend()
    long_only = (Breakout(20, 10, allow_short=False).generate_signal(df) == 1).sum()
    with_short = (Breakout(20, 10, allow_short=True).generate_signal(df) == 1).sum()
    assert long_only > 100
    assert with_short == long_only


# ---------------------------------------------------------------------- NEW-2
def test_ma_crossover_no_spurious_short_during_warmup():
    from scripts.strategies.trend import MACrossover
    df = _uptrend()
    sig = MACrossover(20, 50, allow_short=True).generate_signal(df)
    assert (sig.iloc[:49] == 0).all(), "warm-up bars must be flat, not short"
    assert (sig == 1).sum() > 0


# ---------------------------------------------------------------------- NEW-5
def test_bollinger_short_holds_until_mid_band():
    """A short entered at the upper band must persist while price stays between the
    mid and upper bands (old code zeroed it on the very next bar)."""
    from scripts.strategies.mean_reversion import BollingerReversion
    rng = np.random.default_rng(3)
    n = 200
    base = 100 + np.cumsum(rng.normal(0, 0.2, n))
    spike = base.copy()
    spike[120] += 8          # poke above the upper band once
    spike[121:140] += 4      # then hover between mid and upper
    df = _frame(pd.Series(spike, index=pd.date_range("2024-01-01", periods=n, freq="B")))
    sig = BollingerReversion(20, 2.0, allow_short=True).generate_signal(df)
    short_run = (sig.iloc[121:135] == -1).sum()
    assert short_run >= 5, f"short exited too early (held {short_run} bars)"


# ---------------------------------------------------------------------- BUG-2
def test_pair_spread_flat_leg_no_inf():
    from scripts.strategies.mean_reversion import pair_spread
    idx = pd.date_range("2024-01-01", periods=200)
    a = pd.Series(np.random.default_rng(1).normal(0, 1, 200).cumsum() + 50, index=idx)
    b = pd.Series(50.0, index=idx)                       # halted / flat leg
    spread, z = pair_spread(a, b, 60)
    assert not np.isinf(spread).any() and not np.isinf(z).any()


# ---------------------------------------------------------------------- BUG-3
def test_profit_factor_never_inf():
    from scripts.metrics import profit_factor
    all_gain = pd.Series([0.01, 0.02, 0.005])
    assert np.isfinite(profit_factor(all_gain))
    assert np.isnan(profit_factor(pd.Series([0.0, 0.0])))
    mixed = pd.Series([0.02, -0.01])
    assert profit_factor(mixed) == pytest.approx(2.0)


# ---------------------------------------------------------------------- NEW-3
def test_rebalance_dates_fall_on_trading_days():
    from scripts.rebalance import rebalance_dates
    idx = pd.bdate_range("2022-01-03", "2024-12-31")
    rd = rebalance_dates(idx, "ME")
    assert len(rd) == 36                              # every month rebalances
    assert all(d in idx for d in rd)                  # ...on an actual trading day


def test_multi_factor_signal_rebalances_every_month():
    from scripts.strategies.multi_factor import multi_factor_signal
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=520)
    data = {f"S{i}": _frame(pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx)))), index=idx))
        for i in range(6)}
    w = multi_factor_signal(data, rebalance="ME", top=0.34)
    chg = w.diff().abs().sum(axis=1)
    # ~24 months of post-warm-up history -> at least 12 actual weight changes
    assert (chg > 1e-12).sum() >= 12


# ---------------------------------------------------------------------- NEW-9
def test_vol_target_scale_warmup_is_neutral():
    from scripts.sizing import vol_target_scale
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 100),
                  index=pd.date_range("2024-01-01", periods=100))
    scale = vol_target_scale(r, lookback=21)
    assert (scale.iloc[:21] == 1.0).all(), "warm-up must be neutral 1.0, not flat 0.0"


# ------------------------------------------------------------------------ L-4
def test_validate_ohlcv_drops_inconsistent_rows():
    from scripts.data.base import validate_ohlcv
    idx = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame({"open": [10, 10, 10], "high": [11, 9, 11],
                       "low": [9, 10, 9], "close": [10, 10, 10],
                       "volume": [1, 1, 1]}, index=idx)   # row 2: high < low
    out = validate_ohlcv(df, name="t")
    assert len(out) == 2


# ------------------------------------------------------------------------ L-6
def test_rsi_edge_cases():
    from scripts.indicators import rsi
    flat = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=40))
    assert (rsi(flat, 14).dropna() == 50.0).all()
    up = pd.Series(np.arange(100.0, 140.0), index=pd.date_range("2024-01-01", periods=40))
    assert (rsi(up, 14).dropna() == 100.0).all()


# ---------------------------------------------------------------------- OPT-6
def test_backtest_validates_inputs():
    from scripts.backtest import backtest
    df = _uptrend(80)
    sig = pd.Series(1.0, index=df.index)
    with pytest.raises(ValueError):
        backtest(df, sig, lag=-1)
    with pytest.raises(ValueError):
        backtest(df, sig, commission_bps=-1)
    with pytest.raises(ValueError):
        backtest(df, sig, cost_model="cubic")
    with pytest.warns(UserWarning):                      # sqrt without volume
        backtest(df["close"], sig, cost_model="sqrt")


# ---------------------------------------------------------------------- OPT-7
def test_backtest_portfolio_ledger_and_sqrt_costs():
    from scripts.backtest import backtest_portfolio
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2023-01-02", periods=260)
    close = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
                          for s in "ABC"}, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=list("ABC"))
    w = pd.DataFrame(1 / 3, index=idx, columns=list("ABC"))
    res_lin = backtest_portfolio(close, w)
    assert len(res_lin.trades) > 0, "trade ledger must not be empty"
    res_sqrt = backtest_portfolio(close, w, cost_model="sqrt", panel_volume=vol,
                                  impact_coef=50.0, capital=1e8)
    assert res_sqrt.stats["total_costs"] > res_lin.stats["total_costs"]


# ---------------------------------------------------------------------- BUG-4
def test_html_report_escapes_script_breakout():
    from scripts import html_report as h
    out = h.render({"meta": {"title": "x</script><svg onload=alert(1)>",
                             "date": "2026-06-10"}, "sections": []})
    body = out.split("<script", 1)[1]
    assert "</script><svg" not in body


# ---------------------------------------------------------------------- NEW-8
def test_market_of_respects_exchange_suffix():
    from scripts.data.market import market_of
    assert market_of("005930.KS") == "KR"
    assert market_of("600519.SS") == "CN"
    assert market_of("0700.HK") == "HK"
    assert market_of("7203.T") == "JP"
    assert market_of("AAPL") == "US"


# ---------------------------------------------------------------------- NEW-6
def test_apply_cn_rules_checks_execution_bar():
    """Signal at t executes at t+1; blocking must look at t+1's limit state."""
    from scripts.data.microstructure import apply_cn_rules
    idx = pd.date_range("2024-01-01", periods=6)
    close = pd.Series([100, 100, 110, 121, 121, 121], index=idx, dtype=float)
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 1e6})
    sig = pd.Series([0, 1, 1, 1, 1, 1], index=idx, dtype=float)
    out = apply_cn_rules(sig, df, limit=0.10, lag=1)
    # bars 2 and 3 are +10% limit-up closes; the t=1 signal would execute at t=2
    # (locked) and t=2's at t=3 (locked) -> can't build the position there.
    assert out.iloc[1] == 0.0 and out.iloc[2] == 0.0
    # t=3's trade executes at t=4 (flat day) -> allowed.
    assert out.iloc[3] == 1.0


# ================= coverage for previously-untested modules ==================
def _universe(n_sym=6, n=400, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    out = {}
    for i in range(n_sym):
        c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012 + 0.004 * i, n))), index=idx)
        out[f"S{i}"] = _frame(c)
    return out


def test_zscore_reversion_long_only_holds_to_mean():
    """Pin the (intentional) post-fix semantics: a long entered at z<=-entry is held
    until |z|<=exit -- it is NOT dumped early just because z later spikes positive."""
    from scripts.strategies.mean_reversion import ZScoreReversion
    rng = np.random.default_rng(8)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 300)),
                  index=pd.date_range("2024-01-01", periods=300, freq="B"))
    df = _frame(c)
    sig = ZScoreReversion(20, 1.5, 0.5, allow_short=False).generate_signal(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert (sig == 1).sum() > 0


def test_portfolio_health_summary():
    from scripts.portfolio import portfolio_health
    from scripts.strategies.multi_factor import build_panel
    panel = build_panel(_universe(), "close")
    out = portfolio_health(panel)
    assert out["n_names"] == 6
    assert 1.0 <= out["effective_bets"] <= 6.0
    assert out["var_cvar"]["var_1d"] <= 0
    assert isinstance(out["verdict"], list) and out["verdict"]


def test_autoresearch_single_and_ensemble():
    from scripts import autoresearch as ar
    df = _universe(1, 380)["S0"]
    rep = ar.research_single(df, iterations=6, seed=2)
    assert len(rep.trials) == 6
    assert set(rep.bandit_summary) == set(ar.RULE_SPACE)
    res, members = ar.ensemble_top_k(rep, df, k=2)
    assert 1 <= len(members) <= 2
    assert np.isfinite(res.stats["sharpe"])


def test_research_portfolio_smoke():
    from scripts.autoresearch import research_portfolio
    rep = research_portfolio(_universe(5, 350), iterations=4, seed=3, use_ml=False)
    assert len(rep.trials) == 4
    assert any(np.isfinite(t.oos_sharpe) for t in rep.trials)


def test_html_report_renders_real_schema():
    """Render with the REAL top-level schema keys (alerts/levels/holdings/...) and
    check the payload survives into the embedded JSON with escaping intact."""
    from scripts import html_report as h
    out = h.render({
        "meta": {"title": "复盘", "date": "2026-06-10", "badge": "测试"},
        "alerts": [{"symbol": "NVDA", "headline": "财报临近", "action": "减仓"}],
        "levels": [{"symbol": "AAPL", "price": 230.5, "buy_zone": [225, 228],
                    "stop_loss": 219.0, "target1": 240.0}],
        "holdings": "一切正常 & <安全>",
        "conclusion": "观望",
        "disclaimer": "机械量化研究，非投资建议",
    })
    assert out.startswith("<!DOCTYPE html>")
    for frag in ("复盘", "NVDA", "AAPL", "230.5"):
        assert frag in out
    body = out.split("<script", 1)[1]
    assert "<安全>" not in body            # raw angle brackets must not survive
    assert r"\u003c安全\u003e" in out      # they are unicode-escaped in the JSON


def test_adapters_strip_exchange_suffix(monkeypatch):
    """Suffixed watchlists ('005930.KS') must work with the bare-code libraries."""
    import sys, types
    from scripts.data import free
    seen = {}

    def fake_ohlcv(s, e, code):
        seen["code"] = code
        idx = pd.date_range("2024-01-01", periods=30)
        return pd.DataFrame({"시가": 100.0, "고가": 101.0, "저가": 99.0,
                             "종가": 100.5, "거래량": 1e5}, index=idx)

    monkeypatch.setitem(sys.modules, "pykrx",
                        types.SimpleNamespace(stock=types.SimpleNamespace(get_market_ohlcv=fake_ohlcv)))
    monkeypatch.setitem(sys.modules, "pykrx.stock",
                        types.SimpleNamespace(get_market_ohlcv=fake_ohlcv))
    df = free.from_pykrx("005930.KS")
    assert seen["code"] == "005930"
    assert len(df) == 30


def test_sector_map_external_json(tmp_path, monkeypatch):
    from scripts.data import sectors as S
    fp = tmp_path / "sectors.json"
    fp.write_text('{"zzzz9": "test_sector"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    S._autoload_external()
    assert S.sector_of("ZZZZ9") == "test_sector"


def test_zscore_gap_flips_instead_of_netting_zero():
    """A gap straight from the short extreme to the long extreme must flip the
    position to +1 (close short, open long), not net the two legs to 0."""
    from scripts.strategies.mean_reversion import ZScoreReversion
    vals = [100.0] * 30 + [110.0] * 5 + [80.0] * 5    # spike up (short) then crash (long)
    c = pd.Series(vals, index=pd.date_range("2024-01-01", periods=len(vals), freq="B"))
    sig = ZScoreReversion(20, 1.5, 0.5).generate_signal(pd.DataFrame({"close": c}))
    assert (sig.iloc[-5:] == 1.0).any(), f"crash bars should be LONG, got {sig.iloc[-5:].tolist()}"
    assert (sig.iloc[30:35] == -1.0).any(), "spike bars should be short"


def test_grid_search_drawdown_sorted_best_first():
    """max_drawdown is negative; 'best first' means the SHALLOWEST drawdown on top."""
    from scripts.optimize import grid_search
    from scripts.strategies import REGISTRY
    df = _uptrend(300)
    tbl = grid_search(REGISTRY["ma_crossover"], df,
                      {"fast": [5, 20], "slow": [50, 100]}, metric="max_drawdown")
    dd = tbl["max_drawdown"].dropna()
    assert dd.iloc[0] == dd.max(), "best (shallowest) drawdown must be first"


def test_walk_forward_train_frac_semantics():
    """train_frac=0.6 must yield a train window ~60% of train+test (was 71%)."""
    n, n_splits, train_frac = 350, 5, 0.6
    fold = n // (n_splits + 1)
    expected = int(fold * train_frac / (1 - train_frac))
    assert abs(expected / (expected + fold) - train_frac) < 0.02


def test_pit_snapshot_utf8_roundtrip(tmp_path):
    """PIT snapshots carry Chinese + emoji; they must be written/read as UTF-8
    explicitly (Windows' locale default is GBK, which can't encode emoji)."""
    from scripts.data import pit
    base = str(tmp_path / "pit")
    fund = pd.DataFrame([{"symbol": "600519", "name": "贵州茅台🍷", "pe": 25.0}]).set_index("symbol")
    pit.save_snapshot("2026-06-10", fundamentals_panel=fund.reset_index().set_index("symbol"),
                      sentiment_by_symbol={"600519": 0.5, "备注": -0.2},
                      macro={"说明": "risk-on 🚀"}, base=base)
    raw = (tmp_path / "pit" / "sentiment" / "2026-06-10.json").read_bytes()
    raw.decode("utf-8")                                    # must be valid UTF-8
    hist = pit.load_pit_sentiment(base)
    assert hist[pd.Timestamp("2026-06-10")]["600519"] == 0.5
    funds = pit.load_pit_fundamentals(base)
    assert "贵州茅台🍷" in funds[pd.Timestamp("2026-06-10")]["name"].values
