"""Tests for the broker hand-off (from_columnar on a realistic payload) and the new
cross-sectional compare_tickers convenience — both exercised live during the
SK海力士/三星/美光 analysis that motivated them."""
import numpy as np
import pandas as pd
import pytest


def test_from_columnar_handles_iso_times_and_zero_gaps():
    """The real get_price_history payload has ISO-8601 'time' strings and occasional
    0-value gap bars. from_columnar + validate_ohlcv must parse the dates and DROP the
    zero-close rows (not feed 0 into indicators)."""
    from scripts.data.ibkr import from_columnar
    # row 2 is a gap bar: close==0 but the other fields are real (as the broker ships
    # them) -> must drop on the non-positive-close rule, leaving 3 clean bars.
    payload = {
        "time": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z"],
        "open": [10.0, 11.0, 11.2, 12.0], "high": [10.5, 11.5, 11.7, 12.6],
        "low": [9.5, 10.5, 10.8, 11.8], "close": [10.0, 0.0, 11.4, 12.4],
        "volume": [100, 0, 110, 120],
    }
    df = from_columnar(payload, name="test")
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None
    assert (df["close"] > 0).all()                 # the 0-close gap row dropped
    assert len(df) == 3                            # 4 in, 1 zero-close dropped
    assert df["close"].tolist() == [10.0, 11.4, 12.4]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@pytest.fixture
def universe():
    idx = pd.date_range("2024-01-01", periods=160, freq="B")
    out = {}
    for i, s in enumerate(["AAA", "BBB", "CCC"]):
        c = 100 * np.exp(np.cumsum(np.random.default_rng(i + 5).normal(0.0006 - 0.0002 * i, 0.02, 160)))
        out[s] = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx)
    return out


def test_compare_tickers_shape_and_ranking(universe):
    from scripts.research import compare
    cmp = compare.compare_tickers(universe, bars_per_year=252)
    t = cmp["table"]
    assert set(t.index) == {"AAA", "BBB", "CCC"}
    for col in ["last", "rsi", "ret_1y_%", "ann_vol_%", "pct_from_high_%", "trend"]:
        assert col in t.columns
    assert cmp["correlation"].shape == (3, 3)
    assert len(cmp["rs_rank"]) == 3
    # rs_rank must be sorted by trailing 1y return, descending
    scores = [cmp["rs_scores"][n] for n in cmp["rs_rank"]]
    assert scores == sorted(scores, reverse=True)
    assert 1.0 <= float(cmp["effective_bets"]) <= 3.0


def test_compare_tickers_weekly_bars(universe):
    """Must also work on weekly bars via bars_per_year=52 (the memory-stock case)."""
    from scripts.research import compare
    wk = {k: v.resample("W-MON").last().dropna() for k, v in universe.items()}
    cmp = compare.compare_tickers(wk, bars_per_year=52)
    assert set(cmp["table"].index) == {"AAA", "BBB", "CCC"}
    assert cmp["table"]["ann_vol_%"].notna().all()


def test_html_report_levels_show_pct_and_support():
    """The levels table must render %-vs-current-price (buy/support/stop/target) and a
    support column — both added 2026-06 after they were missing from a live report."""
    from scripts.reporting import html_report as H
    rep = {"meta": {"title": "t", "report_type": "market"},
           "levels": [{"symbol": "X", "price": 1000, "signal": "watch",
                       "buy_low": 850, "buy_high": 920, "support1": 820, "support2": 700,
                       "stop": 780, "target": 1100, "rr": 1.4, "rsi": 70}]}
    html = H.render(rep)
    assert "function vsPrice" in html and "function pctStr" in html   # %-vs-price helpers
    assert "c.supp" in html and "支撑" in html                         # support column
    assert "support1" in html                                          # support data embedded
    assert html.strip().startswith("<!DOCTYPE") and html.strip().endswith("</html>")


def test_newsfeed_to_alerts_and_group():
    """news connector rows -> report alerts + 🗞 news group (the 2026-06 news hand-off)."""
    from scripts.reporting import newsfeed as NF
    items = [{"date": "2026-06-23", "headline": "Chip rout drags Nasdaq", "symbol": "宏观",
              "name": "芯片板块", "level": "high", "detail": "d", "action": "a"},
             {"date": "2026-06-22", "headline": "Micron signs AI deal with Anthropic"},
             {"headline": ""}]                       # skipped (no headline)
    al = NF.to_alerts(items)
    assert len(al) == 2
    assert al[0]["symbol"] == "宏观" and al[0]["level"] == "high"
    assert al[1]["level"] == "mid" and al[1]["signal"] == "watch" and al[1]["symbol"] == "新闻"
    g = NF.to_news_group(items, source="MT Newswires")
    assert g["title"].startswith("🗞") and "Anthropic" in g["body"] and "2026-06-23" in g["body"]
    # and it actually renders inside a report
    from scripts.reporting import html_report as H
    html = H.render({"meta": {"title": "t", "report_type": "market"}, "alerts": al, "groups": [g]})
    assert "Anthropic" in html and html.strip().endswith("</html>")


def test_symbol_order_and_single_env_panel():
    """symbol_order must drive a consistent company order (JS sort + embedded data); a
    lone env panel must go full-width (no empty right column) — the 2026-06 layout fixes."""
    from scripts.reporting import html_report as H
    rep = {"meta": {"title": "t", "report_type": "market"},
           "symbol_order": ["000660", "005930", "MU"],
           "macro": {"title": "M", "risk_score": -0.3, "vix": 20, "rows": []},
           "alerts": [{"symbol": "宏观", "headline": "m"}, {"symbol": "MU", "headline": "mu"}],
           "levels": [{"symbol": "MU", "price": 1, "signal": "watch"},
                      {"symbol": "000660", "price": 1, "signal": "watch"}]}
    html = H.render(rep)
    assert "symbol_order" in html                 # canonical order embedded for the client sort
    assert "_stable" in html                      # stable-sort logic present
    assert "env-grid one" in html                 # single env panel renders full-width
    assert html.strip().endswith("</html>")


def test_verdict_five_levels():
    """综合立场 must support 5 levels with multi-arrow strength + a net-leaning score and an
    explicit score override (2026-06 upgrade)."""
    from scripts.reporting import html_report as H
    html = H.render({"meta": {"title": "t", "report_type": "market"},
                     "verdict": {"stance": "强烈看多 · 满仓", "action": "a"}})
    assert "强烈看多" in html and "强烈看空" in html and "偏空" in html   # level names embedded
    assert "▲▲▲" in html and "▼▼▼" in html                              # multi-arrow strength
    assert "v.score" in html                                            # explicit override path
    assert "bull - bear" in html                                       # net-leaning (not first-match)
    assert html.strip().endswith("</html>")


def test_classify_signal_rules():
    from scripts.risk.levels import classify_signal as cs
    # clean long: uptrend, not overbought, good R/R, in the buy zone
    assert cs(rsi=50, rr=2.0, trend="bull", in_buy_zone=True) == "long"
    assert cs(rsi=82, rr=2.0, trend="bull", in_buy_zone=True) == "watch"   # overbought
    assert cs(rsi=50, rr=0.8, trend="bull", in_buy_zone=True) == "watch"   # poor R/R
    assert cs(rsi=50, rr=2.0, trend="bull", in_buy_zone=False) == "watch"  # extended above zone
    assert cs(rsi=50, rr=2.0, trend="bull", in_buy_zone=True, event=True) == "watch"     # event pending
    assert cs(rsi=50, rr=2.0, trend="bull", in_buy_zone=True, leveraged=True) == "watch" # leveraged
    assert cs(rsi=55, trend="bear") == "short"
    assert cs(rsi=25, trend="bear") == "watch"            # oversold downtrend
    assert cs(rsi=50, rr=2.0, trend="flat", in_buy_zone=True) == "watch"


def test_sigbadge_event_sentiment_with_horizon():
    """Signal badge supports AI-judged event sentiment + holding period (利多·短线 等),
    colored by sentiment word; legacy long/watch/short still work."""
    from scripts.reporting import html_report as H
    html = H.render({"meta": {"title": "t", "report_type": "market"},
                     "alerts": [{"symbol": "MU", "signal": "利多·中线", "headline": "x"},
                                {"symbol": "X", "signal": "利空·短线", "headline": "y"},
                                {"symbol": "Z", "signal": "中性", "headline": "z"}]})
    assert "利多·中线" in html and "利空·短线" in html and "中性" in html   # verbatim labels
    assert "利多|利好" in html                                            # sentiment-coloring regex in JS
    assert "做多" in html and "观望" in html and "做空" in html            # legacy codes intact
    assert html.strip().endswith("</html>")
