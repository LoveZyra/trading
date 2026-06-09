"""Macro risk-on / risk-off layer — real indicators, not just headlines.

Until now the 'market' layer of composite sentiment was only news TEXT (Fed/CPI
headlines scored by the lexicon). This module turns HARD macro data into a single
risk score in [-1, 1] (negative = risk-off / defensive, positive = risk-on), so the
macro layer reflects what actually moves markets:

  * VIX (fear index): elevated level or a spike -> risk-off.
  * Treasury yields (10y, 2y): rising yields are a headwind, esp. for growth/tech.
  * Yield curve (10y - 2y): inverted -> recession risk -> risk-off.
  * Economic surprises (CPI / PPI / NFP / unemployment): actual vs consensus, mapped
    to an equity-risk direction. Inflation/deflation are read off CPI/PPI trend.
  * War / geopolitics: inherently news-based -> folded in via a geo_sentiment number
    (score geopolitics headlines with data.sentiment and pass it here).

How to fetch the inputs (Claude side, hand-off):
  - VIX:  broker search_contracts("VIX", security_type="IND") + get_price_history,
          or yfinance "^VIX".  10y/2y yields: yfinance "^TNX"/"^FVX"/"^TYX" (÷10 for %),
          or use ETFs TLT/IEF/SHY as proxies via the broker.
  - CPI/PPI/NFP/UNEMP: the broker has no econ calendar, so Web-search the latest
          release ("US CPI June 2026 actual vs consensus") and pass the numbers to
          econ_surprise().

These macro->equity relationships are REGIME-DEPENDENT (a hot NFP can be risk-on in a
soft-landing regime and risk-off when the Fed is hawkish). The defaults below assume a
'data-dependent Fed / inflation-sensitive' regime; override the signs if the regime
changes. Treat this as a risk overlay, not a precise forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sentiment as _sent

# Where to fetch each input (for convenience / documentation).
MACRO_TICKERS = {
    "vix": {"yfinance": "^VIX", "broker": ("VIX", "IND")},
    "y10": {"yfinance": "^TNX", "note": "÷10 to get percent"},
    "y2": {"yfinance": "^FVX", "note": "5y proxy; ^UST2YR if available, ÷10"},
    "dollar": {"yfinance": "DX-Y.NYB"},
    "treasury_etf": {"broker": ("TLT", "STK")},  # long bond ETF as a rates proxy
}

# Default equity-risk sign of an UPSIDE surprise for each release.
# +1 means "actual > consensus is risk-ON for equities"; -1 means risk-OFF.
INDICATOR_SIGN = {
    "cpi": -1, "core_cpi": -1, "ppi": -1, "pce": -1,     # hot inflation -> risk-off
    "nfp": -0.3, "nonfarm": -0.3,                         # hot jobs -> hawkish tilt (mild risk-off)
    "unemployment": -1,                                   # higher unemployment -> risk-off
    "gdp": +0.7, "retail_sales": +0.5, "ism": +0.6,      # stronger growth -> risk-on
    "consumer_confidence": +0.4,
}


def _tanh(x):
    return float(np.tanh(x))


def vix_signal(vix_close: pd.Series, lookback: int = 252) -> float:
    """Risk score from the VIX (negative = fearful). Combines the absolute level
    (band-mapped) with a spike term (current vs its recent average)."""
    v = float(vix_close.iloc[-1])
    # level bands -> base score
    if v < 14:
        base = 0.4
    elif v < 18:
        base = 0.15
    elif v < 22:
        base = 0.0
    elif v < 28:
        base = -0.35
    elif v < 35:
        base = -0.7
    else:
        base = -1.0
    # spike term: current vs trailing median
    hist = vix_close.tail(lookback)
    med = float(hist.median()) if len(hist) else v
    spike = _tanh((med - v) / max(med * 0.5, 1e-6))   # v>med -> negative
    return max(-1.0, min(1.0, 0.6 * base + 0.4 * spike))


def rates_signal(y10_close: pd.Series, lookback: int = 63) -> float:
    """Risk score from the 10y yield trend. Rising yields over `lookback` bars are a
    headwind for equities (esp. long-duration/growth) -> negative."""
    if len(y10_close) <= lookback:
        return 0.0
    chg = float(y10_close.iloc[-1] - y10_close.iloc[-lookback])   # in yield points
    return max(-1.0, min(1.0, -_tanh(chg / 0.5)))                 # +0.5pp ~ -0.46


def curve_signal(y10_close: pd.Series, y2_close: pd.Series) -> float:
    """Yield-curve slope (10y - 2y). Inverted (<0) -> recession risk -> risk-off."""
    try:
        slope = float(y10_close.iloc[-1] - y2_close.iloc[-1])
    except Exception:  # noqa: BLE001
        return 0.0
    return max(-1.0, min(1.0, _tanh(slope / 0.5)))               # inverted -> negative


def econ_surprise(releases: list[dict]) -> float:
    """Aggregate economic-release surprises into a risk score.

    releases: list of {"name": "cpi"/"ppi"/"nfp"/..., "actual": x, "consensus": y}
              (optionally "sign" to override INDICATOR_SIGN). Surprise = (actual -
              consensus)/|consensus|, multiplied by the indicator's equity-risk sign.
    """
    if not releases:
        return 0.0
    vals = []
    for r in releases:
        name = str(r.get("name", "")).lower().replace(" ", "_")
        a, c = r.get("actual"), r.get("consensus")
        if a is None or c is None or c == 0:
            continue
        sign = r.get("sign", INDICATOR_SIGN.get(name, 0.0))
        surprise = (float(a) - float(c)) / abs(float(c))
        vals.append(sign * _tanh(surprise * 5))     # scale: 20% surprise ~ saturates
    return float(np.clip(np.mean(vals), -1.0, 1.0)) if vals else 0.0


def macro_score(vix: pd.Series | None = None, y10: pd.Series | None = None,
                y2: pd.Series | None = None, releases: list | None = None,
                geo_sentiment: float | None = None, upcoming: list | None = None,
                today=None, weights: dict | None = None) -> dict:
    """Blend whatever macro inputs you have into one risk-on/off score in [-1, 1].

    Returns {"score": .., plus each component} so a report can show the breakdown.
    Only the components you pass are used; weights renormalize over those present.
    geo_sentiment: a [-1,1] number from scoring war/geopolitics headlines with
    data.sentiment (e.g. negative when escalation news dominates).
    """
    w = {"vix": 0.28, "rates": 0.18, "curve": 0.14, "econ": 0.18, "geo": 0.12, "event": 0.10}
    if weights:
        w.update(weights)
    comp = {}
    if vix is not None and len(vix):
        comp["vix"] = vix_signal(vix)
    if y10 is not None and len(y10):
        comp["rates"] = rates_signal(y10)
    if y10 is not None and y2 is not None and len(y10) and len(y2):
        comp["curve"] = curve_signal(y10, y2)
    if releases:
        comp["econ"] = econ_surprise(releases)
    if geo_sentiment is not None:
        comp["geo"] = float(geo_sentiment)
    pe = None
    if upcoming:
        pe = pre_event_risk(upcoming, today=today)
        comp["event"] = pe["score"]
    if not comp:
        return {"score": 0.0}
    tot = sum(w[k] for k in comp) or 1.0
    score = sum(w[k] * comp[k] for k in comp) / tot
    out = {"score": round(float(np.clip(score, -1, 1)), 3)}
    out.update({k: round(v, 3) for k, v in comp.items()})
    out["regime"] = ("risk-off/避险" if score < -0.25 else
                     "risk-on/偏多" if score > 0.25 else "中性")
    if upcoming and pe is not None:
        out["imminent_events"] = pe["imminent"]
    return out


# ---- forward-looking economic calendar (consider releases BEFORE they print) -
HIGH_IMPACT = {
    "cpi", "core_cpi", "ppi", "pce", "core_pce", "nfp", "nonfarm", "payrolls",
    "fomc", "rate_decision", "fed", "unemployment", "jobs", "gdp", "jackson_hole",
}


def pre_event_risk(upcoming: list, today=None, window_days: int = 4) -> dict:
    """Forward-looking risk from SCHEDULED macro releases not yet out.

    `upcoming`: [{"name":"cpi","date":"2026-06-11"}, {"name":"fomc","date":"2026-06-17"}]
    — get these from a Web-search of the econ calendar. Markets tend to DE-RISK into a
    big print (CPI/PPI/NFP/FOMC), so if a HIGH_IMPACT event lands within `window_days`
    ahead, this returns a small risk-off nudge that grows as the event nears, plus the
    list of imminent events so the report can warn "CPI 明天公布，盘前注意".

    Returns {"score": <=0, "imminent": [{name,date,days}], "next": {...}}.
    """
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.utcnow().normalize().tz_localize(None)
    imminent, worst = [], 0.0
    for e in upcoming or []:
        name = str(e.get("name", "")).lower().replace(" ", "_")
        d = pd.to_datetime(e.get("date"), errors="coerce")
        if pd.isna(d):
            continue
        days = (d.normalize() - today).days
        if 0 <= days <= window_days and any(h in name for h in HIGH_IMPACT):
            # closer event => bigger de-risk nudge (0 days ~ -0.5, window edge ~ -0.1)
            nudge = -0.5 * (1 - days / (window_days + 1))
            worst = min(worst, nudge)
            imminent.append({"name": name, "date": d.strftime("%Y-%m-%d"), "days": int(days)})
    imminent.sort(key=lambda x: x["days"])
    return {"score": round(worst, 3), "imminent": imminent,
            "next": imminent[0] if imminent else None}
