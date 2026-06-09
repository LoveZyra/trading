"""Cross-sectional multi-factor stock selection.

Ranks a *universe* of stocks each rebalance date by a blend of factors, longs the
top bucket (optionally shorts the bottom), and equal-weights within the bucket.
The workhorse of systematic equity: momentum + low-vol (price) plus value + quality
+ growth (fundamentals) plus news sentiment, combined by z-score.

Price inputs are a dict {symbol: OHLCV frame}. Fundamental factors come from a
fundamentals panel (data.fundamentals.load_panel); the news factor from a
{symbol: mean_sentiment} dict (data.news + data.sentiment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind


def _cross_section_z(s: pd.Series) -> pd.Series:
    """Standardize a factor across the universe at one point in time."""
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def momentum_factor(panel_close: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """12-1 style momentum: trailing return excluding the most recent `skip` bars
    (skipping the last month avoids the short-term reversal effect)."""
    return panel_close.shift(skip) / panel_close.shift(skip + lookback) - 1


def low_vol_factor(panel_close: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """Negative realized vol -- low-volatility stocks score high (the low-vol anomaly)."""
    return -panel_close.pct_change().rolling(lookback).std(ddof=0)


def build_panel(data: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Turn {symbol: OHLCV} into a wide DataFrame (index=date, cols=symbols)."""
    return pd.DataFrame({sym: df[field] for sym, df in data.items()}).sort_index()


def rank_and_weight(scores: pd.DataFrame, top: float = 0.2, bottom: float = 0.0,
                    long_short: bool = False) -> pd.DataFrame:
    """Convert per-date factor scores into target weights.

    top/bottom: fraction of the universe to long/short each date.
    Returns weights summing to 1 on the long side (and -1 on the short side if
    long_short). Equal weight within each bucket.
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n == 0:
            continue
        ranked = valid.sort_values(ascending=False)
        n_long = max(1, int(round(n * top)))
        longs = ranked.index[:n_long]
        weights.loc[dt, longs] = 1.0 / n_long
        if long_short and bottom > 0:
            n_short = max(1, int(round(n * bottom)))
            shorts = ranked.index[-n_short:]
            weights.loc[dt, shorts] = -1.0 / n_short
    return weights


# ---- fundamental & news factors (cross-sectional, static snapshot) ---------
def fundamental_factor(fundamentals_panel: pd.DataFrame, field: str) -> pd.Series:
    """One fundamental metric as a *direction-signed* cross-sectional score.

    fundamentals_panel: index=symbol, columns=canonical fields (see
    data/fundamentals.load_panel). `field` e.g. 'pe', 'roe', 'revenue_growth'.
    Cheap-is-good metrics (pe, pb, debt) are negated so that, as always, higher
    score = more attractive. Returns a Series indexed by symbol.
    """
    from ..data.fundamentals import FACTOR_DIRECTION
    if field not in fundamentals_panel.columns:
        return pd.Series(dtype=float)
    raw = pd.to_numeric(fundamentals_panel[field], errors="coerce")
    direction = FACTOR_DIRECTION.get(field, 1)
    return _cross_section_z(raw) * direction


def value_factor(fundamentals_panel: pd.DataFrame) -> pd.Series:
    """Composite 'cheap' score: blend of inverted PE, PB, PS."""
    parts = [fundamental_factor(fundamentals_panel, f) for f in ("pe", "pb", "ps")]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(dtype=float)


def quality_factor(fundamentals_panel: pd.DataFrame) -> pd.Series:
    """Composite 'good business' score: ROE, margins, low leverage."""
    parts = [fundamental_factor(fundamentals_panel, f)
             for f in ("roe", "roa", "net_margin", "gross_margin", "debt_to_equity")]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(dtype=float)


def growth_factor(fundamentals_panel: pd.DataFrame) -> pd.Series:
    parts = [fundamental_factor(fundamentals_panel, f)
             for f in ("revenue_growth", "earnings_growth")]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(dtype=float)


def sentiment_factor(sentiment_by_symbol: dict) -> pd.Series:
    """News-sentiment score per symbol -> cross-sectional z. Feed it the
    {symbol: mean_sentiment} you get from data.news.fetch_with_sentiment /
    sentiment.aggregate_sentiment across the universe."""
    s = pd.Series(sentiment_by_symbol, dtype=float)
    return _cross_section_z(s)


def multi_factor_signal(data: dict,
                        factor_weights: dict | None = None,
                        rebalance: str = "ME", top: float = 0.2,
                        bottom: float = 0.0, long_short: bool = False,
                        fundamentals_panel: pd.DataFrame | None = None,
                        sentiment_by_symbol: dict | None = None) -> pd.DataFrame:
    """Full pipeline: compute factors, blend by z-score, rebalance on a schedule.

    Price factors are time-varying (recomputed each bar). Fundamental and news
    factors are treated as a *static cross-sectional tilt* (a snapshot applied to
    every rebalance) -- the honest way to use a single current snapshot in a price
    backtest without pretending you had point-in-time history you don't. For a true
    point-in-time fundamental backtest, supply dated snapshots and extend this; see
    references/fundamentals_news.md.

    factor_weights: e.g. {"momentum": 0.4, "low_vol": 0.2, "value": 0.2,
                          "quality": 0.1, "sentiment": 0.1}. Defaults to equal blend
                    of whatever factors you supply data for.
    rebalance: pandas offset alias -- 'ME' month-end, 'W-FRI' weekly, 'QE' quarterly.
    fundamentals_panel: from data.fundamentals.load_panel (index=symbol). Enables
                    'value' / 'quality' / 'growth' factors.
    sentiment_by_symbol: {symbol: mean_sentiment}. Enables the 'sentiment' factor.
    Returns a weights panel (index=date, cols=symbols) for backtest_portfolio.
    """
    close = build_panel(data, "close")

    price_factors = {
        "momentum": momentum_factor(close),
        "low_vol": low_vol_factor(close),
    }
    static_factors: dict = {}
    if fundamentals_panel is not None and len(fundamentals_panel):
        fp = fundamentals_panel.reindex(close.columns)
        static_factors["value"] = value_factor(fp)
        static_factors["quality"] = quality_factor(fp)
        static_factors["growth"] = growth_factor(fp)
    if sentiment_by_symbol:
        static_factors["sentiment"] = sentiment_factor(sentiment_by_symbol).reindex(close.columns)

    available = list(price_factors) + [k for k, v in static_factors.items() if v is not None and len(v)]
    factor_weights = factor_weights or {k: 1.0 for k in available}
    factor_weights = {k: v for k, v in factor_weights.items() if k in available}
    total_w = sum(factor_weights.values()) or 1.0

    blended = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name, w in factor_weights.items():
        if name in price_factors:
            z = price_factors[name].apply(_cross_section_z, axis=1)
        else:
            s = static_factors[name].reindex(close.columns).fillna(0.0)
            z = pd.DataFrame([s.values] * len(close), index=close.index, columns=close.columns)
        blended = blended.add(z * (w / total_w), fill_value=0.0)

    rebal_dates = close.resample(rebalance).last().index
    scores_on_dates = blended.reindex(rebal_dates).dropna(how="all")
    weights = rank_and_weight(scores_on_dates, top=top, bottom=bottom, long_short=long_short)
    return weights.reindex(close.index).ffill().fillna(0.0)
