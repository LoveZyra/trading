"""Lexicon-based financial news sentiment -- bilingual (English + 中文), offline.

Why a lexicon and not an LLM call? A scored factor must be reproducible and cheap to
recompute over thousands of headlines across a backtest. A curated finance lexicon
(in the spirit of Loughran-McDonald for English, plus a Chinese finance word list)
gives a deterministic, fast, auditable score. For a richer read on a *handful* of
fresh headlines, Claude can always layer its own judgement on top -- but the factor
that goes into the backtest is this.

score(text) -> float in roughly [-1, 1]
    >0 bullish, <0 bearish, magnitude ~ strength. Negators flip nearby polarity.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- English finance lexicon (compact, polarity-weighted) -------------------
POS_EN = {
    "beat": 1.5, "beats": 1.5, "surge": 1.5, "surges": 1.5, "soar": 1.6, "soars": 1.6,
    "jump": 1.2, "jumps": 1.2, "rally": 1.3, "rallies": 1.3, "gain": 1.0, "gains": 1.0,
    "rise": 0.9, "rises": 0.9, "record": 1.2, "upgrade": 1.4, "upgraded": 1.4,
    "outperform": 1.4, "strong": 1.1, "growth": 1.0, "profit": 1.0, "profitable": 1.1,
    "bullish": 1.5, "boost": 1.1, "boosts": 1.1, "raises": 1.0, "raised": 1.0,
    "expansion": 1.0, "buyback": 1.1, "dividend": 0.7, "approval": 1.2, "approved": 1.2,
    "wins": 1.1, "win": 1.0, "breakthrough": 1.4, "exceeds": 1.4, "topped": 1.3,
    "optimistic": 1.1, "recovery": 1.0, "rebound": 1.2, "accelerate": 1.0,
}
NEG_EN = {
    "miss": -1.5, "misses": -1.5, "missed": -1.5, "plunge": -1.7, "plunges": -1.7,
    "tumble": -1.5, "tumbles": -1.5, "drop": -1.0, "drops": -1.0, "fall": -0.9,
    "falls": -0.9, "slump": -1.4, "crash": -1.8, "crashes": -1.8, "decline": -1.0,
    "downgrade": -1.4, "downgraded": -1.4, "underperform": -1.4, "weak": -1.1,
    "loss": -1.2, "losses": -1.2, "bearish": -1.5, "cut": -1.0, "cuts": -1.0,
    "warning": -1.3, "warn": -1.2, "warns": -1.2, "lawsuit": -1.2, "probe": -1.1,
    "investigation": -1.2, "fraud": -1.8, "bankruptcy": -1.9, "default": -1.7,
    "recall": -1.2, "layoff": -1.2, "layoffs": -1.2, "halt": -1.1, "halted": -1.1,
    "delay": -0.9, "delayed": -0.9, "concern": -0.9, "concerns": -0.9, "risk": -0.7,
    "slowdown": -1.2, "fears": -1.2, "selloff": -1.4, "slashes": -1.3,
}
NEGATORS_EN = {"no", "not", "never", "without", "fails", "fail", "failed", "less", "lower"}

# --- Chinese finance lexicon (中文金融情绪) ----------------------------------
POS_CN = {
    "大涨": 1.6, "涨停": 1.8, "上涨": 1.0, "拉升": 1.2, "走高": 1.0, "新高": 1.3,
    "增长": 1.0, "增持": 1.2, "回购": 1.1, "盈利": 1.0, "扭亏": 1.4, "超预期": 1.5,
    "利好": 1.5, "突破": 1.2, "中标": 1.2, "获批": 1.3, "批准": 1.2, "签约": 1.0,
    "提价": 1.0, "强劲": 1.1, "复苏": 1.0, "反弹": 1.2, "看好": 1.2, "买入": 1.2,
    "推荐": 1.0, "评级上调": 1.4, "业绩预增": 1.5, "分红": 0.7, "创纪录": 1.3,
    "订单": 0.8, "扩产": 1.0, "放量": 0.6, "翻倍": 1.5,
}
NEG_CN = {
    "大跌": -1.6, "跌停": -1.8, "下跌": -1.0, "跳水": -1.6, "走低": -1.0, "新低": -1.3,
    "下滑": -1.1, "减持": -1.3, "亏损": -1.3, "暴跌": -1.8, "利空": -1.5, "预亏": -1.5,
    "业绩预减": -1.5, "下调": -1.2, "评级下调": -1.4, "退市": -1.9, "违约": -1.7,
    "诉讼": -1.2, "处罚": -1.3, "罚款": -1.2, "调查": -1.2, "立案": -1.4, "造假": -1.9,
    "停牌": -1.0, "商誉减值": -1.5, "爆雷": -1.8, "质押": -0.8, "风险": -0.7,
    "警示": -1.2, "问询": -1.0, "减产": -1.1, "裁员": -1.2, "解禁": -0.8, "套现": -1.0,
}
NEGATORS_CN = {"不", "未", "无", "没有", "难以", "下降", "低于"}

_WORD_RE = re.compile(r"[A-Za-z]+")


def _score_english(text: str) -> tuple[float, int]:
    toks = [w.lower() for w in _WORD_RE.findall(text)]
    total, hits = 0.0, 0
    for i, w in enumerate(toks):
        val = POS_EN.get(w, 0.0) + NEG_EN.get(w, 0.0)
        if val:
            window = toks[max(0, i - 3):i]
            if any(n in window for n in NEGATORS_EN):
                val = -val
            total += val
            hits += 1
    return total, hits


def _score_chinese(text: str) -> tuple[float, int]:
    total, hits = 0.0, 0
    for lex in (POS_CN, NEG_CN):
        for term, val in lex.items():
            c = text.count(term)
            if c:
                # crude negation: a negator immediately before the term flips it
                flip = any((term in text) and (neg + term in text) for neg in NEGATORS_CN)
                total += (-val if flip else val) * c
                hits += c
    return total, hits


def score(text: str) -> float:
    """Sentiment of one headline/snippet, squashed to ~[-1, 1]."""
    if not text:
        return 0.0
    en, hen = _score_english(text)
    cn, hcn = _score_chinese(text)
    raw, hits = en + cn, hen + hcn
    if hits == 0:
        return 0.0
    avg = raw / hits
    # squash so a few strong words don't blow past ±1
    return max(-1.0, min(1.0, avg / 1.6))


def score_headlines(items: list[dict], text_keys=("title", "headline", "新闻标题", "summary")) -> pd.DataFrame:
    """Score a list of news dicts. Returns the items plus a 'sentiment' column,
    sorted newest-ish first if a time field is present."""
    rows = []
    for it in items:
        text = next((str(it[k]) for k in text_keys if it.get(k)), "")
        rows.append({**it, "sentiment": score(text)})
    df = pd.DataFrame(rows)
    return df


def headlines_for_llm(items: list[dict], text_keys=("title", "headline", "新闻标题", "summary")) -> list[dict]:
    """Return [{id, text}] for the headlines so Claude can score them itself.

    Why bother when there's already a lexicon? Lopez-Lira & Tang (2304.07619) show an
    LLM reading headlines predicts next-day returns better than dictionary sentiment —
    it understands context, negation and nuance the lexicon misses. The catch: the
    sandbox script can't call an LLM. So it's a hand-off, exactly like prices/news:
    this returns the texts, YOU (Claude) score each in [-1, 1], then feed the scores
    back via `apply_llm_scores`. Use the lexicon for bulk historical scoring and the
    LLM path for the freshest, highest-stakes headlines.
    """
    out = []
    for i, it in enumerate(items):
        text = next((str(it[k]) for k in text_keys if it.get(k)), "")
        out.append({"id": i, "text": text})
    return out


def apply_llm_scores(items: list[dict], scores) -> pd.DataFrame:
    """Attach Claude-provided scores (from `headlines_for_llm`) to the news items.

    scores: either {id: score} or a list aligned to items, each score in [-1, 1].
    Returns the scored DataFrame with a 'sentiment_llm' column. Timing note: news
    sentiment predicts the NEXT bar's return (markets underreact), so align the daily
    aggregate to t and the return to t+1 when using it as a signal — never same-bar.
    """
    if isinstance(scores, dict):
        s = [float(scores.get(i, scores.get(str(i), 0.0))) for i in range(len(items))]
    else:
        s = [float(x) for x in scores]
    rows = [{**it, "sentiment_llm": s[i] if i < len(s) else 0.0} for i, it in enumerate(items)]
    return pd.DataFrame(rows)


def aggregate_sentiment(items: list[dict], **kw) -> dict:
    """Collapse many headlines into one number + breakdown for a symbol/day.
    Returns mean sentiment, count, and bullish/bearish tallies -- ready to use as a
    news factor or to print in a daily report."""
    df = score_headlines(items, **kw)
    if len(df) == 0:
        return {"mean_sentiment": 0.0, "n": 0, "n_pos": 0, "n_neg": 0}
    s = df["sentiment"]
    return {
        "mean_sentiment": float(s.mean()),
        "n": int(len(s)),
        "n_pos": int((s > 0.05).sum()),
        "n_neg": int((s < -0.05).sum()),
        "max": float(s.max()),
        "min": float(s.min()),
    }


# ---- multi-level sentiment: stock + sector/peers + macro -------------------
def composite_sentiment(stock: float = 0.0, sector: float = 0.0, market: float = 0.0,
                        *, w_stock: float = 0.5, w_sector: float = 0.3, w_market: float = 0.2) -> float:
    """Blend stock-specific, sector/peer, and market/macro sentiment into ONE score.

    Why: a name's news-driven move is rarely purely idiosyncratic — the sector cycle
    (whole semi complex moving on an NVDA print) and macro (Fed/CPI/rates) often
    dominate. Default 0.5/0.3/0.2; raise w_sector/w_market for highly beta-driven names
    (ETFs, cyclicals), raise w_stock for single-name event stocks.
    """
    tot = (w_stock + w_sector + w_market) or 1.0
    return (w_stock * float(stock) + w_sector * float(sector) + w_market * float(market)) / tot


def blend_news_layers(stock_items: list | None = None, sector_items: list | None = None,
                      market_items: list | None = None, **weights) -> dict:
    """Score three lists of headlines (own / sector-peers / macro) and return the
    composite plus each layer, ready to use as the sentiment factor for a symbol.

    Each list is whatever you fetched (company news, sector/peer news, macro news) —
    typically via the Web-search hand-off. Returns {composite, stock, sector, market, n_*}.
    """
    def agg(items):
        if not items:
            return 0.0, 0
        a = aggregate_sentiment(items)
        return a["mean_sentiment"], a["n"]
    s, ns = agg(stock_items)
    se, nse = agg(sector_items)
    m, nm = agg(market_items)
    return {"composite": composite_sentiment(s, se, m, **weights),
            "stock": round(s, 3), "sector": round(se, 3), "market": round(m, 3),
            "n_stock": ns, "n_sector": nse, "n_market": nm}


# ---- recency / time-decay weighting ----------------------------------------
def time_weighted_sentiment(items: list, halflife_hours: float = 48.0, now=None,
                            text_keys=("title", "headline", "新闻标题", "summary"),
                            time_keys=("time", "发布时间", "publishedAt", "date", "datetime")) -> dict:
    """Recency-weighted mean sentiment: a headline's weight = 0.5 ** (age / halflife).

    Stale news fades, today's print dominates — markets price the freshest information.
    Each item needs a timestamp (any of `time_keys`); items without a parseable time
    get the median age so they don't dominate. Returns {mean_sentiment, n, newest_age_h,
    eff_n} where mean_sentiment is the time-weighted average.
    """
    now = pd.to_datetime(now, utc=True) if now is not None else pd.Timestamp.now(tz="UTC")
    scores, ages = [], []
    for it in items or []:
        text = next((str(it[k]) for k in text_keys if it.get(k)), "")
        t = None
        for tk in time_keys:
            if it.get(tk) is not None:
                t = pd.to_datetime(it[tk], errors="coerce", utc=True)
                break
        age_h = (now - t).total_seconds() / 3600.0 if (t is not None and not pd.isna(t)) else np.nan
        scores.append(score(text)); ages.append(age_h)
    if not scores:
        return {"mean_sentiment": 0.0, "n": 0, "eff_n": 0.0, "newest_age_h": None}
    ages = np.array(ages, float)
    med = np.nanmedian(ages) if np.isfinite(ages).any() else 0.0
    ages = np.where(np.isnan(ages), med, ages)
    ages = np.clip(ages, 0, None)
    w = 0.5 ** (ages / max(halflife_hours, 1e-6))
    sc = np.array(scores, float)
    mean = float((w * sc).sum() / w.sum()) if w.sum() else 0.0
    return {"mean_sentiment": mean, "n": int(len(sc)), "eff_n": round(float(w.sum()), 2),
            "newest_age_h": round(float(np.nanmin(ages)), 1)}
