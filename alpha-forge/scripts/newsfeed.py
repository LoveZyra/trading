"""newsfeed.py — normalize news-connector rows into report alerts / a news section.

News data comes from a connector that only Claude can call (MT Newswires `fetch`,
FMP `news`, or the broker theme tools). The connector-specific fetch + curation is
Claude's job — pick the few headlines that matter, dedupe, keep them verbatim, attach
a date + source. This module is the small reusable glue that turns those curated items
into the html_report `alerts` list and a 🗞 news `groups` block, so "news → report" is
one call instead of hand-building dicts every time.

    from scripts import newsfeed as NF
    items = [  # assembled by Claude from the feed (see references/data_sources.md)
      {"date":"2026-06-23","headline":"Nasdaq posts sharpest drop in 2 weeks…",
       "symbol":"宏观","name":"芯片板块","level":"high","detail":"…","action":"…"},
      {"date":"2026-06-22","headline":"Micron signs AI agreement with Anthropic"},
    ]
    report["alerts"]  = NF.to_alerts(items) + report.get("alerts", [])
    report.setdefault("groups", []).insert(0, NF.to_news_group(items, source="MT Newswires"))

It never invents content — it only passes through the text you curated, so the report
stays honest and every headline keeps its date + source.
"""
from __future__ import annotations

_ALERT_KEYS = ("level", "symbol", "name", "signal", "hold", "headline", "detail", "action")


def to_alerts(items: list[dict]) -> list[dict]:
    """Curated news items -> html_report `alerts` rows. Pass-through with sane defaults
    (level='mid', signal='watch'); items without a headline are skipped."""
    out = []
    for it in items:
        if not it.get("headline"):
            continue
        row = {k: it[k] for k in _ALERT_KEYS if it.get(k) is not None}
        row.setdefault("level", "mid")
        row.setdefault("signal", "watch")
        row.setdefault("symbol", "新闻")
        out.append(row)
    return out


def to_news_group(items: list[dict], *, title: str | None = None,
                  source: str = "news", tone: str = "neutral") -> dict:
    """Curated headlines -> a 🗞 news `groups` block (date-stamped bullets, source-attributed)."""
    lines = []
    for it in items:
        h = it.get("headline")
        if not h:
            continue
        d = f"({it['date']})" if it.get("date") else ""
        lines.append(f"&bull; {h} {d}".rstrip())
    body = ("<b>来源:" + source + "(实时新闻,逐条注明日期)。</b><br>" + "<br>".join(lines)) if lines else "(无可用新闻)"
    return {"title": title or f"🗞 实时新闻头条({source})", "tag": "news", "tone": tone, "body": body}
