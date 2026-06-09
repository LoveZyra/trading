"""Unified HTML report renderer — turn a markdown report into one clean, self-contained,
good-looking HTML page.

Why: the skill's analysis/复盘 reports are authored as markdown (easy to compose), but
markdown viewers render inline ``<span style=...>`` highlights and dense emoji bullets
poorly, and a sloppily authored table (mismatched column counts) silently collapses
into plain text — that's the "格式乱" problem. This module wraps the SAME markdown in a
polished, print-friendly HTML shell with consistent styling (cards, accent headers,
striped tables, red 🔴 callouts, auto green/red coloring of +x% / -x% figures) AND
normalizes tables so they always render. Output is a single UTF-8 .html file with no
external assets — opens cleanly in any browser.

Usage:
    from scripts import html_report as H
    H.save_html(markdown_text, "trading/reports/美股复盘_2026-06-09.html",
                title="美股盘后复盘", subtitle="2026-06-09 · alpha-forge")
    H.md_file_to_html("trading/reports/x.md")        # writes x.html beside it
    html = H.render(markdown_text, title="...", subtitle="...")
"""
from __future__ import annotations

import re

try:
    import markdown as _md
except Exception:  # pragma: no cover
    _md = None

_CSS = """
:root{--fg:#1f2329;--muted:#6b7280;--line:#e8eaed;--accent:#3a66c4;--accent2:#4a78c8;
--pos:#127a3d;--neg:#c0392b;--bg:#fff;--card:#f7f8fa;}
*{box-sizing:border-box}
body{margin:0;background:#eef0f3;color:var(--fg);
font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;
line-height:1.65;font-size:15px;-webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:24px auto;background:var(--bg);border-radius:14px;
box-shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);overflow:hidden}
.hd{padding:26px 34px 18px;border-bottom:3px solid var(--accent);
background:linear-gradient(180deg,#fafbfd,#fff)}
.hd h1{margin:0;font-size:23px;letter-spacing:.5px}
.hd .sub{color:var(--muted);font-size:13px;margin-top:6px}
.bd{padding:10px 34px 30px}
.bd h2{font-size:18px;margin:26px 0 10px;padding:8px 0 6px 12px;border-left:4px solid var(--accent2);
background:linear-gradient(90deg,#f5f8ff,transparent)}
.bd h3{font-size:15.5px;margin:20px 0 8px;color:#2a2f36;border-bottom:1px solid var(--line);padding-bottom:5px}
.bd p{margin:10px 0}
.bd a{color:var(--accent);text-decoration:none}.bd a:hover{text-decoration:underline}
.bd ul,.bd ol{margin:8px 0 8px 4px;padding-left:22px}.bd li{margin:5px 0}
.bd hr{border:none;border-top:1px solid var(--line);margin:20px 0}
.bd code{background:#f1f3f5;padding:1px 6px;border-radius:5px;font-size:13px;
font-family:"SF Mono",Consolas,Menlo,monospace}
.bd pre{background:#f7f8fa;border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto}
.bd pre code{background:none;padding:0}
blockquote{margin:14px 0;padding:11px 16px;background:var(--card);border-left:4px solid #cbd2dc;
border-radius:0 8px 8px 0;color:#444;font-size:14px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;
box-shadow:0 0 0 1px var(--line);border-radius:8px;overflow:hidden;display:table}
thead th{background:#f0f3f8;color:#3b4658;font-weight:600;text-align:left;
padding:9px 12px;border-bottom:2px solid #dfe4ea;white-space:nowrap}
tbody td{padding:8px 12px;border-bottom:1px solid #eef0f2;vertical-align:top}
tbody tr:nth-child(even){background:#fafbfc}
tbody tr:hover{background:#f4f7fc}
.tablewrap{overflow-x:auto;margin:14px 0}.tablewrap table{margin:0}
span[style*="color:#d62828"],span[style*="color: #d62828"]{
background:#fdecec;padding:1px 6px;border-radius:5px;font-weight:600}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}
.ft{padding:14px 34px 26px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);background:#fafbfc}
@media(max-width:640px){.wrap{margin:0;border-radius:0}.bd,.hd,.ft{padding-left:18px;padding-right:18px}}
@media print{body{background:#fff}.wrap{box-shadow:none;margin:0;max-width:none}}
"""

_PCT = re.compile(r'(?<![\w>])([+\-]\d[\d,]*\.?\d*%)')


def _normalize_tables(md: str) -> str:
    """Make pipe-tables render robustly: ensure a blank line before/after each table and
    REBUILD the separator row to match the header column count, so a sloppily authored
    table (e.g. header 15 cols but separator 14) still renders instead of collapsing."""
    lines = md.split("\n")
    out: list[str] = []

    def is_row(l): return l.lstrip().startswith("|")

    def is_sep(l):
        t = l.strip().strip("|")
        return bool(t) and set(t.replace(" ", "")) <= set("-:|")

    def ncols(l): return len(l.strip().strip("|").split("|"))

    i = 0
    while i < len(lines):
        l = lines[i]
        if is_row(l) and i + 1 < len(lines) and is_row(lines[i + 1]) and is_sep(lines[i + 1]):
            n = ncols(l)
            if out and out[-1].strip() != "":
                out.append("")
            out.append(l)
            out.append("|" + "|".join(["---"] * n) + "|")
            i += 2
            while i < len(lines) and is_row(lines[i]):
                out.append(lines[i]); i += 1
            if i < len(lines) and lines[i].strip() != "":
                out.append("")
            continue
        out.append(l); i += 1
    return "\n".join(out)


def _colorize_pct(html: str) -> str:
    def repl(m):
        v = m.group(1)
        return f'<span class="{"pos" if v[0] == "+" else "neg"}">{v}</span>'
    out, last = [], 0
    for tag in re.finditer(r'<[^>]+>', html):
        out.append(_PCT.sub(repl, html[last:tag.start()]))
        out.append(tag.group(0)); last = tag.end()
    out.append(_PCT.sub(repl, html[last:]))
    return "".join(out)


def _wrap_tables(html: str) -> str:
    """Wrap each <table> in a horizontally-scrollable div (wide tables stay readable)."""
    return html.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")


def render(markdown_text: str, *, title: str = "分析报告", subtitle: str = "",
           colorize_pct: bool = True) -> str:
    """Render markdown -> a complete, self-contained, styled HTML document (string)."""
    if _md is None:
        raise RuntimeError("python-markdown not installed: pip install markdown --break-system-packages")
    body = _md.markdown(_normalize_tables(markdown_text),
                        extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
    body = _wrap_tables(body)
    if colorize_pct:
        body = _colorize_pct(body)
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<div class="hd"><h1>{title}</h1>{sub}</div>
<div class="bd">{body}</div>
<div class="ft">本页由 alpha-forge 技能生成 · 仅复盘研究，非投资建议 · 回测/指标不代表未来。</div>
</div></body></html>"""


def save_html(markdown_text: str, path: str, *, title: str = "分析报告",
              subtitle: str = "", colorize_pct: bool = True) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(markdown_text, title=title, subtitle=subtitle, colorize_pct=colorize_pct))
    return path


def md_file_to_html(md_path: str, html_path: str | None = None, *,
                    title: str | None = None, subtitle: str = "") -> str:
    """Read a .md report and write the styled .html (default: beside it). The leading
    H1 becomes the page header title and is removed from the body to avoid duplication."""
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r'^#\s+(.+)$', text, re.M)
    if title is None:
        title = m.group(1).strip() if m else "分析报告"
    if m:
        text = text[:m.start()] + text[m.end():]
    if html_path is None:
        html_path = re.sub(r'\.md$', '.html', md_path) if md_path.endswith(".md") else md_path + ".html"
    return save_html(text, html_path, title=title, subtitle=subtitle)
