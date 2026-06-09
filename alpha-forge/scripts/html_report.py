"""量化分析报告渲染器 — 把结构化 report dict 渲染成单文件机构研报风 HTML。

用法：
    from scripts import html_report as H
    report = {"meta": {...}, "verdict": {...}, "levels": [...], ...}   # 见 SCHEMA.md
    H.save_html(report, "trading/reports/美股复盘_2026-06-09.html")
    html = H.render(report)                                            # 或拿字符串

颜色为「红涨绿跌」(A股惯例)；如需西式(绿涨红跌)，把下方 _CSS 里 --pos/--neg 两值对调。
自包含、无外部依赖：CSS / JS 已内联，中文字体走系统回退，离线可开。
每个字段「给了就渲染、不给就跳过」，支持单标的 / 组合 / 自选池 / 回测等多种报告。
"""
from __future__ import annotations

import html as _html
import json as _json

_CSS = r"""
/* ============================================================================
   alpha-forge · 量化分析报告模板  —  institutional research-note style
   Light, restrained, print-first. All figures tabular & monospaced.
   ========================================================================== */

/* Fonts: system stacks only (see :root --serif/--sans/--mono). No remote @import, so the
   report is truly self-contained and renders the SAME offline / in a sandboxed preview —
   a remote webfont fails to load there and flashes a thin, washed-out title. */

:root{
  --paper:#f4f2ec;
  --card:#ffffff;
  --ink:#15181e;
  --ink-soft:#39414c;
  --muted:#767c87;
  --faint:#9aa0aa;
  --hair:#e7e3d8;
  --hair-2:#d9d4c7;
  --rule:#1b3a5b;            /* deep institutional navy */
  --accent:#1b3a5b;
  --accent-2:#2f5d8a;
  --accent-wash:#eef2f7;
  --pos:#c0392b;             /* 涨 = 红（A股惯例：红涨绿跌；如需西式改回，对调 pos/neg 即可） */
  --pos-wash:#fbecea;
  --neg:#147a43;             /* 跌 = 绿 */
  --neg-wash:#e8f3ec;
  --warn:#9a6a12;            /* watch / amber */
  --warn-wash:#f6efdf;
  --flag:#bd2a26;            /* 🔴 alert */
  --flag-wash:#fbeceb;
  --long:#c0392b;
  --watch:#9a6a12;
  --short:#147a43;

  --serif:"Source Serif 4","Songti SC",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans","PingFang SC","Microsoft YaHei",system-ui,-apple-system,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;

  --maxw:1080px;
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.6;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
.sheet{
  max-width:var(--maxw);margin:26px auto 60px;background:var(--card);
  border:1px solid var(--hair-2);
  box-shadow:0 1px 2px rgba(20,24,30,.04),0 18px 50px -28px rgba(20,24,30,.28);
}

/* numeric helpers ----------------------------------------------------------*/
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1}
.pos{color:var(--pos)}
.neg{color:var(--neg)}
.muted{color:var(--muted)}
.mono{font-family:var(--mono)}
.tnum{font-variant-numeric:tabular-nums}

/* ===== Masthead =========================================================== */
.masthead{padding:30px 40px 22px;border-bottom:2.5px solid var(--rule);position:relative}
.masthead .kicker{display:flex;align-items:center;gap:10px;margin-bottom:13px;padding-right:150px;
  font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.masthead .kicker .mark{font-family:var(--serif);font-weight:700;letter-spacing:.04em;
  color:var(--accent);text-transform:none;font-size:13px;white-space:nowrap}
.masthead .kicker .spacer{flex:1;height:1px;background:var(--hair)}
.type-badge{font-family:var(--sans);font-weight:600;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:#fff;background:var(--accent);padding:3px 9px;border-radius:2px}
.masthead h1{font-family:var(--serif);font-weight:600;font-size:34px;line-height:1.12;
  letter-spacing:-.01em;margin:0 0 6px;max-width:calc(100% - 150px)}
.masthead .subtitle{font-family:var(--serif);font-size:18px;color:var(--ink-soft);font-weight:400;margin:0}
.masthead .metaline{display:flex;flex-wrap:wrap;gap:6px 20px;margin-top:16px;
  font-size:12px;color:var(--muted)}
.masthead .metaline b{color:var(--ink-soft);font-weight:600}
.masthead .metaline .num{font-size:12px}
.masthead .stamp{position:absolute;top:30px;right:40px;text-align:right;background:var(--card);padding-left:14px;z-index:2}
.masthead .stamp .d{font-family:var(--mono);font-size:20px;font-weight:500;color:var(--ink)}
.masthead .stamp .wd{font-size:11.5px;color:var(--muted);margin-top:2px}

/* ===== Body shell ========================================================= */
.body{padding:6px 40px 40px}
section.block{padding:24px 0 6px;border-top:1px solid var(--hair)}
section.block:first-child{border-top:none}
.sec-head{display:flex;align-items:baseline;gap:12px;margin:0 0 16px}
.sec-head .no{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--accent-2);
  border:1.5px solid var(--accent-2);width:26px;height:26px;flex:none;border-radius:50%;
  display:flex;align-items:center;justify-content:center;line-height:1}
.sec-head h2{font-family:var(--serif);font-weight:600;font-size:21px;letter-spacing:-.005em;margin:0;flex:1}
.sec-head .h-note{font-size:12px;color:var(--muted);font-family:var(--mono);white-space:nowrap}

/* ===== Verdict (结论先行) ================================================= */
.verdict{display:grid;grid-template-columns:auto 1fr;gap:0;border:1px solid var(--hair-2);
  background:linear-gradient(180deg,#fbfaf7,#fff)}
.verdict .stance{padding:22px 24px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:6px;min-width:158px;border-right:1px solid var(--hair-2);text-align:center}
.verdict .stance .lab{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.verdict .stance .val{font-family:var(--serif);font-size:30px;font-weight:700;line-height:1}
.verdict .stance .arrow{font-size:22px;line-height:1}
.verdict.up .val,.verdict.up .arrow{color:var(--pos)}
.verdict.down .val,.verdict.down .arrow{color:var(--neg)}
.verdict.flat .val,.verdict.flat .arrow{color:var(--warn)}
.verdict .body-v{padding:18px 24px}
.verdict .action{font-family:var(--serif);font-size:16.5px;font-weight:600;color:var(--ink);
  line-height:1.4;margin:0 0 8px}
.verdict .summary{font-size:13px;color:var(--ink-soft);line-height:1.62;margin:0}

/* ===== Alerts (重点关注) ================================================== */
.alerts{display:flex;flex-direction:column;gap:0}
.alert{display:grid;grid-template-columns:auto 1fr;gap:14px;padding:14px 2px 14px 0;
  border-bottom:1px solid var(--hair)}
.alert:last-child{border-bottom:none}
.alert .dot{width:9px;height:9px;border-radius:50%;background:var(--flag);margin-top:7px;flex:none;
  box-shadow:0 0 0 3px var(--flag-wash)}
.alert.mid .dot{background:var(--warn);box-shadow:0 0 0 3px var(--warn-wash)}
.alert .a-main{min-width:0}
.alert .a-top{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;margin-bottom:3px}
.alert .sym{font-family:var(--mono);font-weight:600;font-size:14.5px;color:var(--ink);white-space:nowrap}
.alert .nm{font-size:13px;color:var(--muted)}
.alert .chip{font-size:10.5px;font-weight:600;letter-spacing:.04em;padding:1px 7px;border-radius:2px}
.chip.hold{background:var(--accent-wash);color:var(--accent);border:1px solid #cfdae8}
.chip.star::before{content:"★ "}
.alert .headline{font-weight:600;font-size:13.5px;color:var(--ink);line-height:1.45}
.alert .detail{font-size:12.5px;color:var(--ink-soft);margin:5px 0 0;line-height:1.6}
.alert .act{font-size:12.5px;margin:6px 0 0;line-height:1.55;color:var(--ink)}
.alert .act b{color:var(--accent)}

/* ===== Environment grid (regime / macro / calendar) ====================== */
.env-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{border:1px solid var(--hair-2);background:#fcfbf8}
.panel .p-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;
  padding:11px 15px;border-bottom:1px solid var(--hair);flex-wrap:nowrap}
.panel .p-head .t{font-family:var(--serif);font-weight:600;font-size:15px;white-space:nowrap;flex:1 1 auto;min-width:0}
.panel .p-head .score{flex:none}
.panel .p-head .score{font-family:var(--mono);font-weight:600;font-size:15px}
.panel .p-body{padding:13px 15px}
.panel .p-note{font-size:12px;color:var(--ink-soft);line-height:1.6;margin:11px 0 0}

/* score meter */
.meter{margin:4px 0 2px}
.meter .track{position:relative;height:8px;border-radius:5px;
  background:linear-gradient(90deg,var(--neg) 0%,#e8e4d9 50%,var(--pos) 100%);opacity:.9}
.meter .zero{position:absolute;top:-4px;bottom:-4px;left:50%;width:1px;background:var(--ink-soft);opacity:.45}
.meter .needle{position:absolute;top:50%;width:13px;height:13px;border-radius:50%;
  background:#fff;border:2.5px solid var(--ink);transform:translate(-50%,-50%)}
.meter .scale{display:flex;justify-content:space-between;margin-top:6px;
  font-family:var(--mono);font-size:10px;color:var(--faint)}
.meter .label{text-align:center;font-size:12.5px;color:var(--ink-soft);margin-top:7px}
.meter .label b{color:var(--ink)}

/* small definition rows */
.dl{display:flex;flex-direction:column;gap:0;margin:2px 0 0}
.dl .row{display:grid;grid-template-columns:128px 88px 1fr;gap:10px;padding:7px 0;
  border-top:1px dashed var(--hair);font-size:12.5px;align-items:baseline}
.dl .row:first-child{border-top:none}
.dl .row .k{color:var(--muted)}
.dl .row .v{font-family:var(--mono);font-weight:600;text-align:right}
.dl .row .r{color:var(--ink-soft)}

/* calendar */
.cal{display:flex;flex-direction:column}
.cal .ev{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;
  padding:10px 0;border-top:1px solid var(--hair)}
.cal .ev:first-child{border-top:none}
.cal .ev .when{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--ink);min-width:52px}
.cal .ev .nm{font-size:13px}
.cal .ev .nm .imp{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px;vertical-align:middle}
.cal .ev .nm.flag{font-weight:600;color:var(--flag)}
.cal .ev .countdown{font-family:var(--mono);font-size:11.5px;color:var(--muted);white-space:nowrap}
.imp.high{background:var(--flag)} .imp.med{background:var(--warn)} .imp.low{background:var(--faint)}

/* ===== Tables ============================================================= */
.tablewrap{overflow-x:auto;margin:2px 0}
table.grid{border-collapse:collapse;width:100%;font-size:12.5px}
table.grid thead th{background:var(--accent);color:#fff;font-weight:600;font-size:11px;
  letter-spacing:.02em;text-align:right;padding:8px 9px;white-space:nowrap;position:sticky;top:0}
table.grid thead th:first-child,table.grid thead th.l{text-align:left}
table.grid tbody td{padding:7px 9px;border-bottom:1px solid var(--hair);text-align:right;
  white-space:nowrap;font-family:var(--mono);font-variant-numeric:tabular-nums}
table.grid tbody td.l{text-align:left;font-family:var(--sans);white-space:normal}
table.grid tbody td.note{white-space:normal;min-width:150px;max-width:230px;font-size:11.5px;color:var(--ink-soft)}
table.grid.pf tbody td{white-space:normal}
table.grid tbody tr:nth-child(even){background:#faf9f5}
table.grid tbody tr:hover{background:var(--accent-wash)}
table.grid td.sym{font-family:var(--mono);font-weight:600;color:var(--ink)}
table.grid td.name{font-family:var(--sans);color:var(--ink-soft)}
table.grid td.sector{font-family:var(--sans);color:var(--muted);font-size:11.5px}
table.grid .rr-strong{font-weight:700}
table.grid tr.is-watch td{background:var(--warn-wash)}
table.grid tr.is-watch:nth-child(even) td{background:#f3ead7}
table.grid .flagcell{color:var(--flag);font-weight:600}
.colhint{font-size:11.5px;color:var(--muted);margin:9px 0 2px;line-height:1.55}

/* signal badge */
.sig{display:inline-block;font-size:10.5px;font-weight:600;padding:2px 8px;border-radius:2px;
  font-family:var(--sans);letter-spacing:.02em}
.sig.long{background:var(--pos-wash);color:var(--pos)}
.sig.watch{background:var(--warn-wash);color:var(--warn)}
.sig.short{background:var(--neg-wash);color:var(--neg)}

/* inline R/R bar (in tables) */
.rrbar{display:inline-flex;align-items:center;gap:7px;justify-content:flex-end}
.rrbar .bar{width:46px;height:6px;border-radius:4px;background:#ece8dd;overflow:hidden;flex:none}
.rrbar .bar i{display:block;height:100%;border-radius:4px}
.rrbar .v{font-family:var(--mono);font-weight:600;min-width:30px;text-align:right}

/* ===== Level ladder (single-name hero) =================================== */
.ladder-card{border:1px solid var(--hair-2);background:#fcfbf8;padding:20px 22px 14px}
.ladder-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px;margin-bottom:20px}
.ladder-head .px{font-family:var(--mono);font-size:26px;font-weight:600}
.ladder-head .px small{font-size:13px;color:var(--muted);font-weight:400;margin-left:4px}
.ladder-head .rrwrap{margin-left:auto;text-align:right}
.ladder-head .rrwrap .lab{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.ladder-head .rrwrap .val{font-family:var(--serif);font-size:26px;font-weight:700;line-height:1}
.ladder{position:relative;height:78px;margin:0 6px 4px}
.ladder .axis{position:absolute;left:0;right:0;top:46px;height:6px;border-radius:3px;background:#ece8dd}
.ladder .zone{position:absolute;top:46px;height:6px;border-radius:3px}
.ladder .zone.loss{background:var(--neg);opacity:.32}
.ladder .zone.gain{background:var(--pos);opacity:.3}
.ladder .zone.buy{background:var(--accent);opacity:.85;height:8px;top:45px;border-radius:2px}
.ladder .tick{position:absolute;top:40px;width:2px;height:18px;background:var(--ink-soft)}
.ladder .tick.now{top:30px;height:34px;width:2.5px;background:var(--ink)}
.ladder .cap{position:absolute;font-family:var(--mono);font-size:11px;white-space:nowrap;transform:translateX(-50%)}
.ladder .cap.top{top:8px}.ladder .cap.bot{top:62px}
.ladder .cap .k{display:block;font-family:var(--sans);font-size:9.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);text-align:center}
.ladder .cap .pv{font-weight:600}
.ladder .cap.now .pv{color:var(--ink)} .ladder .cap.stop .pv{color:var(--neg)}
.ladder .cap.target .pv{color:var(--pos)} .ladder .cap.buy .pv{color:var(--accent)}
.ladder-foot{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--hair);
  border:1px solid var(--hair);margin-top:14px}
.ladder-foot .cell{background:#fff;padding:9px 12px}
.ladder-foot .cell .k{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.ladder-foot .cell .v{font-family:var(--mono);font-weight:600;font-size:15px;margin-top:2px}

/* ===== Technical stat strip ============================================== */
.statgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));margin:2px 0;
  border-top:1px solid var(--hair);border-left:1px solid var(--hair)}
.statgrid .s{background:#fff;padding:11px 13px;border-right:1px solid var(--hair);border-bottom:1px solid var(--hair)}
.statgrid .s .k{font-size:10.5px;letter-spacing:.04em;color:var(--muted)}
.statgrid .s .v{font-family:var(--mono);font-weight:600;font-size:16px;margin-top:3px}
.statgrid .s .x{font-size:11px;color:var(--muted);margin-top:1px}

/* ===== Sentiment ========================================================== */
.senti{border:1px solid var(--hair-2)}
.senti .s-row{display:grid;grid-template-columns:96px 150px 1fr;gap:14px;align-items:center;
  padding:11px 15px;border-top:1px solid var(--hair)}
.senti .s-row:first-child{border-top:none}
.senti .s-row.comp{background:var(--accent-wash)}
.senti .s-row .lay{font-weight:600;font-size:13px}
.senti .s-row .key{font-size:12px;color:var(--ink-soft);line-height:1.5}
.sbar{position:relative;height:7px;border-radius:4px;background:#ece8dd}
.sbar .fill{position:absolute;top:0;bottom:0;border-radius:4px}
.sbar .mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--ink-soft);opacity:.4}
.sbar-wrap{display:flex;align-items:center;gap:10px}
.sbar-wrap .sv{font-family:var(--mono);font-weight:600;font-size:13px;min-width:42px}

/* ===== Equity / benchmark chart ========================================== */
.chart-card{border:1px solid var(--hair-2);background:#fcfbf8;padding:16px 18px 10px}
.chart-head{display:flex;flex-wrap:wrap;gap:8px 22px;align-items:baseline;margin-bottom:6px}
.chart-head .ct{font-family:var(--serif);font-weight:600;font-size:15px}
.chart-head .legend{display:flex;gap:16px;margin-left:auto;font-size:11.5px;color:var(--muted)}
.chart-head .legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.chart-stats{display:flex;flex-wrap:wrap;gap:6px 26px;margin-top:8px;padding-top:10px;border-top:1px solid var(--hair)}
.chart-stats .cs{font-size:11.5px;color:var(--muted)}
.chart-stats .cs b{font-family:var(--mono);font-size:14px;color:var(--ink);font-weight:600;margin-left:6px}

/* ===== Prose blocks ======================================================= */
.prose{font-size:13px;color:var(--ink-soft);line-height:1.68}
.prose p{margin:10px 0}
.prose b,.prose strong{color:var(--ink);font-weight:600}
.prose ul{margin:8px 0;padding-left:20px}.prose li{margin:5px 0}
.callout-quote{font-size:12.5px;color:var(--ink-soft);background:#fbfaf7;border-left:3px solid var(--hair-2);
  padding:11px 16px;margin:12px 0;line-height:1.62}

/* groups (三组解读 style) */
.groups{display:flex;flex-direction:column;gap:0}
.group{padding:13px 0;border-top:1px solid var(--hair)}
.group:first-child{border-top:none}
.group .gt{font-family:var(--serif);font-weight:600;font-size:14.5px;margin:0 0 4px}
.group .gd{font-size:12.5px;color:var(--ink-soft);line-height:1.6;margin:0}
.group .gt .tag{font-family:var(--sans);font-size:10.5px;font-weight:600;padding:1px 7px;border-radius:2px;
  margin-left:8px;vertical-align:middle}
.tag.strong{background:var(--pos-wash);color:var(--pos)}
.tag.weak{background:var(--neg-wash);color:var(--neg)}
.tag.neutral{background:var(--warn-wash);color:var(--warn)}

/* ===== Footer / disclaimer =============================================== */
.footer{padding:22px 40px 30px;border-top:2.5px solid var(--rule);background:#fbfaf7;margin-top:14px}
.footer .disc{font-size:11.5px;color:var(--muted);line-height:1.62}
.footer .sources{font-size:11.5px;color:var(--muted);margin-top:12px}
.footer .sources a{color:var(--accent-2);text-decoration:none;border-bottom:1px solid #cfdae8}
.footer .sign{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;
  margin-top:14px;padding-top:12px;border-top:1px solid var(--hair);font-size:11px;color:var(--faint)}
.footer .sign .mk{font-family:var(--serif);font-weight:700;color:var(--accent)}

/* ===== Dev data switcher (screen only) =================================== */
.switcher{position:sticky;top:0;z-index:50;display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  background:rgba(244,242,236,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--hair-2);
  padding:9px 16px;font-size:12px}
.switcher .lab{color:var(--muted);font-size:11px;letter-spacing:.05em;text-transform:uppercase}
.switcher .seg{display:inline-flex;border:1px solid var(--hair-2);border-radius:4px;overflow:hidden;background:#fff}
.switcher .seg button{border:none;background:none;font:inherit;font-size:12px;padding:5px 13px;cursor:pointer;
  color:var(--ink-soft);border-left:1px solid var(--hair)}
.switcher .seg button:first-child{border-left:none}
.switcher .seg button.on{background:var(--accent);color:#fff;font-weight:600}
.switcher .hint{margin-left:auto;color:var(--faint);font-size:11px}

/* ===== Responsive ========================================================= */
@media(max-width:760px){
  .sheet{margin:0;border:none}
  .masthead,.body,.footer{padding-left:20px;padding-right:20px}
  .masthead h1{font-size:26px;max-width:none} .masthead .kicker{padding-right:0}
  .masthead .stamp{position:static;text-align:left;margin-top:12px;background:none;padding-left:0}
  .env-grid{grid-template-columns:1fr}
  .verdict{grid-template-columns:1fr}.verdict .stance{border-right:none;border-bottom:1px solid var(--hair-2);flex-direction:row;gap:14px}
  .ladder-foot{grid-template-columns:repeat(2,1fr)}
  .senti .s-row{grid-template-columns:1fr;gap:6px}
}

/* ===== Print ============================================================== */
@media print{
  @page{margin:14mm 12mm}
  body{background:#fff}
  .switcher{display:none !important}
  .sheet{box-shadow:none;border:none;margin:0;max-width:none}
  section.block{break-inside:avoid}
  .alert,.group,.cal .ev,.ladder-card,.chart-card,.panel,.verdict{break-inside:avoid}
  table.grid thead th{position:static;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .tablewrap{overflow:visible}
  table.grid tbody td{padding:5px 7px;font-size:11px}
  *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .masthead .stamp{position:absolute}
}

"""

_JS = r"""
/* ============================================================================
   alpha-forge · 报告渲染器
   Reads a single report object (schema in SCHEMA.md) and builds the DOM.
   Every section is optional — present a key, it renders; omit it, it's skipped.
   ========================================================================== */
(function (global) {
  "use strict";

  /* ---- tiny DOM helper ------------------------------------------------- */
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "style") n.setAttribute("style", attrs[k]);
      else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    }
    if (children != null) (Array.isArray(children) ? children : [children]).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    });
    return n;
  }

  /* ---- number formatting ---------------------------------------------- */
  function commas(x) { return String(x).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function price(v) {
    if (v == null) return "—";
    const a = Math.abs(v);
    const d = a >= 100 ? (Number.isInteger(v) ? 0 : 1) : a >= 10 ? 2 : 2;
    return commas(Number(v).toFixed(d));
  }
  // signed percent where value is already in percent units (e.g. 3.27 -> "+3.27%")
  function pctSigned(v, dp) {
    if (v == null) return "—";
    const s = (v > 0 ? "+" : "") + Number(v).toFixed(dp == null ? (Math.abs(v) >= 100 ? 0 : 1) : dp) + "%";
    return s;
  }
  function signedSpan(v, dp) {
    if (v == null) return el("span", { class: "muted" }, "—");
    return el("span", { class: "num " + (v > 0 ? "pos" : v < 0 ? "neg" : "muted") }, pctSigned(v, dp));
  }
  function scoreStr(v, dp) {
    if (v == null) return "—";
    return (v > 0 ? "+" : "") + Number(v).toFixed(dp == null ? 2 : dp);
  }
  function pct01(x) { return Math.max(0, Math.min(1, x)) * 100; }
  function clampPos(p) { return Math.max(4, Math.min(96, p)); }

  /* ---- score meter (−1 … +1) ------------------------------------------ */
  function scoreMeter(value, opts) {
    opts = opts || {};
    const min = opts.min == null ? -1 : opts.min, max = opts.max == null ? 1 : opts.max;
    const pos = pct01((value - min) / (max - min));
    return el("div", { class: "meter" }, [
      el("div", { class: "track" }, [
        el("span", { class: "zero" }),
        el("span", { class: "needle", style: "left:" + pos + "%" })
      ]),
      el("div", { class: "scale" }, [el("span", null, String(min)), el("span", null, "0"), el("span", null, "+" + max)]),
      opts.label ? el("div", { class: "label", html: opts.label }) : null
    ]);
  }

  /* ---- R/R inline bar -------------------------------------------------- */
  function rrColor(rr) {
    return rr >= 1.8 ? "var(--pos)" : rr >= 1.2 ? "var(--accent)" : rr >= 0.9 ? "var(--warn)" : "var(--neg)";
  }
  function rrBar(rr) {
    if (rr == null) return el("span", { class: "muted" }, "—");
    const w = pct01(rr / 3);
    return el("span", { class: "rrbar" }, [
      el("span", { class: "bar" }, el("i", { style: "width:" + w + "%;background:" + rrColor(rr) })),
      el("span", { class: "v", style: "color:" + rrColor(rr) }, Number(rr).toFixed(2))
    ]);
  }

  /* ---- signal badge ---------------------------------------------------- */
  const SIG = { long: ["long", "做多"], watch: ["watch", "观望"], short: ["short", "做空"] };
  function sigBadge(s, textOverride) {
    const m = SIG[s] || ["watch", s || "—"];
    return el("span", { class: "sig " + m[0] }, textOverride || m[1]);
  }

  /* ====================================================================== */
  /*  SECTION BUILDERS                                                      */
  /* ====================================================================== */

  function masthead(meta) {
    meta = meta || {};
    const TYPE = { single: "个股分析", portfolio: "组合 / 自选池", market: "市场扫描",
      backtest: "策略回测", attribution: "业绩归因", macro: "宏观复盘" };
    const metaItems = [];
    if (meta.market) metaItems.push(el("span", null, [el("b", null, "市场 "), meta.market]));
    if (meta.data_source) metaItems.push(el("span", null, [el("b", null, "数据 "), meta.data_source]));
    if (meta.universe) metaItems.push(el("span", null, [el("b", null, "范围 "), meta.universe]));
    if (meta.tag) metaItems.push(el("span", { class: "flagcell", style: "color:var(--flag)" }, meta.tag));
    return el("header", { class: "masthead" }, [
      el("div", { class: "kicker" }, [
        el("span", { class: "mark" }, meta.generated_by || "alpha-forge"),
        meta.report_type ? el("span", { class: "type-badge" }, TYPE[meta.report_type] || meta.report_type) : null,
        el("span", { class: "spacer" }),
        el("span", null, "量化分析报告")
      ]),
      el("h1", null, meta.title || "量化分析报告"),
      meta.subtitle ? el("p", { class: "subtitle" }, meta.subtitle) : null,
      metaItems.length ? el("div", { class: "metaline" }, metaItems) : null,
      meta.date ? el("div", { class: "stamp" }, [
        el("div", { class: "d num" }, meta.date),
        meta.weekday ? el("div", { class: "wd" }, meta.weekday) : null
      ]) : null
    ]);
  }

  function verdict(v) {
    if (!v) return null;
    const dir = v.stance && /多|涨|强|看多|偏多|bull/i.test(v.stance) ? "up"
      : v.stance && /空|跌|弱|看空|偏空|bear/i.test(v.stance) ? "down" : "flat";
    const arrow = dir === "up" ? "▲" : dir === "down" ? "▼" : "◆";
    return el("div", { class: "verdict " + dir }, [
      el("div", { class: "stance" }, [
        el("div", { class: "lab" }, "综合立场"),
        el("div", { class: "arrow" }, arrow),
        el("div", { class: "val" }, v.stance || "中性")
      ]),
      el("div", { class: "body-v" }, [
        v.action ? el("p", { class: "action", html: v.action }) : null,
        v.summary ? el("p", { class: "summary", html: v.summary }) : null
      ])
    ]);
  }

  function alerts(list) {
    if (!list || !list.length) return null;
    return el("div", { class: "alerts" }, list.map(function (a) {
      const top = [];
      if (a.symbol) top.push(el("span", { class: "sym" }, a.symbol));
      if (a.name) top.push(el("span", { class: "nm" }, a.name));
      if (a.hold) top.push(el("span", { class: "chip hold star" }, a.hold === true ? "持仓" : a.hold));
      if (a.signal) top.push(sigBadge(a.signal));
      return el("div", { class: "alert " + (a.level === "mid" ? "mid" : "") }, [
        el("span", { class: "dot" }),
        el("div", { class: "a-main" }, [
          top.length ? el("div", { class: "a-top" }, top) : null,
          a.headline ? el("div", { class: "headline", html: a.headline }) : null,
          a.detail ? el("div", { class: "detail", html: a.detail }) : null,
          a.action ? el("div", { class: "act", html: "<b>动作 · </b>" + a.action }) : null
        ])
      ]);
    }));
  }

  function envPanel(title, score, bodyChildren, note) {
    return el("div", { class: "panel" }, [
      el("div", { class: "p-head" }, [
        el("div", { class: "t" }, title),
        score != null ? el("div", { class: "score", style: "color:" + (score > 0.05 ? "var(--pos)" : score < -0.05 ? "var(--neg)" : "var(--warn)") }, scoreStr(score)) : null
      ]),
      el("div", { class: "p-body" }, bodyChildren.concat(note ? [el("p", { class: "p-note", html: note })] : []))
    ]);
  }

  function regimePanel(r) {
    const rows = (r.rows || []).map(function (row) {
      return el("div", { class: "row" }, [
        el("div", { class: "k" }, row.item),
        el("div", { class: "v", html: row.value },),
        el("div", { class: "r", html: row.read || "" })
      ]);
    });
    const body = [scoreMeter(r.score, { label: r.label ? "<b>" + scoreStr(r.score) + "</b> · " + r.label : null })];
    if (rows.length) body.push(el("div", { class: "dl" }, rows));
    return envPanel(r.title || "📊 大盘环境", r.score, body, r.note);
  }

  function macroPanel(m) {
    const body = [scoreMeter(m.risk_score, { label: m.label ? "<b>" + scoreStr(m.risk_score) + "</b> · " + m.label : null })];
    const dl = [];
    if (m.vix != null) dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, "VIX"), el("div", { class: "v" }, String(m.vix)), el("div", { class: "r" }, m.vix_note || "")]));
    (m.rows || []).forEach(function (row) {
      dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, row.item), el("div", { class: "v", html: row.value }), el("div", { class: "r", html: row.read || "" })]));
    });
    if (dl.length) body.push(el("div", { class: "dl" }, dl));
    return envPanel(m.title || "🌐 全球宏观", m.risk_score, body, m.note);
  }

  function calendarPanel(cal) {
    const evs = cal.map(function (e) {
      return el("div", { class: "ev" }, [
        el("div", { class: "when" }, e.date),
        el("div", { class: "nm" + (e.flagged ? " flag" : "") }, [
          el("span", { class: "imp " + (e.impact || "med") }), e.event
        ]),
        el("div", { class: "countdown" }, e.in_days != null ? (e.in_days === 0 ? "今日" : e.in_days + "天后") : (e.note || ""))
      ]);
    });
    return el("div", { class: "panel" }, [
      el("div", { class: "p-head" }, el("div", { class: "t" }, "📅 事件前瞻")),
      el("div", { class: "p-body" }, el("div", { class: "cal" }, evs))
    ]);
  }

  /* ---- level ladder (single-name) ------------------------------------- */
  function levelLadder(L) {
    const tgtHi = L.target2 != null ? L.target2 : L.target;
    const lo = Math.min(L.stop, L.buy_low != null ? L.buy_low : L.stop);
    const hi = Math.max(tgtHi, L.price);
    const pad = 7;
    const map = function (p) { return clampPos(pad + (p - lo) / (hi - lo) * (100 - 2 * pad)); };
    const buyMid = L.buy_low != null && L.buy_high != null ? (L.buy_low + L.buy_high) / 2 : L.buy_low;

    const parts = [el("div", { class: "axis" })];
    // zones
    if (L.buy_low != null) parts.push(el("div", { class: "zone loss", style: "left:" + map(L.stop) + "%;width:" + (map(L.buy_low) - map(L.stop)) + "%" }));
    if (L.buy_low != null && L.buy_high != null) parts.push(el("div", { class: "zone buy", style: "left:" + map(L.buy_low) + "%;width:" + (map(L.buy_high) - map(L.buy_low)) + "%" }));
    if (L.target != null) parts.push(el("div", { class: "zone gain", style: "left:" + map(L.buy_high != null ? L.buy_high : buyMid) + "%;width:" + (map(L.target) - map(L.buy_high != null ? L.buy_high : buyMid)) + "%" }));
    // ticks
    parts.push(el("div", { class: "tick", style: "left:" + map(L.stop) + "%" }));
    if (L.target != null) parts.push(el("div", { class: "tick", style: "left:" + map(L.target) + "%" }));
    parts.push(el("div", { class: "tick now", style: "left:" + map(L.price) + "%" }));
    // caps  (top: buy + target ; bottom: stop + now)
    function cap(side, cls, k, v) { return el("div", { class: "cap " + side + " " + cls, style: "left:" + map(v) + "%" }, [el("span", { class: "k" }, k), el("span", { class: "pv num" }, price(v))]); }
    if (buyMid != null) parts.push(el("div", { class: "cap top buy", style: "left:" + map(buyMid) + "%" }, [el("span", { class: "k" }, "买入区"), el("span", { class: "pv num" }, price(L.buy_low) + "–" + price(L.buy_high))]));
    if (L.target != null) parts.push(cap("top", "target", "目标", L.target));
    parts.push(cap("bot", "stop", "止损", L.stop));
    parts.push(cap("bot", "now", "现价", L.price));

    const foot = [
      ["现价", price(L.price)],
      ["买入区", L.buy_low != null ? price(L.buy_low) + "–" + price(L.buy_high) : "—"],
      ["止损", price(L.stop)],
      ["目标", L.target != null ? price(L.target) + (L.target2 ? "→" + price(L.target2) : "") : "—"]
    ];
    return el("div", { class: "ladder-card" }, [
      el("div", { class: "ladder-head" }, [
        el("div", null, [el("span", { class: "px num" }, price(L.price)), L.change_pct != null ? el("small", null, " ") : null, L.change_pct != null ? signedSpan(L.change_pct) : null]),
        el("div", { class: "rrwrap" }, [el("div", { class: "lab" }, "盈亏比 R/R"), el("div", { class: "val", style: "color:" + rrColor(L.rr) }, L.rr != null ? Number(L.rr).toFixed(2) : "—")])
      ]),
      el("div", { class: "ladder" }, parts),
      el("div", { class: "ladder-foot" }, foot.map(function (f) { return el("div", { class: "cell" }, [el("div", { class: "k" }, f[0]), el("div", { class: "v" }, f[1])]); }))
    ]);
  }

  /* ---- technical stat strip ------------------------------------------- */
  function statStrip(stats) {
    return el("div", { class: "statgrid" }, stats.map(function (s) {
      return el("div", { class: "s" }, [
        el("div", { class: "k" }, s.k),
        el("div", { class: "v", html: s.vhtml || (s.v != null ? String(s.v) : "—"), style: s.color ? "color:" + s.color : "" }),
        s.x ? el("div", { class: "x" }, s.x) : null
      ]);
    }));
  }

  /* ---- levels table (portfolio) --------------------------------------- */
  function levelsTable(rows, opts) {
    opts = opts || {};
    const cols = [
      { k: "symbol", h: "标的", cls: "l sym" },
      { k: "name", h: "名称", cls: "l name", opt: true },
      { k: "sector", h: "板块", cls: "l sector", opt: true },
      { k: "price", h: "现价", f: price },
      { k: "change_pct", h: "今日", f: function (v) { return v == null ? "" : pctSigned(v); }, colorPct: true, opt: true },
      { k: "signal", h: "信号", sig: true, cls: "" },
      { k: "regime", h: "仓位", opt: true },
      { k: "buy", h: "建议买入区", buy: true, cls: "" },
      { k: "stop", h: "止损", f: price },
      { k: "target", h: "目标1", f: price },
      { k: "rr", h: "盈亏比 R/R", rr: true },
      { k: "rsi", h: "RSI", f: function (v) { return v == null ? "" : v; } },
      { k: "pctb", h: "%B", f: function (v) { return v == null ? "" : Number(v).toFixed(2); }, opt: true },
      { k: "note", h: "备注", cls: "l", note: true, opt: true }
    ];
    const present = cols.filter(function (c) {
      if (!c.opt) return true;
      return rows.some(function (r) { return c.buy ? r.buy_low != null : r[c.k] != null && r[c.k] !== ""; });
    });
    const thead = el("thead", null, el("tr", null, present.map(function (c) {
      return el("th", { class: /l/.test(c.cls || "") || c.k === "symbol" || c.k === "name" || c.k === "sector" || c.k === "note" ? "l" : "" }, c.h);
    })));
    const tbody = el("tbody", null, rows.map(function (r) {
      const watch = r.signal === "watch";
      return el("tr", { class: watch ? "is-watch" : "" }, present.map(function (c) {
        if (c.sig) return el("td", { class: "" }, sigBadge(r.signal));
        if (c.rr) return el("td", null, rrBar(r.rr));
        if (c.buy) return el("td", null, r.buy_low != null ? price(r.buy_low) + "–" + price(r.buy_high) : "—");
        if (c.note) return el("td", { class: "l note", html: (r.flag ? '<span class="flagcell">🔴 </span>' : "") + (r.note || "") });
        let v = r[c.k];
        if (c.colorPct && v != null && v !== "") {
          return el("td", { class: c.cls || "" }, el("span", { class: v > 0 ? "pos" : v < 0 ? "neg" : "" }, (c.f ? c.f(v) : v)));
        }
        return el("td", { class: c.cls || "" }, c.f ? c.f(v) : (v == null ? "—" : String(v)));
      }));
    }));
    return el("div", null, [
      el("div", { class: "tablewrap" }, el("table", { class: "grid" }, [thead, tbody])),
      opts.hint ? el("div", { class: "colhint", html: opts.hint }) : null
    ]);
  }

  /* ---- factor ranking table ------------------------------------------- */
  function factorTable(fr) {
    const heads = ["排名", "标的", "现价", "6月动量", "12月动量", "年化波动", "信号", "仓位", "综合分"];
    const thead = el("thead", null, el("tr", null, heads.map(function (h, i) {
      return el("th", { class: i === 1 ? "l" : "" }, h);
    })));
    const tbody = el("tbody", null, fr.rows.map(function (r, i) {
      return el("tr", null, [
        el("td", null, r.rank != null ? r.rank : i + 1),
        el("td", { class: "l sym" }, r.symbol + (r.leveraged ? "*" : "")),
        el("td", null, price(r.price)),
        el("td", null, el("span", { class: r.m6 > 0 ? "pos" : "neg" }, pctSigned(r.m6, 0))),
        el("td", null, el("span", { class: r.m12 > 0 ? "pos" : "neg" }, pctSigned(r.m12, 0))),
        el("td", null, r.vol != null ? r.vol + "%" : "—"),
        el("td", null, sigBadge(r.signal)),
        el("td", null, r.regime || "—"),
        el("td", null, el("span", { class: r.score > 0 ? "pos" : "neg", style: "font-weight:700" }, scoreStr(r.score)))
      ]);
    }));
    return el("div", null, [
      el("div", { class: "tablewrap" }, el("table", { class: "grid" }, [thead, tbody])),
      fr.hint ? el("div", { class: "colhint", html: fr.hint }) : null
    ]);
  }

  /* ---- sentiment ------------------------------------------------------- */
  function sentiment(s) {
    function bar(v) {
      const pos = pct01((v + 1) / 2);
      const center = 50, w = Math.abs(pos - center);
      const left = v >= 0 ? center : pos;
      const col = v > 0.05 ? "var(--pos)" : v < -0.05 ? "var(--neg)" : "var(--warn)";
      return el("div", { class: "sbar-wrap" }, [
        el("div", { class: "sbar", style: "flex:1" }, [el("span", { class: "mid" }), el("span", { class: "fill", style: "left:" + left + "%;width:" + w + "%;background:" + col })]),
        el("span", { class: "sv", style: "color:" + col }, scoreStr(v))
      ]);
    }
    const rows = (s.layers || []).map(function (l) {
      return el("div", { class: "s-row" }, [el("div", { class: "lay" }, l.layer), bar(l.score), el("div", { class: "key", html: l.key || "" })]);
    });
    if (s.composite != null) rows.push(el("div", { class: "s-row comp" }, [el("div", { class: "lay" }, "复合"), bar(s.composite), el("div", { class: "key", html: s.note || "三层加权" })]));
    return el("div", { class: "senti" }, rows);
  }

  /* ---- portfolio health ----------------------------------------------- */
  function portfolioHealth(ph) {
    const thead = el("thead", null, el("tr", null, [el("th", { class: "l" }, "指标"), el("th", null, "数值"), el("th", { class: "l" }, "解读")]));
    const tbody = el("tbody", null, ph.rows.map(function (r) {
      return el("tr", null, [
        el("td", { class: "l", style: "font-family:var(--sans)" }, r.metric),
        el("td", { html: r.value }),
        el("td", { class: "l", html: (r.flag ? '<span class="flagcell">🔴 </span>' : "") + (r.read || "") })
      ]);
    }));
    return el("div", null, [
      el("div", { class: "tablewrap" }, el("table", { class: "grid pf" }, [thead, tbody])),
      ph.conclusion ? el("div", { class: "callout-quote", html: ph.conclusion }) : null
    ]);
  }

  /* ---- equity / benchmark chart --------------------------------------- */
  function equityChart(bt) {
    const W = 1000, H = 300, pl = 6, pr = 6, pt = 14, pb = 24;
    const eq = bt.equity || {};
    const A = eq.strategy || [], B = eq.benchmark || [];
    const n = Math.max(A.length, B.length);
    if (n < 2) return null;
    let mn = Infinity, mx = -Infinity;
    [A, B].forEach(function (s) { s.forEach(function (v) { if (v < mn) mn = v; if (v > mx) mx = v; }); });
    const px = function (i) { return pl + i / (n - 1) * (W - pl - pr); };
    const py = function (v) { return pt + (1 - (v - mn) / (mx - mn || 1)) * (H - pt - pb); };
    const path = function (s) { return s.map(function (v, i) { return (i ? "L" : "M") + px(i).toFixed(1) + " " + py(v).toFixed(1); }).join(" "); };
    const grid = [];
    for (let g = 0; g <= 4; g++) { const y = pt + g / 4 * (H - pt - pb); grid.push('<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y + '" y2="' + y + '" stroke="#e7e3d8" stroke-width="1"/>'); }
    const svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" style="width:100%;height:230px;display:block">' +
      grid.join("") +
      (B.length ? '<path d="' + path(B) + '" fill="none" stroke="#9aa0aa" stroke-width="2" stroke-dasharray="5 4"/>' : "") +
      '<path d="' + path(A) + '" fill="none" stroke="#1b3a5b" stroke-width="2.5"/>' +
      '</svg>';
    const stats = (bt.stats || []).map(function (s) { return el("div", { class: "cs", html: s.k + "<b>" + s.v + "</b>" }); });
    return el("div", { class: "chart-card" }, [
      el("div", { class: "chart-head" }, [
        el("div", { class: "legend" }, [
          el("span", null, [el("i", { style: "background:#1b3a5b" }), bt.strategy_label || "策略"]),
          el("span", null, [el("i", { style: "background:#9aa0aa" }), bt.benchmark_label || "买入持有"])
        ])
      ]),
      el("div", { html: svg }),
      stats.length ? el("div", { class: "chart-stats" }, stats) : null
    ]);
  }

  /* ---- generic prose / groups ----------------------------------------- */
  function proseBlock(p) {
    if (typeof p === "string") return el("div", { class: "prose", html: p });
    return el("div", { class: "prose" }, (p || []).map(function (x) { return el("p", { html: x }); }));
  }
  function groups(list) {
    return el("div", { class: "groups" }, list.map(function (g) {
      return el("div", { class: "group" }, [
        el("div", { class: "gt", html: g.title + (g.tag ? '<span class="tag ' + (g.tone || "neutral") + '">' + g.tag + "</span>" : "") }),
        el("div", { class: "gd", html: g.body })
      ]);
    }));
  }

  /* ====================================================================== */
  /*  ORCHESTRATION                                                         */
  /* ====================================================================== */
  function block(no, title, hnote, content) {
    if (!content) return null;
    return el("section", { class: "block" }, [
      el("div", { class: "sec-head" }, [
        no ? el("div", { class: "no" }, no) : null,
        el("h2", null, title),
        hnote ? el("div", { class: "h-note" }, hnote) : null
      ]),
      content
    ]);
  }

  function render(data, mount) {
    mount.innerHTML = "";
    const frag = document.createDocumentFragment();
    frag.appendChild(masthead(data.meta));
    const body = el("div", { class: "body" });

    // verdict is unnumbered, sits at top
    if (data.verdict) body.appendChild(el("section", { class: "block", style: "border-top:none;padding-top:22px" }, verdict(data.verdict)));

    let no = 0;
    function add(title, hnote, content) { if (content) { no++; body.appendChild(block(no, title, hnote, content)); } }

    if (data.alerts) add("🔴 今日重点关注", data.alerts.length + " 项", alerts(data.alerts));

    // environment composite
    if (data.regime || data.macro || data.calendar) {
      const panels = [];
      if (data.regime) panels.push(regimePanel(data.regime));
      if (data.macro) panels.push(macroPanel(data.macro));
      const grid = el("div", { class: "env-grid" }, panels);
      const wrap = el("div", null, [grid, data.calendar ? el("div", { style: "margin-top:18px" }, calendarPanel(data.calendar)) : null]);
      add("市场环境 · 宏观 · 事件", null, wrap);
    }

    if (data.factor_rank) add(data.factor_rank.title || "多因子排序", data.factor_rank.note_head || null, factorTable(data.factor_rank));
    if (data.groups) add(data.groups_title || "分组解读", null, groups(data.groups));

    // single-name hero
    if (data.technical) {
      const parts = [];
      if (data.technical.level) parts.push(levelLadder(data.technical.level));
      if (data.technical.stats) parts.push(el("div", { style: "margin-top:18px" }, statStrip(data.technical.stats)));
      if (data.technical.note) parts.push(el("div", { class: "prose", style: "margin-top:8px" }, proseBlock(data.technical.note)));
      add(data.technical.title || "📈 技术与买卖点", data.technical.signal_note || null, el("div", null, parts));
    }

    if (data.levels) add(data.levels_title || "🎯 建议买卖点", null, levelsTable(data.levels, { hint: data.levels_hint }));
    if (data.backtest) { const c = equityChart(data.backtest); if (c) add(data.backtest.title || "策略回测", data.backtest.head_note || null, c); }
    if (data.sentiment) add(data.sentiment.title || "🗞 三层时效情绪", data.sentiment.composite != null ? "复合 " + scoreStr(data.sentiment.composite) : null, sentiment(data.sentiment));
    if (data.portfolio_health) add(data.portfolio_health.title || "🧩 组合体检", null, portfolioHealth(data.portfolio_health));
    if (data.holdings) add("你的持仓", null, proseBlock(data.holdings));
    if (data.conclusion) add("综合结论", null, proseBlock(data.conclusion));

    frag.appendChild(body);

    // footer
    const foot = el("div", { class: "footer" }, [
      data.disclaimer ? el("div", { class: "disc", html: data.disclaimer }) : null,
      data.sources && data.sources.length ? el("div", { class: "sources", html: "Sources · " + data.sources.map(function (s) { return s.url ? '<a href="' + s.url + '" target="_blank" rel="noopener">' + s.label + "</a>" : s.label; }).join(" · ") }) : null,
      el("div", { class: "sign" }, [
        el("span", { html: '本页由 <span class="mk">' + ((data.meta && data.meta.generated_by) || "alpha-forge") + '</span> 技能生成 · 机械量化研究，非投资建议' }),
        el("span", null, (data.meta && data.meta.date) || "")
      ])
    ]);
    frag.appendChild(foot);

    mount.appendChild(frag);
    document.title = (data.meta && data.meta.title) || "量化分析报告";
  }

  global.QuantReport = { render: render };
})(window);

"""


def _title(report: dict) -> str:
    meta = report.get("meta") or {}
    return meta.get("title") or "量化分析报告"


def render(report: dict) -> str:
    """结构化 report dict -> 一份完整、自包含、带样式的 HTML 字符串。"""
    # 把 JSON 里所有 < 转成 \u003c，确保内嵌的 HTML 片段不会提前关闭 <script> 块
    data = _json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    return (
        '<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>' + _html.escape(_title(report)) + '</title>\n<style>\n' + _CSS + '\n</style></head>\n'
        '<body>\n<div class="sheet" id="sheet"></div>\n'
        '<script id="report-data" type="application/json">' + data + '</script>\n'
        '<script>\n' + _JS + '\n</script>\n'
        '<script>QuantReport.render(JSON.parse('
        'document.getElementById("report-data").textContent),'
        ' document.getElementById("sheet"));</script>\n</body></html>'
    )


def save_html(report: dict, path: str) -> str:
    """把结构化 report dict 写成单文件自包含 .html 报告，返回路径。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(report))
    return path
