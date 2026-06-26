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
.verdict .stance .arrow{font-size:20px;line-height:1;letter-spacing:1px}
.verdict .stance .lvl{font-size:11px;font-weight:700;letter-spacing:.1em;margin-top:5px;font-family:var(--sans)}
.verdict.up .val,.verdict.up .arrow,.verdict.up .lvl{color:var(--pos)}
.verdict.down .val,.verdict.down .arrow,.verdict.down .lvl{color:var(--neg)}
.verdict.flat .val,.verdict.flat .arrow,.verdict.flat .lvl{color:var(--warn)}
.verdict .body-v{padding:18px 24px}
.verdict .action{font-family:var(--serif);font-size:16.5px;font-weight:600;color:var(--ink);
  line-height:1.4;margin:0 0 8px}
.verdict .summary{font-size:13px;color:var(--ink-soft);line-height:1.62;margin:0}
.verdict .vpoints{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:9px}
.verdict .vpt{display:flex;gap:9px;align-items:flex-start;font-size:13px;line-height:1.55;color:var(--ink-soft)}
.verdict .vpt .vpi{flex:0 0 auto;font-size:14px;line-height:1.4}
.verdict .vpt .vptx{flex:1;min-width:0}
.verdict .vpt b{color:var(--ink);font-weight:600}

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
.env-grid.one{grid-template-columns:1fr}
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
.dl .row{display:grid;grid-template-columns:minmax(92px,116px) minmax(78px,34%) minmax(0,1fr);gap:12px;padding:7px 0;
  border-top:1px dashed var(--hair);font-size:12.5px;align-items:baseline}
.dl .row:first-child{border-top:none}
.dl .row .k{color:var(--muted)}
.dl .row .v{font-family:var(--mono);font-weight:600;text-align:left;overflow-wrap:anywhere;line-height:1.45}
.dl .row .v.tag{font-family:var(--sans);font-weight:400}
.vbadge{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:3px;line-height:1.55;white-space:nowrap}
.vbadge.pos{background:var(--pos-wash);color:var(--pos)}
.vbadge.warn{background:var(--warn-wash);color:var(--warn)}
.vbadge.neu{background:var(--hair);color:var(--ink-soft)}
/* methods comparison table (信号多法对照) */
.mtbl{width:100%;border-collapse:collapse;font-size:12px;margin-top:2px}
.mtbl th{background:var(--accent);color:#fff;font-weight:600;font-size:11.5px;padding:7px 9px;text-align:left;white-space:nowrap}
.mtbl td{border-bottom:1px solid var(--hair);padding:8px 9px;vertical-align:top}
.mtbl .mname{font-weight:600;white-space:nowrap}
.mtbl .mdesc{color:var(--muted);font-size:10.5px;line-height:1.4;max-width:140px}
.mtbl .mlab{display:inline-block;font-weight:600;font-size:11.5px;padding:1px 8px;border-radius:3px}
.mtbl .mlab.pos{background:var(--pos-wash);color:var(--pos)}
.mtbl .mlab.neg{background:var(--neg-wash);color:var(--neg)}
.mtbl .mlab.warn{background:var(--warn-wash);color:var(--warn)}
.mtbl .mlab.neu{background:var(--hair);color:var(--ink-soft)}
.mtbl .mdet{font-size:10px;color:var(--muted);line-height:1.38;margin-top:3px}
.mtbl .mrow-old td{background:#fbfaf7}
/* auto-research detail */
.rs-item{border:1px solid var(--hair-2);background:#fcfbf8;padding:14px 16px;margin-bottom:16px}
.rs-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:11px}
.rs-head .rs-name{font-family:var(--serif);font-weight:700;font-size:16px}
.rs-head .rs-win{font-size:12px;font-weight:600;color:var(--accent);background:#eef2f7;padding:2px 9px;border-radius:3px}
.rs-facts{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:0 0 11px}
.rs-facts>div{background:#fff;border:1px solid var(--hair);border-radius:4px;padding:7px 10px}
.rs-facts .rk{display:block;font-size:11px;color:var(--muted)}
.rs-facts .rv{display:block;font-size:13px;font-weight:600;margin-top:2px}
.rs-sel{font-size:11.5px;color:var(--ink-soft);margin:0 0 8px;line-height:1.55}
.rs-sub{font-family:var(--serif);font-weight:600;font-size:13px;margin:12px 0 6px}
.rmtbl{width:100%;border-collapse:collapse;font-size:11px}
.rmtbl .rmbuy{color:var(--pos)}
.rmtbl .rmsell{color:var(--neg)}
.rmtbl th{background:var(--accent);color:#fff;font-weight:600;padding:5px 8px;text-align:left;white-space:nowrap}
.rmtbl td{border-bottom:1px solid var(--hair);padding:5px 8px}
.rmtbl .rmp{color:var(--muted);font-size:11px}
.rmtbl .rmwin{background:var(--pos-wash)}
.rmtbl .rmwin td{font-weight:700}
@media(max-width:760px){.rs-facts{grid-template-columns:repeat(2,1fr)}}
.tc-legend{display:flex;gap:16px;align-items:center;font-size:11.5px;color:var(--ink-soft);padding:2px 2px 6px;flex-wrap:wrap}
.tc-legend .hsw{display:inline-block;width:14px;height:10px;background:#b8923f;opacity:.40;vertical-align:-1px;margin-right:4px}
.tc-legend .tc-now{margin-left:auto;font-weight:600;color:var(--ink)}
.rs-cap{font-size:11.5px;color:var(--ink-soft);line-height:1.55;margin:8px 2px 0;background:#fbfaf7;border-left:3px solid var(--hair-2);padding:8px 12px}
.rs-trig{display:flex;gap:14px;flex-wrap:wrap;align-items:center;background:#eef2f7;border:1px solid #d7e0ea;border-radius:4px;padding:8px 12px;margin:0 0 11px;font-size:12px;line-height:1.5}
.rs-trig .rs-trig-h{font-weight:700;color:var(--accent)}
.rs-trig b{color:var(--ink)}
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
/* groups: news mini-cards (layout:cards) */
.gcards{display:flex;flex-direction:column;gap:9px;margin-top:3px}
.gcard{border:1px solid var(--hair);border-left:3px solid var(--hair-2);border-radius:0 4px 4px 0;background:#fcfbf8;padding:10px 14px}
.gcard.strong{border-left-color:var(--pos)}
.gcard.weak{border-left-color:var(--neg)}
.gcard.neutral{border-left-color:var(--warn)}
.gcard .gch{font-family:var(--serif);font-weight:600;font-size:13.5px;margin:0 0 5px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.gcard .gch .tag{font-family:var(--sans);font-size:10px;font-weight:600;padding:1px 7px;border-radius:2px}
.gcard .gci{margin:0;padding:0 0 0 17px}
.gcard .gci li{font-size:12.5px;color:var(--ink-soft);line-height:1.7;margin:3px 0}
.gfoot{font-size:11.5px;color:var(--faint);font-style:italic;margin:8px 2px 0;line-height:1.55}
/* groups: options implied-vs-actual bar (layout:vol) */
.volwrap{margin-top:4px}
.volbar{position:relative;height:48px;margin:18px 8px 2px}
.vbtrack{position:absolute;top:20px;left:0;right:0;height:6px;background:var(--hair);border-radius:3px}
.vbband{position:absolute;top:20px;height:6px;background:var(--warn-wash);border:1px solid var(--warn);border-radius:3px;box-sizing:border-box}
.vbzero{position:absolute;top:12px;height:22px;width:1px;background:var(--ink-soft);opacity:.5}
.vbtick{position:absolute;top:30px;transform:translateX(-50%);font-size:10.5px;color:var(--faint);white-space:nowrap}
.vbdot{position:absolute;top:14px;width:16px;height:16px;border-radius:50%;transform:translateX(-50%);border:2px solid #fff;box-shadow:0 0 0 1px var(--hair-2)}
.vbdot.pos{background:var(--pos)}
.vbdot.neg{background:var(--neg)}
.vbact{position:absolute;top:-4px;font-size:11px;font-weight:700;white-space:nowrap}
.vbact.pos{color:var(--pos)}
.vbact.neg{color:var(--neg)}
.vblegend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;color:var(--ink-soft);margin:8px 8px 0}
.vblegend .sw{display:inline-block;width:10px;height:10px;margin-right:5px;vertical-align:-1px}
.vblegend .sw.band{background:var(--warn-wash);border:1px solid var(--warn);border-radius:2px}
.vblegend .sw.dot{border-radius:50%}
.vblegend .sw.dot.pos{background:var(--pos)}
.vblegend .sw.dot.neg{background:var(--neg)}
.voliv{font-size:12px;color:var(--ink-soft);margin:10px 0 0;line-height:1.6}
/* conclusion: dimension cards + stance tag (style B) */
.concl{display:flex;flex-direction:column;gap:9px}
.ccard{border:1px solid var(--hair);border-left:3px solid var(--hair-2);border-radius:0 4px 4px 0;background:#fcfbf8;padding:10px 15px}
.ccard.pos{border-left-color:var(--pos)}
.ccard.neg{border-left-color:var(--neg)}
.ccard.warn{border-left-color:var(--warn)}
.ccard.mut{border-left-color:var(--hair-2)}
.ccard .chead{display:flex;align-items:center;gap:8px;margin:0 0 5px}
.ccard .cicon{font-size:14px;line-height:1}
.ccard .clabel{font-family:var(--serif);font-weight:600;font-size:14px}
.ccard .ctag{font-family:var(--sans);font-size:10.5px;font-weight:600;padding:1px 8px;border-radius:2px;margin-left:auto;white-space:nowrap}
.ccard .ctag.pos{background:var(--pos-wash);color:var(--pos)}
.ccard .ctag.neg{background:var(--neg-wash);color:var(--neg)}
.ccard .ctag.warn{background:var(--warn-wash);color:var(--warn)}
.ccard .ctag.mut{background:var(--hair);color:var(--muted)}
.ccard .cbody{font-size:12.5px;color:var(--ink-soft);line-height:1.65}

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
  // Minimal HTML hygiene for DATA-derived strings. Reports are built by the model from
  // curated data, but news / web / AI text can carry stray markup. This strips the
  // genuinely dangerous constructs (script & friends, inline on*= handlers, javascript:
  // URLs) while leaving the benign inline formatting the schema documents (<b>/<i>/<span
  // class=…>) intact. Code-controlled markup (the SVG charts) bypasses this via `raw:`.
  function safeHtml(s) {
    if (s == null) return "";
    s = String(s)
      .replace(/<\s*\/?\s*(script|style|iframe|object|embed|link|meta|svg|img|video|audio|base|form|input)\b[^>]*>/gi, "")
      .replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "")
      .replace(/(href|src)\s*=\s*("\s*javascript:[^"]*"|'\s*javascript:[^']*'|javascript:[^\s>]+)/gi, '$1="#"');
    return s;
  }
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = safeHtml(attrs[k]);   // data -> sanitized
      else if (k === "raw") n.innerHTML = attrs[k];              // code-controlled markup (SVG)
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
  function pctStr(v, dp) { dp = dp == null ? 1 : dp; return (v > 0 ? "+" : "") + Number(v).toFixed(dp) + "%"; }
  function vsPrice(level, p) { return (p != null && p > 0 && level != null) ? pctStr((level / p - 1) * 100) : null; }
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
    // Two badge styles, both supported:
    //  (1) technical codes long/watch/short -> 做多/观望/做空 (legacy).
    //  (2) event-sentiment strings judged by the AI from news, optionally with a holding
    //      period, e.g. "利多·短线" / "利多·中线" / "利空·短线" / "中性". Colored by the
    //      sentiment word (利多/利好→red, 利空→green, else amber); the text shows verbatim.
    var m = SIG[s], cls, label;
    if (m) { cls = m[0]; label = m[1]; }
    else {
      var str = s || "—";
      cls = /利多|利好|偏多|做多|看多|bull/i.test(str) ? "long"
          : /利空|利淡|偏空|做空|看空|bear/i.test(str) ? "short" : "watch";
      label = str;
    }
    return el("span", { class: "sig " + cls }, textOverride || label);
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

  function verdict(v, envScore) {
    if (!v) return null;
    // 5-level stance. Priority: explicit v.score > envScore (objective: blended regime/macro
    // meters) > NET keyword lean (bullish − bearish hits; fixes the old first-match bug).
    var lvl;
    var sc = (typeof v.score === "number") ? v.score : (typeof envScore === "number") ? envScore : null;
    if (sc !== null) {
      var x = sc;
      lvl = x >= 0.6 ? 2 : x >= 0.15 ? 1 : x <= -0.6 ? -2 : x <= -0.15 ? -1 : 0;
    } else {
      var sx = v.stance || "";
      var bull = (sx.match(/多|涨|强|升|新高|突破|反弹|看多|做多|bull|牛/gi) || []).length;
      var bear = (sx.match(/空|跌|弱|新低|破位|回落|下行|看空|做空|bear|熊/gi) || []).length;
      var net = bull - bear;
      if (net === 0) { lvl = 0; }
      else {
        var strong = (/强烈|显著|大幅|坚定|重仓|满仓|全仓/.test(sx) || Math.abs(net) >= 2);
        var temper = /超买|超卖|延伸|不追|谨慎|高位|震荡|观望|温和|分化|控制仓位|中性/.test(sx);
        var mag = (strong && !temper) ? 2 : 1;
        lvl = net > 0 ? mag : -mag;
      }
    }
    var dir = lvl > 0 ? "up" : lvl < 0 ? "down" : "flat";
    var arrow = lvl >= 2 ? "▲▲▲" : lvl === 1 ? "▲" : lvl === 0 ? "◆" : lvl === -1 ? "▼" : "▼▼▼";
    var name = lvl >= 2 ? "强烈看多" : lvl === 1 ? "偏多" : lvl === 0 ? "中性" : lvl === -1 ? "偏空" : "强烈看空";
    return el("div", { class: "verdict " + dir }, [
      el("div", { class: "stance" }, [
        el("div", { class: "lab" }, "综合立场"),
        el("div", { class: "arrow" }, arrow),
        el("div", { class: "val" }, v.stance || name),
        el("div", { class: "lvl" }, name)
      ]),
      el("div", { class: "body-v" }, [
        (v.points && v.points.length) ? el("ul", { class: "vpoints" }, v.points.map(function (pt) {
          var t = (typeof pt === "string") ? { text: pt } : (pt || {});
          return el("li", { class: "vpt" }, [
            t.icon ? el("span", { class: "vpi" }, t.icon) : null,
            el("span", { class: "vptx", html: t.text || "" })
          ]);
        })) : (v.action ? el("p", { class: "action", html: v.action }) : null),
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

  function vCell(value, tone) {
    var t = (value == null) ? "" : String(value);
    if (!/[一-鿿]/.test(t)) return el("div", { class: "v", html: t });  // 数字/英文 -> 等宽
    var tn = tone;
    if (!tn) {
      if (/延伸|偏紧|收紧|紧张|谨慎|降温|回落|承压|高位|拥挤|峰值|震荡|分化|观望|过热|风险|回调|压力|疲软|放缓|降|弱/.test(t)) tn = "warn";
      else if (/加速|超级|强|高|利好|扩张|改善|新高|大超|暴击|领先|饱满|确定|顺风|放量|景气|增长|回暖|向好|缓和|稳健|健康|宽松/.test(t)) tn = "pos";
      else tn = "neu";
    }
    return el("div", { class: "v tag" }, el("span", { class: "vbadge " + tn }, t));
  }
  function regimePanel(r) {
    const rows = (r.rows || []).map(function (row) {
      return el("div", { class: "row" }, [
        el("div", { class: "k" }, row.item),
        vCell(row.value, row.tone),
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
    if (m.vix != null) dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, "VIX"), vCell(m.vix), el("div", { class: "r" }, m.vix_note || "")]));
    (m.rows || []).forEach(function (row) {
      dl.push(el("div", { class: "row" }, [el("div", { class: "k" }, row.item), vCell(row.value, row.tone), el("div", { class: "r", html: row.read || "" })]));
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
    const span = (hi - lo) || 1;          // guard degenerate stop==price==target (no NaN%)
    const pad = 7;
    const map = function (p) { return clampPos(pad + (p - lo) / span * (100 - 2 * pad)); };
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
      ["买入区", L.buy_low != null ? price(L.buy_low) + "–" + price(L.buy_high) + (L.price ? " (" + pctStr((L.buy_low / L.price - 1) * 100) + "~" + pctStr((L.buy_high / L.price - 1) * 100) + ")" : "") : "—"],
      ["止损", price(L.stop) + (L.price ? " (" + pctStr((L.stop / L.price - 1) * 100) + ")" : "")],
      ["目标", L.target != null ? price(L.target) + (L.target2 ? "→" + price(L.target2) : "") + (L.price ? " (" + pctStr((L.target / L.price - 1) * 100) + ")" : "") : "—"]
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
      { k: "support1", h: "支撑", supp: true, opt: true },
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
        if (c.buy) { if (r.buy_low == null) return el("td", null, "—"); var _s = vsPrice(r.buy_low, r.price), _e = vsPrice(r.buy_high, r.price); return el("td", null, [el("span", { class: "num" }, price(r.buy_low) + "–" + price(r.buy_high)), (_s && _e) ? el("small", { class: "muted" }, " (" + _s + "~" + _e + ")") : null]); }
        if (c.supp) { if (r.support1 == null) return el("td", null, "—"); var _s1 = vsPrice(r.support1, r.price); var _kids = [el("span", { class: "num" }, price(r.support1)), _s1 ? el("small", { class: "muted" }, " (" + _s1 + ")") : null]; if (r.support2 != null) { var _s2 = vsPrice(r.support2, r.price); _kids.push(el("small", { class: "muted" }, " / " + price(r.support2) + (_s2 ? " (" + _s2 + ")" : ""))); } return el("td", null, _kids); }
        if (c.note) return el("td", { class: "l note", html: (r.flag ? '<span class="flagcell">🔴 </span>' : "") + (r.note || "") });
        let v = r[c.k];
        if ((c.k === "stop" || c.k === "target") && v != null && v !== "") { var _pp = vsPrice(v, r.price); return el("td", null, [el("span", { class: "num" }, price(v)), _pp ? el("small", { class: "muted" }, " (" + _pp + ")") : null]); }
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
    const heads = ["排名", "标的", "现价", "6月动量", "12月动量", "年化波动", "仓位", "综合分"];
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

  /* ---- generic prose / groups ----------------------------------------- */
  function proseBlock(p) {
    if (typeof p === "string") return el("div", { class: "prose", html: p });
    return el("div", { class: "prose" }, (p || []).map(function (x) { return el("p", { html: x }); }));
  }
  function conclusionBlock(p) {
    if (typeof p === "string") return el("div", { class: "prose", html: p });
    var arr = p || [];
    var structured = arr.some(function (x) { return x && typeof x === "object"; });
    if (!structured) return el("div", { class: "prose" }, arr.map(function (x) { return el("p", { html: x }); }));
    function stanceCls(s) {
      if (!s) return "mut";
      if (/利空|利淡|偏空|做空|看空|bear/i.test(s)) return "neg";
      if (/风险|警惕|谨慎|观望|延伸|watch|caution|risk/i.test(s)) return "warn";
      if (/利多|利好|偏多|做多|看多|强劲|bull/i.test(s)) return "pos";
      return "mut";
    }
    var ICON = { "基本面": "📊", "技术": "📈", "期权": "📐", "情景": "🔭", "场景": "🔭", "宏观": "🌐" };
    return el("div", { class: "concl" }, arr.map(function (x) {
      if (typeof x === "string") return el("div", { class: "ccard mut" }, el("div", { class: "cbody", html: x }));
      var st = stanceCls(x.stance);
      var head = el("div", { class: "chead" }, [
        el("span", { class: "cicon", html: x.icon || ICON[x.label] || "•" }),
        el("span", { class: "clabel", html: x.label || "" }),
        x.stance ? el("span", { class: "ctag " + st, html: x.stance }) : null
      ]);
      return el("div", { class: "ccard " + st }, [head, el("div", { class: "cbody", html: x.text || "" })]);
    }));
  }
  function groups(list) {
    function senti(s) {
      if (!s) return null;
      if (/利多|利好|偏多|做多|看多|bull/i.test(s)) return { cls: "strong", txt: s };
      if (/利空|利淡|偏空|做空|看空|bear/i.test(s)) return { cls: "weak", txt: s };
      return { cls: "neutral", txt: s };
    }
    function chip(s) { var x = senti(s); return x ? '<span class="tag ' + x.cls + '">' + x.txt + "</span>" : ""; }
    function cardLayout(g) {
      var cards = el("div", { class: "gcards" }, (g.cards || []).map(function (c) {
        var x = senti(c.sentiment);
        return el("div", { class: "gcard" + (x ? " " + x.cls : "") }, [
          el("div", { class: "gch", html: (c.label || "") + chip(c.sentiment) }),
          el("ul", { class: "gci" }, (c.items || []).map(function (it) { return el("li", { html: it }); }))
        ]);
      }));
      if (!g.foot) return cards;
      return el("div", null, [cards, el("div", { class: "gfoot", html: g.foot })]);
    }
    function volLayout(g) {
      var v = g.vol || {};
      var lo = (v.implied_low != null) ? v.implied_low : -(v.implied != null ? v.implied : 8);
      var hi = (v.implied_high != null) ? v.implied_high : (v.implied != null ? v.implied : 8);
      var act = (v.actual != null) ? v.actual : 0;
      var span = Math.max(Math.abs(act), Math.abs(lo), Math.abs(hi)) * 1.4;
      if (span < 12) span = 12;
      function pos(x) { return (x + span) / (2 * span) * 100; }
      function pct(x) { return (x > 0 ? "+" : "") + (Math.round(x * 10) / 10) + "%"; }
      var ac = act >= 0 ? "pos" : "neg";
      var pa = pos(act);
      var tShift = pa >= 70 ? "translateX(-100%)" : pa <= 30 ? "translateX(0)" : "translateX(-50%)";
      var bar = el("div", { class: "volbar" }, [
        el("div", { class: "vbtrack" }),
        el("div", { class: "vbband", style: "left:" + pos(lo).toFixed(1) + "%;width:" + (pos(hi) - pos(lo)).toFixed(1) + "%" }),
        el("div", { class: "vbzero", style: "left:" + pos(0).toFixed(1) + "%" }),
        el("div", { class: "vbtick", style: "left:" + pos(0).toFixed(1) + "%", html: "0" }),
        el("div", { class: "vbtick", style: "left:" + pos(lo).toFixed(1) + "%", html: pct(lo) }),
        el("div", { class: "vbtick", style: "left:" + pos(hi).toFixed(1) + "%", html: pct(hi) }),
        el("div", { class: "vbdot " + ac, style: "left:" + pa.toFixed(1) + "%" }),
        el("div", { class: "vbact " + ac, style: "left:" + pa.toFixed(1) + "%;transform:" + tShift, html: "实际 " + pct(act) })
      ]);
      var legend = el("div", { class: "vblegend" }, [
        el("span", { html: '<i class="sw band"></i>期权隐含区间' }),
        el("span", { html: '<i class="sw dot ' + ac + '"></i>实际跳空(冲出区间)' })
      ]);
      var kids = [bar, legend];
      if (v.iv != null) kids.push(el("div", { class: "voliv", html: "IV " + v.iv + "%" + (v.iv_pctile != null ? "(52周 " + v.iv_pctile + " 百分位)" : "") + " → 财报落地后通常回落(vol crush)" }));
      if (v.note) kids.push(el("div", { class: "gd", style: "margin-top:8px", html: v.note }));
      return el("div", { class: "volwrap" }, kids);
    }
    return el("div", { class: "groups" }, list.map(function (g) {
      var inner;
      if (g.layout === "cards" && g.cards) inner = cardLayout(g);
      else if (g.layout === "vol" && g.vol) inner = volLayout(g);
      else inner = el("div", { class: "gd", html: g.body });
      return el("div", { class: "group" }, [
        el("div", { class: "gt", html: g.title + (g.tag ? '<span class="tag ' + (g.tone || "neutral") + '">' + g.tag + "</span>" : "") }),
        inner
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

  function methodsTable(m) {
    var syms = m.symbols || [];
    var data = m.data || {};
    function cell(sym, key) {
      var d = (data[sym] || {})[key];
      if (!d) return el("td", null, "—");
      return el("td", null, [
        el("span", { class: "mlab " + (d.tone || "neu") }, d.label || "—"),
        d.detail ? el("div", { class: "mdet" }, d.detail) : null
      ]);
    }
    var thead = el("thead", null, el("tr", null,
      [el("th", null, "判定方法"), el("th", null, "说明")].concat(
        syms.map(function (s) { return el("th", null, s.name || s.key); }))));
    var tbody = el("tbody", null, (m.rows || []).map(function (r) {
      return el("tr", { class: r.key === "old" ? "mrow-old" : "" },
        [el("td", null, el("div", { class: "mname" }, r.m)),
         el("td", null, el("div", { class: "mdesc" }, r.desc || ""))]
        .concat(syms.map(function (s) { return cell(s.key, r.key); })));
    }));
    var wrap = el("div", { class: "tablewrap" }, el("table", { class: "mtbl" }, [thead, tbody]));
    return m.note ? el("div", null, [wrap, el("p", { class: "p-note", style: "margin-top:10px", html: m.note })]) : wrap;
  }
  function tradesChart(t) {
    var P = t.price || []; var n = P.length; if (n < 2) return null;
    var W = 1000, H = 300, pl = 56, pr = 12, ptp = 10, pb = 20, logy = !!t.logy;
    var mn = Math.min.apply(null, P), mx = Math.max.apply(null, P);
    (t.overlays || []).forEach(function (o) { (o.data || []).forEach(function (v) { if (v != null && isFinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; } }); });
    var llo = Math.log(mn > 0 ? mn : 1e-9), lhi = Math.log(mx > 0 ? mx : 1);
    function X(i) { return pl + i / (n - 1) * (W - pl - pr); }
    function Y(v) { var f = logy ? (Math.log(v) - llo) / ((lhi - llo) || 1) : (v - mn) / ((mx - mn) || 1); return ptp + (1 - f) * (H - ptp - pb); }
    function fmt(v) { var u = t.unit || ""; return v >= 1e6 ? u + (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? u + (v / 1e3).toFixed(0) + "k" : u + v.toFixed(0); }
    function pathOf(a) { var d = "", pen = false; for (var i = 0; i < a.length; i++) { var v = a[i]; if (v == null || !isFinite(v)) { pen = false; continue; } d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1) + " "; pen = true; } return d; }
    var pr2 = [];
    var labs = logy ? [mx, Math.exp((llo + lhi) / 2), mn] : [mx, (mn + mx) / 2, mn];
    [0, 0.5, 1].forEach(function (g, gi) { var y = ptp + g * (H - ptp - pb);
      pr2.push('<line x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y + '" y2="' + y + '" stroke="#e7e3d8" stroke-width="1"/>');
      pr2.push('<text x="' + (pl - 6) + '" y="' + (y + 3.5) + '" text-anchor="end" font-size="11" fill="#8a8474">' + fmt(labs[gi]) + '</text>'); });
    (t.hold || []).forEach(function (sp) { var x1 = X(sp[0]), x2 = X(sp[1]); pr2.push('<rect x="' + x1.toFixed(1) + '" y="' + ptp + '" width="' + Math.max(0.5, x2 - x1).toFixed(1) + '" height="' + (H - ptp - pb) + '" fill="#b8923f" opacity="0.16"/>'); });
    (t.overlays || []).forEach(function (o) { var d = pathOf(o.data || []); if (d) pr2.push('<path d="' + d + '" fill="none" stroke="' + (o.color || "#888") + '" stroke-width="1.2"' + (o.dash ? ' stroke-dasharray="5 4"' : '') + ' opacity="0.95"/>'); });
    pr2.push('<path d="' + pathOf(P) + '" fill="none" stroke="#222" stroke-width="1.5"/>');
    var dlab = (t.dates && (t.buys || []).length + (t.sells || []).length <= 16);
    function dstr(i) { var ss = t.dates && t.dates[i] ? String(t.dates[i]) : ""; return ss.length >= 10 ? ss.slice(5) : ss; }
    (t.buys || []).forEach(function (i) { var x = X(i), y = Y(P[i]); pr2.push('<polygon points="' + x + ',' + (y - 10) + ' ' + (x - 6.5) + ',' + (y + 3) + ' ' + (x + 6.5) + ',' + (y + 3) + '" fill="#c0392b" stroke="#fff" stroke-width="0.8"/>'); if (dlab) pr2.push('<text x="' + x.toFixed(1) + '" y="' + (y + 16) + '" text-anchor="middle" font-size="9.5" fill="#c0392b">' + dstr(i) + '</text>'); });
    (t.sells || []).forEach(function (i) { var x = X(i), y = Y(P[i]); pr2.push('<polygon points="' + x + ',' + (y + 10) + ' ' + (x - 6.5) + ',' + (y - 3) + ' ' + (x + 6.5) + ',' + (y - 3) + '" fill="#147a43" stroke="#fff" stroke-width="0.8"/>'); if (dlab) pr2.push('<text x="' + x.toFixed(1) + '" y="' + (y - 12) + '" text-anchor="middle" font-size="9.5" fill="#147a43">' + dstr(i) + '</text>'); });
    pr2.push('<circle cx="' + X(n - 1) + '" cy="' + Y(P[n - 1]) + '" r="4" fill="#111"/>');
    if (t.date_start) pr2.push('<text x="' + pl + '" y="' + (H - 5) + '" font-size="11" fill="#8a8474">' + t.date_start + '</text>');
    if (t.date_end) pr2.push('<text x="' + (W - pr) + '" y="' + (H - 5) + '" text-anchor="end" font-size="11" fill="#8a8474">' + t.date_end + '</text>');
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto;display:block">' + pr2.join("") + '</svg>';
    var leg = [el("span", { html: '<b style="color:#c0392b">▲</b> 买入' }), el("span", { html: '<b style="color:#147a43">▼</b> 卖出' }), el("span", { html: '<i class="hsw"></i> 持仓期' })];
    (t.overlays || []).forEach(function (o) { leg.push(el("span", { html: '<i style="display:inline-block;width:15px;height:0;border-top:2px ' + (o.dash ? "dashed" : "solid") + ' ' + (o.color || "#888") + ';vertical-align:middle;margin-right:5px"></i>' + (o.label || "") })); });
    if (t.now_label) leg.push(el("span", { class: "tc-now", html: t.now_label }));
    var holder = el("div", null);
    try { var _pd = new DOMParser().parseFromString(svg, "image/svg+xml"); var _r = _pd.documentElement; if (_r && String(_r.nodeName).toLowerCase() === "svg") holder.appendChild(document.importNode(_r, true)); else holder.innerHTML = svg; } catch (e) { holder.innerHTML = svg; }
    return el("div", { class: "chart-card" }, [el("div", { class: "tc-legend" }, leg), holder]);
  }
  function researchSection(r) {
    var kids = (r.items || []).map(function (it) {
      var w = it.winner || {};
      var head = el("div", { class: "rs-head" }, [
        el("span", { class: "rs-name" }, it.name || it.symbol),
        el("span", { class: "rs-win" }, "冠军 " + (w.strategy || "") + (w.params ? (" · " + w.params) : "")),
        w.signal ? el("span", { class: "mlab " + (w.signal_tone || "neu") }, w.signal) : null
      ]);
      function fact(k, v) { return el("div", null, [el("span", { class: "rk" }, k), el("span", { class: "rv" }, v)]); }
      var facts = el("div", { class: "rs-facts" }, [
        fact("OOS 夏普", w.oos_sharpe != null ? String(w.oos_sharpe) : "—"),
        fact("OOS 收益", w.oos_return || "—"),
        fact("当前信号", w.signal || "—"),
        fact("离场线", w.exit || "—")
      ]);
      var tg = w.triggers || null;
      var trig = tg ? el("div", { class: "rs-trig" }, [
        el("span", { class: "rs-trig-h" }, "具体买卖点"),
        el("span", { html: "现在 · <b>" + (tg.action || w.signal || "—") + "</b>" }),
        tg.sell ? el("span", { html: "<b style='color:#147a43'>▼</b> 卖出/离场:<b>" + tg.sell + "</b>" }) : null,
        tg.buy ? el("span", { html: "<b style='color:#c0392b'>▲</b> 买入/回补:<b>" + tg.buy + "</b>" }) : null
      ]) : null;
      var sel = it.selection_text ? el("div", { class: "rs-sel", html: "<b>搜索过程(bandit):</b> " + it.selection_text }) : null;
      var lb = null;
      if (it.leaderboard && it.leaderboard.length) {
        var body = el("tbody", null, it.leaderboard.map(function (x) {
          return el("tr", { class: x.win ? "rmwin" : "" }, [
            el("td", null, String(x.rank)), el("td", null, x.strategy),
            el("td", { class: "rmp" }, x.params || ""),
            el("td", { style: "text-align:right" }, String(x.oos_sharpe)),
            el("td", { style: "text-align:right" }, x.oos_return || ""),
            el("td", null, x.signal || ""),
            el("td", { class: "rmbuy" }, x.buy || ""),
            el("td", { class: "rmsell" }, x.sell || "")
          ]);
        }));
        var thead = el("thead", null, el("tr", null, [el("th", null, "#"), el("th", null, "策略族"),
          el("th", null, "参数"), el("th", { style: "text-align:right" }, "OOS夏普"), el("th", { style: "text-align:right" }, "OOS收益"), el("th", null, "当前"), el("th", null, "买入触发"), el("th", null, "卖出触发")]));
        lb = el("div", null, [el("div", { class: "rs-sub" }, "① 策略选择对比 · 所有模拟策略排行(样本外 walk-forward · 含买卖触发价)"),
          el("div", { class: "tablewrap" }, el("table", { class: "rmtbl" }, [thead, body]))]);
      }
      var chart = null;
      if (it.trades) {
        var statsrow = (it.stats || []).length ? el("div", { class: "chart-stats" }, it.stats.map(function (sx) { return el("div", { class: "cs", html: sx.k + "<b>" + sx.v + "</b>" }); })) : null;
        var cap = el("div", { class: "rs-cap", html: "图说:实线=股价" + (it.trades.logy ? "(对数轴)" : "") + ";<b style='color:#c0392b'>▲买入</b> <b style='color:#147a43'>▼卖出</b> 金色阴影=持仓期。下方数字 <b>策略收益</b>=这套规则的总收益,<b>买入持有</b>=一直拿着不动的总收益(常更高);本策略赢在<b>回撤更小/夏普更高</b>,不是赢在绝对收益。" });
        chart = el("div", null, [el("div", { class: "rs-sub" }, "② 买卖点与持仓(冠军策略在价格上的进出)"), tradesChart(it.trades), statsrow, cap]);
      }
      return el("div", { class: "rs-item" }, [head, facts, trig, sel, lb, chart].filter(Boolean));
    });
    if (r.glossary && r.glossary.length) {
      kids.unshift(el("div", { class: "rs-gloss" }, [
        el("div", { class: "rs-sub" }, "📖 策略方法说明 · 本次测试涵盖的策略族"),
        el("ul", { style: "margin:6px 0 16px;padding-left:18px;font-size:12.5px;line-height:1.75;color:var(--ink-soft)" },
          r.glossary.map(function (g) {
            return el("li", null, [el("b", null, (g.name || g.family) + "："), (g.intro || "") + (g.edge ? " " + g.edge : "")]);
          }))
      ]));
    }
    if (r.note) kids.push(el("p", { class: "p-note", style: "margin-top:4px", html: r.note }));
    return el("div", null, kids);
  }
  function render(data, mount) {
    mount.innerHTML = "";
    // Optional data.symbol_order: stable-sort symbol-bearing sections to ONE canonical
    // company order so every section lines up. Rows whose symbol isn't listed (e.g. a
    // market-wide "宏观" alert) sort first, keeping their relative order. factor_rank is
    // intentionally left as a ranking and never reordered.
    if (data.symbol_order && data.symbol_order.length) {
      var _ord = data.symbol_order;
      var _key = function (r) { var i = _ord.indexOf(r && r.symbol); return i < 0 ? -1 : i; };
      var _stable = function (arr) {
        return arr.map(function (r, i) { return [r, i]; })
          .sort(function (a, b) { return (_key(a[0]) - _key(b[0])) || (a[1] - b[1]); })
          .map(function (x) { return x[0]; });
      };
      if (Array.isArray(data.levels)) data.levels = _stable(data.levels);
      if (Array.isArray(data.alerts)) data.alerts = _stable(data.alerts);
    }
    const frag = document.createDocumentFragment();
    frag.appendChild(masthead(data.meta));
    const body = el("div", { class: "body" });

    // verdict is unnumbered, sits at top
    // Objective stance score: regime (sector) is primary; macro only nudges. No regime score
    // -> envScore stays null and verdict falls back to keyword lean. Set verdict.score to override.
    var _rg = data.regime && typeof data.regime.score === "number" ? data.regime.score : null;
    var _mc = data.macro && typeof data.macro.risk_score === "number" ? data.macro.risk_score : null;
    var _envScore = (_rg !== null) ? (_mc !== null ? 0.75 * _rg + 0.25 * _mc : _rg) : null;
    if (data.verdict) body.appendChild(el("section", { class: "block", style: "border-top:none;padding-top:22px" }, verdict(data.verdict, _envScore)));
    // 综合结论紧跟综合立场,作为顶部"结论"汇总(不编号);其余编号区块顺延
    if (data.conclusion) body.appendChild(block(null, data.conclusion_title || "综合结论", null, conclusionBlock(data.conclusion)));

    let no = 0;
    function add(title, hnote, content) { if (content) { no++; body.appendChild(block(no, title, hnote, content)); } }

    if (data.alerts) add("🔴 今日重点关注", data.alerts.length + " 项", alerts(data.alerts));

    // environment composite
    if (data.regime || data.macro || data.calendar) {
      const panels = [];
      if (data.regime) panels.push(regimePanel(data.regime));
      if (data.macro) panels.push(macroPanel(data.macro));
      const grid = el("div", { class: panels.length === 1 ? "env-grid one" : "env-grid" }, panels);
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
    if (data.sentiment) add(data.sentiment.title || "🗞 三层时效情绪", data.sentiment.composite != null ? "复合 " + scoreStr(data.sentiment.composite) : null, sentiment(data.sentiment));
    if (data.portfolio_health) add(data.portfolio_health.title || "🧩 组合体检", null, portfolioHealth(data.portfolio_health));
    if (data.holdings) add("你的持仓", null, proseBlock(data.holdings));
    if (data.methods) add(data.methods.title || "信号多法对照", null, methodsTable(data.methods));
    if (data.research) add(data.research.title || "自动研究详情", null, researchSection(data.research));

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


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN / ±Inf) with None. Python's json.dumps
    emits bare NaN/Infinity by default — invalid JSON that the BROWSER's JSON.parse THROWS
    on, blanking the entire report. Any metric that came back NaN must serialize as null.
    Also normalizes numpy float subclasses to plain float."""
    import math
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def render(report: dict) -> str:
    """结构化 report dict -> 一份完整、自包含、带样式的 HTML 字符串。"""
    # 完整转义：< > & 防 </script> 变体与 HTML 注入；U+2028/2029 是 JS 字符串里
    # 非法的行分隔符，会让内嵌脚本抛 SyntaxError。NaN/Inf 先转 null,否则浏览器 JSON.parse 抛错。
    data = (_json.dumps(_json_safe(report), ensure_ascii=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
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
