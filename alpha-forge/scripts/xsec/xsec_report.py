"""把横截面记分卡渲染成 markdown / 自包含 HTML(选股报告模板,内联 SVG,无外部依赖)。"""
from __future__ import annotations
import html as _html
import numpy as np, pandas as pd

# ---------- markdown ----------
def scorecard_markdown(result: dict, title="AI 选股 · 横截面预测力") -> str:
    sc = result["scorecard"]
    L = [f"## {title}", "",
         f"**判定: {sc['verdict']}**  (调仓日 {sc['n_dates']}, 平均 {sc['n_names_avg']} 只/日)", "",
         "| 指标 | 值 | 含义 |", "|---|---|---|",
         f"| RankIC | {sc['RankIC']} | 截面排序相关(抗极值,主指标) |",
         f"| RankICIR | {sc['RankICIR']} | 排序信号稳定性(≳0.3 可用) |",
         f"| meanIC | {sc['meanIC']} | 截面 Pearson IC |",
         f"| ICIR | {sc['ICIR']} | IC 稳定性 |",
         f"| IC>0 占比 | {sc['IC_hit']} | 多少调仓日方向对 |",
         f"| 多空年化 | {sc['LS_ann']} | Top-K 多空(扣成本) |",
         f"| 多空 Sharpe | {sc['LS_sharpe']} | 多空夏普 |",
         f"| 分位单调性 | {sc['quantile_monotonicity']} | 高分位是否真更强(→1 理想) |",
         "", "> 非投资建议。RankIC/ICIR 是横截面选股的核心口径;小样本/同业池请谨慎解读。"]
    q = result.get("quantile_fwd")
    if q is not None and len(q):
        L += ["", "分位桶平均前向收益(低→高分位): " + ", ".join(f"{v*100:+.2f}%" for v in q.values)]
    return "\n".join(L)

def scorecard_html(result: dict, title="AI 选股 · 横截面预测力") -> str:
    sc = result["scorecard"]
    rows = "".join(f"<tr><td>{k}</td><td>{sc.get(k)}</td></tr>" for k in
                   ("RankIC","RankICIR","meanIC","ICIR","IC_hit","LS_ann","LS_sharpe","quantile_monotonicity"))
    return (f'<section class="xsec"><h2>{title}</h2>'
            f'<p><b>判定:{sc["verdict"]}</b>(调仓日 {sc["n_dates"]},平均 {sc["n_names_avg"]} 只/日)</p>'
            f'<table border="1" cellspacing="0" cellpadding="4">{rows}</table>'
            f'<p style="color:#888">非投资建议</p></section>')

# ---------- 内联 SVG 图元(自包含,浅色主题) ----------
def _svg_bars(vals, labels, unit="%", w=520, h=210, pad=34):
    if not len(vals): return ""
    lo = min(0, min(vals)); hi = max(vals); rng = (hi - lo) or 1
    gap = (w - 2*pad)/len(vals); bw = gap*0.62
    y0 = h - pad - (0 - lo)/rng*(h - 2*pad)
    cols = ["#B5D4F4","#85B7EB","#378ADD","#185FA5","#0C447C"]
    s = [f'<line x1="{pad}" y1="{y0:.1f}" x2="{w-pad}" y2="{y0:.1f}" stroke="#c3c2b7" stroke-width="1"/>']
    for i,(v,lb) in enumerate(zip(vals,labels)):
        x = pad + i*gap + (gap-bw)/2; yv = h - pad - (v-lo)/rng*(h-2*pad)
        yt = min(yv,y0); hh = abs(yv-y0)
        s.append(f'<rect x="{x:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{hh:.1f}" rx="3" fill="{cols[i%5]}"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{yt-5:.1f}" font-size="12" text-anchor="middle" fill="#0b0b0b">{v:g}{unit}</text>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{h-pad+15:.1f}" font-size="11" text-anchor="middle" fill="#52514e">{_html.escape(str(lb))}</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="分位平均前向收益">{"".join(s)}</svg>'

def _svg_line(vals, dates=None, w=520, h=200, pad=30, base=None):
    if len(vals) < 2: return ""
    lo = min(vals); hi = max(vals); rng = (hi-lo) or 1
    pts = [f"{pad + i/(len(vals)-1)*(w-2*pad):.1f},{h-pad-(v-lo)/rng*(h-2*pad):.1f}" for i,v in enumerate(vals)]
    g = []
    if base is not None and lo <= base <= hi:
        yb = h-pad-(base-lo)/rng*(h-2*pad)
        g.append(f'<line x1="{pad}" y1="{yb:.1f}" x2="{w-pad}" y2="{yb:.1f}" stroke="#c3c2b7" stroke-dasharray="3 3" stroke-width="1"/>')
    g.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#185FA5" stroke-width="2"/>')
    g.append(f'<text x="{w-pad}" y="16" font-size="12" text-anchor="end" fill="#0b0b0b">末值 {vals[-1]:.2f}</text>')
    if dates: g.append(f'<text x="{pad}" y="{h-8}" font-size="11" fill="#898781">{dates[0]}</text><text x="{w-pad}" y="{h-8}" font-size="11" text-anchor="end" fill="#898781">{dates[-1]}</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="组合净值">{"".join(g)}</svg>'

def _svg_spark(vals, w=520, h=90, pad=18):
    if not len(vals): return ""
    n = len(vals); gap = (w-2*pad)/n; bw = gap*0.8
    mx = max(abs(min(vals)), abs(max(vals))) or 1; y0 = h/2
    s = [f'<line x1="{pad}" y1="{y0}" x2="{w-pad}" y2="{y0}" stroke="#c3c2b7" stroke-width="1"/>']
    for i,v in enumerate(vals):
        x = pad + i*gap + (gap-bw)/2; hh = abs(v)/mx*(h/2-6); y = y0-hh if v>=0 else y0
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="{"#1baf7a" if v>=0 else "#e34948"}"/>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="逐期 RankIC">{"".join(s)}</svg>'

_CSS = """*{box-sizing:border-box}body{margin:0;background:#f3f2ec;color:#0b0b0b;font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 60px}h1{font-size:23px;font-weight:600;margin:0 0 4px}h2{font-size:17px;font-weight:600;margin:30px 0 12px}
.sub{color:#52514e;font-size:14px;margin:0 0 14px}.badge{display:inline-block;background:#FAEEDA;color:#854F0B;font-size:13px;padding:4px 12px;border-radius:20px;font-weight:500}
.card{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:12px;padding:16px 18px;margin:10px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}.kpi{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:10px;padding:12px 14px}
.kl{font-size:12px;color:#52514e}.kv{font-size:22px;font-weight:600;margin-top:2px}.ks{font-size:11px;color:#898781}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #e1e0d9}th{color:#52514e;font-weight:500;font-size:12px}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:6px}ul{margin:6px 0;padding-left:20px}li{margin:4px 0;font-size:14px}
.foot{color:#898781;font-size:12px;margin-top:24px;border-top:1px solid #e1e0d9;padding-top:12px}"""

def render_html_report(results: dict, current_rank, sectors=None, title="AI 选股 · 横截面评测",
                       subtitle="", topk=5, out_path=None, caveats=None):
    """选股报告模板(结论排名先行 + 评测依据 + 诚实红线)。
    results: {label: eval_result}(eval_result=xsec_eval.evaluate_cross_section 输出);第一个为主口径,用于图表/KPI。
    current_rank: 已按分降序的 [{'symbol','sector','score'}, ...](最新一期打分)。
    sectors: {symbol: sector}(current_rank 未带 sector 时用)。返回 HTML 字符串;给 out_path 则写文件。
    """
    labels = list(results); primary = results[labels[0]]; sc = primary["scorecard"]
    sectors = sectors or {}
    cr = [dict(symbol=r["symbol"], sector=r.get("sector") or sectors.get(r["symbol"], ""),
               score=float(r["score"])) for r in current_rank]
    n = len(cr); longs = [r["symbol"] for r in cr[:topk]]
    def rowcls(i):
        if i < topk: return ('<span class="tag" style="background:#E1F5EE;color:#0F6E56">做多候选</span>', ' style="background:#F1F7EC"')
        if i >= n-topk: return ('<span class="tag" style="background:#FCEBEB;color:#A32D2D">空头端</span>', '')
        return ('', '')
    rank_rows = ""
    for i,r in enumerate(cr):
        tag,bg = rowcls(i)
        rank_rows += f'<tr{bg}><td>{i+1}</td><td><b>{_html.escape(r["symbol"])}</b></td><td>{_html.escape(r["sector"])}</td><td style="text-align:right">{r["score"]:+.3f}</td><td>{tag}</td></tr>'

    def kpi(l,v,s=""): return f'<div class="kpi"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div></div>'
    kpis = "".join([kpi("RankIC(主)",f'{sc["RankIC"]:+.3f}',"截面排序相关"),
                    kpi("RankICIR",f'{sc["RankICIR"]:.2f}',"稳定性·门槛0.3"),
                    kpi("多空 Sharpe",f'{sc["LS_sharpe"]:.2f}',"Top-K 多空"),
                    kpi("多空年化",f'{sc["LS_ann"]*100:.0f}%',"扣成本"),
                    kpi("IC&gt;0 占比",f'{sc["IC_hit"]*100:.0f}%',"逐月方向"),
                    kpi("分位单调性",f'{sc["quantile_monotonicity"]:.2f}',"1=完美单调")])

    q = primary.get("quantile_fwd"); qsvg = ""
    if q is not None and len(q):
        vals=[round(v*100,2) for v in q.values]; qsvg=_svg_bars(vals,[f"Q{i+1}" for i in range(len(vals))])
    daily = primary.get("daily"); navsvg=sparksvg=""
    if daily is not None and len(daily):
        ls=list(daily["LS"]); nav=[]; v=1.0
        for x in ls: v*=(1+float(x)); nav.append(round(v,4))
        ds=[str(d)[:10] for d in daily["date"]]
        navsvg=_svg_line(nav, ds, base=1.0); sparksvg=_svg_spark([float(x) for x in daily["RankIC"]])

    comp = "".join(f'<tr><td>{_html.escape(l)}</td><td>{results[l]["scorecard"]["RankIC"]:+.3f}</td>'
                   f'<td>{results[l]["scorecard"]["RankICIR"]:.2f}</td>'
                   f'<td>{results[l]["scorecard"]["LS_ann"]*100:.0f}%</td>'
                   f'<td>{results[l]["scorecard"]["LS_sharpe"]:.2f}</td></tr>' for l in labels)

    cav = caveats or [
        "RankICIR &lt; 0.3 属弱信号:IC 平均或为正但逐月不稳,视为组合倾斜而非重仓依据。",
        "线性 ridge 诚实基线;深度模型可经 load_external_scores 接入同口径再比。",
        "样本 regime 可能集中;换市场状态未必延续。",
        "同主题标的彼此相关、离散度有限;幸存者偏差使长回测偏乐观。"]
    cav_html = "".join(f"<li>{c}</li>" for c in cav)

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{_html.escape(title)}</h1>
<p class="sub">{_html.escape(subtitle)}</p>
<p><span class="badge">判定:{sc['verdict']}</span></p>
<p class="sub" style="margin:14px 0 0">结论先行:下方为最终选股排名;评测指标、图表与局限见其后。</p>

<h2>① 本次选股 · 最终排名({sc['n_names_avg']:.0f} 只)</h2>
<div class="card">
<p class="sub" style="margin:0 0 10px">ridge 排序器全历史训练,对最新一期打分排序。<b>做多前 {topk}:{_html.escape(" · ".join(longs))}</b>。这是<b>相对强弱</b>排序,非绝对涨跌预测。</p>
<table><tr><th>#</th><th>标的</th><th>子板块</th><th style="text-align:right">预测分</th><th></th></tr>{rank_rows}</table></div>

<h2>② 评测依据 · 核心指标({labels[0]})</h2>
<div class="kpis">{kpis}</div>

{('<h2>分位桶平均前向收益(低→高档)</h2><div class="card">'+qsvg+'<p class="sub" style="margin:8px 0 0">按预测分档,真实前向收益应逐档递增=排序方向正确。</p></div>') if qsvg else ''}
{('<h2>Top-K 多空组合净值(扣成本)</h2><div class="card">'+navsvg+'</div>') if navsvg else ''}
{('<h2>逐期 RankIC(绿=排序对/红=反)</h2><div class="card">'+sparksvg+'</div>') if sparksvg else ''}

<h2>各周期口径对照</h2>
<div class="card"><table><tr><th>口径</th><th>RankIC</th><th>RankICIR</th><th>多空年化</th><th>多空Sharpe</th></tr>{comp}</table></div>

<h2>诚实红线</h2><ul>{cav_html}</ul>
<div class="foot">由 alpha-forge「AI 选股 / 横截面排序」能力产出:universe → panel → xsec_eval(ridge,purged walk-forward,IC/RankIC/ICIR+分位+Top-K多空)。非投资建议。</div>
</div></body></html>"""
    if out_path:
        open(out_path,"w",encoding="utf-8").write(html)
    return html
