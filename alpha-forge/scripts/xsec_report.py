"""把横截面记分卡渲染成 markdown / 简易 HTML 段(自包含,不改 html_report.py)。"""
from __future__ import annotations
import pandas as pd

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
