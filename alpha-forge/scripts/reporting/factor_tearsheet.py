"""Single-file HTML factor tearsheet — the "is this factor real?" one-pager.

Why a dedicated tearsheet instead of the full html_report pipeline: factor research
iterates fast (dozens of candidate factors per session) and needs a lightweight,
dependency-free artifact you can open anywhere. So this module renders a
self-contained HTML string — inline CSS, hand-rolled SVG polylines, zero external
JS/fonts — independent of scripts.reporting.html_report.

Methodology (mirrors the xsec evaluation口径 so numbers agree across the skill):
  * forward returns: close.shift(-h)/close - 1 — the factor at t predicts returns
    STARTING at t, so everything is causal by construction;
  * daily cross-sectional quantiles via rank(method="first") — deterministic ties;
  * RankIC = daily Spearman(factor_t, fwd_h_t), averaged per horizon;
  * long-short spread = top-quantile mean fwd − bottom-quantile mean fwd;
  * turnover = month-over-month churn of the top-quantile membership set.

Costs: the tearsheet is deliberately GROSS of costs (pure information-content view);
the footer says so explicitly, because a monthly-churn=80% factor with a pretty
gross curve can be net-negative. Use the backtest engine for net numbers.

`tearsheet_data()` returns the plain numbers, `factor_tearsheet()` wraps them in
HTML — split so tests (and agents) can assert on data without parsing markup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

_PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
            "#9d755d", "#b279a2", "#eeca3b", "#bab0ac", "#ff9da6"]


# ---------------------------------------------------------------- statistics

def _spearman_row(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xr = pd.Series(x[m]).rank().values
    yr = pd.Series(y[m]).rank().values
    xd, yd = xr - xr.mean(), yr - yr.mean()
    den = np.sqrt((xd ** 2).sum() * (yd ** 2).sum())
    return float((xd * yd).sum() / den) if den > 0 else np.nan


def tearsheet_data(factor_panel: pd.DataFrame, close_panel: pd.DataFrame, *,
                   quantiles: int = 5, horizons: tuple = (1, 5, 21, 63)) -> dict:
    """Compute every number the tearsheet shows. Returns
    {quantile_cum, ls_cum, quantile_stats, ic_table, ic_series, turnover_monthly, meta}.

    Cumulative curves use the SHORTEST horizon (ideally 1) so daily compounding is
    exact; longer horizons appear in the RankIC table (overlapping windows — fine for
    IC, wrong for compounding, hence the split).
    """
    cols = factor_panel.columns.intersection(close_panel.columns)
    idx = factor_panel.index.intersection(close_panel.index)
    if len(cols) < 3 or len(idx) < max(horizons) + 5:
        raise ValueError(f"tearsheet needs >=3 symbols and > {max(horizons)+5} bars "
                         f"(got {len(cols)} x {len(idx)})")
    f = factor_panel.loc[idx, cols].sort_index()
    c = close_panel.loc[idx, cols].sort_index()
    fwd = {h: c.shift(-h) / c - 1.0 for h in horizons}

    # daily cross-sectional quantile assignment (0 = worst factor, q-1 = best)
    ranks = f.rank(axis=1, method="first")
    nval = f.notna().sum(axis=1).replace(0, np.nan)
    qf = np.floor((ranks - 1).div(nval, axis=0) * quantiles).clip(0, quantiles - 1)

    h0 = int(min(horizons))
    f0 = fwd[h0]
    q_daily = pd.DataFrame({int(k): f0.where(qf == k).mean(axis=1)
                            for k in range(quantiles)})
    # last h0 rows have no realized forward return — drop, don't pad with zeros
    q_daily = q_daily.iloc[:-h0] if h0 > 0 else q_daily
    quantile_cum = (1.0 + q_daily.fillna(0.0)).cumprod()
    ls_daily = q_daily[quantiles - 1] - q_daily[0]
    ls_cum = (1.0 + ls_daily.fillna(0.0)).cumprod()

    # RankIC per horizon
    ic_series = pd.DataFrame({
        h: [_spearman_row(f.iloc[i].values, fwd[h].iloc[i].values) for i in range(len(f))]
        for h in horizons
    })
    ic_series.index = f.index
    rows = []
    for h in horizons:
        s = ic_series[h].dropna()
        std = float(s.std(ddof=0))
        rows.append({"horizon": h, "RankIC": float(s.mean()) if len(s) else np.nan,
                     "ICIR": float(s.mean() / std) if std > 0 else np.nan,
                     "IC_hit": float((s > 0).mean()) if len(s) else np.nan,
                     "n_days": int(len(s))})
    ic_table = pd.DataFrame(rows).set_index("horizon")

    # monthly churn of the top-quantile (long bucket) membership set
    months = f.index.to_period("M")
    is_month_end = pd.Series(months).ne(pd.Series(months).shift(-1)).values
    mends = f.index[is_month_end]
    tops = []
    for d in mends:
        row = qf.loc[d]
        tops.append(frozenset(row.index[row == quantiles - 1]))
    churn = [1.0 - len(a & b) / max(len(b), 1) for a, b in zip(tops[:-1], tops[1:])]
    turnover_monthly = float(np.mean(churn)) if churn else np.nan

    # per-quantile stats on the h0 forward return
    ppy = TRADING_DAYS / h0
    qs = []
    for k in range(quantiles):
        s = q_daily[k].dropna()
        vol = float(s.std(ddof=0))
        qs.append({"quantile": f"Q{k + 1}",
                   "mean_fwd": float(s.mean()) if len(s) else np.nan,
                   "ann_ret": float(s.mean() * ppy) if len(s) else np.nan,
                   "ann_vol": float(vol * np.sqrt(ppy)),
                   "sharpe": float(s.mean() / vol * np.sqrt(ppy)) if vol > 0 else np.nan,
                   "avg_names": float((qf == k).sum(axis=1).mean())})
    quantile_stats = pd.DataFrame(qs).set_index("quantile")

    return {"quantile_cum": quantile_cum, "ls_cum": ls_cum, "ls_daily": ls_daily,
            "quantile_stats": quantile_stats, "ic_table": ic_table,
            "ic_series": ic_series, "turnover_monthly": turnover_monthly,
            "meta": {"n_symbols": int(len(cols)), "n_days": int(len(idx)),
                     "quantiles": quantiles, "horizons": tuple(horizons),
                     "curve_horizon": h0,
                     "start": str(idx.min())[:10], "end": str(idx.max())[:10]}}


# ---------------------------------------------------------------- SVG / HTML

def _svg_lines(df: pd.DataFrame, width: int = 680, height: int = 220,
               hline: float | None = None) -> str:
    """Multi-line SVG chart: one polyline per column, min/max/first/last labels.
    Hand-rolled so the HTML stays dependency-free."""
    d = df.dropna(how="all")
    if d.empty:
        return "<p>(no data)</p>"
    ml, mr, mt, mb = 46, 8, 8, 18
    iw, ih = width - ml - mr, height - mt - mb
    ymin = float(np.nanmin(d.values))
    ymax = float(np.nanmax(d.values))
    if hline is not None:
        ymin, ymax = min(ymin, hline), max(ymax, hline)
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0
    n = len(d)

    def xy(i, v):
        x = ml + iw * (i / max(n - 1, 1))
        y = mt + ih * (1.0 - (v - ymin) / (ymax - ymin))
        return f"{x:.1f},{y:.1f}"

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'xmlns="http://www.w3.org/2000/svg" style="background:#fff">']
    if hline is not None:
        yh = mt + ih * (1.0 - (hline - ymin) / (ymax - ymin))
        parts.append(f'<line x1="{ml}" y1="{yh:.1f}" x2="{width - mr}" y2="{yh:.1f}" '
                     f'stroke="#999" stroke-dasharray="4,3" stroke-width="1"/>')
    for j, col in enumerate(d.columns):
        vals = d[col].values.astype(float)
        pts = " ".join(xy(i, v) for i, v in enumerate(vals) if np.isfinite(v))
        color = _PALETTE[j % len(_PALETTE)]
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    parts.append(f'<text x="4" y="{mt + 10}" font-size="11" fill="#666">{ymax:.3g}</text>')
    parts.append(f'<text x="4" y="{mt + ih}" font-size="11" fill="#666">{ymin:.3g}</text>')
    parts.append(f'<text x="{ml}" y="{height - 4}" font-size="11" fill="#666">{str(d.index[0])[:10]}</text>')
    parts.append(f'<text x="{width - 90}" y="{height - 4}" font-size="11" fill="#666">{str(d.index[-1])[:10]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _legend(labels) -> str:
    spans = []
    for j, lb in enumerate(labels):
        color = _PALETTE[j % len(_PALETTE)]
        spans.append(f'<span style="margin-right:12px"><span style="display:inline-block;'
                     f'width:10px;height:10px;background:{color};border-radius:2px;'
                     f'margin-right:4px"></span>{lb}</span>')
    return f'<div style="font-size:12px;color:#444;margin:4px 0 10px">{"".join(spans)}</div>'


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{h}</th>" for h in [df.index.name or ""] + list(df.columns))
    body = []
    for ix, row in df.iterrows():
        tds = [f"<td><b>{ix}</b></td>"]
        for col in df.columns:
            v = row[col]
            if col in fmt and pd.notna(v):
                tds.append(f"<td>{fmt[col].format(v)}</td>")
            elif isinstance(v, float):
                tds.append("<td>-</td>" if pd.isna(v) else f"<td>{v:.4f}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>')


_CSS = """
body{font-family:-apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
     max-width:860px;margin:24px auto;padding:0 16px;color:#222;background:#fafafa}
h1{font-size:22px;border-bottom:2px solid #4c78a8;padding-bottom:8px}
h2{font-size:16px;margin:26px 0 8px;color:#333}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:right}
th{background:#f0f3f7}td:first-child,th:first-child{text-align:left}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:12px;margin:10px 0}
.meta{font-size:12.5px;color:#555}
.foot{font-size:12px;color:#777;border-top:1px solid #ddd;margin-top:28px;padding-top:10px}
.kpi{display:inline-block;background:#fff;border:1px solid #e3e3e3;border-radius:6px;
     padding:8px 14px;margin:4px 8px 4px 0;font-size:13px}
.kpi b{font-size:17px;display:block}
"""


def factor_tearsheet(factor_panel: pd.DataFrame, close_panel: pd.DataFrame, *,
                     quantiles: int = 5, horizons: tuple = (1, 5, 21, 63),
                     name: str = "factor", out_path: str | None = None) -> str:
    """Render the self-contained HTML tearsheet; optionally write it to out_path.

    Returns the HTML string either way, so callers can embed a section elsewhere.
    """
    data = tearsheet_data(factor_panel, close_panel, quantiles=quantiles, horizons=horizons)
    meta = data["meta"]
    ic = data["ic_table"]
    h_main = meta["curve_horizon"]
    main_ric = float(ic.loc[h_main, "RankIC"]) if h_main in ic.index else float("nan")
    main_icir = float(ic.loc[h_main, "ICIR"]) if h_main in ic.index else float("nan")
    ls_total = float(data["ls_cum"].iloc[-1] - 1.0) if len(data["ls_cum"]) else float("nan")
    to = data["turnover_monthly"]

    q_labels = [f"Q{k + 1}" for k in range(meta["quantiles"])]
    qc = data["quantile_cum"].copy()
    qc.columns = q_labels

    kpis = (f'<span class="kpi">RankIC (h={h_main})<b>{main_ric:.4f}</b></span>'
            f'<span class="kpi">ICIR<b>{main_icir:.3f}</b></span>'
            f'<span class="kpi">多空累计 (毛)<b>{ls_total:+.2%}</b></span>'
            f'<span class="kpi">月均换手 (多头桶)<b>{to:.1%}</b></span>')

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>因子 Tearsheet — {name}</title><style>{_CSS}</style></head><body>
<h1>因子 Tearsheet — {name}</h1>
<p class="meta">样本:{meta['start']} ~ {meta['end']} · {meta['n_symbols']} 只标的 ·
{meta['n_days']} 个交易日 · {meta['quantiles']} 分位 · horizons={list(meta['horizons'])}</p>
<div>{kpis}</div>

<h2>分位累计收益(h={h_main},等权,不扣成本)</h2>
<div class="card">{_legend(q_labels)}{_svg_lines(qc, hline=1.0)}</div>

<h2>逐 horizon 平均 RankIC</h2>
<div class="card">{_table(ic, fmt={'RankIC': '{:.4f}', 'ICIR': '{:.3f}', 'IC_hit': '{:.1%}', 'n_days': '{:.0f}'})}</div>

<h2>RankIC 时序(h={h_main})</h2>
<div class="card">{_svg_lines(data['ic_series'][[h_main]].rename(columns={h_main: f'RankIC h={h_main}'}), hline=0.0)}</div>

<h2>多空价差累计(Q{meta['quantiles']} − Q1,不扣成本)</h2>
<div class="card">{_svg_lines(data['ls_cum'].rename('LS').to_frame(), hline=1.0)}</div>

<h2>分位统计(h={h_main} 前向收益)</h2>
<div class="card">{_table(data['quantile_stats'], fmt={'mean_fwd': '{:.5f}', 'ann_ret': '{:+.2%}', 'ann_vol': '{:.2%}', 'sharpe': '{:.3f}', 'avg_names': '{:.1f}'})}</div>

<div class="foot">口径说明:前向收益 = close.shift(-h)/close − 1(因子在 t 日预测 t 日起的未来收益,
全因果,无前视);分位为逐日截面 rank 分桶;<b>本页所有收益均不扣交易成本与冲击</b>(纯因子信息含量视角)
——月均换手 {to:.0%} 的因子,净收益请以 backtest 扣费口径为准。生成于 alpha-forge factor_tearsheet。</div>
</body></html>"""

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return html
