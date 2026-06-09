# 量化分析报告模板 — 数据契约 (SCHEMA)

一个机构研报风、可打印、JSON 驱动的报告渲染器，给 **alpha-forge** 技能做最终数据展示，替代直接输出 `.md`。

## 就是 `scripts/html_report.py`

报告渲染能力已**内置**在 `scripts/html_report.py`：CSS 与 JS 作为常量内联，无外部依赖。
构造一个 **report dict**（字段见下）交给它即可，产出**单文件、自包含、可打印为 PDF** 的 HTML。

```python
from scripts import html_report as H
H.save_html(report, "trading/reports/美股复盘_2026-06-09.html")   # -> 写单文件
html = H.render(report)                                          # -> 拿字符串
```

> 模板源文件（`assets/report.css`、`assets/render.js`、外壳 `量化分析报告.html`）随包附带，
> 仅作开发/预览用；运行期只依赖 `html_report.py` 一个文件。
> **红涨绿跌**（A股惯例）；如需西式绿涨红跌，把 `html_report.py` 里 `_CSS` 的 `--pos`/`--neg` 两值对调。

---

## 顶层字段（全部可选——给了就渲染，不给就跳过）

章节按下列顺序出现，主章节自动编号 ①②③…

| key | 类型 | 渲染为 |
|---|---|---|
| `meta` | object | 报头：标题/副标题/类型徽章/日期/市场/数据源/免责标签 |
| `verdict` | object | 「结论先行」立场框（▲偏多 / ▼偏空 / ◆中性 自动判别） |
| `alerts` | array | 🔴 今日重点关注（标红/标黄圆点 + 标的 + 动作） |
| `regime` | object | 大盘环境评分计（−1…+1 计量条 + 明细行） |
| `macro` | object | 宏观风险评分计（含 VIX / 分项） |
| `calendar` | array | 事件前瞻（影响等级圆点 + 倒计时） |
| `factor_rank` | object | 多因子排序表（动量/波动/信号/综合分） |
| `groups` | array | 分组解读（强/中/弱 标签色） |
| `technical` | object | 单标的：**价格阶梯**（止损→买区→现价→目标）+ 指标格 |
| `levels` | array | **建议买卖点大表**（核心；含内嵌 R/R 条） |
| `backtest` | object | 策略净值 vs 基准 折线图 + 统计 |
| `sentiment` | object | 三层时效情绪（−1…+1 双向条） |
| `portfolio_health` | object | 组合体检表（ENB/β/VaR…） |
| `holdings` | string/array | 持仓说明 |
| `conclusion` | string/array | 综合结论 |
| `disclaimer` | string | 免责声明 |
| `sources` | array | 来源链接 |

字符串字段支持内联 HTML（`<b>` 加粗、`<span class="pos/neg">` 上色）。

---

## 各字段形状

```jsonc
"meta": {
  "title": "美股盘后复盘", "subtitle": "MU · 美光科技",
  "report_type": "single",        // single | portfolio | market | backtest | attribution | macro
  "date": "2026-06-09", "weekday": "周二",
  "market": "美股（纳指）", "data_source": "IBKR 5年日线（延迟900s）",
  "universe": "watchlist_us 26 只", "tag": "机械量化参考 · 非投资建议",
  "generated_by": "alpha-forge"
},

"verdict": { "stance": "偏多 · 不追高", "action": "<b>等回调 860–915</b>…", "summary": "…" },
// stance 含 多/涨/强→▲红；含 空/跌/弱→▼绿；其余→◆琥珀（红涨绿跌）

"alerts": [{
  "level": "high",               // high=红点 | mid=黄点
  "symbol": "MU", "name": "美光",
  "signal": "long",              // long 做多 | watch 观望 | short 做空（渲染为徽章）
  "hold": "持仓",                 // 可选，★持仓标
  "headline": "…", "detail": "…", "action": "…"
}],

"regime": {                       // macro 同形，键名换成 risk_score
  "title": "📊 美股大盘环境", "score": 0.42, "label": "大盘多头压过宏观谨慎",
  "rows": [{ "item": "大盘趋势 regime", "value": "+0.63", "read": "多头/risk-on…" }],
  "note": "…"
},
"macro": { "title": "🌐 全球宏观", "risk_score": -0.10, "label": "中性偏谨慎",
           "vix": 18.9, "vix_note": "…", "rows": [...], "note": "…" },

"calendar": [{ "event": "CPI", "date": "6/10", "in_days": 1,
               "impact": "high",     // high | med | low（圆点色）
               "flagged": true }],   // 标红强调

"factor_rank": {
  "title": "① 多因子排序", "note_head": "右上小字", "hint": "表下注释",
  "rows": [{ "rank":1, "symbol":"SOXL", "leveraged":true, "price":211.4,
             "m6":355, "m12":1002,      // 动量：百分数（355 → +355%）
             "vol":178, "signal":"long", "regime":"×0.40", "score":1.72 }]
},

"groups": [{ "title":"AI/半导体高动量组", "tag":"强多头·高波动",
             "tone":"strong",          // strong 绿 | neutral 琥珀 | weak 红
             "body":"<b>…</b>" }],

"technical": {
  "title":"📈 MU 技术与买卖点", "signal_note":"ma_cross +1 / ts_mom +1 · ×0.40",
  "level": { "price":949.3, "buy_low":860, "buy_high":915,
             "stop":751, "target":1089, "target2":1160, "rr":1.48,
             "change_pct": 2.3 },      // 可选今日涨跌（带色）
  "stats": [{ "k":"年化波动", "v":"111%", "x":"极高",      // x=小字注脚
              "vhtml":"+774%", "color":"var(--pos)" }],   // vhtml/color 可选上色
  "note": "…"
},

"levels": [{                      // 建议买卖点大表（核心）
  "symbol":"GOOG", "name":"中际旭创", "sector":"光模块",   // name/sector 全空则隐列
  "price":361.2, "change_pct":2.22,                      // change_pct 全空则隐列
  "signal":"long", "regime":"×0.70",
  "buy_low":355, "buy_high":356, "stop":336.7, "target":404.4,
  "rr":2.52,                       // 渲染为彩色 R/R 条：≥1.8 绿 / ≥1.2 蓝 / ≥0.9 琥珀 / <0.9 红
  "rsi":44, "pctb":0.12,
  "flag":true, "note":"🔴 回调到带下沿…"   // flag=true 在备注前加 🔴
}],
"levels_title": "③ 建议买卖点", "levels_hint": "表下说明",

"backtest": {
  "title":"MU · 5年回测", "head_note":"夏普 1.27",
  "strategy_label":"时序动量", "benchmark_label":"买入持有",
  "stats":[{ "k":"夏普", "v":"1.27" }, { "k":"最大回撤", "v":"−50%" }],
  "equity": { "strategy":[1, 1.02, …], "benchmark":[1, 1.01, …] }  // 等长净值数组
},

"sentiment": {
  "title":"🗞 三层时效情绪", "composite":0.39, "note":"…",
  "layers": [{ "layer":"个股", "score":0.61, "key":"HBM 卖光至 2026…" }]
},

"portfolio_health": {
  "title":"🧩 组合体检", "conclusion":"<b>结论…</b>",
  "rows": [{ "metric":"组合 β", "value":"<span class='num'>1.97</span>",
             "flag":true, "read":"偏高，放大涨跌" }]
}
```

---

## 颜色约定

- **涨跌色**：`+x%` 红、`−x%` 绿（**红涨绿跌**，A股惯例）。
  若需西式绿涨红跌，把 `html_report.py` 的 `_CSS` 里 `--pos` / `--neg` 两个变量对调即可。
- **信号徽章**：long 红、watch 琥珀、short 绿（与「做多=红」一致）。
- **R/R 条**：≥1.8 绿、≥1.2 蓝、≥0.9 琥珀、<0.9 红——一眼看出盈亏比好坏。
- 红点 🔴 / 黄点用于「重点关注」与表内 `flag`，与原报告的 🔴 标红习惯一致。

## 打印 / PDF

`⌘/Ctrl + P` 即可。已内置打印样式（去除切换条、避免章节断页、表格紧排）。
> 列数多的买卖点大表在 **横向 (Landscape)** 打印最完整。

## 与 markdown 输出的关系

技能仍可照常先 author markdown 作为草稿/留痕；最终展示改走本模板的结构化 JSON，
即可得到稳定的版式、买卖点为核心的信息层级、以及图表（评分计 / 价格阶梯 / 净值曲线），
不再受 markdown 表格错位、内联 span 渲染差的困扰。
