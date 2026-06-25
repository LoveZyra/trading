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
| `groups` | array | 分组解读（body / cards / vol 三种布局；强/中/弱 标签色） |
| `technical` | object | 单标的：**价格阶梯**（止损→买区→现价→目标）+ 指标格 |
| `levels` | array | **建议买卖点大表**（核心；含内嵌 R/R 条） |
| `backtest` | object | 策略净值 vs 基准 折线图 + 统计 |
| `sentiment` | object | 三层时效情绪（−1…+1 双向条） |
| `portfolio_health` | object | 组合体检表（ENB/β/VaR…） |
| `holdings` | string/array | 持仓说明 |
| `conclusion` | string/array | 综合结论(可结构化为「维度卡 + 态度标」`{label,stance,text}`) |
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

"verdict": { "stance": "偏多 · 不追高",
             "points": [ {"icon":"🔥","text":"<b>财报暴击</b>:营收/EPS/指引全超"},
                         {"icon":"⚠️","text":"技术<b>极度延伸</b> → 回踩更稳"} ],  // 右栏要点列表(推荐)
             "action": "<b>等回调 860–915</b>…", "summary": "…" },  // 不给 points 时回退到整段文字
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

"groups": [
  // 每个分组三选一布局;tone(角标色): strong 红涨 | neutral 琥珀 | weak 绿跌
  // (a) 默认 body:整段 HTML(老格式,仍兼容)
  { "title":"分组标题", "tag":"角标", "tone":"strong", "body":"<b>…</b> &bull;… <br>…" },
  // (b) layout:"cards":新闻/事件 → 子分组小卡(每卡左边框按 sentiment 着色 + 项目符 + 可选 foot 脚注)
  { "title":"🗞 实时新闻头条", "tag":"财报暴击", "tone":"strong", "layout":"cards",
    "cards":[{ "label":"美光财报", "sentiment":"利多",   // sentiment 文案→色:利多/利好/做多→红涨,利空/做空→绿跌,其它→琥珀
               "items":["<b>FQ3</b> 营收 $41.46B…","Q4 指引 $50B…"] }],
    "foot":"灰色小字脚注(可选)" },
  // (c) layout:"vol":期权/财报波动 → 隐含 vs 实际 对比条(数轴上画隐含区间带 + 实际跳空点)
  { "title":"📐 期权/财报波动", "tag":"实际 > 隐含", "tone":"neutral", "layout":"vol",
    "vol":{ "implied_low":-8, "implied_high":8,  // 隐含跳空区间(%);或用 "implied":8 表 ±8
            "actual":16,                          // 实际跳空(%),冲出区间=超预期(红涨绿跌)
            "iv":102.8, "iv_pctile":96,           // 可选:IV 与 52周百分位 → 提示 vol crush
            "note":"一句含义(HTML,可选)" } }
],

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
  "support1":348, "support2":336,  // 支撑位(可选)；有则渲染「支撑」列，自动标注相对现价 %
  "rr":2.52,                       // 渲染为彩色 R/R 条：≥1.8 绿 / ≥1.2 蓝 / ≥0.9 琥珀 / <0.9 红
  // 注：buy_low/buy_high/support1/support2/stop/target 列均自动追加「相对现价的涨跌%」(由 price 算)
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


---

## 排版 / 一致性(2026-06)

- **`symbol_order`(顶层,可选)**:如 `["000660","005930","MU"]` —— 把 `levels` 与带 `symbol` 的
  `alerts` **统一按此公司顺序**展示,避免各区块顺序不一(`factor_rank` 是排名,不参与重排;
  无 symbol 的「宏观」类条目自动排最前)。**多标的报告建议都设它。**
- **市场环境网格**:`regime` 与 `macro` 并排;**只给其一时自动占满整行**(不再在右侧留空白)。
  事件前瞻用 `calendar`(`{event,date,in_days,impact,flagged}`)。


---

## regime / macro 面板行:`value` 放短指标,`read` 放解读(2026-06)

`regime`/`macro` 的 `rows[]` 是 `{item, value, read}` 三栏:**`item`=标签**、**`value`=一个短指标**
(数字或 2–4 字的 tag,如 `19.49` / `创新高` / `2× MU`)、**`read`=较长的解读句**。
`value` 列较窄(为短指标设计)——**不要把整句话塞进 `value`**,否则会挤成多行很难读;长描述一律放 `read`。
(渲染端已加宽容错 + 自动换行,但仍以"短 value / 长 read"为准。)


---

## 综合立场:5 档 + 多箭头强弱(2026-06 升级)

`verdict.stance` 仍是自由文本(大标题),但徽章现在分 **5 档**,并按强弱画多箭头:

| 档位 | 徽章 | 名称 | 颜色 |
|---|---|---|---|
| +2 | ▲▲▲ | 强烈看多 | 红 |
| +1 | ▲ | 偏多 | 红 |
| 0 | ◆ | 中性 | 琥珀 |
| −1 | ▼ | 偏空 | 绿 |
| −2 | ▼▼▼ | 强烈看空 | 绿 |

判定优先级:① 若给了数值 **`verdict.score`∈[-1,1]**,直接据此分档(±0.6 起为强烈、±0.15 起为偏);
② 否则按 `stance` 文字的**净倾向**(看多词数 − 看空词数,修复了"多空词并存时只取第一个"的旧 bug)。
出现 `强烈/满仓/坚定` 等词或净倾向≥2 → 强档;出现 `超买/延伸/不追/谨慎/高位/震荡` 等 → 降为偏档。
**想精确控制档位,直接传 `verdict.score`**(可由 regime/macro 分数推导,更客观)。

**右栏排版(2026-06)**:`verdict.points`(数组,元素为 `{icon,text}` 或纯字符串)→ 渲染成带图标的**要点列表**(粗体只留关键词,比一长段 `action` 易读);不给 `points` 时回退 `action`/`summary` 段落(仍兼容)。`text` 可含 `<b>` 与 `<span class='pos'>`/`<span class='neg'>` 上色。

**综合结论位置(2026-06)**:`conclusion` 现渲染在「综合立场」**正下方**,作为顶部结论汇总(**不编号**);其余编号区块顺延。其下方依次才是 ① 今日重点关注、② 市场环境……


---

## 信号徽章自动判定:`levels.classify_signal`(2026-06)

alert / levels 行的 `signal`(做多/观望/做空)不再手填,可由 `scripts.levels.classify_signal` 按数据算:
```python
from scripts.levels import classify_signal
sig = classify_signal(rsi=47, rr=1.85, trend="bull", in_buy_zone=True, event=False, leveraged=False)
# long: 多头 + 不超买(RSI<70) + R/R≥1.5 + 现价在买区 + 无事件 + 非杠杆;否则 watch;空头未超卖→short
```
报告里把每个标的算出的 `sig` 同时用于 `levels[].signal` 和该标的的 `alert.signal`。当前几只多为 watch,
是因为价格都已跳空在买区上方 / 超买 / 杠杆——回踩进买区且 R/R 够,才会翻成 long。


---

## 信号徽章:事件情绪 + 持有周期(AI 判定,非技术)(2026-06)

`alert.signal` / `levels[].signal` 现支持两种写法:
- 旧:`long/watch/short` → 做多/观望/做空(技术口径,保留兼容)。
- 新(推荐):**事件/新闻情绪 + 持有周期**,如 `"利多·短线"` / `"利多·中线"` / `"利空·短线"` / `"中性"`。
  **由 AI(Claude)读新闻判定**(不看技术指标);`sigBadge` 按情绪词上色(利多/利好→红、利空→绿、其余→琥珀),文字原样显示。
  这样既给方向(利多/利空/中性),又给持有周期(短/中/长线)——例如杠杆 ETF 给 `"利多·短线"`(认可方向、但提示只宜短炒)。
> 判定主体是 AI(静态报告里由 Claude 在生成时判;实时 artifact 可用 `window.cowork.askClaude`)。
> `levels.classify_signal`(技术口径)仍保留为可选工具,但事件情绪徽章不再依赖它。


---

## ## 三条独立判定线(别混用)(2026-06)

报告有三个**互相独立**的判定,别混为一谈:
1. **综合立场 `verdict`(全文最终结论)**:5 档 强烈看多/偏多/中性/偏空/强烈看空,由 `verdict.score`
   或 regime/macro 量化分数得出。**这是整篇的总结论。**
2. **今日重点关注 `alerts[].signal`(逐条新闻/事件)**:**利多 / 利空 / 中性**(可带持有周期,如 `利多·短线`),
   **由 AI 读该条新闻/事件判定**,只针对那一条事件,不是全文结论。
3. **建议买卖点 `levels[].signal`(逐标的技术信号)**:做多/观望/做空,由 `levels.classify_signal`
   按 RSI/RR/趋势/买区算(技术口径)。
> 即:`alerts` 答"这条消息利多还是利空"、`levels` 答"这价位技术上能不能买"、`verdict` 答"整体什么立场"。


---

## 综合结论:维度卡 + 态度标(2026-06)

`conclusion` 现支持两种写法(渲染器自动识别,**老的纯字符串数组仍兼容**):

- **老**:`["<b>基本面:</b>…", "<b>技术:</b>…"]` —— 渲染成平铺段落。
- **新(推荐)**:对象数组,每个维度一张卡 + 右上角「态度标」:
  ```json
  "conclusion": [
    { "label":"基本面", "icon":"📊", "stance":"利好",     "text":"FQ3 暴击 + Q4 指引 $50B,需求验证 —— 真利好。" },
    { "label":"技术",   "icon":"📈", "stance":"谨慎·延伸", "text":"+16% 创新高、跳空在买区上方 → 极度延伸…" },
    { "label":"MUU",    "icon":"⚠️", "stance":"高风险",   "text":"2× MU、止损 −40%、有衰减 —— 仅短线、不可持有。" }
  ]
  ```
  - `stance` 文案→色(红涨绿跌):`利多/利好/做多/强劲`→红;`利空/做空`→绿;`风险/谨慎/观望/延伸`→琥珀;
    `中性/双向` 及其余→灰。卡片左边框与标签同色。
  - `icon`/`label` 可选(缺省按 label 取默认图标)。

**态度标从哪来(重要)**:它**不是引擎算的,也不是单独的 AI 分类调用**,而是 author(生成报告的 Claude)
写结论时一并填的标签,渲染器只显示。为避免变成"凭感觉",**约定 `stance` 必须对齐报告里已有的量化口径**:
基本面←`regime` 财报行、技术←`regime` 趋势/位置、期权←IV 百分位、杠杆 ETF←杠杆属性(固定高风险)、情景←双向(固定中性)。
> 它**不是**「三条独立判定线」之外的第 4 条判定 —— 只是把全文结论按维度拆开展示的**汇总标签**,应与 `verdict`/`regime` 自洽。

## 环境面板 value 列:数字 vs 状态标签(2026-06)

`regime` / `macro` 行的 `value` 现在<b>自动判别</b>:含中文 → 渲染成<b>状态标签</b>(按语义上色),纯数字/英文(如 `VIX 18`、`IV 103%`、`$41.5B`)→ 保持<b>等宽左对齐</b>。整列左对齐共享左边沿,解决了中文词等宽右对齐时的参差。标签配色:利好/扩张类(加速/超级周期/强/高…)→红,警惕类(延伸/偏紧/高位/谨慎…)→琥珀,其余→灰;可用行内 `"tone":"pos|warn|neu"` 强制覆盖自动判断。

## 信号多法对照 `methods` + `signals.py`(2026-06)

`scripts/signals.py` 给单标的提供 **6 种互补判定**(各问不同问题,故结论可不同),配 report 的 `methods` 字段在**报告最底部**渲染成「方法×标的」对照表:

- `m1 tech_rating`:均线族+振荡器投票→[-1,1]→5 档(强力买入/买入/中性/卖出/强力卖出,TradingView 式)
- `m2 strength_score`:0–99 多周期技术强度(SCTR 思路,单标的自有历史口径)
- `m3 walkforward_signal`:MA20/50 walk-forward 样本外 Sharpe(edge)+ 当前方向
- `m4 regime_signal`:趋势/波动态 + 建议敞口(regime_scale)
- `m5 autoresearch_signal`:`autoresearch.research_single` 多策略搜索 + OOS 选最优
- `m6 breakout_signal`:趋势跟随/突破口径(对照「回踩买」)
- `old rule_signal`:旧 `classify_signal`(回踩买点口径)做对照

`signals.all_methods(df, iterations=8)` 一次返回全部;每个方法返回 `{label, tone(pos/neg/warn/neu), detail}`。
report:`methods:{title, note, symbols:[{key,name}], rows:[{m,desc,key}], data:{SYMBOL:{m1..m6,old:{label,tone,detail}}}}`;渲染器 `methodsTable` 自动排在所有编号区块之后(最底部)。
> ⚠️ OOS 指标在**单边趋势 + 短历史**下会被高估(过拟合);无价格序列的标的(如 broker 无 A股)相关方法返回 N/A。

## 自动研究详情 `research`(2026-06 · 含具体买卖点 + 回测)

report 的 `research` → 报告最末「自动研究详情」节(`researchSection`),逐标的三段:
- **策略选择**:`selection_text`(bandit 各族尝试次数/平均分)+ `leaderboard:[{rank,strategy,params,oos_sharpe,oos_return,signal,buy,sell,win}]`(样本外去重 TopN,`win` 高亮;每行含**该策略的当前信号 + 买入/卖出触发价**)
- **策略输出**:`winner:{strategy,params,oos_sharpe,oos_return,signal,signal_tone, triggers:{action,sell,buy}}` —— `triggers` 是规则反推的**具体买卖触发价**(跌破X卖 / 站上Y买,逐日移动,非固定目标价)
- **买卖点图 + 回测**:`trades:{price:[], buys:[idx], sells:[idx], hold:[[s,e]], logy, unit, date_start, date_end, now_label}` → `tradesChart` 画 价格折线 + ▲买入/▼卖出 + 绿阴影持仓;`stats:[{k,v}]` 给"策略收益 vs 买入持有 vs 夏普/回撤"

由 `build_research.py`(`autoresearch.research_single` + `REGISTRY[best.direction](**params)` 回测 + `strat.generate_signal` 取进出点)生成。
⚠️ 触发价随行情移动;OOS 在单边牛市/短历史下高估(冠军总收益常不及买入持有、仅回撤更小);无价格序列标的 `winner=N/A`、`trades=None`。
