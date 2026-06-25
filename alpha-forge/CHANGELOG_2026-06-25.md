# Alpha-Forge — 2026-06-25 优化轮(Round 7)

接 `CHANGELOG_2026-06-23.md`(Round 1–6)。本轮聚焦:**选择层稳健性(CPCV/SPA)**、
**补上 SCHEMA 点名却缺失的策略输出生成器 `build_research.py`**、**策略/方法介绍**、
以及一批**报告渲染的真 bug**。全程 pytest 验证 → **87 passed**。

## 一、validation 稳健性升级(选择层"诚实机器")

| 新增 | 作用 |
|---|---|
| `cpcv(returns, n_groups, k_test, embargo)` | **组合净化交叉验证**:产出 OOS 夏普的**分布**(中位 + 5/95% 带 + 为正占比),而非一个点估计。一个数字掩盖了 edge 是否只靠一个幸运窗口,这个揭示它。 |
| `spa_test(trial_returns)` | **White Reality Check / Hansen SPA**(stationary bootstrap):从 K 个搜过的策略里"挑最好"是不是数据窥探的产物。`spa_p<0.05` 才算通过。deflated Sharpe 校正单条策略,SPA 校正"选择"这一步,二者互补。 |
| `selection_robustness(trial_returns)` | 一站式打包 Deflated Sharpe + PBO + SPA + CPCV,供报告「稳健性体检」直接用。 |

> 还修了 `validation` 里两处早期统计实现:`pbo_cscv` 符号方向、`deflated_sharpe_ratio` 默认 `sr_std` 量纲(默认会让 DSR 恒≈0);均加回归测试。

## 二、`build_research.py`(新文件) — 补上缺失的"策略测试选择"生成器

SCHEMA.md 详细描述了 `research` 块(冠军+触发价、排行每行带**当前信号 + 买卖触发价**、
**买卖点持仓图**、stats、glossary)并**点名由 `build_research.py` 生成**——但该模块一直**不存在**,
导致任何调用方都只能手搓精简版。本轮按规格补上:

- **搜索委托引擎**:逐族经 `autoresearch.screen_rule_strategies` 网格穷举出最优配置 + walk-forward 出样本夏普(冠军/排行的 `params` 一律真实搜索值);**短历史**(杠杆ETF/次新)折回样本内 + `selection_robustness`(CPCV/PBO/SPA),诚实说清"冠军是真 edge 还是搜索幸运"。
- `trades_viz(df, res)` 从(已滞后的)持仓序列生成 `tradesChart` 的 price/▲buy/▼sell/绿阴影持仓。
- `_triggers(name, df)` 按各策略自身逻辑反推**当前买卖触发价**(逐日移动,非固定目标)。
- 返回 `{title, items:[{winner, selection_text, leaderboard, trades, stats}], glossary, robustness}`。

## 三、策略 / 方法介绍(用户需求:策略测试选择部分带每个策略方法的介绍)

- `param_grids.STRATEGY_INFO`:7+1 策略族的一句话原理 + 适用场景(单一事实源,键与 REGISTRY 对齐)。
- `signals.METHOD_INFO` + `signals.methods_report(...)`:「信号多法对照」每行带 `desc`;**默认六法全跑**(含样本外择时 / 自动研究),`heavy=False` 仅作大规模筛选的显式降载开关。
- `autoresearch.strategy_glossary(report)`:按研究排行涵盖的策略族产出说明。
- `html_report.researchSection`:策略选择下方新增 **📖 策略方法说明**。

## 四、报告渲染 —— 修掉几个真 bug

| 修复 | 影响 |
|---|---|
| **`render` 的 NaN/Inf → null**(`_json_safe`) | Python `json.dumps` 默认写出字面 `NaN`,**浏览器 `JSON.parse` 会抛错把整份报告变空白**。任何含 NaN 的报告都会中招(Python 容忍所以一直没暴露)。 |
| `el` 的 `html:` 走 **`safeHtml` 黑名单消毒** | 数据字段(新闻/AI 文本)里的 `<script>/<img onerror>/javascript:` 被剥离,良性 `<b>/<span class>` 保留;SVG 图表改走 `raw:`。 |
| `levelLadder` 除零守卫 `span=(hi-lo)\|\|1` | 退化(止损=现价=目标)不再 NaN 错位。 |
| 表头:删远程 Google Fonts `@import` + 日期戳防与标题/分隔线重叠 | 离线/预览一致渲染,不再发灰/裁切/叠字。 |
| `signals.all_methods` 默认 `heavy=True` | 默认六法全跑;`heavy=False` 是大规模筛选的**显式**降载开关(留空 walk-forward/autoresearch 两栏),非自动静默跳过。 |

## 五、文档

- `优化建议_量化模型与报告_2026-06-25.md`:基于 2025-2026 一手调研(CPCV/SPA、AlphaAgent/AlphaEval、conformal sizing、TabM、FINSABER/StockBench/Profit Mirage 等)的优化路线图。**本轮已落地 A1(CPCV)+ A4(SPA)**;后续 P0:A2 换手成本惩罚、A3 集成升级(rank-average + 时变权重)。

## 六、收口:搜索路径统一 + 模板镜像 + 报告样式同步(Round 8 补遗)

> 用户复核后要求:把 `build_research` 与引擎的**重复搜索**收成一条路径,并确保改动后**其余文件同步、报告样式正常**。

- **唯一搜索路径**:新增 `autoresearch.screen_rule_strategies(df)` —— 对 7 个规则族逐一网格穷举最优配置、回测、再 walk-forward 打出样本夏普(`research_single` bandit 的确定性全覆盖对偶)。`build_research` 退化为**纯展示层**:只消费引擎结果做排名/触发价/稳健性体检,**不再自搜**。删掉 build_research 里的 `optimize`/`PARAM_GRIDS` 依赖。
- **params 真实值守门**:leaderboard 与 winner 的 `params` 必为真实搜索值(如 `fast=20·slow=100`),仅无可搜参数族回退 `默认`;新增回归测试 `test_build_research_full_structure` 断言**绝不出现全 `默认`** 占位。
- **多法默认全跑**:`methods_report` 默认 `heavy=True`,样本外择时(m3)/自动研究(m5)不再静默跳过;DRAM 三家实测 m3 OOS Sharpe 3.3/4.1/2.7、m5 实选最优族。
- **删回测净值图**:模板移除 `equityChart` + `data.backtest` 渲染;`templates/render.js`、`templates/report.css` 重新生成为 `html_report._JS/_CSS` 的**忠实镜像**(原 Jun-9 旧模板已严重漂移)。SCHEMA/SKILL 中"净值曲线"描述同步改为"买卖点图"。
- **买卖点带日期**:`tradesChart` ▲买/▼卖 标记附 `MM-DD`(≤16 个标记时)+ 起止日期;`trades_viz` 输出 `dates`/`date_start`/`date_end`。
- 复跑 DRAM 报告:严格 JSON ✓、无远程字体 ✓、无 `backtest`/`equityChart`/字面 `NaN` ✓。

## 测试
`tests/test_alpha_forge.py` 累计 **87 passed**(本轮新增:CPCV 分布、SPA 显著性方向、selection_robustness、build_research 结构、render NaN-safe、signals heavy、methods/glossary 等回归)。
