# Alpha-Forge — 2026-07-02 优化轮(Round 9)

接 `CHANGELOG_2026-06-25.md`(Round 7-8)。本轮四条线:**目录重组**、**路线图 A2/A3/A5/A6 落地**、
**报告可视化 B2/B3/B4**、**回归测试扩充**。全程 pytest 验证 → **118 passed**(新增 13 个 round-9 回归)。

## 一、scripts/ 目录重组(不留旧路径)

平铺的 ~30 个模块按职责分包;**旧 import 路径(`scripts.backtest` 等)彻底移除,不做兼容别名**:

| 子包 | 内容 |
|---|---|
| `core/` | backtest · metrics · indicators · optimize · validation · rebalance · calendars |
| `risk/` | sizing · regime · portfolio · levels · **conformal(新)** |
| `research/` | autoresearch · models · factor_lab · signals · signal_tracker · param_grids · compare |
| `xsec/` | universe · panel · xsec_eval · xsec_models · xsec_autoresearch · xsec_report |
| `reporting/` | html_report · build_research · report · attribution · newsfeed |
| `trade/` | execution |

`data/`、`strategies/` 不变;CLI 入口仍是 `scripts/run_backtest.py`。全部内部 import、tests、
run_xsec_us.py、examples、SKILL.md/SCHEMA.md/references 同步改写并清查(全仓 grep 无旧路径残留)。

## 二、A2 换手/成本惩罚(turnover-regularized objective, 2407.21791)

- `metrics.cost_stressed_sharpe(stats, extra_bps)`:从已有 stats 精确算出「多收 extra_bps 成本后的夏普」
  (drag = turnover×bps/1e4 / vol),零重跑成本。
- `optimize.grid_search/walk_forward(cost_stress_bps=)`:**选参**用压力夏普,**OOS 报告仍按真实成本**;
  walk_forward 的 `oos_stats` 新增 `turnover_annual`。
- `autoresearch.research_single/research_portfolio(cost_stress_bps=10 默认开)`:bandit 奖励与冠军选择
  用压力分(leaderboard 增 `sel_score` 列),展示的 `oos_sharpe` 不变——高换手纸面冠军自动降权。
- `multi_factor_signal(weight_smoothing=)`:调仓权重向上期收缩(L1 换手惩罚),桶边缘股不再每月进出;
  同 gross 归一。`ml_factor_backtest` 原有 `weight_smoothing` 保持。

## 三、A3 集成升级(简单组合稳定优于单模型)

- `ensemble_top_k` 重写:成员**按族去重**(两个 MA 交叉≈同一笔交易,叠加是集中不是分散);
  `weighting="equal"|"ewma"(默认)|"regime"` —— ewma=对成员近期 EWMA 夏普做 softmax 的**时变权重**
  (AlphaForge 2406.18394 动态组合器,因果 shift);返回 (BacktestResult, members)。
- `models.StackingModel`:**out-of-fold stacking**(时序连续折,成员 OOF 预测→ridge 元学习器,
  防"元学习器奖励记忆者");默认 ridge×2(+lgbm 若装了),注册进 MODEL_REGISTRY,可直接喂
  `ml_factor_backtest` / `evaluate_cross_section`。
- `xsec_autoresearch.ensemble_top(data, lb, k)`:排行榜前 k 配置的预测**逐日转 rank 百分位再平均**
  (不同模型预测尺度不可比,rank 才平权),用同一张诚实记分卡重打分,返回 members。

## 四、A6 regime 条件化集成(修下跌段集体失效)

- `regime.regime_conditional_weights(member_returns, close)`:按 vol 三态×牛熊 组成 regime 状态,
  每个成员的权重 = 它在**过去同状态 bar** 上的 EWMA 夏普(状态内 shift,严格因果)→ softmax;
  样本不足的状态回退等权。接入 `ensemble_top_k(weighting="regime")`。

## 五、A5 共形预测区间 → 仓位(CPPS 2410.16333;新文件 `risk/conformal.py`)

- `split_conformal_qhat(residuals, alpha)`:分布无关、有限样本覆盖的校准区间宽 q̂;
  `adaptive_alpha`(ACI, Gibbs & Candès)应对非平稳;`conviction_scale(pred, q̂)`=|pred|/q̂ 截断 [0,1]。
- `ml_factor_backtest(conformal_alpha=0.2)`:训练窗尾部 20% 做校准,逐名字按 |预测|/q̂ 缩权
  (**只减仓不加仓**;q̂ 无效时不装懂,保持 1.0)。`MLResult.extra` 带 qhat 历史。

## 六、B2/B3/B4 报告可视化(把诚实机器画出来)

- **B2 `robustness`**:Deflated Sharpe / PBO / SPA p / CPCV为正占比 计量表(带阈值人话注解)
  + **CPCV 出样本夏普分布 SVG**(每点一条路径、5-95% 带、中位线、0 虚线)+ 一句判定。
- **B3 `downturn`**:`metrics.downturn_slices` 强制展示基准最差季度/最差滚动63日/最深回撤段的
  策略 vs 买入持有对照(StockBench 教训:别只看顺风段)。
- **B4 `cost_curve`**:`backtest.cost_sensitivity` 同一信号 0/10/30bps 三档重跑 + 夏普对比条。
- `build_research` 自动生成三块;渲染器把 `research.{robustness,downturn,cost_curve}` **自动上提**
  为顶层区块。SCHEMA.md 已补契约。

## 七、测试

新增 `tests/test_round9.py`(13 个):成本压力单调性/优雅降级、walk-forward OOS 换手、sel_score 排序、
smoothing 减换手且 gross 不变、三种集成模式+族去重、stacking 预测力、xsec rank 集成落 [0,1]、
regime 权重因果+归一、共形 q̂ 数值/无效降级、共形门 gross 只减不增、下跌切片形状、成本敏感单调、
build_research 三块齐全+渲染含新区块。**全套 118 passed**。

## 八、巡检修复(独立代码审查子代理确证后修复,均带回归测试 → 全套 123 passed)

| 级别 | 修复 |
|---|---|
| **P0** | `ml_factor_backtest` 权重矩阵 0.0 初始化使结尾 ffill 失效 → **每月只持仓 1 根 bar**(上轮遗留,本轮所有 ML 路径都叠在其上)。改 NaN 初始化,组合真正持有到下次调仓。 |
| **P1** | `weight_smoothing` 把 gross 归一回 1.0 → 抵消 conformal 门的减仓、long-short 杠杆减半。改为重标定到门后目标 gross。 |
| **P1** | `_ewma_softmax_weights` / `regime_conditional_weights`:平躺成员(NaN 夏普)在归一化**之后**被 fillna → 行和 >1 的隐性杠杆、混合信号超 [-1,1]。改为 softmax 内中性 0 分。 |
| **P2** | `downturn_slices` 全面改**位置索引**:非 DatetimeIndex 优雅退出、重复日期不再抛错。 |
| **P2** | `StackingModel` 数据过短改抛 ValueError(原 ImportError 会被 autoresearch 当"缺库"静默吞掉)。 |
| **P3** | walk-forward OOS 换手补上测试窗首根建仓;xsec 集成 `fwd` 不再依赖 0 号成员覆盖面。 |

---

# Round 10(同日)— roadmap v3 阶段 1-6 轻量项落地

按 `references/optimization_roadmap_v3.md`(引用已核验修订版)执行;4 个子代理按文件域并行开发,
互不触碰,主线程整合。**全套 182 passed**(新增 59 个 round-10 测试)。

## 一、P0 因子供给与质量(§2.1-2.4)

| 新模块 | 内容 |
|---|---|
| `research/factor_expr.py` | 因子表达式 DSL:ast 白名单安全求值(禁属性访问/import),**80 个算子**(时序 24 全因果 + 截面 rank/zscore/scale/group_* + K线宏),单标的→Series、{symbol:OHLCV}→宽表;`expr_to_callable` 直通 validate_factor |
| `research/factor_zoo.py` | **Alpha101×30(纯价量)+ Alpha158 子集×84(8 类全覆盖)**;`compute_library` 逐因子容错;`alpha360_panel` 留深度模型输入口 |
| `factor_lab.py` 追加 | `orthogonalize`(逐日截面联合回归取残差)、`incremental_ic`(正交后剩余 RankIC)、`factor_correlation_matrix`、`complexity_control`/`novelty_check`(AST 深度/参数计数/归一化编辑距离,AlphaAgent 式)、`alpha_eval` **五维**(预测力/鲁棒/多样/稳定 纯代码 + 可解释性诚实返回 prompt 交 agent) |
| `research/crowding.py` | 持仓重合 Jaccard、多空收益滚动相关、估值价差、综合 `crowding_score`(>0.7 预警)、`fit_hyperbolic_decay`(α(t)=K/(1+λt),注明源文已撤稿仅作拟合工具) |
| `research/decay_monitor.py` | 滚动 IC、多 horizon IC 衰减、half-life **三法+中位**(AR1/滞后IC/分位价差,持久信号返回 inf 而非乱拟合)、`decay_warning`、MRP |

## 二、P2 横截面评估与选池(§2.7/2.8)

- `panel.cs_zscore(groupby=)`(Qlib CSZScoreNorm 对标)+ `neutralize_panel`(行业组内去均值 + 风格逐日截面回归残差,可叠加)。
- `evaluate_cross_section(neutralize="industry"/"style"/"both")`:**只动特征不动标签**;默认口径逐帧回归确认零破坏;scorecard 记录中性化口径。
- `universe.dynamic_universe / anti_survivorship_pool / rolling_universe`:as-of 选池(≥60 根历史 + 市值/ADV 滤条,缺字段保守保留并 warning),逐调仓日重选。

## 三、P3/P4/P5 轻量项(§2.10/2.11/2.13/2.17-2.20)

- `risk/optimization.py`(纯 numpy):MinVar/MaxSharpe(active-set 处理 long_only,非"清零重归一")、有效前沿(两基金定理)、Black-Litterman(He-Litterman Ω 缺省)、带 cap/行业约束 MVO(投影梯度+POCS)、`views_from_predictions`+LLM prompt 模板。
- `reporting/factor_tearsheet.py`:自包含 HTML 因子体检页(分位累计/逐 horizon RankIC/IC 时序/多空/月换手/分位统计,零 JS 依赖;注脚明示不扣成本)。
- `regime.stock_drift_regime / drift_regime_gate`(§2.13):逐股 drift 门,shift(1) 因果,docstring 照 roadmap 核验结论声明来源数值不可信、阈值须自测。
- 评价补件:`metrics.mae_mfe`(逐持仓段最大不利/有利偏移)、`metrics.capm_decompose`(α/β/Treynor/IR)、`sizing.kelly_fraction/kelly_weights`(对角 Kelly,cap 截断)、`attribution.brinson`(Brinson-Fachler,三项和=超额恒等式精确)。

## 四、本轮不做(依 roadmap 依赖与环境)

§2.5/2.6(listwise/深度模型,需 torch)、§2.9(研究记忆/跨市场迁移,大件)、§2.12/2.15(LLM prompt 生态)、§2.21(RL 执行,需本地 GPU)→ 下一轮;autoresearch 与新质量套件的自动接入(增量 IC 剔冗/拥挤降权)也留下一轮,避免一次改动过宽。

## 五、测试

新增 4 个测试文件共 59 个:`test_round10_expr.py`(19:算子手算对照/因果抽查/白名单拒绝/库全量 compute/端到端 xsec)、`test_round10_factorlab.py`(15)、`test_round10_xsec.py`(11 + 现有 xsec 零破坏回归)、`test_round10_portfolio.py`(14:解析解对照/BL 恒等/约束满足/Brinson 恒等式)。**全仓 182 passed。**
