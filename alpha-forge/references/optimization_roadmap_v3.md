# Alpha-Forge 优化方案 v3 — 选股能力与研究能力深化

> 聚焦 alpha-forge 核心定位: **横截面选股 + 自动研究循环 + 因子工程 + 诚实验证**
> 砍掉交易引擎/实盘运维/交易策略模板(属另一赛道)
> 调研基础: AKQuant 教材 + Qlib/RD-Agent + 2025-2026 学术论文(24篇)
> 编制日期: 2026-07-02 · **修订 2026-07-02**: 全部 24 条引用已逐条访问原文核验
> (2 个并行审查代理,arXiv/ACM/NBER/GitHub 摘要页比对)——来源全部真实存在,
> 但 8 处表述已按原文修正,2 条来源降级(1 撤稿、1 可信度极低);现状描述 3 处
> 与代码比对后修正。核验详情见附录 C 的「核验」列与 C.3 高风险来源说明。
> **落地进度 2026-07-02(Round 10)**: ✅ 已完成 §2.1-2.4, 2.7, 2.8, 2.10, 2.11, 2.13,
> 2.14(纯代码部分), 2.16(纯代码四维+可解释性 prompt), 2.17-2.20;
> ⏭ 待做 §2.5/2.6(需 torch)、2.9(研究记忆/迁移)、2.12/2.15(LLM prompt 生态)、
> 2.21(需本地 GPU)、以及 autoresearch 与质量套件的自动接入(增量IC剔冗/拥挤降权)。

---

## 0. 定位

### 0.1 为什么不补事件引擎/实盘运维

alpha-forge 的护城河是 **研究层**(autoresearch/factor_lab/models/validation)与 **选股层**(xsec/),不是交易系统。事件驱动引擎、MarketRules 撮合、动态策略加载、硬熔断、热启动属于 NautilusTrader/QuantConnect-Lean 赛道。本方案只做能让 **选股更准、研究更快、因子更丰、验证更严** 的事。

### 0.2 skill 现状(选股/研究视角)

**强项(保持)**:
- `xsec/`: universe 选池(内置基座覆盖 US/CN/HK;KR/JP 经 loader/market 层可用但无内置池) + panel 截面因子(**8** 个价量: mom20/60/120, rev5/10, vol20, dist_high60/low60) + eval(IC/RankIC/ICIR/分位/多空) + autoresearch(因子×模型搜索+rank集成) + models(Ridge/LGBM/MLP/TorchRanker)
- `research/autoresearch.py`: RD-Agent 式循环 + UCB1 bandit + factor-model co-opt + holdout tail 验证
- `research/factor_lab.py`: validate_factor(因果性探测) + backtest_custom_factor + factor_ic + scorecard(ic/ic_ir/autocorr/coverage/max_corr)
- `research/models.py`: Ridge/LGBM/MLP + OOF stacking + walk-forward + conformal 门控
- `core/validation.py`: DSR/PBO/CPCV/SPA(过拟合校正)
- `data/pit.py`: PIT dated archive(选股回测诚实性)
- `data/market.py`: 多市场识别(market_of) + 指数 regime(index_for/index_regime,US/CN/HK/KR/JP)
- `risk/`: sizing(RP/IV/VT) + regime(vol/trend/CUSUM/GMM) + conformal(ACI) + portfolio(VaR/CVaR/stress)

**弱项(本方案目标)**:
1. 因子供给: 无表达式 DSL,无 Alpha101/Alpha158 库,仅 7 个手写价量因子
2. 因子质量: 无正交化/增量评估/拥挤度/衰减监控
3. 排序模型: 仅 pointwise(MSE),缺 listwise(ListNet/ListMLE/LS-List)
4. 深度模型: 仅 MLPModel(sklearn)+TorchRanker(简单MLP),无序列模型(LSTM/GRU/Transformer/MASTER)
5. 横截面评估: 无行业/风格中性,无 CSZScoreNorm
6. 研究循环: autoresearch 有 bandit+co-opt 但无知识累积、无跨市场迁移
7. 组合优化: 仅 RP/IV/VT,缺 BL/MinVar/MaxSharpe
8. 因子 tearsheet: 无 Alphalens 式标准化输出
9. 评价体系: 缺 MAE/MFE/Kelly/CAPM/Brinson
10. 选池: 静态选池,无动态/反幸存者
11. 执行: 仅 TWAP/VWAP 静态计划,无 RL 执行,无 IS 质量指标

### 0.3 调研基础

| 调研方向 | 关键发现 | 对应方案项 |
|---|---|---|
| Qlib Alpha158/360 | 158因子7类 + 表达式DSL + CSZScoreNorm + IC/ICIR | 2.1, 2.7 |
| RD-Agent(Q) | Research–Development–Feedback 循环 + Co-STEER 知识库 + 因子-模型协同 + bandit | 2.9 |
| MASTER (AAAI-2024) | Market-Guided Gating + Intra/Inter-Stock Aggregation | 2.6 |
| Stockformer | 小波分解 + 多任务自注意力 + 图嵌入 | 2.6 |
| ListNet/ListMLE/长短仓listwise | listwise > pointwise;长短仓 listwise 论文自报 A股 2006-19 年化38% Sharpe2 | 2.5 |
| 因子正交化/拥挤度/衰减 | Gram-Schmidt + 持仓重合 + half-life + 双曲衰减(⚠️主源已撤稿) | 2.2, 2.3, 2.4 |
| AlphaAgent (2025) | 三重正则化(复杂度/假设对齐/AST新颖性)对抗衰减 | 2.10 |
| Alpha-R1 (2025) | LLM 推理因子经济逻辑+新闻→激活/停用 | 2.12 |
| FactorMiner (2026) | 模块化技能架构 + 经验记忆(成功模式/禁区),对抗知识遗忘 | 2.9 |
| Drift Regime (2025) | regime-conditional 信号激活(🔴其 Sharpe>13 数值不可信,仅借思路) | 2.13 |
| AlphaEval (2025) | **五**维评估(预测力/鲁棒/多样/可解释/稳定),无回测预筛 | 2.16 |
| LLM-Enhanced BL (2025) | LLM 系统性生成 BL views | 2.8 |
| Meta-Learning (2025) | 跨市场参数迁移 + 策略聚类 | 2.14 |
| RL 执行 (2025) | RL IS 优于 TWAP/VWAP | 2.15 |

---

## 1. 设计原则

1. **围绕选股-研究闭环**: 每项必须服务"因子更丰→选股更准→验证更严→研究更快"中的至少一环
2. **因子表达式统一入口**: 字符串 DSL 与 Python callable 同口径,都能进 validate_factor 与 xsec_eval
3. **listwise 优先**: 选股本质是排序问题,排序损失从 pointwise 升级到 listwise
4. **深度模型 skill 内训练**: 训练代码在 skill 内,轻量模型线上训练,重型模型本地 GPU 训练后加载推理
5. **LLM 边界明确**: 纯代码部分在 skill 内;LLM 辅助部分由 agent(Claude)侧执行,skill 提供 prompt 模板+解析函数
6. **诚实性不退步**: 新因子/新模型必须经 validation.py 过拟合校正 + PIT 诚实回测
7. **可选依赖优雅降级**: polars/torch/lightgbm/scipy 缺失时回退
8. **不重复造轮子**: 已有 validation/autoresearch/xsec 不重写,只补缺口与深化

---

## 2. 优化项清单

### P0 — 因子供给与质量

#### 2.1 因子表达式引擎 + Alpha101/Alpha158 库
- **位置**: 新增 `research/factor_expr.py`
- **算子集**(对标 WorldQuant Brain + Qlib):
  - 截面: `rank`, `zscore`, `quantile`, `group_rank`, `group_neutralize`, `vector_neut`
  - 时序: `delay`, `delta`, `ts_mean`, `ts_std`, `ts_rank`, `ts_zscore`, `ts_arg_max/min`, `ts_corr`, `ts_cov`, `ts_decay_linear`, `ts_sum`, `ts_max/min`, `ts_skew/kurt`
  - 算术: `+ - * / log abs sign max min power`
  - Qlib 风格: `Ref`, `EMA`, `SMA`, `WMA`, `Std`, `Corr`, `Cov`, `Rsquare`, `Resi`, `WVMA`, `KMID`, `KLEN`, `ROC/ROCP/ROCR`, `IMAX/IMIN`, `QTLU/QTLD`
- **内置因子库**:
  - Alpha101: ~30 个经典 WorldQuant 公式
  - Alpha158: Qlib 158 因子分 7 类(动量/波动/量能/价格形态/均线/相关/回归)
  - Alpha360: 原始价量序列(供深度模型输入)
- **后端**: 默认 pandas;polars 可选(自动检测,缺失回退)
- **接入**: `factor_lab.eval_expr("rank(ts_delta(close, 5))", data)` → 直接进 validate_factor / xsec_eval / xsec_autoresearch

#### 2.2 因子正交化与增量评估
- **位置**: 扩展 `research/factor_lab.py`
- **新增**:
  - `orthogonalize(factor, reference_factors, method="schmidt") -> Series` — Gram-Schmidt 正交化
  - `incremental_ic(factor, existing_factors, prices, horizon=21) -> dict` — 正交后剩余预测力
  - `factor_correlation_matrix(factors_dict, prices) -> DataFrame` — 因子间相关矩阵
- **接入**: xsec_autoresearch 搜索时自动用增量 IC 剔除冗余因子;scorecard 报告增量贡献

#### 2.3 因子拥挤度监控
- **位置**: 新增 `research/crowding.py`
- **指标**:
  - `holdings_overlap(factor, other_factors) -> float` — 多空组合持仓重合度
  - `factor_correlation_crowding(factor_returns, lookback=63) -> float` — 滚动相关性
  - `valuation_spread(long_leg, short_leg) -> float` — 估值价差(拥挤→反转风险)
  - `crowding_score(factor, prices, *, lookback=252) -> dict` — 综合拥挤度(>0.7 预警)
  - 双曲衰减拟合 `alpha(t)=K/(1+λt)`(arxiv 2512.11913 ⚠️**该文 v2 已被作者撤稿**,称实证部分不足以支撑结论、正在大修——函数形式可作为拟合工具保留,但"机械 vs 判断因子""拥挤预测尾部风险"两条结论降级为待验证假设,勿在报告里当已确立结论引用;实现时以 Man Institute 的拥挤度研究为主要方法学依据)
- **接入**: scorecard 自动附带拥挤度;autoresearch 降权高拥挤因子;HTML 报告增加拥挤度预警

#### 2.4 因子衰减监控与 half-life 估计
- **位置**: 新增 `research/decay_monitor.py` + 扩展 `research/signal_tracker.py`
- **指标**:
  - `rolling_ic(factor, prices, horizon=21, window=63) -> Series` — 滚动 IC
  - `ic_decay(factor, prices, horizons=(1,5,10,21,42,63)) -> dict` — 多 horizon IC 衰减曲线
  - `half_life(factor, prices, *, method="auto") -> float` — 三法估计(AR(1)/IC decay/quantile spread),取中位数
  - `mrp(returns, regime_series) -> float` — Minimum Regime Performance
  - `decay_warning(factor, prices) -> dict` — 自动预警(rolling IC 跌破阈值 / half-life < rebalance周期)
- **接入**: signal_tracker 每日 log 时自动计算衰减;autoresearch 对半衰期过短因子降权

---

### P1 — 排序模型与深度模型

#### 2.5 Listwise 排序损失模型
- **位置**: 扩展 `xsec/xsec_models.py`
- **新增排序器**(均继承 FactorModel,统一 predict 接口):
  - `ListNetModel` — ListNet listwise 损失(softmax top-1 概率交叉熵)
  - `ListMLEModel` — ListMLE(Plackett-Luce 似然损失)
  - `LSListModel` — 长短仓 listwise loss(arxiv 2104.12484,shift-invariant)
  - `LambdaMARTModel` — LightGBM LambdaMART(pairwise + NDCG)
- **接入**: xsec_autoresearch 模型空间增加 listwise;`ml_factor_backtest` 支持 `loss="listnet"/"listmle"/"ls_list"`

#### 2.6 深度选股模型 — skill 内训练管线
- **设计理念**: 训练代码全部在 skill 内。按"线上能否训练得动"分两条路径:
  - **路径 A 线上训练**: 轻量模型(LSTM/GRU/小Transformer/ListwiseTransformer)线上直接 walk-forward 每折 fit,与 Ridge/LGBM 同流程
  - **路径 B 本地 GPU 训练 → 线上推理**: 重型模型(MASTER/Stockformer/AlphaPortfolio)线上训练不动,agent 在本地 GPU 服务器启动训练,模型文件持久化后传线上加载推理
- **划分依据**: 线上单折训练超过 ~10 分钟或显存不够 → 路径 B;轻量模型线上跑得动就直接 walk-forward
- **位置**: 新增 `research/seq_models.py`(模型定义+训练管线) + `scripts/train_seq_model.py`(路径B训练入口) + 扩展 `research/models.py`(序列接口)

**2.6a 序列 FactorModel 接口**
- `SequenceFactorModel(FactorModel)` — 输入 `(n, seq_len, n_features)` 3D 序列,与表格 FactorModel 共享 predict/rank 接口
- `build_sequence_panel(data, *, seq_len=60, fields=["open","high","low","close","volume","amount"])` — 对标 Qlib Alpha360,含 RobustZScoreNorm
- `ml_seq_backtest(data, model, *, seq_len, horizon, ...)` — 序列版 walk-forward,与 ml_factor_backtest 平行,同样遵守 purge + IC 评估 + conformal

**2.6b 模型实现**

| 路径 | 模型 | 论文 | 典型超参 | 线上单折 |
|---|---|---|---|---|
| A 线上 | LSTMModel | Qlib GRU baseline | hidden=64, 2层, epochs=40 | ~2-5分钟 |
| A 线上 | GRUModel | Qlib GRU baseline | hidden=64, 2层, epochs=40 | ~2-5分钟 |
| A 线上 | TransformerModel | Qlib Transformer | d_model=64, 2层, nhead=2 | ~3-8分钟 |
| A 线上 | ListwiseTransformerModel | ListNet+2510.14156 | Transformer + listwise 损失 | ~3-8分钟 |
| B 本地GPU | MASTERModel | MASTER (AAAI-2024) | d_model=64, gating+intra/inter, epochs=60 | ~15-30分钟 |
| B 本地GPU | StockformerModel | Stockformer (2024) | 小波分解+多任务+图嵌入 | ~20-40分钟 |
| B 本地GPU | AlphaPortfolioModel | AlphaPortfolio (NBER) | Transformer+RL+跨资产注意力 | ~60+分钟 |

- `needs_local_gpu(model_name, data_size) -> bool` — 自动判定路径(可 override)
- 均支持 `loss="mse"/"listnet"/"listmle"/"ls_list"`、early stopping、weight decay、dropout
- GPU 优先: `torch.cuda.is_available()` → GPU;无 GPU → CPU 回退 + 警告

**2.6c 路径 B 训练入口**
- `scripts/train_seq_model.py` — agent 在本地 GPU 执行
- CLI: `python scripts/train_seq_model.py --model master --data data/csi300.pkl --device cuda --output models/master.pt`
- 支持: 单模型训练 / walk-forward(每折独立保存) / 超参搜索
- 输出: `model.pt`(state_dict) + `config.json`(超参+数据范围+预处理统计量+git hash) + `metrics.json`(IC/loss曲线)
- agent 工作流: 本地训练 → 模型文件传线上 → `load_trained_seq_model(path)` 加载推理

**2.6d 训练基础设施**(`research/seq_models.py`)
- `SequenceDataset` / `train_loop`(支持 mse/listnet/listmle/ls_list) / `device_selection` / `save_model` / `load_model`
- `cross_sectional_normalize`(CSZScoreNorm) / `drop_extreme_labels`(DropExtremeLabel)
- `wavelet_decompose`(Stockformer) / `stock_graph_embed`(Stockformer)

**2.6e 线上推理接入**
- `load_trained_seq_model(path) -> SequenceFactorModel` — 加载路径B模型
- `trained_seq_factor(model_path, data, *, dates) -> DataFrame` — 截面预测面板
- xsec_autoresearch: 路径A模型在线上 fit;路径B模型 load 后推理
- ensemble_top 可混合表格模型 + 路径A序列模型 + 路径B序列模型
- `load_external_scores_panel` 保留作为旧外部模型兼容入口

**2.6f 诚实性约束**
- 两条路径均遵守 purge + walk-forward + IC 评估 + conformal + validation.py DSR 校正
- 预处理统计量随 model 文件持久化,测试集用训练集统计量
- 序列模型试验数计入 n_trials(Deflated Sharpe)
- **基线准入门槛(核验后新增)**: 任何序列/深度模型必须在**同一张记分卡**上
  (同 universe、同 horizon、同成本)打赢 ridge 基线的 RankICIR,且过 DSR 校正,
  才允许进入 ensemble/报告——这是 v2 路线图核心判断("别追更大的预测器",
  FINSABER/StockBench 证据)在 v3 的延续:深度模型是**待检验的候选**,不是默认升级。

**2.6g 运行环境现实约束(核验后新增)**
- Cowork 沙盒单次 bash 上限 ~45s 且后台进程不保活:路径 A "线上单折 2-8 分钟"
  **不能一把跑完**,训练循环必须原生支持 **checkpoint 断点续跑**(每 epoch 落盘
  state + 已完成折清单,重入自动续),这是 `train_loop` 的硬需求而非可选项;
- 或者干脆把路径 A 也放到本地跑(`train_seq_model.py` 同时服务两条路径),
  线上只做加载推理——实现时按实际环境二选一,默认后者更稳。

---

### P2 — 横截面评估与选池

#### 2.7 横截面评估增强:行业/风格中性 + 截面标准化
- **位置**: 扩展 `xsec/xsec_eval.py` + `xsec/panel.py`
- **新增**:
  - `cs_zscore_normalize(panel, *, groupby=None) -> DataFrame` — 截面 z-score(可选按行业 groupby,对标 Qlib CSZScoreNorm)
  - `industry_neutral_evaluate(data, factor, *, sector_map, horizon=21) -> dict` — 行业中性 IC/RankIC
  - `style_neutral_evaluate(data, factor, *, style_factors=["size","value","momentum"], horizon=21) -> dict` — 风格中性纯选股 IC
  - `purged_label(prices, horizon, *, embargo=5) -> Series` — 标签清洗 + embargo
- **接入**: `evaluate_cross_section` 增加 `neutralize="industry"/"style"/"both"`;报告区分原始 IC vs 中性化 IC

#### 2.8 选池增强:动态 universe 与反幸存者
- **位置**: 扩展 `xsec/universe.py`
- **新增**:
  - `dynamic_universe(meta, prices, *, date, cap_min, adv_min, rebalance="ME") -> list` — 按日期动态选池(只用该日期已上市且满足流动性条件的标的)
  - `sector_rotation_universe(meta, prices, *, sector_scores, per_sector=10) -> list` — 行业轮动选池
  - `anti_survivorship_pool(meta, *, asof_date) -> list` — 反幸存者偏差(剔除 asof_date 后上市/退市的标的)
- **接入**: xsec_autoresearch 支持动态选池回测(每 rebalance 日重新选池);PIT 模式下与 pit.py 联动

---

### P3 — 研究循环与组合优化

#### 2.9 autoresearch 知识累积与跨市场迁移
- **位置**: 深化 `research/autoresearch.py` + `xsec/xsec_autoresearch.py`
- **现状**: 已有 UCB1 bandit + co-opt + holdout,但无知识累积、无跨市场迁移

**2.9a 研究经验记忆(对标 RD-Agent Co-STEER 知识库 + FactorMiner 经验记忆)**
> 核验注: "知识森林"并非 RD-Agent(Q) 摘要用词(其代码生成 agent Co-STEER 自带知识库);
> FactorMiner 的结构是"模块化技能架构 + 一个经验记忆(内分成功模式/禁区)"。
> 类名改为 `ResearchMemory` 更贴近来源,机制不变。
- `ResearchMemory` 类 — 持久化到 JSONL: `{hypothesis, factors_used, model_used, regime, oos_sharpe, timestamp}`
  - 技能记忆 = 可复用因子结构模板(如 "rank+ts_delta" 在趋势市场有效)
  - 经验记忆 = regime→因子成功率映射(如 "drift regime 下 value+reversal 成功率 0.8")
- `propose_hypothesis(memory, regime) -> Hypothesis` — 从记忆采样,而非当前随机/全组合
- `update_memory(memory, trial_result)` — 结果回写(成功模式与禁区分开存,FactorMiner 式)
- 跨 session 持久化;可禁用回退无状态模式

**2.9b 跨市场迁移(对标 arxiv 2505.03659, 2504.09664)**
- skill 已有多市场单市场闭环(US/CN/HK/KR/JP,market_of/index_for/index_regime),但无跨市场迁移
- `transfer_model_params(source_model, target_data, *, n_finetune_epochs=10) -> Model` — 源市场模型参数作为目标市场微调起点
- `cluster_strategies(trial_returns_df, *, n_clusters=5) -> dict` — 聚类 trial 收益曲线,识别低相关高表现策略族
- `warm_start_search(target_data, source_report, *, n_iterations=20) -> Report` — 用源市场 best trial 参数作为目标市场搜索起点
- 接入: `xsec_autoresearch.search(target_data, *, transfer_from=source_report)`

**2.9c 因子-模型协同深化**
- 当前 co-opt 已有,深化为因子空间与模型空间的联合 bandit(含 listwise 模型 2.5 + 序列模型 2.6)

#### 2.10 组合优化模型族
- **位置**: 新增 `risk/optimization.py`
- **新增**:
  - `min_variance_weights(cov) -> Series` — 最小方差
  - `max_sharpe_weights(cov, mu, rf=0.0) -> Series` — 最大夏普(mu 从 ML 预测收益取)
  - `efficient_frontier(cov, mu, n_points=50) -> DataFrame` — 有效前沿
  - `black_litterman(cov, market_w, views_P, views_Q, tau=0.025) -> dict` — BL(市场先验 + 选股 views 融合)
  - `mean_variance_constrained(cov, mu, *, long_only=True, weight_cap=0.1, sector_caps=None) -> Series` — 带约束 MVO
  - `views_from_llm(predictions, confidence) -> (P, Q)` — LLM 生成 BL views 的 prompt 模板+解析(arxiv 2504.14345,LLM 调用由 agent 侧;原文核心提醒:LLM 的选择本质是选**投资风格**,效果取决于风格与当前 regime 的匹配,而非"哪个模型预测最准"——views 置信度应随 regime 匹配度调低)
- **接入**: `multi_factor_signal(..., sizing="black_litterman"/"min_var"/"max_sharpe")`;HTML 报告有效前沿图

#### 2.11 因子 tearsheet
- **位置**: 新增 `reporting/factor_tearsheet.py`
- **新增**: `factor_tearsheet(factor, prices, *, quantiles=5, horizons=(1,5,21,63), neutralize=None) -> HTML`
  - 分位累计收益曲线 / IC 时序 + 衰减曲线 / 多空价差 / 换手率 / 分位统计表
  - 可选行业/风格中性版本
- **接入**: factor_lab.scorecard 自动生成 tearsheet;xsec_autoresearch 对 top 因子自动生成

---

### P4 — 论文驱动的深化项

#### 2.12 LLM 因子相关性推理 [对标 Alpha-R1, arxiv 2512.23515]
- **位置**: 新增 `research/llm_factor_reasoning.py` + `prompts/factor_reasoning.txt`
- **LLM 边界**: skill 侧提供 prompt 模板 + 结果解析;LLM 调用由 agent(Claude)侧执行
- **新增**:
  - `factor_relevance_reasoning(factor_logic, market_context, news_items) -> dict` — 返回 `{relevance_score, activation: bool, reasoning}`
  - `context_aware_factor_gate(factors_dict, market_context, news_items) -> dict` — 批量激活/停用决策
  - `economic_logic_check(factor_expr_or_callable) -> str` — LLM 提取因子经济直觉描述
- **接入**: autoresearch 每轮搜索前筛选因子池;decay_monitor 触发时判断"衰减是真失效还是 regime 切换"
- **数据依赖**: data/news.py 已有基础,提供新闻输入

#### 2.13 Drift Regime 因子激活门 [对标 arxiv 2511.12490, 13-Sharpe]
- **位置**: 扩展 `risk/regime.py`
- **现状**: regime.py 有 vol_regime/trend_regime(市场级),无 per-stock regime
- **新增**:
  - `stock_drift_regime(close, *, window=63, threshold=0.6) -> Series` — per-stock drift regime(63日>60%上涨日)
  - `drift_regime_gate(factor, close, *, activate_in="drift"/"non_drift") -> Series` — binary regime gate
  - `regime_conditional_factor_combo(factors_dict, close, *, regime_gates) -> Series` — 多因子 + 各自 regime gate
- **接入**: xsec_eval 增加 `regime_gate="drift"`;multi_factor_signal 支持每因子配 regime gate
- **注意(核验后加强)**: 该论文为**单一作者、无同行评审**的预印本,自报 S&P500 年化 158%、
  Sharpe 13、回撤 -11.9%——远超顶级对冲基金长期水平(<3),大概率含前视/幸存者/过拟合。
  **只借"regime 条件化激活"这一思路**,任何阈值(63日/60%)都须走 walk-forward 敏感性
  测试 + validation.py 全套体检;其数值结果不得出现在任何报告或预期里。
- **与现有能力衔接**: Round 9 已有**集成层**的 regime 条件化(`regime_conditional_weights`,
  按 vol×牛熊加权集成成员);2.13 是**因子层**的 per-stock gate,二者互补不重复——先复用
  现有 vol/trend 状态机,drift 状态只是新增一种状态定义。

#### 2.14 AlphaAgent 三重正则化因子挖掘 [对标 arxiv 2502.16789]
- **位置**: 深化 `research/factor_lab.py` + `research/autoresearch.py`
- **纯代码部分**(skill 内):
  - `complexity_control(factor_expr, *, max_depth=5, max_params=3) -> bool` — 符号表达式树深度 + 参数计数
  - `novelty_check(factor_expr, existing_library, *, method="ast") -> float` — AST 相似度 vs 已有因子库(0=新颖,1=重复)
- **LLM 辅助部分**(agent 侧,skill 提供 prompt 模板):
  - `hypothesis_alignment(factor_logic, market_hypothesis) -> float` — 语义一致性(0-1)
- **整合**: `regularized_factor_proposal(hypothesis, library, *, complexity=True, novelty=True, alignment=True) -> Factor`
- **接入**: autoresearch 因子生成环节强制纯代码正则化;LLM 对齐作为可选增强

#### 2.15 FactorMAD 多 Agent 辩论因子评估 [对标 FactorMAD, ICAIF 2025, DOI 10.1145/3768292.3770377]
- **核验注**: 原文是多 Agent 辩论式**因子挖掘**框架(两个 LLM agent 轮流提出/评审并迭代改进因子),
  "评估"只是其辩论循环的内部环节——本项只借其**评审辩论**一环,不引入其挖掘循环。
- **与 v2 决策的关系**: v2(2026-06-25)明确不做"多智能体辩论式因子矿工",否决的是
  **N× API 成本的自动挖掘循环**;本项是单次会话内由 agent(Claude)扮演多视角做一轮
  语义综合,成本≈一次推理,不违背该决策——但仍列为**可选增强、最低优先**,纯代码指标
  (2.3/2.4/2.16)永远是主判定,辩论输出只作附注。
- **位置**: 扩展 `research/factor_lab.py` + `prompts/factor_debate.txt`
- **LLM 边界**: 纯代码部分(统计/拥挤/衰减)已被 2.3/2.4 覆盖;辩论是这些指标的"语义综合",由 agent 侧 LLM 执行
- **新增**:
  - `multi_agent_factor_debate(factor, prices, *, agents=["statistician","economist","crowding_analyst","decay_analyst"]) -> dict` — 多视角辩论 prompt 模板 + 结果解析
  - skill 侧汇总 2.3/2.4 的纯代码指标 → 生成辩论输入 → agent 调 LLM → skill 解析返回综合评分
- **接入**: scorecard 增加 `debate=True`;HTML 报告因子块显示辩论摘要

#### 2.16 AlphaEval 五维因子评估 [对标 arxiv 2508.13174]
- **位置**: 扩展 `research/factor_lab.py` 的 scorecard
- **现状**: scorecard 已有 ic/ic_ir/autocorr/coverage/max_corr 五维雏形,2.16 是其升级
- **新增**: `alpha_eval(factor, prices, *, n_trials=20) -> dict`(原文为**五维**,此前漏了预测力)
  - **预测力**(纯代码): IC/RankIC 口径(现有 scorecard 已覆盖,并入统一输出)
  - **鲁棒性**(纯代码): 参数变 ±30% 下 IC 稳定性
  - **多样性**(纯代码): vs 已有因子库的 IC 相关性(低=多样)
  - **稳定性**(纯代码): 跨时间段 IC 一致性(rolling IC 方差)
  - **可解释性/金融逻辑**(LLM 辅助,agent 侧): LLM 评估因子经济逻辑清晰度
- **接入**: scorecard 自动附带五维雷达图;autoresearch 用五维评分排序候选

---

### P5 — 评价体系补全(教材 Ch10)

#### 2.17 MAE/MFE 交易行为分析
- **位置**: `core/metrics.py`
- **新增**: `mae_mfe(trades_df, prices_df) -> DataFrame` — 每笔持仓期最大浮亏/浮盈
- **接入**: build_research 自动附带 MAE/MFE 散点图

#### 2.18 Kelly 仓位
- **位置**: `risk/sizing.py`
- **新增**: `kelly_fraction(win_rate, pl_ratio, *, cap=0.5)` + `kelly_weights(returns_df, *, cap=0.5)`
- **接入**: 与 RP/IV/VT 并列;`multi_factor_signal(..., sizing="kelly")`

#### 2.19 CAPM/Alpha/Beta 标准分解
- **位置**: `core/metrics.py`
- **新增**: `capm_decompose(port_returns, market_returns, rf=0.0) -> dict` → `{beta, alpha, corr, r2, treynor, information_ratio}`
- **接入**: summary() 自动附带;HTML 报告 CAPM 区块

#### 2.20 Brinson 归因
- **位置**: `reporting/attribution.py`
- **新增**: `brinson(port_pnl, bench_pnl, port_w, bench_w, sector_returns) -> dict` — 配置/选股/交互三分解
- **接入**: HTML 报告 attribution 块支持 Brinson 子块

---

### P6 — 执行增强

#### 2.21 RL 执行训练与桥接 [对标 arxiv 2510.04952, 2507.06345]
- **核验注**: 2510.04952 的"RL IS 优于 TWAP/VWAP"结果出自 **ABIDES 多场馆模拟器**而非实盘,
  引用时不得写成实证市场结论;2507.06345 是 market/limit 双订单型 RL(摘要无 cancel 动作、
  无"全 LOB 高维"表述,原方案描述已修正);2601.04896 是**原创 DRL 执行方法论文而非综述**,
  "TWAP/VWAP 适用边界"须由我们自己做参数实验给出,论文里没有现成结论。
- **位置**: 扩展 `trade/execution.py` + 新增 `scripts/train_rl_executor.py`
- **现状**: `order_plan` → TWAP/VWAP 静态计划(不自动提交)
- **设计**: 与 2.6 理念一致,RL 执行策略训练代码在 skill 内,本地 GPU 训练后加载推理
- **新增**:
  - `scripts/train_rl_executor.py` — 本地 GPU 训练入口(ABIDES 风格模拟环境)
  - `load_trained_executor(path) -> ExecutionPolicy` — 加载训练好的 RL 执行策略
  - `implementation_shortfall(fills, benchmark_vwap) -> dict` — IS 执行质量指标
  - `execution_quality_report(fills, prices) -> dict` — IS/方差/延迟影响/市场冲击
- **接入**: `order_plan(..., executor="twap"/"vwap"/"rl")`;执行后自动生成 IS 质量报告
- **文档**: `references/execution_guide.md` 增加 TWAP/VWAP 适用边界 + RL 何时更优

---

## 3. 落地路线图

### 阶段 1: P0 因子供给与质量
| 项 | 模块 | 工程量 | 依赖 |
|---|---|---|---|
| 2.1 因子表达式+Alpha101/158 | `research/factor_expr.py` | **大** | 无 |
| 2.2 因子正交化+增量评估 | `research/factor_lab.py` | 中 | 2.1 |
| 2.3 因子拥挤度 | `research/crowding.py` | 中 | 无 |
| 2.4 因子衰减+half-life | `research/decay_monitor.py` | 中 | 无 |

### 阶段 2: P1 排序模型与深度模型
| 项 | 模块 | 工程量 | 依赖 |
|---|---|---|---|
| 2.5 Listwise 排序模型 | `xsec/xsec_models.py` | 中 | torch/lightgbm 可选 |
| 2.6 深度选股模型 skill 内训练 | `research/seq_models.py` + `scripts/train_seq_model.py` | **大** | torch 可选 |

### 阶段 3: P2 横截面评估与选池
| 项 | 模块 | 工程量 | 依赖 |
|---|---|---|---|
| 2.7 行业/风格中性评估 | `xsec/xsec_eval.py` + `panel.py` | 中 | 无 |
| 2.8 动态选池+反幸存者 | `xsec/universe.py` | 中 | pit.py 联动 |

### 阶段 4: P3 研究循环与组合优化
| 项 | 模块 | 工程量 | 依赖 |
|---|---|---|---|
| 2.9 知识累积+跨市场迁移 | `research/autoresearch.py` | **大** | 2.1/2.5/2.6 |
| 2.10 组合优化族 | `risk/optimization.py` | 中 | scipy/cvxpy 可选 |
| 2.11 因子 tearsheet | `reporting/factor_tearsheet.py` | 中 | 2.1 |

### 阶段 5: P4 论文驱动深化
| 项 | 模块 | 工程量 | 依赖 | 论文 |
|---|---|---|---|---|
| 2.12 LLM 因子相关性推理 | `research/llm_factor_reasoning.py` | 中 | LLM(agent侧) | Alpha-R1 |
| 2.13 Drift Regime 因子门 | `risk/regime.py` | 小 | 无 | 2511.12490 |
| 2.14 AlphaAgent 三重正则化 | `research/factor_lab.py` | 中 | 2.1/2.2 | 2502.16789 |
| 2.15 FactorMAD 多 Agent 辩论 | `research/factor_lab.py` | 中 | LLM(agent侧) | ACM 3768292 |
| 2.16 AlphaEval 四维评估 | `research/factor_lab.py` | 中 | 2.4 | 2508.13174 |

### 阶段 6: P5 评价体系补全
| 项 | 模块 | 工程量 | 依赖 |
|---|---|---|---|
| 2.17 MAE/MFE | `core/metrics.py` | 小 | 无 |
| 2.18 Kelly | `risk/sizing.py` | 小 | 无 |
| 2.19 CAPM 分解 | `core/metrics.py` | 小 | 无 |
| 2.20 Brinson | `reporting/attribution.py` | 中 | benchmark 数据 |

### 阶段 7: P6 执行增强
| 项 | 模块 | 工程量 | 依赖 | 论文 |
|---|---|---|---|---|
| 2.21 RL 执行训练与桥接 | `trade/execution.py` + `scripts/train_rl_executor.py` | 中 | 本地 GPU | 2510.04952 |

---

## 4. 不补充项

| 项 | 原因 |
|---|---|
| 事件驱动引擎/MarketRules/订单状态机 | 交易系统赛道,非选股研究核心 |
| 动态策略加载/硬熔断/热启动 | 实盘运维,非研究 |
| 网格/60-40/可转债交易模板 | 交易策略模板,非选股研究 |
| 指标扩展到 50 个(TA-Lib 全集) | 通用技术分析,Alpha158 子集即可 |
| 过拟合校正(DSR/PBO/CPCV/SPA) | `validation.py` 已比教材 PSR 更全 |
| 期权/期货 | 按要求跳过 |
| 真·历史 PIT 数据 | 代码路径已就绪(`pit.py`),需付费数据 |
| 融券/borrow cost | 按用户既定决策延后 |

---

## 5. 兼容性

- **不重写**: validation.py / xsec/ 现有评估 / models.py 现有模型 / pit.py 保持现状
- **并存**: listwise 模型与现有 pointwise 并存,统一 FactorModel 接口;序列模型与表格模型并存
- **渐进增强**: autoresearch 深化而非重写,research_memory 可禁用回退无状态
- **可选依赖**: polars/torch/lightgbm/scipy/cvxpy 缺失时优雅降级
- **LLM 边界**: 纯代码在 skill 内;LLM 辅助由 agent 侧执行,skill 提供 prompt 模板+解析
- **HTML 报告**: 新区块按 SCHEMA.md "给了就渲染" 原则扩展

---

## 附录 A: 教材章节映射

| 教材章 | 方案项 | 状态 |
|---|---|---|
| Ch1 CAPM/EMH | 2.19 CAPM 分解 | 待补 |
| Ch3 数据获取 | 现有 data/loader.py + pit.py | **已具备** |
| Ch9 MPT/资产配置 | 2.10 组合优化族 | 待补 |
| Ch10 MAE/MFE/Kelly/Brinson/VaR | 2.17-2.20 + 现有 portfolio_var_cvar | 部分 |
| Ch11 WFO | 现有 optimize.walk_forward | **已具备** |
| Ch12 ML | 2.5 listwise + 2.6 深度模型 + 现有 models | 部分 |
| Ch13 可视化 | 现有 html_report.py + 2.11 tearsheet | 部分 |
| Ch14 因子表达式/Alpha101/Polars | 2.1 因子表达式引擎 | 待补 |
| Ch16 指标体系 | Alpha158 子集(2.1 内含) | 待补 |

## 附录 B: 业界对标矩阵

| 能力 | Qlib | RD-Agent | MASTER | ListNet | Alphalens | **skill 现状** | **方案目标** |
|---|---|---|---|---|---|---|---|
| 因子表达式 DSL | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.1 |
| Alpha101/158 库 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.1 |
| 因子正交化/增量 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.2 |
| 因子拥挤度 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.3 |
| 因子衰减/half-life | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.4 |
| Listwise 排序 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ 2.5 |
| 深度模型 skill 内训练 | ✅ | ✅ | ✅原生 | ❌ | ❌ | ❌(仅MLP) | ✅ 2.6 |
| 行业/风格中性 | ✅ CSNorm | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ 2.7 |
| 反幸存者选池 | ✅ | ❌ | ❌ | ❌ | ❌ | 部分 | ✅ 2.8 |
| 知识森林+跨市场迁移 | ❌ | ✅森林 | ❌ | ❌ | ❌ | ❌ | ✅ 2.9 |
| 组合优化族 | ❌ | ❌ | ❌ | ❌ | ❌ | RP/IV/VT | ✅ 2.10 |
| 因子 tearsheet | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ 2.11 |
| LLM 因子推理 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.12 |
| Drift regime gate | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.13 |
| 因子三重正则化 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.14 |
| 多 Agent 辩论 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 2.15(可选/最低优先) |
| 五维因子评估 | ❌ | ❌ | ❌ | ❌ | ❌ | 部分(5维雏形) | ✅ 2.16 |
| 过拟合校正 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ DSR/PBO/SPA | **保持** |
| PIT 诚实回测 | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ pit.py | **保持** |
| 多市场支持 | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ US/CN/HK/KR/JP | **保持** |
| 自动研究循环 | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ autoresearch | ✅ 深化 2.9 |

## 附录 C: 文献索引(2026-07-02 逐条访问原文核验)

核验口径: ✅ 相符 = 声称与原文摘要一致;⚠️ 已修正 = 来源真实但原表述失准(正文已改);
🔴 降级 = 来源真实但可信度不足,只借思路不引数值。**24 条全部真实存在,无捏造条目。**

### 基础调研

| 来源 | 关键贡献 | 对应项 | 核验 |
|---|---|---|---|
| Qlib (microsoft/qlib) | Alpha158/360 + 表达式 DSL + CSZScoreNorm + IC/ICIR | 2.1, 2.7 | ✅ |
| RD-Agent(Q) (arxiv 2505.15155, NeurIPS 2025) | Research–Development–Feedback 循环 + Co-STEER 知识库 + 因子-模型协同 + bandit | 2.9 | ⚠️ 已修正("五阶段/知识森林"非摘要用词) |
| MASTER (arxiv 2312.15235, AAAI-2024) | Market-Guided Gating + Intra/Inter-Stock Aggregation | 2.6 | ✅ |
| Stockformer (arxiv 2401.06139) | 小波分解 + 多任务自注意力 + 图嵌入(投稿中,非已录用) | 2.6 | ✅ |
| AlphaPortfolio (NBER w35195, 2026-05 正式版) | Transformer + RL + 跨资产注意力(SSRN 2019 名作的 NBER 版) | 2.6 | ✅ |
| ListNet (Cao et al., ICML 2007 / MSR-TR-2007-40) | listwise 排序损失 + top-1 概率交叉熵 | 2.5 | ✅ |
| 长短仓 listwise (arxiv 2104.12484) | 长短仓 shift-invariant listwise loss;"LS-List"为本方案自拟简称;38%/Sharpe2 是其 2006-19 A股自报回测,不作为可复现预期 | 2.5 | ⚠️ 已修正 |
| factor-decay-lab (github amit943c) | 6 horizon IC + half-life 三法 + decay warning(个人项目,1 star,仅作工程参考,不引其数字) | 2.4 | ⚠️ 已修正 |
| Man Institute Crowding (man.com/maninstitute/crowding) | 拥挤度 + 持仓重合 + sharp reversal 风险(2.3 的主要方法学依据) | 2.3 | ✅ |
| WorldQuant Alpha101 (Kakushadze) | 101 公式 + 算子集 | 2.1 | ✅ |
| AKQuant 教材 Ch10/Ch14 | MAE/MFE/Kelly/Brinson + 因子表达式/Alpha101 | 2.17-2.20, 2.1 | ✅ |

### 2025-2026 论文

| 论文 | 年份 | 关键贡献 | 对应项 | 核验 |
|---|---|---|---|---|
| AlphaAgent (arxiv 2502.16789) | 2025 | 三重正则化(复杂度/假设对齐/AST新颖性)对抗衰减 | 2.14 | ✅ |
| Alpha-R1 (arxiv 2512.23515) | 2025-12 | 8B RL 推理模型,经济逻辑+新闻→激活/停用(v1 预印本,无同行评审) | 2.12 | ✅ |
| FactorMiner (arxiv 2602.14670) | 2026 | 模块化技能架构 + 经验记忆(成功模式/禁区),Ralph Loop 对抗知识遗忘 | 2.9 | ⚠️ 已修正("双记忆"表述) |
| Drift Regimes (arxiv 2511.12490) | 2025 | regime-conditional 信号激活;自报 Sharpe>13/年化158% 极端反常,单作者无评审 | 2.13 | 🔴 降级(仅借思路,数值不可信) |
| Not All Factors Crowd Equally (arxiv 2512.11913) | 2025 | 双曲衰减 α(t)=K/(1+λt);机械vs判断、拥挤→尾部风险 | 2.3, 2.4 | 🔴 降级(**v2 已撤稿**,实证待大修) |
| AlphaEval (arxiv 2508.13174) | 2025 | **五**维评估(预测力/鲁棒/多样/可解释/稳定),backtest-free 预筛 | 2.16 | ⚠️ 已修正(原写四维) |
| LLM-Enhanced BL (arxiv 2504.14345, CIKM 2025 WS) | 2025 | LLM 预测+不确定性 → BL views/置信度;LLM 选择≈选风格 | 2.10 | ✅ |
| Meta-Learning Optimal Mixture (arxiv 2505.03659) | 2025 | 元学习初始参数 + 聚类低相似高表现策略(场景为在线组合选择) | 2.9 | ✅ |
| Robust Meta-Learning Zero-Shot (arxiv 2504.09664) | 2025 | GMM 软聚类 meta-task + hard task mining 零样本预测(摘要无 alpha360,该词已删) | 2.9 | ⚠️ 已修正 |
| FactorMAD (ICAIF 2025, DOI 10.1145/3768292.3770377) | 2025 | 多 Agent 辩论式**因子挖掘**(内含评审环节;原写"评估"且 DOI 不完整) | 2.15 | ⚠️ 已修正 |
| Safe Cross-Market RL Execution (arxiv 2510.04952) | 2025 | 约束 RL(CMDP+PPO+action shield)跨市场执行;结果出自 ABIDES 模拟 | 2.21 | ⚠️ 已修正(注明模拟) |
| RL Trade Execution (arxiv 2507.06345, Cheridito & Weiss) | 2025 | market/limit 双订单型 RL 执行(摘要无 cancel/全LOB,该表述已删) | 2.21 | ⚠️ 已修正 |
| DRL Optimum Order Execution (arxiv 2601.04896) | 2026 | 原创 DRL 执行方法论文(**非综述**;TWAP/VWAP 边界须自测) | 2.21 | ⚠️ 已修正 |
| Loss Functions for Stock Ranking (arxiv 2510.14156, CIKM 2025) | 2025 | pointwise/pairwise/listwise 损失在 Transformer 选股上的系统评估 | 2.5, 2.6 | ⚠️ 已修正(原标题有误) |

### C.3 高风险来源处理决定

1. **arxiv 2512.11913(已撤稿)**: 双曲衰减函数形式保留为拟合工具;其两条实证结论降级为
   待验证假设。2.3 拥挤度的方法学主依据改为 Man Institute 研究。
2. **arxiv 2511.12490(Sharpe 13)**: 只保留"regime 条件化激活"思路;实现走自有
   walk-forward 敏感性测试 + validation.py 体检,论文数值不进任何报告与预期。
3. **偏弱来源**(factor-decay-lab 个人项目、Alpha-R1/Stockformer 等未评审预印本):
   作为工程参考/思路来源可用,不引用其自报数字,落地一律以本 skill 的诚实评测为准。
