# AI 选股(横截面排序)— 方法与诚实红线

把"预测哪只股票相对更强"做成可评测的能力。与单标的择时**正交、可独立、可协同**:
横截面排序回答**买哪些**,既有的 `backtest/levels/sizing/regime/report` 回答**何时/多少**。

## 三种标的池来源(`universe.py`)
- **list**:显式给代码清单(最可控)。
- **csv**:从文件读清单(可带 sector 列)。
- **index/sector**:A股/港股用 akshare 拉成分;美股用成分快照 CSV。
默认推荐"你给方向 → 组池给候选 → 确认 → 拉数",`build_universe(spec)` 产出可复现的 `universe.json`。

## 评测口径(`xsec_eval.py`,论文口径)
逐调仓日在整截面上打分,purged 扩窗 walk-forward(标签须已实现):
- **IC**:当日预测 vs 实际前向收益的 Pearson 截面相关;**RankIC**:Spearman(抗极值,**主指标**)。
- **ICIR / RankICIR**:IC 均值 / IC 标准差 → 信号**稳定性**。
- **分位单调性**:按预测分桶,看高分位是否真更强(→1 理想)。
- **Top-K 多空**:做多预测前 K、做空后 K,去重叠按 horizon 再平衡、扣交易成本 → 价差/年化/Sharpe。
- **判定**:业界经验 **RankIC ≳ 0.03 且 RankICIR ≳ 0.3** 才算"可用"。

## 诚实红线(都是踩过的坑)
1. **池子要够大且分散**:< ~30 只或单一板块 → 直接告警。同业高度同涨同跌,横截面几乎没有可排序结构(实测:9 只 AI-半导体 5 年,连线性 ridge 的 RankIC ≈ 0)。
2. **purged walk-forward**:标签未实现的样本不进训练,杜绝前视。
3. **扣成本**:多空价差扣佣金+滑点;别用毛收益骗自己。
4. **小样本谨慎**:调仓日少时 ICIR 极不稳,看绝对值也看 n_dates。
5. **幸存者偏差**:免费成分多为当前快照,长回测会偏乐观。

## 模型(`xsec_models.py`)
统一 `fit/predict` 接口:`RidgeModel`(必有)/ `LGBMModel` / `MLPModel`(sklearn 非线性)/ 可选 `TorchRanker`。
深度时序模型(RAVEN/LSTM/TS-foundation)在本机或 GPU 训练后,用 `models.load_external_scores` 挂接其打分,再走同一套 `xsec_eval` 评测——对齐论文配方见 `PAPER_CONFIG`。

## 自动选股研究(`xsec_autoresearch.py`)
搜"因子子集 × 排序器",目标函数 = **RankICIR**(或 LS_sharpe),walk-forward,输出排行榜。

## 快速开始
```python
from scripts.xsec import universe, xsec_eval, xsec_autoresearch, xsec_report
from scripts.data import loader
uni = universe.build_universe({"market":"US","source":"list","symbols":[...]})
data = loader.load_many(uni["symbols"])                 # {sym: OHLCV}
res = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME")
print(res["scorecard"]); print(xsec_report.scorecard_markdown(res))
lb  = xsec_autoresearch.search(data, horizon=21)        # 因子×模型 排行榜
```
非投资建议。

---

## 选池方法(数据驱动:市值/流动性/指数成分 基座 + 热门/龙头 软打分)

回答"每个板块的股票怎么选"的可复现方案。实现于 `scripts/xsec/universe.py`
`build_scored_universe(meta, prices, per_sector, cap_min, adv_min, require_index, sectors_override, weights)`。

**两层结构(软打分 soft-tilt,不做硬淘汰,减少幸存者偏差):**

1. **硬基座门槛(定"能不能进池"):**
   - 市值 ≥ `cap_min`(默认 2e9,即 20 亿美元)—— 真实 marketCap。
   - 美元日均成交额 ADV$ ≥ `adv_min`(默认 2e7,即 2000 万美元)—— price×avgVolume。
   - (可选)指数成分 `require_index`(如 'SP500'/'NDX')—— 真实成分表。
   门槛只决定候选资格,不参与排序权重之外的打分。

2. **软打分(定"板块内谁靠前",在每个板块内取 TopN=per_sector):**
   综合分 = wS·size_z + wL·liq_z + wH·hot_z + wR·hi52_z + wD·lead_z(默认权重 0.20/0.15/0.30/0.15/0.20,全部为截面 z 分,报告展示各分量):
   - **规模 size_z** = z(log10 市值)——大盘更稳、更具代表性。
   - **流动性 liq_z** = z(log10 ADV$)——可交易性。
   - **热门度 hot_z** = z(近 63/126 日动量均值)——"最近热门"。缺价时回退 price/avg200−1。
   - **52周高贴近度 hi52_z** = z(price/52周高)——强势/龙头常态贴近新高。
   - **龙头度 lead_z** = z(板块内相对强弱=长动量−板块中位)×0.6 + size_z×0.4——"板块龙头=又大又强"。

**数据来源(强制真实):** 行情连接器(FMP 系)`quote/batch-quote`(price、marketCap、volume、yearHigh、priceAvg50/200)、
`company/batch-market-cap`、`indexes/sp-500`+`nasdaq`(成分)、`company/profile-symbol`(sector/averageVolume)。
`meta_from_fmp(quotes, profiles, index_members)` 把连接器原始返回解析成 meta。板块用 `sectors_override` 传主题板块
(如 AI基建/核电电力),覆盖连接器的 GICS 行业。

**反后视镜/point-in-time 提示:** 默认用"当前快照"的市值/成分/52周高,适合"今天选池"。做严格历史回测时,
应改用 `indexes/historical-sp-500` 与 `company/historical-market-cap` 取调仓日的历史成分/市值,避免用未来信息选池。
软打分里的动量/52周高本身基于截至打分日的价格,不含未来数据;但"用今天的热门名单去解释过去"仍是幸存者偏差,报告须注明。

**护栏:** 每板块过门槛不足 per_sector 时告警并全取;总数 < 30 时告警(横截面统计意义弱)。
