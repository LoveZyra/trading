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
from scripts import universe, xsec_eval, xsec_autoresearch, xsec_report
from scripts.data import loader
uni = universe.build_universe({"market":"US","source":"list","symbols":[...]})
data = loader.load_many(uni["symbols"])                 # {sym: OHLCV}
res = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME")
print(res["scorecard"]); print(xsec_report.scorecard_markdown(res))
lb  = xsec_autoresearch.search(data, horizon=21)        # 因子×模型 排行榜
```
非投资建议。
