"""Default parameter grids per rule-strategy family -- single source of truth.

Used by autoresearch (as its sampling space) and run_backtest --walk-forward (as
its default grid). Previously duplicated in both files; editing one and missing
the other silently desynced research from the CLI.
"""

PARAM_GRIDS = {
    "ma_crossover": {"fast": [10, 20, 30], "slow": [50, 100, 150]},
    "breakout": {"entry": [20, 40, 55], "exit": [10, 20]},
    "ts_momentum": {"lookback": [60, 90, 120, 180]},
    "macd_trend": {"fast": [8, 12], "slow": [21, 26], "signal": [9]},
    "zscore_reversion": {"lookback": [10, 20, 30], "entry": [1.0, 1.5, 2.0]},
    "bollinger_reversion": {"n": [10, 20, 30], "k": [1.5, 2.0, 2.5]},
    "rsi_reversion": {"n": [7, 14, 21], "oversold": [20, 30]},
}


# One-line intro per strategy family — single source of truth for the report's
# "策略测试选择" glossary, the CLI, and any UI. Keys align with PARAM_GRIDS / REGISTRY
# so every tested strategy can be explained automatically (no hard-coded copy elsewhere).
STRATEGY_INFO = {
    "ma_crossover": {"name": "均线交叉 (MA Crossover)", "kind": "趋势",
        "intro": "快线上穿慢线做多、下穿离场——最经典的趋势跟随,顺势而为。",
        "edge": "适合单边趋势;震荡市易被反复打脸(可加 ADX 趋势过滤)。"},
    "breakout": {"name": "唐奇安通道突破 (Breakout)", "kind": "趋势",
        "intro": "创 N 日新高入场、跌破 M 日新低离场(海龟法则)——吃趋势的肥尾。",
        "edge": "适合突破启动;趋势之间长时间空仓,假突破是主要损耗。"},
    "ts_momentum": {"name": "时序动量 (TSMOM)", "kind": "趋势",
        "intro": "过去 N 日收益为正则持多、否则离场(Moskowitz 2012)——对自身历史回报做趋势。",
        "edge": "跨市场稳健、规则极简;拐点附近滞后。"},
    "macd_trend": {"name": "MACD 趋势", "kind": "趋势",
        "intro": "MACD 柱(快慢 EMA 之差减信号线)为正时持多——动量加速时在场。",
        "edge": "对动量变化敏感;频繁穿越会抬高换手与噪声。"},
    "zscore_reversion": {"name": "Z-Score 均值回归", "kind": "均值回归",
        "intro": "价格偏离均值 z>+入场 做空、z<-入场 做多,回归即离场——赌过度拉伸回弹。",
        "edge": "适合区间震荡;趋势行情'越拉越远'是主要风险。"},
    "bollinger_reversion": {"name": "布林带回归", "kind": "均值回归",
        "intro": "触及/跌破下轨买、回到中轨卖(做空镜像)——用波动带刻画'便宜/贵'。",
        "edge": "高胜率小盈亏;趋势突破带外时会连续亏。"},
    "rsi_reversion": {"name": "RSI 超卖回归", "kind": "均值回归",
        "intro": "RSI<超卖买入、回到中位离场——经典摆动指标抄底。",
        "edge": "震荡市好用;强趋势里 RSI 会长期钝化。"},
    "pairs_trading": {"name": "配对/价差套利 (Pairs)", "kind": "统计套利",
        "intro": "对两只协整标的的价差做 z-score 回归,价差拉大反向、收敛了结——市场中性。",
        "edge": "对冲掉市场 beta、赚相对价值;协整一旦破裂风险大,需先做协整检验。"},
}

