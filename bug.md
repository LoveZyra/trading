# alpha-forge 代码审查报告：问题与优化方案

> 审查日期：2026-06-10
> 范围：`alpha-forge/scripts/` 全部模块（核心引擎、数据层、策略层、工具与报告）
> 现状：`tests/` 下 32 个测试用例全部通过；以下问题均经人工验证确认，已过滤误报

---

## 一、确认的真实 Bug（高优先级，建议立即修复）

### BUG-1: RSIReversion 做空模式下多头信号被完全覆盖

- **文件**: `alpha-forge/scripts/strategies/mean_reversion.py:64-72`
- **严重程度**: 高（直接导致策略行为错误）
- **问题描述**:

  ```python
  pos[r <= self.oversold] = 1.0        # r<=30 做多
  pos[r >= self.exit_level] = 0.0      # r>=50 平仓
  if self.allow_short:
      pos[r >= self.overbought] = -1.0  # r>=70 做空
      pos[r <= self.exit_level] = 0.0   # r<=50 全部清零 ← BUG
  ```

  `allow_short=True` 时，最后一行把所有 `r <= exit_level(50)` 的 bar 置 0，
  其中包括 `r <= oversold(30)` 的多头入场 bar——多头信号刚生成就被抹掉，
  策略实际退化成**只能做空**。
- **优化方案**: 空头的退出条件应该是"RSI 从超买区回落"，用区间条件代替全量覆盖：

  ```python
  pos[r <= self.oversold] = 1.0
  pos[(r >= self.exit_level) & (r < self.overbought)] = 0.0
  if self.allow_short:
      pos[r >= self.overbought] = -1.0
      # 空头回补：RSI 回落到 exit_level 之下、但还没到 oversold（那里是多头入场）
      pos[(r <= self.exit_level) & (r > self.oversold)] = 0.0
  ```

  修复后需补回归测试：构造 RSI 序列依次穿越 20→55→75→45→25，
  断言多头/空头入场与退出都按预期发生。

### BUG-2: pair_spread 对冲比除零产生 inf

- **文件**: `alpha-forge/scripts/strategies/mean_reversion.py:82`
- **严重程度**: 高（A 股停牌/涨跌停场景常见）
- **问题描述**:

  ```python
  hedge = (a.rolling(lookback).cov(b) / b.rolling(lookback).var())
  ```

  当 B 腿在 lookback 窗口内价格不变（停牌、连续涨跌停）时 `var()==0`，
  hedge 变为 `inf`，污染整条 spread 与后续 z-score，PairsTrading 信号失效。
- **优化方案**:

  ```python
  b_var = b.rolling(lookback).var().replace(0, np.nan)
  hedge = a.rolling(lookback).cov(b) / b_var
  ```

  NaN 的 hedge 会让 spread 在该窗口为 NaN，z-score 自然为 NaN、不发信号——
  比 inf 的行为安全。

### BUG-3: profit_factor 返回 inf 污染下游统计

- **文件**: `alpha-forge/scripts/metrics.py:85-88`
- **严重程度**: 高（影响 autoresearch / walk-forward 聚合）
- **问题描述**:

  ```python
  return float(gains / losses) if losses else np.inf
  ```

  短窗口（walk-forward 小分段、autoresearch 采样）出现"零亏损"时返回 `inf`，
  进入报告均值/排序后产生 NaN 传播或排序异常。
- **优化方案**: 区分边界情形并避免 inf 进入聚合：

  ```python
  def profit_factor(returns: pd.Series) -> float:
      gains = returns[returns > 0].sum()
      losses = -returns[returns < 0].sum()
      if losses <= 0:
          return float("nan") if gains <= 0 else float("inf")  # 或封顶如 100.0
      return float(gains / losses)
  ```

  若下游需要数值排序，建议封顶（如 `min(pf, 100.0)`）。

### BUG-4: html_report.py JSON 注入转义不完整

- **文件**: `alpha-forge/scripts/html_report.py:854`
- **严重程度**: 高（潜在 HTML 注入 / 页面破坏）
- **问题描述**:

  ```python
  data = _json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
  ```

  只转义了 `<`，未转义 `>`、`&`、U+2028/U+2029。报告内容若包含
  `</script>` 变体或特殊字符可能破坏内嵌脚本块。
- **优化方案**:

  ```python
  data = (_json.dumps(report, ensure_ascii=False)
          .replace("<", "\\u003c").replace(">", "\\u003e")
          .replace("&", "\\u0026")
          .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
  ```

### BUG-5: signal_tracker.py 文件句柄未关闭

- **文件**: `alpha-forge/scripts/signal_tracker.py:32, 39`
- **严重程度**: 高（Windows 下文件锁可能导致后续写入失败）
- **问题描述**:

  ```python
  return sum(1 for _ in open(path, encoding="utf-8"))          # L32
  rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]  # L39
  ```

  两处 `open()` 未用 `with`，句柄依赖 GC 释放；Windows 上句柄未释放时
  同一日志文件的追加写（`log_signals`）可能被锁。
- **优化方案**:

  ```python
  with open(path, encoding="utf-8") as f:
      return sum(1 for _ in f)

  with open(path, encoding="utf-8") as f:
      rows = [json.loads(l) for l in f if l.strip()]
  ```

---

## 二、值得做的改进（中优先级）

### OPT-1: vol_regime 是 O(n²) 循环

- **文件**: `alpha-forge/scripts/regime.py:41-46`
- **问题**: 每根 bar 对全部历史重新 `dropna()+quantile()`，10 年日线尚可，
  但被 autoresearch 反复调用时成为热点。
- **方案**: 用 expanding 分位数向量化（每个分位一列），再 `searchsorted` 比较：

  ```python
  thr_cols = [rv.expanding(min_periods=min_history).quantile(q) for q in qs]
  # state[t] = sum(rv[t] > thr_q[t] for q in qs)，纯向量化比较
  state = sum((rv > th).astype(float) for th in thr_cols)
  ```

### OPT-2: multi_factor 用 iterrows 逐行加权

- **文件**: `alpha-forge/scripts/strategies/multi_factor.py:53` 附近
- **问题**: `scores.iterrows()` 在大宇宙 + 长历史时性能差。
- **方案**: 向量化为 `scores.rank(axis=1, pct=True)` + 布尔掩码生成多空桶，
  再按行归一化，可整体消除 Python 级循环。

### OPT-3: 数据层时区约定不一致

- **文件**: `alpha-forge/scripts/data/base.py:53-55` vs `data/sentiment.py:224-243`
- **问题**: OHLCV 被强制 tz-naive，而 `time_weighted_sentiment` 用 UTC-aware；
  新闻时间戳跨源（yfinance Unix 秒 / akshare 字符串）解析后混用，
  A 股 + 美股混合组合的新闻时间衰减权重可能算错。
- **方案**: 全项目统一约定（建议内部全 UTC、展示层再本地化），
  在 `base.py` 模块 docstring 写明，并在新闻入口统一
  `pd.to_datetime(..., utc=True, errors="coerce")` + NaT 计数告警。

### OPT-4: loader 缓存健壮性不足

- **文件**: `alpha-forge/scripts/data/loader.py:39-44`
- **问题**: ① 缓存 parquet 损坏时 `read_parquet` 抛异常，没有"删除坏缓存并重新下载"
  的兜底；② `source=="ibkr"` 路径完全绕过缓存。
- **方案**: 缓存读取包 try/except，失败时 `cache.unlink()` 后走下载分支；
  ibkr 路径纳入同一缓存逻辑。

### OPT-5: akshare 财务数据假设第一行是最新报告期

- **文件**: `alpha-forge/scripts/data/fundamentals.py:128`
- **问题**: `row = fin.iloc[0]` 未排序就取首行，API 返回顺序变化时
  会静默取到过期财务数据。
- **方案**: 取数前按报告期/公告日期列显式 `sort_values(ascending=False)`。

### OPT-6: backtest() 缺输入校验，sqrt 成本模型静默退化

- **文件**: `alpha-forge/scripts/backtest.py:43-96`
- **问题**: `lag<0`、负 bps、未知 `cost_model` 字符串都被静默接受；
  `cost_model="sqrt"` 但缺 volume 列时静默退化为 linear，用户无感知。
- **方案**: 函数入口校验 `lag>=0`、costs 非负、`cost_model in {"linear","sqrt"}`、
  `periods_per_year>0`；sqrt 退化时 `warnings.warn(...)`。

### OPT-7: backtest_portfolio 能力与单资产引擎不对齐

- **文件**: `alpha-forge/scripts/backtest.py:122-162`
- **问题**: 不支持 sqrt 市场冲击成本，也不输出 trade ledger（trades 恒为空表）。
- **方案**: 复用单资产的成本逻辑按列计算 participation（需 volume 面板），
  并按 rebalance 日期生成简化 ledger。

### OPT-8: RULE_SPACE 与 run_backtest.py 的 grids 重复定义

- **文件**: `alpha-forge/scripts/autoresearch.py:74-82` vs `scripts/run_backtest.py:94-102`
- **问题**: 同一策略参数网格两处维护，改一处易漏另一处。
- **方案**: 抽到共享模块（如 `scripts/param_grids.py`），两处 import。

### OPT-9: 大量 `except Exception: pass` 静默吞错

- **文件**: `autoresearch.py`、`data/loader.py`、`data/fundamentals.py` 等多处
- **问题**: 数据源失败、回测异常被静默吞掉，问题难以追踪。
- **方案**: 最低限度加 `logging.warning(..., exc_info=True)`；
  能区分的地方捕获具体异常类型（ValueError/KeyError 等）。

---

## 三、低优先级 / 风格类

| # | 文件:行号 | 问题 | 方案 |
|---|-----------|------|------|
| L-1 | `metrics.py` 全文件 | 统一用 `ddof=0`，与 quantstats/pyfolio 的 `ddof=1` 惯例不同，小样本时 Sharpe 略偏高 | 统一为 `ddof=1` 或在 docstring 注明约定 |
| L-2 | `report.py:59` | 用 `v != v` 判 NaN，可读性差 | 改 `math.isnan(v)` |
| L-3 | `requirements.txt` | 依赖只有下界（`pandas>=2.0` 等），大版本升级可能破坏兼容 | 加上界，如 `pandas>=2.0,<3.0` |
| L-4 | `data/base.py:62-70` | `validate_ohlcv` 不检查 `high>=low` 等 OHLC 逻辑关系 | 加一致性检查并剔除/告警异常行 |
| L-5 | `data/microstructure.py:32-40` | `limit_blocked` 只看收盘价，不看开盘/盘中触板 | 用 high/low 对比上下限价判断 |
| L-6 | `indicators.py:37` | RSI 在完全无波动（gain=loss=0）时输出 100 而非中性 50 | 该情形 `fillna(50.0)` |
| L-7 | `levels.py:73-96` | 重复的 `np.isfinite()+round()` 模式 | 抽 `_safe_round()` 辅助函数 |
| L-8 | `run_backtest.py:129` | 字符串 `+` 拼接与 f-string 混用 | 统一 f-string |
| L-9 | `data/sectors.py` | `SECTOR_MAP` 硬编码，未知股票全归 "other" | 提供 `register_sector()` / 外部 JSON 加载 |

---

## 四、审查中排除的误报（验证后确认不是问题）

| 报告项 | 排除理由 |
|--------|----------|
| `sizing.py:82` ERC 判别式可能为负 | `c=-vol²/n≤0, a=cov[i,i]≥0` ⇒ 判别式 `b²-4ac = b²+4a·vol²/n` 恒非负，不会 NaN |
| `regime.py:46` searchsorted 越界 | `thr` 只有 `n_states-1` 个元素，返回值天然落在 `0..n_states-1` 合法区间 |
| `autoresearch.py:53` UCB1 `log(0)` | 未拉过的 arm 在循环中优先返回，走到 `log(total)` 时必有 `total>=1` |
| `backtest.py:85` 首日 turnover "重复计费" | 从 0 建仓收一次成本是正确行为，非 bug |
| `optimize.py:122` walk-forward "warm-up 泄漏" | rolling 指标只向后看；在 train+test 上生成信号再切片是标准且无前视的做法 |
| `factor_lab.py:128` factor_ic "前视偏差" | IC 本身就是用未来收益做的样本内诊断指标，`shift(-horizon)+dropna` 是标准算法 |
| `metrics.py:40-45` Sharpe "分子分母不匹配" | rf 为常数时 `std(excess)==std(returns)`，数学上等价；ddof 选择是惯例问题（见 L-1） |

---

## 五、建议执行顺序

1. **第一批（修 Bug + 回归测试）**: BUG-1 → BUG-2 → BUG-3 → BUG-5 → BUG-4
2. **第二批（正确性相关改进）**: OPT-3 时区统一、OPT-5 财务排序、OPT-6 输入校验、OPT-9 日志
3. **第三批（性能与工程化）**: OPT-1、OPT-2、OPT-4、OPT-7、OPT-8
4. **第四批**: 低优先级清单按顺手程度处理

每批完成后运行 `python -m pytest alpha-forge/tests -q` 确认 32 个既有用例不回归。
