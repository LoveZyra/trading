# alpha-forge 补充审查报告（bug.md 之外的新发现）

> 审查日期：2026-06-10
> 前提：bug.md 中 BUG-1~5、OPT-1~9 **均尚未修复**，本次逐项核对仍存在于代码中。
> 本报告只列 bug.md 未覆盖的新问题；NEW-1/2/3 已用代码实证复现。
> 现状：32 个测试全部通过（测试未覆盖以下场景）。

---

## 一、新发现的真实 Bug（高优先级）

### NEW-1: Breakout 做空模式下多头信号被完全覆盖（与 BUG-1 同族）

- **文件**: `alpha-forge/scripts/strategies/trend.py:47-58`
- **实证**: 构造单边上涨序列，`allow_short=False` 时 278 天持多；`allow_short=True` 时 **0 天持多、0 天持空**。
- **原因**: `hi_s = high.rolling(exit).max()`（10 bar 高点）必然 ≤ `hi`（20 bar 高点），
  所以每个突破 20 日高点的多头入场 bar 同时满足 `close >= hi_s.shift(1)`，
  最后一行 `pos[...] = 0.0` 把多头入场全部抹掉——策略退化为只可能做空（实测连空头也难触发）。
- **修复**: 空头退出条件应排除多头入场区，且按状态区分，例如：

  ```python
  pos[df["close"] >= hi.shift(1)] = 1.0
  pos[(df["close"] <= lo.shift(1)) & (df["close"] > lo_s.shift(1))] = 0.0
  if self.allow_short:
      pos[df["close"] <= lo_s.shift(1)] = -1.0
      pos[(df["close"] >= hi_s.shift(1)) & (df["close"] < hi.shift(1))] = 0.0
  ```

  并补回归测试：上涨序列 allow_short=True 时多头天数应与 allow_short=False 接近。

### NEW-2: MACrossover 做空模式在均线 warm-up 期输出 -1

- **文件**: `alpha-forge/scripts/strategies/trend.py:30-33`
- **实证**: 前 50 根 bar（slow=50 未就绪）信号为 **-1**（应为 0）。
- **原因**: warm-up 期 `f`、`s` 均为 NaN，`(NaN > NaN)=False` → `long=0` → `long*2-1 = -1`，
  在没有任何信息时凭空做空，污染回测前段收益且 walk-forward 每个 fold 开头都会复现。
- **修复**:

  ```python
  sig = long * 2 - 1
  return sig.where(f.notna() & s.notna(), 0.0)
  ```

### NEW-3: 月末再平衡日约 28% 被静默跳过（影响 3 个模块）

- **文件**:
  - `strategies/multi_factor.py:169-170`（`reindex(rebal_dates)` 后全 NaN 行被 dropna）
  - `models.py:201-202`（`[d for d in rebal_dates if d in close.index]`）
  - `sizing.py:53-54`（同上）
- **实证**: 2022-2024 工作日索引上，36 个 `resample("ME")` 标签只有 26 个落在交易日，**28% 的月度再平衡被跳过**（月末逢周末/假日的月份不调仓，仓位静默多持 1 个月）。
- **修复**: 用"每期最后一个实际交易日"替代日历月末标签：

  ```python
  rebal_dates = close.index.to_series().resample(rebalance).last().dropna().values
  ```

  三处统一抽成共享辅助函数（可与 bug.md OPT-8 的 param_grids 一起放公共模块）。

---

## 二、值得修的正确性问题（中优先级）

### NEW-4: debt_to_equity 字段混入两种不同口径

- **文件**: `data/fundamentals.py:145`（akshare）vs `:97-98`（yfinance）
- **问题**: akshare 路径把 **资产负债率（负债/资产，0~1）** 存进 `debt_to_equity`；
  yfinance 路径存的是 **负债/股东权益 ÷100**。两者量纲与含义不同，
  混合市场面板的 quality 因子 z-score 直接失真。
- **修复**: A 股口径换算 `D/E = DA / (1 - DA)` 后再入库（或拆成两个字段并在 FACTOR_DIRECTION 注明）。

### NEW-5: BollingerReversion 空头平仓过早

- **文件**: `strategies/mean_reversion.py:43-50`
- **问题**: "mirror for shorts" 实际不对称：`pos[c >= mid] = 0.0` 写在空头入场之前，
  导致空头只在 `c >= upper` 的当根维持，价格回到 `mid <= c < upper` 区间立即归零——
  空头本应持有至回到中轨，实际几乎都是 1 根 bar 的交易，徒增换手成本。
- **修复**: 与 NEW-1/BUG-1 同方式，用区间条件区分多空退出。

### NEW-6: apply_cn_rules 涨跌停判断与执行 bar 错位

- **文件**: `data/microstructure.py:46-69`
- **问题**: 用 **t 日**的涨跌停状态限制 t 日信号，但引擎 `lag=1` 在 **t+1** 成交；
  真正决定能否成交的是 t+1 的涨跌停。当前实现既可能放行实际不可成交的单，
  也可能误杀可成交的单。另：逐 bar Python 循环，长序列+多标的时偏慢。
- **修复**: 把 `limit_blocked` 先 `shift(-lag)` 对齐执行 bar（或在 backtest 内做约束）；
  循环可用向量化 + 逐段 ffill 替代。

### NEW-7: loader 缓存永不过期（陈旧数据风险）

- **文件**: `data/loader.py:24-44`
- **问题**: 缓存 key 不含"今天"，`load("AAPL")`（period="2y" 无 end）首次缓存后，
  以后每天复盘都返回**首次下载那天**的数据——除非手动删 .cache。
  与 OPT-4（坏缓存兜底）互补但性质不同：这是静默使用过期数据。
- **修复**: 对未指定 `end` 的请求把 `pd.Timestamp.today().date()` 加入 cache key，
  或检查文件 mtime 超过 1 天即重新下载。

### NEW-8: market_of 把韩国 6 位代码误判为 A 股

- **文件**: `data/market.py:38-52`
- **问题**: `market_of("005930")`（三星）→ 6 位且首位为 0 → 返回 "CN"，
  大盘 overlay 会用沪深 300 当三星的"本国指数"。`s[:3].isdigit()` 分支也过宽。
- **修复**: 增加显式 `market=` 参数优先；或维护 watchlist→market 映射，докstring 已承认该局限，但默认行为应更安全（无法判断时返回 default 而非猜 CN）。

### NEW-9: vol_target_scale warm-up 期强制空仓

- **文件**: `sizing.py:91-99`
- **问题**: 前 `lookback` 根 realized vol 为 NaN，`fillna(0.0)` 把杠杆乘数置 0 → 整本书空仓；
  walk-forward 每段开头都损失约 1 个月持仓。也没有 `clip(lower=...)`，
  实际 vol 极低时杠杆可瞬间打到 max_leverage 又骤降。
- **修复**: warm-up 期 `fillna(1.0)`（不缩放）更合理；可选对 scale 做平滑（如 `.rolling(5).mean()`）。

---

## 三、低优先级

| # | 位置 | 问题 | 方案 |
|---|------|------|------|
| N-L1 | `models.py:218` | `close.index.get_loc()` 在双层循环里反复调用，O(n) per call | 预先 `pos = {d: i for i, d in enumerate(close.index)}` |
| N-L2 | `models.py` ml_factor_backtest | 同一 `model` 实例跨 rebalance 反复 fit，自定义模型若有状态会泄漏 | 每次 rebalance clone/重建模型 |
| N-L3 | `data/sentiment.py` `_score_chinese` | 每条文本对约 70 个词条做 `str.find` 全扫描，批量历史新闻时慢 | 预编译一个合并正则（`re.compile("|".join(...))`） |
| N-L4 | `execution.py` order_plan | `days>1` 时 `child_orders` 只给第一天的计划，剩余天数无 schedule | 返回 per-day 列表或注明 |
| N-L5 | `levels.py trade_levels` | `len(c)<200` 时 ma200 为 NaN 进入 support_pool；`atr` 为 NaN 时 buy_zone/stop 全 NaN 无告警 | 入口检查最少 bar 数并给出明确报错 |
| N-L6 | 根目录 `pit_store/` 与 `pit_store_cn/` | env/sentiment 快照只有 1 天（2026-06-09），PIT 回测尚不可用；且未入 git | 确认每日任务在跑；决定是否纳入版本管理 |
| N-L7 | `examples/output/` | 生成物（png/md）提交进 repo | 加入 .gitignore |

---

## 四、与 bug.md 的关系 / 建议执行顺序

bug.md 的修复顺序依然成立。合并后的建议批次：

1. **信号正确性**（影响所有回测结论）: BUG-1 → NEW-1 → NEW-2 → NEW-5 → BUG-2
2. **再平衡与指标**: NEW-3 → BUG-3 → NEW-9
3. **数据层**: NEW-4 → NEW-7 → OPT-5 → OPT-3 → NEW-8 → NEW-6
4. **工程化**: BUG-4 → BUG-5 → OPT-6 → OPT-9 → 其余 OPT/低优先级

每批完成后 `python -m pytest alpha-forge/tests -q`，并为 NEW-1/2/3 补回归测试
（现有 32 个用例对这三类场景零覆盖：allow_short 分支、warm-up 段、月末逢周末）。
