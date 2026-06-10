# alpha-forge 全面修复 + 复审报告

> 日期：2026-06-10 ｜ 测试：48 passed（32 既有 + 16 新增回归），0 warnings
> 范围：bug.md 全部 14 项 + bug2.md 全部 9 项 + 低优先级清单

## 一、已修复（按批次）

### 批次1 信号正确性（影响所有回测结论）
| 项 | 修复内容 |
|---|---|
| BUG-1 | RSIReversion 做空模式多头被覆盖 → 多/空腿独立状态机（新增 `base.positions_from_signals` 辅助） |
| NEW-1 | Breakout 同族 bug → 同方案；上涨行情多头天数恢复 0→279 |
| NEW-2 | MACrossover 做空模式 warm-up 期输出 -1 → 均线未就绪时强制平仓 |
| NEW-5 | BollingerReversion 空头过早平仓 → 空头持有至回中轨 |
| BUG-2 | pair_spread 对冲比除零 inf → var=0 置 NaN，停牌腿不发信号 |

### 批次2 再平衡与指标
| 项 | 修复内容 |
|---|---|
| NEW-3 | 28% 月度再平衡被跳过 → 新增 `scripts/rebalance.py`（每期最后实际交易日），multi_factor / models / sizing 三处统一接入 |
| BUG-3 | profit_factor 返回 inf → 无盈亏 NaN、零亏损封顶 100 |
| NEW-9 | vol_target_scale warm-up 强制空仓 → 中性 1.0 |
| OPT-1 | vol_regime O(n²) 循环 → expanding 分位向量化（已验证与旧实现逐 bar 等价） |
| OPT-2 | rank_and_weight iterrows → rank 向量化（已验证完全等价） |

### 批次3 数据层
| 项 | 修复内容 |
|---|---|
| NEW-4 | akshare 资产负债率(D/A)混入 debt_to_equity → 换算 D/E=DA/(1-DA)，并修正了断言旧错误口径的测试 |
| OPT-5 | akshare 财务取数先按报告期排序再取最新 |
| NEW-7 | loader 缓存永不过期 → 开放式请求按当天日期入 key |
| OPT-4 | 损坏缓存 → 删除并重新下载（ibkr 为本地文件，无需缓存，已注明） |
| OPT-3 | 时区约定成文：OHLCV tz-naive 交易所本地时间；新闻时间戳一律 UTC-aware（yfinance/akshare/json 三个入口已统一） |
| NEW-8 | market_of 优先识别交易所后缀（005930.KS→KR），裸 6 位码歧义已注明 |
| NEW-6 | apply_cn_rules 涨跌停检查对齐执行 bar（新增 lag 参数） |
| L-4 | validate_ohlcv 增加 high≥low 等 OHLC 一致性检查，坏行剔除并告警 |
| L-5 | limit_blocked 增加盘中触板 touched_up/down 列 |
| L-6 | RSI：全涨=100、完全无波动=50（原来都是 NaN/100） |

### 批次4 工程化
| 项 | 修复内容 |
|---|---|
| BUG-4 | html_report JSON 完整转义（& < > U+2028/2029），注入用例实测拦截 |
| BUG-5 | signal_tracker 两处文件句柄 → with 上下文 |
| OPT-6 | backtest 入口校验（lag/费用/cost_model/periods），sqrt 缺 volume 显式 warning |
| OPT-7 | backtest_portfolio 支持 sqrt 市场冲击（panel_volume）+ 输出逐次调仓 ledger（原恒为空） |
| OPT-8 | 参数网格抽到 `scripts/param_grids.py`，autoresearch 与 CLI 共用 |
| OPT-9 | loader/fundamentals/autoresearch 静默吞错处加 logging |
| L-2/L-7/L-8/L-9 | math.isnan、levels._safe_round+入参校验、f-string、sectors register_sector/load_sector_map |
| N-L1~N-L4 | models get_loc 预计算、每次 rebalance deepcopy 模型、中文情绪词典合并正则（顺带消除重叠词条双计数）、execution 多日计划 day_shares |
| L-3/N-L7 | requirements 加版本上界；.gitignore 补 examples/output、.cache、__pycache__ |
| 清理 | pyflakes 全部未用 import 清除（保留 3 处有意导出的 noqa） |

## 二、验证
- 48 个测试全绿；新增 tests/test_fixes.py 16 项回归（allow_short 分支、warm-up、月末逢周末、inf、转义、口径换算、执行 bar 对齐等原零覆盖场景）
- 向量化重写（vol_regime / rank_and_weight）与旧实现逐值等价性验证通过
- 端到端冒烟（warnings 升级为 error）：单资产回测、walk-forward、autoresearch、多因子组合、ML 因子回测、regime、risk parity 全部正常

## 三、复审结论：剩余优化点（均为低风险/运营项）
1. **pit_store 仅 1 天快照**（2026-06-09）：PIT 回测要积累数月才有意义，确认每日快照任务在跑（运营项，非代码）。
2. **apply_cn_rules 仍是逐 bar 循环**：状态依赖难以向量化，日线单标的规模下性能可接受；若未来跑分钟级再优化。
3. **裸 6 位韩股代码仍按约定判为 CN**：根本歧义无法从代码推断，建议 watchlist_kr.txt 统一带 .KS/.KQ 后缀。
4. **行为微调已记录**：ZScoreReversion(allow_short=False) 现持多至 |z|≤exit（旧版在 z≥+entry 也平仓）；中文情绪不再对"评级下调/下调"这类重叠词条双计分——两者均为更合理语义，但与旧数值有差异。
5. **测试覆盖仍有洼地**：html_report（872 行）只有注入转义 1 项直接测试；autoresearch / portfolio 模块覆盖较浅。
6. **SECTOR_MAP 仍以硬编码为主**：接口已加（register_sector/load_sector_map），建议把映射迁到外部 sectors.json。
7. **改动未提交 git**：48 个文件已暂存，建议审阅后 commit（沙盒内 git 偶发 index.lock 告警，必要时在本机执行）。
