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

## 三、复审遗留项的二次处理（2026-06-10 同日完成）
1. ~~apply_cn_rules 逐 bar 循环~~ → 改为 numpy 数组循环（逻辑不变，约 10-50x 提速，5000 bar 实测 <2ms/次）。
2. ~~韩股代码歧义~~ → watchlist_kr.txt 统一带 .KS 后缀；from_pykrx / from_akshare 自动剥离后缀，所有适配器对带后缀代码兼容。
3. ~~SECTOR_MAP 硬编码~~ → 导出 alpha-forge/sectors.json（51 条），sectors.py 在 import 时自动合并工作目录或 skill 根目录的 sectors.json。
4. ~~测试覆盖洼地~~ → 新增 7 项：html_report 各 section 渲染、autoresearch 单资产+ensemble+组合 smoke、portfolio_health、ZScore 新语义 pin、适配器后缀剥离、sectors.json 外挂。测试合计 **55 passed**。
5. ~~git 未提交~~ → 已提交 84a137d（49 文件；之前卡住的 .git/index.lock 已清理）。
6. **行为微调已记录**（保持）：ZScoreReversion 持仓至 |z|≤exit、中文情绪重叠词条不再双计分——已用测试 pin 住新语义。
7. **唯一遗留（需要你决定）**：pit_store 仅 1 天快照，而负责积累快照的 3 个定时任务（A股盘后复盘 / 美股盘后复盘 / 每日早间简报）当前都处于 **禁用** 状态——PIT 数据不会再增长。需要的话重新启用即可。
