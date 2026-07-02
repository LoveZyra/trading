"""研究记忆与跨市场热启动(Round 11,优化方案 §2.9a)。

自动研究每跑一轮就产生一批"假设 -> OOS 结果"的经验,丢掉等于每次都从零开始
交学费。本模块做三件事:
  1. ResearchMemory —— JSONL 持久化的试验日志:log / query / suggest_factors。
     JSONL 而不是数据库:追加写永不损坏已有行、人眼可读、坏行可逐行跳过;
  2. cluster_strategies —— 纯 numpy 的相关距离 k-medoids,把一堆试验的收益
     曲线聚成几个"策略家族"(同家族的策略赚同一份钱,集成时只该派一个代表);
  3. warm_start_search —— 拿源市场 search 排行榜的 top 配置(因子子集×模型)
     到目标市场直接评测(标 origin="transfer"),调用方再续跑本地 search 合并
     ——跨市场研究不必从随机子集冷启动。

优雅回退:path=None(禁用)或文件缺失/行损坏时,log 不落盘、query/suggest
返回空,绝不抛异常打断研究循环。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

#: 一条 trial 建议(不强制)包含的字段;多余字段原样保留。
TRIAL_FIELDS = ("ts", "market", "universe_hash", "hypothesis", "factors",
                "model", "horizon", "oos_rankicir", "regime", "verdict")

_LB_COLS = ("factors", "model", "RankIC", "RankICIR", "ICIR",
            "LS_sharpe", "n_dates", "origin")


class ResearchMemory:
    """JSONL 研究日志。path=None 时整个对象退化为无操作(方便一键禁用)。"""

    def __init__(self, path):
        self.path = Path(path) if path else None

    # ---- 写 ----------------------------------------------------------------
    def log(self, trial: dict) -> dict:
        """追加一条试验记录;自动补 ts(UTC ISO)。返回实际写入的记录。"""
        rec = dict(trial)
        rec.setdefault("ts", pd.Timestamp.now(tz="UTC").isoformat())
        if self.path is None:
            return rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    # ---- 读 ----------------------------------------------------------------
    def _load(self) -> list:
        if self.path is None or not self.path.exists():
            return []
        out = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue                     # 坏行跳过,不让一条脏数据毁掉全库
            if isinstance(rec, dict):
                out.append(rec)
        return out

    @staticmethod
    def _icir(rec) -> float:
        try:
            v = float(rec.get("oos_rankicir"))
        except (TypeError, ValueError):
            return float("-inf")
        return v if np.isfinite(v) else float("-inf")

    def query(self, market=None, regime=None, top: int = 10) -> list:
        """按 market/regime 过滤,按 oos_rankicir 降序,取前 top 条。"""
        rows = self._load()
        if market is not None:
            rows = [r for r in rows if r.get("market") == market]
        if regime is not None:
            rows = [r for r in rows if r.get("regime") == regime]
        rows.sort(key=self._icir, reverse=True)
        return rows[:top] if top else rows

    def suggest_factors(self, regime=None, top: int = 5) -> list:
        """按历史 OOS 成功率排序的因子名(成功 = 该 trial 的 oos_rankicir > 0)。

        排序键:成功率降序,平均 oos_rankicir 次序打破平局。trial 的 factors
        字段既可以是 list,也可以是 search 排行榜的 "a+b+c" 字符串。
        """
        rows = self._load()
        if regime is not None:
            rows = [r for r in rows if r.get("regime") == regime]
        stats: dict = {}                      # name -> [n, n_success, sum_icir]
        for r in rows:
            icir = self._icir(r)
            if icir == float("-inf"):
                continue
            fac = r.get("factors")
            if isinstance(fac, str):
                fac = [x for x in fac.split("+") if x]
            if not isinstance(fac, (list, tuple)):
                continue
            for f in fac:
                s = stats.setdefault(str(f), [0, 0, 0.0])
                s[0] += 1
                s[1] += int(icir > 0)
                s[2] += icir
        ranked = sorted(stats.items(),
                        key=lambda kv: (-kv[1][1] / kv[1][0],
                                        -kv[1][2] / kv[1][0], kv[0]))
        return [name for name, _ in ranked[:top]]


def cluster_strategies(trial_returns_df: pd.DataFrame, n_clusters: int = 5,
                       max_iter: int = 30) -> dict:
    """相关距离 k-medoids 聚类(纯 numpy,无 sklearn)。

    trial_returns_df:date × trial 的收益宽表(列 = 一个试验/策略)。
    距离 = 1 - Pearson 相关:两个策略收益相关 1 -> 距离 0(同一份钱),
    相关 -1 -> 距离 2(互为对冲)。k-medoids 而不是 k-means:相关距离下
    "均值曲线"没有意义,类中心必须是真实存在的策略(medoid),这样
    medoids 可以直接拿去做集成成员。初始化用"最中心点 + 依次取最远点"
    的确定性 farthest-point 播种,结果可复现(不引入随机 seed 依赖)。
    返回 {labels: pd.Series(trial -> 簇号), medoids: [代表策略名, ...]}。
    """
    cols = list(trial_returns_df.columns)
    n = len(cols)
    if n == 0:
        return {"labels": pd.Series(dtype=int), "medoids": []}
    k = int(max(1, min(n_clusters, n)))
    corr = trial_returns_df.corr().values          # pairwise,自动跳 NaN
    D = 1.0 - np.nan_to_num(corr, nan=0.0)         # 相关缺失 -> 中性距离 1
    np.fill_diagonal(D, 0.0)

    medoids = [int(np.argmin(D.sum(axis=1)))]      # 最"居中"的点起步
    while len(medoids) < k:
        d_near = D[:, medoids].min(axis=1)
        d_near[medoids] = -1.0
        medoids.append(int(np.argmax(d_near)))
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = []
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                new_medoids.append(medoids[j])
                continue
            sub = D[np.ix_(members, members)]
            new_medoids.append(int(members[np.argmin(sub.sum(axis=1))]))
        if new_medoids == medoids:
            break
        medoids = new_medoids
    labels = np.argmin(D[:, medoids], axis=1)
    return {"labels": pd.Series(labels, index=cols, name="cluster"),
            "medoids": [cols[m] for m in medoids]}


def warm_start_search(data: dict, source_leaderboard: pd.DataFrame, *, top: int = 5,
                      factor_panels: dict | None = None,
                      candidate_models: dict | None = None,
                      factor_source: str | None = None, zoo_which: str = "alpha158",
                      zoo_max: int = 40, horizon: int = 21, rebalance: str = "ME",
                      objective: str = "RankICIR", **eval_kwargs) -> pd.DataFrame:
    """跨市场热启动:把源市场 leaderboard 的 top 配置搬到目标 `data` 上重评。

    对 source_leaderboard 前 top 行的每个 (因子子集, 模型) 配置,在目标数据上
    调 evaluate_cross_section(同一张诚实记分卡),返回与
    xsec_autoresearch.search 同构的 leaderboard,外加 origin="transfer" 列。
    调用方随后可以续跑本地 search,把两张表 concat 后按 objective 重排——
    转移配置和本地新配置在同一口径下竞争,谁强用谁。

    因子面板:默认与 search 相同(price_factor_panels);源配置若含目标面板里
    没有的因子名(如源市场用了 zoo 因子),按可得子集降级评测,全缺则跳过该行
    ——跨市场缺数据是常态,静默丢弃比报错中断更符合研究循环的用法。
    """
    from ..xsec import panel as PN
    from ..xsec import xsec_autoresearch as XAR
    from ..xsec.xsec_eval import evaluate_cross_section
    from . import models as M

    if factor_panels is None:
        if factor_source is not None:
            factor_panels = XAR.build_factor_source_panels(
                data, factor_source, zoo_which=zoo_which, zoo_max=zoo_max)
        else:
            factor_panels = PN.price_factor_panels(data)
    if candidate_models is None:
        candidate_models = {"ridge": M.RidgeModel(alpha=1.0)}

    rows = []
    if source_leaderboard is not None and len(source_leaderboard):
        for _, r in source_leaderboard.head(top).iterrows():
            sub = [f for f in str(r.get("factors", "")).split("+")
                   if f in factor_panels]
            if not sub:
                continue
            mdl = candidate_models.get(str(r.get("model")), M.RidgeModel(alpha=1.0))
            res = evaluate_cross_section(
                data, model=mdl, panels={f: factor_panels[f] for f in sub},
                horizon=horizon, rebalance=rebalance, **eval_kwargs)
            sc = res["scorecard"]
            rows.append({"factors": "+".join(sub), "model": str(r.get("model")),
                         **{k: sc[k] for k in ("RankIC", "RankICIR", "ICIR",
                                               "LS_sharpe", "n_dates")},
                         "origin": "transfer"})
    lb = pd.DataFrame(rows, columns=list(_LB_COLS))
    if len(lb) and objective in lb.columns:
        lb = lb.sort_values(objective, ascending=False,
                            na_position="last").reset_index(drop=True)
    return lb
