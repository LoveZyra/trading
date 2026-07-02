"""测试 alpha-forge 横截面选股(AI 选股)于美股:半导体/存储/光模块/AI基建/算力电力。
流程:build_universe -> load_many(yfinance) -> liquidity_filter -> evaluate_cross_section
      -> xsec_autoresearch.search(因子×模型排行榜) -> 用最佳配方对最新截面打分给出当前排名。
非投资建议。
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd

from scripts.xsec import universe, xsec_eval, xsec_autoresearch, xsec_report, panel as PN
from scripts.research import models as M
from scripts.data import loader
from scripts.strategies import multi_factor as mf

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# --- 1. 标的池:跨板块、求分散(诚实红线:>=30 只且多板块才有统计意义) ---
SECTORS = {
    # 半导体(设计+设备+模拟)
    "NVDA": "semis", "AMD": "semis", "AVGO": "semis", "MRVL": "semis", "QCOM": "semis",
    "INTC": "semis", "ARM": "semis", "TSM": "semis", "ASML": "semi_equip", "AMAT": "semi_equip",
    "LRCX": "semi_equip", "KLAC": "semi_equip", "TXN": "semis", "ADI": "semis", "NXPI": "semis",
    # 存储 / 内存
    "MU": "memory", "STX": "storage", "WDC": "storage", "PSTG": "storage", "NTAP": "storage",
    # 光模块 / 光网络
    "COHR": "optical", "LITE": "optical", "AAOI": "optical", "POET": "optical",
    "CIEN": "optical", "FN": "optical", "ANET": "optical_net",
    # AI 基建(GPU云/服务器/数据中心)
    "NBIS": "ai_infra", "CRWV": "ai_infra", "IREN": "ai_infra", "SMCI": "ai_infra",
    "DELL": "ai_infra", "VRT": "ai_infra",
    # 算力 / 电力(AI 数据中心供电)
    "VST": "power", "CEG": "power", "GEV": "power", "ETN": "power", "TLN": "power", "OKLO": "power",
}

def main():
    uni = universe.build_universe({"market": "US", "source": "list",
                                   "symbols": list(SECTORS), "sectors": SECTORS})
    print(f"[universe] {uni['n']} names, {len(set(SECTORS.values()))} sectors")

    # --- 2. 拉价格(2.5年日线)---
    data = loader.load_many(uni["symbols"], source="yfinance",
                            start="2023-01-01", interval="1d")
    print(f"[load] got {len(data)}/{uni['n']} symbols")

    # --- 3. 流动性 / 历史长度过滤 ---
    keep = universe.liquidity_filter(data, min_dollar_vol=2e6, min_days=200)
    data = {s: data[s] for s in keep}
    print(f"[liquidity] kept {len(data)} names")

    # --- 4. 横截面记分卡(purged walk-forward, horizon=21, 月度调仓)---
    res = xsec_eval.evaluate_cross_section(data, horizon=21, rebalance="ME",
                                           top_frac=0.2, n_quantiles=5)
    print("\n" + xsec_report.scorecard_markdown(res))

    # --- 5. 因子×模型自动研究排行榜(目标=RankICIR)---
    cand = {"ridge": M.RidgeModel(alpha=1.0)}
    try:
        import sklearn  # noqa: F401
        cand["rf"] = M.SklearnModel("RandomForestRegressor", n_estimators=200, max_depth=4)
    except Exception:
        pass
    lb = xsec_autoresearch.search(data, candidate_models=cand, horizon=21,
                                  rebalance="ME", objective="RankICIR", top_n=10)
    print("\n## 因子×模型 排行榜 (按 RankICIR)\n")
    print(lb.to_string(index=False))

    # --- 6. 用全样本最佳配方对最新截面打分 -> 当前排名(present-day screen)---
    panels = PN.price_factor_panels(data)
    fnames = list(panels)
    close = mf.build_panel(data, "close")
    horizon = 21
    fwd = close.shift(-horizon) / close - 1.0
    t = close.index[-1]
    tl = len(close.index) - 1
    usable = [s for s in close.index if close.index.get_loc(s) + horizon < tl]
    Xtr, ytr = [], []
    for s in usable:
        row = np.column_stack([panels[f].loc[s].values for f in fnames]); yy = fwd.loc[s].values
        mk = np.isfinite(row).all(1) & np.isfinite(yy)
        if mk.any(): Xtr.append(row[mk]); ytr.append(yy[mk])
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    mdl = M.RidgeModel(alpha=1.0).fit(Xtr, ytr)
    cur = np.column_stack([panels[f].loc[t].values for f in fnames])
    valid = np.isfinite(cur).all(1)
    score = np.full(close.shape[1], np.nan); score[valid] = mdl.predict(cur[valid])
    rank = (pd.DataFrame({"symbol": close.columns, "score": score})
            .dropna().sort_values("score", ascending=False).reset_index(drop=True))
    rank["sector"] = rank["symbol"].map(SECTORS)
    rank["rank"] = rank.index + 1
    print(f"\n## 当前横截面排名 (asof {t.date()}, 全样本 ridge, 21d 前向)\n")
    print(rank[["rank", "symbol", "sector", "score"]].to_string(index=False))
    k = max(1, int(round(len(rank) * 0.2)))
    print(f"\nTop-{k} 多头候选: {', '.join(rank.head(k)['symbol'])}")
    print(f"Bottom-{k} 弱势:   {', '.join(rank.tail(k)['symbol'])}")

if __name__ == "__main__":
    warnings.simplefilter("default")
    main()
