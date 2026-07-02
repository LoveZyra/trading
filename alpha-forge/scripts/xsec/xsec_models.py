"""横截面排序器(统一接口,复用 models.FactorModel)。
线性 RidgeModel(必有)/ LGBMModel / MLPModel(sklearn,非线性)/ 可选 torch 深度排序器;
深度时序模型(如 RAVEN/LSTM)在本机/ GPU 训练后,用 models.load_external_scores 挂接其打分。
"""
from __future__ import annotations
import numpy as np
from ..research import models as M

# 论文(RAVEN)配方备忘,供深度排序器对齐:
PAPER_CONFIG = dict(d_model=128, num_experts=3, cit_thresholds=[0.3, 0.6, 0.9],
                    expert_layers=3, lambda_entropy=0.1, optimizer="AdamW",
                    scheduler="cosine", epochs=60, horizon=10)

def ridge(alpha=1.0):  return M.RidgeModel(alpha=alpha)
def lgbm(**kw):        return M.LGBMModel(**kw)
def mlp(**kw):         return M.MLPModel(**kw)

class TorchRanker(M.FactorModel):
    """可选 torch MLP 排序器(缺 torch 时报清晰错误)。截面 factors->fwd return。"""
    def __init__(self, hidden=64, epochs=40, lr=1e-3, seed=42):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.seed)
        X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1, 1)
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-9
        Xt = torch.tensor(((X - self._mu) / self._sd), dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
        self.net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.GELU(), nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad(); loss = ((self.net(Xt) - yt) ** 2).mean(); loss.backward(); opt.step()
        return self
    def predict(self, X):
        import torch
        X = (np.asarray(X, float) - self._mu) / self._sd
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

def external(json_path):
    """挂接外部深度模型(RAVEN/LSTM/TS-foundation)在本机训练后导出的打分。"""
    return M.load_external_scores(json_path)


# ---------------------------------------------------------------------------
# Round11 §2.5 —— Listwise 排序损失(纯 numpy 梯度下降;LambdaMART 需 lightgbm)
#
# 现有 FactorModel 接口是扁平样本 (n_samples, n_factors) -> 前向收益,不带分组
# 信息;listwise 损失需要知道哪些行属于同一天的截面。向后兼容方案:
#   fit(X, y, groups=None) —— groups 为每行的组 id(如日期序号)数组;
#   xsec_eval / ml_factor_backtest 现有调用只传 fit(X, y),模型内部退化为把
#   整个训练窗当"一个大组"计算 listwise 损失。这是近似:跨日收益被放进同一
#   个 softmax / 似然里比较,牺牲了逐日截面结构,但排序方向仍由同一线性信号
#   驱动,作为默认路径可用。
#   TODO(下一轮): 在 xsec_eval.evaluate_cross_section 与 models.
#   ml_factor_backtest 逐日堆叠训练行处把日期组 id 穿透到 fit(groups=...)。
# ---------------------------------------------------------------------------

def _lw_groups(groups, n):
    """groups(每行组 id)-> [组内行号数组];None => 单一大组(近似,见上)。"""
    if groups is None:
        return [np.arange(n)]
    g = np.asarray(groups)
    if g.shape[0] != n:
        raise ValueError(f"groups 长度 {g.shape[0]} != 样本数 {n}")
    return [np.where(g == v)[0] for v in np.unique(g)]


def _lw_softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def _lw_logsoftmax(v):
    m = np.max(v)
    return v - m - np.log(np.exp(v - m).sum())


def _lw_zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


class _GDListwiseBase(M.FactorModel):
    """线性打分 s=Xw 的 listwise 损失基类:标准化输入、全批梯度下降、L2 正则、
    loss 平台 early stop、loss 上升时步长减半(发散保护)。
    子类只需实现 _group_loss_grad(s, y) -> (loss, dL/ds)。
    predict 输出连续分数(只用于排序;不带截距——截距不改变组内排序)。
    deepcopy 安全:状态只有标量超参和 numpy 数组(ml_factor_backtest 每折 deepcopy)。"""

    def __init__(self, lr=0.2, epochs=500, l2=1e-4, tol=1e-7, patience=25):
        self.lr, self.epochs, self.l2 = float(lr), int(epochs), float(l2)
        self.tol, self.patience = float(tol), int(patience)
        self._mu = self._sd = self._w = None

    def _group_loss_grad(self, s, y):
        raise NotImplementedError

    def fit(self, X, y, groups=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        m = np.isfinite(X).all(1) & np.isfinite(y)
        if groups is not None:
            groups = np.asarray(groups)[m]
        X, y = X[m], y[m]
        if len(y) < 3:
            raise ValueError("listwise fit 需要至少 3 行有效样本")
        self._mu, self._sd = X.mean(0), X.std(0)
        self._sd[self._sd == 0] = 1.0
        Xs = (X - self._mu) / self._sd
        grps = [ix for ix in _lw_groups(groups, len(y)) if len(ix) >= 2]
        if not grps:
            raise ValueError("没有 >=2 行的组,无法计算 listwise 损失")
        w = np.zeros(Xs.shape[1]); lr = self.lr
        best, stale = np.inf, 0
        for _ in range(self.epochs):
            s = Xs @ w
            loss, gs = 0.0, np.zeros_like(w)
            for ix in grps:
                li, gi = self._group_loss_grad(s[ix], y[ix])
                loss += li; gs += Xs[ix].T @ gi
            loss = loss / len(grps) + self.l2 * float(w @ w)
            grad = gs / len(grps) + 2.0 * self.l2 * w
            if loss > best + 1e-12:
                lr *= 0.5                                    # 发散保护
            if loss < best - self.tol:
                best, stale = loss, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break                                    # loss 平台 early stop
            w = w - lr * grad
            if lr < 1e-6:
                break
        self._w = w
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self._mu) / self._sd
        return Xs @ self._w


class ListNetModel(_GDListwiseBase):
    """ListNet(top-1)listwise 排序(Cao et al. 2007),纯 numpy 解析梯度。

    组内目标分布 p = softmax(zscore(y)/tau)(y 先组内 z 标准化,使 tau 跨数据
    尺度可比;softmax 对 y 平移常数不变),模型分布 q = softmax(s),
    损失 L = -Σ p·log q,解析梯度 dL/ds = q - p。
    超参:lr / epochs / l2 / tau(tau 越小目标分布越尖、越只关心第一名)。
    fit(X, y, groups=None):不传 groups 时整个训练集按单一大组近似(见模块注)。"""
    name = "listnet"

    def __init__(self, lr=0.2, epochs=500, l2=1e-4, tau=1.0, tol=1e-7, patience=25):
        super().__init__(lr=lr, epochs=epochs, l2=l2, tol=tol, patience=patience)
        self.tau = float(tau)

    def _group_loss_grad(self, s, y):
        p = _lw_softmax(_lw_zscore(y) / self.tau)
        return float(-(p * _lw_logsoftmax(s)).sum()), _lw_softmax(s) - p


class ListMLEModel(_GDListwiseBase):
    """ListMLE(Xia et al. 2008):Plackett-Luce 负对数似然,序取 y 降序。

    组内按 y 降序排出分数 s_(1..n),L = Σ_i [logsumexp(s_i..s_n) - s_i]。
    解析梯度(对排序后分数):dL/ds_j = e^{s_j} · Σ_{i<=j} 1/Z_i - 1,
    其中 Z_i = Σ_{k>=i} e^{s_k};用倒序 cumsum + 前缀 cumsum(1/Z) 向量化。
    损失只依赖 y 的排序 => 对 y 的任意严格单调变换(含平移常数)不变。
    超参:lr / epochs / l2。fit(X, y, groups=None) 同 ListNet。"""
    name = "listmle"

    def _group_loss_grad(self, s, y):
        order = np.argsort(-y, kind="stable")
        ss = s[order]
        mx = ss.max()
        e = np.exp(ss - mx)
        Z = np.cumsum(e[::-1])[::-1]                 # Z_i = Σ_{k>=i} e^{s_k - mx}
        loss = float(np.sum(np.log(Z) + mx - ss))
        gs = e * np.cumsum(1.0 / Z) - 1.0
        g = np.empty_like(s); g[order] = gs
        return loss, g


class LSListModel(_GDListwiseBase):
    """长短仓 listwise 排序(思路对齐 arXiv:2104.12484,shift-invariant)。

    与原文的差异:原文用可微排序算子直接优化多空组合目标;这里实现为
    ListNet 变体——两支 top-1 交叉熵相加,只对 y 两端的名字给目标权重:
      多头支:目标分布仅在 y 的 top top_frac(默认 20%)非零,组内权重
              softmax(zscore(y)/tau),模型分布 softmax(s);
      空头支:目标分布仅在 y 的 bottom top_frac 非零,权重 softmax(-z/tau),
              模型分布 softmax(-s)(分数越低越该做空);
      中间 1-2*top_frac 的名字目标权重为 0 —— 损失只要求两端排对,
      正是 Top-K 多空组合真正用到的部分。
    分位集合(argsort)与 softmax(z 标准化)都对 y 平移常数不变 =>
    y 平移不改变训练出的排序输出(卖点)。
    梯度:dL/ds = (softmax(s) - p_top) - (softmax(-s) - p_bot)。
    超参:lr / epochs / l2 / tau / top_frac。fit(X, y, groups=None) 同 ListNet。"""
    name = "ls_list"

    def __init__(self, lr=0.2, epochs=500, l2=1e-4, tau=1.0, top_frac=0.2,
                 tol=1e-7, patience=25):
        super().__init__(lr=lr, epochs=epochs, l2=l2, tol=tol, patience=patience)
        self.tau, self.top_frac = float(tau), float(top_frac)

    def _group_loss_grad(self, s, y):
        n = len(y)
        k = max(1, int(round(n * self.top_frac)))
        z = _lw_zscore(y)
        order = np.argsort(-y, kind="stable")
        top, bot = order[:k], order[-k:]
        p_top = np.zeros(n); p_top[top] = _lw_softmax(z[top] / self.tau)
        p_bot = np.zeros(n); p_bot[bot] = _lw_softmax(-z[bot] / self.tau)
        loss = float(-(p_top * _lw_logsoftmax(s)).sum()
                     - (p_bot * _lw_logsoftmax(-s)).sum())
        grad = (_lw_softmax(s) - p_top) - (_lw_softmax(-s) - p_bot)
        return loss, grad


class LambdaMARTModel(M.FactorModel):
    """LambdaMART:lightgbm LGBMRanker(objective='lambdarank') 的 lazy 包装。

    需要 lightgbm —— 本环境未装时 fit 抛 ImportError(信息含 'lightgbm'),
    import 推迟到 fit,构造/deepcopy 不需要该库。
    fit(X, y, groups=None):groups 缺省单一大组(近似,见模块头注)。
    lambdarank 需要离散相关性等级:连续 y 在组内按名次分位离散成 n_levels
    (默认 5)档,0=最差 .. n_levels-1=最好(rel = 名次*n_levels//组大小)。
    predict 输出连续分数(排序用)。其余 **kwargs 透传 LGBMRanker。"""
    name = "lambdamart"

    def __init__(self, n_levels=5, **kwargs):
        self.n_levels = int(n_levels)
        self._kwargs = dict(n_estimators=120, learning_rate=0.05, num_leaves=15,
                            min_child_samples=5, verbose=-1)
        self._kwargs.update(kwargs)
        self._est = None

    def fit(self, X, y, groups=None):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LambdaMARTModel 需要 lightgbm(`pip install lightgbm`)") from e
        X = np.asarray(X, float); y = np.asarray(y, float)
        m = np.isfinite(X).all(1) & np.isfinite(y)
        if groups is not None:
            groups = np.asarray(groups)[m]
        X, y = X[m], y[m]
        grps = _lw_groups(groups, len(y))
        rel = np.zeros(len(y), int)
        for ix in grps:                              # 组内名次 -> 5 档相关性等级
            r = np.argsort(np.argsort(y[ix]))
            rel[ix] = (r * self.n_levels) // max(len(ix), 1)
        order = np.concatenate(grps)                 # LGBMRanker 要求同组行连续
        self._est = lgb.LGBMRanker(objective="lambdarank", **self._kwargs)
        self._est.fit(X[order], rel[order], group=[len(ix) for ix in grps])
        return self

    def predict(self, X):
        return self._est.predict(np.asarray(X, float))


# 注册表:key -> 模型类(接入 xsec_autoresearch 由主线程协调,这里只提供映射)
LISTWISE_REGISTRY = {
    "listnet": ListNetModel,
    "listmle": ListMLEModel,
    "ls_list": LSListModel,
    "lambdamart": LambdaMARTModel,
}
