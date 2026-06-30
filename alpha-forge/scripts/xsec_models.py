"""横截面排序器(统一接口,复用 models.FactorModel)。
线性 RidgeModel(必有)/ LGBMModel / MLPModel(sklearn,非线性)/ 可选 torch 深度排序器;
深度时序模型(如 RAVEN/LSTM)在本机/ GPU 训练后,用 models.load_external_scores 挂接其打分。
"""
from __future__ import annotations
import numpy as np
from . import models as M

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
