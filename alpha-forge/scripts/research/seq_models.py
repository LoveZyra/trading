"""Deep sequence models for cross-sectional stock selection (roadmap v3 SS2.6).

Two execution paths (2.6b/2.6c):
  * Path A (in-skill): light models (GRU/LSTM/small Transformer) trained walk-forward
    in ml_seq_backtest, same discipline as ml_factor_backtest.
  * Path B (local GPU): heavy models trained via scripts/train_seq_model.py on the
    user's machine, artifacts (model.pt + config.json) loaded here for inference only.

Hard constraints honoured in this module:
  * torch is OPTIONAL. Everything numpy-only (sequence panel building, robust
    normalization, checkpointing, needs_local_gpu, FlattenRidgeSeqModel, and
    ml_seq_backtest with a numpy model) works without torch. Any torch feature
    imports lazily and raises a clear ImportError telling you to `pip install torch`.
  * 2.6g sandbox reality: a single bash call tops out at ~45s, so train_loop MUST
    checkpoint every epoch (state + epoch counter to json+npz) and resume
    transparently -- re-running the same call continues instead of restarting.
  * 2.6f honesty gate: preprocessing statistics are fitted on TRAIN rows only
    (fit_stats/apply_stats two-step API) and persist with the model file; a deep
    model only earns a seat in the ensemble if it beats the ridge baseline's
    RankICIR on the SAME scorecard and survives DSR correction (see ml_seq_backtest).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import FactorModel, MLResult

DEFAULT_FIELDS = ("open", "high", "low", "close", "volume")


# ----------------------------------------------------------------------------
# Sequence panel building + robust normalization (pure numpy, always available)
# ----------------------------------------------------------------------------
def build_sequence_panel(data: dict, *, seq_len: int = 60,
                         fields=DEFAULT_FIELDS, dates=None,
                         relative: bool = True) -> dict:
    """Per-symbol rolling windows -> one 3D sample tensor (Qlib Alpha360 style).

    data:   {symbol: OHLCV DataFrame} (canonical columns).
    dates:  optional iterable of window END dates to keep (walk-forward uses this).
    relative: divide price fields by the window's last close and volume-ish fields
        by the window's last volume, so levels are comparable across symbols; the
        remaining scale is handled by RobustZScore (fit_stats/apply_stats below).

    Returns {"X": (n_samples, seq_len, n_feat) float array,
             "index": [(end_date, symbol), ...] aligned with X's first axis,
             "feature_names": list(fields)}.
    Windows containing any NaN are dropped. Normalization is NOT applied here:
    fit stats on the TRAIN subset only, then apply everywhere (leak prevention).
    """
    fields = list(fields)
    keep = None if dates is None else set(pd.DatetimeIndex(dates))
    price_like = [i for i, f in enumerate(fields) if f in ("open", "high", "low", "close", "vwap")]
    vol_like = [i for i, f in enumerate(fields) if f in ("volume", "amount")]
    ci = fields.index("close") if "close" in fields else None
    Xs, idx = [], []
    for sym in sorted(data):
        df = data[sym]
        if df is None or len(df) < seq_len or any(f not in df.columns for f in fields):
            continue
        mat = df[fields].to_numpy(float)
        dts = df.index
        for i in range(seq_len - 1, len(df)):
            d = dts[i]
            if keep is not None and d not in keep:
                continue
            w = mat[i - seq_len + 1:i + 1]
            if not np.isfinite(w).all():
                continue
            w = w.copy()
            if relative:
                if ci is not None and w[-1, ci] > 0:
                    w[:, price_like] = w[:, price_like] / w[-1, ci]
                for j in vol_like:
                    ref = w[-1, j]
                    if ref > 0:
                        w[:, j] = w[:, j] / ref
            Xs.append(w)
            idx.append((d, sym))
    X = np.stack(Xs) if Xs else np.empty((0, seq_len, len(fields)))
    return {"X": X, "index": idx, "feature_names": fields}


def fit_stats(X: np.ndarray) -> dict:
    """RobustZScore statistics (median + MAD per feature) from TRAIN samples only.

    Call this on the training tensor, then apply_stats() on train AND test with
    the same stats -- the test set must never contribute to normalization (2.6f)."""
    X = np.asarray(X, float)
    flat = X.reshape(-1, X.shape[-1])
    med = np.nanmedian(flat, axis=0)
    mad = np.nanmedian(np.abs(flat - med), axis=0)
    mad = np.where(mad <= 0, 1.0, mad)
    return {"median": med, "mad": mad, "kind": "robust_zscore"}


def apply_stats(X: np.ndarray, stats: dict, *, clip: float = 5.0) -> np.ndarray:
    """Apply RobustZScore with PRE-FITTED stats: z = (x - median)/(1.4826*MAD),
    clipped to +-clip. Never fits anything -- pass train-fitted stats."""
    X = np.asarray(X, float)
    med = np.asarray(stats["median"], float)
    mad = np.asarray(stats["mad"], float)
    z = (X - med) / (1.4826 * mad)
    return np.clip(z, -clip, clip)


# ----------------------------------------------------------------------------
# Sequence model interface + numpy baseline + path routing
# ----------------------------------------------------------------------------
class SequenceFactorModel(FactorModel):
    """FactorModel over 3D input: X is (n_samples, seq_len, n_features).

    Shares the fit/predict contract with the tabular FactorModel so the ranking /
    portfolio layers don't care which family produced the scores. fit() may accept
    an optional groups= array (int date-group id per sample) for listwise losses."""
    name = "seq"
    seq_input = True

    def fit(self, X: np.ndarray, y: np.ndarray, **kw) -> "SequenceFactorModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FlattenRidgeSeqModel(SequenceFactorModel):
    """Numpy-only sequence baseline: summary-stat flattening -> closed-form ridge.

    Each window becomes [per-feature mean, std, last value, last/first - 1], then a
    ridge fit (same math as models.RidgeModel). This is the in-family stand-in for
    the 2.6f ridge gate: a deep sequence model that cannot beat THIS on RankICIR has
    no business in the ensemble. Always available, no torch."""
    name = "seq_ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._mu = self._sd = self._w = self._b = None

    @staticmethod
    def _flatten(X):
        X = np.asarray(X, float)
        first = np.where(X[:, 0, :] == 0, 1e-12, X[:, 0, :])
        feats = [X.mean(1), X.std(1), X[:, -1, :], X[:, -1, :] / first - 1.0]
        return np.column_stack([f for f in feats])

    def fit(self, X, y, **kw):
        F = self._flatten(X); y = np.asarray(y, float)
        self._mu, self._sd = F.mean(0), F.std(0)
        self._sd[self._sd == 0] = 1.0
        Fs = (F - self._mu) / self._sd
        A = Fs.T @ Fs + self.alpha * np.eye(Fs.shape[1])
        self._w = np.linalg.solve(A, Fs.T @ y)
        self._b = y.mean()
        return self

    def predict(self, X):
        Fs = (self._flatten(X) - self._mu) / self._sd
        return Fs @ self._w + self._b


_HEAVY_MODELS = {"master", "stockformer", "alphaportfolio"}
_LIGHT_MODELS = {"gru", "lstm", "transformer", "listwise_transformer", "seq_ridge"}


def needs_local_gpu(model_name: str, n_samples: int) -> bool:
    """Path A vs path B routing (2.6b): heavy architectures always go to the local
    GPU box; light ones only when the sample count would blow the sandbox budget.
    Unknown names route to local GPU (conservative -- override explicitly if light)."""
    name = str(model_name).lower()
    if name in _HEAVY_MODELS:
        return True
    if name in _LIGHT_MODELS:
        return int(n_samples) > 200_000
    return True


# ----------------------------------------------------------------------------
# Checkpointing (2.6g hard requirement: 45s bash cap -> every-epoch persistence)
# ----------------------------------------------------------------------------
def save_checkpoint(state: dict, path) -> None:
    """Persist a training state dict to `path`.json + `path`.npz (atomic writes).

    Arrays (np.ndarray values, or one-level dicts of arrays such as a state_dict
    converted to numpy) go to the npz; everything else must be json-serializable
    (epoch counter, completed-fold list, loss history, preprocessing stats are
    stored as arrays too). Overwrites any existing checkpoint at `path`."""
    base = Path(str(path))
    base.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta, groups = {}, {}, []
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in v.values()):
            groups.append(k)
            for kk, vv in v.items():
                arrays[f"{k}::{kk}"] = vv
        else:
            meta[k] = v
    meta["__array_groups__"] = groups
    # npz first, json last: load_checkpoint keys off the .json, so a crash between
    # the two writes never leaves a json pointing at a missing/stale npz.
    fd, tmp = tempfile.mkstemp(dir=str(base.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, str(base) + ".npz")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    fd, tmp = tempfile.mkstemp(dir=str(base.parent), suffix=".tmp.json")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(meta, default=float), encoding="utf-8")
        os.replace(tmp, str(base) + ".json")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path) -> dict | None:
    """Load a checkpoint saved by save_checkpoint. Returns None when the checkpoint
    does not exist (fresh start) -- resume code can just `state = load_checkpoint(p)
    or default_state`."""
    base = Path(str(path))
    jpath = Path(str(base) + ".json")
    npath = Path(str(base) + ".npz")
    if not jpath.exists():
        return None
    state = json.loads(jpath.read_text(encoding="utf-8"))
    groups = set(state.pop("__array_groups__", []))
    for g in groups:
        state[g] = {}
    if npath.exists():
        with np.load(str(npath), allow_pickle=False) as z:
            for k in z.files:
                if "::" in k:
                    g, kk = k.split("::", 1)
                    state.setdefault(g, {})[kk] = z[k]
                else:
                    state[k] = z[k]
    return state


@dataclass
class SequenceDataset:
    """Training bundle for train_loop: X (n, seq_len, n_feat), y forward returns,
    groups = int date-group id per sample (listwise losses group by trading date)."""
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None = None
    index: list = field(default_factory=list)

    def __len__(self):
        return len(self.y)


# ----------------------------------------------------------------------------
# Torch implementations (lazy: nothing below imports torch at module import time)
# ----------------------------------------------------------------------------
_TORCH_HINT = ("this feature needs torch, which is not installed in this environment. "
               "Install it with `pip install torch` (or train on a local GPU machine via "
               "scripts/train_seq_model.py -- roadmap v3 SS2.6 path B -- and load the "
               "saved artifact here with load_trained_seq_model).")


def _torch():
    """Lazy torch import. Every torch-touching entry point funnels through here so a
    torch-less sandbox gets ONE consistent, actionable ImportError."""
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:
        raise ImportError(_TORCH_HINT) from e


def pick_device(device: str | None = None):
    """cuda if available else mps else cpu (deep sequence models on CPU are path-B
    candidates, see needs_local_gpu)."""
    torch = _torch()
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _listwise_loss(kind: str, scores, y, groups, torch):
    """Listwise losses grouped by trading date. scores/y: 1D tensors; groups: 1D
    int tensor. listnet = softmax top-1 cross-entropy; listmle = Plackett-Luce."""
    total = scores.new_zeros(())
    n_groups = 0
    for g in torch.unique(groups):
        m = groups == g
        if int(m.sum()) < 2:
            continue
        s, yy = scores[m], y[m]
        if kind == "listnet":
            total = total - (torch.softmax(yy, dim=0) * torch.log_softmax(s, dim=0)).sum()
        else:  # listmle
            order = torch.argsort(yy, descending=True)
            s_sorted = s[order]
            rev_lse = torch.flip(torch.logcumsumexp(torch.flip(s_sorted, [0]), dim=0), [0])
            total = total + (rev_lse - s_sorted).sum() / len(s_sorted)
        n_groups += 1
    return total / max(n_groups, 1)


class _TorchSeqModel(SequenceFactorModel):
    """Shared config/training plumbing for GRU/LSTM/Transformer. Constructing the
    object never touches torch; fit()/predict() do (and raise the install-hint
    ImportError when torch is absent)."""
    name = "torch_seq"

    def __init__(self, *, hidden: int = 64, n_layers: int = 2, dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 20,
                 batch_size: int = 512, patience: int = 5, loss: str = "mse",
                 device: str | None = None, seed: int = 0, val_frac: float = 0.2):
        if loss not in ("mse", "listnet", "listmle"):
            raise ValueError(f"loss must be mse|listnet|listmle, got {loss!r}")
        self.hidden, self.n_layers, self.dropout = hidden, n_layers, dropout
        self.lr, self.weight_decay, self.epochs = lr, weight_decay, epochs
        self.batch_size, self.patience, self.loss = batch_size, patience, loss
        self.device_str, self.seed, self.val_frac = device, seed, val_frac
        self.net = None
        self._stats = None          # preprocessing stats travel WITH the model (2.6f)
        self._meta = {}

    # -- subclass hook ---------------------------------------------------------
    def _build_net(self, n_feat: int):
        raise NotImplementedError

    def config(self) -> dict:
        return {"class": type(self).__name__, "name": self.name, "hidden": self.hidden,
                "n_layers": self.n_layers, "dropout": self.dropout, "lr": self.lr,
                "weight_decay": self.weight_decay, "epochs": self.epochs,
                "batch_size": self.batch_size, "patience": self.patience,
                "loss": self.loss, "seed": self.seed, "val_frac": self.val_frac}

    def fit(self, X, y, *, groups=None, ckpt_path=None, resume=True, **kw):
        ds = SequenceDataset(np.asarray(X, float), np.asarray(y, float),
                             None if groups is None else np.asarray(groups))
        train_loop(self, ds, loss=self.loss, epochs=self.epochs,
                   ckpt_path=ckpt_path, resume=resume)
        return self

    def predict(self, X):
        torch = _torch()
        if self.net is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        X = np.asarray(X, np.float32)
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.tensor(X[i:i + 4096], dtype=torch.float32, device=dev)
                out.append(self.net(xb).squeeze(-1).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    def state_arrays(self) -> dict:
        """torch state_dict -> {name: np.ndarray} for the json+npz checkpoint."""
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def load_state_arrays(self, arrays: dict, n_feat: int | None = None):
        torch = _torch()
        if self.net is None:
            if n_feat is None:
                raise ValueError("need n_feat to rebuild the net before loading weights")
            self.net = self._build_net(n_feat)
        self.net.load_state_dict({k: torch.tensor(np.asarray(v)) for k, v in arrays.items()})
        return self


class _RecurrentSeqModel(_TorchSeqModel):
    _rnn_cls = "GRU"

    def _build_net(self, n_feat: int):
        torch = _torch()
        import torch.nn as nn
        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(s, rnn_cls, n_feat, hidden, n_layers, dropout):
                super().__init__()
                s.rnn = getattr(nn, rnn_cls)(n_feat, hidden, num_layers=n_layers,
                                             batch_first=True,
                                             dropout=dropout if n_layers > 1 else 0.0)
                s.head = nn.Linear(hidden, 1)

            def forward(s, x):
                out, _ = s.rnn(x)
                return s.head(out[:, -1, :])

        return Net(self._rnn_cls, n_feat, self.hidden, self.n_layers, self.dropout)


class GRUModel(_RecurrentSeqModel):
    """Qlib-baseline-style GRU: hidden=64, 2 layers, last hidden state -> linear head.
    Path A (in-skill walk-forward) at typical sample counts; see needs_local_gpu."""
    name = "gru"
    _rnn_cls = "GRU"


class LSTMModel(_RecurrentSeqModel):
    """Qlib-baseline-style LSTM. Same head/config as GRUModel."""
    name = "lstm"
    _rnn_cls = "LSTM"


class TransformerModel(_TorchSeqModel):
    """Small Transformer encoder (d_model=hidden, nhead=2, 2 layers, sinusoidal
    positions, last-token head) -- the Qlib Transformer baseline scale."""
    name = "transformer"

    def __init__(self, *, nhead: int = 2, **kw):
        super().__init__(**kw)
        self.nhead = nhead

    def config(self):
        c = super().config(); c["nhead"] = self.nhead; return c

    def _build_net(self, n_feat: int):
        torch = _torch()
        import torch.nn as nn
        torch.manual_seed(self.seed)
        d_model, nhead, n_layers, dropout = self.hidden, self.nhead, self.n_layers, self.dropout

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj = nn.Linear(n_feat, d_model)
                layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                                   dropout=dropout, batch_first=True)
                s.enc = nn.TransformerEncoder(layer, n_layers)
                s.head = nn.Linear(d_model, 1)

            def forward(s, x):
                T = x.shape[1]
                pos = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(-1)
                div = torch.exp(torch.arange(0, d_model, 2, device=x.device, dtype=torch.float32)
                                * (-np.log(10000.0) / d_model))
                pe = torch.zeros(T, d_model, device=x.device)
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                h = s.enc(s.proj(x) + pe)
                return s.head(h[:, -1, :])

        return Net()


def train_loop(model: "_TorchSeqModel", dataset: SequenceDataset, *, loss: str = "mse",
               epochs: int = 20, ckpt_path=None, resume: bool = True,
               verbose: bool = False) -> dict:
    """Epoch loop with EVERY-EPOCH checkpointing and transparent resume (2.6g).

    The Cowork sandbox caps a bash call at ~45s and does not keep background
    processes alive, so 'train for N epochs' must survive being killed and re-run:
    after each epoch we persist {done_epochs, model weights, best weights, loss
    history, early-stop counter} via save_checkpoint (json+npz); when `resume` and
    a checkpoint exists at ckpt_path, training continues from done_epochs instead
    of restarting. Optimizer moments are rebuilt on resume (documented tradeoff:
    weights+epoch are exact, Adam state is not -- acceptable for these model sizes).

    Validation = trailing `model.val_frac` of the (time-ordered) samples; early
    stopping with `model.patience`; best-val weights are restored at the end.
    loss: "mse" | "listnet" | "listmle" (listwise ones need dataset.groups = date ids).
    Returns {"epochs_run", "best_epoch", "train_loss", "val_loss"}.
    """
    torch = _torch()
    if loss not in ("mse", "listnet", "listmle"):
        raise ValueError(f"unknown loss {loss!r}")
    if loss != "mse" and dataset.groups is None:
        raise ValueError(f"loss={loss!r} needs dataset.groups (date-group id per sample)")
    dev = pick_device(model.device_str)
    X = np.asarray(dataset.X, np.float32)
    y = np.asarray(dataset.y, np.float32)
    g = None if dataset.groups is None else np.asarray(dataset.groups)
    n = len(y)
    n_val = int(round(n * model.val_frac)) if model.val_frac > 0 and n >= 50 else 0
    tr_sl, va_sl = slice(0, n - n_val), slice(n - n_val, n)
    model._meta["n_feat"] = X.shape[-1]
    if model.net is None:
        model.net = model._build_net(X.shape[-1])
    model.net.to(dev)

    hist = {"train_loss": [], "val_loss": []}
    start_epoch, best_epoch, best_val, since_best = 0, -1, float("inf"), 0
    best_state = None
    if ckpt_path and resume:
        st = load_checkpoint(ckpt_path)
        if st is not None and int(st.get("done_epochs", 0)) > 0:
            model.load_state_arrays(st["model_state"], n_feat=X.shape[-1])
            model.net.to(dev)
            start_epoch = int(st["done_epochs"])
            hist["train_loss"] = list(st.get("train_loss", []))
            hist["val_loss"] = list(st.get("val_loss", []))
            best_epoch = int(st.get("best_epoch", -1))
            best_val = float(st.get("best_val", float("inf")))
            since_best = int(st.get("since_best", 0))
            best_state = st.get("best_state") or None

    opt = torch.optim.Adam(model.net.parameters(), lr=model.lr,
                           weight_decay=model.weight_decay)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    gt = None if g is None else torch.tensor(g, dtype=torch.long, device=dev)

    def _epoch_batches(epoch):
        rng = np.random.default_rng(model.seed * 100_003 + epoch)   # deterministic resume
        n_tr = tr_sl.stop
        if loss == "mse":
            order = rng.permutation(n_tr)
            return [order[i:i + model.batch_size] for i in range(0, n_tr, model.batch_size)]
        gids = np.unique(g[:n_tr]); rng.shuffle(gids)
        batches, cur, cnt = [], [], 0
        for gid in gids:
            rows = np.nonzero(g[:n_tr] == gid)[0]
            cur.append(rows); cnt += len(rows)
            if cnt >= model.batch_size:
                batches.append(np.concatenate(cur)); cur, cnt = [], 0
        if cur:
            batches.append(np.concatenate(cur))
        return batches

    def _loss_on(rows):
        scores = model.net(Xt[rows]).squeeze(-1)
        if loss == "mse":
            return torch.mean((scores - yt[rows]) ** 2)
        return _listwise_loss(loss, scores, yt[rows], gt[rows], torch)

    epochs_run = start_epoch
    for epoch in range(start_epoch, epochs):
        model.net.train()
        tl = []
        for rows in _epoch_batches(epoch):
            if len(rows) < 2:
                continue
            opt.zero_grad()
            L = _loss_on(torch.tensor(rows, dtype=torch.long, device=dev))
            L.backward()
            opt.step()
            tl.append(float(L.detach()))
        model.net.eval()
        with torch.no_grad():
            if n_val > 0:
                vrows = torch.arange(va_sl.start, va_sl.stop, device=dev)
                vl = float(_loss_on(vrows))
            else:
                vl = float(np.mean(tl)) if tl else float("nan")
        hist["train_loss"].append(float(np.mean(tl)) if tl else float("nan"))
        hist["val_loss"].append(vl)
        if np.isfinite(vl) and vl < best_val - 1e-12:
            best_val, best_epoch, since_best = vl, epoch, 0
            best_state = model.state_arrays()
        else:
            since_best += 1
        epochs_run = epoch + 1
        if ckpt_path:
            save_checkpoint({"done_epochs": epoch + 1, "epochs_target": epochs,
                             "loss": loss, "best_epoch": best_epoch, "best_val": best_val,
                             "since_best": since_best,
                             "train_loss": hist["train_loss"], "val_loss": hist["val_loss"],
                             "model_state": model.state_arrays(),
                             "best_state": best_state or model.state_arrays()},
                            ckpt_path)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} train={hist['train_loss'][-1]:.6f} val={vl:.6f}")
        if model.patience and since_best >= model.patience:
            break
    if best_state is not None:
        model.load_state_arrays(best_state, n_feat=X.shape[-1])
        model.net.to(dev)
    return {"epochs_run": epochs_run, "best_epoch": best_epoch,
            "train_loss": hist["train_loss"], "val_loss": hist["val_loss"]}


# ----------------------------------------------------------------------------
# Path-B artifacts: model.pt + config.json (hyperparams, data range, preprocess stats)
# ----------------------------------------------------------------------------
def _artifact_paths(path):
    p = Path(str(path))
    if p.suffix == ".pt":
        return p, p.with_suffix(".json")
    return p / "model.pt", p / "config.json"


def save_model(model: "_TorchSeqModel", path, *, extra: dict | None = None) -> dict:
    """Persist a trained torch sequence model: state_dict -> model.pt, and a
    config.json holding hyperparameters, feature list/data range, and the TRAIN
    preprocessing statistics (2.6f: the stats ship with the model so inference
    always normalizes with what the model was fitted on).

    path: a directory (writes model.pt + config.json inside) or a *.pt file
    (writes the sibling <stem>.json). Returns {"model": ..., "config": ...} paths."""
    torch = _torch()
    if model.net is None:
        raise RuntimeError("save_model: model has no trained net")
    pt_path, cfg_path = _artifact_paths(path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.net.state_dict(), str(pt_path))
    cfg = model.config()
    cfg["n_feat"] = int(model._meta.get("n_feat") or 0)
    cfg["meta"] = {k: v for k, v in model._meta.items() if k != "n_feat"}
    if model._stats is not None:
        cfg["preprocess_stats"] = {"median": np.asarray(model._stats["median"]).tolist(),
                                   "mad": np.asarray(model._stats["mad"]).tolist(),
                                   "kind": model._stats.get("kind", "robust_zscore")}
    if extra:
        cfg.update(extra)
    cfg_path.write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    return {"model": str(pt_path), "config": str(cfg_path)}


def load_trained_seq_model(path) -> "_TorchSeqModel":
    """Load a path-B artifact saved by save_model (or scripts/train_seq_model.py):
    rebuilds the architecture from config.json, loads model.pt weights, restores the
    train preprocessing stats onto model._stats. Requires torch (clear ImportError
    with install hint otherwise)."""
    torch = _torch()
    pt_path, cfg_path = _artifact_paths(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config json next to model file: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cls = {"GRUModel": GRUModel, "LSTMModel": LSTMModel,
           "TransformerModel": TransformerModel}.get(cfg.get("class"))
    if cls is None:
        raise ValueError(f"unknown model class in config: {cfg.get('class')!r}")
    hyper = {k: cfg[k] for k in ("hidden", "n_layers", "dropout", "lr", "weight_decay",
                                 "epochs", "batch_size", "patience", "loss", "seed",
                                 "val_frac", "nhead") if k in cfg}
    if cls is not TransformerModel:
        hyper.pop("nhead", None)
    model = cls(**hyper)
    n_feat = int(cfg.get("n_feat") or 0)
    if n_feat <= 0:
        raise ValueError("config.json lacks n_feat; cannot rebuild architecture")
    model._meta["n_feat"] = n_feat
    model.net = model._build_net(n_feat)
    model.net.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    model.net.eval()
    st = cfg.get("preprocess_stats")
    if st:
        model._stats = {"median": np.asarray(st["median"], float),
                        "mad": np.asarray(st["mad"], float),
                        "kind": st.get("kind", "robust_zscore")}
    return model


SEQ_MODEL_REGISTRY = {"gru": GRUModel, "lstm": LSTMModel, "transformer": TransformerModel,
                      "seq_ridge": FlattenRidgeSeqModel}


# ----------------------------------------------------------------------------
# Sequence walk-forward backtest (parallel to models.ml_factor_backtest)
# ----------------------------------------------------------------------------
def ml_seq_backtest(data: dict, model: SequenceFactorModel | None = None, *,
                    seq_len: int = 60, horizon: int = 21, rebalance: str = "ME",
                    train_window: int = 252, min_train: int = 120,
                    top: float = 0.3, long_short: bool = False,
                    commission_bps: float = 1.0, slippage_bps: float = 1.0,
                    fields=DEFAULT_FIELDS) -> MLResult:
    """Sequence-model walk-forward backtest -- ml_factor_backtest's 3D sibling.

    At each rebalance date t: training samples are (window ending s, symbol) with
    label = forward return s -> s+horizon, restricted to the trailing `train_window`
    dates AND s+horizon strictly before t (purge: the label must be fully realized
    at decision time -- no look-ahead). RobustZScore stats are fitted on the TRAIN
    tensor only and applied to both train and the date-t cross-section (2.6f: test
    rows never touch normalization). A fresh model clone is fitted per rebalance,
    ranked into weights via the same rank_and_weight as the tabular path, and the
    realized Spearman IC per rebalance is reported alongside the cost-aware backtest.

    2.6f admission gate (write this into any report that uses a deep model): a
    sequence/deep model may enter the ensemble or be recommended ONLY if, on the
    SAME scorecard (same universe, same horizon, same costs), it beats the ridge
    baseline's RankICIR (FlattenRidgeSeqModel here, or models.RidgeModel on the
    tabular panel) AND survives core/validation.py Deflated Sharpe (DSR) correction with
    the deep-model trials counted in n_trials. Until it clears that bar it is a
    candidate under test, not an upgrade (FINSABER/StockBench evidence, roadmap v2->v3).

    Torch-backed models raise a clear ImportError (with the pip install hint) when
    torch is absent; numpy models (FlattenRidgeSeqModel) always run.
    """
    import copy as _copy

    from ..core import backtest as bt
    from ..core.rebalance import rebalance_dates as _rebal
    from ..strategies import multi_factor as mf

    model = model or FlattenRidgeSeqModel()
    close = mf.build_panel(data, "close")
    fwd = close.shift(-horizon) / close - 1.0
    rebal_dates = _rebal(close.index, rebalance)
    _loc = {d: i for i, d in enumerate(close.index)}

    panel = build_sequence_panel(data, seq_len=seq_len, fields=fields)
    X_all, idx_all = panel["X"], panel["index"]
    by_date: dict = {}
    for pos, (d, sym) in enumerate(idx_all):
        by_date.setdefault(d, []).append(pos)

    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    ic_list: list = []
    n_fits = 0
    for t in rebal_dates:
        hist = close.index[close.index <= t]
        if len(hist) < min_train:
            continue
        train_dates = hist[max(0, len(hist) - train_window):]
        t_loc = _loc[t]
        usable = [s for s in train_dates
                  if s in by_date and _loc[s] + horizon < t_loc]
        if len(usable) < max(20, min_train // 3) or t not in by_date:
            continue
        pos_tr, y_tr, g_tr = [], [], []
        for gi, s in enumerate(usable):
            for p in by_date[s]:
                yy = fwd.at[s, idx_all[p][1]]
                if np.isfinite(yy):
                    pos_tr.append(p); y_tr.append(float(yy)); g_tr.append(gi)
        if len(y_tr) < 20:
            continue
        stats = fit_stats(X_all[pos_tr])                    # TRAIN-only stats
        Xtr = apply_stats(X_all[pos_tr], stats)
        model_t = _copy.deepcopy(model)                     # fresh clone per fold
        model_t.fit(Xtr, np.asarray(y_tr), groups=np.asarray(g_tr))
        n_fits += 1

        pos_t = by_date[t]
        preds = model_t.predict(apply_stats(X_all[pos_t], stats))
        pred_s = pd.Series(np.nan, index=close.columns)
        for p, v in zip(pos_t, preds):
            pred_s[idx_all[p][1]] = v
        real = fwd.loc[t]
        both = pd.concat([pred_s, real], axis=1).dropna()
        if len(both) >= 3:
            ic_list.append(both.rank().corr().iloc[0, 1])
        w_row = mf.rank_and_weight(pred_s.to_frame().T, top=top,
                                   bottom=(top if long_short else 0.0),
                                   long_short=long_short)
        weights.loc[t] = w_row.iloc[0]

    weights = weights.reindex(close.index).ffill().fillna(0.0)
    res = bt.backtest_portfolio(close, weights, commission_bps=commission_bps,
                                slippage_bps=slippage_bps)
    ic = float(np.nanmean(ic_list)) if ic_list else float("nan")
    return MLResult(weights=weights, backtest=res, ic=ic,
                    feature_names=panel["feature_names"],
                    extra={"model": model.name, "seq_len": seq_len, "horizon": horizon,
                           "n_fits": n_fits, "n_ics": len(ic_list)})
