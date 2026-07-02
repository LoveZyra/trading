#!/usr/bin/env python3
"""Path-B training entrypoint for deep sequence stock-selection models (roadmap SS2.6c).

Run this on a machine WITH torch (ideally a GPU box), then copy the output artifacts
(model.pt + config.json + metrics.json) back into the skill and load them with
scripts.research.seq_models.load_trained_seq_model for inference-only use.

    python scripts/train_seq_model.py --model gru --data data/panel.pkl \
        --seq-len 60 --horizon 21 --loss mse --device auto --output models/gru_us

Data format: --data accepts a pickle of {symbol: OHLCV DataFrame} (canonical columns
open/high/low/close/volume, DatetimeIndex) or a directory of <SYMBOL>.csv files with
a date index column. Labels are forward `--horizon` returns; the trailing 20% of
samples (time-ordered) are the early-stopping validation split.

Honesty (2.6f): RobustZScore stats are fitted on the TRAINING samples only and are
persisted inside config.json so online inference normalizes identically. A model
trained here is a CANDIDATE: it must beat the ridge baseline's RankICIR on the same
scorecard and survive DSR correction before entering any ensemble.

Resume (2.6g): --resume continues from the epoch checkpoint (<output>/ckpt.{json,npz})
written after every epoch, so an interrupted run picks up where it stopped.

Exit codes: 0 ok, 2 torch missing (or argparse usage error), 1 bad data/arguments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

TORCH_INSTALL_MSG = (
    "ERROR: torch is not installed, and training a sequence model requires it.\n"
    "Install with:  pip install torch   (CUDA builds: https://pytorch.org/get-started)\n"
    "This script is the roadmap SS2.6 path-B entrypoint -- run it on your local/GPU\n"
    "machine, then load the saved model.pt in the skill with load_trained_seq_model().")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_seq_model.py",
        description="Train a GRU/LSTM/Transformer stock-ranking model (path B, offline).")
    p.add_argument("--model", required=True, choices=["gru", "lstm", "transformer"],
                   help="architecture (hidden=64, 2 layers by default)")
    p.add_argument("--data", required=True,
                   help="pickle of {symbol: OHLCV DataFrame} or a directory of CSVs")
    p.add_argument("--seq-len", type=int, default=60, help="lookback window length")
    p.add_argument("--horizon", type=int, default=21, help="forward-return label horizon")
    p.add_argument("--loss", default="mse", choices=["mse", "listnet", "listmle"],
                   help="pointwise MSE or listwise (grouped by date)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0|mps")
    p.add_argument("--output", required=True,
                   help="output directory (writes model.pt, config.json, metrics.json)")
    p.add_argument("--resume", action="store_true",
                   help="resume from <output>/ckpt if present (2.6g checkpointing)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="cap training samples (0 = no cap); newest samples are kept")
    return p


def load_panel_data(path: str) -> dict:
    """--data loader: .pkl/.pickle of {symbol: DataFrame}, or a directory of CSVs."""
    import pandas as pd
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--data path does not exist: {p}")
    if p.is_dir():
        data = {}
        for f in sorted(p.glob("*.csv")):
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            data[f.stem.upper()] = df
        if not data:
            raise ValueError(f"no *.csv files found in directory {p}")
        return data
    if p.suffix in (".pkl", ".pickle"):
        import pickle
        with open(p, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{p} must contain a non-empty dict of DataFrames")
        return data
    raise ValueError(f"unsupported --data format {p.suffix!r} (want .pkl/.pickle or a CSV dir)")


def prepare_dataset(data: dict, *, seq_len: int, horizon: int):
    """Numpy-only sample building (works without torch): sequences + forward-return
    labels + date-group ids, time-ordered, with TRAIN-fitted RobustZScore stats."""
    import numpy as np
    from scripts.research import seq_models as sm
    panel = sm.build_sequence_panel(data, seq_len=seq_len)
    X, index = panel["X"], panel["index"]
    if len(X) == 0:
        raise ValueError("no usable sequence samples (series shorter than --seq-len, "
                         "missing OHLCV columns, or all-NaN windows)")
    labels, keep, gids = [], [], []
    date_id = {}
    for pos, (d, sym) in enumerate(index):
        df = data[sym]
        loc = df.index.get_loc(d)
        if loc + horizon >= len(df):
            continue
        c0, c1 = float(df["close"].iloc[loc]), float(df["close"].iloc[loc + horizon])
        if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0:
            continue
        keep.append(pos)
        labels.append(c1 / c0 - 1.0)
        gids.append(date_id.setdefault(d, len(date_id)))
    if len(keep) < 50:
        raise ValueError(f"only {len(keep)} labelled samples -- need >=50; supply more "
                         f"history or lower --seq-len/--horizon")
    order = np.argsort(gids, kind="stable")               # time-ordered for the val split
    keep = np.asarray(keep)[order]
    y = np.asarray(labels, float)[order]
    g = np.asarray(gids)[order]
    X = X[keep]
    stats = sm.fit_stats(X[: int(len(X) * 0.8)] if len(X) >= 50 else X)   # train rows only
    X = sm.apply_stats(X, stats)
    dates = sorted(date_id, key=date_id.get)
    return sm.SequenceDataset(X, y, g, index=[index[i] for i in np.asarray(keep)]), stats, {
        "n_samples": int(len(y)), "n_symbols": len(data),
        "date_start": str(dates[0]), "date_end": str(dates[-1]),
        "seq_len": seq_len, "horizon": horizon}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = load_panel_data(args.data)
        dataset, stats, data_range = prepare_dataset(data, seq_len=args.seq_len,
                                                     horizon=args.horizon)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.max_samples and len(dataset) > args.max_samples:
        sl = slice(len(dataset) - args.max_samples, None)
        from scripts.research.seq_models import SequenceDataset
        dataset = SequenceDataset(dataset.X[sl], dataset.y[sl], dataset.groups[sl],
                                  index=dataset.index[sl])
    print(f"data: {data_range['n_samples']} samples / {data_range['n_symbols']} symbols "
          f"({data_range['date_start']} .. {data_range['date_end']})")

    if importlib.util.find_spec("torch") is None:          # AFTER data checks: loading
        print(TORCH_INSTALL_MSG, file=sys.stderr)          # path stays testable w/o torch
        return 2

    from scripts.research import seq_models as sm
    cls = {"gru": sm.GRUModel, "lstm": sm.LSTMModel, "transformer": sm.TransformerModel}[args.model]
    model = cls(hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
                lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
                batch_size=args.batch_size, loss=args.loss,
                device=None if args.device == "auto" else args.device)
    model._stats = stats
    model._meta["data_range"] = data_range
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics = sm.train_loop(model, dataset, loss=args.loss, epochs=args.epochs,
                            ckpt_path=out / "ckpt", resume=args.resume, verbose=True)
    paths = sm.save_model(model, out, extra={"cli_args": vars(args), "data_range": data_range})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float),
                                      encoding="utf-8")
    print(f"saved: {paths['model']}, {paths['config']}, {out / 'metrics.json'}")
    print("Reminder (2.6f): benchmark this model against the ridge baseline's RankICIR "
          "on the SAME scorecard + DSR correction before using it in any ensemble.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
