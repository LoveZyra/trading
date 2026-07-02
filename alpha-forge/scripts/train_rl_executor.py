#!/usr/bin/env python3
"""RL order-execution training entrypoint -- SKELETON (roadmap v3 SS2.21).

What this is
------------
The offline (local GPU) counterpart of trade/execution.py's executor bridge: train a
policy that decides per-slice child-order sizes to minimize implementation shortfall,
save it as policy.pt + config.json, and load it in the skill with
trade.execution.load_trained_executor(path) -> order_plan(..., executor=policy).

Evidence caveat (read before trusting any result)
-------------------------------------------------
The reference result -- RL beating TWAP/VWAP on IS -- comes from arXiv:2510.04952,
whose experiments run in the ABIDES multi-venue SIMULATOR, not live markets. Anything
trained here inherits that caveat: simulator IS improvements are a hypothesis about
live performance, not evidence of it. Report them as simulation results, benchmark
against the static TWAP/VWAP plans on the SAME simulated episodes, and treat live
deployment as its own validation problem (paper-trade IS via
execution_quality_report first).

Environment interface (what a real implementation must provide)
----------------------------------------------------------------
An ABIDES-style episodic env with:
  reset() -> state        state: dict/array with at least
                          {remaining_frac, time_frac, spread_bps, imbalance,
                           recent_return, participation_cap}
  step(action) -> (state, reward, done, info)
                          action: fraction of remaining shares for this slice [0,1]
                          reward: negative incremental implementation shortfall
                                  (bps, vs the episode's decision price), minus an
                                  impact penalty proportional to participation
  Episode = one parent order over N slices; fills modelled with temporary +
  permanent impact; unfilled remainder charged at final price (Perold opportunity
  cost) -- mirror implementation_shortfall() in trade/execution.py so the training
  objective equals the reported metric.
The included SimulatedExecutionEnv is a deliberately tiny random-walk + square-root
impact stand-in so the plumbing is testable; it is NOT a market model. For serious
work plug in ABIDES (https://github.com/abides-sim/abides) or your own LOB simulator.

Status: torch/gym parts are lazy imports; the actual training loop is intentionally
left for the local-GPU environment (this sandbox has no torch and a 45s exec cap).
Exit codes: 0 ok (skeleton run / dry plan), 2 torch missing, 1 bad arguments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

TORCH_INSTALL_MSG = (
    "ERROR: torch is not installed. RL executor training needs torch (and ideally an\n"
    "ABIDES-style simulator):  pip install torch gymnasium\n"
    "Run this on your local GPU machine (roadmap SS2.21); the skill only needs the\n"
    "resulting policy.pt + config.json via trade.execution.load_trained_executor().")


class SimulatedExecutionEnv:
    """Minimal random-walk execution env implementing the interface above.

    Price follows an arithmetic random walk; trading q shares in a slice costs a
    square-root temporary impact (impact_bps * sqrt(q / adv_slice)). Reward is the
    negative incremental IS in bps. This is plumbing-test quality, not a market model."""

    def __init__(self, *, n_slices: int = 8, total_shares: int = 10_000,
                 adv_per_slice: float = 100_000.0, vol_bps: float = 8.0,
                 impact_bps: float = 6.0, seed: int = 0):
        self.n_slices, self.total = n_slices, float(total_shares)
        self.adv, self.vol_bps, self.impact_bps = adv_per_slice, vol_bps, impact_bps
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.remaining = self.total
        self.price = 100.0
        self.decision_price = self.price
        self.cost_bps = 0.0
        return self._state()

    def _state(self):
        return {"remaining_frac": self.remaining / self.total,
                "time_frac": self.t / self.n_slices,
                "spread_bps": 5.0, "imbalance": 0.0,
                "recent_return": (self.price / self.decision_price - 1.0),
                "participation_cap": 0.1}

    def step(self, action: float):
        q = float(np.clip(action, 0.0, 1.0)) * self.remaining
        if self.t == self.n_slices - 1:
            q = self.remaining                                  # force completion
        impact = self.impact_bps * np.sqrt(max(q, 0.0) / self.adv)
        fill_px = self.price * (1 + impact / 1e4)
        inc_cost = (q / self.total) * (fill_px / self.decision_price - 1.0) * 1e4
        self.cost_bps += inc_cost
        self.remaining -= q
        self.price *= 1 + self._rng.normal(0.0, self.vol_bps / 1e4)
        self.t += 1
        done = self.t >= self.n_slices or self.remaining <= 0
        return self._state(), -inc_cost, done, {"filled": q, "is_bps": self.cost_bps}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_rl_executor.py",
        description="Train an RL execution policy offline (SKELETON -- see module docstring; "
                    "results are simulator-based, arXiv:2510.04952 / ABIDES caveat applies).")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--n-slices", type=int, default=8)
    p.add_argument("--total-shares", type=int, default=10_000)
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "reinforce"])
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True, help="directory for policy.pt + config.json")
    p.add_argument("--dry-run", action="store_true",
                   help="roll a TWAP baseline through the toy env and print its IS; no torch needed")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    env = SimulatedExecutionEnv(n_slices=args.n_slices, total_shares=args.total_shares,
                                seed=args.seed)
    if args.dry_run:                       # numpy-only sanity path (works in-sandbox)
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(1.0 / max(args.n_slices - env.t, 1))
        print(json.dumps({"baseline": "twap", "sim_is_bps": round(info["is_bps"], 4),
                          "note": "toy simulator, not market evidence"}, indent=1))
        return 0
    if importlib.util.find_spec("torch") is None:
        print(TORCH_INSTALL_MSG, file=sys.stderr)
        return 2
    # ------------------------------------------------------------------
    # Real training belongs on the local GPU box. Sketch (intentionally not
    # implemented here -- sandbox has no torch and a 45s cap):
    #   1. policy = nn.Sequential(Linear(4, hidden), Tanh(), Linear(hidden, 1))
    #      (4 state features, sigmoid output = fraction of remaining; MUST match
    #       trade.execution._TorchExecutionPolicy.decide's feature layout)
    #   2. optimize with --algo over SimulatedExecutionEnv or a real ABIDES env,
    #      checkpoint every K episodes (seq_models.save_checkpoint pattern)
    #   3. benchmark vs TWAP/VWAP on held-out episodes; only ship if better IS
    #   4. torch.save(policy.state_dict(), out/'policy.pt'); write config.json
    #      {"hidden": ..., "n_features": 4, "algo": ..., "sim": "toy|abides",
    #       "caveat": "simulator result (arXiv:2510.04952), not live evidence"}
    # ------------------------------------------------------------------
    print("Training loop not implemented in-skill: run on a local GPU following the "
          "sketch in this file, then load with trade.execution.load_trained_executor().",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
