#!/usr/bin/env python3
"""Run matched macro vs surface derivation experiments for comparison.

Runs 4 experiments with identical MCTS hyperparameters:
  - Surface D8, Surface D10
  - Macro D8, Macro D10

All use: 80 MCTS sims, 30 games/iter, 10 iterations, seed 42.

Usage:
    PYTHONPATH=src python scripts/run_macro_vs_surface_comparison.py
    PYTHONPATH=src python scripts/run_macro_vs_surface_comparison.py --D 10
    PYTHONPATH=src python scripts/run_macro_vs_surface_comparison.py --skip-surface
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from alphazeropp.instances.doors.dsl.derivation_config import (
    DoorsFactoredD10MacroConfig,
)
from alphazeropp.instances.doors.dsl.surface_derivation_config import (
    DoorsSurfaceDerivationConfig,
)
from alphazeropp.instances.doors.dsl.doors_config import (
    compute_doors_derived_params,
)
from alphazeropp.instances.doors.oracle import optimal_return
from alphazeropp.utils.derivation_utils import run_derivation_training


# ---------------------------------------------------------------------------
# Shared hyperparameters (matched across all experiments)
# ---------------------------------------------------------------------------
N_SIMULATIONS = 80
N_GAMES = 30
N_ITERATIONS = 10
SEED = 42

MCTS_PARAMS = {
    "n_simulations": N_SIMULATIONS,
    "temperature": 1.0,
    "c_exploration": 1.5,
    "dirichlet_alpha": 0.25,
    "dirichlet_epsilon": 0.40,
    "rollout_n": 4,
    "rollout_mode": "max",
    "rollout_blend": 0.3,
    "rollout_budget": 200,
    "backup_rule": "max",
    "backup_topk": 3,
    "backup_tau": 0.1,
}


def _compute_optimal_reward(num_rooms, step_penalty=0.01, unlock_bonus=0.1):
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def _configure_surface(D):
    """Create a surface derivation config for D rooms."""
    cfg = DoorsSurfaceDerivationConfig(num_rooms=D)
    cfg.agent.mcts_params = dict(MCTS_PARAMS)
    cfg.trainer.n_games_per_train = N_GAMES
    cfg.run.n_iterations = N_ITERATIONS
    cfg.agent.random_seeds = {
        "mcts": SEED, "train": SEED + 1,
        "eval": SEED + 2, "external_policy": SEED + 3,
    }
    return cfg


def _configure_macro(D):
    """Create a macro derivation config for D rooms."""
    cfg = DoorsFactoredD10MacroConfig()
    # Override num_rooms and recompute derived params
    derived = compute_doors_derived_params(D, 2)
    cfg.game.kwargs.update({
        "num_rooms": D,
        "locs_per_room": 2,
        "n_sites": derived["n_sites"],
        "budget": derived["budget"],
        "horizon": derived["horizon"],
    })
    cfg.net.kwargs.update({
        "budget": derived["budget"],
        "n_sites": derived["n_sites"],
    })
    cfg.agent.mcts_params = dict(MCTS_PARAMS)
    cfg.trainer.n_games_per_train = N_GAMES
    cfg.run.n_iterations = N_ITERATIONS
    cfg.agent.random_seeds = {
        "mcts": SEED, "train": SEED + 1,
        "eval": SEED + 2, "external_policy": SEED + 3,
    }
    return cfg


def run_experiment(label, cfg, mode, exp_dir):
    """Run a single experiment and save config."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")
    cfg.save(str(exp_dir / "config.json"))

    optimal_reward = _compute_optimal_reward(cfg.game.kwargs["num_rooms"])

    print()
    print("=" * 80)
    print(f"  EXPERIMENT: {label}")
    print(f"  D={cfg.game.kwargs['num_rooms']} | "
          f"sims={cfg.agent.mcts_params['n_simulations']} | "
          f"games={cfg.trainer.n_games_per_train} | "
          f"iters={cfg.run.n_iterations} | seed={SEED}")
    if "budget" in cfg.game.kwargs:
        print(f"  budget={cfg.game.kwargs['budget']} | "
              f"n_sites={cfg.game.kwargs['n_sites']}")
    print(f"  Output: {exp_dir}")
    print("=" * 80)
    print()

    run_derivation_training(cfg, mode, optimal_reward, exp_dir)
    print(f"\n[DONE] {label} complete.\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Macro vs Surface derivation comparison")
    parser.add_argument("--D", nargs="+", type=int, default=[8, 10],
                        help="Door counts to compare (default: 8 10)")
    parser.add_argument("--skip-surface", action="store_true",
                        help="Skip surface experiments")
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip macro experiments")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("experiments") / "macro_vs_surface_comparison" / timestamp

    # Store comparison metadata
    meta = {
        "D_values": args.D,
        "n_simulations": N_SIMULATIONS,
        "n_games": N_GAMES,
        "n_iterations": N_ITERATIONS,
        "seed": SEED,
        "mcts_params": MCTS_PARAMS,
        "timestamp": timestamp,
    }
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / "comparison_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    for D in args.D:
        if not args.skip_surface:
            cfg = _configure_surface(D)
            exp_dir = base_dir / f"surface_D{D}"
            run_experiment(f"Surface D{D}", cfg, "surface", exp_dir)

        if not args.skip_macro:
            cfg = _configure_macro(D)
            exp_dir = base_dir / f"macro_D{D}"
            run_experiment(f"Macro D{D}", cfg, "doors_d10_macro", exp_dir)

    print()
    print("=" * 80)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Results: {base_dir}")
    print()
    print("  Next: generate comparison plots with:")
    print(f"    PYTHONPATH=src python scripts/plot_macro_vs_surface.py {base_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
