"""Diagnostic experiments for grammar-game policy loss decomposition.

Runs 6 controlled experiments at D=5 (K=4), varying one parameter at a time,
to determine which factor most affects network convergence.

All experiments log mcts_target_entropy and policy_kl_gap in train_stats.jsonl,
enabling decomposition of L_policy = H(pi_MCTS) + D_KL.

Usage:
    python scripts/run/run_diagnostic_experiments.py
    python scripts/run/run_diagnostic_experiments.py --experiments 0 1 2
    python scripts/run/run_diagnostic_experiments.py --d 5 --experiments 0
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from alphazeropp.instances.doors.dsl.explicit_surface_derivation_config import (
    DoorsExplicitSurfaceDerivationConfig,
)
from alphazeropp.utils.derivation_utils import run_derivation_training

logger = logging.getLogger(__name__)

# Experiment definitions: (name, overrides_dict, n_iterations)
EXPERIMENTS = [
    ("baseline_h", {}, 15),
    ("long_run", {}, 100),
    ("more_data", {"n_games_per_train": 100}, 15),
    ("less_noise", {"dirichlet_epsilon": 0.10}, 15),
    ("lower_temp", {"temperature": 0.5}, 15),
    ("combined", {"dirichlet_epsilon": 0.10, "temperature": 0.5}, 100),
]


def compute_optimal_reward(cfg):
    gk = cfg.game.kwargs
    num_rooms = gk["num_rooms"]
    step_penalty = gk.get("step_penalty", 0.01)
    unlock_bonus = gk.get("unlock_bonus", 0.1)
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def apply_overrides(cfg, overrides):
    """Apply parameter overrides to a config."""
    for key, value in overrides.items():
        if key == "n_games_per_train":
            cfg.trainer.n_games_per_train = value
        elif key == "n_simulations":
            cfg.agent.mcts_params["n_simulations"] = value
        elif key == "dirichlet_epsilon":
            cfg.agent.mcts_params["dirichlet_epsilon"] = value
        elif key == "temperature":
            cfg.agent.mcts_params["temperature"] = value
        else:
            raise ValueError(f"Unknown override key: {key}")


def run_experiment(exp_idx, D, exp_dir):
    """Run a single experiment."""
    name, overrides, n_iterations = EXPERIMENTS[exp_idx]

    cfg = DoorsExplicitSurfaceDerivationConfig(num_rooms=D)
    cfg.run.n_iterations = n_iterations
    cfg.run.plot_every = max(1, n_iterations // 5)
    apply_overrides(cfg, overrides)

    K = D - 1
    override_str = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "(baseline)"

    print(f"\n{'='*60}")
    print(f"  Experiment {exp_idx}: {name}")
    print(f"  D={D}, K={K}, {n_iterations} iterations")
    print(f"  Overrides: {override_str}")
    print(f"  MCTS sims={cfg.agent.mcts_params['n_simulations']}, "
          f"games={cfg.trainer.n_games_per_train}, "
          f"eps={cfg.agent.mcts_params['dirichlet_epsilon']}, "
          f"tau={cfg.agent.mcts_params['temperature']}")
    print(f"{'='*60}\n")

    run_dir = exp_dir / f"exp{exp_idx}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.run.plot_path = str(run_dir / "training_metrics.png")
    cfg.save(str(run_dir / "config.json"))

    optimal_reward = compute_optimal_reward(cfg)
    run_derivation_training(cfg, "explicit_surface", optimal_reward, run_dir)

    # Read final stats
    stats_path = run_dir / "train_stats.jsonl"
    if stats_path.exists():
        with open(stats_path) as f:
            entries = [json.loads(line) for line in f]
        if entries:
            last = entries[-1]
            return {
                "exp_idx": exp_idx,
                "name": name,
                "D": D,
                "n_iterations": n_iterations,
                "overrides": overrides,
                "final_policy_loss": last.get("train_loss_policy"),
                "final_value_loss": last.get("train_loss_value"),
                "final_mcts_entropy": last.get("mcts_target_entropy"),
                "final_kl_gap": last.get("policy_kl_gap"),
                "final_avg_reward": last.get("avg_reward"),
            }
    return {"exp_idx": exp_idx, "name": name, "error": "no stats"}


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic experiments for grammar-game policy loss"
    )
    parser.add_argument("--d", type=int, default=5,
                        help="D value (default: 5)")
    parser.add_argument("--experiments", type=int, nargs="+",
                        default=list(range(len(EXPERIMENTS))),
                        help="Experiment indices to run (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/diagnostic_D{args.d}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump({"D": args.d, "experiments": args.experiments,
                    "definitions": [(n, o, i) for n, o, i in EXPERIMENTS]}, f, indent=2)

    summaries = []
    for idx in args.experiments:
        summary = run_experiment(idx, args.d, exp_dir)
        summaries.append(summary)
        with open(exp_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")

    # Print comparison
    print(f"\n\n{'='*80}")
    print(f"  DIAGNOSTIC EXPERIMENT RESULTS (D={args.d})")
    print(f"{'='*80}\n")
    print(f"  {'#':>2}  {'Name':<15}  {'L_policy':>8}  {'H_MCTS':>8}  "
          f"{'D_KL':>8}  {'L_value':>8}  {'Reward':>8}")
    print(f"  {'--':>2}  {'----':<15}  {'--------':>8}  {'------':>8}  "
          f"{'----':>8}  {'-------':>8}  {'------':>8}")
    for s in summaries:
        if "error" in s:
            print(f"  {s['exp_idx']:>2}  {s['name']:<15}  ERROR")
            continue
        print(f"  {s['exp_idx']:>2}  {s['name']:<15}  "
              f"{s['final_policy_loss']:>8.4f}  "
              f"{s['final_mcts_entropy']:>8.4f}  "
              f"{s['final_kl_gap']:>8.4f}  "
              f"{s['final_value_loss']:>8.4f}  "
              f"{s['final_avg_reward']:>8.4f}")

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nResults saved to: {exp_dir}/")


if __name__ == "__main__":
    main()
