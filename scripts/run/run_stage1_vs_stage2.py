#!/usr/bin/env python3
"""Stage 1 vs Stage 2 grammar comparison experiment.

Runs AlphaZero training on both the masked grammar (Stage 1) and the
unmasked grammar (Stage 2) across D=3,4,5 with 2 seeds each.

Total: 2 stages × 3 D-values × 2 seeds = 12 runs.

Both stages use identical hyperparameters (from Stage 1 diagnostics):
  - Dirichlet epsilon = 0.10 (best single-param fix)
  - Temperature tau = 0.50 (sharpens MCTS targets)
  - Replay buffer = 10 past iterations (reduce stale targets)
  - 50 iterations, 80 MCTS sims, 30 games/iter

Usage:
    python scripts/run/run_stage1_vs_stage2.py
    python scripts/run/run_stage1_vs_stage2.py --d-values 3 4
    python scripts/run/run_stage1_vs_stage2.py --stages 2
    python scripts/run/run_stage1_vs_stage2.py --seeds 42
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.doors.dsl.explicit_surface_derivation_config import (
    DoorsExplicitSurfaceDerivationConfig,
)
from alphazeropp.instances.doors.dsl.unmasked_surface_derivation_config import (
    DoorsUnmaskedSurfaceDerivationConfig,
)
from alphazeropp.utils.derivation_utils import run_derivation_training

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

D_VALUES = [3, 4, 5]
SEEDS = [42, 137]
N_ITERATIONS = 50


def compute_optimal_reward(num_rooms, step_penalty=0.01, unlock_bonus=0.1):
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def make_config(stage: int, D: int, seed: int):
    """Create a config for the given stage, D value, and seed.

    Both stages use the same tuned hyperparameters so the only
    difference is the grammar (masked vs unmasked).
    """
    if stage == 1:
        cfg = DoorsExplicitSurfaceDerivationConfig(num_rooms=D)
        mode_label = "stage1_masked"
    else:
        cfg = DoorsUnmaskedSurfaceDerivationConfig(
            num_rooms=D, exact_length=True,
        )
        mode_label = "stage2_unmasked"

    # Override seeds
    cfg.agent.random_seeds = {
        "mcts": seed,
        "train": seed + 1,
        "eval": seed + 2,
        "external_policy": seed + 3,
    }

    # Ensure consistent hyperparameters across both stages
    cfg.agent.mcts_params["n_simulations"] = 80
    cfg.agent.mcts_params["temperature"] = 0.5
    cfg.agent.mcts_params["c_exploration"] = 1.5
    cfg.agent.mcts_params["dirichlet_alpha"] = 0.25
    cfg.agent.mcts_params["dirichlet_epsilon"] = 0.10
    cfg.trainer.n_games_per_train = 30
    cfg.trainer.n_past_iterations_to_train = 10
    cfg.run.n_iterations = N_ITERATIONS
    cfg.run.plot_every = 5
    cfg.run.accept_threshold = 0.40

    return cfg, mode_label


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(stage: int, D: int, seed: int, exp_dir: Path):
    """Run one training experiment and return summary stats."""
    cfg, mode_label = make_config(stage, D, seed)
    K = D - 1

    run_dir = exp_dir / f"stage{stage}_D{D}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.run.plot_path = str(run_dir / "training_metrics.png")
    cfg.save(str(run_dir / "config.json"))

    print(f"\n{'='*70}")
    print(f"  Stage {stage} | D={D} (K={K}) | Seed={seed}")
    print(f"  Grammar: {mode_label}")
    print(f"  Language size: ", end="")
    if stage == 1:
        import math
        lang_size = math.factorial(2 * K) // (2 ** K)
        print(f"|L(G1)| = {lang_size:,}")
    else:
        lang_size = (2 * K) ** (2 * K) if K > 0 else 1
        print(f"|L=(G2)| = {lang_size:,}")
    print(f"  Hyperparameters: eps=0.10, tau=0.50, buffer=10, "
          f"sims=80, games=30, iters={N_ITERATIONS}")
    print(f"  Output: {run_dir}/")
    print(f"{'='*70}\n")

    optimal_reward = compute_optimal_reward(D)
    t0 = time.time()
    run_derivation_training(cfg, mode_label, optimal_reward, run_dir)
    wall_time = time.time() - t0

    # Read final stats
    summary = {
        "stage": stage,
        "D": D,
        "K": K,
        "seed": seed,
        "mode": mode_label,
        "n_iterations": N_ITERATIONS,
        "wall_time_s": round(wall_time, 1),
    }

    stats_path = run_dir / "train_stats.jsonl"
    if stats_path.exists():
        with open(stats_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if entries:
            last = entries[-1]
            summary["final_policy_loss"] = last.get("train_loss_policy")
            summary["final_value_loss"] = last.get("train_loss_value")
            summary["final_mcts_entropy"] = last.get("mcts_target_entropy")
            summary["final_kl_gap"] = last.get("policy_kl_gap")
            summary["final_avg_reward"] = last.get("avg_reward")

    prog_path = run_dir / "program_log.jsonl"
    if prog_path.exists():
        with open(prog_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if entries:
            # Find first-solve iteration
            for entry in entries:
                if entry.get("best_solve_rate", 0) >= 1.0:
                    summary["first_solve_iter"] = entry["iteration"]
                    break
            last = entries[-1]
            summary["final_solve_rate"] = last.get("best_solve_rate", 0)
            summary["final_best_reward"] = last.get("best_avg_reward", 0)
            summary["unique_programs"] = last.get("unique_programs", 0)

    return summary


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def print_comparison(summaries, exp_dir):
    """Print and save a comparison table."""
    print(f"\n\n{'='*90}")
    print(f"  STAGE 1 vs STAGE 2 — TRAINING COMPARISON RESULTS")
    print(f"{'='*90}\n")

    # Group by D
    for D in sorted(set(s["D"] for s in summaries)):
        print(f"  D = {D} (K = {D-1})")
        print(f"  {'Stage':>6}  {'Seed':>5}  {'1st Solve':>9}  "
              f"{'Solve%':>7}  {'Reward':>8}  {'L_pi':>7}  "
              f"{'L_v':>7}  {'D_KL':>7}  {'Time':>8}")
        print(f"  {'-----':>6}  {'----':>5}  {'---------':>9}  "
              f"{'------':>7}  {'------':>8}  {'----':>7}  "
              f"{'---':>7}  {'----':>7}  {'----':>8}")

        for s in sorted(
            [x for x in summaries if x["D"] == D],
            key=lambda x: (x["stage"], x["seed"]),
        ):
            first_solve = s.get("first_solve_iter", "never")
            if isinstance(first_solve, int):
                first_solve = f"iter {first_solve}"
            solve_rate = s.get("final_solve_rate", 0)
            reward = s.get("final_best_reward", 0)
            p_loss = s.get("final_policy_loss", float("nan"))
            v_loss = s.get("final_value_loss", float("nan"))
            kl = s.get("final_kl_gap", float("nan"))
            wt = s.get("wall_time_s", 0)

            print(f"  {s['stage']:>6}  {s['seed']:>5}  {first_solve:>9}  "
                  f"{solve_rate:>6.0%}  {reward:>+8.4f}  "
                  f"{p_loss:>7.4f}  {v_loss:>7.4f}  "
                  f"{kl:>7.4f}  {wt:>7.0f}s")
        print()

    # Aggregate: mean across seeds per (stage, D)
    print(f"  {'— AGGREGATED (mean over seeds) —':^90}")
    print(f"  {'Stage':>6}  {'D':>3}  {'1st Solve':>10}  "
          f"{'Solve%':>7}  {'Reward':>8}  {'L_pi':>7}  "
          f"{'L_v':>7}  {'D_KL':>7}")
    print(f"  {'-----':>6}  {'---':>3}  {'----------':>10}  "
          f"{'------':>7}  {'------':>8}  {'----':>7}  "
          f"{'---':>7}  {'----':>7}")

    for stage in [1, 2]:
        for D in sorted(set(s["D"] for s in summaries)):
            group = [s for s in summaries
                     if s["stage"] == stage and s["D"] == D]
            if not group:
                continue

            first_solves = [s.get("first_solve_iter", float("inf"))
                           for s in group]
            avg_first = np.mean([x for x in first_solves if x != float("inf")])
            if np.isnan(avg_first):
                first_str = "never"
            else:
                first_str = f"iter {avg_first:.0f}"

            solve = np.mean([s.get("final_solve_rate", 0) for s in group])
            reward = np.mean([s.get("final_best_reward", 0) for s in group])
            p_loss = np.mean([s.get("final_policy_loss", float("nan"))
                             for s in group])
            v_loss = np.mean([s.get("final_value_loss", float("nan"))
                             for s in group])
            kl = np.mean([s.get("final_kl_gap", float("nan"))
                         for s in group])

            print(f"  {stage:>6}  {D:>3}  {first_str:>10}  "
                  f"{solve:>6.0%}  {reward:>+8.4f}  "
                  f"{p_loss:>7.4f}  {v_loss:>7.4f}  "
                  f"{kl:>7.4f}")

    print(f"\n{'='*90}")
    print(f"  Results saved to: {exp_dir}/")
    print(f"{'='*90}\n")


def plot_comparison(summaries, exp_dir):
    """Generate comparison plots across stages and D values."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Warning] matplotlib/pandas not installed. Skipping plots.")
        return

    # Load per-iteration training stats for learning curves
    curves = {}
    for s in summaries:
        key = (s["stage"], s["D"], s["seed"])
        stats_path = (exp_dir / f"stage{s['stage']}_D{s['D']}_seed{s['seed']}"
                      / "train_stats.jsonl")
        if stats_path.exists():
            with open(stats_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            curves[key] = records

    if not curves:
        return

    d_values = sorted(set(s["D"] for s in summaries))
    fig, axes = plt.subplots(2, len(d_values), figsize=(5 * len(d_values), 8))
    if len(d_values) == 1:
        axes = axes.reshape(-1, 1)

    colors = {1: "tab:blue", 2: "tab:orange"}
    labels = {1: "Stage 1 (masked)", 2: "Stage 2 (unmasked)"}

    for col, D in enumerate(d_values):
        # Top row: policy loss
        ax_p = axes[0, col]
        ax_p.set_title(f"D={D} — Policy Loss")
        ax_p.set_xlabel("Iteration")
        ax_p.set_ylabel("Policy Loss")

        # Bottom row: value loss
        ax_v = axes[1, col]
        ax_v.set_title(f"D={D} — Value Loss")
        ax_v.set_xlabel("Iteration")
        ax_v.set_ylabel("Value Loss")

        for stage in [1, 2]:
            for seed in sorted(set(s["seed"] for s in summaries)):
                key = (stage, D, seed)
                if key not in curves:
                    continue
                records = curves[key]
                iters = [r.get("iteration", i + 1) for i, r in enumerate(records)]
                p_loss = [r.get("train_loss_policy", float("nan"))
                         for r in records]
                v_loss = [r.get("train_loss_value", float("nan"))
                         for r in records]

                alpha = 0.4 if seed != sorted(set(s["seed"] for s in summaries))[0] else 0.8
                ax_p.plot(iters, p_loss, color=colors[stage],
                         alpha=alpha, linewidth=1.5,
                         label=labels[stage] if seed == sorted(set(s["seed"] for s in summaries))[0] else None)
                ax_v.plot(iters, v_loss, color=colors[stage],
                         alpha=alpha, linewidth=1.5,
                         label=labels[stage] if seed == sorted(set(s["seed"] for s in summaries))[0] else None)

        ax_p.legend(fontsize=8)
        ax_v.legend(fontsize=8)
        ax_p.grid(True, alpha=0.3)
        ax_v.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = exp_dir / "stage1_vs_stage2_learning_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {plot_path}")

    # Program quality comparison
    fig2, axes2 = plt.subplots(1, len(d_values), figsize=(5 * len(d_values), 4))
    if len(d_values) == 1:
        axes2 = [axes2]

    for col, D in enumerate(d_values):
        ax = axes2[col]
        ax.set_title(f"D={D} — Best Solve Rate")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Solve Rate")

        for stage in [1, 2]:
            for seed in sorted(set(s["seed"] for s in summaries)):
                prog_path = (exp_dir / f"stage{stage}_D{D}_seed{seed}"
                            / "program_log.jsonl")
                if not prog_path.exists():
                    continue
                with open(prog_path) as f:
                    records = [json.loads(line) for line in f if line.strip()]
                iters = [r["iteration"] for r in records]
                sr = [r.get("best_solve_rate", 0) for r in records]
                alpha = 0.4 if seed != sorted(set(s["seed"] for s in summaries))[0] else 0.8
                ax.plot(iters, sr, color=colors[stage],
                       alpha=alpha, linewidth=1.5,
                       label=labels[stage] if seed == sorted(set(s["seed"] for s in summaries))[0] else None)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plot_path2 = exp_dir / "stage1_vs_stage2_solve_rates.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Saved to {plot_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 vs Stage 2 grammar training comparison")
    parser.add_argument("--d-values", type=int, nargs="+", default=D_VALUES,
                        help=f"D values to test (default: {D_VALUES})")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Random seeds (default: {SEEDS})")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2],
                        help="Stages to run (default: 1 2)")
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS,
                        help=f"Training iterations (default: {N_ITERATIONS})")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    global N_ITERATIONS
    N_ITERATIONS = args.iterations

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d_str = "_".join(str(d) for d in args.d_values)
    exp_dir = Path(f"experiments/stage1_vs_stage2_{timestamp}_D{d_str}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    exp_config = {
        "d_values": args.d_values,
        "seeds": args.seeds,
        "stages": args.stages,
        "n_iterations": N_ITERATIONS,
        "hyperparameters": {
            "n_simulations": 80,
            "temperature": 0.50,
            "dirichlet_alpha": 0.25,
            "dirichlet_epsilon": 0.10,
            "n_games_per_train": 30,
            "n_past_iterations_to_train": 10,
            "accept_threshold": 0.40,
        },
    }
    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    total_runs = len(args.stages) * len(args.d_values) * len(args.seeds)
    print(f"\n{'#'*70}")
    print(f"  STAGE 1 vs STAGE 2 GRAMMAR COMPARISON")
    print(f"  {total_runs} runs: stages={args.stages}, "
          f"D={args.d_values}, seeds={args.seeds}")
    print(f"  {N_ITERATIONS} iterations per run")
    print(f"  Output: {exp_dir}/")
    print(f"{'#'*70}\n")

    summaries = []
    run_idx = 0
    for stage in args.stages:
        for D in args.d_values:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n>>> Run {run_idx}/{total_runs}: "
                      f"Stage {stage}, D={D}, seed={seed}")

                summary = run_single(stage, D, seed, exp_dir)
                summaries.append(summary)

                # Save incrementally
                with open(exp_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(summary) + "\n")

    # Final comparison
    print_comparison(summaries, exp_dir)
    plot_comparison(summaries, exp_dir)

    # Save full summary
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
