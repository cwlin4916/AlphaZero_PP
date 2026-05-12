#!/usr/bin/env python3
"""Systematic epsilon x tau hyperparameter sweep.

Phases:
  1a  Epsilon sweep at fixed tau=0.5
  1b  Tau sweep at fixed eps=0.10
  1c  Diagonal (off-axis) combinations
  2   Fine grid around Phase 1 winner
  3   Simulation and game budget sweep
  4   D=6 validation of top-3 configs

Default screening: D=5, Stage 2 (unmasked grammar), 25 iterations, 80 sims, 50 games/iter.

Usage:
    python scripts/run/run_epsilon_tau_sweep.py --phase 1a
    python scripts/run/run_epsilon_tau_sweep.py --phase 1a 1b 1c
    python scripts/run/run_epsilon_tau_sweep.py --phase 1a 1b 1c --stage 1   # use masked grammar
    python scripts/run/run_epsilon_tau_sweep.py --phase 2 --best-eps 0.05 --best-tau 0.35
    python scripts/run/run_epsilon_tau_sweep.py --phase 3 --best-eps 0.05 --best-tau 0.35
    python scripts/run/run_epsilon_tau_sweep.py --phase 4 --best-eps 0.05 --best-tau 0.35
    python scripts/run/run_epsilon_tau_sweep.py --phase 1a --resume experiments/epsilon_tau_sweep_...
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import csv
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
# Defaults
# ---------------------------------------------------------------------------

SCREENING_D = 5
VALIDATION_D = 6
SCREENING_ITERS = 25
VALIDATION_ITERS = 50
DEFAULT_SIMS = 80
DEFAULT_GAMES = 50
SEEDS_PHASE1 = [42, 137]
SEEDS_PHASE2 = [42, 137, 271]
SEEDS_PHASE3 = [42, 137]
SEEDS_PHASE4 = [42, 137, 271]

# Current best hyperparameters (anchor)
ANCHOR_EPS = 0.10
ANCHOR_TAU = 0.50


def compute_optimal_reward(num_rooms, step_penalty=0.01, unlock_bonus=0.1):
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


# ---------------------------------------------------------------------------
# Phase config generators
# ---------------------------------------------------------------------------

def phase1a_configs():
    """Epsilon sweep at fixed tau=0.5 (only eps=0.00 and 0.03).

    Based on partial results: eps=0.00 dominates (first-solve iter 2),
    higher epsilon uniformly worse. Skip eps>=0.05.
    """
    configs = []
    for eps in [0.00, 0.03]:
        name = f"eps_{eps:.2f}_tau_0.50"
        configs.append((name, {"dirichlet_epsilon": eps, "temperature": 0.50}))
    return configs


def phase1b_configs():
    """Tau sweep at fixed eps=0.00 (the new best epsilon).

    Redesigned: sweep tau at eps=0.00 instead of eps=0.10,
    since the epsilon sweep showed zero noise is optimal.
    """
    configs = []
    for tau in [0.25, 0.35, 0.50, 0.75, 1.00]:
        name = f"eps_0.00_tau_{tau:.2f}"
        configs.append((name, {"dirichlet_epsilon": 0.00, "temperature": tau}))
    return configs


def phase1c_configs():
    """Combos with eps=0.03 at sharper tau values.

    Tests whether tiny noise + sharper tau can compete with eps=0.00.
    """
    combos = [
        (0.03, 0.25, "tiny_noise_sharp"),
        (0.03, 0.35, "tiny_noise_moderate"),
    ]
    configs = []
    for eps, tau, tag in combos:
        name = f"eps_{eps:.2f}_tau_{tau:.2f}_{tag}"
        configs.append((name, {"dirichlet_epsilon": eps, "temperature": tau}))
    return configs


def phase2_configs(best_eps: float, best_tau: float):
    """Fine 3x3 grid around Phase 1 winner."""
    configs = []
    for d_eps in [-0.03, 0.0, 0.03]:
        for d_tau in [-0.10, 0.0, 0.10]:
            eps = round(max(0.0, best_eps + d_eps), 3)
            tau = round(max(0.1, best_tau + d_tau), 3)
            name = f"fine_eps_{eps:.3f}_tau_{tau:.2f}"
            configs.append((name, {"dirichlet_epsilon": eps, "temperature": tau}))
    # Deduplicate by (eps, tau)
    seen = {}
    deduped = []
    for name, overrides in configs:
        key = (overrides["dirichlet_epsilon"], overrides["temperature"])
        if key not in seen:
            seen[key] = True
            deduped.append((name, overrides))
    return deduped


def phase3_configs(best_eps: float, best_tau: float):
    """Simulation and game budget sweep at fixed (eps, tau)."""
    combos = [
        (40, 30),
        (40, 50),
        (80, 30),
        (80, 50),
        (80, 80),
        (160, 50),
        (160, 80),
    ]
    configs = []
    for sims, games in combos:
        name = f"sims_{sims}_games_{games}"
        configs.append((name, {
            "dirichlet_epsilon": best_eps,
            "temperature": best_tau,
            "n_simulations": sims,
            "n_games_per_train": games,
        }))
    return configs


def phase4_configs(top_configs: list[tuple[float, float]], D: int = 6):
    """Validation runs for top (eps, tau) pairs at a given D."""
    configs = []
    for rank, (eps, tau) in enumerate(top_configs, 1):
        name = f"D{D}_rank{rank}_eps_{eps:.3f}_tau_{tau:.2f}"
        configs.append((name, {"dirichlet_epsilon": eps, "temperature": tau}))
    return configs


# ---------------------------------------------------------------------------
# Deduplication across phases
# ---------------------------------------------------------------------------

def deduplicate_configs(all_configs):
    """Remove configs with identical (eps, tau, sims, games) overrides."""
    seen = {}
    deduped = []
    for name, overrides in all_configs:
        key = (
            overrides.get("dirichlet_epsilon"),
            overrides.get("temperature"),
            overrides.get("n_simulations"),
            overrides.get("n_games_per_train"),
        )
        if key not in seen:
            seen[key] = True
            deduped.append((name, overrides))
        else:
            logger.info(f"  Skipping duplicate: {name}")
    return deduped


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def make_config(D: int, seed: int, overrides: dict,
                n_iterations: int, n_sims: int, n_games: int,
                stage: int = 2, n_procs: int | None = None):
    """Create a config with the given overrides.

    stage=1: masked grammar (Stage 1, DoorsExplicitSurfaceDerivationConfig)
    stage=2: unmasked grammar (Stage 2, DoorsUnmaskedSurfaceDerivationConfig)
    n_procs: override parallel workers for trainer and evaluator (None = config default)
    """
    if stage == 1:
        cfg = DoorsExplicitSurfaceDerivationConfig(num_rooms=D)
        mode_label = "stage1_masked"
    else:
        cfg = DoorsUnmaskedSurfaceDerivationConfig(num_rooms=D, exact_length=True)
        mode_label = "stage2_unmasked"

    # Seeds
    cfg.agent.random_seeds = {
        "mcts": seed,
        "train": seed + 1,
        "eval": seed + 2,
        "external_policy": seed + 3,
    }

    # Base hyperparameters
    cfg.agent.mcts_params["n_simulations"] = n_sims
    cfg.agent.mcts_params["temperature"] = ANCHOR_TAU
    cfg.agent.mcts_params["c_exploration"] = 1.5
    cfg.agent.mcts_params["dirichlet_alpha"] = 0.25
    cfg.agent.mcts_params["dirichlet_epsilon"] = ANCHOR_EPS
    cfg.trainer.n_games_per_train = n_games
    cfg.trainer.n_past_iterations_to_train = 10
    cfg.run.n_iterations = n_iterations
    cfg.run.plot_every = max(1, n_iterations // 5)
    cfg.run.accept_threshold = 0.40

    # Override n_procs if specified
    if n_procs is not None:
        cfg.trainer.n_procs = n_procs
        cfg.evaluator.n_procs = n_procs

    # Apply overrides
    for key, val in overrides.items():
        if key == "dirichlet_epsilon":
            cfg.agent.mcts_params["dirichlet_epsilon"] = val
        elif key == "temperature":
            cfg.agent.mcts_params["temperature"] = val
        elif key == "n_simulations":
            cfg.agent.mcts_params["n_simulations"] = val
        elif key == "n_games_per_train":
            cfg.trainer.n_games_per_train = val
        else:
            raise ValueError(f"Unknown override key: {key}")

    return cfg, mode_label


def run_single(D: int, seed: int, config_name: str, overrides: dict,
               exp_dir: Path, n_iterations: int, n_sims: int, n_games: int,
               stage: int = 2, n_procs: int | None = None):
    """Run one training experiment and return summary stats."""
    cfg, mode_label = make_config(D, seed, overrides, n_iterations, n_sims, n_games, stage=stage, n_procs=n_procs)

    run_dir = exp_dir / f"{config_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.run.plot_path = str(run_dir / "training_metrics.png")
    cfg.save(str(run_dir / "config.json"))

    eps_val = cfg.agent.mcts_params["dirichlet_epsilon"]
    tau_val = cfg.agent.mcts_params["temperature"]
    sims_val = cfg.agent.mcts_params["n_simulations"]
    games_val = cfg.trainer.n_games_per_train

    print(f"\n{'='*70}")
    print(f"  Config: {config_name} | D={D} | Seed={seed}")
    print(f"  eps={eps_val}, tau={tau_val}, sims={sims_val}, "
          f"games={games_val}, iters={n_iterations}")
    print(f"  Output: {run_dir}/")
    print(f"{'='*70}\n")

    optimal_reward = compute_optimal_reward(D)
    t0 = time.time()
    run_derivation_training(cfg, mode_label, optimal_reward, run_dir)
    wall_time = time.time() - t0

    # Read final stats
    summary = {
        "config_name": config_name,
        "D": D,
        "seed": seed,
        "n_iterations": n_iterations,
        "wall_time_s": round(wall_time, 1),
        "overrides": overrides,
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
# Resume support
# ---------------------------------------------------------------------------

def load_completed_runs(exp_dir: Path) -> set[str]:
    """Load names of already-completed runs from results.jsonl."""
    completed = set()
    results_path = exp_dir / "results.jsonl"
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = f"{entry['config_name']}_seed{entry['seed']}"
                    completed.add(key)
    return completed


def recover_incomplete_runs(exp_dir: Path, all_configs, seeds, n_iterations):
    """Recover runs that completed training but crashed before writing results.

    Scans for run directories with program_log.jsonl that reached near the
    final iteration, but have no corresponding entry in results.jsonl.
    Synthesizes a results entry from the per-iteration logs.
    """
    completed = load_completed_runs(exp_dir)
    recovered = 0

    for config_name, overrides in all_configs:
        for seed in seeds:
            run_key = f"{config_name}_seed{seed}"
            if run_key in completed:
                continue

            run_dir = exp_dir / run_key
            prog_path = run_dir / "program_log.jsonl"
            stats_path = run_dir / "train_stats.jsonl"

            if not prog_path.exists():
                continue

            with open(prog_path) as f:
                prog_entries = [json.loads(line) for line in f if line.strip()]

            if not prog_entries:
                continue

            last_iter = prog_entries[-1].get("iteration", 0)
            # Recover if within 5 iterations of completion
            if last_iter < n_iterations - 5:
                continue

            # Synthesize summary from logs (same format as run_single)
            summary = {
                "config_name": config_name,
                "D": 0,  # filled below
                "seed": seed,
                "n_iterations": n_iterations,
                "wall_time_s": 0,
                "overrides": overrides,
                "recovered": True,
            }

            # Try to get D from config.json
            config_path = run_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg_data = json.load(f)
                summary["D"] = cfg_data.get("game", {}).get("kwargs", {}).get("num_rooms", 0)

            if stats_path.exists():
                with open(stats_path) as f:
                    stats = [json.loads(line) for line in f if line.strip()]
                if stats:
                    last = stats[-1]
                    summary["final_policy_loss"] = last.get("train_loss_policy")
                    summary["final_value_loss"] = last.get("train_loss_value")
                    summary["final_mcts_entropy"] = last.get("mcts_target_entropy")
                    summary["final_kl_gap"] = last.get("policy_kl_gap")
                    summary["final_avg_reward"] = last.get("avg_reward")

            for entry in prog_entries:
                if entry.get("best_solve_rate", 0) >= 1.0:
                    summary["first_solve_iter"] = entry["iteration"]
                    break
            last_prog = prog_entries[-1]
            summary["final_solve_rate"] = last_prog.get("best_solve_rate", 0)
            summary["final_best_reward"] = last_prog.get("best_avg_reward", 0)
            summary["unique_programs"] = last_prog.get("unique_programs", 0)

            # Append to results.jsonl
            with open(exp_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")

            recovered += 1
            fs = summary.get("first_solve_iter", "never")
            print(f"  RECOVERED: {run_key} (iter {last_iter}/{n_iterations}, "
                  f"first_solve={fs})")

    if recovered:
        print(f"  Recovered {recovered} incomplete run(s).\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(summaries):
    """Print a ranked summary table."""
    if not summaries:
        return

    print(f"\n{'='*100}")
    print(f"  SWEEP RESULTS — RANKED BY COMPOSITE SCORE")
    print(f"{'='*100}\n")

    # Group by config_name, aggregate across seeds
    from collections import defaultdict
    groups = defaultdict(list)
    for s in summaries:
        groups[s["config_name"]].append(s)

    ranked = []
    for config_name, runs in groups.items():
        solved_all = all(r.get("first_solve_iter") is not None for r in runs)
        first_solves = [r.get("first_solve_iter", float("inf")) for r in runs]
        avg_first = np.mean([x for x in first_solves if x != float("inf")])
        if np.isnan(avg_first):
            avg_first = float("inf")
        max_first = max(first_solves)
        avg_reward = np.mean([r.get("final_avg_reward", 0) for r in runs])
        avg_entropy = np.mean([r.get("final_mcts_entropy", 0) for r in runs])
        reward_std = np.std([r.get("final_avg_reward", 0) for r in runs])

        # Extract eps/tau from first run's overrides
        ov = runs[0]["overrides"]
        eps = ov.get("dirichlet_epsilon", ANCHOR_EPS)
        tau = ov.get("temperature", ANCHOR_TAU)

        ranked.append({
            "config_name": config_name,
            "eps": eps,
            "tau": tau,
            "n_seeds": len(runs),
            "solved_all": solved_all,
            "avg_first_solve": avg_first,
            "max_first_solve": max_first,
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "avg_entropy": avg_entropy,
        })

    # Sort: solved first, then by avg_first_solve, then by avg_reward descending
    ranked.sort(key=lambda x: (
        not x["solved_all"],
        x["avg_first_solve"],
        -x["avg_reward"],
    ))

    print(f"  {'Rank':>4}  {'Config':>35}  {'eps':>5}  {'tau':>5}  "
          f"{'Solved':>6}  {'1st(avg)':>8}  {'1st(max)':>8}  "
          f"{'Reward':>8}  {'R_std':>6}  {'Entropy':>8}")
    print(f"  {'----':>4}  {'------':>35}  {'---':>5}  {'---':>5}  "
          f"{'------':>6}  {'--------':>8}  {'--------':>8}  "
          f"{'------':>8}  {'-----':>6}  {'-------':>8}")

    for i, r in enumerate(ranked):
        first_avg = f"{r['avg_first_solve']:.0f}" if r['avg_first_solve'] != float('inf') else "never"
        first_max = f"{r['max_first_solve']:.0f}" if r['max_first_solve'] != float('inf') else "never"
        marker = " *" if r["config_name"] == f"eps_{ANCHOR_EPS:.2f}_tau_{ANCHOR_TAU:.2f}" else ""
        print(f"  {i+1:>4}  {r['config_name']:>35}  {r['eps']:>5.3f}  {r['tau']:>5.2f}  "
              f"{'yes' if r['solved_all'] else 'NO':>6}  {first_avg:>8}  {first_max:>8}  "
              f"{r['avg_reward']:>+8.4f}  {r['reward_std']:>6.4f}  "
              f"{r['avg_entropy']:>8.4f}{marker}")

    print(f"\n  * = anchor (current best)")
    print(f"{'='*100}\n")

    return ranked


def save_csv(summaries, csv_path: Path):
    """Save per-run results to CSV."""
    if not summaries:
        return
    fields = [
        "config_name", "D", "seed", "n_iterations", "wall_time_s",
        "final_policy_loss", "final_value_loss", "final_mcts_entropy",
        "final_kl_gap", "final_avg_reward", "first_solve_iter",
        "final_solve_rate", "final_best_reward", "unique_programs",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    print(f"[CSV] Saved to {csv_path}")


def plot_sweep_results(summaries, exp_dir: Path):
    """Generate bar charts for epsilon sweep and tau sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed. Skipping plots.")
        return

    from collections import defaultdict
    groups = defaultdict(list)
    for s in summaries:
        groups[s["config_name"]].append(s)

    # Separate configs by sweep type
    eps_sweep = []  # configs where tau=0.50
    tau_sweep = []  # configs where eps=0.10

    for config_name, runs in groups.items():
        ov = runs[0]["overrides"]
        eps = ov.get("dirichlet_epsilon", ANCHOR_EPS)
        tau = ov.get("temperature", ANCHOR_TAU)
        avg_reward = np.mean([r.get("final_avg_reward", 0) for r in runs])
        first_solves = [r.get("first_solve_iter", float("inf")) for r in runs]
        avg_first = np.mean([x for x in first_solves if x != float("inf")])

        if abs(tau - 0.50) < 1e-6 and "n_simulations" not in ov:
            eps_sweep.append((eps, avg_reward, avg_first))
        if abs(eps - 0.10) < 1e-6 and "n_simulations" not in ov:
            tau_sweep.append((tau, avg_reward, avg_first))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Epsilon sweep plots
    if eps_sweep:
        eps_sweep.sort()
        eps_vals = [x[0] for x in eps_sweep]
        eps_rewards = [x[1] for x in eps_sweep]
        eps_first = [x[2] if x[2] != float("inf") else 0 for x in eps_sweep]

        ax = axes[0, 0]
        bars = ax.bar(range(len(eps_vals)), eps_rewards, color="steelblue")
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f"{e:.2f}" for e in eps_vals])
        ax.set_xlabel("Dirichlet Epsilon")
        ax.set_ylabel("Avg Reward (final)")
        ax.set_title("Epsilon Sweep (tau=0.50) — Reward")
        ax.grid(True, alpha=0.3, axis="y")
        # Highlight anchor
        for i, e in enumerate(eps_vals):
            if abs(e - ANCHOR_EPS) < 1e-6:
                bars[i].set_color("darkorange")

        ax = axes[0, 1]
        bars = ax.bar(range(len(eps_vals)), eps_first, color="steelblue")
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f"{e:.2f}" for e in eps_vals])
        ax.set_xlabel("Dirichlet Epsilon")
        ax.set_ylabel("First-Solve Iteration (avg)")
        ax.set_title("Epsilon Sweep (tau=0.50) — First Solve")
        ax.grid(True, alpha=0.3, axis="y")
        for i, e in enumerate(eps_vals):
            if abs(e - ANCHOR_EPS) < 1e-6:
                bars[i].set_color("darkorange")

    # Tau sweep plots
    if tau_sweep:
        tau_sweep.sort()
        tau_vals = [x[0] for x in tau_sweep]
        tau_rewards = [x[1] for x in tau_sweep]
        tau_first = [x[2] if x[2] != float("inf") else 0 for x in tau_sweep]

        ax = axes[1, 0]
        bars = ax.bar(range(len(tau_vals)), tau_rewards, color="seagreen")
        ax.set_xticks(range(len(tau_vals)))
        ax.set_xticklabels([f"{t:.2f}" for t in tau_vals])
        ax.set_xlabel("Temperature (tau)")
        ax.set_ylabel("Avg Reward (final)")
        ax.set_title("Tau Sweep (eps=0.10) — Reward")
        ax.grid(True, alpha=0.3, axis="y")
        for i, t in enumerate(tau_vals):
            if abs(t - ANCHOR_TAU) < 1e-6:
                bars[i].set_color("darkorange")

        ax = axes[1, 1]
        bars = ax.bar(range(len(tau_vals)), tau_first, color="seagreen")
        ax.set_xticks(range(len(tau_vals)))
        ax.set_xticklabels([f"{t:.2f}" for t in tau_vals])
        ax.set_xlabel("Temperature (tau)")
        ax.set_ylabel("First-Solve Iteration (avg)")
        ax.set_title("Tau Sweep (eps=0.10) — First Solve")
        ax.grid(True, alpha=0.3, axis="y")
        for i, t in enumerate(tau_vals):
            if abs(t - ANCHOR_TAU) < 1e-6:
                bars[i].set_color("darkorange")

    plt.tight_layout()
    plot_path = exp_dir / "epsilon_tau_sweep.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Systematic epsilon x tau hyperparameter sweep")
    parser.add_argument("--phase", type=str, nargs="+",
                        default=["1a", "1b", "1c"],
                        choices=["1a", "1b", "1c", "2", "3", "4"],
                        help="Phases to run (default: 1a 1b 1c)")
    parser.add_argument("--d", type=int, default=SCREENING_D,
                        help=f"D value for screening (default: {SCREENING_D})")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override iterations (default: 25 for screening, 50 for D=6)")
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                        help=f"MCTS simulations (default: {DEFAULT_SIMS})")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help=f"Games per iteration (default: {DEFAULT_GAMES})")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds (default: phase-dependent)")
    parser.add_argument("--best-eps", type=float, default=ANCHOR_EPS,
                        help=f"Best epsilon from Phase 1 (for Phase 2+, default: {ANCHOR_EPS})")
    parser.add_argument("--best-tau", type=float, default=ANCHOR_TAU,
                        help=f"Best tau from Phase 1 (for Phase 2+, default: {ANCHOR_TAU})")
    parser.add_argument("--top-configs", type=str, default=None,
                        help="Top configs for Phase 4, format: 'eps1,tau1;eps2,tau2;eps3,tau3'")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2],
                        help="Grammar stage: 1=masked, 2=unmasked (default: 2)")
    parser.add_argument("--n-procs", type=int, default=None,
                        help="Override n_procs for trainer and evaluator (default: config default)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing experiment directory")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    phases = args.phase

    # Build config list for requested phases
    all_configs = []
    for phase in phases:
        if phase == "1a":
            all_configs.extend(phase1a_configs())
        elif phase == "1b":
            all_configs.extend(phase1b_configs())
        elif phase == "1c":
            all_configs.extend(phase1c_configs())
        elif phase == "2":
            all_configs.extend(phase2_configs(args.best_eps, args.best_tau))
        elif phase == "3":
            all_configs.extend(phase3_configs(args.best_eps, args.best_tau))
        elif phase == "4":
            if args.top_configs:
                pairs = []
                for pair_str in args.top_configs.split(";"):
                    e, t = pair_str.split(",")
                    pairs.append((float(e), float(t)))
            else:
                # Default: just the current best + two neighbors
                pairs = [
                    (args.best_eps, args.best_tau),
                    (max(0, args.best_eps - 0.03), args.best_tau),
                    (args.best_eps, max(0.1, args.best_tau - 0.10)),
                ]
            all_configs.extend(phase4_configs(pairs, D=args.d))

    # Deduplicate
    all_configs = deduplicate_configs(all_configs)

    if not all_configs:
        print("No configs to run. Check --phase argument.")
        return

    # Determine D, iterations, seeds
    is_validation = "4" in phases
    D = args.d  # always use --d flag; defaults to SCREENING_D (5)
    default_iters = VALIDATION_ITERS if is_validation else SCREENING_ITERS
    n_iterations = args.iterations if args.iterations is not None else default_iters
    n_sims = args.sims
    n_games = args.games

    if args.seeds is not None:
        seeds = args.seeds
    elif is_validation:
        seeds = SEEDS_PHASE4
    elif "2" in phases:
        seeds = SEEDS_PHASE2
    elif "3" in phases:
        seeds = SEEDS_PHASE3
    else:
        seeds = SEEDS_PHASE1

    stage = args.stage
    stage_label = "Stage 1 (masked)" if stage == 1 else "Stage 2 (unmasked)"

    # Setup experiment directory
    if args.resume:
        exp_dir = Path(args.resume)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase_str = "_".join(phases)
        exp_dir = Path(f"experiments/epsilon_tau_sweep_stage{stage}_phase{phase_str}_{timestamp}_D{D}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    exp_config = {
        "phases": phases,
        "stage": stage,
        "D": D,
        "seeds": seeds,
        "n_iterations": n_iterations,
        "n_simulations": n_sims,
        "n_games_per_train": n_games,
        "best_eps": args.best_eps,
        "best_tau": args.best_tau,
        "configs": [(name, ov) for name, ov in all_configs],
    }
    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    total_runs = len(all_configs) * len(seeds)
    print(f"\n{'#'*70}")
    print(f"  EPSILON x TAU SWEEP — Phase(s): {', '.join(phases)}")
    print(f"  Grammar: {stage_label}")
    print(f"  {len(all_configs)} configs x {len(seeds)} seeds = {total_runs} runs")
    print(f"  D={D}, iters={n_iterations}, sims={n_sims}, games={n_games}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {exp_dir}/")
    print(f"{'#'*70}\n")

    # Recover any runs that completed training but crashed before saving
    recover_incomplete_runs(exp_dir, all_configs, seeds, n_iterations)

    # Check for completed runs (resume support)
    completed = load_completed_runs(exp_dir)
    if completed:
        print(f"  Resuming: {len(completed)} runs already completed.\n")

    # Load existing summaries if resuming
    summaries = []
    results_path = exp_dir / "results.jsonl"
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    summaries.append(json.loads(line))

    # Run experiments
    run_idx = 0
    for config_name, overrides in all_configs:
        for seed in seeds:
            run_idx += 1
            run_key = f"{config_name}_seed{seed}"

            if run_key in completed:
                print(f"  [{run_idx}/{total_runs}] SKIP (done): {run_key}")
                continue

            print(f"\n>>> Run {run_idx}/{total_runs}: {config_name}, seed={seed}")

            # For phase 3, override sims/games from config
            run_sims = overrides.get("n_simulations", n_sims)
            run_games = overrides.get("n_games_per_train", n_games)

            summary = run_single(
                D=D, seed=seed, config_name=config_name,
                overrides=overrides, exp_dir=exp_dir,
                n_iterations=n_iterations, n_sims=run_sims, n_games=run_games,
                stage=stage, n_procs=args.n_procs,
            )
            summaries.append(summary)

            # Save incrementally
            with open(exp_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")

    # Final report
    ranked = print_summary_table(summaries)
    save_csv(summaries, exp_dir / "summary.csv")
    plot_sweep_results(summaries, exp_dir)

    # Save full summary
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    # Print top-3 for Phase 2+ input
    if ranked and len(ranked) >= 3:
        print(f"\n  Top 3 configs for --top-configs flag:")
        top3 = ranked[:3]
        pairs = ";".join(f"{r['eps']},{r['tau']}" for r in top3)
        print(f"  --top-configs '{pairs}'")
        print(f"  --best-eps {top3[0]['eps']} --best-tau {top3[0]['tau']}")


if __name__ == "__main__":
    main()
