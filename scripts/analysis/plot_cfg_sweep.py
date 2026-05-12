#!/usr/bin/env python3
"""Plot training curves from the explicit CFG grammar-game sweep.

Creates two plots:
  1. Per-D training curve (solve rate + avg reward vs iteration)
  2. Summary: first-solve iteration vs D
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = Path("experiments/explicit_cfg_sweep_20260406_003202")


def load_d(D):
    log = SWEEP_DIR / f"D{D}" / "program_log.jsonl"
    if not log.exists():
        return None
    with open(log) as f:
        return [json.loads(line) for line in f]


def plot_d8():
    """Detailed training curve for D=8."""
    entries = load_d(8)
    if not entries:
        print("No D=8 data found")
        return

    iters = [e["iteration"] for e in entries]
    solve = [e["best_solve_rate"] for e in entries]
    reward = [e["best_avg_reward"] for e in entries]
    unique = [e["unique_programs"] for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Solve rate
    ax = axes[0]
    ax.plot(iters, solve, "o-", color="#2ca02c", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Solve Rate")
    ax.set_title("D=8 (K=7): Solve Rate")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Solved")
    first_solve = next((e["iteration"] for e in entries if e["best_solve_rate"] > 0), None)
    if first_solve:
        ax.axvline(x=first_solve, color="red", linestyle=":", alpha=0.7,
                   label=f"First solve: iter {first_solve}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[1]
    ax.plot(iters, reward, "s-", color="#1f77b4", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Avg Reward")
    ax.set_title("D=8 (K=7): Best Reward")
    ax.grid(True, alpha=0.3)

    # Unique programs explored
    ax = axes[2]
    ax.plot(iters, [u/1000 for u in unique], "^-", color="#9467bd", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Unique Programs (thousands)")
    ax.set_title("D=8 (K=7): Programs Explored")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Grammar-Game Training: D=8 (K=7, 681M policies, solve density 1/5040)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = SWEEP_DIR / "D8_training_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_summary():
    """Summary across all D values."""
    d_values = list(range(2, 11))
    first_solves = []
    final_solve_rates = []
    final_rewards = []
    wall_times = []

    for D in d_values:
        entries = load_d(D)
        if not entries:
            first_solves.append(None)
            final_solve_rates.append(0)
            final_rewards.append(0)
            wall_times.append(0)
            continue

        fs = next((e["iteration"] for e in entries if e["best_solve_rate"] > 0), None)
        first_solves.append(fs)
        final_solve_rates.append(entries[-1]["best_solve_rate"])
        final_rewards.append(entries[-1]["best_avg_reward"])
        wall_times.append(sum(e["iter_wall_clock"] for e in entries))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # First-solve iteration vs D
    ax = axes[0]
    solved_d = [d for d, fs in zip(d_values, first_solves) if fs is not None]
    solved_fs = [fs for fs in first_solves if fs is not None]
    unsolved_d = [d for d, fs in zip(d_values, first_solves) if fs is None]
    ax.bar(solved_d, solved_fs, color="#2ca02c", alpha=0.8, label="Solved")
    if unsolved_d:
        ax.bar(unsolved_d, [15]*len(unsolved_d), color="#d62728", alpha=0.5, label="Not solved (15 iters)")
    ax.set_xlabel("D (rooms)")
    ax.set_ylabel("First-Solve Iteration")
    ax.set_title("When Does AlphaZero First Solve?")
    ax.set_xticks(d_values)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Final solve rate vs D
    ax = axes[1]
    colors = ["#2ca02c" if sr > 0 else "#d62728" for sr in final_solve_rates]
    ax.bar(d_values, [sr * 100 for sr in final_solve_rates], color=colors, alpha=0.8)
    ax.set_xlabel("D (rooms)")
    ax.set_ylabel("Final Solve Rate (%)")
    ax.set_title("Final Solve Rate After 15 Iterations")
    ax.set_xticks(d_values)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    # Total wall time vs D
    ax = axes[2]
    ax.bar(d_values, [w/3600 for w in wall_times], color="#1f77b4", alpha=0.8)
    ax.set_xlabel("D (rooms)")
    ax.set_ylabel("Total Wall Time (hours)")
    ax.set_title("Compute Cost")
    ax.set_xticks(d_values)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Grammar-Game Sweep: D=2...10 (15 AlphaZero iterations each)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = SWEEP_DIR / "sweep_summary_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_table():
    """Print results table."""
    print(f"\n{'D':>3}  {'K':>3}  {'1st Solve':>9}  {'Final SR':>8}  {'Reward':>8}  {'Wall(h)':>8}  {'Unique':>10}")
    print(f"{'---':>3}  {'---':>3}  {'---------':>9}  {'--------':>8}  {'------':>8}  {'-------':>8}  {'------':>10}")
    for D in range(2, 11):
        entries = load_d(D)
        if not entries:
            print(f"{D:>3}  {D-1:>3}  {'(running)':>9}")
            continue
        fs = next((e["iteration"] for e in entries if e["best_solve_rate"] > 0), None)
        fs_str = str(fs) if fs else "never"
        last = entries[-1]
        wall = sum(e["iter_wall_clock"] for e in entries) / 3600
        print(f"{D:>3}  {D-1:>3}  {fs_str:>9}  {last['best_solve_rate']:>7.0%}  "
              f"{last['best_avg_reward']:>8.2f}  {wall:>7.2f}h  {last['unique_programs']:>10,}")


if __name__ == "__main__":
    print_table()
    plot_d8()
    plot_summary()
