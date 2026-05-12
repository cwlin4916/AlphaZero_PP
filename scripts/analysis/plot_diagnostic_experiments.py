#!/usr/bin/env python3
"""Plot results from diagnostic experiments.

Creates:
  1. Policy loss decomposition (H_MCTS + D_KL) per experiment
  2. D_KL comparison bar chart at final iteration
  3. Value loss comparison

Usage:
    python scripts/analysis/plot_diagnostic_experiments.py experiments/diagnostic_D5_YYYYMMDD_HHMMSS
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_NAMES = {
    0: "baseline",
    1: "long_run (100 iter)",
    2: "more_data (100 games)",
    3: "sharper_mcts (200 sims)",
    4: "less_noise (ε=0.10)",
    5: "lower_temp (τ=0.5)",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_experiment(exp_dir):
    """Load train_stats.jsonl from an experiment directory."""
    stats_path = exp_dir / "train_stats.jsonl"
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        return [json.loads(line) for line in f]


def find_experiments(base_dir):
    """Find all experiment subdirectories."""
    experiments = {}
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("exp"):
            idx = int(d.name.split("_")[0].replace("exp", ""))
            entries = load_experiment(d)
            if entries:
                experiments[idx] = entries
    return experiments


def plot_decomposition(experiments, out_dir):
    """Plot policy loss decomposition for each experiment."""
    n = len(experiments)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for i, (idx, entries) in enumerate(sorted(experiments.items())):
        ax = axes[i // cols][i % cols]
        iters = list(range(1, len(entries) + 1))
        h_mcts = [e["mcts_target_entropy"] for e in entries]
        kl_gap = [e["policy_kl_gap"] for e in entries]
        l_policy = [e["train_loss_policy"] for e in entries]

        ax.fill_between(iters, 0, h_mcts, alpha=0.4, color=COLORS[idx % len(COLORS)],
                         label="H(π_MCTS) (floor)")
        ax.fill_between(iters, h_mcts, l_policy, alpha=0.3, color="gray",
                         label="D_KL (gap)")
        ax.plot(iters, l_policy, "o-", color=COLORS[idx % len(COLORS)],
                markersize=4, linewidth=1.5, label="L_policy")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (nats)")
        name = EXP_NAMES.get(idx, f"exp{idx}")
        ax.set_title(f"Exp {idx}: {name}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(len(experiments), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle("Policy Loss Decomposition: L = H(π_MCTS) + D_KL", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "policy_decomposition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_kl_comparison(experiments, out_dir):
    """Bar chart comparing D_KL at final iteration."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    indices = sorted(experiments.keys())
    names = [EXP_NAMES.get(i, f"exp{i}") for i in indices]
    colors = [COLORS[i % len(COLORS)] for i in indices]

    # D_KL at final iteration
    ax = axes[0]
    kl_values = [experiments[i][-1]["policy_kl_gap"] for i in indices]
    bars = ax.bar(range(len(indices)), kl_values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"E{i}" for i in indices], fontsize=9)
    ax.set_ylabel("D_KL (nats)")
    ax.set_title("Final D_KL (lower = better)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, kl_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # H_MCTS at final iteration
    ax = axes[1]
    h_values = [experiments[i][-1]["mcts_target_entropy"] for i in indices]
    bars = ax.bar(range(len(indices)), h_values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"E{i}" for i in indices], fontsize=9)
    ax.set_ylabel("H(π_MCTS) (nats)")
    ax.set_title("Final MCTS Target Entropy", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Value loss at final iteration
    ax = axes[2]
    v_values = [experiments[i][-1]["train_loss_value"] for i in indices]
    bars = ax.bar(range(len(indices)), v_values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([f"E{i}" for i in indices], fontsize=9)
    ax.set_ylabel("Value Loss (MSE)")
    ax.set_title("Final Value Loss (lower = better)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add legend below
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.8) for c in colors],
               names, loc="lower center", ncol=min(3, len(names)),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = out_dir / "experiment_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_curves_overlay(experiments, out_dir):
    """Overlay D_KL curves from all experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, entries in sorted(experiments.items()):
        iters = list(range(1, len(entries) + 1))
        name = EXP_NAMES.get(idx, f"exp{idx}")
        color = COLORS[idx % len(COLORS)]

        axes[0].plot(iters, [e["policy_kl_gap"] for e in entries],
                     "o-", color=color, markersize=4, linewidth=1.5, label=name)
        axes[1].plot(iters, [e["mcts_target_entropy"] for e in entries],
                     "o-", color=color, markersize=4, linewidth=1.5, label=name)
        axes[2].plot(iters, [e["train_loss_value"] for e in entries],
                     "o-", color=color, markersize=4, linewidth=1.5, label=name)

    axes[0].set_title("D_KL (learnable gap)", fontweight="bold")
    axes[0].set_ylabel("D_KL (nats)")
    axes[1].set_title("H(π_MCTS) (floor)", fontweight="bold")
    axes[1].set_ylabel("H (nats)")
    axes[2].set_title("Value Loss", fontweight="bold")
    axes[2].set_ylabel("MSE")

    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "curves_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def print_summary_table(experiments):
    """Print summary table to stdout."""
    print(f"\n{'='*90}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*90}")
    print(f"  {'#':>2}  {'Name':<25}  {'L_policy':>8}  {'H_MCTS':>8}  "
          f"{'D_KL':>8}  {'L_value':>8}  {'Reward':>8}")
    print(f"  {'--':>2}  {'----':<25}  {'--------':>8}  {'------':>8}  "
          f"{'----':>8}  {'-------':>8}  {'------':>8}")
    for idx, entries in sorted(experiments.items()):
        last = entries[-1]
        name = EXP_NAMES.get(idx, f"exp{idx}")
        print(f"  {idx:>2}  {name:<25}  "
              f"{last['train_loss_policy']:>8.4f}  "
              f"{last['mcts_target_entropy']:>8.4f}  "
              f"{last['policy_kl_gap']:>8.4f}  "
              f"{last['train_loss_value']:>8.4f}  "
              f"{last['avg_reward']:>8.4f}")


def main():
    if len(sys.argv) < 2:
        # Auto-find most recent diagnostic directory
        exp_base = Path("experiments")
        dirs = sorted(exp_base.glob("diagnostic_D*"))
        if not dirs:
            print("Usage: python scripts/analysis/plot_diagnostic_experiments.py <experiment_dir>")
            sys.exit(1)
        base_dir = dirs[-1]
        print(f"Using most recent: {base_dir}")
    else:
        base_dir = Path(sys.argv[1])

    experiments = find_experiments(base_dir)
    if not experiments:
        print(f"No experiments found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments: {sorted(experiments.keys())}")
    print_summary_table(experiments)

    plot_decomposition(experiments, base_dir)
    plot_kl_comparison(experiments, base_dir)
    plot_curves_overlay(experiments, base_dir)


if __name__ == "__main__":
    main()
