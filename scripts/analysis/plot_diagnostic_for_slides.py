#!/usr/bin/env python3
"""Generate presentation-quality plots for diagnostic experiment results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

OUT = Path("docs/presentations/improvementv1")

EXPERIMENTS_D5 = {
    0: ("baseline", "#888888"),
    1: ("long run (100 iter)", "#1f77b4"),
    2: ("more data (100 games)", "#2ca02c"),
    3: ("sharper MCTS (200 sims)", "#ff7f0e"),
    4: ("less noise (ε=0.10)", "#d62728"),
    5: ("lower temp (τ=0.5)", "#9467bd"),
}

EXPERIMENTS_D6 = {
    0: ("baseline", "#888888"),
    1: ("long run (100 iter)", "#1f77b4"),
    2: ("more data (100 games)", "#2ca02c"),
    3: ("less noise (ε=0.10)", "#d62728"),
    4: ("lower temp (τ=0.5)", "#9467bd"),
    5: ("combined (ε=0.10, τ=0.5)", "#e377c2"),
}

# Default
D = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BASE = Path(f"experiments/diagnostic_D{D}_combined")
EXPERIMENTS = EXPERIMENTS_D6 if D == 6 else EXPERIMENTS_D5


def load(idx):
    for pattern in [f"exp{idx}_*"]:
        matches = list(BASE.glob(pattern))
        if matches:
            path = matches[0] / "train_stats.jsonl"
            if path.exists():
                with open(path) as f:
                    return [json.loads(l) for l in f]
    return None


def plot_kl_bar():
    """Bar chart: final D_KL for all experiments."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    indices = sorted(EXPERIMENTS.keys())
    names = []
    kl_vals = []
    colors = []
    for idx in indices:
        entries = load(idx)
        if entries:
            kl_vals.append(entries[-1]["policy_kl_gap"])
            names.append(EXPERIMENTS[idx][0])
            colors.append(EXPERIMENTS[idx][1])

    bars = ax.bar(range(len(kl_vals)), kl_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(kl_vals)))
    ax.set_xticklabels(names, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("$D_{KL}$ (nats)", fontsize=10)
    ax.set_title("Learnable Gap at Final Iteration (lower = better)", fontsize=11, fontweight="bold")
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.6, linewidth=1.5, label="target (0.05)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    out = OUT / f"diag_D{D}_kl_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_kl_curves():
    """D_KL convergence curves for all experiments."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for idx in sorted(EXPERIMENTS.keys()):
        entries = load(idx)
        if entries:
            iters = list(range(1, len(entries) + 1))
            kl = [e["policy_kl_gap"] for e in entries]
            name, color = EXPERIMENTS[idx]
            ax.plot(iters, kl, "o-", color=color, markersize=3, linewidth=1.5, label=name)
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, linewidth=1.5, label="target")
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("$D_{KL}$ (nats)", fontsize=10)
    ax.set_title("$D_{KL}$ Convergence (lower = better)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.01, 0.5)
    plt.tight_layout()
    out = OUT / f"diag_D{D}_kl_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_decomposition_panel():
    """2x3 panel: policy loss decomposed into floor + KL for each experiment."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for i, idx in enumerate(sorted(EXPERIMENTS.keys())):
        ax = axes[i // 3][i % 3]
        entries = load(idx)
        if not entries:
            continue
        iters = list(range(1, len(entries) + 1))
        h = [e["mcts_target_entropy"] for e in entries]
        kl = [e["policy_kl_gap"] for e in entries]
        lp = [e["train_loss_policy"] for e in entries]
        name, color = EXPERIMENTS[idx]

        ax.fill_between(iters, 0, h, alpha=0.4, color=color, label="$H(\\pi_{MCTS})$")
        ax.fill_between(iters, h, lp, alpha=0.25, color="gray", label="$D_{KL}$")
        ax.plot(iters, lp, "-", color=color, linewidth=1.5)
        ax.set_title(f"E{idx}: {name}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Iter", fontsize=8)
        if i % 3 == 0:
            ax.set_ylabel("Loss (nats)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.8)

    fig.suptitle("Policy Loss = $H(\\pi_{MCTS})$ + $D_{KL}$", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT / f"diag_D{D}_decomposition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot_kl_bar()
    plot_kl_curves()
    plot_decomposition_panel()
