#!/usr/bin/env python3
"""Plot training curves for the grammar1_experiment diagnostic report.

Creates three plots saved to docs/presentations/improvementv1/:
  1. D3_D6_D7_policy_loss.png  — policy loss curves
  2. D3_D6_D7_value_loss.png   — value loss curves
  3. D3_D6_D7_avg_reward.png   — average reward curves
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = Path("experiments/explicit_cfg_sweep_20260406_003202")
OUT_DIR = Path("docs/presentations/improvementv1")


def load_train_stats(D):
    path = SWEEP_DIR / f"D{D}" / "train_stats.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    d_values = [3, 6, 7]
    colors = {"3": "#2ca02c", "6": "#1f77b4", "7": "#d62728"}
    markers = {"3": "o", "6": "s", "7": "^"}

    data = {}
    for D in d_values:
        entries = load_train_stats(D)
        data[D] = {
            "iters": list(range(1, len(entries) + 1)),
            "policy": [e["train_loss_policy"] for e in entries],
            "value": [e["train_loss_value"] for e in entries],
            "reward": [e["avg_reward"] for e in entries],
        }

    # --- Plot 1: Policy loss ---
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for D in d_values:
        d = data[D]
        K = D - 1
        ax.plot(d["iters"], d["policy"], f"{markers[str(D)]}-",
                color=colors[str(D)], linewidth=2, markersize=6,
                label=f"D={D} (K={K})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Policy Loss (cross-entropy)", fontsize=11)
    ax.set_title("Policy Loss vs Iteration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    plt.tight_layout()
    out = OUT_DIR / "D3_D6_D7_policy_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 2: Value loss ---
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for D in d_values:
        d = data[D]
        K = D - 1
        ax.plot(d["iters"], d["value"], f"{markers[str(D)]}-",
                color=colors[str(D)], linewidth=2, markersize=6,
                label=f"D={D} (K={K})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Value Loss (MSE)", fontsize=11)
    ax.set_title("Value Loss vs Iteration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    plt.tight_layout()
    out = OUT_DIR / "D3_D6_D7_value_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 3: Average reward ---
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for D in d_values:
        d = data[D]
        K = D - 1
        ax.plot(d["iters"], d["reward"], f"{markers[str(D)]}-",
                color=colors[str(D)], linewidth=2, markersize=6,
                label=f"D={D} (K={K})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Average Reward", fontsize=11)
    ax.set_title("Average Reward vs Iteration", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)
    plt.tight_layout()
    out = OUT_DIR / "D3_D6_D7_avg_reward.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 4: Combined 1x3 panel ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Policy Loss (cross-entropy)", "Value Loss (MSE)", "Average Reward"]
    keys = ["policy", "value", "reward"]
    for idx, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[idx]
        for D in d_values:
            d = data[D]
            K = D - 1
            ax.plot(d["iters"], d[key], f"{markers[str(D)]}-",
                    color=colors[str(D)], linewidth=2, markersize=5,
                    label=f"D={D} (K={K})")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 15.5)
    plt.tight_layout()
    out = OUT_DIR / "D3_D6_D7_training_panel.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
