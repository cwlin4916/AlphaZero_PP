#!/usr/bin/env python3
"""Generate comparison plots for macro vs surface derivation experiments.

Loads train_stats.jsonl from matched experiments and produces:
  1. 2x2 panel: Avg Reward + Gate Score for D8 and D10
  2. 1x2 panel: Policy Loss + Value Loss

Usage:
    PYTHONPATH=src python scripts/plot_macro_vs_surface.py experiments/macro_vs_surface_comparison/TIMESTAMP
    PYTHONPATH=src python scripts/plot_macro_vs_surface.py  # auto-detect latest
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_experiment(exp_dir):
    """Load train_stats and config from an experiment directory."""
    exp_dir = Path(exp_dir)
    train_path = exp_dir / "train_stats.jsonl"
    config_path = exp_dir / "config.json"

    if not train_path.exists():
        return None, None

    train_stats = load_jsonl(train_path)
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    return train_stats, config


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "macro": "#E65100",   # deep orange
    "surface": "#1565C0",  # deep blue
}
LABELS = {
    "macro": "Macro (AST derivation)",
    "surface": "Surface (rule sequencing)",
}


def extract_metric(records, key):
    """Extract a metric from train_stats, returning (iterations, values)."""
    iters = list(range(1, len(records) + 1))
    values = [r.get(key, float("nan")) for r in records]
    return iters, values


def plot_main_comparison(data, D_values, output_path, hp_str):
    """2x2 panel: rows = [Avg Reward, Gate Score], cols = D values."""
    n_cols = len(D_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Macro vs Surface Derivation: Doors Environment\n{hp_str}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    metrics = [
        ("avg_reward", "Avg Reward"),
        ("gate_score", "Gate Score (eval win rate)"),
    ]

    for col, D in enumerate(D_values):
        for row, (metric_key, ylabel) in enumerate(metrics):
            ax = axes[row, col]

            for approach in ["surface", "macro"]:
                key = f"{approach}_D{D}"
                if key not in data or data[key] is None:
                    continue
                records = data[key]
                iters, values = extract_metric(records, metric_key)
                ax.plot(iters, values,
                        color=COLORS[approach],
                        linewidth=2.2,
                        marker="o", markersize=4,
                        label=LABELS[approach])

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(f"D={D} ({2*(D-1)+1} rooms)")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

            if metric_key == "gate_score":
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4,
                           linewidth=1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_loss_comparison(data, D_values, output_path, hp_str):
    """1x2 panel: Policy Loss and Value Loss side by side, one subplot per D."""
    n_cols = len(D_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        f"Training Losses: Macro vs Surface\n{hp_str}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    loss_metrics = [
        ("train_loss_policy", "Policy Loss"),
        ("train_loss_value", "Value Loss"),
    ]

    for col, D in enumerate(D_values):
        for row, (metric_key, ylabel) in enumerate(loss_metrics):
            ax = axes[row, col]

            for approach in ["surface", "macro"]:
                key = f"{approach}_D{D}"
                if key not in data or data[key] is None:
                    continue
                records = data[key]
                iters, values = extract_metric(records, metric_key)
                ax.plot(iters, values,
                        color=COLORS[approach],
                        linewidth=2.2,
                        marker="o", markersize=4,
                        label=LABELS[approach])

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(f"D={D}")
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary_table(data, D_values, configs):
    """Print a summary table comparing final metrics."""
    print()
    print("=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)
    header = f"  {'Experiment':<25} {'Iters':>5} {'Final Reward':>13} {'Gate Score':>11} {'P-Loss':>8} {'V-Loss':>8}"
    print(header)
    print("  " + "-" * 76)

    for D in D_values:
        for approach in ["surface", "macro"]:
            key = f"{approach}_D{D}"
            if key not in data or data[key] is None:
                print(f"  {approach.title()+f' D={D}':<25} {'N/A':>5}")
                continue
            records = data[key]
            last = records[-1] if records else {}
            label = f"{approach.title()} D={D}"
            cfg = configs.get(key, {})
            budget = cfg.get("game", {}).get("kwargs", {}).get("budget", "?")
            n_sites = cfg.get("game", {}).get("kwargs", {}).get("n_sites", "?")
            label_full = f"{label} (B={budget})"
            print(f"  {label_full:<25} {len(records):>5} "
                  f"{last.get('avg_reward', float('nan')):>+13.4f} "
                  f"{last.get('gate_score', float('nan')):>11.3f} "
                  f"{last.get('train_loss_policy', float('nan')):>8.4f} "
                  f"{last.get('train_loss_value', float('nan')):>8.4f}")
    print("=" * 80)


def verify_hyperparams(configs, D_values):
    """Check that MCTS params match across experiments. Print warnings if not."""
    all_mcts = {}
    for D in D_values:
        for approach in ["surface", "macro"]:
            key = f"{approach}_D{D}"
            cfg = configs.get(key)
            if cfg is None:
                continue
            mcts = cfg.get("agent", {}).get("mcts_params", {})
            all_mcts[key] = mcts

    keys_list = list(all_mcts.keys())
    if len(keys_list) < 2:
        return

    ref_key = keys_list[0]
    ref = all_mcts[ref_key]
    mismatches = []
    for other_key in keys_list[1:]:
        other = all_mcts[other_key]
        for param in set(ref) | set(other):
            v1 = ref.get(param)
            v2 = other.get(param)
            if v1 != v2:
                mismatches.append((param, ref_key, v1, other_key, v2))

    if mismatches:
        print()
        print("WARNING: MCTS parameter mismatches detected!")
        for param, k1, v1, k2, v2 in mismatches:
            print(f"  {param}: {k1}={v1} vs {k2}={v2}")
        print()
    else:
        print("MCTS params verified: all experiments match.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_comparison():
    """Find the most recent macro_vs_surface_comparison directory."""
    base = Path("experiments") / "macro_vs_surface_comparison"
    if not base.exists():
        return None
    dirs = sorted(base.iterdir())
    return dirs[-1] if dirs else None


def main():
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = find_latest_comparison()
        if base_dir is None:
            print("No comparison experiments found. Run "
                  "run_macro_vs_surface_comparison.py first.")
            sys.exit(1)
        print(f"Auto-detected: {base_dir}")

    # Load comparison config
    meta_path = base_dir / "comparison_config.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        D_values = meta["D_values"]
    else:
        # Infer from subdirectories
        D_values = sorted({
            int(d.name.split("_D")[1])
            for d in base_dir.iterdir()
            if d.is_dir() and "_D" in d.name
        })

    # Load all experiment data
    data = {}
    configs = {}
    for approach in ["surface", "macro"]:
        for D in D_values:
            key = f"{approach}_D{D}"
            exp_dir = base_dir / f"{approach}_D{D}"
            if not exp_dir.exists():
                print(f"  Skipping {key}: directory not found")
                continue
            records, config = load_experiment(exp_dir)
            if records is None:
                print(f"  Skipping {key}: no train_stats.jsonl")
                continue
            data[key] = records
            if config:
                configs[key] = config
            print(f"  Loaded {key}: {len(records)} iterations")

    if not data:
        print("No experiment data found!")
        sys.exit(1)

    # Verify hyperparameter match
    verify_hyperparams(configs, D_values)

    # Build hyperparameter string for plot titles
    sample_cfg = next(iter(configs.values()), {})
    n_sims = sample_cfg.get("agent", {}).get("mcts_params", {}).get(
        "n_simulations", "?")
    n_games = sample_cfg.get("trainer", {}).get("n_games_per_train", "?")
    n_iters = sample_cfg.get("run", {}).get("n_iterations", "?")
    seed = sample_cfg.get("agent", {}).get("random_seeds", {}).get("mcts", "?")
    hp_str = f"{n_sims} sims | {n_games} games/iter | {n_iters} iterations | seed {seed}"

    # Create output directory
    out_dir = Path("presentations")
    out_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_main_comparison(
        data, D_values,
        out_dir / "macro_vs_surface_D8_D10.png",
        hp_str,
    )
    plot_loss_comparison(
        data, D_values,
        out_dir / "macro_vs_surface_losses.png",
        hp_str,
    )

    # Print summary
    print_summary_table(data, D_values, configs)


if __name__ == "__main__":
    main()
