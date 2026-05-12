#!/usr/bin/env python3
"""Stage 1 vs Stage 2 comparison with plots.

Produces:
  1. Language size scaling (log-scale) for D=2..max_D
  2. Solve density comparison
  3. Unique AST counts
  4. Reward histograms for exhaustively enumerable D values
  5. Search space blowup factors

Saves plots to docs/presentations/improvementv1/ and prints a text report.

Usage:
    python scripts/analysis/compare_stage1_vs_stage2.py [--max-D 7] [--output-dir ...]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.unmasked_surface_cfg import UnmaskedSurfaceCFG
from alphazeropp.instances.doors.dsl.explicit_surface_cfg import SurfaceCFG
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_formula_data(max_D: int):
    """Closed-form counts for all D values."""
    rows = []
    for D in range(2, max_D + 1):
        K = D - 1
        n = 2 * K
        L = 2 * K
        g1 = math.factorial(2 * K) // (2 ** K)
        g2_max = sum(n**l for l in range(L + 1)) if n > 0 else 1
        g2_exact = n ** L if n > 0 else 1
        rows.append({
            "D": D, "K": K,
            "g1": g1,
            "g2_max": g2_max,
            "g2_exact": g2_exact,
            "ratio_max": g2_max / g1,
            "ratio_exact": g2_exact / g1,
        })
    return rows


def collect_exhaustive_data(max_D_exhaust: int, max_enumerate: int = 5_000_000):
    """Exhaustively enumerate and evaluate for small D values."""
    rows = []
    for D in range(2, max_D_exhaust + 1):
        K = D - 1

        # Stage 1
        s1_cfg = SurfaceCFG(K)
        s1_count = s1_cfg.count_words()
        if s1_count > max_enumerate:
            continue

        # Stage 2 exact-length
        s2_cfg = UnmaskedSurfaceCFG(K, exact_length=True)
        s2_count = s2_cfg.count_words()
        if s2_count > max_enumerate:
            continue

        print(f"  Enumerating D={D} (Stage1: {s1_count}, Stage2: {s2_count})...")

        doors_cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
        n_sites = doors_cfg.obs_size()
        x0 = doors_initial_state(doors_cfg)
        le = LeafEvaluator(
            n_sites, [x0], doors_cfg,
            is_solved=doors_cfg.is_solved, metric="solve_rate",
        )

        # Evaluate Stage 1
        s1_words = s1_cfg.enumerate_words()
        s1_rewards = []
        s1_asts = set()
        s1_solve = 0
        for w in s1_words:
            prog = compile_policy(w, doors_cfg)
            r = le(prog)
            s1_rewards.append(r)
            s1_asts.add(prog.pretty())
            if r > 0:
                s1_solve += 1

        # Evaluate Stage 2 exact-length
        s2_words = s2_cfg.enumerate_words()
        s2_rewards = []
        s2_asts = set()
        s2_solve = 0
        for w in s2_words:
            prog = compile_policy(w, doors_cfg)
            r = le(prog)
            s2_rewards.append(r)
            s2_asts.add(prog.pretty())
            if r > 0:
                s2_solve += 1

        rows.append({
            "D": D, "K": K,
            "s1_count": len(s1_words),
            "s1_solve": s1_solve,
            "s1_density": s1_solve / len(s1_words),
            "s1_unique_asts": len(s1_asts),
            "s1_rewards": np.array(s1_rewards),
            "s2_count": len(s2_words),
            "s2_solve": s2_solve,
            "s2_density": s2_solve / len(s2_words),
            "s2_unique_asts": len(s2_asts),
            "s2_rewards": np.array(s2_rewards),
        })

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_language_sizes(formula_data, output_dir: Path):
    """Log-scale plot of language sizes vs D."""
    if not HAS_MPL:
        return
    Ds = [r["D"] for r in formula_data]
    g1 = [r["g1"] for r in formula_data]
    g2_max = [r["g2_max"] for r in formula_data]
    g2_exact = [r["g2_exact"] for r in formula_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Ds, g1, "o-", label="Stage 1 (masked)", linewidth=2, markersize=8)
    ax.semilogy(Ds, g2_exact, "s--", label="Stage 2 exact-length", linewidth=2, markersize=8)
    ax.semilogy(Ds, g2_max, "^:", label="Stage 2 max-length", linewidth=2, markersize=8)
    ax.set_xlabel("D (number of rooms)", fontsize=12)
    ax.set_ylabel("Language size |L|", fontsize=12)
    ax.set_title("Language Size: Stage 1 vs Stage 2", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_vs_stage2_language_size.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stage1_vs_stage2_language_size.png'}")


def plot_blowup_factors(formula_data, output_dir: Path):
    """Blowup factor |G2|/|G1| vs D."""
    if not HAS_MPL:
        return
    Ds = [r["D"] for r in formula_data]
    r_exact = [r["ratio_exact"] for r in formula_data]
    r_max = [r["ratio_max"] for r in formula_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Ds, r_exact, "s-", label="Exact-length ratio", linewidth=2, markersize=8)
    ax.semilogy(Ds, r_max, "^--", label="Max-length ratio", linewidth=2, markersize=8)
    ax.set_xlabel("D (number of rooms)", fontsize=12)
    ax.set_ylabel("Blowup factor |L(G₂)| / |L(G₁)|", fontsize=12)
    ax.set_title("Search Space Blowup: Stage 2 / Stage 1", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_vs_stage2_blowup.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stage1_vs_stage2_blowup.png'}")


def plot_solve_density(exhaust_data, formula_data, output_dir: Path):
    """Solve density comparison."""
    if not HAS_MPL:
        return

    # For exhaustive data we have actual solve densities
    # For formula-only we can compute Stage 1 density = N_solve / |L(G1)|
    # Stage 1 N_solve = (2K-1)!! = (2K)! / (2^K * K!)
    Ds = [r["D"] for r in formula_data]
    s1_density = []
    for r in formula_data:
        K = r["K"]
        n_solve = math.factorial(2 * K) // (2**K * math.factorial(K))
        s1_density.append(n_solve / r["g1"] if r["g1"] > 0 else 0)

    # Stage 2: solve density = (Stage 1 solvers) / |L(G2_exact)|
    # since only Stage 1-valid orderings can solve
    s2_density_exact = []
    for r in formula_data:
        K = r["K"]
        n_solve = math.factorial(2 * K) // (2**K * math.factorial(K))
        s2_density_exact.append(n_solve / r["g2_exact"] if r["g2_exact"] > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Ds, s1_density, "o-", label="Stage 1 (masked)", linewidth=2, markersize=8)
    ax.semilogy(Ds, s2_density_exact, "s--", label="Stage 2 exact-length", linewidth=2, markersize=8)

    # Overlay actual measured densities from exhaustive data
    if exhaust_data:
        ex_Ds = [r["D"] for r in exhaust_data]
        ex_s1 = [r["s1_density"] for r in exhaust_data]
        ex_s2 = [r["s2_density"] for r in exhaust_data]
        ax.semilogy(ex_Ds, ex_s1, "o", color="tab:blue", markersize=12, zorder=5,
                     markerfacecolor="none", markeredgewidth=2, label="Stage 1 (measured)")
        ax.semilogy(ex_Ds, ex_s2, "s", color="tab:orange", markersize=12, zorder=5,
                     markerfacecolor="none", markeredgewidth=2, label="Stage 2 (measured)")

    ax.set_xlabel("D (number of rooms)", fontsize=12)
    ax.set_ylabel("Solve density", fontsize=12)
    ax.set_title("Solve Density: Stage 1 vs Stage 2", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_vs_stage2_solve_density.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stage1_vs_stage2_solve_density.png'}")


def plot_reward_histograms(exhaust_data, output_dir: Path):
    """Side-by-side reward histograms for each D."""
    if not HAS_MPL or not exhaust_data:
        return

    n = len(exhaust_data)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)

    for i, row in enumerate(exhaust_data):
        D = row["D"]
        bins = np.linspace(
            min(row["s1_rewards"].min(), row["s2_rewards"].min()) - 0.1,
            max(row["s1_rewards"].max(), row["s2_rewards"].max()) + 0.1,
            30,
        )

        axes[0, i].hist(row["s1_rewards"], bins=bins, color="tab:blue", alpha=0.7, edgecolor="black")
        axes[0, i].set_title(f"Stage 1, D={D}\n(n={row['s1_count']}, solve={row['s1_solve']})", fontsize=10)
        axes[0, i].set_xlabel("Reward")
        axes[0, i].set_ylabel("Count")

        axes[1, i].hist(row["s2_rewards"], bins=bins, color="tab:orange", alpha=0.7, edgecolor="black")
        axes[1, i].set_title(f"Stage 2, D={D}\n(n={row['s2_count']}, solve={row['s2_solve']})", fontsize=10)
        axes[1, i].set_xlabel("Reward")
        axes[1, i].set_ylabel("Count")

    fig.suptitle("Reward Distributions: Stage 1 vs Stage 2", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_vs_stage2_reward_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stage1_vs_stage2_reward_histograms.png'}")


def plot_unique_asts(exhaust_data, output_dir: Path):
    """Unique AST counts comparison."""
    if not HAS_MPL or not exhaust_data:
        return

    Ds = [r["D"] for r in exhaust_data]
    s1_asts = [r["s1_unique_asts"] for r in exhaust_data]
    s2_asts = [r["s2_unique_asts"] for r in exhaust_data]
    s1_counts = [r["s1_count"] for r in exhaust_data]
    s2_counts = [r["s2_count"] for r in exhaust_data]

    x = np.arange(len(Ds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, s1_asts, width, label="Stage 1 unique ASTs", color="tab:blue", alpha=0.8)
    bars2 = ax.bar(x + width/2, s2_asts, width, label="Stage 2 unique ASTs", color="tab:orange", alpha=0.8)

    # Annotate with derivation counts
    for i in range(len(Ds)):
        ax.text(x[i] - width/2, s1_asts[i] + 0.5, f"/{s1_counts[i]}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width/2, s2_asts[i] + 0.5, f"/{s2_counts[i]}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("D (number of rooms)", fontsize=12)
    ax.set_ylabel("Unique compiled ASTs", fontsize=12)
    ax.set_title("Unique ASTs: Stage 1 vs Stage 2\n(annotated: /total derivations)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in Ds])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_vs_stage2_unique_asts.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stage1_vs_stage2_unique_asts.png'}")


def plot_combined_scaling(formula_data, output_dir: Path):
    """Combined 2x2 plot: language size, blowup, solve density, density ratio."""
    if not HAS_MPL:
        return

    Ds = [r["D"] for r in formula_data]
    g1 = [r["g1"] for r in formula_data]
    g2_exact = [r["g2_exact"] for r in formula_data]
    r_exact = [r["ratio_exact"] for r in formula_data]

    # Solve density
    s1_density = []
    s2_density = []
    density_ratio = []
    for r in formula_data:
        K = r["K"]
        n_solve = math.factorial(2 * K) // (2**K * math.factorial(K))
        d1 = n_solve / r["g1"] if r["g1"] else 0
        d2 = n_solve / r["g2_exact"] if r["g2_exact"] else 0
        s1_density.append(d1)
        s2_density.append(d2)
        density_ratio.append(d1 / d2 if d2 else float("inf"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) Language size
    ax = axes[0, 0]
    ax.semilogy(Ds, g1, "o-", label="Stage 1", linewidth=2, markersize=7)
    ax.semilogy(Ds, g2_exact, "s--", label="Stage 2 (exact)", linewidth=2, markersize=7)
    ax.set_xlabel("D")
    ax.set_ylabel("|L|")
    ax.set_title("Language Size")
    ax.legend()
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)

    # (0,1) Blowup
    ax = axes[0, 1]
    ax.semilogy(Ds, r_exact, "s-", color="tab:red", linewidth=2, markersize=7)
    ax.set_xlabel("D")
    ax.set_ylabel("|L(G₂)| / |L(G₁)|")
    ax.set_title("Search Space Blowup")
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)

    # (1,0) Solve density
    ax = axes[1, 0]
    ax.semilogy(Ds, s1_density, "o-", label="Stage 1", linewidth=2, markersize=7)
    ax.semilogy(Ds, s2_density, "s--", label="Stage 2 (exact)", linewidth=2, markersize=7)
    ax.set_xlabel("D")
    ax.set_ylabel("Solve density")
    ax.set_title("Solve Density")
    ax.legend()
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)

    # (1,1) Density ratio
    ax = axes[1, 1]
    ax.semilogy(Ds, density_ratio, "D-", color="tab:purple", linewidth=2, markersize=7)
    ax.set_xlabel("D")
    ax.set_ylabel("ρ₁ / ρ₂")
    ax.set_title("Solve Density Ratio (Stage 1 / Stage 2)")
    ax.set_xticks(Ds)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Stage 1 vs Stage 2: Scaling Analysis (D=2..{})"
                 .format(max(Ds)), fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_vs_stage2_combined_scaling.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'stage1_vs_stage2_combined_scaling.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-D", type=int, default=7)
    parser.add_argument("--max-D-exhaust", type=int, default=4,
                        help="Max D for exhaustive enumeration+evaluation")
    parser.add_argument("--output-dir", type=str,
                        default="docs/presentations/improvementv1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Stage 1 vs Stage 2: Comparison Report")
    print("=" * 72)

    # 1. Collect formula-based data for all D
    print("\n[1] Computing formula-based counts D=2..{}...".format(args.max_D))
    formula_data = collect_formula_data(args.max_D)

    print(f"\n{'D':>3} {'K':>3} {'|L(G1)|':>14} {'|L=(G2)|':>14} "
          f"{'Ratio':>10} {'ρ1':>12} {'ρ2':>12} {'ρ1/ρ2':>8}")
    print("─" * 80)
    for r in formula_data:
        K = r["K"]
        n_solve = math.factorial(2 * K) // (2**K * math.factorial(K))
        d1 = n_solve / r["g1"] if r["g1"] else 0
        d2 = n_solve / r["g2_exact"] if r["g2_exact"] else 0
        dr = d1 / d2 if d2 else float("inf")
        print(f"{r['D']:>3} {K:>3} {r['g1']:>14,} {r['g2_exact']:>14,} "
              f"{r['ratio_exact']:>9,.0f}x "
              f"{d1:>12.6f} {d2:>12.8f} {dr:>7.0f}x")

    # 2. Exhaustive analysis for small D
    print(f"\n[2] Exhaustive evaluation D=2..{args.max_D_exhaust}...")
    exhaust_data = collect_exhaustive_data(args.max_D_exhaust)

    if exhaust_data:
        print(f"\n{'D':>3} {'Stage':>7} {'|L|':>10} {'Solve':>8} {'Density':>10} "
              f"{'UniqueAST':>10}")
        print("─" * 55)
        for row in exhaust_data:
            print(f"{row['D']:>3} {'S1':>7} {row['s1_count']:>10,} {row['s1_solve']:>8} "
                  f"{row['s1_density']:>10.6f} {row['s1_unique_asts']:>10}")
            print(f"{'':>3} {'S2':>7} {row['s2_count']:>10,} {row['s2_solve']:>8} "
                  f"{row['s2_density']:>10.6f} {row['s2_unique_asts']:>10}")

    # 3. Stage 1 inclusion check
    print(f"\n[3] Stage 1 ⊂ Stage 2 inclusion check...")
    for D in range(2, min(args.max_D_exhaust + 1, 5)):
        K = D - 1
        s1 = SurfaceCFG(K)
        s2 = UnmaskedSurfaceCFG(K, exact_length=True)
        s1_words = {tuple(w.rules) for w in s1.enumerate_words()}
        s2_words = {tuple(w.rules) for w in s2.enumerate_words()}
        included = s1_words.issubset(s2_words)
        strict = len(s2_words - s1_words) > 0
        print(f"  D={D}: S1⊂S2={included}, strict={strict}, "
              f"|S1|={len(s1_words)}, |S2|={len(s2_words)}, "
              f"|S2\\S1|={len(s2_words - s1_words)}")

    # 4. Generate plots
    print(f"\n[4] Generating plots...")
    plot_language_sizes(formula_data, output_dir)
    plot_blowup_factors(formula_data, output_dir)
    plot_solve_density(exhaust_data, formula_data, output_dir)
    plot_reward_histograms(exhaust_data, output_dir)
    plot_unique_asts(exhaust_data, output_dir)
    plot_combined_scaling(formula_data, output_dir)

    print(f"\n{'=' * 72}")
    print("Done. Plots saved to:", output_dir)


if __name__ == "__main__":
    main()
