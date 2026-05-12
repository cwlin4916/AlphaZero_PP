#!/usr/bin/env python3
"""Generate combined Stage 1 vs Stage 2 comparison plots across D=3-7.

Data sources:
  D=3,4,5: experiments/stage1_vs_stage2_20260406_143216_D3_4_5/
  D=6,7:   experiments/epsilon_tau_sweep_stage{1,2}_phase4_*_D{6,7}/

Outputs (saved to docs/presentations/improvementv1/):
  1. stage1_vs_stage2_first_solve_vs_D.pdf   — first-solve iteration vs D
  2. stage1_vs_stage2_solve_curves_D6_D7.pdf — solve-rate learning curves at D=6,7
  3. stage1_vs_stage2_summary_table.pdf      — summary bar chart
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 11})

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "docs" / "presentations" / "improvementv1"

# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_program_log(run_dir: Path):
    """Load per-iteration program log from a run directory."""
    prog_path = run_dir / "program_log.jsonl"
    if not prog_path.exists():
        return []
    with open(prog_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_results():
    """Load all Stage 1 vs Stage 2 results across D=3-7."""
    results = []

    # D=3,4,5 from stage1_vs_stage2 experiment (eps=0.10)
    s1v2_dir = REPO / "experiments" / "stage1_vs_stage2_20260406_143216_D3_4_5"
    if s1v2_dir.exists():
        with open(s1v2_dir / "results.jsonl") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Add program log path
                    stage = entry["stage"]
                    D = entry["D"]
                    seed = entry["seed"]
                    entry["run_dir"] = s1v2_dir / f"stage{stage}_D{D}_seed{seed}"
                    results.append(entry)

    # D=6,7 from sweep experiments (eps=0.00)
    sweep_sources = [
        ("epsilon_tau_sweep_stage2_phase4_20260407_060852_D6", 2, 6, "D6_rank1_eps_0.000_tau_0.50"),
        ("epsilon_tau_sweep_stage2_phase4_20260407_060853_D7", 2, 7, "D7_rank1_eps_0.000_tau_0.50"),
        ("epsilon_tau_sweep_stage1_phase4_20260407_113351_D6", 1, 6, "D6_rank1_eps_0.000_tau_0.50"),
        ("epsilon_tau_sweep_stage1_phase4_20260407_113351_D7", 1, 7, "D7_rank1_eps_0.000_tau_0.50"),
    ]

    for exp_name, stage, D, config_prefix in sweep_sources:
        exp_dir = REPO / "experiments" / exp_name
        results_path = exp_dir / "results.jsonl"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entry["stage"] = stage
                    entry["D"] = D
                    seed = entry["seed"]
                    entry["run_dir"] = exp_dir / f"{config_prefix}_seed{seed}"
                    results.append(entry)

    return results


# -------------------------------------------------------------------------
# Plot 1: First-solve iteration vs D
# -------------------------------------------------------------------------

def plot_first_solve_vs_D(results):
    """Bar chart of first-solve iteration vs D for Stage 1 and Stage 2."""
    fig, ax = plt.subplots(figsize=(8, 5))

    d_values = sorted(set(r["D"] for r in results))

    stage1_means = []
    stage1_mins = []
    stage1_maxs = []
    stage2_means = []
    stage2_mins = []
    stage2_maxs = []
    NEVER = 55  # plot "never" above the iteration axis

    for D in d_values:
        for stage, means, mins, maxs in [
            (1, stage1_means, stage1_mins, stage1_maxs),
            (2, stage2_means, stage2_mins, stage2_maxs),
        ]:
            group = [r for r in results if r["D"] == D and r.get("stage") == stage]
            if not group:
                means.append(NEVER)
                mins.append(NEVER)
                maxs.append(NEVER)
                continue
            solves = [r.get("first_solve_iter", NEVER) for r in group]
            # Replace None/never with NEVER
            solves = [s if isinstance(s, (int, float)) else NEVER for s in solves]
            means.append(np.mean(solves))
            mins.append(np.min(solves))
            maxs.append(np.max(solves))

    x = np.arange(len(d_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, stage1_means, width, label="Stage 1 (masked)",
                   color="tab:blue", alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width/2, stage2_means, width, label="Stage 2 (unmasked)",
                   color="tab:orange", alpha=0.85, zorder=3)

    # Error bars (min/max range)
    for i, D in enumerate(d_values):
        if stage1_maxs[i] != stage1_mins[i]:
            ax.errorbar(i - width/2, stage1_means[i],
                       yerr=[[stage1_means[i] - stage1_mins[i]],
                             [stage1_maxs[i] - stage1_means[i]]],
                       color="tab:blue", capsize=4, capthick=1.5, fmt="none", zorder=4)
        if stage2_maxs[i] != stage2_mins[i] and stage2_means[i] < NEVER:
            ax.errorbar(i + width/2, stage2_means[i],
                       yerr=[[stage2_means[i] - stage2_mins[i]],
                             [stage2_maxs[i] - stage2_means[i]]],
                       color="tab:orange", capsize=4, capthick=1.5, fmt="none", zorder=4)

    # Mark "never solved" bars
    for i, (m1, m2) in enumerate(zip(stage1_means, stage2_means)):
        if m2 >= NEVER:
            ax.annotate("never\nsolved", (i + width/2, NEVER),
                       ha="center", va="bottom", fontsize=8,
                       color="tab:red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_values])
    ax.set_ylabel("First-Solve Iteration")
    ax.set_title("Stage 1 vs Stage 2: First-Solve Iteration by Difficulty")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 60)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(len(d_values) - 0.5, 51, "50 iter budget", ha="right",
            va="bottom", fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)

    # Add annotation about parameter difference
    ax.annotate("D=3-5: eps=0.10\nD=6-7: eps=0.00",
               xy=(0.98, 0.65), xycoords="axes fraction",
               ha="right", va="top", fontsize=8, color="gray",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = OUT_DIR / "stage1_vs_stage2_first_solve_vs_D.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot 1] Saved to {path}")
    # Also save PNG for tex inclusion
    path_png = path.with_suffix(".png")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # Redraw for PNG (matplotlib closes fig)
    fig.savefig(path_png, dpi=150, bbox_inches="tight")


def plot_first_solve_vs_D_v2(results):
    """Bar chart — saves both PDF and PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))

    d_values = sorted(set(r["D"] for r in results))
    NEVER = 55

    stage_data = {}
    for stage in [1, 2]:
        means, err_lo, err_hi = [], [], []
        for D in d_values:
            group = [r for r in results if r["D"] == D and r.get("stage") == stage]
            if not group:
                means.append(NEVER); err_lo.append(0); err_hi.append(0)
                continue
            solves = [r.get("first_solve_iter") for r in group]
            solves = [s if isinstance(s, (int, float)) else NEVER for s in solves]
            m = np.mean(solves)
            means.append(m)
            err_lo.append(m - np.min(solves))
            err_hi.append(np.max(solves) - m)
        stage_data[stage] = (means, err_lo, err_hi)

    x = np.arange(len(d_values))
    width = 0.35

    m1, lo1, hi1 = stage_data[1]
    m2, lo2, hi2 = stage_data[2]

    ax.bar(x - width/2, m1, width, label="Stage 1 (masked)",
           color="tab:blue", alpha=0.85, zorder=3)
    ax.bar(x + width/2, m2, width, label="Stage 2 (unmasked)",
           color="tab:orange", alpha=0.85, zorder=3)

    # Error bars
    ax.errorbar(x - width/2, m1, yerr=[lo1, hi1],
               color="black", capsize=4, capthick=1, fmt="none", zorder=4)
    for i in range(len(d_values)):
        if m2[i] < NEVER:
            ax.errorbar(x[i] + width/2, m2[i], yerr=[[lo2[i]], [hi2[i]]],
                       color="black", capsize=4, capthick=1, fmt="none", zorder=4)

    # Mark "never solved"
    for i in range(len(d_values)):
        if m2[i] >= NEVER:
            ax.annotate("never\nsolved", (x[i] + width/2, NEVER),
                       ha="center", va="bottom", fontsize=8,
                       color="tab:red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in d_values])
    ax.set_ylabel("First-Solve Iteration")
    ax.set_title("Stage 1 (Masked) vs Stage 2 (Unmasked): First-Solve by Difficulty")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 62)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(len(d_values) - 0.5, 51, "50 iter limit", ha="right",
            va="bottom", fontsize=8, color="gray")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        path = OUT_DIR / f"stage1_vs_stage2_first_solve_vs_D{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot 1] Saved to {path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Plot 2: Solve-rate learning curves at D=6 and D=7
# -------------------------------------------------------------------------

def plot_solve_curves_D6_D7(results):
    """Solve-rate learning curves for D=6 and D=7."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors = {1: "tab:blue", 2: "tab:orange"}
    labels = {1: "Stage 1 (masked)", 2: "Stage 2 (unmasked)"}

    for col, D in enumerate([6, 7]):
        ax = axes[col]
        ax.set_title(f"D = {D}")
        ax.set_xlabel("Iteration")
        if col == 0:
            ax.set_ylabel("Best Solve Rate")

        for stage in [1, 2]:
            group = [r for r in results if r["D"] == D and r.get("stage") == stage]
            seed_curves = []

            for r in group:
                run_dir = r.get("run_dir")
                if run_dir is None:
                    continue
                log = load_program_log(Path(run_dir))
                if not log:
                    continue
                iters = [e["iteration"] for e in log]
                rates = [e.get("best_solve_rate", 0) for e in log]
                seed_curves.append((iters, rates))
                ax.plot(iters, rates, color=colors[stage], alpha=0.25,
                       linewidth=1)

            # Mean curve
            if seed_curves:
                max_iter = max(max(c[0]) for c in seed_curves)
                mean_rates = []
                iter_range = range(1, max_iter + 1)
                for it in iter_range:
                    vals = []
                    for iters, rates in seed_curves:
                        if it <= len(rates):
                            vals.append(rates[it - 1])
                    if vals:
                        mean_rates.append(np.mean(vals))
                    else:
                        mean_rates.append(0)
                ax.plot(list(iter_range), mean_rates, color=colors[stage],
                       linewidth=2.5, label=labels[stage])

        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 52)

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        path = OUT_DIR / f"stage1_vs_stage2_solve_curves_D6_D7{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot 2] Saved to {path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Plot 3: Combined summary — first-solve + solve density
# -------------------------------------------------------------------------

def plot_summary_combined(results):
    """Two-panel plot: first-solve vs D (left) and solve density context (right)."""
    import math

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    d_values = [3, 4, 5, 6, 7]
    NEVER = 55

    # Left panel: first-solve vs D
    for stage, color, label in [(1, "tab:blue", "Stage 1 (masked)"),
                                  (2, "tab:orange", "Stage 2 (unmasked)")]:
        means, lo, hi = [], [], []
        for D in d_values:
            group = [r for r in results if r["D"] == D and r.get("stage") == stage]
            if not group:
                means.append(np.nan); lo.append(0); hi.append(0)
                continue
            solves = [r.get("first_solve_iter") for r in group]
            solves = [s if isinstance(s, (int, float)) else NEVER for s in solves]
            m = np.mean(solves)
            means.append(m)
            lo.append(m - np.min(solves))
            hi.append(np.max(solves) - m)

        ax1.errorbar(d_values, means, yerr=[lo, hi], marker="o", markersize=8,
                    linewidth=2, capsize=5, capthick=1.5, label=label, color=color)

    # Mark "never solved" points
    for D in [6, 7]:
        ax1.annotate("never", (D + 0.1, NEVER), fontsize=8, color="tab:red",
                    fontweight="bold", va="bottom")

    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.4)
    ax1.set_xlabel("D (number of rooms)")
    ax1.set_ylabel("First-Solve Iteration")
    ax1.set_title("First-Solve Iteration vs Difficulty")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 62)
    ax1.set_xticks(d_values)
    ax1.grid(axis="y", alpha=0.3)

    # Right panel: solve density comparison
    rho1_vals = []
    rho2_vals = []
    for D in d_values:
        K = D - 1
        lang_g1 = math.factorial(2*K) / (2**K)
        lang_g2 = (2*K) ** (2*K)
        n_solve = 1
        for i in range(1, 2*K, 2):
            n_solve *= i
        rho1_vals.append(n_solve / lang_g1)
        rho2_vals.append(n_solve / lang_g2)

    ax2.semilogy(d_values, rho1_vals, "o-", color="tab:blue", linewidth=2,
                markersize=8, label="Stage 1 solve density ($\\rho_1$)")
    ax2.semilogy(d_values, rho2_vals, "s-", color="tab:orange", linewidth=2,
                markersize=8, label="Stage 2 solve density ($\\rho_2$)")

    # Add shaded region where Stage 2 fails
    ax2.axvspan(5.5, 7.5, alpha=0.1, color="red")
    ax2.text(6.5, 1e-4, "Stage 2\nfails", ha="center", fontsize=9,
            color="tab:red", fontweight="bold")

    ax2.set_xlabel("D (number of rooms)")
    ax2.set_ylabel("Solve Density (log scale)")
    ax2.set_title("Solve Density: Why Stage 2 Fails at D $\\geq$ 6")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xticks(d_values)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        path = OUT_DIR / f"stage1_vs_stage2_combined_summary{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot 3] Saved to {path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Plot 4: Learning curves (policy & value loss) for D=3-7
# -------------------------------------------------------------------------

def load_train_stats(run_dir: Path):
    """Load per-iteration training stats from a run directory."""
    stats_path = run_dir / "train_stats.jsonl"
    if not stats_path.exists():
        return []
    with open(stats_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_learning_curves_all_D(results):
    """Policy loss and value loss learning curves for D=3-7, Stage 1 vs Stage 2."""
    d_values = sorted(set(r["D"] for r in results))
    fig, axes = plt.subplots(2, len(d_values), figsize=(3.2 * len(d_values), 6),
                             sharey="row")

    colors = {1: "tab:blue", 2: "tab:orange"}
    labels = {1: "Stage 1 (masked)", 2: "Stage 2 (unmasked)"}

    for col, D in enumerate(d_values):
        ax_p = axes[0, col]
        ax_v = axes[1, col]
        ax_p.set_title(f"D = {D}", fontsize=10)

        if col == 0:
            ax_p.set_ylabel("Policy Loss")
            ax_v.set_ylabel("Value Loss")
        ax_v.set_xlabel("Iteration")

        for stage in [1, 2]:
            group = [r for r in results if r["D"] == D and r.get("stage") == stage]
            for r in group:
                run_dir = r.get("run_dir")
                if run_dir is None:
                    continue
                stats = load_train_stats(Path(run_dir))
                if not stats:
                    continue
                iters = list(range(1, len(stats) + 1))
                p_loss = [s.get("train_loss_policy", float("nan")) for s in stats]
                v_loss = [s.get("train_loss_value", float("nan")) for s in stats]

                seed = r.get("seed", 0)
                first_seed = min(rr.get("seed", 0) for rr in group)
                alpha = 0.7 if seed == first_seed else 0.3
                lbl = labels[stage] if seed == first_seed else None

                ax_p.plot(iters, p_loss, color=colors[stage], alpha=alpha,
                         linewidth=1.2, label=lbl)
                ax_v.plot(iters, v_loss, color=colors[stage], alpha=alpha,
                         linewidth=1.2)

        ax_p.grid(True, alpha=0.3)
        ax_v.grid(True, alpha=0.3)
        if col == 0:
            ax_p.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        path = OUT_DIR / f"stage1_vs_stage2_learning_curves_D3_to_D7{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot 4] Saved to {path}")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    results = load_all_results()
    print(f"Loaded {len(results)} runs across D=3-7")

    plot_first_solve_vs_D_v2(results)
    plot_solve_curves_D6_D7(results)
    plot_summary_combined(results)
    plot_learning_curves_all_D(results)

    print("\nAll plots saved to:", OUT_DIR)
