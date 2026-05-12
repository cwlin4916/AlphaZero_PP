#!/usr/bin/env python3
"""Stage-2.5 landscape-diagnostic figures for ``docs/notes/stage4/02.md``.

Reads:
  docs/notes/stage4/data/landscape_r1.json   (+ landscape_r1.csv)
  docs/notes/stage4/data/landscape_r3.json   (+ landscape_r3.csv)
  docs/notes/stage4/data/summary.json
  docs/notes/stage4/data/runA_seed{0,1,2}_sims128_best.jsonl

Writes, into ``docs/notes/stage4/figures/``:

  02_landscape_solver_density.png         — B1/B2 solver count per grammar config
  02_landscape_score_variants.png         — (a) current vs progress-only scatter
                                            (b) degenerate-drop rank per variant
  02_landscape_generalization.png         — fraction of B1 solvers solving B2
  02_landscape_pathology_fractions.png    — vacuous-goal & disconnected-goal fractions
  02_landscape_first_solver_index.png     — uniform-MCTS first-solver index across
                                            available (seed, sims) cells (if logs exist)

Run from repo root:

    python scripts/plotting/plot_lifted_gripper_landscape.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "docs" / "notes" / "stage4" / "data"
OUT = REPO_ROOT / "docs" / "notes" / "stage4" / "figures"
DPI = 160

# Colours match plot_lifted_gripper_mcts.py — keep the family consistent.
CONSTRAINT_COLOR = {
    "none": "#1f77b4",
    "goal_predicate": "#ff7f0e",
    "connected": "#2ca02c",
    "both": "#9467bd",
}
CONSTRAINT_ORDER = ("none", "goal_predicate", "connected", "both")
CONSTRAINT_LABELS = {
    "none": "none\n(F,F)",
    "goal_predicate": "goal_predicate\n(T,F)",
    "connected": "connected\n(F,T)",
    "both": "both\n(T,T)",
}
CONFIG_ORDER = ("r1", "r3")
CONFIG_TITLES = {"r1": "r1 (max_rules=1, exhaustive)",
                 "r3": "r3 (max_rules=3, sampled)"}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _load_summary(stem: str) -> dict | None:
    p = DATA / f"landscape_{stem}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_rows(stem: str) -> list[dict] | None:
    p = DATA / f"landscape_{stem}.csv"
    if not p.exists():
        return None
    rows: list[dict] = []
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("num_rules", "num_state_literals", "num_goal_literals",
                      "solves_B1", "solves_B2", "solves_B3", "solves_B4"):
                pass
            rows.append({
                "constraint": r["constraint"],
                "mode": r["mode"],
                "num_rules": int(r["num_rules"]),
                "num_state_literals": int(r["num_state_literals"]),
                "num_goal_literals": int(r["num_goal_literals"]),
                "has_vacuous_goal_predicate": r["has_vacuous_goal_predicate"] == "True",
                "has_disconnected_goal_var": r["has_disconnected_goal_var"] == "True",
                "train_solve_rate": float(r["train_solve_rate"]),
                "eval_solve_rate": float(r["eval_solve_rate"]),
                "current_leaf_score": float(r["current_leaf_score"]),
                "score_no_step_penalty": float(r["score_no_step_penalty"]),
                "score_no_noop_penalty": float(r["score_no_noop_penalty"]),
                "score_progress_only": float(r["score_progress_only"]),
                "score_lexicographic_tuple": tuple(
                    json.loads(r["score_lexicographic_tuple"])
                ),
                "solves_B1": r["solves_B1"] == "True",
                "solves_B2": r["solves_B2"] == "True",
                "solves_B3": r["solves_B3"] == "True",
                "solves_B4": r["solves_B4"] == "True",
                "policy_pretty": r["policy_pretty"],
            })
    return rows


# ---------------------------------------------------------------------------
# F.1 — solver density
# ---------------------------------------------------------------------------

def plot_solver_density() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=False)
    for ax, stem in zip(axes, CONFIG_ORDER):
        summary = _load_summary(stem)
        if summary is None:
            ax.text(0.5, 0.5, f"landscape_{stem}.json missing\n— run make_lifted_gripper_landscape.py",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(CONFIG_TITLES[stem])
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        x = np.arange(len(CONSTRAINT_ORDER))
        b1 = [summary[c]["b1_solver_count"] for c in CONSTRAINT_ORDER]
        b2 = [summary[c]["b2_solver_count"] for c in CONSTRAINT_ORDER]
        n_pol = [summary[c]["unique_policies_evaluated"] for c in CONSTRAINT_ORDER]
        w = 0.38
        bars1 = ax.bar(x - w / 2, b1, width=w, color="#1f77b4", label="B=1 solvers")
        bars2 = ax.bar(x + w / 2, b2, width=w, color="#d62728", label="B=2 solvers")
        for b, count, n in zip(bars1, b1, n_pol):
            frac = count / n if n else 0.0
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{count}\n({100*frac:.2f}%)",
                    ha="center", va="bottom", fontsize=7)
        for b, count, n in zip(bars2, b2, n_pol):
            frac = count / n if n else 0.0
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{count}\n({100*frac:.2f}%)",
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{CONFIG_TITLES[stem]}  —  n_pol per cell ∈ {min(n_pol)}–{max(n_pol)}")
        ax.set_xticks(x)
        ax.set_xticklabels([CONSTRAINT_LABELS[c] for c in CONSTRAINT_ORDER], fontsize=8)
        ax.set_ylabel("# solving policies")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 2.5 — solver density by grammar config", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_landscape_solver_density.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F.2 — current vs progress-only scatter + degenerate-drop rank per variant
# ---------------------------------------------------------------------------

def plot_score_variants() -> None:
    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_rk = fig.add_subplot(gs[0, 1])

    # ---- scatter: current vs progress_only over all policies, r3 sampled
    rows = _load_rows("r3")
    summary_r3 = _load_summary("r3")
    if rows is None or summary_r3 is None:
        ax_sc.text(0.5, 0.5, "landscape_r3.{csv,json} missing", ha="center", va="center",
                   transform=ax_sc.transAxes, color="gray")
    else:
        for c in CONSTRAINT_ORDER:
            sub = [r for r in rows if r["constraint"] == c]
            if not sub:
                continue
            xs = [r["score_progress_only"] for r in sub]
            ys = [r["current_leaf_score"] for r in sub]
            ax_sc.scatter(xs, ys, s=10, alpha=0.55,
                          color=CONSTRAINT_COLOR[c], label=c, edgecolors="none")
        ax_sc.axline((0, 0), slope=1.0, color="black", linewidth=0.8, linestyle=":",
                     label="y = x")
        ax_sc.set_xlabel("score_progress_only  =  solve + 0.25·progress")
        ax_sc.set_ylabel("current_leaf_score  =  + −0.01·steps − 0.05·noops")
        ax_sc.set_title("r3 sampled — current vs progress-only score")
        ax_sc.grid(alpha=0.25)
        ax_sc.legend(loc="lower right", fontsize=8)

    # ---- bar: degenerate-drop rank under each variant (r1 + r3, all constraints)
    variants = ("current", "no_step_penalty", "no_noop_penalty", "progress_only", "lexicographic")
    width = 0.20
    x = np.arange(len(variants))
    plotted_any = False
    for i, stem in enumerate(CONFIG_ORDER):
        summary = _load_summary(stem)
        if summary is None:
            continue
        for j, c in enumerate(CONSTRAINT_ORDER):
            ranks = []
            for v in variants:
                # rank under current is in summary[c]["degenerate_drop_policy_rank"];
                # for other variants we read best_policy_by_variant and reconstruct
                # rank only for ``current`` (deterministic). For the bar, we plot the
                # rank under ``current`` per constraint × config (the most relevant
                # number — does the score reward the do-nothing attractor?).
                ranks.append(summary[c].get("degenerate_drop_policy_rank"))
                break  # plot only current — see comment
            n = summary[c]["unique_policies_evaluated"] or 1
            offset = (i * len(CONSTRAINT_ORDER) + j) * width - 1.5 * width
            # bar height = percentile (lower rank = better → 100·(1 − rank/n))
            r = ranks[0]
            pct = 100.0 * (1.0 - (r or n) / n)
            ax_rk.bar(i + (j - 1.5) * width, pct, width=width,
                      color=CONSTRAINT_COLOR[c],
                      label=c if i == 0 else None,
                      alpha=0.85)
            plotted_any = True
            if r is not None:
                ax_rk.text(i + (j - 1.5) * width, pct,
                           f"#{r}/{n}", ha="center", va="bottom", fontsize=6.5)
    if plotted_any:
        ax_rk.set_xticks(range(len(CONFIG_ORDER)))
        ax_rk.set_xticklabels([CONFIG_TITLES[s] for s in CONFIG_ORDER], fontsize=8)
        ax_rk.set_ylabel("percentile under `current` score   (100 = best)")
        ax_rk.set_title("degenerate ⊤⇒drop policy — rank by constraint × config")
        ax_rk.set_ylim(0, 100)
        ax_rk.grid(axis="y", alpha=0.25)
        ax_rk.legend(loc="lower right", fontsize=7)
    else:
        ax_rk.text(0.5, 0.5, "no landscape data", ha="center", va="center",
                   transform=ax_rk.transAxes, color="gray")

    fig.suptitle("Stage 2.5 — score-variant sensitivity", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_landscape_score_variants.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F.3 — generalisation: fraction of B1 solvers that also solve B2
# ---------------------------------------------------------------------------

def plot_generalization() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    width = 0.38
    x = np.arange(len(CONSTRAINT_ORDER))
    plotted = False
    for i, stem in enumerate(CONFIG_ORDER):
        summary = _load_summary(stem)
        if summary is None:
            continue
        fracs = []
        for c in CONSTRAINT_ORDER:
            n_b1 = summary[c]["b1_solver_count"]
            n_gen = summary[c]["b1_to_b2_generalizing_count"]
            fracs.append(n_gen / n_b1 if n_b1 else 0.0)
        offset = (-0.5 + i) * width
        bars = ax.bar(x + offset, fracs, width=width,
                      color=["#1f77b4", "#d62728"][i],
                      label=CONFIG_TITLES[stem])
        plotted = True
        for b, f, c in zip(bars, fracs, CONSTRAINT_ORDER):
            n_b1 = summary[c]["b1_solver_count"]
            n_gen = summary[c]["b1_to_b2_generalizing_count"]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{n_gen}/{n_b1}", ha="center", va="bottom", fontsize=7)
    if not plotted:
        ax.text(0.5, 0.5, "no landscape data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([CONSTRAINT_LABELS[c] for c in CONSTRAINT_ORDER], fontsize=8)
    ax.set_ylabel("fraction of B=1 solvers that also solve B=2")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Stage 2.5 — B=1 → B=2 generalisation rate by constraint")
    fig.tight_layout()
    fig.savefig(OUT / "02_landscape_generalization.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F.4 — pathology fractions
# ---------------------------------------------------------------------------

def plot_pathology_fractions() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    for ax, stem in zip(axes, CONFIG_ORDER):
        summary = _load_summary(stem)
        if summary is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue
        x = np.arange(len(CONSTRAINT_ORDER))
        n = [summary[c]["unique_policies_evaluated"] or 1 for c in CONSTRAINT_ORDER]
        f_vac = [summary[c]["vacuous_goal_policy_count"] / nn for c, nn in zip(CONSTRAINT_ORDER, n)]
        f_dis = [summary[c]["disconnected_goal_policy_count"] / nn for c, nn in zip(CONSTRAINT_ORDER, n)]
        w = 0.38
        bars1 = ax.bar(x - w / 2, f_vac, width=w, color="#8c564b", label="vacuous goal pred.")
        bars2 = ax.bar(x + w / 2, f_dis, width=w, color="#e377c2", label="disconnected goal var")
        for bars, vals in ((bars1, f_vac), (bars2, f_dis)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{100*v:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([CONSTRAINT_LABELS[c] for c in CONSTRAINT_ORDER], fontsize=8)
        ax.set_title(CONFIG_TITLES[stem])
        ax.set_ylim(0, max(0.05, max(f_vac + f_dis) * 1.25 + 0.05))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("fraction of policies")
    fig.suptitle("Stage 2.5 — pathology fractions: vacuous-goal vs disconnected-goal-var", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_landscape_pathology_fractions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F.5 — uniform-MCTS first-solver index across (seed, sims) (conditional)
# ---------------------------------------------------------------------------

def plot_first_solver_index() -> None:
    summary_path = DATA / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    cells = []  # (label, seed, sims, first_solver_index_or_None, n_unique)
    for key, block in summary.items():
        seed = block.get("seed")
        sims = block.get("mcts_sims")
        fsi = block.get("first_solver_index")
        n = block.get("n_unique", 0)
        cells.append((key, seed, sims, fsi, n))
    if not cells:
        return
    # sort by sims then seed
    cells.sort(key=lambda c: (c[2] or 0, c[1] or 0))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(cells))
    for i, (label, seed, sims, fsi, n) in enumerate(cells):
        if fsi is not None:
            ax.bar(i, fsi, color="#1f77b4")
            ax.text(i, fsi + max(1, 0.02 * max(c[4] or 1 for c in cells)),
                    f"#{fsi}/{n}", ha="center", va="bottom", fontsize=7)
        else:
            ax.bar(i, n, color="#d62728", alpha=0.45)
            ax.text(i, n + max(1, 0.02 * max(c[4] or 1 for c in cells)),
                    f"no solver\n(n={n})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\nseed={seed}\nsims={sims}" for (label, seed, sims, _f, _n) in cells],
        fontsize=7.5,
    )
    ax.set_ylabel("first B=1-solver index (in discovery order)")
    ax.set_title("Stage 2.5 — uniform-MCTS first-solver index across available runs"
                 "  (red = ran but no solver found within `n_unique` policies)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "02_landscape_first_solver_index.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_solver_density()
    plot_score_variants()
    plot_generalization()
    plot_pathology_fractions()
    plot_first_solver_index()
    print(f"[landscape-plots] wrote 02_landscape_*.png to {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
