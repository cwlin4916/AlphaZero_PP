#!/usr/bin/env python3
"""Stage-3-A diagnostic-grid figures for ``docs/notes/stage4/03.md``.

Reads:
  docs/notes/stage4/data/diagnostic_grid/diagnostic_grid.csv          (rolled-up per-cell rows)
  docs/notes/stage4/data/diagnostic_grid/<run_id>/best.jsonl          (best-so-far progressions)
  docs/notes/stage4/data/diagnostic_grid/<run_id>/all.jsonl           (if present — else results/.../all.jsonl)
  results/lifted_gripper_lite/diagnostic_grid/<run_id>/all.jsonl      (full per-policy logs, for occupancy)
  docs/notes/stage4/data/landscape_r3.json                            (Stage-2.5 — legacy 'none' vs strict 'both')

Writes, into ``docs/notes/stage4/figures/``:
  03_diagnostic_solver_rate.png            — solver rate per (grammar, baseline) × sims, Wilson 95% CI
  03_diagnostic_first_solver_index.png     — first-solver index per cell; no-solver cells censored at #unique
  03_diagnostic_best_score_curves.png      — best-so-far score vs unique-policy index, per (grammar, baseline)
  03_diagnostic_pathology_prevalence.png   — vacuous-goal / goal-only-var fractions: legacy vs strict
  03_diagnostic_root_entropy.png           — mean root visit-count entropy vs sim budget, per grammar (mcts cells)
  03_diagnostic_production_occupancy.png    — action-schema occupancy across evaluated policies, legacy vs strict

Figures are generated from whatever cells are present; on the small committed grid (seeds 0-1, sims 64,
8 episodes) several panels are thin — the titles say so. Run from repo root:

    python scripts/plotting/plot_lifted_gripper_diagnostic_grid.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "docs" / "notes" / "stage4" / "data"
GRID_DATA = DATA / "diagnostic_grid"
RAW_GRID = REPO_ROOT / "results" / "lifted_gripper_lite" / "diagnostic_grid"
LANDSCAPE_R3 = DATA / "landscape_r3.json"
OUT = REPO_ROOT / "docs" / "notes" / "stage4" / "figures"
DPI = 160

GRAMMAR_COLOR = {"strict": "#2ca02c", "legacy": "#1f77b4"}
BASELINE_MARK = {"mcts": "o", "random": "s"}
GRAMMAR_ORDER = ("legacy", "strict")
BASELINE_ORDER = ("mcts", "random")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _to_num(x, cast=float):
    if x is None or x == "" or x == "None":
        return None
    try:
        return cast(x)
    except (TypeError, ValueError):
        return None


def load_grid_rows() -> list[dict]:
    p = GRID_DATA / "diagnostic_grid.csv"
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "run_id": r.get("run_id"),
                "run": r.get("run"),
                "grammar": r.get("grammar_config_name"),
                "baseline": r.get("baseline"),
                "seed": _to_num(r.get("seed"), int),
                "sims": _to_num(r.get("sims"), int),
                "episodes": _to_num(r.get("episodes"), int),
                "unique_policies": _to_num(r.get("unique_policies"), int) or 0,
                "best_score": _to_num(r.get("best_score")),
                "first_solver_idx": _to_num(r.get("first_solver_idx"), int),
                "solver_count": _to_num(r.get("solver_count"), int) or 0,
                "root_entropy_mean": _to_num(r.get("root_entropy_mean")),
                "n_vacuous_goal": _to_num(r.get("n_vacuous_goal"), int) or 0,
                "n_goal_only_var": _to_num(r.get("n_goal_only_var"), int) or 0,
                "n_top_drop": _to_num(r.get("n_top_drop"), int) or 0,
            })
    return rows


def _all_jsonl_path(run_id: str) -> Path | None:
    for cand in ((GRID_DATA / run_id / "all.jsonl"), (RAW_GRID / run_id / "all.jsonl")):
        if cand.exists():
            return cand
    return None


def _best_jsonl_path(run_id: str) -> Path | None:
    p = GRID_DATA / run_id / "best.jsonl"
    return p if p.exists() else None


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, point, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), phat, min(1.0, center + half))


def _grid_size_caption(rows: list[dict]) -> str:
    if not rows:
        return "(no diagnostic-grid data found)"
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    sims = sorted({r["sims"] for r in rows if r["sims"] is not None and r["baseline"] == "mcts"})
    eps = sorted({r["episodes"] for r in rows if r["episodes"] is not None})
    sd = f"seeds {seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else (f"seed {seeds[0]}" if seeds else "seeds ?")
    sm = f"sims {','.join(map(str, sims))}" if sims else "sims ?"
    ep = f"{eps[0]} episodes" if len(eps) == 1 else f"episodes {eps}"
    small = " — small grid, illustrative" if (len(seeds) <= 3 or len(sims) <= 1) else ""
    return f"{sd}, {sm}, {ep}{small}"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_solver_rate(rows: list[dict], cap: str):
    """Fraction of seeds in which the cell found ≥1 train-solver, per (grammar, baseline) × sims,
    with Wilson 95% CI. 'random' cells are pooled into a single 'na' sims bucket."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    sims_vals = sorted({r["sims"] for r in rows if r["baseline"] == "mcts" and r["sims"] is not None})
    x_cats = [str(s) for s in sims_vals] + (["random"] if any(r["baseline"] == "random" for r in rows) else [])
    width = 0.18
    series = [(g, b) for g in GRAMMAR_ORDER for b in BASELINE_ORDER
              if any(r["grammar"] == g and r["baseline"] == b for r in rows)]
    for i, (g, b) in enumerate(series):
        xs, los, mids, his, ann = [], [], [], [], []
        for j, cat in enumerate(x_cats):
            if cat == "random":
                cell = [r for r in rows if r["grammar"] == g and r["baseline"] == "random"]
            else:
                cell = [r for r in rows if r["grammar"] == g and r["baseline"] == "mcts"
                        and r["sims"] == int(cat) and b == "mcts"]
            if not cell:
                continue
            n = len(cell)
            k = sum(1 for r in cell if r["solver_count"] > 0)
            lo, mid, hi = wilson_ci(k, n)
            xs.append(j + (i - (len(series) - 1) / 2) * width)
            los.append(mid - lo); mids.append(mid); his.append(hi - mid); ann.append(f"{k}/{n}")
        if not xs:
            continue
        ax.bar(xs, mids, width=width, color=GRAMMAR_COLOR.get(g, "#888"),
               alpha=0.55 if b == "random" else 0.95, edgecolor="black", linewidth=0.4,
               hatch="//" if b == "random" else None, label=f"{g} / {b}")
        ax.errorbar(xs, mids, yerr=[los, his], fmt="none", ecolor="black", capsize=3, linewidth=1.0)
        for x, m, hh, a in zip(xs, mids, his, ann):
            ax.text(x, m + hh + 0.03, a, ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(len(x_cats)))
    ax.set_xticklabels(x_cats)
    ax.set_xlabel("MCTS simulation budget  (rightmost: uniform-random baseline)")
    ax.set_ylabel("fraction of seeds finding ≥1 $B$-train solver")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Solver rate by grammar × baseline × sims (Wilson 95% CI)\n{cap}", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_diagnostic_solver_rate.png", dpi=DPI)
    plt.close(fig)


def fig_first_solver_index(rows: list[dict], cap: str):
    """Per-cell first-solver discovery index; cells that found no solver are drawn as red
    right-censored markers at #unique evaluated policies."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labelled = [r for r in rows if r["run_id"]]
    labelled.sort(key=lambda r: (r["grammar"] or "", r["baseline"] or "", r["sims"] or 0, r["seed"] or 0))
    for i, r in enumerate(labelled):
        if r["first_solver_idx"] is not None:
            ax.plot([i], [r["first_solver_idx"]], marker=BASELINE_MARK.get(r["baseline"], "o"),
                    color=GRAMMAR_COLOR.get(r["grammar"], "#888"), markersize=7, linestyle="none")
        else:
            ax.plot([i], [max(1, r["unique_policies"])], marker="x", color="#d62728", markersize=8,
                    linestyle="none")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labelled)))
    ax.set_xticklabels([r["run_id"] for r in labelled], rotation=90, fontsize=6)
    ax.set_ylabel("first-solver discovery index  (log; ✗ = no solver, plotted at #unique)")
    ax.set_title(f"First $B$-train-solver index per cell\n{cap}", fontsize=10)
    handles = [plt.Line2D([], [], marker=BASELINE_MARK[b], color="black", linestyle="none", label=f"baseline={b}")
               for b in BASELINE_ORDER]
    handles += [plt.Line2D([], [], marker="s", color=GRAMMAR_COLOR[g], linestyle="none", label=f"grammar={g}")
                for g in GRAMMAR_ORDER]
    handles += [plt.Line2D([], [], marker="x", color="#d62728", linestyle="none", label="no solver (censored)")]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_diagnostic_first_solver_index.png", dpi=DPI)
    plt.close(fig)


def fig_best_score_curves(rows: list[dict], cap: str):
    """Best-so-far leaf score vs unique-policy discovery index, one line per (grammar, baseline),
    averaged across seeds with a min/max envelope."""
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    by_series: dict[tuple, list[list[float]]] = defaultdict(list)
    for r in rows:
        p = _best_jsonl_path(r["run_id"]) if r["run_id"] else None
        if p is None:
            continue
        recs = _read_jsonl(p)
        if not recs:
            continue
        # best.jsonl is the best-so-far progression: monotone in `score`, indexed by discovery order
        # is not stored, so use the cumulative max over the progression's own order as a proxy curve.
        curve = []
        cur = -math.inf
        for rec in recs:
            cur = max(cur, rec["score"])
            curve.append(cur)
        by_series[(r["grammar"], r["baseline"])].append(curve)
    if not by_series:
        ax.text(0.5, 0.5, "no best.jsonl progressions found", ha="center", va="center")
    for (g, b), curves in sorted(by_series.items()):
        maxlen = max(len(c) for c in curves)
        padded = [c + [c[-1]] * (maxlen - len(c)) for c in curves]
        cols = list(zip(*padded))
        mean = [sum(col) / len(col) for col in cols]
        lo = [min(col) for col in cols]
        hi = [max(col) for col in cols]
        xs = list(range(1, maxlen + 1))
        ax.plot(xs, mean, color=GRAMMAR_COLOR.get(g, "#888"),
                linestyle="-" if b == "mcts" else "--", marker=BASELINE_MARK.get(b, "o"),
                markersize=3, label=f"{g} / {b}  (n={len(curves)} seeds)")
        ax.fill_between(xs, lo, hi, color=GRAMMAR_COLOR.get(g, "#888"), alpha=0.15)
    ax.set_xlabel("best-so-far progression step (≈ discovery order of new best)")
    ax.set_ylabel("best leaf score so far")
    ax.set_title(f"Best-so-far leaf score, by grammar × baseline\n{cap}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_diagnostic_best_score_curves.png", dpi=DPI)
    plt.close(fig)


def fig_pathology_prevalence(rows: list[dict], cap: str):
    """Vacuous-goal-predicate and goal-only-variable prevalence: Stage-2.5 landscape (legacy 'none'
    vs strict 'both') as the reference, plus the Stage-3-A diagnostic grid's own per-policy fractions."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    # left: Stage-2.5 landscape reference
    ax = axes[0]
    if LANDSCAPE_R3.exists():
        d = json.loads(LANDSCAPE_R3.read_text())
        cats = [("legacy ≈ landscape 'none'", "none"), ("strict ≈ landscape 'both'", "both")]
        xs = range(len(cats))
        vac = []
        goal_only = []
        for _, key in cats:
            v = d.get(key, {})
            n = max(1, v.get("unique_policies_evaluated", 1))
            vac.append(v.get("vacuous_goal_policy_count", 0) / n)
            goal_only.append(v.get("disconnected_goal_policy_count", 0) / n)
        w = 0.35
        ax.bar([x - w / 2 for x in xs], vac, width=w, color="#8c564b", label="vacuous goal predicate")
        ax.bar([x + w / 2 for x in xs], goal_only, width=w, color="#e377c2", label="goal-only variable")
        for x, a, b in zip(xs, vac, goal_only):
            ax.text(x - w / 2, a + 0.01, f"{a:.0%}", ha="center", va="bottom", fontsize=8)
            ax.text(x + w / 2, b + 0.01, f"{b:.0%}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c[0] for c in cats], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction (Stage-2.5 r3 sample, 5000)")
        ax.set_title("Stage-2.5 landscape reference\n(strict = 0/0 by construction)", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "landscape_r3.json not found", ha="center", va="center")
    ax.grid(axis="y", alpha=0.3)
    # right: this grid's own per-policy fractions, legacy vs strict (pooled over cells)
    ax = axes[1]
    pooled: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "vac": 0, "goal_only": 0, "top_drop": 0})
    for r in rows:
        g = r["grammar"]
        if g not in GRAMMAR_COLOR:
            continue
        pooled[g]["n"] += r["unique_policies"]
        pooled[g]["vac"] += r["n_vacuous_goal"]
        pooled[g]["goal_only"] += r["n_goal_only_var"]
        pooled[g]["top_drop"] += r["n_top_drop"]
    if pooled:
        gs = [g for g in GRAMMAR_ORDER if g in pooled]
        xs = range(len(gs))
        w = 0.25
        metrics = [("vac", "#8c564b", "vacuous goal"), ("goal_only", "#e377c2", "goal-only var"),
                   ("top_drop", "#7f7f7f", "⊤⇒drop rule")]
        for mi, (mk, mc, ml) in enumerate(metrics):
            vals = [pooled[g][mk] / max(1, pooled[g]["n"]) for g in gs]
            ax.bar([x + (mi - 1) * w for x in xs], vals, width=w, color=mc, label=ml)
            for x, v in zip(xs, vals):
                ax.text(x + (mi - 1) * w, v + 0.005, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{g}\n(n={pooled[g]['n']})" for g in gs], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of evaluated policies (this grid)")
        ax.set_title("Stage-3-A diagnostic grid (pooled cells)", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no diagnostic-grid rows", ha="center", va="center")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Grammar-pathology prevalence: legacy vs strict\n{cap}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "03_diagnostic_pathology_prevalence.png", dpi=DPI)
    plt.close(fig)


def fig_root_entropy(rows: list[dict], cap: str):
    """Mean root visit-count entropy vs sim budget, per grammar (uniform-MCTS cells only)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    any_pt = False
    for g in GRAMMAR_ORDER:
        cell = [r for r in rows if r["grammar"] == g and r["baseline"] == "mcts"
                and r["root_entropy_mean"] is not None and r["sims"] is not None]
        if not cell:
            continue
        by_sims: dict[int, list[float]] = defaultdict(list)
        for r in cell:
            by_sims[r["sims"]].append(r["root_entropy_mean"])
        sims = sorted(by_sims)
        means = [sum(by_sims[s]) / len(by_sims[s]) for s in sims]
        ax.plot(sims, means, marker="o", color=GRAMMAR_COLOR.get(g, "#888"), label=f"{g} grammar")
        any_pt = True
    if not any_pt:
        ax.text(0.5, 0.5, "no mcts cells with root entropy", ha="center", va="center")
    ax.set_xlabel("MCTS simulation budget")
    ax.set_ylabel("mean visit-count entropy at decision points  (nats)")
    ax.set_title(f"Root visit-count entropy vs sim budget (uniform-prior MCTS)\n{cap}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_diagnostic_root_entropy.png", dpi=DPI)
    plt.close(fig)


def fig_production_occupancy(rows: list[dict], cap: str):
    """Action-schema occupancy across all evaluated policies in the grid, legacy vs strict."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    by_grammar: dict[str, Counter] = defaultdict(Counter)
    totals: dict[str, int] = defaultdict(int)
    for r in rows:
        g = r["grammar"]
        if g not in GRAMMAR_COLOR or not r["run_id"]:
            continue
        p = _all_jsonl_path(r["run_id"])
        if p is None:
            continue
        for rec in _read_jsonl(p):
            pretty = rec.get("policy_pretty", "")
            # schema labels appear as "=> drop(", "=> pick(", "=> move(" in the pretty rendering
            for schema in ("move", "pick", "drop"):
                c = pretty.count(f"{schema}(")
                if c:
                    by_grammar[g][schema] += c
                    totals[g] += c
    if not by_grammar:
        ax.text(0.5, 0.5, "no all.jsonl logs found", ha="center", va="center")
    else:
        schemas = ["move", "pick", "drop"]
        gs = [g for g in GRAMMAR_ORDER if g in by_grammar]
        w = 0.35
        xs = range(len(schemas))
        for gi, g in enumerate(gs):
            tot = max(1, totals[g])
            vals = [by_grammar[g][s] / tot for s in schemas]
            ax.bar([x + (gi - (len(gs) - 1) / 2) * w for x in xs], vals, width=w,
                   color=GRAMMAR_COLOR[g], label=f"{g}  (n_rules={totals[g]})")
            for x, v in zip(xs, vals):
                ax.text(x + (gi - (len(gs) - 1) / 2) * w, v + 0.005, f"{v:.0%}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(schemas)
        ax.set_ylabel("share of rule heads over all evaluated policies")
    ax.set_title(f"Action-schema occupancy across evaluated policies\n{cap}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_diagnostic_production_occupancy.png", dpi=DPI)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_grid_rows()
    cap = _grid_size_caption(rows)
    fig_solver_rate(rows, cap)
    fig_first_solver_index(rows, cap)
    fig_best_score_curves(rows, cap)
    fig_pathology_prevalence(rows, cap)
    fig_root_entropy(rows, cap)
    fig_production_occupancy(rows, cap)
    print(f"[diag-grid-plot] wrote 6 figures to {OUT}  ({cap})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
