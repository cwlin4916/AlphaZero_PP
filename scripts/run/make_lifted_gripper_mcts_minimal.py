#!/usr/bin/env python3
"""Regenerate the Stage-2 *minimal-MCTS* canonical artifacts under
``docs/notes/stage4/data/minimal_mcts/`` (committed) and
``results/lifted_gripper_lite/minimal_mcts/`` (raw, gitignored).

Matrix (same-B train and eval; ``UniformPolicyValueNet`` only — see
``docs/notes/stage4/02_plan.md``):

    B=1 : mcts_sims in {128, 512, 2048}, seeds 0..4, max_rules=3
    B=2 : mcts_sims in {512, 2048, 8192}, seeds 0..4, max_rules=4

``n_episodes`` shrinks as ``mcts_sims`` grows (more search per play -> fewer
plays needed): {128: 64, 512: 32, 2048: 16, 8192: 8}.

The cells are independent (distinct ``balls × mcts_sims × seed``) so they fan
out over ``--jobs`` worker threads (default: ``cpu_count - 1``); the rolled-up
CSV / summary are produced sequentially once every cell has finished. Each cell
is seeded independently, so parallel and sequential runs give byte-identical
per-cell ``*_summary.json`` / ``*_best.jsonl``.

Full run (use all but one core):
    python scripts/run/make_lifted_gripper_mcts_minimal.py --jobs 7

Fast run (omit the slow B=2 / 8192-sim tier — the committed default):
    python scripts/run/make_lifted_gripper_mcts_minimal.py --skip-slow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _parallel import Cell, default_jobs, run_cells  # noqa: E402  (sibling-script import)

_REPO = Path(__file__).resolve().parents[2]
_RUN_ONE = _REPO / "scripts" / "run" / "run_lifted_gripper_mcts_minimal.py"
_DATA_DIR = _REPO / "docs" / "notes" / "stage4" / "data" / "minimal_mcts"
_RAW_DIR = _REPO / "results" / "lifted_gripper_lite" / "minimal_mcts"

_EPISODES_FOR_SIMS = {128: 64, 512: 32, 2048: 16, 8192: 8}

_GRID = {
    1: dict(sims=[128, 512, 2048], seeds=[0, 1, 2, 3, 4], max_rules=3),
    2: dict(sims=[512, 2048, 8192], seeds=[0, 1, 2, 3, 4], max_rules=4),
}

_CSV_COLS = [
    "balls", "mcts_sims", "seed", "n_episodes", "n_unique", "best_score",
    "n_solving", "first_solver_unique_index",
    "count_solver", "count_reasonable", "count_degenerate",
]


def _cell_name(balls: int, sims: int, seed: int) -> str:
    return f"b{balls}_sims{sims}_seed{seed}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-slow", action="store_true",
                    help="omit the B=2 / mcts_sims=8192 tier (the committed default)")
    ap.add_argument("--jobs", type=int, default=default_jobs(),
                    help="number of cells to run concurrently (default: cpu_count - 1)")
    args = ap.parse_args(argv)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- build the cell list ------------------------------------------------
    plan: list[dict] = []
    for balls, spec in _GRID.items():
        for sims in spec["sims"]:
            if args.skip_slow and balls == 2 and sims == 8192:
                continue
            n_episodes = _EPISODES_FOR_SIMS[sims]
            for seed in spec["seeds"]:
                cell = _cell_name(balls, sims, seed)
                raw_jsonl = _RAW_DIR / cell / "all.jsonl"
                raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
                summary_path = _DATA_DIR / f"{cell}_summary.json"
                plan.append(dict(
                    balls=balls, sims=sims, seed=seed, n_episodes=n_episodes,
                    max_rules=spec["max_rules"], cell=cell,
                    raw_jsonl=raw_jsonl, summary_path=summary_path,
                ))

    cell_jobs = [
        Cell(
            name=p["cell"],
            argv=[
                sys.executable, str(_RUN_ONE),
                "--balls", str(p["balls"]),
                "--max-rules", str(p["max_rules"]),
                "--mcts-sims", str(p["sims"]),
                "--n-episodes", str(p["n_episodes"]),
                "--seed", str(p["seed"]),
                "--out-jsonl", str(p["raw_jsonl"]),
                "--out-summary", str(p["summary_path"]),
                "--log-all-terminals",
            ],
            cost_hint=float(p["sims"]),  # schedule the heavy 8192-sim cells first
        )
        for p in plan
    ]

    rc = run_cells(cell_jobs, jobs=args.jobs)
    failed = [name for name, code in rc.items() if code != 0]
    if failed:
        print(f"ERROR: {len(failed)} cell(s) failed: {', '.join(sorted(failed))}", file=sys.stderr)
        return 1

    # --- collect outputs (sequential; every cell has finished) --------------
    rows: list[dict] = []
    for p in plan:
        cell, raw_jsonl, summary_path = p["cell"], p["raw_jsonl"], p["summary_path"]
        # committed best.jsonl = the best_so_far lines only (small)
        best_lines = []
        with raw_jsonl.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("kind") == "best_so_far":
                    best_lines.append(line.rstrip("\n"))
        (_DATA_DIR / f"{cell}_best.jsonl").write_text(
            "\n".join(best_lines) + ("\n" if best_lines else "")
        )
        s = json.loads(summary_path.read_text())
        rows.append({
            "balls": p["balls"], "mcts_sims": p["sims"], "seed": p["seed"],
            "n_episodes": p["n_episodes"], "n_unique": s["n_unique"],
            "best_score": s["best_score"], "n_solving": s["n_solving"],
            "first_solver_unique_index": s["first_solver_unique_index"],
            "count_solver": s["counts"]["solver"],
            "count_reasonable": s["counts"]["reasonable"],
            "count_degenerate": s["counts"]["degenerate"],
        })

    # rolled-up CSV
    csv_path = _DATA_DIR / "minimal_mcts.csv"
    with csv_path.open("w") as fh:
        fh.write(",".join(_CSV_COLS) + "\n")
        for r in sorted(rows, key=lambda r: (r["balls"], r["mcts_sims"], r["seed"])):
            fh.write(",".join(str(r[c]) if r[c] is not None else "" for c in _CSV_COLS) + "\n")

    # top-level summary
    by_b: dict[str, dict] = {}
    for b in sorted({r["balls"] for r in rows}):
        cells = [r for r in rows if r["balls"] == b]
        best = max(cells, key=lambda r: (r["best_score"] if r["best_score"] is not None else -1e9))
        solver_cells = [r for r in cells if r["n_solving"] > 0]
        by_b[str(b)] = {
            "n_cells": len(cells),
            "best_score_over_grid": best["best_score"],
            "best_score_cell": _cell_name(b, best["mcts_sims"], best["seed"]),
            "total_solver_policies": sum(r["count_solver"] for r in cells),
            "total_reasonable_policies": sum(r["count_reasonable"] for r in cells),
            "n_cells_with_a_solver": len(solver_cells),
            "first_solver_cells": [
                _cell_name(b, r["mcts_sims"], r["seed"]) for r in solver_cells
            ],
        }
    (_DATA_DIR / "summary.json").write_text(json.dumps(by_b, indent=2) + "\n")

    print(f"\nWrote {csv_path} ({len(rows)} cells) and {_DATA_DIR / 'summary.json'}.")
    for b, s in by_b.items():
        print(f"  B={b}: best score over grid = {s['best_score_over_grid']}, "
              f"{s['n_cells_with_a_solver']}/{s['n_cells']} cells found a solver, "
              f"{s['total_solver_policies']} solver / {s['total_reasonable_policies']} reasonable policies total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
