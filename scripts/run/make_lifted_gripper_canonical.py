#!/usr/bin/env python3
"""Regenerate the canonical Stage-2 (lifted grammar × MCTS) run artifacts.

Legacy: this driver predates the occurrence-introduced-variable grammar
(``docs/notes/stage4/02_plan.md``) and the current Stage-2 minimal-MCTS data
(``data/minimal_mcts/`` via ``make_lifted_gripper_mcts_minimal.py``); it is kept
as the aux-var-grammar record. If ever re-run, it would adopt the same
``scripts/run/_parallel`` cell fan-out (``--jobs``) as ``make_lifted_gripper_mcts_minimal``.


Runs ``run_lifted_gripper_lite_smoke.py`` for the canonical cells reported in
``docs/notes/stage4/02.md`` §6 — Run A (B=1 train, B=2 eval-out, ≤3 rules, 128
sims, seed 0; also a 512-sim variant for the "more search ≠ better solver"
panel) and Run B (B=2 train, B=3 eval-out, ≤4 rules, 512 sims, seed 0).

Outputs
-------
* ``results/lifted_gripper_lite/{stem}_all.jsonl`` — every distinct evaluated
  policy with full metrics + ``policy_pretty`` (discovery order). These are
  large (~tens of MB) and **gitignored** — regenerate on demand.
* ``docs/notes/stage4/data/{stem}_scores.csv`` — the compact, **committed**
  per-policy view used by the figures: one row per evaluated policy (discovery
  order), columns ``score,train_solve_rate,eval_out_solve_rate,num_rules,num_literals``.
* ``docs/notes/stage4/data/{stem}_best.jsonl`` — best-so-far progression (small).
* ``docs/notes/stage4/data/summary.json`` — headline numbers + the best policy's
  full metrics for each run.

The committed artifacts let ``scripts/plotting/plot_lifted_gripper_mcts.py``
reproduce the Stage-2 figures without the ephemeral ``/tmp`` logs the original
runs used. Run A @ 128 sims is ~1 min; each 512-sim run is ~15 min.

Run from repo root:

    python scripts/run/make_lifted_gripper_canonical.py
    python scripts/run/make_lifted_gripper_canonical.py --skip-slow   # Run A @ 128 only (fast)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "scripts" / "run" / "run_lifted_gripper_lite_smoke.py"
RAW_DIR = REPO_ROOT / "results" / "lifted_gripper_lite"          # transient (gitignored)
DATA_DIR = REPO_ROOT / "docs" / "notes" / "stage4" / "data"      # committed

_SCORE_COLS = ("score", "train_solve_rate", "eval_out_solve_rate", "num_rules", "num_literals")

# summary key -> {stem, slow, smoke args excluding --out-jsonl/--dump-all-jsonl}.
# Stage 2.5: seeds 1 & 2 of Run A @ 128 sims feed the HitRate_B(N) / first-
# solver-index figure (`02_landscape_first_solver_index.png`). Each is ~70s.
RUNS = {
    "runA": dict(
        stem="runA_seed0_sims128", slow=False,
        args=["--n-balls-train", "1", "--n-balls-eval", "2",
              "--max-rules", "3", "--mcts-sims", "128", "--seed", "0"],
    ),
    "runA_seed1": dict(
        stem="runA_seed1_sims128", slow=False,
        args=["--n-balls-train", "1", "--n-balls-eval", "2",
              "--max-rules", "3", "--mcts-sims", "128", "--seed", "1"],
    ),
    "runA_seed2": dict(
        stem="runA_seed2_sims128", slow=False,
        args=["--n-balls-train", "1", "--n-balls-eval", "2",
              "--max-rules", "3", "--mcts-sims", "128", "--seed", "2"],
    ),
    "runA_512sims": dict(
        stem="runA_seed0_sims512", slow=True,
        args=["--n-balls-train", "1", "--n-balls-eval", "2",
              "--max-rules", "3", "--mcts-sims", "512", "--seed", "0"],
    ),
    "runB": dict(
        stem="runB_seed0_sims512", slow=True,
        args=["--n-balls-train", "2", "--n-balls-eval", "3",
              "--max-rules", "4", "--mcts-sims", "512", "--seed", "0"],
    ),
}


def _run_one(label: str, spec: dict) -> dict:
    stem = spec["stem"]
    best_raw = RAW_DIR / f"{stem}_best.jsonl"
    all_raw = RAW_DIR / f"{stem}_all.jsonl"
    # Stage 3-A flipped the grammar-safety flags to default-ON; the Stage-2/2.5
    # canonical numbers in docs/notes/stage4/02.md (Table 1, the figures) were
    # produced under the *legacy* permissive grammar, so pin it here. Use the
    # Stage-3-A diagnostic-grid driver for the strict-vs-legacy comparison.
    cmd = [sys.executable, str(SMOKE), *spec["args"], "--grammar", "legacy",
           "--out-jsonl", str(best_raw), "--dump-all-jsonl", str(all_raw)]
    print(f"[canonical] {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    rows = [json.loads(line) for line in all_raw.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{all_raw} is empty — smoke run produced no policies")

    # committed: compact scores CSV + a copy of the (small) best-so-far progression
    with (DATA_DIR / f"{stem}_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_SCORE_COLS)
        for r in rows:
            w.writerow([r[c] for c in _SCORE_COLS])
    shutil.copyfile(best_raw, DATA_DIR / f"{stem}_best.jsonl")

    best = max(rows, key=lambda r: r["score"])
    r0 = rows[0]
    return {
        "seed": r0["seed"], "mcts_sims": r0["mcts_sims"], "max_rules": r0["max_rules"],
        "n_balls_train": r0["n_balls_train"], "n_balls_eval": r0["n_balls_eval"],
        "n_episodes": 64,
        "n_unique": len(rows),
        "best_score": best["score"],
        "n_solving": sum(1 for r in rows if r["train_solve_rate"] >= 1.0),
        "n_generalising": sum(1 for r in rows
                              if r["train_solve_rate"] >= 1.0 and r["eval_out_solve_rate"] >= 1.0),
        "first_solver_index": next((i for i, r in enumerate(rows)
                                    if r["train_solve_rate"] >= 1.0), None),
        "best_num_rules": best["num_rules"],
        "best_num_literals": best["num_literals"],
        "best_avg_steps": best["avg_steps"],
        "best_train_solve_rate": best["train_solve_rate"],
        "best_eval_out_solve_rate": best["eval_out_solve_rate"],
        "best_policy_pretty": best["policy_pretty"],
        "scores_csv": str((DATA_DIR / f"{stem}_scores.csv").relative_to(REPO_ROOT)),
        "best_jsonl": str((DATA_DIR / f"{stem}_best.jsonl").relative_to(REPO_ROOT)),
        "all_jsonl_raw": str(all_raw.relative_to(REPO_ROOT)) + "  (gitignored — regenerate)",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-slow", action="store_true",
                    help="only regenerate Run A @ 128 sims (~1 min); skip the 512-sim runs")
    args = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_DIR / "summary.json"
    # Preserve entries for runs we are skipping (e.g. --skip-slow) so the figure
    # driver, which needs all three, keeps working.
    summary: dict = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for key, spec in RUNS.items():
        if args.skip_slow and spec["slow"]:
            continue
        summary[key] = _run_one(key, spec)

    summary_path.write_text(json.dumps({k: summary[k] for k in RUNS if k in summary},
                                       indent=2) + "\n")
    print(f"[canonical] wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
