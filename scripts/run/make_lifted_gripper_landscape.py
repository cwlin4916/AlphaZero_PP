#!/usr/bin/env python3
"""Regenerate the committed Stage-2.5 landscape artifacts under
``docs/notes/stage4/data/``.

Legacy: incompatible with the occurrence-introduced-variable grammar — it still
passes ``max_aux_vars`` to ``LiftedGrammarConfig`` (removed; see
``docs/notes/stage4/02_plan.md``). Kept as the aux-var-grammar record; not
ported. If ever revived, it would adopt the ``scripts/run/_parallel`` cell
fan-out (``--jobs``) like ``make_lifted_gripper_mcts_minimal``.

Runs ``analyze_lifted_gripper_landscape.py`` for two canonical configs ×
four grammar-safety constraint settings:

  r1 exhaustive   max_rules=1, max_pre_literals=2, max_goal_literals=1, max_aux_vars=1
                  → every complete policy enumerated under each setting
  r3 sampled      max_rules=3, max_pre_literals=3, max_goal_literals=1, max_aux_vars=1
                  → seeded random walks, up to --max-policies (default 5000)

For each config, all four constraint settings (``none / goal_predicate /
connected / both``) are dumped into ONE summary JSON + ONE per-policy CSV.

Outputs (committed under ``docs/notes/stage4/data/``):

  landscape_r1.json   landscape_r1.csv
  landscape_r3.json   landscape_r3.csv

Run from repo root:

    python scripts/run/make_lifted_gripper_landscape.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANDSCAPE = REPO_ROOT / "scripts" / "run" / "analyze_lifted_gripper_landscape.py"
DATA_DIR = REPO_ROOT / "docs" / "notes" / "stage4" / "data"

CONFIGS = {
    "r1": dict(
        common=["--max-rules", "1", "--max-pre-literals", "2",
                "--max-goal-literals", "1", "--max-aux-vars", "1"],
        max_policies="200000",   # generous so r1 stays exhaustive under every constraint
    ),
    "r3": dict(
        common=["--max-rules", "3", "--max-pre-literals", "3",
                "--max-goal-literals", "1", "--max-aux-vars", "1"],
        max_policies="5000",
    ),
}


def _run_one(key: str, spec: dict, *, seed: int) -> None:
    out_json = DATA_DIR / f"landscape_{key}.json"
    out_csv = DATA_DIR / f"landscape_{key}.csv"
    cmd = [
        sys.executable, str(LANDSCAPE),
        *spec["common"],
        "--constraints", "none", "goal_predicate", "connected", "both",
        "--max-policies", spec["max_policies"],
        "--seed", str(seed),
        "--output-json", str(out_json),
        "--output-csv", str(out_csv),
    ]
    print(f"[landscape-canonical] {key}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the stratified-sampling fallback (r3)")
    ap.add_argument("--only", choices=list(CONFIGS),
                    help="only regenerate one config (debug)")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    keys = [args.only] if args.only else list(CONFIGS)
    for key in keys:
        _run_one(key, CONFIGS[key], seed=args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
