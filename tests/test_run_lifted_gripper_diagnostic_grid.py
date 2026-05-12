"""Stage 3-A smoke test for the reproducible diagnostic-grid driver.

Runs ``scripts/run/run_lifted_gripper_diagnostic_grid.py`` in-process on a tiny grid
(1 seed, sims=8, 1 episode, run A, strict grammar, both baselines) and checks that the
documented artifacts appear with the expected shape. See ``docs/notes/stage4/03_plan.md`` Task D.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DRIVER_PATH = _REPO_ROOT / "scripts" / "run" / "run_lifted_gripper_diagnostic_grid.py"

# CSV columns the plotting script + 03.md rely on.
_REQUIRED_CSV_COLUMNS = {
    "run_id", "grammar_config_name", "baseline", "seed", "sims",
    "unique_policies", "best_score", "first_solver_idx", "solver_count",
    "root_entropy_mean", "root_entropy_final",
    "n_vacuous_goal", "n_goal_only_var", "n_empty_body", "n_top_drop", "n_top_pick", "n_top_move",
    "git_commit",
}
_PATHOLOGY_FIELDS = {
    "has_vacuous_goal_predicate", "has_goal_only_variable", "has_empty_body_rule",
    "has_top_drop_rule", "has_top_pick_rule", "has_top_move_rule",
    "num_goal_literals", "num_negative_goal_literals", "num_rules", "num_body_literals",
}


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("diagnostic_grid_driver", _DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("diagnostic_grid_driver", mod)
    spec.loader.exec_module(mod)
    return mod


def test_diagnostic_grid_smoke(tmp_path, driver):
    out_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    rc = driver.main([
        "--seeds", "0", "--sims", "8", "--episodes", "1",
        "--runs", "A", "--grammars", "strict", "--baselines", "mcts", "random",
        "--out-dir", str(out_dir), "--raw-dir", str(raw_dir),
    ])
    assert rc == 0

    # rolled-up CSV with the documented header + ≥1 row
    csv_path = out_dir / "diagnostic_grid.csv"
    assert csv_path.exists()
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = set(reader.fieldnames or [])
        rows = list(reader)
    assert _REQUIRED_CSV_COLUMNS <= header
    assert len(rows) >= 1
    run_ids = {r["run_id"] for r in rows}
    assert "A_strict_mcts_seed0_sims8" in run_ids
    assert "A_strict_random_seed0_simsna" in run_ids

    # per-cell artifacts for the mcts cell
    mcts_cell = out_dir / "A_strict_mcts_seed0_sims8"
    cfg = json.loads((mcts_cell / "config.json").read_text())
    assert cfg["grammar_config_name"] == "strict"
    assert cfg["grammar_config"]["goal_predicate_relevance"] is True
    assert cfg["grammar_config"]["require_goal_var_connected"] is True
    summary = json.loads((mcts_cell / "summary.json").read_text())
    assert summary["unique_policies"] >= 1
    best_lines = [ln for ln in (mcts_cell / "best.jsonl").read_text().splitlines() if ln.strip()]
    assert len(best_lines) >= 1
    rec0 = json.loads(best_lines[0])
    assert _PATHOLOGY_FIELDS <= set(rec0)          # pathology dict is attached to every record
    assert rec0["grammar_config_name"] == "strict"

    # full per-policy log lives under the raw (gitignored) dir
    all_path = raw_dir / "A_strict_mcts_seed0_sims8" / "all.jsonl"
    assert all_path.exists()
    all_lines = [ln for ln in all_path.read_text().splitlines() if ln.strip()]
    assert len(all_lines) == summary["unique_policies"]
    for ln in all_lines:
        rec = json.loads(ln)
        assert _PATHOLOGY_FIELDS <= set(rec)
        # Stage-3-A strict grammar: no spurious goal-literal pathologies
        assert rec["has_vacuous_goal_predicate"] is False
        assert rec["has_goal_only_variable"] is False

    # the random baseline cell also produced records
    rand_summary = json.loads((out_dir / "A_strict_random_seed0_simsna" / "summary.json").read_text())
    assert rand_summary["baseline"] == "random"
    assert rand_summary["unique_policies"] >= 1
