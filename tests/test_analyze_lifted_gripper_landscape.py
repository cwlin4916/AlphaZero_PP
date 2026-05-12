"""Smoke tests for the Stage-2.5 landscape-enumeration driver.

Acceptance criterion 4 of the plan: the script must run at least one small
exhaustive config and one sampled larger config. These tests check both and
assert the per-constraint summary schema + the reproducibility of seeded
sampling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run.analyze_lifted_gripper_landscape import main as landscape_main

# Stage 2 (occurrence-introduced grammar): the Stage-2.5 landscape driver is a
# *legacy* artifact of the aux-var grammar — its ``_grammar_config(max_aux_vars=…)``
# helper no longer matches ``LiftedGrammarConfig`` (``max_aux_vars`` was removed in
# favour of ``max_body_local_vars``). The driver is intentionally left unported (see
# ``docs/notes/stage4/02_plan.md`` — "legacy artifacts of the aux-var grammar"), so
# these smoke tests are skipped rather than fixed.
pytestmark = pytest.mark.skip(
    reason="legacy Stage-2.5 landscape driver — incompatible with the occurrence-"
           "introduced grammar (max_aux_vars removed); see docs/notes/stage4/02_plan.md"
)

_REQUIRED_KEYS = {
    "mode",
    "total_policies_evaluated",
    "unique_policies_evaluated",
    "solver_count_by_B",
    "b1_solver_count",
    "b1_to_b2_generalizing_count",
    "b2_solver_count",
    "hand_policy_score",
    "hand_policy_rank",
    "degenerate_drop_policy_score",
    "degenerate_drop_policy_rank",
    "vacuous_goal_policy_count",
    "disconnected_goal_policy_count",
    "best_policy_by_variant",
    "size_strata",
    "wall_seconds",
}


def _run(args, tmp_path: Path) -> dict:
    out_json = tmp_path / "landscape.json"
    out_csv = tmp_path / "landscape.csv"
    rc = landscape_main(args + ["--output-json", str(out_json), "--output-csv", str(out_csv)])
    assert rc == 0
    return json.loads(out_json.read_text())


def test_landscape_smoke_exhaustive(tmp_path):
    """Tiny r1 config (max_rules=1, max_pre_literals=1) → both constraint
    settings hit exhaustive mode and the required summary fields are present.
    ``both`` setting must have zero vacuous/disconnected pathologies."""
    summary = _run(
        [
            "--max-rules", "1", "--max-pre-literals", "1",
            "--constraints", "none", "both",
            "--max-policies", "200000",
        ],
        tmp_path,
    )
    assert "none" in summary and "both" in summary
    for key in ("none", "both"):
        block = summary[key]
        assert block["mode"] == "exhaustive"
        assert _REQUIRED_KEYS <= set(block.keys()), (
            f"missing keys in {key}: {_REQUIRED_KEYS - set(block.keys())}"
        )
        assert isinstance(block["solver_count_by_B"], dict)
        assert set(map(int, block["solver_count_by_B"].keys())) == {1, 2, 3, 4}
    # the safety flags must eliminate both pathologies by construction
    assert summary["both"]["vacuous_goal_policy_count"] == 0
    assert summary["both"]["disconnected_goal_policy_count"] == 0
    # and the `none` baseline must show *some* pathology (otherwise the
    # filters would be vacuous tests)
    assert (summary["none"]["vacuous_goal_policy_count"]
            + summary["none"]["disconnected_goal_policy_count"]) > 0


def test_landscape_smoke_sampled(tmp_path):
    """Mid-size r3 config (max_rules=3, max_pre_literals=3) → sampling mode at
    --max-policies 50; two runs with the same --seed produce identical per-
    policy CSVs (modulo wall-time fields in the JSON)."""
    args = [
        "--max-rules", "3", "--max-pre-literals", "3",
        "--constraints", "none",
        "--max-policies", "50", "--seed", "7",
    ]
    out_json_a = tmp_path / "a.json"
    out_csv_a = tmp_path / "a.csv"
    out_json_b = tmp_path / "b.json"
    out_csv_b = tmp_path / "b.csv"
    assert landscape_main(args + ["--output-json", str(out_json_a),
                                  "--output-csv", str(out_csv_a)]) == 0
    summary = json.loads(out_json_a.read_text())
    assert summary["none"]["mode"] == "sampled"
    assert summary["none"]["unique_policies_evaluated"] <= 50

    assert landscape_main(args + ["--output-json", str(out_json_b),
                                  "--output-csv", str(out_csv_b)]) == 0
    # The CSV is byte-for-byte stable across same-seed runs.
    assert out_csv_a.read_bytes() == out_csv_b.read_bytes()
