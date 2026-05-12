"""Stage 2 (minimal) — smoke tests for the uniform-MCTS run script.

Runs ``scripts/run/run_lifted_gripper_mcts_minimal.py`` in-process on a tiny
budget for B=1 and B=2 and checks the JSONL + summary shape, the
solver/reasonable/degenerate bucketing, and that the hand policy classifies as
a solver. See ``docs/notes/stage4/02_plan.md``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_PATH = _REPO_ROOT / "scripts" / "run" / "run_lifted_gripper_mcts_minimal.py"


@pytest.fixture(scope="module")
def runmod():
    spec = importlib.util.spec_from_file_location("lifted_gripper_mcts_minimal", _RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("lifted_gripper_mcts_minimal", mod)
    spec.loader.exec_module(mod)
    return mod


def _run(runmod, tmp_path, balls, sims=16, episodes=3, seed=0):
    out_jsonl = tmp_path / f"b{balls}.jsonl"
    out_summary = tmp_path / f"b{balls}_summary.json"
    rc = runmod.main([
        "--balls", str(balls), "--mcts-sims", str(sims), "--n-episodes", str(episodes),
        "--seed", str(seed), "--out-jsonl", str(out_jsonl), "--out-summary", str(out_summary),
        "--log-all-terminals",
    ])
    assert rc == 0
    summary = json.loads(out_summary.read_text())
    records = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
    return summary, records


@pytest.mark.parametrize("balls", [1, 2])
def test_run_smoke_shape(runmod, tmp_path, balls):
    summary, records = _run(runmod, tmp_path, balls)
    # summary keys
    for k in ("balls", "max_rules", "mcts_sims", "n_episodes", "seed", "n_unique",
              "best_score", "best_policy_pretty", "best_metrics", "n_solving",
              "first_solver_episode", "first_solver_unique_index", "counts", "git_commit"):
        assert k in summary, k
    assert summary["balls"] == balls
    assert summary["max_rules"] == (3 if balls == 1 else 4)
    assert set(summary["counts"]) == {"solver", "reasonable", "degenerate"}
    assert sum(summary["counts"].values()) == summary["n_unique"]
    # there is at least one best_so_far record
    assert any(r.get("kind") == "best_so_far" for r in records)
    # every terminal record carries a classification in the three buckets
    for r in records:
        if r.get("kind") == "terminal":
            assert r["classification"] in {"solver", "reasonable", "degenerate"}


@pytest.mark.parametrize("balls", [1, 2])
def test_classification_buckets_consistent(runmod, tmp_path, balls):
    summary, records = _run(runmod, tmp_path, balls, episodes=4)
    terminals = [r for r in records if r.get("kind") == "terminal"]
    assert len(terminals) == summary["n_unique"]
    for r in terminals:
        if r["classification"] == "solver":
            assert r["solved"] is True
        if r["classification"] == "degenerate":
            # degenerate ⇒ no progress, or it stalled (steps == 0 with a noop)
            assert (r["progress"] == 0) or (r["steps"] == 0.0)
    n_by_bucket = {b: sum(1 for r in terminals if r["classification"] == b)
                   for b in ("solver", "reasonable", "degenerate")}
    assert n_by_bucket == summary["counts"]


@pytest.mark.parametrize("balls", [1, 2])
def test_hand_policy_classifies_as_solver(runmod, balls):
    from alphazeropp.instances.gripper_lite.policies import hand_policy
    info = runmod.classify_policy(hand_policy(), balls)
    assert info["classification"] == "solver"
    assert info["solved"] is True
    assert set(info["schemas_fired"]) >= {"pick", "move", "drop"}
