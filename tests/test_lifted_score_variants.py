"""Stage 2.5 tests for :mod:`alphazeropp.synthesis.lifted_score_variants`.

Pins the five score-variant formulae to the spec values and asserts that the
Stage-2 do-nothing attractor ``⊤ ⇒ drop(?b_0, ?r_1)`` is *not* described as a
success by any of them — it may have nonzero leaf score (≈ −0.05) but it is
neither a positive hit nor a solver.
"""

from __future__ import annotations

import math

import pytest

from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import degenerate_drop_policy
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator
from alphazeropp.synthesis.lifted_score_variants import (
    SCORE_VARIANT_NAMES,
    all_score_variants,
    score_current,
    score_lexicographic,
    score_no_noop_penalty,
    score_no_step_penalty,
    score_progress_only,
)


def test_score_variants_on_solving_policy():
    """Variant formulae match the Stage-2.5 spec on a synthetic solver-shaped
    aggregate (``solve_rate=1, progress=1, steps=7, noops=0``)."""
    m = {"solve_rate": 1.0, "avg_progress": 1.0, "avg_steps": 7.0, "num_noops": 0}
    assert math.isclose(score_current(m), 1.0 + 0.25 - 0.07)        # 1.18
    assert math.isclose(score_no_step_penalty(m), 1.25)
    assert math.isclose(score_no_noop_penalty(m), 1.0 + 0.25 - 0.07)
    assert math.isclose(score_progress_only(m), 1.25)
    assert score_lexicographic(m) == (1.0, 1.0, -7.0, 0.0)

    bundle = all_score_variants(m)
    assert set(bundle.keys()) == set(SCORE_VARIANT_NAMES)
    assert bundle["current"] == pytest.approx(1.18)
    assert bundle["progress_only"] == pytest.approx(1.25)
    assert bundle["lexicographic"] == (1.0, 1.0, -7.0, 0.0)


def test_degenerate_drop_policy_not_a_success():
    """``⊤ ⇒ drop(?b_0, ?r_1)`` — drop is never legal from the start, so the
    rollout stalls on step 0 with one noop and zero steps. Score ≈ −0.05 (one
    noop penalty), train_solve_rate 0, eval-out solve rate 0, lex tuple's
    first coord 0. Nothing should describe this as a hit or a solver."""
    pi = degenerate_drop_policy()
    ev = LiftedLeafEvaluator(
        train_instances=[GripperLiteEnv(n_balls=1)],
        eval_in_instances=[GripperLiteEnv(n_balls=1)],
        eval_out_instances=[GripperLiteEnv(n_balls=2)],
    )
    diag = ev.metrics_for(pi)
    assert diag["train_solve_rate"] == 0.0
    assert diag["eval_out_solve_rate"] == 0.0
    assert diag["avg_steps"] == 0.0
    assert diag["num_noops"] == 1
    assert diag["score"] == pytest.approx(-0.05)

    m = ev.aggregate_metrics_for(pi)
    variants = all_score_variants(m)
    assert variants["current"] <= 0.0
    assert variants["current"] == pytest.approx(-0.05)
    assert variants["progress_only"] == pytest.approx(0.0)
    # lex tuple — first coord is solve_rate, must be 0 (not a solver).
    assert variants["lexicographic"][0] == 0.0

    # explicit "is this a solver?" assertion (the only definition of solver).
    assert diag["train_solve_rate"] < 1.0
