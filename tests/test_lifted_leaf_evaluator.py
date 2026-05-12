"""Stage 2.5 tests for ``LiftedLeafEvaluator``.

Covers the C.1–C.3 spec items: ``num_rule_evals`` is the preferred Stage-2.5
key and ``num_binding_attempts`` is a documented-deprecated alias that still
emits the same value; ``aggregate_metrics_for`` returns exactly the shape the
score-variant module consumes.
"""

from __future__ import annotations

import inspect

import alphazeropp.synthesis.lifted_leaf_evaluator as lle
from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator


def _evaluator():
    train = [GripperLiteEnv(n_balls=1, seed=0)]
    eval_in = [GripperLiteEnv(n_balls=1, seed=1)]
    eval_out = [GripperLiteEnv(n_balls=2, seed=2)]
    return LiftedLeafEvaluator(train, eval_in, eval_out)


def test_evaluator_returns_num_rule_evals_and_alias():
    """``metrics_for`` must emit both ``num_rule_evals`` (preferred) and
    ``num_binding_attempts`` (deprecated alias), with identical values; the
    module docstring must mark ``num_binding_attempts`` deprecated."""
    ev = _evaluator()
    metrics = ev.metrics_for(hand_policy())
    assert "num_rule_evals" in metrics
    assert "num_binding_attempts" in metrics
    assert metrics["num_rule_evals"] == metrics["num_binding_attempts"]
    # both are non-negative ints (they sum `rule_idx + 1` per firing step).
    assert isinstance(metrics["num_rule_evals"], int)
    assert metrics["num_rule_evals"] >= 0

    # docstring documents the deprecation.
    src = inspect.getsource(lle)
    assert "deprecated" in src.lower()
    assert "num_binding_attempts" in src


def test_aggregate_metrics_for_shape():
    """``aggregate_metrics_for`` returns exactly the four keys the score-variant
    module consumes — ``solve_rate / avg_progress / avg_steps / num_noops``."""
    ev = _evaluator()
    m = ev.aggregate_metrics_for(hand_policy())
    assert set(m.keys()) == {"solve_rate", "avg_progress", "avg_steps", "num_noops"}
    # hand policy solves B=1 in 3 steps → solve_rate 1.0, avg_steps 3.0, no noops.
    assert m["solve_rate"] == 1.0
    assert m["avg_steps"] == 3.0
    assert m["num_noops"] == 0
