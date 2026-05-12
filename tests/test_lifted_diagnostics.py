"""Stage 3-A tests for the per-policy grammar-pathology report.

See ``docs/notes/stage4/03_plan.md`` Task C and ``src/alphazeropp/synthesis/lifted_diagnostics.py``.
"""

from __future__ import annotations

from alphazeropp.instances.gripper_lite.policies import (
    degenerate_drop_policy,
    hand_policy,
)
from alphazeropp.synthesis.lifted_diagnostics import (
    analyze_policy_pathologies,
    analyze_policy_pathologies_for,
)
from alphazeropp.synthesis.lifted_dsl import (
    LiftedAction,
    Policy,
    Rule,
    Var,
    goal_lit,
    state_lit,
)
from alphazeropp.synthesis.lifted_grammar import gripper_lite_signature

_GRIPPER_GOAL_PREDS = ("at_ball",)

_EXPECTED_KEYS = {
    "has_vacuous_goal_predicate",
    "has_goal_only_variable",
    "has_empty_body_rule",
    "has_top_drop_rule",
    "has_top_pick_rule",
    "has_top_move_rule",
    "num_goal_literals",
    "num_negative_goal_literals",
    "num_rules",
    "num_body_literals",
}


def _brittle_runA_policy() -> Policy:
    """The Stage-2 brittle "best" $B=1$ policy (02.md): a vacuous goal predicate
    (``Goal[carrying(?b_0)]``) and a goal-only auxiliary variable (``?aux_0`` in
    ``Goal[at_ball(?aux_0, ?r_1)]``)."""
    b0, r1, r0 = Var("?b_0", "ball"), Var("?r_1", "room"), Var("?r_0", "room")
    aux0 = Var("?aux_0", "ball")
    rho1 = Rule(
        vars=(b0, r1, aux0),
        body=(state_lit("at_robot", r1), state_lit("carrying", b0), goal_lit("at_ball", aux0, r1)),
        action=LiftedAction("drop", (b0, r1)),
    )
    rho2 = Rule(
        vars=(b0, r1),
        body=(state_lit("at_robot", r1), goal_lit("carrying", b0, negated=True)),
        action=LiftedAction("pick", (b0, r1)),
    )
    rho3 = Rule(vars=(r0, r1), body=(), action=LiftedAction("move", (r0, r1)))
    return Policy((rho1, rho2, rho3))


def test_keys_are_exactly_the_documented_set():
    rep = analyze_policy_pathologies(hand_policy(), goal_predicate_names=_GRIPPER_GOAL_PREDS)
    assert set(rep) == _EXPECTED_KEYS


def test_hand_policy_has_no_pathologies():
    rep = analyze_policy_pathologies(hand_policy(), goal_predicate_names=_GRIPPER_GOAL_PREDS)
    assert rep["has_vacuous_goal_predicate"] is False
    assert rep["has_goal_only_variable"] is False
    assert rep["has_empty_body_rule"] is False
    assert rep["has_top_drop_rule"] is False
    assert rep["has_top_pick_rule"] is False
    assert rep["has_top_move_rule"] is False
    assert rep["num_rules"] == 4
    assert rep["num_goal_literals"] == 4
    assert rep["num_negative_goal_literals"] == 2  # ρ₃ and ρ₄ use ¬Goal[at_ball(...)]
    assert rep["num_body_literals"] == sum(len(r.body) for r in hand_policy().rules)


def test_hand_policy_for_signature_matches_explicit():
    sig = gripper_lite_signature()
    assert analyze_policy_pathologies_for(hand_policy(), sig) == analyze_policy_pathologies(
        hand_policy(), goal_predicate_names=sig.goal_predicate_names
    )


def test_no_whitelist_means_no_vacuous_predicate():
    """With ``goal_predicate_names=None`` there is no notion of "vacuous", so the
    flag is False even for ``Goal[carrying(...)]``."""
    rep = analyze_policy_pathologies(_brittle_runA_policy(), goal_predicate_names=None)
    assert rep["has_vacuous_goal_predicate"] is False
    # the disconnected-aux pathology is structural and still detected
    assert rep["has_goal_only_variable"] is True


def test_brittle_runA_policy_flagged():
    rep = analyze_policy_pathologies(_brittle_runA_policy(), goal_predicate_names=_GRIPPER_GOAL_PREDS)
    assert rep["has_vacuous_goal_predicate"] is True   # Goal[carrying(?b_0)]
    assert rep["has_goal_only_variable"] is True       # ?aux_0 only in Goal[at_ball(?aux_0, ?r_1)]
    assert rep["has_empty_body_rule"] is True          # ρ₃: ⊤ ⇒ move(?r_0, ?r_1)
    assert rep["has_top_move_rule"] is True
    assert rep["has_top_drop_rule"] is False
    assert rep["num_rules"] == 3
    assert rep["num_goal_literals"] == 2
    assert rep["num_negative_goal_literals"] == 1      # ¬Goal[carrying(?b_0)]


def test_degenerate_drop_policy_flagged():
    rep = analyze_policy_pathologies(degenerate_drop_policy(), goal_predicate_names=_GRIPPER_GOAL_PREDS)
    assert rep["has_empty_body_rule"] is True
    assert rep["has_top_drop_rule"] is True
    assert rep["has_top_move_rule"] is False
    assert rep["has_top_pick_rule"] is False
    assert rep["has_vacuous_goal_predicate"] is False
    assert rep["has_goal_only_variable"] is False
    assert rep["num_rules"] == 1
    assert rep["num_goal_literals"] == 0
    assert rep["num_body_literals"] == 0


def test_pure_and_deterministic():
    p = _brittle_runA_policy()
    a = analyze_policy_pathologies(p, goal_predicate_names=_GRIPPER_GOAL_PREDS)
    b = analyze_policy_pathologies(p, goal_predicate_names=_GRIPPER_GOAL_PREDS)
    assert a == b
