"""Stage 1 DSL tests for the synthesis-side lifted DSL.

Named ``_v2`` to avoid collision with the existing Doors-specific
``tests/test_lifted_dsl.py``.
"""

from __future__ import annotations

import pytest

from alphazeropp.synthesis.lifted_dsl import (
    Atom,
    LiftedAction,
    PredicateSchema,
    Rule,
    Var,
    check_atom_against_schema,
    goal_lit,
    state_lit,
)


def test_type_check_rejects_bad_atom():
    """A room-typed arg position must reject a ball-typed variable."""
    at_ball_schema = PredicateSchema("at_ball", ("ball", "room"))
    bad_atom = Atom("at_ball", (Var("?b", "ball"), Var("?r", "ball")))
    # var_types maps variable name -> declared type
    var_types = {"?b": "ball", "?r": "ball"}
    with pytest.raises(ValueError, match="expects 'room'"):
        check_atom_against_schema(bad_atom, at_ball_schema, var_types)


def test_safe_negation_rejected_at_rule_construction():
    """A negative literal whose variables are not positively bound must raise."""
    b = Var("?b", "ball")
    r = Var("?r", "room")
    # Negative literal mentions ?r but ?r appears nowhere positive and is
    # not in the action args either → unsafe.
    with pytest.raises(ValueError, match="unsafe negation"):
        Rule(
            vars=(b, r),
            body=(
                state_lit("carrying", b),
                goal_lit("at_ball", b, r, negated=True),
            ),
            action=LiftedAction("drop", (b,)),
        )


def test_safe_negation_accepts_action_bound_var():
    """A negative-literal var that appears in the action args is safe."""
    b = Var("?b", "ball")
    r = Var("?r", "room")
    # ?r is in action.args, so the negative goal literal mentioning ?r is safe.
    Rule(
        vars=(b, r),
        body=(
            state_lit("carrying", b),
            goal_lit("at_ball", b, r, negated=True),
        ),
        action=LiftedAction("drop", (b, r)),
    )


def test_rule_rejects_undeclared_variable():
    b = Var("?b", "ball")
    r = Var("?r", "room")
    with pytest.raises(ValueError, match="not declared"):
        Rule(
            vars=(b,),
            body=(state_lit("at_ball", b, r),),
            action=LiftedAction("drop", (b,)),
        )


def test_atom_pretty_renders_vars_and_constants():
    a = Atom("at_ball", (Var("?b", "ball"), "room_b"))
    assert a.pretty() == "at_ball(?b, room_b)"
