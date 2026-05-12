"""Stage 1 interpreter tests for ``synthesis/lifted_interpreter.py``."""

from __future__ import annotations

import pytest

from alphazeropp.synthesis.lifted_dsl import (
    GroundAction,
    LiftedAction,
    Policy,
    RelState,
    Rule,
    Var,
    goal_lit,
    state_lit,
)
from alphazeropp.synthesis.lifted_interpreter import (
    find_bindings,
    interpret,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def objs() -> dict[str, tuple[str, ...]]:
    return {"ball": ("ball_0", "ball_1"), "room": ("room_a", "room_b")}


@pytest.fixture
def state() -> RelState:
    return {
        "at_robot": {("room_a",)},
        "at_ball": {("ball_0", "room_a"), ("ball_1", "room_a")},
        "carrying": set(),
        "handempty": {()},
    }


@pytest.fixture
def goal() -> RelState:
    return {
        "at_robot": set(),
        "at_ball": {("ball_0", "room_b"), ("ball_1", "room_b")},
        "carrying": set(),
        "handempty": set(),
    }


# ---------------------------------------------------------------------------
# Positive / negative matching
# ---------------------------------------------------------------------------

def test_positive_state_matching(state, goal, objs):
    b = Var("?b", "ball")
    r = Var("?r", "room")
    rule = Rule(
        vars=(b, r),
        body=(state_lit("at_ball", b, r),),
        action=LiftedAction("pick", (b, r)),
    )
    # Every theta must satisfy at_ball(?b, ?r) ∈ state["at_ball"].
    bindings = find_bindings(
        rule, state, goal, objs,
        legal_actions={
            GroundAction("pick", ("ball_0", "room_a")),
            GroundAction("pick", ("ball_1", "room_a")),
        },
    )
    got = {(th["?b"], th["?r"]) for th in bindings}
    assert got == {("ball_0", "room_a"), ("ball_1", "room_a")}


def test_positive_goal_matching(state, goal, objs):
    """Same rule shape, but read from goal_atoms via LiteralSource.GOAL."""
    b = Var("?b", "ball")
    r = Var("?r", "room")
    rule = Rule(
        vars=(b, r),
        body=(goal_lit("at_ball", b, r),),
        action=LiftedAction("pick", (b, r)),
    )
    legal = {
        GroundAction("pick", ("ball_0", "room_b")),
        GroundAction("pick", ("ball_1", "room_b")),
    }
    bindings = find_bindings(rule, state, goal, objs, legal_actions=legal)
    got = {(th["?b"], th["?r"]) for th in bindings}
    # The state has balls in room_a, but the GOAL says room_b — so we must
    # only see room_b bindings here, proving the discriminator routes correctly.
    assert got == {("ball_0", "room_b"), ("ball_1", "room_b")}


def test_negative_goal_matching(state, goal, objs):
    """¬Goal at_ball(?b, ?r) should pick out (ball, room) pairs NOT in goal."""
    b = Var("?b", "ball")
    r = Var("?r", "room")
    # Positive at_ball ties down ?b and ?r to {(ball_0,room_a),(ball_1,room_a)};
    # ¬Goal at_ball(?b, ?r) further requires (b, r) ∉ goal["at_ball"].
    # Goal["at_ball"] has (b, room_b) pairs only, so room_a survives.
    rule = Rule(
        vars=(b, r),
        body=(
            state_lit("at_ball", b, r),
            goal_lit("at_ball", b, r, negated=True),
        ),
        action=LiftedAction("pick", (b, r)),
    )
    legal = {
        GroundAction("pick", ("ball_0", "room_a")),
        GroundAction("pick", ("ball_1", "room_a")),
    }
    bindings = find_bindings(rule, state, goal, objs, legal_actions=legal)
    got = {(th["?b"], th["?r"]) for th in bindings}
    assert got == {("ball_0", "room_a"), ("ball_1", "room_a")}


# ---------------------------------------------------------------------------
# Determinism, "no rule", no-full-grounding
# ---------------------------------------------------------------------------

def test_interpret_is_deterministic(state, goal, objs):
    b = Var("?b", "ball")
    r = Var("?r", "room")
    rule = Rule(
        vars=(b, r),
        body=(
            state_lit("at_ball", b, r),
            state_lit("at_robot", r),
            state_lit("handempty"),
            goal_lit("at_ball", b, r, negated=True),
        ),
        action=LiftedAction("pick", (b, r)),
    )
    policy = Policy((rule,))
    legal = {
        GroundAction("pick", ("ball_0", "room_a")),
        GroundAction("pick", ("ball_1", "room_a")),
    }
    a1 = interpret(policy, state, goal, objs, legal, trace=True)
    a2 = interpret(policy, state, goal, objs, legal, trace=True)
    assert a1 == a2
    action, idx, theta = a1
    # lex-min must pick ball_0 before ball_1
    assert action == GroundAction("pick", ("ball_0", "room_a"))
    assert idx == 0
    assert theta == {"?b": "ball_0", "?r": "room_a"}


def test_no_rule_returns_none(state, goal, objs):
    """A policy whose sole rule is unsatisfiable returns None."""
    b = Var("?b", "ball")
    rule = Rule(
        vars=(b,),
        body=(state_lit("carrying", b),),  # nothing is being carried
        action=LiftedAction("drop", (b,)),
    )
    legal: set[GroundAction] = set()
    out = interpret(Policy((rule,)), state, goal, objs, legal)
    assert out is None


def test_no_full_grounding_for_two_var_rule(state, goal, objs):
    """A two-var rule with one positive literal should yield O(|state[at_ball]|)
    bindings, not O(|balls| × |rooms|).

    This is the §5.3 Option-B contract: enumerate by rows, not by Cartesian
    product of types.
    """
    b = Var("?b", "ball")
    r = Var("?r", "room")
    rule = Rule(
        vars=(b, r),
        body=(state_lit("at_ball", b, r),),
        action=LiftedAction("pick", (b, r)),
    )
    legal = {GroundAction("pick", (bn, rn))
             for bn in objs["ball"] for rn in objs["room"]}
    bindings = find_bindings(rule, state, goal, objs, legal_actions=legal)
    # state["at_ball"] has exactly 2 rows; the full Cartesian would be 2*2=4.
    # The interpreter must yield exactly 2.
    assert len(bindings) == 2
