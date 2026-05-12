"""Hand-written lifted policy for Gripper-lite.

The four rules below are the Stage 1 reference policy. They solve any
``GripperLiteEnv`` instance regardless of ball count, with the lex-min
tie-break in the interpreter selecting ``ball_0`` first, then ``ball_1``,
and so on.

Rule order matters — first-applicable semantics:

1. carrying(?b) ∧ at_robot(?r) ∧ Goal at_ball(?b, ?r)        ⇒ drop(?b, ?r)
2. carrying(?b) ∧ at_robot(?from) ∧ Goal at_ball(?b, ?to)    ⇒ move(?from, ?to)
3. at_ball(?b, ?r) ∧ at_robot(?r) ∧ handempty()
                  ∧ ¬Goal at_ball(?b, ?r)                    ⇒ pick(?b, ?r)
4. at_robot(?from) ∧ at_ball(?b, ?to) ∧ handempty()
                  ∧ ¬Goal at_ball(?b, ?to)                   ⇒ move(?from, ?to)
"""

from __future__ import annotations

from alphazeropp.synthesis.lifted_dsl import (
    LiftedAction,
    Policy,
    Rule,
    Var,
    goal_lit,
    state_lit,
)


def hand_policy() -> Policy:
    b = Var("?b", "ball")
    r = Var("?r", "room")
    rfrom = Var("?from", "room")
    rto = Var("?to", "room")

    rule1 = Rule(
        vars=(b, r),
        body=(
            state_lit("carrying", b),
            state_lit("at_robot", r),
            goal_lit("at_ball", b, r),
        ),
        action=LiftedAction("drop", (b, r)),
    )

    rule2 = Rule(
        vars=(b, rfrom, rto),
        body=(
            state_lit("carrying", b),
            state_lit("at_robot", rfrom),
            goal_lit("at_ball", b, rto),
        ),
        action=LiftedAction("move", (rfrom, rto)),
    )

    rule3 = Rule(
        vars=(b, r),
        body=(
            state_lit("at_ball", b, r),
            state_lit("at_robot", r),
            state_lit("handempty"),
            goal_lit("at_ball", b, r, negated=True),
        ),
        action=LiftedAction("pick", (b, r)),
    )

    rule4 = Rule(
        vars=(b, rfrom, rto),
        body=(
            state_lit("at_robot", rfrom),
            state_lit("at_ball", b, rto),
            state_lit("handempty"),
            goal_lit("at_ball", b, rto, negated=True),
        ),
        action=LiftedAction("move", (rfrom, rto)),
    )

    return Policy((rule1, rule2, rule3, rule4))


def degenerate_drop_policy() -> Policy:
    """The Stage-2 do-nothing attractor: ``⊤ ⇒ drop(?b_0, ?r_1)``.

    ``drop`` is never legal from any reachable state where the robot is not
    carrying a ball, so :func:`interpret` returns ``None`` on the first step
    and the rollout stalls. The leaf score is roughly ``-0.05`` (one noop
    penalty, no progress). Stage 2.5 uses this policy as the reference
    "non-success" — it has nonzero leaf score but is **not** a positive hit
    and **not** a solver. Variable names follow the grammar's schema-position
    convention (``?b_0`` for ball arg 0, ``?r_1`` for room arg 1).
    """
    b0 = Var("?b_0", "ball")
    r1 = Var("?r_1", "room")
    rule = Rule(
        vars=(b0, r1),
        body=(),
        action=LiftedAction("drop", (b0, r1)),
    )
    return Policy((rule,))
