"""Hand-written lifted policy for the Doors PDDL relational adapter (Stage 1 rev. 1).

The three rules below are the Stage-1 reference policy for the Doors domain (see
``docs/notes/stage4/01_plan.md``). They dispatch through the generic
:func:`alphazeropp.synthesis.lifted_interpreter.interpret` over
:class:`alphazeropp.instances.doors.doors_pddl_lifted.DoorsPDDLLiteRelationalEnv`, and happen to
solve the two shipped layouts — D2 in 3 steps, D3 in 5 steps (a chained-unlock layout with K keys
takes ``2K + 1`` steps). This is a *semantic-core demonstration on a second domain*, not a
generalization claim and not an MCTS result — Stage 1 has no grammar and no search.

First-applicable semantics — rule order matters:

1. Goal[at_loc(?l)] ∧ unlocked(?r) ∧ loc_in_room(?l, ?r)              ⇒ move_to(?l)
2. at_loc(?l) ∧ key_at(?k, ?l) ∧ key_avail(?k)                       ⇒ pick(?k)
3. key_at(?k, ?l) ∧ key_avail(?k) ∧ loc_in_room(?l, ?r) ∧ unlocked(?r) ⇒ move_to(?l)

ρ₁: if the goal location's room is reachable (unlocked), walk there. ρ₂: if standing on an
available key, pick it (unlocking a room). ρ₃: otherwise, walk to a reachable available key.
No negation appears, so safe-negation is vacuously satisfied.
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


def doors_hand_policy() -> Policy:
    l = Var("?l", "location")
    r = Var("?r", "room")
    k = Var("?k", "key")

    rho1 = Rule(  # goal location's room is unlocked -> walk to the goal
        vars=(l, r),
        body=(
            goal_lit("at_loc", l),
            state_lit("unlocked", r),
            state_lit("loc_in_room", l, r),
        ),
        action=LiftedAction("move_to", (l,)),
    )

    rho2 = Rule(  # standing on an available key -> pick it
        vars=(l, k),
        body=(
            state_lit("at_loc", l),
            state_lit("key_at", k, l),
            state_lit("key_avail", k),
        ),
        action=LiftedAction("pick", (k,)),
    )

    rho3 = Rule(  # an available key is in a reachable (unlocked) room -> walk to it
        vars=(l, r, k),
        body=(
            state_lit("key_at", k, l),
            state_lit("key_avail", k),
            state_lit("loc_in_room", l, r),
            state_lit("unlocked", r),
        ),
        action=LiftedAction("move_to", (l,)),
    )

    return Policy((rho1, rho2, rho3))


def degenerate_noop_policy() -> Policy:
    """``⊤ ⇒ noop()`` — always fires ``noop``; the rollout never solves and the state never
    changes. The Stage-1 "trivial / no-progress" reference policy, mirroring
    :func:`alphazeropp.instances.gripper_lite.policies.degenerate_drop_policy` for Gripper-lite.
    """
    return Policy((Rule(vars=(), body=(), action=LiftedAction("noop", ())),))
