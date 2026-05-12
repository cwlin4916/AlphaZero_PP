"""Stage 1 (rev. 1) — the generic lifted interpreter dispatching policies over Doors PDDL.

The same ``alphazeropp.synthesis.lifted_interpreter.interpret`` that drives Gripper-lite must
drive ``DoorsPDDLLiteRelationalEnv``: pick legal Doors actions, expose a deterministic trace, and
(with the hand-written 3-rule policy) solve the two shipped layouts. See
``docs/notes/stage4/01_plan.md``. No grammar, no MCTS — Stage 1.
"""

from __future__ import annotations

import pytest

from alphazeropp.instances.doors.doors_pddl_lifted import DoorsPDDLLiteRelationalEnv
from alphazeropp.instances.doors.doors_pddl_policies import (
    degenerate_noop_policy,
    doors_hand_policy,
)
from alphazeropp.synthesis.lifted_dsl import (
    GroundAction,
    LiftedAction,
    Policy,
    Rule,
    Var,
    state_lit,
)
from alphazeropp.synthesis.lifted_interpreter import interpret


def _rollout(env: DoorsPDDLLiteRelationalEnv):
    """Run a policy through ``env`` to solve / no-firing / horizon.

    Returns ``(actions, traces)`` where each trace is ``(rule_index, theta)``.
    """
    policy = doors_hand_policy()
    actions: list[GroundAction] = []
    traces: list[tuple[int, dict[str, str]]] = []
    for _ in range(env.horizon):
        if env.is_solved():
            break
        out = interpret(
            policy,
            env.get_state_atoms(),
            env.get_goal_atoms(),
            env.get_objects_by_type(),
            env.legal_actions(),
            trace=True,
        )
        if out is None:
            break
        action, rule_idx, theta = out
        actions.append(action)
        traces.append((rule_idx, theta))
        env.step(action)
    return actions, traces


# ---------------------------------------------------------------------------
# item e — a lifted rule produces a legal PICK when the agent is at a key
# ---------------------------------------------------------------------------

def test_pick_rule_picks_colocated_available_key():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    env.step(GroundAction("move_to", ("loc_1",)))     # key_0 lives at loc_1
    l = Var("?l", "location")
    k = Var("?k", "key")
    rule = Rule(
        vars=(l, k),
        body=(state_lit("at_loc", l), state_lit("key_at", k, l), state_lit("key_avail", k)),
        action=LiftedAction("pick", (k,)),
    )
    a = interpret(
        Policy((rule,)),
        env.get_state_atoms(), env.get_goal_atoms(),
        env.get_objects_by_type(), env.legal_actions(),
    )
    assert a == GroundAction("pick", ("key_0",))
    assert a in env.legal_actions()


# ---------------------------------------------------------------------------
# item f — a lifted rule produces a legal MOVE_TO in a small fixture
# ---------------------------------------------------------------------------

def test_move_rule_produces_legal_move():
    env = DoorsPDDLLiteRelationalEnv.make_d2()       # at loc_0, only room_0 unlocked
    l = Var("?l", "location")
    r = Var("?r", "room")
    rule = Rule(
        vars=(l, r),
        body=(state_lit("unlocked", r), state_lit("loc_in_room", l, r)),
        action=LiftedAction("move_to", (l,)),
    )
    a = interpret(
        Policy((rule,)),
        env.get_state_atoms(), env.get_goal_atoms(),
        env.get_objects_by_type(), env.legal_actions(),
    )
    assert a is not None and a.schema == "move_to"
    assert a in env.legal_actions()
    # the chosen location must be in an unlocked room (i.e. not a move into the locked room)
    loc_id = env._loc_id[a.args[0]]
    room_id = env.base.loc_room[loc_id]
    assert ("room_%d" % room_id,) in env.get_state_atoms()["unlocked"]
    assert a != GroundAction("move_to", ("loc_2",))   # loc_2 is in the locked room_1


# ---------------------------------------------------------------------------
# item g — interpret(trace=True) returns the expected rule index + binding
# ---------------------------------------------------------------------------

def test_interpret_trace_doors_d2_step0():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    out = interpret(
        doors_hand_policy(),
        env.get_state_atoms(), env.get_goal_atoms(),
        env.get_objects_by_type(), env.legal_actions(),
        trace=True,
    )
    assert out == (
        GroundAction("move_to", ("loc_1",)),
        2,                                            # rho_3 (index 2): walk to the reachable key
        {"?k": "key_0", "?l": "loc_1", "?r": "room_0"},
    )


def test_interpret_is_deterministic_doors():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    args = (
        doors_hand_policy(),
        env.get_state_atoms(), env.get_goal_atoms(),
        env.get_objects_by_type(), env.legal_actions(),
    )
    assert interpret(*args, trace=True) == interpret(*args, trace=True)


def test_no_rule_returns_none_doors():
    """A rule whose binding exists but is pruned by ``legal_actions`` does not fire."""
    env = DoorsPDDLLiteRelationalEnv.make_d2()        # at loc_0; key_0 is at loc_1, so PICK is illegal here
    k = Var("?k", "key")
    rule = Rule(vars=(k,), body=(state_lit("key_avail", k),), action=LiftedAction("pick", (k,)))
    out = interpret(
        Policy((rule,)),
        env.get_state_atoms(), env.get_goal_atoms(),
        env.get_objects_by_type(), env.legal_actions(),
    )
    assert out is None


# ---------------------------------------------------------------------------
# the 3-rule hand policy solves the shipped layouts
# ---------------------------------------------------------------------------

def test_doors_hand_policy_solves_d2():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    actions, _ = _rollout(env)
    assert env.is_solved(), [a.pretty() for a in actions]
    assert [a.schema for a in actions] == ["move_to", "pick", "move_to"]
    assert len(actions) == 3
    assert [a.args[0] for a in actions if a.schema == "pick"] == ["key_0"]


def test_doors_hand_policy_solves_d3():
    env = DoorsPDDLLiteRelationalEnv.make_d3()
    actions, _ = _rollout(env)
    assert env.is_solved(), [a.pretty() for a in actions]
    assert [a.schema for a in actions] == ["move_to", "pick", "move_to", "pick", "move_to"]
    assert len(actions) == 5
    assert [a.args[0] for a in actions if a.schema == "pick"] == ["key_0", "key_1"]


# ---------------------------------------------------------------------------
# object renaming preserves the lifted structure (for an order-preserving sigma)
# ---------------------------------------------------------------------------

def test_doors_object_renaming_preserves_structure():
    canon = DoorsPDDLLiteRelationalEnv.make_d2()
    canon_actions, canon_traces = _rollout(canon)

    renamed = DoorsPDDLLiteRelationalEnv.make_d2(
        name_fmt={"location": "site_{}", "room": "area_{}", "key": "tool_{}"}
    )
    renamed_actions, renamed_traces = _rollout(renamed)

    # sigma: same integer ids, different prefixes (order-preserving)
    sigma = {}
    for i in range(canon.base.M):
        sigma[f"loc_{i}"] = f"site_{i}"
    for i in range(canon.base.D):
        sigma[f"room_{i}"] = f"area_{i}"
    for i in range(canon.base.K):
        sigma[f"key_{i}"] = f"tool_{i}"

    assert len(canon_actions) == len(renamed_actions)
    for a_c, a_r in zip(canon_actions, renamed_actions):
        assert a_c.schema == a_r.schema
        assert tuple(sigma[x] for x in a_c.args) == a_r.args
    assert [t[0] for t in canon_traces] == [t[0] for t in renamed_traces]   # same rule fired each step


def test_degenerate_noop_policy_stalls():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    s0 = env.get_state_atoms()
    a = interpret(
        degenerate_noop_policy(),
        env.get_state_atoms(), env.get_goal_atoms(),
        env.get_objects_by_type(), env.legal_actions(),
    )
    assert a == GroundAction("noop", ())
    for _ in range(5):
        env.step(GroundAction("noop", ()))
    assert not env.is_solved()
    assert env.get_state_atoms() == s0
