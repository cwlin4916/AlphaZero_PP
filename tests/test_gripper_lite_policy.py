"""Stage 1 end-to-end: hand-written lifted policy must solve Gripper-lite
for B = 1, 2, 3, and the abstract action sequence must be invariant under
object renaming.
"""

from __future__ import annotations

import pytest

from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.lifted_dsl import GroundAction
from alphazeropp.synthesis.lifted_interpreter import interpret


def _rollout(env: GripperLiteEnv) -> tuple[list[GroundAction], list[tuple[int, dict[str, str]]]]:
    """Run the hand policy through ``env`` to completion or no-firing/horizon.

    Returns ``(actions, traces)`` where each trace is ``(rule_index, theta)``.
    """
    policy = hand_policy()
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


@pytest.mark.parametrize("n_balls,expected_steps", [(1, 3), (2, 7), (3, 11)])
def test_hand_policy_solves(n_balls, expected_steps):
    env = GripperLiteEnv(n_balls=n_balls)
    actions, _traces = _rollout(env)
    assert env.is_solved(), f"unsolved at n_balls={n_balls}; trace: {[a.pretty() for a in actions]}"
    assert len(actions) == expected_steps, (
        f"expected {expected_steps} steps for n_balls={n_balls}, got {len(actions)}"
    )


def test_hand_policy_solves_1_ball():
    """B=1 separate test (named per spec) — uses the parametrized worker."""
    env = GripperLiteEnv(n_balls=1)
    actions, _ = _rollout(env)
    assert env.is_solved()
    assert [a.schema for a in actions] == ["pick", "move", "drop"]


def test_hand_policy_solves_2_balls():
    env = GripperLiteEnv(n_balls=2)
    actions, _ = _rollout(env)
    assert env.is_solved()
    schemas = [a.schema for a in actions]
    assert schemas == ["pick", "move", "drop", "move", "pick", "move", "drop"]
    # lex-min: ball_0 must be transported before ball_1
    pick_balls = [a.args[0] for a in actions if a.schema == "pick"]
    assert pick_balls == ["ball_0", "ball_1"]


def test_hand_policy_solves_3_balls():
    env = GripperLiteEnv(n_balls=3)
    actions, _ = _rollout(env)
    assert env.is_solved()
    pick_balls = [a.args[0] for a in actions if a.schema == "pick"]
    assert pick_balls == ["ball_0", "ball_1", "ball_2"]


def test_object_renaming_preserves_behavior():
    """Renaming ball_0→foo and room_a→left_room must not change the lifted
    structure of the rollout (same schemas in the same order; same rule indices;
    bindings differ only by the renaming map)."""
    rename_ball = {"ball_0": "foo"}
    rename_room = {"room_a": "left_room", "room_b": "room_b"}

    canon_env = GripperLiteEnv(n_balls=1)
    canon_actions, canon_traces = _rollout(canon_env)

    renamed_env = GripperLiteEnv(
        n_balls=1,
        rooms=(rename_room["room_a"], rename_room["room_b"]),
        ball_names=(rename_ball["ball_0"],),
    )
    renamed_actions, renamed_traces = _rollout(renamed_env)

    assert len(canon_actions) == len(renamed_actions)
    for a_c, a_r in zip(canon_actions, renamed_actions):
        assert a_c.schema == a_r.schema
        # Map each canonical arg through the renaming, compare to renamed arg.
        mapped_args = tuple(
            rename_ball.get(x, rename_room.get(x, x)) for x in a_c.args
        )
        assert mapped_args == a_r.args

    # Rule indices must match step-by-step (proves the SAME rule fired at each step).
    canon_rule_ids = [t[0] for t in canon_traces]
    renamed_rule_ids = [t[0] for t in renamed_traces]
    assert canon_rule_ids == renamed_rule_ids
