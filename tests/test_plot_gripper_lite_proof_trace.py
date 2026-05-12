"""Smoke test for the §7 proof-trace figure.

``plot_proof_trace`` internally asserts that its per-state ``_diagnose_rule``
verdict agrees with ``interpret`` at every step, so simply running it for
B = 2, 3 exercises that the figure stays faithful to the interpreter. We also
check the recorded fired-rule sequence matches the §7 corollary
``(ρ3,ρ2,ρ1)(ρ4,ρ3,ρ2,ρ1)^(B-1)`` (0-indexed: 2,1,0 then 3,2,1,0 repeated).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.lifted_interpreter import interpret


def _load_plot_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "plotting" / "plot_gripper_lite_rollout.py"
    spec = importlib.util.spec_from_file_location("_plot_gripper_lite_rollout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot_mod = _load_plot_module()


def _expected_fired_indices(n_balls: int) -> list[int]:
    return [2, 1, 0] + [3, 2, 1, 0] * (n_balls - 1)


@pytest.mark.parametrize("n_balls", [2, 3])
def test_diagnose_rule_matches_interpret(n_balls):
    env = GripperLiteEnv(n_balls=n_balls)
    policy = hand_policy()
    fired: list[int] = []
    for _ in range(env.horizon):
        if env.is_solved():
            break
        state = env.get_state_atoms()
        goal = env.get_goal_atoms()
        objs = env.get_objects_by_type()
        legal = env.legal_actions()
        rows = [plot_mod._diagnose_rule(r, state, goal, objs, legal) for r in policy.rules]
        diag_idx = next(i for i, (fires, _, _) in enumerate(rows) if fires)
        out = interpret(policy, state, goal, objs, legal, trace=True)
        assert out is not None and out[1] == diag_idx
        assert sum(1 for fires, _, _ in rows if fires) == 1
        assert rows[diag_idx][2] == out[0]
        fired.append(diag_idx)
        env.step(out[0])
    assert env.is_solved()
    assert len(fired) == 4 * n_balls - 1
    assert fired == _expected_fired_indices(n_balls)


@pytest.mark.parametrize("n_balls", [2, 3])
def test_plot_proof_trace_writes_png(tmp_path, monkeypatch, n_balls):
    monkeypatch.setattr(plot_mod, "OUT", tmp_path)
    out = plot_mod.plot_proof_trace(n_balls=n_balls)
    assert out == tmp_path / f"gripper_lite_proof_trace_B{n_balls}.png"
    assert out.exists() and out.stat().st_size > 0
