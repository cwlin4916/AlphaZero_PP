"""Stage 2 tests for LiftedDerivationGame + LiftedLeafEvaluator + MCTS wiring.

See ``docs/notes/stage4/02_plan.md`` §3.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.core.mcts import MCTS
from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.derivation_game import UniformPolicyValueNet
from alphazeropp.synthesis.lifted_derivation import (
    LiftedDerivationGame,
    LiftedDerivationState,
)
from alphazeropp.synthesis.lifted_dsl import Policy
from alphazeropp.synthesis.lifted_grammar import (
    LiftedGrammarConfig,
    enumerate_productions,
    gripper_lite_signature,
)
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator


def _evaluator(n_train=1, n_eval_out=2):
    train = [GripperLiteEnv(n_balls=n_train)]
    return LiftedLeafEvaluator(train, train, [GripperLiteEnv(n_balls=n_eval_out)])


def _first_action(game) -> int:
    return int(np.flatnonzero(game.get_action_mask())[0])


# ---------------------------------------------------------------------------

def test_lifted_derivation_reaches_terminal_policy():
    cfg = LiftedGrammarConfig(max_rules=1)
    sig = gripper_lite_signature()
    s = LiftedDerivationState.initial()
    steps = 0
    while not s.is_terminal() and steps < 50:
        prods = enumerate_productions(s, cfg, sig)
        assert prods
        s = s.apply(prods[0])
        steps += 1
    assert s.is_terminal()
    prog = s.to_program()
    assert isinstance(prog, Policy)
    assert len(prog.rules) == 1
    # the policy is also runnable through the evaluator without raising
    assert isinstance(_evaluator()(prog), float)


def test_terminal_policy_evaluator_runs_without_exception():
    ev = LiftedLeafEvaluator(
        [GripperLiteEnv(n_balls=2)],
        [GripperLiteEnv(n_balls=2)],
        [GripperLiteEnv(n_balls=3)],
    )
    score = ev(hand_policy())
    assert isinstance(score, float)
    m = ev.metrics_for(hand_policy())
    assert m["train_solve_rate"] == 1.0
    assert m["eval_out_solve_rate"] == 1.0          # hand policy generalises to B=3
    assert score > 0.0


def test_action_mask_matches_legal_productions():
    cfg = LiftedGrammarConfig(max_rules=2)
    sig = gripper_lite_signature()
    game = LiftedDerivationGame(cfg, sig, _evaluator())
    game.reset_wrapper()
    steps = 0
    while not game.terminated and not game.truncated and steps < 60:
        n = len(game._current_productions)
        mask = game.get_action_mask()
        assert mask.shape == (game._max_productions,)
        assert mask[:n].all()
        assert not mask[n:].any()
        game.step_wrapper(_first_action(game))
        steps += 1
    assert game.terminated


def test_clone_and_stash_state_round_trip():
    cfg = LiftedGrammarConfig(max_rules=2)
    sig = gripper_lite_signature()
    game = LiftedDerivationGame(cfg, sig, _evaluator())
    game.reset_wrapper()
    for _ in range(3):
        game.step_wrapper(_first_action(game))

    snapshot = game.stash_state()
    pre_obs = game.obs.copy()
    pre_pretty = game.hashable_obs

    game.step_wrapper(_first_action(game))
    assert game.hashable_obs != pre_pretty

    game.unstash_state(snapshot)
    assert game.hashable_obs == pre_pretty
    assert np.array_equal(game.obs, pre_obs)

    twin = game.clone()
    game.step_wrapper(_first_action(game))
    assert twin.hashable_obs == pre_pretty          # clone unaffected by the original advancing
    assert twin.leaf_evaluator is game.leaf_evaluator  # evaluator stays shared, not deep-copied


@pytest.mark.parametrize(
    "n_balls,max_rules,n_sims",
    [(1, 3, 16), (1, 3, 32), (2, 4, 16)],   # a B=1 and a B=2 uniform-MCTS smoke
)
def test_derivation_game_smoke_with_uniform_mcts(n_balls, max_rules, n_sims):
    cfg = LiftedGrammarConfig(max_rules=max_rules)
    sig = gripper_lite_signature()
    # Stage-2 minimal: same-B train and eval (no generalization claim).
    ev = _evaluator(n_train=n_balls, n_eval_out=n_balls)
    game = LiftedDerivationGame(cfg, sig, ev)
    game.reset_wrapper()
    net = UniformPolicyValueNet(game._max_productions)
    mcts = MCTS(game, net, n_simulations=n_sims, temperature=0.5, c_exploration=1.5)

    steps = 0
    while not game.terminated and not game.truncated and steps < 80:
        probs = np.asarray(mcts.perform_simulations(None), dtype=np.float64)
        assert probs.shape == (game._max_productions,)
        a = int(np.argmax(probs))
        game.step_wrapper(a)
        steps += 1

    assert game.terminated
    program = game.get_program()
    assert isinstance(program, Policy)
    # the terminal program was scored and cached
    assert program.pretty() in ev._cache
    assert isinstance(ev._cache[program.pretty()]["score"], float)
