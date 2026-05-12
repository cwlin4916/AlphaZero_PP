"""Tests for ExplicitSurfaceDerivationGame.

Verifies that the grammar-based game produces identical behavior
to the bitmask-based SurfaceDerivationGame.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.explicit_surface_cfg import SurfaceCFG
from alphazeropp.instances.doors.dsl.explicit_surface_derivation_game import (
    ExplicitSurfaceDerivationGame,
)
from alphazeropp.instances.doors.dsl.surface_derivation_game import (
    SurfaceDerivationGame,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy, enumerate_relaxed_policies,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_games(num_rooms: int):
    """Create matched pair of old and new games."""
    cfg = DoorsGameConfig(num_rooms=num_rooms, locs_per_room=2)
    n_sites = cfg.obs_size()
    x0 = doors_initial_state(cfg)
    le = LeafEvaluator(
        n_sites, [x0], cfg,
        is_solved=cfg.is_solved, metric="solve_rate",
    )

    old_game = SurfaceDerivationGame(num_rooms, le, cfg)
    new_game = ExplicitSurfaceDerivationGame(num_rooms, le, cfg)
    return old_game, new_game


def _play_policy(game, policy):
    """Play a policy through a game, return (obs_list, mask_list, reward)."""
    obs, _ = game.reset()
    obs_list = [obs.copy()]
    mask_list = [game.get_action_mask().copy()]

    for rule in policy.rules:
        action = game._rule_to_action(rule)
        obs, reward, terminated, _, info = game.step(action)
        obs_list.append(obs.copy())
        if not terminated:
            mask_list.append(game.get_action_mask().copy())

    return obs_list, mask_list, reward


# ---------------------------------------------------------------------------
# Test: Action mask equivalence
# ---------------------------------------------------------------------------

class TestActionMaskEquivalence:
    """At every reachable state, grammar mask == bitmask mask."""

    @pytest.mark.parametrize("D", [2, 3])
    def test_masks_match_on_all_policies(self, D):
        old_game, new_game = _make_games(D)
        policies = enumerate_relaxed_policies(D)

        for policy in policies:
            _, old_masks, _ = _play_policy(old_game, policy)
            _, new_masks, _ = _play_policy(new_game, policy)
            assert len(old_masks) == len(new_masks)
            for step, (om, nm) in enumerate(zip(old_masks, new_masks)):
                np.testing.assert_array_equal(
                    om, nm,
                    err_msg=f"Mask mismatch at step {step} for policy {policy.pretty()}"
                )


# ---------------------------------------------------------------------------
# Test: Observation equivalence
# ---------------------------------------------------------------------------

class TestObservationEquivalence:

    @pytest.mark.parametrize("D", [2, 3])
    def test_obs_match_on_all_policies(self, D):
        old_game, new_game = _make_games(D)
        policies = enumerate_relaxed_policies(D)

        for policy in policies:
            old_obs, _, _ = _play_policy(old_game, policy)
            new_obs, _, _ = _play_policy(new_game, policy)
            assert len(old_obs) == len(new_obs)
            for step, (oo, no) in enumerate(zip(old_obs, new_obs)):
                np.testing.assert_array_equal(
                    oo, no,
                    err_msg=f"Obs mismatch at step {step} for policy {policy.pretty()}"
                )


# ---------------------------------------------------------------------------
# Test: Reward equivalence
# ---------------------------------------------------------------------------

class TestRewardEquivalence:

    @pytest.mark.parametrize("D", [2, 3])
    def test_rewards_match_on_all_policies(self, D):
        old_game, new_game = _make_games(D)
        policies = enumerate_relaxed_policies(D)

        for policy in policies:
            _, _, old_reward = _play_policy(old_game, policy)
            _, _, new_reward = _play_policy(new_game, policy)
            assert old_reward == new_reward, (
                f"Reward mismatch for {policy.pretty()}: "
                f"old={old_reward}, new={new_reward}"
            )


# ---------------------------------------------------------------------------
# Test: Canonical policy episode
# ---------------------------------------------------------------------------

class TestCanonicalEpisode:

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_canonical_episode(self, D):
        _, new_game = _make_games(D)
        policy = canonical_policy(D)
        _, _, reward = _play_policy(new_game, policy)
        # Canonical policy always solves
        assert reward > 0


# ---------------------------------------------------------------------------
# Test: Random play always terminates
# ---------------------------------------------------------------------------

class TestRandomPlay:

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_random_play_terminates(self, D):
        _, game = _make_games(D)
        K = D - 1
        rng = np.random.default_rng(42)

        for _ in range(20):
            game.reset()
            for step in range(2 * K + 1):
                mask = game.get_action_mask()
                legal = np.where(mask)[0]
                assert len(legal) > 0, f"No legal actions at step {step}"
                action = rng.choice(legal)
                _, _, terminated, _, _ = game.step(action)
                if terminated:
                    assert step == 2 * K, f"Terminated at step {step}, expected {2*K}"
                    break
            assert terminated, "Episode did not terminate"


# ---------------------------------------------------------------------------
# Test: Stash/unstash
# ---------------------------------------------------------------------------

class TestStateManagement:

    def test_stash_unstash(self):
        _, game = _make_games(3)
        game.reset()

        # Play one step
        mask = game.get_action_mask()
        action = np.where(mask)[0][0]
        game.step(action)

        # Stash
        state = game.stash_state()
        saved_mask = game.get_action_mask().copy()

        # Play another step
        mask2 = game.get_action_mask()
        action2 = np.where(mask2)[0][0]
        game.step(action2)

        # Unstash — should restore
        game.unstash_state(state)
        restored_mask = game.get_action_mask()
        np.testing.assert_array_equal(saved_mask, restored_mask)

    def test_clone_independent(self):
        _, game = _make_games(3)
        game.reset()

        mask = game.get_action_mask()
        action = np.where(mask)[0][0]
        game.step(action)

        clone = game.clone()
        clone_mask = clone.get_action_mask().copy()

        # Advance original
        mask2 = game.get_action_mask()
        action2 = np.where(mask2)[0][0]
        game.step(action2)

        # Clone should be unchanged
        np.testing.assert_array_equal(clone_mask, clone.get_action_mask())


# ---------------------------------------------------------------------------
# Test: Exact counts
# ---------------------------------------------------------------------------

class TestExactCounts:

    def test_d2_exactly_1_policy(self):
        _, game = _make_games(2)
        game.reset()
        mask = game.get_action_mask()
        # Only one legal action at each step for K=1
        assert mask.sum() == 1

    def test_d3_branching(self):
        _, game = _make_games(3)
        game.reset()
        mask = game.get_action_mask()
        # At start with K=2: PickRule(0) and PickRule(1) are legal
        assert mask.sum() == 2
