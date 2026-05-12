"""Tests for UnmaskedSurfaceDerivationGame (Stage 2).

Verifies grammar counts, language inclusion, compilation totality,
and game playability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.unmasked_surface_cfg import (
    UnmaskedSurfaceCFG,
)
from alphazeropp.instances.doors.dsl.unmasked_surface_derivation_game import (
    UnmaskedSurfaceDerivationGame,
)
from alphazeropp.instances.doors.dsl.explicit_surface_cfg import SurfaceCFG
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickRule, MoveRule, GoalRule, SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy, enumerate_relaxed_policies,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(num_rooms: int, exact_length: bool = False):
    cfg = DoorsGameConfig(num_rooms=num_rooms, locs_per_room=2)
    n_sites = cfg.obs_size()
    x0 = doors_initial_state(cfg)
    le = LeafEvaluator(
        n_sites, [x0], cfg,
        is_solved=cfg.is_solved, metric="solve_rate",
    )
    game = UnmaskedSurfaceDerivationGame(
        num_rooms, le, cfg, exact_length=exact_length,
    )
    return game, cfg, le


def _play_policy(game, policy):
    """Play a policy through the game, return reward."""
    obs, _ = game.reset()
    for rule in policy.rules:
        action = game._rule_to_action(rule)
        obs, reward, terminated, _, info = game.step(action)
    return reward, info


# ---------------------------------------------------------------------------
# Test: Exact language counts
# ---------------------------------------------------------------------------

class TestExactCounts:

    @pytest.mark.parametrize("K, expected", [(1, 7), (2, 341)])
    def test_max_length_counts(self, K, expected):
        cfg = UnmaskedSurfaceCFG(K, exact_length=False)
        assert cfg.count_words() == expected
        words = cfg.enumerate_words()
        assert len(words) == expected

    @pytest.mark.parametrize("K, expected", [(1, 4), (2, 256)])
    def test_exact_length_counts(self, K, expected):
        cfg = UnmaskedSurfaceCFG(K, exact_length=True)
        assert cfg.count_words() == expected
        words = cfg.enumerate_words()
        assert len(words) == expected

    def test_max_length_formula(self):
        """Verify count_words matches sum formula for K=1..4."""
        for K in range(1, 5):
            cfg = UnmaskedSurfaceCFG(K, exact_length=False)
            n = 2 * K
            L = 2 * K
            expected = sum(n**l for l in range(L + 1))
            assert cfg.count_words() == expected

    def test_exact_length_formula(self):
        """Verify count_words matches (2K)^{2K} for K=1..4."""
        for K in range(1, 5):
            cfg = UnmaskedSurfaceCFG(K, exact_length=True)
            expected = (2 * K) ** (2 * K)
            assert cfg.count_words() == expected


# ---------------------------------------------------------------------------
# Test: Stage 1 inclusion
# ---------------------------------------------------------------------------

class TestStage1Inclusion:

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_stage1_subset_of_stage2_exact(self, D):
        """Every Stage 1 word is a valid Stage 2 exact-length word."""
        K = D - 1
        stage1 = SurfaceCFG(K)
        stage2 = UnmaskedSurfaceCFG(K, exact_length=True)

        stage1_words = stage1.enumerate_words()
        stage2_words = stage2.enumerate_words()
        stage2_set = {tuple(w.rules) for w in stage2_words}

        for word in stage1_words:
            assert tuple(word.rules) in stage2_set, (
                f"Stage 1 word {word.pretty()} not in Stage 2"
            )

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_stage1_strict_subset(self, D):
        """Stage 2 exact-length has words not in Stage 1."""
        K = D - 1
        stage1 = SurfaceCFG(K)
        stage2 = UnmaskedSurfaceCFG(K, exact_length=True)

        stage1_set = {tuple(w.rules) for w in stage1.enumerate_words()}
        stage2_words = stage2.enumerate_words()

        extra = [w for w in stage2_words if tuple(w.rules) not in stage1_set]
        assert len(extra) > 0, "Stage 2 should be strictly larger than Stage 1"

    @pytest.mark.parametrize("D", [2, 3])
    def test_exact_stage1_count(self, D):
        """Stage 1 count matches (2K)!/2^K."""
        K = D - 1
        stage1 = SurfaceCFG(K)
        expected = math.factorial(2 * K) // (2 ** K)
        assert stage1.count_words() == expected


# ---------------------------------------------------------------------------
# Test: Every derivation compiles and evaluates
# ---------------------------------------------------------------------------

class TestCompilationTotality:

    @pytest.mark.parametrize("D", [2, 3])
    def test_all_max_length_compile(self, D):
        K = D - 1
        cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
        grammar = UnmaskedSurfaceCFG(K, exact_length=False)
        words = grammar.enumerate_words()
        for word in words:
            prog = compile_policy(word, cfg)
            assert prog is not None

    @pytest.mark.parametrize("D", [2, 3])
    def test_all_exact_length_compile(self, D):
        K = D - 1
        cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
        grammar = UnmaskedSurfaceCFG(K, exact_length=True)
        words = grammar.enumerate_words()
        for word in words:
            prog = compile_policy(word, cfg)
            assert prog is not None

    @pytest.mark.parametrize("D", [2, 3])
    def test_all_evaluate(self, D):
        """Every compiled program returns a finite reward."""
        K = D - 1
        cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
        n_sites = cfg.obs_size()
        x0 = doors_initial_state(cfg)
        le = LeafEvaluator(
            n_sites, [x0], cfg,
            is_solved=cfg.is_solved, metric="solve_rate",
        )
        grammar = UnmaskedSurfaceCFG(K, exact_length=True)
        words = grammar.enumerate_words()
        for word in words:
            prog = compile_policy(word, cfg)
            reward = le(prog)
            assert np.isfinite(reward), (
                f"Non-finite reward for {word.pretty()}: {reward}"
            )


# ---------------------------------------------------------------------------
# Test: Game terminates correctly
# ---------------------------------------------------------------------------

class TestGamePlayability:

    @pytest.mark.parametrize("D", [2, 3, 4])
    @pytest.mark.parametrize("exact", [True, False])
    def test_random_play_terminates(self, D, exact):
        game, _, _ = _make_game(D, exact_length=exact)
        K = D - 1
        rng = np.random.default_rng(42)

        for _ in range(20):
            game.reset()
            terminated = False
            for step in range(2 * K + 2):  # generous bound
                mask = game.get_action_mask()
                legal = np.where(mask)[0]
                assert len(legal) > 0, f"No legal actions at step {step}"
                action = rng.choice(legal)
                _, _, terminated, _, _ = game.step(action)
                if terminated:
                    break
            assert terminated, "Episode did not terminate"

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_canonical_policy_works(self, D):
        """The canonical Stage 1 policy plays correctly in Stage 2."""
        game, _, _ = _make_game(D, exact_length=True)
        policy = canonical_policy(D)
        reward, info = _play_policy(game, policy)
        assert reward > 0, f"Canonical policy should solve, got reward={reward}"

    def test_action_mask_no_domain_logic(self):
        """Mask depends only on level, not on placed rules."""
        game, _, _ = _make_game(3, exact_length=True)
        K = 2

        # Play P_0 first
        game.reset()
        game.step(0)  # P_0
        mask_after_p0 = game.get_action_mask().copy()

        # Play M_0 first (not legal in Stage 1!)
        game.reset()
        game.step(K)  # M_0
        mask_after_m0 = game.get_action_mask().copy()

        # Both should have identical masks (all non-goal tokens legal)
        np.testing.assert_array_equal(mask_after_p0, mask_after_m0)


# ---------------------------------------------------------------------------
# Test: State management
# ---------------------------------------------------------------------------

class TestStateManagement:

    def test_stash_unstash(self):
        game, _, _ = _make_game(3, exact_length=True)
        game.reset()

        mask = game.get_action_mask()
        action = np.where(mask)[0][0]
        game.step(action)

        state = game.stash_state()
        saved_mask = game.get_action_mask().copy()

        mask2 = game.get_action_mask()
        action2 = np.where(mask2)[0][0]
        game.step(action2)

        game.unstash_state(state)
        restored_mask = game.get_action_mask()
        np.testing.assert_array_equal(saved_mask, restored_mask)

    def test_clone_independent(self):
        game, _, _ = _make_game(3, exact_length=True)
        game.reset()

        mask = game.get_action_mask()
        action = np.where(mask)[0][0]
        game.step(action)

        clone = game.clone()
        clone_mask = clone.get_action_mask().copy()

        mask2 = game.get_action_mask()
        action2 = np.where(mask2)[0][0]
        game.step(action2)

        np.testing.assert_array_equal(clone_mask, clone.get_action_mask())


# ---------------------------------------------------------------------------
# Test: Random play coverage
# ---------------------------------------------------------------------------

class TestCoverage:

    def test_random_covers_d2_exact(self):
        """Random play eventually hits all 4 exact-length D=2 derivations."""
        game, _, _ = _make_game(2, exact_length=True)
        rng = np.random.default_rng(42)
        seen: set[tuple] = set()

        for _ in range(200):
            game.reset()
            terminated = False
            while not terminated:
                mask = game.get_action_mask()
                legal = np.where(mask)[0]
                action = rng.choice(legal)
                _, _, terminated, _, info = game.step(action)

            if "policy" in info:
                seen.add(tuple(info["policy"].rules))

        assert len(seen) == 4, f"Expected 4 derivations, saw {len(seen)}"
