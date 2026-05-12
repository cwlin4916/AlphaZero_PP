"""Tests for the explicit right-linear CFG (explicit_surface_cfg.py).

Verifies structural properties, language equality against the existing
masked generator oracles, AST equality, and reward equality.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.explicit_surface_cfg import (
    UNTOUCHED, PICKED, MOVED,
    StatusVector, Nonterminal, Production, SurfaceCFG,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.instances.doors.dsl.surface_dsl import (
    GoalRule, MoveRule, PickRule, SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy,
    count_relaxed_policies,
    enumerate_relaxed_policies,
    enumerate_surface_prefixes,
)
from alphazeropp.synthesis.interpreter import run_policy_episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_d2():
    return DoorsGameConfig(num_rooms=2, locs_per_room=2)


@pytest.fixture
def cfg_d3():
    params = compute_doors_derived_params(3, 2)
    return DoorsGameConfig(num_rooms=3, locs_per_room=2, horizon=params["horizon"])


# ---------------------------------------------------------------------------
# Test: Structural properties
# ---------------------------------------------------------------------------

class TestStructuralProperties:
    """Verify nonterminal/production counts match formulas."""

    @pytest.mark.parametrize("D", [2, 3, 4, 5])
    def test_nonterminal_count(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        expected = 3 ** K if K > 0 else 1
        assert len(cfg.nonterminals) == expected
        assert cfg.count_nonterminals() == expected

    @pytest.mark.parametrize("D", [2, 3, 4, 5])
    def test_production_count(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        expected = 2 * K * (3 ** (K - 1)) + 1 if K > 0 else 1
        assert len(cfg.productions) == expected
        assert cfg.count_productions() == expected

    @pytest.mark.parametrize("D", [2, 3, 4, 5])
    def test_word_count(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        assert cfg.count_words() == count_relaxed_policies(D)

    @pytest.mark.parametrize("D", [2, 3, 4, 5])
    def test_start_symbol(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        assert cfg.start.is_start()
        assert cfg.start.status == StatusVector.initial(K)

    @pytest.mark.parametrize("D", [2, 3, 4, 5])
    def test_exactly_one_terminal_production(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        terminal_prods = [p for p in cfg.productions if p.is_terminal()]
        assert len(terminal_prods) == 1
        assert isinstance(terminal_prods[0].terminal, GoalRule)
        assert terminal_prods[0].lhs.is_pre_goal()


# ---------------------------------------------------------------------------
# Test: Language equality
# ---------------------------------------------------------------------------

class TestLanguageEquality:
    """Verify CFG generates exactly the same language as the masked generator."""

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_words_match_relaxed_policies(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        cfg_words = cfg.enumerate_words()
        oracle_words = enumerate_relaxed_policies(D)
        assert set(cfg_words) == set(oracle_words)
        assert len(cfg_words) == len(oracle_words)

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_prefixes_match_surface_prefixes(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        cfg_prefixes = cfg.enumerate_prefixes()
        oracle_prefixes = enumerate_surface_prefixes(D)
        assert set(cfg_prefixes) == set(oracle_prefixes)
        assert len(cfg_prefixes) == len(oracle_prefixes)


# ---------------------------------------------------------------------------
# Test: AST equality
# ---------------------------------------------------------------------------

class TestASTEquality:
    """Verify compiled ASTs are identical for both enumeration paths."""

    @pytest.mark.parametrize("D", [2, 3])
    def test_compiled_asts_match(self, D):
        K = D - 1
        doors_cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
        cfg = SurfaceCFG(K)
        cfg_words = cfg.enumerate_words()
        oracle_words = enumerate_relaxed_policies(D)

        # Build policy → AST pretty string mappings
        cfg_asts = {}
        for policy in cfg_words:
            prog = compile_policy(policy, doors_cfg)
            cfg_asts[policy] = prog.pretty()

        oracle_asts = {}
        for policy in oracle_words:
            prog = compile_policy(policy, doors_cfg)
            oracle_asts[policy] = prog.pretty()

        # Same set of policies, same ASTs
        assert set(cfg_asts.keys()) == set(oracle_asts.keys())
        for policy in cfg_asts:
            assert cfg_asts[policy] == oracle_asts[policy]


# ---------------------------------------------------------------------------
# Test: Reward equality
# ---------------------------------------------------------------------------

class TestRewardEquality:
    """Verify compiled policies produce identical rewards."""

    @pytest.mark.parametrize("D", [2, 3])
    def test_rewards_match(self, D):
        K = D - 1
        doors_cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
        params = compute_doors_derived_params(D, 2)
        n_sites = doors_cfg.obs_size()
        x0 = doors_initial_state(doors_cfg)

        cfg = SurfaceCFG(K)
        cfg_words = cfg.enumerate_words()
        oracle_words = enumerate_relaxed_policies(D)

        def run_reward(policy):
            prog = compile_policy(policy, doors_cfg)
            env = doors_cfg.make_env(n_sites, frozen_states=[x0])
            result = run_policy_episode(
                env, prog, x0=x0, is_solved=doors_cfg.is_solved
            )
            return result.cumulative_reward, result.solved

        cfg_rewards = {p: run_reward(p) for p in cfg_words}
        oracle_rewards = {p: run_reward(p) for p in oracle_words}

        assert set(cfg_rewards.keys()) == set(oracle_rewards.keys())
        for policy in cfg_rewards:
            assert cfg_rewards[policy] == oracle_rewards[policy]


# ---------------------------------------------------------------------------
# Test: Bitmask bijection
# ---------------------------------------------------------------------------

class TestBitmaskBijection:
    """Verify StatusVector ↔ bitmask conversion is a bijection."""

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_roundtrip(self, D):
        K = D - 1
        # Enumerate all valid (picked, moved) pairs where moved ⊆ picked
        for picked in range(1 << K):
            for moved in range(1 << K):
                if moved & ~picked:
                    continue  # skip invalid: moved not subset of picked
                sv = StatusVector.from_bitmasks(picked, moved, K)
                p2, m2 = sv.to_bitmasks()
                assert (p2, m2) == (picked, moved), (
                    f"Roundtrip failed: ({picked}, {moved}) → {sv} → ({p2}, {m2})"
                )

    @pytest.mark.parametrize("D", [2, 3, 4])
    def test_bijection_covers_all_status_vectors(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        seen = set()
        for picked in range(1 << K):
            for moved in range(1 << K):
                if moved & ~picked:
                    continue
                sv = StatusVector.from_bitmasks(picked, moved, K)
                seen.add(sv)
        # Every nonterminal's status vector should be reachable
        for nt in cfg.nonterminals:
            assert nt.status in seen


# ---------------------------------------------------------------------------
# Test: Pretty-printing and derivation traces
# ---------------------------------------------------------------------------

class TestPrettyPrinting:

    @pytest.mark.parametrize("D", [2, 3])
    def test_pretty_print_nonempty(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        text = cfg.pretty()
        assert len(text) > 0
        assert "N_" in text
        assert "→" in text

    @pytest.mark.parametrize("D", [2, 3])
    def test_derivation_trace(self, D):
        K = D - 1
        cfg = SurfaceCFG(K)
        policy = canonical_policy(D)
        trace = cfg.derivation_trace(policy)
        assert len(trace) > 0
        assert "⇒" in trace
        # Should mention GoalRule
        assert "GoalRule" in trace


# ---------------------------------------------------------------------------
# Test: Edge case K=0
# ---------------------------------------------------------------------------

class TestEdgeCaseK0:
    """D=1 means K=0: one room, no keys, just GoalRule."""

    def test_k0_grammar(self):
        cfg = SurfaceCFG(0)
        assert len(cfg.nonterminals) == 1
        assert len(cfg.productions) == 1
        assert cfg.productions[0].is_terminal()
        words = cfg.enumerate_words()
        assert len(words) == 1
        assert words[0] == SurfacePolicy((GoalRule(),))
