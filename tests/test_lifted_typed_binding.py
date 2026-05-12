"""Stage 3.1 tests: typed binding engine for Doors lifted policies."""

from __future__ import annotations

import pytest

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state, compute_doors_derived_params,
)
from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.lifted_typed_dsl import (
    Atom, Rule, Var, LiftedDecisionList, manual_three_rule_policy,
)
from alphazeropp.instances.doors.dsl.binding_engine import (
    BindingEngine, run_typed_lifted_episode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(D: int) -> DoorsGameConfig:
    p = compute_doors_derived_params(D, 2)
    return DoorsGameConfig(num_rooms=D, locs_per_room=2, horizon=p["horizon"])


@pytest.fixture
def setup_d3():
    cfg = _make_cfg(3)
    rt = DoorsRelationalRuntime(cfg)
    engine = BindingEngine(rt)
    return cfg, rt, engine


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

class TestThreeRulePolicySolves:
    def test_solves_d3(self, setup_d3):
        cfg, _, engine = setup_d3
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        res = run_typed_lifted_episode(
            env, engine, manual_three_rule_policy(),
            x0=x0, is_solved=cfg.is_solved,
        )
        assert res.solved
        assert res.total_env_steps == 2 * (cfg.D - 1) + 1

    @pytest.mark.parametrize("D", [4, 5, 10], ids=["D4", "D5", "D10"])
    def test_solves_larger_D_same_policy(self, D):
        cfg = _make_cfg(D)
        rt = DoorsRelationalRuntime(cfg)
        engine = BindingEngine(rt)
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        res = run_typed_lifted_episode(
            env, engine, manual_three_rule_policy(),
            x0=x0, is_solved=cfg.is_solved,
        )
        assert res.solved, f"D={D} did not solve"
        assert res.total_env_steps == 2 * (D - 1) + 1


# ---------------------------------------------------------------------------
# Binding-engine semantics
# ---------------------------------------------------------------------------

class TestBindingEngineSemantics:
    def test_lex_tiebreak_picks_smallest_object(self, setup_d3):
        # rule with no constraints → returns θ where all vars bind to obj 0
        cfg, _, engine = setup_d3
        x0 = doors_initial_state(cfg)
        rule = Rule(
            pre=(Atom("goal_loc", (Var("g", "loc"),)),),
            action="move_to",
            action_args=(Var("g", "loc"),),
        )
        # Construct a rule whose state-condition is always true for some loc:
        # at_loc(u) holds for u = current agent location only — so we test
        # tie-break with two `loc_of_key` candidates.  At step 0 (D=3), the
        # smallest k with key_available(k) is k=0.
        u = Var("u", "loc")
        l = Var("l", "loc")
        k = Var("k", "key")
        r = Var("r", "room")
        move_rule = Rule(
            pre=(
                Atom("loc_of_key", (k, l)),
                Atom("key_for", (k, r)),
                Atom("locked", (r,)),
                Atom("key_available", (k,)),
            ),
            action="move_to",
            action_args=(l,),
        )
        firing = engine.bind_rule(move_rule, x0)
        assert firing is not None
        binding = dict(firing.binding)
        assert binding["k"] == 0  # lex-smallest key satisfies guard

    def test_type_respecting_no_cross_type_binding(self, setup_d3):
        # A rule that asks `loc_of_key(k:key, l:loc)` must never bind k to a
        # room ID; the static signature in PREDICATES enforces this at AST
        # construction time.
        with pytest.raises(TypeError, match="loc_of_key"):
            Atom("loc_of_key", (Var("r", "room"), Var("l", "loc")))

    def test_unsatisfiable_guard_returns_none(self, setup_d3):
        cfg, _, engine = setup_d3
        x0 = doors_initial_state(cfg)
        # rule: locked(r) ∧ goal_loc(g) — but agent variable is unrelated.
        # Make it unsatisfiable by requiring a non-existent predicate combo.
        # We use locked(r=0) which is always false (room 0 is always open).
        # Encode as: at_loc(u) ∧ goal_loc(u) ∧ locked(r) — agent is at loc 0
        # initially, goal is loc 5, so at_loc(u) ∧ goal_loc(u) cannot hold.
        u = Var("u", "loc")
        r = Var("r", "room")
        rule = Rule(
            pre=(
                Atom("at_loc", (u,)),
                Atom("goal_loc", (u,)),
                Atom("locked", (r,)),
            ),
            action="move_to",
            action_args=(u,),
        )
        assert engine.bind_rule(rule, x0) is None


# ---------------------------------------------------------------------------
# Object-renaming invariance — the mandatory mask-leakage diagnostic
# ---------------------------------------------------------------------------

class TestObjectRenamingInvariant:
    def test_d3_renamed_still_solves(self):
        # Standard D=3 layout: rooms {0,1,2}, keys {0,1}, locs {0..5}.
        # Permute keys (0↔1) and the locations not at room 0 / not goal.
        cfg = _make_cfg(3)
        rt = DoorsRelationalRuntime(cfg)
        # Permute room labels: 1↔2 (but leave 0 and goal-room invariant?).
        # Doors needs room 0 (start) and room D-1 (goal) preserved or the
        # whole instance changes meaning.  The renaming test should preserve
        # *static-fact structure* — `next` and `key_for` — under the
        # permutation, which is exactly what permute_objects() does.
        perm_keys = {0: 1, 1: 0}
        rt2 = rt.permute_objects(perm_keys=perm_keys)
        engine2 = BindingEngine(rt2)
        x0_new = doors_initial_state(rt2.cfg)
        env_new = rt2.cfg.make_env(rt2.cfg.obs_size(), frozen_states=[x0_new])
        res = run_typed_lifted_episode(
            env_new, engine2, manual_three_rule_policy(),
            x0=x0_new, is_solved=rt2.cfg.is_solved,
        )
        assert res.solved
        assert res.total_env_steps == 2 * (rt2.cfg.D - 1) + 1

    def test_episode_lengths_equal_under_renaming(self):
        cfg = _make_cfg(4)
        rt = DoorsRelationalRuntime(cfg)
        engine = BindingEngine(rt)
        x0 = doors_initial_state(cfg)
        env = cfg.make_env(cfg.obs_size(), frozen_states=[x0])
        baseline = run_typed_lifted_episode(
            env, engine, manual_three_rule_policy(),
            x0=x0, is_solved=cfg.is_solved,
        )
        # Permute keys (reverse order)
        perm_keys = {i: cfg.K - 1 - i for i in range(cfg.K)}
        rt2 = rt.permute_objects(perm_keys=perm_keys)
        engine2 = BindingEngine(rt2)
        x0_new = doors_initial_state(rt2.cfg)
        env_new = rt2.cfg.make_env(rt2.cfg.obs_size(), frozen_states=[x0_new])
        renamed = run_typed_lifted_episode(
            env_new, engine2, manual_three_rule_policy(),
            x0=x0_new, is_solved=rt2.cfg.is_solved,
        )
        assert baseline.solved and renamed.solved
        assert baseline.total_env_steps == renamed.total_env_steps


# ---------------------------------------------------------------------------
# Pretty-print diagnostic
# ---------------------------------------------------------------------------

class TestPolicyPrettyPrintNoIndices:
    def test_pretty_contains_only_variable_names(self):
        text = manual_three_rule_policy().pretty()
        assert "u" in text and "l" in text and "k" in text and "r" in text
        for forbidden in ["r_0", "r_1", "k_0", "k_1", "loc_0", "loc_1"]:
            assert forbidden not in text
        for forbidden in ["IsZero", "Flip(", "obs[", "cfg."]:
            assert forbidden not in text
