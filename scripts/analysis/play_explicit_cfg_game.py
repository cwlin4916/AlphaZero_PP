#!/usr/bin/env python3
"""Play the ExplicitSurfaceDerivationGame and verify against the old game.

Reports:
  1. Exhaustive play for D=2 and D=3 — all derivations with rewards
  2. Random play for D=3 and D=4 — solve rate and reward distribution
  3. Side-by-side comparison for canonical policy
  4. Exact count verification for D=2..4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def make_game(D: int, game_cls):
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
    n_sites = cfg.obs_size()
    x0 = doors_initial_state(cfg)
    le = LeafEvaluator(
        n_sites, [x0], cfg,
        is_solved=cfg.is_solved, metric="solve_rate",
    )
    return game_cls(D, le, cfg), cfg


def play_policy(game, policy):
    """Play a policy through a game, return (reward, solved)."""
    game.reset()
    for rule in policy.rules:
        action = game._rule_to_action(rule)
        _, reward, terminated, _, info = game.step(action)
    return reward, info.get("leaf_value", reward)


def main():
    # ------------------------------------------------------------------
    # 1. Exhaustive Play D=2, D=3
    # ------------------------------------------------------------------
    for D in [2, 3]:
        section(f"Exhaustive Play — D={D} (K={D-1})")
        game, cfg = make_game(D, ExplicitSurfaceDerivationGame)
        policies = enumerate_relaxed_policies(D)

        solve_count = 0
        print(f"  {'#':>3}  {'Policy':<45}  {'Reward':>8}  {'Solved':>6}")
        print(f"  {'---':>3}  {'-----':<45}  {'------':>8}  {'------':>6}")

        for i, policy in enumerate(policies):
            reward, _ = play_policy(game, policy)
            solved = reward > 0
            if solved:
                solve_count += 1
            print(f"  {i+1:>3}  {policy.pretty():<45}  {reward:>8.4f}  {'YES' if solved else 'no':>6}")

        print(f"\n  Total: {len(policies)} derivations, {solve_count} solving")
        print(f"  Solve fraction: {solve_count}/{len(policies)} = {solve_count/len(policies):.4f}")

    # ------------------------------------------------------------------
    # 2. Random Play D=3, D=4
    # ------------------------------------------------------------------
    for D in [3, 4]:
        section(f"Random Play — D={D} (K={D-1}), N=200 episodes")
        game, _ = make_game(D, ExplicitSurfaceDerivationGame)
        rng = np.random.default_rng(42)
        K = D - 1

        rewards = []
        solves = 0
        for _ in range(200):
            game.reset()
            for step in range(2 * K + 1):
                mask = game.get_action_mask()
                legal = np.where(mask)[0]
                action = rng.choice(legal)
                _, reward, terminated, _, info = game.step(action)
                if terminated:
                    break
            rewards.append(reward)
            if reward > 0:
                solves += 1

        rewards = np.array(rewards)
        print(f"  Episodes:   200")
        print(f"  Solve rate: {solves}/200 = {solves/200:.3f}")
        print(f"  Reward:     mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        print(f"              min={rewards.min():.4f}, max={rewards.max():.4f}")

    # ------------------------------------------------------------------
    # 3. Side-by-side Comparison — Canonical Policy
    # ------------------------------------------------------------------
    section("Side-by-side Comparison — D=3 Canonical Policy")
    D = 3
    policy = canonical_policy(D)
    print(f"  Policy: {policy.pretty()}")

    old_game, _ = make_game(D, SurfaceDerivationGame)
    new_game, cfg = make_game(D, ExplicitSurfaceDerivationGame)

    # Play through old game
    old_game.reset()
    new_game.reset()

    print(f"\n  {'Step':>4}  {'Rule':<20}  {'Old mask':>20}  {'New mask':>20}  {'Match':>6}")
    print(f"  {'----':>4}  {'----':<20}  {'--------':>20}  {'--------':>20}  {'-----':>6}")

    all_match = True
    for i, rule in enumerate(policy.rules):
        old_mask = old_game.get_action_mask()
        new_mask = new_game.get_action_mask()
        match = np.array_equal(old_mask, new_mask)
        if not match:
            all_match = False

        action = old_game._rule_to_action(rule)
        old_mask_str = ''.join(['1' if m else '0' for m in old_mask])
        new_mask_str = ''.join(['1' if m else '0' for m in new_mask])

        print(f"  {i:>4}  {rule.pretty():<20}  {old_mask_str:>20}  {new_mask_str:>20}  {'OK' if match else 'FAIL':>6}")

        old_game.step(action)
        new_game.step(action)

    print(f"\n  All masks match: {'YES' if all_match else 'NO'}")

    # Show compiled AST
    prog = compile_policy(policy, cfg)
    print(f"\n  Compiled AST:")
    for line in prog.pretty().split('\n'):
        print(f"    {line}")

    # ------------------------------------------------------------------
    # 4. Grammar derivation trace
    # ------------------------------------------------------------------
    section("Grammar Derivation Trace — D=3 Canonical")
    grammar = SurfaceCFG(D - 1)
    print(f"  {grammar.derivation_trace(policy)}")

    # ------------------------------------------------------------------
    # 5. Exact Counts
    # ------------------------------------------------------------------
    section("Exact Count Verification")
    print(f"  {'D':>3}  {'K':>3}  {'|N|':>6}  {'|P|':>6}  {'|L|':>8}  {'Solve':>6}  {'rho':>8}")
    print(f"  {'---':>3}  {'---':>3}  {'---':>6}  {'---':>6}  {'---':>8}  {'-----':>6}  {'---':>8}")

    for D in range(2, 5):
        K = D - 1
        g = SurfaceCFG(K)
        game, cfg = make_game(D, ExplicitSurfaceDerivationGame)
        policies = enumerate_relaxed_policies(D)

        solve_count = sum(1 for p in policies if play_policy(game, p)[0] > 0)
        rho = solve_count / len(policies) if policies else 1.0

        print(f"  {D:>3}  {K:>3}  {g.count_nonterminals():>6}  "
              f"{g.count_productions():>6}  {g.count_words():>8}  "
              f"{solve_count:>6}  {rho:>8.4f}")

    print()
    print("  All experiments completed successfully.")
    print()


if __name__ == "__main__":
    main()
