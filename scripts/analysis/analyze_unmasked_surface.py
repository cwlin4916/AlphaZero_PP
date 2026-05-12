#!/usr/bin/env python3
"""Exact analysis of the Stage 2 unmasked surface grammar.

For each D value, enumerates all derivations, compiles and evaluates them,
and reports language sizes, solve counts, unique ASTs, and reward histograms.

Usage:
    python scripts/analysis/analyze_unmasked_surface.py [--max-D 4]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from alphazeropp.instances.doors.dsl.doors_config import (
    DoorsGameConfig, doors_initial_state,
)
from alphazeropp.instances.doors.dsl.unmasked_surface_cfg import UnmaskedSurfaceCFG
from alphazeropp.instances.doors.dsl.explicit_surface_cfg import SurfaceCFG
from alphazeropp.instances.doors.dsl.surface_dsl import SurfacePolicy
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


def analyze_grammar(D: int, exact_length: bool, max_enumerate: int = 5_000_000):
    """Analyze one grammar configuration exhaustively."""
    K = D - 1
    mode = "exact" if exact_length else "max"

    grammar = UnmaskedSurfaceCFG(K, exact_length=exact_length)
    word_count = grammar.count_words()

    result = {
        "D": D,
        "K": K,
        "mode": mode,
        "word_count_formula": word_count,
    }

    if word_count > max_enumerate:
        result["enumerated"] = False
        result["note"] = f"Too large to enumerate ({word_count:,} words)"
        return result

    # Enumerate
    t0 = time.time()
    words = grammar.enumerate_words(max_enumerate=max_enumerate)
    t_enum = time.time() - t0
    result["enumerated"] = True
    result["word_count_actual"] = len(words)
    result["enum_time_s"] = round(t_enum, 2)

    assert len(words) == word_count, (
        f"Count mismatch: formula={word_count}, actual={len(words)}"
    )

    # Set up compiler and evaluator
    cfg = DoorsGameConfig(num_rooms=D, locs_per_room=2)
    n_sites = cfg.obs_size()
    x0 = doors_initial_state(cfg)
    le = LeafEvaluator(
        n_sites, [x0], cfg,
        is_solved=cfg.is_solved, metric="solve_rate",
    )

    # Compile and evaluate all
    t0 = time.time()
    rewards = []
    ast_to_derivations: dict[str, list[str]] = defaultdict(list)
    solved_derivations = []

    for word in words:
        prog = compile_policy(word, cfg)
        reward = le(prog)
        rewards.append(reward)
        ast_key = prog.pretty()
        ast_to_derivations[ast_key].append(word.pretty())
        if reward > 0:
            solved_derivations.append(word)

    t_eval = time.time() - t0

    rewards_arr = np.array(rewards)
    result["eval_time_s"] = round(t_eval, 2)
    result["unique_asts"] = len(ast_to_derivations)
    result["solve_count"] = len(solved_derivations)
    result["solve_density"] = len(solved_derivations) / len(words)
    result["reward_mean"] = float(np.mean(rewards_arr))
    result["reward_std"] = float(np.std(rewards_arr))
    result["reward_min"] = float(np.min(rewards_arr))
    result["reward_max"] = float(np.max(rewards_arr))

    # Reward histogram buckets
    bins = [-np.inf, -0.5, 0.0, 0.5, 1.0, np.inf]
    hist, _ = np.histogram(rewards_arr, bins=bins)
    result["reward_histogram"] = {
        "(-inf,-0.5]": int(hist[0]),
        "(-0.5,0.0]": int(hist[1]),
        "(0.0,0.5]": int(hist[2]),
        "(0.5,1.0]": int(hist[3]),
        "(1.0,+inf)": int(hist[4]),
    }

    # AST collisions (different derivations -> same AST)
    collision_counts = [len(v) for v in ast_to_derivations.values() if len(v) > 1]
    result["ast_collisions"] = len(collision_counts)
    result["max_collision_size"] = max(collision_counts) if collision_counts else 0

    # Shortest solving derivation
    if solved_derivations:
        shortest = min(solved_derivations, key=lambda w: len(w.rules))
        result["shortest_solving"] = shortest.pretty()
        result["shortest_solving_len"] = len(shortest.rules)

    # Sample bad derivation
    bad = [w for w, r in zip(words, rewards) if r <= 0]
    if bad:
        sample = bad[0]
        prog = compile_policy(sample, cfg)
        result["sample_bad"] = {
            "derivation": sample.pretty(),
            "ast": prog.pretty(),
            "reward": float(rewards[words.index(sample)]),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-D", type=int, default=7)
    parser.add_argument("--max-enumerate", type=int, default=5_000_000)
    args = parser.parse_args()

    print("=" * 72)
    print("Stage 2 Unmasked Surface Grammar — Exact Analysis")
    print("=" * 72)

    for D in range(2, args.max_D + 1):
        K = D - 1
        print(f"\n{'─' * 72}")
        print(f"D = {D}  (K = {K})")
        print(f"{'─' * 72}")

        for exact in [True, False]:
            mode = "exact-length" if exact else "max-length"
            result = analyze_grammar(D, exact, max_enumerate=args.max_enumerate)

            print(f"\n  [{mode}]")
            print(f"    Language size (formula): {result['word_count_formula']:>15,}")

            if not result.get("enumerated"):
                print(f"    {result.get('note', 'Skipped')}")
                continue

            print(f"    Language size (actual):  {result['word_count_actual']:>15,}")
            print(f"    Unique ASTs:            {result['unique_asts']:>15,}")
            print(f"    Solving derivations:    {result['solve_count']:>15,}")
            print(f"    Solve density:          {result['solve_density']:>15.6f}")
            print(f"    Reward: mean={result['reward_mean']:.4f}, "
                  f"std={result['reward_std']:.4f}, "
                  f"range=[{result['reward_min']:.4f}, {result['reward_max']:.4f}]")
            print(f"    Reward histogram: {result['reward_histogram']}")
            print(f"    AST collisions: {result['ast_collisions']} groups, "
                  f"max size {result['max_collision_size']}")
            if "shortest_solving" in result:
                print(f"    Shortest solving ({result['shortest_solving_len']} rules): "
                      f"{result['shortest_solving']}")
            if "sample_bad" in result:
                sb = result["sample_bad"]
                print(f"    Sample bad derivation: {sb['derivation']}")
                print(f"      AST: {sb['ast'][:80]}...")
                print(f"      Reward: {sb['reward']}")
            print(f"    Time: enum={result['enum_time_s']}s, eval={result['eval_time_s']}s")

    # Stage 1 comparison summary
    print(f"\n{'=' * 72}")
    print("Stage 1 vs Stage 2 — Language Size Summary")
    print(f"{'=' * 72}")
    print(f"{'D':>3} {'K':>3} {'|L(G1)|':>12} {'|L<=(G2)|':>14} {'|L=(G2)|':>14} "
          f"{'Ratio<=':>10} {'Ratio=':>10}")
    print(f"{'─' * 72}")

    for D in range(2, args.max_D + 1):
        K = D - 1
        g1 = math.factorial(2 * K) // (2 ** K)
        n = 2 * K
        L = 2 * K
        g2_max = sum(n**l for l in range(L + 1)) if n > 0 else 1
        g2_exact = n ** L if n > 0 else 1
        r_max = g2_max / g1 if g1 > 0 else float("inf")
        r_exact = g2_exact / g1 if g1 > 0 else float("inf")
        print(f"{D:>3} {K:>3} {g1:>12,} {g2_max:>14,} {g2_exact:>14,} "
              f"{r_max:>9.0f}x {r_exact:>9.0f}x")


if __name__ == "__main__":
    main()
