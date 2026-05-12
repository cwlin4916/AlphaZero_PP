#!/usr/bin/env python3
"""Inspect the explicit right-linear CFG for the Doors surface DSL.

Reports:
  1. Full grammar printout for D=2
  2. Derivation trace for the D=2 canonical policy
  3. Scaling table D=2..6
  4. Language equality check D=2..4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphazeropp.instances.doors.dsl.explicit_surface_cfg import SurfaceCFG
from alphazeropp.instances.doors.dsl.surface_grammar import (
    canonical_policy,
    count_relaxed_policies,
    enumerate_relaxed_policies,
    enumerate_surface_prefixes,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    # ------------------------------------------------------------------
    # 1. Full Grammar for D=2
    # ------------------------------------------------------------------
    section("D=2 Grammar (K=1)")
    cfg = SurfaceCFG(1)
    print(cfg.pretty())

    # ------------------------------------------------------------------
    # 2. Derivation Trace
    # ------------------------------------------------------------------
    section("D=2 Canonical Policy — Derivation Trace")
    policy = canonical_policy(2)
    print(f"  Policy:  {policy.pretty()}")
    print(f"  Trace:   {cfg.derivation_trace(policy)}")

    # Also show D=3
    section("D=3 Canonical Policy — Derivation Trace")
    cfg3 = SurfaceCFG(2)
    policy3 = canonical_policy(3)
    print(f"  Policy:  {policy3.pretty()}")
    print(f"  Trace:   {cfg3.derivation_trace(policy3)}")

    # ------------------------------------------------------------------
    # 3. Scaling Table
    # ------------------------------------------------------------------
    section("Scaling Table: D=2..6")
    print(f"  {'D':>3}  {'K':>3}  {'|N|':>6}  {'|P|':>8}  {'|L|':>10}  {'Relaxed':>10}")
    print(f"  {'---':>3}  {'---':>3}  {'---':>6}  {'---':>8}  {'---':>10}  {'-------':>10}")
    for d in range(2, 7):
        K = d - 1
        g = SurfaceCFG(K)
        relaxed = count_relaxed_policies(d)
        print(
            f"  {d:>3}  {K:>3}  {g.count_nonterminals():>6}  "
            f"{g.count_productions():>8}  {g.count_words():>10,}  "
            f"{relaxed:>10,}"
        )

    # ------------------------------------------------------------------
    # 4. Language Equality Check
    # ------------------------------------------------------------------
    section("Language Equality Verification (D=2..4)")
    for d in range(2, 5):
        K = d - 1
        g = SurfaceCFG(K)

        # Words
        cfg_words = set(g.enumerate_words())
        oracle_words = set(enumerate_relaxed_policies(d))
        words_ok = cfg_words == oracle_words

        # Prefixes
        cfg_prefixes = set(g.enumerate_prefixes())
        oracle_prefixes = set(enumerate_surface_prefixes(d))
        prefixes_ok = cfg_prefixes == oracle_prefixes

        status = "PASS" if (words_ok and prefixes_ok) else "FAIL"
        print(
            f"  D={d}:  words={len(cfg_words):>4} "
            f"({'✓' if words_ok else '✗'})  "
            f"prefixes={len(cfg_prefixes):>4} "
            f"({'✓' if prefixes_ok else '✗'})  "
            f"[{status}]"
        )

    print()


if __name__ == "__main__":
    main()
