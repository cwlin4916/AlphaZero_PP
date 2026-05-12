"""Unmasked right-linear CFG for the Doors surface DSL (Stage 2).

Removes all domain-specific constraints from SurfaceCFG (Stage 1).
The grammar generates all macro sequences of length 0..L_max over
{P_0, ..., P_{K-1}, M_0, ..., M_{K-1}}, terminated by GoalRule.

No precedence, uniqueness, or completeness masks.
Bad programs fail by reward, not by generation-time filtering.

Two modes:
  - max-length:   S_l -> G | T S_{l-1}  (can stop early)
  - exact-length: S_l -> T S_{l-1}       (must use exactly L_max tokens)
                  S_0 -> G               (then terminate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from alphazeropp.instances.doors.dsl.surface_dsl import (
    GoalRule, MoveRule, PickRule, SurfacePolicy, SurfaceRule,
)


class UnmaskedSurfaceCFG:
    """Unmasked right-linear grammar G_2.

    Nonterminals are S_0, S_1, ..., S_{L_max} (indexed by remaining slots).
    Start symbol is S_{L_max}.
    """

    def __init__(self, K: int, *, exact_length: bool = False):
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
        self.K = K
        self.L_max = 2 * K
        self.exact_length = exact_length

        # Terminal alphabet
        self.non_goal_terminals: list[SurfaceRule] = (
            [PickRule(k) for k in range(K)]
            + [MoveRule(k) for k in range(K)]
        )
        self.terminals: list[SurfaceRule] = (
            self.non_goal_terminals + [GoalRule()]
        )

        self.start_level = self.L_max

    # -- Queries -----------------------------------------------------------

    def legal_actions(self, level: int) -> list[SurfaceRule]:
        """Legal terminal symbols from nonterminal S_level."""
        if level == 0:
            return [GoalRule()]
        if self.exact_length:
            return list(self.non_goal_terminals)
        else:
            return list(self.non_goal_terminals) + [GoalRule()]

    def successor(self, level: int, rule: SurfaceRule) -> int | None:
        """Apply production: returns next level, or None if GoalRule."""
        if isinstance(rule, GoalRule):
            return None
        if level <= 0:
            raise ValueError(
                f"Cannot apply non-goal rule at level 0"
            )
        return level - 1

    # -- Enumeration -------------------------------------------------------

    def enumerate_words(
        self, *, max_enumerate: int = 1_000_000
    ) -> list[SurfacePolicy]:
        """DFS enumeration of all complete derivations."""
        count = self.count_words()
        if count > max_enumerate:
            raise ValueError(
                f"Word count {count} for K={self.K} "
                f"(exact={self.exact_length}) exceeds "
                f"max_enumerate={max_enumerate}"
            )

        results: list[SurfacePolicy] = []
        buf: list[SurfaceRule] = []

        def dfs(level: int) -> None:
            for rule in self.legal_actions(level):
                buf.append(rule)
                nxt = self.successor(level, rule)
                if nxt is None:
                    results.append(SurfacePolicy(tuple(buf)))
                else:
                    dfs(nxt)
                buf.pop()

        dfs(self.start_level)
        return results

    # -- Counting ----------------------------------------------------------

    def count_words(self) -> int:
        """Closed-form word count."""
        n = 2 * self.K  # number of non-goal terminals
        L = self.L_max
        if self.exact_length:
            # Exactly L non-goal tokens then G
            return n ** L if n > 0 else 1
        else:
            # 0..L non-goal tokens then G
            if n <= 1:
                return L + 1 if n == 1 else 1
            return (n ** (L + 1) - 1) // (n - 1)

    def count_nonterminals(self) -> int:
        return self.L_max + 1

    def count_productions(self) -> int:
        """Total number of grammar productions."""
        n = 2 * self.K
        if self.exact_length:
            # S_0 -> G  (1 production)
            # S_l -> n non-goal productions each, for l=1..L_max
            return 1 + n * self.L_max
        else:
            # S_0 -> G  (1 production)
            # S_l -> G + n non-goal = (n+1) productions each, for l=1..L_max
            return 1 + (n + 1) * self.L_max

    # -- Pretty-printing ---------------------------------------------------

    def pretty(self) -> str:
        mode = "exact-length" if self.exact_length else "max-length"
        lines = [f"Unmasked right-linear grammar G_2 ({mode})"]
        lines.append(f"  K = {self.K}")
        lines.append(f"  L_max = {self.L_max}")
        lines.append(f"  |N| = {self.count_nonterminals()}")
        lines.append(f"  |Σ| = {len(self.terminals)}")
        lines.append(f"  |P| = {self.count_productions()}")
        lines.append(f"  |L| = {self.count_words()}")
        lines.append(f"  S   = S_{self.start_level}")
        lines.append("")
        lines.append("Productions:")
        lines.append(f"  S_0 -> G")
        for l in range(1, self.L_max + 1):
            rhs_parts = [
                f"{r.pretty()} S_{l-1}"
                for r in self.non_goal_terminals
            ]
            if not self.exact_length:
                rhs_parts.append("G")
            lines.append(f"  S_{l} -> " + " | ".join(rhs_parts))
        return "\n".join(lines)

    def derivation_trace(self, policy: SurfacePolicy) -> str:
        """Step-by-step derivation showing level rewriting."""
        steps: list[str] = []
        level = self.start_level
        placed: list[str] = []

        for rule in policy.rules:
            form = " ".join(placed) + (" " if placed else "") + f"S_{level}"
            steps.append(form.strip())
            placed.append(rule.pretty())
            nxt = self.successor(level, rule)
            if nxt is None:
                steps.append(" ".join(placed))
                break
            level = nxt

        return " => ".join(steps)
