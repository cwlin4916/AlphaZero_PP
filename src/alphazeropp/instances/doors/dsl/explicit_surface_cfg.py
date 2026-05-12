"""Explicit right-linear CFG for the Doors surface DSL.

Extracts the implicit grammar encoded by the runtime bitmask logic in
surface_derivation_game.py into an explicit grammar object G_K = (N, Σ, P, S).

Nonterminals are indexed by status vectors s ∈ {0, P, M}^K, where each
component tracks the stage of a key: untouched (0), picked (P), or moved (M).
The start symbol is the all-zeros vector. The unique terminal production is
at the all-M vector, producing GoalRule.

The generated language is exactly the set of relaxed policies: all orderings
of {PickRule(k), MoveRule(k)} satisfying Pick-before-Move per key, terminated
by GoalRule.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Union

from alphazeropp.instances.doors.dsl.surface_dsl import (
    GoalRule, MoveRule, PickRule, SurfacePolicy, SurfaceRule,
)

# Status constants
UNTOUCHED = 0
PICKED = 1
MOVED = 2

_STATUS_LABELS = {UNTOUCHED: "0", PICKED: "P", MOVED: "M"}


# ---------------------------------------------------------------------------
# StatusVector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatusVector:
    """Per-key status vector s ∈ {0, P, M}^K."""

    statuses: tuple[int, ...]

    def __post_init__(self):
        for v in self.statuses:
            if v not in (UNTOUCHED, PICKED, MOVED):
                raise ValueError(f"Invalid status {v}, must be 0, 1, or 2")

    @property
    def K(self) -> int:
        return len(self.statuses)

    @staticmethod
    def initial(K: int) -> StatusVector:
        """All-zeros: (0, 0, ..., 0)."""
        return StatusVector(tuple(UNTOUCHED for _ in range(K)))

    @staticmethod
    def from_bitmasks(picked_mask: int, moved_mask: int, K: int) -> StatusVector:
        """Bijection from (picked_mask, moved_mask) → StatusVector.

        Requires: moved_mask is a subset of picked_mask.
        """
        if moved_mask & ~picked_mask:
            raise ValueError("moved_mask must be a subset of picked_mask")
        statuses = []
        for k in range(K):
            if moved_mask & (1 << k):
                statuses.append(MOVED)
            elif picked_mask & (1 << k):
                statuses.append(PICKED)
            else:
                statuses.append(UNTOUCHED)
        return StatusVector(tuple(statuses))

    def to_bitmasks(self) -> tuple[int, int]:
        """Inverse bijection: StatusVector → (picked_mask, moved_mask)."""
        picked = 0
        moved = 0
        for k, s in enumerate(self.statuses):
            if s == PICKED:
                picked |= (1 << k)
            elif s == MOVED:
                picked |= (1 << k)
                moved |= (1 << k)
        return picked, moved

    def with_update(self, k: int, new_status: int) -> StatusVector:
        """Return copy with position k set to new_status."""
        lst = list(self.statuses)
        lst[k] = new_status
        return StatusVector(tuple(lst))

    def is_all_moved(self) -> bool:
        """True iff all statuses are MOVED."""
        return all(s == MOVED for s in self.statuses)

    def pretty(self) -> str:
        labels = [_STATUS_LABELS[s] for s in self.statuses]
        return "(" + ",".join(labels) + ")"


# ---------------------------------------------------------------------------
# Nonterminal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Nonterminal:
    """Grammar nonterminal N_s indexed by StatusVector."""

    status: StatusVector

    def is_start(self) -> bool:
        return all(s == UNTOUCHED for s in self.status.statuses)

    def is_pre_goal(self) -> bool:
        return self.status.is_all_moved()

    def pretty(self) -> str:
        return f"N_{self.status.pretty()}"


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Production:
    """Grammar production: lhs → terminal [rhs].

    If rhs is None, this is a terminal production (GoalRule).
    """

    lhs: Nonterminal
    terminal: SurfaceRule
    rhs: Nonterminal | None

    def is_terminal(self) -> bool:
        return self.rhs is None

    def pretty(self) -> str:
        lhs_str = self.lhs.pretty()
        term_str = self.terminal.pretty()
        if self.rhs is None:
            return f"{lhs_str} → {term_str}"
        return f"{lhs_str} → {term_str} {self.rhs.pretty()}"


# ---------------------------------------------------------------------------
# SurfaceCFG
# ---------------------------------------------------------------------------

class SurfaceCFG:
    """Explicit right-linear grammar G_K = (N, Σ, P, S).

    Constructed eagerly from K (number of keys = D - 1).
    """

    def __init__(self, K: int):
        if K < 0:
            raise ValueError(f"K must be >= 0, got {K}")
        self.K = K

        # Build all 3^K nonterminals
        if K == 0:
            all_statuses = [StatusVector(())]
        else:
            all_statuses = [
                StatusVector(combo)
                for combo in itertools.product(
                    [UNTOUCHED, PICKED, MOVED], repeat=K
                )
            ]
        self.nonterminals = [Nonterminal(s) for s in all_statuses]

        # Terminal alphabet
        self.terminals: list[SurfaceRule] = (
            [PickRule(k) for k in range(K)]
            + [MoveRule(k) for k in range(K)]
            + [GoalRule()]
        )

        # Build productions
        self.productions: list[Production] = []
        self._prod_by_lhs: dict[StatusVector, list[Production]] = {}

        for nt in self.nonterminals:
            s = nt.status
            prods: list[Production] = []
            for k in range(K):
                if s.statuses[k] == UNTOUCHED:
                    s_new = s.with_update(k, PICKED)
                    prods.append(Production(nt, PickRule(k), Nonterminal(s_new)))
                elif s.statuses[k] == PICKED:
                    s_new = s.with_update(k, MOVED)
                    prods.append(Production(nt, MoveRule(k), Nonterminal(s_new)))
            if s.is_all_moved():
                prods.append(Production(nt, GoalRule(), rhs=None))
            self.productions.extend(prods)
            self._prod_by_lhs[s] = prods

        # Start symbol
        self.start = Nonterminal(StatusVector.initial(K))

    # -- Queries -----------------------------------------------------------

    def legal_actions(self, nt: Nonterminal) -> list[SurfaceRule]:
        """Legal terminal symbols from this nonterminal."""
        return [p.terminal for p in self._prod_by_lhs[nt.status]]

    def successor(
        self, nt: Nonterminal, rule: SurfaceRule
    ) -> Nonterminal | None:
        """Apply production: returns successor nonterminal, or None if terminal."""
        for p in self._prod_by_lhs[nt.status]:
            if p.terminal == rule:
                return p.rhs
        raise ValueError(
            f"No production {nt.pretty()} → {rule.pretty()}"
        )

    # -- Enumeration -------------------------------------------------------

    def enumerate_words(
        self, *, max_enumerate: int = 100_000
    ) -> list[SurfacePolicy]:
        """Grammar-guided DFS of all complete derivations."""
        count = self.count_words()
        if count > max_enumerate:
            raise ValueError(
                f"Word count {count} for K={self.K} exceeds "
                f"max_enumerate={max_enumerate}"
            )

        results: list[SurfacePolicy] = []
        buf: list[SurfaceRule] = []

        def dfs(nt: Nonterminal) -> None:
            for prod in self._prod_by_lhs[nt.status]:
                buf.append(prod.terminal)
                if prod.rhs is None:
                    results.append(SurfacePolicy(tuple(buf)))
                else:
                    dfs(prod.rhs)
                buf.pop()

        dfs(self.start)
        return results

    def enumerate_prefixes(
        self, *, max_enumerate: int = 1_000_000
    ) -> list[tuple[SurfaceRule, ...]]:
        """All valid partial derivation strings, including empty prefix."""
        prefixes: list[tuple[SurfaceRule, ...]] = [()]
        buf: list[SurfaceRule] = []

        def dfs(nt: Nonterminal) -> None:
            for prod in self._prod_by_lhs[nt.status]:
                buf.append(prod.terminal)
                prefixes.append(tuple(buf))
                if len(prefixes) > max_enumerate:
                    raise ValueError(
                        f"Prefix count exceeds max_enumerate={max_enumerate} "
                        f"for K={self.K}"
                    )
                if prod.rhs is not None:
                    dfs(prod.rhs)
                buf.pop()

        dfs(self.start)
        return prefixes

    # -- Counting ----------------------------------------------------------

    def count_words(self) -> int:
        """Closed-form: (2K)!/2^K."""
        K = self.K
        if K == 0:
            return 1
        return math.factorial(2 * K) // (2 ** K)

    def count_nonterminals(self) -> int:
        return 3 ** self.K if self.K > 0 else 1

    def count_productions(self) -> int:
        K = self.K
        if K == 0:
            return 1
        return 2 * K * (3 ** (K - 1)) + 1

    # -- Pretty-printing ---------------------------------------------------

    def pretty(self) -> str:
        """Full grammar printout."""
        lines = [f"Right-linear grammar G_{self.K}"]
        lines.append(f"  K = {self.K}")
        lines.append(f"  |N| = {len(self.nonterminals)}")
        lines.append(f"  |Σ| = {len(self.terminals)}")
        lines.append(f"  |P| = {len(self.productions)}")
        lines.append(f"  |L| = {self.count_words()}")
        lines.append(f"  S   = {self.start.pretty()}")
        lines.append("")
        lines.append("Nonterminals:")
        for nt in self.nonterminals:
            label = ""
            if nt.is_start():
                label = "  (start)"
            elif nt.is_pre_goal():
                label = "  (pre-goal)"
            lines.append(f"  {nt.pretty()}{label}")
        lines.append("")
        lines.append("Terminals:")
        for t in self.terminals:
            lines.append(f"  {t.pretty()}")
        lines.append("")
        lines.append("Productions:")
        for p in self.productions:
            lines.append(f"  {p.pretty()}")
        return "\n".join(lines)

    def derivation_trace(self, policy: SurfacePolicy) -> str:
        """Step-by-step derivation showing nonterminal rewriting."""
        steps: list[str] = []
        nt = self.start
        placed: list[str] = []

        for rule in policy.rules:
            # Show current sentential form
            form = " ".join(placed) + (" " if placed else "") + nt.pretty()
            steps.append(form.strip())

            # Apply production
            placed.append(rule.pretty())
            next_nt = self.successor(nt, rule)
            if next_nt is None:
                # Terminal production — show final word
                steps.append(" ".join(placed))
                break
            nt = next_nt

        return " ⇒ ".join(steps)
