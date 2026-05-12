"""Stage 3.1 typed-variable lifted DSL for Doors.

Parallel to ``lifted_dsl.py``: where the existing module uses hardcoded
selectors (``NextLockedRoom``, ``KeyFor``, ``CurrentRoom``), this module
expresses lifted rules with explicit typed variables that are bound at
runtime by the binding engine.  The two modules coexist; nothing here
modifies the existing canonical_lifted_policy() or its tests.

The predicate / action signature here adapts to the actual Doors env
(pick auto-unlocks; no inventory predicate).  ``next(room, room)`` is
included for completeness — it is used in Stage 3.2 variant B.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Typed predicate / action signatures
# ---------------------------------------------------------------------------

PREDICATES: dict[str, tuple[str, ...]] = {
    "at_loc":        ("loc",),
    "locked":        ("room",),
    "key_available": ("key",),
    "key_for":       ("key", "room"),
    "loc_of_key":    ("key", "loc"),
    "loc_in_room":   ("loc", "room"),
    "next":          ("room", "room"),
    "goal_loc":      ("loc",),
}

ACTIONS: dict[str, tuple[str, ...]] = {
    "pick":    ("key",),
    "move_to": ("loc",),
}


# ---------------------------------------------------------------------------
# AST nodes (frozen dataclasses, matching lifted_dsl.py style)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:
    name: str
    type: str

    def __post_init__(self):
        if self.type not in {"room", "key", "loc"}:
            raise ValueError(f"Unknown type: {self.type!r}")

    def pretty(self) -> str:
        return f"{self.name}:{self.type}"


@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[Var, ...]

    def __post_init__(self):
        sig = PREDICATES.get(self.pred)
        if sig is None:
            raise ValueError(f"Unknown predicate: {self.pred!r}")
        if len(self.args) != len(sig):
            raise ValueError(
                f"{self.pred} takes {len(sig)} args, got {len(self.args)}"
            )
        for v, expected_t in zip(self.args, sig):
            if v.type != expected_t:
                raise TypeError(
                    f"{self.pred} arg {v.name}:{v.type} expects {expected_t}"
                )

    def pretty(self) -> str:
        args = ", ".join(v.name for v in self.args)
        return f"{self.pred}({args})"


@dataclass(frozen=True)
class Rule:
    pre: tuple[Atom, ...]
    action: str
    action_args: tuple[Var, ...]

    def __post_init__(self):
        sig = ACTIONS.get(self.action)
        if sig is None:
            raise ValueError(f"Unknown action: {self.action!r}")
        if len(self.action_args) != len(sig):
            raise ValueError(
                f"{self.action} takes {len(sig)} args, got {len(self.action_args)}"
            )
        for v, expected_t in zip(self.action_args, sig):
            if v.type != expected_t:
                raise TypeError(
                    f"{self.action} arg {v.name}:{v.type} expects {expected_t}"
                )
        # Action vars must appear in some pre-atom (PG3 well-formedness).
        guard_vars = {v for atom in self.pre for v in atom.args}
        for v in self.action_args:
            if v not in guard_vars:
                raise ValueError(
                    f"action var {v.name} not bound by guard"
                )

    def variables(self) -> tuple[Var, ...]:
        seen: dict[Var, None] = {}
        for atom in self.pre:
            for v in atom.args:
                seen.setdefault(v, None)
        for v in self.action_args:
            seen.setdefault(v, None)
        return tuple(seen)

    def pretty(self) -> str:
        guard = " ∧ ".join(a.pretty() for a in self.pre) or "⊤"
        action = f"{self.action}({', '.join(v.name for v in self.action_args)})"
        return f"if {guard} then {action}"


@dataclass(frozen=True)
class LiftedDecisionList:
    rules: tuple[Rule, ...]

    def pretty(self) -> str:
        return "\n".join(f"ρ_{i}: {r.pretty()}" for i, r in enumerate(self.rules, 1))


# ---------------------------------------------------------------------------
# Manual three-rule Doors policy (the Stage 3.1 fixture)
# ---------------------------------------------------------------------------

def manual_three_rule_policy() -> LiftedDecisionList:
    # Frontier abstraction by binding-search order: in `pre`, room var `r`
    # appears before key var `k`, so lex-min over r picks the smallest locked
    # room first.  This makes the policy invariant under key-renaming
    # (validated by the object-renaming test; see <a> §4).  Room ordering is
    # *not* permuted — Doors has semantic room roles (0=start, D-1=goal).
    u = Var("u", "loc")
    l = Var("l", "loc")
    r = Var("r", "room")
    k = Var("k", "key")
    g = Var("g", "loc")

    rho_pick = Rule(
        pre=(
            Atom("locked", (r,)),
            Atom("key_for", (k, r)),
            Atom("loc_of_key", (k, u)),
            Atom("at_loc", (u,)),
            Atom("key_available", (k,)),
        ),
        action="pick",
        action_args=(k,),
    )

    rho_move_to_key = Rule(
        pre=(
            Atom("locked", (r,)),
            Atom("key_for", (k, r)),
            Atom("loc_of_key", (k, l)),
            Atom("key_available", (k,)),
        ),
        action="move_to",
        action_args=(l,),
    )

    rho_finish = Rule(
        pre=(Atom("goal_loc", (g,)),),
        action="move_to",
        action_args=(g,),
    )

    return LiftedDecisionList(rules=(rho_pick, rho_move_to_key, rho_finish))
