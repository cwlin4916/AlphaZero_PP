"""Typed lifted decision-list DSL.

Domain-agnostic counterpart to ``instances/doors/dsl/lifted_typed_dsl.py``.
Stage 1 of the lifted-policy program-synthesis project — see
``docs/notes/stage4/01_plan.md``.

Design points worth flagging for future readers:

* Object identity is ``str``. Doors uses int obs-indices; this module deliberately
  diverges so that synthesis-time policies are readable and renaming-invariant.
* ``Rule.body`` is a single ordered tuple of ``Literal``. The state/goal split is
  carried by ``Literal.source``, not by separate ``pre`` / ``goal`` fields.
* Safe-negation is enforced at ``Rule.__post_init__``: every variable in a
  negative literal must also appear positively or in the action arguments,
  so the interpreter can assume neg-vars are bound by the positive pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


# ---------------------------------------------------------------------------
# Type / signature catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredicateSchema:
    name: str
    arg_types: tuple[str, ...]


@dataclass(frozen=True)
class ActionSchema:
    name: str
    arg_types: tuple[str, ...]


# ---------------------------------------------------------------------------
# Syntactic atoms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:
    name: str
    type_name: str

    def pretty(self) -> str:
        return self.name


Arg = Union[Var, str]


@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[Arg, ...]

    def pretty(self) -> str:
        parts = [a.pretty() if isinstance(a, Var) else a for a in self.args]
        return f"{self.pred}({', '.join(parts)})"

    def variables(self) -> tuple[Var, ...]:
        return tuple(a for a in self.args if isinstance(a, Var))


class LiteralSource(Enum):
    STATE = "state"
    GOAL = "goal"


@dataclass(frozen=True)
class Literal:
    atom: Atom
    negated: bool = False
    source: LiteralSource = LiteralSource.STATE

    def pretty(self) -> str:
        prefix = "¬" if self.negated else ""
        if self.source is LiteralSource.GOAL:
            return f"{prefix}Goal[{self.atom.pretty()}]"
        return f"{prefix}{self.atom.pretty()}"


def state_lit(pred: str, *args: Arg, negated: bool = False) -> Literal:
    return Literal(Atom(pred, tuple(args)), negated=negated, source=LiteralSource.STATE)


def goal_lit(pred: str, *args: Arg, negated: bool = False) -> Literal:
    return Literal(Atom(pred, tuple(args)), negated=negated, source=LiteralSource.GOAL)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedAction:
    schema: str
    args: tuple[Var, ...]

    def pretty(self) -> str:
        return f"{self.schema}({', '.join(v.name for v in self.args)})"


@dataclass(frozen=True)
class GroundAction:
    schema: str
    args: tuple[str, ...]

    def pretty(self) -> str:
        return f"{self.schema}({', '.join(self.args)})"


# ---------------------------------------------------------------------------
# Rules / policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    vars: tuple[Var, ...]
    body: tuple[Literal, ...]
    action: LiftedAction

    def __post_init__(self) -> None:
        # 1. vars must be unique by name and each must carry a type.
        seen: dict[str, str] = {}
        for v in self.vars:
            if v.name in seen and seen[v.name] != v.type_name:
                raise ValueError(
                    f"variable {v.name!r} declared with conflicting types "
                    f"{seen[v.name]!r} and {v.type_name!r}"
                )
            seen[v.name] = v.type_name

        # 2. every Var appearing in body or action must be declared in self.vars.
        declared = {v.name: v.type_name for v in self.vars}
        for lit in self.body:
            for v in lit.atom.variables():
                if v.name not in declared:
                    raise ValueError(
                        f"variable {v.name!r} used in literal {lit.pretty()} "
                        f"is not declared in rule.vars"
                    )
                if declared[v.name] != v.type_name:
                    raise ValueError(
                        f"variable {v.name!r} typed {v.type_name!r} in literal "
                        f"but declared {declared[v.name]!r}"
                    )
        for v in self.action.args:
            if v.name not in declared:
                raise ValueError(
                    f"action variable {v.name!r} is not declared in rule.vars"
                )
            if declared[v.name] != v.type_name:
                raise ValueError(
                    f"action variable {v.name!r} typed {v.type_name!r} but "
                    f"declared {declared[v.name]!r}"
                )

        # 3. safe-negation: vars appearing in negative literals must also appear
        #    in some positive literal or in self.action.args.
        positive_vars: set[str] = set()
        for lit in self.body:
            if not lit.negated:
                positive_vars.update(v.name for v in lit.atom.variables())
        positive_vars.update(v.name for v in self.action.args)
        for lit in self.body:
            if not lit.negated:
                continue
            for v in lit.atom.variables():
                if v.name not in positive_vars:
                    raise ValueError(
                        f"unsafe negation: variable {v.name!r} in negative "
                        f"literal {lit.pretty()} does not appear positively "
                        f"or in the action arguments"
                    )

    def pretty(self) -> str:
        body_str = " ∧ ".join(lit.pretty() for lit in self.body) or "⊤"
        return f"{body_str} ⇒ {self.action.pretty()}"


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...]

    def pretty(self) -> str:
        return "\n".join(f"ρ_{i + 1}: {r.pretty()}" for i, r in enumerate(self.rules))


# ---------------------------------------------------------------------------
# Relational state alias
# ---------------------------------------------------------------------------

RelState = dict[str, set[tuple[str, ...]]]


# ---------------------------------------------------------------------------
# Type-checking helpers
# ---------------------------------------------------------------------------

def check_atom_against_schema(
    atom: Atom,
    schema: PredicateSchema,
    var_types: dict[str, str],
) -> None:
    """Verify ``atom`` is well-typed under ``schema``.

    For each variable argument, the variable's declared type must match the
    schema's type at that position. For each ground (str) argument, no type
    check is performed at the DSL level (instance code can enforce membership
    at runtime).
    """
    if atom.pred != schema.name:
        raise ValueError(
            f"atom predicate {atom.pred!r} does not match schema {schema.name!r}"
        )
    if len(atom.args) != len(schema.arg_types):
        raise ValueError(
            f"atom {atom.pretty()} has {len(atom.args)} args but schema "
            f"{schema.name!r} expects {len(schema.arg_types)}"
        )
    for i, (a, expected_type) in enumerate(zip(atom.args, schema.arg_types)):
        if isinstance(a, Var):
            declared = var_types.get(a.name, a.type_name)
            if declared != expected_type:
                raise ValueError(
                    f"atom {atom.pretty()} arg {i}: variable {a.name!r} has "
                    f"type {declared!r} but schema expects {expected_type!r}"
                )


def check_action_against_schema(act: LiftedAction, schema: ActionSchema) -> None:
    if act.schema != schema.name:
        raise ValueError(
            f"action schema {act.schema!r} does not match {schema.name!r}"
        )
    if len(act.args) != len(schema.arg_types):
        raise ValueError(
            f"action {act.pretty()} has {len(act.args)} args but schema "
            f"{schema.name!r} expects {len(schema.arg_types)}"
        )
    for i, (v, expected_type) in enumerate(zip(act.args, schema.arg_types)):
        if v.type_name != expected_type:
            raise ValueError(
                f"action {act.pretty()} arg {i}: variable {v.name!r} has type "
                f"{v.type_name!r} but schema expects {expected_type!r}"
            )
