"""Online unification interpreter for lifted decision-list policies.

PG3-style first-applicable semantics under safe negation. No full grounding:
``find_bindings`` enumerates substitutions rule-by-rule, constraining variables
through positive literals before falling back to type-based enumeration of any
free action variables.

See ``docs/notes/stage4/01_plan.md`` §"Module/class design" for the contract.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Optional, Union

from alphazeropp.synthesis.lifted_dsl import (
    Atom,
    GroundAction,
    Literal,
    LiteralSource,
    Policy,
    RelState,
    Rule,
    Var,
)


# A binding maps Var.name -> object-name.
Binding = dict[str, str]


def _atom_row_matches(
    atom: Atom,
    row: tuple[str, ...],
    theta: Binding,
) -> Optional[Binding]:
    """Try to unify ``atom.args`` with the ground tuple ``row`` extending ``theta``.

    Returns the extended binding on success, or ``None`` on conflict. Mutates
    nothing — returns a fresh dict so callers can backtrack cheaply.
    """
    if len(atom.args) != len(row):
        return None
    new_theta = dict(theta)
    for arg, obj in zip(atom.args, row):
        if isinstance(arg, Var):
            existing = new_theta.get(arg.name)
            if existing is None:
                new_theta[arg.name] = obj
            elif existing != obj:
                return None
        else:
            # ground constant in the atom must match the row position literally
            if arg != obj:
                return None
    return new_theta


def _eval_negative_literal(
    lit: Literal,
    theta: Binding,
    state_atoms: RelState,
    goal_atoms: RelState,
) -> bool:
    """Closed-world evaluation of a negative literal under a complete-enough
    binding ``theta``. Safe-negation enforcement at rule construction guarantees
    every variable in ``lit.atom`` is already bound here.
    """
    source = goal_atoms if lit.source is LiteralSource.GOAL else state_atoms
    rows = source.get(lit.atom.pred, set())
    grounded: list[str] = []
    for arg in lit.atom.args:
        if isinstance(arg, Var):
            grounded.append(theta[arg.name])
        else:
            grounded.append(arg)
    return tuple(grounded) not in rows


def _positive_literal_rows(
    lit: Literal,
    state_atoms: RelState,
    goal_atoms: RelState,
) -> Iterable[tuple[str, ...]]:
    source = goal_atoms if lit.source is LiteralSource.GOAL else state_atoms
    return source.get(lit.atom.pred, set())


def find_bindings(
    rule: Rule,
    state_atoms: RelState,
    goal_atoms: RelState,
    objects_by_type: dict[str, tuple[str, ...]],
    legal_actions: set[GroundAction],
) -> list[Binding]:
    """Enumerate all valid substitutions for ``rule`` under the given context.

    Returned bindings are sorted by ``tuple(sorted(theta.items()))`` for
    determinism — the first element is therefore the lex-min binding.
    """
    positive = [lit for lit in rule.body if not lit.negated]
    negative = [lit for lit in rule.body if lit.negated]

    # 1. Iteratively extend theta over positive literals.
    partial: list[Binding] = [dict()]
    for lit in positive:
        next_partial: list[Binding] = []
        rows = _positive_literal_rows(lit, state_atoms, goal_atoms)
        for theta in partial:
            for row in rows:
                extended = _atom_row_matches(lit.atom, row, theta)
                if extended is not None:
                    next_partial.append(extended)
        partial = next_partial
        if not partial:
            return []

    # 2. Enumerate any rule variables (declared or used in action) still unbound,
    #    over their declared types.
    declared_vars: list[Var] = list(rule.vars)
    # variables that might still be free after the positive pass:
    free_vars = [v for v in declared_vars if v.name not in (partial[0] if partial else {})]
    # The above check is a heuristic over the first binding; we re-check per-theta
    # below because different positive matches may leave different vars free.
    completed: list[Binding] = []
    for theta in partial:
        unbound = [v for v in declared_vars if v.name not in theta]
        if not unbound:
            completed.append(theta)
            continue
        domains = []
        for v in unbound:
            objs = objects_by_type.get(v.type_name, ())
            domains.append(objs)
        if any(len(d) == 0 for d in domains):
            continue
        for tup in product(*domains):
            new_theta = dict(theta)
            for v, obj in zip(unbound, tup):
                new_theta[v.name] = obj
            completed.append(new_theta)

    # 3. Filter by negative literals (CWA).
    filtered: list[Binding] = []
    for theta in completed:
        if all(_eval_negative_literal(lit, theta, state_atoms, goal_atoms) for lit in negative):
            filtered.append(theta)

    # 4. Filter by legal_actions.
    legal: list[Binding] = []
    for theta in filtered:
        ground_args = tuple(theta[v.name] for v in rule.action.args)
        if GroundAction(rule.action.schema, ground_args) in legal_actions:
            legal.append(theta)

    # 5. Deterministic sort: lex-min over (var-name-sorted) -> value tuples.
    legal.sort(key=lambda th: tuple(sorted(th.items())))
    return legal


def interpret(
    policy: Policy,
    state_atoms: RelState,
    goal_atoms: RelState,
    objects_by_type: dict[str, tuple[str, ...]],
    legal_actions: set[GroundAction],
    *,
    trace: bool = False,
) -> Union[GroundAction, None, tuple[GroundAction, int, Binding]]:
    """First-applicable rule + lex-min binding → grounded action.

    If ``trace=True``, returns ``(action, rule_index, binding)`` so callers
    can inspect *which* rule fired and *which* binding selected the action.
    Returns ``None`` (or ``(None, ...)``-equivalent: just ``None``) when no
    rule applies.
    """
    for i, rule in enumerate(policy.rules):
        bindings = find_bindings(
            rule, state_atoms, goal_atoms, objects_by_type, legal_actions,
        )
        if not bindings:
            continue
        theta = bindings[0]
        ground_args = tuple(theta[v.name] for v in rule.action.args)
        action = GroundAction(rule.action.schema, ground_args)
        if trace:
            return action, i, theta
        return action
    return None
