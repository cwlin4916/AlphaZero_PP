"""Typed attributed grammar for lifted decision-list policies.

Stage 2 of the lifted-policy program-synthesis project — see
``docs/notes/stage4/02_plan.md``. This module defines the *grammar*: a
configuration, a domain signature (reusing ``lifted_dsl`` schemas), the
production type, and the per-hole production enumerator. The *derivation
state machine* that consumes these productions lives in ``lifted_derivation``.

Design points:

* Variables are named by **schema position**: ``move(?r_0, ?r_1)``,
  ``pick(?b_0, ?r_1)``, ``drop(?b_0, ?r_1)``; an auxiliary variable is
  ``?aux_0``. Two rules with the same shape therefore pretty-print
  identically — no separate α-canonicalisation pass at production time.
* Literal lists are kept canonical at production time: a literal hole only
  offers literals strictly greater than the last accepted one under a fixed
  lex key, so no duplicates and no order permutations are ever generated.
* Safe negation is enforced *before* a negated goal literal is offered: its
  variables must already appear in a positive (state) literal or in the
  action arguments, so every ``Rule`` produced passes ``Rule.__post_init__``.
* State literals are never negated in this stage (``allow_state_negation`` is
  ``False``); goal literals may be negated (``allow_goal_negation`` ``True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

from alphazeropp.synthesis.lifted_dsl import (
    ActionSchema,
    Atom,
    Literal,
    LiteralSource,
    PredicateSchema,
    Rule,
    Var,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alphazeropp.synthesis.lifted_derivation import LiftedDerivationState


# ---------------------------------------------------------------------------
# Configuration & domain signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedGrammarConfig:
    max_rules: int = 4
    max_pre_literals: int = 3
    max_goal_literals: int = 1
    max_aux_vars: int = 1
    allow_goal_negation: bool = True
    allow_state_negation: bool = False
    allow_disjunction: bool = False
    allow_constants: bool = False
    # --- Stage 2.5 grammar-safety ablations (default-off) ---
    goal_predicate_relevance: bool = False
    """If True, only predicate schemas listed in ``DomainSignature.goal_predicate_names``
    may appear inside a goal literal (positive or negated). A ``None`` whitelist on the
    signature makes this flag inert — currently behaviour is preserved."""

    require_goal_var_connected: bool = False
    """If True, a goal literal is only offered when every variable in it occurs either in
    the rule's action arguments or in at least one positive (state) precondition literal
    already added to the partial rule. This is the *local* connectedness rule (see Stage
    2.5 plan §B.2): the goal literal must not be the sole binding site of any variable.
    The literal transitive-closure-from-action-args reading from the draft would reject
    hand-policy ρ₂ (where ``?aux_0`` is bound only via ``carrying(?aux_0)``); that rule
    must remain expressible, so we use the local rule."""


@dataclass(frozen=True)
class DomainSignature:
    """A typed first-order signature. Reuses ``lifted_dsl`` schema classes.

    ``goal_predicate_names`` (Stage 2.5) is an optional whitelist of predicate names that
    are semantically meaningful inside a goal literal. ``None`` means "no whitelist", in
    which case ``LiftedGrammarConfig.goal_predicate_relevance=True`` is a no-op.
    """
    types: tuple[str, ...]
    predicates: tuple[PredicateSchema, ...]
    action_schemas: tuple[ActionSchema, ...]
    goal_predicate_names: tuple[str, ...] | None = None


def gripper_lite_signature() -> DomainSignature:
    """Built from the Gripper-lite env's existing schema constants.

    The Gripper-lite goal only ever uses ``at_ball`` atoms (see
    :meth:`alphazeropp.instances.gripper_lite.env.GripperLiteEnv._initial_goal`), so
    ``goal_predicate_names = ("at_ball",)`` — ``Goal[carrying(...)]`` and
    ``Goal[handempty()]`` are vacuous / spurious for this domain.
    """
    from alphazeropp.instances.gripper_lite.env import (
        ACTION_SCHEMAS,
        PREDICATE_SCHEMAS,
    )
    return DomainSignature(
        types=("ball", "room"),
        predicates=tuple(PREDICATE_SCHEMAS),
        action_schemas=tuple(ACTION_SCHEMAS),
        goal_predicate_names=("at_ball",),
    )


# ---------------------------------------------------------------------------
# Productions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedProduction:
    """One grammar production. ``payload`` is a tagged tuple consumed by
    ``LiftedDerivationState.apply`` — it is self-contained (no signature lookup
    needed at apply time)."""
    hole_kind: str   # "policy" | "action_schema" | "aux_var" | "pre_lit" | "goal_lit"
    label: str
    payload: tuple[Any, ...]


# -- naming helpers ---------------------------------------------------------

def _type_prefix(type_name: str) -> str:
    return type_name[0] if type_name else "x"


def action_vars_for_schema(schema: ActionSchema) -> tuple[Var, ...]:
    return tuple(Var(f"?{_type_prefix(t)}_{i}", t) for i, t in enumerate(schema.arg_types))


def aux_var_name(index: int) -> str:
    return f"?aux_{index}"


# -- literal enumeration ----------------------------------------------------

def _vars_of_type(scope: tuple[Var, ...], type_name: str) -> list[Var]:
    return [v for v in scope if v.type_name == type_name]


def _lit_key(lit: Literal) -> tuple:
    return (
        lit.atom.pred,
        tuple(a.name if isinstance(a, Var) else a for a in lit.atom.args),
        lit.negated,
    )


def literal_candidates(
    scope: tuple[Var, ...],
    sig: DomainSignature,
    kind: str,                       # "pre" (state) or "goal"
    *,
    last_lit: Literal | None,
    safe_var_names: frozenset[str] | set[str],
    cfg: LiftedGrammarConfig,
) -> list[Literal]:
    """All well-typed literals over ``scope`` that may legally extend the
    current literal list (strictly greater than ``last_lit`` under ``_lit_key``).

    No constants are introduced (``allow_constants`` is ``False``). State
    literals are never negated; goal literals may be negated only when every
    variable is in ``safe_var_names``.
    """
    source = LiteralSource.STATE if kind == "pre" else LiteralSource.GOAL
    out: list[Literal] = []
    for pred in sig.predicates:
        # Stage 2.5 — goal_predicate_relevance whitelist (no-op for state literals
        # and when the signature carries no whitelist).
        if (
            kind == "goal"
            and cfg.goal_predicate_relevance
            and sig.goal_predicate_names is not None
            and pred.name not in sig.goal_predicate_names
        ):
            continue
        choices = [_vars_of_type(scope, t) for t in pred.arg_types]
        if any(len(c) == 0 for c in choices):
            # some position has no compatible variable in scope (only possible
            # when arity > 0); skip this predicate
            continue
        for args in product(*choices):  # product() with no lists -> [()]
            atom = Atom(pred.name, tuple(args))
            negations = [False]
            if kind == "goal" and cfg.allow_goal_negation:
                if all(v.name in safe_var_names for v in atom.variables()):
                    negations.append(True)
            for neg in negations:
                lit = Literal(atom, negated=neg, source=source)
                if last_lit is not None and _lit_key(lit) <= _lit_key(last_lit):
                    continue
                out.append(lit)
    out.sort(key=_lit_key)
    return out


# -- the per-hole enumerator ------------------------------------------------

def enumerate_productions(
    state: "LiftedDerivationState",
    cfg: LiftedGrammarConfig,
    sig: DomainSignature,
) -> list[LiftedProduction]:
    h = state.current_hole
    if h is None:
        return []

    if h == "policy":
        prods: list[LiftedProduction] = []
        if len(state.completed_rules) < cfg.max_rules and len(sig.action_schemas) > 0:
            prods.append(LiftedProduction("policy", "ADD_RULE", ("add_rule",)))
        if len(state.completed_rules) >= 1:
            prods.append(LiftedProduction("policy", "STOP_POLICY", ("stop",)))
        return prods

    p = state.partial
    assert p is not None

    if h == "action_schema":
        return [
            LiftedProduction("action_schema", f"schema={s.name}",
                             ("schema", s.name, tuple(s.arg_types)))
            for s in sig.action_schemas
        ]

    if h == "aux_var":
        prods = [LiftedProduction("aux_var", "SKIP_AUX", ("skip_aux",))]
        if cfg.max_aux_vars >= 1 and len(p.aux_vars) < cfg.max_aux_vars:
            for t in sig.types:
                prods.append(LiftedProduction("aux_var", f"add_aux:{t}", ("add_aux", t)))
        return prods

    if h == "pre_lit":
        scope = p.all_vars()
        last = p.state_lits[-1] if p.state_lits else None
        prods = []
        if len(p.state_lits) < cfg.max_pre_literals:
            for lit in literal_candidates(scope, sig, "pre", last_lit=last,
                                          safe_var_names=set(), cfg=cfg):
                prods.append(LiftedProduction("pre_lit", f"pre:{lit.pretty()}", ("add", lit)))
        prods.append(LiftedProduction("pre_lit", "STOP_PRE", ("stop_pre",)))
        return prods

    if h == "goal_lit":
        scope = p.all_vars()
        last = p.goal_lits[-1] if p.goal_lits else None
        safe = p.positive_var_names()
        prods = []
        if len(p.goal_lits) < cfg.max_goal_literals:
            action_arg_names = {v.name for v in p.action_args}
            for lit in literal_candidates(scope, sig, "goal", last_lit=last,
                                          safe_var_names=safe, cfg=cfg):
                # Stage 2.5 — require every goal-literal variable to be bound somewhere
                # other than the goal literal itself (action args ∪ positive state lits).
                # See the `require_goal_var_connected` docstring for the local-rule rationale.
                if cfg.require_goal_var_connected and not goal_vars_locally_connected(
                    lit, action_arg_names=action_arg_names, state_lits=p.state_lits
                ):
                    continue
                prods.append(LiftedProduction("goal_lit", f"goal:{lit.pretty()}", ("add", lit)))
        prods.append(LiftedProduction("goal_lit", "FINISH_RULE", ("finish",)))
        return prods

    raise ValueError(f"unknown hole kind {h!r}")


# ---------------------------------------------------------------------------
# Action-space upper bound
# ---------------------------------------------------------------------------

def compute_max_productions(cfg: LiftedGrammarConfig, sig: DomainSignature) -> int:
    """A valid upper bound on ``len(enumerate_productions(state, cfg, sig))``
    over every reachable derivation state.

    The literal holes dominate. For each (action schema, aux-var choice) the
    maximal scope is taken and the literal-candidate count is computed with
    *nothing filtered* (``last_lit=None``) and *every* variable assumed safe
    (so negation is always allowed). Any reachable state has a real
    ``last_lit`` (only shrinks the set) and a real safe-set (subset), so this
    over-estimates. The fixed holes contribute small constants.
    """
    candidates = [
        2,                                                 # policy: ADD_RULE + STOP_POLICY
        max(1, len(sig.action_schemas)),                   # action_schema
        1 + (len(sig.types) if cfg.max_aux_vars >= 1 else 0),  # aux_var: SKIP + per-type
    ]
    aux_choices: list[str | None] = [None]
    if cfg.max_aux_vars >= 1:
        aux_choices += list(sig.types)
    pre_max = goal_max = 0
    for schema in sig.action_schemas:
        avars = action_vars_for_schema(schema)
        for ac in aux_choices:
            scope = avars + ((Var(aux_var_name(0), ac),) if ac is not None else ())
            safe = {v.name for v in scope}
            pre_c = literal_candidates(scope, sig, "pre", last_lit=None,
                                       safe_var_names=safe, cfg=cfg)
            goal_c = literal_candidates(scope, sig, "goal", last_lit=None,
                                        safe_var_names=safe, cfg=cfg)
            pre_max = max(pre_max, len(pre_c))
            goal_max = max(goal_max, len(goal_c))
    candidates.append(pre_max + 1)   # + STOP_PRE
    candidates.append(goal_max + 1)  # + FINISH_RULE
    return max(candidates)


# ---------------------------------------------------------------------------
# Stage 2.5 — pathology predicates & connectedness helper
# ---------------------------------------------------------------------------

def goal_predicate_is_relevant(pred_name: str, sig: DomainSignature) -> bool:
    """True iff ``pred_name`` appears in ``sig.goal_predicate_names``. When the
    signature has no whitelist (``goal_predicate_names is None``) every predicate
    is considered relevant — the constraint is then a no-op."""
    if sig.goal_predicate_names is None:
        return True
    return pred_name in sig.goal_predicate_names


def goal_literal_var_names(lit: Literal) -> set[str]:
    """The set of variable names appearing in the goal literal's atom."""
    return {v.name for v in lit.atom.variables()}


def goal_vars_locally_connected(
    goal_lit: Literal,
    *,
    action_arg_names: set[str],
    state_lits: tuple[Literal, ...],
) -> bool:
    """Stage 2.5 — local connectedness check.

    Returns ``True`` iff every variable in ``goal_lit`` occurs in
    ``action_arg_names`` or in at least one positive state literal of
    ``state_lits``. Equivalently: the goal literal does not introduce a
    variable that is bound nowhere else.

    This is the *local* rule used by ``require_goal_var_connected``. A literal
    transitive-closure-from-action-args reading rejects hand-policy ρ₂
    (``carrying(?aux_0) ∧ … ⇒ move(?r_0,?r_1)``), which Stage 2.5 explicitly
    requires to remain expressible (see the plan and ``02.md`` §H4)."""
    state_var_names: set[str] = set()
    for L in state_lits:
        for v in L.atom.variables():
            state_var_names.add(v.name)
    binding_sites = action_arg_names | state_var_names
    return all(name in binding_sites for name in goal_literal_var_names(goal_lit))


def rule_has_vacuous_goal_predicate(rule: Rule, sig: DomainSignature) -> bool:
    """True iff ``rule`` has at least one goal literal whose predicate is not in
    ``sig.goal_predicate_names`` (signature whitelist required). With no
    whitelist this returns ``False`` — there is no notion of "vacuous" then."""
    if sig.goal_predicate_names is None:
        return False
    for L in rule.body:
        if L.source is LiteralSource.GOAL and L.atom.pred not in sig.goal_predicate_names:
            return True
    return False


def rule_has_disconnected_goal_var(rule: Rule) -> bool:
    """True iff some variable inside a goal literal of ``rule`` occurs neither
    in the rule's action arguments nor in any positive state literal of the
    rule (i.e. is bound only via the goal literal)."""
    action_arg_names = {v.name for v in rule.action.args}
    state_lits = tuple(L for L in rule.body if L.source is LiteralSource.STATE)
    state_var_names: set[str] = set()
    for L in state_lits:
        for v in L.atom.variables():
            state_var_names.add(v.name)
    binding_sites = action_arg_names | state_var_names
    for L in rule.body:
        if L.source is not LiteralSource.GOAL:
            continue
        for v in L.atom.variables():
            if v.name not in binding_sites:
                return True
    return False


# ---------------------------------------------------------------------------
# α-canonical form (for tests / equivalence checks)
# ---------------------------------------------------------------------------

def canonical_rule_form(rule) -> tuple:
    """A representation of ``rule`` invariant under variable renaming and
    body-literal reordering. Two rules are α-equivalent iff their canonical
    forms are equal.

    Action-argument variables are renamed to their schema-position names
    (``?{prefix}_{i}``); any remaining (auxiliary) variable becomes
    ``?aux_0`` — Stage 2 caps ``max_aux_vars`` at 1, so there is at most one.
    """
    sigma: dict[str, str] = {}
    for i, v in enumerate(rule.action.args):
        sigma[v.name] = f"?{_type_prefix(v.type_name)}_{i}"
    remaining: list[str] = []
    for v in rule.vars:
        if v.name not in sigma and v.name not in remaining:
            remaining.append(v.name)
    for lit in rule.body:
        for a in lit.atom.args:
            if isinstance(a, Var) and a.name not in sigma and a.name not in remaining:
                remaining.append(a.name)
    for j, name in enumerate(remaining):
        sigma[name] = aux_var_name(j)

    def ren(a):
        return sigma[a.name] if isinstance(a, Var) else a

    canon_body = sorted(
        (lit.source.value, lit.negated, lit.atom.pred, tuple(ren(a) for a in lit.atom.args))
        for lit in rule.body
    )
    canon_action = (rule.action.schema, tuple(sigma[v.name] for v in rule.action.args))
    return (canon_action, tuple(canon_body))
