"""Typed attributed grammar for lifted decision-list policies.

Stage 2 of the lifted-policy program-synthesis project — see
``docs/notes/stage4/02_plan.md``. This module defines the *grammar*: a
configuration, a domain signature (reusing ``lifted_dsl`` schemas), the
production type, and the per-hole production enumerator. The *derivation
state machine* that consumes these productions lives in ``lifted_derivation``.

Design points (the **occurrence-introduced-variable** grammar — orientation
doc ``01_draft_lifted_policy_az.md`` §8):

* **No standalone ``Aux`` phase.** Variables enter scope only where they first
  *occur*. Action variables are introduced by choosing the action schema and
  are named by **schema position**: ``move(?r_0, ?r_1)``, ``pick(?b_0, ?r_1)``,
  ``drop(?b_0, ?r_1)``. A state (precondition) literal *may* introduce a fresh
  typed **body-local variable** ``?v_0, ?v_1, …`` in any argument position that
  uses it (capped by ``max_body_local_vars``). A goal literal introduces *no*
  variables — every variable in it must already be an action argument or a
  variable of a positive state literal already added to the rule. Hence
  goal-literal-variable connectedness is *structural*, not a filter.
* Two rules with the same shape pretty-print identically — no separate
  α-canonicalisation pass at production time (action vars by position; body-local
  vars by introduction order).
* Literal lists are kept canonical at production time: a literal hole only
  offers literals strictly greater than the last accepted one under a fixed
  lex key ``(predicate, args, negated)``, so no duplicates and no order
  permutations are ever generated; fresh body-local vars are introduced
  left-to-right within a literal, so the naming is deterministic.
* Safe negation is enforced *before* a negated goal literal is offered: its
  variables must already appear in a positive (state) literal or in the action
  arguments, so every ``Rule`` produced passes ``Rule.__post_init__``.
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
    max_body_local_vars: int = 2
    """Cap on the number of fresh body-local variables a single rule may
    introduce (via its state literals). The Gripper hand policy needs ≤ 1
    (ρ₂'s ``?b`` / ρ₄'s ``?b``); the default ``2`` is a generous margin that
    keeps :func:`compute_max_productions` closed-form and the observation
    encoding fixed-width. (Variables enter scope only at the literal occurrence
    that first uses them — there is no dedicated variable-introduction phase.)"""

    allow_goal_negation: bool = True
    allow_state_negation: bool = False
    allow_disjunction: bool = False
    allow_constants: bool = False
    # --- Grammar-safety constraints (Stage 2.5 introduced these as default-OFF
    # ablations; Stage 3-A promoted them to defaults — see docs/notes/stage4/legacy/03_plan.md).
    # Use ``legacy_grammar_config()`` to recover the pre-Stage-3-A permissive grammar. ---
    goal_predicate_relevance: bool = True
    """If True, only predicate schemas listed in ``DomainSignature.goal_predicate_names``
    may appear inside a goal literal (positive or negated). A ``None`` whitelist on the
    signature makes this flag inert. For Gripper-lite the whitelist is ``("at_ball",)``,
    so ``Goal[carrying(...)]`` / ``Goal[handempty()]`` candidates are never offered."""

    require_goal_var_connected: bool = True
    """**Vacuous-by-construction under the occurrence-introduced grammar** — goal
    literals introduce no variables (their arguments must already be action
    arguments or positive-state-literal variables), so every goal-literal
    variable is connected by construction. The flag is retained for config-shape
    stability and is a no-op; the old ``goal_vars_locally_connected`` filter is
    kept only for diagnostics on hand-written / legacy policies."""


def legacy_grammar_config(**overrides) -> LiftedGrammarConfig:
    """The pre-Stage-3-A *permissive* grammar: both safety flags off.

    ``**overrides`` are forwarded to :class:`LiftedGrammarConfig`
    (e.g. ``max_rules=3``). Note: under the occurrence-introduced grammar the
    ``require_goal_var_connected`` flag is already a no-op, so this differs from
    the strict config only by ``goal_predicate_relevance``."""
    overrides.setdefault("goal_predicate_relevance", False)
    overrides.setdefault("require_goal_var_connected", False)
    return LiftedGrammarConfig(**overrides)


def strict_grammar_config(**overrides) -> LiftedGrammarConfig:
    """The Stage-3-A default grammar (both safety flags on), named for call-site
    clarity. Equivalent to ``LiftedGrammarConfig(**overrides)`` with the current
    defaults; spell it out so the diff between strict / legacy call sites is obvious."""
    overrides.setdefault("goal_predicate_relevance", True)
    overrides.setdefault("require_goal_var_connected", True)
    return LiftedGrammarConfig(**overrides)


@dataclass(frozen=True)
class DomainSignature:
    """A typed first-order signature. Reuses ``lifted_dsl`` schema classes.

    ``goal_predicate_names`` is an optional whitelist of predicate names that
    are semantically meaningful inside a goal literal. ``None`` means "no
    whitelist", in which case ``LiftedGrammarConfig.goal_predicate_relevance=True``
    is a no-op.
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


def doors_pddl_signature() -> DomainSignature:
    """Built from the Doors relational adapter's schema constants (Stage 1 rev. 1).

    See :mod:`alphazeropp.instances.doors.doors_pddl_lifted`. The Doors goal is always a single
    goal location, so ``goal_predicate_names = ("at_loc",)``.
    """
    from alphazeropp.instances.doors.doors_pddl_lifted import (
        ACTION_SCHEMAS,
        PREDICATE_SCHEMAS,
    )
    return DomainSignature(
        types=("room", "location", "key"),
        predicates=tuple(PREDICATE_SCHEMAS),
        action_schemas=tuple(ACTION_SCHEMAS),
        goal_predicate_names=("at_loc",),
    )


# ---------------------------------------------------------------------------
# Productions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiftedProduction:
    """One grammar production. ``payload`` is a tagged tuple consumed by
    ``LiftedDerivationState.apply`` — it is self-contained (no signature lookup
    needed at apply time). For a state-literal ``add`` it is
    ``("add", literal, new_body_local_vars)``; for a goal-literal ``add`` it is
    ``("add", literal, ())`` (goal literals introduce no variables)."""
    hole_kind: str   # "policy" | "action_schema" | "pre_lit" | "goal_lit"
    label: str
    payload: tuple[Any, ...]


# -- naming helpers ---------------------------------------------------------

def _type_prefix(type_name: str) -> str:
    return type_name[0] if type_name else "x"


def action_vars_for_schema(schema: ActionSchema) -> tuple[Var, ...]:
    return tuple(Var(f"?{_type_prefix(t)}_{i}", t) for i, t in enumerate(schema.arg_types))


def body_local_var_name(index: int) -> str:
    """Name of the ``index``-th body-local variable of a rule (introduction
    order). The token used by :func:`canonical_rule_form` for any non-action
    variable, and by the live ``pre_lit`` enumerator for freshly-introduced
    variables."""
    return f"?v_{index}"


# -- literal enumeration ----------------------------------------------------

def _vars_of_type(scope: tuple[Var, ...], type_name: str) -> list[Var]:
    return [v for v in scope if v.type_name == type_name]


def _lit_key(lit: Literal) -> tuple:
    return (
        lit.atom.pred,
        tuple(a.name if isinstance(a, Var) else a for a in lit.atom.args),
        lit.negated,
    )


# sentinel used inside the per-position option lists for "introduce a fresh var here"
_FRESH = object()


def state_literal_candidates(
    scope: tuple[Var, ...],
    sig: DomainSignature,
    *,
    last_lit: Literal | None,
    n_body_local: int,
    cfg: LiftedGrammarConfig,
) -> list[tuple[Literal, tuple[Var, ...]]]:
    """All well-typed state (precondition) literals over ``scope`` that may
    legally extend the current precondition list — strictly greater than
    ``last_lit`` under ``_lit_key`` — paired with the tuple of fresh body-local
    variables each literal introduces.

    Each argument position may be filled by an existing in-scope variable of the
    required type *or* (when the rule has room left under
    ``cfg.max_body_local_vars``) a freshly-introduced typed variable. Fresh
    variables are named by introduction order — ``?v_{n_body_local}``,
    ``?v_{n_body_local+1}``, … left-to-right within the literal. State literals
    are never negated. No constants (``allow_constants`` is ``False``).
    """
    room_for_fresh = max(0, cfg.max_body_local_vars - n_body_local)
    out: list[tuple[Literal, tuple[Var, ...]]] = []
    for pred in sig.predicates:
        pos_options: list[list[Any]] = []
        for t in pred.arg_types:
            opts: list[Any] = list(_vars_of_type(scope, t))
            opts.append((_FRESH, t))
            pos_options.append(opts)
        for combo in product(*pos_options):  # product() over [] -> [()]
            n_fresh = sum(1 for c in combo if isinstance(c, tuple) and c[0] is _FRESH)
            if n_fresh > room_for_fresh:
                continue
            args: list[Var] = []
            new_vars: list[Var] = []
            idx = n_body_local
            for c in combo:
                if isinstance(c, tuple) and c[0] is _FRESH:
                    v = Var(body_local_var_name(idx), c[1])
                    args.append(v)
                    new_vars.append(v)
                    idx += 1
                else:
                    args.append(c)
            lit = Literal(Atom(pred.name, tuple(args)), negated=False,
                          source=LiteralSource.STATE)
            if last_lit is not None and _lit_key(lit) <= _lit_key(last_lit):
                continue
            out.append((lit, tuple(new_vars)))
    out.sort(key=lambda x: _lit_key(x[0]))
    return out


def goal_literal_candidates(
    scope: tuple[Var, ...],
    sig: DomainSignature,
    *,
    last_lit: Literal | None,
    safe_var_names: frozenset[str] | set[str],
    cfg: LiftedGrammarConfig,
) -> list[Literal]:
    """All well-typed goal literals over ``scope`` that may legally extend the
    current goal list — strictly greater than ``last_lit`` under ``_lit_key``.

    Goal literals introduce **no** variables: every argument must be a variable
    already in ``scope``. Predicates are restricted to
    ``sig.goal_predicate_names`` when ``cfg.goal_predicate_relevance`` (and the
    signature carries a whitelist). A negated ``Goal[ℓ]`` is offered only when
    every variable of ``ℓ`` is in ``safe_var_names`` (action arguments ∪
    positive-state-literal variables).
    """
    out: list[Literal] = []
    for pred in sig.predicates:
        if (
            cfg.goal_predicate_relevance
            and sig.goal_predicate_names is not None
            and pred.name not in sig.goal_predicate_names
        ):
            continue
        choices = [_vars_of_type(scope, t) for t in pred.arg_types]
        if any(len(c) == 0 for c in choices):
            continue
        for args in product(*choices):
            atom = Atom(pred.name, tuple(args))
            negations = [False]
            if cfg.allow_goal_negation and all(v.name in safe_var_names for v in atom.variables()):
                negations.append(True)
            for neg in negations:
                lit = Literal(atom, negated=neg, source=LiteralSource.GOAL)
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

    if h == "pre_lit":
        prods = []
        if len(p.state_lits) < cfg.max_pre_literals:
            scope = p.all_vars()
            last = p.state_lits[-1] if p.state_lits else None
            for lit, new_vars in state_literal_candidates(
                scope, sig, last_lit=last, n_body_local=len(p.body_local_vars), cfg=cfg
            ):
                prods.append(LiftedProduction("pre_lit", f"pre:{lit.pretty()}",
                                              ("add", lit, new_vars)))
        prods.append(LiftedProduction("pre_lit", "STOP_PRE", ("stop_pre",)))
        return prods

    if h == "goal_lit":
        prods = []
        if len(p.goal_lits) < cfg.max_goal_literals:
            scope = p.all_vars()
            last = p.goal_lits[-1] if p.goal_lits else None
            safe = p.positive_var_names()
            for lit in goal_literal_candidates(scope, sig, last_lit=last,
                                               safe_var_names=safe, cfg=cfg):
                prods.append(LiftedProduction("goal_lit", f"goal:{lit.pretty()}",
                                              ("add", lit, ())))
        prods.append(LiftedProduction("goal_lit", "FINISH_RULE", ("finish",)))
        return prods

    raise ValueError(f"unknown hole kind {h!r}")


# ---------------------------------------------------------------------------
# Action-space upper bound
# ---------------------------------------------------------------------------

def compute_max_productions(cfg: LiftedGrammarConfig, sig: DomainSignature) -> int:
    """A valid upper bound on ``len(enumerate_productions(state, cfg, sig))``
    over every reachable derivation state.

    The literal holes dominate. For each action schema the *maximal scope* is
    taken — its action variables plus ``max_body_local_vars`` synthetic
    variables of *each* type — and the literal-candidate count is computed with
    *nothing filtered* (``last_lit=None``), ``n_body_local=0`` (so the full
    ``max_body_local_vars`` worth of fresh slots is still available), and *every*
    variable assumed safe (so a negated goal literal is always allowed). Any
    reachable state has scope ⊆ that maximal scope (as a multiset of
    variables-per-type), at most as many fresh slots left, and a real
    ``last_lit`` (only shrinks the set) — so this over-estimates. The fixed
    holes contribute small constants.
    """
    candidates = [
        2,                                # policy: ADD_RULE + STOP_POLICY
        max(1, len(sig.action_schemas)),  # action_schema
    ]
    pre_max = goal_max = 0
    for schema in sig.action_schemas:
        avars = action_vars_for_schema(schema)
        synthetic = tuple(
            Var(f"?syn_{t}_{j}", t)
            for t in sig.types
            for j in range(cfg.max_body_local_vars)
        )
        scope = avars + synthetic
        safe = {v.name for v in scope}
        pre_c = state_literal_candidates(scope, sig, last_lit=None, n_body_local=0, cfg=cfg)
        goal_c = goal_literal_candidates(scope, sig, last_lit=None, safe_var_names=safe, cfg=cfg)
        pre_max = max(pre_max, len(pre_c))
        goal_max = max(goal_max, len(goal_c))
    candidates.append(pre_max + 1)   # + STOP_PRE
    candidates.append(goal_max + 1)  # + FINISH_RULE
    return max(candidates)


# ---------------------------------------------------------------------------
# Pathology predicates & connectedness helper (used by lifted_diagnostics on
# arbitrary hand-written / legacy policies; vacuous-by-construction for the
# occurrence-introduced grammar's own output).
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
    """Returns ``True`` iff every variable in ``goal_lit`` occurs in
    ``action_arg_names`` or in at least one positive state literal of
    ``state_lits``. Under the occurrence-introduced grammar this is always
    ``True`` for grammar-produced rules (goal literals only reference in-scope
    variables); kept for diagnostics on arbitrary policies."""
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
    rule (i.e. is bound only via the goal literal). Under the occurrence-
    introduced grammar this is ``False`` for every grammar-produced rule;
    kept for diagnostics on arbitrary policies."""
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
    (``?{prefix}_{i}``); any remaining (body-local) variable becomes ``?v_0``,
    ``?v_1``, … in first-occurrence order.
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
        sigma[name] = body_local_var_name(j)

    def ren(a):
        return sigma[a.name] if isinstance(a, Var) else a

    canon_body = sorted(
        (lit.source.value, lit.negated, lit.atom.pred, tuple(ren(a) for a in lit.atom.args))
        for lit in rule.body
    )
    canon_action = (rule.action.schema, tuple(sigma[v.name] for v in rule.action.args))
    return (canon_action, tuple(canon_body))
