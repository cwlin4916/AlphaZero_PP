"""Stage 2 tests for the lifted-policy grammar (occurrence-introduced variables).

See ``docs/notes/stage4/02_plan.md``. There is **no** ``aux_var`` phase: action
variables enter scope when the schema is chosen; body-local variables
(``?v_0, ?v_1, …``) are born inside the precondition literal that first uses
them; goal literals introduce none.
"""

from __future__ import annotations

import random

import pytest

from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.lifted_derivation import LiftedDerivationState
from alphazeropp.synthesis.lifted_dsl import Policy, Rule, Var
from alphazeropp.synthesis.lifted_grammar import (
    LiftedGrammarConfig,
    canonical_rule_form,
    compute_max_productions,
    enumerate_productions,
    gripper_lite_signature,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _only_prod(s, cfg, sig, label):
    for p in enumerate_productions(s, cfg, sig):
        if p.label == label:
            return p
    raise AssertionError(
        f"production {label!r} not available; have "
        f"{[p.label for p in enumerate_productions(s, cfg, sig)]}"
    )


def _find_lit_prod(s, cfg, sig, hole_kind, pred, args, neg, source):
    for p in enumerate_productions(s, cfg, sig):
        if p.hole_kind != hole_kind or p.payload[0] != "add":
            continue
        lit = p.payload[1]
        lit_args = tuple(a.name if isinstance(a, Var) else a for a in lit.atom.args)
        if (lit.atom.pred == pred and lit_args == args
                and lit.negated == neg and lit.source.value == source):
            return p
    raise AssertionError(
        f"no {hole_kind} production for {source} {'¬' if neg else ''}{pred}{args}; have "
        f"{[p.label for p in enumerate_productions(s, cfg, sig) if p.payload[0] == 'add']}"
    )


def _derive_one_rule(target: Rule, cfg, sig) -> Rule:
    """Drive the grammar deterministically to a rule α-equivalent to ``target``.

    Uses ``canonical_rule_form`` (action args by schema position, body-local
    vars ``?v_j`` by first occurrence) — the same naming the grammar emits — so
    the canonical body's literals can be matched to grammar productions by
    ``(predicate, argument-names, negated)``."""
    canon = canonical_rule_form(target)            # ((schema, action_arg_names), sorted_body)
    s = LiftedDerivationState.initial()
    s = s.apply(_only_prod(s, cfg, sig, "ADD_RULE"))
    s = s.apply(_only_prod(s, cfg, sig, f"schema={target.action.schema}"))
    for (_src, neg, pred, args) in [e for e in canon[1] if e[0] == "state"]:
        s = s.apply(_find_lit_prod(s, cfg, sig, "pre_lit", pred, args, neg, "state"))
    s = s.apply(_only_prod(s, cfg, sig, "STOP_PRE"))
    for (_src, neg, pred, args) in [e for e in canon[1] if e[0] == "goal"]:
        s = s.apply(_find_lit_prod(s, cfg, sig, "goal_lit", pred, args, neg, "goal"))
    s = s.apply(_only_prod(s, cfg, sig, "FINISH_RULE"))
    return s.completed_rules[-1]


def _drive_to_first_goal_hole(cfg, sig, *, schema: str, state_lit_specs):
    """Build a derivation up to the first ``goal_lit`` hole with the given action
    schema and (canonically-ordered) positive state literals ``[(pred, args), …]``."""
    s = LiftedDerivationState.initial()
    s = s.apply(_only_prod(s, cfg, sig, "ADD_RULE"))
    s = s.apply(_only_prod(s, cfg, sig, f"schema={schema}"))
    for pred, args in state_lit_specs:
        s = s.apply(_find_lit_prod(s, cfg, sig, "pre_lit", pred, args, False, "state"))
    s = s.apply(_only_prod(s, cfg, sig, "STOP_PRE"))
    assert s.current_hole == "goal_lit"
    return s


def _walk_random(cfg, sig, rng, max_steps=200):
    """Yield every state visited along one random complete derivation."""
    s = LiftedDerivationState.initial()
    yield s
    steps = 0
    while not s.is_terminal() and steps < max_steps:
        prods = enumerate_productions(s, cfg, sig)
        assert prods, f"non-terminal state with no productions: {s.pretty()}"
        s = s.apply(rng.choice(prods))
        steps += 1
        yield s
    assert s.is_terminal(), "random walk did not terminate"


# ---------------------------------------------------------------------------
# core grammar
# ---------------------------------------------------------------------------

def test_grammar_generates_well_typed_rules():
    """Every rule the grammar finishes must construct without error
    (``Rule.__post_init__`` runs at construction time)."""
    cfg = LiftedGrammarConfig(max_rules=2)
    sig = gripper_lite_signature()
    rng = random.Random(0)
    for _ in range(40):
        last = None
        for s in _walk_random(cfg, sig, rng):
            last = s
        prog = last.to_program()
        assert isinstance(prog, Policy)
        for r in prog.rules:
            assert isinstance(r, Rule)


def test_grammar_can_express_hand_policy_rules():
    """Each of the four Stage-1 hand-policy rules is reachable in the grammar
    (up to α-renaming) — including ρ₂/ρ₄ whose body-local ``?b`` is born inside
    a precondition literal, not in a separate phase."""
    cfg = LiftedGrammarConfig(max_rules=1)
    sig = gripper_lite_signature()
    for rule in hand_policy().rules:
        produced = _derive_one_rule(rule, cfg, sig)
        assert canonical_rule_form(produced) == canonical_rule_form(rule), (
            f"grammar produced {produced.pretty()!r}, "
            f"not α-equivalent to {rule.pretty()!r}"
        )


def test_literal_lists_are_canonical():
    """A literal hole offers literals in strictly increasing key order — no
    duplicates, no permutations, nothing <= the last accepted literal."""
    cfg = LiftedGrammarConfig(max_rules=1)
    sig = gripper_lite_signature()
    s = LiftedDerivationState.initial()
    s = s.apply(_only_prod(s, cfg, sig, "ADD_RULE"))
    s = s.apply(_only_prod(s, cfg, sig, "schema=pick"))    # action args ?b_0, ?r_1

    add0 = [p for p in enumerate_productions(s, cfg, sig) if p.payload[0] == "add"]
    keys0 = [(p.payload[1].atom.pred, tuple(a.name for a in p.payload[1].atom.args)) for p in add0]
    assert keys0 == sorted(keys0)
    assert len(keys0) == len(set(keys0))

    # adding the lex-max candidate leaves nothing further to add
    s_max = s.apply(add0[-1])
    add_after_max = [p for p in enumerate_productions(s_max, cfg, sig) if p.payload[0] == "add"]
    assert add_after_max == []
    assert any(p.label == "STOP_PRE" for p in enumerate_productions(s_max, cfg, sig))

    # adding a middle candidate forbids re-adding it or anything smaller
    s_mid = s.apply(add0[1])
    offered = {(p.payload[1].atom.pred, tuple(a.name for a in p.payload[1].atom.args))
               for p in enumerate_productions(s_mid, cfg, sig) if p.payload[0] == "add"}
    assert keys0[1] not in offered
    assert keys0[0] not in offered


def test_alpha_equivalent_rules_have_same_pretty():
    """Two α-equivalent rules canonicalise identically, and a grammar-produced
    rule canonicalises to the same form as the hand rule it represents."""
    rho3 = hand_policy().rules[2]               # at_ball ∧ at_robot ∧ handempty ∧ ¬Goal[at_ball] ⇒ pick
    ren = {"?b": "?x", "?r": "?y"}
    def rv(v: Var) -> Var:
        return Var(ren.get(v.name, v.name), v.type_name)
    from alphazeropp.synthesis.lifted_dsl import Atom, LiftedAction, Literal
    rho3_renamed = Rule(
        vars=tuple(rv(v) for v in rho3.vars),
        body=tuple(
            Literal(Atom(l.atom.pred, tuple(rv(a) if isinstance(a, Var) else a for a in l.atom.args)),
                    negated=l.negated, source=l.source)
            for l in rho3.body
        ),
        action=LiftedAction(rho3.action.schema, tuple(rv(v) for v in rho3.action.args)),
    )
    assert canonical_rule_form(rho3) == canonical_rule_form(rho3_renamed)

    cfg = LiftedGrammarConfig(max_rules=1)
    sig = gripper_lite_signature()
    produced = _derive_one_rule(rho3, cfg, sig)
    assert canonical_rule_form(produced) == canonical_rule_form(rho3)


def test_safe_negation_at_grammar_level():
    """A goal-literal hole offers a negated literal only when every variable in
    it already appears in a positive (state) literal or in the action args.

    Under the occurrence-introduced grammar a goal-literal variable is always
    either an action argument or a body-local variable born in a positive state
    literal — so safe negation is automatic — but the grammar must still never
    offer a negated literal over an un-covered variable, and it *does* offer a
    negated ``Goal[…]`` once the variable is positively bound."""
    sig = gripper_lite_signature()
    cfg = LiftedGrammarConfig(max_rules=3, goal_predicate_relevance=False)
    rng = random.Random(7)
    for _ in range(40):
        for s in _walk_random(cfg, sig, rng):
            if s.partial is None:
                continue
            covered = s.partial.positive_var_names()
            for p in enumerate_productions(s, cfg, sig):
                if p.hole_kind == "goal_lit" and p.payload[0] == "add" and p.payload[1].negated:
                    for v in p.payload[1].atom.variables():
                        assert v.name in covered, (
                            f"unsafe negated literal offered: {p.payload[1].pretty()}"
                        )
    # a negated Goal[at_ball(?v_0, ?r_1)] *is* offered once ?v_0 is bound by carrying.
    s = _drive_to_first_goal_hole(
        cfg, sig, schema="move",
        state_lit_specs=[("at_robot", ("?r_0",)), ("carrying", ("?v_0",))],
    )
    assert any(p.payload[0] == "add" and p.payload[1].negated
               and any(v.name == "?v_0" for v in p.payload[1].atom.variables())
               for p in enumerate_productions(s, cfg, sig))


@pytest.mark.parametrize("max_rules", [1, 3, 4])
@pytest.mark.parametrize("max_body_local_vars", [0, 1, 2])
def test_compute_max_productions_is_a_real_upper_bound(max_rules, max_body_local_vars):
    """The action-space size must dominate the production count at every
    reachable derivation state."""
    cfg = LiftedGrammarConfig(max_rules=max_rules, max_body_local_vars=max_body_local_vars)
    sig = gripper_lite_signature()
    bound = compute_max_productions(cfg, sig)
    assert bound >= 1
    rng = random.Random(1234)
    for _ in range(50):
        for s in _walk_random(cfg, sig, rng):
            assert len(enumerate_productions(s, cfg, sig)) <= bound


# ---------------------------------------------------------------------------
# grammar-safety constraints (goal_predicate_relevance + the structural
# connectedness guarantee that replaces require_goal_var_connected)
# ---------------------------------------------------------------------------

_CFG_PERMISSIVE = LiftedGrammarConfig(
    max_rules=1, goal_predicate_relevance=False, require_goal_var_connected=False,
)
_CFG_RELEVANCE = LiftedGrammarConfig(
    max_rules=1, goal_predicate_relevance=True, require_goal_var_connected=False,
)
_CFG_BOTH = LiftedGrammarConfig(
    max_rules=1, goal_predicate_relevance=True, require_goal_var_connected=True,
)


def test_current_grammar_still_expresses_hand_policy_rules():
    """The *current* default ``LiftedGrammarConfig`` (strict — both safety flags
    on; ``require_goal_var_connected`` now vacuous-by-construction) must still
    express every Stage-1 hand-policy rule."""
    sig = gripper_lite_signature()
    for rule in hand_policy().rules:
        produced = _derive_one_rule(rule, LiftedGrammarConfig(max_rules=1), sig)
        assert canonical_rule_form(produced) == canonical_rule_form(rule)


def test_hand_policy_still_expressible_after_goal_filters():
    """With *both* grammar-safety filters on (``_CFG_BOTH`` ≡ the strict default),
    each of the four Stage-1 hand-policy rules is still reachable up to α-renaming."""
    sig = gripper_lite_signature()
    for rule in hand_policy().rules:
        produced = _derive_one_rule(rule, _CFG_BOTH, sig)
        assert canonical_rule_form(produced) == canonical_rule_form(rule), (
            f"strict grammar produced {produced.pretty()!r}, "
            f"not α-equivalent to {rule.pretty()!r}"
        )


def test_goal_predicate_relevance_blocks_vacuous_goal_carrying():
    """With ``goal_predicate_relevance=True`` and the Gripper-lite whitelist
    ``("at_ball",)``, no production at a ``goal_lit`` hole offers a ``carrying``
    or ``handempty`` goal literal. Under the *permissive* grammar such
    productions do appear (subject to safe-negation)."""
    sig = gripper_lite_signature()
    s_permissive = _drive_to_first_goal_hole(
        _CFG_PERMISSIVE, sig, schema="drop", state_lit_specs=[("at_robot", ("?r_1",))],
    )
    preds_permissive = {p.payload[1].atom.pred
                        for p in enumerate_productions(s_permissive, _CFG_PERMISSIVE, sig)
                        if p.payload[0] == "add"}
    assert "carrying" in preds_permissive  # baseline: Goal[carrying(...)] is on offer

    s_relevance = _drive_to_first_goal_hole(
        _CFG_RELEVANCE, sig, schema="drop", state_lit_specs=[("at_robot", ("?r_1",))],
    )
    preds_filtered = {p.payload[1].atom.pred
                      for p in enumerate_productions(s_relevance, _CFG_RELEVANCE, sig)
                      if p.payload[0] == "add"}
    assert preds_filtered <= {"at_ball"}
    assert "carrying" not in preds_filtered
    assert "handempty" not in preds_filtered


def test_goal_literal_only_uses_in_scope_vars():
    """A ``goal_lit`` hole never introduces a fresh variable — every variable in
    every offered goal literal is already an action argument or a body-local
    variable of a positive state literal. (This is the structural replacement
    for ``require_goal_var_connected``.)"""
    sig = gripper_lite_signature()
    cfg = LiftedGrammarConfig(max_rules=3)
    rng = random.Random(99)
    for _ in range(60):
        for s in _walk_random(cfg, sig, rng):
            if s.partial is None:
                continue
            scope_names = {v.name for v in s.partial.all_vars()}
            for p in enumerate_productions(s, cfg, sig):
                if p.hole_kind == "goal_lit" and p.payload[0] == "add":
                    for v in p.payload[1].atom.variables():
                        assert v.name in scope_names, (
                            f"goal_lit production introduced a non-scope var: {p.payload[1].pretty()}"
                        )


def test_goal_literal_over_body_local_var_offered():
    """Hand-policy ρ₂ — ``carrying(?v_0) ∧ at_robot(?r_0) ∧ Goal[at_ball(?v_0, ?r_1)] ⇒
    move(?r_0,?r_1)`` — remains expressible: ``?v_0`` is born in the positive
    precondition ``carrying(?v_0)``, so the ``goal_lit`` hole offers
    ``Goal[at_ball(?v_0, ?r_1)]`` over it."""
    sig = gripper_lite_signature()
    s = _drive_to_first_goal_hole(
        _CFG_BOTH, sig, schema="move",
        state_lit_specs=[("at_robot", ("?r_0",)), ("carrying", ("?v_0",))],
    )
    offered = {
        (p.payload[1].atom.pred, tuple(a.name for a in p.payload[1].atom.args), p.payload[1].negated)
        for p in enumerate_productions(s, _CFG_BOTH, sig)
        if p.payload[0] == "add"
    }
    assert ("at_ball", ("?v_0", "?r_1"), False) in offered
    # and the full hand-policy ρ₂ is derivable under the strict grammar.
    rho2 = hand_policy().rules[1]
    produced = _derive_one_rule(rho2, _CFG_BOTH, sig)
    assert canonical_rule_form(produced) == canonical_rule_form(rho2)


def test_goal_literal_cannot_reference_unintroduced_var():
    """At a ``goal_lit`` hole for ``drop(?b_0,?r_1)`` with positive state
    literals ``at_robot(?r_1) ∧ carrying(?b_0)`` (both over action arguments — no
    body-local var introduced), the grammar offers ``Goal[at_ball(?b_0, ?r_1)]``
    but never ``Goal[at_ball(?v_0, ?r_1)]`` — no ``?v_0`` exists, and a goal
    literal cannot create one."""
    sig = gripper_lite_signature()
    s = _drive_to_first_goal_hole(
        LiftedGrammarConfig(max_rules=1), sig, schema="drop",
        state_lit_specs=[("at_robot", ("?r_1",)), ("carrying", ("?b_0",))],
    )
    offered = {
        (p.payload[1].atom.pred, tuple(a.name for a in p.payload[1].atom.args), p.payload[1].negated)
        for p in enumerate_productions(s, LiftedGrammarConfig(max_rules=1), sig)
        if p.payload[0] == "add"
    }
    assert ("at_ball", ("?b_0", "?r_1"), False) in offered
    assert all(va not in {"?v_0", "?v_1", "?aux_0"} for (_p, args, _n) in offered for va in args)


def test_spurious_runA_solver_no_longer_expressible():
    """The brittle "best" B=1 policy uniform-prior MCTS found in *legacy* Stage 2
    (``docs/notes/stage4/legacy/02.md`` — its solve leaned on a vacuous goal
    predicate and a disconnected auxiliary variable) is no longer generable:

      ρ₁ leaned on ``Goal[at_ball(?aux_0, ?r_1)]`` with ``?aux_0`` bound nowhere
         else — under the occurrence-introduced grammar a goal literal cannot
         introduce a variable, so this shape is unreachable by construction.
      ρ₂ leaned on ``Goal[carrying(?b_0)]`` — rejected by ``goal_predicate_relevance``.
      ρ₃ ``⊤ ⇒ move(?r_0, ?r_1)`` — still generable (a body-less action rule is
         legitimate; ρ₃ alone does not solve B=1).
    """
    from alphazeropp.synthesis.lifted_dsl import LiftedAction, goal_lit, state_lit

    cfg = LiftedGrammarConfig(max_rules=1)
    sig = gripper_lite_signature()
    b0, r1 = Var("?b_0", "ball"), Var("?r_1", "room")
    r0 = Var("?r_0", "room")
    aux0 = Var("?aux_0", "ball")

    rho1 = Rule(
        vars=(b0, r1, aux0),
        body=(state_lit("at_robot", r1), state_lit("carrying", b0), goal_lit("at_ball", aux0, r1)),
        action=LiftedAction("drop", (b0, r1)),
    )
    rho2 = Rule(
        vars=(b0, r1),
        body=(state_lit("at_robot", r1), goal_lit("carrying", b0, negated=True)),
        action=LiftedAction("pick", (b0, r1)),
    )
    rho3 = Rule(vars=(r0, r1), body=(), action=LiftedAction("move", (r0, r1)))

    with pytest.raises(AssertionError):  # goal literal cannot introduce a body-local var
        _derive_one_rule(rho1, cfg, sig)
    with pytest.raises(AssertionError):  # goal_predicate_relevance blocks Goal[carrying(?b_0)]
        _derive_one_rule(rho2, cfg, sig)
    produced = _derive_one_rule(rho3, cfg, sig)
    assert canonical_rule_form(produced) == canonical_rule_form(rho3)


@pytest.mark.parametrize("max_rules", [1, 3, 4])
@pytest.mark.parametrize(
    "flags",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["none", "relevance", "connected", "both"],
)
def test_compute_max_productions_is_valid_under_constraints(max_rules, flags):
    """``compute_max_productions`` must dominate the per-state production count
    at every reachable state, under every combination of the safety flags."""
    rel, conn = flags
    cfg = LiftedGrammarConfig(
        max_rules=max_rules, goal_predicate_relevance=rel, require_goal_var_connected=conn,
    )
    sig = gripper_lite_signature()
    bound = compute_max_productions(cfg, sig)
    assert bound >= 1
    rng = random.Random(20250511)
    for _ in range(40):
        for s in _walk_random(cfg, sig, rng):
            assert len(enumerate_productions(s, cfg, sig)) <= bound
