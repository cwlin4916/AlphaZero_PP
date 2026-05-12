"""Stage 2 tests for the lifted-policy grammar.

See ``docs/notes/stage4/02_plan.md`` §3.
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
    raise AssertionError(f"no {hole_kind} production for {source} {'¬' if neg else ''}{pred}{args}")


def _derive_one_rule(target: Rule, cfg, sig) -> Rule:
    """Drive the grammar deterministically to a rule α-equivalent to ``target``."""
    canon = canonical_rule_form(target)            # ((schema, action_arg_names), sorted_body)
    s = LiftedDerivationState.initial()
    s = s.apply(_only_prod(s, cfg, sig, "ADD_RULE"))
    s = s.apply(_only_prod(s, cfg, sig, f"schema={target.action.schema}"))

    action_names = {v.name for v in target.action.args}
    aux_vars = [v for v in target.vars if v.name not in action_names]
    if aux_vars:
        assert len(aux_vars) == 1, "Stage 2 caps max_aux_vars at 1"
        s = s.apply(_only_prod(s, cfg, sig, f"add_aux:{aux_vars[0].type_name}"))
    else:
        s = s.apply(_only_prod(s, cfg, sig, "SKIP_AUX"))

    for (_src, neg, pred, args) in [e for e in canon[1] if e[0] == "state"]:
        s = s.apply(_find_lit_prod(s, cfg, sig, "pre_lit", pred, args, neg, "state"))
    s = s.apply(_only_prod(s, cfg, sig, "STOP_PRE"))

    for (_src, neg, pred, args) in [e for e in canon[1] if e[0] == "goal"]:
        s = s.apply(_find_lit_prod(s, cfg, sig, "goal_lit", pred, args, neg, "goal"))
    s = s.apply(_only_prod(s, cfg, sig, "FINISH_RULE"))
    return s.completed_rules[-1]


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
# tests
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
    (up to α-renaming)."""
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
    s = s.apply(_only_prod(s, cfg, sig, "SKIP_AUX"))

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
    # rename ?b -> ?x, ?r -> ?y
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


@pytest.mark.parametrize("max_rules", [1, 3, 4])
def test_compute_max_productions_is_a_real_upper_bound(max_rules):
    """The action-space size must dominate the production count at every
    reachable derivation state."""
    cfg = LiftedGrammarConfig(max_rules=max_rules)
    sig = gripper_lite_signature()
    bound = compute_max_productions(cfg, sig)
    assert bound >= 1
    rng = random.Random(1234)
    for _ in range(50):
        for s in _walk_random(cfg, sig, rng):
            assert len(enumerate_productions(s, cfg, sig)) <= bound


def test_safe_negation_at_grammar_level():
    """A goal-literal hole offers a negated literal only when every variable in
    it already appears in a positive (state) literal or in the action args."""
    cfg = LiftedGrammarConfig(max_rules=1)
    sig = gripper_lite_signature()

    # move + aux:ball + no state literals -> ?aux_0 is unbound, hence unsafe
    s = LiftedDerivationState.initial()
    s = s.apply(_only_prod(s, cfg, sig, "ADD_RULE"))
    s = s.apply(_only_prod(s, cfg, sig, "schema=move"))
    s = s.apply(_only_prod(s, cfg, sig, "add_aux:ball"))
    s = s.apply(_only_prod(s, cfg, sig, "STOP_PRE"))
    add_prods = [p for p in enumerate_productions(s, cfg, sig) if p.payload[0] == "add"]
    for p in add_prods:
        lit = p.payload[1]
        if lit.negated:
            assert "?aux_0" not in {v.name for v in lit.atom.variables()}, (
                f"unsafe negated literal offered: {lit.pretty()}"
            )
    # a positive goal literal over ?aux_0 is still offered
    assert any((not p.payload[1].negated)
               and any(v.name == "?aux_0" for v in p.payload[1].atom.variables())
               for p in add_prods)

    # bind ?aux_0 with a positive state literal -> negated literals over it appear
    s2 = LiftedDerivationState.initial()
    s2 = s2.apply(_only_prod(s2, cfg, sig, "ADD_RULE"))
    s2 = s2.apply(_only_prod(s2, cfg, sig, "schema=move"))
    s2 = s2.apply(_only_prod(s2, cfg, sig, "add_aux:ball"))
    s2 = s2.apply(_find_lit_prod(s2, cfg, sig, "pre_lit", "carrying", ("?aux_0",), False, "state"))
    s2 = s2.apply(_only_prod(s2, cfg, sig, "STOP_PRE"))
    assert any(p.payload[0] == "add" and p.payload[1].negated
               and any(v.name == "?aux_0" for v in p.payload[1].atom.variables())
               for p in enumerate_productions(s2, cfg, sig))


# ---------------------------------------------------------------------------
# Stage 2.5 — grammar-safety ablations
# ---------------------------------------------------------------------------

# Constraint preset cfgs (max_rules=1 keeps test derivations short).
_CFG_DEFAULT = LiftedGrammarConfig(max_rules=1)
_CFG_RELEVANCE = LiftedGrammarConfig(max_rules=1, goal_predicate_relevance=True)
_CFG_CONNECTED = LiftedGrammarConfig(max_rules=1, require_goal_var_connected=True)
_CFG_BOTH = LiftedGrammarConfig(
    max_rules=1, goal_predicate_relevance=True, require_goal_var_connected=True,
)


def test_current_grammar_still_expresses_hand_policy_rules():
    """Default ``LiftedGrammarConfig`` (Stage-2.5 flags off) must still express
    every Stage-1 hand-policy rule — the new fields must be backwards-compat."""
    sig = gripper_lite_signature()
    for rule in hand_policy().rules:
        produced = _derive_one_rule(rule, _CFG_DEFAULT, sig)
        assert canonical_rule_form(produced) == canonical_rule_form(rule)


def _drive_to_first_goal_hole(cfg, sig, *, schema: str, aux_type: str | None,
                              state_lit_specs):
    """Build a derivation up to the first ``goal_lit`` hole with the given
    action schema, aux var, and (ordered) positive state literals."""
    s = LiftedDerivationState.initial()
    s = s.apply(_only_prod(s, cfg, sig, "ADD_RULE"))
    s = s.apply(_only_prod(s, cfg, sig, f"schema={schema}"))
    if aux_type is None:
        s = s.apply(_only_prod(s, cfg, sig, "SKIP_AUX"))
    else:
        s = s.apply(_only_prod(s, cfg, sig, f"add_aux:{aux_type}"))
    for pred, args in state_lit_specs:
        s = s.apply(_find_lit_prod(s, cfg, sig, "pre_lit", pred, args, False, "state"))
    s = s.apply(_only_prod(s, cfg, sig, "STOP_PRE"))
    assert s.current_hole == "goal_lit"
    return s


def test_goal_predicate_relevance_blocks_vacuous_goal_carrying():
    """With ``goal_predicate_relevance=True`` and the Gripper-lite whitelist
    ``("at_ball",)``, no production at a ``goal_lit`` hole offers a
    ``carrying`` or ``handempty`` goal literal. With it ``False`` such
    productions do appear (subject to safe-negation)."""
    sig = gripper_lite_signature()
    # drive to a goal_lit hole that, under default cfg, offers a Goal[carrying(?b_0)]
    # (positive — safe since ?b_0 is an action arg of drop).
    s_default = _drive_to_first_goal_hole(
        _CFG_DEFAULT, sig, schema="drop", aux_type=None,
        state_lit_specs=[("at_robot", ("?r_1",))],
    )
    preds_default = {p.payload[1].atom.pred
                     for p in enumerate_productions(s_default, _CFG_DEFAULT, sig)
                     if p.payload[0] == "add"}
    assert "carrying" in preds_default  # baseline: Goal[carrying(...)] is on offer

    s_relevance = _drive_to_first_goal_hole(
        _CFG_RELEVANCE, sig, schema="drop", aux_type=None,
        state_lit_specs=[("at_robot", ("?r_1",))],
    )
    preds_filtered = {p.payload[1].atom.pred
                      for p in enumerate_productions(s_relevance, _CFG_RELEVANCE, sig)
                      if p.payload[0] == "add"}
    assert preds_filtered <= {"at_ball"}
    assert "carrying" not in preds_filtered
    assert "handempty" not in preds_filtered


def test_connectedness_blocks_free_aux_goal_literal():
    """With ``require_goal_var_connected=True`` and a partial rule
    ``drop(?b_0,?r_1)`` + aux ``?aux_0:ball`` + state lit ``at_robot(?r_1)``,
    the spurious ``Goal[at_ball(?aux_0, ?r_1)]`` is not on offer (``?aux_0``
    occurs nowhere outside the goal literal). With it ``False`` it is."""
    sig = gripper_lite_signature()
    state_lit_specs = [("at_robot", ("?r_1",))]

    def offered_goal_atoms(cfg):
        s = _drive_to_first_goal_hole(
            cfg, sig, schema="drop", aux_type="ball", state_lit_specs=state_lit_specs,
        )
        return [
            (p.payload[1].atom.pred,
             tuple(a.name for a in p.payload[1].atom.args),
             p.payload[1].negated)
            for p in enumerate_productions(s, cfg, sig)
            if p.payload[0] == "add"
        ]

    bad = ("at_ball", ("?aux_0", "?r_1"), False)
    assert bad in offered_goal_atoms(_CFG_DEFAULT)
    assert bad not in offered_goal_atoms(_CFG_CONNECTED)


def test_connectedness_allows_hand_policy_move_toward_goal():
    """Hand-policy ρ₂ — ``carrying(?aux_0) ∧ at_robot(?r_0) ∧
    Goal[at_ball(?aux_0, ?r_1)] ⇒ move(?r_0,?r_1)`` — must remain expressible
    under ``require_goal_var_connected=True``: ``?aux_0`` is bound by the
    ``carrying(?aux_0)`` precond (a positive state literal), so the goal lit
    is locally connected."""
    sig = gripper_lite_signature()
    # the grammar lists state literals in canonical lex order: at_robot < carrying.
    state_lit_specs = [("at_robot", ("?r_0",)), ("carrying", ("?aux_0",))]
    s = _drive_to_first_goal_hole(
        _CFG_CONNECTED, sig, schema="move", aux_type="ball",
        state_lit_specs=state_lit_specs,
    )
    offered = [
        (p.payload[1].atom.pred,
         tuple(a.name for a in p.payload[1].atom.args),
         p.payload[1].negated)
        for p in enumerate_productions(s, _CFG_CONNECTED, sig)
        if p.payload[0] == "add"
    ]
    assert ("at_ball", ("?aux_0", "?r_1"), False) in offered

    # cross-check: full hand-policy ρ₂ is still derivable under (T,T).
    rho2 = hand_policy().rules[1]
    produced = _derive_one_rule(rho2, _CFG_BOTH, sig)
    assert canonical_rule_form(produced) == canonical_rule_form(rho2)


def test_safe_negation_still_enforced():
    """Even with both Stage-2.5 flags on, negated goal literals are offered
    only when every variable is positively bound (or an action arg). The new
    filters must not weaken safe negation."""
    sig = gripper_lite_signature()
    # move + aux:ball + NO state lits: ?aux_0 is unbound → no negated lit over it.
    s = _drive_to_first_goal_hole(
        _CFG_BOTH, sig, schema="move", aux_type="ball", state_lit_specs=[],
    )
    for p in enumerate_productions(s, _CFG_BOTH, sig):
        if p.payload[0] != "add":
            continue
        lit = p.payload[1]
        if lit.negated:
            assert "?aux_0" not in {v.name for v in lit.atom.variables()}, (
                f"unsafe negated literal offered: {lit.pretty()}"
            )


@pytest.mark.parametrize("max_rules", [1, 3, 4])
@pytest.mark.parametrize(
    "flags",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
    ids=["none", "relevance", "connected", "both"],
)
def test_compute_max_productions_is_valid_under_constraints(max_rules, flags):
    """``compute_max_productions`` must dominate the per-state production count
    at every reachable state, under every combination of the Stage-2.5 flags."""
    rel, conn = flags
    cfg = LiftedGrammarConfig(
        max_rules=max_rules,
        goal_predicate_relevance=rel,
        require_goal_var_connected=conn,
    )
    sig = gripper_lite_signature()
    bound = compute_max_productions(cfg, sig)
    assert bound >= 1
    rng = random.Random(20250511)
    for _ in range(40):
        for s in _walk_random(cfg, sig, rng):
            assert len(enumerate_productions(s, cfg, sig)) <= bound
