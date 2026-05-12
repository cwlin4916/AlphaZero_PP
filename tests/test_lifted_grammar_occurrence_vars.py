"""Stage 2 — the occurrence-introduced-variable grammar redesign.

These tests pin the *structural* properties the redesign buys (orientation doc
``01_draft_lifted_policy_az.md`` §8 / ``docs/notes/stage4/02_plan.md``):
no ``aux_var`` phase; state literals may introduce a fresh body-local variable;
goal literals introduce none; the disconnected-goal-variable pathology is
unreachable by construction; the Gripper-lite goal-predicate whitelist still
holds; all four Stage-1 hand-policy rules remain expressible.
"""

from __future__ import annotations

import random

import pytest

from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.lifted_derivation import LiftedDerivationState
from alphazeropp.synthesis.lifted_grammar import (
    LiftedGrammarConfig,
    canonical_rule_form,
    enumerate_productions,
    gripper_lite_signature,
    rule_has_disconnected_goal_var,
    strict_grammar_config,
)

SIG = gripper_lite_signature()


def _walk_random(cfg, sig, rng, max_steps=200):
    s = LiftedDerivationState.initial()
    yield s
    steps = 0
    while not s.is_terminal() and steps < max_steps:
        prods = enumerate_productions(s, cfg, sig)
        assert prods
        s = s.apply(rng.choice(prods))
        steps += 1
        yield s
    assert s.is_terminal()


# ---------------------------------------------------------------------------

def test_no_aux_var_hole_reachable():
    """``current_hole`` is never ``"aux_var"`` along any derivation, and the
    encoding's hole table has no such id."""
    from alphazeropp.synthesis.lifted_encoding import _HOLE_ID
    assert "aux_var" not in _HOLE_ID
    cfg = strict_grammar_config(max_rules=3)
    rng = random.Random(0)
    for _ in range(80):
        for s in _walk_random(cfg, SIG, rng):
            assert s.current_hole in {"policy", "action_schema", "pre_lit", "goal_lit", None}


def test_no_aux_production_labels():
    """No production label ever mentions an auxiliary-variable phase."""
    cfg = strict_grammar_config(max_rules=3)
    rng = random.Random(1)
    banned = ("aux", "Aux", "add_aux", "SKIP_AUX")
    for _ in range(80):
        for s in _walk_random(cfg, SIG, rng):
            for p in enumerate_productions(s, cfg, SIG):
                assert not any(b in p.label for b in banned), p.label


def test_state_literal_can_introduce_fresh_body_local_var():
    """After ``schema=move`` (action args ``?r_0, ?r_1``), a ``pre_lit`` hole
    offers ``carrying(?v_0)`` with ``?v_0`` a *fresh* ball variable; applying it
    grows ``partial.body_local_vars`` by exactly that variable."""
    cfg = strict_grammar_config(max_rules=1)
    s = LiftedDerivationState.initial()
    s = s.apply(next(p for p in enumerate_productions(s, cfg, SIG) if p.label == "ADD_RULE"))
    s = s.apply(next(p for p in enumerate_productions(s, cfg, SIG) if p.label == "schema=move"))
    assert s.current_hole == "pre_lit"
    assert {v.name for v in s.partial.all_vars()} == {"?r_0", "?r_1"}
    carrying_prod = next(
        p for p in enumerate_productions(s, cfg, SIG)
        if p.payload[0] == "add" and p.payload[1].atom.pred == "carrying"
        and any(v.name == "?v_0" for v in p.payload[1].atom.variables())
    )
    new_vars = carrying_prod.payload[2]
    assert len(new_vars) == 1 and new_vars[0].name == "?v_0" and new_vars[0].type_name == "ball"
    s2 = s.apply(carrying_prod)
    assert {v.name for v in s2.partial.body_local_vars} == {"?v_0"}
    assert any(v.type_name == "ball" and v.name == "?v_0" for v in s2.partial.body_local_vars)


def test_goal_literal_cannot_introduce_fresh_var():
    """Every ``goal_lit`` production over every reachable state references only
    variables already in scope — its ``payload`` carries no new variables."""
    cfg = strict_grammar_config(max_rules=3)
    rng = random.Random(2)
    for _ in range(80):
        for s in _walk_random(cfg, SIG, rng):
            if s.partial is None:
                continue
            scope = {v.name for v in s.partial.all_vars()}
            for p in enumerate_productions(s, cfg, SIG):
                if p.hole_kind == "goal_lit" and p.payload[0] == "add":
                    # goal-lit add payloads are ("add", lit, ()) — no new vars
                    assert p.payload[2] == ()
                    for v in p.payload[1].atom.variables():
                        assert v.name in scope


def test_disconnected_goal_var_impossible_by_construction():
    """No grammar-produced rule has a goal-literal variable bound nowhere else."""
    cfg = strict_grammar_config(max_rules=4)
    rng = random.Random(3)
    for _ in range(400):
        last = None
        for s in _walk_random(cfg, SIG, rng):
            last = s
        for r in last.to_program().rules:
            assert rule_has_disconnected_goal_var(r) is False, r.pretty()


def test_vacuous_goal_predicate_not_offered_gripper():
    """Under the strict grammar + Gripper-lite signature, no ``goal_lit`` hole
    ever offers a ``carrying`` or ``handempty`` goal literal (whitelist =
    ``("at_ball",)``)."""
    cfg = strict_grammar_config(max_rules=3)
    rng = random.Random(4)
    for _ in range(60):
        for s in _walk_random(cfg, SIG, rng):
            for p in enumerate_productions(s, cfg, SIG):
                if p.hole_kind == "goal_lit" and p.payload[0] == "add":
                    assert p.payload[1].atom.pred == "at_ball", p.payload[1].pretty()


def test_all_four_hand_policy_rules_expressible():
    """Each Stage-1 hand-policy rule is reachable (up to α-renaming) under the
    strict default grammar."""
    from tests.test_lifted_grammar import _derive_one_rule  # reuse the deterministic driver
    cfg = LiftedGrammarConfig(max_rules=1)
    for rule in hand_policy().rules:
        produced = _derive_one_rule(rule, cfg, SIG)
        assert canonical_rule_form(produced) == canonical_rule_form(rule)


def test_terminal_policies_pass_post_init():
    """Every random complete derivation's rules were constructed without raising
    (``Rule.__post_init__`` validates typed vars + safe negation at build time)."""
    cfg = strict_grammar_config(max_rules=3)
    rng = random.Random(5)
    for _ in range(120):
        last = None
        for s in _walk_random(cfg, SIG, rng):
            last = s
        prog = last.to_program()  # would have raised already if a rule were ill-formed
        for r in prog.rules:
            # safe negation: every var in a negated literal is positively covered
            pos = {v.name for lit in r.body if not lit.negated for v in lit.atom.variables()}
            pos |= {v.name for v in r.action.args}
            for lit in r.body:
                if lit.negated:
                    assert all(v.name in pos for v in lit.atom.variables())


@pytest.mark.parametrize("max_body_local_vars", [0, 1, 2])
def test_body_local_var_cap_respected(max_body_local_vars):
    """No grammar-produced rule has more than ``max_body_local_vars`` body-local
    variables (a body-local variable = a declared var that is not an action arg)."""
    cfg = strict_grammar_config(max_rules=3, max_body_local_vars=max_body_local_vars)
    rng = random.Random(6)
    for _ in range(120):
        last = None
        for s in _walk_random(cfg, SIG, rng):
            last = s
        for r in last.to_program().rules:
            action_names = {v.name for v in r.action.args}
            n_body_local = len({v.name for v in r.vars if v.name not in action_names})
            assert n_body_local <= max_body_local_vars, r.pretty()
