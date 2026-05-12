"""Stage 3-A — pure per-policy *grammar-pathology* report for lifted decision-list policies.

A "pathology" here is a structural feature of a complete :class:`~alphazeropp.synthesis.lifted_dsl.Policy`
that the Stage-2.5 landscape study (``docs/notes/stage4/02.md`` §Interpretation) and the Stage-3-A grammar
filters (``docs/notes/stage4/03_plan.md``) flagged as a spurious-solver enabler on Gripper-lite:

* **vacuous goal predicate** — a goal literal whose predicate is not in ``sig.goal_predicate_names``
  (e.g. ``Goal[carrying(...)]`` / ``Goal[handempty()]`` for Gripper-lite, whose goals only ever use
  ``at_ball``). Excluded by construction once ``LiftedGrammarConfig.goal_predicate_relevance`` is on
  (the Stage-3-A default).
* **goal-only variable** — a variable that occurs *only* inside a goal literal of its rule (bound by no
  action argument and no positive state literal). Excluded by construction once
  ``LiftedGrammarConfig.require_goal_var_connected`` is on (the Stage-3-A default).
* **⊤-headed action rule** — a rule with an empty body (``⊤ ⇒ a``). The grammar legitimately allows these
  (a body-less ``move`` can be a sensible default), so this is *reported*, not forbidden; the ``⊤ ⇒ drop``
  variant is the "do-nothing" attractor of 02.md.

This module is pure and side-effect-free; the diagnostic-grid driver and the smoke script attach
:func:`analyze_policy_pathologies` to every evaluated policy's JSONL record.
"""

from __future__ import annotations

from alphazeropp.synthesis.lifted_dsl import LiteralSource, Policy
from alphazeropp.synthesis.lifted_grammar import (
    DomainSignature,
    rule_has_disconnected_goal_var,
)


def analyze_policy_pathologies(
    policy: Policy,
    *,
    goal_predicate_names: tuple[str, ...] | None,
) -> dict[str, bool | int]:
    """Structural pathology flags + counts for one complete lifted policy.

    Pure. ``goal_predicate_names is None`` ⇒ there is no notion of a "vacuous goal predicate" for the
    domain, so ``has_vacuous_goal_predicate`` is always ``False`` (everything is considered relevant).
    Typically pass ``gripper_lite_signature().goal_predicate_names`` (see
    :func:`analyze_policy_pathologies_for`).

    Returns a dict with exactly these keys::

        has_vacuous_goal_predicate : bool   # ∃ goal literal whose predicate ∉ goal_predicate_names
        has_goal_only_variable     : bool   # ∃ rule with a variable bound only inside its goal literal
        has_empty_body_rule        : bool   # ∃ rule with an empty body (⊤ ⇒ a)
        has_top_drop_rule          : bool   # ∃ empty-body rule whose action schema is "drop"
        has_top_pick_rule          : bool   # ∃ empty-body rule whose action schema is "pick"
        has_top_move_rule          : bool   # ∃ empty-body rule whose action schema is "move"
        num_goal_literals          : int    # total goal literals across all rules
        num_negative_goal_literals : int    # of those, how many are negated
        num_rules                  : int
        num_body_literals          : int    # total body literals (state + goal) across all rules
    """
    rules = policy.rules
    goal_lits = [L for r in rules for L in r.body if L.source is LiteralSource.GOAL]
    vacuous = goal_predicate_names is not None and any(
        L.atom.pred not in goal_predicate_names for L in goal_lits
    )
    empty_body_rules = [r for r in rules if len(r.body) == 0]
    empty_body_schemas = {r.action.schema for r in empty_body_rules}
    return {
        "has_vacuous_goal_predicate": bool(vacuous),
        "has_goal_only_variable": any(rule_has_disconnected_goal_var(r) for r in rules),
        "has_empty_body_rule": bool(empty_body_rules),
        "has_top_drop_rule": "drop" in empty_body_schemas,
        "has_top_pick_rule": "pick" in empty_body_schemas,
        "has_top_move_rule": "move" in empty_body_schemas,
        "num_goal_literals": len(goal_lits),
        "num_negative_goal_literals": sum(1 for L in goal_lits if L.negated),
        "num_rules": len(rules),
        "num_body_literals": sum(len(r.body) for r in rules),
    }


def analyze_policy_pathologies_for(policy: Policy, sig: DomainSignature) -> dict[str, bool | int]:
    """:func:`analyze_policy_pathologies` with the goal-predicate whitelist taken from ``sig``."""
    return analyze_policy_pathologies(policy, goal_predicate_names=sig.goal_predicate_names)
