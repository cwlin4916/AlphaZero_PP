"""Fixed-width observation encoding for partial lifted-policy ASTs.

Stage 2 only ever drives ``LiftedDerivationGame`` with uniform-prior MCTS
(``UniformPolicyValueNet`` ignores the observation), so this encoding only has
to be **valid and deterministic** — a stable preorder serialisation of the
partial AST into a fixed-length ``float32`` vector. A learned network is a
Stage 3 concern. See ``docs/notes/stage4/02_plan.md`` §2.3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alphazeropp.synthesis.lifted_derivation import LiftedDerivationState
    from alphazeropp.synthesis.lifted_grammar import DomainSignature, LiftedGrammarConfig


N_FIELDS_PER_NODE = 7  # (node_kind, predicate_id, action_id, type_id, var_local_id, source_id, negated_bit)

# node-kind ids (small integers; 0 is padding)
_KIND_RULE_HEADER = 20
_KIND_PARTIAL_HEADER = 21
_KIND_AUX_VAR = 22
_KIND_STATE_LIT = 10
_KIND_GOAL_LIT = 11
_KIND_HOLE_BASE = 30  # + hole id

_HOLE_ID = {None: 0, "policy": 1, "action_schema": 2, "aux_var": 3, "pre_lit": 4, "goal_lit": 5}


def encode_max_len(cfg: "LiftedGrammarConfig") -> int:
    """Worst-case encoded length. One node per finished literal/var + one per
    rule header + one trailing hole-marker node, all times ``N_FIELDS_PER_NODE``."""
    nodes_per_rule = 1 + cfg.max_pre_literals + cfg.max_goal_literals + cfg.max_aux_vars
    n_nodes = cfg.max_rules * nodes_per_rule + 1
    return N_FIELDS_PER_NODE * n_nodes


def encode_state(
    state: "LiftedDerivationState",
    sig: "DomainSignature",
    max_len: int,
) -> np.ndarray:
    pred_id = {p.name: i + 1 for i, p in enumerate(sig.predicates)}
    act_id = {a.name: i + 1 for i, a in enumerate(sig.action_schemas)}
    type_id = {t: i + 1 for i, t in enumerate(sig.types)}

    def lit_node(lit) -> tuple:
        kind = _KIND_GOAL_LIT if lit.source.value == "goal" else _KIND_STATE_LIT
        source_id = 2 if lit.source.value == "goal" else 1
        return (kind, pred_id.get(lit.atom.pred, 0), 0, 0,
                len(lit.atom.args), source_id, 1 if lit.negated else 0)

    items: list[tuple] = []
    for r in state.completed_rules:
        items.append((_KIND_RULE_HEADER, 0, act_id.get(r.action.schema, 0), 0,
                      len(r.action.args), 0, 0))
        for lit in r.body:
            items.append(lit_node(lit))
    if state.partial is not None:
        p = state.partial
        items.append((_KIND_PARTIAL_HEADER, 0,
                      act_id.get(p.schema, 0) if p.schema else 0, 0,
                      len(p.action_args), 0, 0))
        for lit in p.state_lits:
            items.append(lit_node(lit))
        for lit in p.goal_lits:
            items.append(lit_node(lit))
        for v in p.aux_vars:
            items.append((_KIND_AUX_VAR, 0, 0, type_id.get(v.type_name, 0), 0, 0, 0))
    items.append((_KIND_HOLE_BASE + _HOLE_ID.get(state.current_hole, 0), 0, 0, 0, 0, 0, 0))

    flat: list[float] = []
    for tup in items:
        flat.extend(tup)
    arr = np.zeros(max_len, dtype=np.float32)
    n = min(len(flat), max_len)
    if n:
        arr[:n] = np.asarray(flat[:n], dtype=np.float32)
    return arr
