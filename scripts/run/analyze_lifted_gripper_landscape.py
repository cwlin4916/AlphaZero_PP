#!/usr/bin/env python3
"""Stage 2.5 — enumerate / sample the lifted-policy program space for
Gripper-lite and log per-policy leaf metrics, grammar-pathology flags, and
score-variant rankings.

The Stage-2 figures showed uniform-prior MCTS lives in a deceptive sparse-
reward landscape; this driver quantifies that landscape directly. For one or
more grammar-safety constraint settings, the script either *exhaustively*
enumerates every complete lifted policy (when the count fits within
``--max-policies``) or falls back to *reproducible stratified random
sampling* over derivation depths and policy sizes.

For every policy it logs:

  policy_pretty                — the human-readable rendering (cache key)
  num_rules / num_state_literals / num_goal_literals
  has_vacuous_goal_predicate   — uses a Goal[pred(...)] with pred ∉ sig.goal_predicate_names
  has_disconnected_goal_var    — a goal-literal variable not bound in action / pre lits
  train_solve_rate             — fraction of --train-balls instances solved
  eval_solve_rate              — fraction of --eval-balls instances solved
  progress / steps / noops     — train-instance averages from LiftedLeafEvaluator
  current_leaf_score           — the Stage-2 MCTS reward
  score_no_step_penalty / score_no_noop_penalty / score_progress_only
  score_lexicographic_tuple    — ranking-only tuple
  solves_B1..B4                — single-instance solve booleans

The summary JSON adds total/unique policy counts, per-B solver counts,
B1↔B2 generalisation counts, hand-policy and degenerate-drop-policy
score+rank, vacuous-goal / disconnected-goal counts, and the best policy
under each score variant.

Run from repo root:

    python scripts/run/analyze_lifted_gripper_landscape.py \\
        --max-rules 1 --max-pre-literals 2 \\
        --constraints none both \\
        --max-policies 200000 \\
        --output-json /tmp/landscape_r1.json --output-csv /tmp/landscape_r1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import (
    degenerate_drop_policy,
    hand_policy,
)
from alphazeropp.synthesis.lifted_derivation import LiftedDerivationState
from alphazeropp.synthesis.lifted_dsl import Policy
from alphazeropp.synthesis.lifted_grammar import (
    DomainSignature,
    LiftedGrammarConfig,
    enumerate_productions,
    gripper_lite_signature,
    rule_has_disconnected_goal_var,
    rule_has_vacuous_goal_predicate,
)
from alphazeropp.synthesis.lifted_interpreter import interpret
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator
from alphazeropp.synthesis.lifted_score_variants import (
    SCORE_VARIANT_NAMES,
    all_score_variants,
    score_lexicographic,
)

# ---------------------------------------------------------------------------
# constraint presets
# ---------------------------------------------------------------------------

CONSTRAINT_CHOICES = ("none", "goal_predicate", "connected", "both")


def _cfg_for(constraint: str, *, max_rules: int, max_pre_literals: int,
             max_goal_literals: int, max_aux_vars: int) -> LiftedGrammarConfig:
    rel = constraint in {"goal_predicate", "both"}
    conn = constraint in {"connected", "both"}
    return LiftedGrammarConfig(
        max_rules=max_rules,
        max_pre_literals=max_pre_literals,
        max_goal_literals=max_goal_literals,
        max_aux_vars=max_aux_vars,
        goal_predicate_relevance=rel,
        require_goal_var_connected=conn,
    )


# ---------------------------------------------------------------------------
# enumeration / sampling
# ---------------------------------------------------------------------------

@dataclass
class _Stratum:
    num_rules: int
    num_body_literals: int


def _enumerate_exhaustive(
    cfg: LiftedGrammarConfig, sig: DomainSignature, *, max_policies: int
) -> tuple[list[Policy], dict[str, int], bool]:
    """DFS the derivation tree, collecting every terminal ``Policy``. If the
    distinct-policy count crosses ``max_policies`` (or a hard node budget),
    abort and signal the caller to fall back to sampling. Dedupe by
    ``Policy.pretty()``."""
    seen: dict[str, Policy] = {}
    strata: dict[str, int] = {}
    stack: list[LiftedDerivationState] = [LiftedDerivationState.initial()]
    # Node budget keeps "infeasible" enumerations from runaway. 10× max_policies
    # is generous — at typical grammars the DFS expands few internal nodes per
    # terminal.
    node_budget = max(100_000, 25 * max_policies)
    nodes_visited = 0
    while stack:
        s = stack.pop()
        nodes_visited += 1
        if nodes_visited > node_budget:
            return [], {}, False
        if s.is_terminal():
            policy = s.to_program()
            key = policy.pretty()
            if key not in seen:
                seen[key] = policy
                stratum_key = f"{len(policy.rules)}r_{sum(len(r.body) for r in policy.rules)}lit"
                strata[stratum_key] = strata.get(stratum_key, 0) + 1
                if len(seen) > max_policies:
                    return [], {}, False
            continue
        prods = enumerate_productions(s, cfg, sig)
        # Reverse for left-to-right DFS order (cosmetic — output is set-keyed).
        for prod in reversed(prods):
            stack.append(s.apply(prod))
    return list(seen.values()), strata, True


def _sample_stratified(
    cfg: LiftedGrammarConfig, sig: DomainSignature, *,
    max_policies: int, seed: int,
) -> tuple[list[Policy], dict[str, int]]:
    """Reproducible random walks with stratum tracking. Uniform legal
    production at each hole. Dedupe by ``Policy.pretty()``. Stops at
    ``max_policies`` distinct policies or after ``50 * max_policies`` walks
    (whichever comes first)."""
    rng = random.Random(seed)
    seen: dict[str, Policy] = {}
    strata: dict[str, int] = {}
    walk_cap = max(1000, 50 * max_policies)
    walks = 0
    while len(seen) < max_policies and walks < walk_cap:
        walks += 1
        s = LiftedDerivationState.initial()
        while not s.is_terminal():
            prods = enumerate_productions(s, cfg, sig)
            if not prods:
                break
            s = s.apply(rng.choice(prods))
        if not s.is_terminal():
            continue
        policy = s.to_program()
        key = policy.pretty()
        if key in seen:
            continue
        seen[key] = policy
        stratum_key = f"{len(policy.rules)}r_{sum(len(r.body) for r in policy.rules)}lit"
        strata[stratum_key] = strata.get(stratum_key, 0) + 1
    return list(seen.values()), strata


def _enumerate_or_sample(
    cfg: LiftedGrammarConfig, sig: DomainSignature, *,
    max_policies: int, seed: int,
) -> tuple[str, list[Policy], dict[str, int]]:
    policies, strata, ok = _enumerate_exhaustive(cfg, sig, max_policies=max_policies)
    if ok:
        return "exhaustive", policies, strata
    sampled, strata = _sample_stratified(cfg, sig, max_policies=max_policies, seed=seed)
    return "sampled", sampled, strata


# ---------------------------------------------------------------------------
# per-policy evaluation
# ---------------------------------------------------------------------------

def _solves(policy: Policy, env: GripperLiteEnv) -> bool:
    """Single-instance solve check by running ``interpret`` to horizon."""
    env.reset()
    h = env.horizon
    for _ in range(h):
        if env.is_solved():
            return True
        action = interpret(
            policy,
            env.get_state_atoms(),
            env.get_goal_atoms(),
            env.get_objects_by_type(),
            env.legal_actions(),
        )
        if action is None:
            break
        env.step(action)
    return env.is_solved()


def _per_b_solve_map(
    policy: Policy, *, b_range: tuple[int, ...] = (1, 2, 3, 4)
) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for b in b_range:
        env = GripperLiteEnv(n_balls=b, seed=0)
        out[b] = _solves(policy, env)
    return out


def _policy_row(
    policy: Policy, sig: DomainSignature, *,
    constraint: str, mode: str,
    train_balls: tuple[int, ...], eval_balls: tuple[int, ...],
    evaluator: LiftedLeafEvaluator,
) -> dict[str, Any]:
    diag = evaluator.metrics_for(policy)
    m_agg = evaluator.aggregate_metrics_for(policy)
    variants = all_score_variants(m_agg)
    has_vac = any(rule_has_vacuous_goal_predicate(r, sig) for r in policy.rules)
    has_disc = any(rule_has_disconnected_goal_var(r) for r in policy.rules)
    solves = _per_b_solve_map(policy)
    train_solve_rate = sum(int(solves[b]) for b in train_balls) / max(1, len(train_balls))
    eval_solve_rate = sum(int(solves[b]) for b in eval_balls) / max(1, len(eval_balls))
    n_state = sum(1 for r in policy.rules for L in r.body if L.source.value == "state")
    n_goal = sum(1 for r in policy.rules for L in r.body if L.source.value == "goal")
    return {
        "constraint": constraint,
        "mode": mode,
        "policy_pretty": diag["policy_pretty"],
        "num_rules": len(policy.rules),
        "num_state_literals": n_state,
        "num_goal_literals": n_goal,
        "has_vacuous_goal_predicate": has_vac,
        "has_disconnected_goal_var": has_disc,
        "train_solve_rate": train_solve_rate,
        "eval_solve_rate": eval_solve_rate,
        "progress": m_agg["avg_progress"],
        "steps": m_agg["avg_steps"],
        "noops": m_agg["num_noops"],
        "current_leaf_score": variants["current"],
        "score_no_step_penalty": variants["no_step_penalty"],
        "score_no_noop_penalty": variants["no_noop_penalty"],
        "score_progress_only": variants["progress_only"],
        "score_lexicographic_tuple": list(variants["lexicographic"]),
        "solves_B1": bool(solves.get(1, False)),
        "solves_B2": bool(solves.get(2, False)),
        "solves_B3": bool(solves.get(3, False)),
        "solves_B4": bool(solves.get(4, False)),
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _rank_of(target_pretty: str | None, rows: list[dict]) -> tuple[float | None, int | None]:
    """1-indexed rank of ``target_pretty`` when ``rows`` are sorted by
    ``current_leaf_score`` desc. Returns ``(score, rank)`` or ``(None, None)``."""
    if target_pretty is None:
        return None, None
    sorted_rows = sorted(rows, key=lambda r: r["current_leaf_score"], reverse=True)
    for i, r in enumerate(sorted_rows, start=1):
        if r["policy_pretty"] == target_pretty:
            return r["current_leaf_score"], i
    return None, None


def _best_by_variant(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not rows:
        return out
    for name in SCORE_VARIANT_NAMES:
        if name == "current":
            key_fn = lambda r: r["current_leaf_score"]
        elif name == "lexicographic":
            key_fn = lambda r: tuple(r["score_lexicographic_tuple"])
        else:
            key_fn = lambda r, n=name: r[f"score_{n}"]
        best = max(rows, key=key_fn)
        value = key_fn(best)
        out[name] = {
            "policy_pretty": best["policy_pretty"],
            "value": list(value) if isinstance(value, tuple) else value,
        }
    return out


def _summarise(
    rows: list[dict], *, mode: str,
    hand_pretty: str | None, degenerate_pretty: str | None,
    strata: dict[str, int],
) -> dict[str, Any]:
    n = len(rows)
    solver_count_by_B = {
        b: sum(1 for r in rows if r[f"solves_B{b}"])
        for b in (1, 2, 3, 4)
    }
    b1_to_b2 = sum(1 for r in rows if r["solves_B1"] and r["solves_B2"])
    hand_score, hand_rank = _rank_of(hand_pretty, rows)
    deg_score, deg_rank = _rank_of(degenerate_pretty, rows)
    return {
        "mode": mode,
        "total_policies_evaluated": n,
        "unique_policies_evaluated": n,
        "solver_count_by_B": solver_count_by_B,
        "b1_solver_count": solver_count_by_B[1],
        "b1_to_b2_generalizing_count": b1_to_b2,
        "b2_solver_count": solver_count_by_B[2],
        "hand_policy_score": hand_score,
        "hand_policy_rank": hand_rank,
        "degenerate_drop_policy_score": deg_score,
        "degenerate_drop_policy_rank": deg_rank,
        "vacuous_goal_policy_count": sum(1 for r in rows if r["has_vacuous_goal_predicate"]),
        "disconnected_goal_policy_count": sum(1 for r in rows if r["has_disconnected_goal_var"]),
        "best_policy_by_variant": _best_by_variant(rows),
        "size_strata": dict(sorted(strata.items())),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "constraint", "mode", "policy_pretty", "num_rules", "num_state_literals",
    "num_goal_literals", "has_vacuous_goal_predicate", "has_disconnected_goal_var",
    "train_solve_rate", "eval_solve_rate", "progress", "steps", "noops",
    "current_leaf_score", "score_no_step_penalty", "score_no_noop_penalty",
    "score_progress_only", "score_lexicographic_tuple",
    "solves_B1", "solves_B2", "solves_B3", "solves_B4",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rules", type=int, default=1)
    ap.add_argument("--max-pre-literals", type=int, default=2)
    ap.add_argument("--max-goal-literals", type=int, default=1)
    ap.add_argument("--max-aux-vars", type=int, default=1)
    ap.add_argument("--train-balls", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--eval-balls", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--constraints", nargs="+", default=["none"],
                    choices=CONSTRAINT_CHOICES)
    ap.add_argument("--max-policies", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args(argv)

    sig = gripper_lite_signature()
    train_balls = tuple(args.train_balls)
    eval_balls = tuple(args.eval_balls)
    hand_pretty = hand_policy().pretty()
    degenerate_pretty = degenerate_drop_policy().pretty()

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summary: dict[str, Any] = {
        "meta": {
            "max_rules": args.max_rules,
            "max_pre_literals": args.max_pre_literals,
            "max_goal_literals": args.max_goal_literals,
            "max_aux_vars": args.max_aux_vars,
            "train_balls": list(train_balls),
            "eval_balls": list(eval_balls),
            "constraints": list(args.constraints),
            "max_policies": args.max_policies,
            "seed": args.seed,
            "hand_policy_pretty": hand_pretty,
            "degenerate_drop_policy_pretty": degenerate_pretty,
        },
    }

    for constraint in args.constraints:
        cfg = _cfg_for(
            constraint,
            max_rules=args.max_rules, max_pre_literals=args.max_pre_literals,
            max_goal_literals=args.max_goal_literals, max_aux_vars=args.max_aux_vars,
        )
        t0 = time.time()
        mode, policies, strata = _enumerate_or_sample(
            cfg, sig, max_policies=args.max_policies, seed=args.seed,
        )
        # Build a *fresh* evaluator per constraint so caches don't leak across
        # configs (and so the train/eval env state is clean).
        evaluator = LiftedLeafEvaluator(
            train_instances=[GripperLiteEnv(n_balls=b, seed=0) for b in train_balls],
            eval_in_instances=[GripperLiteEnv(n_balls=b, seed=0) for b in train_balls],
            eval_out_instances=[GripperLiteEnv(n_balls=b, seed=0) for b in eval_balls],
        )
        rows = [
            _policy_row(p, sig, constraint=constraint, mode=mode,
                        train_balls=train_balls, eval_balls=eval_balls,
                        evaluator=evaluator)
            for p in policies
        ]
        all_rows.extend(rows)
        summary[constraint] = _summarise(
            rows, mode=mode,
            hand_pretty=hand_pretty, degenerate_pretty=degenerate_pretty,
            strata=strata,
        )
        summary[constraint]["wall_seconds"] = round(time.time() - t0, 3)
        print(
            f"[landscape] constraint={constraint} mode={mode} "
            f"n={len(rows)} solvers_B1={summary[constraint]['b1_solver_count']} "
            f"vacuous={summary[constraint]['vacuous_goal_policy_count']} "
            f"disconnected={summary[constraint]['disconnected_goal_policy_count']} "
            f"({summary[constraint]['wall_seconds']}s)",
            flush=True,
        )

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in all_rows:
            row = dict(r)
            row["score_lexicographic_tuple"] = json.dumps(row["score_lexicographic_tuple"])
            w.writerow(row)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[landscape] wrote {out_json}")
    print(f"[landscape] wrote {out_csv}  ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
