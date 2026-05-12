#!/usr/bin/env python3
"""Stage-2 smoke test: uniform-MCTS synthesis of lifted policies on Gripper-lite.

Wires the lifted grammar -> ``LiftedDerivationGame`` -> the existing ``MCTS``
(with ``UniformPolicyValueNet``) -> ``LiftedLeafEvaluator`` and runs a handful
of search episodes, logging every new best-so-far policy to JSONL. No baseline
comparison; this only checks the integration works end to end and that some
nonzero-reward (ideally solving) policy is found. See
``docs/notes/stage4/02_plan.md`` / ``02.md``.

Usage:
    python scripts/run/run_lifted_gripper_lite_smoke.py \
        --n-balls-train 1 --n-balls-eval 2 --max-rules 3 --mcts-sims 128 \
        --seed 0 --out-jsonl /tmp/lifted_gripper_b1_seed0.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is importable even without an editable install.
_src = Path(__file__).resolve().parents[2] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from alphazeropp.core.mcts import MCTS
from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.synthesis.derivation_game import UniformPolicyValueNet
from alphazeropp.synthesis.lifted_derivation import LiftedDerivationGame
from alphazeropp.synthesis.lifted_diagnostics import analyze_policy_pathologies
from alphazeropp.synthesis.lifted_grammar import (
    gripper_lite_signature,
    legacy_grammar_config,
    strict_grammar_config,
)
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator

_GRAMMAR_CONFIGS = {"strict": strict_grammar_config, "legacy": legacy_grammar_config}


def git_commit_short() -> str:
    """Short HEAD hash, or ``""`` outside a git checkout (so JSONL stays valid)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-balls-train", type=int, default=1)
    ap.add_argument("--n-balls-eval", type=int, default=2)
    ap.add_argument("--max-rules", type=int, default=3)
    ap.add_argument("--mcts-sims", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-jsonl", type=str, required=True)
    ap.add_argument("--grammar", choices=sorted(_GRAMMAR_CONFIGS), default="strict",
                    help="grammar-safety preset: 'strict' (Stage-3-A default — goal-predicate "
                         "relevance + goal-var connectedness on) or 'legacy' (pre-Stage-3-A, both off)")
    # Stage-2 extras (not in the original spec; harmless defaults):
    ap.add_argument("--n-episodes", type=int, default=64,
                    help="number of independent MCTS search episodes to run")
    ap.add_argument("--dump-all-jsonl", type=str, default=None,
                    help="optional path: dump every distinct evaluated policy's metrics")
    return ap.parse_args(argv)


def _run_episode(cfg, sig, evaluator, net, sims):
    game = LiftedDerivationGame(cfg, sig, evaluator)
    game.reset_wrapper()
    mcts = MCTS(game, net, n_simulations=sims, temperature=1.0, c_exploration=1.5)
    while not game.terminated and not game.truncated:
        probs = np.asarray(mcts.perform_simulations(None), dtype=np.float64)
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            mask = game.get_action_mask()
            a = int(np.flatnonzero(mask)[0])
        else:
            probs = probs / total
            a = int(np.random.choice(len(probs), p=probs))
        game.step_wrapper(a)
    return game.get_program()


def main(argv=None):
    args = _parse_args(argv)
    np.random.seed(args.seed)

    cfg = _GRAMMAR_CONFIGS[args.grammar](max_rules=args.max_rules)  # other knobs at Stage-2 defaults
    sig = gripper_lite_signature()
    git_commit = git_commit_short()
    train_envs = [GripperLiteEnv(n_balls=args.n_balls_train, seed=args.seed)]
    eval_in_envs = train_envs  # single-instance run: eval-in == train (each rollout resets)
    eval_out_envs = [GripperLiteEnv(n_balls=args.n_balls_eval, seed=args.seed)]
    evaluator = LiftedLeafEvaluator(train_envs, eval_in_envs, eval_out_envs)
    net = UniformPolicyValueNet(LiftedDerivationGame(cfg, sig, evaluator)._max_productions)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Run the search episodes. Each terminal program reached by MCTS — both the
    # episode endpoints and every terminal hit during simulations — is scored
    # and recorded in evaluator._cache (insertion order = discovery order).
    for _ in range(args.n_episodes):
        _run_episode(cfg, sig, evaluator, net, args.mcts_sims)

    wall = time.time() - t0

    def _record_from(m):
        rec = {
            "seed": args.seed,
            "mcts_sims": args.mcts_sims,
            "grammar_config_name": args.grammar,
            "git_commit": git_commit,
            "n_balls_train": args.n_balls_train,
            "n_balls_eval": args.n_balls_eval,
            "max_rules": args.max_rules,
            "train_solve_rate": m["train_solve_rate"],
            "eval_in_solve_rate": m["eval_in_solve_rate"],
            "eval_out_solve_rate": m["eval_out_solve_rate"],
            "score": m["score"],
            "avg_steps": m["avg_steps"],
            "num_rules": m["num_rules"],
            "num_literals": m["num_literals"],
            "num_noops": m["num_noops"],
            # Stage 2.5 — `num_rule_evals` is the preferred key;
            # `num_binding_attempts` is a deprecated alias kept for compat.
            "num_rule_evals": m["num_rule_evals"],
            "num_binding_attempts": m["num_binding_attempts"],
            "wall_time": wall,
            "policy_pretty": m["policy_pretty"],
        }
        # Stage 3-A — per-policy structural pathology report (lifted_diagnostics).
        rec.update(analyze_policy_pathologies(
            evaluator.program_for(m["policy_pretty"]),
            goal_predicate_names=sig.goal_predicate_names,
        ))
        return rec

    # Best-so-far progression over every evaluated policy, in discovery order.
    best_score = float("-inf")
    n_lines = 0
    best_metrics = None
    with out_path.open("w") as fh:
        for m in evaluator.all_metrics():
            if m["score"] > best_score:
                best_score = m["score"]
                best_metrics = m
                fh.write(json.dumps(_record_from(m)) + "\n")
                n_lines += 1

    if args.dump_all_jsonl:
        ap = Path(args.dump_all_jsonl)
        ap.parent.mkdir(parents=True, exist_ok=True)
        with ap.open("w") as fh:
            for m in evaluator.all_metrics():
                fh.write(json.dumps(_record_from(m)) + "\n")

    n_unique = len(evaluator._cache)
    n_solving = sum(1 for m in evaluator.all_metrics() if m["train_solve_rate"] >= 1.0)
    n_gen = sum(1 for m in evaluator.all_metrics()
                if m["train_solve_rate"] >= 1.0 and m["eval_out_solve_rate"] >= 1.0)
    print(
        f"[lifted-gripper-smoke] grammar={args.grammar} seed={args.seed} sims={args.mcts_sims} "
        f"episodes={args.n_episodes} unique_policies={n_unique} "
        f"best_score={best_score:.4f} solving_train={n_solving} generalising={n_gen} "
        f"jsonl_lines={n_lines} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
