#!/usr/bin/env python3
"""Stage 3-A — reproducible diagnostic grid for lifted-policy synthesis on Gripper-lite.

For each cell ``(run, grammar, baseline, seed, sims)`` this drives the lifted grammar →
``LiftedDerivationGame`` → either the unchanged ``MCTS`` (uniform prior) **or** a uniform-random
terminal-policy sampler → ``LiftedLeafEvaluator``, and writes a self-contained, claim-safe record:
config JSON (incl. git commit), best-so-far JSONL, summary JSON, the full per-policy JSONL (with the
``lifted_diagnostics.analyze_policy_pathologies`` report), and a rolled-up CSV. **No neural network** —
this is the diagnostic baseline that precedes any learned prior (see ``docs/notes/stage4/03_plan.md`` /
``03.md``; the Stage-2 deceptive-landscape finding is in ``02.md``).

Runs / grammars / baselines
---------------------------
* ``--runs A`` : train ``B=1`` / eval-out ``B=2`` / ``max_rules=3``  (Stage-2 "Run A").
* ``--runs B`` : train ``B=2`` / eval-out ``B=3`` / ``max_rules=4``  (Stage-2 "Run B").
* ``--grammars strict`` : Stage-3-A default grammar (goal-predicate relevance + goal-var connectedness on).
* ``--grammars legacy`` : pre-Stage-3-A permissive grammar (both off) — the strict/legacy contrast.
* ``--baselines mcts``   : uniform-prior MCTS, ``--episodes`` independent search episodes per cell.
* ``--baselines random`` : uniform-random complete derivations until the cell's unique-policy count
  matches the paired ``mcts`` cell (or ``--random-budget`` / 4096 if no paired ``mcts`` cell ran).

Outputs
-------
* ``{raw-dir}/{run_id}/all.jsonl``         — every distinct evaluated policy: smoke-style metrics +
  pathology dict + ``grammar_config_name`` + ``baseline`` + ``git_commit`` (large; under ``results/`` —
  gitignored / regenerable).
* ``{out-dir}/{run_id}/config.json``       — full cell config + grammar-config dataclass dump + git commit.
* ``{out-dir}/{run_id}/best.jsonl``        — best-so-far progression (small).
* ``{out-dir}/{run_id}/summary.json``      — headline numbers + aggregate pathology counts + root entropy.
* ``{out-dir}/diagnostic_grid.csv``        — one row per cell (header documented below; appended/merged
  across invocations on ``run_id``).

``run_id = f"{run}_{grammar}_{baseline}_seed{seed}_sims{sims}"`` (``sims`` is ``na`` for ``random``).

The acceptance-criterion smoke command::

    python scripts/run/run_lifted_gripper_diagnostic_grid.py --seeds 0 1 --sims 64 --episodes 8 --strict-only

The committed small grid (both grammars, both baselines)::

    python scripts/run/run_lifted_gripper_diagnostic_grid.py --seeds 0 1 --sims 64 --episodes 8

The full sweep (``--seeds 0..9 --sims 64 128 256 512 --episodes 64``, both grammars/baselines/runs) is
*runnable but intentionally not committed* — running it and reporting its numbers would be an unrun claim
until it is actually executed.

Parallelism: this driver runs its cells *in-process* (``_run_mcts_cell`` / ``_run_random_cell``), and a
``random`` cell's unique-policy budget is read from its paired ``mcts`` cell — so it is left sequential
rather than fanned out via ``scripts/run/_parallel``. (The ``make_lifted_gripper_mcts_minimal`` driver,
whose cells are subprocess-isolated and independent, is the one that parallelises — ``--jobs``.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_src = _REPO_ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from alphazeropp.core.mcts import MCTS
from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.synthesis.derivation_game import UniformPolicyValueNet
from alphazeropp.synthesis.lifted_derivation import (
    LiftedDerivationGame,
    LiftedDerivationState,
)
from alphazeropp.synthesis.lifted_diagnostics import analyze_policy_pathologies
from alphazeropp.synthesis.lifted_grammar import (
    enumerate_productions,
    gripper_lite_signature,
    legacy_grammar_config,
    strict_grammar_config,
)
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator


def git_commit_short() -> str:
    """Short HEAD hash, or ``""`` outside a git checkout (so JSONL/JSON stays valid)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""

_GRAMMARS = {"strict": strict_grammar_config, "legacy": legacy_grammar_config}
# run -> (n_balls_train, n_balls_eval_out, max_rules)
_RUNS = {"A": (1, 2, 3), "B": (2, 3, 4)}
_DEFAULT_RANDOM_BUDGET = 4096

_CSV_COLUMNS = (
    "run_id", "run", "grammar_config_name", "baseline", "seed", "sims", "episodes",
    "train_B", "eval_B", "max_rules",
    "unique_policies", "terminal_evals", "best_score",
    "train_solve_rate_best", "eval_out_solve_rate_best",
    "first_solver_idx", "solver_count",
    "root_entropy_mean", "root_entropy_final",
    "n_vacuous_goal", "n_goal_only_var", "n_empty_body", "n_top_drop", "n_top_pick", "n_top_move",
    "git_commit",
)


# ---------------------------------------------------------------------------
# per-policy record (shared shape with run_lifted_gripper_lite_smoke._record_from,
# plus baseline / pathology fields)
# ---------------------------------------------------------------------------

def _policy_record(m: dict, program, *, sig, run, grammar_name, baseline, seed, sims,
                   train_B, eval_B, max_rules, git_commit) -> dict:
    rec = {
        "run": run, "grammar_config_name": grammar_name, "baseline": baseline,
        "seed": seed, "sims": sims, "git_commit": git_commit,
        "n_balls_train": train_B, "n_balls_eval": eval_B, "max_rules": max_rules,
        "train_solve_rate": m["train_solve_rate"],
        "eval_in_solve_rate": m["eval_in_solve_rate"],
        "eval_out_solve_rate": m["eval_out_solve_rate"],
        "score": m["score"],
        "avg_steps": m["avg_steps"],
        "num_rules": m["num_rules"],
        "num_literals": m["num_literals"],
        "num_noops": m["num_noops"],
        "num_rule_evals": m["num_rule_evals"],
        "num_binding_attempts": m["num_binding_attempts"],
        "policy_pretty": m["policy_pretty"],
    }
    rec.update(analyze_policy_pathologies(program, goal_predicate_names=sig.goal_predicate_names))
    return rec


def _shannon_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0.0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# the two baselines
# ---------------------------------------------------------------------------

def _run_mcts_cell(cfg, sig, evaluator, *, n_episodes: int, sims: int, rng_seed: int):
    """Run ``n_episodes`` independent uniform-prior MCTS search episodes; populate ``evaluator``.
    Returns the list of decision-point visit-distribution entropies, in order, with the index of the
    last decision point of the last episode (for ``root_entropy_final``)."""
    np.random.seed(rng_seed)
    net = UniformPolicyValueNet(LiftedDerivationGame(cfg, sig, evaluator)._max_productions)
    entropies: list[float] = []
    last_episode_entropies: list[float] = []
    for _ in range(n_episodes):
        game = LiftedDerivationGame(cfg, sig, evaluator)
        game.reset_wrapper()
        mcts = MCTS(game, net, n_simulations=sims, temperature=1.0, c_exploration=1.5)
        ep_entropies: list[float] = []
        while not game.terminated and not game.truncated:
            probs = np.asarray(mcts.perform_simulations(None), dtype=np.float64)
            total = probs.sum()
            if total <= 0 or not np.isfinite(total):
                mask = game.get_action_mask()
                a = int(np.flatnonzero(mask)[0])
            else:
                probs = probs / total
                if int((probs > 0).sum()) > 1:
                    h = _shannon_entropy(probs)
                    entropies.append(h)
                    ep_entropies.append(h)
                a = int(np.random.choice(len(probs), p=probs))
            game.step_wrapper(a)
        last_episode_entropies = ep_entropies or last_episode_entropies
    return entropies, last_episode_entropies


def _run_random_cell(cfg, sig, evaluator, *, target_unique: int, rng_seed: int):
    """Draw uniform-random complete derivations (uniform legal production at each hole), evaluate each,
    until ``target_unique`` distinct policies have been evaluated (or a walk cap is hit)."""
    rng = random.Random(rng_seed)
    walk_cap = max(2000, 80 * target_unique)
    walks = 0
    while len(evaluator._cache) < target_unique and walks < walk_cap:
        walks += 1
        s = LiftedDerivationState.initial()
        while not s.is_terminal():
            prods = enumerate_productions(s, cfg, sig)
            if not prods:
                break
            s = s.apply(rng.choice(prods))
        if not s.is_terminal():
            continue
        evaluator(s.to_program())
    return walks


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def _summarise(records: list[dict], *, run_id, run, grammar_name, baseline, seed, sims, episodes,
               train_B, eval_B, max_rules, entropies, last_ep_entropies, git_commit, walks=None):
    n_unique = len(records)
    solver_idxs = [i for i, r in enumerate(records) if r["train_solve_rate"] >= 1.0]
    held_idxs = [i for i in solver_idxs if records[i]["eval_out_solve_rate"] >= 1.0]
    best = max(records, key=lambda r: r["score"]) if records else None
    ent_mean = float(np.mean(entropies)) if entropies else None
    ent_final = float(np.mean(last_ep_entropies)) if last_ep_entropies else None
    return {
        "run_id": run_id, "run": run, "grammar_config_name": grammar_name, "baseline": baseline,
        "seed": seed, "sims": sims, "episodes": episodes,
        "train_B": train_B, "eval_B": eval_B, "max_rules": max_rules,
        "unique_policies": n_unique,
        "terminal_evals": n_unique,  # one record per distinct evaluated policy
        "random_walks": walks,
        "best_score": (best["score"] if best else None),
        "train_solve_rate_best": (best["train_solve_rate"] if best else None),
        "eval_out_solve_rate_best": (best["eval_out_solve_rate"] if best else None),
        "best_num_rules": (best["num_rules"] if best else None),
        "best_num_literals": (best["num_literals"] if best else None),
        "best_policy_pretty": (best["policy_pretty"] if best else None),
        "first_solver_idx": (solver_idxs[0] if solver_idxs else None),
        "solver_count": len(solver_idxs),
        "held_out_solver_count": len(held_idxs),
        "root_entropy_mean": ent_mean,
        "root_entropy_final": ent_final,
        "n_vacuous_goal": sum(1 for r in records if r["has_vacuous_goal_predicate"]),
        "n_goal_only_var": sum(1 for r in records if r["has_goal_only_variable"]),
        "n_empty_body": sum(1 for r in records if r["has_empty_body_rule"]),
        "n_top_drop": sum(1 for r in records if r["has_top_drop_rule"]),
        "n_top_pick": sum(1 for r in records if r["has_top_pick_rule"]),
        "n_top_move": sum(1 for r in records if r["has_top_move_rule"]),
        "git_commit": git_commit,
    }


def _csv_row(summary: dict) -> dict:
    return {c: summary.get(c) for c in _CSV_COLUMNS}


def run_cell(*, run: str, grammar_name: str, baseline: str, seed: int, sims: int, episodes: int,
             out_dir: Path, raw_dir: Path, random_budget: int, matched_unique: int | None,
             git_commit: str) -> dict:
    train_B, eval_B, max_rules = _RUNS[run]
    cfg = _GRAMMARS[grammar_name](max_rules=max_rules)
    sig = gripper_lite_signature()
    sims_label = sims if baseline == "mcts" else "na"
    run_id = f"{run}_{grammar_name}_{baseline}_seed{seed}_sims{sims_label}"

    train_envs = [GripperLiteEnv(n_balls=train_B, seed=seed)]
    eval_in_envs = train_envs
    eval_out_envs = [GripperLiteEnv(n_balls=eval_B, seed=seed)]
    evaluator = LiftedLeafEvaluator(train_envs, eval_in_envs, eval_out_envs)

    t0 = time.time()
    entropies: list[float] = []
    last_ep_entropies: list[float] = []
    walks = None
    if baseline == "mcts":
        entropies, last_ep_entropies = _run_mcts_cell(
            cfg, sig, evaluator, n_episodes=episodes, sims=sims, rng_seed=seed,
        )
    elif baseline == "random":
        target = matched_unique if matched_unique is not None else random_budget
        walks = _run_random_cell(cfg, sig, evaluator, target_unique=target, rng_seed=seed)
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(f"unknown baseline {baseline!r}")
    wall = time.time() - t0

    records = [
        _policy_record(m, p, sig=sig, run=run, grammar_name=grammar_name, baseline=baseline,
                       seed=seed, sims=sims_label, train_B=train_B, eval_B=eval_B,
                       max_rules=max_rules, git_commit=git_commit)
        for (m, p) in evaluator.all_metrics_with_programs()
    ]
    summary = _summarise(
        records, run_id=run_id, run=run, grammar_name=grammar_name, baseline=baseline,
        seed=seed, sims=sims_label, episodes=(episodes if baseline == "mcts" else None),
        train_B=train_B, eval_B=eval_B, max_rules=max_rules,
        entropies=entropies, last_ep_entropies=last_ep_entropies, git_commit=git_commit, walks=walks,
    )
    summary["wall_time"] = wall

    cell_out = out_dir / run_id
    cell_raw = raw_dir / run_id
    cell_out.mkdir(parents=True, exist_ok=True)
    cell_raw.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id, "run": run, "grammar_config_name": grammar_name, "baseline": baseline,
        "seed": seed, "sims": sims_label, "episodes": (episodes if baseline == "mcts" else None),
        "train_B": train_B, "eval_B": eval_B, "max_rules": max_rules,
        "grammar_config": asdict(cfg), "git_commit": git_commit,
        "random_budget": (random_budget if baseline == "random" and matched_unique is None else None),
        "matched_unique_budget": matched_unique if baseline == "random" else None,
    }
    (cell_out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (cell_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # best-so-far progression (discovery order)
    best_score = -math.inf
    with (cell_out / "best.jsonl").open("w") as fh:
        for r in records:
            if r["score"] > best_score:
                best_score = r["score"]
                fh.write(json.dumps(r) + "\n")
    # full per-policy log (raw / gitignored)
    with (cell_raw / "all.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"[diag-grid] {run_id}: unique={summary['unique_policies']} "
          f"best={summary['best_score']} first_solver={summary['first_solver_idx']} "
          f"solvers={summary['solver_count']} held_out={summary['held_out_solver_count']} "
          f"H_root~{summary['root_entropy_mean']} ({wall:.1f}s)", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--sims", type=int, nargs="+", default=[64, 128, 256, 512],
                    help="MCTS simulation budgets (one cell per budget; ignored by the 'random' baseline)")
    ap.add_argument("--episodes", type=int, default=64,
                    help="independent MCTS search episodes per (mcts) cell")
    ap.add_argument("--grammars", choices=sorted(_GRAMMARS), nargs="+", default=sorted(_GRAMMARS))
    ap.add_argument("--strict-only", action="store_true",
                    help="shortcut for --grammars strict")
    ap.add_argument("--runs", choices=sorted(_RUNS), nargs="+", default=sorted(_RUNS))
    ap.add_argument("--baselines", choices=("mcts", "random"), nargs="+", default=["mcts", "random"])
    ap.add_argument("--random-budget", type=int, default=_DEFAULT_RANDOM_BUDGET,
                    help="unique-policy target for a 'random' cell that has no paired 'mcts' cell")
    ap.add_argument("--out-dir", type=str,
                    default=str(_REPO_ROOT / "docs" / "notes" / "stage4" / "data" / "diagnostic_grid"))
    ap.add_argument("--raw-dir", type=str,
                    default=str(_REPO_ROOT / "results" / "lifted_gripper_lite" / "diagnostic_grid"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    grammars = ["strict"] if args.strict_only else list(args.grammars)
    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    git_commit = git_commit_short()

    do_mcts = "mcts" in args.baselines
    do_random = "random" in args.baselines

    summaries: list[dict] = []
    for run in args.runs:
        for grammar in grammars:
            for seed in args.seeds:
                # MCTS cells first (one per sims budget); remember the largest-budget unique count
                # to "match" the random baseline against (a coarse but reproducible budget).
                matched_unique = None
                if do_mcts:
                    for sims in args.sims:
                        s = run_cell(run=run, grammar_name=grammar, baseline="mcts", seed=seed,
                                     sims=sims, episodes=args.episodes, out_dir=out_dir, raw_dir=raw_dir,
                                     random_budget=args.random_budget, matched_unique=None,
                                     git_commit=git_commit)
                        summaries.append(s)
                        matched_unique = max(matched_unique or 0, int(s["unique_policies"]))
                if do_random:
                    s = run_cell(run=run, grammar_name=grammar, baseline="random", seed=seed,
                                 sims=0, episodes=args.episodes, out_dir=out_dir, raw_dir=raw_dir,
                                 random_budget=args.random_budget, matched_unique=matched_unique,
                                 git_commit=git_commit)
                    summaries.append(s)

    # rolled-up CSV — merge with any existing rows on run_id (so re-running a subset updates in place).
    csv_path = out_dir / "diagnostic_grid.csv"
    existing: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                existing[row["run_id"]] = row
    for s in summaries:
        existing[s["run_id"]] = _csv_row(s)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        w.writeheader()
        for run_id in sorted(existing):
            w.writerow({c: existing[run_id].get(c) for c in _CSV_COLUMNS})

    grid_summary_path = out_dir / "grid_summary.json"
    grid_summary_path.write_text(json.dumps(
        {"git_commit": git_commit, "cells": {s["run_id"]: s for s in summaries}}, indent=2) + "\n")
    print(f"[diag-grid] wrote {csv_path} ({len(existing)} rows total), "
          f"{grid_summary_path}, and {len(summaries)} per-cell dirs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
