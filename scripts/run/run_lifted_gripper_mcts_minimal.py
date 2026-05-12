#!/usr/bin/env python3
"""Stage-2 (minimal): uniform-prior MCTS over the redesigned (occurrence-introduced-
variable) lifted grammar, on Gripper-lite, **same-B train and eval only**.

The single research question: can uniform MCTS over the redesigned grammar find
*reasonable* Gripper-lite policies for B=1 and B=2? No held-out generalization,
no learned net, no Doors, no landscape — see ``docs/notes/stage4/02_plan.md``.

Per *episode* we run one MCTS play (PUCT with a flat policy / zero value, i.e.
``UniformPolicyValueNet``, over ``LiftedDerivationGame``) to a terminal policy,
score it with ``LiftedLeafEvaluator`` on a frozen ``GripperLiteEnv(B)``, and
classify it by a cheap classification rollout through the Stage-1 ``interpret``:

  * **solver**     — the rollout reaches ``env.is_solved()`` within the horizon for that B;
  * **reasonable** — not a solver, but ``progress > 0`` *and* the rollout fires
                     >= 1 legal ``pick``, >= 1 legal ``move``, >= 1 legal ``drop``;
  * **degenerate** — the first ``interpret`` returns ``None`` (immediate stall),
                     or only ``None``/illegal attempts, or ``progress == 0``.

Usage:
    python scripts/run/run_lifted_gripper_mcts_minimal.py --balls 1 --mcts-sims 128 \
        --n-episodes 8 --seed 0 --out-jsonl /tmp/mcts_b1.jsonl \
        --out-summary /tmp/mcts_b1_summary.json --log-all-terminals

``--episode-jobs N`` (N>1) runs the N episodes in parallel processes — only worth it
when running ONE cell directly; under ``make_lifted_gripper_mcts_minimal`` (which already
parallelises across cells) keep it at the default 1 to avoid oversubscription.
"""

from __future__ import annotations

import argparse
import json
import random
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
from alphazeropp.synthesis.lifted_grammar import (
    gripper_lite_signature,
    strict_grammar_config,
)
from alphazeropp.synthesis.lifted_interpreter import interpret
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator

_TASK_SCHEMAS = ("pick", "move", "drop")


def git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def default_max_rules(balls: int) -> int:
    return 3 if balls == 1 else 4


# ---------------------------------------------------------------------------
# classification rollout (separate from the leaf-evaluator score path)
# ---------------------------------------------------------------------------

def classify_policy(program, balls: int) -> dict:
    """Roll ``program`` out on a fresh ``GripperLiteEnv(balls)`` via ``interpret``
    and bucket it. Returns ``{classification, solved, progress, steps,
    schemas_fired (sorted unique), first_step_stalled}``."""
    env = GripperLiteEnv(n_balls=balls)
    env.reset()
    horizon = getattr(env, "horizon", 4 * balls + 4)
    schemas_fired: set[str] = set()
    steps = 0
    first_step_stalled = False
    for t in range(horizon):
        if env.is_solved():
            break
        out = interpret(
            program,
            env.get_state_atoms(),
            env.get_goal_atoms(),
            env.get_objects_by_type(),
            env.legal_actions(),
        )
        if out is None:
            if t == 0:
                first_step_stalled = True
            break
        # interpret() already filters to legal actions, so `out` is legal here.
        schemas_fired.add(out.schema)
        env.step(out)
        steps += 1
    solved = bool(env.is_solved())
    goal_at = env.get_goal_atoms().get("at_ball", set())
    state_at = env.get_state_atoms().get("at_ball", set())
    progress = (len(state_at & goal_at) / len(goal_at)) if goal_at else 1.0
    if solved:
        classification = "solver"
    elif progress > 0 and all(s in schemas_fired for s in _TASK_SCHEMAS):
        classification = "reasonable"
    else:
        classification = "degenerate"
    return {
        "classification": classification,
        "solved": solved,
        "progress": progress,
        "steps": steps,
        "schemas_fired": sorted(schemas_fired),
        "first_step_stalled": first_step_stalled,
    }


# ---------------------------------------------------------------------------
# one MCTS episode
# ---------------------------------------------------------------------------

def run_episode(cfg, sig, evaluator, mcts_sims: int, c_exploration: float):
    game = LiftedDerivationGame(cfg, sig, evaluator)
    game.reset_wrapper()
    net = UniformPolicyValueNet(game._max_productions)
    mcts = MCTS(game, net, n_simulations=mcts_sims, c_exploration=c_exploration)
    while not game.terminated and not game.truncated:
        probs = mcts.perform_simulations(None)
        probs = np.asarray(probs, dtype=np.float64)
        s = probs.sum()
        if s <= 0:
            # dead end with no legal move should not happen (a STOP_* is always
            # offered until terminal); guard anyway.
            break
        a = int(np.random.choice(len(probs), p=probs / s))
        game.step_wrapper(a)
    return game.get_program()


def _episode_worker(task):
    """Top-level (picklable) worker for ``--episode-jobs > 1``: re-seed, build a
    fresh grammar/signature/evaluator, run one MCTS play, return the terminal
    ``Policy`` (or ``None``). Scoring/classification/dedup all happen back in the
    parent over the returned programs, in episode order — so the only difference
    from ``--episode-jobs 1`` is that each episode gets its own seed
    (``base_seed + ep``) instead of drawing from one global RNG stream; results
    are still deterministic for a given ``--episode-jobs`` value."""
    balls, max_rules, mcts_sims, c_exploration, seed = task
    random.seed(seed)
    np.random.seed(seed)
    sig = gripper_lite_signature()
    cfg = strict_grammar_config(max_rules=max_rules)
    evaluator = LiftedLeafEvaluator(
        [GripperLiteEnv(n_balls=balls)],
        [GripperLiteEnv(n_balls=balls)],
        [GripperLiteEnv(n_balls=balls)],
    )
    return run_episode(cfg, sig, evaluator, mcts_sims, c_exploration)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--balls", type=int, choices=(1, 2), required=True)
    ap.add_argument("--max-rules", type=int, default=None,
                    help="default 3 for B=1, 4 for B=2")
    ap.add_argument("--mcts-sims", type=int, default=256)
    ap.add_argument("--n-episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-exploration", type=float, default=1.5)
    ap.add_argument("--episode-jobs", type=int, default=1,
                    help="run this many independent MCTS plays in parallel processes "
                         "(default 1 = the in-process loop; use >1 only when running ONE "
                         "cell directly — under the grid driver keep it 1 to avoid "
                         "oversubscription)")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--log-all-terminals", action="store_true",
                    help="log every distinct terminal policy (not just best-so-far)")
    args = ap.parse_args(argv)

    max_rules = args.max_rules if args.max_rules is not None else default_max_rules(args.balls)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sig = gripper_lite_signature()
    cfg = strict_grammar_config(max_rules=max_rules)
    # Same-B: train == eval (Stage 2 makes no generalization claim).
    evaluator = LiftedLeafEvaluator(
        [GripperLiteEnv(n_balls=args.balls)],
        [GripperLiteEnv(n_balls=args.balls)],
        [GripperLiteEnv(n_balls=args.balls)],
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    seen: dict[str, dict] = {}          # policy_pretty -> record
    best_score = -float("inf")
    first_solver_episode = None
    first_solver_unique_index = None
    counts = {"solver": 0, "reasonable": 0, "degenerate": 0}
    best_record = None
    t0 = time.time()

    def ingest(fh, program, ep: int) -> None:
        """Score + classify + dedup a terminal policy from episode ``ep`` and
        append the right JSONL record(s). Mutates ``seen``/``counts`` and the
        ``best_*`` / ``first_solver_*`` accumulators."""
        nonlocal best_score, best_record, first_solver_episode, first_solver_unique_index
        if program is None:
            return
        key = program.pretty()
        score = float(evaluator(program))
        m = evaluator.metrics_for(program)
        if key not in seen:
            cls = classify_policy(program, args.balls)
            unique_index = len(seen)
            rec = {
                "balls": args.balls,
                "episode": ep,
                "unique_index": unique_index,
                "score": score,
                "solved": bool(m["train_solve_rate"] >= 1.0),
                "progress": m["avg_progress"],
                "steps": m["avg_steps"],
                "noops": m["num_noops"],
                "num_rules": m["num_rules"],
                "num_literals": m["num_literals"],
                "policy_pretty": key,
                "classification": cls["classification"],
                "schemas_fired": cls["schemas_fired"],
            }
            seen[key] = rec
            counts[cls["classification"]] += 1
            if cls["classification"] == "solver" and first_solver_episode is None:
                first_solver_episode = ep
                first_solver_unique_index = unique_index
            if args.log_all_terminals:
                fh.write(json.dumps({**rec, "kind": "terminal"}) + "\n")
        else:
            rec = seen[key]
        if score > best_score:
            best_score = score
            best_record = dict(rec)
            fh.write(json.dumps({**rec, "kind": "best_so_far", "episode": ep}) + "\n")

    with args.out_jsonl.open("w") as fh:
        if args.episode_jobs <= 1:
            for ep in range(args.n_episodes):
                ingest(fh, run_episode(cfg, sig, evaluator, args.mcts_sims, args.c_exploration), ep)
        else:
            from concurrent.futures import ProcessPoolExecutor
            tasks = [
                (args.balls, max_rules, args.mcts_sims, args.c_exploration, args.seed + ep)
                for ep in range(args.n_episodes)
            ]
            with ProcessPoolExecutor(max_workers=args.episode_jobs) as ex:
                for ep, program in enumerate(ex.map(_episode_worker, tasks)):
                    ingest(fh, program, ep)
        fh.flush()

    n_unique = len(seen)
    n_solving = counts["solver"]
    summary = {
        "balls": args.balls,
        "max_rules": max_rules,
        "mcts_sims": args.mcts_sims,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "c_exploration": args.c_exploration,
        "n_unique": n_unique,
        "best_score": best_score if n_unique else None,
        "best_policy_pretty": best_record["policy_pretty"] if best_record else None,
        "best_metrics": {
            "score": best_record["score"],
            "solved": best_record["solved"],
            "progress": best_record["progress"],
            "steps": best_record["steps"],
            "noops": best_record["noops"],
            "num_rules": best_record["num_rules"],
            "num_literals": best_record["num_literals"],
            "classification": best_record["classification"],
        } if best_record else None,
        "n_solving": n_solving,
        "first_solver_episode": first_solver_episode,
        "first_solver_unique_index": first_solver_unique_index,
        "counts": dict(counts),
        "elapsed_sec": round(time.time() - t0, 2),
        "git_commit": git_commit_short(),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"B={args.balls} sims={args.mcts_sims} seed={args.seed} episodes={args.n_episodes}: "
          f"{n_unique} unique policies, best score {best_score:.4f}, "
          f"solver/reasonable/degenerate = {counts['solver']}/{counts['reasonable']}/{counts['degenerate']}, "
          f"first solver: ep {first_solver_episode}")
    if best_record:
        print("best policy:")
        print(best_record["policy_pretty"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
