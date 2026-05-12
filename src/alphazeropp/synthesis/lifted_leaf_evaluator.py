"""Leaf evaluator for lifted decision-list policies.

Stage 2 — see ``docs/notes/stage4/02_plan.md`` §2.4. Runs episodic rollouts on
Gripper-lite instances through the Stage 1 :func:`interpret`, scores them, and
caches by ``Policy.pretty()`` (same cache-key discipline as
:class:`alphazeropp.synthesis.leaf_evaluator.LeafEvaluator`).

Score (per the spec):

    solve_bonus  = train solve-rate (0 or 1 for a single instance)
    progress     = |state["at_ball"] ∩ goal["at_ball"]| / |goal["at_ball"]|  at final state
    step_penalty = 0.01 * steps
    noop_penalty = 0.05 * num_noops      (num_noops = # times interpret() returned None)
    score        = solve_bonus + 0.25 * progress - step_penalty - noop_penalty

``num_rule_evals`` (Stage 2.5 — preferred name) is a *rule-evaluation proxy*:
``rule_idx + 1`` on a firing step, ``len(policy.rules)`` on a stall — it does
*not* instrument ``find_bindings`` itself (which would mean touching a Stage 1
module). ``num_binding_attempts`` is kept as a backwards-compatible **alias**
for the same value but is **deprecated** — new code should read
``num_rule_evals``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from alphazeropp.synthesis.lifted_dsl import Policy
from alphazeropp.synthesis.lifted_interpreter import interpret


class LiftedLeafEvaluator:
    def __init__(
        self,
        train_instances: Iterable,        # GripperLiteEnv objects; reset() before each rollout
        eval_in_instances: Iterable,      # may alias train_instances
        eval_out_instances: Iterable,
        *,
        step_penalty: float = 0.01,
        noop_penalty: float = 0.05,
        progress_weight: float = 0.25,
        horizon: Optional[int] = None,
    ):
        self.train = list(train_instances)
        self.eval_in = list(eval_in_instances)
        self.eval_out = list(eval_out_instances)
        self.step_penalty = step_penalty
        self.noop_penalty = noop_penalty
        self.progress_weight = progress_weight
        self.horizon = horizon
        self._cache: dict[str, dict] = {}
        # Stage 3-A — keep the Policy object alongside its cached metrics so callers
        # (the smoke / diagnostic-grid scripts) can attach a structural pathology
        # report (see alphazeropp.synthesis.lifted_diagnostics) to each JSONL record.
        self._programs: dict[str, Policy] = {}

    # -- rollout --
    def _rollout_one(self, env, program: Policy) -> dict:
        env.reset()
        h = self.horizon if self.horizon is not None else getattr(env, "horizon", 50)
        n_rules = len(program.rules)
        steps = 0
        num_noops = 0
        num_binding_attempts = 0
        for _ in range(h):
            if env.is_solved():
                break
            out = interpret(
                program,
                env.get_state_atoms(),
                env.get_goal_atoms(),
                env.get_objects_by_type(),
                env.legal_actions(),
                trace=True,
            )
            if out is None:
                num_noops += 1
                num_binding_attempts += n_rules
                break
            action, rule_idx, _theta = out
            num_binding_attempts += rule_idx + 1
            env.step(action)
            steps += 1
        solved = env.is_solved()
        goal_at = env.get_goal_atoms().get("at_ball", set())
        state_at = env.get_state_atoms().get("at_ball", set())
        progress = (len(state_at & goal_at) / len(goal_at)) if goal_at else 1.0
        return {
            "solved": bool(solved),
            "steps": steps,
            "num_noops": num_noops,
            # `num_rule_evals` is the preferred name (Stage 2.5);
            # `num_binding_attempts` is a deprecated alias for the same value.
            "num_rule_evals": num_binding_attempts,
            "num_binding_attempts": num_binding_attempts,
            "progress": progress,
        }

    def _aggregate(self, envs: list, program: Policy) -> dict:
        rs = [self._rollout_one(e, program) for e in envs]
        n = len(rs) or 1
        # `num_rule_evals` is the preferred Stage-2.5 name; `num_binding_attempts`
        # is the deprecated alias kept for backwards compatibility.
        rule_evals = sum(r["num_rule_evals"] for r in rs)
        return {
            "solve_rate": sum(r["solved"] for r in rs) / n,
            "avg_steps": sum(r["steps"] for r in rs) / n,
            "avg_progress": sum(r["progress"] for r in rs) / n,
            "num_noops": sum(r["num_noops"] for r in rs),
            "num_rule_evals": rule_evals,
            "num_binding_attempts": rule_evals,
        }

    def _score_from(self, m: dict) -> float:
        return (
            m["solve_rate"]
            + self.progress_weight * m["avg_progress"]
            - self.step_penalty * m["avg_steps"]
            - self.noop_penalty * m["num_noops"]
        )

    # -- public --
    def __call__(self, program: Policy) -> float:
        key = program.pretty()
        cached = self._cache.get(key)
        if cached is not None:
            return cached["score"]
        tm = self._aggregate(self.train, program)
        em_in = self._aggregate(self.eval_in, program)
        em_out = self._aggregate(self.eval_out, program)
        score = self._score_from(tm)
        diag = {
            "score": score,
            "train_solve_rate": tm["solve_rate"],
            "eval_in_solve_rate": em_in["solve_rate"],
            "eval_out_solve_rate": em_out["solve_rate"],
            "avg_steps": tm["avg_steps"],
            "avg_progress": tm["avg_progress"],
            "num_rules": len(program.rules),
            "num_literals": sum(len(r.body) for r in program.rules),
            "num_noops": tm["num_noops"],
            # Stage 2.5 — preferred key + deprecated alias (same value).
            "num_rule_evals": tm["num_rule_evals"],
            "num_binding_attempts": tm["num_rule_evals"],
            "policy_pretty": key,
        }
        self._cache[key] = diag
        self._programs[key] = program
        return score

    def metrics_for(self, program: Policy) -> dict:
        key = program.pretty()
        if key not in self._cache:
            self.__call__(program)
        return dict(self._cache[key])

    def aggregate_metrics_for(self, program: Policy) -> dict:
        """Stage 2.5 — return the *aggregate-shaped* train-instance metrics dict.

        Keys: ``{"solve_rate", "avg_progress", "avg_steps", "num_noops"}`` —
        exactly the shape consumed by
        :mod:`alphazeropp.synthesis.lifted_score_variants`. Lets callers compute
        score variants without reaching into the cached ``diag`` dict (whose
        ``train_solve_rate`` key uses a different name)."""
        d = self.metrics_for(program)
        return {
            "solve_rate": d["train_solve_rate"],
            "avg_progress": d["avg_progress"],
            "avg_steps": d["avg_steps"],
            "num_noops": d["num_noops"],
        }

    def all_metrics(self) -> list[dict]:
        return [dict(v) for v in self._cache.values()]

    def program_for(self, policy_pretty: str) -> Policy:
        """The :class:`Policy` whose ``pretty()`` equals ``policy_pretty`` (must have been evaluated)."""
        return self._programs[policy_pretty]

    def all_metrics_with_programs(self) -> list[tuple[dict, Policy]]:
        """``(metrics, program)`` pairs in discovery order — for callers that attach
        per-policy diagnostics (Stage 3-A)."""
        return [(dict(v), self._programs[k]) for k, v in self._cache.items()]
