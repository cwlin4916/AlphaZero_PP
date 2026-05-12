"""Stage 3.1 deterministic typed binding engine for Doors lifted policies.

Given a :class:`LiftedDecisionList` and an environment observation, the
:class:`BindingEngine` finds the first applicable rule under its first
lex-minimal type-respecting binding ``θ`` and returns the grounded action
together with ``θ`` for logging / renaming-test inspection.

This module is the runtime counterpart to ``lifted_typed_dsl.py``.  It
queries the existing :class:`DoorsRelationalRuntime` for both dynamic
predicates (``at_loc``, ``locked``, ``key_available``) and static ones
(``key_for``, ``loc_of_key``, ``next``, ``goal_loc``), so it works on
any DoorsGameConfig, including renamed ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from alphazeropp.instances.doors.dsl.relational_runtime import (
    DoorsRelationalRuntime,
)
from alphazeropp.instances.doors.dsl.lifted_typed_dsl import (
    Atom, LiftedDecisionList, Rule, Var,
)


@dataclass(frozen=True)
class GroundAction:
    name: str
    args: tuple[int, ...]


@dataclass(frozen=True)
class FiringRecord:
    rule_index: int
    action: GroundAction
    binding: tuple[tuple[str, int], ...]  # (var_name, object_id), ordered

    def pretty(self) -> str:
        bind = ", ".join(f"{n}={oid}" for n, oid in self.binding)
        return f"ρ_{self.rule_index + 1}: {self.action.name}({list(self.action.args)}) | θ = {{{bind}}}"


class BindingEngine:
    def __init__(self, runtime: DoorsRelationalRuntime):
        self.rt = runtime

    def _domain(self, var_type: str) -> list[int]:
        return self.rt.objects_of_type(var_type)

    def _atom_holds(
        self,
        atom: Atom,
        theta: dict[Var, int],
        state: np.ndarray,
    ) -> bool:
        args = tuple(theta[v] for v in atom.args)
        return self.rt.query_predicate(atom.pred, args, state)

    def bind_rule(
        self,
        rule: Rule,
        state: np.ndarray,
    ) -> FiringRecord | None:
        variables = rule.variables()
        domains = [self._domain(v.type) for v in variables]
        # itertools.product respects the order of the input domains, and each
        # domain is in numerical (= lex) order — so the first satisfying
        # tuple is the lex-minimal binding.
        for tup in product(*domains):
            theta: dict[Var, int] = dict(zip(variables, tup))
            if all(self._atom_holds(a, theta, state) for a in rule.pre):
                ground_args = tuple(theta[v] for v in rule.action_args)
                return FiringRecord(
                    rule_index=-1,  # filled in by caller
                    action=GroundAction(rule.action, ground_args),
                    binding=tuple(
                        (v.name, theta[v]) for v in variables
                    ),
                )
        return None

    def bind_decision_list(
        self,
        policy: LiftedDecisionList,
        state: np.ndarray,
    ) -> FiringRecord | None:
        for i, rule in enumerate(policy.rules):
            firing = self.bind_rule(rule, state)
            if firing is not None:
                return FiringRecord(
                    rule_index=i,
                    action=firing.action,
                    binding=firing.binding,
                )
        return None

    def encode_action(self, action: GroundAction) -> int:
        if action.name == "pick":
            (k,) = action.args
            return self.rt.action_pick(k)  # type: ignore[arg-type]
        if action.name == "move_to":
            (l,) = action.args
            return self.rt.action_move_to(l)  # type: ignore[arg-type]
        raise ValueError(f"Unknown action: {action.name!r}")


# ---------------------------------------------------------------------------
# Episode driver — runs a typed lifted policy through a Doors env.
# ---------------------------------------------------------------------------

@dataclass
class TypedEpisodeResult:
    solved: bool
    total_env_steps: int
    firings: list[FiringRecord]


def run_typed_lifted_episode(
    env,
    engine: BindingEngine,
    policy: LiftedDecisionList,
    *,
    x0: np.ndarray | None = None,
    is_solved,
    max_steps: int | None = None,
) -> TypedEpisodeResult:
    obs, _ = env.reset()
    if x0 is not None:
        env.state = x0.copy()
        obs = x0.copy()

    cap = max_steps if max_steps is not None else engine.rt.cfg.horizon

    firings: list[FiringRecord] = []
    for step in range(cap):
        if is_solved(obs):
            return TypedEpisodeResult(True, step, firings)
        firing = engine.bind_decision_list(policy, obs)
        if firing is None:
            return TypedEpisodeResult(False, step, firings)
        firings.append(firing)
        action_idx = engine.encode_action(firing.action)
        obs, _reward, terminated, truncated, _info = env.step(action_idx)
        if terminated or truncated:
            return TypedEpisodeResult(
                bool(is_solved(obs)), step + 1, firings,
            )
    return TypedEpisodeResult(bool(is_solved(obs)), cap, firings)
