"""Derivation state machine + ``Game`` wrapper for the lifted-policy grammar.

Stage 2 — see ``docs/notes/stage4/02_plan.md`` §2.2. ``LiftedDerivationState``
is a frozen value; ``apply`` returns a fresh state and never mutates ``self``.
``LiftedDerivationGame`` mirrors ``DerivationGame``'s ``Game`` surface so the
existing ``MCTS`` engine drives it unchanged; ``stash_state`` keeps the leaf
evaluator *shared* (the base-class deepcopy default would clone its cache on
every simulation).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.synthesis.lifted_dsl import (
    LiftedAction,
    Literal,
    Policy,
    Rule,
    Var,
)
from alphazeropp.synthesis.lifted_encoding import encode_max_len, encode_state
from alphazeropp.synthesis.lifted_grammar import (
    DomainSignature,
    LiftedGrammarConfig,
    LiftedProduction,
    aux_var_name,
    compute_max_productions,
    enumerate_productions,
)


# ---------------------------------------------------------------------------
# Partial rule accumulator + derivation state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartialRule:
    schema: str | None
    action_args: tuple[Var, ...]
    aux_vars: tuple[Var, ...]
    state_lits: tuple[Literal, ...]
    goal_lits: tuple[Literal, ...]

    @staticmethod
    def empty() -> "PartialRule":
        return PartialRule(schema=None, action_args=(), aux_vars=(),
                           state_lits=(), goal_lits=())

    def all_vars(self) -> tuple[Var, ...]:
        seen: set[str] = set()
        out: list[Var] = []
        for v in self.action_args + self.aux_vars:
            if v.name not in seen:
                seen.add(v.name)
                out.append(v)
        return tuple(out)

    def positive_var_names(self) -> set[str]:
        names = {v.name for v in self.action_args}
        for lit in self.state_lits:           # all state lits are positive in Stage 2
            names.update(v.name for v in lit.atom.variables())
        return names


@dataclass(frozen=True)
class LiftedDerivationState:
    """``current_hole`` is one of ``"policy"`` / ``"action_schema"`` /
    ``"aux_var"`` / ``"pre_lit"`` / ``"goal_lit"`` while building, or ``None``
    once the policy is finished (terminal). ``partial`` is non-``None`` iff a
    rule is in progress."""
    completed_rules: tuple[Rule, ...]
    partial: PartialRule | None
    current_hole: str | None

    @staticmethod
    def initial() -> "LiftedDerivationState":
        return LiftedDerivationState(completed_rules=(), partial=None, current_hole="policy")

    # -- grammar bridge --
    def legal_productions(self, cfg: LiftedGrammarConfig, sig: DomainSignature) -> list[LiftedProduction]:
        return enumerate_productions(self, cfg, sig)

    def apply(self, prod: LiftedProduction) -> "LiftedDerivationState":
        kind = prod.hole_kind
        tag = prod.payload[0]

        if kind == "policy":
            if tag == "add_rule":
                return replace(self, current_hole="action_schema", partial=PartialRule.empty())
            if tag == "stop":
                return replace(self, current_hole=None, partial=None)

        p = self.partial
        assert p is not None, f"production {prod.label!r} requires a partial rule"

        if kind == "action_schema":               # ("schema", name, arg_types)
            _, schema_name, arg_types = prod.payload
            action_args = tuple(
                Var(f"?{t[0] if t else 'x'}_{i}", t) for i, t in enumerate(arg_types)
            )
            return replace(self, current_hole="aux_var",
                           partial=replace(p, schema=schema_name, action_args=action_args))

        if kind == "aux_var":
            if tag == "skip_aux":
                return replace(self, current_hole="pre_lit")
            if tag == "add_aux":                  # ("add_aux", type_name)
                t = prod.payload[1]
                new_aux = p.aux_vars + (Var(aux_var_name(len(p.aux_vars)), t),)
                # Stage 2: max_aux_vars == 1, so go straight to pre_lit.
                return replace(self, current_hole="pre_lit", partial=replace(p, aux_vars=new_aux))

        if kind == "pre_lit":
            if tag == "add":                      # ("add", literal)
                lit = prod.payload[1]
                return replace(self, partial=replace(p, state_lits=p.state_lits + (lit,)))
            if tag == "stop_pre":
                return replace(self, current_hole="goal_lit")

        if kind == "goal_lit":
            if tag == "add":                      # ("add", literal)
                lit = prod.payload[1]
                return replace(self, partial=replace(p, goal_lits=p.goal_lits + (lit,)))
            if tag == "finish":
                rule = Rule(
                    vars=p.all_vars(),
                    body=tuple(p.state_lits + p.goal_lits),
                    action=LiftedAction(p.schema, p.action_args),
                )
                return replace(self, completed_rules=self.completed_rules + (rule,),
                               partial=None, current_hole="policy")

        raise ValueError(f"cannot apply production {prod!r} in hole {self.current_hole!r}")

    # -- queries --
    def is_terminal(self) -> bool:
        return self.current_hole is None

    def to_program(self) -> Policy:
        return Policy(self.completed_rules)

    def pretty(self) -> str:
        lines = [f"ρ_{i + 1}: {r.pretty()}" for i, r in enumerate(self.completed_rules)]
        if self.partial is not None:
            p = self.partial
            body = " ∧ ".join(l.pretty() for l in (p.state_lits + p.goal_lits)) or "⊤"
            act = (f"{p.schema}({', '.join(v.name for v in p.action_args)})"
                   if p.schema else "?")
            # variable list carries types: an auxiliary variable's name (?aux_0)
            # alone is type-ambiguous (ball vs room), and two such states have
            # different production sets — so the type must be in the node key.
            vsig = ",".join(f"{v.name}:{v.type_name}" for v in (p.action_args + p.aux_vars))
            lines.append(f"<@{self.current_hole}: {body} ⇒ {act} | vars=[{vsig}]>")
        elif self.current_hole == "policy":
            lines.append("<@policy>")
        # current_hole is None -> terminal: no extra marker
        return "\n".join(lines) if lines else "<empty>"


# ---------------------------------------------------------------------------
# Game wrapper
# ---------------------------------------------------------------------------

class LiftedDerivationGame(Game):
    """Single-player game: actions are lifted-grammar productions applied to
    the current hole; terminal reward = ``leaf_evaluator(policy)``.

    Mirrors :class:`alphazeropp.synthesis.derivation_game.DerivationGame` but
    over the lifted grammar. The core MCTS engine and ``UniformPolicyValueNet``
    are reused unchanged.
    """

    def __init__(
        self,
        cfg: LiftedGrammarConfig,
        sig: DomainSignature,
        leaf_evaluator,
    ):
        super().__init__()
        self.cfg = cfg
        self.sig = sig
        self.leaf_evaluator = leaf_evaluator
        self._max_productions = compute_max_productions(cfg, sig)
        self._obs_len = encode_max_len(cfg)
        self.action_space = spaces.Discrete(self._max_productions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_len,), dtype=np.float32,
        )
        self._deriv_state: LiftedDerivationState | None = None
        self._current_productions: list[LiftedProduction] = []

    # -- internals --
    def _legal(self) -> list[LiftedProduction]:
        return enumerate_productions(self._deriv_state, self.cfg, self.sig)

    def _encode(self) -> np.ndarray:
        return encode_state(self._deriv_state, self.sig, self._obs_len)

    # -- Game protocol --
    def reset(self, **kwargs):
        self._deriv_state = LiftedDerivationState.initial()
        self._current_productions = self._legal()
        return self._encode(), {}

    def step(self, action: int):
        prod = self._current_productions[action]
        self._deriv_state = self._deriv_state.apply(prod)
        self._current_productions = self._legal()

        is_complete = self._deriv_state.is_terminal()
        is_dead_end = (not is_complete) and len(self._current_productions) == 0
        terminated = is_complete
        truncated = is_dead_end
        info: dict[str, Any] = {
            "production": prod.label,
            "hole_kind": prod.hole_kind,
            "branching": len(self._current_productions),
            "is_complete": is_complete,
            "is_dead_end": is_dead_end,
        }
        if is_complete:
            program = self._deriv_state.to_program()
            reward = float(self.leaf_evaluator(program))
            info["program"] = program
            info["leaf_value"] = reward
        else:
            reward = 0.0
        return self._encode(), reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        n = len(self._current_productions)
        if n > self._max_productions:
            raise RuntimeError(
                f"{n} legal productions exceeds action space {self._max_productions} "
                f"in hole {self._deriv_state.current_hole!r} — compute_max_productions is wrong"
            )
        mask = np.zeros(self._max_productions, dtype=bool)
        mask[:n] = True
        return mask

    @property
    def hashable_obs(self) -> str:
        return self._deriv_state.pretty()

    def get_program(self) -> Policy | None:
        if self._deriv_state is not None and self._deriv_state.is_terminal():
            return self._deriv_state.to_program()
        return None

    # -- stash / unstash (avoid deepcopy-ing the evaluator) --
    def stash_state(self) -> tuple:
        return (
            self._deriv_state,
            self._current_productions,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        )

    def unstash_state(self, state: tuple) -> "LiftedDerivationGame":
        (
            self._deriv_state,
            self._current_productions,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> "LiftedDerivationGame":
        new = LiftedDerivationGame(self.cfg, self.sig, self.leaf_evaluator)
        new.unstash_state(self.stash_state())
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new
