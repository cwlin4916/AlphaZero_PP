"""ExplicitSurfaceDerivationGame: grammar-production-based derivation game.

Identical interface to SurfaceDerivationGame, but derives legal actions
from SurfaceCFG productions instead of bitmask logic. This proves that
the explicit grammar is a faithful, playable representation of the
masked generator.

Fixed-length episodes of exactly 2K+1 steps, no dead ends.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium.spaces as spaces
import numpy as np

from alphazeropp.core.game import Game
from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.instances.doors.dsl.explicit_surface_cfg import (
    SurfaceCFG, Nonterminal,
)
from alphazeropp.instances.doors.dsl.surface_dsl import (
    PickRule, MoveRule, GoalRule, SurfaceRule, SurfacePolicy,
)
from alphazeropp.instances.doors.dsl.surface_compiler import compile_policy
from alphazeropp.instances.doors.dsl.surface_derivation_game import (
    SURFACE_TOKEN_IDS,
)
from alphazeropp.synthesis.leaf_evaluator import LeafEvaluator


class ExplicitSurfaceDerivationGame(Game):
    """Single-player game where actions are grammar productions.

    Action layout (identical to SurfaceDerivationGame):
      0..K-1    -> PickRule(k)
      K..2K-1   -> MoveRule(k-K)
      2K        -> GoalRule

    Legal mask derived from SurfaceCFG.legal_actions(current_nonterminal).
    """

    def __init__(
        self,
        num_rooms: int,
        leaf_evaluator: LeafEvaluator,
        doors_cfg: DoorsGameConfig,
        *,
        allow_early_goal: bool = False,
    ):
        super().__init__()
        self.num_rooms = num_rooms
        self.K = num_rooms - 1
        self.leaf_evaluator = leaf_evaluator
        self.doors_cfg = doors_cfg
        self.allow_early_goal = allow_early_goal

        self._grammar = SurfaceCFG(self.K)
        self._max_steps = 2 * self.K + 1
        self._n_actions = self._max_steps

        self.action_space = spaces.Discrete(self._n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * self._max_steps,), dtype=np.float32,
        )

        self._nt: Nonterminal | None = None
        self._rules: list[SurfaceRule] = []

    # -- Action <-> Rule mapping (same as SurfaceDerivationGame) --

    def _action_to_rule(self, action: int) -> SurfaceRule:
        if action < self.K:
            return PickRule(action)
        elif action < 2 * self.K:
            return MoveRule(action - self.K)
        else:
            return GoalRule()

    def _rule_to_action(self, rule: SurfaceRule) -> int:
        if isinstance(rule, PickRule):
            return rule.k
        elif isinstance(rule, MoveRule):
            return self.K + rule.k
        else:
            return 2 * self.K

    # -- Game interface --

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self._nt = self._grammar.start
        self._rules = []
        obs = self._encode_obs()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        rule = self._action_to_rule(action)
        self._rules.append(rule)

        next_nt = self._grammar.successor(self._nt, rule)
        is_terminal = next_nt is None

        info: dict[str, Any] = {
            "rule": rule,
            "n_rules_placed": len(self._rules),
        }

        if is_terminal:
            policy = SurfacePolicy(tuple(self._rules))
            program = compile_policy(policy, self.doors_cfg)
            reward = self.leaf_evaluator(program)
            self.leaf_evaluator._surface_labels[program.pretty()] = policy.pretty()
            info["program"] = program
            info["policy"] = policy
            info["leaf_value"] = reward
        else:
            self._nt = next_nt
            reward = 0.0

        obs = self._encode_obs()
        return obs, reward, is_terminal, False, info

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self._n_actions, dtype=bool)
        for rule in self._grammar.legal_actions(self._nt):
            mask[self._rule_to_action(rule)] = True
        return mask

    # -- Observation encoding (identical to SurfaceDerivationGame) --

    def _encode_obs(self) -> np.ndarray:
        obs = np.zeros(2 * self._max_steps, dtype=np.float32)
        for i, rule in enumerate(self._rules):
            if isinstance(rule, PickRule):
                obs[2 * i] = SURFACE_TOKEN_IDS["PICK"]
                obs[2 * i + 1] = float(rule.k)
            elif isinstance(rule, MoveRule):
                obs[2 * i] = SURFACE_TOKEN_IDS["MOVE"]
                obs[2 * i + 1] = float(rule.k)
            elif isinstance(rule, GoalRule):
                obs[2 * i] = SURFACE_TOKEN_IDS["GOAL"]
                obs[2 * i + 1] = 0.0
        return obs

    # -- Hashable obs --

    @property
    def hashable_obs(self) -> tuple:
        return tuple(self._rules)

    # -- Stash / Unstash --

    def stash_state(self) -> tuple:
        return (
            self._nt,
            list(self._rules),
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        )

    def unstash_state(self, state: tuple):
        (
            self._nt,
            self._rules,
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.info,
            self.step_count,
        ) = state
        return self

    def clone(self) -> ExplicitSurfaceDerivationGame:
        new = ExplicitSurfaceDerivationGame(
            self.num_rooms, self.leaf_evaluator, self.doors_cfg,
            allow_early_goal=self.allow_early_goal,
        )
        new.unstash_state(self.stash_state())
        new._rules = list(self._rules)  # independent copy
        if self.obs is not None:
            new.obs = self.obs.copy()
        return new
