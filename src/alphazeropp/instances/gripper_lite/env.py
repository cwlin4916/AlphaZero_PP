"""Minimal Gripper-lite relational environment.

Pure relational env (no gym.Env subclass, no flat numpy obs) used by Stage 1
of the lifted-policy program-synthesis project to exercise the
``synthesis/lifted_interpreter.py`` core. See
``docs/notes/stage4/01_plan.md`` for the design contract.

Domain:
    Types:        ball, room.
    Predicates:   at_robot(room), at_ball(ball, room), carrying(ball), handempty().
    Actions:      move(room, room), pick(ball, room), drop(ball, room).
    Goal:         every ball at room_b.
"""

from __future__ import annotations

from copy import deepcopy

from alphazeropp.synthesis.lifted_dsl import (
    ActionSchema,
    GroundAction,
    PredicateSchema,
    RelState,
)


ROOM_NAMES_DEFAULT: tuple[str, ...] = ("room_a", "room_b")

PREDICATE_SCHEMAS: tuple[PredicateSchema, ...] = (
    PredicateSchema("at_robot", ("room",)),
    PredicateSchema("at_ball", ("ball", "room")),
    PredicateSchema("carrying", ("ball",)),
    PredicateSchema("handempty", ()),
)

ACTION_SCHEMAS: tuple[ActionSchema, ...] = (
    ActionSchema("move", ("room", "room")),
    ActionSchema("pick", ("ball", "room")),
    ActionSchema("drop", ("ball", "room")),
)


class GripperLiteEnv:
    def __init__(
        self,
        n_balls: int = 1,
        seed: int | None = None,
        rooms: tuple[str, str] = ROOM_NAMES_DEFAULT,
        ball_names: tuple[str, ...] | None = None,
    ):
        if n_balls < 1:
            raise ValueError("n_balls must be >= 1")
        if len(rooms) != 2:
            raise ValueError("gripper_lite has exactly 2 rooms")
        self.n_balls = n_balls
        self.rooms = tuple(rooms)
        self.balls = (
            tuple(ball_names)
            if ball_names is not None
            else tuple(f"ball_{i}" for i in range(n_balls))
        )
        if len(self.balls) != n_balls:
            raise ValueError("ball_names length must equal n_balls")
        self.horizon = 4 * n_balls + 4
        self._seed = seed
        self.reset(seed=seed)

    # ------------------------------------------------------------------ state

    def _initial_state(self) -> RelState:
        room_start = self.rooms[0]
        return {
            "at_robot": {(room_start,)},
            "at_ball": {(b, room_start) for b in self.balls},
            "carrying": set(),
            "handempty": {()},
        }

    def _initial_goal(self) -> RelState:
        room_target = self.rooms[1]
        return {
            "at_robot": set(),
            "at_ball": {(b, room_target) for b in self.balls},
            "carrying": set(),
            "handempty": set(),
        }

    def reset(
        self,
        seed: int | None = None,
        n_balls: int | None = None,
    ) -> RelState:
        if n_balls is not None and n_balls != self.n_balls:
            self.n_balls = n_balls
            self.balls = tuple(f"ball_{i}" for i in range(n_balls))
            self.horizon = 4 * n_balls + 4
        if seed is not None:
            self._seed = seed
        self.state: RelState = self._initial_state()
        self.goal: RelState = self._initial_goal()
        self.step_count = 0
        return deepcopy(self.state)

    # ------------------------------------------------------- relational API

    def get_state_atoms(self) -> RelState:
        return deepcopy(self.state)

    def get_goal_atoms(self) -> RelState:
        return deepcopy(self.goal)

    def get_objects_by_type(self) -> dict[str, tuple[str, ...]]:
        return {"ball": self.balls, "room": self.rooms}

    def legal_actions(self) -> set[GroundAction]:
        actions: set[GroundAction] = set()
        # move(from, to)
        ((robot_room,),) = tuple(self.state["at_robot"])  # exactly one
        for to in self.rooms:
            if to != robot_room:
                actions.add(GroundAction("move", (robot_room, to)))
        # pick(b, r): robot at r, ball at r, handempty
        if self.state["handempty"]:
            for b, r in self.state["at_ball"]:
                if r == robot_room:
                    actions.add(GroundAction("pick", (b, r)))
        # drop(b, r): robot at r, carrying b
        for (b,) in self.state["carrying"]:
            actions.add(GroundAction("drop", (b, robot_room)))
        return actions

    # ------------------------------------------------------- episodic API

    def is_solved(self) -> bool:
        return self.state["at_ball"] == self.goal["at_ball"]

    def step(self, action: GroundAction) -> tuple[RelState, float, bool, dict]:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action.pretty()}")

        reward = -1.0
        if action.schema == "move":
            _from, to = action.args
            self.state["at_robot"] = {(to,)}
        elif action.schema == "pick":
            b, r = action.args
            self.state["at_ball"].discard((b, r))
            self.state["carrying"].add((b,))
            self.state["handempty"].clear()
        elif action.schema == "drop":
            b, r = action.args
            self.state["carrying"].discard((b,))
            self.state["at_ball"].add((b, r))
            self.state["handempty"].add(())
        else:
            raise ValueError(f"unknown schema: {action.schema}")

        self.step_count += 1
        solved = self.is_solved()
        if solved:
            reward += 100.0
        done = solved or self.step_count >= self.horizon
        return deepcopy(self.state), reward, done, {"solved": solved}
