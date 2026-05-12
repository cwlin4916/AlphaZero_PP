"""Relational adapter exposing :class:`DoorsPDDLLiteEnv` through the lifted env contract.

``DoorsPDDLLiteEnv`` ([doors_pddl_lite.py]) is a flat-vector gymnasium env wired into the
AlphaZero benchmark harness — ``spaces.Box`` observations, ``spaces.Discrete`` actions, integer
object ids. This wrapper presents the *same* relational interface ``GripperLiteEnv`` presents
(:mod:`alphazeropp.synthesis.lifted_dsl` ``RelState`` / ``GroundAction`` over string object
names), so the generic :func:`alphazeropp.synthesis.lifted_interpreter.interpret` runs unchanged
on the Doors domain. See ``docs/notes/stage4/01_plan.md`` (Stage 1 rev. 1).

It does **not** touch ``doors_pddl_lite.py``; the gym interface is untouched and reachable via
``.base``. The underlying env keeps its integer ids internally — only the surface object names are
strings, and they are configurable (``name_fmt``) so a relabeled instance can be built for the
object-renaming test.

Domain vocabulary
-----------------
========  ===================================================================
Types     ``room``, ``location``, ``key``
Fluents   ``at_loc(location)``, ``unlocked(room)``, ``key_avail(key)``
Static    ``loc_in_room(location, room)``, ``key_at(key, location)``, ``key_unlocks(key, room)``
Actions   ``move_to(location)``, ``pick(key)``, ``noop()``
Goal      ``{ at_loc(<goal location>) }``
========  ===================================================================

The static relations (``loc_in_room`` / ``key_at`` / ``key_unlocks``) are the persistent
instance structure that ``DoorsPDDLLiteEnv`` stores as ``loc_room`` / ``key_loc`` /
``key_unlocks``; in PDDL terms they are static predicates of the problem. The orientation doc
§4.2 sketched only the three fluents and flagged the contract "not yet implemented"; wiring it for
real adds the statics so meaningful lifted Doors rules are expressible.
"""

from __future__ import annotations

from typing import Optional

from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.synthesis.lifted_dsl import (
    ActionSchema,
    GroundAction,
    PredicateSchema,
    RelState,
)


# ---------------------------------------------------------------------------
# Schemas (consumed by alphazeropp.synthesis.lifted_grammar.doors_pddl_signature)
# ---------------------------------------------------------------------------

PREDICATE_SCHEMAS: tuple[PredicateSchema, ...] = (
    PredicateSchema("at_loc", ("location",)),
    PredicateSchema("unlocked", ("room",)),
    PredicateSchema("key_avail", ("key",)),
    PredicateSchema("loc_in_room", ("location", "room")),
    PredicateSchema("key_at", ("key", "location")),
    PredicateSchema("key_unlocks", ("key", "room")),
)

ACTION_SCHEMAS: tuple[ActionSchema, ...] = (
    ActionSchema("move_to", ("location",)),
    ActionSchema("pick", ("key",)),
    ActionSchema("noop", ()),
)

#: Default surface-name templates per type. ``"{}"`` is filled with the integer id.
DEFAULT_NAME_FMT: dict[str, str] = {
    "location": "loc_{}",
    "room": "room_{}",
    "key": "key_{}",
}


# ---------------------------------------------------------------------------
# Module-level name helpers (the *default* naming; the adapter may override it)
# ---------------------------------------------------------------------------

def loc_name(l: int) -> str:
    return DEFAULT_NAME_FMT["location"].format(l)


def room_name(r: int) -> str:
    return DEFAULT_NAME_FMT["room"].format(r)


def key_name(k: int) -> str:
    return DEFAULT_NAME_FMT["key"].format(k)


# ---------------------------------------------------------------------------
# DoorsPDDLLiteRelationalEnv
# ---------------------------------------------------------------------------

class DoorsPDDLLiteRelationalEnv:
    """Relational view of a :class:`DoorsPDDLLiteEnv`.

    Construct directly with an existing base env, or via :meth:`make_d2` / :meth:`make_d3` which
    build the preset layout. The base env is reset on construction so ``get_state_atoms()`` is
    valid immediately (mirrors :class:`GripperLiteEnv`).
    """

    def __init__(
        self,
        base: DoorsPDDLLiteEnv,
        *,
        name_fmt: Optional[dict[str, str]] = None,
    ):
        self._base = base

        fmt = dict(DEFAULT_NAME_FMT)
        if name_fmt:
            fmt.update(name_fmt)
        self._name_fmt = fmt

        self._loc_name: tuple[str, ...] = tuple(fmt["location"].format(i) for i in range(base.M))
        self._room_name: tuple[str, ...] = tuple(fmt["room"].format(i) for i in range(base.D))
        self._key_name: tuple[str, ...] = tuple(fmt["key"].format(i) for i in range(base.K))
        self._loc_id: dict[str, int] = {n: i for i, n in enumerate(self._loc_name)}
        self._room_id: dict[str, int] = {n: i for i, n in enumerate(self._room_name)}
        self._key_id: dict[str, int] = {n: i for i, n in enumerate(self._key_name)}

        self._base.reset()

    # -- constructors --

    @classmethod
    def make_d2(cls, *, name_fmt: Optional[dict[str, str]] = None, **kwargs) -> "DoorsPDDLLiteRelationalEnv":
        """D=2, M=4, K=1 preset (key at loc_1, goal loc_3)."""
        return cls(DoorsPDDLLiteEnv.make_d2(**kwargs), name_fmt=name_fmt)

    @classmethod
    def make_d3(cls, *, name_fmt: Optional[dict[str, str]] = None, **kwargs) -> "DoorsPDDLLiteRelationalEnv":
        """D=3, M=6, K=2 preset (sequential unlock, goal loc_5)."""
        return cls(DoorsPDDLLiteEnv.make_d3(**kwargs), name_fmt=name_fmt)

    # -- passthrough properties --

    @property
    def base(self) -> DoorsPDDLLiteEnv:
        return self._base

    @property
    def horizon(self) -> int:
        return self._base.horizon

    @property
    def step_count(self) -> int:
        return self._base.step_count

    # ------------------------------------------------------- relational API

    def get_state_atoms(self) -> RelState:
        b = self._base
        s = b._state
        uoff, koff = b._unlocked_offset, b._key_offset
        at_loc = {(self._loc_name[l],) for l in range(b.M) if s[l] == 1.0}
        unlocked = {(self._room_name[r],) for r in range(b.D) if s[uoff + r] == 1.0}
        key_avail = {(self._key_name[k],) for k in range(b.K) if s[koff + k] == 1.0}
        loc_in_room = {
            (self._loc_name[l], self._room_name[b.loc_room[l]]) for l in range(b.M)
        }
        key_at = {(self._key_name[k], self._loc_name[b.key_loc[k]]) for k in range(b.K)}
        key_unlocks = {(self._key_name[k], self._room_name[b.key_unlocks[k]]) for k in range(b.K)}
        return {
            "at_loc": at_loc,
            "unlocked": unlocked,
            "key_avail": key_avail,
            "loc_in_room": loc_in_room,
            "key_at": key_at,
            "key_unlocks": key_unlocks,
        }

    def get_goal_atoms(self) -> RelState:
        return {
            "at_loc": {(self._loc_name[self._base.goal_loc],)},
            "unlocked": set(),
            "key_avail": set(),
            "loc_in_room": set(),
            "key_at": set(),
            "key_unlocks": set(),
        }

    def get_objects_by_type(self) -> dict[str, tuple[str, ...]]:
        return {
            "room": self._room_name,
            "location": self._loc_name,
            "key": self._key_name,
        }

    def legal_actions(self) -> set[GroundAction]:
        b = self._base
        s = b._state
        uoff, koff = b._unlocked_offset, b._key_offset
        actions: set[GroundAction] = {GroundAction("noop", ())}
        for l in range(b.M):
            if s[uoff + b.loc_room[l]] == 1.0:
                actions.add(GroundAction("move_to", (self._loc_name[l],)))
        for k in range(b.K):
            if s[b.key_loc[k]] == 1.0 and s[koff + k] == 1.0:
                actions.add(GroundAction("pick", (self._key_name[k],)))
        return actions

    # ------------------------------------------------------- episodic API

    def reset(self, seed: int | None = None) -> RelState:
        self._base.reset(seed=seed)
        return self.get_state_atoms()

    def is_solved(self) -> bool:
        return bool(self._base.is_solved(self._base.state))

    def step(self, action: GroundAction) -> tuple[RelState, float, bool, dict]:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action.pretty()}")
        b = self._base
        if action.schema == "move_to":
            idx = b.encode_action("move", self._loc_id[action.args[0]])
        elif action.schema == "pick":
            idx = b.encode_action("pick", self._key_id[action.args[0]])
        elif action.schema == "noop":
            idx = b.encode_action("noop")
        else:  # pragma: no cover - legal_actions already filters these out
            raise ValueError(f"unknown schema: {action.schema}")
        _obs, reward, terminated, truncated, info = b.step(idx)
        solved = self.is_solved()
        return (
            self.get_state_atoms(),
            float(reward),
            bool(terminated or truncated),
            {**info, "solved": solved},
        )
