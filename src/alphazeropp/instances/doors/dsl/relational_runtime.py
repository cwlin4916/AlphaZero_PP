"""Relational compile-time query interface for the Doors environment.

Provides typed entity resolution and observation-index mapping that the
lifted DSL compiler uses to generate raw AST.  All queries are pure
functions of DoorsGameConfig — no runtime state inspection.

This module is the SOLE interface between the lifted compiler and
DoorsGameConfig.  The lifted compiler should never read cfg.key_loc,
cfg.key_unlocks, cfg.loc_room, or cfg.goal_loc directly.

Two modes:
  known_map  (default) — key locations are known from the instance
                         descriptor; loc_of_key() returns a concrete LocId.
  partial_map          — key locations are unknown until observed;
                         loc_of_key() returns None (⊥).
"""

from __future__ import annotations

from typing import NewType

from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig
from alphazeropp.synthesis.ast_nodes import (
    Flip, IsZero, Not, And,
    Condition,
)


# ---------------------------------------------------------------------------
# Typed entity aliases (NewType — zero runtime cost)
# ---------------------------------------------------------------------------

RoomId = NewType("RoomId", int)   # room index 0..D-1
KeyId  = NewType("KeyId", int)    # key index 0..K-1
LocId  = NewType("LocId", int)    # location index 0..M-1
ObsIdx = NewType("ObsIdx", int)   # raw observation vector index


# ---------------------------------------------------------------------------
# DoorsRelationalRuntime
# ---------------------------------------------------------------------------

class DoorsRelationalRuntime:
    """Compile-time relational query interface over a DoorsGameConfig.

    Constructed once per lifted compilation.  All methods are pure and
    deterministic.

    Parameters
    ----------
    cfg : DoorsGameConfig
        The game configuration to wrap.
    known_map : bool
        If True (default), key locations are available from ``cfg.key_loc``.
        If False, ``loc_of_key`` and derived methods return None.
    """

    __slots__ = ("cfg", "known_map", "_room_to_key")

    def __init__(self, cfg: DoorsGameConfig, *, known_map: bool = True):
        self.cfg = cfg
        self.known_map = known_map

        # Build inverse map: room r → key k such that key_unlocks[k] == r
        inv: dict[int, int] = {}
        for k, r in enumerate(cfg.key_unlocks):
            inv[r] = k
        self._room_to_key = inv

    # -------------------------------------------------------------------
    # Entity enumeration
    # -------------------------------------------------------------------

    def lockable_rooms(self) -> list[RoomId]:
        """Rooms that require keys, in canonical order [1, 2, ..., D-1].

        Room 0 is always unlocked.  The lifted compiler iterates over
        these when expanding ``NextLockedRoom`` selectors.
        """
        return [RoomId(r) for r in range(1, self.cfg.D)]

    def all_locations(self) -> list[LocId]:
        """All location indices [0, 1, ..., M-1]."""
        return [LocId(l) for l in range(self.cfg.M)]

    def locations_in_room(self, r: RoomId) -> list[LocId]:
        """Locations belonging to room *r*."""
        return [
            LocId(l)
            for l, rm in enumerate(self.cfg.loc_room)
            if rm == int(r)
        ]

    def num_keys(self) -> int:
        """Number of keys (K = D - 1)."""
        return self.cfg.K

    def num_rooms(self) -> int:
        """Number of rooms (D)."""
        return self.cfg.D

    # -------------------------------------------------------------------
    # Relational lookups (config queries)
    # -------------------------------------------------------------------

    def key_for_room(self, r: RoomId) -> KeyId | None:
        """Which key unlocks room *r*?  None if room 0 (always open)."""
        k = self._room_to_key.get(int(r))
        return KeyId(k) if k is not None else None

    def room_unlocked_by(self, k: KeyId) -> RoomId:
        """Which room does key *k* unlock?"""
        _check_key_id(k, self.cfg)
        return RoomId(self.cfg.key_unlocks[int(k)])

    def loc_of_key(self, k: KeyId) -> LocId | None:
        """Location of key *k*.

        Returns None in ``partial_map`` mode (key location unknown).
        """
        _check_key_id(k, self.cfg)
        if not self.known_map:
            return None
        return LocId(self.cfg.key_loc[int(k)])

    def goal_location(self) -> LocId:
        """The goal location."""
        return LocId(self.cfg.goal_loc)

    def room_of_location(self, l: LocId) -> RoomId:
        """Which room contains location *l*?"""
        _check_loc_id(l, self.cfg)
        return RoomId(self.cfg.loc_room[int(l)])

    # -------------------------------------------------------------------
    # Observation index queries
    # -------------------------------------------------------------------

    def obs_at_loc(self, l: LocId) -> ObsIdx:
        """Obs index for 'agent is at location *l*': ``state[l]``."""
        return ObsIdx(int(l))

    def obs_room_unlocked(self, r: RoomId) -> ObsIdx:
        """Obs index for 'room *r* is unlocked': ``state[M + r]``."""
        _check_room_id(r, self.cfg)
        return ObsIdx(self.cfg.M + int(r))

    def obs_key_avail(self, k: KeyId) -> ObsIdx:
        """Obs index for 'key *k* is available': ``state[M + D + k]``."""
        _check_key_id(k, self.cfg)
        return ObsIdx(self.cfg.M + self.cfg.D + int(k))

    # -------------------------------------------------------------------
    # Action index queries
    # -------------------------------------------------------------------

    def action_move_to(self, l: LocId) -> int:
        """Flip index for MOVE_TO(l)."""
        return int(l)

    def action_pick(self, k: KeyId) -> int:
        """Flip index for PICK(k): ``M + k``."""
        _check_key_id(k, self.cfg)
        return self.cfg.M + int(k)

    # -------------------------------------------------------------------
    # AST fragment builders
    # -------------------------------------------------------------------

    def cond_at_loc(self, l: LocId) -> Condition:
        """AST condition: agent is at location *l*."""
        return Not(IsZero(self.obs_at_loc(l)))

    def cond_room_locked(self, r: RoomId) -> Condition:
        """AST condition: room *r* is locked."""
        return IsZero(self.obs_room_unlocked(r))

    def cond_key_avail(self, k: KeyId) -> Condition:
        """AST condition: key *k* is available."""
        return Not(IsZero(self.obs_key_avail(k)))

    def cond_pick_ready(self, k: KeyId) -> Condition | None:
        """AST condition: agent at key *k*'s location AND key available.

        Returns None in ``partial_map`` mode (key location unknown).
        """
        loc = self.loc_of_key(k)
        if loc is None:
            return None
        return And(self.cond_at_loc(loc), self.cond_key_avail(k))

    def cond_need_key(self, k: KeyId) -> Condition:
        """AST condition: the room that key *k* unlocks is still locked."""
        r = self.room_unlocked_by(k)
        return self.cond_room_locked(r)

    def flip_pick(self, k: KeyId) -> Flip:
        """AST action: pick up key *k*."""
        return Flip(self.action_pick(k))

    def flip_move_to_key(self, k: KeyId) -> Flip | None:
        """AST action: move to key *k*'s location.

        Returns None in ``partial_map`` mode (key location unknown).
        """
        loc = self.loc_of_key(k)
        if loc is None:
            return None
        return Flip(self.action_move_to(loc))

    def flip_move_to_goal(self) -> Flip:
        """AST action: move to goal location."""
        return Flip(self.action_move_to(self.goal_location()))

    # -------------------------------------------------------------------
    # Stage 3.1: typed-binding queries
    # -------------------------------------------------------------------

    def objects_of_type(self, t: str) -> list[int]:
        if t == "room":
            return list(range(self.cfg.D))
        if t == "key":
            return list(range(self.cfg.K))
        if t == "loc":
            return list(range(self.cfg.M))
        raise ValueError(f"Unknown type: {t!r}")

    def is_next(self, r1: int, r2: int) -> bool:
        return 0 <= r1 < self.cfg.D - 1 and r2 == r1 + 1

    def query_predicate(
        self,
        name: str,
        args: tuple[int, ...],
        state,
    ) -> bool:
        if name == "at_loc":
            (l,) = args
            return bool(state[self.obs_at_loc(LocId(l))]) and 0 <= l < self.cfg.M
        if name == "locked":
            (r,) = args
            if r == 0 or not (0 <= r < self.cfg.D):
                return False
            return not bool(state[self.obs_room_unlocked(RoomId(r))])
        if name == "key_available":
            (k,) = args
            if not (0 <= k < self.cfg.K):
                return False
            return bool(state[self.obs_key_avail(KeyId(k))])
        if name == "key_for":
            k, r = args
            return 0 <= k < self.cfg.K and self.cfg.key_unlocks[k] == r
        if name == "loc_of_key":
            k, l = args
            return 0 <= k < self.cfg.K and self.cfg.key_loc[k] == l
        if name == "loc_in_room":
            l, r = args
            return 0 <= l < self.cfg.M and self.cfg.loc_room[l] == r
        if name == "next":
            r1, r2 = args
            return self.is_next(r1, r2)
        if name == "goal_loc":
            (l,) = args
            return l == self.cfg.goal_loc
        raise ValueError(f"Unknown predicate: {name!r}")

    def permute_objects(
        self,
        perm_rooms: dict[int, int] | None = None,
        perm_keys: dict[int, int] | None = None,
        perm_locs: dict[int, int] | None = None,
    ) -> "DoorsRelationalRuntime":
        # Test-only helper: relabel object IDs while preserving static-fact structure.
        # The returned runtime answers the same relational queries up to renaming.
        from alphazeropp.instances.doors.dsl.doors_config import DoorsGameConfig

        pr = perm_rooms or {i: i for i in range(self.cfg.D)}
        pk = perm_keys or {i: i for i in range(self.cfg.K)}
        pl = perm_locs or {i: i for i in range(self.cfg.M)}

        new_loc_room = [0] * self.cfg.M
        for old_l, old_r in enumerate(self.cfg.loc_room):
            new_loc_room[pl[old_l]] = pr[old_r]

        new_key_loc = [0] * self.cfg.K
        new_key_unlocks = [0] * self.cfg.K
        for old_k in range(self.cfg.K):
            new_key_loc[pk[old_k]] = pl[self.cfg.key_loc[old_k]]
            new_key_unlocks[pk[old_k]] = pr[self.cfg.key_unlocks[old_k]]

        new_cfg = DoorsGameConfig(
            num_rooms=self.cfg.D,
            locs_per_room=self.cfg.locs_per_room,
            horizon=self.cfg.horizon,
            key_loc=new_key_loc,
            key_unlocks=new_key_unlocks,
            loc_room=new_loc_room,
            goal_loc=pl[self.cfg.goal_loc],
            start_loc=pl[self.cfg.start_loc],
        )
        return DoorsRelationalRuntime(new_cfg, known_map=self.known_map)

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def validate(self) -> None:
        """Check internal consistency of the config mapping.

        Raises AssertionError on any inconsistency.
        """
        cfg = self.cfg
        # Every key_unlocks[k] is a valid room > 0
        for k in range(cfg.K):
            r = cfg.key_unlocks[k]
            assert 0 < r < cfg.D, (
                f"key_unlocks[{k}]={r} not in (0, {cfg.D})"
            )
        # Inverse map covers all lockable rooms
        for r in range(1, cfg.D):
            assert r in self._room_to_key, f"No key unlocks room {r}"
        # Key locations in valid range (if known_map)
        if self.known_map:
            for k in range(cfg.K):
                loc = cfg.key_loc[k]
                assert 0 <= loc < cfg.M, (
                    f"key_loc[{k}]={loc} not in [0, {cfg.M})"
                )


# ---------------------------------------------------------------------------
# Bounds checking helpers
# ---------------------------------------------------------------------------

def _check_key_id(k: KeyId, cfg: DoorsGameConfig) -> None:
    if not (0 <= int(k) < cfg.K):
        raise ValueError(f"KeyId {k} out of range [0, {cfg.K})")


def _check_room_id(r: RoomId, cfg: DoorsGameConfig) -> None:
    if not (0 <= int(r) < cfg.D):
        raise ValueError(f"RoomId {r} out of range [0, {cfg.D})")


def _check_loc_id(l: LocId, cfg: DoorsGameConfig) -> None:
    if not (0 <= int(l) < cfg.M):
        raise ValueError(f"LocId {l} out of range [0, {cfg.M})")
