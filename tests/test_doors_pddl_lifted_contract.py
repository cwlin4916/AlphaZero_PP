"""Stage 1 (rev. 1) — the Doors PDDL relational adapter contract.

``DoorsPDDLLiteRelationalEnv`` (src/alphazeropp/instances/doors/doors_pddl_lifted.py) must expose
the same lifted env contract ``GripperLiteEnv`` exposes, over the Doors vocabulary, *without*
touching the existing flat-vector / gymnasium ``DoorsPDDLLiteEnv``. See
``docs/notes/stage4/01_plan.md``.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from alphazeropp.instances.doors import doors_pddl_lifted, doors_pddl_policies
from alphazeropp.instances.doors.doors_pddl_lifted import (
    DoorsPDDLLiteRelationalEnv,
    key_name,
    loc_name,
)
from alphazeropp.instances.doors.doors_pddl_lite import DoorsPDDLLiteEnv
from alphazeropp.synthesis.lifted_dsl import GroundAction
from alphazeropp.synthesis.lifted_grammar import doors_pddl_signature


_FLUENT_ARITY = {"at_loc": 1, "unlocked": 1, "key_avail": 1, "loc_in_room": 2, "key_at": 2, "key_unlocks": 2}
_ARG_TYPES = {
    "at_loc": ("location",),
    "unlocked": ("room",),
    "key_avail": ("key",),
    "loc_in_room": ("location", "room"),
    "key_at": ("key", "location"),
    "key_unlocks": ("key", "room"),
}


# ---------------------------------------------------------------------------
# item a — state atoms non-empty and typed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker,D,M,K", [("make_d2", 2, 4, 1), ("make_d3", 3, 6, 2)])
def test_state_atoms_nonempty_and_typed(maker, D, M, K):
    env = getattr(DoorsPDDLLiteRelationalEnv, maker)()
    state = env.get_state_atoms()
    objs = env.get_objects_by_type()

    # every declared predicate is present with the right arity
    assert set(state) == set(_FLUENT_ARITY)
    for pred, arity in _FLUENT_ARITY.items():
        for tup in state[pred]:
            assert len(tup) == arity, (pred, tup)
            for obj, ty in zip(tup, _ARG_TYPES[pred]):
                assert obj in objs[ty], (pred, obj, ty)

    # fluents at reset: exactly one location, room_0 unlocked, all K keys available
    assert state["at_loc"] == {("loc_0",)}
    assert ("room_0",) in state["unlocked"]
    assert len(state["unlocked"]) == 1                       # only room_0 unlocked at reset
    assert len(state["key_avail"]) == K
    # statics: M loc_in_room, K key_at, K key_unlocks
    assert len(state["loc_in_room"]) == M
    assert len(state["key_at"]) == K
    assert len(state["key_unlocks"]) == K
    # not empty
    assert any(state[p] for p in state)


# ---------------------------------------------------------------------------
# item b — goal atoms exposed and typed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker,goal", [("make_d2", "loc_3"), ("make_d3", "loc_5")])
def test_goal_atoms_exposed_and_typed(maker, goal):
    env = getattr(DoorsPDDLLiteRelationalEnv, maker)()
    g = env.get_goal_atoms()
    assert set(g) == set(_FLUENT_ARITY)
    assert g["at_loc"] == {(goal,)}
    for pred in g:
        if pred != "at_loc":
            assert g[pred] == set()
    assert g["at_loc"].issubset({(o,) for o in env.get_objects_by_type()["location"]})


# ---------------------------------------------------------------------------
# item c — objects_by_type includes room / location / key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker,D,M,K", [("make_d2", 2, 4, 1), ("make_d3", 3, 6, 2)])
def test_objects_by_type_has_room_location_key(maker, D, M, K):
    env = getattr(DoorsPDDLLiteRelationalEnv, maker)()
    objs = env.get_objects_by_type()
    assert set(objs) == {"room", "location", "key"}
    assert objs["room"] == tuple(f"room_{r}" for r in range(D))
    assert objs["location"] == tuple(f"loc_{l}" for l in range(M))
    assert objs["key"] == tuple(f"key_{k}" for k in range(K))


# ---------------------------------------------------------------------------
# item d — legal actions round-trip as GroundAction
# ---------------------------------------------------------------------------

def test_legal_actions_roundtrip_as_groundaction():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    objs = env.get_objects_by_type()
    for a in env.legal_actions():
        assert isinstance(a, GroundAction)
        assert a.schema in {"move_to", "pick", "noop"}
        if a.schema == "move_to":
            assert len(a.args) == 1 and a.args[0] in objs["location"]
        elif a.schema == "pick":
            assert len(a.args) == 1 and a.args[0] in objs["key"]
        else:
            assert a.args == ()
    # at D2 reset: room_1 locked, not standing on a key -> only the two room_0 moves + noop
    assert env.legal_actions() == {
        GroundAction("move_to", ("loc_0",)),
        GroundAction("move_to", ("loc_1",)),
        GroundAction("noop", ()),
    }


def _decode_mask_to_groundactions(base: DoorsPDDLLiteEnv) -> set[GroundAction]:
    mask = base.action_masks("precondition")
    out: set[GroundAction] = set()
    for i, ok in enumerate(mask):
        if not ok:
            continue
        atype, param = base.decode_action(i)
        if atype == "move":
            out.add(GroundAction("move_to", (loc_name(param),)))
        elif atype == "pick":
            out.add(GroundAction("pick", (key_name(param),)))
        elif atype == "noop":
            out.add(GroundAction("noop", ()))
    return out


def test_legal_actions_mirrors_base_action_masks():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    assert env.legal_actions() == _decode_mask_to_groundactions(env.base)
    env.step(GroundAction("move_to", ("loc_1",)))
    assert env.legal_actions() == _decode_mask_to_groundactions(env.base)
    env.step(GroundAction("pick", ("key_0",)))
    assert env.legal_actions() == _decode_mask_to_groundactions(env.base)


# ---------------------------------------------------------------------------
# adapter ≡ underlying env dynamics
# ---------------------------------------------------------------------------

def test_step_parity_with_base_env():
    adapter = DoorsPDDLLiteRelationalEnv.make_d2()
    seq = [
        GroundAction("move_to", ("loc_1",)),
        GroundAction("pick", ("key_0",)),
        GroundAction("move_to", ("loc_3",)),
    ]
    for a in seq:
        adapter.step(a)
    # a fresh base env, stepped with the matching integer actions, reaches the same raw state
    ref = DoorsPDDLLiteEnv.make_d2()
    ref.reset()
    for i in (1, 4, 3):           # move_to(loc_1)=1, pick(key_0)=M+0=4, move_to(loc_3)=3
        ref.step(i)
    assert np.array_equal(adapter.base._state, ref._state)
    # and the relational view of the goal-reached state is what we expect
    assert adapter.is_solved()
    s = adapter.get_state_atoms()
    assert s["at_loc"] == {("loc_3",)}
    assert s["unlocked"] == {("room_0",), ("room_1",)}
    assert s["key_avail"] == set()


def test_illegal_action_raises():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    # room_1 is locked at reset -> moving into it is illegal
    with pytest.raises(ValueError, match="illegal action"):
        env.step(GroundAction("move_to", ("loc_2",)))


# ---------------------------------------------------------------------------
# the flat-vector / gymnasium interface is untouched
# ---------------------------------------------------------------------------

def test_gym_interface_untouched():
    env = DoorsPDDLLiteRelationalEnv.make_d2()
    assert isinstance(env.base, DoorsPDDLLiteEnv)
    obs, info = env.base.reset()
    assert isinstance(obs, np.ndarray) and obs.shape == (7,) and obs.dtype == np.float32
    assert isinstance(info, dict)
    assert env.base.action_space.n == 6           # M + K + 1 = 4 + 1 + 1
    assert env.base.observation_space.shape == (7,)
    # original module still imports & behaves
    import alphazeropp.instances.doors.doors_pddl_lite as _m  # noqa: F401
    e2 = DoorsPDDLLiteEnv.make_d2()
    o2, _ = e2.reset()
    o2, r, term, trunc, _ = e2.step(1)
    assert o2[1] == 1.0 and not term


# ---------------------------------------------------------------------------
# the DomainSignature later stages will consume
# ---------------------------------------------------------------------------

def test_doors_pddl_signature_shape():
    sig = doors_pddl_signature()
    assert sig.types == ("room", "location", "key")
    assert len(sig.predicates) == 6
    assert len(sig.action_schemas) == 3
    assert sig.goal_predicate_names == ("at_loc",)
    pred_by_name = {p.name: p.arg_types for p in sig.predicates}
    assert pred_by_name == _ARG_TYPES
    act_by_name = {a.name: a.arg_types for a in sig.action_schemas}
    assert act_by_name == {"move_to": ("location",), "pick": ("key",), "noop": ()}


# ---------------------------------------------------------------------------
# Stage 1 is grammar / MCTS independent
# ---------------------------------------------------------------------------

def test_no_grammar_or_mcts_imports_in_adapter_modules():
    for mod in (doors_pddl_lifted, doors_pddl_policies):
        src = inspect.getsource(mod)
        import_lines = [ln for ln in src.splitlines() if re.match(r"\s*(import |from )", ln)]
        blob = " ".join(import_lines).lower()
        for bad in ("mcts", "derivation", "leaf_evaluator", "lifted_grammar", "derivation_game"):
            assert bad not in blob, f"{mod.__name__} imports something matching {bad!r}"
