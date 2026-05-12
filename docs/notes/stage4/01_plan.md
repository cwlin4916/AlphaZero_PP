# Stage 1 (rev. 1) — Lifted semantic core on two relational contracts: Gripper-lite + Doors PDDL

> Companion to be written **after** code runs: [01.md](01.md).
> Orientation doc: [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md).
> v0 (single-domain) record: [legacy/01_plan.md](legacy/01_plan.md) / [legacy/01.md](legacy/01.md).

## Context

The orientation doc [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md) frames a two-level architecture: a
**task level** (relational MDPs) and a **synthesis level** (a grammar-derivation game searched by MCTS), connected by a
**lifted decision-list policy**. Stage 1 is the bottom of that stack — *"does the lifted-policy dispatch primitive even
work?"* — and is **deliberately independent of the derivation grammar and of MCTS**: no grammar, no derivation game, no
leaf-evaluator wiring here. The grammar and MCTS are Stage 2; a learned net is Stage 4.

The v0 Stage 1 ([legacy/01_plan.md](legacy/01_plan.md) / [legacy/01.md](legacy/01.md)) built the domain-agnostic
lifted DSL + online-unification interpreter + a tiny Gripper-lite relational env, and proved a hand-written 4-rule
policy solves Gripper-lite for `B ∈ {1,2,3}` with identical rule structure across sizes and under object renaming. All
of that already exists and passes:
[lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py),
[lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py),
[gripper_lite/env.py](../../../src/alphazeropp/instances/gripper_lite/env.py),
[gripper_lite/policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py), and
`tests/test_lifted_dsl_v2.py` / `tests/test_lifted_interpreter.py` / `tests/test_gripper_lite_policy.py` (18 tests).

**What this rev adds.** The orientation doc §4.2 sketches a *Doors PDDL relational contract* — `at_loc(location)`,
`unlocked(room)`, `key_avail(key)`, actions `move_to(location)` / `pick(key)` / `noop` — and flags it **not yet
implemented** (no file imports `doors_pddl_lite` alongside the lifted interpreter). Stage 1 rev. 1 *implements* it: a
thin **relational adapter** over the existing `DoorsPDDLLiteEnv`
([doors/doors_pddl_lite.py](../../../src/alphazeropp/instances/doors/doors_pddl_lite.py)) that exposes the same lifted
env contract Gripper-lite exposes, so the *same* generic `interpret(...)` runs on a second, structurally different
domain. The contract sketch is **refined while wiring it**: the adapter additionally exposes three *static* relations —
`loc_in_room(location, room)`, `key_at(key, location)`, `key_unlocks(key, room)` — so meaningful Doors rules are
expressible, and a hand-written 3-rule Doors policy actually solves the two shipped layouts (D2: 3 steps, D3: 5 steps),
mirroring the Gripper-lite story. This is a **semantic-core demonstration on a second domain**, not an MCTS experiment
and not a generalization claim.

Success here re-confirms the dispatch primitive is genuinely domain-agnostic before Stage 2 couples it to the grammar.

## Refinement table — input prompt item → refinement → reason

| Input-prompt item | Refinement | Reason |
|---|---|---|
| "Reuse or implement the lifted DSL: `PredicateSchema`/`ActionSchema`/`Var`/`Atom`/`Literal`/`LiteralSource`/`LiftedAction`/`GroundAction`/`Rule`/`Policy`/`RelState`" | **Reuse as-is** — all already in [lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py). No edits. | v0 Stage 1 built them; they pass `tests/test_lifted_dsl_v2.py`. The shapes match the prompt exactly (single `Rule.body` tuple discriminated by `Literal.source`; safe-negation at `Rule.__post_init__`; `RelState = dict[str, set[tuple[str,...]]]`). |
| "Reuse or implement `find_bindings(...)` / `interpret(..., trace=False)` with first-applicable, typed vars, state/goal source, CWA negation, safe negation, deterministic lex-min, no full grounding" | **Reuse as-is** — all in [lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py). No edits. | Already implements every listed semantic; passes `tests/test_lifted_interpreter.py`. `find_bindings` walks positive literals row-by-row before type-enumerating any still-free vars, then CWA-filters negatives, then filters by `legal_actions`, then sorts by `tuple(sorted(theta.items()))` (lex-min first). |
| "Ensure `GripperLiteEnv` exposes `get_state_atoms`/`get_goal_atoms`/`get_objects_by_type`/`legal_actions`/`step`/`reset`/`is_solved`; `hand_policy()` solves B=1,2,3; plan lengths 3,7,11 (`4B−1`)" | **Already true** — [gripper_lite/env.py](../../../src/alphazeropp/instances/gripper_lite/env.py) + [gripper_lite/policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py); verified by the existing parametrized test `test_hand_policy_solves[(1,3),(2,7),(3,11)]`. No edits. | The prompt's "B=2 → 7, B=3 → 11" already corrects the v0 plan's `3B` typo; `horizon = 4·n_balls + 4 ≥ 4B−1`. |
| "Inspect `doors_pddl_lite.py`; add a relational adapter or direct env methods exposing the lifted contract; do not break the gym interface; keep int ids internally, expose stable string names" | **New wrapper class `DoorsPDDLLiteRelationalEnv`** in a new file `src/alphazeropp/instances/doors/doors_pddl_lifted.py`. It holds a `DoorsPDDLLiteEnv` (`.base`), maps ints↔stable strings (`loc_l`/`room_r`/`key_k`), and exposes `get_state_atoms/get_goal_atoms/get_objects_by_type/legal_actions/step(GroundAction)/reset/is_solved`. `doors_pddl_lite.py` is **left 100 % untouched**. | A wrapper, not direct methods: relational `step(action: GroundAction) -> 4-tuple` would collide with gym's `step(action: int) -> 5-tuple`. A wrapper is the lowest-blast-radius way to satisfy "don't break the gym interface" and "keep the existing Doors tests passing". |
| "`get_state_atoms() -> RelState`, `get_goal_atoms() -> RelState`, `get_objects_by_type() -> dict[str, tuple[str,...]]`, `legal_actions() -> set[GroundAction]`; objects include room/location/key" | Implemented. **Vocabulary refined vs orientation §4.2**: besides the 3 fluents `at_loc(location)` / `unlocked(room)` / `key_avail(key)`, the adapter exposes 3 *static* relations `loc_in_room(location, room)` / `key_at(key, location)` / `key_unlocks(key, room)` read from `base.loc_room` / `base.key_loc` / `base.key_unlocks`. `get_goal_atoms()` returns `{"at_loc": {(loc_<goal>,)}}` and empty sets for the rest. | The §4.2 box was an explicitly *not-yet-wired* sketch; wiring it for real, the static maps that already define the Doors instance are part of the relational world state (PDDL: `(loc-in-room l r)` etc.). Without them no rule can reason about *where* a key is. **The orientation doc §4.2 box + §10 ladder are updated to this refined vocabulary.** |
| "Add a `doors_pddl_signature()` returning `DomainSignature`-style schemas if that type exists; else document the schema in the adapter" | `DomainSignature` exists ([lifted_grammar.py](../../../src/alphazeropp/synthesis/lifted_grammar.py)). Add `doors_pddl_signature() -> DomainSignature` next to the existing `gripper_lite_signature()`, importing the Doors schema constants lazily. `goal_predicate_names = ("at_loc",)`. | Mirrors `gripper_lite_signature()` exactly. `DomainSignature` is a plain dataclass — importing it does *not* pull in any grammar/MCTS machinery, so Stage 1 stays grammar-independent. |
| "Add minimal hand-written Doors rules or trace policies sufficient to show the interpreter can choose legal Doors actions; need not solve arbitrary instances" | **`doors_hand_policy() -> Policy`** in a new file `src/alphazeropp/instances/doors/doors_pddl_policies.py` (mirrors `gripper_lite/policies.py`): the 3-rule policy ρ₁/ρ₂/ρ₃ below. Also export `degenerate_noop_policy()` (`⊤ ⇒ noop()`) as a "trivial / no-progress" reference, mirroring `degenerate_drop_policy()`. | A real 3-rule policy that solves D2/D3 is a *stronger* demonstration than vestigial rules, parallels the Gripper-lite story, and costs nothing extra. Framed in [01.md](01.md) as evidence the dispatch primitive works on a second domain — **not** a generalization or MCTS claim. |
| (item e) "a simple lifted rule can produce a legal `PICK` when the agent is at a key" | `test_pick_rule_picks_colocated_available_key` in `tests/test_doors_pddl_lifted_interpreter.py` — agent moved onto `loc_1` (D2), rule `at_loc(?l) ∧ key_at(?k,?l) ∧ key_avail(?k) ⇒ pick(?k)`; assert `interpret(...) == GroundAction("pick", ("key_0",))` ∈ `legal_actions()`. | Direct realization of item e. |
| (item f) "a simple lifted rule can produce a legal `MOVE_TO` in a small fixture" | `test_move_rule_produces_legal_move` — D2 initial state, rule `unlocked(?r) ∧ loc_in_room(?l,?r) ⇒ move_to(?l)`; result is a `move_to` to a location in an unlocked room and ∈ `legal_actions()`. | Direct realization of item f. |
| (item g) "`interpret(..., trace=True)` returns the expected rule index and binding for one deterministic Doors fixture" | `test_interpret_trace_doors_d2_step0` — `doors_hand_policy()` on the D2 initial state; assert `interpret(..., trace=True) == (GroundAction("move_to", ("loc_1",)), 2, {"?k":"key_0","?l":"loc_1","?r":"room_0"})` (ρ₃ = index 2; ρ₁/ρ₂ have no binding — room 1 locked / no key at loc 0). | Direct realization of item g; the binding is hand-checked in §"Worked Doors traces". |
| Tests: "`tests/test_lifted_dsl_v2.py`, `test_lifted_interpreter.py`, `test_gripper_lite_policy.py`, `test_doors_pddl_lifted_contract.py`, `test_doors_pddl_lifted_interpreter.py`; preserve existing; avoid changing grounded Doors behavior" | First three: **preserved unchanged** (already exist, 18 tests). Last two: **new**. No `doors_pddl_lite.py` / grounded-`interpreter.py` edits, so existing Doors tests (`test_doors_pddl_lite.py`, `test_doors_direct.py`, …) are untouched. | Matches the acceptance criteria verbatim. |
| (implicit) reproducible Doors-rollout driver, like `scripts/run/run_lifted_gripper_lite_smoke.py` | Add `scripts/run/run_lifted_doors_pddl_smoke.py`: imports the adapter + `doors_hand_policy()` + `interpret`, rolls out on D2 and D3, prints the trace tables, exits non-zero if either is unsolved. ASCII trace tables are the load-bearing artifact for [01.md](01.md) §6 (no new figure required; the orientation §4.2 has no Doors figure either). | Mirrors the existing gripper smoke; gives [01.md](01.md) a copy-pasteable, drift-proof source for the trace tables. |

## DSL / interpreter contract (recap — unchanged in Stage 1 rev. 1)

The lifted env contract every relational env must satisfy (already met by `GripperLiteEnv`; newly met by
`DoorsPDDLLiteRelationalEnv`):

```
get_state_atoms()          -> RelState                               # dict[pred_name, set[tuple[obj_name,...]]]
get_goal_atoms()           -> RelState
get_objects_by_type()      -> dict[type_name, tuple[obj_name, ...]]
legal_actions()            -> set[GroundAction]
step(action: GroundAction) -> tuple[RelState, float, bool, dict]     # (next_state, reward, done, info)
reset(...)                 -> RelState
is_solved()                -> bool
```

The interpreter consumes the first four:

```
interpret(policy: Policy,
          state_atoms: RelState, goal_atoms: RelState,
          objects_by_type: dict[str, tuple[str,...]],
          legal_actions: set[GroundAction],
          *, trace: bool = False)
    -> GroundAction | None | tuple[GroundAction, int, Binding]
```

First-applicable rule; lex-min binding under `tuple(sorted(theta.items()))`; STATE literals read against `state_atoms`,
GOAL literals against `goal_atoms`; closed-world negation evaluated after the positive pass; safe-negation enforced at
`Rule.__post_init__`; never materializes the Cartesian product over all objects when positive literals constrain the
search. **The same semantics, the same code, applied to whatever env satisfies the contract above.**

## Module / class design (the new code)

### `src/alphazeropp/instances/doors/doors_pddl_lifted.py` (new)

A wrapper class `DoorsPDDLLiteRelationalEnv` over `DoorsPDDLLiteEnv`, exposing the lifted env contract over the Doors
vocabulary:

```
Types:      room, location, key
Fluents:    at_loc(location), unlocked(room), key_avail(key)
Static:     loc_in_room(location, room), key_at(key, location), key_unlocks(key, room)
Actions:    move_to(location), pick(key), noop()
Goal:       { at_loc(<goal location>) }
```

Module-level constants `PREDICATE_SCHEMAS` / `ACTION_SCHEMAS` (the `PredicateSchema` / `ActionSchema` tuples above) and
name helpers `loc_name(l)` / `room_name(r)` / `key_name(k)` / `_parse_id(name)`. A `name_fmt` constructor arg
(`{"location": "loc_{}", "room": "room_{}", "key": "key_{}"}` by default) lets a relabeled instance be built for the
renaming test — only the surface strings change; underlying int ids are untouched.

```python
class DoorsPDDLLiteRelationalEnv:
    def __init__(self, base: DoorsPDDLLiteEnv, *, name_fmt: dict[str, str] | None = None): ...
    @classmethod
    def make_d2(cls, *, name_fmt=None, **kw) -> "DoorsPDDLLiteRelationalEnv": ...   # DoorsPDDLLiteEnv.make_d2(**kw)
    @classmethod
    def make_d3(cls, *, name_fmt=None, **kw) -> "DoorsPDDLLiteRelationalEnv": ...

    @property
    def base(self) -> DoorsPDDLLiteEnv: ...
    @property
    def horizon(self) -> int: ...        # = base.horizon
    @property
    def step_count(self) -> int: ...     # = base.step_count

    def get_state_atoms(self) -> RelState: ...
    def get_goal_atoms(self) -> RelState: ...
    def get_objects_by_type(self) -> dict[str, tuple[str, ...]]: ...
    def legal_actions(self) -> set[GroundAction]: ...

    def reset(self, seed: int | None = None) -> RelState: ...           # base.reset(seed=seed)[0] -> RelState
    def is_solved(self) -> bool: ...                                    # base.is_solved(base.state)
    def step(self, action: GroundAction) -> tuple[RelState, float, bool, dict]: ...
```

`get_state_atoms()` reads `base._state` via the precomputed offsets `base.M / base._unlocked_offset / base._key_offset`
for the fluents, and `base.loc_room / base.key_loc / base.key_unlocks` for the static relations. `get_goal_atoms()`
returns `{"at_loc": {(loc_name(base.goal_loc),)}}` plus empty sets for the other five predicates. `legal_actions()` is
the relational mirror of `base.action_masks("precondition")`: `move_to(loc_l)` iff `unlocked[loc_room[l]]`;
`pick(key_k)` iff agent is at `key_loc[k]` and `key_available[k]`; `noop()` always. `step()` translates the
`GroundAction` to an int via `base.encode_action(...)`, raises `ValueError` if the action ∉ `legal_actions()`, calls
`base.step(idx)`, and returns `(get_state_atoms(), reward, terminated or truncated, {**info, "solved": is_solved()})`.

Determinism caveat (same as Gripper-lite, orientation §7): the interpreter's lex-min tie-break is over *raw* object
names, so `"loc_10" < "loc_2"` lexicographically. The shipped D2 (M=4) / D3 (M=6) layouts have only single-digit ids
and no contested tie in the hand-policy rollout, so this is moot here; zero-padding the names would be the fix and is
out of Stage-1 scope.

### `src/alphazeropp/instances/doors/doors_pddl_policies.py` (new)

`doors_hand_policy() -> Policy` — the 3-rule reference policy (first-applicable order matters):

$$
\boxed{\;\rho_1:\quad \mathrm{Goal}[\mathrm{at\_loc}(?l)]\;\wedge\;\mathrm{unlocked}(?r)\;\wedge\;\mathrm{loc\_in\_room}(?l,?r)\;\Longrightarrow\;\mathrm{move\_to}(?l)\;}
$$

$$
\boxed{\;\rho_2:\quad \mathrm{at\_loc}(?l)\;\wedge\;\mathrm{key\_at}(?k,?l)\;\wedge\;\mathrm{key\_avail}(?k)\;\Longrightarrow\;\mathrm{pick}(?k)\;}
$$

$$
\boxed{\;\rho_3:\quad \mathrm{key\_at}(?k,?l)\;\wedge\;\mathrm{key\_avail}(?k)\;\wedge\;\mathrm{loc\_in\_room}(?l,?r)\;\wedge\;\mathrm{unlocked}(?r)\;\Longrightarrow\;\mathrm{move\_to}(?l)\;}
$$

(`?l : location`, `?r : room`, `?k : key`.) Intuition: ρ₁ — if the goal location's room is unlocked, walk to the goal;
ρ₂ — if standing on an available key, pick it (which unlocks a room); ρ₃ — otherwise walk to a reachable available key.
No negation, so safe-negation is vacuous. Also `degenerate_noop_policy() -> Policy` = `⊤ ⇒ noop()` (always fires noop;
the rollout never solves and the state never changes) — the Stage-1 "trivial / no-progress" reference, mirroring
`degenerate_drop_policy()` in `gripper_lite/policies.py`.

### `src/alphazeropp/synthesis/lifted_grammar.py` (extend — one new function)

```python
def doors_pddl_signature() -> DomainSignature:
    from alphazeropp.instances.doors.doors_pddl_lifted import ACTION_SCHEMAS, PREDICATE_SCHEMAS
    return DomainSignature(
        types=("room", "location", "key"),
        predicates=tuple(PREDICATE_SCHEMAS),
        action_schemas=tuple(ACTION_SCHEMAS),
        goal_predicate_names=("at_loc",),
    )
```

This is the *only* edit to `lifted_grammar.py` (the existing `aux_var` grammar code stays untouched — the §8.2 redesign
is a later stage). No other `src/` file is edited; `lifted_dsl.py`, `lifted_interpreter.py`, `gripper_lite/*`,
`doors_pddl_lite.py`, the grounded `interpreter.py` all stay byte-for-byte the same.

### `scripts/run/run_lifted_doors_pddl_smoke.py` (new)

Mirrors `scripts/run/run_lifted_gripper_lite_smoke.py`'s shape (path-bootstrap, no MCTS): for `layout ∈ {d2, d3}`,
build `DoorsPDDLLiteRelationalEnv.make_{d2,d3}()`, roll out `doors_hand_policy()` via `interpret(..., trace=True)`,
print a `step | rule | grounded action | θ` table, assert `is_solved()`; exit non-zero on failure. Used to regenerate
the trace tables in [01.md](01.md).

## Worked Doors traces (hand-checked — the source of [01.md](01.md) §6 and of test g)

Object names: `room_r`, `loc_l`, `key_k`. Layouts from `doors_pddl_lite.py`:
- **D2**: D=2, M=4, K=1; `loc_room=[0,0,1,1]`, `key_loc=[1]`, `key_unlocks=[1]`, start=0, goal=3.
- **D3**: D=3, M=6, K=2; `loc_room=[0,0,1,1,2,2]`, `key_loc=[1,2]`, `key_unlocks=[1,2]`, start=0, goal=5.

Static relations: `loc_in_room = {(loc_l, room_{loc_room[l]})}`; `key_at = {(key_k, loc_{key_loc[k]})}`;
`key_unlocks = {(key_k, room_{key_unlocks[k]})}`.

**D2** — initial fluents `at_loc={(loc_0,)}`, `unlocked={(room_0,)}`, `key_avail={(key_0,)}`; goal `at_loc={(loc_3,)}`:

| step | rule | grounded action | θ | why this rule (first-applicable) |
|---:|---:|---|---|---|
| 0 | ρ₃ (idx 2) | `move_to(loc_1)` | `?k=key_0, ?l=loc_1, ?r=room_0` | ρ₁ fails: `loc_in_room(loc_3, room_0)` ∉ state (loc_3 ∈ room_1, locked). ρ₂ fails: no `key_at(?k, loc_0)`. ρ₃: key_0 at loc_1 ∈ room_0 (unlocked). |
| 1 | ρ₂ (idx 1) | `pick(key_0)` | `?l=loc_1, ?k=key_0` | now `at_loc(loc_1)` ∧ `key_at(key_0, loc_1)` ∧ `key_avail(key_0)`. Picking unlocks room_1. |
| 2 | ρ₁ (idx 0) | `move_to(loc_3)` | `?l=loc_3, ?r=room_1` | `Goal[at_loc(loc_3)]` ∧ `unlocked(room_1)` ∧ `loc_in_room(loc_3, room_1)`. → goal_loc. **solved, 3 steps.** |

**D3** — initial fluents `at_loc={(loc_0,)}`, `unlocked={(room_0,)}`, `key_avail={(key_0,),(key_1,)}`; goal `at_loc={(loc_5,)}`:

| step | rule | grounded action | θ |
|---:|---:|---|---|
| 0 | ρ₃ (idx 2) | `move_to(loc_1)` | `?k=key_0, ?l=loc_1, ?r=room_0` |
| 1 | ρ₂ (idx 1) | `pick(key_0)` | `?l=loc_1, ?k=key_0` |
| 2 | ρ₃ (idx 2) | `move_to(loc_2)` | `?k=key_1, ?l=loc_2, ?r=room_1` |
| 3 | ρ₂ (idx 1) | `pick(key_1)` | `?l=loc_2, ?k=key_1` |
| 4 | ρ₁ (idx 0) | `move_to(loc_5)` | `?l=loc_5, ?r=room_2` |

**solved, 5 steps.** Step 0 ρ₃ binding is unambiguous — of `{(key_0,loc_1,room_0),(key_1,loc_2,room_1)}` only the
key_0 row survives `unlocked(?r)` (room_1 still locked). No contested lex-min tie at any step in either rollout.

> Observation: with `K` keys arranged in a chain, this policy solves the chained-unlock Doors layout in `2K + 1` steps
> (K `move_to` + K `pick` + 1 final `move_to`): D2 → 3, D3 → 5. This is stated as a fact about *this hand policy* on
> *these shipped layouts* — **not** a generalization claim about synthesized policies.

## Tests and acceptance criteria

Acceptance run set:
`pytest tests/test_lifted_dsl_v2.py tests/test_lifted_interpreter.py tests/test_gripper_lite_policy.py tests/test_doors_pddl_lifted_contract.py tests/test_doors_pddl_lifted_interpreter.py -v`

### Preserved unchanged (already pass — 18 tests)
- `tests/test_lifted_dsl_v2.py` — DSL type-checking & safe-negation (5 tests).
- `tests/test_lifted_interpreter.py` — positive/negative matching, determinism, no-rule, no-full-grounding (6 tests).
- `tests/test_gripper_lite_policy.py` — hand-policy solves B=1,2,3; object-renaming invariance (7 tests).

### New: `tests/test_doors_pddl_lifted_contract.py` — the adapter contract

| Test | What it checks | Why it matters |
|---|---|---|
| `test_state_atoms_nonempty_and_typed` | `get_state_atoms()` on D2 reset has non-empty `at_loc` (exactly one tuple), `unlocked` (≥ room_0), `key_avail` (= K keys), and the 3 static relations with the right arities; every object name is one of `get_objects_by_type()`'s names of the right type | item a — state non-empty and typed |
| `test_goal_atoms_exposed_and_typed` | `get_goal_atoms()` = `{"at_loc": {("loc_3",)}, …all-others-empty}` for D2; `("loc_5",)` for D3 | item b — pins the new `get_goal_atoms()` contract addition |
| `test_objects_by_type_has_room_location_key` | keys are exactly `{"room","location","key"}`; counts `(D,M,K)` = `(2,4,1)` (D2) and `(3,6,2)` (D3); names `room_0..`, `loc_0..`, `key_0..` | item c |
| `test_legal_actions_roundtrip_as_groundaction` | every element of `legal_actions()` is a `GroundAction` with `.schema ∈ {"move_to","pick","noop"}` and type-correct args; on D2 reset it equals `{move_to(loc_0), move_to(loc_1), noop()}` | item d |
| `test_legal_actions_mirrors_base_action_masks` | for D2 reset, after `step(move_to(loc_1))`, and after `step(pick(key_0))`: legal `GroundAction` set equals the set decoded from `base.action_masks("precondition")` via `base.decode_action` | adapter ≡ env precondition semantics |
| `test_step_parity_with_base_env` | run `move_to(loc_1) → pick(key_0) → move_to(loc_3)` on the adapter (D2); `is_solved()`, final `at_loc == {("loc_3",)}`, room_1 unlocked, key_0 unavailable; a fresh base env stepped with the matching int actions decodes to the same `get_state_atoms()` | adapter doesn't fork the dynamics |
| `test_illegal_action_raises` | `step(GroundAction("move_to", ("loc_2",)))` from D2 reset (room_1 locked) raises `ValueError` | matches `GripperLiteEnv.step` |
| `test_gym_interface_untouched` | `base` is a `DoorsPDDLLiteEnv`; `base.reset()` → `(np.ndarray shape (7,), dict)` for D2; `base.action_space.n == 6`; importing `doors_pddl_lite` still works | "do not break the flat-vector interface" |
| `test_doors_pddl_signature_shape` | `doors_pddl_signature()` has `types == ("room","location","key")`, `len(predicates) == 6`, `len(action_schemas) == 3`, `goal_predicate_names == ("at_loc",)`, and every schema arg-type tuple matches the vocabulary above | the signature later stages consume is correct |
| `test_no_mcts_imports_in_adapter` | source of `doors_pddl_lifted.py` / `doors_pddl_policies.py` contains no `mcts` / `derivation` / `leaf_evaluator` substring | Stage 1 is grammar/MCTS-independent |

### New: `tests/test_doors_pddl_lifted_interpreter.py` — the interpreter on Doors

| Test | What it checks | Why it matters |
|---|---|---|
| `test_pick_rule_picks_colocated_available_key` | `make_d2()`, `step(move_to(loc_1))`; rule `at_loc(?l) ∧ key_at(?k,?l) ∧ key_avail(?k) ⇒ pick(?k)`; `interpret(...) == GroundAction("pick", ("key_0",))` ∈ `legal_actions()` | **item e** |
| `test_move_rule_produces_legal_move` | `make_d2()` reset; rule `unlocked(?r) ∧ loc_in_room(?l,?r) ⇒ move_to(?l)`; result is a `move_to` to a location in an unlocked room, ∈ `legal_actions()`, and ≠ a move into the locked room | **item f** |
| `test_interpret_trace_doors_d2_step0` | `interpret(doors_hand_policy(), <D2 initial>, …, trace=True) == (GroundAction("move_to", ("loc_1",)), 2, {"?k":"key_0","?l":"loc_1","?r":"room_0"})` | **item g** — deterministic rule index + binding |
| `test_interpret_is_deterministic_doors` | two `interpret(…, trace=True)` calls on the same D2 fixture return identical results | reproducibility |
| `test_no_rule_returns_none_doors` | 1-rule policy `key_avail(?k) ⇒ pick(?k)` on the D2 *initial* state (agent at loc_0, key at loc_1 — no legal pick) → `interpret(...) is None` | clean "no firing" signal; exercises the `legal_actions` prune |
| `test_doors_hand_policy_solves_d2` | roll out `doors_hand_policy()` on `make_d2()`; `is_solved()`; schemas == `["move_to","pick","move_to"]`; exactly 3 steps; picked key `key_0` | semantic-core works on Doors (D2) |
| `test_doors_hand_policy_solves_d3` | same on `make_d3()`; `is_solved()`; schemas == `["move_to","pick","move_to","pick","move_to"]`; exactly 5 steps; picked keys `["key_0","key_1"]` | same 3 rules, same canonical order, larger instance |
| `test_doors_object_renaming_preserves_structure` | renamed adapter (`loc_l→site_l`, `room_r→area_r`, `key_k→tool_k` — order-preserving σ); roll out the *same* `doors_hand_policy()`; step-for-step: same `action.schema`, same rule index, `σ(canonical_args) == renamed_args` | the lifting point — `find_bindings` reads relational structure, not raw names (orientation §7 caveat: order-preserving σ) |
| `test_degenerate_noop_policy_stalls` | `interpret(degenerate_noop_policy(), <D2 initial>, …) == GroundAction("noop", ())`; rolling it out never solves and the state never changes | the "trivial" reference policy behaves as documented |

All new tests use the deterministic preset layouts only; no randomness, no MCTS, no grammar imports.

### Acceptance criteria

(i) the five-file `pytest` set passes; (ii) existing Doors tests still pass (`test_doors_pddl_lite.py`,
`test_doors_direct.py`, `test_relational_runtime.py`, …); (iii) the full suite passes except any pre-existing
documented archive collection error (`archive/test_zoning_game.py`); (iv) Stage-1 docs do not describe MCTS as
implemented work and do not claim grammar-search success; (v) Doors PDDL is presented as a semantic-core setup
(relational adapter + hand policy), not as an MCTS experiment; (vi) `doors_pddl_lite.py` and the grounded
`interpreter.py` are unchanged (`git diff main -- <each>` empty).

## What [01.md](01.md) (results companion) will report

Written **after** the acceptance set passes. Sections: **§0** intro & scope (in: semantic core on two contracts;
out: grammar, derivation game, MCTS, learned net, leaf evaluator, multi-domain *generalization*); **§1** module diagram
(`{GripperLiteEnv | DoorsPDDLLiteRelationalEnv} → get_state/goal_atoms, get_objects_by_type, legal_actions →
interpret(policy,…) → GroundAction → env.step`; zero `mcts | derivation | leaf_evaluator` imports in the new modules);
**§2** Gripper-lite recap (the 4 hand rules + boxed math; one worked B=2 binding; `figures/gripper_lite_domain.png`);
**§3** Gripper-lite rollout traces B=1/2/3 (3/7/11 steps; boxed `steps(B)=4B−1`; `figures/gripper_lite_rollout_B2.png`);
**§4** Gripper-lite object-renaming (canonical vs renamed side-by-side; name-agnostic-matching caveat); **§5** Doors PDDL
relational adapter (the refined vocabulary box; a printed `get_state_atoms`/`get_goal_atoms`/`get_objects_by_type`/
`legal_actions` example for the D2 reset; int↔string name map; mirror of `action_masks("precondition")`; optional ASCII
D2 schematic); **§6** Doors PDDL rollout traces (the D2 3-step / D3 5-step tables, regenerated by
`run_lifted_doors_pddl_smoke.py`; the `2K+1`-on-these-layouts observation flagged as not a generalization claim; the
Doors renaming-structure example); **§7** test summary (auto from `pytest --tb=no -v` over the 5 files; plus the
`pytest tests/ -q` regression count vs `main`); **§8** refinement deltas vs this plan; **§9** limitations (hand-written
policies only; Doors solves are dispatch demonstrations on two fixed layouts; renaming holds for order-preserving σ;
`pick` arity differs Gripper vs Doors but they're never co-resident; Doors `noop` exposed but never needed); **§10** how
this sets up Stage 2 (same `interpret(...)` and env contract are what the grammar's productions must target;
`gripper_lite_signature()` / `doors_pddl_signature()` are the `DomainSignature`s the derivation game consumes; Stage 2
replaces the hand policies with grammar-derived ones and wires the leaf evaluator — none of which exists yet).

Also: update [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md) §4.2 (status → implemented; `P_Doors` box
gains the 3 static relations), §10 ladder Stage-1 row (point at this `01_plan.md` / `01.md`; "Doors adapter: not yet" →
"done"), and the references list — `legacy/01_plan.md` / `legacy/01.md` stay linked as the v0 record.

## Critical files

**To create** — `src/alphazeropp/instances/doors/doors_pddl_lifted.py`,
`src/alphazeropp/instances/doors/doors_pddl_policies.py`, `scripts/run/run_lifted_doors_pddl_smoke.py`,
`tests/test_doors_pddl_lifted_contract.py`, `tests/test_doors_pddl_lifted_interpreter.py`, `docs/notes/stage4/01.md`.

**To extend** — [src/alphazeropp/synthesis/lifted_grammar.py](../../../src/alphazeropp/synthesis/lifted_grammar.py)
(add `doors_pddl_signature()`); [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md) (§4.2, §10, references).

**To reference (read-only)** — [lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py) /
[lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py) (the contract being satisfied);
[gripper_lite/env.py](../../../src/alphazeropp/instances/gripper_lite/env.py) /
[gripper_lite/policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py) (the pattern mirrored);
[doors/doors_pddl_lite.py](../../../src/alphazeropp/instances/doors/doors_pddl_lite.py) (env wrapped:
`_state`, offsets, `loc_room`/`key_loc`/`key_unlocks`/`goal_loc`, `encode_action`/`decode_action`, `action_masks`,
`is_solved`); `DomainSignature` / `gripper_lite_signature()` in `lifted_grammar.py`;
[doors/dsl/relational_runtime.py](../../../src/alphazeropp/instances/doors/dsl/relational_runtime.py) (a *different*
compile-time relational view over `DoorsGameConfig` for the flat-vector DSL stack — **not** reused here);
`scripts/run/run_lifted_gripper_lite_smoke.py` (the smoke-driver shape mirrored).

**To leave untouched (acceptance depends on it)** —
[doors/doors_pddl_lite.py](../../../src/alphazeropp/instances/doors/doors_pddl_lite.py) (byte-for-byte),
[synthesis/interpreter.py](../../../src/alphazeropp/synthesis/interpreter.py) (grounded interpreter),
`tests/test_lifted_dsl_v2.py` / `tests/test_lifted_interpreter.py` / `tests/test_gripper_lite_policy.py`, all existing
Doors / synthesis / reactive tests.

## Verification

1. **Acceptance test set:** `pytest tests/test_lifted_dsl_v2.py tests/test_lifted_interpreter.py tests/test_gripper_lite_policy.py tests/test_doors_pddl_lifted_contract.py tests/test_doors_pddl_lifted_interpreter.py -v` → all green.
2. **No regression:** `pytest tests/ -q` → same pass/fail/skip count as `main` plus the new passes (the pre-existing `archive/test_zoning_game.py` collection error, if surfaced, is unrelated and documented in `legacy/01.md`).
3. **Doors env untouched:** `git diff main -- src/alphazeropp/instances/doors/doors_pddl_lite.py` and `git diff main -- src/alphazeropp/synthesis/interpreter.py` are empty.
4. **Smoke driver:** `python scripts/run/run_lifted_doors_pddl_smoke.py` prints the D2 (3-step) / D3 (5-step) trace tables and exits 0.
5. **No MCTS/grammar coupling:** `grep -rEn "mcts|derivation|leaf_evaluator" src/alphazeropp/instances/doors/doors_pddl_lifted.py src/alphazeropp/instances/doors/doors_pddl_policies.py` → empty.
6. **Docs render:** `01_plan.md` / `01.md` open cleanly; trace tables, module diagram, and orientation-doc cross-links resolve.

Stage 1 rev. 1 is complete when checks 1–6 pass and [01.md](01.md) has been written.

## What is not claimed

- No grammar, no derivation game, no MCTS, no learned net, no leaf-evaluator wiring in Stage 1.
- The Gripper-lite and Doors hand policies are hand-written — Stage 1 does not show any search discovers them.
- The D2/D3 solves (and the `2K+1`-step observation) are dispatch-primitive demonstrations on fixed shipped layouts, **not** generalization to larger / structurally-different instances.
- Object-renaming invariance holds for order-preserving relabelings (raw-name lex-min tie-break), not arbitrary permutations (orientation §7 caveat).
- Full classical Gripper (explicit grippers, `carry(b,g)` / `free(g)`) and arbitrary Doors layouts remain future extensions.
