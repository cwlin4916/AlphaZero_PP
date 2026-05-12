# Stage 1 — Lifted policy semantic core (refined plan)

> Companion to be written **after** code runs: [01.md](01.md).
> Upstream draft: [00_draft_lifted_policy_az.md](../00_draft_lifted_policy_az.md).

## Context

Stage 0 ([00_draft_lifted_policy_az.md](../00_draft_lifted_policy_az.md)) documents the open question — can AlphaZero MCTS over a typed lifted-policy grammar synthesize PG3-style decision-list policies that generalize across object counts? Stage 0 settled on PG3-style first-applicable semantics under unification (§5.1) and recommended online unification as the interpreter strategy (§5.3, Option B). Stage 1 is the first executable slice: build the **lifted DSL + online-unification interpreter + a minimal relational env** with no MCTS coupling, and prove via a hand-written 4-rule Gripper-lite policy that the semantic core works end-to-end.

Stage 1 is deliberately scoped to the semantic core because the existing Doors-side `BindingEngine` ([src/alphazeropp/instances/doors/dsl/binding_engine.py](../../../../src/alphazeropp/instances/doors/dsl/binding_engine.py)) is too tied to Doors to reuse as-is for synthesis: it accepts a numpy-array state, uses integer object ids, lacks negative literals, and has no state-vs-goal `LiteralSource` distinction. Stage 1 builds the clean, instance-agnostic counterpart under `src/alphazeropp/synthesis/`, leaves Doors untouched, and exercises the new core on a fresh tiny Gripper-lite domain that exists only to stress universal-goal quantification (the key contrast Stage 0 §4.2 calls out).

Success at Stage 1 unblocks Stage 2 (grammar + production-emitter for `DerivationGame`) and Stage 3 (LeafEvaluator/env-contract wiring).

## Refinement table

DRAFT items rebased onto the codebase. Items unchanged from the input prompt are not listed.

| DRAFT item | Refinement | Reason |
|---|---|---|
| `tests/test_lifted_dsl.py` | Rename → `tests/test_lifted_dsl_v2.py` | Hard collision with existing 234-LOC doors-specific [tests/test_lifted_dsl.py](../../../../tests/test_lifted_dsl.py). (`test_lifted_interpreter.py` and `test_gripper_lite_policy.py` do not collide.) |
| Object identity uses `str` names | Keep `str` names in the new DSL/interpreter | PG3-aligned, matches DRAFT spec, avoids Doors-style numeric obs-index machinery. Doors stays on its own `int` representation in `BindingEngine`. |
| `LiteralSource` enum (`"state"`/`"goal"`) | Implement as `Enum` not bare str-literal; helper constructors `state_lit(...)`, `goal_lit(...)` | One source of truth for the discriminator; safer match statements. |
| `Atom(pred, args: tuple[Var \| str, ...])` | Keep as-is; `str` arg = ground constant, `Var` = variable | Supports rules with mixed bound constants (e.g. `at_robot(room_b)`) even though Stage-1 policy doesn't need it. |
| Safe-negation enforcement | Raise `ValueError` *at `Rule.__post_init__`*, not lazily at first `interpret()` call | Fail fast; aligned with the `lifted_typed_dsl.py:__post_init__` pattern. |
| `Rule.pre` and `Rule.goal` separate fields | Replace with a single `Rule.body: tuple[Literal, ...]` where each `Literal.source` discriminates | Single ordered literal list keeps the binding loop simple; the `source` field already carries the state/goal distinction. The four-rule Gripper-lite policy is straightforward to express this way. |
| Hand-written policy (rules 1–4) | Live in `src/alphazeropp/instances/gripper_lite/policies.py`, not in the test file | Reusable from both tests and downstream demos; tests import it. |
| `obs` returned from `reset()` / `step()` | Return `RelState` (the relational dict), not `np.ndarray` | Stage 1 has no MCTS network; flat-vector obs is gratuitous. `gripper_lite` is a pure relational env. |
| Reward = `-1` per step, `+100` on solve | Keep, but tests assert on `is_solved()` and action sequence only | Reward shape is irrelevant until Stage 3 wires `LeafEvaluator`. |
| Horizon `4 * n_balls + 4` | Keep | Comfortable margin: tight bound is `4 * n_balls`. |
| `interpret(...)` returns `GroundAction \| None` | Also return rule index + binding via an optional `trace` parameter | Lets the rollout test inspect which rule fired, which is what the "object-renaming preserves behavior" test compares. Default `trace=False` keeps the simple API. |
| Test `test_no_rule_returns_none()` from spec list | Keep, plus add `test_safe_negation_rejected_at_rule_construction()` | The safe-negation requirement is load-bearing; promote to its own test. |
| `find_bindings(...) -> list[dict[str, str]]` | Keep, but document deterministic ordering: lex-min over (var-name-sorted) → object-name-sorted | Required for the determinism test and for the object-renaming test. |
| Goal predicate set | Goal atoms are a `RelState` with the SAME predicate schema as state atoms | Avoids a separate `Goal[pred]` namespace; the `LiteralSource` discriminator is what routes lookups. |
| Reward shaping for `gripper_lite` | Use a fixed seed so `frozen_states` is unnecessary for Stage 1 | Stage 1 tests are deterministic; no curriculum sampler needed. |

## Module / class design

### `src/alphazeropp/synthesis/lifted_dsl.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

# --- type / signature catalog ---

@dataclass(frozen=True)
class PredicateSchema:
    name: str
    arg_types: tuple[str, ...]

@dataclass(frozen=True)
class ActionSchema:
    name: str
    arg_types: tuple[str, ...]

# --- syntactic atoms ---

@dataclass(frozen=True)
class Var:
    name: str          # e.g. "?b"
    type_name: str     # e.g. "ball"

Arg = Union[Var, str]  # str = ground constant

@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[Arg, ...]
    def pretty(self) -> str: ...

class LiteralSource(Enum):
    STATE = "state"
    GOAL = "goal"

@dataclass(frozen=True)
class Literal:
    atom: Atom
    negated: bool = False
    source: LiteralSource = LiteralSource.STATE

def state_lit(pred: str, *args: Arg, negated: bool = False) -> Literal: ...
def goal_lit(pred: str, *args: Arg, negated: bool = False) -> Literal: ...

# --- actions ---

@dataclass(frozen=True)
class LiftedAction:
    schema: str
    args: tuple[Var, ...]

@dataclass(frozen=True)
class GroundAction:
    schema: str
    args: tuple[str, ...]              # object NAMES, str
    def pretty(self) -> str: ...

# --- rules / policy ---

@dataclass(frozen=True)
class Rule:
    vars: tuple[Var, ...]
    body: tuple[Literal, ...]
    action: LiftedAction

    def __post_init__(self):
        # type-check, enforce safe-negation:
        # every var in any negative literal must appear in some positive literal
        # or in self.action.args; else raise ValueError.
        ...

@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...]
    def pretty(self) -> str: ...

# --- relational state alias ---
RelState = dict[str, set[tuple[str, ...]]]   # pred_name -> set of arg-tuples

# --- type-checking helpers (used by Atom and Rule post-init) ---
def check_atom_against_schema(atom: Atom, schema: PredicateSchema, var_types: dict[str, str]) -> None: ...
def check_action_against_schema(act: LiftedAction, schema: ActionSchema) -> None: ...
```

### `src/alphazeropp/synthesis/lifted_interpreter.py`

```python
from .lifted_dsl import (
    Policy, Rule, Literal, LiteralSource, Var, Atom,
    LiftedAction, GroundAction, RelState,
)

def find_bindings(
    rule: Rule,
    state_atoms: RelState,
    goal_atoms: RelState,
    objects_by_type: dict[str, tuple[str, ...]],
    legal_actions: set[GroundAction],
) -> list[dict[str, str]]:
    """
    Enumerate substitutions θ: Var.name -> object_name such that:
      - every positive literal (source=STATE) ⊆ state_atoms[pred]
      - every positive literal (source=GOAL)  ⊆ goal_atoms[pred]
      - every negative literal evaluated CWA after θ binds its vars
      - free action-only vars enumerated over objects_by_type[type_name]
      - grounded action is in legal_actions
    Returns bindings in deterministic order (sorted by tuple of var-name→value).
    """

def interpret(
    policy: Policy,
    state_atoms: RelState,
    goal_atoms: RelState,
    objects_by_type: dict[str, tuple[str, ...]],
    legal_actions: set[GroundAction],
    *,
    trace: bool = False,
) -> GroundAction | None | tuple[GroundAction, int, dict[str, str]]:
    """
    First-applicable rule + lex-min binding -> GroundAction.
    If trace=True, return (action, rule_index, binding).
    Returns None if no rule applies.
    """
```

Implementation notes:

- `find_bindings` walks positive literals first to constrain variables (uses `state_atoms[pred]` / `goal_atoms[pred]` row enumeration), then enumerates remaining free variables by type, then evaluates negative literals as CWA filters, then filters by `legal_actions`. This is the §5.3 Option-B online unifier; no full ground policy is materialized.
- Determinism: returned bindings are sorted by `tuple(sorted(theta.items()))` so the first element is lex-min.
- Safe-negation is enforced at `Rule.__post_init__` so the interpreter can assume every neg-var is bound by the positive pass.

### `src/alphazeropp/instances/gripper_lite/env.py`

```python
from dataclasses import dataclass
from alphazeropp.synthesis.lifted_dsl import GroundAction, RelState

@dataclass
class GripperLiteEnv:
    n_balls: int
    horizon: int

    def __init__(self, n_balls: int = 1, seed: int | None = None): ...

    # relational API (used by the lifted interpreter)
    def get_state_atoms(self) -> RelState: ...
    def get_goal_atoms(self) -> RelState: ...
    def get_objects_by_type(self) -> dict[str, tuple[str, ...]]: ...
    def legal_actions(self) -> set[GroundAction]: ...

    # episodic API
    def reset(self, seed: int | None = None, n_balls: int | None = None) -> RelState: ...
    def step(self, action: GroundAction) -> tuple[RelState, float, bool, dict]: ...
    def is_solved(self) -> bool: ...
```

Domain constants (in `env.py`):

- Types: `("ball", "room")`.
- Rooms: `("room_a", "room_b")` (length-2 tuple, the order is the canonical iteration order — important for renaming test).
- Balls: `("ball_0", ..., f"ball_{n_balls-1}")`.
- Predicate schemas: `at_robot(room)`, `at_ball(ball, room)`, `carrying(ball)`, `handempty()`.
- Action schemas: `move(room, room)`, `pick(ball, room)`, `drop(ball, room)`.
- Goal: `{ at_ball(b, "room_b") for b in balls }`.
- Initial state: `{ at_robot("room_a"), handempty(), at_ball(b, "room_a") for b in balls }`.

### `src/alphazeropp/instances/gripper_lite/policies.py`

Exports `hand_policy() -> Policy` — the four-rule lifted policy:

1. `carrying(?b) ∧ at_robot(?r) ∧ Goal at_ball(?b, ?r) ⇒ drop(?b, ?r)`
2. `carrying(?b) ∧ at_robot(?from) ∧ Goal at_ball(?b, ?to) ⇒ move(?from, ?to)`
3. `at_ball(?b, ?r) ∧ at_robot(?r) ∧ handempty() ∧ ¬Goal at_ball(?b, ?r) ⇒ pick(?b, ?r)`
4. `at_robot(?from) ∧ at_ball(?b, ?to) ∧ handempty() ∧ ¬Goal at_ball(?b, ?to) ⇒ move(?from, ?to)`

`?from`, `?to` are typed `room`; `?b` is `ball`; `?r` is `room`.

### `src/alphazeropp/instances/gripper_lite/__init__.py`

Empty (consistent with `instances/doors/__init__.py`).

## Tests

Test files (path-final after the naming-collision fix):

- `tests/test_lifted_dsl_v2.py`
- `tests/test_lifted_interpreter.py`
- `tests/test_gripper_lite_policy.py`

| Test | What it checks | Why it matters |
|---|---|---|
| `test_type_check_rejects_bad_atom` | `Atom("at_ball", (Var("?b","ball"), Var("?r","ball")))` raises (room arg given ball var) when type-checked against the schema catalog | Catches schema mismatches at construction, not at runtime |
| `test_safe_negation_rejected_at_rule_construction` | Constructing a `Rule` with a negative literal whose variables don't appear positively (and aren't action-args) raises `ValueError` | Stage-0 §5.3 Q3.4: safe-negation is the contract the interpreter relies on |
| `test_positive_state_matching` | `find_bindings` on rule `at_ball(?b, ?r)` returns exactly the (b, r) pairs in `state_atoms["at_ball"]` | Core unification correctness |
| `test_positive_goal_matching` | Same rule with `LiteralSource.GOAL` reads from `goal_atoms`, not `state_atoms` | The state/goal discriminator is load-bearing |
| `test_negative_goal_matching` | `¬Goal at_ball(?b, ?r)` (vars also bound positively) yields the room-ball pairs NOT in the goal set | Negation under CWA, post-bind evaluation |
| `test_no_full_grounding` | Inspect `find_bindings` output ordering or call count — only typed-bind expansions happen; no Cartesian product over (`balls × rooms × balls × rooms`) for a rule with only 2 vars | Stage 0 §5.3 Option-B requirement |
| `test_interpret_is_deterministic` | Two calls with same `(state, goal, objects, legal)` return identical `GroundAction` and identical trace `(rule_index, binding)` | MCTS reproducibility relies on this |
| `test_no_rule_returns_none` | A pathological `state` (e.g. solved goal) where no rule's body is satisfiable → `interpret(...) is None` | The interpreter must signal "no firing" cleanly |
| `test_hand_policy_solves_1_ball` | Rollout hand-policy through `GripperLiteEnv(n_balls=1)`; assert `is_solved()` within horizon | End-to-end smoke |
| `test_hand_policy_solves_2_balls` | Same for `n_balls=2` | Universal-goal quantification: same rules must apply twice |
| `test_hand_policy_solves_3_balls` | Same for `n_balls=3` | The B=3 case is where lex-min tie-break determinism becomes visible |
| `test_object_renaming_preserves_behavior` | Build a renamed env (ball_0→foo, room_a→left_room) and verify the abstract action sequence (action-schema + variable-binding pattern, with renaming applied) equals the original | The whole point of lifting — Stage 0 §1.1 |

Test scaffolding:

- All tests use a fixed seed (`0`); no randomness in Stage 1.
- `conftest.py` is not extended in Stage 1 (no shared fixtures cross test files).
- `pytest -q` discovers all three files via the existing top-level `tests/` collection (no new `pytest.ini` change needed).

## Visualizations and what `01.md` (results companion) will report

[01.md](01.md) is the results companion that gets written **after** the code passes pytest. It will contain:

### §0 Introduction
- One-paragraph reminder of what the stage was for: build the lifted DSL + online-unification interpreter + minimal relational env, and prove a hand-written 4-rule policy solves Gripper-lite for B=1,2,3.
- Pointer to upstream [00_draft_lifted_policy_az.md](../00_draft_lifted_policy_az.md) (Stage 0 question and design space) and this plan.
- Statement of what is in-scope (semantic core, hand policy) and out-of-scope (grammar, MCTS, learning, multi-domain transfer) for Stage 1.

### §1 Module diagram
ASCII (or mermaid) figure showing the data flow:

```
+-------------------------+      +---------------------------+
|  GripperLiteEnv         |      |  hand_policy() : Policy   |
|  (instances/            |      |  (instances/gripper_lite/ |
|   gripper_lite/env.py)  |      |   policies.py)            |
+-----------+-------------+      +-------------+-------------+
            |                                  |
   get_state_atoms()                           |
   get_goal_atoms()                            |
   get_objects_by_type()                       |
   legal_actions()                             |
            |                                  |
            v                                  v
   +-----------------------------------------------+
   |   interpret(policy, state, goal, objs, legal) |
   |   (synthesis/lifted_interpreter.py)           |
   +-----------------------+-----------------------+
                           |
                           v
                     GroundAction
                           |
                           v
                     env.step(action)
```

### §2 The four rules, with worked binding example
A table of the four hand-rules with their body literals, action, and a concrete `θ` for one step of the B=2 rollout. Show how rule 4 fires *first* on the initial state (handempty, balls at room_a, robot at room_a, goal room_b) and binds `?from=room_a, ?b=ball_0 (lex-min), ?to=room_b`.

### §3 Rollout traces
For B=1, 2, 3 print the action sequence:
- B=1: 3 steps (`move(room_a, room_b)` is NOT first — rule 4 fires first because handempty and a misplaced ball exists, so the sequence is `pick → move → drop`). Verify and print the exact trace from the test run.
- B=2: 6 steps with explicit lex-min tie-break showing `ball_0` chosen before `ball_1`.
- B=3: 9 steps.

Render as an ASCII step table (preferred; renders consistently across viewers).

### §4 Object-renaming invariance
Print the action sequence under the renamed instance side-by-side with the canonical one. Highlight that the *schema* and the *bound variable pattern* are unchanged; only the object names differ. This is the figure that visually justifies "the same lifted policy generalizes."

### §5 Test summary table
Auto-generated table from `pytest --tb=no -v` output: 12 tests, all passing, runtimes. (Inserted by hand from the actual run; no test-introspection script is added in Stage 1.)

### §6 What's next
Pointer forward: Stage 2 will replace `hand_policy()` with a CFG-driven production emitter that `DerivationGame.step()` can drive; the interpreter built here is the LeafEvaluator's dispatch primitive.

## Critical files

### To create

- [src/alphazeropp/synthesis/lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py)
- [src/alphazeropp/synthesis/lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py)
- `src/alphazeropp/instances/gripper_lite/__init__.py`
- `src/alphazeropp/instances/gripper_lite/env.py`
- `src/alphazeropp/instances/gripper_lite/policies.py`
- `tests/test_lifted_dsl_v2.py`
- `tests/test_lifted_interpreter.py`
- `tests/test_gripper_lite_policy.py`
- [01.md](01.md) (results companion, written after tests pass)

### To reference (not modify)

- [src/alphazeropp/instances/doors/dsl/binding_engine.py](../../../../src/alphazeropp/instances/doors/dsl/binding_engine.py) — design reference for lex-min unification implementation; the new `find_bindings` mirrors its `itertools.product` iteration discipline but generalizes to (a) string objects, (b) goal/state sources, (c) negative literals.
- [src/alphazeropp/instances/doors/dsl/lifted_typed_dsl.py](../../../../src/alphazeropp/instances/doors/dsl/lifted_typed_dsl.py) — design reference for the `Var`/`Atom`/`Rule` dataclass shapes and `__post_init__` type-checking pattern.
- [src/alphazeropp/synthesis/protocols.py](../../../../src/alphazeropp/synthesis/protocols.py) — make sure the new modules don't shadow existing protocols.
- [src/alphazeropp/instances/doors/doors_pddl_lite.py](../../../../src/alphazeropp/instances/doors/doors_pddl_lite.py) — design reference for env file layout (`__init__`, `reset`, `step`, `is_solved` shape) even though `gripper_lite` will be a non-gym pure-relational env.

### To leave untouched

- [src/alphazeropp/synthesis/interpreter.py](../../../../src/alphazeropp/synthesis/interpreter.py) — the grounded interpreter. Acceptance requires NO behavior change to this file.
- All existing `tests/test_lifted_*.py` and `tests/test_relational_runtime.py` — Doors-specific, must continue passing.

## Verification

End-to-end checks:

1. **Unit tests pass.** `pytest tests/test_lifted_dsl_v2.py tests/test_lifted_interpreter.py tests/test_gripper_lite_policy.py -q` returns 12/12 passing.
2. **Existing tests unchanged.** `pytest tests/ -q` returns the same pass/fail count as `main`, plus the 12 new passes. (Doors-side `test_lifted_dsl.py` and `test_lifted_typed_binding.py` must continue to pass — sanity check that Stage 1 had no Doors blast radius.)
3. **No grounded-interpreter regression.** Confirm `src/alphazeropp/synthesis/interpreter.py` is unchanged via `git diff main -- src/alphazeropp/synthesis/interpreter.py`.
4. **Hand-rollout sanity check.** Run a small driver script that imports `GripperLiteEnv`, `hand_policy`, and `interpret`, resets the env with `n_balls=3`, runs the policy-driven loop, and observes a 9-step action trace ending in `is_solved()=True`.
5. **Manual inspection of [01.md](01.md).** Confirm §1 module diagram, §3 rollout tables, and §4 renaming side-by-side render correctly in the local Markdown viewer.
6. **No MCTS imports in new modules.** `grep -r "mcts\|derivation_game\|leaf_evaluator" src/alphazeropp/synthesis/lifted_*.py src/alphazeropp/instances/gripper_lite/` returns empty — Stage 1 has no MCTS coupling.

Stage 1 is considered complete when checks 1–6 pass and the results companion [01.md](01.md) has been written.
