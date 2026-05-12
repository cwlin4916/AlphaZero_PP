# Stage 2 — Plug the lifted DSL into the grammar-MCTS path (refined plan)

> Refined from the draft prompt against the codebase. Upstream: [01.md](01.md). Results companion (written after the runs): [02.md](02.md).

## Context

Stage 1 delivered the **semantic core** — a lifted decision-list DSL ([src/alphazeropp/synthesis/lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py)), an online-unification interpreter ([src/alphazeropp/synthesis/lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py)), and a relational Gripper-lite env ([src/alphazeropp/instances/gripper_lite/](../../../src/alphazeropp/instances/gripper_lite/)) — proven by a hand-written 4-rule policy that solves B = 1, 2, 3. None of these talk to MCTS.

Stage 2 wires the lifted core into the **search** loop. The load-bearing question:

> Can the existing grammar-MCTS infrastructure generate complete lifted policies, evaluate them through the Stage 1 interpreter, and discover at least a tiny solving policy on Gripper-lite?

This is a systems smoke test. No baselines, no PG3 comparison, no learned network. Acceptable result: uniform-MCTS finds nonzero-reward policies, and at least one seed solves the training instance (B = 2). Generalization to B = 3 would be a bonus.

## §0 Design decisions resolved up front

Three choices were locked in via clarification before writing this plan:

- **Integration strategy:** new `LiftedDerivationGame` (mirrors the `Game` protocol). `DerivationGame.__init__` calls `compute_max_productions(...)` with a grounded-grammar-specific signature, and `_encode_obs` hardcodes `NODE_TYPE_IDS` for `Flip/IsZero/Not/And/Ite/Default`. Refactoring the existing class to be grammar-agnostic has a large blast radius on the grounded test suite and is deferred.
- **α-canonicalization:** schema-position variable names. Action variables are named by schema parameter position (`drop(?b_0, ?r_1)`, `move(?r_0, ?r_1)`). Aux variable, if added, is `?aux_0`. Two rules with the same shape pretty-print identically with no separate canonicalization pass.
- **`num_noops`:** count of `interpret()` returning `None`. Each such step terminates the rollout. Simple, matches "policy stalled."

## §1 Refinement table — draft → refined plan

| Draft item | Refined | Reason |
|---|---|---|
| "Reuse the existing DerivationGame class if it only needs a derivation state with legal_productions, apply, is_terminal, to_program" | Implement **LiftedDerivationGame** in `lifted_derivation.py` that mirrors the `Game` protocol (step/reset/get_action_mask + stash/unstash overrides). | [derivation_game.py:121–151](../../../src/alphazeropp/synthesis/derivation_game.py#L121) calls `compute_max_productions(budget, n_sites, mode, allow_and, allow_not, n_actions)` and [_encode_obs:273–286](../../../src/alphazeropp/synthesis/derivation_game.py#L273) hardcodes grounded NODE_TYPE_IDS. The class is not actually grammar-agnostic at the `__init__` boundary. Falls back to the explicit Fallback path. |
| Rule structure `(PAR, PRE, GOAL, ACT)` | Rule structure `(vars, body, action)` where `body: tuple[Literal, ...]` and each `Literal` carries `source: STATE\|GOAL` and `negated: bool`. | Stage 1's [lifted_dsl.Rule:123–184](../../../src/alphazeropp/synthesis/lifted_dsl.py#L123) has one body list, not a (PRE,GOAL) split. Source is encoded structurally on each literal. Grammar must emit this exact shape, otherwise [interpret()](../../../src/alphazeropp/synthesis/lifted_interpreter.py#L161) will not run the program. |
| Grammar parameters: `max_rules=4, max_pre_literals=3, max_goal_literals=1, max_aux_vars=1, allow_goal_negation=True, allow_state_negation=False` | Same parameters. Stored on a `LiftedGrammarConfig` dataclass passed to `LiftedDerivationState.initial(...)`. | Direct lift; matches the Stage 1 hand-policy: ρ₃ has 3 state literals + 1 negated goal literal; ρ₂/ρ₄ have one aux var. |
| "Canonical literal lists" | Enforced at production time: a `PreLiteralHole` only emits literals `>` the last accepted literal under a fixed lex key `(source.value, predicate_name, args)`. Same for `GoalLiteralHole`. | Eliminates `And`-style commutativity classes at the source — analogous to the [grounded-grammar canonicalization in _condition_productions](../../../src/alphazeropp/synthesis/derivation.py#L336). No post-hoc dedup needed. |
| "Canonicalize alpha-equivalent variable naming" | Schema-position naming. Action vars: `?{type_prefix}_{position_index}` where type_prefix is one char (`b` for ball, `r` for room). Aux var: `?aux_0`. No renaming pass; names are fixed at action-schema-choice time. | Locked in §0. Two rules with the same shape pretty-print identically by construction. |
| "Enforce safe negation" | Negated **goal** literals must use only variables that already appear in a positive (state) literal or in the action's args. Enforced at the grammar level: `GoalLiteralHole` only offers negation when the candidate goal literal's vars are already covered. State literals can't be negated (config flag). | Matches Stage 1 [Rule.__post_init__:164–180](../../../src/alphazeropp/synthesis/lifted_dsl.py#L164); grammar-level pre-filtering avoids generating programs the DSL constructor will reject. |
| `compute_max_productions` upper bound | Compute analytically: for each hole kind, count the max number of legal productions across all reachable scopes. Bound is dominated by the literal-choice hole: `O(\|preds\| × max_var_combos)`. With predicates {at_robot, at_ball, carrying, handempty}, action schemas {move, pick, drop}, and ≤ 4 vars per rule scope, the bound is < 100 per hole. | Required by [`DerivationGame.action_space = spaces.Discrete(self._max_productions)`](../../../src/alphazeropp/synthesis/derivation_game.py#L147) and by the lifted analogue. Keep the calculation in a single helper with a unit test. |
| Encoding "fixed-width for existing network interface" | Stage 2: deterministic preorder encoding into `np.ndarray(shape=(max_len,), dtype=float32)` with **7 fields per node** and `max_len = 7 * (max_rules * (1 + max_pre_literals + max_goal_literals + max_aux_vars) + 1)` (the `+1` is the policy-stop node). For Stage 2 only uniform MCTS reads this; network adequacy is a Stage 3 concern. | "For a uniform-MCTS smoke test, the encoding only needs to be valid and deterministic." A pinned `max_len` keeps `observation_space` shape stable across runs; avoids premature design and unblocks the smoke test. |
| LeafEvaluator: `solve_bonus + 0.25*progress − 0.01*steps − 0.05*noops` | Same. Concrete definitions: `progress = \|state["at_ball"] ∩ goal["at_ball"]\| / \|goal["at_ball"]\|`. `steps = env steps taken`. `num_noops = number of times interpret() returned None during rollout` (0 if rollout completes by hitting horizon or solving). Cap rollout at `env.horizon`. | RelState is a `dict[str, set[tuple]]` per [env.py:106–113](../../../src/alphazeropp/instances/gripper_lite/env.py#L106). Set intersection is the natural progress metric. |
| Cache by `program.pretty()` | Cache by `Policy.pretty()`. Use a plain `dict[str, dict]` keyed on the pretty string; cached value stores the full per-instance metrics dict (used both for `score` and for the JSONL log fields). | Matches [LeafEvaluator.__call__:98–115](../../../src/alphazeropp/synthesis/leaf_evaluator.py#L98) caching shape; richer per-policy diagnostics for the log. |
| "Run uniform MCTS or existing AlphaZero with an untrained network" | Use the existing [`UniformPolicyValueNet(action_size)`](../../../src/alphazeropp/synthesis/derivation_game.py#L293) and [`MCTS`](../../../src/alphazeropp/core/mcts.py#L39) classes unchanged. | No new infrastructure. Both already accept any `Game` and any `PolicyValueNet`. |
| Smoke runs | Run A: `--n-balls-train 1 --n-balls-eval 2`, sims ∈ {128, 512}, seeds ∈ {0, 1, 2}, `--max-rules 3`. Run B: `--n-balls-train 2 --n-balls-eval 3`, sims ∈ {512, 2048}, seeds ∈ {0, 1, 2}, `--max-rules 4`. | Lifted from the goal block. The script will accept these args; the docs companion will run the cross-product. |
| "At least one seed solves training" | **Required** for B=2 Run B; **desired but not required** for the B=3 eval-out. Required for every run: ≥1 nonzero-reward policy. | The goal explicitly weakens generalization to "desired but not required." |
| Tests (7 from draft) | Renamed to match Stage 1 style (`test_lifted_grammar_*`, `test_lifted_derivation_*`). Added one extra test: explicit α-equivalence check that two rule constructions with reordered literal additions produce the same `.pretty()`. | Mirrors the additions Stage 1 made (see [01.md §6](01.md#§6-refinement-deltas-vs-the-plan)) — adding tests during planning is expected. |
| Domain signature representation (draft implied raw dicts) | Reuse [`PredicateSchema` / `ActionSchema`](../../../src/alphazeropp/synthesis/lifted_dsl.py#L29) from `lifted_dsl.py`; `gripper_lite_signature()` is assembled from the existing module constants [`PREDICATE_SCHEMAS` / `ACTION_SCHEMAS`](../../../src/alphazeropp/instances/gripper_lite/env.py#L29). | The canonical typed-schema representation already exists and is already instantiated for Gripper-lite — re-inventing it as `dict[str, tuple]` would diverge from the rest of the lifted stack. |

## §2 Module / class design

### §2.1 `src/alphazeropp/synthesis/lifted_grammar.py`

```python
@dataclass(frozen=True)
class LiftedGrammarConfig:
    max_rules: int = 4
    max_pre_literals: int = 3
    max_goal_literals: int = 1
    max_aux_vars: int = 1
    allow_goal_negation: bool = True
    allow_state_negation: bool = False
    allow_disjunction: bool = False
    allow_constants: bool = False

@dataclass(frozen=True)
class DomainSignature:
    types: tuple[str, ...]                       # e.g. ("ball", "room")
    predicates: tuple[PredicateSchema, ...]      # reuse lifted_dsl.PredicateSchema
    action_schemas: tuple[ActionSchema, ...]     # reuse lifted_dsl.ActionSchema

def gripper_lite_signature() -> DomainSignature: ...
    # built from gripper_lite.env.PREDICATE_SCHEMAS / ACTION_SCHEMAS — do not re-declare

@dataclass(frozen=True)
class LiftedProduction:
    """Mirror of synthesis.derivation.Production but lifted-aware."""
    hole_kind: str          # "policy" | "action_schema" | "aux_var" | "pre_lit" | "goal_lit"
    label: str              # human-readable, e.g. "ADD_RULE", "schema=move", "lit=at_robot(?r_0)"
    payload: Any            # the lifted-dsl fragment to splice in (or sentinel for STOP/SKIP)

def enumerate_productions(state: "LiftedDerivationState", cfg: LiftedGrammarConfig,
                          sig: DomainSignature) -> list[LiftedProduction]: ...

def compute_max_productions(cfg: LiftedGrammarConfig, sig: DomainSignature) -> int: ...
    # Closed-form upper bound. Unit-tested directly.
```

Key invariants enforced inside `enumerate_productions`:
- A `pre_lit` hole only offers literals strictly `>` the last accepted body literal under key `(source.value, predicate, args)`.
- A `goal_lit` hole offers `(literal, negated=False)` and, if `cfg.allow_goal_negation` and all the literal's vars are safe (covered by positive body literals or action args), also `(literal, negated=True)`.
- An `aux_var` hole offers `SKIP` plus one introduction per type in `sig.types` (if `current_aux_count < cfg.max_aux_vars`).
- An `action_schema` hole offers `STOP_POLICY` (terminate the rule list) or one `schema=X` choice per `X in sig.action_schemas`. Choosing a schema fixes the action vars by **schema-position naming** (`?{prefix}_{i}`).
- The `vars` tuple emitted with each finished `Rule` includes **every** action var **and** the aux var (if added) — [`Rule.__post_init__`](../../../src/alphazeropp/synthesis/lifted_dsl.py#L128) rejects any var used in body/action that is not declared in `rule.vars`.
- Predicate literals are well-typed **by construction** — the grammar only instantiates `PredicateSchema`-compatible arg tuples. Note `Rule` itself does *not* check predicate arities (only `check_atom_against_schema` / `check_action_against_schema` do, and `Rule` does not call them), so the grammar is the sole guarantor of predicate well-typedness.

### §2.2 `src/alphazeropp/synthesis/lifted_derivation.py`

```python
@dataclass
class LiftedDerivationState:
    cfg: LiftedGrammarConfig
    sig: DomainSignature
    completed_rules: tuple[Rule, ...]              # lifted_dsl.Rule
    current_partial_rule: Optional["PartialRule"]  # None when no rule in progress
    current_hole: str                              # "policy" | "action_schema" | ...

    @staticmethod
    def initial(cfg: LiftedGrammarConfig, sig: DomainSignature) -> "LiftedDerivationState": ...

    def legal_productions(self) -> list[LiftedProduction]: ...
    def apply(self, prod: LiftedProduction) -> "LiftedDerivationState": ...  # pure: returns a fresh state, never mutates self
    def is_terminal(self) -> bool: ...
    def to_program(self) -> Policy: ...
    def pretty(self) -> str: ...

class LiftedDerivationGame(Game):
    """Mirrors DerivationGame but for the lifted grammar.
    Drives MCTS via reset/step/get_action_mask (+ stash/unstash for tree search)."""

    def __init__(self, cfg: LiftedGrammarConfig, sig: DomainSignature,
                 leaf_evaluator: "LiftedLeafEvaluator"): ...

    # -- strictly required by the Game protocol --
    def reset(self, **kw) -> tuple[np.ndarray, dict]: ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def get_action_mask(self) -> np.ndarray: ...
    # action_space = spaces.Discrete(self._max_productions); observation_space = spaces.Box(...)

    # -- overridden for performance / correctness (defaults would deepcopy the evaluator) --
    def stash_state(self) -> tuple: ...     # (deriv_state, current_productions, obs, reward, terminated, truncated, info, step_count) — leaf_evaluator stays shared
    def unstash_state(self, s: tuple) -> "LiftedDerivationGame": ...
    def clone(self) -> "LiftedDerivationGame": ...  # new game pointing at the same leaf_evaluator, then unstash_state(self.stash_state())

    @property
    def hashable_obs(self) -> str: ...      # self._deriv_state.pretty() — canonical program string as the MCTS node key
```

Notes on the `Game` interface (cross-checked against [core/game.py](../../../src/alphazeropp/core/game.py:12)):
- **Strictly required:** `step`, `reset`, `get_action_mask`, plus the `action_space` and `observation_space` fields. `reset_wrapper` / `step_wrapper` are provided by the base class.
- **`stash_state` / `unstash_state` / `clone` are overridden** to keep `leaf_evaluator` *shared* — the base-class deepcopy default would clone the evaluator's cache dict on every MCTS simulation (a perf trap). Mirror [`DerivationGame.stash_state`](../../../src/alphazeropp/synthesis/derivation_game.py#L224) (saves only the derivation state + the base-class scalars).
- **`hashable_obs` is overridden** to return `state.pretty()` (a canonical program string), so MCTS deduplicates α-equivalent partial programs for free — mirror [`DerivationGame.hashable_obs`](../../../src/alphazeropp/synthesis/derivation_game.py#L212).
- **`apply` is purely functional** — it builds and returns a new `LiftedDerivationState` (including a fresh `PartialRule`) and never mutates `self`; this is what makes `stash_state` returning the bare state object safe (mirror [`DerivationState.apply`](../../../src/alphazeropp/synthesis/derivation.py#L484)).

### §2.3 `src/alphazeropp/synthesis/lifted_encoding.py`

```python
N_FIELDS_PER_NODE = 7  # (node_kind_id, predicate_id, action_id, type_id, var_local_id, source_id, negated_bit)

def encode_max_len(cfg: LiftedGrammarConfig) -> int:
    # one node per finished literal + one per finished rule + one policy-stop node
    return N_FIELDS_PER_NODE * (cfg.max_rules * (1 + cfg.max_pre_literals + cfg.max_goal_literals + cfg.max_aux_vars) + 1)

def encode_state(state: LiftedDerivationState, sig: DomainSignature,
                 max_len: int) -> np.ndarray:
    """Deterministic preorder encoding of the partial AST.
    Returns float32 array of shape (max_len,).
    For each node in preorder, emits N_FIELDS_PER_NODE values; padded/truncated with zeros."""
```

`LiftedDerivationGame.__init__` sets `observation_space = spaces.Box(low=-inf, high=inf, shape=(encode_max_len(cfg),), dtype=np.float32)`. Node-kind id table is internal to this module; documented inline. Stage 2 only needs determinism — no training reads this.

### §2.4 `src/alphazeropp/synthesis/lifted_leaf_evaluator.py`

```python
class LiftedLeafEvaluator:
    """Cache-by-pretty leaf evaluator for lifted policies."""

    def __init__(self,
                 train_instances: list[GripperLiteEnv],     # one env per n_balls; reset() before each rollout (the env is mutable, not immutable)
                 eval_in_instances: list[GripperLiteEnv],   # may alias train_instances for single-instance B=1/B=2 runs
                 eval_out_instances: list[GripperLiteEnv],
                 *,
                 step_penalty: float = 0.01,
                 noop_penalty: float = 0.05,
                 progress_weight: float = 0.25,
                 horizon: Optional[int] = None): ...

    def __call__(self, program: Policy) -> float:
        """Return the training score. Side effects: populate self._cache[key]
        with full diagnostics dict {train_solve_rate, eval_in_solve_rate,
        eval_out_solve_rate, avg_steps, num_noops, num_binding_attempts, ...}."""

    def metrics_for(self, program: Policy) -> dict: ...  # read from cache

    def _rollout_one(self, env, program) -> dict:
        """Run interpret() in a loop. Returns dict with solved, steps,
        num_noops, num_binding_attempts, final_progress."""
```

Per-instance rollout pseudo:

```python
state = env.reset()
num_noops = num_binding_attempts = 0
for _ in range(horizon):
    if env.is_solved(): break
    out = interpret(program, env.get_state_atoms(), env.get_goal_atoms(),
                    env.get_objects_by_type(), env.legal_actions(), trace=True)
    num_binding_attempts += rules_tried_proxy(out, program)
    if out is None:
        num_noops += 1
        break    # policy stalled
    action, _rule_idx, _theta = out
    env.step(action)
```

Where `rules_tried_proxy = (rule_idx + 1) if out is not None else len(program.rules)`. Records the load-bearing "graded reward for partial programs" signal the goal asks for.

**Definition of `num_binding_attempts` (JSONL key kept for spec-compatibility with the draft).** It is `Σ` over rollout steps of *the number of rules evaluated before one fired* — i.e. `rule_idx + 1` on a firing step, `len(policy.rules)` on a stall. This is a **rule-evaluation proxy**, not the literal count of unification attempts inside [`find_bindings`](../../../src/alphazeropp/synthesis/lifted_interpreter.py#L89); instrumenting `find_bindings` would mean touching a Stage 1 module, which Stage 2 avoids. The proxy is monotone in the real quantity and adequate for the "is MCTS exploring complete programs?" debugging question. (Recorded as a refinement delta in `02.md` §6.)

### §2.5 `scripts/run/run_lifted_gripper_lite_smoke.py`

Placed under `scripts/run/` to match the 40+ existing run-scripts (the draft wrote `scripts/run_lifted_gripper_lite_smoke.py`; corrected to repo convention). Argparse surface matches the draft exactly:

```
--n-balls-train INT         (default 1)
--n-balls-eval  INT         (default 2)
--max-rules     INT         (default 3)
--mcts-sims     INT         (default 128)
--seed          INT         (default 0)
--out-jsonl     PATH        (required)
```

Behavior:
1. Build `cfg = LiftedGrammarConfig(max_rules=args.max_rules, ...)`.
2. Build `sig = gripper_lite_signature()`.
3. Build `train_envs = [GripperLiteEnv(n_balls=n_balls_train, seed=args.seed)]`, `eval_in_envs = train_envs` (the *same* object — every rollout `reset()`s the env first, so the alias is safe and intentional), `eval_out_envs = [GripperLiteEnv(n_balls=n_balls_eval, seed=args.seed)]`.
4. Build `evaluator = LiftedLeafEvaluator(...)`.
5. Build `game = LiftedDerivationGame(cfg, sig, evaluator)`; reset.
6. `net = UniformPolicyValueNet(game._max_productions)`.
7. `mcts = MCTS(game, net, n_simulations=args.mcts_sims, temperature=0.1, c_exploration=1.5)` (parameters cribbed from the existing [scripts/run/run_derivation_mcts.py](../../../scripts/run/run_derivation_mcts.py#L281)).
8. Round loop: until game terminated, `probs = mcts.perform_simulations(None); a = argmax(probs); game.step_wrapper(a)`. At terminal, read `program = state.to_program()` and `score = evaluator(program)`; if `score > best_so_far`, append a JSONL line.
9. JSONL line fields (one per "new best"): `{seed, mcts_sims, train_solve_rate, eval_in_solve_rate, eval_out_solve_rate, score, avg_steps, num_rules, num_literals, num_noops, num_binding_attempts, wall_time, policy_pretty}`.

## §3 Tests

All tests under `tests/`. Mirror Stage 1's pattern ([test_lifted_dsl_v2.py](../../../tests/test_lifted_dsl_v2.py), [test_lifted_interpreter.py](../../../tests/test_lifted_interpreter.py)) with pytest fixtures and direct assertions.

### `tests/test_lifted_grammar.py`

| Test | What it checks | Why it matters |
|---|---|---|
| `test_grammar_generates_well_typed_rules` | Apply a chain of productions from initial state through one full rule; assert the resulting `Rule` passes `Rule.__post_init__` (no exception). | Production fragments must compose into objects the Stage 1 DSL accepts. |
| `test_grammar_can_express_hand_policy_rules` | For each of the four hand-policy rules in [policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py), find a sequence of productions that produces a `Rule` α-equivalent to it. | Proves the grammar's expressivity bound is correct — the load-bearing claim that "Stage 2 can in principle find the Stage 1 policy." |
| `test_literal_lists_are_canonical` | After applying productions that try to add a duplicate literal or a literal smaller than the last accepted one, the grammar offers neither. | Eliminates α/`And`-commutativity equivalence classes from the search space. |
| `test_alpha_equivalent_rules_have_same_pretty` | Construct rule R₁ and rule R₂ by adding the same literals in the same canonical order under schema-position naming; assert `R₁.pretty() == R₂.pretty()`. | Validates the §0 decision: schema-position naming kills α-equivalence by construction. |
| `test_compute_max_productions_is_a_real_upper_bound` | For 50 random derivation paths, `len(state.legal_productions()) ≤ compute_max_productions(cfg, sig)` at every state visited. | The `Discrete(action_space)` dimension must not be undercounted, otherwise MCTS will index out of bounds. |
| `test_safe_negation_at_grammar_level` | A `goal_lit` hole offers a negated literal only when its vars are covered by already-added positive literals or action args. | Prevents the grammar from emitting programs the DSL `Rule.__post_init__` will reject — keeps "found a complete policy" probability high. |

### `tests/test_lifted_derivation_game.py`

| Test | What it checks | Why it matters |
|---|---|---|
| `test_lifted_derivation_reaches_terminal_policy` | From `state = LiftedDerivationState.initial(cfg, sig)`, apply a hand-picked sequence of productions; assert `state.is_terminal()` and `state.to_program()` is a `Policy` whose pretty contains the expected rules. | The derivation state actually terminates. |
| `test_terminal_policy_evaluator_runs_without_exception` | Take a terminal `Policy` (the hand-policy from Stage 1), pass it through `LiftedLeafEvaluator`, assert it returns a `float` and that `train_solve_rate == 1.0` for n_balls=2. | The evaluator–interpreter wiring is correct. |
| `test_derivation_game_smoke_with_uniform_mcts` | Build `LiftedDerivationGame`, `UniformPolicyValueNet`, `MCTS(n_simulations=32)`. Run one episode end-to-end with `game.step_wrapper(int(np.argmax(mcts.perform_simulations(None))))`. Assert: no exceptions, `game.terminated` reached, a `Policy` was scored. | Stage 2's load-bearing systems check: MCTS drives the lifted game. |
| `test_action_mask_matches_legal_productions` | At each state, `get_action_mask()[i] == True` iff `i < len(state.legal_productions())`. | Action-space contract for MCTS expansion. |
| `test_clone_and_stash_state_round_trip` | After `state2 = game.clone()` then `game.step_wrapper(a)`, `state2`'s observation equals the pre-step observation. Same for `stash_state`/`unstash_state`. | MCTS deep-copies game state in `_expand_node`; cloning must be sound. |

Regression: run `pytest tests/ -q --ignore=tests/test_zoning_game.py` and assert no new failures relative to Stage 1's 1144 passed / 2 skipped baseline.

## §4 Visualizations and what `02.md` will report

`02.md` is the results companion. After the smoke runs complete, it will contain:

- **§0 Introduction.** What Stage 2 wired up and the load-bearing claim ("uniform MCTS over the lifted grammar finds a nonzero-reward policy and at least one seed solves B=2").
- **§1 Module diagram.** ASCII flow `LiftedGrammarConfig + DomainSignature → LiftedDerivationState → LiftedDerivationGame → MCTS(UniformPolicyValueNet) → terminal Policy → LiftedLeafEvaluator → score`. Cross-link each box to its file.
- **§2 Grammar surface.** The four hand-policy rules expressed as production sequences, side by side with their `.pretty()`. Demonstrates the expressivity test.
- **§3 Smoke run results.** Two tables:
  - **Run A (B=1 train, B=2 eval-out):** rows = (seed, mcts_sims), columns = `best_score (dimensionless)`, `train_solve_rate`, `eval_out_solve_rate`, `num_rules`, `num_literals`, `wall_time (s)`. One row per (seed, sims) combination — 6 rows.
  - **Run B (B=2 train, B=3 eval-out):** same shape, 6 rows.
- **§4 The best-found policies.** For each run, print the highest-scoring `Policy.pretty()` and a short remark on whether it matches one of the four hand-policy rules. If a policy generalizes to B=3, call that out.
- **§5 Graded-reward diagnostic** *(first-class figure, not optional)*. A scatter of all evaluated policies in Run B at sims=2048: **x-axis** `num_literals` (total body literals across rules), **y-axis** `leaf score` (dimensionless). Annotate the max-score policy on the plot; draw a horizontal reference line at `score = 0` (the "any nonzero reward" bar) and shade the band `score ≥ 1.0` (solved). For very small N a table is an acceptable fallback, but the scatter is the headline artifact — it is the visual answer to "do partial-shape policies receive graded scores rather than all-0/all-1?".
- **§6 Refinement deltas.** Anything that changed between this plan and reality (per Stage 1's §6 convention) — plan vs actual. Must include: the `num_binding_attempts` rule-evaluation proxy (see §2.4), any line-count drift in the cited file ranges, and the final `compute_max_productions` / `encode_max_len` numbers.
- **§7 What's next.** Stage 3 will replace `UniformPolicyValueNet` with a learned policy/value net; the encoding in `lifted_encoding.py` becomes load-bearing.

## §5 Critical files

**To create:**
- [src/alphazeropp/synthesis/lifted_grammar.py](../../../src/alphazeropp/synthesis/lifted_grammar.py)
- [src/alphazeropp/synthesis/lifted_derivation.py](../../../src/alphazeropp/synthesis/lifted_derivation.py)
- [src/alphazeropp/synthesis/lifted_encoding.py](../../../src/alphazeropp/synthesis/lifted_encoding.py)
- [src/alphazeropp/synthesis/lifted_leaf_evaluator.py](../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py)
- [scripts/run/run_lifted_gripper_lite_smoke.py](../../../scripts/run/run_lifted_gripper_lite_smoke.py)
- [tests/test_lifted_grammar.py](../../../tests/test_lifted_grammar.py)
- [tests/test_lifted_derivation_game.py](../../../tests/test_lifted_derivation_game.py)
- [docs/notes/stage4/02.md](02.md) (results companion, written after the runs)

**To reuse unmodified:**
- [src/alphazeropp/core/mcts.py](../../../src/alphazeropp/core/mcts.py) — MCTS class, `perform_simulations`.
- [src/alphazeropp/core/game.py](../../../src/alphazeropp/core/game.py) — `Game` abstract base.
- [src/alphazeropp/synthesis/derivation_game.py:293–323](../../../src/alphazeropp/synthesis/derivation_game.py#L293) — `UniformPolicyValueNet`.
- [src/alphazeropp/synthesis/lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py) — `Var`, `Atom`, `Literal`, `Rule`, `Policy`, `LiftedAction`, `GroundAction`.
- [src/alphazeropp/synthesis/lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py) — `interpret`, `find_bindings`.
- [src/alphazeropp/instances/gripper_lite/env.py](../../../src/alphazeropp/instances/gripper_lite/env.py) — `GripperLiteEnv`.
- [src/alphazeropp/instances/gripper_lite/policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py) — `hand_policy` (used in tests as the reference target).

**To reference for style:**
- [scripts/run/run_derivation_mcts.py](../../../scripts/run/run_derivation_mcts.py) — argparse + JSONL logging pattern.
- [tests/test_derivation_game.py](../../../tests/test_derivation_game.py) — `TestMCTSIntegration` fixture + smoke-test idiom.

## §6 Verification

End-to-end checks, in order:

1. **Unit tests pass.** `pytest tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py -v` — all 11 tests above pass.
2. **No regressions.** `pytest tests/ -q --ignore=tests/test_zoning_game.py` — count remains 1144 passed (or higher) / 2 skipped.
3. **Smoke run A.**
   ```
   python scripts/run/run_lifted_gripper_lite_smoke.py \
     --n-balls-train 1 --n-balls-eval 2 \
     --max-rules 3 --mcts-sims 128 --seed 0 \
     --out-jsonl /tmp/lifted_gripper_b1_seed0.jsonl
   ```
   Acceptance: script exits 0; `/tmp/lifted_gripper_b1_seed0.jsonl` has ≥ 1 line; the highest-`score` line has `score > 0.0`.
4. **Smoke run B.**
   ```
   python scripts/run/run_lifted_gripper_lite_smoke.py \
     --n-balls-train 2 --n-balls-eval 3 \
     --max-rules 4 --mcts-sims 512 --seed 0 \
     --out-jsonl /tmp/lifted_gripper_b2_seed0.jsonl
   ```
   Acceptance: same as Run A. Bonus: any seed in {0, 1, 2} at sims=2048 gets `train_solve_rate == 1.0`.
5. **Manual: results companion `02.md`.** Once the runs complete, write `docs/notes/stage4/02.md` per §4 above. Re-run `/refine-results` on it before declaring Stage 2 done.

## §7 Out of scope

Per the goal's "Success criteria" — these are explicitly **not** part of Stage 2:

- Any baseline comparison (PG3, grounded-grammar AlphaZero, hand-written-policy benchmarks).
- Learned policy/value network. Stage 2 uses `UniformPolicyValueNet` only.
- Doors-domain integration. Gripper-lite is the sole domain.
- Performance characterization / wall-clock claims. We log wall_time per JSONL line as a sanity number, not as a result.
- Generalization claim for B=3. Recorded if it happens; not a success criterion.
