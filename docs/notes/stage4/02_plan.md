# Stage 2 — minimal uniform-MCTS over the redesigned (occurrence-introduced-variable) lifted grammar

> Companion to be written **after** code runs: [02.md](02.md).
> Orientation doc: [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md) — §8 specifies the grammar redesign this plan lands; §9 the synthesis game / leaf evaluator.
> Legacy Stage-2 / 2.5 / 3 record (the *aux-var* grammar): [legacy/02.md](legacy/02.md) / [legacy/02_plan.md](legacy/02_plan.md) / [legacy/02_plan2.md](legacy/02_plan2.md) / [legacy/03.md](legacy/03.md) / [legacy/03_plan.md](legacy/03_plan.md).

## Context

The orientation doc [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md) §8 proposes replacing the grammar's **standalone `Aux` phase** — the search picks a body-local variable (`current_hole == "aux_var"`, `add_aux:{type}` / `SKIP_AUX` productions, `max_aux_vars`, `?aux_i` names, `PartialRule.aux_vars`, the `_KIND_AUX_VAR` encoding node) *before any literal justified it* — with **occurrence-introduced variables**: variables enter scope only where they first occur. §8.5 records that the live code on branch `feature/grammar-redesign` still ships the `aux_var` hole and that the cutover is "a future stage". **This is that stage.**

The decoupling of variable-introduction from literal-occurrence is the root cause of the disconnected-variable pathology Stage 2/2.5 documented ([legacy/02.md](legacy/02.md) §H4 / Table 2): uniform-prior MCTS placed `?aux_0` in `Goal[at_ball(?aux_0, ?r_1)]` with `?aux_0` occurring nowhere else, getting the unintended existential reading. Stage 2.5's `require_goal_var_connected` flag (now a default) patches the symptom; the redesign removes the cause. It does **not** remove body-local variables — hand-policy ρ₂ (`carrying(?b) ∧ at_robot(?from) ∧ Goal[at_ball(?b,?to)] ⇒ move(?from,?to)`) needs `?b`, which is not a `move` argument; under the redesign `?b` is born inside the state literal `carrying(?b)` that uses it.

Alongside the grammar cutover this stage wires a **minimal** uniform-prior MCTS run on Gripper-lite. The single research question:

> Can uniform MCTS over the redesigned lifted derivation grammar find *reasonable* Gripper-lite policies for B=1 and B=2?

**Out of scope** (explicitly): held-out generalization (no B=3 evaluation, no `J_out`); Stage-2.5 landscape enumeration; a learned policy/value net; Doors MCTS; any PG3 comparison; any claim of "learned AlphaZero" success. The MCTS prior is `UniformPolicyValueNet` — a policy it turns up is a *search artifact*. Same-`B` train *and* eval only.

The previous Stage-2/2.5/3 drivers — `scripts/run/make_lifted_gripper_canonical.py`, `scripts/run/make_lifted_gripper_landscape.py`, `scripts/run/run_lifted_gripper_diagnostic_grid.py` — and their committed artifacts under `docs/notes/stage4/data/` (`runA_*`, `runB_*`, `landscape_*`, `diagnostic_grid/`) are **legacy artifacts of the aux-var grammar**; they are left in place untouched and are *not* re-pinned to the new grammar. New runs land under `docs/notes/stage4/data/minimal_mcts/`.

## Refinement table — draft brief item → refinement → reason

| Brief item | Refinement | Reason |
|---|---|---|
| "No `aux_var` hole / no standalone `Aux` production / no `max_aux_vars` as the main way variables enter" | Holes become `policy → action_schema → pre_lit → goal_lit → (FINISH_RULE → policy) → STOP_POLICY → ⊥`. `LiftedGrammarConfig.max_aux_vars` is **removed**; add `max_body_local_vars: int = 2` (a generous cap; the Gripper hand policy needs ≤ 1). The `action_schema` production now leaves the hole at `pre_lit` (was `aux_var`). | Matches 01_draft §8.2 / §8.4. The cap keeps `compute_max_productions` closed-form and the encoding fixed-width. |
| "Action variables introduced when the schema is chosen, named by schema position" | Already so — `action_vars_for_schema(schema)` → `?{type_prefix}_{i}` (`move(?r_0,?r_1)`, `pick(?b_0,?r_1)`, `drop(?b_0,?r_1)`). Unchanged. | Existing behaviour in `lifted_grammar.py` / `lifted_derivation.py`. |
| "State literals may introduce fresh body-local vars at the literal where first used; a fresh var only because the selected literal uses it" | `literal_candidates(..., kind="pre")` gains a *fresh-var path*: for each predicate `p(t_1,…,t_n)`, each argument position is filled by an existing in-scope var of the right type **or** a freshly-introduced typed body-local var. Fresh vars are named `?v_0, ?v_1, …` by introduction order in the rule, introduced left-to-right within a literal — and since `pre_lit` only offers literals strictly greater than the last accepted one under the lex key `(source, predicate, args, negated)`, each rule body's literal order is fixed, so the naming is deterministic. No fresh-var option once the rule already has `max_body_local_vars` body-local vars; still capped by `max_pre_literals`. | This is the §8.2 rule (2). It is exactly what ρ₂'s `carrying(?b)` and ρ₄'s `at_ball(?b, ?r_1)` need. |
| "Goal literals may not introduce fresh vars; every goal-literal var must already occur in an action arg or a positive state literal; negated goal lits must satisfy safe negation; goal lits restricted to domain-goal predicates" | `literal_candidates(..., kind="goal")`: predicates restricted to `sig.goal_predicate_names` when `goal_predicate_relevance` (Gripper: `("at_ball",)`); **all arguments must be variables already in scope** — no fresh-var path; a negated `Goal[ℓ]` offered only when every variable in `ℓ` is positively covered (action args ∪ positive state-literal vars). | §8.2 rules (3)/(4). Consequence: `rule_has_disconnected_goal_var(r) == False` for every grammar-produced `r` *by construction* — `require_goal_var_connected` becomes vacuous-by-construction (kept as a config field, documented as a no-op, defaults `True`). |
| "State negation remains disabled unless already implemented/tested" | `allow_state_negation = False` (unchanged); `allow_goal_negation = True` (unchanged). | No state negation exists; out of scope. |
| "Preserve canonical literal ordering and typed schema checking" | `_lit_key` ordering kept; `Atom`/`LiftedAction` type checks via `Rule.__post_init__` kept. The lex key over a fresh-var literal uses the fresh var's (deterministic) name, so canonicalisation is preserved. | Existing invariants in `lifted_grammar.py`. |
| "The grammar must still express all four Stage-1 hand-policy rules" | ρ₁ (`carrying(?b_0)∧at_robot(?r_1)∧Goal[at_ball(?b_0,?r_1)] ⇒ drop(?b_0,?r_1)`) — all vars are `drop` args, no fresh var. ρ₃ (similar, `pick` args). ρ₂ (`carrying(?v_0)∧at_robot(?r_0)∧Goal[at_ball(?v_0,?r_1)] ⇒ move(?r_0,?r_1)`) — `?v_0:ball` is a body-local var born in `carrying`. ρ₄ (`at_robot(?r_0)∧at_ball(?v_0,?r_1)∧handempty()∧¬Goal[at_ball(?v_0,?r_1)] ⇒ move(?r_0,?r_1)`) — `?v_0:ball` born in `at_ball`, positively covered ⇒ safe negation OK. All four reachable. | Required; tested (§Tests). |
| "Refactor encoding — remove aux-node encoding; encode variables by first-use order; deterministic obs shape" | `lifted_encoding.py`: remove `_KIND_AUX_VAR`; remove `"aux_var"` from `_HOLE_ID` (renumber); drop the per-aux-var node loop — body-local vars are implicit in the encoded state literals (the literal node already carries arity + predicate). `encode_max_len`: `nodes_per_rule = 1 + cfg.max_pre_literals + cfg.max_goal_literals`. | Uniform-prior MCTS ignores the obs anyway (`UniformPolicyValueNet`); the encoding only needs to be valid + deterministic. Observation length shrinks — fine. |
| "Refactor `compute_max_productions` — sound upper bound under the new literal arg choices; randomized tests" | Drop the `1 + len(types)` aux term. For each action schema take the maximal scope = `action_vars_for_schema(s)` + `max_body_local_vars` fresh vars **per type** (over-approx), enumerate `pre` candidates (with the fresh-var path) and `goal` candidates (no fresh-var path, every var assumed safe so negation always allowed), `max(...)` + 1 for `STOP_PRE` / `FINISH_RULE`; the fixed holes contribute their small constants. | A reachable state has a real `last_lit` (only shrinks the set) and ≤ `max_body_local_vars` body-local vars (subset of the over-approx scope), so this over-estimates. Tested over many random derivations and across the `goal_predicate_relevance × require_goal_var_connected` combos. |
| "Leaf evaluator — keep `score = solve_rate + 0.25·progress − 0.01·steps − 0.05·noops`; Stage 2 trains/evaluates on the same B; log solved/progress/steps/noops/policy_pretty" | `LiftedLeafEvaluator` score formula untouched. Stage-2 instantiation: `train = eval_in = eval_out = [GripperLiteEnv(n_balls=B)]` (one frozen instance; `eval_out` aliased to avoid an empty list). Cached metrics already carry `train_solve_rate` (⇒ `solved`), `avg_progress`, `avg_steps`, `num_noops`, `policy_pretty`. | No evaluator code change. The run-script's solver/reasonable/degenerate classification (below) is done by a separate cheap classification rollout in the script (`interpret(...)` on a fresh env, ≤ horizon steps) — it needs the per-rollout *schema multiset*, which the evaluator doesn't track and which we don't add to it (keeps the score path clean). |
| "MCTS script `run_lifted_gripper_mcts_minimal.py` — `--balls{1,2} --max-rules --mcts-sims --seed --out-jsonl --out-summary --log-all-terminals`; B=1 default `max_rules=3`, B=2 default `max_rules=4`; `UniformPolicyValueNet` only; existing MCTS core unchanged; log every distinct terminal policy if flagged; always log best-so-far" | Add `--n-episodes INT` (independent MCTS plays; default 64) and optional `--c-exploration FLOAT` (default 1.5, matching the Stage-2 smoke). Uses `strict_grammar_config(max_rules=…)` + `gripper_lite_signature()` + `LiftedDerivationGame` + `UniformPolicyValueNet(game._max_productions)` + `MCTS(game, net, n_simulations=…, c_exploration=…)`, `random.seed(seed)` / `np.random.seed(seed)`. | An MCTS *episode* yields one terminal policy; you need several plays to cover the space. The Stage-2 smoke (`run_lifted_gripper_lite_smoke.py`) already used `--n-episodes 64` + `c_exploration 1.5`. `MCTS.perform_simulations` returns visit-count probs; sample a move, `game.step_wrapper`, repeat to terminal — the core is reused unchanged. |
| "Canonical driver `make_lifted_gripper_mcts_minimal.py` — B=1: sims 128/512/2048 seeds 0..4 max_rules=3; B=2: sims 512/2048/8192 seeds 0..4 max_rules=4; write under `docs/notes/stage4/data/minimal_mcts/`; `--skip-slow` may omit 8192 but keep the command documented" | Per cell, raw `all.jsonl` → `results/lifted_gripper_lite/minimal_mcts/<cell>/all.jsonl` (gitignored). Committed under `docs/notes/stage4/data/minimal_mcts/`: per-cell `<cell>_best.jsonl` + `<cell>_summary.json`, a rolled-up `minimal_mcts.csv` (one row per cell), and a top-level `summary.json`. `n_episodes` modest for cheap cells (64) and smaller for `sims=8192` (8). `--skip-slow` omits the `sims=8192` B=2 cell; the full command is documented in the driver docstring and in [02.md](02.md) §4. | `results/` is gitignored (`.gitignore` line 64); committed data must be small. Mirrors the layout `make_lifted_gripper_canonical.py` used. |
| (implicit) "a figure for 02.md" | Optional new `scripts/plotting/plot_lifted_gripper_mcts_minimal.py` (or extend `scripts/plotting/plot_lifted_gripper_mcts.py`): best-score and solver-rate vs. sims, one panel per `B`. **Tables are the primary artifact**; the figure is nice-to-have. | The brief asks for results tables; a small curve helps but the brief explicitly forbids landscape / score-variant plots. |
| Tests | NEW `tests/test_lifted_grammar_occurrence_vars.py`, `tests/test_lifted_gripper_mcts_minimal.py`. UPDATE `tests/test_lifted_grammar.py` (rewrite ~8 aux-referencing tests/helpers), `tests/test_lifted_derivation_game.py` (drop aux refs; keep action-mask + clone/stash + uniform-MCTS smoke; add a B=2 smoke), `tests/test_lifted_leaf_evaluator.py` (same-`B` setup; keep metric-keys + aggregate-shape; hand policy B=1 ⇒ `solve_rate=1.0, avg_steps=3.0, num_noops=0`). KEEP `tests/test_lifted_diagnostics.py` (the analyzer must still flag arbitrary hand-built bad policies — including a disconnected non-action var; optionally rename the `?aux_0` fixture token to `?v_0`). RUN `tests/test_run_lifted_gripper_diagnostic_grid.py`; minimal fix or `xfail` (with a pointer to this plan) if it can't be salvaged cheaply. | Acceptance criteria. The diagnostic-grid driver runs over `strict_grammar_config` — it exercises the new grammar; its pathology-count assertions (`has_vacuous_goal_predicate` / `has_goal_only_variable` both `False` under strict) should still hold. |
| "02_plan.md concise executable plan; 02.md reports grammar redesign summary, tests, B=1 results, B=2 results, best-policy examples, failure analysis if no solver, non-claims (no held-out B=3, no landscape, no score-variant plots)" | This file is `02_plan.md`. [02.md](02.md) outline in §"What 02.md will report" below. Also update [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md): §0 (drop "still ships the live `aux_var` hole"), §8.5 (proposed → **landed in Stage 2**), §10 ladder (Stage-2 row → the new minimal-MCTS result; Stage-5 row → drop the "§8.2 cutover" item), changelog/refs. Keep `legacy/02.md` / `legacy/03.md` linked as the aux-var record. | Brief; keeps the orientation doc honest about landed code. |

## The grammar redesign (occurrence-introduced variables)

**Holes & transitions** (`LiftedDerivationState.current_hole`):

```
                ADD_RULE            schema=X                       STOP_PRE
   policy ───────────────▶ action_schema ──────────────▶ pre_lit ──────────────▶ goal_lit
     │  ▲                                                  │ ⟲ pre:ℓ                │ ⟲ goal:ℓ
     │  │                                                  (≤ L_S, may introduce    (≤ L_G, vars must
     │  │ FINISH_RULE (seal q into C, hole → policy)         a fresh ?v_i body-local  already be in scope)
     │  └──────────────────────────────────────────────────────────────────────────────┘
     │ STOP_POLICY (#completed rules ≥ 1)
     ▼
     ⊥   (terminal: q = ∅, h = ⊥)
```

The `aux_var` hole and the `SKIP_AUX` / `add_aux:{type}` productions are deleted. `schema=X` introduces the schema-position action variables and leaves the hole at `pre_lit`.

**`pre_lit` productions.** For predicate `p(t_1,…,t_n)` and the current rule's variable scope `V` (= action args ∪ body-local vars introduced so far):

- if `n == 0` (nullary, e.g. `handempty()`): the single literal `p()` (if greater than `last_lit` under `_lit_key`);
- otherwise, for each tuple `(a_1,…,a_n)` where `a_j ∈ {existing vars of type t_j in V}` **and**, if the rule has `< max_body_local_vars` body-local vars, optionally a single shared placeholder per "fresh slot" — fresh vars introduced left-to-right and named `?v_{k}`, `?v_{k+1}`, … where `k` = current body-local-var count: emit `p(a_1,…,a_n)` (if greater than `last_lit`). State literals are never negated.

A fresh var is added to `partial.body_local_vars` *only* when the chosen `pre_lit` literal actually uses it. Cap: `len(partial.state_lits) < max_pre_literals` and `len(partial.body_local_vars) ≤ max_body_local_vars`.

**`goal_lit` productions.** For predicate `p(t_1,…,t_n)` with `p ∈ sig.goal_predicate_names` (when `goal_predicate_relevance`): for each tuple of *existing* in-scope vars `(a_1,…,a_n)` of matching types (no fresh-var path), emit positive `Goal[p(a_1,…,a_n)]` (if greater than `last_lit`), plus the negated form when `allow_goal_negation` and every `a_j` is positively covered. `FINISH_RULE` always offered. Cap: `len(partial.goal_lits) < max_goal_literals`.

**`FINISH_RULE`** builds `Rule(vars = action_args + body_local_vars, body = state_lits + goal_lits, action = LiftedAction(schema, action_args))`. By construction this passes `Rule.__post_init__`: every body/action var is declared (action args + body-local vars cover everything a literal can mention); safe negation holds (negated goal lits only offered with all-covered vars); no goal-only var (goal literals add nothing); no unused declared var (a body-local var is born inside the state literal that uses it). Bodyless rules (`⊤ ⇒ move(?r_0,?r_1)`) remain legal.

**`compute_max_productions(cfg, sig)`** (new): `max(2, len(action_schemas), pre_max + 1, goal_max + 1)` where `pre_max` / `goal_max` are the largest `len(literal_candidates(scope, sig, kind, last_lit=None, all-safe, cfg))` over each action schema `s`, with `scope = action_vars_for_schema(s) + (max_body_local_vars fresh vars per type)`; `pre` uses the fresh-var path, `goal` does not. Sound upper bound (tested).

## Module / class changes

- **`src/alphazeropp/synthesis/lifted_grammar.py`** — `LiftedGrammarConfig`: drop `max_aux_vars`, add `max_body_local_vars: int = 2`. `legacy_grammar_config()` / `strict_grammar_config()` unchanged in spirit (no longer mention aux vars). Remove the `aux_var` branch from `enumerate_productions`; `action_schema` is unchanged. Extend `literal_candidates` (or add a `pre_literal_candidates` helper) with the fresh-var path; `goal` candidates unchanged except they already use only `scope` vars. Rewrite `compute_max_productions`. Rename `aux_var_name` → `body_local_var_name` (or inline); in `canonical_rule_form` the canonical filler token `?aux_j` → `?v_j`. `goal_vars_locally_connected` / `rule_has_disconnected_goal_var` / `rule_has_vacuous_goal_predicate` kept (still used by `lifted_diagnostics.py` on arbitrary policies).
- **`src/alphazeropp/synthesis/lifted_derivation.py`** — `PartialRule.aux_vars` → `body_local_vars` (grown lazily in the `pre_lit` "add" branch when the literal carries a fresh var). `all_vars()` = `action_args + body_local_vars` (de-duped). `LiftedDerivationState.current_hole ∈ {"policy","action_schema","pre_lit","goal_lit",None}`. `apply`: `action_schema` → hole `"pre_lit"`; delete the `aux_var` branch; `pre_lit` "add" also extends `body_local_vars` from any fresh var in the added literal (the production payload should carry the literal *and* the list of newly-introduced vars, to keep `apply` self-contained). `pretty()` drops aux commentary, keeps the typed var signature. `LiftedDerivationGame` / `_max_productions` / `_obs_len` / `stash` / `unstash` / `clone` otherwise unchanged. **`core/mcts.py`, `core/game.py`, `UniformPolicyValueNet` untouched.**
- **`src/alphazeropp/synthesis/lifted_encoding.py`** — remove `_KIND_AUX_VAR`; remove `"aux_var"` from `_HOLE_ID` (renumber `{None:0, policy:1, action_schema:2, pre_lit:3, goal_lit:4}`); drop the `for v in p.aux_vars` node loop in `encode_state`; `encode_max_len`: `nodes_per_rule = 1 + cfg.max_pre_literals + cfg.max_goal_literals`.
- **`src/alphazeropp/synthesis/lifted_leaf_evaluator.py`** — no change to the score path. (If a future need arises for per-rollout schema counts, add them as extra metric keys; not needed here.)
- **`scripts/run/run_lifted_gripper_mcts_minimal.py`** (new) — see §"The minimal MCTS run" below.
- **`scripts/run/make_lifted_gripper_mcts_minimal.py`** (new) — see §"Canonical driver & data layout".
- **`scripts/plotting/plot_lifted_gripper_mcts_minimal.py`** (optional new) — best-score / solver-rate vs. sims, one panel per `B`, from `docs/notes/stage4/data/minimal_mcts/minimal_mcts.csv`.
- **`docs/notes/stage4/01_draft_lifted_policy_az.md`** — §0, §8.5, §10, changelog/refs (above).
- **Untouched:** `lifted_interpreter.py`, `lifted_dsl.py`, `lifted_diagnostics.py`, the Gripper-lite / Doors envs and policies, the legacy `make_lifted_gripper_canonical.py` / `make_lifted_gripper_landscape.py` / `run_lifted_gripper_diagnostic_grid.py` drivers and their `data/` outputs.

## The minimal MCTS run — `scripts/run/run_lifted_gripper_mcts_minimal.py`

```
--balls {1,2}              (required)
--max-rules INT            (default: 3 for B=1, 4 for B=2)
--mcts-sims INT            (default: 256)
--n-episodes INT           (default: 64)         independent MCTS plays
--seed INT                 (default: 0)
--c-exploration FLOAT      (default: 1.5)
--out-jsonl PATH           (required)            best-so-far records (always) + every distinct terminal policy (if --log-all-terminals)
--out-summary PATH         (required)            one summary JSON
--log-all-terminals        (flag)
```

Per episode: fresh `LiftedDerivationGame(strict_grammar_config(max_rules=…), gripper_lite_signature(), LiftedLeafEvaluator([GripperLiteEnv(B)], [GripperLiteEnv(B)], [GripperLiteEnv(B)]))`; `MCTS(game, UniformPolicyValueNet(game._max_productions), n_simulations=…, c_exploration=…)`; loop `probs = mcts.perform_simulations(None); a = sample(probs); game.step_wrapper(a)` until `game.terminated`; `program = game.get_program()`; `score = evaluator(program)`; `metrics = evaluator.metrics_for(program)`.

**Classification** (separate cheap rollout on a fresh `GripperLiteEnv(B)` via `interpret(...)`, recording the multiset of *legal* schemas fired and the progress):

- **solver** — the rollout reaches `env.is_solved()` within the horizon for that `B`;
- **reasonable** — not a solver, but `progress > 0` **and** the rollout fires ≥ 1 legal `pick`, ≥ 1 legal `move`, ≥ 1 legal `drop`;
- **degenerate** — the first `interpret(...)` is `None` (immediate stall), or only `None`/illegal attempts, or `progress == 0`.

(Per the brief: a "reasonable policy" is a solver *or* a positive-progress policy that uses all three task verbs; everything else is degenerate.)

`--out-jsonl` records (one JSON object per line): for the best-so-far stream — `{episode, unique_index, score, solved, progress, steps, noops, num_rules, num_literals, policy_pretty, classification}`; for `--log-all-terminals` — the same fields for every distinct `policy_pretty`. `--out-summary` JSON: `{balls, max_rules, mcts_sims, n_episodes, seed, c_exploration, n_unique, best_score, best_policy_pretty, best_metrics, n_solving, first_solver_episode, first_solver_unique_index, counts: {solver, reasonable, degenerate}, git_commit}`.

Smoke commands (used as the acceptance demo):

```
python scripts/run/run_lifted_gripper_mcts_minimal.py --balls 1 --mcts-sims 128 --n-episodes 8 --seed 0 \
    --out-jsonl /tmp/mcts_b1.jsonl --out-summary /tmp/mcts_b1_summary.json --log-all-terminals
python scripts/run/run_lifted_gripper_mcts_minimal.py --balls 2 --mcts-sims 256 --n-episodes 8 --seed 0 \
    --out-jsonl /tmp/mcts_b2.jsonl --out-summary /tmp/mcts_b2_summary.json --log-all-terminals
```

## Canonical driver & data layout — `scripts/run/make_lifted_gripper_mcts_minimal.py`

Matrix:
- **B=1**: `mcts_sims ∈ {128, 512, 2048}`, `seeds ∈ {0,1,2,3,4}`, `max_rules = 3`, `n_episodes = 64`.
- **B=2**: `mcts_sims ∈ {512, 2048, 8192}`, `seeds ∈ {0,1,2,3,4}`, `max_rules = 4`, `n_episodes = 64` for `sims ≤ 2048`, `n_episodes = 8` for `sims = 8192`.
- `--skip-slow` omits the `sims = 8192` B=2 cell. Full command (without `--skip-slow`) documented here and in [02.md](02.md) §4.

Per cell `<cell> = b{B}_sims{S}_seed{K}`: raw `results/lifted_gripper_lite/minimal_mcts/<cell>/all.jsonl` (gitignored). Committed under `docs/notes/stage4/data/minimal_mcts/`: `<cell>_best.jsonl`, `<cell>_summary.json`; plus `minimal_mcts.csv` (columns: `balls, mcts_sims, seed, n_episodes, n_unique, best_score, n_solving, first_solver_unique_index, count_solver, count_reasonable, count_degenerate`) and `summary.json` (per-`B` headline: best score over the grid, total solvers, first cell to find a solver, the best policy's pretty + metrics).

## Tests and acceptance criteria

Acceptance run set: `pytest tests/test_lifted_grammar_occurrence_vars.py tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py tests/test_lifted_leaf_evaluator.py tests/test_lifted_gripper_mcts_minimal.py -v`.

### New: `tests/test_lifted_grammar_occurrence_vars.py`
| Test | Checks |
|---|---|
| `test_no_aux_var_hole_reachable` | over many random derivations from `LiftedDerivationState.initial()`, `state.current_hole` is never `"aux_var"`; `_HOLE_ID` has no `"aux_var"` key. |
| `test_no_aux_production_labels` | no `LiftedProduction.label` ever contains `"aux"` / `"Aux"` / `"add_aux"` / `"SKIP_AUX"` (sampled across holes). |
| `test_state_literal_can_introduce_fresh_body_local_var` | after `ADD_RULE`, `schema=move`, the `pre_lit` productions include one whose added literal is `carrying(?v_0)` with `?v_0:ball` *not* in `{?r_0,?r_1}` — and applying it grows `partial.body_local_vars` by exactly that var. |
| `test_goal_literal_cannot_introduce_fresh_var` | at any `goal_lit` hole, every offered `goal:` literal's variables ⊆ current scope (action args ∪ body-local vars); no production introduces a new var. |
| `test_disconnected_goal_var_impossible_by_construction` | over many random *complete* derivations, `rule_has_disconnected_goal_var(r) is False` for every rule `r` in the resulting policy. |
| `test_vacuous_goal_predicate_not_offered_gripper` | with `gripper_lite_signature()` + `strict_grammar_config()`, no `goal_lit` hole ever offers a `Goal[carrying(...)]` or `Goal[handempty()]`. |
| `test_all_four_hand_policy_rules_expressible` | each of `hand_policy().rules` has a reachable derivation (matched by `canonical_rule_form`). |
| `test_terminal_policies_pass_post_init` | every random complete derivation's `to_program()` rules already satisfy `Rule.__post_init__` (constructing them didn't raise). |
| `test_compute_max_productions_is_an_upper_bound` | over many random walks, `len(state.legal_productions(cfg, sig)) <= compute_max_productions(cfg, sig)`; parametrised across the 4 `(goal_predicate_relevance, require_goal_var_connected)` flag combos and `max_rules ∈ {1,3,4}`, `max_body_local_vars ∈ {0,1,2}`. |

### New: `tests/test_lifted_gripper_mcts_minimal.py`
| Test | Checks |
|---|---|
| `test_run_b1_smoke` | `run_lifted_gripper_mcts_minimal.main([... --balls 1 --mcts-sims 16 --n-episodes 2 ...])` completes; `--out-jsonl` is non-empty valid JSONL; `--out-summary` parses and has the documented keys incl. `counts == {solver,reasonable,degenerate}`. |
| `test_run_b2_smoke` | same for `--balls 2 --mcts-sims 16 --n-episodes 2`. |
| `test_classification_buckets_are_disjoint_and_total` | for a tiny `--log-all-terminals` run, `count_solver + count_reasonable + count_degenerate == n_unique`, and every solver has `solved == True`, every degenerate has `progress == 0` or stalled. |
| `test_hand_policy_classifies_as_solver` | the script's classifier applied to `hand_policy()` on B=1 and B=2 returns `"solver"`. |

### Updated
- `tests/test_lifted_grammar.py` — rewrite `_derive_one_rule`, `_drive_to_first_goal_hole` (no aux step), `test_safe_negation_at_grammar_level`, `test_connectedness_*`, `test_positive_goal_literal_cannot_introduce_goal_only_aux_var` (now: a `goal_lit` can't introduce *any* var; the would-be disconnected literal is simply never offered), `test_spurious_runA_solver_no_longer_expressible` (the `?aux_0`-in-goal-literal shape is unreachable by construction; the bodyless `⊤ ⇒ move(...)` fragment is still derivable), `test_compute_max_productions_*`. Keep the well-typed-rules / canonical-order / α-equivalence / hand-policy-expressibility tests, updated for the new production sequence.
- `tests/test_lifted_derivation_game.py` — drop aux references; keep `test_action_mask_matches_legal_productions`, `test_clone_and_stash_state_round_trip`, `test_derivation_game_smoke_with_uniform_mcts` (B=1); add `test_derivation_game_smoke_with_uniform_mcts_b2`.
- `tests/test_lifted_leaf_evaluator.py` — same-`B` setup (`train=eval_in=eval_out=[GripperLiteEnv(B)]`); keep `test_evaluator_returns_num_rule_evals_and_alias`, `test_aggregate_metrics_for_shape` (hand policy B=1 ⇒ `solve_rate=1.0, avg_steps=3.0, num_noops=0`).
- `tests/test_lifted_diagnostics.py` — keep (the pathology analyzer must still flag a hand-built rule with a disconnected non-action var); optionally rename the `?aux_0` fixture token to `?v_0`.
- `tests/test_run_lifted_gripper_diagnostic_grid.py` — run; minimal fix or `xfail` (with a pointer to this plan) if a stale assumption breaks.

Regression: `pytest tests/ -q --ignore=tests/test_zoning_game.py` — no new failures vs. before (the pre-existing archive collection error, if surfaced, is unrelated).

Acceptance criteria: (i) the five-file `pytest` set passes; (ii) the B=1 and B=2 smoke commands run and write well-formed `--out-jsonl` / `--out-summary` with `counts` distinguishing solver / reasonable / degenerate; (iii) `grep -rn "aux_var\|max_aux_vars\|SKIP_AUX\|add_aux\|_KIND_AUX_VAR" src/` is empty; (iv) `core/mcts.py` and `lifted_interpreter.py` are unchanged (`git diff` empty); (v) `02.md` reports B=1 and B=2 results honestly (incl. a failure analysis if no B=2 solver) and claims nothing about generalization / learned AlphaZero / Doors; (vi) optionally `make_lifted_gripper_mcts_minimal.py --skip-slow` populates `docs/notes/stage4/data/minimal_mcts/`.

## What [02.md](02.md) (results companion) will report

Written **after** the acceptance set passes and the canonical driver (at least `--skip-slow`) has run. Sections:

- **§0** intro & scope — in: the grammar cutover + a minimal uniform-MCTS feasibility run on Gripper-lite B∈{1,2}, same-`B`; out: held-out generalization, landscape, learned net, Doors MCTS, PG3.
- **§1** grammar redesign summary — before/after hole-state machine diagram; the production catalogue diff (the two aux rows removed; the `pre_lit` row gains "may introduce a fresh `?v_i`"; the `goal_lit` row — connectedness now structural); a worked ρ₂ derivation showing `pre:carrying(?v_0)` introducing the body-local var; `compute_max_productions` old (13) vs new value.
- **§2** test summary — auto from `pytest --tb=no -v` over the five files; the `pytest tests/ -q` regression count vs `main`/before; the `grep` "no `aux_var` in `src/`" check.
- **§3** B=1 MCTS results — a table over (`mcts_sims` × `seed`): `n_unique`, `best_score`, `n_solving`, `first_solver_unique_index`, `count_solver / reasonable / degenerate`; a one-line summary of how sims affects solver discovery.
- **§4** B=2 MCTS results — the same table; the full canonical command (incl. `sims=8192`) documented even if `--skip-slow` was used for the committed run.
- **§5** best-policy examples — the best policy per `B` pretty-printed, with a step-by-step rollout trace (rule / grounded action / θ); a couple of representative *reasonable non-solver* policies and where they help.
- **§6** failure analysis — if no B=2 solver: what the best B=2 policies look like, exactly where they stall (which rule fires / which doesn't, missing the return-trip `move`), the logs preserved under `data/minimal_mcts/` as evidence.
- **§7** refinement deltas vs this plan — table of anything that changed during implementation.
- **§8** limitations — uniform-prior MCTS, not learned AlphaZero (any policy is a search artifact); same-`B` only — no generalization claim; no Doors MCTS; the old `data/runA_*` / `runB_*` / `landscape_*` / `diagnostic_grid/` are legacy aux-var-grammar artifacts; the optional figure (if produced) is the only plot — no landscape / score-variant plots.
- **§9** what is not claimed — the brief's non-claims, verbatim-in-spirit.

Also update [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md): §0 (drop the "still ships the live `aux_var` hole" sentence), §8.5 (proposed → landed in Stage 2; the cutover is done), §10 ladder (Stage-2 row → the new minimal-MCTS result with a pointer to `02.md`; Stage-5 row → drop the "§8.2 occurrence-introduced cutover" item), the changelog item #9 and the references list.

## Critical files

**To create** — `scripts/run/run_lifted_gripper_mcts_minimal.py`, `scripts/run/make_lifted_gripper_mcts_minimal.py`, `tests/test_lifted_grammar_occurrence_vars.py`, `tests/test_lifted_gripper_mcts_minimal.py`, `docs/notes/stage4/data/minimal_mcts/` (committed artifacts), `docs/notes/stage4/02.md`; optionally `scripts/plotting/plot_lifted_gripper_mcts_minimal.py`.
**To edit** — `src/alphazeropp/synthesis/lifted_grammar.py`, `src/alphazeropp/synthesis/lifted_derivation.py`, `src/alphazeropp/synthesis/lifted_encoding.py`, `tests/test_lifted_grammar.py`, `tests/test_lifted_derivation_game.py`, `tests/test_lifted_leaf_evaluator.py`, `tests/test_run_lifted_gripper_diagnostic_grid.py` (minimal), `docs/notes/stage4/01_draft_lifted_policy_az.md` (§0, §8.5, §10, changelog, refs).
**To reference (read-only)** — `src/alphazeropp/synthesis/lifted_dsl.py` / `lifted_interpreter.py` / `lifted_diagnostics.py` (unchanged); `src/alphazeropp/core/mcts.py` / `core/game.py` and `UniformPolicyValueNet` in `src/alphazeropp/synthesis/derivation_game.py` (unchanged); `src/alphazeropp/instances/gripper_lite/{env,policies}.py`; `scripts/run/run_lifted_gripper_lite_smoke.py` / `make_lifted_gripper_canonical.py` (the legacy run-script shape mirrored); `docs/notes/stage4/notes/derivation_game.md` (to be rewritten in a follow-up — see `notes/rewrite.md`).
**To leave untouched** — `lifted_interpreter.py`, `lifted_dsl.py`, `core/mcts.py`, `core/game.py`, the Gripper-lite / Doors envs and policies, the legacy `make_lifted_gripper_canonical.py` / `make_lifted_gripper_landscape.py` / `run_lifted_gripper_diagnostic_grid.py` and their committed `data/` outputs.

## Verification

1. **Acceptance set:** `pytest tests/test_lifted_grammar_occurrence_vars.py tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py tests/test_lifted_leaf_evaluator.py tests/test_lifted_gripper_mcts_minimal.py -v` → all green.
2. **No regression:** `pytest tests/ -q --ignore=tests/test_zoning_game.py` → same pass/fail/skip count as before plus the new passes.
3. **No aux-var residue in `src/`:** `grep -rn "aux_var\|max_aux_vars\|SKIP_AUX\|add_aux\|_KIND_AUX_VAR" src/` → empty (in `tests/` only intentional historical mentions, if any).
4. **Core untouched:** `git diff -- src/alphazeropp/core/mcts.py src/alphazeropp/core/game.py src/alphazeropp/synthesis/lifted_interpreter.py src/alphazeropp/synthesis/lifted_dsl.py` → empty.
5. **Smoke runs:** the two B=1 / B=2 smoke commands above run and produce well-formed `--out-jsonl` (≥ 1 best-so-far record) + `--out-summary` (the documented keys, `counts` summing to `n_unique`).
6. **Canonical driver:** `python scripts/run/make_lifted_gripper_mcts_minimal.py --skip-slow` populates `docs/notes/stage4/data/minimal_mcts/` (per-cell `*_best.jsonl` + `*_summary.json`, `minimal_mcts.csv`, `summary.json`); raw `all.jsonl` under `results/lifted_gripper_lite/minimal_mcts/` (gitignored).
7. **Docs render:** `02_plan.md` / `02.md` open cleanly; cross-links resolve; `02.md` reports B=1 and B=2 results (and an honest failure analysis if no B=2 solver) and makes no generalization / learned-AlphaZero / Doors claim; `01_draft_lifted_policy_az.md` no longer calls the redesign "proposed / not landed".

Stage 2 is complete when checks 1–7 pass and [02.md](02.md) has been written.

## What is not claimed

- The MCTS prior is `UniformPolicyValueNet` — this is **uniform-prior MCTS, not learned AlphaZero**; any policy it finds is a *search artifact*, not a learned policy.
- Stage 2 trains and evaluates on the **same `B`** — there is **no held-out generalization** result, no `J_out`, no B=3 evaluation.
- No Stage-2.5 landscape enumeration; no score-variant plots.
- No Doors MCTS — Doors has a relational adapter (Stage 1 rev. 1) and a hand policy, but no synthesized Doors policy.
- No PG3 comparison.
- Removing the `aux_var` *syntax* does not remove body-local variables — state literals still introduce them (hand-policy ρ₂ needs one).
- If no B=2 solver is found, that is reported honestly with the logs as evidence — it is **not** spun as a success.
