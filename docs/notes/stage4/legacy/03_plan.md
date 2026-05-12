# Stage 3-A — Lifted-grammar safety cleanup, reproducible diagnostics, baselines

> Plan for the Stage 3-A coding pass. Companion (post-execution) results document: [03.md](03.md).
> Builds on [02_plan.md](02_plan.md) / [02_plan2.md](02_plan2.md) (Stage 2 / 2.5) and [02.md](02.md).

## Context

Stage 2 / 2.5 ([02.md](02.md)) established: the typed lifted grammar generates well-typed policies,
expresses the four Stage-1 hand-policy rules, and the unchanged MCTS engine drives the
`LiftedDerivationGame`. But uniform-prior MCTS only found a `B=1` solver at one seed/budget cell, never a
`B=2` solver, and the "best" found policy is brittle — it leans on a vacuous goal literal
`Goal[carrying(?b_0)]` and a disconnected auxiliary variable in `Goal[at_ball(?aux_0, ?r_1)]`. Stage 2.5
*measured* these pathologies behind two **default-off ablation flags** (`goal_predicate_relevance`,
`require_goal_var_connected`) and a landscape sampler.

Stage 3-A is the **cleanup before any learned prior** (H5 in 02.md): (1) promote the two safety flags to
grammar defaults so the pathologies are unreachable by construction; (2) add a pure per-policy pathology
report and log it everywhere; (3) add a reproducible diagnostic-grid driver (uniform MCTS vs uniform-random
baseline, seed × sims grid) that writes config JSON + git hash + JSONL + summary + CSV — nothing
claim-bearing left only in `/tmp`; (4) add a diagnostics plot script; (5) make minimal forward-pointer edits
to 02.md and write the results companion 03.md. **No neural policy/value network.** No new interpreter
semantics. No claims from unrun experiments.

Much of the draft's Task A/B/E is *already implemented* in Stage 2.5 under different names — the refinement
table maps each draft item onto what exists, so this pass is mostly: flip two defaults, update the contrast
tests, add one diagnostics module, two scripts, one new grammar test, and the docs.

## Refinement table (draft item → refinement → reason)

| # | Draft asks | Refinement | Reason |
|---|---|---|---|
| A.1 | New config fields `restrict_goal_predicates_to_goal_schema=True`, `require_goal_literal_vars_bound=True`, `allow_goal_only_existential_vars=False`, `goal_predicate_names` | **Reuse the Stage-2.5 flags, flip their defaults to `True`.** `goal_predicate_relevance` *is* `restrict_goal_predicates_to_goal_schema`; `require_goal_var_connected` *is* `require_goal_literal_vars_bound` (its negation *is* `allow_goal_only_existential_vars`). Do **not** add duplicate fields. `goal_predicate_names` already lives on `DomainSignature` (`gripper_lite_signature()` sets `("at_ball",)`) — keep it there. | The Stage-2.5 flags ([lifted_grammar.py:59-72](../../../../src/alphazeropp/synthesis/lifted_grammar.py#L59-L72)) already implement exactly the requested semantics, with passing tests. Parallel fields = dead weight. The only behavioural change Stage 3-A needs is the default flip. |
| A.2 | Goal predicate set inferred as `{"at_ball"}` | Already done: `gripper_lite_signature().goal_predicate_names == ("at_ball",)`. With `goal_predicate_relevance` now default-`True`, `Goal[carrying(...)]` / `Goal[handempty()]` candidates are filtered by default. | No change beyond the default flip. |
| A.3 | Positive-goal var safety: every var in a positive goal literal appears in action args or a positive state precond | Already implemented by `goal_vars_locally_connected` / the `require_goal_var_connected` gate in `enumerate_productions`'s `goal_lit` branch. It applies to **all** goal literals (positive and negated), subsuming A.3 + A.4. Default flip turns it on. | The "local rule" (not "transitive closure from action args") was deliberate so hand-policy ρ₂ (`?aux_0` bound only by `carrying(?aux_0)`) stays expressible — see 02.md §H4. |
| A.4 | Negative-literal safety must remain | Unchanged: enforced at grammar level (`literal_candidates`) and again in `Rule.__post_init__`. | Already correct; default flip doesn't touch it. |
| A.5 | All four hand-policy rules stay expressible | Already proven under both flags by `test_current_grammar_still_expresses_hand_policy_rules` / `test_connectedness_allows_hand_policy_move_toward_goal`. After the flip, the default-config test `test_grammar_can_express_hand_policy_rules` also exercises the strict grammar. Add a thin alias test `test_hand_policy_still_expressible_after_goal_filters`. | Regression guard; draft wants the named test. |
| B.1 | `test_gripper_goal_literals_only_use_goal_predicates` | Add: under **default** `LiftedGrammarConfig(max_rules=1)`, no production at any reachable `goal_lit` hole offers a `carrying`/`handempty` goal literal. | Pins the new default. |
| B.2 | `test_positive_goal_literal_cannot_introduce_goal_only_aux_var` | Add: at a `goal_lit` hole for `drop(?b_0,?r_1)` with state lits `at_robot(?r_1) ∧ carrying(?b_0)` and aux `?aux_0:ball`, `Goal[at_ball(?aux_0,?r_1)]` is not offered under the default config (mirror `test_connectedness_blocks_free_aux_goal_literal`). | Pins the new default. |
| B.3 | `test_hand_policy_still_expressible_after_goal_filters` | Add (see A.5). | Draft. |
| B.4 | `test_spurious_runA_solver_no_longer_expressible` | Add. Encode the Stage-2 brittle policy's three rules: `ρ₁: at_robot(?r_1) ∧ carrying(?b_0) ∧ Goal[at_ball(?aux_0,?r_1)] ⇒ drop(?b_0,?r_1)`; `ρ₂: at_robot(?r_1) ∧ ¬Goal[carrying(?b_0)] ⇒ pick(?b_0,?r_1)`; `ρ₃: ⊤ ⇒ move(?r_0,?r_1)`. Assert under the default config: (i) ρ₁'s `Goal[at_ball(?aux_0,?r_1)]` is not generable — **rejected by `require_goal_var_connected`**; (ii) ρ₂'s `Goal[carrying(?b_0)]` is not generable — **rejected by `goal_predicate_relevance`**; ρ₃ (`⊤⇒move`) *remains* generable (grammar doesn't forbid ⊤-headed action rules — note in the test). Implement via `_derive_one_rule` + `pytest.raises`, or by probing `enumerate_productions`/`literal_candidates`. | Acceptance #4. The Stage-2.5 Table-3 argmax (`at_robot(?aux_0) ⇒ move`) is *not* a good probe — its disconnected aux is in a *state* literal, which the grammar legitimately allows. |
| B.5 | Regression: `pytest tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py tests/test_gripper_lite_policy.py -v` | Keep; also run the wider Stage-2.5 suite and `pytest tests/ -q --ignore=tests/test_zoning_game.py`. Add `_CFG_PERMISSIVE = LiftedGrammarConfig(max_rules=1, goal_predicate_relevance=False, require_goal_var_connected=False)` and rebase the contrast tests (`test_goal_predicate_relevance_blocks_vacuous_goal_carrying`, `test_connectedness_blocks_free_aux_goal_literal`, and anything asserting pathologies *appear* under the bare default) onto it. Config-driven tests (`test_safe_negation_at_grammar_level`, `test_literal_lists_are_canonical`, `test_compute_max_productions_is_a_real_upper_bound`) need no change. | Default flip moves the blast radius into the contrast tests; the fix makes them explicit about which config they probe. |
| C | `analyze_policy_pathologies(policy, goal_predicate_names) -> dict`; add fields to full JSONL | New module `src/alphazeropp/synthesis/lifted_diagnostics.py`. Reuse `rule_has_vacuous_goal_predicate`, `rule_has_disconnected_goal_var`. Add the dict to `_record_from()` in `run_lifted_gripper_lite_smoke.py` and the new diagnostic-grid script. | Draft. New module keeps `lifted_leaf_evaluator.py` focused. |
| D | Reproducible runs; new `scripts/run/run_lifted_gripper_diagnostic_grid.py`; legacy vs strict configs, Run A (B1→B2, max_rules 3) & Run B (B2→B3, max_rules 4), seeds 0-9, sims {64,128,256,512}, episodes 64 (CLI), baselines uniform-MCTS + uniform-random; config JSON + git hash + JSONL + summary + best-so-far + CSV per run | Add the script, fully parameterized via CLI, but **commit only the small smoke grid** (`--seeds 0 1 --sims 64 --episodes 8`, both grammars + a strict-only shortcut) to `docs/notes/stage4/data/diagnostic_grid/`. The full sweep is *runnable but not run* (else it would be unrun claims). Add `--grammar {strict,legacy}` to `run_lifted_gripper_lite_smoke.py`; pin `make_lifted_gripper_canonical.py` to `--grammar legacy` so 02.md's Table 1 stays reproducible. Reuse the random-terminal walker from the landscape path rather than reinventing. | Draft full grid ≈ 320 MCTS runs of up to 512 sims — large; non-goals pin the committed scope. The canonical generator must not silently change 02.md's numbers. |
| E | New `scripts/plotting/plot_lifted_gripper_diagnostic_grid.py`; 6 figures | Add. Fig 4 (pathology prevalence before vs after the grammar filters) reads the **existing** `data/landscape_r3.json` — strong even without the grid. Figs 3 (best-score curves) and 6 (production-label occupancy) are meaningful from the small grid. Figs 1 (solver-rate, Wilson CIs), 2 (median first-solver idx, censored), 5 (root-entropy vs sims) need the full grid; generated anyway with a "small grid — illustrative" annotation. Add a tiny `_wilson_ci(k, n, z=1.96)` inside the plot script. | Draft; be honest about thin panels on the committed grid. |
| F | Edit 02.md: rename "learned policies"; soften "Run B @ 2048 would not change the conclusion"; add "Known grammar artifacts" para; add "Stage 3-A" para; keep math minimal | **Minimal, conservative.** 02.md already scare-quotes `"best learned policy"` and already carries the `(Non-claim.)` framing; it doesn't currently contain "Run B @ 2048 would not change the conclusion" (grep — if absent it's a documented no-op). Edits: (a) one-clause "'learned' = found by uniform-prior MCTS, not a trained net (that is Stage 3)" at first use; (b) leave figure **filenames** `02_learned_*.png` untouched, tweak prose captions where natural; (c) a 2-3 sentence Stage-3-A forward-pointer linking 03.md at the end of "Hypotheses and next experiments"; (d) the "Known grammar artifacts" content already lives in 02.md §Interpretation item 2 + Fig L4 — don't duplicate; 03.md's §Minimal setup names the three artifacts and cross-links 02.md. | 02.md is the finished Stage-2/2.5 record; heavy retrofitting risks inconsistency with its own Tables 1-2. Stage-3-A material → 03.md, per the project's plan/results convention. |
| G | Acceptance criteria | Folded into Verification. | — |

## Module / class design

### `src/alphazeropp/synthesis/lifted_grammar.py` (edit)

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
    # Grammar-safety (Stage 2.5 ablation flags, *promoted to defaults in Stage 3-A*).
    # `goal_predicate_relevance`  ≡ draft's `restrict_goal_predicates_to_goal_schema`
    # `require_goal_var_connected` ≡ draft's `require_goal_literal_vars_bound`
    #   (its negation ≡ draft's `allow_goal_only_existential_vars`)
    goal_predicate_relevance: bool = True       # was False (Stage 2.5)
    require_goal_var_connected: bool = True      # was False (Stage 2.5)

def legacy_grammar_config(**overrides) -> LiftedGrammarConfig:
    """Pre-Stage-3-A permissive grammar (both safety flags off). Used by
    make_lifted_gripper_canonical.py to keep 02.md's Table 1 reproducible."""
    return LiftedGrammarConfig(goal_predicate_relevance=False,
                               require_goal_var_connected=False, **overrides)

def strict_grammar_config(**overrides) -> LiftedGrammarConfig:
    """Stage-3-A default grammar, named for call-site clarity."""
    return LiftedGrammarConfig(**overrides)
```

No change to `enumerate_productions`, `literal_candidates`, `goal_vars_locally_connected`,
`rule_has_vacuous_goal_predicate`, `rule_has_disconnected_goal_var`, `compute_max_productions` — they read
`cfg`. (`compute_max_productions` under the new default returns a smaller `M` than 13; fine —
`LiftedDerivationGame._max_productions` follows. 02.md's "default cfg = 13" is the Stage-2 value and stays
as a historical note; 03.md records the new one.)

### `src/alphazeropp/synthesis/lifted_diagnostics.py` (new)

```python
def analyze_policy_pathologies(
    policy: Policy, *, goal_predicate_names: tuple[str, ...] | None,
) -> dict[str, bool | int]:
    # has_vacuous_goal_predicate, has_goal_only_variable, has_empty_body_rule,
    # has_top_drop_rule, has_top_pick_rule, has_top_move_rule,
    # num_goal_literals, num_negative_goal_literals, num_rules, num_body_literals
    ...

def analyze_policy_pathologies_for(policy: Policy, sig: DomainSignature) -> dict[str, bool | int]:
    return analyze_policy_pathologies(policy, goal_predicate_names=sig.goal_predicate_names)
```

`goal_predicate_names is None` ⇒ `has_vacuous_goal_predicate` always False. Pure; reuses
`rule_has_disconnected_goal_var` (goal-only var) and the predicate-whitelist check.

### `scripts/run/run_lifted_gripper_lite_smoke.py` (edit)

- `--grammar {strict,legacy}` (default `strict`) → `strict_grammar_config(max_rules=...)` / `legacy_grammar_config(max_rules=...)`.
- `_record_from(m)` gains the `analyze_policy_pathologies(...)` dict (so `--dump-all-jsonl` carries it), `"grammar_config_name"`, and `"git_commit"`. Needs the `Policy` handle for each cached metric — see the `LiftedLeafEvaluator` change below.

### `src/alphazeropp/synthesis/lifted_leaf_evaluator.py` (edit)

Keep the `Policy` object alongside the cached metrics and expose it (e.g. add `program` to the metrics dict
or a parallel accessor) so callers can attach the pathology report. Additive; no behaviour change to `__call__`/scores.

### `scripts/run/run_lifted_gripper_diagnostic_grid.py` (new)

```
CLI: --seeds (0..9) --sims (64 128 256 512) --episodes (64) --grammars {strict,legacy}
     --strict-only --runs {A,B} --baselines {mcts,random} --out-dir --raw-dir [--random-budget]
Per (run, grammar, baseline, seed, sims) cell -> run_id = f"{run}_{grammar}_{baseline}_seed{seed}_sims{sims}":
  mcts:   reuse run_lifted_gripper_lite_smoke._run_episode × episodes; capture root entropy
          H(normalised root action_N) from mcts.nodes[game.hashable_obs] each episode.
  random: random complete derivations (reuse landscape walker) until #unique == matched mcts budget
          (or --random-budget / 4096 if no paired mcts cell).
  Outputs: {raw-dir}/{run_id}/all.jsonl  (full per-policy + pathology dict; gitignored)
           {out-dir}/{run_id}/config.json, best.jsonl, summary.json  (committed; small)
           {out-dir}/diagnostic_grid.csv  rows:
             run_id, seed, sims, grammar_config_name, baseline, train_B, eval_B,
             unique_policies, terminal_evals, best_score, train_solve_rate_best, eval_out_solve_rate_best,
             first_solver_idx, solver_count, root_entropy_mean, root_entropy_final,
             n_vacuous_goal, n_goal_only_var, n_empty_body, n_top_drop, n_top_pick, n_top_move, git_commit
Committed small grid: --seeds 0 1 --sims 64 --episodes 8 --runs A B --baselines mcts random (both grammars);
the --strict-only shortcut is the acceptance-criterion command.
```

### `scripts/plotting/plot_lifted_gripper_diagnostic_grid.py` (new)

Reads `data/diagnostic_grid/diagnostic_grid.csv` (+ per-cell `summary.json`) and `data/landscape_r3.json`
(Fig 4). Writes to `docs/notes/stage4/figures/`: `03_diagnostic_solver_rate.png` (Wilson CIs),
`03_diagnostic_first_solver_index.png` (censored), `03_diagnostic_best_score_curves.png`,
`03_diagnostic_pathology_prevalence.png` (legacy=landscape `none` vs strict=`both`, plus grid cross-check),
`03_diagnostic_root_entropy.png` (mcts cells), `03_diagnostic_production_occupancy.png`. matplotlib + numpy,
Agg backend, dpi 160; titles carry the grid size + "illustrative" caveat on the small grid.

## Tests

| Test (file) | Checks | Why |
|---|---|---|
| `test_hand_policy_still_expressible_after_goal_filters` (`test_lifted_grammar.py`) | every `hand_policy().rules` reachable up to α-renaming under the **default** (now strict) config | Acceptance #3 |
| `test_gripper_goal_literals_only_use_goal_predicates` (`test_lifted_grammar.py`) | no `carrying`/`handempty` goal literal offered at any reachable `goal_lit` hole under the default config | Acceptance #4a |
| `test_positive_goal_literal_cannot_introduce_goal_only_aux_var` (`test_lifted_grammar.py`) | `Goal[at_ball(?aux_0,?r_1)]` not offered for `drop(?b_0,?r_1)` with state lits `at_robot(?r_1) ∧ carrying(?b_0)` + aux `?aux_0:ball` under the default config | Acceptance #4b |
| `test_spurious_runA_solver_no_longer_expressible` (`test_lifted_grammar.py`) | ρ₁'s `Goal[at_ball(?aux_0,?r_1)]` and ρ₂'s `Goal[carrying(?b_0)]` non-generable under the default config (comment names which filter); ρ₃ `⊤⇒move` still generable | Acceptance #4 |
| updated `test_goal_predicate_relevance_blocks_vacuous_goal_carrying`, `test_connectedness_blocks_free_aux_goal_literal` (`test_lifted_grammar.py`) | rebased onto `_CFG_PERMISSIVE` for the "present" side | keeps Stage-2.5 contrast after the flip |
| `test_analyze_policy_pathologies_*` (`test_lifted_diagnostics.py`, new) | hand_policy → all-False / counts (4 rules, 4 goal lits, 2 negative); Stage-2 brittle policy → vacuous + goal-only-var True; `⊤⇒drop` → empty-body + top-drop True; pure | Task C |
| `test_diagnostic_grid_smoke` (`test_run_lifted_gripper_diagnostic_grid.py`, new, fast) | in-process `seeds=[0] sims=[8] episodes=1 runs=["A"] grammars=["strict"] baselines=["mcts","random"]` → exit 0; writes `config.json`, `summary.json`, ≥1 line `best.jsonl`, CSV with documented header incl. pathology columns | Acceptance #5/#6 in miniature |
| existing suites | `test_lifted_grammar test_lifted_derivation_game test_gripper_lite_policy test_lifted_interpreter test_lifted_leaf_evaluator test_lifted_score_variants test_analyze_lifted_gripper_landscape` pass; `pytest tests/ -q --ignore=tests/test_zoning_game.py` ≥ prior count | Acceptance #1/#2 |

## What `03.md` will report (results companion, written after the run)

Stage-4 house style: executive summary; minimal setup pinning new symbols (and naming the three artifacts —
vacuous goal predicate, goal-only variable, ⊤-headed action rule — cross-linking 02.md); systems result
tied to passing tests + the new `M` value; empirical result with a per-cell table
`(run ∈ {A,B}) × (grammar ∈ {legacy,strict}) × (baseline ∈ {mcts,random})` at seeds 0-1 / sims 64 / 8
episodes (`#unique`, `#train solvers`, `#held-out solvers`, `first-solver idx`, `#vacuous-goal`,
`#goal-only-var`, `#⊤-headed-action`, `root entropy`) and the six `03_diagnostic_*.png` figures inline with
*Read / Mathematical / Interp.* captions and the small-grid caveat; interpretation with `(Non-claim.)` /
`(Limit.)` tags (strict grammar removes the goal-literal pathologies by construction without losing the hand
policy, but does *not* by itself fix the deceptive sparse-reward landscape — the `⊤⇒drop` attractor and low
solver density survive, see 02.md Table 2 `r3 both`; the small grid is not powered for an MCTS-vs-random
solver-rate comparison); Appendix A code-fidelity audit incl. the draft↔code name mapping; Appendix B
reproduction commands; Appendix C **What remains untested** (full 10-seed × 4-sims × 2-config × 2-task ×
2-baseline grid — script ready, not run; whether the strict grammar raises MCTS HitRate at realistic
budgets; any neural policy/value network).

## Critical files

**Create:** `docs/notes/stage4/03_plan.md` (this file); `src/alphazeropp/synthesis/lifted_diagnostics.py`;
`scripts/run/run_lifted_gripper_diagnostic_grid.py`; `scripts/plotting/plot_lifted_gripper_diagnostic_grid.py`;
`tests/test_lifted_diagnostics.py`; `tests/test_run_lifted_gripper_diagnostic_grid.py`;
`docs/notes/stage4/data/diagnostic_grid/` (committed: per-cell `config.json`/`best.jsonl`/`summary.json` + `diagnostic_grid.csv`);
`results/lifted_gripper_lite/diagnostic_grid/` (gitignored: `all.jsonl`); `docs/notes/stage4/03.md`;
`docs/notes/stage4/figures/03_diagnostic_*.png`.

**Extend:** `src/alphazeropp/synthesis/lifted_grammar.py` (flip defaults; docstrings; `legacy_/strict_grammar_config`);
`src/alphazeropp/synthesis/lifted_leaf_evaluator.py` (expose cached `Policy`);
`scripts/run/run_lifted_gripper_lite_smoke.py` (`--grammar`; pathology + `grammar_config_name` + `git_commit` in `_record_from`);
`scripts/run/make_lifted_gripper_canonical.py` (pass `--grammar legacy`);
`tests/test_lifted_grammar.py` (new tests + `_CFG_PERMISSIVE`);
`docs/notes/stage4/02.md` ("learned" clarification; prose-caption tweaks; Stage-3-A forward-pointer).

**Reference (read-only):** `lifted_derivation.py` (`LiftedDerivationGame(cfg, sig, evaluator)`, `_max_productions`),
`lifted_dsl.py` (`Policy`/`Rule`/`Literal`/`LiteralSource`), `lifted_encoding.py`, `core/mcts.py`
(`MCTS.nodes[...].action_N` for root entropy), `derivation_game.py` (`UniformPolicyValueNet`),
`instances/gripper_lite/{env,policies}.py`, `scripts/run/make_lifted_gripper_landscape.py` /
`analyze_lifted_gripper_landscape.py` (random-terminal walker; `data/landscape_r3.json` for Fig 4),
`scripts/plotting/plot_lifted_gripper_mcts.py` / `plot_lifted_gripper_landscape.py` (figure style).

## Verification (→ draft acceptance criteria G1-G7)

1. **G1** — `pytest tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py tests/test_lifted_interpreter.py tests/test_gripper_lite_policy.py tests/test_lifted_leaf_evaluator.py tests/test_lifted_score_variants.py tests/test_analyze_lifted_gripper_landscape.py -v`; `pytest tests/ -q --ignore=tests/test_zoning_game.py` (≥ prior count + new tests).
2. **G2** — `pytest tests/test_lifted_grammar.py::test_hand_policy_still_expressible_after_goal_filters tests/test_lifted_grammar.py::test_gripper_goal_literals_only_use_goal_predicates tests/test_lifted_grammar.py::test_positive_goal_literal_cannot_introduce_goal_only_aux_var tests/test_lifted_grammar.py::test_spurious_runA_solver_no_longer_expressible tests/test_lifted_diagnostics.py tests/test_run_lifted_gripper_diagnostic_grid.py -v`.
3. **G3** — `test_hand_policy_still_expressible_after_goal_filters` + existing `test_hand_policy_solves[1|2|3]` / `test_terminal_policy_evaluator_runs_without_exception` (unchanged).
4. **G4** — `test_spurious_runA_solver_no_longer_expressible`.
5. **G5** — `python scripts/run/run_lifted_gripper_diagnostic_grid.py --seeds 0 1 --sims 64 --episodes 8 --strict-only` exits 0.
6. **G6** — after (5): `docs/notes/stage4/data/diagnostic_grid/diagnostic_grid.csv` exists with the documented header + ≥1 row; per-cell `summary.json` exist; raw `all.jsonl` under `results/lifted_gripper_lite/diagnostic_grid/`. For 03.md's tables also run the legacy counterpart and `python scripts/plotting/plot_lifted_gripper_diagnostic_grid.py`; eyeball the six figures.
7. **G7** — `grep -n "would not change the conclusion\|@ *2048\|learned" docs/notes/stage4/02.md` reviewed; 03.md's "What remains untested" lists the full grid + neural prior; no 03.md table reports an unrun cell.
8. Manual figure check — Fig 4 against `landscape_r3.json`'s `none`/`both` numbers.
