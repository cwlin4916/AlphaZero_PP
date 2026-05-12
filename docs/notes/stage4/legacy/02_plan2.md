# Stage 2.5 — landscape diagnostics, grammar-safety ablations, report restructure

> **Deliverable note.** This plan is to be written verbatim (sections *Context* … *Verification*) to
> **`docs/notes/stage4/02_plan2.md`** as the executable Stage-2.5 plan. The companion results doc that
> execution must (re)populate is **`docs/notes/stage4/02.md`** (restructured per *Part A* below); after
> the code runs and `02.md` has real numbers, run **`/refine-results docs/notes/stage4/02.md`**.
> Stage-2.5 adds *no* learned network and *no* MCTS-core changes (no test forces one).

---

## Context

Stage 2 (`docs/notes/stage4/02.md`) showed the lifted grammar + `LiftedDerivationGame` + `LiftedLeafEvaluator`
mechanically integrate with the unchanged MCTS core, but uniform-prior MCTS mostly collapses onto degenerate
policies: Run A (train `B=1`, seed 0, 128 sims) finds eight `B=1` solvers and no generalizing `B=2` solver;
Run A′ (512 sims) and Run B (train `B=2`, 512 sims) find none. The best learned `B=1` solver is *spurious* —
it uses a vacuous goal literal (`Goal[carrying(...)]`, always false) and a disconnected auxiliary variable
(`Goal[at_ball(?aux_0, ?r_1)]` with `?aux_0` not tied to the dropped ball).

Stage 2.5 does three things, **without** building the learned policy/value net (Stage 3) and **without**
touching the MCTS core: (1) restructure `02.md` so claims / non-claims / hypotheses / evidence / next steps
are cleanly separated and the notation stops overloading symbols; (2) add two *optional, default-off*
grammar-safety constraints (`goal_predicate_relevance`, `require_goal_var_connected`) that excise exactly
those two pathologies while still expressing the Stage-1 hand policy; (3) add a reproducible
**landscape-enumeration / sampling** script + score-variant utilities + figures that quantify solver density,
generalization rate, pathology fractions, and reward-shape sensitivity over the program space — turning the
Stage-2 "the landscape looks deceptive" hand-wave into measured numbers and pinning down which knob (density,
reward design, connectedness) the Stage-3 learned prior actually has to fight.

---

## Refinement table

| DRAFT (`<a>`) item | Refinement | Reason |
|---|---|---|
| **A.1** rewrite `02.md` to *Executive summary / Minimal setup / Systems result / Empirical result / Interpretation / Hypotheses / Appendices* | Keep — but **preserve** the existing substantive content (formulation diagrams D1–D3, code-fidelity audit table, reproduction block, figure captions) by relocating it into the Appendices, not deleting it. Add a new *Empirical result* sub-block for the Stage-2.5 landscape tables. Companion link becomes `02_plan2.md`. | The refine-results discipline wants every claim/figure traceable; the diagrams and audit are still load-bearing. |
| **A.1** "more search makes it worse" | Restate as: *observed under the current seed/budget/grid; should be tested* — and the Stage-2.5 `HitRate_B(N)` panel (now 3 seeds @128, see clarification) is the start of that test. | Avoid an over-broad universal claim from a single trajectory. |
| **A.2** audit renaming language in `00_draft_lifted_policy_az.md` & `01.md` | `00` §5 already has the precise "Renaming caveat" — leave, add one cross-pointer sentence. `01.md` §4 (titled "Object-renaming invariance", states `rollout(σ s₀, σ g) = σ rollout(s₀,g)` as "the property under test") gets a **precision paragraph**: the interpreter is relational/name-agnostic *except* deterministic raw-object-name lexicographic tie-breaking; the test's σ is order-preserving so it demonstrates name-agnostic matching, **not** full permutation equivariance (which needs object order transported/canonicalized independently of raw names) — cross-link `00` §5. **Do not rename** `test_object_renaming_preserves_behavior` (code symbol referenced from both docs; out of scope) — only tighten prose; flag the symbol name as a future tidy-up. | The math must match what `find_bindings` actually does; renaming a referenced test risks dangling links. |
| **A.3** notation: don't reuse `A`; `cfg = (R, L_S, L_G, V_aux)`; `U_B(π)`; `p` = productions; `a` = grounded actions | Adopt exactly. Concrete remapping in `02.md`: config tuple `cfg = (R, L_S, L_G, V_aux) = (max_rules, max_pre_literals, max_goal_literals, max_aux_vars)`; legal-production set `𝒫(s)` (was `𝒜(s)`); a production `p ∈ LiftedProduction` (was `a`); task-level grounded actions stay `a`, task action sets stay `𝒜_B`/`𝒜_B(s)`; MCTS action-head width `M(cfg, Σ)`; MCTS exploration constant stays `c` (distinct letter, no clash); policy class `Π_{Γ,cfg}` (avoids the `c` clash in the draft's `Π_{Γ,c}`). | `A` was action-set ∧ aux-cap; `a` was production ∧ task action — both overloaded today. |
| **B.1** `goal_predicate_relevance: bool = False` on `LiftedGrammarConfig` | Add. Implement via a new **`goal_predicate_names: tuple[str, ...] | None = None`** field on `DomainSignature` (explicit whitelist, the draft's preferred option). `gripper_lite_signature()` sets `goal_predicate_names=("at_ball",)`. When `cfg.goal_predicate_relevance and sig.goal_predicate_names is not None`, the `goal`-kind branch of `literal_candidates` skips any predicate not in the whitelist. `None` ⇒ no-op even when the flag is `True` (documented). | Whitelist is cleanest; putting the filter inside `literal_candidates` makes `compute_max_productions` tighten automatically. |
| **B.2** `require_goal_var_connected: bool = False` — "every var in a goal literal connected to an action arg through direct occurrence or a positive precondition containing an already-connected var" | Add the flag. **Deviation (forced):** the draft's *transitive-closure-from-action-args* reading **rejects hand-policy ρ₂** `carrying(?aux_0) ∧ at_robot(?r_0) ∧ Goal[at_ball(?aux_0, ?r_1)] ⇒ move(?r_0, ?r_1)` (`?aux_0` co-occurs with no action arg) — but the draft *explicitly requires* ρ₂ to remain allowed. So implement the *semantically equivalent local rule*: **a candidate goal literal is offered only if every variable in it occurs in the rule's action arguments or in at least one positive (state) precondition literal already added** — i.e. the goal literal must not be the sole binding site of any variable (the natural generalization of safe-negation to *positive* goal literals). This rejects the spurious `… Goal[at_ball(?aux_0, ?r_1)] ⇒ drop(?b_0, ?r_1)` (`?aux_0` only in the goal lit) and allows all four hand-policy rules (verified by hand: ρ₁ `?b_0,?r_1` ✓; ρ₂ `?aux_0` in `carrying(?aux_0)` ✓; ρ₃/ρ₄ `?aux_0` in `at_ball(?aux_0,?r_1)` ✓). Document the deviation in `02_plan2.md` and in `02.md` §H4. Applies to positive *and* negated goal literals (a no-op for negated ones — already safe). | A literal reading of the draft contradicts its own acceptance example; resolve toward the example. |
| **B.3** preserve current behavior by default | Both new flags default `False`; `gripper_lite_signature().goal_predicate_names` is set but inert unless the flag is on. | Backwards compatibility (criterion 6 of stage 2 still holds). |
| **B.4** new grammar tests | Adopt all six names verbatim (see *Tests*). | — |
| **C.1–C.3** rename `num_binding_attempts` → add `num_rule_evals` (alias kept, deprecated) + test | Add `num_rule_evals` to `LiftedLeafEvaluator._rollout_one` / `_aggregate` / the `diag` dict (same value as `num_binding_attempts`, which stays). Update the module docstring and the `02.md` code-fidelity audit row to mark `num_binding_attempts` deprecated. Thread `num_rule_evals` through the run-script JSONL (`run_lifted_gripper_lite_smoke.py`, `make_lifted_gripper_canonical.py`); leave the committed `*_scores.csv` columns unchanged (they never carried it). | Minimal diff; CSV schema stable. |
| **D** new `scripts/run/analyze_lifted_gripper_landscape.py` with the listed CLI; exhaustive if feasible else reproducible stratified sampling; per-policy + summary JSON | Adopt. CLI exactly as listed (`--constraints` is `nargs="+"` over `{none, goal_predicate, connected, both}`). Enumeration = bounded DFS over `LiftedDerivationState`; if the complete-policy count would exceed `--max-policies`, fall back to seeded random walks (uniform legal production per hole), deduped by `Policy.pretty()`, recording the (`num_rules`, body-literal-count) strata so the sample distribution is reportable. Per-B solve booleans computed via fresh `GripperLiteEnv(n_balls=B)` rollouts through `interpret`; `train_solve_rate`/`eval_solve_rate` derived from `--train-balls`/`--eval-balls`; aggregate `progress/steps/noops/current_leaf_score` and all score variants reused from `LiftedLeafEvaluator` + the new score-variant module. Add a thin `scripts/run/make_lifted_gripper_landscape.py` canonical driver (mirrors `make_lifted_gripper_canonical.py`) that runs `r1` exhaustive + `r3` sampled (`--max-policies 5000`), each × `{none, goal_predicate, connected, both}`, writing committed `docs/notes/stage4/data/landscape_*.json` + capped `landscape_*.csv` (per the user's clarification). | Matches existing house pattern (`make_*` canonical driver feeding committed `data/`). |
| **E** score-variant utility (5 variants incl. lexicographic tuple) + test that `⊤⇒drop` is not a "success" | Adopt as a **new module** `src/alphazeropp/synthesis/lifted_score_variants.py` (clean import for the landscape script & plots & tests). Functions take the aggregate-metrics dict (`solve_rate`, `avg_progress`, `avg_steps`, `num_noops`); weights default to `(0.25, 0.01, 0.05)`. `lexicographic` returns `(solve_rate, avg_progress, -avg_steps, -num_noops)` and is documented "ranking only — not an MCTS scalar unless explicitly converted". `LiftedLeafEvaluator` gains `aggregate_metrics_for(policy)` (rebuilds the aggregate-shaped dict from the cached `diag`) so callers don't reach into `diag` key-renames. | Decouples score shapes from the evaluator; `⊤⇒drop` test belongs with the variants. |
| **F** six figures regenerable | Adopt as a new `scripts/plotting/plot_lifted_gripper_landscape.py` reading the committed `landscape_*.json/.csv` (+ existing `data/summary.json` / `*_best.jsonl` for the MCTS panel). Figure files: `02_landscape_solver_density.png`, `02_landscape_score_variants.png` (current-vs-progress-only scatter ⊕ degenerate-rank bars), `02_landscape_generalization.png`, `02_landscape_pathology_fractions.png`, `02_landscape_first_solver_index.png` (now 3 seeds @128 — see clarification). Reuse the existing colour constants (`RUN_A_COLOR` etc.); no new style module. | Keeps the `02_*.png` naming `02.md` already links; matches `plot_lifted_gripper_mcts.py` conventions. |
| **G** acceptance criteria | Kept as the *Verification* section, with one explicit relaxation recorded: criterion-1 pytest must pass for the four named files + all new tests; the `require_goal_var_connected` semantics is the *local* rule above (deviation documented), which still satisfies criteria 2 & 3 (hand policy expressible; `Goal[carrying(...)]` blocked under `goal_predicate_relevance`; disconnected goal vars blocked under `require_goal_var_connected`). | Criterion 3's "disconnected goal variables" is exactly what the local rule blocks. |
| (clarification) MCTS re-runs for F.6 | Re-run **Run A @128 sims for seeds 1 and 2** (~70 s each), commit their `*_best.jsonl` + `*_scores.csv` to `data/`, extend `summary.json` with `runA_seed1_sims128` / `runA_seed2_sims128`; `make_lifted_gripper_canonical.py` gains these as canonical runs. | User chose "Add seeds 1–2 for Run A @128". |
| (clarification) committed landscape configs | `r1` exhaustive + `r3` sampled (≤5k), all four constraint settings, summary JSON + capped per-policy CSV committed. | User chose option 1. |

---

## Module / class design

### 1. `src/alphazeropp/synthesis/lifted_grammar.py` (extend)

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
    goal_predicate_relevance: bool = False      # NEW — restrict goal-literal predicates to sig.goal_predicate_names
    require_goal_var_connected: bool = False    # NEW — every goal-literal var must be bound outside the goal literal

@dataclass(frozen=True)
class DomainSignature:
    types: tuple[str, ...]
    predicates: tuple[PredicateSchema, ...]
    action_schemas: tuple[ActionSchema, ...]
    goal_predicate_names: tuple[str, ...] | None = None   # NEW — None ⇒ goal_predicate_relevance is inert

def gripper_lite_signature() -> DomainSignature:
    ...  # add goal_predicate_names=("at_ball",)

# new reusable predicates (also used by the landscape script for the pathology flags):
def goal_predicate_is_relevant(pred_name: str, sig: DomainSignature) -> bool: ...
def goal_literal_var_names(lit: Literal) -> set[str]: ...
def goal_vars_locally_connected(
    goal_lit: Literal, *, action_arg_names: set[str], state_lits: tuple[Literal, ...]
) -> bool:
    """True iff every variable in goal_lit occurs in action_arg_names or in some positive state literal."""

def rule_has_vacuous_goal_predicate(rule: Rule, sig: DomainSignature) -> bool: ...
def rule_has_disconnected_goal_var(rule: Rule) -> bool: ...
```

`literal_candidates(...)`: in the `kind == "goal"` loop, after picking `pred`, `continue` if
`cfg.goal_predicate_relevance and sig.goal_predicate_names is not None and pred.name not in sig.goal_predicate_names`.

`enumerate_productions(...)`, `goal_lit` branch: after building the candidate list from `literal_candidates`,
if `cfg.require_goal_var_connected`, drop any candidate `lit` where
`not goal_vars_locally_connected(lit, action_arg_names={v.name for v in p.action_args}, state_lits=p.state_lits)`.
(Keep this filter *out* of `literal_candidates` / `compute_max_productions` — it can only shrink the set, so the
existing over-estimate bound still holds; `compute_max_productions` therefore needs no change beyond automatically
picking up the `goal_predicate_relevance` tightening through `literal_candidates`.)

### 2. `src/alphazeropp/synthesis/lifted_score_variants.py` (new)

```python
SCORE_VARIANT_NAMES = ("current", "no_step_penalty", "no_noop_penalty", "progress_only", "lexicographic")

def score_current(m, *, w_prog=0.25, w_step=0.01, w_noop=0.05) -> float: ...
def score_no_step_penalty(m, *, w_prog=0.25, w_noop=0.05) -> float: ...
def score_no_noop_penalty(m, *, w_prog=0.25, w_step=0.01) -> float: ...
def score_progress_only(m, *, w_prog=0.25) -> float: ...
def score_lexicographic(m) -> tuple[float, float, float, float]:   # (solve_rate, avg_progress, -avg_steps, -num_noops); RANKING ONLY
    ...
def all_score_variants(m) -> dict[str, float | tuple]: ...   # {name: value for name in SCORE_VARIANT_NAMES}
```

`m` is the aggregate-metrics dict: keys `solve_rate`, `avg_progress`, `avg_steps`, `num_noops` (accepts
`train_solve_rate` as a fallback for `solve_rate`). `score_current` reproduces `LiftedLeafEvaluator._score_from`'s
formula with the spec weights; the docstring states the MCTS reward = the `"current"` variant.

### 3. `src/alphazeropp/synthesis/lifted_leaf_evaluator.py` (extend)

- `_rollout_one`: keep the accumulator; return it under **both** `"num_binding_attempts"` and `"num_rule_evals"`.
- `_aggregate`: sum both.
- `__call__` `diag`: add `"num_rule_evals"` (= `"num_binding_attempts"`).
- New `def aggregate_metrics_for(self, program) -> dict`: ensures cached, returns
  `{"solve_rate": d["train_solve_rate"], "avg_progress": d["avg_progress"], "avg_steps": d["avg_steps"], "num_noops": d["num_noops"]}`.
- Module docstring: note `num_binding_attempts` is **deprecated** in favour of `num_rule_evals` (same value; both still emitted).

### 4. `src/alphazeropp/instances/gripper_lite/policies.py` (extend)

```python
def degenerate_drop_policy() -> Policy:
    """The do-nothing attractor: ⊤ ⇒ drop(?b_0, ?r_1) — drop is never legal from the start."""
    b0, r1 = Var("?b_0", "ball"), Var("?r_1", "room")
    return Policy((Rule(vars=(b0, r1), body=(), action=LiftedAction("drop", (b0, r1))),))
```

### 5. `scripts/run/analyze_lifted_gripper_landscape.py` (new)

```
argparse:
  --max-rules INT (default 1)         --max-pre-literals INT (default 2)
  --max-goal-literals INT (default 1) --max-aux-vars INT (default 1)
  --train-balls INT+ (default 1 2)    --eval-balls INT+ (default 3 4)
  --constraints {none,goal_predicate,connected,both}+ (default none)
  --max-policies INT (default 5000)   --seed INT (default 0)
  --output-json PATH (required)       --output-csv PATH (required)
```

For each constraint setting → `LiftedGrammarConfig(...)` with the right flags →
`_enumerate_or_sample(cfg, sig, max_policies, seed)`:
- DFS expand `LiftedDerivationState.initial()` over `legal_productions`; collect terminal `Policy`s deduped by `.pretty()`.
- Abort to sampling if the live count crosses `max_policies` (or a node-budget guard) → seeded random walks
  (`random.Random(seed)`), uniform legal production per hole, dedupe, until `max_policies` distinct or attempt cap.
- Returns `(mode: "exhaustive"|"sampled", policies: list[Policy], strata: Counter[(num_rules, num_body_lits)])`.

Per policy row (one CSV row, `constraint` column added): `constraint, mode, policy_pretty, num_rules,
num_state_literals, num_goal_literals, has_vacuous_goal_predicate, has_disconnected_goal_var,
train_solve_rate, eval_solve_rate, progress, steps, noops, current_leaf_score, score_no_step_penalty,
score_no_noop_penalty, score_progress_only, score_lexicographic_tuple, solves_B1, solves_B2, solves_B3, solves_B4`.
(`progress/steps/noops` = train-instance aggregates from `LiftedLeafEvaluator`; `solves_Bk` from a fresh
`GripperLiteEnv(n_balls=k)` rollout; flags from `rule_has_vacuous_goal_predicate` / `rule_has_disconnected_goal_var`
over the policy's rules.)

Summary JSON (per constraint, plus a top-level `meta` with the CLI): `mode`, `total_policies_evaluated`,
`unique_policies_evaluated`, `solver_count_by_B` ({1:…,2:…,3:…,4:…}), `b1_solver_count`,
`b1_to_b2_generalizing_count`, `b2_solver_count`, `hand_policy_score`/`hand_policy_rank` (rank by
`current_leaf_score` desc, 1-indexed; `null` if not in the set), `degenerate_drop_policy_score`/`…_rank`,
`vacuous_goal_policy_count`, `disconnected_goal_policy_count`, `best_policy_by_variant`
({variant_name: {policy_pretty, value}}), `size_strata` (the Counter as a dict).
`main(argv=None)` so the smoke test can call it.

### 6. `scripts/run/make_lifted_gripper_landscape.py` (new, thin)

Runs `analyze_lifted_gripper_landscape` for `r1` (exhaustive: `--max-rules 1 --max-pre-literals 2`) and `r3`
(sampled: `--max-rules 3 --max-pre-literals 3 --max-policies 5000`), each over `--constraints none goal_predicate
connected both`, writing `docs/notes/stage4/data/landscape_r1.json`, `landscape_r1.csv`, `landscape_r3.json`,
`landscape_r3.csv` (the CSV stays committable at ≤5k rows/constraint).

### 7. `scripts/run/run_lifted_gripper_lite_smoke.py` / `make_lifted_gripper_canonical.py` (extend)

- JSONL output dict: add `num_rule_evals` (read from evaluator metrics).
- `make_lifted_gripper_canonical.py`: add `runA_seed1_sims128`, `runA_seed2_sims128` (`--mcts-sims 128 --seed {1,2}`,
  Run-A config) to the canonical runs; `summary.json` gains those keys.

### 8. `scripts/plotting/plot_lifted_gripper_landscape.py` (new)

Reads `data/landscape_r1.json`, `data/landscape_r3.json` (+ `.csv` for scatter), `data/summary.json`,
`data/runA_seed{0,1,2}_sims128_best.jsonl`. Writes to `docs/notes/stage4/figures/`:

| File | Content |
|---|---|
| `02_landscape_solver_density.png` | grouped bars: B1 / B2 solver count (and solver fraction) per constraint setting, for `r1` and `r3`. |
| `02_landscape_score_variants.png` | (a) scatter `current_leaf_score` vs `score_progress_only` over all policies (one panel per setting or coloured); (b) bar of `degenerate_drop_policy` rank under each of the 5 variants. |
| `02_landscape_generalization.png` | bar: fraction of B1-solvers that also solve B2, per constraint setting (`r1` and `r3`). |
| `02_landscape_pathology_fractions.png` | bar: fraction of policies with a vacuous goal predicate / a disconnected goal var, per setting — `none` > 0, `both` = 0. |
| `02_landscape_first_solver_index.png` | uniform-MCTS first-solver index across the available `(seed, sims)` cells from `summary.json`/`*_best.jsonl` (now seeds 0–2 @128, plus 512 cells); annotate cells with no solver. |

### 9. Docs

- **`docs/notes/stage4/02.md` — full rewrite to:** `## Executive summary` (what Stage 2 proves; what it does
  not prove; one-sentence negative finding "uniform-prior MCTS faces a deceptive program-search landscape") ·
  `## Minimal setup` (define `𝓜_B`; `Π_{Γ,cfg}`; `U_B(π)`; `J_train`, `J_out`; `HitRate_B(N)`) ·
  `## Systems result` (grammar generates complete well-typed policies; expresses the Stage-1 hand policy up to
  α-renaming — **and still does under `goal_predicate_relevance` / `require_goal_var_connected`**; the new flags
  are default-off ablations; `LiftedDerivationGame` drives MCTS and calls `LiftedLeafEvaluator`) ·
  `## Empirical result` (Table 1: Run A / Run A′ / Run B — columns evaluable-complete-policy count, nonzero-but-
  negative-utility count, positive-score count, train-solver count, held-out-solver count, first-solver index;
  Table 2 *(Stage 2.5)*: landscape per constraint setting × {`r1` exhaustive, `r3` sampled} — #policies, #B1
  solvers, #B1→B2 generalizers, #B2 solvers, #vacuous-goal, #disconnected-goal, hand-policy rank, degenerate-drop
  rank; Table 3 *(Stage 2.5)*: best policy + degenerate-drop rank under each of the 5 score variants) ·
  `## Interpretation` (core failure = deceptive sparse terminal search in program space; "more search made it
  worse" was *observed* at the current seed/budget/grid — the `HitRate_B(N)` panel begins the test; the safety
  flags remove the two obvious grammar pathologies without losing the target) · `## Hypotheses and next
  experiments` (H1 solver-density — *partially measured by Table 2*; H2 attractor / reward-design — *probed by
  the score variants, Table 3*; H3 B=1 identifiability — *probed by per-B solver columns*; H4 grammar-
  connectedness — *implemented as the `require_goal_var_connected` ablation; document the local-rule deviation
  here*; H5 learned-prior → Stage 3) · `## Appendices` (A code-fidelity audit — updated rows: `num_rule_evals`
  added & `num_binding_attempts` deprecated, two new `LiftedGrammarConfig` fields, `DomainSignature.
  goal_predicate_names`; B module / pipeline diagram — updated; C reproduction commands — add
  `make_lifted_gripper_landscape.py`, `plot_lifted_gripper_landscape.py`, the new pytest files, the seed-1/2
  canonical runs; D figure-by-figure captions — existing D1–D3, Figs 1–5, plus the five `02_landscape_*` figs).
  Companion link → `02_plan2.md`. Notation per *Refinement table* row A.3.
- **`docs/notes/stage4/01.md` §4:** add the precision paragraph (relational/name-agnostic except raw-name lex
  tie-break; the tested σ is order-preserving ⇒ name-agnostic matching, not full permutation equivariance; needs
  object-order transport/canonicalization for that); cross-link `00` §5.
- **`docs/notes/stage4/00_draft_lifted_policy_az.md` §5/§10:** already precise — add one cross-pointer sentence
  tying §5's caveat to `01.md` §4 and to `02.md` §H4 (connectedness ablation).
- **`docs/notes/stage4/02_plan2.md`:** this plan, verbatim (sections Context … Verification).

---

## Tests

`tests/test_lifted_grammar.py` (extend):
- `test_current_grammar_still_expresses_hand_policy_rules` — default `LiftedGrammarConfig` (both flags `False`):
  for each of the four hand-policy rules there is a production sequence whose result is α-equivalent
  (`canonical_rule_form`) to it. *Why:* the Stage-2 expressibility guarantee is unchanged by the new fields.
- `test_goal_predicate_relevance_blocks_vacuous_goal_carrying` — drive a derivation to a `goal_lit` hole; with
  `goal_predicate_relevance=True` no production's `payload` literal has predicate `carrying` or `handempty`
  (only `at_ball`); with it `False` such productions reappear. *Why:* `Goal[carrying(...)]` is the spurious
  Run-A guard — the flag must excise it (criterion 3).
- `test_connectedness_blocks_free_aux_goal_literal` — build a partial rule: action `drop(?b_0,?r_1)`, aux
  `?aux_0:ball`, state lit `at_robot(?r_1)`, hole `goal_lit`; with `require_goal_var_connected=True`,
  `Goal[at_ball(?aux_0, ?r_1)]` is **not** offered (`?aux_0` ∉ action args, ∉ any positive precond); with it
  `False` it is. *Why:* this is the exact disconnected-aux pathology from the best learned policy (criterion 3).
- `test_connectedness_allows_hand_policy_move_toward_goal` — partial rule: action `move(?r_0,?r_1)`, aux
  `?aux_0:ball`, state lits `at_robot(?r_0)`, `carrying(?aux_0)`, hole `goal_lit`; with
  `require_goal_var_connected=True`, `Goal[at_ball(?aux_0, ?r_1)]` **is** offered. *Why:* guards the forced
  local-rule deviation — hand-policy ρ₂ must survive (criterion 2).
- `test_safe_negation_still_enforced` — with both flags `True`, a negated goal literal is offered only when
  every variable is positively bound or an action arg (re-assert the Stage-2 invariant under the new flags).
  *Why:* the new filters must not weaken safe negation.
- `test_compute_max_productions_is_valid_under_constraints` — for `cfg`s with each flag on/off (and `R∈{1,3,4}`),
  random derivations: `len(state.legal_productions(cfg, sig)) ≤ compute_max_productions(cfg, sig)` at every step.
  *Why:* the MCTS action-head bound must stay sound under the ablations.

`tests/test_lifted_leaf_evaluator.py` (new):
- `test_evaluator_returns_num_rule_evals_and_alias` — `metrics_for(hand_policy())` has both `num_rule_evals`
  and `num_binding_attempts`, equal; `LiftedLeafEvaluator.__doc__` (or module docstring) contains "deprecated".
  *Why:* C.3 — both keys present and the deprecation documented.
- `test_aggregate_metrics_for_shape` — `aggregate_metrics_for(hand_policy())` returns exactly the four keys
  `score_*` consume. *Why:* keeps the score-variant adapter honest.

`tests/test_lifted_score_variants.py` (new):
- `test_score_variants_on_solving_policy` — for a synthetic aggregate of a `B=2` solver (`solve_rate=1`,
  `avg_progress=1`, `avg_steps=7`, `num_noops=0`): `current ≈ 1.18`, `progress_only = 1.25`,
  `lexicographic = (1.0, 1.0, -7, 0)`. *Why:* the formulas match the spec.
- `test_degenerate_drop_policy_not_a_success` — run `degenerate_drop_policy()` through a `LiftedLeafEvaluator`
  on `B∈{1,2}`: `train_solve_rate == 0.0`, `eval_out_solve_rate == 0.0`, `current_leaf_score ≤ 0` (≈ −0.05),
  `all_score_variants(...)["current"] ≤ 0`, `score_lexicographic(...)[0] == 0.0`; assert it is **not** a solver
  (`train_solve_rate < 1.0`). *Why:* E — `⊤⇒drop` may have nonzero score but is neither a positive hit nor a
  solver, and nothing in the new code should describe it as success.

`tests/test_analyze_lifted_gripper_landscape.py` (new):
- `test_landscape_smoke_exhaustive` — call `analyze_lifted_gripper_landscape.main([...])` with `--max-rules 1
  --max-pre-literals 1 --constraints none both --max-policies 100000 -> tmp paths`: summary JSON has both
  constraint keys, each with `mode == "exhaustive"`, the required summary fields, `disconnected_goal_policy_count
  == 0` and `vacuous_goal_policy_count == 0` under `both`. *Why:* criterion 4 (a small exhaustive config runs)
  and a regression hook for the summary schema.
- `test_landscape_smoke_sampled` — `--max-rules 3 --max-policies 50 --constraints none -> tmp paths`:
  `mode == "sampled"`, `unique_policies_evaluated <= 50`, run is deterministic for a fixed `--seed`. *Why:*
  criterion 4 (a sampled larger config runs) + reproducibility.

Regression: `pytest tests/ -q --ignore=tests/test_zoning_game.py` must stay green (`test_zoning_game.py` has the
pre-existing 3.11 collection error). Run the four criterion-1 files explicitly: `pytest
tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py tests/test_lifted_interpreter.py
tests/test_gripper_lite_policy.py -v`.

---

## Visualizations and what RESULTS OUT will report

**RESULTS OUT = `docs/notes/stage4/02.md`** (the results companion). After the code runs it is rewritten to the
*Part-A structure* (see *Docs* above) and must contain:

- **Narrative:** Executive summary (3 bullets) · Minimal-setup definitions · Systems result (now incl. "hand
  policy still expressible under the two safety flags") · Interpretation (deceptive sparse terminal search;
  "more search hurt" is conditioned on seed/budget/grid) · Hypotheses H1–H5 each annotated with which Stage-2.5
  artifact bears on it.
- **Tables:**
  - *Table 1 — MCTS runs* (Run A / Run A′ / Run B, + seed-1/2 @128 rows): #unique policies, #evaluable complete
    policies, #nonzero-but-negative-utility, #positive-score, #train solvers, #held-out solvers, first-solver
    index — built from `data/summary.json` + `data/*_scores.csv`.
  - *Table 2 — landscape by grammar config* (`none / goal_predicate / connected / both` × `{r1 exhaustive, r3
    sampled}`): #policies (mode-tagged), #B1 solvers, #B1→B2 generalizers, #B2 solvers, #vacuous-goal,
    #disconnected-goal, hand-policy rank, degenerate-drop rank — from `data/landscape_r1.json` / `landscape_r3.json`.
  - *Table 3 — reward-shape sensitivity*: best policy (pretty + value) under each of `current / no_step_penalty /
    no_noop_penalty / progress_only / lexicographic`, and the degenerate-drop policy's rank under each — from
    the `best_policy_by_variant` block of the landscape summaries.
- **Figures (regenerated by `plot_lifted_gripper_landscape.py`, captioned in Appendix D):**
  `02_landscape_solver_density.png` · `02_landscape_score_variants.png` · `02_landscape_generalization.png` ·
  `02_landscape_pathology_fractions.png` · `02_landscape_first_solver_index.png` — plus the existing D1–D3 and
  Figs 1–5 carried into the appendix unchanged (the `02_*` filenames are stable).
- **Then run `/refine-results docs/notes/stage4/02.md`** to tighten structure, math notation, figure clarity,
  code-fidelity, claim discipline, concision and reproducibility per that skill.

---

## Critical files

**Create**
- `src/alphazeropp/synthesis/lifted_score_variants.py` — the 5 score variants + `all_score_variants`.
- `scripts/run/analyze_lifted_gripper_landscape.py` — landscape enumeration / sampling driver (`main(argv)`).
- `scripts/run/make_lifted_gripper_landscape.py` — thin canonical driver (`r1` exhaustive + `r3` sampled × 4 constraint settings → `data/landscape_*.json/.csv`).
- `scripts/plotting/plot_lifted_gripper_landscape.py` — the five `02_landscape_*.png` figures.
- `tests/test_lifted_leaf_evaluator.py`, `tests/test_lifted_score_variants.py`, `tests/test_analyze_lifted_gripper_landscape.py`.
- `docs/notes/stage4/02_plan2.md` — this plan, verbatim.

**Extend**
- `src/alphazeropp/synthesis/lifted_grammar.py` — `LiftedGrammarConfig` (+2 fields), `DomainSignature` (+`goal_predicate_names`), `gripper_lite_signature`, `literal_candidates` (goal-predicate filter), `enumerate_productions` (connectedness filter), new helper predicates `goal_predicate_is_relevant` / `goal_vars_locally_connected` / `rule_has_vacuous_goal_predicate` / `rule_has_disconnected_goal_var`.
- `src/alphazeropp/synthesis/lifted_leaf_evaluator.py` — emit `num_rule_evals` (alias `num_binding_attempts` kept, deprecated), add `aggregate_metrics_for`, update docstring.
- `src/alphazeropp/instances/gripper_lite/policies.py` — add `degenerate_drop_policy()`.
- `scripts/run/run_lifted_gripper_lite_smoke.py`, `scripts/run/make_lifted_gripper_canonical.py` — JSONL gains `num_rule_evals`; canonical driver gains `runA_seed{1,2}_sims128`.
- `tests/test_lifted_grammar.py` — six new tests above.
- `docs/notes/stage4/02.md` — full rewrite (Part A).
- `docs/notes/stage4/01.md` §4, `docs/notes/stage4/00_draft_lifted_policy_az.md` §5/§10 — renaming-precision edits.

**Reference (read-only, the APIs everything builds on)**
- `src/alphazeropp/synthesis/lifted_derivation.py` — `PartialRule` (`action_args`, `aux_vars`, `state_lits`, `goal_lits`, `all_vars()`, `positive_var_names()`), `LiftedDerivationState` (`initial`, `apply`, `legal_productions`, `current_hole`, `to_program`, `pretty`), `LiftedDerivationGame`.
- `src/alphazeropp/synthesis/lifted_dsl.py` — `Var`, `Atom.variables()`, `Literal`, `LiteralSource`, `Rule.__post_init__` (safe-negation), `Policy.pretty()`, `LiftedAction`.
- `src/alphazeropp/synthesis/lifted_interpreter.py` — `interpret(..., trace=True)`.
- `src/alphazeropp/synthesis/lifted_grammar.py` — `enumerate_productions`, `compute_max_productions`, `canonical_rule_form`, `action_vars_for_schema`, `aux_var_name`.
- `src/alphazeropp/instances/gripper_lite/env.py` — `GripperLiteEnv(n_balls=…)`, `ACTION_SCHEMAS`, `PREDICATE_SCHEMAS`, `horizon = 4B+4`.
- `src/alphazeropp/instances/gripper_lite/policies.py` — `hand_policy()`.
- `scripts/run/make_lifted_gripper_canonical.py`, `scripts/plotting/plot_lifted_gripper_mcts.py` — house patterns / colour constants / `data/` layout to mirror.
- `core/mcts.py`, `synthesis/derivation_game.py` (`UniformPolicyValueNet`) — **unchanged**.

---

## Verification (acceptance criteria, runnable)

1. **Tests pass.**
   - `pytest tests/test_lifted_grammar.py tests/test_lifted_derivation_game.py tests/test_lifted_interpreter.py tests/test_gripper_lite_policy.py -v` → all pass (incl. the 6 new grammar tests).
   - `pytest tests/test_lifted_leaf_evaluator.py tests/test_lifted_score_variants.py tests/test_analyze_lifted_gripper_landscape.py -v` → all pass.
   - `pytest tests/ -q --ignore=tests/test_zoning_game.py` → no regressions vs the pre-Stage-2.5 count.
2. **Hand policy still expressible under the constrained grammar** — `test_current_grammar_still_expresses_hand_policy_rules` and `test_connectedness_allows_hand_policy_move_toward_goal`; plus a manual check that `LiftedLeafEvaluator` scores `hand_policy()` ≈ 1.18 on `B=2` under `LiftedGrammarConfig(goal_predicate_relevance=True, require_goal_var_connected=True)` (the evaluator doesn't read `cfg`, but the production-sequence test does).
3. **Constrained grammar blocks the pathologies** — `test_goal_predicate_relevance_blocks_vacuous_goal_carrying` (no `Goal[carrying(...)]` when `goal_predicate_relevance=True`) and `test_connectedness_blocks_free_aux_goal_literal` (no disconnected goal var when `require_goal_var_connected=True`); cross-checked by `vacuous_goal_policy_count == 0` / `disconnected_goal_policy_count == 0` under the `both` setting in `data/landscape_r1.json`.
4. **Landscape script runs both modes** — `python scripts/run/make_lifted_gripper_landscape.py` produces `docs/notes/stage4/data/landscape_r1.json` (`mode=="exhaustive"` per constraint), `landscape_r3.json` (`mode=="sampled"`), and the matching `.csv`s; re-running with the same `--seed` reproduces `landscape_r3.*` byte-for-byte.
5. **`02.md` cleanly separates claims / non-claims / hypotheses / evidence / next experiments** — visual review of the rewritten file against the Part-A outline; every figure/table in it is regenerated by a committed driver (`make_lifted_gripper_canonical.py`, `make_lifted_gripper_landscape.py`, `plot_lifted_gripper_mcts.py`, `plot_lifted_gripper_landscape.py`, `plot_derivation_game_diagrams.py`, `plot_gripper_lite_rollout.py`) listed in its reproduction block; `00`/`01` renaming caveats updated.
6. **No over-claiming** — `02.md`'s Executive summary and Interpretation state: Stage 2 validates grammar↔MCTS integration; uniform-prior MCTS exposes a deceptive sparse-reward program-search landscape; Stage 2.5 quantifies that landscape (Tables 2–3, the `02_landscape_*` figs) and removes the obvious grammar/reward pathologies (the two default-off ablations); Stage 3 will test learned priors. No claim of learned-AlphaZero success anywhere.
7. **End-to-end demo run** (manual sanity, not committed beyond `data/`):
   ```bash
   python scripts/run/make_lifted_gripper_canonical.py --skip-slow         # Run A @128 seeds 0,1,2 (~3–4 min)
   python scripts/run/make_lifted_gripper_landscape.py                     # landscape r1 exhaustive + r3 sampled × 4 settings
   python scripts/plotting/plot_lifted_gripper_mcts.py                     # existing 5 figs
   python scripts/plotting/plot_lifted_gripper_landscape.py               # 5 new 02_landscape_* figs
   python scripts/plotting/plot_derivation_game_diagrams.py               # D1–D3
   ```
   then eyeball the five `02_landscape_*.png` for sane axes/labels/annotations, and confirm `data/summary.json` has the seed-1/2 keys and `data/landscape_*.json` have all four constraint keys with non-empty `best_policy_by_variant`.
8. **Then:** `/refine-results docs/notes/stage4/02.md`.
