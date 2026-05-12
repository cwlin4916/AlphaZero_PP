# Plan — rewrite the Stage-4 orientation doc (`00_draft → 01_draft`) + archive the legacy stage notes

> Plan of record for the docs refactor that produced [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md)
> and the [`legacy/`](legacy/) reorganization. Companion: the rewritten orientation doc itself
> ([01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md)); upstream: the now-superseded
> [00_draft_lifted_policy_az.md](00_draft_lifted_policy_az.md).

## Context

Two coupled, **docs-only** cleanup tasks on `docs/notes/stage4/` (no source or test changes).

**Task A — rewrite the orientation doc.** [00_draft_lifted_policy_az.md](00_draft_lifted_policy_az.md)
is the project's orientation/theory document. Its derivation-grammar section (00 §6) exposed a
*standalone `Aux` phase*: the search picked a body-local variable (`current_hole = "aux_var"`,
`add_aux:{type}` productions, `max_aux_vars`, `?aux_i` names, `PartialRule.aux_vars`) **before any
literal justified why that variable was needed**. That decoupling is the root cause of the
disconnected-variable pathology Stage 2/2.5 documented ([legacy/02.md](legacy/02.md) §H4 / Table 2):
uniform-prior MCTS placed `?aux_0` in `Goal[at_ball(?aux_0, ?r_1)]` with `?aux_0` occurring nowhere
else, getting the unintended existential-goal reading. Stage 2.5's `require_goal_var_connected` flag
patches the symptom; the cause is the `Aux` phase. We rewrote the doc as a new file
[01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md) whose grammar section presents a
**proposed occurrence-introduced-variable grammar** — no `Aux` nonterminal; variables enter scope by
occurrence (action arguments → state literals may introduce fresh typed body-local vars → goal
literals introduce nothing) — keeping everything else (typed DSL, `LiteralSource`, safe negation,
first-applicable semantics, online unification, the two-level MDP / derivation-game architecture, the
Gripper-lite MDP, MCTS over the derivation game). It is framed as a *proposed* redesign because the
code on branch `feature/grammar-redesign` still ships the live `aux_var` hole (the new doc's §8.5
records this). The rewrite **keeps body-local variables** — hand-policy ρ₂
(`carrying(?b) ∧ at_robot(?from) ∧ Goal[at_ball(?b, ?to)] ⇒ move(?from, ?to)`) needs `?b`, which is
*not* a `move` argument — so the redesign removes the `Aux` *syntax*, not body-local variables. The
rewrite also adds a literature-positioning section (§2) and a Doors-PDDL-contract section (§4.2), and
sharpens the non-claims (uniform MCTS ≠ learned AlphaZero; the Doors relational adapter is not yet
implemented).

**Task B — archive the numbered stage notes.** The seven numbered stage notes (`01.md`, `01_plan.md`,
`02.md`, `02_plan.md`, `02_plan2.md`, `03.md`, `03_plan.md`) moved into a new `docs/notes/stage4/legacy/`
directory, with every relative link they contain (and every link *to* them) fixed.
[00_draft_lifted_policy_az.md](00_draft_lifted_policy_az.md) and the auxiliary notes
([literature_orientation.md](literature_orientation.md),
[notes_hand_policy_plan_length.md](notes_hand_policy_plan_length.md),
[proposals_lifted_policy_az.md](proposals_lifted_policy_az.md)) stay at the top level, as do
`figures/`, `data/`, and `notes/`.

There is **no separate post-run results companion** — this is a docs refactor, not an experiment, so
the "result" is the rewritten doc itself; the full content of `01_draft_lifted_policy_az.md` is
specified by the section outline below.

## Refinement table (brief item → refinement → reason)

| Brief / DRAFT item | Refinement | Reason |
|---|---|---|
| Output doc name `01_draft_lifted_policy_az.md` | Used verbatim. After Task B it is the only `01*` file at top level. | User specified it verbatim. |
| "Make `goal_predicate_relevance` / `require_goal_var_connected` grammar invariants, not Stage-2.5 patches" | They are **already** defaults in `LiftedGrammarConfig` (`= True`; `legacy_grammar_config()` recovers the permissive grammar) — Stage-3-A promoted them. The doc states them as current invariants. Under the §8 redesign, goal-var connectedness becomes *structural* (goal literals introduce no variables) so `require_goal_var_connected` is vacuous-by-construction; `goal_predicate_relevance` stays a `DomainSignature.goal_predicate_names` whitelist. | Matches the code (`src/alphazeropp/synthesis/lifted_grammar.py`, `goal_vars_locally_connected`). |
| "Remove `Aux` nonterminal / `aux_var` hole / `max_aux_vars`" | Presented in §8 as the *proposed* grammar; §8.5 records that the live code still has `current_hole = "aux_var"`, `add_aux:{t}` productions, `max_aux_vars`, `PartialRule.aux_vars`, `?aux_i` names, and the `_KIND_AUX_VAR` encoding node — cutover deferred. | User chose "present as a proposed redesign"; branch `feature/grammar-redesign` is mid-migration. |
| "Doors PDDL is part of the Stage-1 semantic-core setup" + §4.2 "Doors PDDL contract" + §10 "Stage 1: … + Doors PDDL adapters, no MCTS" | Presented the **relational adapter** as a *planned* Stage-1 deliverable: `DoorsPDDLLiteEnv` is currently a gymnasium flat-vector env (`step(action:int) -> (np.ndarray, …)`, `reset(...) -> (np.ndarray, dict)`), has **no** `get_state_atoms/get_goal_atoms/get_objects_by_type`, and is **not** wired into the lifted interpreter. §4.2 spells out the adapter (types `{room, location, key}`; predicates `{at_loc(l), unlocked(r), key_avail(k)}`; actions `{move_to(l), pick(k), noop}`) and the `get_goal_atoms()` env-contract addition, with an explicit "not yet implemented" flag. | Honest with the codebase; consistent with the rigor constraints. |
| "Don't call uniform-MCTS-found policies 'learned policies'" | Prose says "search artifact." Figures named `02_learned_policies*.png` are **not** renamed (out of scope). | Rigor constraint; figure renames are noise. |
| "Mention B=2 needs the 4-rule policy and plan length 4B−1" | §6 carries the boxed 4-rule hand policy and `plan_length(B) = 4B − 1`, noting ρ₄'s return-trip `move(room_b, room_a)` between drops is why B≥2 needs all four rules; points to `notes_hand_policy_plan_length.md`. | Brief + `legacy/01.md` §3. |
| Renaming caveat | Kept in §7: name-agnostic matching ≠ arbitrary permutation equivariance; raw-object-name lexicographic tie-break; the Stage-1 renaming test (`ball_0→foo`, `room_a→left_room`, B=1) demonstrates the weaker property only. | Rigor constraint; matches `legacy/01.md` §4. |
| Legacy move scope | Moved exactly: `01.md`, `01_plan.md`, `02.md`, `02_plan.md`, `02_plan2.md`, `03.md`, `03_plan.md` → `docs/notes/stage4/legacy/`. Kept `00_draft_*`, `literature_orientation.md`, `notes_hand_policy_plan_length.md`, `proposals_lifted_policy_az.md`, `notes/`, `figures/`, `data/` at top level. | User choice: "numbered stage notes only". |

## Document outline — `01_draft_lifted_policy_az.md`

Twelve sections matching the brief's required structure (boxed-math house style; verbatim formalism
from `00_draft` where unchanged):

- **§0 Status and reading guide** — what this doc is; the grammar section is a *proposed* redesign,
  not yet in code; supersedes `00_draft`; pointer to the changelog; reading guide for §1–§12.
- **§1 Problem statement** — `{M_i}`, typed grammar `Γ` over `Π_Γ`, `J_train` / `J_out`, the
  research question (generic AlphaZero MCTS over `Γ` vs PG3's GBFS + STRIPS-A* score).
- **§2 Literature positioning** — generalized planning; PDDL/STRIPS domains; first-order / lifted
  decision-list policies; PG3 as the closest baseline (keep its policy form, swap GBFS + STRIPS-A*
  for MCTS + leaf-rollout reward); MCTS-vs-learned-AlphaZero distinction (*uniform MCTS is not
  learned AlphaZero; a policy it finds is a search artifact*). Cites the PDFs already in `papers/`.
- **§3 Two-level architecture** — task MDPs `M_B` vs synthesis derivation game `D_Γ`; the table.
- **§4 Task domains.** §4.1 Gripper-lite — the full MDP formalism (types, objects, predicates,
  `RelState` + invariants, `s⁰_B` / `G_B`, ground/legal actions, `T_B`, `R_B`, `H_B = 4B+4`, the env
  contract). §4.2 Doors PDDL contract — `DoorsPDDLLiteEnv` is a gymnasium flat-vector env, not
  relational; the planned relational adapter (vocab + `get_goal_atoms()`), flagged not-yet-built.
- **§5 Lifted decision-list semantics** — `π = [ρ₁,…,ρ_m]`, `ρ = ⟨X_ρ, body_ρ, α_ρ⟩`,
  `L_j = (atom, source ∈ {STATE,GOAL}, negated)`, type-respecting binding `θ: X_ρ → O`; maps to
  `Rule(vars, body, action)` / `Policy(rules)`; the PG3 `⟨PAR,PRE,GOAL,ACT⟩`-vs-folded-`body` note.
- **§6 The Gripper hand policy and its plan length** — boxed ρ₁…ρ₄; ρ₂'s `?b` flagged as a
  body-local variable; `plan_length(B) = 4B − 1`; ρ₄'s return trip ⇒ B≥2 needs all four rules;
  pointer to `notes_hand_policy_plan_length.md`.
- **§7 The online-unification interpreter** — closed-world `⊨_θ` (state/goal, positive/negated);
  safe negation; `B_ρ(S,G)`; lex-min tie-break `θ*_ρ`; `Exec`, `Interpret` first-applicable; the
  no-full-grounding note; the renaming caveat verbatim.
- **§8 The redesigned derivation grammar (proposed).** §8.1 why the standalone `Aux` phase is
  removed (the disconnected-`?aux_0` pathology; the `require_goal_var_connected` patch fixes the
  symptom not the cause). §8.2 occurrence-introduced variables: no `Aux` nonterminal / `aux_var`
  hole / `max_aux_vars`; action args → state literals may introduce fresh body-local vars → goal
  literals introduce none ⇒ connectedness structural; rename "auxiliary" → "body-local"; removing
  the `aux_var` syntax does not ban body-local vars (ρ₂ needs one). §8.3 example derivations for ρ₃
  and ρ₂ (step tables). §8.4 invariants of the strict grammar (well-typed literals; canonical
  literal order; schema-position action-var names; body-local vars only via state literals; goal
  literals introduce none; safe negation; `goal_predicate_relevance` whitelist; bounded caps with
  `max_aux_vars` removed; the 7-field per-node encoding minus `_KIND_AUX_VAR`); the corresponding
  `LiftedGrammarConfig(...)` and `legacy_grammar_config()`. §8.5 implementation status — the live
  code still has the `aux_var` hole etc.; the two safety flags are already defaults; the cutover is
  a future stage.
- **§9 Synthesis game and leaf evaluator** — `D_Γ = ⟨Z, P_Γ, T_Γ, R_Γ⟩`,
  `R_Γ(z_T) = LeafEval(to_program(z_T))`;
  `LeafEval(π) = solve_rate + 0.25·avg_progress − 0.01·avg_steps − 0.05·num_noops`,
  `progress = |s_T|_at_ball ∩ G|_at_ball| / |G|_at_ball|`; MCTS engine + `Game` protocol reused;
  `LiftedDerivationGame` new; `to_program(z_T)` executable by `interpret(...)`; uniform-prior MCTS
  (`UniformPolicyValueNet`), learned net later; the strict grammar / redesign makes the §8.1
  pathology unreachable.
- **§10 Experimental ladder** — Stage 0 (this doc); Stage 1 (semantic core **done** — DSL +
  interpreter + Gripper-lite + 4-rule policy solving B∈{1,2,3}; Doors adapter **not yet**); Stage 2
  (uniform-MCTS feasibility on Gripper B∈{1,2}, **landed but weak**); Stage 3 (strict-grammar
  defaults + diagnostic grid, **done**); Stage 4 (learned prior, future); Stage 5 (scale /
  generalization to larger B & Doors + the §8.2 cutover, future).
- **§11 What is not claimed** — hand-written policy, not MCTS discovery; uniform MCTS ≠ learned
  AlphaZero, the B=1 solver is a search artifact; removing the `aux_var` syntax doesn't remove
  body-local vars; name-agnostic matching ≠ arbitrary permutation equivariance; Gripper-lite ≠ Doors
  (adapter not built); full classical Gripper is a separate extension; no baseline grid / seed
  budget / learning hyperparameters beyond `legacy/02.md` / `legacy/03.md`.
- **§12 References** — PG3, AlphaZero, decision-lists PDFs; the semantic-core and synthesis source
  files; the `legacy/` stage notes; the companion notes; this plan.
- **Changelog** — the nine conceptual changes vs `00_draft` (removed the `Aux` phase / introduced
  occurrence-introduced variables; renamed "auxiliary" → "body-local"; promoted the two safety flags
  to invariants; added §2; added §4.2; restructured into 12 sections; sharpened non-claims; updated
  cross-links to `legacy/`; framed §8 as proposed with a §8.5 status note).

## Task B — file moves and link surgery (executed)

1. `mkdir docs/notes/stage4/legacy`.
2. `git mv` the five tracked notes (`01.md`, `01_plan.md`, `02.md`, `02_plan.md`, `02_plan2.md`) into
   `legacy/`; `mv` the two untracked notes (`03.md`, `03_plan.md`) into `legacy/` (they were not yet
   under version control). History follows the `git mv`'d files (`git log --follow legacy/01.md`).
3. Inside the seven moved files (`docs/notes/stage4/legacy/*.md`): prepend `../` to relative targets
   now one level deeper — `](../../../…` → `](../../../../…` (repo-root-relative `src/`, `tests/`,
   `scripts/` links), `](figures/` → `](../figures/`, `](data/` → `](../data/`, `](notes/` →
   `](../notes/`, `](00_draft_lifted_policy_az.md` → `](../00_draft_lifted_policy_az.md` (anchors
   preserved). Cross-references *among the seven moved files* (e.g. `03.md` → `](02.md)`) stay —
   they all moved together. `02_plan2.md` contained no markdown links (no-op). All link targets in
   the moved files re-checked to resolve.
4. `00_draft_lifted_policy_az.md` (stays at top level): refs to moved files rewritten with the
   `legacy/` prefix (`](01_plan.md)` → `](legacy/01_plan.md)`, `](01.md…)` → `](legacy/01.md…)`,
   `](02_plan.md)` → `](legacy/02_plan.md)`, `](02.md)` → `](legacy/02.md)`); a one-line
   "superseded by `01_draft_lifted_policy_az.md`" banner added under the title.
5. `01_draft_lifted_policy_az.md` (new, top level): refs to the moved notes use the `legacy/`
   prefix; refs to `notes_hand_policy_plan_length.md` / `literature_orientation.md` /
   `proposals_lifted_policy_az.md` / `notes/*.md` / `figures/*` use the top-level relative paths;
   `../../../src/…` for source files and `../../../papers/…` for the PDF citations.
6. `rewrite00_plan.md` (this file): repo-tracked plan of record.

## Critical files

**Created:** `01_draft_lifted_policy_az.md`, `rewrite00_plan.md`, `legacy/` (dir).
**Moved into `legacy/`:** `01.md`, `01_plan.md`, `02.md`, `02_plan.md`, `02_plan2.md`, `03.md`,
`03_plan.md`.
**Edited (link surgery only):** the seven moved files; `00_draft_lifted_policy_az.md` (banner +
`legacy/` prefixes).
**Referenced (not modified):** `src/alphazeropp/synthesis/{lifted_dsl,lifted_interpreter,lifted_grammar,lifted_derivation,lifted_encoding,lifted_leaf_evaluator,lifted_diagnostics}.py`;
`src/alphazeropp/instances/gripper_lite/{env,policies}.py`;
`src/alphazeropp/instances/doors/doors_pddl_lite.py`; `legacy/02.md` (§H4, Table 2),
`legacy/02_plan2.md` (§B), `legacy/01.md` (renaming caveat); `notes_hand_policy_plan_length.md`.
**Untouched:** any `.py` source or test, `figures/`, `data/`, `notes/`, and the
`feature/grammar-redesign` working-tree changes already in `git status`.

## Verification

1. `git status` shows: new `01_draft_lifted_policy_az.md`, new `rewrite00_plan.md`, five `git mv`
   renames into `legacy/`, two new untracked `legacy/03*.md`, modified `00_draft_lifted_policy_az.md`;
   nothing else changed.
2. `git log --follow docs/notes/stage4/legacy/01.md` shows pre-move history.
3. Link check — for every `.md` under `docs/notes/stage4/` (top level + `legacy/`), every relative
   `](path)` (minus `#anchor` / `:Lnn`) resolves: `01_draft` → `legacy/01.md`, `legacy/02.md`,
   `legacy/03.md`, `legacy/{01,02,02_plan2,03}_plan.md`, `figures/gripper_lite_domain.png`,
   `notes_hand_policy_plan_length.md`, `notes/{pddl_background,derivation_game}.md`, and the
   `../../../src/…` / `../../../papers/…` paths; each `legacy/*.md` → `../figures/…`, `../data/…`,
   `../notes/…`, `../00_draft_lifted_policy_az.md`, `../../../../src|tests|scripts/…`; `00_draft` →
   `legacy/01.md` etc.
4. `grep -rn "stage4/0" docs/ README* src/` shows no broken external references.
5. Render `01_draft_lifted_policy_az.md`: 12 numbered sections, boxed math, the two §8.3 derivation
   tables, the changelog; the Gripper-lite domain figure displays.
6. Read-through: `01_draft` does **not** claim MCTS success beyond the 1-solver-at-1-seed Stage-2
   result; does **not** claim arbitrary permutation equivariance (keeps the raw-name lex tie-break
   caveat); does **not** call uniform-MCTS-found policies "learned policies"; states that removing
   the `Aux` syntax does not remove body-local variables; states ρ₂ needs `?b` and
   `plan_length(B) = 4B − 1`; flags the Doors relational adapter as not yet implemented; §8 is
   framed as *proposed* with the §8.5 status note.
7. `00_draft_lifted_policy_az.md` retains all its content — `git diff` touches only link targets and
   the new banner line.
