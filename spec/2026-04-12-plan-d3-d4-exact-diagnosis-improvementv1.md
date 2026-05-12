# Plan: D=3/4 Exact-Diagnosis Regime for Doors AlphaZero

**Final spec destination (created during execution, not now):**
`spec/2026-04-12-plan-d3-d4-exact-diagnosis.md`

This plan file is the planning artifact. The spec file above is the deliverable.

---

## Context

The Doors AlphaZero synthesis pipeline is now mature enough that the bottleneck is *diagnosis*, not scale. Prior reports ([2026-03-26 alphazero-synthesis-diagnostic](../spec/2026-03-26-report-alphazero-synthesis-diagnostic.md), [stage2_D6_failure_diagnostic](../docs/specs/stage2_D6_failure_diagnostic.md)) establish the core failure mode: AlphaZero only scales if (a) search finds better derivations and (b) the network absorbs that improvement into its raw policy. At D≥6 we cannot tell these apart from rollouts alone.

The key shift this plan makes: **treat D=3 and D=4 as exact-diagnosis regimes, not miniature performance runs.** At these depths the derivation-prefix DAG is enumerable. We can compute an oracle over full prefixes and score both MCTS and the raw policy against ground truth instead of against each other (KL-to-MCTS). This turns D=3 into a unit test and D=4 into the smallest true diagnosis case, and it makes the two central questions ("is MCTS doing the work?" and "have we learned the correct grammar policy?") measurable rather than interpretive.

Two additional commitments:
- **State = full emitted-token prefix**, not status vector. Token order becomes runtime guard priority in [surface_compiler.py](../src/alphazeropp/instances/doors/dsl/surface_compiler.py) (`compile_policy` chains `Ite` right-to-left), so status-equivalent prefixes can compile to different policies. Collapse only after proving equivalence.
- **Two scenarios, not a grid.** Dirichlet ε=0, temperature=0.5 fixed. Train-sim ∈ {4, 32}. Budget goes into exact statewise evaluation, not a hyperparameter sweep.

Literature anchors: AlphaZero/ExIt frame search as the expert and the net as apprentice; [Grill et al. 2020](https://arxiv.org/pdf/2007.12509) formalizes visit-count targets as regularized policy improvement; [Danihelka et al. 2022 (Gumbel AZ)](https://openreview.net/pdf?id=bERaNdoegnO) show visit-count targets do *not* guarantee improvement at low sim budget — which is exactly the regime we probe. [Willemsen et al. 2021](https://ala2020.vub.ac.be/papers/ALA2020_paper_18.pdf) show on-policy value targets can miss the greedy-optimal policy, motivating exact `V^π` vs `V*` audits.

## Scope

In-scope: Doors derivation game at D=3, D=4; surface-DSL grammar; existing [DerivationPolicyValueNet](../src/alphazeropp/synthesis/derivation_network.py), [MCTS](../src/alphazeropp/core/mcts.py), [Trainer](../src/alphazeropp/training/trainer.py); two training scenarios only.

Out-of-scope: D≥5 sweeps; new grammar variants; new MCTS variants (Gumbel/KataGo targets are follow-up work gated on Stage 6 memo).

## Key Quantities

- `illegal_mass(x) = Σ_{a∉A_legal(x)} p_raw(a|x)`
- `regret_π(x) = max_a Q*(x,a) − Σ_a π(a|x) Q*(x,a)`
- `search_lift = V^{π_MCTS}(root) − V^{π_net}(root)`
- `learned_search_gain = V^{π_MCTS, learned}(root) − V^{π_MCTS, uniform}(root)`

## Stages

Each stage produces (i) a markdown audit under `analysis/`, (ii) machine-readable artifacts, (iii) a short validation section. Stages 1 and 2 are non-negotiable prerequisites.

### Stage 1 — Exact oracle over full prefixes (D=3, D=4)
- Enumerate all legal derivation prefixes using the real grammar ([budget_grammar.py](../src/alphazeropp/synthesis/budget_grammar.py), [derivation_game.py](../src/alphazeropp/synthesis/derivation_game.py)).
- For every complete derivation, compute terminal reward via the true compile path: [surface_compiler.py](../src/alphazeropp/instances/doors/dsl/surface_compiler.py) → [interpreter.py](../src/alphazeropp/synthesis/interpreter.py) → [LeafEvaluator](../src/alphazeropp/synthesis/leaf_evaluator.py). Reuse [oracle.py](../src/alphazeropp/instances/doors/oracle.py) frozen-state cycling; do not reimplement semantics.
- Back up exact `Q*(prefix, a)`, `V*(prefix)`, and optimal-action sets (with ties) over the prefix DAG.
- Validate prefix counts against expected combinatorics; D=3 reachable env states = 12 ([test_doors_oracle.py:71](../tests/test_doors_oracle.py#L71)).
- Artifacts: `analysis/d3_d4_stage1_oracle.{md,parquet,csv}`.

### Stage 2 — Exact evaluator for any (checkpoint, actor)
- Load checkpoint via [CheckpointManager](../src/alphazeropp/utils/checkpoint.py); reuse its `agent_checkpoint.pkl` + `.pt` layout.
- Actor modes: `raw_policy_masked`, `mcts_learned`, `mcts_uniform` (shim net, uniform logits, constant V); optional `mcts_prior_only`, `mcts_value_only`.
- For every oracle prefix emit: raw logits, pre-mask illegal mass, masked policy, MCTS visit distribution, exact `V^π(prefix)` via oracle backup (not MC), exact occupancy from root under π, `regret_π`.
- Artifacts: `analysis/d3_d4_stage2_eval/{ckpt}_{actor}.parquet`.

### Stage 3 — Two training scenarios only
Hold fixed: `dirichlet_epsilon=0`, `temperature=0.5`, all other [derivation_config.py](../src/alphazeropp/instances/doors/dsl/derivation_config.py) defaults.

- **Scenario A `learning_visible`:** `n_simulations_train=4`.
- **Scenario B `search_supported`:** `n_simulations_train=32`.

Pipeline: D=3 smoke test first (single seed), then D=4 primary run (≥3 seeds). Save checkpoints every few iterations so Stage 2 can attach to the full curve. Register any existing 80-sim checkpoint as a *reference only*; do not make it part of the primary sweep. Eval-sim probe grid at analysis time: `{0,1,2,4,8,16,32}`.

### Stage 4 — "Is MCTS doing the work?"
Per checkpoint: exact `V` for all actor modes; compute `search_lift` and `learned_search_gain`; break down by prefix depth. Plots: root value, search_lift, learned_search_gain vs iteration; statewise heatmap of where search helps. Secondary: env-rollout solve-rate curves for sanity.

### Stage 5 — "Have we learned the correct grammar policy?"
Separate *legality* from *semantics*.
- Pre-mask: `illegal_mass`, top-unmasked-illegal rate, by depth.
- Post-mask: `regret_π` vs `Q*`, mass-on-optimal-set, top-1 optimal-action accuracy; policy↔MCTS KL as *secondary only*.
- Anchor-state plots: 8–12 D=4 prefixes (root, high-tie, max net-vs-MCTS disagreement, legal-but-bad, deepest, high illegal-mass). For each: raw masked dist, MCTS visits, oracle-optimal set, Q-gap annotation, across checkpoints.

### Stage 6 — Value, coverage, decision memo
- Per prefix: `v_θ`, exact `V^π_current`, `V*`. Compute `v_θ − V^π_current` (approximation error) and `V^π_current − V*` (suboptimality gap), both depth- and occupancy-weighted.
- Occupancy from root under π; cross-check with replay/self-play empirical visitation if available ([trainer.py](../src/alphazeropp/training/trainer.py)).
- Classify D=4 failure: {search, representation, policy-target, value/replay, amortization-works}. Recommend exactly one intervention before D=5.

## Decision Logic (for Stage 6 memo)

| Observation | Diagnosis |
|---|---|
| Large `search_lift`, tiny `learned_search_gain` | Search does the work; uniform net would suffice |
| MCTS near `V*`, raw policy not | Search correct; learning is the bottleneck |
| High pre-mask illegal_mass | Net hasn't learned legality; mask hides it |
| Low illegal_mass, high `regret_π` | Legality learned, semantics not |
| `v_θ ≈ V^π_current` but `V^π_current ≪ V*` | Target/search issue, not value fitting |
| Bad states concentrated at low occupancy | Coverage bottleneck |
| D=3 fails any of the above | Halt; debug before D=4 |

## Critical files (modify / reuse)

- Enumeration: [budget_grammar.py](../src/alphazeropp/synthesis/budget_grammar.py), [derivation_game.py](../src/alphazeropp/synthesis/derivation_game.py)
- Reward path: [surface_compiler.py](../src/alphazeropp/instances/doors/dsl/surface_compiler.py), [interpreter.py](../src/alphazeropp/synthesis/interpreter.py), [leaf_evaluator.py](../src/alphazeropp/synthesis/leaf_evaluator.py)
- Net + MCTS: [derivation_network.py](../src/alphazeropp/synthesis/derivation_network.py), [mcts.py](../src/alphazeropp/core/mcts.py)
- Config + training: [derivation_config.py](../src/alphazeropp/instances/doors/dsl/derivation_config.py), [trainer.py](../src/alphazeropp/training/trainer.py), [run_doors_derivation.py](../scripts/run/run_doors_derivation.py)
- Checkpoints: [checkpoint.py](../src/alphazeropp/utils/checkpoint.py)
- Oracle baseline: [oracle.py](../src/alphazeropp/instances/doors/oracle.py), [test_doors_oracle.py](../tests/test_doors_oracle.py)

New code lives under `src/alphazeropp/analysis/exact/` (oracle builder, evaluator) and `scripts/analysis/d3_d4_*`.

## Verification

- **Stage 1**: D=3 prefix count and terminal-reward totals match independent combinatorial expectation and [test_doors_oracle.py](../tests/test_doors_oracle.py). `V*(root)` at D=3 equals optimal_return(3) ≈ 1.06 when the grammar admits an optimal program; otherwise flag gap explicitly.
- **Stage 2**: For a uniform-logits shim checkpoint, `raw_policy_masked` reproduces grammar-uniform distributions; `V^π` from backup equals MC rollout mean within noise on a small check set.
- **Stage 3**: Smoke test completes on D=3 end-to-end; checkpoints loadable by Stage 2 evaluator.
- **Stages 4–6**: Each stage's markdown contains the plots named in its section and a conclusion paragraph for D=3 and D=4 separately.

## Resolved parameters

- **D=4 seeds per scenario:** 3.
- **Sequencing:** Stage 3 is gated on Stages 1–2 completing. Oracle + evaluator are prerequisites, not parallel work.
- **80-sim checkpoint:** included in Stage 4 plots as a reference-only series (distinct style), excluded from primary sweep stats.
