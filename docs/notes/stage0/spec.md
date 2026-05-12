# Stage 0 — Doors-Direct AlphaZero: Baseline Plot and Minimal-MCTS Sweep

**Status:** Forward-looking design spec (pre-registration). No results yet.
**Scope:** AlphaZero applied *directly* to the Doors MDP — no grammar, no program synthesis.
**Instance:** Primary D=3, robustness check at D=4.

## 0. Framing: Why Stage 0 Exists

The project's stages 1–3 run AlphaZero on a program-synthesis *derivation* game: each action expands a policy-program AST under grammar $G_i$, the reward is the compiled program's return on Doors. Stage 1's [exp1](../stage1/exp1_baseline_D3_D6_D7.md) already observed the now-famous paradox — "MCTS solves it at iter 1, the network doesn't learn the MCTS policy" — *but on the grammar game*. That paradox rides on top of an unstated baseline: whether AlphaZero even makes sense on Doors-*direct* in the first place, and whether the "MCTS does the work" pattern is grammar-specific or already visible without the grammar.

This spec writes down the missing progenesis. Doors-direct is the raw MDP (agent picks keys, walks through unlocked rooms, reaches the goal); AlphaZero is applied straight to that MDP with `DoorsDirectGame`. Stage 0 answers two concrete questions at the smallest instances — not whether AlphaZero *works* (it obviously can — D=3 has solve density $\rho = 1/2$), but how much of its working is MCTS and how little MCTS suffices.

**Key boundary:** Any mention of "grammar", "derivation", "program", "AST" is out of scope for stage 0. Those start in stage 1.

## 1. The Doors Direct Game at D=3

Cross-reference [overview_project.md §1](../overview_project.md) for the full Doors environment description. The stage-0-relevant specifics:

| Quantity | D=3 | D=4 |
|---|---:|---:|
| Keys $K$ | 2 | 3 |
| Locations $M = D \cdot \text{locs\_per\_room}$ | 6 | 8 |
| Observation size $M + 2D - 1$ | 11 | 15 |
| Action size $M + K + 1$ | 9 | 12 |
| Optimal steps | 5 | 7 |
| Horizon $= \max(15, 5 \cdot \text{opt})$ | 25 | 35 |
| Optimal reward $1.0 + (D-1)(0.1) - \text{opt}(0.01)$ | 1.15 | 1.23 |
| Solve density under uniform-random policy | $\rho \sim 1/2$ | $\rho \sim 1/6$ |

Solve density is the structural reason D=3 is "easy": half of random plays hit the key → room ordering correctly. D=4 falls off by a factor of ~3 and is the smallest instance where we expect MCTS-alone to noticeably struggle.

Game wrapper: `DoorsDirectGame` in [src/alphazeropp/instances/doors/game.py](../../../src/alphazeropp/instances/doors/game.py). Environment: [src/alphazeropp/instances/doors/doors_pddl_lite.py](../../../src/alphazeropp/instances/doors/doors_pddl_lite.py).

## 2. Research Questions

**Q1 — MCTS contribution decomposition.** What fraction of end-of-training solve performance at D=3 comes from MCTS search alone, the learned policy prior, and the learned value head respectively? Concretely: under the seven-way ablation of §5, how close is each ablated mode to the `full` baseline?

**Q2 — Minimal MCTS budget.** What is the smallest `n_simulations` such that the full AlphaZero loop reaches `solve_rate ≥ 0.95` by iteration 30, across ≥ 3 seeds, at D=3 and D=4?

Q1 probes the *horizontal* decomposition — which component. Q2 probes the *vertical* axis — how much compute.

## 3. Experimental Design

### E1 — Simulation-budget sweep (primary)

| Parameter | Value |
|---|---|
| Grid | `n_simulations ∈ {3, 6, 9, 12, 15, 18, 21, 24}` |
| $D$ | ∈ {3, 4} |
| Seeds | {0, 1, 2} |
| `n_iterations` | 30 (fixed — "within 30 iterations" is the Q2 criterion) |
| `n_games_per_train` | 20 |
| Network | 1-hidden-layer MLP, defaults from `BASELINE` in [sweep_doors_hyperparams.py:41](../../../scripts/benchmark/sweep_doors_hyperparams.py#L41) |
| `locs_per_room` | 2 |
| Other MCTS knobs | `c_exploration=1.5`, `dirichlet_alpha=0.25`, `dirichlet_epsilon=0.40` (stage-1 values held constant) |

Total: 2 × 8 × 3 = **48 training runs**. Logged per iteration via `StatisticsManager.save_jsonl`: `solve_rate`, `avg_reward`, `policy_loss`, `value_loss`.

The grid is concentrated in the **low** regime because prior observation at D=3 is that anything with `n_simulations ≥ 25` already solves; the threshold of interest lives well below 25. Steps of 3 give dense sampling with tolerable cost.

### E2 — Component ablations (answers Q1)

After E1 completes, pick two sim budgets that bracket the observed threshold — one "just below" (where solve_rate < 0.5 at D=4) and one "clearly above" (where solve_rate ≥ 0.95 at both D). Starting picks: `n_simulations ∈ {6, 18}`, revised after seeing E1.

For each $(D, \text{sims})$ pair, seed 0, at end of training, run all 7 ablation modes via [`compute_solve_rate_ablated`](../../../src/alphazeropp/instances/doors/eval_ablations.py#L88) with `n_episodes=20`. All wrappers preserve the trained agent state (see [network_ablations.py](../../../src/alphazeropp/instances/doors/network_ablations.py)) — no retraining needed.

### E3 — Untrained-MCTS floor (answers Q1 structurally)

For each $(D, \text{sims})$ in the E1 grid, run [run_doors_mcts_baseline.py](../../../scripts/run/run_doors_mcts_baseline.py) with `n_rounds=100` and `UniformPolicyValueNet`. This is the "MCTS with *zero* training" bound. If untrained MCTS already matches trained AlphaZero at some budget, the neural network isn't contributing there.

## 4. The Plot

Single figure, 2 rows × 3 columns, saved to `experiments/stage0_doors_low_sim/plots/stage0_overview.png`. Top row = D=3, bottom row = D=4.

| Column | Content | Answers |
|---|---|---|
| **1 — Convergence by sim budget** | `solve_rate` vs iteration, one line per `n_simulations`, mean over 3 seeds, shaded ±σ band. Horizontal dashed line at 0.95. | Q2 — first curve to cross 0.95 by iter 30 = minimum sims. |
| **2 — Component ablation bars** | End-of-training `solve_rate` across 7 modes (§5), grouped by the two bracketing sim budgets. | Q1 — which component carries how much. |
| **3 — Trained vs untrained MCTS** | Paired bars at each sim budget: "trained AZ" (end-of-training `solve_rate`) vs "untrained MCTS" (E3). | Q1 structurally — gap width = NN contribution. |

Stacking D=3 over D=4 lets the reader eyeball whether the threshold shifts rightward with D and whether a trained-vs-untrained gap that is invisible at D=3 opens at D=4.

## 5. Standard-in-Practice AlphaZero Ablation Checklist

The seven-row decomposition below is the standard "how much does each piece contribute" study used across AlphaGo Zero, AlphaZero, and MuZero. Every row already has a corresponding mode in [eval_ablations.py](../../../src/alphazeropp/instances/doors/eval_ablations.py) — the checklist is **already implemented**; stage 0 just names it and runs it. Future stages can reuse exactly this template.

| # | Ablation | What it isolates | Repo mode |
|---|---|---|---|
| 1 | Full system (trained net + MCTS) | Baseline | `full` |
| 2 | Policy network only, no MCTS | Value of MCTS | `policy-only` |
| 3 | Value-only 1-step lookahead | Value head quality in isolation | `value-only-1step` |
| 4 | MCTS + uniform prior + learned value | Contribution of the learned *prior* | `uniform-prior` |
| 5 | MCTS + learned prior + zero value | Contribution of the learned *value* | `zero-value` |
| 6 | MCTS + uniform prior + zero value | Pure MCTS with shaped reward only | `unguided-search` |
| 7 | MCTS + random-weight network | What a fresh, untrained net adds | `random-net` |
| +1 | Simulation-budget sweep | MCTS depth-vs-width sensitivity | grid over `n_simulations` |
| +2 | ≥ 3 seeds, per-iter curves + bars | Noise robustness and readability | standard statistics practice |

### Reading the rows against the full baseline

- If row 6 ≈ row 1, MCTS alone suffices — the network is decorative.
- If row 2 ≈ row 1, the policy head memorized the trajectory and MCTS is redundant.
- If rows 4 and 5 are both < row 1, prior and value are each non-trivial contributors.
- If row 7 ≈ row 6 but < row 1, training the network is what matters, not the architecture per se.

These inequalities, reported with numbers, constitute the answer to Q1.

## 6. Execution Recipe

Commands are reproducible from a clean checkout; they reference two thin scripts (`stage0_doors_low_sim_sweep.py`, `plot_stage0_overview.py`) that are part of a follow-up task — not written in this spec pass.

```bash
# E1 — sim-budget sweep at D=3 and D=4 (writes iteration_log.jsonl per config)
for D in 3 4; do
    python scripts/benchmark/stage0_doors_low_sim_sweep.py \
        --num-rooms $D --locs-per-room 2 \
        --sims 3 6 9 12 15 18 21 24 --seeds 0 1 2 \
        --out experiments/stage0_doors_low_sim/D${D}/
done

# E2 — 7-mode ablation on trained checkpoints (seed 0, two bracketing sims picked from E1)
for D in 3 4; do
    python scripts/benchmark/stage0_doors_low_sim_sweep.py \
        --num-rooms $D --ablate-only --sims 6 18 --seeds 0 \
        --out experiments/stage0_doors_low_sim/D${D}/
done

# E3 — untrained-MCTS floor
for D in 3 4; do
    for S in 3 6 9 12 15 18 21 24; do
        python scripts/run/run_doors_mcts_baseline.py \
            --num_rooms $D --n_rounds 100 --n_simulations $S --no-interactive \
            --out experiments/stage0_doors_low_sim/D${D}/untrained_mcts/sims_$S/
    done
done

# Plot
python scripts/plotting/plot_stage0_overview.py \
    --root experiments/stage0_doors_low_sim/ \
    --out experiments/stage0_doors_low_sim/plots/stage0_overview.png
```

Existing infrastructure the scripts above wrap: `build_config` in [sweep_doors_hyperparams.py:132](../../../scripts/benchmark/sweep_doors_hyperparams.py#L132), `compute_solve_rate_ablated` in [eval_ablations.py:88](../../../src/alphazeropp/instances/doors/eval_ablations.py#L88), the JSONL-reading pattern from [plot_macro_vs_surface.py](../../../scripts/plotting/plot_macro_vs_surface.py).

## 7. Pre-registered Predictions

Stated before running to guard against hindsight rationalization. Priors: D=3 solve density $\rho = 1/2$, stage-1's "MCTS solves at iter 1" observation, and the user's prior experience that sub-25-sim configurations already solve D=3.

- **P1** — At D=3, even `n_simulations = 6` converges to solve_rate ≥ 0.95 by iter 30. The threshold sits in $[3, 6]$.
- **P2** — At D=4, the threshold shifts rightward but remains inside the grid — expected around `n_simulations ∈ [12, 18]`. If the D=4 threshold exceeds 24, we re-run with an extended grid.
- **P3** — Untrained MCTS (E3) reaches solve_rate ≥ 0.8 at `n_simulations ≥ 12` for D=3. If true, MCTS-alone carries most of the work at that budget.
- **P4** — `unguided-search` ≈ `full` in the E2 bars at D=3 — direct evidence for Q1 = "MCTS does nearly everything at D=3". At D=4, a gap opens.
- **P5** — `policy-only` is weak at iter 30 even with a trained net — the value head may memorize, but the prior does not carry the full burden.

**Implication if P3+P4 hold.** D=3 is MCTS-dominated and is therefore a poor testbed for studying what the neural network contributes. The D=4 panel is the diagnostic: if the trained-vs-untrained gap visibly opens there, D=4 is the smallest instance where the NN starts earning its keep, and stages 1+ correctly moved to D=6, D=7 to see clear network-contribution effects. Stage 0 thus motivates the scaling choice made downstream.

## 8. Success Criteria

- `stage0_overview.png` exists with 2×3 panels populated — 3 seeds in col 1, 7 modes in col 2, 8 sim budgets in col 3, for both D=3 (top) and D=4 (bottom).
- `summary.csv` across the sim sweep is sorted and contains `best_solve_rate` and `final_solve_rate` per $(D, \text{sims})$ config.
- All 7 ablation modes produce numbers for the two chosen sim budgets at each $D$.
- A short "Results" subsection appended to this spec (or a sibling `results.md`) answers Q1 and Q2 with numbers for both D=3 and D=4, and states which of P1–P5 held.

## 9. Takeaways that Will Carry Forward to Stage 1

To be filled in after the run. Placeholder structure:

1. Whether MCTS-dominance at D=3 is confirmed (determines framing of the stage-1 paradox).
2. Minimum viable `n_simulations` for each $D$ (sets defaults for subsequent stages, lowers compute cost).
3. Where the trained-vs-untrained gap first opens (identifies the smallest instance where the NN matters — the natural floor for stage-1 grammar experiments).
