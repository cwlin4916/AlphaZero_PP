# Stage 2 / Experiment 4 — Training Scaling: Stage 1 vs Stage 2 at D=3–7

## Question

With the tuned hyperparameters ($\epsilon = 0$, $\tau = 0.5$), can AlphaZero on the unmasked grammar match Stage 1's first-solve and convergence across increasing problem sizes? Where (and why) does it break?

## Setup

- **D=3–5:** 12 runs total (2 stages × 3 $D$-values × 2 seeds), 50 iterations each, $\epsilon = 0.10, \tau = 0.5$.
- **D=6, 7:** both stages, $\epsilon = 0, \tau = 0.5$, 80 sims, 50 games/iter, 50 iterations, 3 seeds. **Same hyperparameters ⇒ grammar is the only variable.**

## Search-difficulty prediction (pre-experiment)

| $D$ | $\lvert L_=(G_2)\rvert$ | $\rho_2$ | $1/\rho_2$ | 80 sims? | Prediction |
|---:|---:|---:|---:|---|---|
| 3 | 256 | 0.012 | 85 | Yes | 80 sims ≈ $1/\rho_2$ |
| 4 | 46,656 | $3.2\times 10^{-4}$ | 3,111 | Hard | ≫ 80 sims |
| 5 | 16.8M | $6.3\times 10^{-6}$ | 159K | No | Random chance ≈ 0 |

Stage 1 MCTS solves all three easily because $1/\rho_1$ grows only as $K!$ (2, 6, 24, 120) which stays below the MCTS budget.

## Results

### D=3, 4, 5 — first-solve and final losses

| $D$ | Stage | Seed | First solve | Solve rate | $L_\pi$ | $L_v$ |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 42 | iter 1 | 100% | 0.1126 | 0.0003 |
| 3 | 1 | 137 | iter 1 | 100% | 0.0008 | 0.0002 |
| 3 | **2** | **42** | **iter 1** | **100%** | 0.0145 | 0.0004 |
| 3 | **2** | **137** | **iter 1** | **100%** | 0.0006 | 0.0002 |
| 4 | 1 | 42 | iter 1 | 100% | 0.1109 | 0.0003 |
| 4 | 1 | 137 | iter 1 | 100% | 0.1141 | 0.0004 |
| 4 | **2** | **42** | **iter 1** | **100%** | 0.0016 | 0.0002 |
| 4 | **2** | **137** | **iter 3** | **100%** | 0.0011 | 0.0002 |
| 5 | 1 | 42 | iter 1 | 100% | 0.1478 | 0.0002 |
| 5 | 1 | 137 | iter 1 | 100% | 0.0886 | 0.0003 |
| 5 | **2** | **42** | **iter 1** | **100%** | 0.1327 | 0.0011 |
| 5 | **2** | **137** | **iter 17** | **100%** | 0.0302 | 0.0006 |

Stage 2 solves all cases at $D \leq 5$. The gap is visible at $D=5$ (worst-case iter 17).

### D=6, 7 — the cliff

| $D$ | Stage | Seed | First solve | Solve rate | Best reward |
|---:|---:|---:|---|---|---:|
| 6 | 1 | 42 | iter 1 | 100% | +1.117 |
| 6 | 1 | 137 | iter 1 | 100% | +1.117 |
| 6 | 1 | 271 | iter 1 | 100% | +1.117 |
| 6 | **2** | **42** | **never** | **0%** | −0.150 |
| 6 | **2** | **137** | **never** | **0%** | −0.150 |
| 6 | **2** | **271** | **never** | **0%** | −0.150 |
| 7 | 1 | 42 | iter 1 | 100% | +1.141 |
| 7 | 1 | 137 | iter 1 | 100% | +1.141 |
| 7 | 1 | 271 | iter 1 | 100% | +1.141 |
| 7 | **2** | **42** | **never** | **0%** | −0.150 |
| 7 | **2** | **137** | **never** | **0%** | −0.150 |
| 7 | **2** | **271** | **never** | **0%** | −0.150 |

**Decisive result.** Stage 1 solves $D=6$ and $D=7$ at iter 1 for all seeds. Stage 2 never solves (0/6 seeds across 50 iterations each). With identical hyperparameters, grammar masks are the only difference — and they are necessary at $D \geq 6$.

### Learning curves and solve summaries

![Learning curves D3-D7](../../presentations/improvementv1/stage1_vs_stage2_learning_curves_D3_to_D7.png)

Top: policy loss. At $D \leq 5$ both stages converge. At $D = 6, 7$: Stage 1 converges normally; Stage 2 plateaus at high loss — the network learns the junk-reward distribution but never sees a solver. Bottom: value loss. Stage 1 → near-zero at all $D$. Stage 2 at $D \geq 6$ cannot learn meaningful predictions from the near-constant negative reward landscape.

![Combined summary](../../presentations/improvementv1/stage1_vs_stage2_combined_summary.png)

Left: first-solve iteration vs $D$. Stage 1 solves at iter 1 everywhere; Stage 2 solves $D \leq 5$ but fails at $D \geq 6$. Right: solve density (log) explains the failure — $\rho_2$ drops below $10^{-7}$ at $D=6$.

![First-solve vs D](../../presentations/improvementv1/stage1_vs_stage2_first_solve_vs_D.png)

![Solve curves D6 D7](../../presentations/improvementv1/stage1_vs_stage2_solve_curves_D6_D7.pdf)

## Why Stage 2 fails — the bootstrap problem

| $D$ | Stage 1 first-solve | $\lvert L_=(G_2)\rvert$ | $\rho_2$ | Programs explored | P(find solver) |
|---:|---:|---:|---:|---:|---:|
| 5 | iter 1 | 16.8M | $6.3\times 10^{-6}$ | ~90K | **43%** |
| 6 | iter 1 | 10B | $9.5\times 10^{-8}$ | ~195K | **1.8%** |
| 7 | iter 1 | 8.9T | $1.2\times 10^{-9}$ | ~200K | **0.02%** |

Chicken and egg:
1. The network needs a *solver example* to learn what good programs look like.
2. MCTS needs a *good prior* to find a solver in $10^{10}$ candidates.
3. At $D \leq 5$, random rollouts have a 43% chance of stumbling on a solver — bootstrap succeeds.
4. At $D = 6$ the probability drops to 1.8%. The bootstrap fails.

Stage 1 is immune: the masked grammar compresses search ~88,000× at $D=6$, leaving only 113,400 legal programs. MCTS trivially finds a solver at iter 1.

## Predictions vs reality

| $D$ | Predicted | Observed | Assessment |
|---:|---|---|---|
| 3 | Solve iter 1–3, converge | Solve iter 1 (both seeds) | ✓ Correct |
| 4 | Delayed first-solve (iter 5–15) | Solve iter 1 and iter 3 | ≈ Overpredicted difficulty |
| 5 | May not solve in 50 iters | Solve iter 1 and iter 17 | ✗ Hypothesis refuted |

Why too pessimistic for $D \leq 5$:
1. **Tuned hyperparameters.** $\epsilon = 0.10, \tau = 0.5$ (carried over from Stage 1 diagnostics) are far more effective than the baseline $\epsilon = 0.40, \tau = 1.0$.
2. **Rollouts.** 4 random completions per leaf provide reward signal even before the network learns, bootstrapping first solves.
3. **Solve density ≠ search difficulty.** $1/\rho_2$ measures random enumeration cost; MCTS with a prior is exponentially better than random search.

## Grammar bias spectrum

| Stage | Grammar | $\lvert L\rvert$ at $D=5$ | Bias level |
|---|---|---:|---|
| **Stage 1** | $G_1$: masked. Precedence, uniqueness, completeness. | 2,520 | Full domain |
| **Stage 2** | $G_2$: unmasked. Only episode length enforced. | 16.8M | Length only |
| (Stage 3) | Typed CFG with lifted variables; structural bounds. | — | Structural / portable |

## Takeaways

1. **Masks are not necessary for $D \leq 5$.** AlphaZero solves a 159,000× larger search space when hyperparameters are tuned.
2. **Masks are decisive at $D \geq 6$.** 0/6 seeds, no partial credit. The transition from "possible" to "impossible" is abrupt — one step of $D$.
3. **The boundary is sharp and lives at the bootstrap probability.** Below ~5% random-solver probability, the learning loop cannot start.
4. **Three interventions to cross the boundary without full masks:**
   - *Curriculum learning* — transfer weights from $D-1$ to $D$ to bootstrap the prior from an easier problem.
   - *Partial masking* — keep precedence, drop uniqueness (the "Stage 2.5" option).
   - *Stage 3 lifted schema* — provide inductive bias that is not domain-specific but still structurally narrowing (typed variables, bounded decision-list depth).
5. **Scaling law match:** $\rho_1/\rho_2$ predicts first-solve delay within an order of magnitude for $D \leq 5$ but not the cliff at $D = 6$. MCTS + learned prior beats the random-enumeration bound until the bootstrap probability falls below a threshold.

## Source slides

Frames 28–31 (training results $D=3,4,5$), frames 36–37 ($D=6,7$ results), frames 38–39 (bootstrap problem, first-solve scaling), frames 40–42 (learning curves, predictions vs reality, what the results tell us).
