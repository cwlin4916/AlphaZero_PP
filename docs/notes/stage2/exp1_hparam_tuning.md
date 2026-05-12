# Stage 2 / Experiment 1 — Hyperparameter Tuning for the Unmasked Grammar

## Question

Without domain-specific masks, what hyperparameter configuration lets AlphaZero match Stage 1's first-solve performance on the unmasked grammar $G_2$?

## Setup

Problem: **D=5 unmasked** ($K=4$, $|L_=(G_2)| = 16.8\text{M}$, $\rho_2 = 6.3 \times 10^{-6}$).

19-run $\epsilon \times \tau$ sweep:
- $\epsilon \in \{0, 0.03, 0.05, 0.10\}$ (Dirichlet root noise)
- $\tau \in \{0.25, 0.35, 0.50, 0.75, 1.0\}$ (MCTS visit-count temperature)
- 80 sims, 50 games/iter, 25 iterations, 2 seeds each.

## Why hyperparameters matter more without masks

Dirichlet noise mixes $\pi_{\mathrm{root}} = (1-\epsilon)\pi_{\mathrm{net}} + \epsilon\,\mathrm{Dir}(\alpha)$ into the root prior. In board games ($\epsilon = 0.25$ standard), noise ensures MCTS explores opponent responses. In program synthesis there is no opponent — **noise dilutes the learned prior in a search space where >99.99% of programs are junk**.

Stage 1 (masked) is robust to $\epsilon$: the grammar constrains search to a small, high-quality region regardless of noise. Stage 2 is not robust — every wasted simulation matters because the bootstrap problem (see [exp4](exp4_training_scaling.md)) depends on finding a solver at all.

## Results

Top and bottom of the 19-run ranking (by first-solve iteration):

| Rank | $\epsilon$ | $\tau$ | Solved? | Avg 1st Solve | Max 1st Solve | Note |
|---:|---:|---:|---|---:|---:|---|
| **1** | **0.00** | **0.50** | yes | **2** | **2** | **Best — robust** |
| 2 | 0.00 | 0.25 | yes | 6 | 11 | High variance |
| 3 | 0.00 | 1.00 | yes | 8 | 13 | |
| 4 | 0.05 | 0.50 | yes | 11 | 14 | |
| 5 | 0.10 | 0.50 | yes | 11 | 11 | Prior best (Stage 1) |
| 6 | 0.03 | 0.50 | yes | 12 | 13 | |
| 7 | 0.00 | 0.35 | yes | 12 | 16 | High variance |
| ⋮ | | | | | | |
| 8 | 0.03 | 0.25 | no | — | never | |
| 9 | 0.03 | 0.35 | no | — | never | |
| 10 | 0.00 | 0.75 | no | — | never | Too soft |

## Analysis

- **All top-3 configs use $\epsilon = 0$.** Zero noise is strictly better in Stage 2.
- **$\tau = 0.5$ dominates across $\epsilon$ levels** — sharp enough to concentrate training signal, soft enough to preserve diversity across seeds.
- **$\tau = 0.75$ with $\epsilon = 0$ fails to solve.** Targets too flat to concentrate MCTS on the solver neighborhood.
- **$\tau = 0.35$ is brittle:** with $\epsilon = 0$ it works (rank 7) but variance is high; with any noise it breaks.
- Compared to the prior Stage 1 best ($\epsilon = 0.10, \tau = 0.5$, first-solve iter 11), the winner is **5× faster**.

## Takeaways

1. **$\epsilon = 0, \tau = 0.5$** is the canonical Stage 2 / Stage 3 hyperparameter setting. Every subsequent Stage 2 training run uses it.
2. For synthesis without a masked grammar, Dirichlet noise is harmful — it spreads MCTS across a junk-dominated action space. Any future domain adapter should default to $\epsilon = 0$.
3. Visit-count temperature $\tau = 0.5$ is the sweet spot: lower $\tau$ is unstable, higher $\tau$ flattens targets below the concentration threshold.

## Source slides

Frames 13–15 (why hparams matter without masks, $\epsilon \times \tau$ sweep table).
