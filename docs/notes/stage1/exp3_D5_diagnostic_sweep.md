# Stage 1 / Experiment 3 — D=5 Diagnostic Hyperparameter Sweep

## Question

At the smallest "uncertain convergence" size (D=5, $K=4$), which single hyperparameter change most reduces $D_{\mathrm{KL}}$ — patience, data, search budget, exploration noise, or visit-count temperature?

## Setup

Target problem: **D=5**, $K=4$. 9 steps/episode, $3^4 = 81$ nonterminals, $\rho = 1/24$, 3024 legal policies. Time per iter $\approx 30\,\text{s}$.

Success metric: $D_{\mathrm{KL}} = L_\pi - \bar H_{\mathrm{MCTS}} < 0.05$ at the final iteration.

| # | Experiment | Change from baseline | Hypothesis | Est. time |
|---|---|---|---|---:|
| 0 | Baseline + $\bar H$ logging | — (80 sims, 30 games, $\epsilon=0.40$, $\tau=1.0$, 15 iter) | Measure the true floor | 5 min |
| 1 | Long run | 100 iterations | Patience suffices | 30 min |
| 2 | More data | 100 games/iter | Coverage is the bottleneck | 15 min |
| 3 | Sharper MCTS | 200 simulations | Noisy MCTS targets are the issue | 20 min |
| 4 | Less noise | $\epsilon = 0.10$ | Exploration noise dominates | 5 min |
| 5 | Lower temp | $\tau = 0.5$ | Flat $\boldsymbol\pi$ is the issue | 5 min |

## Method

Run each configuration independently with $\bar H_{\mathrm{MCTS}}$ logging enabled. Compute $D_{\mathrm{KL}} = L_\pi - \bar H_{\mathrm{MCTS}}$ at the final iteration and track the convergence curve.

## Results

### Final-iteration decomposition

![D5 KL bar](../../presentations/improvementv1/diag_D5_kl_bar.png)

| # | Experiment | $L_\pi$ | $\bar H_{\mathrm{MCTS}}$ | $D_{\mathrm{KL}}$ | $L_v$ |
|---|---|---:|---:|---:|---:|
| 0 | Baseline (15 iter) | 0.622 | 0.436 | 0.186 | 0.171 |
| 1 | Long run (100 iter) | 0.574 | 0.479 | **0.095** | 0.089 |
| 2 | More data (100 games) | 0.567 | 0.435 | 0.133 | 0.127 |
| 3 | Sharper MCTS (200 sims) | 0.611 | 0.428 | 0.182 | 0.144 |
| 4 | Less noise ($\epsilon = 0.10$) | 0.510 | 0.388 | 0.122 | 0.099 |
| 5 | Lower temp ($\tau = 0.5$) | 0.475 | 0.239 | 0.236 | **0.064** |

### Convergence curves

![D5 KL curves](../../presentations/improvementv1/diag_D5_kl_curves.png)

Per-iteration $D_{\mathrm{KL}}$ for all six configs. Long run (blue) keeps decreasing through iter 100. Less-noise (red) is the best single-parameter fix at 15 iter.

### Decomposition detail

![D5 decomposition](../../presentations/improvementv1/diag_D5_decomposition.png)

Per-config breakdown showing how $\bar H_{\mathrm{MCTS}}$ and $D_{\mathrm{KL}}$ trade off.

## Analysis

| Experiment | Finding | Interpretation |
|---|---|---|
| E1: long run (100 iter) | $D_{\mathrm{KL}}$: $0.186 \to 0.095$, still decreasing | **Patience works.** The network is learning, just slowly. 15 iters is insufficient. |
| E2: more data (100 games) | $D_{\mathrm{KL}}$: $0.186 \to 0.133$ | More examples help. Coverage per state $20\times \to 67\times$. |
| E3: sharper MCTS (200 sims) | $D_{\mathrm{KL}}$: $0.186 \to 0.182$ | **No effect.** 80 sims is already enough at $K=4$. Not the bottleneck. |
| E4: less noise ($\epsilon = 0.10$) | $D_{\mathrm{KL}}$: $0.186 \to 0.122$; $\bar H$: $0.436 \to 0.388$ | Reduces both the floor and the gap. **Best single-parameter fix at 15 iter.** |
| E5: lower temp ($\tau = 0.5$) | $\bar H$: $0.436 \to 0.239$; $D_{\mathrm{KL}}$: $0.186 \to 0.236$ | Targets become sharper (lower $\bar H$), but the network can't keep up, so the KL gap widens. Best $L_v = 0.064$. |

The long run and the less-noise experiments operate on different parts of the decomposition: long run shrinks $D_{\mathrm{KL}}$ while $\bar H_{\mathrm{MCTS}}$ actually *grows* slightly (0.436 → 0.479) as MCTS explores more broadly with a better prior; less-noise shrinks both simultaneously by sharpening the root prior. Temperature goes the other way — it sharpens targets faster than the network can match them.

Projected: extrapolating the long-run curve, $D_{\mathrm{KL}} < 0.05$ would be reached around iter 200.

## Takeaways

1. **Patience is the primary fix.** Long run alone reduces $D_{\mathrm{KL}}$ by 49% (0.186 → 0.095).
2. **Dirichlet noise is the best 15-iter intervention.** $\epsilon = 0.40 \to 0.10$ gives $-34\%$ without needing more compute.
3. **Doubling MCTS simulations does not help.** 80 sims is enough at $K=4$; more are wasted.
4. **Lower temperature trades ease for sharpness.** Best $L_v$, but worst $D_{\mathrm{KL}}$ at 15 iter — the network needs more time to match the sharper targets.
5. **Recommended config going forward:** $\epsilon = 0.10$, $\tau = 0.5$, 100 iterations, 80 sims, 30 games/iter. Combine patience with a sharp prior.

## Source slides

Frames 41–46 (where non-convergence begins, diagnostic experiments plan), frames 47–51 (results table, bar chart, convergence curves, per-parameter analysis, conclusions).
