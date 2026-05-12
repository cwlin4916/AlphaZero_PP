# Stage 1 / Experiment 1 — Baseline Training at D=3, D=6, D=7

## Question

Does the AlphaZero loop converge — policy and value loss, on a cross-entropy floor basis — when applied to the masked grammar derivation game at increasing problem sizes?

## Setup

| Dimension | D=3 | D=6 | D=7 |
|---|---:|---:|---:|
| Keys $K$ | 2 | 5 | 6 |
| Actions ($2K+1$) | 5 | 11 | 13 |
| Policies $\lvert L(G_1)\rvert$ | 6 | 113,400 | 7,484,400 |
| Solve density $\rho = 1/K!$ | 1/2 | 1/120 | 1/720 |
| Rooms $D = K+1$ | 3 | 6 | 7 |

All three cases use the default Stage 1 config (80 MCTS sims, 30 games/iter, Dirichlet $\epsilon=0.40$, $\tau=1.0$, 15 iterations).

## Method

Run `ExplicitSurfaceDerivationGame` with AlphaZero. At each iteration, log policy cross-entropy $L_\pi$, value MSE $L_v$, and average rollout reward `avg_reward`. All three instances solve at iteration 1 (`best_solve_rate = 1.0`) — the question is not *whether* a solving program is found, but whether the network learns to reproduce MCTS's policy.

## Results

### Policy loss

![Policy loss D3/D6/D7](../../presentations/improvementv1/D3_D6_D7_policy_loss.png)

| Iteration | D=3 | D=6 | D=7 |
|---:|---:|---:|---:|
| 1 | 1.375 | 2.353 | 2.500 |
| 5 | 0.289 | 1.354 | 1.674 |
| 10 | 0.245 | 0.941 | 1.143 |
| 15 | 0.236 | 0.783 | 0.920 |

D=3 converges by iter 5. D=6 and D=7 are still decreasing at iter 15.

### Value loss and average reward

![Value loss D3/D6/D7](../../presentations/improvementv1/D3_D6_D7_value_loss.png)
![Avg reward D3/D6/D7](../../presentations/improvementv1/D3_D6_D7_avg_reward.png)

- **Value loss:** D=3 converges to $\approx 0.036$. D=7 *increases* early and plateaus around 0.41.
- **Average reward:** D=3 hits ~1.0 from iter 1; D=6, D=7 climb slowly to 0.8–1.0.

### Uniform-entropy floor comparison

If MCTS visits were uniform over legal actions, the per-step entropy floor would be $\bar H_{\mathrm{unif}} = \frac{1}{2K+1}\sum_t \log m_t$. For the canonical D=3 derivation (steps 0,1 have $m=2$; steps 2–4 are forced, $m=1$), $\bar H_{\mathrm{unif}} = (0.693+0.693)/5 = 0.277$.

| $D$ | $\bar H_{\mathrm{unif}}$ | Observed $L_\pi^{(15)}$ | Relationship |
|---:|---:|---:|---|
| 3 | 0.277 | 0.236 | $L < \bar H_{\mathrm{unif}}$ |
| 6 | 0.871 | 0.783 | $L < \bar H_{\mathrm{unif}}$ |
| 7 | 1.012 | 0.920 | $L < \bar H_{\mathrm{unif}}$ |

Since $L_\pi = H(\pi_{\mathrm{MCTS}}) + D_{\mathrm{KL}} \geq H(\pi_{\mathrm{MCTS}})$ and $L < \bar H_{\mathrm{unif}}$, MCTS targets are sharper than uniform — but we can't tell how much of $L$ is floor vs. KL gap without logging the true $\bar H_{\mathrm{MCTS}}$ (see [exp2](exp2_ideal_loss_theory.md)).

### Scaling table across $D = 3 \ldots 7$ (policy-loss delta in last 5 iterations)

| $D$ | $K$ | $L_\pi^{(1)}$ | $L_\pi^{(15)}$ | $\bar H_{\mathrm{unif}}$ | $\Delta L_\pi^{(11\text{-}15)}$ | $L_v^{(15)}$ | Status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 2 | 1.375 | 0.236 | 0.277 | 0.007 | 0.036 | Converged |
| 4 | 3 | 1.808 | 0.469 | 0.512 | 0.019 | 0.103 | ≈ converged |
| 5 | 4 | 2.091 | 0.605 | 0.706 | 0.040 | 0.166 | Slow |
| 6 | 5 | 2.353 | 0.783 | 0.871 | 0.111 | 0.291 | Not converged |
| 7 | 6 | 2.499 | 0.920 | 1.012 | 0.163 | 0.407 | Not converged |

D=5 is the smallest case where convergence is uncertain — the right target for the diagnostic sweep in [exp3](exp3_D5_diagnostic_sweep.md).

## Analysis

MCTS does all the work. With $\rho = 1/2$ at D=3, every random-prior MCTS tree stumbles on a solving word immediately; the training buffer is all-solve from iter 1 and $\mathrm{Var}(z) \to 0$, so the value head can trivially learn the constant $\hat v(s) \approx 1.15$. At D=6 ($\rho = 1/120$), the buffer contains a mix of solves and failures for many iterations; the value head cannot even fit the mean because the same state $s$ appears with different $z$ values as the policy shifts. The value loss is dominated by **policy variance**, not reward variance.

Five diagnostic hypotheses for why the gap grows with $K$ (formally tested in later experiments):

1. **Insufficient data.** Coverage $C$ = tuples/iter × iters / $3^K$ drops from 250× at D=3 to 8× at D=7.
2. **Non-stationary targets.** The replay buffer holds 20 past iterations; stale targets conflict with recent ones once MCTS starts sharpening.
3. **Observation encoding.** The 2-layer transformer sees a flat token sequence and must implicitly parse the status vector $\mathbf s \in \{0,P,M\}^K$ — feasible at D=3 (2-dim), hard at D=7 (6-dim).
4. **Reward sparsity.** Under a random policy, expected return is 0.58 at D=3 but 0.002 at D=7.
5. **Exploration noise.** $\epsilon=0.40$ means 40% of the root prior is Dirichlet noise; MCTS visit distributions stay flat even when the solution is known.

## Takeaways

- "MCTS solves it" ≠ "the network learned it." Convergence must be measured against $H(\pi_{\mathrm{MCTS}})$, not against 0.
- Convergence visibly degrades starting at **D=5** ($K=4$). D=5 is the natural target for controlled ablations.
- The combined value-loss pathology at D=6 (increases early, plateaus at 0.29) is a *policy-variance* artifact, not a reward-variance artifact.
- Before running any more experiments, add $\bar H_{\mathrm{MCTS}}$ logging to the training loop so every subsequent diagnostic gets a principled $D_{\mathrm{KL}}$ number.

## Source slides

Frames 3–5 (what we ran, observed losses), 21–27 (scaling table, where non-convergence begins), 38–41 (diagnosis 1–5). Context frames 9–20 explain the grammar, compiler, and runtime semantics and are summarized in [appendix/alphazero_grammar_framework.md](../appendix/alphazero_grammar_framework.md) where relevant.
