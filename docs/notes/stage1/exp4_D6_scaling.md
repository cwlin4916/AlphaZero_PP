# Stage 1 / Experiment 4 — D=6 Scaling Under the Best Config

## Question

Do the rankings from the D=5 sweep ([exp3](exp3_D5_diagnostic_sweep.md)) hold at D=6 ($K=5$, 243 nonterminals), and what is the coverage ceiling for further scaling?

## Setup

Target problem: **D=6**, $K=5$. 11 steps/episode, $3^5 = 243$ nonterminals, $\rho = 1/120$, 113,400 legal policies. Time per iter $\approx 2\,\text{min}$.

Same methodology as D=5, but replace "sharper MCTS" (no effect at D=5) with a **combined** experiment using the two best single-parameter fixes.

| # | Experiment | Change from baseline | Iterations |
|---|---|---|---:|
| 0 | Baseline + $\bar H$ logging | — | 15 |
| 1 | Long run | 100 iterations | 100 |
| 2 | More data | 100 games/iter | 15 |
| 3 | Less noise | $\epsilon = 0.10$ | 15 |
| 4 | Lower temp | $\tau = 0.5$ | 15 |
| 5 | **Combined** | $\epsilon = 0.10$, $\tau = 0.5$ | 100 |

D=5 vs D=6 comparison:

| | D=5 ($K=4$) | D=6 ($K=5$) |
|---|---:|---:|
| Nonterminals ($3^K$) | 81 | 243 |
| Policies | 3,024 | 113,400 |
| Solve density ($1/K!$) | 1/24 | 1/120 |
| Steps/game | 9 | 11 |
| Time/iter | ~30s | ~2 min |

## Results

### Final-iteration decomposition

![D6 KL bar](../../presentations/improvementv1/diag_D6_kl_bar.png)

| # | Experiment | $L_\pi$ | $\bar H_{\mathrm{MCTS}}$ | $D_{\mathrm{KL}}$ | $L_v$ |
|---|---|---:|---:|---:|---:|
| 0 | Baseline (15 iter) | 0.832 | 0.509 | 0.322 | 0.283 |
| 1 | Long run (100 iter) | 0.690 | 0.560 | **0.130** | 0.143 |
| 2 | More data (100 games) | 0.745 | 0.515 | 0.230 | 0.251 |
| 3 | Less noise ($\epsilon = 0.10$) | 0.609 | 0.408 | 0.201 | 0.204 |
| 4 | Lower temp ($\tau = 0.5$) | 0.583 | 0.193 | 0.390 | 0.115 |
| 5 | Combined$^\*$ | scheduled overnight | | | |

$^\*$ See `experiments/overnight_run.md`.

### Convergence curves

![D6 KL curves](../../presentations/improvementv1/diag_D6_kl_curves.png)

Long run (blue) trajectory: $0.42 \to 0.24 \to 0.15 \to 0.13$ over 100 iterations. Slower than D=5 (which reached 0.095). All curves show the same shape: rapid initial drop, long slow tail.

### Value loss and reward

![D6 value/reward](../../presentations/improvementv1/diag_D6_value_reward.png)

- Lower temp ($\tau = 0.5$): best value loss at 0.115 — sharper targets help the value head most.
- Baseline: value loss still 0.283 at iter 15; the head needs either more iterations or a sharper signal.
- All configs reach positive reward by iter 5 — MCTS finds solutions fast regardless of network quality.

### D=5 vs D=6 side-by-side

![D5 vs D6 KL](../../presentations/improvementv1/diag_D5_vs_D6_kl.png)

Same interventions, same ranking, shifted upward:
- D=6 curves are uniformly above D=5 at every iteration.
- Relative ordering is preserved: long run > less noise > more data > baseline > lower temp.
- The gap narrows with more iterations (long run: +37% vs baseline: +73%).

### D=6 decomposition

![D6 decomposition](../../presentations/improvementv1/diag_D6_decomposition.png)

Per-config $\bar H_{\mathrm{MCTS}}$ vs $D_{\mathrm{KL}}$ at D=6.

## Scaling analysis — why $D_{\mathrm{KL}}$ grows with $K$

The network must learn $\hat\pi(a\!\mid\! s)$ per nonterminal. Difficulty scales with:

- **State space:** $\lvert N\rvert = 3^K$.
- **Coverage:** $C = n_{\mathrm{past}} \cdot n_{\mathrm{games}} \cdot (2K+1) / 3^K$.

| $D$ | $K$ | $3^K$ | Buffer tuples | $C$ | $D_{\mathrm{KL}}^{(15)}$ |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 9 | 3,000 | 333× | 0.064 |
| 5 | 4 | 81 | 5,400 | 67× | 0.186 |
| 6 | 5 | 243 | 6,600 | 27× | 0.322 |
| 7 | 6 | 729 | 7,800 | 11× | — |

**Non-uniform visitation:** The root $N_{(0,\ldots,0)}$ is visited every game (30/iter), but a nonterminal at depth $d$ gets $\sim 30/K^d$ visits/iter — deep states are severely under-sampled.

**Target quality:** Nonterminals at D=6 average ~3.2 legal actions vs ~2.6 at D=5 — higher $\bar H_{\mathrm{MCTS}}$ and a noisier gradient signal.

Empirical scaling law: $D_{\mathrm{KL}} \sim 3^K / C \sim 3^K / (n_{\mathrm{games}} \cdot K)$. To hold $D_{\mathrm{KL}}$ fixed, data must grow as $\Theta(3^K)$.

## Coverage — the bottleneck

Coverage table (tuples per nonterminal in the replay buffer):

| | | | **30 games/iter** | | **100 games/iter** | |
|---:|---:|---:|---:|---:|---:|---:|
| $D$ | $K$ | $3^K$ | $C_{15}$ | $C_{100}$ | $C_{15}$ | $C_{100}$ |
| 3 | 2 | 9 | 250× | 333× | 833× | 1111× |
| 5 | 4 | 81 | 50× | 67× | 167× | 222× |
| 6 | 5 | 243 | 20× | 27× | 68× | 91× |
| 7 | 6 | 729 | 8× | 11× | 27× | 36× |

Empirical threshold: D=3 converges at $C \approx 250\times$; D=5 reaches $D_{\mathrm{KL}} = 0.095$ at $C = 67\times$ (long run). Reaching $D_{\mathrm{KL}} < 0.05$ at D=6 likely needs $C > 100\times$ — either ~300 games/iter or ~300 iterations, or both.

## Takeaways

1. **The D=5 ranking holds at D=6, shifted upward.** Long run remains the best single intervention (best $D_{\mathrm{KL}} = 0.130$ at iter 100).
2. **Best-config recommendation for Stage 2 onward:** $\epsilon = 0.10$, $\tau = 0.5$, long-iteration budget. Stage 2's hyperparameter sweep ([stage2/exp1](../stage2/exp1_hparam_tuning.md)) will retune these — the unmasked grammar responds to $\epsilon$ very differently.
3. **Coverage scales $\Theta(3^K)$ — this is a fundamental limit.** Without architectural changes (weight sharing across nonterminals, status-vector input features, lifted representations), coverage will always be the bottleneck. This directly motivates Stage 3.
4. The combined-config overnight run (entry 5) was scheduled but not reported here; see `experiments/overnight_run.md` for status.

## Source slides

Frames 52–58 (D=6 experiment design, results, convergence curves, value/reward, D=5 vs D=6 comparison, scaling analysis, coverage bottleneck).
