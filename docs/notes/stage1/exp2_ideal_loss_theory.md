# Stage 1 / Experiment 2 — Ideal-Loss Theory and the $\bar H_{\mathrm{MCTS}}$ Floor

## Question

What is the irreducible floor of the AlphaZero policy loss, and how do we measure it so that "not converged" can be distinguished from "converged to a naturally noisy target"?

## Setup

No experiment is run here. This is the theoretical decomposition that turns the baseline logs ([exp1](exp1_baseline_D3_D6_D7.md)) into a principled convergence criterion used in [exp3](exp3_D5_diagnostic_sweep.md) and [exp4](exp4_D6_scaling.md).

## Policy loss decomposition

The AlphaZero loss (Silver et al. 2017, Eq. 1) is
$$
 l(\theta) = (z - v_\theta)^2 \;-\; \boldsymbol\pi^\top \log \mathbf p_\theta \;+\; c\lVert\theta\rVert^2,
$$
where $\boldsymbol\pi = \pi_{\mathrm{MCTS}}$ is a **fixed target** (MCTS visit-count distribution at state $s$) and $\mathbf p_\theta = \hat\pi_\theta$ is the **learnable** network policy.

The policy term is a cross-entropy $H(\pi_{\mathrm{MCTS}}, \hat\pi_\theta)$ and decomposes as
$$
 L_{\mathrm{policy}} \;=\; \underbrace{H(\pi_{\mathrm{MCTS}})}_{\text{irreducible floor}} \;+\; \underbrace{D_{\mathrm{KL}}(\pi_{\mathrm{MCTS}} \,\Vert\, \hat\pi_\theta)}_{\text{learnable gap} \geq 0}.
$$
The entropy term does not depend on $\theta$, so $\nabla_\theta L_{\mathrm{policy}} = \nabla_\theta D_{\mathrm{KL}}$. Minimizing the policy loss is equivalent to making $\hat\pi_\theta \to \pi_{\mathrm{MCTS}}$.

**Consequence.** Rearranging gives a measurable convergence criterion:
$$
 D_{\mathrm{KL}} = L_{\mathrm{policy}} - H(\pi_{\mathrm{MCTS}}).
$$
- $D_{\mathrm{KL}} \to 0$: the network has matched MCTS.
- $D_{\mathrm{KL}} \gg 0$: the network is still learning.
- $L_{\mathrm{policy}} \to 0$ is **not** the right target — $L_{\mathrm{policy}} \to H(\pi_{\mathrm{MCTS}})$ is.

## Bounding the floor with uniform entropy

Without logging $\bar H_{\mathrm{MCTS}}$, a rough upper bound is $H_{\mathrm{unif}} = \log m$ where $m$ is the number of legal actions at each state. For D=3 on the canonical derivation: steps 0 and 1 have $m=2$, steps 2–4 are forced ($m=1$), so $\bar H_{\mathrm{unif}} = (0.693 + 0.693)/5 = 0.277$.

| $D$ | $\bar H_{\mathrm{unif}}$ | $L_\pi^{(15)}$ | Relationship |
|---:|---:|---:|---|
| 3 | 0.277 | 0.236 | $L < \bar H_{\mathrm{unif}}$ |
| 6 | 0.871 | 0.783 | $L < \bar H_{\mathrm{unif}}$ |
| 7 | 1.012 | 0.920 | $L < \bar H_{\mathrm{unif}}$ |

$L < \bar H_{\mathrm{unif}}$ confirms MCTS targets are sharper than uniform, but doesn't pin the true floor.

## Computing the true floor

For each training tuple $(s_t, \boldsymbol\pi_t, z)$:
1. Compute per-example entropy $H(\boldsymbol\pi_t) = -\sum_a \pi_t(a\!\mid\! s_t) \log \pi_t(a\!\mid\! s_t)$.
2. Average over the training batch: $\bar H_{\mathrm{MCTS}} = \frac{1}{N} \sum_i H(\boldsymbol\pi_i)$.
3. Compute the learnable gap: $D_{\mathrm{KL}} = L_{\mathrm{policy}} - \bar H_{\mathrm{MCTS}}$.

The ordering
$$
 0 \;\leq\; H(\pi_{\mathrm{MCTS}}) \;\leq\; L_{\mathrm{policy}} \;\leq\; \bar H_{\mathrm{unif}} \qquad(\text{when } D_{\mathrm{KL}} \text{ is small})
$$
pins the floor between 0 and $\bar H_{\mathrm{unif}}$.

## Value loss floor

The value term is $(z - v_\theta)^2$ with label $z = r_T$ (terminal reward, $\gamma = 1$). At Doors, $r_T = 1.0\!\cdot\![\text{goal reached}] + K\!\cdot\! 0.1\!\cdot\![\text{rooms unlocked}] - (2K+1)\!\cdot\! 0.01\!\cdot\![\text{step penalty}]$.

| | D=3 ($K=2$) | D=6 ($K=5$) |
|---|---|---|
| $r_{\mathrm{solve}}$ (all rooms, goal) | $1.0 + 0.2 - 0.05 = 1.15$ | $1.0 + 0.5 - 0.11 = 1.39$ |
| $r_{\mathrm{fail}}$ (partial, no goal) | $\approx 0.05$ | $\approx 0.09$ |

If a fraction $p$ of games solve, the best constant predictor is $\hat v^\star = p\, r_{\mathrm{solve}} + (1-p)\, r_{\mathrm{fail}}$ with residual
$$
 L_{\mathrm{value}}^\star \;\geq\; \mathrm{Var}(z) \;=\; p(1-p)(r_{\mathrm{solve}} - r_{\mathrm{fail}})^2.
$$

| $D$ | $p = 1/K!$ | $\mathrm{Var}(z)$ | $L_v^{(15)}$ | Status |
|---:|---:|---:|---:|---|
| 3 | 0.500 | 0.30 | 0.036 | $L_v \ll \mathrm{Var}$ — converged |
| 6 | 0.008 | 0.01 | 0.291 | $L_v \gg \mathrm{Var}$ — not converged |

**Why D=3 converges.** $p = 1/2$ so the replay buffer quickly becomes all-solve. $\mathrm{Var}(z) \to 0$ and the value head fits the constant 1.15.

**Why D=6 doesn't.** $L_v = 0.291 \gg \mathrm{Var}(z) = 0.01$. The value head can't even learn the mean: the same state $s$ appears with different $z$ across games because the *policy* is changing. Value loss is dominated by policy variance, not reward variance.

## Takeaways

1. The policy-loss target is $\bar H_{\mathrm{MCTS}}$, not 0. Every subsequent experiment must log it.
2. $D_{\mathrm{KL}} = L_\pi - \bar H_{\mathrm{MCTS}}$ is the *true* convergence metric and the success criterion for the D=5 and D=6 sweeps.
3. Value loss above $\mathrm{Var}(z)$ signals policy non-convergence — the value head is chasing a moving target defined by a still-shifting policy.

## Source slides

Frames 6–8 (ideal policy loss and entropy bounds), frames 17–19 (ideal value loss setup and variance floor).
