# Stage 1 — Masked Grammar $G_1$: Diagnostic Report

**Source:** [stage1_grammar_experiment.tex](../../presentations/improvementv1/stage1_grammar_experiment.tex) (62 frames, April 2026)

## Narrative

Stage 1 runs AlphaZero on the Doors derivation game with the domain-specific right-linear grammar $G_1$ (defined below).

The headline paradox: **MCTS solves the game at iteration 1 for every $D \in \{3, 6, 7\}$, but the neural network does not learn the MCTS policy.** At $D=7$ the policy loss is 0.92 after 15 iterations, barely below the uniform-random floor of 1.01.

Stage 1 diagnoses this by (a) deriving the ideal-loss floor $\bar H_{\mathrm{MCTS}}$ that lets us decompose policy loss into a reducible gap $D_{\mathrm{KL}}$ plus an irreducible entropy term, and (b) sweeping one hyperparameter at a time at $D=5$ (the smallest "uncertain convergence" case) and $D=6$. The diagnosis concludes that the non-convergence is **primarily a patience problem** — the network learns, just slowly — and that **Dirichlet noise $\epsilon$ is the most impactful knob**.

## Grammar $G_1$

$G_1$ is a **right-linear context-free grammar** $G_K = (N, \Sigma, P, S)$, parameterised by the number of keys $K = D - 1$. It is right-linear because every production has at most one nonterminal on its right-hand side, and when present that nonterminal is rightmost — so derivations generate words strictly left-to-right, one terminal at a time. This is exactly the CFG class recognised by a (deterministic) finite automaton; $G_1$'s nonterminals *are* the states of that automaton.

**Nonterminals.** $N = \{N_{\vec s} : \vec s \in \{0, P, M\}^K\}$, so $|N| = 3^K$. Each component $s_k$ tracks key $k$'s stage: untouched ($0$), picked ($P$), or moved/used ($M$).

**Terminals.** $\Sigma = \{P_0, \dots, P_{K-1}, M_0, \dots, M_{K-1}, G\}$ (the $2K+1$ macro rules): pick-up-key-$k$, use-key-$k$, and the goal-reached marker.

**Start symbol.** $S = N_{(0, 0, \dots, 0)}$ — all keys untouched.

**Production schema.** For each $\vec s \in \{0, P, M\}^K$ and each key index $k$:

$$
\begin{aligned}
N_{\vec s} &\to P_k \, N_{\vec s[k \mapsto P]} &&\text{if } s_k = 0, \\
N_{\vec s} &\to M_k \, N_{\vec s[k \mapsto M]} &&\text{if } s_k = P, \\
N_{(M, \dots, M)} &\to G &&\text{(only terminating production)}.
\end{aligned}
$$

Three structural properties fall out of the schema — and these are precisely the three "masks":

1. **Per-key precedence** ($P_k$ before $M_k$): $M_k$ is only reachable from $s_k = P$.
2. **Token uniqueness**: once $s_k = M$, no further $P_k$ or $M_k$ production applies.
3. **Completeness**: $G$ is only legal at $\vec s = (M, \dots, M)$.

Because the schema already enforces these, "masked" here is a runtime statement about MCTS, not about $G_1$ itself: the game's action mask is exactly `SurfaceCFG.legal_actions(nt)` ([explicit_surface_cfg.py:208](../../../src/alphazeropp/instances/doors/dsl/explicit_surface_cfg.py#L208)), which zeroes every illegal terminal in the neural-network policy head. The search tree only ever branches on legal productions.

**Language size.** A legal word is an interleaving of the $K$ pairs $(P_k, M_k)$ with $P_k$ preceding $M_k$ and a final $G$. This is the number of ways to arrange $2K$ ordered tokens modulo the $2^K$ within-pair swaps:

$$
|L(G_1)| = \frac{(2K)!}{2^K}.
$$

**Example ($K = 2$, $D = 3$).** $|N| = 9$, $|\Sigma| = 5$, $|L(G_1)| = 6$. A full derivation from $S = N_{(0,0)}$:

$$
N_{(0,0)} \xrightarrow{P_0} N_{(P,0)} \xrightarrow{M_0} N_{(M,0)} \xrightarrow{P_1} N_{(M,P)} \xrightarrow{M_1} N_{(M,M)} \xrightarrow{G} \varepsilon.
$$

**Implementation.** The CFG is constructed eagerly at game init ([`SurfaceCFG` in explicit_surface_cfg.py:154](../../../src/alphazeropp/instances/doors/dsl/explicit_surface_cfg.py#L154)) and wrapped as an MDP by [`ExplicitSurfaceDerivationGame`](../../../src/alphazeropp/instances/doors/dsl/explicit_surface_derivation_game.py). Macro tokens compile down to the base AST DSL via the surface compiler; see [overview_project.md §2–3](../overview_project.md) for how $G_1$ relates to $G_2$ (unmasked) and $G_3$ (lifted typed CFG).

## Experiments

| # | File | Headline result |
|---|---|---|
| 1 | [exp1_baseline_D3_D6_D7.md](exp1_baseline_D3_D6_D7.md) | MCTS solves at iter 1 for D=3,6,7; network converges only at D=3. |
| 2 | [exp2_ideal_loss_theory.md](exp2_ideal_loss_theory.md) | Decompose $L_\pi = \bar H_{\mathrm{MCTS}} + D_{\mathrm{KL}}$; need to log $\bar H_{\mathrm{MCTS}}$ to separate "noisy targets" from "network failed". |
| 3 | [exp3_D5_diagnostic_sweep.md](exp3_D5_diagnostic_sweep.md) | 6-config sweep at D=5: long run and $\epsilon=0.10$ are the two effective single-parameter fixes. |
| 4 | [exp4_D6_scaling.md](exp4_D6_scaling.md) | Same ranking at D=6, shifted upward. Coverage $C = $ tuples per nonterminal is the fundamental scaling bottleneck. |

## Results

### Experiment 1 — Baseline at $D \in \{3, 6, 7\}$

- **Headline.** Policy loss converges cleanly at $D=3$ (1.375 → 0.236 over 15 iterations) but is still decreasing at $D=6$ (→ 0.783) and $D=7$ (→ 0.920). Value loss tracks: $L_v = 0.036$ at $D=3$, 0.291 at $D=6$, 0.407 at $D=7$. All three sizes solve at iteration 1 — the question is whether the network reproduces MCTS, not whether MCTS finds a solution.
- **Methodology.** Default Stage 1 config (80 sims, 30 games/iter, $\epsilon = 0.40$, $\tau = 1.0$, 15 iterations).
- **Critical threshold.** Policy-loss delta over iterations 11–15 across $D = 3 \ldots 7$: 0.007, 0.019, 0.040, 0.111, 0.163. $D=5$ is the smallest case where convergence is uncertain — the natural target for the diagnostic sweep.
- **Takeaway.** "MCTS solves it" $\neq$ "the network learned it"; convergence must be measured against $H(\pi_{\mathrm{MCTS}})$, not zero. Five hypotheses are listed for the growing gap with $K$: insufficient data coverage, non-stationary targets, observation encoding, reward sparsity, and exploration noise.

### Experiment 2 — Ideal-Loss Decomposition

- **Headline.** Theoretical: $L_{\mathrm{policy}} = H(\pi_{\mathrm{MCTS}}) + D_{\mathrm{KL}}(\pi_{\mathrm{MCTS}} \,\Vert\, \hat\pi_\theta)$, so the right convergence target is $\bar H_{\mathrm{MCTS}}$ rather than zero.
- **Methodology.** No new run; this is the decomposition that turns baseline logs into a principled convergence criterion used by exp3 and exp4.
- **Key bounds.** Uniform-entropy upper bounds $\bar H_{\mathrm{unif}}$: $D=3 \to 0.277$, $D=6 \to 0.871$, $D=7 \to 1.012$. At every $D$, observed $L_\pi^{(15)} < \bar H_{\mathrm{unif}}$, confirming MCTS targets are sharper than uniform but not pinning the true floor.
- **Value-loss diagnostic.** $\mathrm{Var}(z) = 0.30$ at $D=3$ vs observed $L_v = 0.036$ (head fits the constant); $\mathrm{Var}(z) = 0.01$ at $D=6$ vs $L_v = 0.291$ (the value head is chasing a moving target driven by policy non-convergence).
- **Takeaway.** Every subsequent experiment must log $\bar H_{\mathrm{MCTS}}$ so that $D_{\mathrm{KL}} = L_\pi - \bar H_{\mathrm{MCTS}}$ becomes the success criterion.

### Experiment 3 — $D=5$ Diagnostic Sweep

- **Headline.** Of six single-parameter interventions at $D=5$ ($K=4$, $|N| = 81$, 3,024 legal policies), patience and reduced Dirichlet noise are the two effective fixes; doubling MCTS sims has no effect.
- **Methodology.** Six independent configs vary one knob each — iteration count, games-per-iter, MCTS budget, $\epsilon$, $\tau$ — under the success criterion $D_{\mathrm{KL}} < 0.05$ at the final iteration.
- **Final-iteration $D_{\mathrm{KL}}$.** Baseline 0.186 (15 iter) · long run **0.095** (100 iter, best overall) · $\epsilon = 0.10$ **0.122** (best at fixed compute, $-34\%$ vs baseline) · more data 0.133 · 200 sims 0.182 (no effect) · $\tau = 0.5$ 0.236 (worst $D_{\mathrm{KL}}$, best $L_v = 0.064$).
- **Takeaway.** Patience reduces $D_{\mathrm{KL}}$ by 49% on its own; $\epsilon = 0.10$ is the best 15-iter intervention. Recommended config going forward: $\epsilon = 0.10$, $\tau = 0.5$, 100 iterations, 80 sims, 30 games/iter — combine patience with a sharp prior.

### Experiment 4 — $D=6$ Scaling

- **Headline.** The $D=5$ ranking holds at $D=6$ ($K=5$, $|N| = 243$, 113,400 legal policies), shifted uniformly upward. Coverage $C$ — tuples per nonterminal in the replay buffer — is the fundamental scaling bottleneck.
- **Methodology.** Same six-intervention structure as $D=5$, with "sharper MCTS" replaced by a combined ($\epsilon = 0.10, \tau = 0.5$, long-run) experiment.
- **Final-iteration $D_{\mathrm{KL}}$.** Baseline 0.322 · long run **0.130** (still above $D=5$'s 0.095) · $\epsilon = 0.10$ 0.201 · more data 0.230 · $\tau = 0.5$ 0.390 (worst $D_{\mathrm{KL}}$, best $L_v = 0.115$).
- **Coverage scaling.** $C = 333\times$ at $D=3$, $67\times$ at $D=5$ (long run), $27\times$ at $D=6$ (long run), $11\times$ at $D=7$. Empirical scaling law $D_{\mathrm{KL}} \sim 3^K / C$ — to hold $D_{\mathrm{KL}}$ fixed, data must grow as $\Theta(3^K)$.
- **Takeaway.** Reaching $D_{\mathrm{KL}} < 0.05$ at $D=6$ likely needs $C > 100\times$ (≈ 300 games/iter or 300 iterations). Without architectural changes (weight sharing across nonterminals, lifted representations), coverage caps further scaling — directly motivating Stage 3.

## Configuration (used unless noted otherwise)

| Component | Value |
|---|---|
| Game | `ExplicitSurfaceDerivationGame` |
| Grammar | Right-linear $G_1$; $\lvert N\rvert = 3^K$, $\lvert L(G_1)\rvert = (2K)!/2^K$ |
| Network | `DerivationPolicyValueNet`: 2-layer transformer, $d=64$, 4 heads |
| MCTS | 80 sims, $c_{\mathrm{PUCT}}=1.5$, Dirichlet $\alpha=0.25$, $\epsilon=0.40$ |
| Training | 30 games/iter, 5 epochs, batch 32, lr $3\!\times\!10^{-4}$, 15 iterations |

## Takeaways that carry forward to Stage 2

1. The non-convergence is **not a bug** — it is a scaling property of the learner, not the grammar.
2. The masked grammar is **search-robust**: MCTS finds solutions regardless of hyperparameters, because the grammar pre-filters junk.
3. Hyperparameters tuned here ($\epsilon=0.10$, $\tau=0.5$) become the Stage 2 starting point. Stage 2 will retune them because the unmasked grammar responds to $\epsilon$ differently.
4. Coverage $C$ = training tuples per nonterminal is the scaling-limiting quantity. To hold $D_{\mathrm{KL}}$ fixed, data must grow as $\Theta(3^K)$.
