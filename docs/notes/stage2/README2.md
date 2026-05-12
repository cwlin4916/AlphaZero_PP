# Stage 2 — Unmasked Grammar $G_2$: Ablation Report

**Source:** [stage2_grammar_experiment.tex](../../presentations/improvementv1/stage2_grammar_experiment.tex) (48 frames, April 2026)

## Narrative

Stage 2 removes every domain-specific constraint from the Stage 1 grammar while holding everything else fixed — same tokens, same compiler, same Doors evaluator, same network, same MCTS. The unmasked grammar $G_2$ uses exact-length productions $S_\ell \to T\,S_{\ell-1}$ for $1 \leq \ell \leq 2K$ and $S_0 \to G$, giving $|L_=(G_2)| = (2K)^{2K}$ words of length $2K+1$. This eliminates the three Stage 1 masks (precedence $P_k < M_k$, uniqueness, completeness) and keeps only structural length.

The central research question: **how much of Stage 1's learning success was due to the masks, and how much was due to the optimizer?**

Three findings emerge:

1. **The masks filter only junk.** Exhaustive enumeration at $D \in \{2,3,4\}$ confirms $N_{\mathrm{solve}}$ is *identical* in $G_1$ and $G_2$. The masks remove zero-reward programs, not promising ones.
2. **The boundary is sharp at $D=6$.** With optimized hyperparameters ($\epsilon = 0$, $\tau = 0.5$), Stage 2 matches Stage 1 at $D \leq 5$ (first-solve iter 2 at $D=5$ for best seed). At $D \geq 6$ Stage 2 **never solves** (0/6 seeds in 50 iterations) while Stage 1 solves at iter 1 under the same hyperparameters.
3. **Hyperparameters matter more without masks.** Stage 1 is robust to $\epsilon$; Stage 2 needs $\epsilon = 0$ (zero Dirichlet noise) because 99.99% of the search space is junk and any noise dilutes a hard-won prior.

## Experiments

| # | File | Headline result |
|---|---|---|
| 1 | [exp1_hparam_tuning.md](exp1_hparam_tuning.md) | 19-run $\epsilon\times\tau$ sweep: winner $\epsilon=0,\tau=0.5$ (first-solve iter 2, 5× faster than prior best). |
| 2 | [exp2_language_size.md](exp2_language_size.md) | $\lvert L_=(G_2)\rvert / \lvert L(G_1)\rvert \sim e^{2K}/\sqrt{4\pi K}$. At $D=7$ the ratio is $1.19\times 10^6$. |
| 3 | [exp3_exhaustive_reward_ast.md](exp3_exhaustive_reward_ast.md) | Same $N_{\mathrm{solve}}$ in both grammars; zero AST collisions; solve density collapses super-exponentially. |
| 4 | [exp4_training_scaling.md](exp4_training_scaling.md) | Stage 2 solves $D \leq 5$; fails completely at $D \geq 6$ (bootstrap problem). |

## Results

### Experiment 1 — $\epsilon \times \tau$ Hyperparameter Sweep

- **Headline.** A 19-run $\epsilon \times \tau$ sweep at $D=5$ unmasked ($\lvert L_=(G_2)\rvert = 16.8\text{M}$) finds $\epsilon = 0$, $\tau = 0.5$ as the canonical Stage 2 setting — first-solve at iteration 2 (max 2), 5× faster than Stage 1's prior best of iteration 11 ($\epsilon = 0.10, \tau = 0.5$).
- **Methodology.** $\epsilon \in \{0, 0.03, 0.05, 0.10\}$ × $\tau \in \{0.25, 0.35, 0.50, 0.75, 1.0\}$, 80 sims, 50 games/iter, 25 iterations, 2 seeds each.
- **Pattern.** All top-3 configs use $\epsilon = 0$; $\tau = 0.5$ dominates across noise levels. $\tau = 0.75$ at $\epsilon = 0$ fails to solve (targets too flat); $\tau = 0.35$ is brittle (works at $\epsilon = 0$ with high variance, breaks under any noise).
- **Takeaway.** Without masks, Dirichlet noise dilutes a hard-won prior across a junk-dominated action space, so $\epsilon = 0$ is canonical; $\tau = 0.5$ is the sweet spot between concentration and seed diversity.

### Experiment 2 — Language-Size Blowup

- **Headline.** Removing masks causes a super-exponential explosion: $\lvert L_=(G_2)\rvert / \lvert L(G_1)\rvert = (2K)^{2K} \cdot 2^K / (2K)! \sim e^{2K}/\sqrt{4\pi K}$, with base $e^2 \approx 7.4$ per additional key.
- **Methodology.** Closed-form enumeration of $\lvert L(G_1)\rvert = (2K)!/2^K$ and $\lvert L_=(G_2)\rvert = (2K)^{2K}$ for $D = 2 \ldots 7$, verified by exhaustive enumeration at $D \in \{2, 3, 4\}$.
- **Ratios.** $D=2$: 4× · $D=3$: 43× · $D=4$: 518× · $D=5$: 6,658× · $D=6$: 88,183× · $D=7$: 1.19M× (8.9T words in $G_2$ vs 7.5M in $G_1$).
- **Why super-exponential.** Stage 1 averages $\sim K$ legal actions per step (constrained walk through $3^K$ nonterminals); Stage 2 sees $2K$ legal non-goal actions every step (sampling with replacement). The branching ratio compounds at each step.
- **Takeaway.** Masks provided exponential search-space compression. Without them, any framework needs an equally strong inductive bias (learned prior, lifted typing) or accepts the search-space penalty.

### Experiment 3 — Exhaustive Reward / AST / Solve Density

- **Headline.** Stage 2's larger language contains **no new solvers**: at $D \in \{2, 3, 4\}$ both grammars yield the same number of solving programs (1, 3, 15 respectively). Masks were filtering junk, not promising candidates.
- **Methodology.** Enumerate every exact-length Stage 2 derivation at $D = 2, 3, 4$, compile via `surface_compiler.compile_policy`, evaluate with `LeafEvaluator`. 33/33 spot checks compile without crashing; every compiled program evaluates to a finite reward.
- **AST uniqueness.** Zero AST collisions across 46,656 $D=4$ derivations — every Stage 2 derivation is a distinct program; there is no syntactic deduplication benefit.
- **Solve-density collapse.** $\rho_2 = 0.25$ ($D=2$) → 0.012 ($D=3$) → $3.2 \times 10^{-4}$ ($D=4$) → $6.3 \times 10^{-6}$ ($D=5$) → $9.4 \times 10^{-8}$ ($D=6$) → $1.2 \times 10^{-10}$ ($D=7$). The density ratio $\rho_1 / \rho_2$ tracks the language blowup factor exactly.
- **What the masks remove.** Three categories of compileable junk: repeated tokens (e.g. $P_0 P_0 P_0 P_0 G$), move-before-pick orderings (e.g. $M_0 P_0 \ldots$), and (in max-length mode) early-termination programs. All compile fine; all earn reward 0.
- **Takeaway.** Solve density falls with the language blowup — this is the quantitative source of the bootstrap problem in exp4.

### Experiment 4 — Training Scaling

- **Headline.** Stage 2 solves $D \leq 5$ with the tuned hyperparameters but **fails completely at $D \geq 6$**: 0/6 seeds in 50 iterations (3 seeds at $D=6$ + 3 seeds at $D=7$), while Stage 1 solves at iteration 1 for every $D$ under identical hyperparameters.
- **Methodology.** $D = 3, 4, 5$: 12 runs (2 stages × 3 $D$-values × 2 seeds) at $\epsilon = 0.10, \tau = 0.5$, 50 iterations. $D = 6, 7$: 6 runs (2 stages × 2 $D$-values × 3 seeds) at the optimized $\epsilon = 0, \tau = 0.5$. Same hyperparameters across stages — grammar is the only variable.
- **$D \leq 5$ behaviour.** Stage 2 first-solve always within iteration 17 (worst case: $D=5$, seed 137). Stage 1 always at iteration 1.
- **The cliff.** At $D = 6, 7$ Stage 2 plateaus at high policy loss with best reward $-0.150$ — the network learns the junk-reward distribution but never sees a solver.
- **Why — bootstrap probability.** Probability that a random rollout finds a solver: 43% ($D=5$) → **1.8%** ($D=6$) → 0.02% ($D=7$). Below ~5% the learning loop cannot start.
- **Takeaway.** Masks are not necessary up to $D=5$ but are decisive at $D \geq 6$. Three forward paths are flagged: curriculum learning (transfer weights from $D-1$), partial masking ("Stage 2.5"), or Stage 3's lifted schema.

## Configuration

| Component | Value |
|---|---|
| Game | `UnmaskedSurfaceDerivationGame` (exact-length mode) |
| Grammar | $G_2$: $L_{\max}+1$ nonterminals, $(2K)^{2K}$ words |
| Compiler | `surface_compiler.compile_policy` (unchanged from Stage 1) |
| Evaluator | `LeafEvaluator` with `solve_rate` metric (unchanged) |
| Tuned MCTS | $\epsilon = 0$ (zero noise), $\tau = 0.5$, 80 sims, 50 games/iter, 2–3 seeds |
| Training runs | $D=3,4,5$: 50 iters with $\epsilon=0.10$; $D=6,7$: 50 iters with optimized $\epsilon=0, \tau=0.5$ |

## Stage 1 $\subset$ Stage 2, verified

Computationally for $D \in \{2,3,4\}$:

| $D$ | $\lvert L(G_1)\rvert$ | $\lvert L_=(G_2)\rvert$ | $\lvert L_=(G_2) \setminus L(G_1)\rvert$ |
|---:|---:|---:|---:|
| 2 | 1 | 4 | 3 |
| 3 | 6 | 256 | 250 |
| 4 | 90 | 46,656 | 46,566 |

## Takeaways that carry forward to Stage 3

1. The masks gave an **exponential search-space compression** without removing any solvers — the archetype of "cheap" inductive bias.
2. The bootstrap problem (P(random solver) = 1.8% at $D=6$) is why Stage 2 cannot cold-start at large $D$. Any Stage 3 framework that wants to match Stage 1 without domain-specific masks must provide equivalent structural compression.
3. Decision-list execution semantics already make token order matter (priority ordering) — Stage 3 inherits this from the Stage 1/2 compiler.
