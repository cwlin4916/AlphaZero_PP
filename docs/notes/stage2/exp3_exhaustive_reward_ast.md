# Stage 2 / Experiment 3 — Exhaustive Analysis: Reward, AST, Solve Density

## Question

When we enumerate *every* Stage 2 derivation at small $D$, how many compile? How many solve? How is reward distributed? And do different derivations collapse to the same AST?

## Setup

For $D \in \{2, 3, 4\}$ we enumerated every exact-length Stage 2 derivation, compiled it with `surface_compiler.compile_policy`, and evaluated the resulting program on Doors via `LeafEvaluator`. 33/33 test cases (spot checks) compile without crashing; every compiled program evaluates to a finite reward.

## Results

### Solve counts — the critical finding

| $D$ | $\lvert L(G_1)\rvert$ | $\lvert L_=(G_2)\rvert$ | $G_1$ solve | $G_2$ solve | $\rho_1$ | $\rho_2$ | $\rho_1/\rho_2$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 4 | 1 | 1 | 1.000 | 0.2500 | 4× |
| 3 | 6 | 256 | 3 | 3 | 0.500 | 0.0117 | 43× |
| 4 | 90 | 46,656 | 15 | 15 | 0.167 | 0.000322 | 518× |

**The number of solving programs is identical in $G_1$ and $G_2$.** Removing the masks adds only junk — zero new solvers. The masks were filtering programs that *cannot* solve, not programs that *might* solve.

### Reward distribution

![Reward histograms G1 vs G2](../../presentations/improvementv1/stage1_vs_stage2_reward_histograms.png)

- **Top row (Stage 1):** rewards are spread across the range — a significant fraction of programs solve.
- **Bottom row (Stage 2):** almost all programs receive reward 0. The solving spike at $r > 1$ is tiny.

The reward landscape becomes extremely sparse without masks.

### AST uniqueness

![Unique ASTs](../../presentations/improvementv1/stage1_vs_stage2_unique_asts.png)

| $D$ | Derivations | Unique ASTs |
|---:|---:|---:|
| 2 | 4 | 4 |
| 3 | 256 | 256 |
| 4 | 46,656 | 46,656 |

Zero AST collisions — every Stage 2 derivation compiles to a distinct AST. There is no deduplication benefit; the entire language blowup translates directly into distinct programs the optimizer must sift through.

### Solve density

![Solve density](../../presentations/improvementv1/stage1_vs_stage2_solve_density.png)

Solve density $\rho = N_{\mathrm{solve}} / |L|$ (closed form below):

| $D$ | $\rho_1 = 1/K!$ | $\rho_2$ | $\rho_1 / \rho_2$ |
|---:|---:|---:|---:|
| 2 | 1.0 | 0.25 | 4× |
| 3 | 0.50 | 0.012 | 43× |
| 4 | 0.167 | $3.2 \times 10^{-4}$ | 518× |
| 5 | 0.042 | $6.3 \times 10^{-6}$ | 6,658× |
| 6 | 0.0083 | $9.4 \times 10^{-8}$ | 88,183× |
| 7 | 0.0014 | $1.2 \times 10^{-10}$ | 1.19M× |

Because $N_{\mathrm{solve}}$ is the same, the density ratio equals the language blowup factor.

### Combined scaling view

![Combined scaling D2-D7](../../presentations/improvementv1/stage1_vs_stage2_combined_scaling.png)

Four panels: language size (log, top-left), blowup factor (top-right), solve density (log, bottom-left), density ratio $\rho_1/\rho_2$ (bottom-right).

## What the masks were doing

Each mask removes a category of junk programs:

1. **Repeated tokens** (uniqueness violation). E.g. $P_0 P_0 P_0 P_0 G$ — compiles to nested `Ite` checking `PickReady(0)` four times. Reward 0.
2. **Move-before-Pick** (precedence violation). E.g. $M_0 P_0 M_1 P_1 G$ — compiles fine, but `MoveToKey(0)` fires before key 0 is picked, wrong room. Reward 0.
3. **Early termination** (completeness violation, only in max-length mode). E.g. $G$ alone — agent walks straight to goal without any keys. Reward 0.

All three categories compile and evaluate without crashing. They simply produce programs that don't solve the puzzle.

## Language inclusion — verified

**Theorem (verified for $D \in \{2,3,4\}$):** $L(G_1) \subsetneq L_=(G_2)$.

*Proof sketch.* Every Stage 1 word is an ordering of $\{P_0, M_0, \ldots, P_{K-1}, M_{K-1}\}$ with $P_k$ before $M_k$ followed by $G$ — exactly $2K+1$ symbols. Every exact-length Stage 2 word is any sequence of $2K$ non-goal tokens followed by $G$ — exactly $2K+1$ symbols. Same tokens, same length, so every Stage 1 word is automatically a Stage 2 exact-length word. Strict inclusion: Stage 2 contains e.g. $P_0 P_0 \ldots G$ which violates uniqueness. $\square$

## Takeaways

1. **Masks filter only junk.** The Stage 2 search space adds no solvers; it only adds zero-reward decoys.
2. **Zero AST collisions** — no syntactic deduplication is available; the optimizer must genuinely navigate the full expanded space.
3. **Solve density collapses super-exponentially**: $\rho_2$ falls from 0.25 at $D=2$ to $1.2\times 10^{-10}$ at $D=7$. This is the quantitative source of the bootstrap problem in [exp4](exp4_training_scaling.md).
4. The theoretical formulas $|L(G_1)| = (2K)!/2^K$, $|L_=(G_2)| = (2K)^{2K}$, $N_{\mathrm{solve}} = (2K-1)!!$ close the analysis and agree exactly with the exhaustive counts.

## Source slides

Frames 19–24 (exhaustive solve counts, reward histograms, AST uniqueness, solve density), frame 25 (three-mask breakdown), frame 26 (language-inclusion theorem), frame 32 (combined scaling plot).
