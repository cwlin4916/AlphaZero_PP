# Stage 2 / Experiment 2 — Language Size and Super-Exponential Blowup

## Question

How much larger is the unmasked grammar $G_2$ than the masked grammar $G_1$, and what is the asymptotic form of the ratio?

## Setup

Closed-form enumeration of $|L(G_1)| = (2K)!/2^K$ and $|L_=(G_2)| = (2K)^{2K}$ for $D = 2 \ldots 7$ ($K = D-1$), verified by exhaustive enumeration for $D \in \{2,3,4\}$ (see [exp3](exp3_exhaustive_reward_ast.md)).

## Results

### Language sizes

![Language size G1 vs G2](../../presentations/improvementv1/stage1_vs_stage2_language_size.png)

| $D$ | $\lvert L(G_1)\rvert$ | $\lvert L_=(G_2)\rvert$ | Ratio |
|---:|---:|---:|---:|
| 2 | 1 | 4 | 4× |
| 3 | 6 | 256 | 43× |
| 4 | 90 | 46,656 | 518× |
| 5 | 2,520 | 16.8M | 6,658× |
| 6 | 113,400 | 10B | 88,183× |
| 7 | 7.5M | 8.9T | 1.19M× |

### Blowup factor

![Blowup factor](../../presentations/improvementv1/stage1_vs_stage2_blowup.png)

Ratio formula:
$$
 \frac{\lvert L_=(G_2)\rvert}{\lvert L(G_1)\rvert} = \frac{(2K)^{2K} \cdot 2^K}{(2K)!} \;\stackrel{\text{Stirling}}{\sim}\; \frac{e^{2K}}{\sqrt{4\pi K}}.
$$

The blowup is exponential in $K$ with base $e^2 \approx 7.4$ — each additional key multiplies search difficulty by ~7×.

## Why super-exponential

**Stage 1 (masked):** at each step, only ~$K$ of $2K+1$ actions are legal. Each token is used exactly once with ordering constraints. The game is a directed walk through the $3^K$-nonterminal graph. Branching factor ~$K$ (average).

**Stage 2 (unmasked):** at each step, all $2K$ non-goal tokens are legal. Each of $2K$ slots chooses from $2K$ tokens with replacement. Branching factor = $2K$ (constant).

Effective branching ratio: $\tfrac{\text{Stage 2 branching}}{\text{Stage 1 branching}} = \tfrac{2K}{\bar b_1}$, compounding at each step.

**Concrete example.** At $D=7$ ($K=6$): Stage 1 averages ~5 legal actions per step, Stage 2 sees 12. Over 12 steps: $12^{12} / 5^{12} \approx 3.6 \times 10^5$.

## Analysis

Removing each mask contributes one factor of $2^K$, $2^K$, or a $2K{+}1$ option at each step to the language size. Concretely (for words of length $2K+1$):

| Mask removed | Effect | Multiplier |
|---|---|---:|
| Precedence ($P_k$ before $M_k$) | Each slot can choose P or M freely | $2^K$ |
| Uniqueness | Tokens can repeat | large |
| Early-goal (if also removed in max-length mode) | Could shorten words | not in exact-length |

In the exact-length variant used here, completeness is enforced structurally (every word has length exactly $2K+1$), so early termination is not available.

## Takeaways

1. The masks provided an **exponential compression** of the search space, with asymptotic base $e^2 \approx 7.4$ per additional key.
2. At $D=7$, Stage 2 has **1.19 million times** more words than Stage 1. Any framework that removes masks must either live with this search-space penalty or replace the inductive bias with something equally strong (e.g., a learned prior trained on smaller instances, or lifted typing as in Stage 3).
3. The formulas above — $|L(G_1)| = (2K)!/2^K$, $|L_=(G_2)| = (2K)^{2K}$, $N_{\mathrm{solve}} = (2K-1)!!$ — are the scaffolding for every comparison in this stage and reappear in [exp3](exp3_exhaustive_reward_ast.md) and [exp4](exp4_training_scaling.md).

## Source slides

Frames 16–18 (language size table, blowup factor), frames 32–35 (combined scaling, theoretical formulas, super-exponential solve density).
