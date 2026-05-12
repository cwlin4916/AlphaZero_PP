# Appendix — AlphaZero + CFG Framework

**Source:** [alphazero_grammar_beamer.tex](../../presentations/alphazero_grammar_beamer.tex) (~37 frames). Also see `docs/presentations/new_games_old_engine_report.md` for the longer narrative version.

This note summarizes the shared framework that Stages 1 and 2 both use: a two-level MDP where the outer level is CFG derivation and the inner level evaluates the derived program on a small game. The notation here is the foundation for every experiment in Stages 1, 2, and 3.

## 1. The core question

When should we search in program space vs. action space? Two approaches:

1. **Primitive actions.** Agent selects one action per env step; many steps per episode.
2. **Synthesize programs via grammar.** Agent selects grammar productions until a derivation terminates. The derivation is compiled into a reactive policy that is then run on the base environment.

The grammar provides inductive bias and reduces the effective horizon, but may increase branching. The trade-off is what Stage 2 quantifies ([stage2/exp2](../stage2/exp2_language_size.md)).

## 2. Placement in program synthesis

| Paradigm | Method | Grammar? | Learning? |
|---|---|---|---|
| Enumerative | Size-ordered enumeration | yes | no |
| Constraint-based (CEGIS, Sketch) | SMT / synthesis by sketching | partial | no |
| Stochastic search | MCMC, evolutionary | sometimes | limited |
| Neural-guided (DeepCoder, etc.) | Learned priors over DSL | yes | offline |
| **This work** | **MCTS + neural net + self-play on grammar derivation** | **yes** | **online** |

Distinctive features: online learning during search, action masking for grammar validity, reward-driven specification instead of logical.

## 3. Context-free grammar (formal)

A CFG is a 4-tuple $G = (V, \Sigma, R, S)$ with nonterminals $V$, terminals $\Sigma$, productions $R \subseteq V \times (V \cup \Sigma)^\ast$, and start symbol $S$. Each symbol gets a unique integer token; a padding token is introduced for fixed-length encodings.

### Worked example — bitstring grammar

```
S  → S₁ | S₂
S₁ → C A
S₂ → C A C A
C  → bit0 | bit1
A  → set0 | set1
```

Language size: $\lvert L(G)\rvert = 20$ programs (2 choices for $S$, each expanding to 4 or 16 via 2 × 2 terminals per $(C, A)$ pair). Two sample derivations:

```
S → S₁ → C A → bit0 A → bit0 set1
S → S₂ → C A C A → ... → bit0 set1 bit1 set0
```

Every derivation is a sequence of state vectors: at each step, one nonterminal is replaced by its RHS, the suffix is shifted, and the padding token fills unused slots. An **action** is a pair (position, production index). The **action mask** at state $s$ marks positions that contain a nonterminal and productions that match that nonterminal.

## 4. The two-level MDP

### Outer MDP — grammar derivation

| | Definition |
|---|---|
| State | Padded token vector of length $L_{\max}$ |
| Action | (position, production index) |
| Transitions | Deterministic: replace the nonterminal at the chosen position with the chosen production's RHS, shift suffix right, re-pad |
| Terminal | No nonterminals remain |
| Intermediate reward | 0 |
| Terminal reward | $R_{\mathrm{inner}}(s)$ — output of the inner game on the derived program |

### Inner MDP — game-specific evaluator

In the bitstring game ("OneMax"): state is an $N$-bit vector initialized with $k$ random 1s; an action flips one bit; reward is $+1$ if 0→1, $-1$ if 1→0 or invalid; terminal when all bits are 1 or $t \geq 2N$.

More generally the inner game is:
- **GrammarNetEnv** for bitstring tasks (Stage 1/2 use a Doors-specific inner env, not this).
- **GrammarSREnv** for symbolic regression (reward = $R^2$ fit).
- **GrammarOnesEnv** for a pure-counting warm-up.

## 5. Action masking — the crucial mechanism

The mask $m(s) \in \{0,1\}^{L_{\max} \times n_{\mathrm{prod}}}$ marks action $(p, j)$ valid iff:

1. The symbol at position $p$ is a nonterminal, and
2. Production $j$ has this nonterminal on its LHS, and
3. $\ell(s) + \lvert\mathrm{rhs}(j)\rvert - 1 < L_{\max}$ (no buffer overflow).

Failed overflow terminates the episode with $r = -1$. Masking reduces branching from raw ~48 actions to typically 2–4 valid actions.

### Where masking plugs in

- **Policy head.** Invalid actions get logit $-10^8$, so $p_\theta(a\!\mid\! s) \approx 0$ after softmax.
- **PUCT selection.** Invalid actions get an additional $-10^8$ penalty on the UCB bonus.
- **Grammar compression.** Example state `[C, A, pad, pad, pad, pad]` has only 4 valid actions of 48 after masking — a 12× reduction in per-step branching.

Masking is essentially a "grammar-validity oracle" analogous to legal-move generation in chess.

## 6. Coupling outer and inner

The two levels are coupled through the reward:

1. Agent plays the outer MDP via MCTS, producing a derivation.
2. The derivation is parsed into a program / rule.
3. The program is interpreted as a policy for the inner MDP.
4. The inner MDP is rolled out (possibly from a random start).
5. The cumulative inner reward becomes the outer terminal reward.

The outer reward is sparse (only at terminal). MCTS value bootstrapping handles credit assignment across deep derivations — this is why the neural value head matters even though all in-episode rewards are 0.

## 7. AlphaZero components

### Neural network $f_\theta$

Maps state to $(\mathbf p(s), v(s)) \in \Delta^{\lvert\mathcal A\rvert} \times \mathbb R$.

Architecture options (transformer is default):
- **Transformer encoder:** special `[CLS]` token prepended, padding mask for real-length states, 2–6 layers, 4 heads, $d_{\mathrm{model}} \in \{64, 128\}$. Stage 1 uses 2 layers $d=64$; the shared framework supports up to 6 layers $d=128$.
- **MLP / LSTM:** alternative baselines.

Output heads:
- **Policy:** Linear → logits → mask → softmax.
- **Value:** Linear → scalar $\hat v \in \mathbb R$.

### MCTS — PUCT selection

$$
a^\ast = \arg\max_a \left[ Q(s, a) + c_{\mathrm{PUCT}} \cdot p_\theta(a\!\mid\! s) \cdot \frac{\sqrt{N(s) + 1}}{N(s, a) + 1} \right].
$$

Prior penalty: actions with $p_\theta(a\!\mid\! s) < 10^{-8}$ get a $-10^8$ score — they are effectively pruned.

### MCTS simulation and backup

```
Select via PUCT until leaf.
If terminal: G ← 0. Else: expand, G ← v_θ(s).
Backup: G ← r(s, a) + γ·G; update N(s, a) and W(s, a) along the path.
```

### Training targets

- **Policy target:** $\pi_{\mathrm{MCTS}}(a\!\mid\! s) = N(s, a)^{1/\tau} / \sum_{a'} N(s, a')^{1/\tau}$ with temperature $\tau$ (default 1.0; Stages 1–2 tune to 0.5).
- **Value target:** $\hat V(s) = \sum_a \frac{N(s, a)}{\sum_{a'} N(s, a')} Q(s, a)$.
- **Action selection:** sample from $\pi_{\mathrm{MCTS}}$.
- **Tree reuse:** after committing to an action, that subtree becomes the new root.

### Loss and optimization

$$
 \mathcal L(\theta) = \sum_{(s, \pi, z)} \left[ (z - v_\theta(s))^2 + \mathrm{CE}(\pi, p_\theta(\cdot\!\mid\! s)) \right]
$$

plus an L2 regularizer. RMSProp, lr $10^{-4}$, gradient clip $\leq 0.5$, batch 32, replay buffer 1000, one epoch per episode.

### Training loop (abridged)

```
for episode in range(N):
  root ← init MCTS at s₀
  for t in range(T):
    run n_mcts simulations from root
    extract π_t and V̂_t
    store (s_t, π_t, V̂_t)
    sample a_t from π_t
    execute a_t, advance root to child
  after episode: reshuffle replay buffer, train 1 epoch
```

## 8. Worked walk-through — bitstring game

### Grammar and tokenization

$L_{\max} = 6$, $n_{\mathrm{sym}} = 9$, pad = 9, $n_{\mathrm{prod}} = 8$. Raw action space is $6 \times 8 = 48$ before masking. Five nonterminals + four terminals map to tokens 0–8.

### One derivation

```
Step 0  [S, 9, 9, 9, 9, 9]         action=(0, S→S₁) → [S₁, 9, 9, 9, 9, 9]
Step 1  [S₁, 9, 9, 9, 9, 9]         action=(0, S₁→CA) → [C, A, 9, 9, 9, 9]
Step 2  [C, A, 9, 9, 9, 9]          action=(0, C→bit0) → [bit0, A, 9, 9, 9, 9]
Step 3  [bit0, A, 9, 9, 9, 9]       action=(1, A→set1) → [bit0, set1, 9, 9, 9, 9]
Final   [5, 8, 9, 9, 9, 9] decoded as "bit0 set1" — the optimal single-rule policy.
```

### Parse tree and interpretation

The parse tree is $S \to S_1 \to (C, A) \to (\text{bit0}, \text{set1})$. Compiled as a one-rule reactive policy: "when you see a 0 bit, set a 1." That policy is then run on the inner OneMax game.

### Inner game execution

N=10, k=3 random 1s. Over 8 timesteps, the policy flips bits one at a time; cumulative inner reward = 7. This becomes the outer MDP's terminal reward.

### Training data from this episode

5 outer-level tuples $(s_t, \pi_t, \hat V_t)$, all tagged with terminal reward 7. After many episodes: the network learns to prefer $S \to S_1$, $C \to \text{bit0}$, $A \to \text{set1}$.

## 9. Other grammar games

**Symbolic regression (`GrammarSREnv`).** Grammar produces prefix-notation expressions; pipeline: grammar derivation → prefix → infix → sympy parse → fit coefficients. Reward: $\mathrm{sign}(R^2) \cdot \lvert R^2\rvert^p$. No inner sub-game — the grammar directly produces the output.

**Ones game (`GrammarOnesEnv`).** Grammar: $S \to S\,S \mid \texttt{'1'}$. Reward = count of `'1'` tokens. Optimal policy: expand $S \to S\,S$ until the buffer nearly fills, then fill with $S \to \texttt{'1'}$. Serves as a validation benchmark with a known optimal policy.

## 10. Trade-off: when does grammar help?

- **Wins** when the grammar reduces the search space ($B_{\mathrm{CFG}}^L \ll N^{N-k}$) without exploding branching. Stage 1's masked grammar is an archetype.
- **Loses** when program length $L$ is large, the grammar is ambiguous, or branching $B_{\mathrm{CFG}}$ is high. Stage 2 is the archetype here — removing masks explodes $L_=(G_2) = (2K)^{2K}$.

## Summary

1. **Grammar as game.** CFG derivation is modeled as an MDP; each production application is a transition.
2. **AlphaZero integration.** MCTS selects grammar derivations guided by a learned network; the network is trained on MCTS visit-count targets.
3. **Two-level architecture.** The outer grammar MDP produces a program; the inner sub-game evaluates it. This is the shared scaffold for Stages 1 and 2.
4. **Synthesis perspective.** Neural-guided grammar-constrained synthesis — a fifth program synthesis paradigm distinct from enumerative, constraint-based, stochastic, and fixed neural-prior methods.

## Why this matters for Stages 1–3

- **Stage 1** treats the Doors-grammar game as an outer MDP over the masked CFG $G_1$ and measures whether the network can learn the MCTS policy on it.
- **Stage 2** keeps the entire framework fixed and changes only the grammar ($G_1 \to G_2$) — an ablation of the masking mechanism described here.
- **Stage 3** generalizes the grammar from a right-linear surface grammar to a typed CFG with lifted variables. The outer-MDP machinery above is unchanged; only the grammar generator and the interpreter are replaced. See [../stage3/02_doors_adapter.md](../stage3/02_doors_adapter.md) for the impact map.

## References

- D. Silver et al., *Mastering the game of Go without human knowledge*, Nature 2017 (AlphaZero).
- R. Alur et al., *Syntax-Guided Synthesis*, FMCAD 2013.
- A. Solar-Lezama, *Program Synthesis by Sketching*, PhD thesis 2008.
- M. Balog et al., *DeepCoder: Learning to Write Programs*, ICLR 2017.
- K. Ellis et al., *DreamCoder*, PLDI 2021.
- A. Kalyan et al., *Neural-Guided Deductive Search*, ICLR 2018.
- L. Kocsis, C. Szepesvári, *Bandit Based Monte-Carlo Planning*, ECML 2006 (UCT).
- A. Novikov et al., *AlphaEvolve*, 2024.
- R. Sutton et al., *Between MDPs and Semi-MDPs: A framework for temporal abstraction*, AIJ 1999 (options).
