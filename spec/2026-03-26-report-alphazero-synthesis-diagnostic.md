# AlphaZero for Program Synthesis: Diagnostic Comparison Report

**Date:** 2026-03-26
**Purpose:** Systematic component-level comparison of standard AlphaZero (board games) vs. program synthesis variants, identifying WHERE and WHY the algorithm breaks down.
**Audience:** LLM assistants performing architectural diagnosis and researchers comparing approaches.
**Format:** Each section is self-contained. Tables use explicit contrast columns. Every component comparison ends with a verdict.

---

## 1. Executive Summary

The AlphaZero algorithm is structurally intact in this implementation — the MCTS, neural network, self-play loop, and training pipeline are correctly implemented. However, **every assumption that makes AlphaZero work in board games is violated by the program synthesis domain.** The result is that the learning loop contributes nothing: solvers are found by random exploration (MCTS noise + random network priors), not by learned guidance. The neural network never learns to prefer good partial programs over bad ones.

| # | AlphaZero Assumption | Board Games | Program Synthesis | Status |
|---|---|---|---|---|
| 1 | Intermediate states have predictable value | Board positions have intrinsic quality | Partial ASTs have no predictive value — quality depends entirely on unfilled holes | **VIOLATED** |
| 2 | Reward has high variance | Win/loss/draw gives ~50% variance | 99% of programs score identically (-0.075) | **VIOLATED** |
| 3 | Opponent creates natural curriculum | Weak opponents early, strong later | No opponent — problem difficulty is fixed | **VIOLATED** |
| 4 | Value head can learn from training data | Training targets span [0, 1] with meaningful spread | Training targets are ~constant (-0.075 for 99% of data) | **VIOLATED** |
| 5 | MCTS exploitation improves with learning | Q-value spread grows → exploitation kicks in | Q_min ≈ Q_max → UCB = pure exploration indefinitely | **VIOLATED** |

---

## 2. Algorithm Structure Overview

### 2.1 Standard AlphaZero Loop (Board Games)

```
for iteration in 1..N:
    1. SELF-PLAY: Play n_games using MCTS + neural_network
       - At each position: run n_simulations MCTS traversals
       - Select move proportional to visit counts
       - Collect (board_state, MCTS_policy, game_outcome)

    2. TRAIN: Update network on recent examples
       - Policy loss: CrossEntropy(network_policy, MCTS_policy)
       - Value loss: MSE(network_value, game_outcome)

    3. GATE: Pit new network vs old network
       - If win_rate >= threshold: keep new weights
       - Else: revert
```

**Key property:** Game outcomes ({-1, 0, +1}) create a natural training signal. As the network improves, MCTS produces stronger policies, which produce better training data — a virtuous cycle.

### 2.2 Program Synthesis Variant Loop

```
for iteration in 1..N:
    1. SELF-PLAY: Play n_games of the "derivation game"
       - State = partial AST with holes (or partial rule sequence)
       - Action = grammar production (or rule selection)
       - At each step: run n_simulations MCTS traversals
       - Select production proportional to visit counts
       - Terminal reward = leaf_evaluator(completed_program)
       - Collect (partial_AST_obs, MCTS_policy, terminal_reward)    # ← all steps get SAME terminal reward

    2. TRAIN: Update network on recent examples
       - Policy loss: CrossEntropy(network_policy, MCTS_policy)
       - Value loss: MSE(network_value, terminal_reward)            # ← terminal_reward ≈ -0.075 for 99% of data

    3. GATE: Pit new network vs old network on fresh games
       - Both produce ~identical program distributions → tie → no selection pressure
```

**Key difference:** Terminal reward is sparse (0 at intermediate steps, scalar at end) and near-constant across programs. The virtuous cycle never starts.

### 2.3 Structural Diff Table

| Component | Board Games (Go/Chess) | Program Synthesis (Derivation) | Identical? |
|---|---|---|---|
| State | Board position (grid/bitboard) | Partial AST encoded as (type_id, param) vector | Different representation, same role |
| Action | Place stone / move piece | Apply grammar production to leftmost hole | Same mechanism, different semantics |
| Reward timing | Terminal only ({-1, 0, +1}) | Terminal only (continuous scalar) | **Same** (both sparse) |
| Reward variance | ~50% (win vs loss) | ~1% (99% score identically) | **Critical difference** |
| Training target | game_outcome ∈ {-1, 0, +1} | leaf_eval ∈ [-0.075, +1.045] but 99% ≈ -0.075 | Same structure, collapsed signal |
| Value prediction task | "Who is winning from this position?" | "How good will the final program be from this partial AST?" | Same question, fundamentally harder |
| Opponent | Yes (self-play creates curriculum) | No (single-player construction) | **Missing** |
| Gate mechanism | Win rate of new vs old | Win rate of new vs old (but always ~0.50) | Same code, vacuous in synthesis |
| Data collection | Parallel self-play games | Parallel derivation games | **Identical** |

---

## 3. Component-by-Component Comparison

### 3.1 State Representation

**Board games:** A board position encodes the full game state — piece locations, control of territory, material balance. A human expert can look at a Go position and estimate who is winning. The state carries intrinsic value information.

**Derivation game:** A partial AST with holes. Example at step 3 of a D=3 derivation:
```
Ite(ConditionHole(5), Flip(6), ProgramHole(26))
```
Encoded as a fixed-length vector: `[5, 5, 1, 6, 7, 26, 0, 0, ..., 0]` (68 floats for budget=34).

The partial AST tells you *what has been decided so far* but provides almost no information about *what will be decided for the remaining holes*. The terminal reward depends entirely on how the 26-budget ProgramHole and 5-budget ConditionHole are filled — which is exactly what MCTS is searching for.

**Surface game:** A partial rule sequence. Example at step 2 of a D=3 surface game:
```
(PickRule(0), MoveRule(0))
```
Encoded as: `[1, 0, 2, 0, 0, 0, 0, 0, 0, 0]` (10 floats for D=3).

Simpler than the derivation state, but still provides limited value prediction signal because the terminal reward depends on the ordering of remaining rules.

| Property | Board Games | Derivation Game | Surface Game |
|---|---|---|---|
| State content | Full game position | Partial AST + holes | Partial rule sequence |
| Encoding size | Fixed (e.g., 19x19x17 for Go) | 2 * budget (68 for D=3) | 2 * (2K+1) (10 for D=3) |
| Intrinsic value signal | High (position quality is estimable) | Near-zero (quality depends on unfilled holes) | Low (quality depends on rule ordering) |
| State space size | ~10^170 (Go) | Exponential in budget | (2K)!/2^K (6 for D=3) |

**Verdict:** Partial ASTs carry almost no predictive information about terminal quality. The value prediction task is information-theoretically much harder than in board games because the state does not encode the decisions that determine the outcome.

---

### 3.2 Action Space

**Board games:** Legal moves depend on position. Go has ~250 legal moves on average but narrows as the board fills. Chess has ~30 average legal moves. The branching factor is moderate and varies with game phase.

**Synthesis variants:**

| Variant | Actions per step | Example (D=3) | Position-dependent narrowing? |
|---|---|---|---|
| Derivation (flat) | ~200-300 productions | 279 actions | Slight (budget decreases → fewer productions) |
| Derivation (factored+macros) | ~15-39 per phase | 39 structure + 11 parameter | Moderate (phases alternate) |
| Surface | 2K+1 rules | 5 actions | Yes (placed rules removed from pool) |

**Key difference:** Board games have natural narrowing — as the game progresses, fewer moves are available. The derivation game's branching stays high throughout because each hole expansion creates new holes with their own production sets. The surface game narrows naturally (from 2K+1 down to 1).

**Verdict:** Action space breadth is comparable to Go, but board games benefit from position-dependent narrowing and the fact that most moves are "reasonable" (only a few are clearly bad). In the derivation game, the vast majority of productions lead to useless programs, and there is no signal to distinguish them.

---

### 3.3 Episode Structure

| Property | Board Games (Go) | Derivation (flat) | Factored+Macros | Surface |
|---|---|---|---|---|
| Episode length | ~200 moves | ~15-100 steps | ~10-40 steps | Fixed 2K+1 |
| Length variance | High (depends on play) | High (depends on budget distribution) | Moderate | **Zero** |
| Dead ends | No (always legal moves) | Yes (budget exhaustion) | Yes (budget exhaustion) | **No** |
| Opponent interaction | Yes (alternating) | No (single-player) | No | No |
| Phase structure | Opening/midgame/endgame | None (monotonic growth) | Structure/parameter alternation | None (sequence append) |
| Terminal condition | Two passes / resignation | All holes filled | All holes filled | GoalRule placed |

**Board games** have rich episode dynamics: opening theory, middlegame tactics, endgame technique. Different phases require different strategies, creating natural structure for the value function to learn.

**Derivation games** are monotonic — the AST only grows. There is no strategic phase structure. Every step is "fill the next hole."

**Surface games** are the simplest: append one rule to the sequence, exactly 2K+1 times.

**Verdict:** Board game episodes have rich temporal structure (phases, opponent interaction, narrowing endgame) that provides a natural curriculum for value learning. Synthesis episodes are monotonic and structureless — the value function has nothing to "latch onto" across different game phases.

---

### 3.4 Reward Signal

This is the **primary failure point**.

**Board games:** Terminal reward is {-1, 0, +1} (loss/draw/win). In any reasonably balanced game, roughly 50% of outcomes are wins and 50% are losses. The value head has clear, high-variance training signal.

**Program synthesis:** Terminal reward is `leaf_evaluator(program)`, a continuous scalar. The distribution is:

| Program quality (D=3) | Env reward | Weighted metric (alpha=0.7) | Frequency |
|---|---|---|---|
| Garbage (NOOP/stuck) | -0.25 | -0.075 | ~91% |
| Moves but no picks | ~-0.25 | ~-0.075 | ~5% |
| Picks 1 key | -0.15 | -0.045 | ~3% |
| Picks 2 keys | -0.05 | -0.015 | ~0.9% |
| **Solves** (2 keys + goal) | +1.15 | +1.045 | **<0.001%** |

**Critical numbers:**
- Non-solving reward range: 0.060 units (from -0.075 to -0.015)
- Cliff to solving: 1.060 units
- Cliff-to-range ratio: **18x**
- Programs at modal reward (-0.075): **91-99%**

The value head is asked to distinguish between programs scoring -0.075, -0.045, and -0.015 — a range of 0.060 units — while 91-99% of its training data is at -0.075. This is like asking a classifier to learn from a dataset where 99% of labels are identical.

**Reward compression analysis:**
```
Layer 1 — Environment:
  unlock_bonus = 0.10
  total_step_penalty = 25 * 0.01 = 0.25
  Signal-to-noise ratio: 0.10 / 0.25 = 0.40

Layer 2 — Weighted metric (alpha=0.7):
  When solve_rate=0: weighted = 0.3 * avg_reward
  Compression factor: 3.3x

Combined non-solving range: 0.060 / 3.3 ≈ 0.018 units after weighting
```

**Verdict: CRITICAL FAILURE POINT.** The reward landscape is a desert — 99% of programs score identically. The value head cannot learn because there is nothing to learn from near-constant training targets. This is the root cause of all downstream failures. Board games avoid this because win/loss provides guaranteed 50%+ variance.

---

### 3.5 MCTS Search

The MCTS implementation is identical to standard AlphaZero. The UCB formula:

```
UCB(a) = Q_norm(a) + c_exploration * P(a) * sqrt(N) / (1 + n(a))

where:
  Q_norm(a) = (Q(a) - Q_min) / (Q_max - Q_min)    # min-max normalization
  P(a) = neural network policy for action a
  N = total visits to current node
  n(a) = visits to action a
  c_exploration = 1.5
```

**How the same formula behaves differently:**

| UCB Component | Board Games | Program Synthesis |
|---|---|---|
| Q_norm(a) | Meaningful spread [0, 1] — different actions lead to different outcomes | Q_min ≈ Q_max → Q_norm = 0.5 for all actions (degenerate) |
| P(a) | Learned prior that improves over training | Random-initialized prior that never improves |
| Exploration term | Balanced with exploitation | **Dominates** (Q term is constant 0.5) |
| Net effect | Exploitation + guided exploration | Pure noise-driven exploration |

**When Q_min == Q_max** (which happens when all leaf evaluations return ~-0.075):
```
Q_norm(a) = 0.5 for all a
UCB(a) = 0.5 + 1.5 * P(a) * sqrt(N) / (1 + n(a))
```
The Q term contributes nothing. Visit counts are driven entirely by the policy prior P(a) and Dirichlet noise. MCTS becomes a fancy way to sample from a noise-perturbed uniform distribution.

**Dirichlet noise:**
```
P_noisy = (1 - epsilon) * P_network + epsilon * Dir(alpha)
epsilon = 0.40, alpha = 0.25
```
In board games, this adds mild perturbation to a learned prior. In synthesis, the network prior is itself noise, so Dirichlet noise adds noise to noise.

**Rollout evaluation** (optional, rollout_n=4, blend=0.3):
In board games, random rollouts from a position give crude but informative value estimates. In synthesis, random rollouts complete the AST randomly, producing random programs that score ~-0.075, providing no additional signal.

**Backup rules:**
The implementation offers mean, max, topk, and softmax backup. Current setting: `backup_rule="max"`. Max backup is an attempt to address the reward desert — by keeping the best value ever seen through an action, it should propagate rare good programs more effectively. However, when rare good programs are never found (1 in 100K+), max backup has nothing to propagate.

**Verdict:** The MCTS machinery is correctly implemented and would work perfectly in a domain with meaningful reward signal. In the synthesis domain, Q-normalization collapses to a constant, making MCTS degenerate to noise-driven exploration. This is not a bug — it is the correct behavior of MCTS when reward variance is near-zero.

---

### 3.6 Value Head

**Board games:** The value head predicts "who is winning from this position." Training targets span [-1, +1] with roughly equal mass at each extreme. As training progresses, the value head learns to evaluate positions with increasing accuracy, which improves MCTS exploitation, which produces better training data.

**Program synthesis:** The value head predicts "how good will the completed program be from this partial AST." Training targets:
```
99% of targets ≈ -0.075  (garbage programs)
<1% of targets in [-0.045, -0.015]  (partial progress)
<0.001% of targets ≈ +1.045  (solvers)
```

With discount=1.0, **every step in an episode gets the same target** (the terminal reward). A 15-step derivation game episode produces 15 training examples, all with the same value target.

**What the value head actually learns:**
```
V(s) ≈ -0.075 for all states s

Value loss ≈ 0.0001 (noise floor)
```
This is the optimal predictor given the training distribution — a constant function that minimizes MSE on near-constant targets. The value head has learned the correct answer: "from any partial AST, the expected program quality is -0.075." This is statistically accurate and completely useless for guiding search.

**Why the prediction task is impossible:**
Consider a partial AST at step 3: `Ite(And(Not(IsZero(1)), Not(IsZero(9))), Flip(6), ProgramHole(26))`. This is the beginning of a correct PickRule(0) — a "good" start. But whether the final program solves depends entirely on the remaining 26-budget ProgramHole, which could be filled in billions of ways. The value head would need to learn: "partial ASTs that start with PickRule-like patterns tend to produce better programs." But this requires seeing a mix of good and bad programs during training — which doesn't happen when 99% of programs score identically.

**Verdict: CRITICAL FAILURE POINT.** The value head learns a constant predictor because training targets are near-constant. This is not a network architecture problem — no architecture can learn a non-constant function from constant data. The value head collapse causes MCTS to degenerate (Section 3.5), which causes policy head collapse (Section 3.7).

---

### 3.7 Policy Head

**Board games:** The policy head predicts which moves to explore first. Training targets are MCTS visit-count distributions, which reflect real strategic preferences (exploitation of high-value moves). Over training iterations, the policy head learns to predict which moves MCTS would favor, bootstrapping future searches.

**Program synthesis:** The policy head's training targets come from MCTS visit counts. But when MCTS is degenerate (Section 3.5), visit counts are driven by Dirichlet noise, not by exploitation:

```
Visit counts ∝ P_noisy(a) = (0.6) * P_network(a) + (0.4) * Dir(0.25)
```

Since P_network is itself noise (untrained or trained on noise), the visit-count targets are noisy distributions over legal actions. Training the policy to reproduce these targets teaches the network to output noise.

**The vicious cycle:**
```
Noisy policy → Noisy MCTS (no exploitation) → Noisy visit counts → Train on noise → Noisy policy
```

Board games break this cycle at the "Noisy MCTS" step: even a noisy policy, combined with exploitation of Q-values, produces visit counts that favor good moves. In synthesis, there is no Q-value signal to exploit.

**Verdict:** The policy head is a downstream casualty of value head collapse. It learns to reproduce noise because its training targets are noise. This is not independently fixable — fixing the value signal would automatically fix the policy.

---

### 3.8 Training Loop and Acceptance Gate

**The training loop** collects `n_games_per_train=30` games per iteration, keeps the last `n_past_iterations=20` iterations in a sliding buffer, and trains the network for 5 epochs.

**Board games:** Early iterations produce diverse game outcomes (wins and losses from different positions). As the network improves, it plays differently, generating new positions and outcomes. The training distribution evolves over time — this is the curriculum.

**Program synthesis:** Every iteration produces the same distribution: ~99% garbage programs scoring -0.075. The training distribution does not evolve because the network never learns anything useful. Iterations 1 and 30 produce statistically identical data.

**The acceptance gate** (gated trainer):
```
Play n_eval games with new network vs old network
If win_rate >= 0.40: accept new weights
```

In board games, a better network produces genuinely better programs/moves, so the gate provides selection pressure.

In synthesis, both networks produce identical program distributions (random programs scoring -0.075). Win rate is always ~0.50 (coin flip). The gate accepts or rejects at random, with no selection pressure.

**Verdict:** The training loop and gate are structurally correct but vacuous in the synthesis domain. They cannot create a curriculum when the training data distribution is stationary, and the gate cannot select for improvement when there is no improvement to select for.

---

### 3.9 Self-Play and Curriculum

**Board games (the virtuous cycle):**
```
Iteration 1: Random play → diverse outcomes → network learns basic patterns
Iteration 10: Better play → more sophisticated positions → network learns tactics
Iteration 100: Strong play → subtle endgame positions → network learns strategy
```

Each iteration's self-play data is qualitatively different from the previous iteration. The opponent (self) grows stronger, creating a natural curriculum from simple to complex positions.

**Program synthesis (the vicious cycle):**
```
Iteration 1: Random programs → 99% score -0.075 → network learns V(s) ≈ -0.075
Iteration 10: Still random programs → still 99% at -0.075 → network unchanged
Iteration 100: Still random programs → still 99% at -0.075 → network unchanged
```

There is no opponent to create curriculum. The problem difficulty is fixed at D=3 from iteration 1. The network cannot improve because the training data never changes, and the training data never changes because the network cannot improve.

**The missing ingredient:** Board games have a built-in curriculum via opponent strength. Program synthesis has no analogue. The system never sees "easy" programs first and "hard" programs later — it sees the full D=3 difficulty from the start.

**Verdict: CRITICAL MISSING INGREDIENT.** The absence of curriculum means the system has no path from "bad" to "good." In board games, the system bootstraps from random play through the virtuous cycle. In synthesis, the system is stuck at random forever because there is no curriculum mechanism to break the vicious cycle.

---

## 4. Game Variant Comparison Matrix

| Dimension | Derivation (flat) | Factored+Macros | Surface |
|---|---|---|---|
| **State type** | Partial AST with holes | Partial AST with holes | Flat rule sequence |
| **Action type** | Grammar production | Structure template or parameter | Semantic rule (PickRule/MoveRule/GoalRule) |
| **Branching factor (D=3)** | ~279 | ~39 structure + ~11 parameter | 5 |
| **Episode length (D=3)** | ~15 (variable) | ~10 (variable) | 5 (fixed) |
| **Dead ends** | Yes | Yes | No |
| **Reward at intermediate steps** | 0 | 0 | 0 |
| **Terminal reward source** | leaf_evaluator(program) | leaf_evaluator(program) | leaf_evaluator(compiled_program) |
| **Programs reachable (D=3)** | ~52 billion | ~52 billion (same space) | 6 |
| **Solver frequency (D=3)** | < 1 in 5M | ~1 in 100K-200K | 3 out of 6 (50%) |
| **D=2 solved?** | Yes (iter 7-8) | Yes (iter 1) | Yes (trivially) |
| **D=3 solved?** | **Never** (0/13 runs) | **Sometimes** (~60-70% of runs) | Yes (easily) |
| **Does learning contribute?** | No | No (solvers found at iter 1) | No (but search space is small enough) |
| **Value head useful?** | No (constant predictor) | No (constant predictor) | N/A (search space too small to need it) |
| **Domain knowledge encoded** | None (generic grammar) | Macro templates (PickRule/MoveRule shape) | Full rule structure (conditions + actions paired) |
| **Search problem type** | Tree construction | Factored tree construction | Permutation selection |
| **Observation encoding** | (type_id, param) pairs, 2*budget | (type_id, param) + phase, 2*budget+2 | (token_id, param) pairs, 2*(2K+1) |

**Key insight from this table:** Success correlates perfectly with domain knowledge — the more structure encoded in the grammar, the smaller the search space, and the more likely the system is to find a solver. The AlphaZero learning loop contributes nothing in any variant. Success is determined entirely by whether random exploration can find a solver within the search space.

---

## 5. Failure Chain Analysis

### 5.1 The Causal Failure Chain (Program Synthesis)

```
Step 1: REWARD DESERT
  99% of programs score -0.075
  Reward range for non-solvers: 0.060 units
      │
      ▼
Step 2: VALUE HEAD COLLAPSES
  Training targets ≈ constant → V(s) ≈ -0.075 for all s
  Value loss → noise floor (~0.0001)
      │
      ▼
Step 3: Q-VALUES BECOME UNIFORM
  Q_min ≈ Q_max → Q_normalized = 0.5 for all actions
  No action is preferred by exploitation
      │
      ▼
Step 4: MCTS DEGENERATES TO RANDOM SEARCH
  UCB = 0.5 + exploration_term (no exploitation)
  Visit counts driven by Dirichlet noise + prior noise
      │
      ▼
Step 5: POLICY TARGETS ARE NOISE
  Visit-count distributions reflect noise, not strategy
  Training the policy on noise produces noise
      │
      ▼
Step 6: NO IMPROVEMENT ACROSS ITERATIONS
  Network weights change but behavior doesn't improve
  Training distribution remains stationary
      │
      ▼
Step 7: GATE IS VACUOUS
  New network ≈ old network → win_rate ≈ 0.50
  No selection pressure
      │
      ▼
  ┌───────────────────┐
  │ RETURN TO STEP 1  │ ← vicious cycle repeats indefinitely
  └───────────────────┘
```

### 5.2 Why Board Games Break This Chain

| Chain Step | Board Games (why it doesn't fail) |
|---|---|
| **Step 1** (Reward desert) | **Broken.** Win/loss gives ~50% variance. Games have clear outcomes. |
| **Step 2** (Value collapse) | **Broken.** Targets span [-1, +1]. Value head has meaningful gradients. |
| **Step 3** (Uniform Q) | **Broken.** Different moves lead to genuinely different outcomes. Q-values spread naturally. |
| **Step 4** (Degenerate MCTS) | **Broken.** Exploitation drives MCTS toward winning moves. |
| **Step 5** (Noise policy) | **Broken.** Visit counts reflect real preferences (exploitation of Q). |
| **Step 6** (No improvement) | **Broken.** Better policy → different games → new training data → improvement. |
| **Step 7** (Vacuous gate) | **Broken.** Better network genuinely wins more → gate selects for improvement. |

Board games break the chain at **Step 1** (the root), which prevents all downstream failures. The key property is **reward variance** — in any competitive game, approximately half the outcomes are wins and half are losses, providing the signal gradient that powers the entire learning loop.

### 5.3 Where Each Game Variant Breaks in the Chain

| Failure Step | Derivation (flat) | Factored+Macros | Surface |
|---|---|---|---|
| 1. Reward desert | **Severe.** 99% identical scores. | **Severe.** 99% identical scores. | **Mild.** 50% of policies solve (D=3). |
| 2. Value collapse | **Yes.** Constant predictor. | **Yes.** Constant predictor. | **Mild.** More variance but still sparse. |
| 3. Uniform Q | **Yes.** Q_min ≈ Q_max. | **Yes.** Q_min ≈ Q_max. | **Partially.** Some Q spread possible. |
| 4. Degenerate MCTS | **Yes.** Pure noise. | **Yes.** Pure noise. | **Partially.** Some exploitation possible. |
| 5. Noise policy | **Yes.** | **Yes.** | **Mild.** |
| 6. No improvement | **Yes.** | **Yes** (solvers found by chance). | **N/A** (space small enough). |
| 7. Vacuous gate | **Yes.** | **Yes.** | **N/A.** |

**The surface game's advantage is not that it fixes the algorithm — it shrinks the search space until the algorithm's failures don't matter.** With only 6 possible policies (D=3) and 50% solving, random exploration finds a solver almost immediately. The AlphaZero learning loop is still vacuous, but the problem is easy enough that brute force suffices.

---

## 6. Direct RL Comparison

### 6.1 Why Direct RL Works

In **direct play**, the neural network maps Doors observations directly to actions. There is no program synthesis layer.

```
State: Doors observation vector [at_loc, room_status, key_status]
Action: MOVE_TO(loc) / PICK(key) / NOOP
Reward: -0.01 per step, +0.10 per unlock, +1.00 at goal
```

**Result:** Direct RL (AlphaZero) on Doors D=10 solves in **5 iterations**. The same algorithm that fails at D=3 synthesis solves D=10 direct play trivially.

**Why it works:**

| Property | Direct RL | Synthesis |
|---|---|---|
| Reward frequency | Every step (-0.01, +0.10, +1.00) | Only at terminal (single scalar) |
| Reward variance | High (different trajectories score differently) | Near-zero (99% identical scores) |
| Value prediction | "From this position, what is my expected return?" → tractable | "From this partial AST, what will the program score?" → intractable |
| State information | Full game state (location, keys, doors) | Partial AST (no game state visible) |
| Curriculum | MCTS explores different trajectories with different outcomes | All programs score the same |

### 6.2 The Synthesis Layer as Information Bottleneck

The synthesis layer sits between the search algorithm and the environment, collapsing the rich per-step reward signal into a single terminal scalar:

```
Environment:  step → (-0.01, obs₁) → step → (+0.10, obs₂) → step → (-0.01, obs₃) → ... → (+1.00, obs_N)
                ↕                        ↕                        ↕                            ↕
              25 reward events with graded feedback at every step

Synthesis:    production → (0, partial_AST₁) → production → (0, partial_AST₂) → ... → (leaf_eval, done)
                ↕                                  ↕                                       ↕
              N-1 zero-reward steps + 1 terminal scalar
```

**Information loss:**
- Environment provides ~25 reward events per episode (step penalties, unlock bonuses, goal reward)
- Synthesis provides 1 reward event per episode (terminal leaf evaluation)
- Information compression ratio: ~25:1

The leaf evaluator runs the completed program on the environment and observes all 25 reward events — but then compresses them into a single scalar (`weighted = 0.7 * solve_rate + 0.3 * avg_reward`). The rich per-step feedback that makes direct RL work is lost.

**Verdict:** The synthesis layer is a massive information bottleneck. It takes a domain where AlphaZero succeeds (direct RL on Doors) and makes it one where AlphaZero fails, by collapsing the reward signal from 25 informative events into 1 near-constant scalar.

---

## 7. Quantitative Summary Tables

### 7.1 Master Comparison Table

| Metric | Board Games (Go) | Derivation (flat, D=3) | Factored+Macros (D=3) | Surface (D=3) | Direct RL (D=10) |
|---|---|---|---|---|---|
| State space | ~10^170 | ~52 × 10^9 programs | ~52 × 10^9 programs | 6 policies | ~10^8 trajectories |
| Branching factor | ~250 | ~279 | ~39 | 5 | 9 |
| Episode length | ~200 | ~15 | ~10 | 5 | ~25 |
| Reward values | {-1, 0, +1} | [-0.075, +1.045] | [-0.075, +1.045] | [-0.075, +1.045] | [-0.25, +1.15] |
| % at modal reward | ~50% | ~99% | ~99% | ~50% | ~30% |
| Reward range (non-optimal) | 2.0 | 0.060 | 0.060 | 0.060 | 1.40 |
| Solver frequency | N/A (win/loss) | <1 in 5M | ~1 in 100K | 3 in 6 | N/A |
| Value head useful? | Yes | No | No | Marginally | Yes |
| Learning contributes? | Yes (core mechanism) | No | No | No (unnecessary) | Yes |
| D=3 or equiv. solved? | N/A | Never | Sometimes | Always | D=10 solved |

### 7.2 Assumption Violation Severity Matrix

| AlphaZero Assumption | Board Games | Derivation | Factored+Macros | Surface | Direct RL |
|---|---|---|---|---|---|
| Intermediate value predictable | MET | **SEVERE** | **SEVERE** | MILD | MET |
| Reward has high variance | MET | **SEVERE** | **SEVERE** | MILD | MET |
| Opponent creates curriculum | MET | **SEVERE** | **SEVERE** | **SEVERE** | MILD* |
| Value head can learn | MET | **SEVERE** | **SEVERE** | MILD | MET |
| MCTS exploitation works | MET | **SEVERE** | **SEVERE** | MILD | MET |
| Overall | **5/5 MET** | **0/5 MET** | **0/5 MET** | **2/5 MET** | **4/5 MET** |

*Direct RL has no opponent but has rich per-step rewards that create implicit curriculum through diverse trajectories.

---

## 8. Implications for Fixes

### 8.1 Fixes That Address Root Causes

These fixes target the causal chain at its source (Step 1: reward desert).

| Fix | What it does | Which chain link it breaks | Expected impact |
|---|---|---|---|
| **Richer intermediate rewards** | Add per-step rewards during derivation (e.g., reward for placing a PickRule-like pattern) | Breaks Step 1 (reward desert) | Could enable value learning, but risks reward hacking |
| **Curriculum over D** | Train on D=1 → D=2 → D=3, transferring weights | Breaks Step 6 (no improvement) by providing progressive difficulty | Network learns structural patterns on easy problems first |
| **Per-key progress metric** | `reward = keys_picked / total_keys + 0.1 * avg_reward` | Breaks Step 1 (reward desert) by widening non-solving reward range | Uniform spacing (0, 0.5, 1.0 for D=3) instead of (-0.075, -0.045, -0.015) |
| **Alternative search algorithm** | Replace MCTS with beam search, evolutionary search, or LLM-guided sampling | Bypasses Steps 2-5 entirely | Avoids MCTS degeneration; works better with sparse rewards |
| **Grammar redesign** | Increase solver density by encoding more domain knowledge (like surface DSL) | Breaks Step 1 by increasing % of programs that score distinctly | Surface grammar (50% solve rate) vs. flat grammar (<0.001%) |

### 8.2 Fixes That Address Symptoms Only

These fixes do not target the root cause and are unlikely to help.

| Fix | Why it won't help |
|---|---|
| **Hyperparameter tuning** (c_exploration, dirichlet, etc.) | MCTS behavior is dominated by flat Q-values, not hyperparameters. Tuning c_exploration changes exploration speed but not the absence of exploitation signal. |
| **Network architecture changes** (more layers, different model) | The problem is not representation capacity — it is training data quality. No architecture can learn a non-constant function from constant data. |
| **More compute** (more games, more iterations, more simulations) | Random exploration with more compute explores more programs, but the probability of finding a solver scales linearly while the search space scales exponentially. One run explored 688K programs over 100 iterations without solving. |
| **Different backup rule** (mean vs max vs softmax) | Max backup helps propagate rare good values, but when good values are never encountered (1 in 100K+), there is nothing to propagate. |
| **Larger replay buffer** | More data from the same degenerate distribution does not improve learning. |

### 8.3 Ranking of Root-Cause Fixes

| Priority | Fix | Rationale |
|---|---|---|
| 1 | **Grammar redesign (increase solver density)** | Proven to work: surface grammar solves D=3 trivially. The question is whether intermediate grammars can balance expressivity with solver density. |
| 2 | **Curriculum over D** | Low-cost, high-potential: D=2 is reliably solved, so D=2 → D=3 transfer is plausible. Network may learn "PickRule patterns are good" from D=2. |
| 3 | **Per-key progress metric** | Easy to implement (already in code). Widens reward range from 0.060 to 1.0+. Risk: may not provide enough gradient for value head. |
| 4 | **Alternative search** | Highest potential but highest cost. Abandoning MCTS for beam search or evolutionary methods could fundamentally change the approach. |
| 5 | **Richer intermediate rewards** | Hardest to design correctly. What intermediate reward for partial ASTs? Risk of reward hacking (ASTs that look like solvers but aren't). |

---

## Appendix A: Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| **Problem (D=3)** | rooms (D) | 3 |
| | locations (M) | 6 |
| | keys (K) | 2 |
| | n_sites | 11 |
| | n_actions | 9 |
| | budget (L) | 34 |
| | horizon | 25 |
| **MCTS** | n_simulations | 80 |
| | c_exploration | 1.5 |
| | dirichlet_alpha | 0.25 |
| | dirichlet_epsilon | 0.40 |
| | rollout_n | 4 |
| | rollout_mode | "max" |
| | rollout_blend | 0.3 |
| | rollout_budget | 200 |
| | backup_rule | "max" |
| **Network** | architecture | Transformer encoder |
| | d_model | 64 |
| | n_heads | 4 |
| | n_layers | 2 |
| | lr | 3e-4 |
| | epochs | 5 |
| | batch_size | 32 |
| | policy_weight | 2.0 |
| **Training** | n_games_per_train | 30 |
| | n_past_iterations | 20 |
| | n_procs | 8 |
| | reward_discount | 1.0 |
| **Evaluation** | n_games | 20 |
| | accept_threshold | 0.40 |
| | eval_temperature | 0.05 |

## Appendix B: Code Pointers

| Component | File |
|---|---|
| MCTS (UCB, backup, noise, Q-norm) | `src/alphazeropp/core/mcts.py` |
| Agent (self-play, data collection) | `src/alphazeropp/core/agent.py` |
| Game interface | `src/alphazeropp/core/game.py` |
| Policy-value network | `src/alphazeropp/core/policy_value_net.py` |
| Training loop | `src/alphazeropp/training/trainer.py` |
| Acceptance gate | `src/alphazeropp/training/gated_trainer.py` |
| Evaluator | `src/alphazeropp/training/evaluator.py` |
| Derivation game | `src/alphazeropp/synthesis/derivation_game.py` |
| Factored derivation game | `src/alphazeropp/synthesis/factored_derivation_game.py` |
| AST nodes | `src/alphazeropp/synthesis/ast_nodes.py` |
| Budget grammar | `src/alphazeropp/synthesis/budget_grammar.py` |
| Leaf evaluator | `src/alphazeropp/synthesis/leaf_evaluator.py` |
| Surface derivation game | `src/alphazeropp/instances/doors/dsl/surface_derivation_game.py` |
| Doors macros | `src/alphazeropp/instances/doors/dsl/doors_macros.py` |
| Surface compiler | `src/alphazeropp/instances/doors/dsl/surface_compiler.py` |
| Doors environment | `src/alphazeropp/instances/doors/doors_pddl_lite.py` |
| Direct play game | `src/alphazeropp/instances/doors/game.py` |
| Config classes | `src/alphazeropp/core/config.py` |
| Derivation configs | `src/alphazeropp/instances/doors/dsl/derivation_config.py` |
| Surface config | `src/alphazeropp/instances/doors/dsl/surface_derivation_config.py` |

## Appendix C: Glossary

| Term | Definition |
|---|---|
| **AlphaZero** | RL algorithm combining MCTS with a neural network (policy + value heads), trained via self-play |
| **MCTS** | Monte Carlo Tree Search — lookahead search that builds a tree of states, selecting actions via UCB |
| **UCB** | Upper Confidence Bound — formula balancing exploitation (high Q) and exploration (low visit count) |
| **Derivation game** | Program synthesis cast as a game: state = partial AST, action = grammar production, reward = program quality |
| **Surface game** | High-level variant: state = partial rule sequence, action = semantic rule, reward = compiled program quality |
| **Factored game** | Derivation variant that splits productions into structure (template) and parameter phases |
| **Macro production** | Domain-specific production that expands a hole into a multi-node template (e.g., PickRule = 7 AST nodes) |
| **Leaf evaluator** | Runs a completed program on frozen environment states and returns a scalar quality metric |
| **Reward desert** | Condition where 99%+ of programs score identically, providing no gradient for learning |
| **Value head** | Neural network output that predicts expected terminal reward from current state |
| **Policy head** | Neural network output that predicts probability distribution over actions |
| **Q-normalization** | Scaling Q-values to [0, 1] using global min/max from the current search tree |
| **Dirichlet noise** | Random noise added to root policy prior to encourage exploration: P_noisy = (1-eps)*P + eps*Dir(alpha) |
| **Backup rule** | How MCTS aggregates backed-up values at edges: mean, max, topk, or softmax |
| **Acceptance gate** | Mechanism that keeps new network weights only if they beat the old network in head-to-head play |
| **Budget** | Maximum number of AST nodes allowed in a derivation (controls program space size) |
| **PickRule(k)** | Surface/macro rule: "if key k is available and agent is at key k's location, pick it up" (7 AST nodes) |
| **MoveRule(k)** | Surface/macro rule: "if room that key k unlocks is locked, move to key k's location" (3 AST nodes) |
| **GoalRule** | Surface rule: "move to goal location" (2 AST nodes, always last in sequence) |
| **Doors** | Navigation environment: D rooms, K=D-1 keys, agent must collect keys and reach goal |
