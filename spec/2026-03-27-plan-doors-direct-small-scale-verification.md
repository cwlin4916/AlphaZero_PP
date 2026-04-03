# Doors Direct Play: Small-Scale AlphaZero Verification Plan

**Date:** 2026-03-27
**Goal:** Systematically verify AlphaZero works on Doors D=2..6, identify minimum compute budgets, and build confidence before scaling to D=14.

---

## 1. Problem Dimensions

| D | obs | actions (K) | opt_steps | horizon (H) | opt_reward | net params |
|---|-----|-------------|-----------|--------------|------------|------------|
| 2 |  9  |      8      |     3     |      15      |   +1.07    |   1,225    |
| 3 | 14  |     12      |     5     |      25      |   +1.15    |   1,805    |
| 4 | 19  |     16      |     7     |      35      |   +1.23    |   2,829    |
| 5 | 24  |     20      |     9     |      45      |   +1.31    |   4,437    |
| 6 | 29  |     24      |    11     |      55      |   +1.39    |   6,405    |

**State space upper bound:** M × 2^D × 2^(D-1) where M = 3D. Ranges from 48 (D=2) to 36,864 (D=6).

**Prior empirical results:** D=4 and D=6 both solve by iteration 2 with sims=60, games=30. This confirms the algorithm works for comfortable budgets.

---

## 2. Mathematical Framework for Hyperparameter Choices

### 2.1 MCTS Simulation Budget: Why sims/action ratio matters

The MCTS UCB selection rule (from `mcts.py:422`) is:

```
UCB(s,a) = Q_norm(s,a) + c · π(a) · √N_total / (1 + N(s,a))
```

where c=1.5, π(a) is the neural network prior, N_total is total root visits, and N(s,a) is per-action visits.

**With an untrained network** (uniform prior π(a) = 1/K for all K actions), the exploration term dominates and drives roughly uniform visitation: N(s,a) ≈ N/K. The resulting visit-count policy π_MCTS(a) = N(s,a)/N becomes the training target for the policy head.

**Two conditions must hold for π_MCTS to be a useful training signal:**

**(A) Tree depth must cover the reward horizon.** In Doors without masking, wrong actions are silent no-ops (no penalty beyond step cost). The value difference ΔQ between correct and incorrect actions only becomes visible when the tree expands deep enough to see the consequence of a correct action — specifically, the chain MOVE_TO(key_loc) → PICK(key) → unlock_bonus (+0.1). This requires tree depth ≥ 2.

With uniform visitation, expected tree depth ≈ log_K(N). To reach depth 2:

```
log_K(N) ≥ 2   →   N ≥ K²
```

| D | K (actions) | K²  | Implication                            |
|---|-------------|-----|----------------------------------------|
| 2 |      8      |  64 | N≥64 for guaranteed depth-2            |
| 4 |     16      | 256 | N≥256 for guaranteed depth-2           |
| 6 |     24      | 576 | N≥576 for guaranteed depth-2           |

These bounds are conservative — UCB is not uniform; actions with higher Q get more visits. As the value head improves, the tree focuses on promising branches and reaches much greater depth at the same N. But in early iterations (random network), these bounds explain why very low N fails.

**(B) Visit counts must differentiate good from bad actions above noise.** The multinomial standard deviation of the visit count for any single action under uniform visitation is √(N/K · (1 - 1/K)) ≈ √(N/K). For the visit-count shift δN toward a good action to exceed this noise:

```
δN > √(N/K)   →   need N/K > 1/ΔQ²  (roughly)
```

But because ΔQ in Doors is substantial once the tree reaches depth 2 (unlock_bonus = 0.1 creates ΔQ ≈ 0.1–0.3), the binding constraint is (A), not (B). **Tree depth, not visit-count noise, is the bottleneck for small N.**

**Practical takeaway:** N/K ≈ 1 (each action visited once on average) is the absolute minimum. N/K ≈ 2–3 gives meaningful depth-2 exploration and is the regime where learning begins.

**Sims/action ratio table:**

| D | K  | N=5  | N=10 | N=25 | N=50 |
|---|----|------|------|------|------|
| 2 |  8 | 0.62 | 1.25 | 3.12 | 6.25 |
| 3 | 12 | 0.42 | 0.83 | 2.08 | 4.17 |
| 4 | 16 | 0.31 | 0.62 | 1.56 | 3.12 |
| 5 | 20 | 0.25 | 0.50 | 1.25 | 2.50 |
| 6 | 24 | 0.21 | 0.42 | 1.04 | 2.08 |

### 2.2 Data Volume: Why games/iteration matters

Each self-play game produces approximately H state-action-reward tuples (H = horizon for failing agents, ≈ opt_steps for solving agents). The network trains on accumulated data from the last `n_past_iterations_to_train = 20` iterations.

**Effective training set at iteration i:** min(i, 20) × games × avg_episode_length

**Network capacity constraint:** The network has P parameters (ranging from 1,225 at D=2 to 6,405 at D=6). Training with 10 epochs and batch_size=32 means P/32 × 10 gradient steps per iteration's data. For convergence, a rough requirement is that the training set should eventually reach 2–5× the parameter count.

| D | params |games=5/iter|games=10|games=20|games=40| Iter to reach 2×params (buffer=20) |
|---|--------|------------|--------|--------|--------|------------------------------------|
| 2 | 1,225  | 75/iter    | 150    | 300    | 600    | games=5: iter 33, games=20: iter 9 |
| 4 | 2,829  | 175/iter   | 350    | 700    | 1400   | games=5: iter 33, games=20: iter 9 |
| 6 | 6,405  | 275/iter   | 550    | 1100   | 2200   | games=5: iter 47, games=20: iter 12|

**With games=5, the replay buffer doesn't reach 2× params until iteration ~33-47.** This means the network trains on insufficient data for most of the run, leading to overfitting and the spike-and-collapse instability seen in the D=14 failure.

**With games=20, the buffer reaches 2× params by iteration ~9-12.** This is why games=20 is the natural lower bound for reliable learning.

### 2.3 Why wrong actions are no-ops matters

In Doors without precondition masking (`use_precondition_mask=False`), invalid actions (MOVE_TO locked room, PICK unavailable key) are silently converted to no-ops. This means:

- **No negative reward signal** for bad actions — just a step_penalty of 0.01 (same as for correct actions)
- **MCTS must discover positive signal** (unlock_bonus, goal reward) by chance to differentiate actions
- The fraction of valid actions at any state is roughly 1/D + 1/(D-1) ≈ 2/D for large D

At D=4, roughly 4–5 of 16 actions are valid at any given state (~25-30%). At D=6, roughly 5-6 of 24 are valid (~21-25%). MCTS must waste most of its budget visiting actions that are silently ignored.

---

## 3. Fixed Hyperparameters (Do Not Sweep)

| Parameter | Value | Justification |
|---|---|---|
| locs_per_room | 3 | Standard; changing it confounds D scaling analysis |
| temperature | 1.0 | Standard self-play exploration temperature |
| c_exploration | 1.5 | UCB constant; standard for AlphaZero variants |
| dirichlet_alpha | 0.25 | Root noise concentration; α ≈ 10/K is standard (AlphaZero used 0.03 for Go with K=362) |
| dirichlet_epsilon | 0.40 | Noise weight at root; slightly aggressive to aid exploration in small problems |
| reward_discount | 1.0 | Undiscounted; episodes are short and rewards are terminal/near-terminal |
| n_past_iterations_to_train | 20 | Replay buffer depth; keeps 20 iterations of data |
| accept_threshold | 0.0 | No gating — deliberately disabled so we observe raw learning dynamics including instability |
| n_procs | 8 | Parallelism for self-play; matches typical CPU count |
| use_precondition_mask | False | No domain knowledge; agent must learn which actions are valid |

---

## 4. Experiment Design

### Experiment A: Sanity Check — Does the AlphaZero Loop Work? (2 runs)

**Purpose:** Verify the full AlphaZero pipeline (self-play → train → improve) produces correct learning dynamics on the simplest non-trivial instance.

**Why D=2 with sims=25 is a rigorous sanity check:**

1. **MCTS covers the full solution.** The optimal plan is 3 steps: MOVE_TO(key_loc) → PICK(0) → MOVE_TO(goal). With sims=25 and K=8 actions, sims/action = 3.12. The expected tree depth is log₈(25) ≈ 1.5, meaning MCTS regularly reaches depth 2–3 on some branches. Since the full plan is depth 3, MCTS can see the entire solution within a single tree search — even with an untrained network.

2. **State space is exhaustively coverable.** D=2 has at most 48 distinct states. With games=20 and horizon=15, each iteration generates ~300 state-action tuples. After 3 iterations the replay buffer has ~900 examples, covering the state space ~19× over. The network (1,225 params) has more than enough data.

3. **Random exploration can discover partial solutions.** P(random solve) ≈ (1/8)³ = 1/512 per trajectory. With 20 games/iteration of 15 steps each, the expected number of random solves is ~20 × 15/3 × (1/512) ≈ 0.2 per iteration. Not guaranteed, but plausible within a few iterations — and MCTS with even noisy value estimates will far exceed random.

4. **If D=2 fails with these settings, no hyperparameter issue can explain it.** The compute budget is generous in every dimension. Failure would indicate a code bug in the self-play loop, the training pipeline, or the MCTS implementation.

| Run | D | sims | games | iters | Purpose |
|-----|---|------|-------|-------|---------|
| A1  | 2 |  25  |  20   |  15   | Generous budget — must solve |
| A2  | 3 |  25  |  20   |  15   | Slightly harder — should still solve |

```bash
python scripts/run_doors_direct.py --D 2 --n-simulations 25 --n-games-per-train 20 --n-iterations 15 --non-interactive
python scripts/run_doors_direct.py --D 3 --n-simulations 25 --n-games-per-train 20 --n-iterations 15 --non-interactive
```

**Pass criteria:**
- A1: solve_rate = 1.0 by iteration 5. Eval reward reaches +1.07 (optimal). Policy loss drops well below ln(8) = 2.08.
- A2: solve_rate = 1.0 by iteration 10. Eval reward reaches +1.15.

**If A1 fails → STOP.** Debug the code. Check: (a) are MCTS visit counts non-uniform? (b) does the policy loss decrease at all? (c) is the value head predicting anything other than the mean?

**If A1 passes but A2 fails → suspicious.** D=3 has only 12 actions and 5 optimal steps. Failure would suggest the network architecture or training loop degrades even with small scaling. Investigate policy loss trajectory.

---

### Experiment B: MCTS Simulation Sweep at D=4 (4 runs)

**Purpose:** Isolate the effect of MCTS budget on learning, holding data volume constant. Answers: "How many simulations per move does D=4 need?"

**Why D=4 is the right testbed:** D=4 has 16 actions and 7 optimal steps — large enough that MCTS budget matters, small enough that each run completes in minutes. The prior result (sims=60, games=30 solves at iter 2) gives us a known-good reference point.

**Why these simulation values:**

| Run | sims | sims/K | Expected tree depth (log₁₆(N)) | Prediction |
|-----|------|--------|--------------------------------|------------|
| B1  |   5  |  0.31  | 0.58 — tree barely reaches depth 1 | **Fail.** Most actions unvisited. Visit counts are noise. |
| B2  |  10  |  0.63  | 0.83 — depth 1 on average | **Marginal.** Each action visited <1 time. Cannot see key→unlock chain. |
| B3  |  25  |  1.56  | 1.16 — some depth-2 branches | **Borderline.** First regime where learning is possible. |
| B4  |  50  |  3.12  | 1.41 — reliable depth-2 | **Should solve.** Prior with sims=60 solved at iter 2. |

All with games=20, iters=30.

```bash
python scripts/run_doors_direct.py --D 4 --n-simulations 5  --n-games-per-train 20 --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 4 --n-simulations 10 --n-games-per-train 20 --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 4 --n-simulations 25 --n-games-per-train 20 --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 4 --n-simulations 50 --n-games-per-train 20 --n-iterations 30 --non-interactive
```

**Diagnostic analysis after running:**
1. Plot policy loss vs iteration for all 4 runs. The curve should decrease faster with higher sims. If sims=50 and sims=25 have identical policy loss curves → simulations don't help beyond 25, and the bottleneck is elsewhere (data volume or network capacity).
2. Check whether B1 (sims=5) shows **anti-learning** (policy loss increases) or just **stagnation** (policy loss flat). Anti-learning = MCTS producing worse-than-uniform targets (possible if visit counts are dominated by a single random branch). Stagnation = targets are near-uniform, no signal.
3. Record the iteration at which solve_rate first reaches 1.0 for B3, B4. This gives the "MCTS threshold" for D=4.

---

### Experiment C: Data Volume Sweep at D=4 (4 runs)

**Purpose:** Isolate the effect of self-play data volume, holding MCTS budget constant. Answers: "How many games per iteration does D=4 need?"

**Why sims=25 is the right anchor:** From Experiment B, sims=25 is the borderline regime. By using it here, we test data volume in the regime where MCTS quality is adequate but not lavish — the regime most relevant to understanding the D=14 failure.

**Why these game counts:**

| Run | games | examples/iter | buffer at iter 10 | buffer/params | Prediction |
|-----|-------|---------------|--------------------|--------------:|------------|
| C1  |   5   |     ~175      |      1,750         |    0.62×      | **Unstable.** Spike-and-collapse expected (same mechanism as D=14 failure). |
| C2  |  10   |     ~350      |      3,500         |    1.24×      | **Slow, possibly noisy.** Data barely reaches parity with params. |
| C3  |  20   |     ~700      |      7,000         |    2.47×      | **Should work.** Buffer reaches 2× params by iter 9. |
| C4  |  40   |    ~1400      |     14,000         |    4.95×      | **Must work.** Generous data. If this fails, something else is wrong. |

All with sims=25, iters=30.

```bash
python scripts/run_doors_direct.py --D 4 --n-simulations 25 --n-games-per-train 5  --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 4 --n-simulations 25 --n-games-per-train 10 --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 4 --n-simulations 25 --n-games-per-train 20 --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 4 --n-simulations 25 --n-games-per-train 40 --n-iterations 30 --non-interactive
```

**Critical diagnostic:** If C1 (games=5) shows the same spike-and-collapse pattern as the D=14 run, that **reproduces the failure mode at small scale**. This is the single most important result of the entire plan — it confirms data starvation is the root cause and that the D=14 failure is not a code bug.

---

### Experiment D: Difficulty Scaling (3 runs)

**Purpose:** With a fixed moderate budget, how does performance degrade as D increases from 4 to 6? (D=2,3 already covered by Experiment A.)

**Why sims=25, games=20:** This is the "lean but adequate" configuration identified by Experiments B and C. Using it for difficulty scaling reveals whether the budget that works at D=4 transfers to D=5,6, or whether the problem demands more resources at larger D.

| Run | D | sims | sims/K | games | iters | Prediction |
|-----|---|------|--------|-------|-------|------------|
| D1  | 4 |  25  |  1.56  |  20   |  30   | Should solve (same as B3/C3) |
| D2  | 5 |  25  |  1.25  |  20   |  40   | **Uncertain.** sims/K = 1.25 is thin. May solve slowly or not at all. |
| D3  | 6 |  25  |  1.04  |  20   |  50   | **Likely fails.** sims/K ≈ 1.0. Expected to show same stagnation as starved runs. |

```bash
python scripts/run_doors_direct.py --D 4 --n-simulations 25 --n-games-per-train 20 --n-iterations 30 --non-interactive
python scripts/run_doors_direct.py --D 5 --n-simulations 25 --n-games-per-train 20 --n-iterations 40 --non-interactive
python scripts/run_doors_direct.py --D 6 --n-simulations 25 --n-games-per-train 20 --n-iterations 50 --non-interactive
```

**Note:** D1 overlaps with B3/C3. If already run, reuse those results.

**Diagnostic:** Record the iteration of first solve for each D. If D=5 and D=6 fail, check whether it's the same failure mode (policy loss flat near ln(K)) as the D=14 case. If so, the difficulty-compute curve is steeper than linear and D=14 requires substantially more than proportional scaling of sims and games.

---

## 5. Run Summary

| Phase | Runs | Experiment | Wall-clock (est.) | Dependency |
|-------|------|------------|--------------------|-----------:|
| 1     | 2    | A (sanity) | ~3-5 min           | None — run first |
| 2     | 8    | B + C (sweeps) | ~30-60 min     | A must pass |
| 3     | 3    | D (scaling) | ~30-60 min        | B,C inform interpretation |

**Total: 13 runs (with D1 reused from B3/C3), all ≤ 50 iterations.**

---

## 6. Decision Table: What Results Mean

| Outcome | Interpretation | Next Action |
|---|---|---|
| A1 (D=2) fails | **Code bug.** The problem is trivially solvable with this budget. | Debug MCTS, training loop, or reward pipeline |
| A1 passes, A2 (D=3) fails | Suspicious. 12 actions is small. | Check if network architecture is too small; check action masking logic |
| B4 (sims=50) fails | sims aren't the issue — data volume or training is broken | Focus on Experiment C results |
| B1 (sims=5) shows spike-and-collapse | MCTS noise can cause instability even at D=4 | Confirms MCTS quality is a prerequisite for stable training |
| C1 (games=5) shows spike-and-collapse | **Reproduces D=14 failure at small scale.** Data starvation confirmed. | games/iter is the primary lever; D=14 needs games≥50 |
| C3 (games=20) solves, C1 fails | Clean threshold effect for data volume | games≥20 is the minimum for D=4 |
| D2 (D=5, sims=25) fails | sims/K=1.25 is below threshold | D=5+ needs sims≥40 (sims/K≥2.0) |
| D3 (D=6, sims=25) fails while prior sims=60 solved | Confirms sims/K≥2.5 needed | D=14 (K=56) needs sims≥112 |
| All experiments pass | AlphaZero is working; D=14 failure was purely resource starvation | Rerun D=14 with sims=120, games=50, iters=30 |

**Red flags that indicate a code bug (not just hyperparameters):**
- Policy loss never decreases below ln(K) in any run (network not learning from MCTS targets)
- Value loss increases over training (value head anti-learning)
- sims=50, games=40 at D=4 fails to solve within 30 iterations
- Reward consistently goes *down* as training progresses
