# Doors D=14 Direct Play: Failure Diagnosis

**Date:** 2026-03-26
**Run:** `20260326_155122_D14_L3_nomask_mcts120_games20_iter20`
**Status:** Failed — 0% solve rate at termination (10/20 iterations completed)

---

## 1. Run Configuration

| Parameter | Value | Notes |
|---|---|---|
| D (rooms) | 14 | 42 locations, 13 keys, 56 actions |
| MCTS simulations | 120 | |
| Games per iteration | **20** | Prior D=14 runs used **50** |
| Training iterations | 20 (10 completed) | Prior runs used 15 |
| Replay buffer depth | 20 iterations | |
| Gate threshold | 0.0 | No quality gating — every network accepted |
| Action masking | None (structural) | Agent must learn preconditions from experience |

Optimal plan: 27 steps, optimal reward: +2.03.

---

## 2. What the Plots Show

### Evaluation Reward (top-left)
- Iterations 1-3: flat at roughly -1.0 (agent flails, hits horizon).
- Iterations 4-8: gradual improvement to -0.25 range, with high variance (swings between -0.25 and -0.85).
- **Iteration 9: spike to +0.95** — the agent briefly solves the task.
- **Iteration 10: collapse back to -0.55** — the solution is immediately lost.
- The optimal line (+2.03) is never approached.

### Training Losses (top-right)
- Policy loss: 3.99 → 3.80 over 10 iterations. Since ln(56) ≈ 4.03 (uniform random over 56 actions), the policy remains near-uniform throughout. The network is barely learning which actions to take.
- Value loss: stays low (0.05-0.10) — the value head predicts roughly the mean reward but doesn't differentiate states well.

### Solve Rate (bottom-left)
- 0% for all iterations except iteration 9 (100%), then back to 0%.
- A single lucky iteration, not stable learning.

### Combined Metrics (bottom-right)
- Confirms the spike-and-crash pattern. No sustained improvement trend.

---

## 3. Systematic Diagnosis

### 3.1 Primary Cause: Insufficient Self-Play Data

**20 games/iteration is too few for D=14.**

- Each game produces ~135 state-action examples (horizon = 135 for D=14).
- 20 games × 135 = **2,700 examples per iteration**.
- The network has 69 inputs, 276 hidden units, 56+1 outputs — roughly **40K parameters**.
- With 56 possible actions and a 27-step optimal plan, the policy landscape is complex. The MCTS visit distributions from only 20 games provide a very noisy training signal.
- **Comparison:** Prior D=14 runs used 50 games/iteration (6,750 examples/iter), and even those struggled. Prior D=12 runs with 50 games showed more stable learning.

### 3.2 Spike-and-Collapse Instability

The iteration 9→10 pattern (solve_rate 1.0 → 0.0) is a textbook sign of **catastrophic forgetting under low-data regimes**:

1. At iteration 9, the network happened to produce a policy that could solve the task (avg_train_reward jumped from -0.445 to -0.149).
2. The training data from iteration 9 (which includes some successful trajectories) shifts the loss landscape.
3. At iteration 10, the updated network overfits to the new data distribution and destroys the fragile policy that was working.

**Contributing factors:**
- **No gating (threshold=0.0):** Every network update is accepted, even if it degrades performance. With a threshold of 0.55, the iter-10 regression would have been rejected and the iter-9 weights preserved.
- **Full replay buffer still dominated by failure data:** Even at iteration 9, the buffer contains 9 iterations of mostly-failing trajectories vs 1 iteration of success. The network trains predominantly on "how to fail."

### 3.3 Policy Loss Near Uniform

The policy loss hovering at ~3.8-3.9 (vs 4.03 for uniform) means the network has barely differentiated between the 56 actions. In a 56-action space:
- Most actions are invalid or useless at any given state (e.g., MOVE_TO a locked room, PICK a key you're not near).
- With no precondition masking, the MCTS must discover through simulation that ~80% of actions are no-ops. 120 simulations spread over 56 actions gives only ~2 visits per action on average, which is inadequate for accurate visit-count-based policy targets.

### 3.4 The Run Stopped at 10/20 Iterations

Only 10 of the requested 20 iterations completed (the log has 10 entries). This may indicate a crash, user interruption, or timeout. Regardless, 10 iterations is not enough for D=14 to stabilize — the prior D=12 ablation runs with 50 games needed 15 iterations to show clear trends.

---

## 4. Comparison with Prior D=14 Runs

The prior D=14 experiments in `experiments/doors_direct/` all used **mcts=120, games=50, iter=15**. This run used **games=20**, which is 40% of that budget. The self-play data volume is the single largest difference and the most likely root cause.

---

## 5. Recommended Fixes (Ranked by Impact)

| Fix | Expected Impact | Cost |
|---|---|---|
| **Increase games/iteration to 50-80** | High — directly addresses data starvation | ~2.5-4× wall-clock per iteration |
| **Enable gating (threshold ≥ 0.55)** | Medium — prevents collapse by rejecting regressions | Near-zero (adds eval games) |
| **Increase MCTS simulations to 200+** | Medium — better action discrimination in 56-action space | ~1.7× per game |
| **Run for 30-50 iterations** | Medium — allows the slow learning signal to accumulate | Proportional wall-clock |
| **Enable precondition masking** | High for search quality — prunes ~80% of dead actions | Changes the experimental question (provides domain knowledge) |

### Minimal Rerun Command

```bash
python scripts/run_doors_direct.py \
  --D 14 \
  --n-simulations 120 \
  --n-games-per-train 50 \
  --n-iterations 30 \
  --n-procs 8 \
  --non-interactive
```

This matches the proven configuration from prior D=14 runs with an extended iteration budget.

---

## 6. Summary

The D=14 direct play run failed due to **data starvation** (20 games/iter is insufficient for a 56-action, 27-step planning problem) compounded by **no quality gating** (allowing destructive network updates). The network briefly found a solution at iteration 9 but immediately lost it — a spike-and-collapse pattern characteristic of high-variance, low-data AlphaZero training. The policy loss remaining near-uniform throughout confirms the network never reliably learned action preferences. Increasing self-play games to 50+ and enabling acceptance gating are the highest-priority fixes.
