#!/bin/bash
# Schedule Stage 1 (masked) runs to launch after Stage 2 (unmasked) finishes.
# Stage 1 uses identical hyperparameters (eps=0.00, tau=0.50) for a clean ablation.
#
# Usage:
#   nohup bash scripts/run/run_stage1_after_stage2.sh > experiments/stage1_schedule.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."  # repo root

STAGE2_D6_PID=11526
STAGE2_D7_PID=11534

echo "[$(date)] Waiting for Stage 2 D=6 (PID $STAGE2_D6_PID) and D=7 (PID $STAGE2_D7_PID) to finish..."

# macOS doesn't support tail --pid, so poll with kill -0
while kill -0 $STAGE2_D6_PID 2>/dev/null || kill -0 $STAGE2_D7_PID 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Stage 2 runs finished. Launching Stage 1 (masked) at D=6 and D=7..."
echo "[$(date)] Hyperparameters: eps=0.00, tau=0.50, sims=80, games=50, iters=50, n_procs=3"

# Launch Stage 1 D=6 and D=7 in parallel (each uses 3 cores = 6 total)
python scripts/run/run_epsilon_tau_sweep.py --phase 4 --stage 1 --d 6 --n-procs 3 \
    --best-eps 0.0 --best-tau 0.5 --top-configs '0.0,0.5' --iterations 50 \
    > experiments/stage1_D6_run.log 2>&1 &
STAGE1_D6_PID=$!

python scripts/run/run_epsilon_tau_sweep.py --phase 4 --stage 1 --d 7 --n-procs 3 \
    --best-eps 0.0 --best-tau 0.5 --top-configs '0.0,0.5' --iterations 50 \
    > experiments/stage1_D7_run.log 2>&1 &
STAGE1_D7_PID=$!

echo "[$(date)] Stage 1 D=6 launched (PID $STAGE1_D6_PID)"
echo "[$(date)] Stage 1 D=7 launched (PID $STAGE1_D7_PID)"

# Wait for both to complete
wait $STAGE1_D6_PID
echo "[$(date)] Stage 1 D=6 finished (exit code $?)"

wait $STAGE1_D7_PID
echo "[$(date)] Stage 1 D=7 finished (exit code $?)"

echo "[$(date)] All experiments complete."
