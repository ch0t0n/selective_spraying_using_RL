#!/bin/bash
# ============================================================
# Step 2 — Hyperparameter tuning (Optuna)
#
# One job per algorithm (6 total).  Each job runs 20 sequential
# Optuna trials × 500 k timesteps.  All jobs write to the
# same best_hyperparams.json (read-modify-write is safe here
# because the 6 jobs write different keys).
#
# Grid: 6 algorithms  →  array=0-5
#   0 → A2C   (CPU)
#   1 → ARS   (CPU)
#   2 → PPO   (CPU)
#   3 → TRPO  (CPU)
#   4 → CrossQ (GPU)
#   5 → TQC   (GPU)
# ============================================================

#SBATCH --array=0-5
#SBATCH --job-name=s2_tune
#SBATCH --output=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_logs/step2_tune/outputs/%x_%j.out
#SBATCH --error=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_logs/step2_tune/errors/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --export=NONE

# GPU indices: CrossQ (4) and TQC (5) need a GPU
GPU_INDICES=(4 5)
algorithms=("A2C" "ARS" "PPO" "TRPO" "CrossQ" "TQC")

index=$SLURM_ARRAY_TASK_ID
algorithm=${algorithms[$index]}

# Request GPU only for CrossQ and TQC
needs_gpu=0
for gi in "${GPU_INDICES[@]}"; do
    if [ "$index" -eq "$gi" ]; then
        needs_gpu=1
        break
    fi
done

if [ "$needs_gpu" -eq 1 ]; then
    device="cuda"
    # Ask SLURM for a GPU (SLURM cannot conditionally request resources,
    # so this script is submitted to a GPU partition; CPU-only algs
    # will simply not use the GPU)
else
    device="cpu"
fi

BEST_JSON="/homes/choton/rl4pag/selective_spraying_using_RL/logs/best_hyperparams.json"

echo "S2-tune | algorithm=$algorithm | device=$device | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n rl4pag python3 tune.py \
    --algorithm   $algorithm \
    --device      $device \
    --n_trials    20 \
    --tune_steps  500000 \
    --output_json $BEST_JSON

wait