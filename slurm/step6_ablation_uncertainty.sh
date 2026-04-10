#!/bin/bash
# ============================================================
# Step 6 — Ablation: Physical uncertainty model (CrossQ, GPU)
#
# Fixed: CrossQ, N = 3, env variation 1, 0.5 M timesteps
# Grid:  4 uncertainty conditions × 5 seeds = 20 jobs
#
# uncertainty_mode:
#   0 → full          (wind + actuation + spray noise — default)
#   1 → wind_only     (only wind noise active)
#   2 → act_only      (only actuation noise active)
#   3 → deterministic (all noise sources disabled)
#
# Each trained policy is evaluated under all four conditions
# inside train.py to measure the stochasticity gap — no extra
# scripts needed.
#
# Index layout:
#   seed_idx = index % 5
#   cond_idx = index // 5
# ============================================================

#SBATCH --array=0-19
#SBATCH --job-name=s6_ablation_uncertainty
#SBATCH --output=/homes/choton/rl4pag/neurips_experiments/logs/slurm_outputs/s6_ablation_uncertainty/%x_%A_%a.out
#SBATCH --error=/homes/choton/rl4pag/neurips_experiments/logs/slurm_errors/s6_ablation_uncertainty/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]

uncertainty_modes=("full" "wind_only" "act_only" "deterministic")
seeds=(0 42 123 2024 9999)

num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
cond_idx=$(( index / num_seeds ))

uncertainty_mode=${uncertainty_modes[$cond_idx]}
seed=${seeds[$seed_idx]}

echo "S6-ablation-uncertainty | mode=$uncertainty_mode | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n robot_env python3 train.py \
    --algorithm   "CrossQ" \
    --set         1 \
    --num_robots  3 \
    --seed        $seed \
    --steps       1000000 \
    --experiment  ablation_uncertainty \
    --ablation    $uncertainty_mode \
    --verbose     1 \
    --log_steps   10000 \
    --device      cuda

wait