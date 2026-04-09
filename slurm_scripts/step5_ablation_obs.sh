#!/bin/bash
# ============================================================
# Step 5 — Ablation: Observation space (CrossQ, GPU)
#
# Fixed: CrossQ, N = 3, env variation 1, 0.5 M timesteps
# Grid:  5 obs conditions × 5 seeds = 25 jobs
#
# obs_mode:
#   0 → base          (original: positions+vel+cap+infection, 5N+M)
#   1 → full          (base + wind vector + spray history, 6N+M+2)
#   2 → no_wind       (base + spray history, no wind estimate)
#   3 → no_spray_hist (base + wind vector, no spray history)
#   4 → pos_only      (robot positions only, 2N)
#
# Index layout:
#   seed_idx = index % 5
#   cond_idx = index // 5
# ============================================================

#SBATCH --array=0-24
#SBATCH --job-name=s5_ablation_obs
#SBATCH --output=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_logs/step5_ab_obs/outputs/%x_%j.out
#SBATCH --error=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_logs/step5_ab_obs/errors/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

obs_modes=("base" "full" "no_wind" "no_spray_hist" "pos_only")
seeds=(0 42 123 2024 9999)

num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
cond_idx=$(( index / num_seeds ))

obs_mode=${obs_modes[$cond_idx]}
seed=${seeds[$seed_idx]}

echo "S5-ablation-obs | obs_mode=$obs_mode | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n rl4pag python3 train.py \
    --algorithm   CrossQ \
    --set         1 \
    --num_robots  3 \
    --seed        $seed \
    --steps       500000 \
    --experiment  ablation_obs \
    --ablation    $obs_mode \
    --verbose     1 \
    --log_steps   5000 \
    --device      cuda

wait