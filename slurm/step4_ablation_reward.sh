#!/bin/bash
# ============================================================
# Step 4 — Ablation: Reward function components (CrossQ, GPU)
#
# Fixed: CrossQ, N = 3, env variation 1, 0.5 M timesteps
# Grid:  4 reward conditions × 5 seeds = 20 jobs
#
# conditions:
#   0 → full     (all reward terms — baseline)
#   1 → no_col   (collision penalty + termination disabled)
#   2 → no_cov   (coverage terms disabled)
#   3 → no_eff   (efficiency terms disabled)
#
# Index layout:
#   seed_idx = index % 5
#   cond_idx = index // 5
# ============================================================

#SBATCH --array=0-19
#SBATCH --job-name=s4_ablation_reward
#SBATCH --output=/homes/choton/rl4pag/neurips_experiments/logs/slurm_outputs/s4_ablation_reward/%x_%A_%a.out
#SBATCH --error=/homes/choton/rl4pag/neurips_experiments/logs/slurm_errors/s4_ablation_reward/%x_%A_%a.err
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

conditions=("full" "no_col" "no_cov" "no_eff")
seeds=(0 42 123 2024 9999)

num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
cond_idx=$(( index / num_seeds ))

condition=${conditions[$cond_idx]}
seed=${seeds[$seed_idx]}

echo "S4-ablation-reward | condition=$condition | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n robot_env python3 train.py \
    --algorithm   "CrossQ" \
    --set         1 \
    --num_robots  3 \
    --seed        $seed \
    --steps       1000000 \
    --experiment  ablation_reward \
    --ablation    $condition \
    --verbose     1 \
    --log_steps   10000 \
    --device      cuda

wait