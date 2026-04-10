#!/bin/bash
# ============================================================
# Step 7 — Domain Randomization (CrossQ, GPU)
#
# Three DR training conditions:
#   0 → none   (standard, no DR)
#   1 → wind   (re-sample wind each episode: U(0,1) m/s, U(0,2π))
#   2 → full   (wind + actuation noise + spray radius + mass + thrust)
#
# Grid: 3 DR modes × 10 env sets × 4 robot counts × 5 seeds
# Total jobs: 600  →  array=0-599
#
# Index layout (innermost → outermost):
#   seed_idx   = index % 5
#   robot_idx  = (index // 5) % 4
#   set_idx    = (index // 20) % 10
#   dr_idx     = index // 200
# ============================================================

#SBATCH --array=0-599
#SBATCH --job-name=s7_dr
#SBATCH --output=/homes/choton/rl4pag/neurips_experiments/logs/slurm_outputs/s7_dr/%x_%A_%a.out
#SBATCH --error=/homes/choton/rl4pag/neurips_experiments/logs/slurm_errors/s7_dr/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]

dr_modes=("none" "wind" "full")
sets=(1 2 3 4 5 6 7 8 9 10)
robots=(2 3 4 5)
seeds=(0 42 123 2024 9999)

num_dr=${#dr_modes[@]}
num_sets=${#sets[@]}
num_robots=${#robots[@]}
num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
robot_idx=$(( (index / num_seeds) % num_robots ))
set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
dr_idx=$(( index / (num_seeds * num_robots * num_sets) ))

dr_mode=${dr_modes[$dr_idx]}
set=${sets[$set_idx]}
num_robots_value=${robots[$robot_idx]}
seed=${seeds[$seed_idx]}
steps=1000000

echo "S7-DR | dr_mode=$dr_mode | set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n robot_env python3 train.py \
    --algorithm   "CrossQ" \
    --set         $set \
    --num_robots  $num_robots_value \
    --seed        $seed \
    --steps       $steps \
    --experiment  dr \
    --ablation    $dr_mode \
    --verbose     1 \
    --log_steps   10000 \
    --device      cuda

wait