#!/bin/bash

# Run all experiments with: sbatch slurm_scripts/train_default_all_3_robots.sh

#SBATCH --array=0-49
#SBATCH --job-name=RL4PA_d3r_James
#SBATCH --output=slurm_scripts/slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_scripts/slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_scripts/slurm_out

export HOME=/homes/jameschapman
export WANDB_MODE=offline
export WANDB_SILENT=true

source /homes/jameschapman/miniforge3/etc/profile.d/conda.sh
conda activate rl4pag

# Modify these for other experiments
algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO")
sets=(1 2 3 4 5 6 7 8 9 10)
seed=(33)

# IMPORTANT: array job length = num_algorithms * num_sets * num_seeds - 1
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}
num_seeds=${#seed[@]}

index=$((SLURM_ARRAY_TASK_ID))

algorithm_index=$((index / (num_sets * num_seeds)))
set_index=$(( (index / num_seeds) % num_sets ))
seed_index=$((index % num_seeds))

algorithm=${algorithms[$algorithm_index]}
set=${sets[$set_index]}
seed=${seed[$seed_index]}

RUN_NAME="${algorithm}_set${set}_seed${seed}_default_3_robots_cpu"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "ALG=$algorithm SET=$set SEED=$seed DEVICE=cpu"
echo "RUN_NAME=$RUN_NAME"

steps=1000000
num_robots=3

python3 train_default.py \
  --algorithm $algorithm \
  --set $set \
  --seed $seed \
  --steps $steps \
  --num_robots $num_robots \
  --run_name $RUN_NAME \
  --device "cpu"

