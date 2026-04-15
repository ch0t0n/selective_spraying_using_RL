#!/bin/bash

# Run all experiments with: sbatch slurm_scripts/crossq_transfer_3_robots.sh

#SBATCH --array=0-8
#SBATCH --job-name=crossq_t3r_James
#SBATCH --output=slurm_scripts/slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_scripts/slurm_out/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=160:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_scripts/slurm_out

export HOME=/homes/jameschapman
export WANDB_MODE=offline
export WANDB_SILENT=true

source /homes/jameschapman/miniforge3/etc/profile.d/conda.sh
conda activate rl4pag

# Modify these for other experiments
algorithms=("CrossQ")
sets=(2 3 4 5 6 7 8 9 10)
seed=(33)

# IMPORTANT: array job length = num_algorithms * num_sets - 1
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}
num_seeds=${#seed[@]}

index=$((SLURM_ARRAY_TASK_ID))

algorithm_index=$((index / num_sets))
set_index=$((index % num_sets ))
seed_index=$((index % num_seeds))

algorithm=${algorithms[$algorithm_index]}
set=${sets[$set_index]}
seed=${seed[$seed_index]}

load_set=1
RUN_NAME="${algorithm}_from_set${load_set}_to_set${set}_seed${seed}_transfer_3_robots_cuda"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "ALG=$algorithm LOAD_SET=$load_set SET=$set SEED=$seed DEVICE=cuda"
echo "RUN_NAME=$RUN_NAME"

steps=1000000
num_robots=3

python3 transfer.py       --algorithm $algorithm       --load_set $load_set       --set $set       --seed $seed       --steps $steps       --num_robots $num_robots       --run_name $RUN_NAME       --device "cuda"
