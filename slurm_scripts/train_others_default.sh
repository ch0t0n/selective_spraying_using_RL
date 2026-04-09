#!/bin/bash

#SBATCH --array=0-199
#SBATCH --job-name=others_default
#SBATCH --output=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_outputs/%x_%j.out
#SBATCH --error=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_outputs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --export=NONE

# Modify these for other experiments
algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO")
sets=(1 2 3 4 5 6 7 8 9 10)
robots=(2 3 4 5)

# IMPORTANT: array job length = num_algorithms * num_sets * num_robots - 1
# Total jobs = alg * sets * robots = 5 * 10 * 4 = 200 → array=0-199
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}
num_robots=${#robots[@]}

index=$((SLURM_ARRAY_TASK_ID))
algorithm_index=$(( index / (num_sets * num_robots) ))
set_index=$(( (index % (num_sets * num_robots)) / num_robots ))
robot_index=$(( index % num_robots ))
algorithm=${algorithms[$algorithm_index]}
set=${sets[$set_index]}
steps=2000000
num_robots_value=${robots[$robot_index]}

# ================================
# Run training
# ================================
echo "Running: alg=$algorithm | set=$set | robots=$num_robots_value | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n rl4pag python3 train_default.py \
    --algorithm $algorithm \
    --set $set \
    --verbose 1 \
    --log_steps 10000 \
    --seed 33 \
    --steps $steps \
    --num_robots $num_robots_value

wait