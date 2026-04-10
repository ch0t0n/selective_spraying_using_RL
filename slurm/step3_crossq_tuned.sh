#!/bin/bash
# ============================================================
# Step 3 — Main results, TUNED hyperparameters, CrossQ (GPU)
#
# Identical grid to Step 1 but passes --hyperparams_json so
# train.py loads the Optuna-tuned HPs produced by Step 2.
#
# Grid: 1 alg × 10 env sets × 4 robot counts × 5 seeds
# Total jobs: 200  →  array=0-199
# ============================================================

#SBATCH --array=0-199
#SBATCH --job-name=s3_crossq_tuned
#SBATCH --output=/homes/choton/rl4pag/neurips_experiments/logs/slurm_outputs/s3_crossq_tuned/%x_%A_%a.out
#SBATCH --error=/homes/choton/rl4pag/neurips_experiments/logs/slurm_errors/s3_crossq_tuned/%x_%A_%a.err
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

algorithms=("CrossQ")
sets=(1 2 3 4 5 6 7 8 9 10)
robots=(2 3 4 5)
seeds=(0 42 123 2024 9999)

num_sets=${#sets[@]}
num_robots=${#robots[@]}
num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
robot_idx=$(( (index / num_seeds) % num_robots ))
set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
alg_idx=$(( index / (num_seeds * num_robots * num_sets) ))

algorithm=${algorithms[$alg_idx]}
set=${sets[$set_idx]}
num_robots_value=${robots[$robot_idx]}
seed=${seeds[$seed_idx]}
steps=2000000

BEST_JSON="/homes/choton/rl4pag/neurips_experiments/logs/best_hyperparams.json"

echo "S3-CrossQ-tuned | alg=$algorithm | set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n robot_env python3 train.py \
    --algorithm        $algorithm \
    --set              $set \
    --num_robots       $num_robots_value \
    --seed             $seed \
    --steps            $steps \
    --experiment       main \
    --hyperparams_json $BEST_JSON \
    --verbose          1 \
    --log_steps        10000 \
    --device           cuda

wait