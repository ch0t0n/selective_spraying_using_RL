#!/bin/bash
# ============================================================
# eval_main.sh — Run evaluate.py for all main-experiment runs.
#
# This must be submitted AFTER steps 1 and 3 are complete.
#
# Grid: 6 algs × 10 sets × 4 robot counts × 5 seeds = 1200 jobs
# Both HP tags (default and tuned) use the same array size.
#
# Submit each section individually:
#   sbatch --array=0-1199 eval_main.sh default
#   sbatch --array=0-1199 eval_main.sh tuned
#
# NOTE: The --array flag MUST be passed on the sbatch command line.
#       SLURM does not read #SBATCH directives from inside shell
#       conditionals, so there is no default array size set here.
# ============================================================

#SBATCH --job-name=eval_main
#SBATCH --output=/homes/choton/rl4pag/neurips_experiments/slurm_outputs/%x_%j.out
#SBATCH --error=/homes/choton/rl4pag/neurips_experiments/slurm_outputs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --export=NONE

HP_TAG=${1:-default}    # "default" or "tuned"  — pass as sbatch arg

algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO" "CrossQ")
sets=(1 2 3 4 5 6 7 8 9 10)
robots=(2 3 4 5)
seeds=(0 42 123 2024 9999)

num_algs=${#algorithms[@]}
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

OUT_CSV="/homes/choton/rl4pag/neurips_experiments/results/results_${HP_TAG}.csv"

echo "eval_main | alg=$algorithm | set=$set | robots=$num_robots_value | seed=$seed | hp=$HP_TAG"

conda run --no-capture-output -n robot_env python3 evaluate.py \
    --algorithm  $algorithm \
    --set        $set \
    --num_robots $num_robots_value \
    --seed       $seed \
    --experiment main \
    --hp_tag     $HP_TAG \
    --log_root   logs \
    --output_csv $OUT_CSV \
    --n_eval_eps 50

wait