#!/bin/bash
# ============================================================
# Step 3 — Main results, TUNED hyperparameters, Others (CPU)
#
# Grid: 5 algs × 10 env sets × 4 robot counts × 5 seeds
# Total jobs: 1000  →  array=0-999
# ============================================================

#SBATCH --array=0-999
#SBATCH --job-name=s3_others_tuned
#SBATCH --output=logs/slurm_outputs/s3_others_tuned/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s3_others_tuned/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --export=NONE

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/train.py" ] || [ -f "$SCRIPT_DIR/tune.py" ] || [ -f "$SCRIPT_DIR/evaluate.py" ]; then
    DEFAULT_PROJECT_ROOT="$SCRIPT_DIR"
else
    DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
cd "$PROJECT_ROOT"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"

algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO")
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

BEST_JSON="${BEST_JSON:-$LOG_ROOT/best_hyperparams.json}"

echo "S3-others-tuned | alg=$algorithm | set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

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
    --n_eval_eps        20 \
    --log_root          "$LOG_ROOT"

wait
