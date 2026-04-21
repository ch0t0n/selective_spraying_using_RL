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
#SBATCH --output=logs/slurm_outputs/eval_main/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/eval_main/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --export=NONE

HP_TAG=${1:-default}    # "default" or "tuned"  — pass as sbatch arg
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/train.py" ] || [ -f "$SCRIPT_DIR/tune.py" ] || [ -f "$SCRIPT_DIR/evaluate.py" ]; then
    DEFAULT_PROJECT_ROOT="$SCRIPT_DIR"
else
    DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
cd "$PROJECT_ROOT"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"


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

OUT_CSV="$RESULTS_DIR/results_${HP_TAG}.csv"

echo "eval_main | alg=$algorithm | set=$set | robots=$num_robots_value | seed=$seed | hp=$HP_TAG"

conda run --no-capture-output -n robot_env python3 evaluate.py \
    --algorithm  $algorithm \
    --set        $set \
    --num_robots $num_robots_value \
    --seed       $seed \
    --experiment main \
    --hp_tag     $HP_TAG \
    --log_root   "$LOG_ROOT" \
    --output_csv "$OUT_CSV" \
    --n_eval_eps 50 \
    --device     cpu

wait
