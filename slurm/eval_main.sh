#!/bin/bash
# ============================================================
# eval_main.sh — Run evaluate.py for all main-experiment runs.
#
# This must be submitted AFTER the corresponding training step is complete.
# Raw result CSVs include terminal episode metrics from info["episode_metrics"].
#
# Default/tuned grid: 6 algs × 10 sets × 4 robot counts × 5 seeds = 1200 jobs
# Transfer grid:      6 algs ×  9 sets × 4 robot counts × 5 seeds = 1080 jobs
#
# Submit each section individually:
#   sbatch --array=0-1199 eval_main.sh default
#   sbatch --array=0-1199 eval_main.sh tuned
#   sbatch --array=0-1079 eval_main.sh transfer
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
#SBATCH --export=ALL

set -euo pipefail

HP_TAG=${1:-default}    # "default", "tuned", or "transfer"  — pass as sbatch arg
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/train.py" ] || [ -f "$SCRIPT_DIR/tune.py" ] || [ -f "$SCRIPT_DIR/evaluate.py" ]; then
    DEFAULT_PROJECT_ROOT="$SCRIPT_DIR"
else
    DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
cd "$PROJECT_ROOT"

if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found. Load conda or set PATH before sbatch." >&2
        exit 1
    fi
fi

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
mkdir -p "$LOG_ROOT" "$RESULTS_DIR"

algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO" "CrossQ")
if [ "$HP_TAG" == "transfer" ]; then
    sets=(2 3 4 5 6 7 8 9 10)
    pretrain_steps=2000000
    finetune_steps=2000000
    hparam_search_steps_per_algorithm=0
elif [ "$HP_TAG" == "tuned" ]; then
    sets=(1 2 3 4 5 6 7 8 9 10)
    pretrain_steps=0
    finetune_steps=2000000
    hparam_search_steps_per_algorithm=$((50 * 500000))
else
    sets=(1 2 3 4 5 6 7 8 9 10)
    pretrain_steps=0
    finetune_steps=2000000
    hparam_search_steps_per_algorithm=0
fi
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
    --pretrain_steps $pretrain_steps \
    --finetune_steps $finetune_steps \
    --hparam_search_steps_per_algorithm $hparam_search_steps_per_algorithm \
    --log_root   "$LOG_ROOT" \
    --output_csv "$OUT_CSV" \
    --n_eval_eps 50 \
    --device     cpu
