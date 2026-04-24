#!/bin/bash
# ============================================================
# Step 2 — Transfer learning, Others (CPU)
#
# Source: Step 1 default checkpoint on env set 1.
# Target: env sets 2–10, same algorithm, robot count, and seed.
#
# Grid: 5 algs × 9 target env sets × 4 robot counts × 5 seeds
# Total jobs: 900  →  array=0-899
#
# Index layout (innermost → outermost):
#   seed_idx  = index % 5
#   robot_idx = (index // 5) % 4
#   set_idx   = (index // 20) % 9
#   alg_idx   = index // 180
# ============================================================

#SBATCH --array=0-899
#SBATCH --job-name=s2_others_transfer
#SBATCH --output=logs/slurm_outputs/s2_others_transfer/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s2_others_transfer/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]

set -euo pipefail

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
mkdir -p "$LOG_ROOT"

algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO")
sets=(2 3 4 5 6 7 8 9 10)
robots=(2 3 4 5)
seeds=(0 42 123 2024 9999)
source_set=1

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

SOURCE_MODEL="$LOG_ROOT/main_default/${algorithm}_N${num_robots_value}_env${source_set}_seed${seed}/best_model/best_model.zip"

echo "S2-others-transfer | alg=$algorithm | source_set=$source_set | target_set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n robot_env python3 train.py \
    --algorithm     $algorithm \
    --set           $set \
    --num_robots    $num_robots_value \
    --seed          $seed \
    --steps         $steps \
    --experiment    main \
    --transfer_from "$SOURCE_MODEL" \
    --verbose       1 \
    --log_steps     10000 \
    --n_eval_eps     20 \
    --log_root       "$LOG_ROOT"
