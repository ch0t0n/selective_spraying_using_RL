#!/bin/bash
# ============================================================
# Step 3 — Hyperparameter tuning, CPU algorithms only.
#
# Submit with:
#   sbatch step3_tune_cpu.sh
#
# Grid: 4 CPU algorithms × 50 trials = 200 jobs → array=0-199
# ============================================================

#SBATCH --array=0-199
#SBATCH --job-name=s3_tune_cpu
#SBATCH --output=logs/slurm_outputs/s3_tune_cpu/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s3_tune_cpu/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --export=NONE

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
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
BEST_JSON="${BEST_JSON:-$LOG_ROOT/best_hyperparams.json}"
JOURNAL_DIR="${JOURNAL_DIR:-$LOG_ROOT/optuna_studies}"
mkdir -p "$JOURNAL_DIR"

algorithms=("A2C" "ARS" "PPO" "TRPO")

alg_idx=$(( SLURM_ARRAY_TASK_ID / 50 ))
trial_idx=$(( SLURM_ARRAY_TASK_ID % 50 ))
algorithm=${algorithms[$alg_idx]}
storage="${JOURNAL_DIR}/${algorithm}_journal.log"

echo "S3-tune-cpu | alg=${algorithm} | trial=${trial_idx} | job=${SLURM_ARRAY_TASK_ID}"

conda run --no-capture-output -n robot_env python3 tune.py \
    --algorithm   "$algorithm" \
    --device      cpu \
    --n_trials    1 \
    --tune_steps  500000 \
    --storage     "$storage" \
    --study_name  "${algorithm}_tune" \
    --output_json "$BEST_JSON" \
    --log_root    "$LOG_ROOT/step3_tune" \
    --set         1 \
    --num_robots  3 \
    --tune_seed   42
