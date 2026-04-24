#!/bin/bash
# ============================================================
# Step 7 — Ablation: Physical uncertainty model (CrossQ, GPU)
#
# Fixed: CrossQ, N = 3, env variation 1, 1,000,000 timesteps
# Grid:  4 uncertainty conditions × 5 seeds = 20 jobs
#
# uncertainty_mode:
#   0 → full          (wind + wind-direction + actuation + spray
#                       + observation + initial-position noise — default)
#   1 → wind_only     (only wind noise active)
#   2 → act_only      (only actuation noise active)
#   3 → deterministic (all noise sources disabled)
#
# This script only trains the four uncertainty-mode policies.
# Cross-evaluation under all train × eval uncertainty conditions
# is performed later by evaluate.py via eval_ablations.sh.
#
# Index layout:
#   seed_idx = index % 5
#   cond_idx = index // 5
# ============================================================

#SBATCH --array=0-19
#SBATCH --job-name=s7_ablation_uncertainty
#SBATCH --output=logs/slurm_outputs/s7_ablation_uncertainty/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s7_ablation_uncertainty/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
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

uncertainty_modes=("full" "wind_only" "act_only" "deterministic")
seeds=(0 42 123 2024 9999)

num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
cond_idx=$(( index / num_seeds ))

uncertainty_mode=${uncertainty_modes[$cond_idx]}
seed=${seeds[$seed_idx]}

echo "S7-ablation-uncertainty | mode=$uncertainty_mode | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

conda run --no-capture-output -n robot_env python3 train.py \
    --algorithm   "CrossQ" \
    --set         1 \
    --num_robots  3 \
    --seed        $seed \
    --steps       1000000 \
    --experiment  ablation_uncertainty \
    --ablation    $uncertainty_mode \
    --verbose     1 \
    --log_steps   10000 \
    --n_eval_eps   20 \
    --log_root     "$LOG_ROOT" \
    --device      cuda
