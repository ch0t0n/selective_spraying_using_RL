#!/bin/bash
# ============================================================
# eval_wind_sweep.sh — Wind sensitivity sweep for Figure 4.
#
# Evaluates CrossQ under 10 wind-speed bands for both standard
# and DR-trained policies.  Run AFTER step8_dr.sh completes.
# Raw result CSVs include terminal episode metrics from info["episode_metrics"].
#
# Grid: 2 dr_modes × 10 wind bins × 5 seeds = 100 jobs
# ============================================================

#SBATCH --array=0-99
#SBATCH --job-name=eval_wind_sweep
#SBATCH --output=logs/slurm_outputs/eval_wind_sweep/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/eval_wind_sweep/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --export=ALL

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
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
mkdir -p "$LOG_ROOT" "$RESULTS_DIR"
eval_dr_mode="${EVAL_DR_MODE:-none}"

seeds=(0 42 123 2024 9999)
dr_modes=("none" "full")
# 10 equally-spaced wind bins spanning [0, 2.0] m/s
wind_mins=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8)
wind_maxs=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

num_seeds=${#seeds[@]}
num_dr=${#dr_modes[@]}
num_bins=${#wind_mins[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
bin_idx=$(( (index / num_seeds) % num_bins ))
dr_idx=$(( index / (num_seeds * num_bins) ))

seed=${seeds[$seed_idx]}
dr_mode=${dr_modes[$dr_idx]}
wind_min=${wind_mins[$bin_idx]}
wind_max=${wind_maxs[$bin_idx]}

OUT_CSV="$RESULTS_DIR/wind_sweep.csv"

echo "wind_sweep | dr_mode=$dr_mode | eval_dr=$eval_dr_mode | wind=[$wind_min,$wind_max] | seed=$seed"

conda run --no-capture-output -n robot_env python3 evaluate.py \
    --algorithm      CrossQ \
    --set            1 \
    --num_robots     3 \
    --seed           $seed \
    --experiment     dr \
    --ablation       $dr_mode \
    --eval_dr_mode   $eval_dr_mode \
    --eval_wind_min  $wind_min \
    --eval_wind_max  $wind_max \
    --freeze_eval_wind_noise \
    --log_root       "$LOG_ROOT" \
    --output_csv     "$OUT_CSV" \
    --n_eval_eps     50 \
    --device         cpu
