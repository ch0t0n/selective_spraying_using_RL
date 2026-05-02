#!/bin/bash
# ============================================================
# Step 3 — Hyperparameter tuning, CrossQ (GPU)
#
# One SLURM job per trial.
# All CrossQ jobs share one JournalStorage log file — append-only
# writes make it safe on NFS/Lustre/GPFS.
# (SQLite is NOT safe on HPC shared filesystems.)
#
# Grid: 1 algorithm × 50 trials = 50 jobs  →  array=0-49
#
# Index layout:
#   alg_idx   = index // 50        (always 0 here)
#   trial_idx = index  % 50        (0–49, for logging only)
#
# Algorithm order:
#   0 → CrossQ (cuda)
# ============================================================

#SBATCH --array=0-49
#SBATCH --job-name=s3_crossq_tune
#SBATCH --output=logs/slurm_outputs/s3_crossq_tune/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s3_crossq_tune/%x_%A_%a.err
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

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/beocat_env.sh"

# ── Paths ──────────────────────────────────────────────────────────
BEST_JSON="logs/best_hyperparams.json"
JOURNAL_DIR="logs/optuna_studies"
mkdir -p "$JOURNAL_DIR"

# ── Algorithm table ────────────────────────────────────────────────
algorithms=("CrossQ")
device="cuda"

# ── Decode index ───────────────────────────────────────────────────
index=$((SLURM_ARRAY_TASK_ID))
alg_idx=$(( index / 50 ))
trial_idx=$(( index % 50 ))

algorithm=${algorithms[$alg_idx]}

# Plain file path — not a sqlite:/// URL
storage="${JOURNAL_DIR}/${algorithm}_journal.log"

echo "S3-CrossQ-tune | alg=${algorithm} | trial=${trial_idx} | device=${device} | job=${SLURM_ARRAY_TASK_ID}"

"$PYTHON_BIN" tune.py \
    --algorithm   "$algorithm" \
    --device      "$device" \
    --n_trials    1 \
    --tune_steps  2000000 \
    --storage     "$storage" \
    --study_name  "${algorithm}_tune" \
    --output_json "$BEST_JSON"

wait
