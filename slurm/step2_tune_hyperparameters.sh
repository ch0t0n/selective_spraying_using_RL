#!/bin/bash
# ============================================================
# Step 2 — Hyperparameter tuning (Optuna, fully parallelised)
#
# One SLURM job per trial per algorithm.
# All jobs for the same algorithm share one JournalStorage log
# file — append-only writes make it safe on NFS/Lustre/GPFS.
# (SQLite is NOT safe on HPC shared filesystems.)
#
# Grid: 6 algorithms × 50 trials = 300 jobs  →  array=0-299
#
# Index layout:
#   alg_idx   = index // 50        (0–5)
#   trial_idx = index  % 50        (0–49, for logging only)
#
# Algorithm order:
#   0 → A2C    (cpu)
#   1 → ARS    (cpu)
#   2 → PPO    (cpu)
#   3 → TRPO   (cpu)
#   4 → CrossQ (cuda)
#   5 → TQC    (cuda)
# ============================================================

#SBATCH --array=0-299
#SBATCH --job-name=s2_tune
#SBATCH --output=logs/slurm_outputs/s2_tune/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s2_tune/%x_%A_%a.err
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

# ── Paths ──────────────────────────────────────────────────────────
BEST_JSON="logs/best_hyperparams.json"
JOURNAL_DIR="logs/optuna_studies"
mkdir -p "$JOURNAL_DIR"

# ── Algorithm table ────────────────────────────────────────────────
algorithms=("A2C" "ARS" "PPO" "TRPO" "CrossQ" "TQC")
devices=("cpu" "cpu" "cpu" "cpu" "cuda" "cuda")

# ── Decode index ───────────────────────────────────────────────────
alg_idx=$(( SLURM_ARRAY_TASK_ID / 50 ))
trial_idx=$(( SLURM_ARRAY_TASK_ID % 50 ))

algorithm=${algorithms[$alg_idx]}
device=${devices[$alg_idx]}

# Plain file path — not a sqlite:/// URL
storage="${JOURNAL_DIR}/${algorithm}_journal.log"

echo "S2-tune | alg=${algorithm} | trial=${trial_idx} | device=${device} | job=${SLURM_ARRAY_TASK_ID}"

/homes/choton/miniconda3/envs/robot_env/bin/python tune.py \
    --algorithm   "$algorithm" \
    --device      "$device" \
    --n_trials    1 \
    --tune_steps  2000000 \
    --storage     "$storage" \
    --study_name  "${algorithm}_tune" \
    --output_json "$BEST_JSON"

wait