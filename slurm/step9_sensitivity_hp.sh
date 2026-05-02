#!/bin/bash
# ============================================================
# step9_sensitivity_hp.sh — Hyperparameter sensitivity (Table 6)
#
# One job per algorithm.
# Each job sweeps ALL hyperparameters and grid points.
#
# Total jobs: 6  →  array=0-5
#
# After all jobs finish:
#   python sensitivity_hp.py --write_latex_only --results_dir logs/results
# ============================================================

#SBATCH --array=0-5
#SBATCH --job-name=s9_sensitivity_hp
#SBATCH --output=logs/slurm_outputs/step9_sens/%x_%j.out
#SBATCH --error=logs/slurm_errors/step9_sens/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/beocat_env.sh"

algorithms=("A2C" "ARS" "PPO" "TRPO" "CrossQ" "TQC")

index=$SLURM_ARRAY_TASK_ID
algorithm=${algorithms[$index]}

# Device logic
if [[ "$algorithm" == "CrossQ" || "$algorithm" == "TQC" ]]; then
    device="cuda"
else
    device="cpu"
fi

echo "S9-sensitivity | alg=$algorithm | job=$SLURM_ARRAY_TASK_ID"

"$PYTHON_BIN" sensitivity_hp.py \
    --algorithm   $algorithm \
    --results_dir logs/results \
    --device      $device

wait
