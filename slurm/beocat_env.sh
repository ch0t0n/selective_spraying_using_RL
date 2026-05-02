#!/bin/bash
# Shared Beocat setup for Slurm jobs.
#
# Use:
#   cd /path/to/selective_spraying_using_RL
#   bash slurm/beocat_prepare_dirs.sh
#   sbatch slurm/step1_crossq_default.sh
#
# Each user edits only the PYTHON_BIN line below.

# Required: path to this user's Python inside the robot_env environment.

# PYTHON_BIN="/homes/choton/miniconda3/envs/robot_env/bin/python"
PYTHON_BIN="/homes/jameschapman/miniforge3/envs/robot_env/bin/python"

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ ! -f "$REPO_DIR/train.py" || ! -d "$REPO_DIR/src" ]]; then
    echo "ERROR: Submit from the repository root." >&2
    echo "Example:" >&2
    echo "  cd /path/to/selective_spraying_using_RL" >&2
    echo "  sbatch slurm/step1_crossq_default.sh" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable:" >&2
    echo "  $PYTHON_BIN" >&2
    echo "Edit slurm/beocat_env.sh and set PYTHON_BIN for your Beocat account." >&2
    exit 1
fi

cd "$REPO_DIR" || exit 1

export PROJECT_ROOT="$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHON_BIN

echo "Repo root: $PROJECT_ROOT"
echo "Python:    $PYTHON_BIN"
