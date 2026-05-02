#!/bin/bash
# ============================================================
# Step 2 — Transfer Learning, Others (CPU)
#
# Uses Step 1 default-policy checkpoints from Set 1 as the
# source models, then fine-tunes on target Sets 2–10.
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
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --export=NONE

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/beocat_env.sh"

algorithms=("A2C" "ARS" "PPO" "TQC" "TRPO")
sets=(2 3 4 5 6 7 8 9 10)
robots=(2 3 4 5)
seeds=(0 42 123 2024 9999)

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

TRANSFER_FROM="logs/main_default/${algorithm}_N${num_robots_value}_env1_seed${seed}/best_model/best_model.zip"

echo "S2-others-transfer | alg=$algorithm | set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"
echo "Transfer source: $TRANSFER_FROM"

"$PYTHON_BIN" train.py \
    --algorithm      $algorithm \
    --set            $set \
    --num_robots     $num_robots_value \
    --seed           $seed \
    --steps          $steps \
    --experiment     main \
    --transfer_from  $TRANSFER_FROM \
    --verbose        1 \
    --log_steps      10000

wait
