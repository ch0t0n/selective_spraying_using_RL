#!/bin/bash
# ============================================================
# eval_ablations.sh — evaluate.py for ablation + DR experiments.
#
# Submit after steps 5, 6, 7, 8 are complete.
# Usage:
#   sbatch --array=0-19  eval_ablations.sh ablation_reward
#   sbatch --array=0-24  eval_ablations.sh ablation_obs
#   sbatch --array=0-79  eval_ablations.sh ablation_uncertainty
#   sbatch --array=0-599 eval_ablations.sh dr
# ============================================================

#SBATCH --job-name=eval_ablation
#SBATCH --output=logs/slurm_outputs/eval_ablation/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/eval_ablation/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --export=NONE

set -euo pipefail

EXPERIMENT=${1:-ablation_reward}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/train.py" ] || [ -f "$SCRIPT_DIR/tune.py" ] || [ -f "$SCRIPT_DIR/evaluate.py" ]; then
    DEFAULT_PROJECT_ROOT="$SCRIPT_DIR"
else
    DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
cd "$PROJECT_ROOT"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
seeds=(0 42 123 2024 9999)
num_seeds=${#seeds[@]}
index=$SLURM_ARRAY_TASK_ID

# ── Ablation: reward ──────────────────────────────────────────
if [ "$EXPERIMENT" == "ablation_reward" ]; then
    conditions=("full" "no_col" "no_cov" "no_eff")
    seed_idx=$(( index % num_seeds ))
    cond_idx=$(( index / num_seeds ))
    condition=${conditions[$cond_idx]}
    seed=${seeds[$seed_idx]}
    echo "eval | ablation_reward | condition=$condition | seed=$seed"
    conda run --no-capture-output -n robot_env python3 evaluate.py \
        --algorithm  CrossQ --set 1 --num_robots 3 --seed $seed \
        --experiment ablation_reward --ablation $condition \
        --eval_reward_ablation full \
        --log_root "$LOG_ROOT" \
        --output_csv "$RESULTS_DIR/ablation_reward.csv" --n_eval_eps 50 \
        --device cpu

# ── Ablation: observation ─────────────────────────────────────
elif [ "$EXPERIMENT" == "ablation_obs" ]; then
    obs_modes=("base" "full" "no_wind" "no_spray_hist" "pos_only")
    seed_idx=$(( index % num_seeds ))
    cond_idx=$(( index / num_seeds ))
    obs_mode=${obs_modes[$cond_idx]}
    seed=${seeds[$seed_idx]}
    echo "eval | ablation_obs | obs_mode=$obs_mode | seed=$seed"
    conda run --no-capture-output -n robot_env python3 evaluate.py \
        --algorithm  CrossQ --set 1 --num_robots 3 --seed $seed \
        --experiment ablation_obs --ablation $obs_mode \
        --log_root "$LOG_ROOT" \
        --output_csv "$RESULTS_DIR/ablation_obs.csv" --n_eval_eps 50 \
        --device cpu

# ── Ablation: uncertainty (cross-evaluation) ──────────────────
elif [ "$EXPERIMENT" == "ablation_uncertainty" ]; then
    # Train condition × Eval condition pairs
    # Index encodes: eval_cond_idx*20 + seed_idx*4 + train_cond_idx
    train_modes=("full" "wind_only" "act_only" "deterministic")
    eval_modes=("full" "wind_only" "act_only" "deterministic")
    seed_idx=$(( index % num_seeds ))
    train_idx=$(( (index / num_seeds) % 4 ))
    eval_idx=$(( index / (num_seeds * 4) ))
    train_mode=${train_modes[$train_idx]}
    eval_mode=${eval_modes[$eval_idx]}
    seed=${seeds[$seed_idx]}
    echo "eval | ablation_uncertainty | train=$train_mode | eval=$eval_mode | seed=$seed"
    conda run --no-capture-output -n robot_env python3 evaluate.py \
        --algorithm  CrossQ --set 1 --num_robots 3 --seed $seed \
        --experiment ablation_uncertainty --ablation $train_mode \
        --eval_uncertainty_mode $eval_mode \
        --log_root "$LOG_ROOT" \
        --output_csv "$RESULTS_DIR/ablation_uncertainty.csv" --n_eval_eps 50 \
        --device cpu

# ── Domain randomization ──────────────────────────────────────
elif [ "$EXPERIMENT" == "dr" ]; then
    dr_modes=("none" "wind" "full")
    sets=(1 2 3 4 5 6 7 8 9 10)
    robots=(2 3 4 5)
    num_dr=${#dr_modes[@]}
    num_sets=${#sets[@]}
    num_robots=${#robots[@]}
    eval_dr_mode="${EVAL_DR_MODE:-none}"
    seed_idx=$(( index % num_seeds ))
    robot_idx=$(( (index / num_seeds) % num_robots ))
    set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
    dr_idx=$(( index / (num_seeds * num_robots * num_sets) ))
    dr_mode=${dr_modes[$dr_idx]}
    set=${sets[$set_idx]}
    num_robots_value=${robots[$robot_idx]}
    seed=${seeds[$seed_idx]}
    echo "eval | dr | dr_mode=$dr_mode | eval_dr=$eval_dr_mode | set=$set | robots=$num_robots_value | seed=$seed"

    # DR training samples wind in [0.0, 1.0] m/s, so in-distribution covers that full range.
    # OOD starts at 1.0 m/s to avoid mixing seen and unseen wind speeds.
    # In-distribution evaluation (wind ∈ [0.0, 1.0] m/s)
    conda run --no-capture-output -n robot_env python3 evaluate.py \
        --algorithm  CrossQ --set $set --num_robots $num_robots_value --seed $seed \
        --experiment dr --ablation $dr_mode \
        --eval_dr_mode $eval_dr_mode \
        --eval_wind_min 0.0 --eval_wind_max 1.0 \
        --freeze_eval_wind_noise \
        --log_root "$LOG_ROOT" \
        --output_csv "$RESULTS_DIR/dr_inDist.csv" --n_eval_eps 50 \
        --device cpu

    # OOD evaluation (wind ∈ [1.0, 2.0] m/s)
    conda run --no-capture-output -n robot_env python3 evaluate.py \
        --algorithm  CrossQ --set $set --num_robots $num_robots_value --seed $seed \
        --experiment dr --ablation $dr_mode \
        --eval_dr_mode $eval_dr_mode \
        --eval_wind_min 1.0 --eval_wind_max 2.0 \
        --freeze_eval_wind_noise \
        --log_root "$LOG_ROOT" \
        --output_csv "$RESULTS_DIR/dr_OOD.csv" --n_eval_eps 50 \
        --device cpu
fi
