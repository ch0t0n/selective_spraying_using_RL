#!/bin/bash
# ============================================================
# eval_ablations.sh — SAFE version (per-job CSVs, no conflicts)
#
# Covers experiments that require separate evaluation passes:
#   ablation_reward       — 50-ep eval for terminal-condition tracking
#   ablation_uncertainty  — cross-evaluation matrix (train × eval noise)
#   dr                    — in-distribution and OOD wind evaluation
#
# Note: ablation_obs results are still read directly from training
# evaluations.npz files by analyze_results.py; no separate eval
# pass is needed for that experiment.
#
# Changes vs previous version
# ----------------------------
#   NEW: ablation_reward block added.  evaluate.py now supports
#     --experiment ablation_reward and writes sprayed_pct,
#     collision_pct, and max_steps_pct columns to the CSV.
#     analyze_results.py prefers this CSV (results/ablation_reward.csv)
#     over the training NPZ fallback to report dominant terminal
#     condition in tab:ablation_reward.
#     Grid: 4 conditions × 10 sets × 5 seeds = 200 jobs.
#   REQ (2) + BUG FIX: ablation_uncertainty block now iterates over
#     all 10 env sets (was hardcoded to --set 1).  Index math extended
#     with a set_idx dimension.  Array size updated from 0-79 to 0-799
#     (4 train × 4 eval × 10 sets × 5 seeds = 800 jobs).
#   REQ (3) / evaluate.py: evaluate.py itself skips the run when the
#     per-job output CSV already contains a matching row, so re-submitting
#     failed array indices is safe with no double-appending.
#
# Submit:
#   sbatch --array=0-199 eval_ablations.sh ablation_reward
#   sbatch --array=0-799 eval_ablations.sh ablation_uncertainty
#   sbatch --array=0-599 eval_ablations.sh dr
# ============================================================
 
#SBATCH --job-name=eval_ablation
#SBATCH --output=logs/slurm_outputs/eval_ablations/%x_%j.out
#SBATCH --error=logs/slurm_outputs/eval_ablations/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --export=NONE

EXPERIMENT=${1:-ablation_uncertainty}

# Per-experiment output root
OUT_ROOT="logs/results/tmp/${EXPERIMENT}"
mkdir -p "$OUT_ROOT"

seeds=(0 42 123 2024 9999)
num_seeds=${#seeds[@]}
index=$SLURM_ARRAY_TASK_ID

if [ "$EXPERIMENT" == "ablation_reward" ]; then
    conditions=("full" "no_term" "no_spr" "no_path")
    sets=(1 2 3 4 5 6 7 8 9 10)

    num_conditions=${#conditions[@]}
    num_sets=${#sets[@]}

    seed_idx=$(( index % num_seeds ))
    set_idx=$(( (index / num_seeds) % num_sets ))
    cond_idx=$(( index / (num_seeds * num_sets) ))

    condition=${conditions[$cond_idx]}
    set=${sets[$set_idx]}
    seed=${seeds[$seed_idx]}

    OUT_DIR="${OUT_ROOT}/${condition}/set${set}"
    mkdir -p "$OUT_DIR"
    OUT_CSV="${OUT_DIR}/result_${index}.csv"

    echo "eval | ablation_reward | condition=$condition | set=$set | seed=$seed"
    echo "Output: $OUT_CSV"

    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --set $set --num_robots 3 --seed $seed \
        --experiment ablation_reward --ablation $condition \
        --output_csv $OUT_CSV --n_eval_eps 50

# ============================================================
# ── Ablation: uncertainty ────────────────────────────────────
#
# REQ (2) + BUG FIX: now iterates over all 10 env sets.
#
# Grid: 4 train_modes × 4 eval_modes × 10 sets × 5 seeds = 800 jobs
# Submit: sbatch --array=0-799 eval_ablations.sh ablation_uncertainty
#
# Index layout (innermost → outermost):
#   seed_idx   = index % 5
#   set_idx    = (index / 5)  % 10
#   train_idx  = (index / 50) % 4
#   eval_idx   = index / 200
# ============================================================
elif [ "$EXPERIMENT" == "ablation_uncertainty" ]; then
    train_modes=("full" "wind_only" "act_only" "deterministic")
    eval_modes=("full" "wind_only" "act_only" "deterministic")
    sets=(1 2 3 4 5 6 7 8 9 10)

    num_sets=${#sets[@]}

    seed_idx=$(( index % num_seeds ))
    set_idx=$(( (index / num_seeds) % num_sets ))
    train_idx=$(( (index / (num_seeds * num_sets)) % 4 ))
    eval_idx=$(( index / (num_seeds * num_sets * 4) ))

    train_mode=${train_modes[$train_idx]}
    eval_mode=${eval_modes[$eval_idx]}
    seed=${seeds[$seed_idx]}
    set=${sets[$set_idx]}

    OUT_DIR="${OUT_ROOT}/train_${train_mode}_eval_${eval_mode}/set${set}"
    mkdir -p "$OUT_DIR"
    OUT_CSV="${OUT_DIR}/result_${index}.csv"

    echo "eval | ablation_uncertainty | train=$train_mode | eval=$eval_mode | set=$set | seed=$seed"
    echo "Output: $OUT_CSV"

    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --set $set --num_robots 3 --seed $seed \
        --experiment ablation_uncertainty --ablation $train_mode \
        --eval_uncertainty_mode $eval_mode \
        --output_csv $OUT_CSV --n_eval_eps 50

# ============================================================
# ── Domain randomization ─────────────────────────────────────
#
# Grid: 3 dr_modes × 10 sets × 4 robots × 5 seeds = 600 jobs
# Submit: sbatch --array=0-599 eval_ablations.sh dr
# ============================================================
elif [ "$EXPERIMENT" == "dr" ]; then

    dr_modes=("none" "wind" "full")
    sets=(1 2 3 4 5 6 7 8 9 10)
    robots=(2 3 4 5)

    num_sets=${#sets[@]}
    num_robots=${#robots[@]}

    seed_idx=$(( index % num_seeds ))
    robot_idx=$(( (index / num_seeds) % num_robots ))
    set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
    dr_idx=$(( index / (num_seeds * num_robots * num_sets) ))

    dr_mode=${dr_modes[$dr_idx]}
    set=${sets[$set_idx]}
    num_robots_value=${robots[$robot_idx]}
    seed=${seeds[$seed_idx]}

    OUT_DIR="${OUT_ROOT}/${dr_mode}/set${set}_N${num_robots_value}"
    mkdir -p "$OUT_DIR"

    echo "eval | dr | mode=$dr_mode | set=$set | robots=$num_robots_value | seed=$seed"

    # In-distribution
    OUT_CSV_IN="${OUT_DIR}/inDist_${index}.csv"
    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --set $set --num_robots $num_robots_value --seed $seed \
        --experiment dr --ablation $dr_mode \
        --eval_wind_min 0.0 --eval_wind_max 0.5 \
        --output_csv $OUT_CSV_IN --n_eval_eps 50

    # OOD
    OUT_CSV_OOD="${OUT_DIR}/OOD_${index}.csv"
    /homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
        --algorithm  CrossQ --set $set --num_robots $num_robots_value --seed $seed \
        --experiment dr --ablation $dr_mode \
        --eval_wind_min 0.5 --eval_wind_max 2.0 \
        --output_csv $OUT_CSV_OOD --n_eval_eps 50

else
    echo "ERROR: unknown experiment '${EXPERIMENT}'."
    echo "Usage: sbatch --array=0-199 eval_ablations.sh ablation_reward"
    echo "       sbatch --array=0-799 eval_ablations.sh ablation_uncertainty"
    echo "       sbatch --array=0-599 eval_ablations.sh dr"
    exit 1
fi

wait