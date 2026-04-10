#!/bin/bash
# ============================================================
# eval_wind_sweep.sh — Wind sensitivity sweep for Figure 4.
#
# Evaluates CrossQ under 10 wind-speed bands for both standard
# and DR-trained policies.  Run AFTER step7_dr.sh completes.
#
# Grid: 2 dr_modes × 10 wind bins × 5 seeds = 100 jobs
# ============================================================

#SBATCH --array=0-99
#SBATCH --job-name=eval_wind_sweep
#SBATCH --output=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_outputs/%x_%j.out
#SBATCH --error=/homes/choton/rl4pag/selective_spraying_using_RL/slurm_outputs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --export=NONE

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

OUT_CSV="/homes/choton/rl4pag/selective_spraying_using_RL/results/wind_sweep.csv"

echo "wind_sweep | dr_mode=$dr_mode | wind=[$wind_min,$wind_max] | seed=$seed"

conda run --no-capture-output -n rl4pag python3 evaluate.py \
    --algorithm      CrossQ \
    --set            1 \
    --num_robots     3 \
    --seed           $seed \
    --experiment     dr \
    --ablation       $dr_mode \
    --eval_wind_min  $wind_min \
    --eval_wind_max  $wind_max \
    --output_csv     $OUT_CSV \
    --n_eval_eps     50

wait
