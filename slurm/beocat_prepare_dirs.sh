#!/bin/bash
# Create directories required before sbatch opens --output/--error files.
# Note: This creates extra folders for step 3 (tuning) depending on the Slurm script you run
set -euo pipefail

mkdir -p \
  logs \
  logs/optuna_studies \
  logs/results/tmp \
  logs/slurm_outputs/s1_crossq_default \
  logs/slurm_outputs/s1_others_default \
  logs/slurm_outputs/s2_crossq_transfer \
  logs/slurm_outputs/s2_others_transfer \
  logs/slurm_outputs/s3_tune \
  logs/slurm_outputs/s3_crossq_tune \
  logs/slurm_outputs/s3_others_tune \
  logs/slurm_outputs/s4_crossq_tuned \
  logs/slurm_outputs/s4_others_tuned \
  logs/slurm_outputs/s5_6_7_ablations \
  logs/slurm_outputs/s8_dr \
  logs/slurm_outputs/eval_ablations \
  logs/slurm_outputs/eval_wind_sweep \
  logs/slurm_outputs/step9_sens \
  logs/slurm_errors/s1_crossq_default \
  logs/slurm_errors/s1_others_default \
  logs/slurm_errors/s2_crossq_transfer \
  logs/slurm_errors/s2_others_transfer \
  logs/slurm_errors/s3_tune \
  logs/slurm_errors/s3_crossq_tune \
  logs/slurm_errors/s3_others_tune \
  logs/slurm_errors/s4_crossq_tuned \
  logs/slurm_errors/s4_others_tuned \
  logs/slurm_errors/s5_6_7_ablations \
  logs/slurm_errors/s8_dr \
  logs/slurm_errors/step9_sens \
  figures

echo "Beocat output/result directories are ready."
