#!/bin/bash

# Run a single experiment with: sbatch slurm_scripts/train_one.sh

#SBATCH --job-name=crossq_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

algorithm="CrossQ"
set=1
steps=100000
num_robots=2

conda run --no-capture-output -n rl4pag python3 train_default.py --algorithm $algorithm --set $set --verbose 1 --steps $steps --num_robots $num_robots --device "cuda"

wait