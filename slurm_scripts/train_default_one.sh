#!/bin/bash

# Run a single experiment with: sbatch slurm_scripts/train_one.sh

#SBATCH --job-name=RL4PAg_2_robots
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --export=NONE

algorithm="A2C"
set=1
steps=100000
num_robots=2

conda run --no-capture-output -n rl4pag python3 train_default.py --algorithm $algorithm --set $set --verbose 1 --steps $steps --num_robots $num_robots

wait