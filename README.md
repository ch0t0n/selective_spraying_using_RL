# Formal Environment Design with Correctness Guarantees for Multi-Robot Selective Spraying

This is the codebase for the paper "Formal Environment Design with Correctness Guarantees for Multi-Robot Selective Spraying". In this paper, we present a Reinforcement Learning (RL) solution for multi-robot systems used for spraying herbicide.  

## Setup  

It is recommended to run this codebase on Linux. The necessary packages and libraries needed to run the code are provided in the `environment.yaml` conda file. If you do not have conda installed on your machine, download it from [here](https://docs.anaconda.com/miniconda/miniconda-install/). Once it is installed, run the following command to set up the environment: 

```
conda env create -f environment.yaml
```

If you update the environment by installing or removing packages, please update the conda file with the following command:

```
conda env export --no-builds > environment.yaml
```


## Training 

### On your local machine  

To train an algorithm with the default configuration, run the following command:
```
python3 train_default.py --algorithm A2C --set 1
```
change the file to `train_random.py` if you want to train it using random hyperparameters. The `--algorithm` and `--set` are the required arguments for running this file. Other optional arguments are as follows:
```
--verbose # The verbosity level: 0 no output, 1 info, 2 debug
--steps # The number of steps to train the DRL model
--num_robots # Number of robots. Currently supports 2-5 robots.
--seed # A random number to use for seeding
--log_steps # The number of steps between each log entry
--resume # If true, loads an existing model to resume training. If false, trains a new model
--device # The device to train on (cpu or cuda)
```

### On Compute Clusters  

Slurm scripts for training the model are provided in the `slurm_scripts` directory. To run all non-GPU training experiments on 2 robots using default hyperparameters, use the command:
```
sbatch slurm_scripts/train_default_all_2_robots.sh
``` 
To run all GPU training experiments (currently just CrossQ) for 2 robots using the default hyperparameters, use this command: 
```
sbatch slurm_scripts/crossq_default_2_robots.sh
``` 
Use other scripts as needed. 
  
### Viewing Results  
Please check and run the jupyter notebook `\plotting\plot_results.ipynb` for plotting the training results. 
  
  
## Hyperparameter Tuning
Please check and run the jupyter notebook `hyperparameter_tuning\env1_all_optuna.ipynb` for finding random hyperparameters using Optuna.   

## Simulation  

### CoppeliaSim

First, you will need to download and install the CoppeliaSim robotics simulator from [here](https://coppeliarobotics.com/). Once it is installed, open the `simulation\sim_envs\coppeliasim_scene_for_spraying.ttt` scene file in the simulator. Then follow the instructions given in the jupyter notebook: `simulation\new_env_sim.ipynb`.  

**IMPORTANT:** You will need to reopen the scene each time before running the simulation. Never save changes to the scene file when closing.