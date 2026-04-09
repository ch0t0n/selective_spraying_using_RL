#!/usr/bin/env python3
"""
tune.py — Step 2: Optuna hyperparameter tuning.

Runs N_TRIALS per algorithm (0.5 M timesteps each) on env
variation 1, N = 3 robots.  Writes best_hyperparams.json which
is consumed by Step 3 via  train.py --hyperparams_json <path>.

One SLURM job per algorithm (see step2_tune.sh):
  python tune.py --algorithm CrossQ --device cuda
  python tune.py --algorithm PPO    --device cpu

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime

import optuna
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiRobotEnv
from src.utils import load_experiment_dict_json

# ================================================================
# Constants
# ================================================================

JSON_PATH  = os.path.join('exp_sets', 'stochastic_envs_v2.json')
NUM_ENVS   = 4
NUM_ROBOTS = 3
MAX_STEPS  = 1000
ENV_VAR    = 1      # always tune on variation 1
TUNE_SEED  = 42
N_EVAL_EPS = 10

ALGORITHMS = {
    "A2C":    (A2C,    "MlpPolicy"),
    "ARS":    (ARS,    "LinearPolicy"),
    "PPO":    (PPO,    "MlpPolicy"),
    "TRPO":   (TRPO,   "MlpPolicy"),
    "CrossQ": (CrossQ, "MlpPolicy"),
    "TQC":    (TQC,    "MlpPolicy"),
}

# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm",   type=str, required=True,
                   choices=list(ALGORITHMS.keys()))
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--n_trials",    type=int, default=20)
    p.add_argument("--tune_steps",  type=int, default=int(5e5))
    p.add_argument("--output_json", type=str,
                   default=os.path.join('logs', 'best_hyperparams.json'))
    p.add_argument("--log_root",    type=str,
                   default=os.path.join('logs', 'step2_tune'))
    return p.parse_args()

# ================================================================
# IQM helper
# ================================================================

def compute_iqm(rewards):
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))

# ================================================================
# Search-space samplers (ranges consistent across algorithms)
# ================================================================

def sample_a2c(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda",    0.90, 1.00),
        "vf_coef":       trial.suggest_float("vf_coef",       0.20, 0.70),
        "ent_coef":      trial.suggest_float("ent_coef",      0.00, 0.05),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.30, 0.99),
    }

def sample_ars(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "delta_std":     trial.suggest_float("delta_std",     0.01, 0.30),
        "n_delta":       trial.suggest_int(  "n_delta",       8,    64),
    }

def sample_ppo(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda",    0.90, 1.00),
        "vf_coef":       trial.suggest_float("vf_coef",       0.20, 0.70),
        "ent_coef":      trial.suggest_float("ent_coef",      0.00, 0.05),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.30, 0.99),
        "clip_range":    trial.suggest_float("clip_range",    0.10, 0.40),
        "n_epochs":      trial.suggest_int(  "n_epochs",      3,    20),
    }

def sample_trpo(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda",    0.90, 1.00),
        "target_kl":     trial.suggest_float("target_kl",     1e-3, 5e-2),
        "cg_max_steps":  trial.suggest_int(  "cg_max_steps",  5,    20),
    }

def sample_crossq(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "buffer_size":   trial.suggest_int(  "buffer_size",   1_000, 100_000),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        # "tau":           trial.suggest_float("tau",            1e-3, 5e-2),
    }

def sample_tqc(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "buffer_size":   trial.suggest_int(  "buffer_size",   1_000, 100_000),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "tau":           trial.suggest_float("tau",            1e-3, 5e-2),
        "top_quantiles_to_drop_per_net":
                         trial.suggest_int("top_quantiles_to_drop_per_net", 0, 5),
    }

SAMPLERS = {"A2C": sample_a2c, "ARS": sample_ars, "PPO": sample_ppo,
            "TRPO": sample_trpo, "CrossQ": sample_crossq, "TQC": sample_tqc}

# ================================================================
# Optuna objective
# ================================================================

def make_objective(alg_name, AlgClass, policy, env_kwargs, device, tune_steps):
    def objective(trial):
        params   = SAMPLERS[alg_name](trial)
        vec_env  = make_vec_env("MultiRobotEnv-v0", env_kwargs=env_kwargs,
                                n_envs=NUM_ENVS, seed=TUNE_SEED)
        eval_env = make_vec_env("MultiRobotEnv-v0", env_kwargs=env_kwargs,
                                n_envs=1, seed=TUNE_SEED + 1)
        score = float("-inf")
        try:
            model = AlgClass(policy, vec_env, verbose=0,
                             device=device, seed=TUNE_SEED, **params)
            model.learn(total_timesteps=tune_steps)
            ep_r, _ = evaluate_policy(model, eval_env,
                                      n_eval_episodes=N_EVAL_EPS,
                                      deterministic=True,
                                      return_episode_rewards=True)
            score = compute_iqm(np.array(ep_r, dtype=np.float32))
            print(f"  trial {trial.number:3d} | IQM={score:8.3f} | {params}")
        except Exception as e:
            print(f"  trial {trial.number:3d} FAILED: {e}")
        finally:
            vec_env.close()
            eval_env.close()
            if "model" in locals():
                del model
        return score
    return objective

# ================================================================
# Entry point
# ================================================================

def run_tuning(args):
    os.makedirs(args.log_root, exist_ok=True)

    json_dict  = load_experiment_dict_json(JSON_PATH)
    env_kwargs = dict(field_info=json_dict[f"set{ENV_VAR}"],
                      num_robots=NUM_ROBOTS, max_steps=MAX_STEPS, render_mode=None)

    if "MultiRobotEnv-v0" not in gym.envs.registry:
        gym.register(id="MultiRobotEnv-v0", entry_point=MultiRobotEnv,
                     max_episode_steps=MAX_STEPS)

    alg_name         = args.algorithm
    AlgClass, policy = ALGORITHMS[alg_name]

    print(f"\n{'='*60}")
    print(f"  Tuning : {alg_name}  |  {args.n_trials} trials × {args.tune_steps:,} steps")
    print(f"  Device : {args.device}  |  Output: {args.output_json}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{alg_name}_tune",
        sampler=optuna.samplers.TPESampler(seed=TUNE_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(
        make_objective(alg_name, AlgClass, policy,
                       env_kwargs, args.device, args.tune_steps),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest IQM = {best.value:.4f}")
    print(f"Best params: {best.params}")

    # Save per-algorithm study CSV
    study.trials_dataframe().to_csv(
        os.path.join(args.log_root, f"optuna_{alg_name}.csv"), index=False)

    # Update shared best_hyperparams.json (read-modify-write)
    best_all = {}
    json_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(json_dir, exist_ok=True)
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            best_all = json.load(f)
    best_all[alg_name] = {"iqm": best.value, "params": best.params}
    with open(args.output_json, "w") as f:
        json.dump(best_all, f, indent=2)
    print(f"Updated → {args.output_json}")


if __name__ == "__main__":
    run_tuning(parse_args())