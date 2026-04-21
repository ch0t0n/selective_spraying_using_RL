#!/usr/bin/env python3
"""
tune.py — Optuna distributed tuning (HPC-safe version)

Design:
- Each SLURM job = 1 Optuna worker
- Workers share a single study via JournalStorage (append-only log file)
- JournalStorage is safe on NFS/Lustre/GPFS shared filesystems;
  SQLite is NOT (file-locking on network mounts corrupts the database)
- Optuna handles trial scheduling
"""

import os
import json
import argparse
import time
import fcntl
import numpy as np

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiRobotEnv
from src.utils import load_experiment_dict_json, set_global_seeds


# ================================================================
# CONSTANTS
# ================================================================

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(PROJECT_ROOT, 'exp_sets', 'stochastic_envs_v2.json')
NUM_ENVS   = 4
NUM_ROBOTS = 3
MAX_STEPS  = 1000
ENV_VAR    = 1
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
# ARG PARSING
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm",   required=True, choices=list(ALGORITHMS))
    p.add_argument("--device",      default="cpu")
    p.add_argument("--n_trials",    type=int, default=1)
    p.add_argument("--tune_steps",  type=int, default=500_000)
    p.add_argument("--storage",     required=True,
                   help="Path to the journal log file, e.g. "
                        "logs/optuna_studies/CrossQ_journal.log")
    p.add_argument("--study_name",  required=True)
    p.add_argument("--output_json", default=os.path.join(PROJECT_ROOT, "logs", "best_hyperparams.json"))
    p.add_argument("--log_root",    default=os.path.join(PROJECT_ROOT, "logs", "step2_tune"))
    p.add_argument("--set",         type=int, default=ENV_VAR)
    p.add_argument("--num_robots",  type=int, default=NUM_ROBOTS)
    p.add_argument("--tune_seed",   type=int, default=TUNE_SEED)
    return p.parse_args()


# ================================================================
# METRIC
# ================================================================

def compute_iqm(rewards: np.ndarray) -> float:
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))


# ================================================================
# SAMPLERS
# ================================================================

def sample_a2c(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda",    0.90, 1.00),
        "vf_coef":       trial.suggest_float("vf_coef",       0.20, 0.70),
        "ent_coef":      trial.suggest_float("ent_coef",      0.00, 0.05),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.30, 0.99),
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


def sample_crossq(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "buffer_size":   trial.suggest_int(  "buffer_size",   1_000, 50_000),
        "batch_size":    trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }


def sample_tqc(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "buffer_size":   trial.suggest_int(  "buffer_size",   1_000, 50_000),
        "batch_size":    trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "tau":           trial.suggest_float("tau",            1e-3, 5e-2),
        "top_quantiles_to_drop_per_net":
                         trial.suggest_int("top_quantiles_to_drop_per_net", 0, 5),
    }


def sample_trpo(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "gae_lambda":    trial.suggest_float("gae_lambda",    0.90, 1.00),
        "target_kl":     trial.suggest_float("target_kl",     1e-3, 5e-2),
        "cg_max_steps":  trial.suggest_int(  "cg_max_steps",  5,    20),
    }


def sample_ars(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "delta_std":     trial.suggest_float("delta_std",     0.01, 0.30),
        "n_delta":       trial.suggest_int(  "n_delta",       8,    64),
    }


SAMPLERS = {
    "A2C":    sample_a2c,
    "PPO":    sample_ppo,
    "CrossQ": sample_crossq,
    "TQC":    sample_tqc,
    "TRPO":   sample_trpo,
    "ARS":    sample_ars,
}


def update_best_hyperparams(output_json: str, alg_name: str, best_trial, context: dict) -> None:
    """Update the shared best-HP JSON under a file lock."""
    output_dir = os.path.dirname(os.path.abspath(output_json))
    os.makedirs(output_dir, exist_ok=True)
    lock_path = f"{output_json}.lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            best_all = {}
            if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
                with open(output_json) as f:
                    best_all = json.load(f)

            previous = best_all.get(alg_name)
            previous_iqm = float(previous.get("iqm", "-inf")) if isinstance(previous, dict) else float("-inf")
            if best_trial.value >= previous_iqm:
                best_all[alg_name] = {
                    "iqm": best_trial.value,
                    "params": best_trial.params,
                    "context": context,
                }

            tmp_path = f"{output_json}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(best_all, f, indent=2)
            os.replace(tmp_path, output_json)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# ================================================================
# OBJECTIVE
# ================================================================

def make_objective(alg_name, AlgClass, policy, env_kwargs, device, tune_steps, tune_seed):

    def objective(trial):
        params = SAMPLERS[alg_name](trial)

        vec_env = make_vec_env(
            "MultiRobotEnv-v0",
            env_kwargs=env_kwargs,
            n_envs=NUM_ENVS,
            seed=tune_seed,
        )

        eval_env = make_vec_env(
            "MultiRobotEnv-v0",
            env_kwargs=env_kwargs,
            n_envs=1,
            seed=tune_seed + 1,
        )

        model = None
        try:
            model = AlgClass(
                policy,
                vec_env,
                device=device,
                verbose=0,
                seed=tune_seed,
                **params,
            )

            model.learn(total_timesteps=tune_steps)

            ep_r, _ = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=N_EVAL_EPS,
                deterministic=True,
                return_episode_rewards=True,
            )

            score = compute_iqm(np.array(ep_r, dtype=np.float32))
            print(f"[trial {trial.number}] IQM={score:.3f} | {params}")
            return score

        except Exception as e:
            print(f"[trial {trial.number}] FAILED: {e}")
            return float("-inf")

        finally:
            vec_env.close()
            eval_env.close()
            if model is not None:
                del model

    return objective


# ================================================================
# SAFE STUDY CREATION
#
# JournalStorage + JournalFileBackend replaces RDBStorage/SQLite.
#
# Why JournalStorage is safe on HPC shared filesystems:
#   - Writes are append-only: no in-place mutation means no
#     corruption window if two workers write simultaneously.
#   - Uses fcntl advisory locking (not SQLite's POSIX locks),
#     which NFS/Lustre honour correctly.
#   - The entire study state is reconstructed by replaying the
#     log on read, so a partial write never corrupts prior trials.
#
# --storage is now a plain file path (not a sqlite:/// URL).
# ================================================================

def create_study_safe(args):
    for attempt in range(10):
        try:
            storage = JournalStorage(
                JournalFileBackend(args.storage)
            )
            return optuna.create_study(
                direction="maximize",
                study_name=args.study_name,
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=args.tune_seed),
                pruner=optuna.pruners.MedianPruner(),
            )
        except Exception as e:
            wait = 2 ** attempt
            print(f"Study init retry {attempt + 1}/10 (wait {wait}s): {e}")
            time.sleep(wait)
    raise RuntimeError("Failed to create Optuna study after 10 retries")


# ================================================================
# MAIN
# ================================================================

def run_tuning(args):
    set_global_seeds(args.tune_seed)
    os.makedirs(args.log_root, exist_ok=True)

    json_dict = load_experiment_dict_json(JSON_PATH)
    env_kwargs = dict(
        field_info=json_dict[f"set{args.set}"],
        num_robots=args.num_robots,
        max_steps=MAX_STEPS,
        render_mode=None,
    )

    try:
        gym.register(
            id="MultiRobotEnv-v0",
            entry_point=MultiRobotEnv,
            max_episode_steps=MAX_STEPS,
        )
    except Exception:
        pass

    alg_name = args.algorithm
    AlgClass, policy = ALGORITHMS[alg_name]

    print(f"Algorithm : {alg_name}")
    print(f"Device    : {args.device}")
    print(f"Storage   : {args.storage}")
    print(f"Study     : {args.study_name}")

    study = create_study_safe(args)

    study.optimize(
        make_objective(
            alg_name, AlgClass, policy,
            env_kwargs, args.device, args.tune_steps, args.tune_seed,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"BEST (so far): IQM={best.value:.4f} | {best.params}")

    context = {
        "set": args.set,
        "num_robots": args.num_robots,
        "tune_seed": args.tune_seed,
        "tune_steps": args.tune_steps,
    }
    update_best_hyperparams(args.output_json, alg_name, best, context)

    print(f"Updated → {args.output_json}")


if __name__ == "__main__":
    run_tuning(parse_args())