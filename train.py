#!/usr/bin/env python3
"""
train.py — unified training script for all experiments.

Handles all 8 experiment steps via command-line arguments:
  Step 1  --experiment main     (default HPs)
  Step 2  --experiment main     --transfer_from <path>      (transfer learning)
  Step 4  --experiment main     --hyperparams_json <path>   (tuned HPs)
  Step 5  --experiment ablation_reward  --ablation {full|no_term|no_spr|no_path}
  Step 6  --experiment ablation_obs     --ablation {full|no_pos|no_inf_hist|pos_only}
  Step 7  --experiment ablation_uncertainty  --ablation {full|wind_only|act_only|deterministic}
  Step 8  --experiment dr       --ablation {none|wind|full}

Usage example (equivalent to old train_default.py call):
  python train.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42
                  --steps 1000000 --device cuda --experiment main
"""

import os
import csv
import json
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CallbackList, LogEveryNTimesteps)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiRobotEnv
from src.utils import load_experiment_dict_json, set_global_seeds

# ================================================================
# Constants
# ================================================================

ALGORITHMS = {
    "A2C":    (A2C,    "MlpPolicy"),
    "ARS":    (ARS,    "LinearPolicy"),
    "PPO":    (PPO,    "MlpPolicy"),
    "TRPO":   (TRPO,   "MlpPolicy"),
    "CrossQ": (CrossQ, "MlpPolicy"),
    "TQC":    (TQC,    "MlpPolicy"),
}

# Experiment → env constructor kwarg name
EXPERIMENT_MAP = {
    "main":                  None,               # no ablation kwarg
    "ablation_reward":       "reward_ablation",
    "ablation_obs":          "obs_mode",
    "ablation_uncertainty":  "uncertainty_mode",
    "dr":                    "dr_mode",
}

# Valid ablation choices per experiment — must match env.py assertions exactly
VALID_ABLATIONS = {
    "main":                  {None},
    "ablation_reward":       {"full", "no_term", "no_spr", "no_path"},
    "ablation_obs":          {"full", "no_pos", "no_inf_hist", "pos_only"},
    "ablation_uncertainty":  {"full", "wind_only", "act_only", "deterministic"},
    "dr":                    {"none", "wind", "full"},
}

# Default ablation value per experiment type (used when --ablation not given)
# BUG FIX: was "base" for ablation_obs — "base" is not a valid obs_mode in env.py
EXPERIMENT_DEFAULTS = {
    "main":                  None,
    "ablation_reward":       "full",
    "ablation_obs":          "full",        # fixed: was "base"
    "ablation_uncertainty":  "full",
    "dr":                    "none",
}

EPISODE_METRIC_KEYS = [
    "episode_step_count",
    "episode_spray_used",
    "episode_spray_wasted",
    "episode_spray_applied",
    "episode_wasted_fraction",
    "episode_spray_empty_capacity",
    "episode_boundary_viol_count",
    "episode_remaining_infection_sum",
    "episode_remaining_infection_fraction",
    "episode_time_limit_reached",
    "episode_collision_occurred",
    "episode_success",
]

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(PROJECT_ROOT, 'exp_sets', 'stochastic_envs_v2.json')

# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified RL training script")

    # ── original arguments (backward-compatible with train_default.py) ──
    p.add_argument("--algorithm",   type=str,   required=True,
                   choices=list(ALGORITHMS.keys()))
    p.add_argument("--set",         type=int,   required=True,
                   help="Environment variation index (1–10)")
    p.add_argument("--num_robots",  type=int,   default=3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--steps",       type=int,   default=int(1e6))
    p.add_argument("--verbose",     type=int,   default=1)
    p.add_argument("--log_steps",   type=int,   default=10_000)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--num_envs",    type=int,   default=4,
                   help="Number of parallel training envs")
    p.add_argument("--max_steps",   type=int,   default=1000,
                   help="Maximum steps per episode")
    p.add_argument("--n_eval_eps",  type=int,   default=5,
                   help="Episodes per EvalCallback evaluation")

    # ── new experiment-control arguments ──
    p.add_argument("--experiment",  type=str,   default="main",
                   choices=list(EXPERIMENT_MAP.keys()),
                   help="Which experiment this run belongs to")
    p.add_argument("--ablation",    type=str,   default=None,
                   help="Ablation condition within the selected experiment")
    p.add_argument("--hyperparams_json", type=str, default=None,
                   help="Path to best_hyperparams.json (Step 4 tuned HPs). "
                        "If omitted, SB3 defaults are used.")
    p.add_argument("--transfer_from", type=str, default=None,
                   help="Path to a Step 1 default-policy checkpoint for "
                        "Step 2 transfer learning. If omitted, training "
                        "starts from scratch.")
    p.add_argument("--log_root",    type=str,
                   default=os.path.join(PROJECT_ROOT, 'logs'),
                   help="Root directory for all logs and saved models")

    args = p.parse_args()

    # ── Validate --ablation against the chosen experiment ────────────
    # Fail fast here with a clear message rather than inside the env constructor
    effective_ablation = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
    valid = VALID_ABLATIONS[args.experiment]
    if effective_ablation not in valid:
        p.error(
            f"--ablation '{effective_ablation}' is not valid for "
            f"--experiment '{args.experiment}'. "
            f"Choose from: {sorted(v for v in valid if v is not None)}"
        )

    return args


# ================================================================
# Episode metric logging
# ================================================================

class EpisodeMetricsCallback(BaseCallback):
    """Write terminal episode metrics exposed by MultiRobotEnv to CSV."""

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._file = None
        self._writer = None

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        write_header = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0

        self._file = open(self.log_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(["num_timesteps", "env_index"] + EPISODE_METRIC_KEYS)
            self._file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_index, info in enumerate(infos):
            metrics = info.get("episode_metrics")
            if not metrics:
                continue

            self._writer.writerow(
                [self.num_timesteps, env_index] +
                [metrics.get(k, "") for k in EPISODE_METRIC_KEYS]
            )
            self._file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

# ================================================================
# Utilities
# ================================================================

def load_hyperparams(json_path: str, algorithm: str) -> dict:
    """Load tuned hyperparameters for one algorithm from JSON (Step 3 output)."""
    if json_path is None:
        return {}
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Hyperparameter JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)
    # if hyperparameters are missing, do NOT silently use defaults
    if algorithm not in data:
        raise KeyError(
            f"Algorithm '{algorithm}' not found in hyperparameter JSON: {json_path}"
        )
    if not isinstance(data[algorithm], dict) or "params" not in data[algorithm]:
        raise KeyError(
            f"Missing 'params' for algorithm '{algorithm}' in: {json_path}"
        )
    ctx = data[algorithm].get("context")
    if ctx:
        print(f"  HP tuning context for {algorithm}: {ctx}")
    hp = data[algorithm]["params"]
    if not isinstance(hp, dict):
        raise TypeError(
            f"Expected dict for '{algorithm}' params in {json_path}, got {type(hp).__name__}"
        )
    print(f"  Loaded tuned HPs for {algorithm}: {hp}")
    return hp


def build_env_kwargs(args, field_info: dict) -> dict:
    """Assemble the keyword-argument dict for MultiRobotEnv."""
    kwargs = dict(
        field_info=field_info,
        num_robots=args.num_robots,
        max_steps=args.max_steps,
        render_mode=None,
    )

    # Map the --ablation value to the correct env constructor kwarg
    kwarg_name = EXPERIMENT_MAP[args.experiment]
    if kwarg_name is not None:
        ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
        kwargs[kwarg_name] = ablation_val

    return kwargs


def build_log_dir(args) -> str:
    """
    Build a unique log directory that encodes all run parameters so
    different experiments never write to the same location.

    Structure:
        logs/{version}/{algorithm}_N{num_robots}_env{set}_seed{seed}/

    The seed is included in the leaf directory so that parallel jobs
    for the same (algorithm, N, env_set) never overwrite each other's
    best_model.zip or eval_logs/evaluations.npz.

    The date prefix has been removed: it caused jobs from the same
    SLURM array that crossed midnight to land in different parent
    directories, breaking the glob in evaluate.py.
    """
    if args.experiment == "main":
        if args.transfer_from is not None:
            hp_tag = "transfer"
        else:
            hp_tag = "tuned" if args.hyperparams_json else "default"
        version = f"main_{hp_tag}"
    elif args.experiment == "dr":
        version = f"dr_{args.ablation or 'none'}"
    else:
        # ablation_reward / ablation_obs / ablation_uncertainty
        version = f"{args.experiment}_{args.ablation or EXPERIMENT_DEFAULTS[args.experiment]}"

    tag = (f"{args.algorithm}_N{args.num_robots}"
           f"_env{args.set}_seed{args.seed}")   # ← seed added here

    return os.path.join(args.log_root, version, tag)


# ================================================================
# Main training function
# ================================================================

def train(args):
    set_global_seeds(args.seed)

    if args.transfer_from is not None:
        if args.experiment != "main":
            raise ValueError("--transfer_from is only supported for --experiment main")
        if args.hyperparams_json is not None:
            raise ValueError("--transfer_from and --hyperparams_json should not be used together")
        if not os.path.exists(args.transfer_from):
            raise FileNotFoundError(f"Transfer checkpoint not found: {args.transfer_from}")

    # ── Load environment config ──────────────────────────────────
    json_dict  = load_experiment_dict_json(JSON_PATH)
    field_info = json_dict[f"set{args.set}"]
    env_kwargs = build_env_kwargs(args, field_info)

    # ── Gym registration ─────────────────────────────────────────
    env_id = "MultiRobotEnv-v0"
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=MultiRobotEnv,
                     max_episode_steps=args.max_steps)

    # ── Vectorised training env ──────────────────────────────────
    vec_env = make_vec_env(
        env_id,
        env_kwargs=env_kwargs,
        n_envs=args.num_envs,
        seed=args.seed,
    )

    # ── Vectorised evaluation env ────────────────────────────────
    eval_vec = make_vec_env(
        env_id,
        env_kwargs=env_kwargs,
        n_envs=1,
        seed=args.seed + 1,
    )

    # ── Directories ──────────────────────────────────────────────
    log_dir = build_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"  Algorithm : {args.algorithm}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Ablation  : {args.ablation}")
    print(f"  Env set   : {args.set}  |  N robots: {args.num_robots}")
    print(f"  Seed      : {args.seed}")
    print(f"  Steps     : {args.steps:,}")
    if args.transfer_from is not None:
        print(f"  Transfer  : {args.transfer_from}")
    print(f"  Log dir   : {log_dir}")
    print("=" * 60)

    # ── Callbacks ────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(args.log_steps // args.num_envs, 1),
        n_eval_episodes=args.n_eval_eps,
        deterministic=True,
        render=False,
    )
    log_cb = LogEveryNTimesteps(n_steps=args.log_steps)
    metrics_cb = EpisodeMetricsCallback(os.path.join(log_dir, "episode_metrics.csv"))
    callback = CallbackList([log_cb, eval_cb, metrics_cb])

    # ── SB3 logger ───────────────────────────────────────────────
    logger = configure(log_dir, ["stdout", "log", "csv", "tensorboard"])

    # ── Hyperparameters ──────────────────────────────────────────
    hp = load_hyperparams(args.hyperparams_json, args.algorithm)

    # ── Model ────────────────────────────────────────────────────
    AlgClass, policy = ALGORITHMS[args.algorithm]
    if args.transfer_from is not None:
        model = AlgClass.load(
            args.transfer_from,
            env=vec_env,
            device=args.device,
        )
        if hasattr(model, "set_random_seed"):
            model.set_random_seed(args.seed)
        model.verbose = args.verbose
    else:
        model = AlgClass(
            policy, vec_env,
            verbose=args.verbose,
            device=args.device,
            seed=args.seed,
            **hp,
        )
    model.set_logger(logger)

    # ── Training ─────────────────────────────────────────────────
    print(f"\nTraining {args.algorithm} ...")
    model.learn(total_timesteps=args.steps, callback=callback)

    # ── Save final model ─────────────────────────────────────────
    save_path = os.path.join(
        log_dir,
        f"{args.algorithm}_N{args.num_robots}_env{args.set}",
    )
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")

    # ── Cleanup ──────────────────────────────────────────────────
    vec_env.close()
    eval_vec.close()
    del model


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)