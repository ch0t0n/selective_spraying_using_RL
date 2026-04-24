#!/usr/bin/env python3
"""
evaluate.py — post-training evaluation for ALL experiment types.

Loads the best saved model for a given (algorithm, set, num_robots, seed,
experiment, ablation) combination, runs N_EVAL_EPISODES episodes, and
appends one row to a results CSV.

Changes vs previous version
----------------------------
  BUG FIX  : EXPERIMENT_DEFAULTS["ablation_obs"] was "base" (invalid obs_mode
              in env.py); corrected to "full" — matching train.py.
  REQ (3)  : evaluate() now checks whether a matching row already exists in
              output_csv before running any episodes.  If found, it prints a
              [SKIP] message and returns immediately.  This prevents
              double-appending when a SLURM job is re-submitted after partial
              completion without needing to delete the per-job CSV first.
  MINOR    : find_model_path() sorts glob matches for determinism.

Usage examples:
  # Main results (default or tuned HPs):
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \\
                     --experiment main --hp_tag default \\
                     --log_root logs --output_csv results/results_default.csv

  # Reward ablation:
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \\
                     --experiment ablation_reward --ablation no_col \\
                     --output_csv results/ablation_reward.csv

  # DR:
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \\
                     --experiment dr --ablation wind \\
                     --output_csv results/dr_results.csv \\
                     --eval_wind_min 0.0 --eval_wind_max 2.0

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import csv
import glob
import time
import fcntl
import argparse
import numpy as np
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

JSON_PATH = os.path.join('exp_sets', 'stochastic_envs_v2.json')
N_EVAL_EPISODES = 50
MAX_STEPS = 1000

ALGORITHMS = {
    "A2C":    A2C,
    "ARS":    ARS,
    "PPO":    PPO,
    "TRPO":   TRPO,
    "CrossQ": CrossQ,
    "TQC":    TQC,
}

EXPERIMENT_MAP = {
    "main":                 None,
    "ablation_reward":      "reward_ablation",
    "ablation_obs":         "obs_mode",
    "ablation_uncertainty": "uncertainty_mode",
    "dr":                   "dr_mode",
}

# BUG FIX: "ablation_obs" default was "base" — not a valid obs_mode in env.py.
# train.py already had this corrected to "full"; evaluate.py now matches.
EXPERIMENT_DEFAULTS = {
    "main":                 None,
    "ablation_reward":      "full",
    "ablation_obs":         "full",   # was "base" — invalid; corrected to "full"
    "ablation_uncertainty": "full",
    "dr":                   "none",
}

# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Post-training evaluation")
    p.add_argument("--algorithm",    type=str, required=True,
                   choices=list(ALGORITHMS.keys()))
    p.add_argument("--set",          type=int, required=True)
    p.add_argument("--num_robots",   type=int, default=3)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--experiment",   type=str, default="main",
                   choices=list(EXPERIMENT_MAP.keys()))
    p.add_argument("--ablation",     type=str, default=None)
    p.add_argument("--hp_tag",       type=str, default="default",
                   choices=["default", "tuned"],
                   help="For experiment=main: which HP set was used")
    p.add_argument("--log_root",     type=str, default="logs")
    p.add_argument("--output_csv",   type=str, required=True)
    p.add_argument("--n_eval_eps",   type=int, default=N_EVAL_EPISODES)
    p.add_argument("--device",       type=str, default="cpu")
    # Wind sweep options (for DR and wind sensitivity eval)
    p.add_argument("--eval_wind_min",  type=float, default=None,
                   help="If set, uniformly sample wind speed in [min,max] for each eval episode")
    p.add_argument("--eval_wind_max",  type=float, default=None)
    # For uncertainty ablation cross-evaluation
    p.add_argument("--eval_uncertainty_mode", type=str, default=None,
                   help="Override environment uncertainty_mode at eval time")
    return p.parse_args()

# ================================================================
# Helpers
# ================================================================

def compute_iqm(rewards: np.ndarray) -> float:
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))


def _csv_str(v) -> str:
    """Serialise a value the same way csv.writer does for None/float.
    csv.writer writes None as an empty string, so we must match that."""
    return "" if v is None else str(v)


def already_evaluated(output_csv: str, args, ablation_val: str) -> bool:
    """
    Return True when output_csv already contains a row whose key fields
    match the current run exactly.

    REQ (3): This is the skip-if-present guard.  Fields checked:
        algorithm, experiment, ablation, num_robots, env_set, seed,
        eval_wind_min, eval_wind_max, eval_uncertainty_mode.

    Using csv.DictReader instead of pandas so that evaluate.py has no
    extra dependencies beyond the standard library.
    """
    if not os.path.exists(output_csv):
        return False
    try:
        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("algorithm")             == args.algorithm
                        and row.get("experiment")    == args.experiment
                        and row.get("ablation")      == ablation_val
                        and row.get("num_robots")    == str(args.num_robots)
                        and row.get("env_set")       == str(args.set)
                        and row.get("seed")          == str(args.seed)
                        and row.get("eval_wind_min")         == _csv_str(args.eval_wind_min)
                        and row.get("eval_wind_max")         == _csv_str(args.eval_wind_max)
                        and row.get("eval_uncertainty_mode") == _csv_str(args.eval_uncertainty_mode)):
                    return True
    except Exception:
        pass
    return False


def find_model_path(log_root: str, algorithm: str, num_robots: int,
                    env_set: int, experiment: str, hp_tag: str,
                    ablation: str, seed: int) -> str:
    """
    Search log_root for the best_model directory matching the given run.
    Directory layout (produced by the fixed train.py):
        logs/{version}/{algorithm}_N{num_robots}_env{set}_seed{seed}/
                        best_model/best_model.zip
    """
    if experiment == "main":
        version = f"main_{hp_tag}"
    elif experiment == "dr":
        version = f"dr_{ablation or 'none'}"
    else:
        version = f"{experiment}_{ablation or EXPERIMENT_DEFAULTS[experiment]}"

    tag     = f"{algorithm}_N{num_robots}_env{env_set}_seed{seed}"
    pattern = os.path.join(log_root, version, tag, "best_model", "best_model.zip")
    matches = sorted(glob.glob(pattern))  # sorted for determinism

    if not matches:
        raise FileNotFoundError(
            f"No model found for pattern:\n  {pattern}\n"
            "Have you completed training?")

    path = matches[0]
    print(f"  Using model: {path}")
    return os.path.splitext(path)[0]


def build_eval_env_kwargs(args, field_info: dict,
                           wind_speed: float = None,
                           wind_dir: float = None) -> dict:
    """Build env kwargs, optionally overriding wind and uncertainty_mode."""
    kwargs = dict(
        field_info=field_info,
        num_robots=args.num_robots,
        max_steps=MAX_STEPS,
        render_mode=None,
    )

    # Pass ablation kwarg if applicable
    kwarg_name = EXPERIMENT_MAP[args.experiment]
    if kwarg_name is not None:
        ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
        kwargs[kwarg_name] = ablation_val

    # Override uncertainty_mode at eval time (cross-evaluation)
    if args.eval_uncertainty_mode is not None:
        kwargs["uncertainty_mode"] = args.eval_uncertainty_mode

    # Override wind speed and direction for DR / wind sensitivity sweep
    if wind_speed is not None:
        direction = wind_dir if wind_dir is not None else 0.0
        kwargs["wind_par"] = [wind_speed, direction]

    return kwargs


# ================================================================
# Main evaluation
# ================================================================

def evaluate(args):
    ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]

    # ── REQ (3): skip immediately if this exact run is already recorded ──
    if already_evaluated(args.output_csv, args, ablation_val):
        print(
            f"  [SKIP] Result already present in {args.output_csv} for\n"
            f"         {args.algorithm} | {args.experiment} | {ablation_val} | "
            f"N={args.num_robots} | set={args.set} | seed={args.seed}\n"
            f"         eval_wind=[{args.eval_wind_min},{args.eval_wind_max}] | "
            f"eval_uncertainty_mode={args.eval_uncertainty_mode}\n"
            f"         Skipping re-evaluation."
        )
        return

    json_dict  = load_experiment_dict_json(JSON_PATH)
    field_info = json_dict[f"set{args.set}"]

    env_id = "MultiRobotEnv-v0"
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=MultiRobotEnv,
                     max_episode_steps=MAX_STEPS)

    # Find and load model
    model_path = find_model_path(
        args.log_root, args.algorithm, args.num_robots, args.set,
        args.experiment, args.hp_tag, args.ablation, args.seed)
    AlgClass = ALGORITHMS[args.algorithm]
    model    = AlgClass.load(model_path, device=args.device)

    print(f"\n{'='*60}")
    print(f"  Evaluating : {args.algorithm} | {args.experiment} | {ablation_val}")
    print(f"  Env set    : {args.set}  |  N robots: {args.num_robots}  |  Seed: {args.seed}")
    print(f"  Episodes   : {args.n_eval_eps}")
    print(f"{'='*60}")

    # ── Run episodes ─────────────────────────────────────────────
    all_rewards    = []
    all_ep_lengths = []
    start = time.perf_counter()

    for ep in range(args.n_eval_eps):
        # Optionally sweep wind speed across eval episodes
        if args.eval_wind_min is not None and args.eval_wind_max is not None:
            ep_rng = np.random.default_rng(args.seed + ep)
            w     = ep_rng.uniform(args.eval_wind_min, args.eval_wind_max)
            w_dir = ep_rng.uniform(0, 360)
        else:
            w     = None
            w_dir = None

        env_kwargs = build_eval_env_kwargs(args, field_info, wind_speed=w, wind_dir=w_dir)
        eval_env   = make_vec_env(env_id, env_kwargs=env_kwargs,
                                  n_envs=1, seed=args.seed + ep + 1000)

        ep_rewards, ep_lengths = evaluate_policy(
            model, eval_env,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True,
        )
        all_rewards.append(ep_rewards[0])
        all_ep_lengths.append(ep_lengths[0])
        eval_env.close()

    elapsed = time.perf_counter() - start

    rewards_arr = np.array(all_rewards,    dtype=np.float32)
    ep_lens_arr = np.array(all_ep_lengths, dtype=np.float32)
    mean_r      = float(np.mean(rewards_arr))
    std_r       = float(np.std(rewards_arr))
    max_r       = float(np.max(rewards_arr))
    iqm_r       = compute_iqm(rewards_arr)
    # CVaR_0.1: expected reward of worst 10 % of episodes
    cutoff      = int(np.ceil(0.1 * len(rewards_arr)))
    cvar        = float(np.mean(np.sort(rewards_arr)[:cutoff]))
    mean_ep_len = float(np.mean(ep_lens_arr))
    iqm_ep_len  = compute_iqm(ep_lens_arr)

    print(f"\n  mean={mean_r:.3f}  std={std_r:.3f}  max={max_r:.3f}  "
          f"IQM={iqm_r:.3f}  CVaR_0.1={cvar:.3f}")
    print(f"  mean_ep_len={mean_ep_len:.1f}  iqm_ep_len={iqm_ep_len:.1f}  elapsed={elapsed:.1f}s")

    # ── Append to CSV ────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)          # acquire exclusive lock
        try:
            write_header = os.fstat(f.fileno()).st_size == 0
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "algorithm", "experiment", "ablation", "hp_tag",
                    "num_robots", "env_set", "seed",
                    "eval_wind_min", "eval_wind_max", "eval_uncertainty_mode",
                    "mean_reward", "std_reward", "max_reward", "iqm",
                    "cvar_0.1", "mean_ep_length", "iqm_ep_length",
                    "n_episodes", "elapsed_s",
                ])
            writer.writerow([
                args.algorithm, args.experiment, ablation_val, args.hp_tag,
                args.num_robots, args.set, args.seed,
                args.eval_wind_min, args.eval_wind_max, args.eval_uncertainty_mode,
                f"{mean_r:.4f}", f"{std_r:.4f}", f"{max_r:.4f}", f"{iqm_r:.4f}",
                f"{cvar:.4f}", f"{mean_ep_len:.2f}", f"{iqm_ep_len:.2f}",
                args.n_eval_eps, f"{elapsed:.1f}",
            ])
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  # release lock
    print(f"  Appended to {args.output_csv}")


if __name__ == "__main__":
    evaluate(parse_args())