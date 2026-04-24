#!/usr/bin/env python3
"""
evaluate.py — post-training evaluation for ALL experiment types.

Loads the best saved model for a given (algorithm, set, num_robots, seed,
experiment, ablation) combination, runs N_EVAL_EPISODES episodes, and
appends one row to a results CSV.

Usage examples:
  # Main results (default, tuned, or transfer):
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \
                     --experiment main --hp_tag default \
                     --log_root logs --output_csv results/results_default.csv

  # Main results, transfer learning:
  python evaluate.py --algorithm CrossQ --set 2 --num_robots 3 --seed 42 \
                     --experiment main --hp_tag transfer \
                     --log_root logs --output_csv results/results_transfer.csv

  # Reward ablation:
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \
                     --experiment ablation_reward --ablation no_col \
                     --output_csv results/ablation_reward.csv

  # DR:
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \
                     --experiment dr --ablation wind \
                     --output_csv results/dr_results.csv \
                     --eval_wind_min 0.0 --eval_wind_max 2.0

The script searches for the best model under log_root using the fixed
training layout: logs/{version}/{algorithm}_N{N}_env{set}_seed{seed}/.

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import csv
import glob
import json
import time
import fcntl
import argparse
import inspect
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiRobotEnv
from src.utils import load_experiment_dict_json, set_global_seeds

# ================================================================
# Constants
# ================================================================

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(PROJECT_ROOT, 'exp_sets', 'stochastic_envs_v2.json')
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

EXPERIMENT_DEFAULTS = {
    "main":                 None,
    "ablation_reward":      "full",
    "ablation_obs":         "base",
    "ablation_uncertainty": "full",
    "dr":                   "none",
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
                   choices=["default", "tuned", "transfer"],
                   help="For experiment=main: default, tuned, or transfer run")
    p.add_argument("--log_root",     type=str, default=os.path.join(PROJECT_ROOT, "logs"))
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
    p.add_argument(
        "--eval_reward_ablation",
        type=str,
        default=None,
        choices=["full", "no_col", "no_cov", "no_eff"],
        help="For experiment=ablation_reward: reward function used for evaluation scoring. "
             "If omitted, uses the training reward_ablation.",
    )
    p.add_argument(
        "--freeze_eval_wind_noise",
        action="store_true",
        help="When eval wind is overridden, set wind and wind-direction noise to zero if the env supports it.",
    )
    p.add_argument(
        "--eval_dr_mode",
        type=str,
        default=None,
        choices=["none", "wind", "full"],
        help="For experiment=dr: environment dr_mode used during evaluation. "
             "If omitted, uses the training DR mode from --ablation.",
    )
    p.add_argument("--pretrain_steps", type=int, default=0)
    p.add_argument("--finetune_steps", type=int, default=0)
    p.add_argument(
        "--hparam_search_steps_per_algorithm",
        type=int,
        default=0,
        help="Hyperparameter-search steps spent per algorithm before final training. "
             "Use 0 for default and transfer runs.",
    )
    return p.parse_args()

# ================================================================
# Helpers
# ================================================================

def compute_iqm(rewards: np.ndarray) -> float:
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))


def clean_episode_metrics(metrics: dict) -> dict:
    """Convert terminal episode metrics to JSON-safe Python scalars."""
    cleaned = {}
    if not isinstance(metrics, dict):
        return cleaned
    for key in EPISODE_METRIC_KEYS:
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, np.generic):
            val = val.item()
        if isinstance(val, bool):
            cleaned[key] = bool(val)
        elif isinstance(val, (int, np.integer)):
            cleaned[key] = int(val)
        elif isinstance(val, (float, np.floating)):
            cleaned[key] = float(val)
        else:
            cleaned[key] = val
    return cleaned


def aggregate_episode_metrics(metrics_list: list) -> dict:
    """Aggregate terminal episode metrics over one evaluated policy."""
    if not metrics_list:
        return {}

    def _mean_float(key):
        vals = []
        for metrics in metrics_list:
            if key not in metrics:
                continue
            try:
                vals.append(float(metrics[key]))
            except (TypeError, ValueError):
                pass
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_bool(key):
        vals = [bool(metrics.get(key, False)) for metrics in metrics_list]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "success_rate": _mean_bool("episode_success"),
        "collision_rate": _mean_bool("episode_collision_occurred"),
        "time_limit_rate": _mean_bool("episode_time_limit_reached"),
        "mean_episode_spray_used": _mean_float("episode_spray_used"),
        "mean_episode_spray_wasted": _mean_float("episode_spray_wasted"),
        "mean_episode_spray_applied": _mean_float("episode_spray_applied"),
        "mean_episode_wasted_fraction": _mean_float("episode_wasted_fraction"),
        "mean_episode_spray_empty_capacity": _mean_float("episode_spray_empty_capacity"),
        "mean_episode_boundary_viol_count": _mean_float("episode_boundary_viol_count"),
        "mean_episode_remaining_infection_sum": _mean_float("episode_remaining_infection_sum"),
        "mean_episode_remaining_infection_fraction": _mean_float("episode_remaining_infection_fraction"),
    }


def run_eval_episode(model, eval_env):
    """Run one deterministic episode and return reward, length, and terminal metrics."""
    obs = eval_env.reset()
    done = False
    ep_reward = 0.0
    ep_length = 0
    ep_metrics = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        ep_reward += float(rewards[0])
        ep_length += 1
        done = bool(dones[0])
        if done:
            info = infos[0] if len(infos) else {}
            ep_metrics = clean_episode_metrics(info.get("episode_metrics", {}))

    return ep_reward, ep_length, ep_metrics


def upsert_csv_row_locked(csv_path: str, header: list, row: list, key_cols: list) -> None:
    """Insert or replace one CSV row under an inter-process file lock."""
    output_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    lock_path = f"{csv_path}.lock"
    row_dict = {k: ("" if v is None else v) for k, v in zip(header, row)}

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            rows = []
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for existing in reader:
                        same_key = all(
                            str(existing.get(k, "")) == str(row_dict.get(k, ""))
                            for k in key_cols
                        )
                        if not same_key:
                            rows.append(existing)

            rows.append(row_dict)

            tmp_path = f"{csv_path}.tmp.{os.getpid()}"
            try:
                with open(tmp_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(rows)
                os.replace(tmp_path, csv_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _env_accepts_kwarg(name: str) -> bool:
    try:
        return name in inspect.signature(MultiRobotEnv.__init__).parameters
    except (TypeError, ValueError):
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
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No model found for pattern:\n  {pattern}\n"
            "Have you completed training?")

    path = matches[0]
    print(f"  Using model: {path}")
    return path.replace(".zip", "")


def build_eval_env_kwargs(args, field_info: dict,
                           wind_speed: float = None) -> dict:
    """Build env kwargs, optionally overriding wind and uncertainty_mode."""
    kwargs = dict(
        field_info=field_info,
        num_robots=args.num_robots,
        max_steps=MAX_STEPS,
        render_mode=None,
    )

    # Pass ablation kwarg if applicable. For DR, --ablation identifies
    # the trained checkpoint, while --eval_dr_mode can set a common
    # evaluation randomization mode for controlled comparisons.
    kwarg_name = EXPERIMENT_MAP[args.experiment]
    if kwarg_name is not None:
        ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
        if args.experiment == "dr" and args.eval_dr_mode is not None:
            kwargs[kwarg_name] = args.eval_dr_mode
        else:
            kwargs[kwarg_name] = ablation_val

    if args.experiment == "ablation_reward" and args.eval_reward_ablation is not None:
        kwargs["reward_ablation"] = args.eval_reward_ablation

    # Override uncertainty_mode at eval time (cross-evaluation)
    if args.eval_uncertainty_mode is not None:
        kwargs["uncertainty_mode"] = args.eval_uncertainty_mode

    # Override wind speed for DR / wind sensitivity sweep
    if wind_speed is not None:
        kwargs["wind_par"] = [wind_speed, np.random.uniform(0, 360)]
        kwargs["wind_override"] = True
        if args.freeze_eval_wind_noise:
            if _env_accepts_kwarg("wind_noise_override"):
                kwargs["wind_noise_override"] = 0.0
            if _env_accepts_kwarg("wind_dir_noise_override"):
                kwargs["wind_dir_noise_override"] = 0.0

    return kwargs


# ================================================================
# Main evaluation
# ================================================================

def evaluate(args):
    set_global_seeds(args.seed)
    eval_rng = np.random.default_rng(args.seed + 10_000)

    json_dict  = load_experiment_dict_json(JSON_PATH)
    field_info = json_dict[f"set{args.set}"]

    env_id = "MultiRobotEnv-v0"
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=MultiRobotEnv,
                     max_episode_steps=MAX_STEPS)

    # Find and load model
    model_path   = find_model_path(
        args.log_root, args.algorithm, args.num_robots, args.set,
        args.experiment, args.hp_tag, args.ablation, args.seed)
    AlgClass = ALGORITHMS[args.algorithm]
    model    = AlgClass.load(model_path, device=args.device)

    ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
    total_train_steps = args.pretrain_steps + args.finetune_steps
    total_train_steps_plus_search = (
        total_train_steps + args.hparam_search_steps_per_algorithm
    )

    print(f"\n{'='*60}")
    print(f"  Evaluating : {args.algorithm} | {args.experiment} | {ablation_val}")
    print(f"  Env set    : {args.set}  |  N robots: {args.num_robots}  |  Seed: {args.seed}")
    if args.experiment == "dr":
        print(f"  Eval DR    : {args.eval_dr_mode or ablation_val}")
    print(f"  Episodes   : {args.n_eval_eps}")
    print(f"{'='*60}")

    # ── Run episodes ─────────────────────────────────────────────
    all_rewards = []
    all_ep_lengths = []
    all_episode_metrics = []
    start = time.perf_counter()

    for ep in range(args.n_eval_eps):
        # Optionally sweep wind speed across eval episodes
        if args.eval_wind_min is not None and args.eval_wind_max is not None:
            w = float(eval_rng.uniform(args.eval_wind_min, args.eval_wind_max))
        else:
            w = None

        env_kwargs = build_eval_env_kwargs(args, field_info, wind_speed=w)
        eval_env   = make_vec_env(env_id, env_kwargs=env_kwargs,
                                  n_envs=1, seed=args.seed + ep + 1000)

        try:
            ep_reward, ep_length, ep_metrics = run_eval_episode(model, eval_env)
            all_rewards.append(ep_reward)
            all_ep_lengths.append(ep_length)
            all_episode_metrics.append(ep_metrics)
        finally:
            eval_env.close()

    elapsed = time.perf_counter() - start

    rewards_arr = np.array(all_rewards, dtype=np.float32)
    mean_r  = float(np.mean(rewards_arr))
    std_r   = float(np.std(rewards_arr))
    max_r   = float(np.max(rewards_arr))
    iqm_r   = compute_iqm(rewards_arr)
    # CVaR_0.1: expected reward of worst 10% of episodes
    cutoff  = int(np.ceil(0.1 * len(rewards_arr)))
    cvar    = float(np.mean(np.sort(rewards_arr)[:cutoff]))
    mean_ep_len = float(np.mean(all_ep_lengths))
    episode_rewards_json = json.dumps([float(x) for x in all_rewards])
    episode_lengths_json = json.dumps([int(x) for x in all_ep_lengths])
    episode_metrics_json = json.dumps(all_episode_metrics)
    episode_metric_aggs = aggregate_episode_metrics(all_episode_metrics)

    print(f"\n  mean={mean_r:.3f}  std={std_r:.3f}  max={max_r:.3f}  "
          f"IQM={iqm_r:.3f}  CVaR_0.1={cvar:.3f}")
    print(f"  mean_ep_len={mean_ep_len:.1f}  elapsed={elapsed:.1f}s")
    if episode_metric_aggs:
        print(
            f"  success={episode_metric_aggs['success_rate']:.3f}  "
            f"collision={episode_metric_aggs['collision_rate']:.3f}  "
            f"remaining_frac={episode_metric_aggs['mean_episode_remaining_infection_fraction']:.3f}"
        )

    # ── Upsert to CSV ────────────────────────────────────────────
    eval_reward_ablation = (
        args.eval_reward_ablation if args.experiment == "ablation_reward" else None
    )
    eval_dr_mode = None
    if args.experiment == "dr":
        eval_dr_mode = args.eval_dr_mode or ablation_val
    header = [
        "algorithm", "experiment", "ablation", "hp_tag",
        "num_robots", "env_set", "seed",
        "pretrain_steps", "finetune_steps", "total_train_steps",
        "hparam_search_steps_per_algorithm", "total_train_steps_plus_search",
        "eval_wind_min", "eval_wind_max", "eval_uncertainty_mode",
        "eval_reward_ablation", "eval_dr_mode",
        "mean_reward", "std_reward", "max_reward", "iqm",
        "cvar_0.1", "mean_ep_length", "n_episodes", "elapsed_s",
        "success_rate", "collision_rate", "time_limit_rate",
        "mean_episode_spray_used", "mean_episode_spray_wasted",
        "mean_episode_spray_applied", "mean_episode_wasted_fraction",
        "mean_episode_spray_empty_capacity",
        "mean_episode_boundary_viol_count",
        "mean_episode_remaining_infection_sum",
        "mean_episode_remaining_infection_fraction",
        "episode_rewards_json", "episode_lengths_json", "episode_metrics_json",
    ]
    row = [
        args.algorithm, args.experiment, ablation_val, args.hp_tag,
        args.num_robots, args.set, args.seed,
        args.pretrain_steps, args.finetune_steps, total_train_steps,
        args.hparam_search_steps_per_algorithm, total_train_steps_plus_search,
        args.eval_wind_min, args.eval_wind_max, args.eval_uncertainty_mode,
        eval_reward_ablation, eval_dr_mode,
        f"{mean_r:.4f}", f"{std_r:.4f}", f"{max_r:.4f}", f"{iqm_r:.4f}",
        f"{cvar:.4f}", f"{mean_ep_len:.2f}", args.n_eval_eps, f"{elapsed:.1f}",
        f"{episode_metric_aggs.get('success_rate', float('nan')):.4f}",
        f"{episode_metric_aggs.get('collision_rate', float('nan')):.4f}",
        f"{episode_metric_aggs.get('time_limit_rate', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_spray_used', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_spray_wasted', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_spray_applied', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_wasted_fraction', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_spray_empty_capacity', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_boundary_viol_count', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_remaining_infection_sum', float('nan')):.4f}",
        f"{episode_metric_aggs.get('mean_episode_remaining_infection_fraction', float('nan')):.4f}",
        episode_rewards_json, episode_lengths_json, episode_metrics_json,
    ]
    key_cols = [
        "algorithm", "experiment", "ablation", "hp_tag",
        "num_robots", "env_set", "seed",
        "eval_wind_min", "eval_wind_max", "eval_uncertainty_mode",
        "eval_reward_ablation", "eval_dr_mode",
    ]
    upsert_csv_row_locked(args.output_csv, header, row, key_cols)
    print(f"  Upserted into {args.output_csv}")


if __name__ == "__main__":
    evaluate(parse_args())