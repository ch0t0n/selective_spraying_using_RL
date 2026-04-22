#!/usr/bin/env python3
"""
plot_figures.py — Generate all paper figures from training logs and
evaluation CSVs.

Figures produced (saved to figures/):
  default_*_robots.png        → fig:default_hyp  (learning curves, default HPs)
  random_*_robots.png         → fig:random_hyp   (learning curves, tuned HPs)
  transfer_*_robots.png       → fig:transfer_hyp (learning curves, transfer)
  scalability.png             → fig:scalability  (IQM + ep-length vs N)
  transfer_scalability.png    → optional transfer scalability figure
  wind_sensitivity.png        → fig:wind_sensitivity
  dr_curves.png               → fig:dr_curves    (DR training curves)

Usage:
  python plot_figures.py --log_root logs --results_dir results \
                         --figures_dir figures

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================================================================
# Style
# ================================================================

ALG_COLORS = {
    "A2C":    "#e41a1c",
    "ARS":    "#ff7f00",
    "PPO":    "#4daf4a",
    "TQC":    "#984ea3",
    "TRPO":   "#377eb8",
    "CrossQ": "#000000",
}
ALG_ORDER = ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"]

DR_COLORS = {
    "none": "#e41a1c",
    "wind": "#ff7f00",
    "full": "#377eb8",
}
DR_LABELS = {
    "none": "(A) No DR",
    "wind": "(B) Wind DR",
    "full": "(C) Full DR",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
})

# ================================================================
# Log loading
# ================================================================

def load_eval_npz(log_root: str, version_fragment: str,
                  algorithm: str, num_robots: int, env_set: int,
                  seeds=(0, 42, 123, 2024, 9999), strict: bool = False,
                  normalize_timesteps: bool = False):
    """
    Load eval_logs/evaluations.npz for one (alg, N, set), averaged
    across all seeds.  Matches the fixed train.py directory layout:
        logs/{version}/{alg}_N{N}_env{set}_seed{seed}/eval_logs/evaluations.npz
    Returns (timesteps, mean_rewards, std_rewards) or None.
    """
    all_runs = []
    for seed in seeds:
        tag     = f"{algorithm}_N{num_robots}_env{env_set}_seed{seed}"
        pattern = os.path.join(log_root, version_fragment, tag,
                               "eval_logs", "evaluations.npz")
        matches = glob.glob(pattern)
        for m in matches:
            try:
                data = np.load(m, allow_pickle=True)
                ep_rewards = data["results"].mean(axis=1)
                timesteps  = data["timesteps"]
                if normalize_timesteps and len(timesteps) > 0:
                    timesteps = timesteps - timesteps[0]
                all_runs.append((timesteps, ep_rewards))
            except Exception as e:
                print(f"  [WARN] Could not load {m}: {e}")

    expected = len(seeds)
    if len(all_runs) < expected:
        msg = (
            f"Missing eval logs for {algorithm}, N={num_robots}, set={env_set}: "
            f"found {len(all_runs)}/{expected} seeds."
        )
        if strict:
            raise FileNotFoundError(msg)
        print(f"  [WARN] {msg}")

    if not all_runs:
        return None

    min_len = min(len(ts) for ts, _ in all_runs)
    ts_ref  = all_runs[0][0][:min_len]
    for ts, _ in all_runs:
        if not np.array_equal(ts[:min_len], ts_ref):
            msg = (
                f"Non-aligned timesteps for {algorithm}, N={num_robots}, "
                f"set={env_set}."
            )
            if strict:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")
            break
    stacked = np.array([r[:min_len] for _, r in all_runs])
    return ts_ref, stacked.mean(axis=0), stacked.std(axis=0)


def load_learning_curves_for_n(log_root: str, version_fragment: str,
                                num_robots: int, n_sets: int = 10,
                                strict: bool = False, env_sets=None,
                                normalize_timesteps: bool = False):
    """
    Aggregate learning curves across env sets and seeds.
    Returns dict: algorithm → (timesteps, mean_rewards, std_rewards)
    """
    curves = {}
    if env_sets is None:
        env_sets = range(1, n_sets + 1)

    for alg in ALG_ORDER:
        all_runs_ts = []
        all_runs_r  = []
        for s in env_sets:
            result = load_eval_npz(log_root, version_fragment, alg, num_robots, s,
                                   strict=strict,
                                   normalize_timesteps=normalize_timesteps)
            if result is None:
                continue
            ts, mean_r, _ = result
            all_runs_ts.append(ts)
            all_runs_r.append(mean_r)

        if not all_runs_r:
            continue

        min_len = min(len(r) for r in all_runs_r)
        ts_ref  = all_runs_ts[0][:min_len]
        for ts in all_runs_ts:
            if not np.array_equal(ts[:min_len], ts_ref):
                msg = f"Non-aligned set-level timesteps for {alg}, N={num_robots}."
                if strict:
                    raise ValueError(msg)
                print(f"  [WARN] {msg}")
                break
        stacked = np.array([r[:min_len] for r in all_runs_r])
        curves[alg] = (ts_ref, stacked.mean(axis=0), stacked.std(axis=0))

    return curves


# ================================================================
# Figure 1 & 2: Learning curves (default / tuned / transfer)
# ================================================================

def plot_learning_curves(log_root: str, version_fragment: str,
                         hp_label: str, figures_dir: str, strict: bool = False,
                         env_sets=None, title_label: str = None,
                         normalize_timesteps: bool = False):
    """
    Produce a 1×4 figure of learning curves (one panel per N).
    Saved as figures/{hp_label}_{N}_robots.png and a combined figure.
    """
    robot_counts = [2, 3, 4, 5]
    title = title_label or f"{hp_label.replace('_', ' ').title()} Hyperparameters"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    fig.suptitle(f"Learning Curves — {title}", fontsize=13, y=1.02)

    for ax, N in zip(axes, robot_counts):
        curves = load_learning_curves_for_n(log_root, version_fragment, N,
                                            strict=strict, env_sets=env_sets,
                                            normalize_timesteps=normalize_timesteps)
        for alg in ALG_ORDER:
            if alg not in curves:
                continue
            ts, mean_r, std_r = curves[alg]
            ts_m = ts / 1e6   # x-axis in millions of timesteps
            color = ALG_COLORS[alg]
            ax.plot(ts_m, mean_r, label=alg, color=color)
            ax.fill_between(ts_m, mean_r - std_r, mean_r + std_r,
                            alpha=0.15, color=color)
        ax.set_title(f"$N = {N}$")
        ax.set_xlabel("Timesteps (×10⁶)")
        if N == 2:
            ax.set_ylabel("Mean Episodic Reward")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # Single legend
    handles = [mpatches.Patch(color=ALG_COLORS[a], label=a) for a in ALG_ORDER]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(ALG_ORDER), bbox_to_anchor=(0.5, -0.06), fontsize=9)

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, f"{hp_label}_learning_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

    # Also save individual files to match \includegraphics in tex
    for N in robot_counts:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        curves = load_learning_curves_for_n(log_root, version_fragment, N,
                                            strict=strict, env_sets=env_sets,
                                            normalize_timesteps=normalize_timesteps)
        for alg in ALG_ORDER:
            if alg not in curves:
                continue
            ts, mean_r, std_r = curves[alg]
            ts_m = ts / 1e6
            ax2.plot(ts_m, mean_r, label=alg, color=ALG_COLORS[alg])
            ax2.fill_between(ts_m, mean_r - std_r, mean_r + std_r,
                             alpha=0.15, color=ALG_COLORS[alg])
        ax2.set_title(f"$N = {N}$ — {hp_label}")
        ax2.set_xlabel("Timesteps (×10⁶)")
        ax2.set_ylabel("Mean Episodic Reward")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        per_n_path = os.path.join(figures_dir, f"{hp_label}_{N}_robots.png")
        plt.savefig(per_n_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {per_n_path}")


# ================================================================
# Figure 3: Scalability (IQM + ep-length vs N)
# ================================================================

def plot_scalability(results_dir: str, figures_dir: str,
                     summary_filename: str = "main_tuned_summary.csv",
                     output_filename: str = "scalability.png",
                     title: str = "Scalability: IQM and Episode Length vs Number of Robots"):
    """
    Reads a main-summary CSV and plots IQM and mean episode length
    vs number of robots for the best algorithm (TRPO) and CrossQ.
    """
    csv_path = os.path.join(results_dir, summary_filename)
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping scalability plot.")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(title, fontsize=12)

    for alg in ["TRPO", "CrossQ", "PPO"]:
        sub = df[df["algorithm"] == alg].sort_values("num_robots")
        if sub.empty:
            continue
        ax1.plot(sub["num_robots"], sub["iqm"], marker="o",
                 label=alg, color=ALG_COLORS[alg])

    ax1.set_xlabel("Number of robots $N$")
    ax1.set_ylabel("IQM")
    ax1.set_xticks([2, 3, 4, 5])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode length — check if column present
    if "mean_ep_length" in df.columns:
        for alg in ["TRPO", "CrossQ"]:
            sub = df[df["algorithm"] == alg].sort_values("num_robots")
            if sub.empty:
                continue
            ax2.plot(sub["num_robots"], sub["mean_ep_length"], marker="s",
                     label=alg, color=ALG_COLORS[alg])
        ax2.set_xlabel("Number of robots $N$")
        ax2.set_ylabel("Mean episode length (steps)")
        ax2.set_xticks([2, 3, 4, 5])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "mean_ep_length not available\n(run evaluate.py with --n_eval_eps 50)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=9)

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, output_filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ================================================================
# Figure 4: Wind sensitivity
# ================================================================

def plot_wind_sensitivity(results_dir: str, log_root: str, figures_dir: str):
    """
    Plot IQM vs inference-time wind speed for standard vs DR-trained CrossQ.
    Requires wind-sweep evaluation CSVs (generated by eval_wind_sweep.sh).
    """
    wind_csv = os.path.join(results_dir, "wind_sweep.csv")
    if not os.path.exists(wind_csv):
        print(f"  [WARN] {wind_csv} not found — skipping wind sensitivity plot.")
        print("         Run eval_wind_sweep.sh to generate this file.")
        return

    df = pd.read_csv(wind_csv)
    df["mean_reward"]  = df["mean_reward"].astype(float)
    df["iqm"] = df["iqm"].astype(float)
    df["eval_wind_mid"] = (df["eval_wind_min"].astype(float) +
                           df["eval_wind_max"].astype(float)) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    for dr_mode in ["none", "full"]:
        sub = df[df["ablation"] == dr_mode].sort_values("eval_wind_mid")
        if sub.empty:
            continue
        r = sub.groupby("eval_wind_mid")["iqm"]
        means = r.mean()
        stds  = r.std().fillna(0.0)
        label = "Standard training" if dr_mode == "none" else "Full DR training"
        ax.plot(means.index, means.values, marker="o",
                label=label, color=DR_COLORS[dr_mode])
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=DR_COLORS[dr_mode])

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="$v_{\\text{clip}}/10$")
    ax.axvline(x=1.0, color="orange", linestyle="--", alpha=0.7, label="DR training max")
    ax.set_xlabel("Wind speed $v_a$ (m/s)")
    ax.set_ylabel("IQM") # IQM vs mean__reward
    ax.set_title("Wind Sensitivity: Standard vs DR-trained Policy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "wind_sensitivity.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ================================================================
# Figure 5: DR training curves
# ================================================================

def plot_dr_curves(log_root: str, figures_dir: str, N: int = 3, strict: bool = False):
    """
    Learning curves for DR conditions (A), (B), (C) — CrossQ, N=3.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for dr_mode, label in DR_LABELS.items():
        version_fragment = f"dr_{dr_mode}"
        # Average across all sets
        all_runs = []
        for s in range(1, 11):
            result = load_eval_npz(log_root, version_fragment, "CrossQ", N, s,
                                   strict=strict)
            if result is None:
                continue
            ts, mean_r, _ = result
            all_runs.append((ts, mean_r))

        if not all_runs:
            print(f"  [WARN] No DR logs found for mode={dr_mode}")
            continue

        min_len = min(len(r) for _, r in all_runs)
        stacked = np.array([r[:min_len] for _, r in all_runs])
        ts_ref  = all_runs[0][0][:min_len] / 1e6
        mean_r  = stacked.mean(axis=0)
        std_r   = stacked.std(axis=0)

        ax.plot(ts_ref, mean_r, label=label, color=DR_COLORS[dr_mode])
        ax.fill_between(ts_ref, mean_r - std_r, mean_r + std_r,
                        alpha=0.15, color=DR_COLORS[dr_mode])

    ax.set_xlabel("Timesteps (×10⁶)")
    ax.set_ylabel("Mean Episodic Reward")
    ax.set_title(f"Domain Randomization Training Curves (CrossQ, $N={N}$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "dr_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ================================================================
# Entry point
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_root",     type=str, default="logs")
    p.add_argument("--results_dir",  type=str, default="results")
    p.add_argument("--figures_dir",  type=str, default="figures")
    p.add_argument("--skip_curves",  action="store_true",
                   help="Skip learning-curve plots (slow if many log files)")
    p.add_argument("--strict", action="store_true",
                   help="Fail if expected training logs are missing")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    if not args.skip_curves:
        print("\n── Figure 1: Learning curves (default HPs) ─────────────────")
        plot_learning_curves(args.log_root, "main_default",
                             "default", args.figures_dir, strict=args.strict)

        print("\n── Figure 2: Learning curves (tuned HPs) ───────────────────")
        plot_learning_curves(args.log_root, "main_tuned",
                             "random", args.figures_dir, strict=args.strict)

        print("\n── Transfer learning curves ───────────────────────────────")
        plot_learning_curves(args.log_root, "main_transfer",
                             "transfer", args.figures_dir, strict=args.strict,
                             env_sets=list(range(2, 11)),
                             title_label="Transfer Fine-Tuning",
                             normalize_timesteps=True)

    print("\n── Figure 3: Scalability ───────────────────────────────────")
    plot_scalability(args.results_dir, args.figures_dir)
    plot_scalability(
        args.results_dir,
        args.figures_dir,
        summary_filename="main_transfer_summary.csv",
        output_filename="transfer_scalability.png",
        title="Transfer Scalability: IQM and Episode Length vs Number of Robots",
    )

    print("\n── Figure 4: Wind sensitivity ──────────────────────────────")
    plot_wind_sensitivity(args.results_dir, args.log_root, args.figures_dir)

    print("\n── Figure 5: DR training curves ────────────────────────────")
    plot_dr_curves(args.log_root, args.figures_dir, strict=args.strict)

    print("\n✓ plot_figures.py complete.  All figures in:", args.figures_dir)


if __name__ == "__main__":
    main()
