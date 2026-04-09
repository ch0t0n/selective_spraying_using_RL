#!/usr/bin/env python3
"""
analyze_results.py — Aggregate raw per-run evaluation CSVs into
table-ready summary CSVs that directly map to the LaTeX tables in
full_experiments.tex.

Outputs (written to results/):
  results_default.csv         → tab:default_hyp
  results_tuned.csv           → tab:random_hyp
  ablation_reward_agg.csv     → tab:ablation_reward
  ablation_obs_agg.csv        → tab:ablation_obs
  ablation_uncertainty_agg.csv→ tab:ablation_uncertainty
  dr_results_agg.csv          → tab:dr_results

Usage:
  python analyze_results.py --results_dir results

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ranksums

# ================================================================
# Helpers
# ================================================================

def compute_iqm(vals: np.ndarray) -> float:
    q25, q75 = np.percentile(vals, [25, 75])
    mask = (vals >= q25) & (vals <= q75)
    return float(np.mean(vals[mask])) if mask.any() else float(np.mean(vals))


def cvar_0_1(vals: np.ndarray) -> float:
    n = max(1, int(np.ceil(0.1 * len(vals))))
    return float(np.mean(np.sort(vals)[:n]))


def wilcoxon_pval(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = ranksums(a, b)
    return float(p)


def cell_str(mean: float, std: float, iqm: float) -> str:
    return f"{mean:.1f} ± {std:.1f} ({iqm:.1f})"


def mark_best(df_summary: pd.DataFrame,
              value_col: str = "mean_reward",
              group_cols: list = None,
              alpha: float = 0.05) -> pd.DataFrame:
    """
    Add a 'is_best' boolean column — True for the best (highest value_col)
    algorithm per group, provided it is statistically significantly better
    than the second-best (Wilcoxon, α=0.05).
    """
    df_summary = df_summary.copy()
    df_summary["is_best"] = False
    if group_cols is None:
        groups = [None]
        df_iter = [(None, df_summary)]
    else:
        df_iter = df_summary.groupby(group_cols)
        groups = None

    for key, grp in (df_iter if groups is None else df_summary.groupby(group_cols)):
        sorted_grp = grp.sort_values(value_col, ascending=False)
        if len(sorted_grp) < 2:
            df_summary.loc[sorted_grp.index[0], "is_best"] = True
            continue
        best_idx = sorted_grp.index[0]
        second_idx = sorted_grp.index[1]
        # Retrieve raw reward lists for significance test
        best_rewards   = grp.loc[best_idx,   "raw_rewards"]
        second_rewards = grp.loc[second_idx, "raw_rewards"]
        p = wilcoxon_pval(np.array(best_rewards), np.array(second_rewards))
        df_summary.loc[best_idx, "is_best"] = (p < alpha)

    return df_summary


# ================================================================
# Main results (default / tuned HPs)
# ================================================================

def process_main(results_dir: str, hp_tag: str) -> pd.DataFrame:
    """
    Aggregate results_default.csv or results_tuned.csv.
    Returns a DataFrame with one row per (algorithm, num_robots),
    columns: mean_reward, std_reward, iqm, max_reward, raw_rewards.
    """
    csv_path = os.path.join(results_dir, f"results_{hp_tag}.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping.")
        return None

    df = pd.read_csv(csv_path)

    rows = []
    for (alg, N), grp in df.groupby(["algorithm", "num_robots"]):
        r = grp["mean_reward"].astype(float).values   # one row per seed×set
        row = dict(
            algorithm=alg,
            num_robots=N,
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            max_reward=float(np.max(r)),
            iqm=compute_iqm(r),
            raw_rewards=list(r),
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = mark_best(summary, "mean_reward", group_cols=["num_robots"])

    # Write machine-readable CSV
    out = summary.drop(columns=["raw_rewards"])
    out_path = os.path.join(results_dir, f"main_{hp_tag}_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")

    # Write LaTeX-ready table CSV
    _write_latex_main(summary, hp_tag, results_dir)
    return summary


def _write_latex_main(summary: pd.DataFrame, hp_tag: str, results_dir: str):
    algorithms = ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"]
    robot_counts = [2, 3, 4, 5]

    lines = []
    lines.append(f"% LaTeX table rows for tab:{'default' if hp_tag == 'default' else 'random'}_hyp")
    lines.append(f"% hp_tag = {hp_tag}")
    lines.append("")

    for alg in algorithms:
        cells = []
        for N in robot_counts:
            row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if row.empty:
                cells.append("---")
            else:
                r = row.iloc[0]
                dagger = r"$^\dagger$" if r["is_best"] else ""
                s = f"${r['mean_reward']:.1f} \\pm {r['std_reward']:.1f}\\ ({r['iqm']:.1f}){dagger}$"
                if r["is_best"]:
                    s = r"\mathbf{" + s[1:-1] + "}" 
                    s = "$" + s + "$"
                cells.append(s)
        lines.append(f"{alg} & " + " & ".join(cells) + r" \\")

    out_path = os.path.join(results_dir, f"main_{hp_tag}_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Reward ablation
# ================================================================

def process_ablation_reward(results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, "ablation_reward.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping.")
        return None

    df = pd.read_csv(csv_path)
    df["mean_reward"] = df["mean_reward"].astype(float)

    condition_labels = {
        "full":   ("Full reward",      "---"),
        "no_col": (r"No $R_\text{col}$", "collision"),
        "no_cov": (r"No $R_\text{cov}$", "coverage"),
        "no_eff": (r"No $R_\text{eff}$", "efficiency"),
    }

    rows = []
    for cond, (label, removed) in condition_labels.items():
        grp = df[df["ablation"] == cond]
        if grp.empty:
            continue
        r = grp["mean_reward"].values
        rows.append(dict(
            condition=label,
            removed_term=removed,
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            iqm=compute_iqm(r),
            raw_rewards=list(r),
        ))

    summary = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "ablation_reward_agg.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_reward(summary, results_dir)
    return summary


def _write_latex_ablation_reward(summary: pd.DataFrame, results_dir: str):
    lines = ["% LaTeX table rows for tab:ablation_reward", ""]
    best_iqm = summary["iqm"].max()
    for _, r in summary.iterrows():
        bold = r["iqm"] == best_iqm
        mean_s = rf"\mathbf{{{r['mean_reward']:.1f}}}" if bold else f"{r['mean_reward']:.1f}"
        iqm_s  = rf"\mathbf{{{r['iqm']:.1f}}}" if bold else f"{r['iqm']:.1f}"
        lines.append(
            f"{r['condition']} & {r['removed_term']} & "
            f"${mean_s}$ & ${r['std_reward']:.1f}$ & ${iqm_s}$ & --- \\\\"
        )
    out_path = os.path.join(results_dir, "ablation_reward_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Observation ablation
# ================================================================

def process_ablation_obs(results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, "ablation_obs.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping.")
        return None

    df = pd.read_csv(csv_path)
    df["mean_reward"] = df["mean_reward"].astype(float)

    obs_dims = {
        "full":          "5N+M+2",
        "base":          "5N+M",
        "no_wind":       "6N+M",
        "no_spray_hist": "5N+M+2",
        "pos_only":      "2N",
    }
    labels = {
        "full":          "Full obs.",
        "base":          "Full obs. (base)",
        "no_wind":       "No wind estimate",
        "no_spray_hist": "No spray history",
        "pos_only":      "Positions only",
    }

    rows = []
    for cond in ["full", "no_wind", "no_spray_hist", "pos_only"]:
        grp = df[df["ablation"] == cond]
        if grp.empty:
            continue
        r = grp["mean_reward"].values
        rows.append(dict(
            condition=labels.get(cond, cond),
            obs_dim=obs_dims.get(cond, "?"),
            mean_reward=float(np.mean(r)),
            iqm=compute_iqm(r),
            raw_rewards=list(r),
        ))

    summary = pd.DataFrame(rows)
    ref_iqm = summary.iloc[0]["iqm"] if not summary.empty else 1.0
    summary["delta_iqm_pct"] = (summary["iqm"] - ref_iqm) / (abs(ref_iqm) + 1e-9) * 100

    out_path = os.path.join(results_dir, "ablation_obs_agg.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_obs(summary, results_dir)
    return summary


def _write_latex_ablation_obs(summary: pd.DataFrame, results_dir: str):
    lines = ["% LaTeX table rows for tab:ablation_obs", ""]
    for i, r in summary.iterrows():
        delta = "---" if i == 0 else f"${r['delta_iqm_pct']:.1f}$"
        mean_s = rf"\mathbf{{{r['mean_reward']:.1f}}}" if i == 0 else f"{r['mean_reward']:.1f}"
        iqm_s  = rf"\mathbf{{{r['iqm']:.1f}}}" if i == 0 else f"{r['iqm']:.1f}"
        lines.append(
            f"{r['condition']} & ${r['obs_dim']}$ & ${mean_s}$ & ${iqm_s}$ & {delta} \\\\"
        )
    out_path = os.path.join(results_dir, "ablation_obs_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Uncertainty ablation (cross-evaluation matrix)
# ================================================================

def process_ablation_uncertainty(results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, "ablation_uncertainty.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping.")
        return None

    df = pd.read_csv(csv_path)
    df["mean_reward"] = df["mean_reward"].astype(float)
    df["iqm"]         = df["iqm"].astype(float)

    train_modes = ["full", "wind_only", "act_only", "deterministic"]
    eval_modes  = ["full", "deterministic"]    # "same", "full_stoch", "det"

    rows = []
    for train_mode in train_modes:
        # "Eval (same)": eval under the same condition as training
        grp_same = df[(df["ablation"] == train_mode) &
                      (df["eval_uncertainty_mode"] == train_mode)]
        # "Eval (full stoch.)": eval under full uncertainty
        grp_full = df[(df["ablation"] == train_mode) &
                      (df["eval_uncertainty_mode"] == "full")]
        # "Eval (det.)": eval deterministically
        grp_det  = df[(df["ablation"] == train_mode) &
                      (df["eval_uncertainty_mode"] == "deterministic")]

        rows.append(dict(
            train_condition=train_mode,
            eval_same_iqm=compute_iqm(grp_same["mean_reward"].values) if not grp_same.empty else float("nan"),
            eval_full_iqm=compute_iqm(grp_full["mean_reward"].values) if not grp_full.empty else float("nan"),
            eval_det_iqm =compute_iqm(grp_det["mean_reward"].values)  if not grp_det.empty  else float("nan"),
        ))

    summary = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "ablation_uncertainty_agg.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_uncertainty(summary, results_dir)
    return summary


def _write_latex_ablation_uncertainty(summary: pd.DataFrame, results_dir: str):
    noise_labels = {
        "full":          "wind + actuation + spray",
        "wind_only":     "wind",
        "act_only":      "actuation",
        "deterministic": "none",
    }
    lines = ["% LaTeX table rows for tab:ablation_uncertainty", ""]
    for _, r in summary.iterrows():
        tc = r["train_condition"]
        lines.append(
            f"{tc.replace('_',' ').title()} & {noise_labels.get(tc,'?')} & "
            f"${r['eval_same_iqm']:.1f}$ & ${r['eval_full_iqm']:.1f}$ & ${r['eval_det_iqm']:.1f}$ \\\\"
        )
    out_path = os.path.join(results_dir, "ablation_uncertainty_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Domain randomization
# ================================================================

def process_dr(results_dir: str) -> pd.DataFrame:
    in_dist_path = os.path.join(results_dir, "dr_inDist.csv")
    ood_path     = os.path.join(results_dir, "dr_OOD.csv")

    if not os.path.exists(in_dist_path) or not os.path.exists(ood_path):
        print(f"  [WARN] DR CSVs not found — skipping.")
        return None

    df_in  = pd.read_csv(in_dist_path)
    df_ood = pd.read_csv(ood_path)
    df_in["mean_reward"]  = df_in["mean_reward"].astype(float)
    df_ood["mean_reward"] = df_ood["mean_reward"].astype(float)

    dr_conditions = [("none", "(A) no DR", "none (standard training)"),
                     ("wind", "(B) wind DR", "wind speed + direction"),
                     ("full", "(C) full DR", "all parameters")]

    rows = []
    for ablation, label, rand_params in dr_conditions:
        r_in  = df_in[df_in["ablation"]  == ablation]["mean_reward"].values
        r_ood = df_ood[df_ood["ablation"] == ablation]["mean_reward"].values
        all_r = np.concatenate([r_in, r_ood]) if len(r_in) and len(r_ood) else np.array([])
        rows.append(dict(
            condition=label,
            rand_params=rand_params,
            in_dist_iqm=compute_iqm(r_in)  if len(r_in)  else float("nan"),
            ood_iqm    =compute_iqm(r_ood)  if len(r_ood) else float("nan"),
            cvar_0_1   =cvar_0_1(all_r)     if len(all_r) else float("nan"),
        ))

    summary = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "dr_results_agg.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_dr(summary, results_dir)
    return summary


def _write_latex_dr(summary: pd.DataFrame, results_dir: str):
    lines = ["% LaTeX table rows for tab:dr_results", ""]
    best_ood  = summary["ood_iqm"].max()
    best_cvar = summary["cvar_0_1"].max()
    for _, r in summary.iterrows():
        ood_s  = rf"\mathbf{{{r['ood_iqm']:.1f}}}^\dagger" if r["ood_iqm"]  == best_ood  else f"{r['ood_iqm']:.1f}"
        cvar_s = rf"\mathbf{{{r['cvar_0_1']:.1f}}}^\dagger" if r["cvar_0_1"] == best_cvar else f"{r['cvar_0_1']:.1f}"
        lines.append(
            f"{r['condition']} & {r['rand_params']} & "
            f"${r['in_dist_iqm']:.1f}$ & ${ood_s}$ & ${cvar_s}$ \\\\"
        )
    out_path = os.path.join(results_dir, "dr_results_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Print summary to stdout for quick copy-paste into paper
# ================================================================

def print_summary_table(summary: pd.DataFrame, label: str, hp_tag: str):
    if summary is None:
        return
    algorithms = ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"]
    robot_counts = [2, 3, 4, 5]
    print(f"\n{'='*80}")
    print(f"  {label} ({hp_tag})")
    print(f"{'='*80}")
    header = f"{'Algorithm':10s}" + "".join(f"  N={N:5}" for N in robot_counts)
    print(header)
    print("-" * len(header))
    for alg in algorithms:
        row = f"{alg:10s}"
        for N in robot_counts:
            r = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if r.empty:
                row += "     ---   "
            else:
                ri = r.iloc[0]
                flag = "*" if ri["is_best"] else " "
                row += f"  {ri['mean_reward']:6.1f}{flag}"
        print(row)
    print(f"{'='*80}")


# ================================================================
# Entry point
# ================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results",
                   help="Directory containing all raw evaluation CSVs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n── Main results: default HPs ───────────────────────────────")
    default_summary = process_main(args.results_dir, "default")
    print_summary_table(default_summary, "Main results", "default")

    print("\n── Main results: tuned HPs ─────────────────────────────────")
    tuned_summary = process_main(args.results_dir, "tuned")
    print_summary_table(tuned_summary, "Main results", "tuned")

    print("\n── Reward ablation ─────────────────────────────────────────")
    process_ablation_reward(args.results_dir)

    print("\n── Observation ablation ────────────────────────────────────")
    process_ablation_obs(args.results_dir)

    print("\n── Uncertainty ablation ────────────────────────────────────")
    process_ablation_uncertainty(args.results_dir)

    print("\n── Domain randomization ────────────────────────────────────")
    process_dr(args.results_dir)

    print("\n✓ analyze_results.py complete.")
    print("  LaTeX row files are in results/*_latex_rows.txt")
    print("  Copy each file's contents into the corresponding table in full_experiments.tex.")


if __name__ == "__main__":
    main()
