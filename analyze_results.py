#!/usr/bin/env python3
"""
analyze_results.py — Aggregate raw per-run evaluation CSVs into
table-ready summary CSVs that directly map to the LaTeX tables in
full_experiments.tex.

Outputs (written to results/):
  results_default.csv         → tab:default_hyp
  results_tuned.csv           → tab:random_hyp
  results_transfer.csv        → tab:transfer_hyp
  ablation_reward_agg.csv     → tab:ablation_reward
  ablation_obs_agg.csv        → tab:ablation_obs
  ablation_uncertainty_agg.csv→ tab:ablation_uncertainty
  dr_results_agg.csv          → tab:dr_results

Usage:
  python analyze_results.py --results_dir results

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ranksums, wilcoxon

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TUNING_TRIALS_PER_ALGORITHM = 50
DEFAULT_TUNING_STEPS_PER_TRIAL = 500_000
DEFAULT_HPARAM_SEARCH_STEPS_PER_ALGORITHM = (
    DEFAULT_TUNING_TRIALS_PER_ALGORITHM * DEFAULT_TUNING_STEPS_PER_TRIAL
)

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


def extract_episode_rewards(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)

    if "episode_rewards_json" in df.columns:
        vals = []
        for x in df["episode_rewards_json"].dropna():
            try:
                vals.extend(float(v) for v in json.loads(x))
            except Exception:
                pass
        if vals:
            return np.array(vals, dtype=float)

    if "mean_reward" in df.columns:
        return df["mean_reward"].astype(float).to_numpy()

    return np.array([], dtype=float)


def extract_run_mean_rewards(df: pd.DataFrame) -> np.ndarray:
    if df.empty or "mean_reward" not in df.columns:
        return np.array([], dtype=float)

    vals = pd.to_numeric(df["mean_reward"], errors="coerce").dropna()
    return vals.to_numpy(dtype=float)


OPTIONAL_METRIC_COLS = [
    "success_rate",
    "collision_rate",
    "time_limit_rate",
    "mean_episode_spray_used",
    "mean_episode_spray_wasted",
    "mean_episode_spray_applied",
    "mean_episode_wasted_fraction",
    "mean_episode_spray_empty_capacity",
    "mean_episode_boundary_viol_count",
    "mean_episode_remaining_infection_sum",
    "mean_episode_remaining_infection_fraction",
]


def add_optional_metric_means(row: dict, grp: pd.DataFrame,
                              prefix: str = "") -> dict:
    """Attach mean diagnostic metrics when the raw CSV contains them."""
    for col in OPTIONAL_METRIC_COLS:
        if col not in grp.columns:
            continue
        vals = pd.to_numeric(grp[col], errors="coerce").dropna()
        if len(vals):
            row[f"{prefix}{col}"] = float(vals.mean())
    return row


def describe_eval_dr_mode(df: pd.DataFrame) -> str:
    if "eval_dr_mode" not in df.columns:
        return "unknown"

    vals = (
        df["eval_dr_mode"]
        .fillna("")
        .astype(str)
        .replace("", "unspecified")
        .unique()
        .tolist()
    )
    vals = sorted(vals)
    return vals[0] if len(vals) == 1 else "mixed:" + ",".join(vals)


def require_grid_complete(df: pd.DataFrame, expected: dict, label: str) -> None:
    import itertools

    cols = list(expected.keys())
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{label}: missing required grid columns: {missing_cols}")

    expected_keys = set(itertools.product(*[expected[c] for c in cols]))
    actual_keys = set(
        tuple(row[c] for c in cols)
        for _, row in df[cols].drop_duplicates().iterrows()
    )

    missing = expected_keys - actual_keys
    if missing:
        sample = list(missing)[:10]
        raise ValueError(
            f"{label}: missing {len(missing)} expected combinations. "
            f"Examples: {sample}"
        )

# The previous helper was named wilcoxon_pval but used scipy.stats.ranksums,
# which is an unpaired rank-sum test. Main-result comparisons are paired because
# each algorithm is evaluated on the same env_set × seed combinations. Therefore,
# we first collapse duplicate rows to one value per algorithm × N × env_set × seed,
# then align algorithms by env_set and seed before applying scipy.stats.wilcoxon,
# the paired signed-rank test. This preserves the matched-run structure that the
# rank-sum test ignored.

# def wilcoxon_pval(a: np.ndarray, b: np.ndarray) -> float:
#     if len(a) < 2 or len(b) < 2:
#         return 1.0
#     _, p = ranksums(a, b)
#     return float(p)

def wilcoxon_pval(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    # If all paired differences are zero, scipy can fail or return edge-case behavior.
    if np.allclose(a, b):
        return 1.0
    _, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(p)

def ranksum_pval(a: np.ndarray, b: np.ndarray) -> float:
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

    for key, grp in df_iter:
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

def paired_wilcoxon_between_algs(
    df_pairs: pd.DataFrame,
    num_robots: int,
    alg_a: str,
    alg_b: str,
    value_col: str = "mean_reward",
) -> float:
    sub = df_pairs[
        (df_pairs["num_robots"] == num_robots)
        & (df_pairs["algorithm"].isin([alg_a, alg_b]))
    ]

    pivot = sub.pivot_table(
        index=["env_set", "seed"],
        columns="algorithm",
        values=value_col,
        aggfunc="mean",
    )

    if alg_a not in pivot.columns or alg_b not in pivot.columns:
        return 1.0

    paired = pivot[[alg_a, alg_b]].dropna()

    return wilcoxon_pval(
        paired[alg_a].to_numpy(dtype=float),
        paired[alg_b].to_numpy(dtype=float),
    )


def mark_best_main_paired(
    summary: pd.DataFrame,
    df_pairs: pd.DataFrame,
    rank_col: str = "iqm",
    test_col: str = "mean_reward",
    alpha: float = 0.05,
) -> pd.DataFrame:
    summary = summary.copy()
    summary["is_best"] = False

    for N, grp in summary.groupby("num_robots"):
        sorted_grp = grp.sort_values(rank_col, ascending=False)

        if len(sorted_grp) < 2:
            summary.loc[sorted_grp.index[0], "is_best"] = True
            continue

        best_idx = sorted_grp.index[0]
        second_idx = sorted_grp.index[1]

        best_alg = sorted_grp.loc[best_idx, "algorithm"]
        second_alg = sorted_grp.loc[second_idx, "algorithm"]

        p = paired_wilcoxon_between_algs(
            df_pairs=df_pairs,
            num_robots=N,
            alg_a=best_alg,
            alg_b=second_alg,
            value_col=test_col,
        )

        summary.loc[best_idx, "is_best"] = p < alpha

    return summary

# ================================================================
# Main results (default / tuned / transfer)
# ================================================================

def process_main(results_dir: str, hp_tag: str, strict: bool = False) -> pd.DataFrame:
    """
    Aggregate results_default.csv, results_tuned.csv, or results_transfer.csv.
    Returns a DataFrame with one row per (algorithm, num_robots),
    columns: mean_reward, std_reward, iqm, max_reward, raw_rewards.
    """
    csv_path = os.path.join(results_dir, f"results_{hp_tag}.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping.")
        return None

    df = pd.read_csv(csv_path)

    df["mean_reward"] = df["mean_reward"].astype(float)
    if "mean_ep_length" in df.columns:
        df["mean_ep_length"] = df["mean_ep_length"].astype(float)
    for c in OPTIONAL_METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "hparam_search_steps_per_algorithm" not in df.columns:
        df["hparam_search_steps_per_algorithm"] = (
            DEFAULT_HPARAM_SEARCH_STEPS_PER_ALGORITHM if hp_tag == "tuned" else 0
        )

    if "total_train_steps_plus_search" not in df.columns:
        if "total_train_steps" in df.columns:
            df["total_train_steps_plus_search"] = (
                pd.to_numeric(df["total_train_steps"], errors="coerce").fillna(0)
                + pd.to_numeric(
                    df["hparam_search_steps_per_algorithm"], errors="coerce"
                ).fillna(0)
            )
        else:
            df["total_train_steps_plus_search"] = np.nan

    compute_cols = [
        "pretrain_steps",
        "finetune_steps",
        "total_train_steps",
        "hparam_search_steps_per_algorithm",
        "total_train_steps_plus_search",
    ]
    for c in compute_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # checks
    required_cols = {"algorithm", "num_robots", "env_set", "seed", "mean_reward"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    
    if strict:
        expected_sets = list(range(2, 11)) if hp_tag == "transfer" else list(range(1, 11))
        require_grid_complete(
            df,
            expected={
                "algorithm": ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"],
                "num_robots": [2, 3, 4, 5],
                "env_set": expected_sets,
                "seed": [0, 42, 123, 2024, 9999],
            },
            label=f"main {hp_tag}",
        )

    # One value per algorithm × N × env_set × seed.
    # This also prevents duplicate appended rows from overweighting a run.
    agg_map = {"mean_reward": "mean"}
    if "mean_ep_length" in df.columns:
        agg_map["mean_ep_length"] = "mean"
    for c in OPTIONAL_METRIC_COLS:
        if c in df.columns:
            agg_map[c] = "mean"
    for c in compute_cols:
        if c in df.columns:
            agg_map[c] = "mean"
    df_pairs = (
        df.groupby(["algorithm", "num_robots", "env_set", "seed"], as_index=False)
          .agg(agg_map)
    )

    rows = []
    for (alg, N), grp in df_pairs.groupby(["algorithm", "num_robots"]):
        r = grp["mean_reward"].astype(float).values   # one row per seed×set

        # source_grp = df[
        #     (df["algorithm"] == alg) &
        #     (df["num_robots"] == N)
        # ]
        # ep_r = extract_episode_rewards(source_grp)

        row = dict(
            algorithm=alg,
            num_robots=N,
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            max_reward=float(np.max(r)),
            iqm=compute_iqm(r),
            metric_scope="run_mean_reward",
            # metric_scope="run_mean",
            # mean_reward_episode=float(np.mean(ep_r)) if len(ep_r) else float("nan"),
            # std_reward_episode=float(np.std(ep_r)) if len(ep_r) else float("nan"),
            # max_reward_episode=float(np.max(ep_r)) if len(ep_r) else float("nan"),
            # iqm_episode=compute_iqm(ep_r) if len(ep_r) else float("nan"),
            raw_rewards=list(r),
        )
        if "mean_ep_length" in grp.columns:
            row["mean_ep_length"] = float(np.mean(grp["mean_ep_length"].astype(float).values))
        for c in OPTIONAL_METRIC_COLS:
            if c in grp.columns:
                vals = grp[c].dropna().astype(float).values
                if len(vals):
                    row[c] = float(np.mean(vals))
        for c in compute_cols:
            if c in grp.columns:
                vals = grp[c].dropna().astype(float).values
                if len(vals):
                    row[c] = int(round(float(np.mean(vals))))

        # if hp_tag == "tuned":
        #     row["hparam_search_steps_per_algorithm"] = 50 * 500_000
        # else:
        #     row["hparam_search_steps_per_algorithm"] = 0

        # row["total_train_steps_plus_search"] = int(
        #     row.get("total_train_steps", 0) +
        #     row["hparam_search_steps_per_algorithm"]
        # )
        rows.append(row)

    summary = pd.DataFrame(rows)
    # summary = mark_best(summary, "mean_reward", group_cols=["num_robots"])
    # Use paired Wilcoxon, explicitly aligned by env_set and seed.
    summary = mark_best_main_paired(
        summary,
        df_pairs=df_pairs,
        rank_col="iqm",
        test_col="mean_reward",
        alpha=0.05,
    )

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

    table_labels = {
        "default":  "default_hyp",
        "tuned":    "random_hyp",
        "transfer": "transfer_hyp",
    }

    lines = []
    lines.append(f"% LaTeX table rows for tab:{table_labels.get(hp_tag, hp_tag)}")
    lines.append(f"% hp_tag = {hp_tag}")
    lines.append("% Cell format: mean ± std (IQM).")
    lines.append("% mean/std/IQM are computed over one mean_reward per algorithm × N × env_set × seed.")
    lines.append("% These rows report final-policy evaluation reward, not equalized total compute.")
    if hp_tag == "tuned":
        lines.append("% Tuned-policy rows include an additional hyperparameter-search budget in main_tuned_summary.csv.")
    if hp_tag == "transfer":
        lines.append("% Transfer rows include source pretraining and target fine-tuning steps in main_transfer_summary.csv.")
    lines.append("")

    for alg in algorithms:
        cells = []
        for N in robot_counts:
            row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if row.empty:
                cells.append("---")
            else:
                r = row.iloc[0]
                dagger = r"^\dagger" if r["is_best"] else ""
                body = f"{r['mean_reward']:.1f} \\pm {r['std_reward']:.1f}\\ ({r['iqm']:.1f}){dagger}"
                if r["is_best"]:
                    body = r"\mathbf{" + body + "}"
                cells.append("$" + body + "$")
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
    if "eval_reward_ablation" not in df.columns:
        raise ValueError(
            "ablation_reward.csv is missing eval_reward_ablation. "
            "Re-run reward-ablation evaluation with --eval_reward_ablation full."
        )
    df = df[df["eval_reward_ablation"] == "full"].copy()
    if df.empty:
        raise ValueError("No reward-ablation rows evaluated with eval_reward_ablation='full'.")
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
        r = extract_run_mean_rewards(grp)
        row = dict(
            condition=label,
            removed_term=removed,
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            iqm=compute_iqm(r),
            metric_scope="run_mean_reward",
            raw_rewards=list(r),
        )
        rows.append(add_optional_metric_means(row, grp))

    summary = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "ablation_reward_agg.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_reward(summary, results_dir)
    return summary


def _write_latex_ablation_reward(summary: pd.DataFrame, results_dir: str):
    lines = [
        "% LaTeX table rows for tab:ablation_reward",
        "% All rows must be evaluated with eval_reward_ablation='full'.",
        "% No automatic bolding: these are ablation diagnostics.",
        "",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['condition']} & {r['removed_term']} & "
            f"${r['mean_reward']:.1f}$ & ${r['std_reward']:.1f}$ & "
            f"${r['iqm']:.1f}$ & --- \\\\"
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
        "full":          "6N+3M+2",
        "base":          "5N+3M",
        "no_wind":       "6N+3M",
        "no_spray_hist": "5N+3M+2",
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
    for cond in ["full", "base", "no_wind", "no_spray_hist", "pos_only"]:
        grp = df[df["ablation"] == cond]
        if grp.empty:
            continue
        r = extract_run_mean_rewards(grp)
        row = dict(
            condition=labels.get(cond, cond),
            obs_dim=obs_dims.get(cond, "?"),
            mean_reward=float(np.mean(r)),
            iqm=compute_iqm(r),
            metric_scope="run_mean_reward",
            raw_rewards=list(r),
        )
        rows.append(add_optional_metric_means(row, grp))

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
    eval_modes  = ["full", "wind_only", "act_only", "deterministic"]

    rows = []
    for train_mode in train_modes:
        row = {"train_condition": train_mode}
        for eval_mode in eval_modes:
            grp = df[(df["ablation"] == train_mode) &
                     (df["eval_uncertainty_mode"] == eval_mode)]
            vals = extract_run_mean_rewards(grp)
            row[f"eval_{eval_mode}_iqm"] = compute_iqm(vals) if len(vals) else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "ablation_uncertainty_agg.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_uncertainty(summary, results_dir)
    return summary


def _write_latex_ablation_uncertainty(summary: pd.DataFrame, results_dir: str):
    noise_labels = {
        "full":          "wind + actuation + spray + obs + init",
        "wind_only":     "wind",
        "act_only":      "actuation",
        "deterministic": "none",
    }
    lines = ["% LaTeX table rows for tab:ablation_uncertainty", ""]
    for _, r in summary.iterrows():
        tc = r["train_condition"]
        lines.append(
            f"{tc.replace('_',' ').title()} & {noise_labels.get(tc,'?')} & "
            f"${r['eval_full_iqm']:.1f}$ & ${r['eval_wind_only_iqm']:.1f}$ & "
            f"${r['eval_act_only_iqm']:.1f}$ & ${r['eval_deterministic_iqm']:.1f}$ \\\\"
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
        sub_in = df_in[df_in["ablation"] == ablation]
        sub_ood = df_ood[df_ood["ablation"] == ablation]
        r_in = extract_run_mean_rewards(sub_in) if not sub_in.empty else np.array([])
        r_ood = extract_run_mean_rewards(sub_ood) if not sub_ood.empty else np.array([])
        all_r = np.concatenate([r_in, r_ood]) if len(r_in) and len(r_ood) else np.array([])
        in_eval_dr_mode = describe_eval_dr_mode(sub_in) if not sub_in.empty else "unknown"
        ood_eval_dr_mode = describe_eval_dr_mode(sub_ood) if not sub_ood.empty else "unknown"
        eval_protocol = (
            f"in_dist eval_dr_mode={in_eval_dr_mode}; "
            f"OOD eval_dr_mode={ood_eval_dr_mode}; "
            "wind ranges are controlled by eval_wind_min/eval_wind_max"
        )
        row = dict(
            condition=label,
            rand_params=rand_params,
            eval_protocol=eval_protocol,
            in_dist_iqm=compute_iqm(r_in)  if len(r_in)  else float("nan"),
            ood_iqm    =compute_iqm(r_ood)  if len(r_ood) else float("nan"),
            cvar_0_1   =cvar_0_1(all_r)     if len(all_r) else float("nan"),
            metric_scope="run_mean_reward",
        )
        row = add_optional_metric_means(row, sub_in, prefix="in_dist_")
        row = add_optional_metric_means(row, sub_ood, prefix="ood_")
        rows.append(row)
    summary = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "dr_results_agg.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_dr(summary, results_dir)
    return summary


def _write_latex_dr(summary: pd.DataFrame, results_dir: str):
    lines = [
        "% LaTeX table rows for tab:dr_results",
        "% rand_params describes training-time randomization.",
        "% Evaluation protocol is written to dr_results_agg.csv as eval_protocol.",
        "",
    ]
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
    p.add_argument("--results_dir", type=str, default=os.path.join(PROJECT_ROOT, "results"),
                   help="Directory containing all raw evaluation CSVs")
    p.add_argument("--strict", action="store_true",
                   help="Fail if expected result-grid combinations are missing")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n── Main results: default HPs ───────────────────────────────")
    default_summary = process_main(args.results_dir, "default", strict=args.strict)
    print_summary_table(default_summary, "Main results", "default")

    print("\n── Main results: tuned HPs ─────────────────────────────────")
    tuned_summary = process_main(args.results_dir, "tuned", strict=args.strict)
    print_summary_table(tuned_summary, "Main results", "tuned")

    print("\n── Main results: transfer learning ─────────────────────────")
    transfer_summary = process_main(args.results_dir, "transfer", strict=args.strict)
    print_summary_table(transfer_summary, "Main results", "transfer")

    print("\n── Reward ablation ─────────────────────────────────────────")
    process_ablation_reward(args.results_dir)

    print("\n── Observation ablation ────────────────────────────────────")
    process_ablation_obs(args.results_dir)

    print("\n── Uncertainty ablation ────────────────────────────────────")
    process_ablation_uncertainty(args.results_dir)

    print("\n── Domain randomization ────────────────────────────────────")
    process_dr(args.results_dir)

    print("\n✓ analyze_results.py complete.")
    print(f"  LaTeX row files are in {args.results_dir}/*_latex_rows.txt")
    print("  Copy each file's contents into the corresponding table in full_experiments.tex.")


if __name__ == "__main__":
    main()
