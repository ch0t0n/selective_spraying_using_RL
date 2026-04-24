#!/usr/bin/env python3
"""
analyze_results.py — Aggregate per-run evaluations.npz files (written by
SB3's EvalCallback during training) into table-ready summary CSVs that
directly map to the LaTeX tables in full_experiments.tex.

Log folder layout produced by train.py:
  logs/{version}/{alg}_N{num_robots}_env{set}_seed{seed}/eval_logs/evaluations.npz

  version examples:
    main_default          — Step 1 / Step 3 (default HPs)
    main_tuned            — Step 3 (tuned HPs)
    ablation_reward_full  — Step 4 reward ablations
    ablation_obs_no_pos   — Step 5 obs ablations
    ablation_uncertainty_wind_only  — Step 6 uncertainty ablations
    dr_wind               — Step 7 domain randomisation

The "best mean reward" for a run is max(mean reward over evaluation episodes)
across all EvalCallback checkpoints — the checkpoint at which best_model.zip
was saved.

Changes vs previous version
----------------------------
  REQ (1) : process_ablation_reward now reports mean episode length (at the
            best-reward checkpoint) instead of IQM of episode reward.
            New helper load_npz_ep_length_at_best() and helper function
            collect_rewards_and_ep_lengths() support this.
            _write_latex_ablation_reward() emits episode-length columns.
  REQ (2) : process_ablation_reward and process_ablation_obs now use
            ENV_SETS (all 10 sets) instead of the hard-coded [1].
  BUG FIX : same set_list=[1] bug in process_ablation_obs corrected.

Outputs (written to --results_dir, default "results/"):
  main_default_summary.csv / main_tuned_summary.csv
  main_default_latex_rows.txt / main_tuned_latex_rows.txt
  main_combined_latex_rows.txt          → tab:main  (default+tuned IQM)
  ablation_reward_agg.csv               → tab:ablation_reward
  ablation_obs_agg.csv                  → tab:ablation_obs
  ablation_uncertainty_agg.csv          → tab:ablation_uncertainty
  dr_results_agg.csv                    → tab:dr_results
  combined_ablations_dr_latex_rows.txt  → tab:ablations_and_dr

Usage:
  python analyze_results.py --log_root logs --results_dir results

Author: Jahid Chowdhury Choton (choton@ksu.edu)
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ranksums

# ================================================================
# Experiment metadata — mirrors the constants in train.py
# ================================================================

ALGORITHMS   = ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"]
ROBOT_COUNTS = [2, 3, 4, 5]
ENV_SETS     = list(range(1, 11))
SEEDS        = [0, 42, 123, 2024, 9999]

# ================================================================
# NPZ loading helpers
# ================================================================

def load_npz_best_reward(npz_path: str) -> float:
    """
    Load evaluations.npz written by SB3's EvalCallback and return the
    *best mean reward* seen across all evaluation checkpoints.

    evaluations.npz keys:
      timesteps  : 1-D int array, length n_evals
      results    : 2-D float array, shape (n_evals, n_eval_episodes)
      ep_lengths : 2-D int   array, shape (n_evals, n_eval_episodes)
    """
    data    = np.load(npz_path)
    results = data["results"]          # (n_evals, n_eps)
    means   = results.mean(axis=1)     # mean reward per checkpoint
    return float(np.max(means))


def load_npz_ep_length_at_best(npz_path: str) -> float:
    """
    REQ (1): Load evaluations.npz and return the mean episode length at
    the checkpoint where mean episode reward was highest (i.e., the
    checkpoint from which best_model.zip was saved).

    Returns NaN if ep_lengths is absent from the file (older logs).
    """
    data     = np.load(npz_path)
    results  = data["results"]                 # (n_evals, n_eps)
    best_idx = int(np.argmax(results.mean(axis=1)))
    if "ep_lengths" in data:
        return float(data["ep_lengths"][best_idx].mean())
    return float("nan")


def find_npz(log_root: str, version: str, tag: str) -> str | None:
    """Return path to evaluations.npz for (version, tag) or None if missing."""
    path = os.path.join(log_root, version, tag, "eval_logs", "evaluations.npz")
    return path if os.path.exists(path) else None


def collect_rewards(log_root: str, version: str,
                    alg_list, robot_list, set_list, seed_list,
                    extra_fields: dict | None = None) -> list[dict]:
    """
    Scan the log tree for all (alg, N, env_set, seed) combinations and
    return a list of dicts with fields:
      algorithm, num_robots, env_set, seed, mean_reward
    plus any key/value pairs in *extra_fields*.

    Missing NPZ files are silently skipped with a debug message.
    """
    rows = []
    missing = 0
    for alg in alg_list:
        for N in robot_list:
            for s in set_list:
                for seed in seed_list:
                    tag      = f"{alg}_N{N}_env{s}_seed{seed}"
                    npz_path = find_npz(log_root, version, tag)
                    if npz_path is None:
                        missing += 1
                        continue
                    r = load_npz_best_reward(npz_path)
                    row = dict(algorithm=alg, num_robots=N,
                               env_set=s, seed=seed, mean_reward=r)
                    if extra_fields:
                        row.update(extra_fields)
                    rows.append(row)
    if missing:
        print(f"    [INFO] {missing} NPZ file(s) not found under "
              f"{log_root}/{version}/ — those runs are skipped.")
    return rows


def collect_rewards_and_ep_lengths(log_root: str, version: str,
                                    alg_list, robot_list, set_list, seed_list,
                                    extra_fields: dict | None = None) -> list[dict]:
    """
    REQ (1): Like collect_rewards but also records mean_ep_length at the
    best-reward checkpoint.  Used exclusively for the reward ablation table.
    """
    rows = []
    missing = 0
    for alg in alg_list:
        for N in robot_list:
            for s in set_list:
                for seed in seed_list:
                    tag      = f"{alg}_N{N}_env{s}_seed{seed}"
                    npz_path = find_npz(log_root, version, tag)
                    if npz_path is None:
                        missing += 1
                        continue
                    r      = load_npz_best_reward(npz_path)
                    ep_len = load_npz_ep_length_at_best(npz_path)
                    row = dict(algorithm=alg, num_robots=N,
                               env_set=s, seed=seed,
                               mean_reward=r, mean_ep_length=ep_len)
                    if extra_fields:
                        row.update(extra_fields)
                    rows.append(row)
    if missing:
        print(f"    [INFO] {missing} NPZ file(s) not found under "
              f"{log_root}/{version}/ — those runs are skipped.")
    return rows


# ================================================================
# Statistical helpers
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


def mark_best(df_summary: pd.DataFrame,
              value_col: str = "mean_reward",
              group_cols: list = None,
              alpha: float = 0.05) -> pd.DataFrame:
    """
    Add 'is_best' boolean column — True for the highest *value_col*
    algorithm per group, provided it is statistically significantly better
    than the second-best (Wilcoxon rank-sum, α=0.05).
    """
    df_summary = df_summary.copy()
    df_summary["is_best"] = False

    df_iter = (df_summary.groupby(group_cols)
               if group_cols else [(None, df_summary)])

    for _, grp in df_iter:
        sorted_grp = grp.sort_values(value_col, ascending=False)
        if len(sorted_grp) < 2:
            df_summary.loc[sorted_grp.index[0], "is_best"] = True
            continue
        best_idx   = sorted_grp.index[0]
        second_idx = sorted_grp.index[1]
        a = np.array(grp.loc[best_idx,   "raw_rewards"])
        b = np.array(grp.loc[second_idx, "raw_rewards"])
        p = wilcoxon_pval(a, b)
        df_summary.loc[best_idx, "is_best"] = (p < alpha)

    return df_summary


# ================================================================
# Main results (default / tuned HPs)
# ================================================================

def process_main(log_root: str, results_dir: str, hp_tag: str) -> pd.DataFrame:
    """
    Collect best-mean-reward per run from the training NPZs, then
    aggregate across env_sets and seeds into a per-(algorithm, num_robots)
    summary.
    """
    version = f"main_{hp_tag}"
    print(f"  Scanning {log_root}/{version}/")

    rows = collect_rewards(log_root, version,
                           ALGORITHMS, ROBOT_COUNTS, ENV_SETS, SEEDS)
    if not rows:
        print(f"  [WARN] No data found for main_{hp_tag} — skipping.")
        return None

    df = pd.DataFrame(rows)

    summary_rows = []
    for (alg, N), grp in df.groupby(["algorithm", "num_robots"]):
        r = grp["mean_reward"].astype(float).values
        summary_rows.append(dict(
            algorithm=alg,
            num_robots=N,
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            max_reward=float(np.max(r)),
            iqm=compute_iqm(r),
            raw_rewards=list(r),
            n_runs=len(r),
        ))

    summary = pd.DataFrame(summary_rows)
    summary = mark_best(summary, "mean_reward", group_cols=["num_robots"])

    # Machine-readable CSV
    out = summary.drop(columns=["raw_rewards"])
    out_path = os.path.join(results_dir, f"main_{hp_tag}_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")

    _write_latex_main(summary, hp_tag, results_dir)
    return summary


def _write_latex_main(summary: pd.DataFrame, hp_tag: str, results_dir: str):
    lines = [
        f"% LaTeX table rows for tab:{'default' if hp_tag == 'default' else 'random'}_hyp",
        f"% hp_tag = {hp_tag}", "",
    ]
    for alg in ALGORITHMS:
        cells = []
        for N in ROBOT_COUNTS:
            row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if row.empty:
                cells.append("---")
            else:
                r      = row.iloc[0]
                dagger = r"^\dagger" if r["is_best"] else ""
                s = (f"${r['mean_reward']:.1f} \\pm {r['std_reward']:.1f}"
                     f"\\ ({r['iqm']:.1f}){dagger}$")
                if r["is_best"]:
                    s = "$" + r"\mathbf{" + s[1:-1] + "}" + "$"
                cells.append(s)
        lines.append(f"{alg} & " + " & ".join(cells) + r" \\")

    out_path = os.path.join(results_dir, f"main_{hp_tag}_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Combined main table — tab:main  (IQM only, default + tuned)
# ================================================================

def write_latex_main_combined(default_summary: pd.DataFrame,
                               tuned_summary:   pd.DataFrame,
                               results_dir:     str):
    def _remark_iqm(summary):
        if summary is None:
            return None
        return mark_best(summary.copy(), value_col="iqm", group_cols=["num_robots"])

    def _iqm_cell(summary, alg, N):
        if summary is None:
            return "---"
        row = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
        if row.empty:
            return "---"
        r   = row.iloc[0]
        val = f"{r['iqm']:.1f}"
        return (rf"$\mathbf{{{val}}}^\dagger$" if r["is_best"]
                else f"${val}$")

    default_iqm = _remark_iqm(default_summary)
    tuned_iqm   = _remark_iqm(tuned_summary)

    lines = [
        r"% LaTeX table rows for tab:main (NeurIPS compressed)",
        r"% IQM only, default + tuned side-by-side.",
        r"% Paste between \midrule and \bottomrule.", "",
    ]
    for alg in ALGORITHMS:
        default_cells = [_iqm_cell(default_iqm, alg, N) for N in ROBOT_COUNTS]
        tuned_cells   = [_iqm_cell(tuned_iqm,   alg, N) for N in ROBOT_COUNTS]
        lines.append(
            f"    {alg:<6} & " + " & ".join(default_cells + tuned_cells) + r" \\"
        )

    out_path = os.path.join(results_dir, "main_combined_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Reward ablation  (Step 4)
# ================================================================

def process_ablation_reward(log_root: str, results_dir: str) -> pd.DataFrame:
    """
    CrossQ, N=3, ALL env sets (REQ 2), seeds × conditions.
    version pattern: ablation_reward_{condition}

    REQ (1): Reports mean episode length at best-reward checkpoint
             instead of IQM of episode reward.
    REQ (2): Uses all ENV_SETS (was hardcoded to [1]).
    """
    condition_labels = {
        "full":    "Full reward",
        "no_term": r"No $R_\text{term}$",
        "no_spr":  r"No $R_\text{spr}$",
        "no_path": r"No $R_\text{path}$",
    }
    condition_removed = {
        "full":    "---",
        "no_term": "collision penalty + success bonus",
        "no_spr":  "remaining-infection + useless spray penalty",
        "no_path": "energy + speed + path + time penalty",
    }

    summary_rows = []
    for cond, label in condition_labels.items():
        version = f"ablation_reward_{cond}"
        print(f"  Scanning {log_root}/{version}/")
        # REQ (2): was collect_rewards(..., [1], ...) — now all ENV_SETS
        rows = collect_rewards_and_ep_lengths(
            log_root, version, ["CrossQ"], [3], ENV_SETS, SEEDS)
        if not rows:
            print(f"    [WARN] No data for {version}")
            continue
        r       = np.array([row["mean_reward"]    for row in rows])
        ep_lens = np.array([row["mean_ep_length"] for row in rows])
        summary_rows.append(dict(
            condition=label,
            removed_term=condition_removed[cond],
            mean_reward=float(np.mean(r)),
            std_reward=float(np.std(r)),
            iqm=compute_iqm(r),
            # REQ (1): episode length statistics
            mean_ep_length=float(np.nanmean(ep_lens)),
            std_ep_length=float(np.nanstd(ep_lens)),
            raw_rewards=list(r),
        ))

    if not summary_rows:
        print("  [WARN] Reward ablation — no data found, skipping.")
        return None

    summary  = pd.DataFrame(summary_rows)
    out_path = os.path.join(results_dir, "ablation_reward_agg.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_reward(summary, results_dir)
    return summary


def _write_latex_ablation_reward(summary: pd.DataFrame, results_dir: str):
    """
    REQ (1): Emit mean episode length ± std as the primary metric instead
    of IQM of reward.  Shortest episode = fastest completion = best.

    LaTeX column layout:
      Condition & Removed terms & Mean reward & Std & Mean ep. length ± Std
    """
    lines = ["% LaTeX table rows for tab:ablation_reward", ""]
    # Best = lowest mean episode length (task completed fastest)
    best_ep_len = summary["mean_ep_length"].min()
    for _, r in summary.iterrows():
        bold = (r["mean_ep_length"] == best_ep_len)
        ep_s = (
            rf"\mathbf{{{r['mean_ep_length']:.1f} \pm {r['std_ep_length']:.1f}}}"
            if bold
            else rf"{r['mean_ep_length']:.1f} \pm {r['std_ep_length']:.1f}"
        )
        lines.append(
            f"{r['condition']} & {r['removed_term']} & "
            f"${r['mean_reward']:.1f}$ & ${r['std_reward']:.1f}$ & ${ep_s}$ \\\\"
        )
    out_path = os.path.join(results_dir, "ablation_reward_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Observation ablation  (Step 5)
# ================================================================

def process_ablation_obs(log_root: str, results_dir: str) -> pd.DataFrame:
    """
    CrossQ, N=3, ALL env sets (REQ 2), seeds × obs conditions.
    version pattern: ablation_obs_{condition}

    REQ (2): Uses all ENV_SETS (was hardcoded to [1]).
    """
    obs_dims = {"full": "5N+M", "no_pos": "N+M",
                "no_inf_hist": "5N", "pos_only": "2N"}
    labels   = {"full":        "Full obs.",
                "no_pos":      "No positions",
                "no_inf_hist": "No infection history",
                "pos_only":    "Positions only"}

    summary_rows = []
    for cond in ["full", "no_pos", "no_inf_hist", "pos_only"]:
        version = f"ablation_obs_{cond}"
        print(f"  Scanning {log_root}/{version}/")
        # REQ (2): was collect_rewards(..., [1], ...) — now all ENV_SETS
        rows = collect_rewards(log_root, version,
                               ["CrossQ"], [3], ENV_SETS, SEEDS)
        if not rows:
            print(f"    [WARN] No data for {version}")
            continue
        r = np.array([row["mean_reward"] for row in rows])
        summary_rows.append(dict(
            condition=labels.get(cond, cond),
            obs_dim=obs_dims.get(cond, "?"),
            mean_reward=float(np.mean(r)),
            iqm=compute_iqm(r),
            raw_rewards=list(r),
        ))

    if not summary_rows:
        print("  [WARN] Obs ablation — no data found, skipping.")
        return None

    summary  = pd.DataFrame(summary_rows)
    ref_iqm  = summary.iloc[0]["iqm"] if not summary.empty else 1.0
    summary["delta_iqm_pct"] = (
        (summary["iqm"] - ref_iqm) / (abs(ref_iqm) + 1e-9) * 100
    )
    out_path = os.path.join(results_dir, "ablation_obs_agg.csv")
    summary.drop(columns=["raw_rewards"]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_ablation_obs(summary, results_dir)
    return summary


def _write_latex_ablation_obs(summary: pd.DataFrame, results_dir: str):
    lines = ["% LaTeX table rows for tab:ablation_obs", ""]
    for _, r in summary.iterrows():
        is_baseline = (r["condition"] == "Full obs.")
        delta  = "---" if is_baseline else f"${r['delta_iqm_pct']:.1f}$"
        mean_s = (rf"\mathbf{{{r['mean_reward']:.1f}}}" if is_baseline
                  else f"{r['mean_reward']:.1f}")
        iqm_s  = (rf"\mathbf{{{r['iqm']:.1f}}}" if is_baseline
                  else f"{r['iqm']:.1f}")
        lines.append(
            f"{r['condition']} & ${r['obs_dim']}$ & "
            f"${mean_s}$ & ${iqm_s}$ & {delta} \\\\"
        )
    out_path = os.path.join(results_dir, "ablation_obs_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Uncertainty ablation  (Step 6)
# — reads results_dir/ablation_uncertainty.csv produced by eval_ablations.sh
# ================================================================

def process_ablation_uncertainty(results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, "ablation_uncertainty.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — skipping.")
        return None

    df = pd.read_csv(csv_path)
    df["mean_reward"] = df["mean_reward"].astype(float)

    train_modes = ["full", "wind_only", "act_only", "deterministic"]

    rows = []
    for train_mode in train_modes:
        grp_same = df[(df["ablation"] == train_mode) &
                      (df["eval_uncertainty_mode"] == train_mode)]
        grp_full = df[(df["ablation"] == train_mode) &
                      (df["eval_uncertainty_mode"] == "full")]
        grp_det  = df[(df["ablation"] == train_mode) &
                      (df["eval_uncertainty_mode"] == "deterministic")]
        rows.append(dict(
            train_condition=train_mode,
            eval_same_iqm=compute_iqm(grp_same["mean_reward"].values) if not grp_same.empty else float("nan"),
            eval_full_iqm=compute_iqm(grp_full["mean_reward"].values) if not grp_full.empty else float("nan"),
            eval_det_iqm =compute_iqm(grp_det["mean_reward"].values)  if not grp_det.empty  else float("nan"),
        ))

    summary  = pd.DataFrame(rows)
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

    def _fmt(v):
        return "---" if (isinstance(v, float) and np.isnan(v)) else f"${v:.1f}$"

    lines = ["% LaTeX table rows for tab:ablation_uncertainty", ""]
    for _, r in summary.iterrows():
        tc = r["train_condition"]
        lines.append(
            f"{tc.replace('_',' ').title()} & {noise_labels.get(tc,'?')} & "
            f"{_fmt(r['eval_same_iqm'])} & "
            f"{_fmt(r['eval_full_iqm'])} & "
            f"{_fmt(r['eval_det_iqm'])} \\\\"
        )
    out_path = os.path.join(results_dir, "ablation_uncertainty_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Domain randomization  (Step 7)
# — reads results_dir/dr_inDist.csv and dr_OOD.csv produced by eval_ablations.sh dr
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

    dr_conditions = [("none", "(A) no DR",   "none (standard training)"),
                     ("wind", "(B) wind DR",  "wind speed + direction"),
                     ("full", "(C) full DR",  "all parameters")]

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

    summary  = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "dr_results_agg.csv")
    summary.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    _write_latex_dr(summary, results_dir)
    return summary


def _write_latex_dr(summary: pd.DataFrame, results_dir: str):
    lines = ["% LaTeX table rows for tab:dr_results", ""]

    def _fmt(v, bold=False):
        if isinstance(v, float) and np.isnan(v):
            return "---"
        s = f"{v:.1f}"
        return (rf"\mathbf{{{s}}}^\dagger" if bold else s)

    valid_ood  = summary["ood_iqm"].dropna()
    valid_cvar = summary["cvar_0_1"].dropna()
    best_ood   = valid_ood.max()  if not valid_ood.empty  else None
    best_cvar  = valid_cvar.max() if not valid_cvar.empty else None

    for _, r in summary.iterrows():
        ood_s  = _fmt(r["ood_iqm"],  bold=(best_ood  is not None and r["ood_iqm"]  == best_ood))
        cvar_s = _fmt(r["cvar_0_1"], bold=(best_cvar is not None and r["cvar_0_1"] == best_cvar))
        lines.append(
            f"{r['condition']} & {r['rand_params']} & "
            f"${_fmt(r['in_dist_iqm'])}$ & ${ood_s}$ & ${cvar_s}$ \\\\"
        )
    out_path = os.path.join(results_dir, "dr_results_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Combined ablations + DR table — tab:ablations_and_dr
# ================================================================

def write_latex_combined_ablations_dr(reward_df:  pd.DataFrame,
                                       obs_df:     pd.DataFrame,
                                       unc_df:     pd.DataFrame,
                                       dr_df:      pd.DataFrame,
                                       results_dir: str):
    """
    10-column side-by-side table:
      reward_cond | reward_ep_len | obs_cond | obs_iqm | unc_cond | unc_iqm
      | dr_cond | in_dist_iqm | ood_iqm | cvar_0.1

    REQ (1): reward block now shows mean episode length instead of IQM.
    4 rows each for reward / obs / uncertainty; DR fills rows 0–2.
    """
    OBS_LABEL_MAP = {
        "Full obs.":            "Full",
        "No positions":         "No pos.",
        "No infection history": "No inf.",
        "Positions only":       "Pos. only",
    }
    UNC_LABEL_MAP = {
        "full":          "Full",
        "wind_only":     "Wind only",
        "act_only":      "Act. only",
        "deterministic": "Deterministic",
    }

    # Flatten uncertainty to one IQM per train condition (diagonal only)
    unc_simple = None
    if unc_df is not None:
        unc_rows = []
        for _, r in unc_df.iterrows():
            tc = r["train_condition"]
            unc_rows.append(dict(
                condition=UNC_LABEL_MAP.get(tc, tc),
                iqm=float(r.get("eval_same_iqm", float("nan"))),
            ))
        unc_simple = pd.DataFrame(unc_rows)

    # ── helpers ──────────────────────────────────────────────────

    def _safe_val(df, col, i):
        if df is None or i >= len(df):
            return float("nan")
        v = df[col].iloc[i]
        return float("nan") if (isinstance(v, float) and np.isnan(v)) else v

    def _fmt(val, is_best=False):
        if isinstance(val, float) and np.isnan(val):
            return ""
        s = f"{val:.1f}"
        return (rf"$\mathbf{{{s}}}^\dagger$" if is_best else f"${s}$")

    def _best_mask_iqm(df):
        if df is None or df.empty:
            return []
        vals       = df["iqm"].values
        sorted_idx = np.argsort(vals)[::-1]
        mask       = [False] * len(df)
        if len(sorted_idx) >= 2 and "raw_rewards" in df.columns:
            a = np.array(df["raw_rewards"].iloc[sorted_idx[0]])
            b = np.array(df["raw_rewards"].iloc[sorted_idx[1]])
            if wilcoxon_pval(a, b) < 0.05:
                mask[sorted_idx[0]] = True
        elif sorted_idx.size >= 1:
            mask[sorted_idx[0]] = True
        return mask

    def _best_mask_ep_len(df):
        """REQ (1): best = lowest mean_ep_length."""
        if df is None or df.empty or "mean_ep_length" not in df.columns:
            return []
        vals = df["mean_ep_length"].values
        mask = [False] * len(df)
        if not np.all(np.isnan(vals)):
            mask[int(np.nanargmin(vals))] = True
        return mask

    def _dr_best_mask(col):
        if dr_df is None or col not in dr_df.columns:
            return [False] * (len(dr_df) if dr_df is not None else 0)
        vals = dr_df[col].values
        m    = [False] * len(dr_df)
        if not np.all(np.isnan(vals)):
            m[int(np.nanargmax(vals))] = True
        return m

    # REQ (1): reward block uses episode-length best mask
    rew_best  = _best_mask_ep_len(reward_df)
    obs_best  = _best_mask_iqm(obs_df)
    unc_best  = [False] * (len(unc_simple) if unc_simple is not None else 0)
    if unc_simple is not None and not unc_simple.empty:
        non_nan = ~np.isnan(unc_simple["iqm"].values)
        if non_nan.any():
            unc_best[int(np.nanargmax(unc_simple["iqm"].values))] = True

    dr_ood_best  = _dr_best_mask("ood_iqm")
    dr_cvar_best = _dr_best_mask("cvar_0_1")

    # ── build rows ────────────────────────────────────────────────
    lines = [
        r"% LaTeX table rows for tab:ablations_and_dr (NeurIPS compressed)",
        r"% Paste between \midrule and \bottomrule.", "",
    ]

    for i in range(4):
        # reward block — REQ (1): show mean_ep_length instead of IQM
        r_cond   = (str(_safe_val(reward_df, "condition", i))
                    if reward_df is not None and i < len(reward_df) else "")
        r_ep_len = _fmt(_safe_val(reward_df, "mean_ep_length", i),
                        rew_best[i] if i < len(rew_best) else False)

        # obs block
        raw_o  = (str(_safe_val(obs_df, "condition", i))
                  if obs_df is not None and i < len(obs_df) else "")
        o_cond = OBS_LABEL_MAP.get(raw_o, raw_o)
        o_iqm  = _fmt(_safe_val(obs_df, "iqm", i),
                      obs_best[i] if i < len(obs_best) else False)

        # uncertainty block
        u_cond = (str(_safe_val(unc_simple, "condition", i))
                  if unc_simple is not None and i < len(unc_simple) else "")
        u_iqm  = _fmt(_safe_val(unc_simple, "iqm", i),
                      unc_best[i] if i < len(unc_best) else False)

        # DR block (3 rows only; row 3 is blank)
        if i < 3 and dr_df is not None and i < len(dr_df):
            d_cond   = str(dr_df["condition"].iloc[i])
            d_indist = _fmt(dr_df["in_dist_iqm"].iloc[i], False)
            d_ood    = _fmt(dr_df["ood_iqm"].iloc[i],
                            dr_ood_best[i] if i < len(dr_ood_best) else False)
            d_cvar   = _fmt(dr_df["cvar_0_1"].iloc[i],
                            dr_cvar_best[i] if i < len(dr_cvar_best) else False)
        else:
            d_cond, d_indist, d_ood, d_cvar = "", "", "", ""

        lines.append(
            f"    {r_cond} & {r_ep_len}"
            f" & {o_cond} & {o_iqm}"
            f" & {u_cond} & {u_iqm}"
            f" & {d_cond} & {d_indist} & {d_ood} & {d_cvar}"
            r" \\"
        )

    out_path = os.path.join(results_dir, "combined_ablations_dr_latex_rows.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


# ================================================================
# Console summary
# ================================================================

def print_summary_table(summary: pd.DataFrame, label: str, hp_tag: str):
    if summary is None:
        return
    print(f"\n{'=' * 80}")
    print(f"  {label} ({hp_tag})")
    print(f"{'=' * 80}")
    header = f"{'Algorithm':10s}" + "".join(f"  N={N:5}" for N in ROBOT_COUNTS)
    print(header)
    print("-" * len(header))
    for alg in ALGORITHMS:
        row_str = f"{alg:10s}"
        for N in ROBOT_COUNTS:
            r = summary[(summary["algorithm"] == alg) & (summary["num_robots"] == N)]
            if r.empty:
                row_str += "     ---   "
            else:
                ri   = r.iloc[0]
                flag = "*" if ri["is_best"] else " "
                row_str += f"  {ri['mean_reward']:6.1f}{flag}"
        print(row_str)
    print(f"{'=' * 80}")


# ================================================================
# Entry point
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate evaluations.npz from training logs into "
                    "LaTeX-ready result tables.")
    p.add_argument("--log_root",    type=str, default="logs",
                   help="Root directory written by train.py (default: logs)")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Output directory for CSVs and LaTeX rows "
                        "(default: results)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nLog root    : {args.log_root}")
    print(f"Results dir : {args.results_dir}")

    print("\n── Main results: default HPs ───────────────────────────────")
    default_summary = process_main(args.log_root, args.results_dir, "default")
    print_summary_table(default_summary, "Main results", "default")

    print("\n── Main results: tuned HPs ─────────────────────────────────")
    tuned_summary = process_main(args.log_root, args.results_dir, "tuned")
    print_summary_table(tuned_summary, "Main results", "tuned")

    print("\n── Reward ablation ─────────────────────────────────────────")
    reward_summary = process_ablation_reward(args.log_root, args.results_dir)

    print("\n── Observation ablation ────────────────────────────────────")
    obs_summary = process_ablation_obs(args.log_root, args.results_dir)

    print("\n── Uncertainty ablation ────────────────────────────────────")
    unc_summary = process_ablation_uncertainty(args.results_dir)

    print("\n── Domain randomization ────────────────────────────────────")
    dr_summary = process_dr(args.results_dir)

    # NeurIPS compressed tables
    write_latex_main_combined(default_summary, tuned_summary, args.results_dir)
    write_latex_combined_ablations_dr(
        reward_summary, obs_summary, unc_summary, dr_summary, args.results_dir
    )

    print("\n✓ analyze_results.py complete.")
    print("  LaTeX row files are in results/*_latex_rows.txt")
    print("  NOTE: uncertainty ablation and DR results are read from CSVs in")
    print("  --results_dir (produced by eval_ablations.sh).  All other results")
    print("  are read directly from training evaluations.npz files in --log_root.")


if __name__ == "__main__":
    main()