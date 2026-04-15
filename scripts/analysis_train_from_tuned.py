from __future__ import annotations

# EDITED:
"""Analyze train-from-tuned results and compare them against default training.

This continues the existing analysis pattern but centers the report on the
``train_from_tuned.py`` stage. The script summarizes:
- coverage of the expected train-from-tuned runs implied by the Slurm scripts
- learning-curve and per-run metrics for the train-from-tuned experiments
- direct matched comparisons against the default training runs
- provenance links from each train-from-tuned run back to the tuning run that
  supplied its hyperparameters
- optional comparisons against the best Optuna tuning objective of the source
  tuning run

Outputs are written to ``plotting/results`` and ``plotting/plots``.
"""

# EDITED:
import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# EDITED:
from analysis import (
    add_artifact,
    build_expected_inventory,
    build_training_algorithm_summary,
    build_training_run_summary,
    build_training_set_summary,
    build_tuning_run_summary,
    coverage_summary_from_inventory,
    dataframe_for_csv,
    discover_training_progress,
    discover_tuning_trials,
    find_project_root,
    format_number,
    format_reward_millions,
    markdown_table,
    ordered_algorithms,
    parse_slurm_spec,
    plot_boxplot,
    plot_comparison_bars,
    plot_coverage_heatmap,
    plot_mean_std_curves,
    plot_reward_heatmap,
    resolve_output_root,
    top_algorithm,
    build_algorithm_curves,
)


# EDITED:
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run_name>\S+)")
TUNING_RUN_RE = re.compile(
    r"Loaded tuned hyperparameters from (?P<tuning_run_name>\S+):\s*(?P<params>\{.*\})"
)
TUNED_PARAMS_RE = re.compile(r"Tuned hyperparameters used for training:\s*(?P<params>\{.*\})")
TUNED_SOURCES_RE = re.compile(r"Tuned hyperparameter sources:\s*(?P<sources>\{.*\})")


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze train-from-tuned results, compare them with default training, "
            "and write tables, figures, and a markdown report to plotting/results "
            "and plotting/plots."
        )
    )
    parser.add_argument("--project-root", type=str, default=None, help="Optional repository root override.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional override for the plotting output directory.")
    parser.add_argument("--from-tuned-exp-name", type=str, default="from_tuned", help="Training experiment name used in logs/training_<exp_name>_logs for the train-from-tuned runs.")
    parser.add_argument("--baseline-exp-name", type=str, default="default", help="Baseline training experiment name used in logs/training_<exp_name>_logs.")
    parser.add_argument("--tuning-exp-name", type=str, default="random", help="Tuning experiment name encoded in run names such as *_random_3_robots_*.")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to match in run names.")
    parser.add_argument("--from-tuned-log-root", type=str, default=None, help="Optional explicit train-from-tuned log root.")
    parser.add_argument("--baseline-log-root", type=str, default=None, help="Optional explicit default-training log root.")
    parser.add_argument("--slurm-out-dir", type=str, default=None, help="Optional override for slurm_scripts/slurm_out.")
    parser.add_argument(
        "--from-tuned-slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/train_from_tuned_all_3_robots.sh",
            "slurm_scripts/crossq_from_tuned_3_robots.sh",
        ],
        help="Train-from-tuned Slurm scripts whose expected runs should be summarized.",
    )
    parser.add_argument(
        "--tuning-slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/train_random_all_3_robots.sh",
            "slurm_scripts/crossq_random_3_robots.sh",
        ],
        help="Optional tuning Slurm scripts used only for contextual reporting.",
    )
    parser.add_argument("--points", type=int, default=200, help="Interpolation points for mean/std line plots.")
    return parser.parse_args()


# EDITED:
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


# EDITED:
def safe_literal_eval(raw: Optional[str]):
    if raw is None:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


# EDITED:
def discover_from_tuned_provenance(slurm_out_dir: Path, exp_name: str, num_robots: int) -> pd.DataFrame:
    if not slurm_out_dir.exists():
        return pd.DataFrame()

    pair_map: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for path in sorted(slurm_out_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".out", ".err"}:
            continue

        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        job_name, job_id, task_text = parts
        try:
            task_id = int(task_text)
        except ValueError:
            continue

        key = (job_name, job_id, task_id)
        pair = pair_map.get(key)
        if pair is None:
            pair = {
                "job_name": job_name,
                "job_id": job_id,
                "task_id": task_id,
                "out_path": None,
                "err_path": None,
                "out_text": "",
                "err_text": "",
                "mtime": -1,
            }
            pair_map[key] = pair

        if path.suffix == ".out":
            pair["out_path"] = path
            pair["out_text"] = read_text(path)
        else:
            pair["err_path"] = path
            pair["err_text"] = read_text(path)

        try:
            pair["mtime"] = max(pair["mtime"], int(path.stat().st_mtime))
        except OSError:
            pass

    latest_by_run_name: Dict[str, Tuple[int, int, Dict[str, object]]] = {}
    for pair in pair_map.values():
        combined_text = "\n".join(part for part in [pair["out_text"], pair["err_text"]] if part)
        match = RUN_NAME_RE.search(combined_text)
        if match is None:
            continue

        run_name = match.group("run_name")
        if f"_{exp_name}_{num_robots}_robots_" not in run_name:
            continue

        tuning_run_name = None
        tuned_params = {}
        tuned_sources = {}

        tuning_matches = list(TUNING_RUN_RE.finditer(combined_text))
        if tuning_matches:
            tuning_run_name = tuning_matches[-1].group("tuning_run_name")
            maybe_params = safe_literal_eval(tuning_matches[-1].group("params"))
            if isinstance(maybe_params, dict):
                tuned_params = maybe_params

        tuned_param_matches = list(TUNED_PARAMS_RE.finditer(combined_text))
        if tuned_param_matches:
            maybe_params = safe_literal_eval(tuned_param_matches[-1].group("params"))
            if isinstance(maybe_params, dict):
                tuned_params = maybe_params

        tuned_source_matches = list(TUNED_SOURCES_RE.finditer(combined_text))
        if tuned_source_matches:
            maybe_sources = safe_literal_eval(tuned_source_matches[-1].group("sources"))
            if isinstance(maybe_sources, dict):
                tuned_sources = maybe_sources

        record = {
            "run_name": run_name,
            "tuning_run_name": tuning_run_name,
            "tuned_hyperparameters_json": json.dumps(tuned_params, sort_keys=True) if tuned_params else "",
            "tuned_param_count": len(tuned_params),
            "tuned_hyperparameter_sources_json": json.dumps(tuned_sources, sort_keys=True) if tuned_sources else "",
            "slurm_job_name": pair["job_name"],
            "slurm_job_id": pair["job_id"],
            "slurm_task_id": pair["task_id"],
            "slurm_out": str(pair["out_path"]) if pair["out_path"] is not None else None,
            "slurm_err": str(pair["err_path"]) if pair["err_path"] is not None else None,
        }

        job_rank = int(pair["job_id"]) if str(pair["job_id"]).isdigit() else -1
        rank = (job_rank, int(pair["mtime"]))
        previous = latest_by_run_name.get(run_name)
        if previous is None or rank >= (previous[0], previous[1]):
            latest_by_run_name[run_name] = (rank[0], rank[1], record)

    if not latest_by_run_name:
        return pd.DataFrame()

    return pd.DataFrame([item[2] for item in latest_by_run_name.values()]).sort_values("run_name").reset_index(drop=True)


# EDITED:
def build_matched_training_comparison(from_tuned_run_summary: pd.DataFrame, baseline_run_summary: pd.DataFrame) -> pd.DataFrame:
    if from_tuned_run_summary.empty or baseline_run_summary.empty:
        return pd.DataFrame()

    left = from_tuned_run_summary.copy()
    right = baseline_run_summary.copy()

    join_cols = ["algorithm", "set", "seed", "device"]
    merged = left.merge(right, on=join_cols, how="inner", suffixes=("_from_tuned", "_default"))
    if merged.empty:
        return merged

    merged["final_reward_delta"] = merged["final_reward_from_tuned"] - merged["final_reward_default"]
    merged["best_reward_delta"] = merged["best_reward_from_tuned"] - merged["best_reward_default"]
    merged["auc_reward_delta"] = merged["auc_reward_from_tuned"] - merged["auc_reward_default"]
    merged["first_reward_delta"] = merged["first_reward_from_tuned"] - merged["first_reward_default"]
    merged["improved_final_reward"] = merged["final_reward_delta"] > 0
    merged["improved_best_reward"] = merged["best_reward_delta"] > 0
    merged["improved_auc_reward"] = merged["auc_reward_delta"] > 0
    return merged.sort_values(["algorithm", "set", "seed", "device"]).reset_index(drop=True)


# EDITED:
def build_training_delta_algorithm_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()

    summary = (
        comparison_df.groupby("algorithm")
        .agg(
            matched_runs=("run_name_from_tuned", "nunique"),
            mean_final_reward_default=("final_reward_default", "mean"),
            mean_final_reward_from_tuned=("final_reward_from_tuned", "mean"),
            mean_best_reward_default=("best_reward_default", "mean"),
            mean_best_reward_from_tuned=("best_reward_from_tuned", "mean"),
            mean_auc_reward_default=("auc_reward_default", "mean"),
            mean_auc_reward_from_tuned=("auc_reward_from_tuned", "mean"),
            mean_final_reward_delta=("final_reward_delta", "mean"),
            mean_best_reward_delta=("best_reward_delta", "mean"),
            mean_auc_reward_delta=("auc_reward_delta", "mean"),
            max_best_reward_delta=("best_reward_delta", "max"),
            min_best_reward_delta=("best_reward_delta", "min"),
            improved_best_fraction=("improved_best_reward", "mean"),
            improved_final_fraction=("improved_final_reward", "mean"),
            improved_auc_fraction=("improved_auc_reward", "mean"),
        )
        .reset_index()
    )
    ordered = ordered_algorithms(summary["algorithm"].tolist())
    return summary.set_index("algorithm").loc[ordered].reset_index()


# EDITED:
def build_training_delta_set_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()
    summary = (
        comparison_df.groupby(["algorithm", "set"])
        .agg(
            matched_runs=("run_name_from_tuned", "nunique"),
            mean_final_reward_delta=("final_reward_delta", "mean"),
            mean_best_reward_delta=("best_reward_delta", "mean"),
            mean_auc_reward_delta=("auc_reward_delta", "mean"),
            improved_best_fraction=("improved_best_reward", "mean"),
            improved_final_fraction=("improved_final_reward", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["algorithm", "set"]).reset_index(drop=True)


# EDITED:
def build_from_tuned_vs_tuning_run_comparison(from_tuned_run_summary: pd.DataFrame, tuning_run_summary: pd.DataFrame) -> pd.DataFrame:
    if from_tuned_run_summary.empty or tuning_run_summary.empty or "tuning_run_name" not in from_tuned_run_summary.columns:
        return pd.DataFrame()

    left = from_tuned_run_summary[from_tuned_run_summary["tuning_run_name"].notna()].copy()
    right = tuning_run_summary.copy().rename(
        columns={
            "run_name": "tuning_run_name",
            "best_reward": "tuning_best_reward",
            "best_trial": "tuning_best_trial",
            "completed_trials": "tuning_completed_trials",
            "failed_trials": "tuning_failed_trials",
            "pruned_trials": "tuning_pruned_trials",
        }
    )

    merged = left.merge(
        right[
            [
                "tuning_run_name",
                "tuning_best_reward",
                "tuning_best_trial",
                "tuning_completed_trials",
                "tuning_failed_trials",
                "tuning_pruned_trials",
            ]
        ],
        on="tuning_run_name",
        how="left",
    )
    if merged.empty:
        return merged

    merged["from_tuned_minus_tuning_best"] = merged["best_reward"] - merged["tuning_best_reward"]
    merged["from_tuned_final_minus_tuning_best"] = merged["final_reward"] - merged["tuning_best_reward"]
    return merged.sort_values(["algorithm", "set", "seed", "run_name"]).reset_index(drop=True)


# EDITED:
def build_from_tuned_vs_tuning_algorithm_summary(run_comparison_df: pd.DataFrame) -> pd.DataFrame:
    if run_comparison_df.empty:
        return pd.DataFrame()
    summary = (
        run_comparison_df.groupby("algorithm")
        .agg(
            matched_runs=("run_name", "nunique"),
            mean_best_reward_from_tuned=("best_reward", "mean"),
            mean_tuning_best_reward=("tuning_best_reward", "mean"),
            mean_from_tuned_minus_tuning_best=("from_tuned_minus_tuning_best", "mean"),
            max_from_tuned_minus_tuning_best=("from_tuned_minus_tuning_best", "max"),
            min_from_tuned_minus_tuning_best=("from_tuned_minus_tuning_best", "min"),
        )
        .reset_index()
    )
    ordered = ordered_algorithms(summary["algorithm"].tolist())
    return summary.set_index("algorithm").loc[ordered].reset_index()


# EDITED:
def build_provenance_algorithm_summary(from_tuned_run_summary: pd.DataFrame) -> pd.DataFrame:
    if from_tuned_run_summary.empty or "tuning_run_name" not in from_tuned_run_summary.columns:
        return pd.DataFrame()
    summary = (
        from_tuned_run_summary.groupby("algorithm")
        .agg(
            runs=("run_name", "nunique"),
            linked_tuning_runs=("tuning_run_name", lambda series: series.dropna().nunique()),
            linked_runs=("tuning_run_name", lambda series: series.notna().sum()),
            mean_tuned_param_count=("tuned_param_count", "mean"),
            mean_best_reward=("best_reward", "mean"),
            mean_final_reward=("final_reward", "mean"),
        )
        .reset_index()
    )
    ordered = ordered_algorithms(summary["algorithm"].tolist())
    return summary.set_index("algorithm").loc[ordered].reset_index()


# EDITED:
def plot_delta_heatmap(summary_df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    if summary_df.empty or value_col not in summary_df.columns:
        return

    pivot = summary_df.pivot_table(index="algorithm", columns="set", values=value_col, aggfunc="mean")
    if pivot.empty:
        return

    ordered = [algo for algo in ordered_algorithms(pivot.index.tolist()) if algo in pivot.index]
    pivot = pivot.loc[ordered]
    display = pivot / 1_000_000.0

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(display.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(display.shape[1]))
    ax.set_xticklabels(display.columns.tolist())
    ax.set_yticks(np.arange(display.shape[0]))
    ax.set_yticklabels(display.index.tolist())
    ax.set_xlabel("Set")
    ax.set_ylabel("Algorithm")
    ax.set_title(title)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display.iat[i, j]
            text = "-" if pd.isna(value) else f"{value:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label=r"Delta reward (x$10^6$)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_delta_boxplot(run_comparison_df: pd.DataFrame, value_col: str, ylabel: str, title: str, path: Path) -> None:
    if run_comparison_df.empty or value_col not in run_comparison_df.columns:
        return

    labels = []
    data = []
    for algorithm in ordered_algorithms(run_comparison_df["algorithm"].unique()):
        values = (run_comparison_df.loc[run_comparison_df["algorithm"] == algorithm, value_col] / 1_000_000.0).dropna().to_list()
        if not values:
            continue
        labels.append(algorithm)
        data.append(values)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def build_report(
    report_path: Path,
    artifacts_df: pd.DataFrame,
    warnings: List[str],
    from_tuned_specs: List[object],
    coverage_summary: pd.DataFrame,
    from_tuned_roots: List[Path],
    from_tuned_algo_summary: pd.DataFrame,
    baseline_algo_summary: pd.DataFrame,
    provenance_algo_summary: pd.DataFrame,
    delta_algo_summary: pd.DataFrame,
    tuning_link_algo_summary: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Train-from-tuned analysis report")
    lines.append("")
    lines.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    lines.append("")
    lines.append(
        "This report summarizes the train-from-tuned experiments, compares them against the default training runs, and tracks each run back to the tuning run that supplied its hyperparameters when that provenance could be parsed from Slurm output files."
    )
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Parsed train-from-tuned Slurm scripts")
    lines.append("")
    if from_tuned_specs:
        rows = []
        for spec in from_tuned_specs:
            rows.append(
                {
                    "script": spec.path.name,
                    "algorithms": len(spec.algorithms),
                    "sets": len(spec.sets),
                    "seeds": len(spec.seeds),
                    "steps": spec.steps,
                    "exp_name": spec.exp_name or "-",
                }
            )
        lines.append(markdown_table(pd.DataFrame(rows), max_rows=20))
    else:
        lines.append("_No train-from-tuned Slurm scripts were parsed._")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    if coverage_summary.empty:
        lines.append("_No coverage summary could be produced._")
    else:
        found_total = int(coverage_summary["found_runs"].sum())
        expected_total = int(coverage_summary["expected_runs"].sum())
        lines.append(f"Across the train-from-tuned launch scripts, **{found_total}** outputs were found out of **{expected_total}** expected runs.")
        lines.append("")
        lines.append(markdown_table(coverage_summary, formatters={"coverage_ratio": lambda v: format_number(v, 2)}, max_rows=30))
    lines.append("")

    lines.append("## Train-from-tuned performance")
    lines.append("")
    if from_tuned_algo_summary.empty:
        lines.append("_No train-from-tuned progress logs were discovered._")
    else:
        lines.append(f"Usable train-from-tuned learning curves were discovered under: {', '.join(str(path) for path in from_tuned_roots) if from_tuned_roots else '-' }.")
        lines.append("")
        best_peak_algo, best_peak_value = top_algorithm(from_tuned_algo_summary, "mean_best_reward", higher_is_better=True)
        best_final_algo, best_final_value = top_algorithm(from_tuned_algo_summary, "mean_final_reward", higher_is_better=True)
        if best_peak_algo is not None:
            lines.append(f"Highest mean peak reward after training from tuned hyperparameters: **{best_peak_algo}** at **{format_reward_millions(best_peak_value)} x10^6**.")
        if best_final_algo is not None:
            lines.append(f"Highest mean final logged reward in the train-from-tuned runs: **{best_final_algo}** at **{format_reward_millions(best_final_value)} x10^6**.")
        lines.append("")
        lines.append(markdown_table(
            from_tuned_algo_summary[[
                "algorithm",
                "runs",
                "mean_final_reward",
                "mean_best_reward",
                "std_best_reward",
                "best_of_best_reward",
                "mean_auc_reward",
            ]],
            formatters={
                "mean_final_reward": format_reward_millions,
                "mean_best_reward": format_reward_millions,
                "std_best_reward": format_reward_millions,
                "best_of_best_reward": format_reward_millions,
                "mean_auc_reward": format_reward_millions,
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Comparison against default training")
    lines.append("")
    if delta_algo_summary.empty:
        lines.append("_No matched default-vs-from-tuned comparison could be produced._")
    else:
        uplift_algo, uplift_value = top_algorithm(delta_algo_summary, "mean_best_reward_delta", higher_is_better=True)
        stable_uplift_algo, stable_uplift_value = top_algorithm(delta_algo_summary, "mean_auc_reward_delta", higher_is_better=True)
        if uplift_algo is not None:
            lines.append(f"Largest mean best-reward uplift over default training: **{uplift_algo}** at **{format_reward_millions(uplift_value)} x10^6**.")
        if stable_uplift_algo is not None:
            lines.append(f"Largest mean AUC-style reward uplift over default training: **{stable_uplift_algo}** at **{format_reward_millions(stable_uplift_value)} x10^6**.")
        lines.append("")
        lines.append(markdown_table(
            delta_algo_summary[[
                "algorithm",
                "matched_runs",
                "mean_best_reward_default",
                "mean_best_reward_from_tuned",
                "mean_best_reward_delta",
                "mean_auc_reward_delta",
                "improved_best_fraction",
                "improved_final_fraction",
            ]],
            formatters={
                "mean_best_reward_default": format_reward_millions,
                "mean_best_reward_from_tuned": format_reward_millions,
                "mean_best_reward_delta": format_reward_millions,
                "mean_auc_reward_delta": format_reward_millions,
                "improved_best_fraction": lambda v: format_number(100 * float(v), 1) + "%",
                "improved_final_fraction": lambda v: format_number(100 * float(v), 1) + "%",
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Tuning provenance")
    lines.append("")
    if provenance_algo_summary.empty:
        lines.append("_No train-from-tuned provenance could be parsed from Slurm outputs._")
    else:
        lines.append("These summaries show how many train-from-tuned runs could be linked back to their source tuning runs, and how many tuned parameters were typically injected into each training job.")
        lines.append("")
        lines.append(markdown_table(
            provenance_algo_summary,
            formatters={
                "mean_tuned_param_count": lambda v: format_number(v, 2),
                "mean_best_reward": format_reward_millions,
                "mean_final_reward": format_reward_millions,
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Train-from-tuned versus tuning objective")
    lines.append("")
    if tuning_link_algo_summary.empty:
        lines.append("_No from-tuned-versus-tuning comparison could be produced._")
    else:
        uplift_algo, uplift_value = top_algorithm(tuning_link_algo_summary, "mean_from_tuned_minus_tuning_best", higher_is_better=True)
        if uplift_algo is not None:
            lines.append(f"Largest mean gap between the final train-from-tuned peak and the best source tuning objective: **{uplift_algo}** at **{format_reward_millions(uplift_value)} x10^6**.")
            lines.append("")
        lines.append(markdown_table(
            tuning_link_algo_summary,
            formatters={
                "mean_best_reward_from_tuned": format_reward_millions,
                "mean_tuning_best_reward": format_reward_millions,
                "mean_from_tuned_minus_tuning_best": format_reward_millions,
                "max_from_tuned_minus_tuning_best": format_reward_millions,
                "min_from_tuned_minus_tuning_best": format_reward_millions,
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Generated artifacts")
    lines.append("")
    if artifacts_df.empty:
        lines.append("_No artifacts were recorded._")
    else:
        lines.append(markdown_table(artifacts_df[["category", "kind", "filename", "description"]], max_rows=200))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# EDITED:
def main() -> None:
    args = parse_args()

    project_root = find_project_root(args.project_root)
    output_root = resolve_output_root(project_root, args.output_root)
    results_dir = output_root / "results"
    plots_dir = output_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    slurm_out_dir = Path(args.slurm_out_dir).expanduser().resolve() if args.slurm_out_dir else (project_root / "slurm_scripts" / "slurm_out").resolve()

    warnings: List[str] = []

    from_tuned_specs = []
    for script_relpath in args.from_tuned_slurm_scripts:
        script_path = (project_root / script_relpath).resolve()
        if not script_path.exists():
            warnings.append(f"Missing train-from-tuned Slurm script: {script_path}")
            continue
        try:
            from_tuned_specs.append(parse_slurm_spec(script_path))
        except Exception as exc:
            warnings.append(f"Failed to parse train-from-tuned Slurm script {script_path}: {exc}")

    # Tuning scripts are optional context only; parse failures should not stop the report.
    for script_relpath in args.tuning_slurm_scripts:
        script_path = (project_root / script_relpath).resolve()
        if not script_path.exists():
            warnings.append(f"Missing tuning Slurm script: {script_path}")

    inventory_df = build_expected_inventory(from_tuned_specs)

    from_tuned_raw_df, from_tuned_roots = discover_training_progress(
        project_root=project_root,
        exp_name=args.from_tuned_exp_name,
        num_robots=args.num_robots,
        explicit_log_root=args.from_tuned_log_root,
    )
    if from_tuned_raw_df.empty:
        warnings.append("No usable train-from-tuned progress.csv files were found.")
    else:
        from_tuned_raw_df["reward_scaled"] = from_tuned_raw_df["reward"] / 1_000_000.0
        from_tuned_raw_df["reward_scaled"] = from_tuned_raw_df["reward_scaled"].clip(lower=-2)

    baseline_raw_df, baseline_roots = discover_training_progress(
        project_root=project_root,
        exp_name=args.baseline_exp_name,
        num_robots=args.num_robots,
        explicit_log_root=args.baseline_log_root,
    )
    if baseline_raw_df.empty:
        warnings.append("No usable baseline default-training progress.csv files were found.")
    else:
        baseline_raw_df["reward_scaled"] = baseline_raw_df["reward"] / 1_000_000.0
        baseline_raw_df["reward_scaled"] = baseline_raw_df["reward_scaled"].clip(lower=-2)

    tuning_trial_df, tuning_run_files_df = discover_tuning_trials(
        slurm_out_dir=slurm_out_dir,
        exp_name=args.tuning_exp_name,
        num_robots=args.num_robots,
    )
    if not tuning_trial_df.empty:
        tuning_trial_df["reward_scaled"] = tuning_trial_df["reward"] / 1_000_000.0
        tuning_trial_df["reward_scaled"] = tuning_trial_df["reward_scaled"].clip(lower=-2)
    else:
        warnings.append("No usable Optuna tuning trial rows were found in Slurm output files.")

    from_tuned_run_summary = build_training_run_summary(from_tuned_raw_df)
    from_tuned_set_summary = build_training_set_summary(from_tuned_run_summary)
    from_tuned_algo_summary = build_training_algorithm_summary(from_tuned_run_summary)

    baseline_run_summary = build_training_run_summary(baseline_raw_df)
    baseline_set_summary = build_training_set_summary(baseline_run_summary)
    baseline_algo_summary = build_training_algorithm_summary(baseline_run_summary)

    tuning_run_summary = build_tuning_run_summary(tuning_trial_df)

    provenance_df = discover_from_tuned_provenance(slurm_out_dir, args.from_tuned_exp_name, args.num_robots)
    if provenance_df.empty:
        warnings.append("No train-from-tuned provenance rows were parsed from Slurm output files.")
    elif not from_tuned_run_summary.empty:
        from_tuned_run_summary = from_tuned_run_summary.merge(provenance_df, on="run_name", how="left")

    provenance_algo_summary = build_provenance_algorithm_summary(from_tuned_run_summary)

    training_delta_run_df = build_matched_training_comparison(from_tuned_run_summary, baseline_run_summary)
    training_delta_algo_summary = build_training_delta_algorithm_summary(training_delta_run_df)
    training_delta_set_summary = build_training_delta_set_summary(training_delta_run_df)

    from_tuned_vs_tuning_run_df = build_from_tuned_vs_tuning_run_comparison(from_tuned_run_summary, tuning_run_summary)
    from_tuned_vs_tuning_algo_summary = build_from_tuned_vs_tuning_algorithm_summary(from_tuned_vs_tuning_run_df)

    discovered_from_tuned_names = set(from_tuned_run_summary["run_name"]) if not from_tuned_run_summary.empty else set()
    if not inventory_df.empty:
        inventory_df = inventory_df.copy()
        inventory_df["expected_output_found"] = inventory_df["run_name"].isin(discovered_from_tuned_names)
        inventory_df["coverage_status"] = np.where(inventory_df["expected_output_found"], "FOUND", "MISSING")
    coverage_summary = coverage_summary_from_inventory(inventory_df) if not inventory_df.empty else pd.DataFrame()

    artifacts: List[Dict[str, str]] = []

    from_tuned_raw_path = results_dir / "analysis_from_tuned_raw.csv"
    from_tuned_run_summary_path = results_dir / "analysis_from_tuned_run_summary.csv"
    from_tuned_set_summary_path = results_dir / "analysis_from_tuned_set_summary.csv"
    from_tuned_algo_summary_path = results_dir / "analysis_from_tuned_algorithm_summary.csv"
    baseline_run_summary_path = results_dir / "analysis_default_baseline_run_summary.csv"
    baseline_set_summary_path = results_dir / "analysis_default_baseline_set_summary.csv"
    baseline_algo_summary_path = results_dir / "analysis_default_baseline_algorithm_summary.csv"
    provenance_path = results_dir / "analysis_from_tuned_provenance.csv"
    provenance_algo_summary_path = results_dir / "analysis_from_tuned_provenance_algorithm_summary.csv"
    training_delta_run_path = results_dir / "analysis_from_tuned_vs_default_run_comparison.csv"
    training_delta_set_path = results_dir / "analysis_from_tuned_vs_default_set_comparison.csv"
    training_delta_algo_path = results_dir / "analysis_from_tuned_vs_default_algorithm_comparison.csv"
    tuning_run_summary_path = results_dir / "analysis_tuning_run_summary_for_context.csv"
    from_tuned_vs_tuning_run_path = results_dir / "analysis_from_tuned_vs_tuning_run_comparison.csv"
    from_tuned_vs_tuning_algo_path = results_dir / "analysis_from_tuned_vs_tuning_algorithm_comparison.csv"
    inventory_path = results_dir / "analysis_from_tuned_run_inventory.csv"
    coverage_summary_path = results_dir / "analysis_from_tuned_coverage_summary.csv"
    manifest_path = results_dir / "analysis_from_tuned_artifact_manifest.csv"
    report_path = results_dir / "analysis_from_tuned_report.md"

    dataframe_for_csv(from_tuned_raw_df).to_csv(from_tuned_raw_path, index=False)
    dataframe_for_csv(from_tuned_run_summary).to_csv(from_tuned_run_summary_path, index=False)
    dataframe_for_csv(from_tuned_set_summary).to_csv(from_tuned_set_summary_path, index=False)
    dataframe_for_csv(from_tuned_algo_summary).to_csv(from_tuned_algo_summary_path, index=False)
    dataframe_for_csv(baseline_run_summary).to_csv(baseline_run_summary_path, index=False)
    dataframe_for_csv(baseline_set_summary).to_csv(baseline_set_summary_path, index=False)
    dataframe_for_csv(baseline_algo_summary).to_csv(baseline_algo_summary_path, index=False)
    dataframe_for_csv(provenance_df).to_csv(provenance_path, index=False)
    dataframe_for_csv(provenance_algo_summary).to_csv(provenance_algo_summary_path, index=False)
    dataframe_for_csv(training_delta_run_df).to_csv(training_delta_run_path, index=False)
    dataframe_for_csv(training_delta_set_summary).to_csv(training_delta_set_path, index=False)
    dataframe_for_csv(training_delta_algo_summary).to_csv(training_delta_algo_path, index=False)
    dataframe_for_csv(tuning_run_summary).to_csv(tuning_run_summary_path, index=False)
    dataframe_for_csv(from_tuned_vs_tuning_run_df).to_csv(from_tuned_vs_tuning_run_path, index=False)
    dataframe_for_csv(from_tuned_vs_tuning_algo_summary).to_csv(from_tuned_vs_tuning_algo_path, index=False)
    dataframe_for_csv(inventory_df).to_csv(inventory_path, index=False)
    dataframe_for_csv(coverage_summary).to_csv(coverage_summary_path, index=False)

    add_artifact(artifacts, "results", "table", from_tuned_raw_path, "Raw train-from-tuned reward traces loaded from progress.csv files.")
    add_artifact(artifacts, "results", "table", from_tuned_run_summary_path, "Per-run train-from-tuned summary metrics including best, final, and AUC-style reward.")
    add_artifact(artifacts, "results", "table", from_tuned_set_summary_path, "Train-from-tuned summary aggregated by algorithm and environment set.")
    add_artifact(artifacts, "results", "table", from_tuned_algo_summary_path, "Train-from-tuned summary aggregated by algorithm.")
    add_artifact(artifacts, "results", "table", baseline_run_summary_path, "Per-run baseline default-training summary used for matched comparison.")
    add_artifact(artifacts, "results", "table", baseline_set_summary_path, "Baseline default-training summary aggregated by algorithm and set.")
    add_artifact(artifacts, "results", "table", baseline_algo_summary_path, "Baseline default-training summary aggregated by algorithm.")
    add_artifact(artifacts, "results", "table", provenance_path, "Run-level mapping from each train-from-tuned run to its source tuning run and tuned hyperparameters.")
    add_artifact(artifacts, "results", "table", provenance_algo_summary_path, "Algorithm-level summary of train-from-tuned provenance coverage and tuned-parameter counts.")
    add_artifact(artifacts, "results", "table", training_delta_run_path, "Matched run-level comparison between train-from-tuned and default training.")
    add_artifact(artifacts, "results", "table", training_delta_set_path, "Algorithm/set level delta summary for train-from-tuned versus default training.")
    add_artifact(artifacts, "results", "table", training_delta_algo_path, "Algorithm-level delta summary for train-from-tuned versus default training.")
    add_artifact(artifacts, "results", "table", tuning_run_summary_path, "Tuning summary used as contextual provenance for the source runs.")
    add_artifact(artifacts, "results", "table", from_tuned_vs_tuning_run_path, "Run-level comparison between train-from-tuned results and the best objective of the linked tuning run.")
    add_artifact(artifacts, "results", "table", from_tuned_vs_tuning_algo_path, "Algorithm-level summary comparing train-from-tuned peaks against linked tuning best values.")
    add_artifact(artifacts, "results", "table", inventory_path, "Expected-run inventory derived from the parsed train-from-tuned Slurm scripts.")
    add_artifact(artifacts, "results", "table", coverage_summary_path, "Coverage summary showing discovered versus expected train-from-tuned runs by script and algorithm.")

    if not coverage_summary.empty:
        coverage_plot_path = plots_dir / "analysis_from_tuned_coverage_heatmap.png"
        plot_coverage_heatmap(coverage_summary, coverage_plot_path)
        add_artifact(artifacts, "plots", "figure", coverage_plot_path, "Coverage heatmap annotated as found/expected for each train-from-tuned Slurm script and algorithm.")

    if not from_tuned_raw_df.empty:
        common_steps = np.linspace(float(from_tuned_raw_df["step"].min()), float(from_tuned_raw_df["step"].max()), args.points)
        from_tuned_curves = build_algorithm_curves(from_tuned_raw_df, "step", "reward_scaled", common_steps)
        learning_curve_path = plots_dir / "analysis_from_tuned_learning_curves.png"
        plot_mean_std_curves(
            curves=from_tuned_curves,
            common_x=common_steps,
            xlabel="Step",
            ylabel=r"Reward (x$10^6$)",
            title="Train-from-tuned learning curves",
            path=learning_curve_path,
        )
        add_artifact(artifacts, "plots", "figure", learning_curve_path, "Mean train-from-tuned learning curves with one-standard-deviation shading across sets.")

        final_reward_boxplot_path = plots_dir / "analysis_from_tuned_final_reward_boxplot.png"
        plot_boxplot(
            run_summary=from_tuned_run_summary,
            value_col="final_reward",
            ylabel=r"Final reward (x$10^6$)",
            title="Distribution of final logged rewards for train-from-tuned runs",
            path=final_reward_boxplot_path,
        )
        add_artifact(artifacts, "plots", "figure", final_reward_boxplot_path, "Boxplots of each run's final logged reward for the train-from-tuned experiments.")

        best_reward_heatmap_path = plots_dir / "analysis_from_tuned_best_reward_heatmap.png"
        plot_reward_heatmap(
            summary_df=from_tuned_set_summary,
            value_col="mean_best_reward",
            title="Train-from-tuned best reward by algorithm and set",
            path=best_reward_heatmap_path,
        )
        add_artifact(artifacts, "plots", "figure", best_reward_heatmap_path, "Heatmap of train-from-tuned mean best reward by algorithm and environment set.")

    if not training_delta_algo_summary.empty:
        default_vs_from_tuned_plot_df = training_delta_algo_summary[[
            "algorithm",
            "mean_best_reward_default",
            "mean_best_reward_from_tuned",
        ]].rename(
            columns={
                "mean_best_reward_default": "mean_best_reward_training",
                "mean_best_reward_from_tuned": "mean_best_reward_tuning",
            }
        )
        comparison_plot_path = plots_dir / "analysis_from_tuned_vs_default_comparison.png"
        plot_comparison_bars(default_vs_from_tuned_plot_df, comparison_plot_path)
        add_artifact(artifacts, "plots", "figure", comparison_plot_path, "Grouped bar chart comparing algorithm-level mean best rewards from default training and train-from-tuned runs.")

        delta_heatmap_path = plots_dir / "analysis_from_tuned_vs_default_best_delta_heatmap.png"
        plot_delta_heatmap(
            training_delta_set_summary,
            value_col="mean_best_reward_delta",
            title="Train-from-tuned minus default mean best reward by algorithm and set",
            path=delta_heatmap_path,
        )
        add_artifact(artifacts, "plots", "figure", delta_heatmap_path, "Heatmap of the mean best-reward delta between train-from-tuned and default training by algorithm and set.")

        delta_boxplot_path = plots_dir / "analysis_from_tuned_vs_default_best_delta_boxplot.png"
        plot_delta_boxplot(
            training_delta_run_df,
            value_col="best_reward_delta",
            ylabel=r"Best reward delta (x$10^6$)",
            title="Distribution of train-from-tuned minus default best reward",
            path=delta_boxplot_path,
        )
        add_artifact(artifacts, "plots", "figure", delta_boxplot_path, "Boxplots of run-level best-reward deltas between train-from-tuned and default training.")

    if not from_tuned_vs_tuning_run_df.empty:
        tuning_delta_boxplot_path = plots_dir / "analysis_from_tuned_vs_tuning_best_delta_boxplot.png"
        plot_delta_boxplot(
            from_tuned_vs_tuning_run_df,
            value_col="from_tuned_minus_tuning_best",
            ylabel=r"Best reward delta (x$10^6$)",
            title="Distribution of train-from-tuned peak minus source tuning best",
            path=tuning_delta_boxplot_path,
        )
        add_artifact(artifacts, "plots", "figure", tuning_delta_boxplot_path, "Boxplots of run-level deltas between train-from-tuned best reward and the linked tuning best objective.")

    add_artifact(artifacts, "results", "report", report_path, "Narrative markdown report explaining the train-from-tuned tables and figures.")
    add_artifact(artifacts, "results", "table", manifest_path, "Machine-readable manifest of every table, figure, and report created by analysis_train_from_tuned.py.")

    artifacts_df = pd.DataFrame(artifacts)
    build_report(
        report_path=report_path,
        artifacts_df=artifacts_df,
        warnings=warnings,
        from_tuned_specs=from_tuned_specs,
        coverage_summary=coverage_summary,
        from_tuned_roots=from_tuned_roots,
        from_tuned_algo_summary=from_tuned_algo_summary,
        baseline_algo_summary=baseline_algo_summary,
        provenance_algo_summary=provenance_algo_summary,
        delta_algo_summary=training_delta_algo_summary,
        tuning_link_algo_summary=from_tuned_vs_tuning_algo_summary,
    )

    artifacts_df = pd.DataFrame(artifacts)
    artifacts_df.to_csv(manifest_path, index=False)

    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Train-from-tuned log roots used: {', '.join(str(path) for path in from_tuned_roots) if from_tuned_roots else '-'}")
    print(f"Baseline log roots used: {', '.join(str(path) for path in baseline_roots) if baseline_roots else '-'}")
    print(f"Slurm out directory: {slurm_out_dir}")
    print(f"Parsed train-from-tuned Slurm scripts: {len(from_tuned_specs)}")
    print(f"Train-from-tuned runs summarized: {0 if from_tuned_run_summary.empty else from_tuned_run_summary.shape[0]}")
    print(f"Matched default-vs-from-tuned comparisons: {0 if training_delta_run_df.empty else training_delta_run_df.shape[0]}")
    print(f"Matched from-tuned-vs-tuning comparisons: {0 if from_tuned_vs_tuning_run_df.empty else from_tuned_vs_tuning_run_df.shape[0]}")
    print(f"Saved report to: {report_path}")
    print(f"Saved artifact manifest to: {manifest_path}")


# EDITED:
if __name__ == "__main__":
    main()
