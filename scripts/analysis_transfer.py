# EDITED:
"""Create tables, plots, and a written report for transfer-learning runs.

The script focuses on:
- run coverage versus the expected transfer runs implied by the transfer Slurm scripts
- transfer-learning learning curves and per-run summaries
- comparison against the default-training baseline on the same target sets
- a concise markdown report that highlights where transfer appears to help or hurt
"""

from __future__ import annotations

# EDITED:
import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# EDITED:
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# EDITED:
from analysis import (
    REWARD_CANDIDATES,
    STEP_CANDIDATES,
    add_artifact,
    build_training_algorithm_summary,
    build_training_run_summary,
    build_training_set_summary,
    coverage_summary_from_inventory,
    dataframe_for_csv,
    discover_training_progress,
    find_project_root,
    format_number,
    format_reward_millions,
    markdown_table,
    ordered_algorithms,
    parse_python_entry,
    parse_run_name_template,
    parse_sbatch_directive,
    parse_scalar_assignment,
    parse_shell_list,
    plot_boxplot,
    plot_coverage_heatmap,
    plot_mean_std_curves,
    plot_reward_heatmap,
    render_shell_template,
    resolve_output_root,
    safe_float,
    top_algorithm,
)


# EDITED:
ALGO_SET_RE = re.compile(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)$")
# EDITED:
TRANSFER_RUN_RE = re.compile(
    r"^(?P<algorithm>[A-Za-z0-9]+)_from_set-?(?P<source_set>\d+)_to_set-?(?P<target_set>\d+)"
    r"_seed-?(?P<seed>-?\d+)_(?P<exp_name>[^_]+)_(?P<num_robots>\d+)_robots_(?P<device>cpu|cuda)$"
)


# EDITED:
@dataclass
class TransferSlurmSpec:
    path: Path
    job_name: str
    algorithms: List[str]
    target_sets: List[int]
    seeds: List[int]
    load_set: int
    device: str
    steps: int
    num_robots: int
    run_name_template: str
    entry_script: str
    exp_name: Optional[str]


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze transfer-learning results and compare them to the default-training "
            "baseline on the same target sets."
        )
    )
    parser.add_argument("--project-root", type=str, default=None, help="Optional repository root override.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional override for the plotting output directory.")
    parser.add_argument("--transfer-exp-name", type=str, default="transfer", help="Transfer experiment name used in logs/training_<exp_name>_logs.")
    parser.add_argument("--default-exp-name", type=str, default="default", help="Default-training experiment name used in logs/training_<exp_name>_logs.")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to match in run names.")
    parser.add_argument("--transfer-log-root", type=str, default=None, help="Optional explicit transfer log root.")
    parser.add_argument("--default-log-root", type=str, default=None, help="Optional explicit default-training log root.")
    parser.add_argument(
        "--slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/transfer_all_3_robots.sh",
            "slurm_scripts/crossq_transfer_3_robots.sh",
        ],
        help="Transfer Slurm scripts whose expected runs should be summarized.",
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
def first_matching_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


# EDITED:
def parse_transfer_run_bits(run_name: str) -> Optional[Dict[str, object]]:
    match = TRANSFER_RUN_RE.match(run_name)
    if match is None:
        return None
    return {
        "algorithm": match.group("algorithm"),
        "source_set": int(match.group("source_set")),
        "target_set": int(match.group("target_set")),
        "set": int(match.group("target_set")),
        "seed": int(match.group("seed")),
        "exp_name": match.group("exp_name"),
        "num_robots": int(match.group("num_robots")),
        "device": match.group("device"),
    }


# EDITED:
def parse_transfer_slurm_spec(path: Path) -> TransferSlurmSpec:
    text = read_text(path)
    if not text:
        raise FileNotFoundError(path)

    job_name = parse_sbatch_directive(text, "job-name") or path.stem
    algorithms = parse_shell_list(text, "algorithms")
    target_sets = [int(value) for value in parse_shell_list(text, "sets")]
    seeds_raw = parse_shell_list(text, "seed") or parse_shell_list(text, "seeds")
    seeds = [int(value) for value in seeds_raw]
    load_set = int(parse_scalar_assignment(text, "load_set") or 0)
    device = "cpu"
    run_name_template = parse_run_name_template(text) or ""
    entry_script = parse_python_entry(text) or ""
    steps = int(parse_scalar_assignment(text, "steps") or 0)
    num_robots = int(parse_scalar_assignment(text, "num_robots") or 0)

    exp_name = None
    if run_name_template and algorithms and target_sets and seeds:
        rendered = render_shell_template(
            run_name_template,
            {
                "algorithm": algorithms[0],
                "load_set": load_set,
                "set": target_sets[0],
                "seed": seeds[0],
            },
        )
        bits = parse_transfer_run_bits(rendered)
        if bits is not None:
            device = str(bits["device"])
            exp_name = str(bits["exp_name"])
            num_robots = int(bits["num_robots"])

    missing = []
    if not algorithms:
        missing.append("algorithms")
    if not target_sets:
        missing.append("sets")
    if not seeds:
        missing.append("seeds")
    if not load_set:
        missing.append("load_set")
    if not run_name_template:
        missing.append("run_name_template")
    if missing:
        raise ValueError(f"Unable to parse {path}: missing {', '.join(missing)}")

    return TransferSlurmSpec(
        path=path,
        job_name=job_name,
        algorithms=algorithms,
        target_sets=target_sets,
        seeds=seeds,
        load_set=load_set,
        device=device,
        steps=steps,
        num_robots=num_robots,
        run_name_template=run_name_template,
        entry_script=entry_script,
        exp_name=exp_name,
    )


# EDITED:
def build_expected_inventory(specs: Sequence[TransferSlurmSpec]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in specs:
        for algorithm in spec.algorithms:
            for target_set in spec.target_sets:
                for seed in spec.seeds:
                    run_name = render_shell_template(
                        spec.run_name_template,
                        {
                            "algorithm": algorithm,
                            "load_set": spec.load_set,
                            "set": target_set,
                            "seed": seed,
                        },
                    )
                    bits = parse_transfer_run_bits(run_name) or {}
                    rows.append(
                        {
                            "script_label": spec.path.stem,
                            "script_path": str(spec.path),
                            "job_name": spec.job_name,
                            "kind": "transfer",
                            "entry_script": spec.entry_script,
                            "exp_name": bits.get("exp_name", spec.exp_name),
                            "algorithm": algorithm,
                            "source_set": int(bits.get("source_set", spec.load_set)),
                            "target_set": int(bits.get("target_set", target_set)),
                            "set": int(bits.get("set", target_set)),
                            "seed": int(bits.get("seed", seed)),
                            "device": bits.get("device", spec.device),
                            "steps": spec.steps,
                            "num_robots": int(bits.get("num_robots", spec.num_robots)),
                            "run_name": run_name,
                        }
                    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["algorithm", "source_set", "target_set", "seed"]).reset_index(drop=True)


# EDITED:
def candidate_transfer_log_roots(project_root: Path, exp_name: str, explicit_log_root: Optional[str] = None) -> List[Path]:
    if explicit_log_root:
        return [Path(explicit_log_root).expanduser().resolve()]
    return [
        project_root / "logs" / f"training_{exp_name}_logs",
        project_root / f"training_{exp_name}_logs",
    ]


# EDITED:
def discover_transfer_progress(
    project_root: Path,
    exp_name: str,
    num_robots: int,
    explicit_log_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Path]]:
    frames: List[pd.DataFrame] = []
    used_roots: List[Path] = []

    for base in candidate_transfer_log_roots(project_root, exp_name, explicit_log_root):
        if not base.exists():
            continue
        for progress_csv in base.rglob("progress.csv"):
            if "__old_" in progress_csv.parts:
                continue

            try:
                rel_parts = progress_csv.relative_to(base).parts
            except ValueError:
                continue

            if len(rel_parts) != 4 or rel_parts[1] != "logs" or rel_parts[-1] != "progress.csv":
                continue

            algo_match = ALGO_SET_RE.match(rel_parts[2])
            if algo_match is None:
                continue

            run_root = rel_parts[0]
            bits = parse_transfer_run_bits(run_root)
            if bits is None:
                continue
            if bits["exp_name"] != exp_name or bits["num_robots"] != num_robots:
                continue
            if int(bits["target_set"]) != int(algo_match.group("set")):
                continue

            frame = pd.read_csv(progress_csv)
            if frame.empty:
                continue

            step_col = first_matching_column(frame, STEP_CANDIDATES)
            reward_col = first_matching_column(frame, REWARD_CANDIDATES)
            if step_col is None or reward_col is None:
                continue

            out = frame[[step_col, reward_col]].copy()
            out.columns = ["step", "reward"]
            out["step"] = pd.to_numeric(out["step"], errors="coerce")
            out["reward"] = pd.to_numeric(out["reward"], errors="coerce")
            out = out.dropna(subset=["step", "reward"]).copy()
            if out.empty:
                continue

            out["step"] = out["step"].astype(int)
            out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")
            out["algorithm"] = algo_match.group("algorithm")
            out["source_set"] = int(bits["source_set"])
            out["target_set"] = int(bits["target_set"])
            out["set"] = int(bits["target_set"])
            out["run_name"] = run_root
            out["run_id"] = str(progress_csv.parent)
            out["seed"] = int(bits["seed"])
            out["device"] = bits["device"]
            out["progress_csv"] = str(progress_csv)
            out["exp_name"] = exp_name
            frames.append(out)
            used_roots.append(base)

    if not frames:
        return pd.DataFrame(), sorted(set(used_roots))

    df = pd.concat(frames, ignore_index=True)
    cols = [
        "algorithm",
        "source_set",
        "target_set",
        "set",
        "run_name",
        "run_id",
        "seed",
        "device",
        "step",
        "reward",
        "progress_csv",
        "exp_name",
    ]
    return df[cols], sorted(set(used_roots))


# EDITED:
def attach_transfer_metadata(summary_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or raw_df.empty:
        return summary_df

    meta = (
        raw_df[["run_name", "source_set", "target_set"]]
        .drop_duplicates(subset=["run_name"])
        .reset_index(drop=True)
    )
    return summary_df.merge(meta, on="run_name", how="left")


# EDITED:
def build_transfer_vs_default_algorithm_comparison(
    transfer_algo_summary: pd.DataFrame,
    default_algo_summary: pd.DataFrame,
) -> pd.DataFrame:
    if transfer_algo_summary.empty or default_algo_summary.empty:
        return pd.DataFrame()

    merged = transfer_algo_summary.merge(
        default_algo_summary,
        on="algorithm",
        how="inner",
        suffixes=("_transfer", "_default"),
    )
    if merged.empty:
        return merged

    merged["transfer_minus_default_mean_final"] = merged["mean_final_reward_transfer"] - merged["mean_final_reward_default"]
    merged["transfer_minus_default_mean_best"] = merged["mean_best_reward_transfer"] - merged["mean_best_reward_default"]
    merged["transfer_minus_default_best_of_best"] = merged["best_of_best_reward_transfer"] - merged["best_of_best_reward_default"]
    ordered = ordered_algorithms(merged["algorithm"].tolist())
    return merged.set_index("algorithm").loc[ordered].reset_index()


# EDITED:
def build_transfer_vs_default_set_comparison(
    transfer_set_summary: pd.DataFrame,
    default_set_summary: pd.DataFrame,
) -> pd.DataFrame:
    if transfer_set_summary.empty or default_set_summary.empty:
        return pd.DataFrame()

    merged = transfer_set_summary.merge(
        default_set_summary,
        on=["algorithm", "set"],
        how="inner",
        suffixes=("_transfer", "_default"),
    )
    if merged.empty:
        return merged

    merged["transfer_minus_default_mean_final"] = merged["mean_final_reward_transfer"] - merged["mean_final_reward_default"]
    merged["transfer_minus_default_mean_best"] = merged["mean_best_reward_transfer"] - merged["mean_best_reward_default"]
    merged["transfer_minus_default_mean_auc"] = merged["mean_auc_reward_transfer"] - merged["mean_auc_reward_default"]
    return merged.sort_values(["algorithm", "set"]).reset_index(drop=True)


# EDITED:
def plot_algorithm_comparison_bars(comparison_df: pd.DataFrame, path: Path) -> None:
    if comparison_df.empty:
        return

    labels = ordered_algorithms(comparison_df["algorithm"].tolist())
    df = comparison_df.set_index("algorithm").loc[labels].reset_index()
    x = np.arange(df.shape[0])
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(x - width / 2, df["mean_best_reward_default"] / 1_000_000.0, width, label="Default target mean best")
    ax.bar(x + width / 2, df["mean_best_reward_transfer"] / 1_000_000.0, width, label="Transfer mean best")
    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"].tolist())
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(r"Reward (x$10^6$)")
    ax.set_title("Transfer versus default-target mean best reward")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


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
    ax.set_xlabel("Target set")
    ax.set_ylabel("Algorithm")
    ax.set_title(title)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display.iat[i, j]
            text = "-" if pd.isna(value) else f"{value:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label=r"Reward delta (x$10^6$)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def build_report(
    report_path: Path,
    artifacts_df: pd.DataFrame,
    warnings: List[str],
    slurm_specs: Sequence[TransferSlurmSpec],
    inventory_df: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    transfer_roots: List[Path],
    transfer_run_summary: pd.DataFrame,
    transfer_algo_summary: pd.DataFrame,
    default_target_algo_summary: pd.DataFrame,
    source_default_algo_summary: pd.DataFrame,
    comparison_algo_df: pd.DataFrame,
    comparison_set_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Transfer-learning analysis report")
    lines.append("")
    lines.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    lines.append("")
    lines.append("This report summarizes the transfer-learning experiments that load a model trained on a source set and continue training on a different target set. All detailed tables are written to `plotting/results`, and all figures are written to `plotting/plots`.")
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Parsed transfer Slurm scripts")
    lines.append("")
    if slurm_specs:
        rows = []
        for spec in slurm_specs:
            rows.append(
                {
                    "script": spec.path.name,
                    "algorithms": len(spec.algorithms),
                    "target_sets": len(spec.target_sets),
                    "seeds": len(spec.seeds),
                    "load_set": spec.load_set,
                    "steps": spec.steps,
                    "exp_name": spec.exp_name or "-",
                }
            )
        lines.append(markdown_table(pd.DataFrame(rows), max_rows=20))
    else:
        lines.append("_No transfer Slurm scripts were parsed._")
    lines.append("")

    lines.append("## Run coverage")
    lines.append("")
    if inventory_df.empty:
        lines.append("_No expected transfer-run inventory could be built from the supplied Slurm scripts._")
    else:
        expected_total = int(inventory_df.shape[0])
        found_total = int(inventory_df["expected_output_found"].sum())
        missing_total = int((~inventory_df["expected_output_found"]).sum())
        lines.append(f"The inventory contains **{expected_total}** expected transfer runs. Matching outputs were found for **{found_total}** of them, leaving **{missing_total}** runs without a matching result artifact.")
        lines.append("")
        if not coverage_summary.empty:
            lines.append(markdown_table(coverage_summary, formatters={"coverage_ratio": lambda v: format_number(v, 2)}, max_rows=30))
    lines.append("")

    lines.append("## Transfer-learning results")
    lines.append("")
    if transfer_run_summary.empty:
        lines.append("_No transfer-learning progress logs were discovered._")
    else:
        lines.append(f"Usable transfer-learning curves were discovered under: {', '.join(str(path) for path in transfer_roots) if transfer_roots else '-'}.")
        lines.append("")
        best_peak_algo, best_peak_value = top_algorithm(transfer_algo_summary, "mean_best_reward", higher_is_better=True)
        best_final_algo, best_final_value = top_algorithm(transfer_algo_summary, "mean_final_reward", higher_is_better=True)
        stable_algo, stable_value = top_algorithm(transfer_algo_summary, "std_best_reward", higher_is_better=False)
        if best_peak_algo is not None:
            lines.append(f"Highest mean peak transfer reward: **{best_peak_algo}** at **{format_reward_millions(best_peak_value)} x10^6** reward units.")
        if best_final_algo is not None:
            lines.append(f"Highest mean final logged transfer reward: **{best_final_algo}** at **{format_reward_millions(best_final_value)} x10^6**.")
        if stable_algo is not None:
            lines.append(f"Lowest cross-run variability in transfer peak reward: **{stable_algo}** with a standard deviation of **{format_reward_millions(stable_value)} x10^6**.")
        lines.append("")
        lines.append(markdown_table(
            transfer_algo_summary[[
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

    lines.append("## Transfer versus default-target baseline")
    lines.append("")
    if comparison_algo_df.empty:
        lines.append("_A transfer-versus-default comparison could not be produced because either transfer logs or default target-set logs were missing._")
    else:
        best_gain_algo, best_gain_value = top_algorithm(comparison_algo_df, "transfer_minus_default_mean_best", higher_is_better=True)
        worst_gain_algo, worst_gain_value = top_algorithm(comparison_algo_df, "transfer_minus_default_mean_best", higher_is_better=False)
        if best_gain_algo is not None:
            lines.append(f"Largest algorithm-level gain from transfer in mean best reward: **{best_gain_algo}** at **{format_reward_millions(best_gain_value)} x10^6** compared with training directly on the same target sets.")
        if worst_gain_algo is not None:
            lines.append(f"Largest algorithm-level drop from transfer in mean best reward: **{worst_gain_algo}** at **{format_reward_millions(worst_gain_value)} x10^6** relative to the default target-set baseline.")
        lines.append("")
        lines.append(markdown_table(
            comparison_algo_df[[
                "algorithm",
                "mean_best_reward_default",
                "mean_best_reward_transfer",
                "transfer_minus_default_mean_best",
                "mean_final_reward_default",
                "mean_final_reward_transfer",
                "transfer_minus_default_mean_final",
            ]],
            formatters={
                "mean_best_reward_default": format_reward_millions,
                "mean_best_reward_transfer": format_reward_millions,
                "transfer_minus_default_mean_best": format_reward_millions,
                "mean_final_reward_default": format_reward_millions,
                "mean_final_reward_transfer": format_reward_millions,
                "transfer_minus_default_mean_final": format_reward_millions,
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Source-set context")
    lines.append("")
    if source_default_algo_summary.empty:
        lines.append("_No default-training logs for the source set were discovered._")
    else:
        lines.append("The table below summarizes how each algorithm performed on the source set that supplied the pretrained weights. This is useful context, but it should not be over-interpreted as a direct success criterion for transfer, because the target sets are different tasks.")
        lines.append("")
        lines.append(markdown_table(
            source_default_algo_summary[[
                "algorithm",
                "runs",
                "mean_final_reward",
                "mean_best_reward",
                "best_of_best_reward",
            ]],
            formatters={
                "mean_final_reward": format_reward_millions,
                "mean_best_reward": format_reward_millions,
                "best_of_best_reward": format_reward_millions,
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    if artifacts_df.empty:
        lines.append("_No artifacts were registered._")
    else:
        lines.append(markdown_table(artifacts_df[["category", "kind", "filename", "description"]], max_rows=200))

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

    warnings: List[str] = []

    slurm_specs: List[TransferSlurmSpec] = []
    for script_relpath in args.slurm_scripts:
        script_path = (project_root / script_relpath).resolve()
        slurm_specs.append(parse_transfer_slurm_spec(script_path))

    inventory_df = build_expected_inventory(slurm_specs)

    transfer_raw_df, transfer_roots = discover_transfer_progress(
        project_root=project_root,
        exp_name=args.transfer_exp_name,
        num_robots=args.num_robots,
        explicit_log_root=args.transfer_log_root,
    )
    if not transfer_raw_df.empty:
        transfer_raw_df["reward_scaled"] = (transfer_raw_df["reward"] / 1_000_000.0).clip(lower=-2)

    transfer_run_summary = build_training_run_summary(transfer_raw_df) if not transfer_raw_df.empty else pd.DataFrame()
    transfer_run_summary = attach_transfer_metadata(transfer_run_summary, transfer_raw_df)
    transfer_set_summary = build_training_set_summary(transfer_run_summary) if not transfer_run_summary.empty else pd.DataFrame()
    transfer_algo_summary = build_training_algorithm_summary(transfer_run_summary) if not transfer_run_summary.empty else pd.DataFrame()

    default_raw_df, default_roots = discover_training_progress(
        project_root=project_root,
        exp_name=args.default_exp_name,
        num_robots=args.num_robots,
        explicit_log_root=args.default_log_root,
    )
    if not default_raw_df.empty:
        default_raw_df["reward_scaled"] = (default_raw_df["reward"] / 1_000_000.0).clip(lower=-2)

    target_sets = sorted(set(transfer_run_summary["set"].tolist())) if not transfer_run_summary.empty else sorted(set(inventory_df["target_set"].tolist())) if not inventory_df.empty else []
    source_sets = sorted(set(transfer_run_summary["source_set"].tolist())) if not transfer_run_summary.empty and "source_set" in transfer_run_summary.columns else sorted(set(inventory_df["source_set"].tolist())) if not inventory_df.empty else []

    if not default_raw_df.empty and target_sets:
        default_target_raw_df = default_raw_df[default_raw_df["set"].isin(target_sets)].copy()
    else:
        default_target_raw_df = pd.DataFrame()

    default_target_run_summary = build_training_run_summary(default_target_raw_df) if not default_target_raw_df.empty else pd.DataFrame()
    default_target_set_summary = build_training_set_summary(default_target_run_summary) if not default_target_run_summary.empty else pd.DataFrame()
    default_target_algo_summary = build_training_algorithm_summary(default_target_run_summary) if not default_target_run_summary.empty else pd.DataFrame()

    if not default_raw_df.empty and source_sets:
        source_default_raw_df = default_raw_df[default_raw_df["set"].isin(source_sets)].copy()
    else:
        source_default_raw_df = pd.DataFrame()

    source_default_run_summary = build_training_run_summary(source_default_raw_df) if not source_default_raw_df.empty else pd.DataFrame()
    source_default_algo_summary = build_training_algorithm_summary(source_default_run_summary) if not source_default_run_summary.empty else pd.DataFrame()

    comparison_algo_df = build_transfer_vs_default_algorithm_comparison(
        transfer_algo_summary,
        default_target_algo_summary,
    )
    comparison_set_df = build_transfer_vs_default_set_comparison(
        transfer_set_summary,
        default_target_set_summary,
    )

    discovered_transfer_names = set(transfer_run_summary["run_name"]) if not transfer_run_summary.empty else set()
    if not inventory_df.empty:
        inventory_df = inventory_df.copy()
        inventory_df["expected_output_found"] = inventory_df["run_name"].isin(discovered_transfer_names)
        inventory_df["coverage_status"] = np.where(inventory_df["expected_output_found"], "FOUND", "MISSING")
    else:
        inventory_df = pd.DataFrame(columns=["expected_output_found"])
    coverage_summary = coverage_summary_from_inventory(inventory_df) if not inventory_df.empty else pd.DataFrame()

    artifacts: List[Dict[str, str]] = []

    transfer_raw_path = results_dir / "analysis_transfer_raw.csv"
    transfer_run_summary_path = results_dir / "analysis_transfer_run_summary.csv"
    transfer_set_summary_path = results_dir / "analysis_transfer_set_summary.csv"
    transfer_algo_summary_path = results_dir / "analysis_transfer_algorithm_summary.csv"
    default_target_algo_summary_path = results_dir / "analysis_transfer_default_target_algorithm_summary.csv"
    source_default_algo_summary_path = results_dir / "analysis_transfer_source_default_algorithm_summary.csv"
    comparison_algo_path = results_dir / "analysis_transfer_vs_default_algorithm_comparison.csv"
    comparison_set_path = results_dir / "analysis_transfer_vs_default_set_comparison.csv"
    inventory_path = results_dir / "analysis_transfer_run_inventory.csv"
    coverage_summary_path = results_dir / "analysis_transfer_run_coverage_summary.csv"
    manifest_path = results_dir / "analysis_transfer_artifact_manifest.csv"
    report_path = results_dir / "analysis_transfer_report.md"

    dataframe_for_csv(transfer_raw_df).to_csv(transfer_raw_path, index=False)
    dataframe_for_csv(transfer_run_summary).to_csv(transfer_run_summary_path, index=False)
    dataframe_for_csv(transfer_set_summary).to_csv(transfer_set_summary_path, index=False)
    dataframe_for_csv(transfer_algo_summary).to_csv(transfer_algo_summary_path, index=False)
    dataframe_for_csv(default_target_algo_summary).to_csv(default_target_algo_summary_path, index=False)
    dataframe_for_csv(source_default_algo_summary).to_csv(source_default_algo_summary_path, index=False)
    dataframe_for_csv(comparison_algo_df).to_csv(comparison_algo_path, index=False)
    dataframe_for_csv(comparison_set_df).to_csv(comparison_set_path, index=False)
    dataframe_for_csv(inventory_df).to_csv(inventory_path, index=False)
    dataframe_for_csv(coverage_summary).to_csv(coverage_summary_path, index=False)

    add_artifact(artifacts, "results", "table", transfer_raw_path, "Raw transfer-learning reward traces loaded from progress.csv files.")
    add_artifact(artifacts, "results", "table", transfer_run_summary_path, "Per-run transfer-learning summary metrics such as final reward, best reward, and AUC-like average reward.")
    add_artifact(artifacts, "results", "table", transfer_set_summary_path, "Transfer-learning summary aggregated by algorithm and target set.")
    add_artifact(artifacts, "results", "table", transfer_algo_summary_path, "Transfer-learning summary aggregated by algorithm.")
    add_artifact(artifacts, "results", "table", default_target_algo_summary_path, "Default-training baseline summary aggregated by algorithm across the transfer target sets.")
    add_artifact(artifacts, "results", "table", source_default_algo_summary_path, "Default-training summary aggregated by algorithm on the source set used for transfer.")
    add_artifact(artifacts, "results", "table", comparison_algo_path, "Algorithm-level comparison between transfer-learning runs and direct default training on the same target sets.")
    add_artifact(artifacts, "results", "table", comparison_set_path, "Per-target-set comparison between transfer-learning runs and direct default training.")
    add_artifact(artifacts, "results", "table", inventory_path, "Expected transfer-run inventory derived from the parsed transfer Slurm scripts.")
    add_artifact(artifacts, "results", "table", coverage_summary_path, "Coverage summary showing discovered versus expected transfer runs by script and algorithm.")

    if not coverage_summary.empty:
        coverage_plot_path = plots_dir / "analysis_transfer_run_coverage_heatmap.png"
        plot_coverage_heatmap(coverage_summary, coverage_plot_path)
        add_artifact(artifacts, "plots", "figure", coverage_plot_path, "Coverage heatmap annotated as found/expected for each transfer Slurm-script and algorithm combination.")

    if not transfer_raw_df.empty:
        common_steps = np.linspace(float(transfer_raw_df["step"].min()), float(transfer_raw_df["step"].max()), args.points)
        transfer_curves = {}
        for algorithm, algo_df in transfer_raw_df.groupby("algorithm"):
            set_curves = []
            for _target_set, set_df in algo_df.groupby("target_set"):
                run_curves = []
                for _run_name, run_df in set_df.groupby("run_name"):
                    sub = run_df[["step", "reward_scaled"]].dropna().copy()
                    if sub.empty:
                        continue
                    x = sub["step"].to_numpy(dtype=float)
                    y = sub["reward_scaled"].to_numpy(dtype=float)
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                    x_unique, unique_idx = np.unique(x, return_index=True)
                    y_unique = y[unique_idx]
                    if x_unique.size == 1:
                        interp = np.full_like(common_steps, y_unique[0], dtype=float)
                    else:
                        interp = np.interp(common_steps, x_unique, y_unique)
                    run_curves.append(interp)
                if run_curves:
                    set_curves.append(np.mean(np.vstack(run_curves), axis=0))
            if set_curves:
                transfer_curves[algorithm] = np.vstack(set_curves)

        transfer_curve_path = plots_dir / "analysis_transfer_learning_curves.png"
        plot_mean_std_curves(
            curves=transfer_curves,
            common_x=common_steps,
            xlabel="Step",
            ylabel=r"Reward (x$10^6$)",
            title="Transfer-learning learning curves",
            path=transfer_curve_path,
        )
        add_artifact(artifacts, "plots", "figure", transfer_curve_path, "Mean transfer-learning learning curves with one-standard-deviation shading across target sets.")

        transfer_boxplot_path = plots_dir / "analysis_transfer_final_reward_boxplot.png"
        plot_boxplot(
            run_summary=transfer_run_summary,
            value_col="final_reward",
            ylabel=r"Final reward (x$10^6$)",
            title="Distribution of final logged rewards for transfer learning",
            path=transfer_boxplot_path,
        )
        add_artifact(artifacts, "plots", "figure", transfer_boxplot_path, "Boxplots of each transfer run's final logged reward.")

        transfer_heatmap_path = plots_dir / "analysis_transfer_best_reward_heatmap.png"
        plot_reward_heatmap(
            summary_df=transfer_set_summary,
            value_col="mean_best_reward",
            title="Transfer-learning best reward by algorithm and target set",
            path=transfer_heatmap_path,
        )
        add_artifact(artifacts, "plots", "figure", transfer_heatmap_path, "Heatmap of transfer-learning mean best reward by algorithm and target set.")

    if not comparison_algo_df.empty:
        comparison_plot_path = plots_dir / "analysis_transfer_vs_default_algorithm_comparison.png"
        plot_algorithm_comparison_bars(comparison_algo_df, comparison_plot_path)
        add_artifact(artifacts, "plots", "figure", comparison_plot_path, "Grouped bar chart comparing transfer-learning and direct default-training mean best reward by algorithm.")

    if not comparison_set_df.empty:
        delta_heatmap_path = plots_dir / "analysis_transfer_gain_heatmap.png"
        plot_delta_heatmap(
            comparison_set_df,
            "transfer_minus_default_mean_best",
            "Transfer gain/loss in mean best reward by algorithm and target set",
            delta_heatmap_path,
        )
        add_artifact(artifacts, "plots", "figure", delta_heatmap_path, "Heatmap of transfer-learning mean-best-reward delta relative to direct default training on each target set.")

    add_artifact(artifacts, "results", "report", report_path, "Narrative markdown report explaining the generated transfer-learning tables and figures.")
    add_artifact(artifacts, "results", "table", manifest_path, "Machine-readable manifest of every table, figure, and report created by analysis_transfer.py.")

    artifacts_df = pd.DataFrame(artifacts)
    build_report(
        report_path=report_path,
        artifacts_df=artifacts_df,
        warnings=warnings,
        slurm_specs=slurm_specs,
        inventory_df=inventory_df,
        coverage_summary=coverage_summary,
        transfer_roots=transfer_roots,
        transfer_run_summary=transfer_run_summary,
        transfer_algo_summary=transfer_algo_summary,
        default_target_algo_summary=default_target_algo_summary,
        source_default_algo_summary=source_default_algo_summary,
        comparison_algo_df=comparison_algo_df,
        comparison_set_df=comparison_set_df,
    )

    artifacts_df = pd.DataFrame(artifacts)
    artifacts_df.to_csv(manifest_path, index=False)

    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Transfer log roots used: {', '.join(str(path) for path in transfer_roots) if transfer_roots else '-'}")
    print(f"Default log roots used: {', '.join(str(path) for path in default_roots) if default_roots else '-'}")
    print(f"Parsed transfer Slurm scripts: {len(slurm_specs)}")
    print(f"Transfer runs summarized: {0 if transfer_run_summary.empty else transfer_run_summary.shape[0]}")
    print(f"Saved report to: {report_path}")
    print(f"Saved artifact manifest to: {manifest_path}")


# EDITED:
if __name__ == "__main__":
    main()
