from __future__ import annotations

# EDITED:
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# EDITED:
import numpy as np
import pandas as pd

# EDITED:
from plot_results_rewritten import (
    ALGO_SET_RE,
    REWARD_CANDIDATES,
    STEP_CANDIDATES,
    build_algorithm_curves,
    find_project_root,
    first_matching_column,
    ordered_algorithms,
    plot_results,
    resolve_output_root,
)


# EDITED:
TRANSFER_RUN_RE = re.compile(
    r"^(?P<algorithm>[A-Za-z0-9]+)_from_set-?(?P<source_set>\d+)_to_set-?(?P<target_set>\d+)"
    r"_seed-?(?P<seed>-?\d+)_(?P<exp_name>[^_]+)_(?P<num_robots>\d+)_robots_(?P<device>cpu|cuda)$"
)


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot transfer-learning Stable-Baselines3 training results using "
            "progress.csv files from logs/training_transfer_logs."
        )
    )
    parser.add_argument("--exp-name", type=str, default="transfer", help="Experiment name used in training_<exp_name>_logs.")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to filter the matching runs.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional repo root override.")
    parser.add_argument("--log-root", type=str, default=None, help="Optional explicit log root.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional directory for plots/ and results/.")
    parser.add_argument("--points", type=int, default=200, help="Number of interpolation points for the common x-axis.")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    return parser.parse_args()


# EDITED:
def candidate_log_roots(project_root: Path, exp_name: str, explicit_log_root: Optional[str] = None) -> List[Path]:
    if explicit_log_root:
        return [Path(explicit_log_root).expanduser().resolve()]

    return [
        project_root / "logs" / f"training_{exp_name}_logs",
        project_root / f"training_{exp_name}_logs",
    ]


# EDITED:
def classify_progress_path(rel_parts: Tuple[str, ...], exp_name: str, num_robots: int) -> Optional[str]:
    if len(rel_parts) != 4:
        return None
    if rel_parts[1] != "logs" or rel_parts[-1] != "progress.csv":
        return None
    if ALGO_SET_RE.match(rel_parts[2]) is None:
        return None

    run_root = rel_parts[0]
    match = TRANSFER_RUN_RE.match(run_root)
    if match is None:
        return None
    if match.group("exp_name") != exp_name:
        return None
    if int(match.group("num_robots")) != num_robots:
        return None
    return "transfer_run_name"


# EDITED:
def discover_progress_files(
    project_root: Path,
    exp_name: str,
    num_robots: int,
    explicit_log_root: Optional[str] = None,
) -> Tuple[List[Tuple[Path, Path]], Optional[str]]:
    discovered: List[Tuple[Path, Path]] = []

    for base in candidate_log_roots(project_root, exp_name, explicit_log_root):
        if not base.exists():
            continue
        for progress_csv in base.rglob("progress.csv"):
            if "__old_" in progress_csv.parts:
                continue
            rel_parts = progress_csv.relative_to(base).parts
            layout = classify_progress_path(rel_parts, exp_name, num_robots)
            if layout is None:
                continue
            discovered.append((base, progress_csv))

    items = sorted(set(discovered), key=lambda item: str(item[1]))
    if items:
        return items, "transfer_run_name"
    return [], None


# EDITED:
def parse_progress_metadata(base: Path, progress_csv: Path, layout: str) -> Dict[str, object]:
    rel_parts = progress_csv.relative_to(base).parts
    run_root_name = rel_parts[0]
    algo_dir = rel_parts[2]
    match = ALGO_SET_RE.match(algo_dir)
    if match is None:
        raise ValueError(f"Could not parse algorithm/set from {progress_csv}")

    run_match = TRANSFER_RUN_RE.match(run_root_name)
    if run_match is None:
        raise ValueError(f"Could not parse transfer run name from {run_root_name}")

    target_set = int(run_match.group("target_set"))
    logger_set = int(match.group("set"))
    if logger_set != target_set:
        raise ValueError(
            f"Logger directory set ({logger_set}) does not match run-name target set ({target_set}) for {progress_csv}"
        )

    return {
        "algorithm": match.group("algorithm"),
        "set": target_set,
        "target_set": target_set,
        "source_set": int(run_match.group("source_set")),
        "run_name": run_root_name,
        "run_id": str(progress_csv.parent),
        "seed": int(run_match.group("seed")),
        "device": run_match.group("device"),
        "layout": layout,
        "progress_csv": str(progress_csv),
    }


# EDITED:
def load_progress_rows(progress_csv: Path, metadata: Dict[str, object]) -> pd.DataFrame:
    frame = pd.read_csv(progress_csv)
    if frame.empty:
        return pd.DataFrame()

    step_col = first_matching_column(frame, STEP_CANDIDATES)
    reward_col = first_matching_column(frame, REWARD_CANDIDATES)
    if step_col is None or reward_col is None:
        return pd.DataFrame()

    out = frame[[step_col, reward_col]].copy()
    out.columns = ["step", "reward"]
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["reward"] = pd.to_numeric(out["reward"], errors="coerce")
    out = out.dropna(subset=["step", "reward"]).copy()
    if out.empty:
        return out

    out["step"] = out["step"].astype(int)
    out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")

    for key, value in metadata.items():
        out[key] = value

    cols = [
        "algorithm",
        "set",
        "target_set",
        "source_set",
        "run_name",
        "run_id",
        "seed",
        "device",
        "layout",
        "step",
        "reward",
        "progress_csv",
    ]
    return out[cols]


# EDITED:
def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("algorithm")
    summary = grouped["reward"].agg(["mean", "max", "std", "min"]).reset_index()
    summary = summary.rename(
        columns={
            "algorithm": "Algorithm",
            "mean": "Mean",
            "max": "Max",
            "std": "SD",
            "min": "Min",
        }
    )
    summary["Range"] = summary["Max"] - summary["Min"]
    summary["Runs"] = grouped["run_id"].nunique().to_numpy()
    summary["SourceSets"] = grouped["source_set"].nunique().to_numpy()
    summary["TargetSets"] = grouped["target_set"].nunique().to_numpy()
    summary["Points"] = grouped.size().to_numpy()

    for col in ["Mean", "Max", "SD", "Range", "Min"]:
        summary[col] = summary[col] / 1_000_000.0

    ordered = ordered_algorithms(summary["Algorithm"].tolist())
    return summary.set_index("Algorithm").loc[ordered].reset_index()


# EDITED:
def make_target_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["algorithm", "target_set"])
        .agg(
            mean_reward=("reward", "mean"),
            max_reward=("reward", "max"),
            std_reward=("reward", "std"),
            min_reward=("reward", "min"),
            runs=("run_id", "nunique"),
            points=("reward", "size"),
            source_set=("source_set", "min"),
        )
        .reset_index()
        .sort_values(["algorithm", "target_set"])
        .reset_index(drop=True)
    )

    grouped["reward_range"] = grouped["max_reward"] - grouped["min_reward"]
    for col in ["mean_reward", "max_reward", "std_reward", "min_reward", "reward_range"]:
        grouped[col] = grouped[col] / 1_000_000.0
    return grouped


# EDITED:
def main() -> None:
    args = parse_args()

    project_root = find_project_root(args.project_root)
    output_root = resolve_output_root(project_root, args.output_root)

    results_dir = output_root / "results"
    plots_dir = output_root / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    discovered, layout = discover_progress_files(
        project_root=project_root,
        exp_name=args.exp_name,
        num_robots=args.num_robots,
        explicit_log_root=args.log_root,
    )
    if not discovered:
        searched = candidate_log_roots(project_root, args.exp_name, args.log_root)
        searched_str = ", ".join(str(path) for path in searched)
        raise SystemExit(
            "No matching transfer progress.csv files were found. "
            f"Searched under: {searched_str}"
        )

    frames: List[pd.DataFrame] = []
    for base, progress_csv in discovered:
        metadata = parse_progress_metadata(base, progress_csv, layout or "unknown")
        frame = load_progress_rows(progress_csv, metadata)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit(
            "Log files were found, but none contained a usable reward column. "
            f"Expected one of {list(REWARD_CANDIDATES)} and one of {list(STEP_CANDIDATES)}."
        )

    df = pd.concat(frames, ignore_index=True)
    df["reward_scaled"] = (df["reward"] / 1_000_000.0).clip(lower=-2)

    min_step = float(df["step"].min())
    max_step = float(df["step"].max())
    common_steps = np.linspace(min_step, max_step, args.points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = results_dir / f"{args.exp_name}_raw_for_{args.num_robots}_robots_{timestamp}.csv"
    summary_csv_path = results_dir / f"{args.exp_name}_results_for_{args.num_robots}_robots.csv"
    target_summary_csv_path = results_dir / f"{args.exp_name}_results_by_target_for_{args.num_robots}_robots.csv"
    plot_path = plots_dir / f"{args.exp_name}_graph_for_{args.num_robots}_robots.png"

    df.to_csv(raw_csv_path, index=False)
    make_summary_table(df).to_csv(summary_csv_path, index=False)
    make_target_summary_table(df).to_csv(target_summary_csv_path, index=False)
    plot_results(df, common_steps, plot_path, args.show)

    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Log layout used: {layout}")
    print(f"Progress files used: {len(discovered)}")
    print(f"Source sets used: {sorted(df['source_set'].dropna().unique().tolist())}")
    print(f"Target sets used: {sorted(df['target_set'].dropna().unique().tolist())}")
    print(f"Saved raw data to: {raw_csv_path}")
    print(f"Saved summary to: {summary_csv_path}")
    print(f"Saved target summary to: {target_summary_csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
