from __future__ import annotations

# EDITED:
import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# EDITED:
DEFAULT_ALGO_ORDER: Tuple[str, ...] = ("A2C", "ARS", "CrossQ", "PPO", "TQC", "TRPO")
ALGO_SET_RE = re.compile(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)$")
SEED_RE = re.compile(r"_seed(?P<seed>-?\d+)")
DEVICE_RE = re.compile(r"_(?P<device>cpu|cuda)$")
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run_name>\S+)")
TUNING_RUN_RE = re.compile(
    r"Loaded tuned hyperparameters from (?P<tuning_run_name>\S+):\s*(?P<params>\{.*\})"
)
TUNED_PARAMS_RE = re.compile(r"Tuned hyperparameters used for training:\s*(?P<params>\{.*\})")
TUNED_SOURCES_RE = re.compile(r"Tuned hyperparameter sources:\s*(?P<sources>\{.*\})")
STEP_CANDIDATES: Tuple[str, ...] = (
    "time/total_timesteps",
    "total_timesteps",
    "timesteps",
    "step",
)
REWARD_CANDIDATES: Tuple[str, ...] = (
    "rollout/ep_rew_mean",
    "ep_rew_mean",
)


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot and summarize train-from-tuned Stable-Baselines3 training results "
            "using progress.csv files. Outputs are written to plotting/results and "
            "plotting/plots when those folders exist."
        )
    )
    parser.add_argument("--exp-name", type=str, default="from_tuned", help="Experiment name used in training_<exp_name>_logs.")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to filter the matching runs.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional repo root override.")
    parser.add_argument("--log-root", type=str, default=None, help="Optional explicit log root.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional directory for plots/ and results/.")
    parser.add_argument("--slurm-out-dir", type=str, default=None, help="Optional slurm_scripts/slurm_out override for tuned-parameter provenance.")
    parser.add_argument("--points", type=int, default=200, help="Number of interpolation points for the common x-axis.")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    return parser.parse_args()


# EDITED:
def _candidate_roots(start: Path) -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for base in [start, Path.cwd().resolve(), Path(__file__).resolve().parent]:
        for cand in [base, *base.parents]:
            if cand not in seen:
                roots.append(cand)
                seen.add(cand)
    return roots


# EDITED:
def find_project_root(explicit_root: Optional[str] = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    start = Path(__file__).resolve().parent
    markers = ("exp_sets", "src", "train_default.py", "plotting", "logs")
    for cand in _candidate_roots(start):
        if any((cand / marker).exists() for marker in markers):
            return cand
    return start


# EDITED:
def resolve_output_root(project_root: Path, explicit_root: Optional[str] = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "plotting":
        return script_dir

    plotting_dir = project_root / "plotting"
    if plotting_dir.exists():
        return plotting_dir

    return script_dir / "plotting"


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
    if run_root == f"{num_robots}_robots":
        return "bucket"
    if f"_{exp_name}_{num_robots}_robots" in run_root:
        return "run_name"
    return None


# EDITED:
def discover_progress_files(
    project_root: Path,
    exp_name: str,
    num_robots: int,
    explicit_log_root: Optional[str] = None,
) -> Tuple[List[Tuple[Path, Path]], Optional[str]]:
    by_layout: Dict[str, List[Tuple[Path, Path]]] = {"run_name": [], "bucket": []}

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
            by_layout[layout].append((base, progress_csv))

    for layout in ("run_name", "bucket"):
        items = sorted(set(by_layout[layout]), key=lambda item: str(item[1]))
        if items:
            return items, layout

    return [], None


# EDITED:
def parse_progress_metadata(base: Path, progress_csv: Path, layout: str) -> Dict[str, object]:
    rel_parts = progress_csv.relative_to(base).parts
    run_root_name = rel_parts[0]
    algo_dir = rel_parts[2]
    match = ALGO_SET_RE.match(algo_dir)
    if match is None:
        raise ValueError(f"Could not parse algorithm/set from {progress_csv}")

    seed_match = SEED_RE.search(run_root_name)
    device_match = DEVICE_RE.search(run_root_name)

    return {
        "algorithm": match.group("algorithm"),
        "set": int(match.group("set")),
        "run_name": run_root_name,
        "run_id": str(progress_csv.parent),
        "seed": int(seed_match.group("seed")) if seed_match else None,
        "device": device_match.group("device") if device_match else None,
        "layout": layout,
        "progress_csv": str(progress_csv),
    }


# EDITED:
def first_matching_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


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
def interpolate_run(run_df: pd.DataFrame, common_steps: np.ndarray) -> np.ndarray:
    x = run_df["step"].to_numpy(dtype=float)
    y = run_df["reward_scaled"].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    if len(x_unique) == 1:
        return np.full_like(common_steps, y_unique[0], dtype=float)

    return np.interp(common_steps, x_unique, y_unique)


# EDITED:
def ordered_algorithms(names: Iterable[str]) -> List[str]:
    names = list(dict.fromkeys(names))
    ordered = [algo for algo in DEFAULT_ALGO_ORDER if algo in names]
    extras = sorted([algo for algo in names if algo not in DEFAULT_ALGO_ORDER])
    return ordered + extras


# EDITED:
def build_algorithm_curves(df: pd.DataFrame, common_steps: np.ndarray) -> Dict[str, np.ndarray]:
    curves: Dict[str, np.ndarray] = {}
    for algorithm, algo_df in df.groupby("algorithm"):
        set_curves: List[np.ndarray] = []

        for _set_id, set_df in algo_df.groupby("set"):
            run_curves: List[np.ndarray] = []
            for _run_id, run_df in set_df.groupby("run_id"):
                if run_df.empty:
                    continue
                run_curves.append(interpolate_run(run_df, common_steps))

            if run_curves:
                set_curves.append(np.mean(np.vstack(run_curves), axis=0))

        if set_curves:
            curves[algorithm] = np.vstack(set_curves)

    return curves


# EDITED:
def build_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for run_name, run_df in df.groupby("run_name"):
        run_df = run_df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        rewards = run_df["reward"].to_numpy(dtype=float)
        steps = run_df["step"].to_numpy(dtype=float)
        best_idx = int(np.nanargmax(rewards))
        min_reward = float(np.nanmin(rewards))
        final_reward = float(rewards[-1])
        first_reward = float(rewards[0])
        first_step = int(steps[0])
        last_step = int(steps[-1])
        if steps.size > 1 and steps[-1] > steps[0]:
            auc_reward = float(np.trapezoid(rewards, steps) / (steps[-1] - steps[0]))
        else:
            auc_reward = final_reward

        meta = run_df.iloc[0]
        rows.append(
            {
                "algorithm": meta["algorithm"],
                "set": int(meta["set"]),
                "run_name": run_name,
                "run_id": meta["run_id"],
                "seed": int(meta["seed"]) if pd.notna(meta["seed"]) else np.nan,
                "device": meta["device"],
                "n_points": int(run_df.shape[0]),
                "first_step": first_step,
                "last_step": last_step,
                "first_reward": first_reward,
                "final_reward": final_reward,
                "best_reward": float(rewards[best_idx]),
                "best_step": int(steps[best_idx]),
                "min_reward": min_reward,
                "reward_range": float(np.nanmax(rewards) - min_reward),
                "curve_mean_reward": float(np.nanmean(rewards)),
                "curve_std_reward": float(np.nanstd(rewards, ddof=1)) if rewards.size > 1 else 0.0,
                "auc_reward": auc_reward,
                "progress_csv": meta["progress_csv"],
            }
        )

    return pd.DataFrame(rows).sort_values(["algorithm", "set", "seed", "run_name"]).reset_index(drop=True)


# EDITED:
def build_set_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    summary = (
        run_summary.groupby(["algorithm", "set"])
        .agg(
            runs=("run_name", "nunique"),
            mean_final_reward=("final_reward", "mean"),
            std_final_reward=("final_reward", "std"),
            mean_best_reward=("best_reward", "mean"),
            std_best_reward=("best_reward", "std"),
            mean_auc_reward=("auc_reward", "mean"),
            mean_last_step=("last_step", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["algorithm", "set"]).reset_index(drop=True)


# EDITED:
def build_algorithm_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    summary = (
        run_summary.groupby("algorithm")
        .agg(
            runs=("run_name", "nunique"),
            sets=("set", "nunique"),
            mean_final_reward=("final_reward", "mean"),
            std_final_reward=("final_reward", "std"),
            mean_best_reward=("best_reward", "mean"),
            std_best_reward=("best_reward", "std"),
            best_of_best_reward=("best_reward", "max"),
            mean_curve_reward=("curve_mean_reward", "mean"),
            mean_auc_reward=("auc_reward", "mean"),
            mean_last_step=("last_step", "mean"),
        )
        .reset_index()
    )
    ordered = ordered_algorithms(summary["algorithm"].tolist())
    return summary.set_index("algorithm").loc[ordered].reset_index()


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
def discover_tuned_provenance(slurm_out_dir: Path, exp_name: str, num_robots: int) -> pd.DataFrame:
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

        tuned_used_matches = list(TUNED_PARAMS_RE.finditer(combined_text))
        if tuned_used_matches:
            maybe_params = safe_literal_eval(tuned_used_matches[-1].group("params"))
            if isinstance(maybe_params, dict):
                tuned_params = maybe_params

        source_matches = list(TUNED_SOURCES_RE.finditer(combined_text))
        if source_matches:
            maybe_sources = safe_literal_eval(source_matches[-1].group("sources"))
            if isinstance(maybe_sources, dict):
                tuned_sources = maybe_sources

        record = {
            "run_name": run_name,
            "slurm_job_name": pair["job_name"],
            "slurm_job_id": pair["job_id"],
            "slurm_task_id": pair["task_id"],
            "slurm_out": str(pair["out_path"]) if pair["out_path"] is not None else None,
            "slurm_err": str(pair["err_path"]) if pair["err_path"] is not None else None,
            "tuning_run_name": tuning_run_name,
            "tuned_hyperparameters_json": json.dumps(tuned_params, sort_keys=True) if tuned_params else "",
            "tuned_param_count": len(tuned_params),
            "tuned_hyperparameter_sources_json": json.dumps(tuned_sources, sort_keys=True) if tuned_sources else "",
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
def plot_learning_curves(df: pd.DataFrame, common_steps: np.ndarray, plot_path: Path, show: bool) -> None:
    curves = build_algorithm_curves(df, common_steps)
    if not curves:
        raise ValueError("No curves could be built from the discovered logs.")

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in ordered_algorithms(curves.keys()):
        set_curves = curves[algorithm]
        mean_rewards = set_curves.mean(axis=0)
        std_rewards = set_curves.std(axis=0)

        ax.plot(common_steps, mean_rewards, label=algorithm, linewidth=2)
        ax.fill_between(
            common_steps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(r"Reward (x$10^6$)")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)

    if show and "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


# EDITED:
def plot_final_reward_boxplot(run_summary: pd.DataFrame, plot_path: Path) -> None:
    if run_summary.empty:
        return

    labels = []
    data = []
    for algorithm in ordered_algorithms(run_summary["algorithm"].unique()):
        values = (run_summary.loc[run_summary["algorithm"] == algorithm, "final_reward"] / 1_000_000.0).dropna().to_list()
        if not values:
            continue
        labels.append(algorithm)
        data.append(values)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(r"Final reward (x$10^6$)")
    ax.set_title("Distribution of final logged rewards for train-from-tuned runs")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_best_reward_heatmap(set_summary: pd.DataFrame, plot_path: Path) -> None:
    if set_summary.empty:
        return

    pivot = set_summary.pivot_table(index="algorithm", columns="set", values="mean_best_reward", aggfunc="mean")
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
    ax.set_title("Train-from-tuned mean best reward by algorithm and set")

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display.iat[i, j]
            text = "-" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label=r"Reward (x$10^6$)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


# EDITED:
def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) else ("" if value is None else str(value)))
    return out


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
            "No matching progress.csv files were found. "
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

    run_summary = build_run_summary(df)
    set_summary = build_set_summary(run_summary)
    algorithm_summary = build_algorithm_summary(run_summary)

    slurm_out_dir = Path(args.slurm_out_dir).expanduser().resolve() if args.slurm_out_dir else (project_root / "slurm_scripts" / "slurm_out").resolve()
    provenance_df = discover_tuned_provenance(slurm_out_dir, args.exp_name, args.num_robots)
    if not provenance_df.empty and not run_summary.empty:
        run_summary = run_summary.merge(provenance_df, on="run_name", how="left")

    min_step = float(df["step"].min())
    max_step = float(df["step"].max())
    common_steps = np.linspace(min_step, max_step, args.points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv_path = results_dir / f"{args.exp_name}_raw_for_{args.num_robots}_robots_{timestamp}.csv"
    run_summary_path = results_dir / f"{args.exp_name}_run_summary_for_{args.num_robots}_robots.csv"
    set_summary_path = results_dir / f"{args.exp_name}_set_summary_for_{args.num_robots}_robots.csv"
    algorithm_summary_path = results_dir / f"{args.exp_name}_results_for_{args.num_robots}_robots.csv"
    provenance_path = results_dir / f"{args.exp_name}_tuned_provenance_for_{args.num_robots}_robots.csv"
    curve_plot_path = plots_dir / f"{args.exp_name}_graph_for_{args.num_robots}_robots.png"
    boxplot_path = plots_dir / f"{args.exp_name}_final_reward_boxplot_for_{args.num_robots}_robots.png"
    heatmap_path = plots_dir / f"{args.exp_name}_best_reward_heatmap_for_{args.num_robots}_robots.png"

    dataframe_for_csv(df).to_csv(raw_csv_path, index=False)
    dataframe_for_csv(run_summary).to_csv(run_summary_path, index=False)
    dataframe_for_csv(set_summary).to_csv(set_summary_path, index=False)
    dataframe_for_csv(algorithm_summary).to_csv(algorithm_summary_path, index=False)
    dataframe_for_csv(provenance_df).to_csv(provenance_path, index=False)

    plot_learning_curves(df, common_steps, curve_plot_path, args.show)
    plot_final_reward_boxplot(run_summary, boxplot_path)
    plot_best_reward_heatmap(set_summary, heatmap_path)

    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Log layout used: {layout}")
    print(f"Progress files used: {len(discovered)}")
    print(f"Slurm provenance rows used: {0 if provenance_df.empty else provenance_df.shape[0]}")
    print(f"Saved raw data to: {raw_csv_path}")
    print(f"Saved run summary to: {run_summary_path}")
    print(f"Saved set summary to: {set_summary_path}")
    print(f"Saved algorithm summary to: {algorithm_summary_path}")
    print(f"Saved tuned provenance to: {provenance_path}")
    print(f"Saved learning-curve plot to: {curve_plot_path}")
    print(f"Saved final-reward boxplot to: {boxplot_path}")
    print(f"Saved best-reward heatmap to: {heatmap_path}")


if __name__ == "__main__":
    main()
