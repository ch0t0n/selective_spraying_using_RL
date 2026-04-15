# EDITED:
"""Create tables, plots, and a written report for default training and Optuna tuning runs.

The script is designed to summarize the runs launched by the project Slurm scripts,
with a focus on:
- run coverage versus expected runs implied by the Slurm scripts
- default training learning curves and per-run summaries
- Optuna tuning trial histories, best-so-far behavior, and best-parameter summaries
- a direct training-versus-tuning comparison at the algorithm level

Summary tables are written to plotting/results and figures are written to plotting/plots.
"""

from __future__ import annotations

# EDITED:
import argparse
import ast
import itertools
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _np_trapezoid_compat(y, x=None, dx=1.0, axis=-1):
    trapezoid_fn = getattr(np, "trapezoid", None)
    if trapezoid_fn is not None:
        return trapezoid_fn(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)

# EDITED:
DEFAULT_ALGO_ORDER: Tuple[str, ...] = ("A2C", "ARS", "CrossQ", "PPO", "TQC", "TRPO")
ALGO_SET_RE = re.compile(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)$")
SEED_RE = re.compile(r"_seed(?P<seed>-?\d+)")
DEVICE_RE = re.compile(r"_(?P<device>cpu|cuda)$")
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run_name>\S+)")
ALG_LINE_RE = re.compile(
    r"ALG=(?P<algorithm>\S+)\s+SET=(?P<set>\d+)\s+SEED=(?P<seed>-?\d+)\s+DEVICE=(?P<device>\S+)"
)
RUN_INFO_RE = re.compile(
    r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)_seed(?P<seed>-?\d+)_"
    r"(?P<exp_name>[^_]+)_(?P<num_robots>\d+)_robots_(?P<device>cpu|cuda)$"
)
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
OPTUNA_FINISHED_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+finished\s+with\s+value:\s*"
    r"(?P<value>[-+0-9.eEinfINFnanNAN]+)\s+and\s+parameters:\s*"
    r"(?P<params>\{.*\})"
)
OPTUNA_FAILED_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+failed(?:\s+with\s+parameters:\s*(?P<params>\{.*\}))?"
)
OPTUNA_PRUNED_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+pruned(?:\s+with\s+parameters:\s*(?P<params>\{.*\}))?"
)
USER_TRIAL_RESULT_RE = re.compile(
    r"TRIAL_RESULT\s+trial=(?P<trial>\d+)\s+mean_reward=(?P<value>[-+0-9.eEinfINFnanNAN]+)\s+params=(?P<params>\{.*\})"
)


# EDITED:
@dataclass
class SlurmSpec:
    path: Path
    job_name: str
    algorithms: List[str]
    sets: List[int]
    seeds: List[int]
    device: str
    steps: int
    num_robots: int
    run_name_template: str
    entry_script: str
    kind: str
    exp_name: Optional[str]


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze default training and Optuna tuning results, then write tables, "
            "figures, and a markdown report to plotting/results and plotting/plots."
        )
    )
    parser.add_argument("--project-root", type=str, default=None, help="Optional repository root override.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional override for the plotting output directory.")
    parser.add_argument("--training-exp-name", type=str, default="default", help="Training experiment name used in logs/training_<exp_name>_logs.")
    parser.add_argument("--tuning-exp-name", type=str, default="random", help="Tuning experiment name encoded in run names such as *_random_3_robots_*.")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to match in run names.")
    parser.add_argument("--training-log-root", type=str, default=None, help="Optional explicit training log root.")
    parser.add_argument("--slurm-out-dir", type=str, default=None, help="Optional override for slurm_scripts/slurm_out.")
    parser.add_argument(
        "--slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/train_default_all_3_robots.sh",
            "slurm_scripts/crossq_default_3_robots.sh",
            "slurm_scripts/train_random_all_3_robots.sh",
            "slurm_scripts/crossq_random_3_robots.sh",
        ],
        help="Slurm scripts whose expected runs should be summarized.",
    )
    parser.add_argument("--points", type=int, default=200, help="Interpolation points for mean/std line plots.")
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

    return plotting_dir


# EDITED:
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


# EDITED:
def ordered_algorithms(names: Iterable[str]) -> List[str]:
    names = list(dict.fromkeys(names))
    ordered = [algo for algo in DEFAULT_ALGO_ORDER if algo in names]
    extras = sorted([algo for algo in names if algo not in DEFAULT_ALGO_ORDER])
    return ordered + extras


# EDITED:
def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(parsed):
        return float("nan")
    return parsed


# EDITED:
def safe_literal_eval(text: Optional[str]) -> Optional[object]:
    if text is None:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


# EDITED:
def jsonify_object(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


# EDITED:
def parse_shell_list(text: str, variable_name: str) -> List[str]:
    pattern = re.compile(rf"^{re.escape(variable_name)}=\((.*?)\)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return []
    raw = match.group(1)
    items = re.findall(r'"([^"]+)"|([^\s]+)', raw)
    values = []
    for quoted, bare in items:
        values.append(quoted or bare)
    return values


# EDITED:
def parse_scalar_assignment(text: str, variable_name: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(variable_name)}=(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


# EDITED:
def parse_sbatch_directive(text: str, directive_name: str) -> Optional[str]:
    pattern = re.compile(rf"^#SBATCH\s+--{re.escape(directive_name)}(?:=(.+)|\s+(.+))$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip()


# EDITED:
def parse_run_name_template(text: str) -> Optional[str]:
    pattern = re.compile(r'^RUN_NAME="(.+)"$', re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


# EDITED:
def parse_python_entry(text: str) -> Optional[str]:
    pattern = re.compile(r"python3?\s+([^\s\\]+\.py)")
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


# EDITED:
def render_shell_template(template: str, variables: Dict[str, object]) -> str:
    def replace_braced(match: re.Match) -> str:
        key = match.group(1)
        return str(variables[key])

    rendered = re.sub(r"\$\{(\w+)\}", replace_braced, template)
    rendered = re.sub(r"\$(\w+)", replace_braced, rendered)
    return rendered


# EDITED:
def parse_run_name_bits(run_name: str) -> Optional[Dict[str, object]]:
    match = RUN_INFO_RE.match(run_name)
    if match is None:
        return None
    return {
        "algorithm": match.group("algorithm"),
        "set": int(match.group("set")),
        "seed": int(match.group("seed")),
        "exp_name": match.group("exp_name"),
        "num_robots": int(match.group("num_robots")),
        "device": match.group("device"),
    }


# EDITED:
def parse_slurm_spec(path: Path) -> SlurmSpec:
    text = read_text(path)
    if not text:
        raise FileNotFoundError(path)

    job_name = parse_sbatch_directive(text, "job-name") or path.stem
    algorithms = parse_shell_list(text, "algorithms")
    sets = [int(value) for value in parse_shell_list(text, "sets")]
    seeds_raw = parse_shell_list(text, "seed") or parse_shell_list(text, "seeds")
    seeds = [int(value) for value in seeds_raw]
    device = "cpu"
    run_name_template = parse_run_name_template(text) or ""
    entry_script = parse_python_entry(text) or ""
    steps = int(parse_scalar_assignment(text, "steps") or 0)
    num_robots = int(parse_scalar_assignment(text, "num_robots") or 0)

    if run_name_template:
        first_run = render_shell_template(
            run_name_template,
            {
                "algorithm": algorithms[0] if algorithms else "ALG",
                "set": sets[0] if sets else 0,
                "seed": seeds[0] if seeds else 0,
            },
        )
        bits = parse_run_name_bits(first_run)
        if bits is not None:
            device = str(bits["device"])
            exp_name = str(bits["exp_name"])
        else:
            exp_name = None
    else:
        exp_name = None

    kind = "tuning" if Path(entry_script).stem == "tuning" or "tuning" in Path(entry_script).stem else "training"

    missing = []
    if not algorithms:
        missing.append("algorithms")
    if not sets:
        missing.append("sets")
    if not seeds:
        missing.append("seeds")
    if not run_name_template:
        missing.append("run_name_template")
    if missing:
        raise ValueError(f"Unable to parse {path}: missing {', '.join(missing)}")

    return SlurmSpec(
        path=path,
        job_name=job_name,
        algorithms=algorithms,
        sets=sets,
        seeds=seeds,
        device=device,
        steps=steps,
        num_robots=num_robots,
        run_name_template=run_name_template,
        entry_script=entry_script,
        kind=kind,
        exp_name=exp_name,
    )


# EDITED:
def build_expected_inventory(specs: Sequence[SlurmSpec]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in specs:
        for algorithm, set_id, seed in itertools.product(spec.algorithms, spec.sets, spec.seeds):
            run_name = render_shell_template(
                spec.run_name_template,
                {"algorithm": algorithm, "set": set_id, "seed": seed},
            )
            bits = parse_run_name_bits(run_name) or {}
            rows.append(
                {
                    "script_label": spec.path.stem,
                    "script_path": str(spec.path),
                    "job_name": spec.job_name,
                    "kind": spec.kind,
                    "entry_script": spec.entry_script,
                    "exp_name": bits.get("exp_name", spec.exp_name),
                    "algorithm": algorithm,
                    "set": int(bits.get("set", set_id)),
                    "seed": int(bits.get("seed", seed)),
                    "device": bits.get("device", spec.device),
                    "steps": spec.steps,
                    "num_robots": int(bits.get("num_robots", spec.num_robots)),
                    "run_name": run_name,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["kind", "script_label", "algorithm", "set", "seed"]).reset_index(drop=True)


# EDITED:
def candidate_training_log_roots(project_root: Path, exp_name: str, explicit_log_root: Optional[str] = None) -> List[Path]:
    if explicit_log_root:
        return [Path(explicit_log_root).expanduser().resolve()]
    return [
        project_root / "logs" / f"training_{exp_name}_logs",
        project_root / f"training_{exp_name}_logs",
    ]


# EDITED:
def first_matching_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


# EDITED:
def discover_training_progress(
    project_root: Path,
    exp_name: str,
    num_robots: int,
    explicit_log_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Path]]:
    frames: List[pd.DataFrame] = []
    used_roots: List[Path] = []

    for base in candidate_training_log_roots(project_root, exp_name, explicit_log_root):
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
            if run_root != f"{num_robots}_robots" and f"_{exp_name}_{num_robots}_robots" not in run_root:
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
            out["set"] = int(algo_match.group("set"))
            out["run_name"] = run_root
            out["run_id"] = str(progress_csv.parent)

            seed_match = SEED_RE.search(run_root)
            device_match = DEVICE_RE.search(run_root)
            out["seed"] = int(seed_match.group("seed")) if seed_match else np.nan
            out["device"] = device_match.group("device") if device_match else None
            out["progress_csv"] = str(progress_csv)
            out["exp_name"] = exp_name
            frames.append(out)
            used_roots.append(base)

    if not frames:
        return pd.DataFrame(), sorted(set(used_roots))

    df = pd.concat(frames, ignore_index=True)
    cols = [
        "algorithm",
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
def parse_tuning_output_metadata(text: str) -> Optional[Dict[str, object]]:
    run_name_match = RUN_NAME_RE.search(text)
    run_name = run_name_match.group("run_name") if run_name_match else None
    if not run_name:
        return None

    alg_line_match = ALG_LINE_RE.search(text)
    if alg_line_match is not None:
        return {
            "run_name": run_name,
            "algorithm": alg_line_match.group("algorithm"),
            "set": int(alg_line_match.group("set")),
            "seed": int(alg_line_match.group("seed")),
            "device": alg_line_match.group("device"),
            **({} if parse_run_name_bits(run_name) is None else {"exp_name": parse_run_name_bits(run_name)["exp_name"], "num_robots": parse_run_name_bits(run_name)["num_robots"]}),
        }

    bits = parse_run_name_bits(run_name)
    if bits is None:
        return None
    return {
        "run_name": run_name,
        "algorithm": bits["algorithm"],
        "set": int(bits["set"]),
        "seed": int(bits["seed"]),
        "device": bits["device"],
        "exp_name": bits["exp_name"],
        "num_robots": int(bits["num_robots"]),
    }


# EDITED:
def has_optuna_trial_lines(text: str) -> bool:
    return any(
        pattern.search(text) is not None
        for pattern in (OPTUNA_FINISHED_RE, OPTUNA_FAILED_RE, OPTUNA_PRUNED_RE, USER_TRIAL_RESULT_RE)
    )


# EDITED:
def parse_trial_rows(text: str, metadata: Dict[str, object], source_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen_keys = set()
    log_mtime = datetime.fromtimestamp(source_path.stat().st_mtime)

    for line in text.splitlines():
        state = None
        trial_match = USER_TRIAL_RESULT_RE.search(line)
        if trial_match is not None:
            state = "finished"
        else:
            trial_match = OPTUNA_FINISHED_RE.search(line)
            if trial_match is not None:
                state = "finished"
            else:
                trial_match = OPTUNA_FAILED_RE.search(line)
                if trial_match is not None:
                    state = "failed"
                else:
                    trial_match = OPTUNA_PRUNED_RE.search(line)
                    if trial_match is not None:
                        state = "pruned"

        if trial_match is None or state is None:
            continue

        trial = int(trial_match.group("trial"))
        params = safe_literal_eval(trial_match.groupdict().get("params"))
        reward = np.nan
        if state == "finished":
            reward = safe_float(trial_match.groupdict().get("value"))

        key = (trial, state)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        rows.append(
            {
                "run_name": metadata["run_name"],
                "algorithm": metadata["algorithm"],
                "set": int(metadata["set"]),
                "seed": int(metadata["seed"]),
                "device": metadata["device"],
                "trial": trial,
                "reward": reward,
                "state": state,
                "params": params if isinstance(params, dict) else None,
                "params_json": jsonify_object(params if isinstance(params, dict) else params),
                "slurm_out": str(source_path),
                "log_mtime": log_mtime,
            }
        )
    return rows


# EDITED:
def expand_param_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "params" not in df.columns:
        return df

    param_keys = sorted({key for params in df["params"] if isinstance(params, dict) for key in params})
    for key in param_keys:
        df[f"param_{key}"] = [params.get(key) if isinstance(params, dict) else np.nan for params in df["params"]]
    return df


# EDITED:
def discover_tuning_trials(
    slurm_out_dir: Path,
    exp_name: Optional[str],
    num_robots: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest_by_run_name: Dict[str, Tuple[Path, datetime, str, Dict[str, object]]] = {}

    if not slurm_out_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for path in sorted(slurm_out_dir.glob("*.out")):
        text = read_text(path)
        metadata = parse_tuning_output_metadata(text)
        if metadata is None:
            continue

        run_name = str(metadata["run_name"])
        if exp_name is not None and f"_{exp_name}_{num_robots}_robots_" not in run_name:
            bits = parse_run_name_bits(run_name)
            if bits is None or bits.get("exp_name") != exp_name or bits.get("num_robots") != num_robots:
                continue

        if not has_optuna_trial_lines(text):
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        current = latest_by_run_name.get(run_name)
        if current is None or mtime > current[1]:
            latest_by_run_name[run_name] = (path, mtime, text, metadata)

    run_rows: List[Dict[str, object]] = []
    trial_rows: List[Dict[str, object]] = []
    for run_name, (path, mtime, text, metadata) in sorted(latest_by_run_name.items()):
        run_rows.append(
            {
                "run_name": run_name,
                "algorithm": metadata["algorithm"],
                "set": int(metadata["set"]),
                "seed": int(metadata["seed"]),
                "device": metadata["device"],
                "slurm_out": str(path),
                "log_mtime": mtime,
            }
        )
        trial_rows.extend(parse_trial_rows(text, metadata, path))

    trial_df = pd.DataFrame(trial_rows)
    run_df = pd.DataFrame(run_rows)
    if trial_df.empty:
        return trial_df, run_df

    trial_df = trial_df.sort_values(["run_name", "trial", "state", "log_mtime"]).drop_duplicates(
        subset=["run_name", "trial", "state"],
        keep="last",
    )
    trial_df = expand_param_columns(trial_df)
    return trial_df.reset_index(drop=True), run_df.reset_index(drop=True)


# EDITED:
def interpolate_series(x: np.ndarray, y: np.ndarray, common_x: np.ndarray) -> np.ndarray:
    if x.size == 0 or y.size == 0:
        return np.full_like(common_x, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    if x_unique.size == 1:
        return np.full_like(common_x, y_unique[0], dtype=float)
    return np.interp(common_x, x_unique, y_unique)


# EDITED:
def build_algorithm_curves(df: pd.DataFrame, x_col: str, y_col: str, common_x: np.ndarray) -> Dict[str, np.ndarray]:
    curves: Dict[str, np.ndarray] = {}
    for algorithm, algo_df in df.groupby("algorithm"):
        set_curves: List[np.ndarray] = []
        for _set_id, set_df in algo_df.groupby("set"):
            run_curves: List[np.ndarray] = []
            for _run_name, run_df in set_df.groupby("run_name"):
                sub = run_df[[x_col, y_col]].dropna().copy()
                if sub.empty:
                    continue
                run_curves.append(
                    interpolate_series(
                        sub[x_col].to_numpy(dtype=float),
                        sub[y_col].to_numpy(dtype=float),
                        common_x,
                    )
                )
            if run_curves:
                set_curves.append(np.nanmean(np.vstack(run_curves), axis=0))
        if set_curves:
            curves[algorithm] = np.vstack(set_curves)
    return curves


# EDITED:
def build_training_run_summary(df: pd.DataFrame) -> pd.DataFrame:
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
            # auc_reward = float(np.trapz(rewards, steps) / (steps[-1] - steps[0]))
            # auc_reward = float(np.trapezoid(rewards, steps) / (steps[-1] - steps[0]))
            auc_reward = float(_np_trapezoid_compat(rewards, steps) / (steps[-1] - steps[0]))
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
def build_training_set_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
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
def build_training_algorithm_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
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
def build_tuning_run_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for run_name, all_df in trial_df.groupby("run_name"):
        all_df = all_df.sort_values(["trial", "state"]).copy()
        finished = all_df[(all_df["state"] == "finished") & all_df["reward"].notna()].sort_values("trial")
        meta = all_df.iloc[0]
        state_counts = all_df.groupby("state")["trial"].nunique().to_dict()

        record: Dict[str, object] = {
            "algorithm": meta["algorithm"],
            "set": int(meta["set"]),
            "run_name": run_name,
            "seed": int(meta["seed"]),
            "device": meta["device"],
            "observed_trials": int(all_df["trial"].nunique()),
            "max_trial": int(all_df["trial"].max()) if not all_df.empty else np.nan,
            "completed_trials": int(state_counts.get("finished", 0)),
            "failed_trials": int(state_counts.get("failed", 0)),
            "pruned_trials": int(state_counts.get("pruned", 0)),
            "slurm_out": meta["slurm_out"],
            "log_mtime": meta["log_mtime"],
        }

        if finished.empty:
            record.update(
                {
                    "first_reward": np.nan,
                    "last_reward": np.nan,
                    "mean_reward": np.nan,
                    "median_reward": np.nan,
                    "std_reward": np.nan,
                    "min_reward": np.nan,
                    "best_reward": np.nan,
                    "best_trial": np.nan,
                    "reward_range": np.nan,
                    "improvement_best_minus_first": np.nan,
                    "improvement_best_minus_median": np.nan,
                    "best_params_json": "",
                }
            )
            rows.append(record)
            continue

        rewards = finished["reward"].to_numpy(dtype=float)
        trials = finished["trial"].to_numpy(dtype=int)
        best_idx = int(np.nanargmax(rewards))
        best_row = finished.iloc[best_idx]
        best_params = best_row["params"] if isinstance(best_row["params"], dict) else {}

        record.update(
            {
                "first_reward": float(rewards[0]),
                "last_reward": float(rewards[-1]),
                "mean_reward": float(np.nanmean(rewards)),
                "median_reward": float(np.nanmedian(rewards)),
                "std_reward": float(np.nanstd(rewards, ddof=1)) if rewards.size > 1 else 0.0,
                "min_reward": float(np.nanmin(rewards)),
                "best_reward": float(rewards[best_idx]),
                "best_trial": int(trials[best_idx]),
                "reward_range": float(np.nanmax(rewards) - np.nanmin(rewards)),
                "improvement_best_minus_first": float(rewards[best_idx] - rewards[0]),
                "improvement_best_minus_median": float(rewards[best_idx] - np.nanmedian(rewards)),
                "best_params_json": jsonify_object(best_params),
            }
        )
        for key, value in best_params.items():
            record[f"best_param_{key}"] = value
        rows.append(record)

    return pd.DataFrame(rows).sort_values(["algorithm", "set", "seed", "run_name"]).reset_index(drop=True)


# EDITED:
def build_tuning_set_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    summary = (
        run_summary.groupby(["algorithm", "set"])
        .agg(
            runs=("run_name", "nunique"),
            mean_best_reward=("best_reward", "mean"),
            std_best_reward=("best_reward", "std"),
            mean_first_reward=("first_reward", "mean"),
            mean_last_reward=("last_reward", "mean"),
            mean_improvement=("improvement_best_minus_first", "mean"),
            mean_completed_trials=("completed_trials", "mean"),
            mean_failed_trials=("failed_trials", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["algorithm", "set"]).reset_index(drop=True)


# EDITED:
def build_tuning_algorithm_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    summary = (
        run_summary.groupby("algorithm")
        .agg(
            runs=("run_name", "nunique"),
            sets=("set", "nunique"),
            mean_best_reward=("best_reward", "mean"),
            std_best_reward=("best_reward", "std"),
            best_of_best_reward=("best_reward", "max"),
            mean_first_reward=("first_reward", "mean"),
            mean_last_reward=("last_reward", "mean"),
            mean_improvement=("improvement_best_minus_first", "mean"),
            mean_completed_trials=("completed_trials", "mean"),
            mean_failed_trials=("failed_trials", "mean"),
        )
        .reset_index()
    )
    ordered = ordered_algorithms(summary["algorithm"].tolist())
    return summary.set_index("algorithm").loc[ordered].reset_index()


# EDITED:
def build_tuning_hyperparameter_summary(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame()

    finished = trial_df[(trial_df["state"] == "finished") & trial_df["reward"].notna()].copy()
    if finished.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    param_cols = [col for col in finished.columns if col.startswith("param_")]
    for algorithm, algo_df in finished.groupby("algorithm"):
        for param_col in param_cols:
            numeric = pd.to_numeric(algo_df[param_col], errors="coerce")
            valid = algo_df.loc[numeric.notna(), ["reward"]].copy()
            if valid.empty:
                continue
            valid[param_col] = numeric.loc[numeric.notna()].to_numpy()
            if valid.shape[0] >= 2 and valid[param_col].nunique() >= 2 and valid["reward"].nunique() >= 2:
                spearman_corr = valid[param_col].corr(valid["reward"], method="spearman")
            else:
                spearman_corr = np.nan

            top_quantile = 0.8 if valid.shape[0] >= 5 else 0.5
            threshold = valid["reward"].quantile(top_quantile)
            top_valid = valid[valid["reward"] >= threshold]
            rows.append(
                {
                    "algorithm": algorithm,
                    "parameter": param_col.replace("param_", ""),
                    "n_trials": int(valid.shape[0]),
                    "unique_values": int(valid[param_col].nunique()),
                    "spearman_corr": float(spearman_corr) if pd.notna(spearman_corr) else np.nan,
                    "overall_mean": float(valid[param_col].mean()),
                    "overall_median": float(valid[param_col].median()),
                    "overall_min": float(valid[param_col].min()),
                    "overall_max": float(valid[param_col].max()),
                    "top_bucket_mean": float(top_valid[param_col].mean()),
                    "top_bucket_median": float(top_valid[param_col].median()),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["algorithm", "parameter"]).reset_index(drop=True)


# EDITED:
def build_comparison_table(training_algorithm_summary: pd.DataFrame, tuning_algorithm_summary: pd.DataFrame) -> pd.DataFrame:
    if training_algorithm_summary.empty or tuning_algorithm_summary.empty:
        return pd.DataFrame()

    merged = training_algorithm_summary.merge(
        tuning_algorithm_summary,
        on="algorithm",
        how="inner",
        suffixes=("_training", "_tuning"),
    )
    if merged.empty:
        return merged

    merged["tuning_minus_training_mean_best"] = merged["mean_best_reward_tuning"] - merged["mean_best_reward_training"]
    merged["tuning_minus_training_best_of_best"] = merged["best_of_best_reward_tuning"] - merged["best_of_best_reward_training"]
    ordered = ordered_algorithms(merged["algorithm"].tolist())
    return merged.set_index("algorithm").loc[ordered].reset_index()


# EDITED:
def add_artifact(artifacts: List[Dict[str, str]], category: str, kind: str, path: Path, description: str) -> None:
    artifacts.append(
        {
            "category": category,
            "kind": kind,
            "path": str(path),
            "filename": path.name,
            "description": description,
        }
    )


# EDITED:
def plot_mean_std_curves(
    curves: Dict[str, np.ndarray],
    common_x: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    if not curves:
        return

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(11, 7))
    for algorithm in ordered_algorithms(curves.keys()):
        algo_curves = curves[algorithm]
        mean_values = np.nanmean(algo_curves, axis=0)
        std_values = np.nanstd(algo_curves, axis=0)
        ax.plot(common_x, mean_values, label=algorithm, linewidth=2)
        ax.fill_between(common_x, mean_values - std_values, mean_values + std_values, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_boxplot(run_summary: pd.DataFrame, value_col: str, ylabel: str, title: str, path: Path) -> None:
    if run_summary.empty or value_col not in run_summary.columns:
        return

    labels = []
    data = []
    for algorithm in ordered_algorithms(run_summary["algorithm"].unique()):
        values = (run_summary.loc[run_summary["algorithm"] == algorithm, value_col] / 1_000_000.0).dropna().to_list()
        if not values:
            continue
        labels.append(algorithm)
        data.append(values)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    # ax.boxplot(data, labels=labels)
    # EDITED:
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_reward_heatmap(summary_df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
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
            text = "-" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label=r"Reward (x$10^6$)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_coverage_heatmap(coverage_summary: pd.DataFrame, path: Path) -> None:
    if coverage_summary.empty:
        return

    coverage_summary = coverage_summary.copy()
    coverage_summary["row_label"] = coverage_summary["script_label"] + " (" + coverage_summary["kind"] + ")"
    pivot_ratio = coverage_summary.pivot_table(index="row_label", columns="algorithm", values="coverage_ratio", aggfunc="mean")
    if pivot_ratio.empty:
        return

    ordered_cols = [algo for algo in ordered_algorithms(pivot_ratio.columns.tolist()) if algo in pivot_ratio.columns]
    pivot_ratio = pivot_ratio[ordered_cols]
    pivot_found = coverage_summary.pivot_table(index="row_label", columns="algorithm", values="found_runs", aggfunc="sum").reindex(index=pivot_ratio.index, columns=pivot_ratio.columns)
    pivot_expected = coverage_summary.pivot_table(index="row_label", columns="algorithm", values="expected_runs", aggfunc="sum").reindex(index=pivot_ratio.index, columns=pivot_ratio.columns)

    fig, ax = plt.subplots(figsize=(12, max(4, 1.2 * pivot_ratio.shape[0] + 2)))
    im = ax.imshow(pivot_ratio.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(pivot_ratio.shape[1]))
    ax.set_xticklabels(pivot_ratio.columns.tolist())
    ax.set_yticks(np.arange(pivot_ratio.shape[0]))
    ax.set_yticklabels(pivot_ratio.index.tolist())
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Slurm script")
    ax.set_title("Coverage ratio by script and algorithm")

    for i in range(pivot_ratio.shape[0]):
        for j in range(pivot_ratio.shape[1]):
            ratio = pivot_ratio.iat[i, j]
            found = pivot_found.iat[i, j]
            expected = pivot_expected.iat[i, j]
            text = "-" if pd.isna(ratio) else f"{int(found)}/{int(expected)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Coverage ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_comparison_bars(comparison_df: pd.DataFrame, path: Path) -> None:
    if comparison_df.empty:
        return

    labels = ordered_algorithms(comparison_df["algorithm"].tolist())
    df = comparison_df.set_index("algorithm").loc[labels].reset_index()
    x = np.arange(df.shape[0])
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(x - width / 2, df["mean_best_reward_training"] / 1_000_000.0, width, label="Default training mean best")
    ax.bar(x + width / 2, df["mean_best_reward_tuning"] / 1_000_000.0, width, label="Tuning mean best")
    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"].tolist())
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(r"Reward (x$10^6$)")
    ax.set_title("Algorithm-level comparison: default training vs tuning")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def plot_hyperparameter_correlation(hyper_df: pd.DataFrame, algorithm: str, path: Path) -> None:
    algo_df = hyper_df[hyper_df["algorithm"] == algorithm].copy()
    if algo_df.empty:
        return

    algo_df = algo_df.sort_values("spearman_corr", key=lambda series: series.abs(), ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * algo_df.shape[0] + 2)))
    ax.barh(algo_df["parameter"], algo_df["spearman_corr"])
    ax.set_xlabel("Spearman correlation with objective value")
    ax.set_ylabel("Hyperparameter")
    ax.set_title(f"Hyperparameter correlation summary for {algorithm}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# EDITED:
def coverage_summary_from_inventory(inventory_df: pd.DataFrame) -> pd.DataFrame:
    if inventory_df.empty:
        return pd.DataFrame()
    summary = (
        inventory_df.groupby(["script_label", "kind", "algorithm"])
        .agg(
            expected_runs=("run_name", "size"),
            found_runs=("expected_output_found", "sum"),
        )
        .reset_index()
    )
    summary["coverage_ratio"] = np.where(
        summary["expected_runs"] > 0,
        summary["found_runs"] / summary["expected_runs"],
        np.nan,
    )
    return summary.sort_values(["kind", "script_label", "algorithm"]).reset_index(drop=True)


# EDITED:
def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(jsonify_object)
    return out


# EDITED:
def format_reward_millions(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    try:
        return f"{float(value) / 1_000_000.0:.3f}"
    except (TypeError, ValueError):
        return str(value)


# EDITED:
def format_number(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


# EDITED:
def markdown_table(df: pd.DataFrame, formatters: Optional[Dict[str, callable]] = None, max_rows: int = 10) -> str:
    if df.empty:
        return "_No data available._"

    preview = df.head(max_rows).copy()
    headers = preview.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in preview.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if formatters and col in formatters:
                values.append(formatters[col](value))
            else:
                values.append(format_number(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if df.shape[0] > max_rows:
        lines.append("")
        lines.append(f"_Showing first {max_rows} of {df.shape[0]} rows._")
    return "\n".join(lines)


# EDITED:
def top_algorithm(summary_df: pd.DataFrame, metric_col: str, higher_is_better: bool = True) -> Tuple[Optional[str], Optional[float]]:
    if summary_df.empty or metric_col not in summary_df.columns:
        return None, None
    valid = summary_df[["algorithm", metric_col]].dropna()
    if valid.empty:
        return None, None
    row = valid.sort_values(metric_col, ascending=not higher_is_better).iloc[0]
    return str(row["algorithm"]), float(row[metric_col])


# EDITED:
def build_report(
    report_path: Path,
    artifacts_df: pd.DataFrame,
    warnings: List[str],
    slurm_specs: Sequence[SlurmSpec],
    inventory_df: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    training_roots: List[Path],
    training_run_summary: pd.DataFrame,
    training_algo_summary: pd.DataFrame,
    tuning_run_summary: pd.DataFrame,
    tuning_algo_summary: pd.DataFrame,
    tuning_hyper_summary: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Experiment analysis report")
    lines.append("")
    lines.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    lines.append("")
    lines.append("This report summarizes the available outputs from the default training runs and the Optuna tuning runs implied by the Slurm launch scripts. All detailed tables are written to `plotting/results`, and all figures are written to `plotting/plots`.")
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Parsed Slurm scripts")
    lines.append("")
    if slurm_specs:
        rows = []
        for spec in slurm_specs:
            rows.append(
                {
                    "script": spec.path.name,
                    "kind": spec.kind,
                    "algorithms": len(spec.algorithms),
                    "sets": len(spec.sets),
                    "seeds": len(spec.seeds),
                    "steps": spec.steps,
                    "exp_name": spec.exp_name or "-",
                }
            )
        lines.append(markdown_table(pd.DataFrame(rows), max_rows=20))
    else:
        lines.append("_No Slurm scripts were parsed._")
    lines.append("")

    lines.append("## Run coverage")
    lines.append("")
    if inventory_df.empty:
        lines.append("_No expected-run inventory could be built from the supplied Slurm scripts._")
    else:
        expected_total = int(inventory_df.shape[0])
        found_total = int(inventory_df["expected_output_found"].sum())
        missing_total = int((~inventory_df["expected_output_found"]).sum())
        lines.append(f"The inventory contains **{expected_total}** expected runs. Matching outputs were found for **{found_total}** of them, leaving **{missing_total}** runs without a matching result artifact.")
        lines.append("")
        lines.append("`analysis_run_inventory.csv` contains one row per expected run. `analysis_run_coverage_summary.csv` aggregates that inventory by Slurm script and algorithm.")
        lines.append("")
        if not coverage_summary.empty:
            lines.append("`analysis_run_coverage_heatmap.png` shows the ratio of discovered outputs to expected outputs for each script/algorithm combination. Each cell is annotated as `found/expected`.")
            lines.append("")
            lines.append(markdown_table(coverage_summary, formatters={"coverage_ratio": lambda v: format_number(v, 2)}, max_rows=30))
    lines.append("")

    lines.append("## Default training analysis")
    lines.append("")
    if training_run_summary.empty:
        lines.append("_No default training progress logs were discovered._")
    else:
        lines.append(f"Usable default-training learning curves were discovered under: {', '.join(str(path) for path in training_roots) if training_roots else '-'}.")
        lines.append("")
        best_peak_algo, best_peak_value = top_algorithm(training_algo_summary, "mean_best_reward", higher_is_better=True)
        best_final_algo, best_final_value = top_algorithm(training_algo_summary, "mean_final_reward", higher_is_better=True)
        stable_algo, stable_value = top_algorithm(training_algo_summary, "std_best_reward", higher_is_better=False)
        if best_peak_algo is not None:
            lines.append(f"Highest mean peak training reward: **{best_peak_algo}** at **{format_reward_millions(best_peak_value)} x10^6** reward units.")
        if best_final_algo is not None:
            lines.append(f"Highest mean final logged training reward: **{best_final_algo}** at **{format_reward_millions(best_final_value)} x10^6**.")
        if stable_algo is not None:
            lines.append(f"Lowest cross-run variability in peak reward: **{stable_algo}** with a standard deviation of **{format_reward_millions(stable_value)} x10^6**.")
        lines.append("")
        lines.append("`analysis_training_learning_curves.png` overlays the mean default-training learning curve for each algorithm with one-standard-deviation shading across sets.")
        lines.append("")
        lines.append("`analysis_training_final_reward_boxplot.png` summarizes the distribution of each run's final logged reward, while `analysis_training_best_reward_heatmap.png` shows which algorithms were strongest on which environment sets.")
        lines.append("")
        lines.append(markdown_table(
            training_algo_summary[[
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

    lines.append("## Tuning analysis")
    lines.append("")
    if tuning_run_summary.empty:
        lines.append("_No Optuna tuning trial histories were discovered in the Slurm `.out` files._")
    else:
        best_tuning_algo, best_tuning_value = top_algorithm(tuning_algo_summary, "mean_best_reward", higher_is_better=True)
        biggest_improve_algo, biggest_improve_value = top_algorithm(tuning_algo_summary, "mean_improvement", higher_is_better=True)
        stable_tuning_algo, stable_tuning_value = top_algorithm(tuning_algo_summary, "std_best_reward", higher_is_better=False)
        if best_tuning_algo is not None:
            lines.append(f"Highest mean best tuning objective: **{best_tuning_algo}** at **{format_reward_millions(best_tuning_value)} x10^6**.")
        if biggest_improve_algo is not None:
            lines.append(f"Largest average improvement from first completed trial to best trial: **{biggest_improve_algo}** at **{format_reward_millions(biggest_improve_value)} x10^6**.")
        if stable_tuning_algo is not None:
            lines.append(f"Lowest variability in best tuning value across runs: **{stable_tuning_algo}** with **{format_reward_millions(stable_tuning_value)} x10^6** standard deviation.")
        lines.append("")
        lines.append("`analysis_tuning_trial_curves.png` shows mean objective value by trial index with one-standard-deviation shading across sets. `analysis_tuning_best_so_far_curves.png` shows how quickly each algorithm improves its incumbent best value as trials accumulate.")
        lines.append("")
        lines.append("`analysis_tuning_best_reward_boxplot.png` compares the best trial achieved in each run, and `analysis_tuning_best_reward_heatmap.png` highlights which algorithms performed best on each set.")
        lines.append("")
        if not tuning_hyper_summary.empty:
            lines.append("The hyperparameter-correlation tables and per-algorithm bar charts summarize which suggested parameters had the strongest monotonic relationship with objective value. These are descriptive diagnostics rather than causal claims.")
            lines.append("")
        lines.append(markdown_table(
            tuning_algo_summary[[
                "algorithm",
                "runs",
                "mean_best_reward",
                "std_best_reward",
                "mean_first_reward",
                "mean_improvement",
                "mean_completed_trials",
                "mean_failed_trials",
            ]],
            formatters={
                "mean_best_reward": format_reward_millions,
                "std_best_reward": format_reward_millions,
                "mean_first_reward": format_reward_millions,
                "mean_improvement": format_reward_millions,
                "mean_completed_trials": lambda v: format_number(v, 1),
                "mean_failed_trials": lambda v: format_number(v, 1),
            },
            max_rows=20,
        ))
    lines.append("")

    lines.append("## Default training versus tuning")
    lines.append("")
    if comparison_df.empty:
        lines.append("_A direct algorithm-level comparison could not be produced because one of the two result families is missing._")
    else:
        lines.append("`analysis_training_vs_tuning_comparison.png` compares each algorithm's mean best default-training reward against its mean best tuning objective value.")
        lines.append("")
        training_steps = sorted({spec.steps for spec in slurm_specs if spec.kind == "training" and spec.steps})
        tuning_steps = sorted({spec.steps for spec in slurm_specs if spec.kind == "tuning" and spec.steps})
        if training_steps and tuning_steps and training_steps != tuning_steps:
            lines.append(
                f"The default-training Slurm scripts request {training_steps} steps, while the tuning Slurm scripts request {tuning_steps} steps per trial. The comparison is therefore useful for direction and scale, but it is not a strict equal-budget benchmark."
            )
            lines.append("")

        uplift_algo, uplift_value = top_algorithm(comparison_df, "tuning_minus_training_mean_best", higher_is_better=True)
        if uplift_algo is not None:
            lines.append(f"Largest tuning-minus-training gap in mean best reward: **{uplift_algo}** at **{format_reward_millions(uplift_value)} x10^6**.")
            lines.append("")

        lines.append(markdown_table(
            comparison_df[[
                "algorithm",
                "mean_best_reward_training",
                "mean_best_reward_tuning",
                "tuning_minus_training_mean_best",
                "best_of_best_reward_training",
                "best_of_best_reward_tuning",
            ]],
            formatters={
                "mean_best_reward_training": format_reward_millions,
                "mean_best_reward_tuning": format_reward_millions,
                "tuning_minus_training_mean_best": format_reward_millions,
                "best_of_best_reward_training": format_reward_millions,
                "best_of_best_reward_tuning": format_reward_millions,
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
    slurm_specs: List[SlurmSpec] = []
    for script_relpath in args.slurm_scripts:
        script_path = (project_root / script_relpath).resolve()
        if not script_path.exists():
            warnings.append(f"Missing Slurm script: {script_path}")
            continue
        try:
            slurm_specs.append(parse_slurm_spec(script_path))
        except Exception as exc:
            warnings.append(f"Failed to parse Slurm script {script_path}: {exc}")

    inventory_df = build_expected_inventory(slurm_specs)

    training_raw_df, training_roots = discover_training_progress(
        project_root=project_root,
        exp_name=args.training_exp_name,
        num_robots=args.num_robots,
        explicit_log_root=args.training_log_root,
    )
    if training_raw_df.empty:
        warnings.append("No usable default training progress.csv files were found.")
    else:
        training_raw_df["reward_scaled"] = training_raw_df["reward"] / 1_000_000.0
        training_raw_df["reward_scaled"] = training_raw_df["reward_scaled"].clip(lower=-2)

    tuning_trial_df, tuning_run_files_df = discover_tuning_trials(
        slurm_out_dir=slurm_out_dir,
        exp_name=args.tuning_exp_name,
        num_robots=args.num_robots,
    )
    if tuning_trial_df.empty:
        warnings.append("No usable Optuna tuning trial rows were found in Slurm output files.")
    else:
        tuning_trial_df["reward_scaled"] = tuning_trial_df["reward"] / 1_000_000.0
        tuning_trial_df["reward_scaled"] = tuning_trial_df["reward_scaled"].clip(lower=-2)

    training_run_summary = build_training_run_summary(training_raw_df)
    training_set_summary = build_training_set_summary(training_run_summary)
    training_algo_summary = build_training_algorithm_summary(training_run_summary)

    if not tuning_trial_df.empty:
        finished_tuning_df = tuning_trial_df[(tuning_trial_df["state"] == "finished") & tuning_trial_df["reward"].notna()].copy()
        if not finished_tuning_df.empty:
            finished_tuning_df["reward_best_so_far"] = finished_tuning_df.groupby("run_name")["reward"].cummax()
            finished_tuning_df["reward_best_so_far_scaled"] = finished_tuning_df["reward_best_so_far"] / 1_000_000.0
            finished_tuning_df["reward_best_so_far_scaled"] = finished_tuning_df["reward_best_so_far_scaled"].clip(lower=-2)
        else:
            finished_tuning_df = pd.DataFrame()
    else:
        finished_tuning_df = pd.DataFrame()

    tuning_run_summary = build_tuning_run_summary(tuning_trial_df)
    tuning_set_summary = build_tuning_set_summary(tuning_run_summary)
    tuning_algo_summary = build_tuning_algorithm_summary(tuning_run_summary)
    tuning_hyper_summary = build_tuning_hyperparameter_summary(tuning_trial_df)
    comparison_df = build_comparison_table(training_algo_summary, tuning_algo_summary)

    discovered_training_names = set(training_run_summary["run_name"]) if not training_run_summary.empty else set()
    discovered_tuning_names = set(tuning_run_summary["run_name"]) if not tuning_run_summary.empty else set()
    discovered_tuning_names.update(set(tuning_run_files_df["run_name"]) if not tuning_run_files_df.empty else set())

    if not inventory_df.empty:
        inventory_df = inventory_df.copy()
        inventory_df["training_found"] = inventory_df["run_name"].isin(discovered_training_names)
        inventory_df["tuning_found"] = inventory_df["run_name"].isin(discovered_tuning_names)
        inventory_df["expected_output_found"] = np.where(
            inventory_df["kind"] == "training",
            inventory_df["training_found"],
            inventory_df["tuning_found"],
        )
        inventory_df["coverage_status"] = np.where(inventory_df["expected_output_found"], "FOUND", "MISSING")
    else:
        inventory_df = pd.DataFrame(columns=["expected_output_found"])

    coverage_summary = coverage_summary_from_inventory(inventory_df) if not inventory_df.empty else pd.DataFrame()

    artifacts: List[Dict[str, str]] = []

    training_raw_path = results_dir / "analysis_training_raw.csv"
    training_run_summary_path = results_dir / "analysis_training_run_summary.csv"
    training_set_summary_path = results_dir / "analysis_training_set_summary.csv"
    training_algo_summary_path = results_dir / "analysis_training_algorithm_summary.csv"
    tuning_trial_raw_path = results_dir / "analysis_tuning_trials_raw.csv"
    tuning_run_summary_path = results_dir / "analysis_tuning_run_summary.csv"
    tuning_set_summary_path = results_dir / "analysis_tuning_set_summary.csv"
    tuning_algo_summary_path = results_dir / "analysis_tuning_algorithm_summary.csv"
    tuning_hyper_summary_path = results_dir / "analysis_tuning_hyperparameter_summary.csv"
    comparison_path = results_dir / "analysis_training_vs_tuning_comparison.csv"
    inventory_path = results_dir / "analysis_run_inventory.csv"
    coverage_summary_path = results_dir / "analysis_run_coverage_summary.csv"
    manifest_path = results_dir / "analysis_artifact_manifest.csv"
    report_path = results_dir / "analysis_report.md"

    dataframe_for_csv(training_raw_df).to_csv(training_raw_path, index=False)
    dataframe_for_csv(training_run_summary).to_csv(training_run_summary_path, index=False)
    dataframe_for_csv(training_set_summary).to_csv(training_set_summary_path, index=False)
    dataframe_for_csv(training_algo_summary).to_csv(training_algo_summary_path, index=False)
    dataframe_for_csv(tuning_trial_df).to_csv(tuning_trial_raw_path, index=False)
    dataframe_for_csv(tuning_run_summary).to_csv(tuning_run_summary_path, index=False)
    dataframe_for_csv(tuning_set_summary).to_csv(tuning_set_summary_path, index=False)
    dataframe_for_csv(tuning_algo_summary).to_csv(tuning_algo_summary_path, index=False)
    dataframe_for_csv(tuning_hyper_summary).to_csv(tuning_hyper_summary_path, index=False)
    dataframe_for_csv(comparison_df).to_csv(comparison_path, index=False)
    dataframe_for_csv(inventory_df).to_csv(inventory_path, index=False)
    dataframe_for_csv(coverage_summary).to_csv(coverage_summary_path, index=False)

    add_artifact(artifacts, "results", "table", training_raw_path, "Raw default-training reward traces loaded from progress.csv files.")
    add_artifact(artifacts, "results", "table", training_run_summary_path, "Per-run default-training summary metrics such as final reward, best reward, and AUC-like average reward.")
    add_artifact(artifacts, "results", "table", training_set_summary_path, "Default-training summary aggregated by algorithm and environment set.")
    add_artifact(artifacts, "results", "table", training_algo_summary_path, "Default-training summary aggregated by algorithm.")
    add_artifact(artifacts, "results", "table", tuning_trial_raw_path, "Raw Optuna trial rows parsed from Slurm output files.")
    add_artifact(artifacts, "results", "table", tuning_run_summary_path, "Per-run Optuna tuning summary including best trial value and best parameters.")
    add_artifact(artifacts, "results", "table", tuning_set_summary_path, "Tuning summary aggregated by algorithm and environment set.")
    add_artifact(artifacts, "results", "table", tuning_algo_summary_path, "Tuning summary aggregated by algorithm.")
    add_artifact(artifacts, "results", "table", tuning_hyper_summary_path, "Spearman-correlation style hyperparameter diagnostics for tuning trials.")
    add_artifact(artifacts, "results", "table", comparison_path, "Algorithm-level comparison between default-training rewards and tuning best values.")
    add_artifact(artifacts, "results", "table", inventory_path, "Expected-run inventory derived from the parsed Slurm scripts.")
    add_artifact(artifacts, "results", "table", coverage_summary_path, "Coverage summary showing discovered versus expected runs by script and algorithm.")

    if not coverage_summary.empty:
        coverage_plot_path = plots_dir / "analysis_run_coverage_heatmap.png"
        plot_coverage_heatmap(coverage_summary, coverage_plot_path)
        add_artifact(artifacts, "plots", "figure", coverage_plot_path, "Coverage heatmap annotated as found/expected for each Slurm-script and algorithm combination.")

    if not training_raw_df.empty:
        common_steps = np.linspace(float(training_raw_df["step"].min()), float(training_raw_df["step"].max()), args.points)
        training_curves = build_algorithm_curves(training_raw_df, "step", "reward_scaled", common_steps)
        training_curve_path = plots_dir / "analysis_training_learning_curves.png"
        plot_mean_std_curves(
            curves=training_curves,
            common_x=common_steps,
            xlabel="Step",
            ylabel=r"Reward (x$10^6$)",
            title="Default training learning curves",
            path=training_curve_path,
        )
        add_artifact(artifacts, "plots", "figure", training_curve_path, "Mean default-training learning curves with one-standard-deviation shading across sets.")

        training_boxplot_path = plots_dir / "analysis_training_final_reward_boxplot.png"
        plot_boxplot(
            run_summary=training_run_summary,
            value_col="final_reward",
            ylabel=r"Final reward (x$10^6$)",
            title="Distribution of final logged rewards for default training",
            path=training_boxplot_path,
        )
        add_artifact(artifacts, "plots", "figure", training_boxplot_path, "Boxplots of each run's final logged reward for the default-training experiments.")

        training_heatmap_path = plots_dir / "analysis_training_best_reward_heatmap.png"
        plot_reward_heatmap(
            summary_df=training_set_summary,
            value_col="mean_best_reward",
            title="Default training best reward by algorithm and set",
            path=training_heatmap_path,
        )
        add_artifact(artifacts, "plots", "figure", training_heatmap_path, "Heatmap of default-training mean best reward by algorithm and environment set.")

    if not finished_tuning_df.empty:
        common_trials = np.linspace(float(finished_tuning_df["trial"].min()), float(finished_tuning_df["trial"].max()), args.points)
        tuning_curves = build_algorithm_curves(finished_tuning_df, "trial", "reward_scaled", common_trials)
        tuning_curve_path = plots_dir / "analysis_tuning_trial_curves.png"
        plot_mean_std_curves(
            curves=tuning_curves,
            common_x=common_trials,
            xlabel="Trial",
            ylabel=r"Reward (x$10^6$)",
            title="Optuna tuning objective by trial",
            path=tuning_curve_path,
        )
        add_artifact(artifacts, "plots", "figure", tuning_curve_path, "Mean Optuna objective value by trial index with one-standard-deviation shading across sets.")

        best_so_far_curves = build_algorithm_curves(finished_tuning_df, "trial", "reward_best_so_far_scaled", common_trials)
        best_so_far_path = plots_dir / "analysis_tuning_best_so_far_curves.png"
        plot_mean_std_curves(
            curves=best_so_far_curves,
            common_x=common_trials,
            xlabel="Trial",
            ylabel=r"Best-so-far reward (x$10^6$)",
            title="Optuna incumbent best value by trial",
            path=best_so_far_path,
        )
        add_artifact(artifacts, "plots", "figure", best_so_far_path, "Mean best-so-far Optuna objective curve by algorithm.")

    if not tuning_run_summary.empty:
        tuning_boxplot_path = plots_dir / "analysis_tuning_best_reward_boxplot.png"
        plot_boxplot(
            run_summary=tuning_run_summary,
            value_col="best_reward",
            ylabel=r"Best trial reward (x$10^6$)",
            title="Distribution of best trial values for tuning runs",
            path=tuning_boxplot_path,
        )
        add_artifact(artifacts, "plots", "figure", tuning_boxplot_path, "Boxplots of the best trial value achieved in each tuning run.")

        tuning_heatmap_path = plots_dir / "analysis_tuning_best_reward_heatmap.png"
        plot_reward_heatmap(
            summary_df=tuning_set_summary,
            value_col="mean_best_reward",
            title="Best tuning value by algorithm and set",
            path=tuning_heatmap_path,
        )
        add_artifact(artifacts, "plots", "figure", tuning_heatmap_path, "Heatmap of mean best tuning value by algorithm and environment set.")

    if not comparison_df.empty:
        comparison_plot_path = plots_dir / "analysis_training_vs_tuning_comparison.png"
        plot_comparison_bars(comparison_df, comparison_plot_path)
        add_artifact(artifacts, "plots", "figure", comparison_plot_path, "Grouped bar chart comparing algorithm-level mean best rewards from default training and tuning.")

    if not tuning_hyper_summary.empty:
        for algorithm in ordered_algorithms(tuning_hyper_summary["algorithm"].unique()):
            algo_plot_path = plots_dir / f"analysis_tuning_hyperparameter_correlation_{algorithm}.png"
            plot_hyperparameter_correlation(tuning_hyper_summary, algorithm, algo_plot_path)
            add_artifact(artifacts, "plots", "figure", algo_plot_path, f"Spearman-correlation summary of hyperparameters versus tuning objective value for {algorithm}.")

    # EDITED:
    add_artifact(artifacts, "results", "report", report_path, "Narrative markdown report explaining the generated tables and figures.")
    # EDITED:
    add_artifact(artifacts, "results", "table", manifest_path, "Machine-readable manifest of every table, figure, and report created by analysis.py.")

    artifacts_df = pd.DataFrame(artifacts)
    build_report(
        report_path=report_path,
        artifacts_df=artifacts_df,
        warnings=warnings,
        slurm_specs=slurm_specs,
        inventory_df=inventory_df,
        coverage_summary=coverage_summary,
        training_roots=training_roots,
        training_run_summary=training_run_summary,
        training_algo_summary=training_algo_summary,
        tuning_run_summary=tuning_run_summary,
        tuning_algo_summary=tuning_algo_summary,
        tuning_hyper_summary=tuning_hyper_summary,
        comparison_df=comparison_df,
    )

    artifacts_df = pd.DataFrame(artifacts)
    artifacts_df.to_csv(manifest_path, index=False)

    print(f"Project root: {project_root}")
    print(f"Output root: {output_root}")
    print(f"Training log roots used: {', '.join(str(path) for path in training_roots) if training_roots else '-'}")
    print(f"Slurm out directory: {slurm_out_dir}")
    print(f"Parsed Slurm scripts: {len(slurm_specs)}")
    print(f"Default training runs summarized: {0 if training_run_summary.empty else training_run_summary.shape[0]}")
    print(f"Tuning runs summarized: {0 if tuning_run_summary.empty else tuning_run_summary.shape[0]}")
    print(f"Saved report to: {report_path}")
    print(f"Saved artifact manifest to: {manifest_path}")


# EDITED:
TRIAL_STATE_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+(?P<state>finished|pruned|failed)\b",
    re.IGNORECASE,
)
VALUE_RE = re.compile(r"value:\s*(?P<value>[-+0-9.eEinfINFnanNAN]+)")
PARAMS_RE = re.compile(r"parameters:\s*(?P<params>\{.*\})")


# EDITED:
def discover_tuning_text_pairs(slurm_out_dir: Path) -> List[Tuple[Path, Optional[Path], str, datetime]]:
    pairs: List[Tuple[Path, Optional[Path], str, datetime]] = []
    if not slurm_out_dir.exists():
        return pairs

    for out_path in sorted(slurm_out_dir.glob("*.out")):
        err_path = out_path.with_suffix(".err")
        out_text = read_text(out_path)
        err_text = read_text(err_path) if err_path.exists() else ""
        combined_text = "\n".join(part for part in [out_text, err_text] if part)

        mtime_candidates = [datetime.fromtimestamp(out_path.stat().st_mtime)]
        if err_path.exists():
            mtime_candidates.append(datetime.fromtimestamp(err_path.stat().st_mtime))
        combined_mtime = max(mtime_candidates)
        pairs.append((out_path, err_path if err_path.exists() else None, combined_text, combined_mtime))

    return pairs


# EDITED:
def has_optuna_trial_lines(text: str) -> bool:
    return TRIAL_STATE_RE.search(text) is not None or USER_TRIAL_RESULT_RE.search(text) is not None


# EDITED:
def parse_trial_rows(text: str, metadata: Dict[str, object], source_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen_keys = set()
    log_mtime = datetime.fromtimestamp(source_path.stat().st_mtime)

    for line in text.splitlines():
        state = None
        trial = None
        reward = np.nan
        params = None

        user_match = USER_TRIAL_RESULT_RE.search(line)
        if user_match is not None:
            state = "finished"
            trial = int(user_match.group("trial"))
            reward = safe_float(user_match.group("value"))
            params = safe_literal_eval(user_match.groupdict().get("params"))
        else:
            trial_state_match = TRIAL_STATE_RE.search(line)
            if trial_state_match is None:
                continue
            state = trial_state_match.group("state").lower()
            trial = int(trial_state_match.group("trial"))
            value_match = VALUE_RE.search(line)
            if state == "finished" and value_match is not None:
                reward = safe_float(value_match.group("value"))
            params_match = PARAMS_RE.search(line)
            if params_match is not None:
                params = safe_literal_eval(params_match.group("params"))

        if trial is None or state is None:
            continue

        key = (trial, state)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        rows.append(
            {
                "run_name": metadata["run_name"],
                "algorithm": metadata["algorithm"],
                "set": int(metadata["set"]),
                "seed": int(metadata["seed"]),
                "device": metadata["device"],
                "trial": trial,
                "reward": reward,
                "state": state,
                "params": params if isinstance(params, dict) else None,
                "params_json": jsonify_object(params if isinstance(params, dict) else params),
                "slurm_out": str(source_path),
                "log_mtime": log_mtime,
            }
        )
    return rows


# EDITED:
def discover_tuning_trials(
    slurm_out_dir: Path,
    exp_name: Optional[str],
    num_robots: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest_by_run_name: Dict[str, Tuple[Path, datetime, str, Dict[str, object]]] = {}

    if not slurm_out_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for out_path, err_path, combined_text, combined_mtime in discover_tuning_text_pairs(slurm_out_dir):
        metadata = parse_tuning_output_metadata(combined_text)
        if metadata is None:
            continue

        run_name = str(metadata["run_name"])
        if exp_name is not None and f"_{exp_name}_{num_robots}_robots_" not in run_name:
            bits = parse_run_name_bits(run_name)
            if bits is None or bits.get("exp_name") != exp_name or bits.get("num_robots") != num_robots:
                continue

        if not has_optuna_trial_lines(combined_text):
            continue

        current = latest_by_run_name.get(run_name)
        if current is None or combined_mtime > current[1]:
            latest_by_run_name[run_name] = (out_path, combined_mtime, combined_text, metadata)

    run_rows: List[Dict[str, object]] = []
    trial_rows: List[Dict[str, object]] = []
    for run_name, (path, mtime, combined_text, metadata) in sorted(latest_by_run_name.items()):
        run_rows.append(
            {
                "run_name": run_name,
                "algorithm": metadata["algorithm"],
                "set": int(metadata["set"]),
                "seed": int(metadata["seed"]),
                "device": metadata["device"],
                "slurm_out": str(path),
                "log_mtime": mtime,
            }
        )
        trial_rows.extend(parse_trial_rows(combined_text, metadata, path))

    trial_df = pd.DataFrame(trial_rows)
    run_df = pd.DataFrame(run_rows)
    if trial_df.empty:
        return trial_df, run_df

    trial_df = trial_df.sort_values(["run_name", "trial", "state", "log_mtime"]).drop_duplicates(
        subset=["run_name", "trial", "state"],
        keep="last",
    )
    trial_df = expand_param_columns(trial_df)
    return trial_df.reset_index(drop=True), run_df.reset_index(drop=True)


# EDITED:
if __name__ == "__main__":
    main()
