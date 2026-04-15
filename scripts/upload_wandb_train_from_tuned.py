#!/usr/bin/env python3
# EDITED:
"""Upload train-from-tuned SB3 logs to Weights & Biases.

This mirrors ``upload_wandb.py`` but defaults to the
``logs/training_from_tuned_logs`` tree and enriches each W&B run with the
provenance of the tuned hyperparameters when that information can be parsed from
Slurm stdout/stderr.
"""

from __future__ import annotations

# EDITED:
import argparse
import ast
import csv
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# EDITED:
try:
    import wandb
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "wandb is not installed. Install it with `pip install wandb`."
    ) from exc


# EDITED:
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run_name>\S+)")
TUNING_RUN_RE = re.compile(
    r"Loaded tuned hyperparameters from (?P<tuning_run_name>\S+):\s*(?P<params>\{.*\})"
)
TUNED_PARAMS_RE = re.compile(r"Tuned hyperparameters used for training:\s*(?P<params>\{.*\})")
TUNED_SOURCES_RE = re.compile(r"Tuned hyperparameter sources:\s*(?P<sources>\{.*\})")
ALGO_SET_RE = re.compile(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)$")
SEED_RE = re.compile(r"(?:^|_)seed(?P<seed>-?\d+)(?:_|$)")
ROBOTS_RE = re.compile(r"(?:^|_)(?P<num_robots>\d+)_robots(?:_|$)")
DEVICE_RE = re.compile(r"(?:^|_)(?P<device>cpu|cuda)(?:_|$)")


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload existing train-from-tuned SB3 progress.csv logs to Weights & Biases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        default=None,
        help="Root directory containing run folders. Defaults to logs/training_from_tuned_logs.",
    )
    parser.add_argument(
        "--slurm_out_dir",
        type=Path,
        default=None,
        help="Optional Slurm output directory used to enrich runs with tuned-parameter provenance.",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project name to upload into",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Optional W&B entity (user or team)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode. Use online to send runs to the browser immediately.",
    )
    parser.add_argument(
        "--include_glob",
        type=str,
        default="progress*.csv",
        help="Glob used to find SB3 CSV log files under log_root",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List the runs that would be uploaded without creating W&B runs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information while uploading",
    )
    return parser.parse_args()


# EDITED:
def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# EDITED:
def resolve_log_root(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (default_repo_root() / "logs" / "training_from_tuned_logs").resolve()


# EDITED:
def resolve_slurm_out_dir(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (default_repo_root() / "slurm_scripts" / "slurm_out").resolve()


# EDITED:
def discover_progress_files(log_root: Path, include_glob: str) -> List[Path]:
    return sorted(path for path in log_root.rglob(include_glob) if path.is_file())


# EDITED:
def infer_run_name(progress_csv: Path, log_root: Path) -> str:
    try:
        rel_parts = progress_csv.relative_to(log_root).parts
    except ValueError:
        return progress_csv.parent.name

    if len(rel_parts) >= 3 and rel_parts[1] == "logs":
        return rel_parts[0]
    return progress_csv.parent.name


# EDITED:
def parse_run_metadata(run_name: str, logger_dir_name: str) -> Dict[str, object]:
    metadata: Dict[str, object] = {"run_name": run_name, "experiment_family": "from_tuned"}

    algo_set_match = ALGO_SET_RE.match(logger_dir_name)
    if algo_set_match:
        metadata["algorithm"] = algo_set_match.group("algorithm")
        metadata["set"] = int(algo_set_match.group("set"))

    seed_match = SEED_RE.search(run_name)
    if seed_match:
        metadata["seed"] = int(seed_match.group("seed"))

    robots_match = ROBOTS_RE.search(run_name)
    if robots_match:
        metadata["num_robots"] = int(robots_match.group("num_robots"))

    device_match = DEVICE_RE.search(run_name)
    if device_match:
        metadata["device"] = device_match.group("device")

    return metadata


# EDITED:
def stable_run_id(progress_csv: Path, log_root: Path) -> str:
    try:
        key = progress_csv.relative_to(log_root).as_posix()
    except ValueError:
        key = str(progress_csv.resolve())
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"sb3-from-tuned-{digest}"


# EDITED:
def parse_float(raw_value: Optional[str]) -> Optional[float]:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if value == "":
        return None

    try:
        parsed = float(value)
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


# EDITED:
def choose_step(metrics: Dict[str, float], fallback_step: int) -> int:
    for key in ("time/total_timesteps", "global_step", "step"):
        if key in metrics:
            return int(metrics[key])
    return fallback_step


# EDITED:
def iter_csv_metrics(progress_csv: Path) -> Iterable[Dict[str, float]]:
    with progress_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics: Dict[str, float] = {}
            for key, raw_value in row.items():
                if not key:
                    continue
                parsed = parse_float(raw_value)
                if parsed is not None:
                    metrics[key] = parsed
            if metrics:
                yield metrics


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
def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


# EDITED:
def discover_latest_slurm_metadata(slurm_out_dir: Path) -> Dict[str, Dict[str, object]]:
    if not slurm_out_dir.exists():
        return {}

    pair_map: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for path in sorted(slurm_out_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".out", ".err"}:
            continue

        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        job_name, job_id, task_text = parts
        task_id = parse_int(task_text)
        if task_id is None:
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
        tuning_run_name = None
        tuned_params = None
        tuned_sources = None

        tuning_match = None
        tuning_matches = list(TUNING_RUN_RE.finditer(combined_text))
        if tuning_matches:
            tuning_match = tuning_matches[-1]
            tuning_run_name = tuning_match.group("tuning_run_name")
            tuned_params = safe_literal_eval(tuning_match.group("params"))

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

        job_id_rank = int(pair["job_id"]) if str(pair["job_id"]).isdigit() else -1
        rank = (job_id_rank, int(pair["mtime"]))
        record = {
            "run_name": run_name,
            "slurm_job_name": pair["job_name"],
            "slurm_job_id": pair["job_id"],
            "slurm_task_id": pair["task_id"],
            "slurm_out_path": str(pair["out_path"]) if pair["out_path"] is not None else None,
            "slurm_err_path": str(pair["err_path"]) if pair["err_path"] is not None else None,
            "tuning_run_name": tuning_run_name,
            "tuned_hyperparameters": tuned_params if isinstance(tuned_params, dict) else {},
            "tuned_hyperparameter_sources": tuned_sources if isinstance(tuned_sources, dict) else {},
        }

        previous = latest_by_run_name.get(run_name)
        if previous is None or rank >= (previous[0], previous[1]):
            latest_by_run_name[run_name] = (rank[0], rank[1], record)

    return {run_name: triple[2] for run_name, triple in latest_by_run_name.items()}


# EDITED:
def upload_run(
    progress_csv: Path,
    log_root: Path,
    args: argparse.Namespace,
    slurm_metadata_by_run_name: Dict[str, Dict[str, object]],
) -> int:
    run_name = infer_run_name(progress_csv, log_root)
    logger_dir_name = progress_csv.parent.name
    metadata = parse_run_metadata(run_name, logger_dir_name)
    slurm_metadata = slurm_metadata_by_run_name.get(run_name, {})

    try:
        rel_path = progress_csv.relative_to(log_root).as_posix()
    except ValueError:
        rel_path = str(progress_csv)

    if args.dry_run:
        tuning_run_name = slurm_metadata.get("tuning_run_name")
        print(
            f"[DRY-RUN] {rel_path} -> project={args.project}, run_name={run_name}, "
            f"tuning_run_name={tuning_run_name or '-'}"
        )
        return 0

    config = dict(metadata)
    config.update(
        {
            "source_progress_csv": str(progress_csv),
            "source_log_root": str(log_root),
        }
    )
    if slurm_metadata:
        config.update(
            {
                "tuning_run_name": slurm_metadata.get("tuning_run_name"),
                "slurm_job_name": slurm_metadata.get("slurm_job_name"),
                "slurm_job_id": slurm_metadata.get("slurm_job_id"),
                "slurm_task_id": slurm_metadata.get("slurm_task_id"),
                "slurm_out_path": slurm_metadata.get("slurm_out_path"),
                "slurm_err_path": slurm_metadata.get("slurm_err_path"),
                "tuned_hyperparameters": slurm_metadata.get("tuned_hyperparameters", {}),
                "tuned_hyperparameter_sources": slurm_metadata.get("tuned_hyperparameter_sources", {}),
            }
        )

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        id=stable_run_id(progress_csv, log_root),
        resume="allow",
        mode=args.mode,
        job_type="historical-sb3-from-tuned-upload",
        config=config,
        reinit=True,
        save_code=False,
    )

    rows_uploaded = 0
    last_existing_step = -1
    last_logged_step = -1
    last_step = None
    last_reward = None
    best_reward = None

    try:
        try:
            summary_step = run.summary.get("_step", -1)
            last_existing_step = int(summary_step) if summary_step is not None else -1
        except Exception:
            last_existing_step = -1

        for row_index, metrics in enumerate(iter_csv_metrics(progress_csv)):
            step = choose_step(metrics, row_index)
            reward = metrics.get("rollout/ep_rew_mean", metrics.get("ep_rew_mean"))
            if reward is not None:
                last_reward = reward
                best_reward = reward if best_reward is None else max(best_reward, reward)
            last_step = step

            if step <= last_existing_step:
                continue

            if step <= last_logged_step:
                step = last_logged_step + 1

            run.log(metrics, step=step)
            last_logged_step = step
            rows_uploaded += 1

        run.summary["progress_rows_total"] = row_index + 1 if 'row_index' in locals() else 0
        if last_step is not None:
            run.summary["last_timesteps"] = int(last_step)
        if last_reward is not None:
            run.summary["final_reward"] = float(last_reward)
        if best_reward is not None:
            run.summary["best_reward"] = float(best_reward)

        if slurm_metadata:
            tuning_run_name = slurm_metadata.get("tuning_run_name")
            if tuning_run_name:
                run.summary["tuning_run_name"] = tuning_run_name
            tuned_params = slurm_metadata.get("tuned_hyperparameters", {})
            if isinstance(tuned_params, dict):
                run.summary["tuned_param_count"] = len(tuned_params)
                for key, value in tuned_params.items():
                    run.summary[f"tuned_params/{key}"] = value
            tuned_sources = slurm_metadata.get("tuned_hyperparameter_sources", {})
            if isinstance(tuned_sources, dict):
                for key, value in tuned_sources.items():
                    run.summary[f"tuned_param_sources/{key}"] = value

        if args.verbose:
            print(
                f"[INFO] Uploaded {rows_uploaded} rows from {rel_path} "
                f"(existing_step={last_existing_step})"
            )
        elif rows_uploaded > 0:
            print(f"[OK] {rel_path} ({rows_uploaded} rows)")
        else:
            print(f"[SKIP] {rel_path} (already up to date)")
    finally:
        run.finish()

    return rows_uploaded


# EDITED:
def main() -> int:
    args = parse_args()
    log_root = resolve_log_root(args.log_root)
    slurm_out_dir = resolve_slurm_out_dir(args.slurm_out_dir)

    if not log_root.exists():
        print(f"Log root does not exist: {log_root}", file=sys.stderr)
        return 1

    progress_files = discover_progress_files(log_root, args.include_glob)
    if not progress_files:
        print(
            f"No files matching '{args.include_glob}' were found under {log_root}",
            file=sys.stderr,
        )
        return 1

    slurm_metadata_by_run_name = discover_latest_slurm_metadata(slurm_out_dir)

    print(f"Found {len(progress_files)} progress file(s) under {log_root}")
    if slurm_out_dir.exists():
        print(f"Parsed {len(slurm_metadata_by_run_name)} train-from-tuned Slurm run(s) under {slurm_out_dir}")
    else:
        print(f"Slurm output directory not found; continuing without provenance enrichment: {slurm_out_dir}")

    total_rows = 0
    for progress_csv in progress_files:
        total_rows += upload_run(progress_csv, log_root, args, slurm_metadata_by_run_name)

    if args.dry_run:
        print("Dry run finished. No W&B runs were created.")
    else:
        print(f"Finished. Uploaded {total_rows} new row(s) to project '{args.project}'.")
    return 0


# EDITED:
if __name__ == "__main__":
    raise SystemExit(main())
