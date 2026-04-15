#!/usr/bin/env python3
# EDITED:
"""Upload Optuna tuning history to a Weights & Biases project.

This uploader is for ``tuning.py`` runs that store their useful history in the
Slurm stdout/stderr files rather than SB3 ``progress.csv`` files. It creates one
W&B run per tuning run and logs one W&B step per Optuna trial.
"""

# EDITED:
from __future__ import annotations

# EDITED:
import argparse
import ast
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
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
TRIAL_STATE_RE = re.compile(r"Trial\s+(?P<trial>\d+)\s+(?P<state>finished|pruned|failed)\b", re.IGNORECASE)
VALUE_RE = re.compile(r"value:\s*(?P<value>[-+0-9.eE]+)")
PARAMS_RE = re.compile(r"parameters:\s*(?P<params>\{.*\})")
BEST_VALUE_RE = re.compile(r"Best mean reward:\s*(?P<value>[-+0-9.eE]+)")
BEST_PARAMS_RE = re.compile(r"Best hyperparameters:\s*(?P<params>\{.*\})")
SEED_RE = re.compile(r"(?:^|_)seed(?P<seed>-?\d+)(?:_|$)")
ROBOTS_RE = re.compile(r"(?:^|_)(?P<num_robots>\d+)_robots(?:_|$)")
DEVICE_RE = re.compile(r"(?:^|_)(?P<device>cpu|cuda)(?:_|$)")
ALGO_SET_RE = re.compile(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)")
N_TRIALS_RE = re.compile(r"n_trials=(?P<n_trials>\d+)")


# EDITED:
@dataclass
class TrialRecord:
    trial_number: int
    state: str
    value: Optional[float] = None
    params: Dict[str, object] = field(default_factory=dict)
    raw_line: str = ""


# EDITED:
@dataclass
class SlurmPair:
    job_name: Optional[str]
    job_id: Optional[str]
    task_id: Optional[int]
    out_path: Optional[Path] = None
    err_path: Optional[Path] = None
    out_text: str = ""
    err_text: str = ""


# EDITED:
@dataclass
class ParsedRun:
    run_name: str
    metadata: Dict[str, object]
    trials: List[TrialRecord]
    tuning_started: bool
    tuning_ended: bool
    objective_exception_count: int
    best_value: Optional[float]
    best_params: Dict[str, object]
    expected_trials: Optional[int]
    slurm_pair: SlurmPair
    run_home: Path
    plot_path: Path
    plot_exists: bool


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload historical Optuna tuning results to Weights & Biases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        required=True,
        help="Root directory containing tuning run folders, e.g. logs/training_tuning_logs",
    )
    parser.add_argument(
        "--slurm_out_dir",
        type=Path,
        required=True,
        help="Directory containing Slurm .out/.err files for the tuning jobs",
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
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


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
def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


# EDITED:
def safe_literal_eval(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


# EDITED:
def discover_slurm_pairs(slurm_out_dir: Path) -> List[SlurmPair]:
    pair_map: Dict[Tuple[str, str, int], SlurmPair] = {}

    for path in sorted(slurm_out_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".out", ".err"}:
            continue

        stem = path.stem
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        job_name, job_id, task_text = parts
        task_id = parse_int(task_text)
        if task_id is None:
            continue

        key = (job_name, job_id, task_id)
        pair = pair_map.get(key)
        if pair is None:
            pair = SlurmPair(job_name=job_name, job_id=job_id, task_id=task_id)
            pair_map[key] = pair

        if path.suffix == ".out":
            pair.out_path = path
            pair.out_text = read_text(path)
        else:
            pair.err_path = path
            pair.err_text = read_text(path)

    return list(pair_map.values())


# EDITED:
def infer_run_name(slurm_pair: SlurmPair) -> Optional[str]:
    combined_text = "\n".join(part for part in [slurm_pair.out_text, slurm_pair.err_text] if part)
    match = re.search(r"RUN_NAME=(\S+)", combined_text)
    if match:
        return match.group(1)
    return None


# EDITED:
def infer_run_metadata(run_name: str) -> Dict[str, object]:
    metadata: Dict[str, object] = {"run_name": run_name}

    algo_set_match = ALGO_SET_RE.match(run_name)
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
def parse_trial_records(text: str) -> List[TrialRecord]:
    records: Dict[int, TrialRecord] = {}
    for line in text.splitlines():
        match = TRIAL_STATE_RE.search(line)
        if not match:
            continue

        trial_number = int(match.group("trial"))
        state = match.group("state").lower()
        value_match = VALUE_RE.search(line)
        value = parse_float(value_match.group("value")) if value_match else None

        params: Dict[str, object] = {}
        params_match = PARAMS_RE.search(line)
        if params_match:
            parsed = safe_literal_eval(params_match.group("params"))
            if isinstance(parsed, dict):
                params = parsed

        candidate = TrialRecord(
            trial_number=trial_number,
            state=state,
            value=value,
            params=params,
            raw_line=line.strip(),
        )

        existing = records.get(trial_number)
        if existing is None:
            records[trial_number] = candidate
            continue

        if existing.value is None and candidate.value is not None:
            existing.value = candidate.value
        if not existing.params and candidate.params:
            existing.params = candidate.params
        if len(candidate.raw_line) > len(existing.raw_line):
            existing.raw_line = candidate.raw_line
        existing.state = candidate.state or existing.state

    return sorted(records.values(), key=lambda record: record.trial_number)


# EDITED:
def parse_best_value(text: str) -> Optional[float]:
    matches = list(BEST_VALUE_RE.finditer(text))
    if not matches:
        return None
    return parse_float(matches[-1].group("value"))


# EDITED:
def parse_best_params(text: str) -> Dict[str, object]:
    matches = list(BEST_PARAMS_RE.finditer(text))
    if not matches:
        return {}
    parsed = safe_literal_eval(matches[-1].group("params"))
    return parsed if isinstance(parsed, dict) else {}


# EDITED:
def parse_expected_trials(text: str) -> Optional[int]:
    matches = list(N_TRIALS_RE.finditer(text))
    if not matches:
        return None
    return parse_int(matches[-1].group("n_trials"))


# EDITED:
def stable_run_id(run_name: str) -> str:
    digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:16]
    return f"optuna-{digest}"


# EDITED:
def discover_latest_runs(log_root: Path, slurm_out_dir: Path) -> List[ParsedRun]:
    latest_by_run_name: Dict[str, Tuple[int, int, ParsedRun]] = {}

    for slurm_pair in discover_slurm_pairs(slurm_out_dir):
        run_name = infer_run_name(slurm_pair)
        if not run_name:
            continue

        metadata = infer_run_metadata(run_name)
        combined_text = "\n".join(part for part in [slurm_pair.out_text, slurm_pair.err_text] if part)
        trials = parse_trial_records(combined_text)
        tuning_started = "Tuning started on" in combined_text
        tuning_ended = "Tuning ended on" in combined_text
        objective_exception_count = combined_text.count("Training failed with error:")
        best_value = parse_best_value(combined_text)
        best_params = parse_best_params(combined_text)
        expected_trials = parse_expected_trials(combined_text)

        algorithm = metadata.get("algorithm")
        set_id = metadata.get("set")
        run_home = log_root / run_name
        if algorithm is not None and set_id is not None:
            plot_path = run_home / "outputs" / f"{algorithm}_set{set_id}_optuna_rewards.png"
        else:
            plot_path = run_home / "outputs"
        plot_exists = plot_path.exists()

        parsed_run = ParsedRun(
            run_name=run_name,
            metadata=metadata,
            trials=trials,
            tuning_started=tuning_started,
            tuning_ended=tuning_ended,
            objective_exception_count=objective_exception_count,
            best_value=best_value,
            best_params=best_params,
            expected_trials=expected_trials,
            slurm_pair=slurm_pair,
            run_home=run_home,
            plot_path=plot_path,
            plot_exists=plot_exists,
        )

        job_id_rank = int(slurm_pair.job_id) if slurm_pair.job_id and slurm_pair.job_id.isdigit() else -1
        mtime_candidates = []
        if slurm_pair.out_path and slurm_pair.out_path.exists():
            mtime_candidates.append(int(slurm_pair.out_path.stat().st_mtime))
        if slurm_pair.err_path and slurm_pair.err_path.exists():
            mtime_candidates.append(int(slurm_pair.err_path.stat().st_mtime))
        mtime_rank = max(mtime_candidates) if mtime_candidates else -1

        previous = latest_by_run_name.get(run_name)
        if previous is None or (job_id_rank, mtime_rank) >= (previous[0], previous[1]):
            latest_by_run_name[run_name] = (job_id_rank, mtime_rank, parsed_run)

    return sorted((item[2] for item in latest_by_run_name.values()), key=lambda run: run.run_name)


# EDITED:
def iter_numeric_trial_metrics(trial: TrialRecord) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "trial/is_finished": 1 if trial.state == "finished" else 0,
        "trial/is_pruned": 1 if trial.state == "pruned" else 0,
        "trial/is_failed": 1 if trial.state == "failed" else 0,
    }
    if trial.value is not None:
        metrics["trial/value"] = trial.value

    for key, value in trial.params.items():
        if isinstance(value, bool):
            metrics[f"params/{key}"] = int(value)
        elif isinstance(value, (int, float)):
            metrics[f"params/{key}"] = value

    return metrics


# EDITED:
def upload_run(parsed_run: ParsedRun, args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            f"[DRY-RUN] {parsed_run.run_name} -> project={args.project} "
            f"trials={len(parsed_run.trials)} plot_exists={parsed_run.plot_exists}"
        )
        return 0

    config = dict(parsed_run.metadata)
    config.update(
        {
            "expected_trials": parsed_run.expected_trials,
            "slurm_job_name": parsed_run.slurm_pair.job_name,
            "slurm_job_id": parsed_run.slurm_pair.job_id,
            "slurm_task_id": parsed_run.slurm_pair.task_id,
            "run_home": str(parsed_run.run_home),
            "plot_path": str(parsed_run.plot_path),
        }
    )

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=parsed_run.run_name,
        id=stable_run_id(parsed_run.run_name),
        resume="allow",
        mode=args.mode,
        job_type="historical-optuna-upload",
        config=config,
        reinit=True,
        save_code=False,
    )

    rows_uploaded = 0
    last_existing_step = -1
    best_so_far = None

    try:
        try:
            summary_step = run.summary.get("_step", -1)
            last_existing_step = int(summary_step) if summary_step is not None else -1
        except Exception:
            last_existing_step = -1

        for trial in parsed_run.trials:
            step = int(trial.trial_number)
            if step <= last_existing_step:
                if trial.value is not None:
                    best_so_far = trial.value if best_so_far is None else max(best_so_far, trial.value)
                continue

            metrics = iter_numeric_trial_metrics(trial)
            if trial.value is not None:
                best_so_far = trial.value if best_so_far is None else max(best_so_far, trial.value)
                metrics["trial/best_so_far"] = best_so_far
            run.log(metrics, step=step)
            rows_uploaded += 1

        run.summary["completed_trials"] = len(parsed_run.trials)
        run.summary["finished_trials"] = sum(1 for trial in parsed_run.trials if trial.state == "finished")
        run.summary["pruned_trials"] = sum(1 for trial in parsed_run.trials if trial.state == "pruned")
        run.summary["failed_trials"] = sum(1 for trial in parsed_run.trials if trial.state == "failed")
        run.summary["objective_exception_count"] = parsed_run.objective_exception_count
        run.summary["tuning_started"] = parsed_run.tuning_started
        run.summary["tuning_ended"] = parsed_run.tuning_ended
        run.summary["plot_exists"] = parsed_run.plot_exists
        if parsed_run.expected_trials is not None:
            run.summary["expected_trials"] = parsed_run.expected_trials
        if parsed_run.best_value is not None:
            run.summary["best_value"] = parsed_run.best_value
        for key, value in parsed_run.best_params.items():
            run.summary[f"best_params/{key}"] = value

        if args.verbose:
            print(
                f"[INFO] Uploaded {rows_uploaded} trial row(s) for {parsed_run.run_name} "
                f"(existing_step={last_existing_step})"
            )
        elif rows_uploaded > 0:
            print(f"[OK] {parsed_run.run_name} ({rows_uploaded} trial row(s))")
        else:
            print(f"[SKIP] {parsed_run.run_name} (already up to date)")
    finally:
        run.finish()

    return rows_uploaded


# EDITED:
def main() -> int:
    args = parse_args()
    log_root = args.log_root.expanduser().resolve()
    slurm_out_dir = args.slurm_out_dir.expanduser().resolve()

    if not log_root.exists():
        print(f"Log root does not exist: {log_root}", file=sys.stderr)
        return 1
    if not slurm_out_dir.exists():
        print(f"Slurm output directory does not exist: {slurm_out_dir}", file=sys.stderr)
        return 1

    parsed_runs = discover_latest_runs(log_root, slurm_out_dir)
    if not parsed_runs:
        print(
            f"No tuning runs with parseable RUN_NAME entries were found under {slurm_out_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(parsed_runs)} tuning run(s) using Slurm outputs under {slurm_out_dir}")

    total_rows = 0
    for parsed_run in parsed_runs:
        total_rows += upload_run(parsed_run, args)

    if args.dry_run:
        print("Dry run finished. No W&B runs were created.")
    else:
        print(f"Finished. Uploaded {total_rows} new trial row(s) to project '{args.project}'.")
    return 0


# EDITED:
if __name__ == "__main__":
    raise SystemExit(main())
