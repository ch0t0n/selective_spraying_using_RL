#!/usr/bin/env python3
# EDITED:
"""Upload transfer-learning Stable-Baselines3 CSV logs to a Weights & Biases project.

This script mirrors scripts/upload_wandb.py, but it extracts transfer-specific
metadata such as the source set and target set from the run name.
"""

from __future__ import annotations

# EDITED:
import argparse
import csv
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# EDITED:
try:
    import wandb
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "wandb is not installed. Install it with `pip install wandb`."
    ) from exc


# EDITED:
TRANSFER_RUN_RE = re.compile(
    r"^(?P<algorithm>[A-Za-z0-9]+)_from_set-?(?P<source_set>\d+)_to_set-?(?P<target_set>\d+)"
    r"_seed-?(?P<seed>-?\d+)(?:_(?P<exp_name>[^_]+)_(?P<num_robots>\d+)_robots_(?P<device>cpu|cuda))?$"
)


# EDITED:
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload existing transfer-learning SB3 progress.csv logs to Weights & Biases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        required=True,
        help="Root directory containing transfer run folders, e.g. logs/training_transfer_logs",
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
    metadata: Dict[str, object] = {
        "run_name": run_name,
        "experiment_type": "transfer",
    }

    transfer_match = TRANSFER_RUN_RE.match(run_name)
    if transfer_match:
        metadata["algorithm"] = transfer_match.group("algorithm")
        metadata["source_set"] = int(transfer_match.group("source_set"))
        metadata["target_set"] = int(transfer_match.group("target_set"))
        metadata["set"] = int(transfer_match.group("target_set"))

        if transfer_match.group("seed") is not None:
            metadata["seed"] = int(transfer_match.group("seed"))
        if transfer_match.group("num_robots") is not None:
            metadata["num_robots"] = int(transfer_match.group("num_robots"))
        if transfer_match.group("device") is not None:
            metadata["device"] = transfer_match.group("device")
        if transfer_match.group("exp_name") is not None:
            metadata["exp_name"] = transfer_match.group("exp_name")

    algo_set_match = re.match(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)$", logger_dir_name)
    if algo_set_match:
        metadata.setdefault("algorithm", algo_set_match.group("algorithm"))
        metadata.setdefault("target_set", int(algo_set_match.group("set")))
        metadata.setdefault("set", int(algo_set_match.group("set")))

    return metadata


# EDITED:
def stable_run_id(progress_csv: Path, log_root: Path) -> str:
    try:
        key = progress_csv.relative_to(log_root).as_posix()
    except ValueError:
        key = str(progress_csv.resolve())
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"transfer-sb3-{digest}"


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
def upload_run(progress_csv: Path, log_root: Path, args: argparse.Namespace) -> int:
    run_name = infer_run_name(progress_csv, log_root)
    logger_dir_name = progress_csv.parent.name
    metadata = parse_run_metadata(run_name, logger_dir_name)

    try:
        rel_path = progress_csv.relative_to(log_root).as_posix()
    except ValueError:
        rel_path = str(progress_csv)

    if args.dry_run:
        print(f"[DRY-RUN] {rel_path} -> project={args.project}, run_name={run_name}")
        return 0

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        id=stable_run_id(progress_csv, log_root),
        resume="allow",
        mode=args.mode,
        job_type="historical-transfer-upload",
        config=metadata,
        reinit=True,
        save_code=False,
    )

    rows_uploaded = 0
    last_existing_step = -1
    last_logged_step = -1

    try:
        try:
            summary_step = run.summary.get("_step", -1)
            last_existing_step = int(summary_step) if summary_step is not None else -1
        except Exception:
            last_existing_step = -1

        for row_index, metrics in enumerate(iter_csv_metrics(progress_csv)):
            step = choose_step(metrics, row_index)
            if step <= last_existing_step:
                continue

            if step <= last_logged_step:
                step = last_logged_step + 1

            run.log(metrics, step=step)
            last_logged_step = step
            rows_uploaded += 1

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
    log_root = args.log_root.expanduser().resolve()

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

    print(f"Found {len(progress_files)} progress file(s) under {log_root}")

    total_rows = 0
    for progress_csv in progress_files:
        total_rows += upload_run(progress_csv, log_root, args)

    if args.dry_run:
        print("Dry run finished. No W&B runs were created.")
    else:
        print(f"Finished. Uploaded {total_rows} new row(s) to project '{args.project}'.")
    return 0


# EDITED:
if __name__ == "__main__":
    raise SystemExit(main())
