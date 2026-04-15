#!/usr/bin/env python3
# EDITED:
"""Check transfer-learning status across Slurm outputs, transfer logs, and source pretrained models.

This mirrors scripts/check_status_training.py, but it also verifies that each
expected transfer run has access to its prerequisite source model from
logs/training_default_logs.
"""

from __future__ import annotations

# EDITED:
import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# EDITED:
REPO_ROOT = Path(__file__).resolve().parents[1]
# EDITED:
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# EDITED:
from check_status_training import (
    ACTIVE_PENDING_STATES,
    ACTIVE_RUNNING_STATES,
    FAILED_STATES,
    SchedulerInfo,
    SlurmFileRecord,
    append_jsonl_record,
    compress_task_ids,
    discover_slurm_files,
    dt_from_epoch,
    format_datetime,
    format_steps,
    group_safe_reruns,
    has_nonempty_err,
    latest_heartbeat,
    load_failed_runs_file,
    normalize_state,
    now_utc,
    parse_arithmetic_assignment,
    parse_progress_csv,
    parse_python_arg,
    parse_run_name_template,
    parse_sbatch_directive,
    parse_scalar_assignment,
    parse_shell_list,
    parse_slurm_array_indices,
    print_compact_runs,
    print_detailed_runs,
    print_status_section,
    print_summary_table,
    query_sacct,
    query_squeue,
    read_text,
    render_shell_template,
    safe_eval_arithmetic,
    scheduler_job_id,
)


# EDITED:
@dataclass
class SlurmScriptSpec:
    path: Path
    job_name: str
    array_spec: str
    array_indices: List[int]
    algorithms: List[str]
    sets: List[int]
    seeds: List[int]
    steps: int
    num_robots: int
    load_set: int
    device: str
    run_name_template: str
    index_expr: str
    algorithm_index_expr: str
    set_index_expr: str
    seed_index_expr: str


# EDITED:
@dataclass
class RunSpec:
    slurm_script: Path
    job_name: str
    task_id: int
    algorithm: str
    set_id: int
    seed: int
    device: str
    steps: int
    num_robots: int
    load_set: int
    source_run_name: str
    run_name: str


# EDITED:
@dataclass
class ProgressInfo:
    path: Optional[Path] = None
    last_timesteps: Optional[int] = None
    row_count: int = 0
    mtime: Optional[datetime] = None
    header: List[str] = field(default_factory=list)
    parse_error: Optional[str] = None


# EDITED:
@dataclass
class RunArtifacts:
    run_home: Path
    progress: ProgressInfo
    weights_path: Path
    weights_exists: bool
    source_run_home: Path
    source_weights_path: Path
    source_weights_exists: bool
    output_logs: List[Path]
    output_log_mtime: Optional[datetime]


# EDITED:
@dataclass
class StatusRecord:
    run_spec: RunSpec
    slurm_files: Optional[SlurmFileRecord]
    artifacts: RunArtifacts
    scheduler: SchedulerInfo
    status: str
    safe_to_rerun: bool
    expected_steps: int
    actual_steps: Optional[int]
    steps_verified: bool
    last_heartbeat: Optional[datetime]
    heartbeat_age_hours: Optional[float]
    failure_reason: Optional[str]
    notes: List[str] = field(default_factory=list)


# EDITED:
def build_source_run_name(algorithm: str, load_set: int, seed: int, num_robots: int, device: str) -> str:
    return f"{algorithm}_set{load_set}_seed{seed}_default_{num_robots}_robots_{device}"


# EDITED:
def parse_slurm_script(path: Path) -> SlurmScriptSpec:
    text = read_text(path)

    job_name = parse_sbatch_directive(text, "job-name")
    array_spec = parse_sbatch_directive(text, "array")
    algorithms = parse_shell_list(text, "algorithms")
    sets = [int(value) for value in parse_shell_list(text, "sets")]
    seeds_raw = parse_shell_list(text, "seed") or parse_shell_list(text, "seeds")
    seeds = [int(value) for value in seeds_raw]
    steps = int(parse_scalar_assignment(text, "steps") or 0)
    num_robots = int(parse_scalar_assignment(text, "num_robots") or 0)
    load_set = int(parse_scalar_assignment(text, "load_set") or 0)
    device = parse_python_arg(text, "device") or "cpu"
    run_name_template = parse_run_name_template(text) or ""
    index_expr = parse_arithmetic_assignment(text, "index") or "SLURM_ARRAY_TASK_ID"
    algorithm_index_expr = parse_arithmetic_assignment(text, "algorithm_index") or "0"
    set_index_expr = parse_arithmetic_assignment(text, "set_index") or "0"
    seed_index_expr = parse_arithmetic_assignment(text, "seed_index") or "0"

    missing = []
    if not job_name:
        missing.append("job_name")
    if not array_spec:
        missing.append("array_spec")
    if not algorithms:
        missing.append("algorithms")
    if not sets:
        missing.append("sets")
    if not seeds:
        missing.append("seeds")
    if not steps:
        missing.append("steps")
    if not num_robots:
        missing.append("num_robots")
    if not load_set:
        missing.append("load_set")
    if not run_name_template:
        missing.append("run_name_template")

    if missing:
        raise ValueError(f"Unable to parse {path}: missing {', '.join(missing)}")

    return SlurmScriptSpec(
        path=path,
        job_name=job_name,
        array_spec=array_spec,
        array_indices=parse_slurm_array_indices(array_spec),
        algorithms=algorithms,
        sets=sets,
        seeds=seeds,
        steps=steps,
        num_robots=num_robots,
        load_set=load_set,
        device=device,
        run_name_template=run_name_template,
        index_expr=index_expr,
        algorithm_index_expr=algorithm_index_expr,
        set_index_expr=set_index_expr,
        seed_index_expr=seed_index_expr,
    )


# EDITED:
def expand_expected_runs(script_spec: SlurmScriptSpec) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for task_id in script_spec.array_indices:
        variables = {
            "SLURM_ARRAY_TASK_ID": task_id,
            "num_algorithms": len(script_spec.algorithms),
            "num_sets": len(script_spec.sets),
            "num_seeds": len(script_spec.seeds),
        }
        index_value = safe_eval_arithmetic(script_spec.index_expr, variables)
        variables["index"] = index_value
        algorithm_index = safe_eval_arithmetic(script_spec.algorithm_index_expr, variables)
        set_index = safe_eval_arithmetic(script_spec.set_index_expr, variables)
        seed_index = safe_eval_arithmetic(script_spec.seed_index_expr, variables)

        algorithm = script_spec.algorithms[algorithm_index]
        set_id = script_spec.sets[set_index]
        seed = script_spec.seeds[seed_index]

        render_vars = {
            "algorithm": algorithm,
            "load_set": script_spec.load_set,
            "set": set_id,
            "seed": seed,
        }
        run_name = render_shell_template(script_spec.run_name_template, render_vars)
        source_run_name = build_source_run_name(
            algorithm,
            script_spec.load_set,
            seed,
            script_spec.num_robots,
            script_spec.device,
        )

        runs.append(
            RunSpec(
                slurm_script=script_spec.path,
                job_name=script_spec.job_name,
                task_id=task_id,
                algorithm=algorithm,
                set_id=set_id,
                seed=seed,
                device=script_spec.device,
                steps=script_spec.steps,
                num_robots=script_spec.num_robots,
                load_set=script_spec.load_set,
                source_run_name=source_run_name,
                run_name=run_name,
            )
        )
    return runs


# EDITED:
def discover_run_artifacts(logs_dir: Path, run_spec: RunSpec) -> RunArtifacts:
    run_home = logs_dir / run_spec.run_name
    progress_files = sorted(run_home.glob("logs/**/progress.csv"))
    progress_info = ProgressInfo()
    if progress_files:
        latest_progress = max(progress_files, key=lambda candidate: candidate.stat().st_mtime)
        progress_info = parse_progress_csv(latest_progress)

    weights_path = run_home / "weights" / f"env{run_spec.set_id}_{run_spec.algorithm}.zip"

    source_run_home = REPO_ROOT / "logs" / "training_default_logs" / run_spec.source_run_name
    source_weights_path = source_run_home / "weights" / f"env{run_spec.load_set}_{run_spec.algorithm}.zip"

    output_logs = sorted(run_home.glob("outputs/*.log"))
    output_log_mtime = None
    if output_logs:
        latest_output_log = max(output_logs, key=lambda candidate: candidate.stat().st_mtime)
        output_log_mtime = dt_from_epoch(latest_output_log.stat().st_mtime)

    return RunArtifacts(
        run_home=run_home,
        progress=progress_info,
        weights_path=weights_path,
        weights_exists=weights_path.exists(),
        source_run_home=source_run_home,
        source_weights_path=source_weights_path,
        source_weights_exists=source_weights_path.exists(),
        output_logs=output_logs,
        output_log_mtime=output_log_mtime,
    )


# EDITED:
def classify_run(
    run_spec: RunSpec,
    slurm_files: Optional[SlurmFileRecord],
    artifacts: RunArtifacts,
    scheduler: SchedulerInfo,
    stale_hours: float,
    failed_runs_index: Dict[str, List[dict]],
) -> StatusRecord:
    actual_steps = artifacts.progress.last_timesteps
    expected_steps = run_spec.steps
    steps_verified = actual_steps is not None and actual_steps >= expected_steps
    last_heartbeat = latest_heartbeat(
        artifacts.progress.mtime,
        artifacts.output_log_mtime,
        slurm_files.out_mtime if slurm_files else None,
        slurm_files.err_mtime if slurm_files else None,
    )
    heartbeat_age_hours = None
    if last_heartbeat is not None:
        heartbeat_age_hours = (now_utc() - last_heartbeat).total_seconds() / 3600.0

    has_training_output = slurm_files is not None and (slurm_files.training_started or slurm_files.out_path is not None)
    has_any_artifact = (
        has_training_output
        or artifacts.progress.path is not None
        or artifacts.weights_exists
        or bool(artifacts.output_logs)
    )

    notes: List[str] = []
    failure_reason = None
    status = "UNKNOWN"
    safe_to_rerun = False

    existing_failed_entries = failed_runs_index.get(run_spec.run_name, [])

    if artifacts.weights_exists and steps_verified:
        status = "COMPLETE_VERIFIED"
        if slurm_files is not None and not slurm_files.training_ended:
            notes.append("Weights and timesteps verify completion, but the explicit training-end marker was not found.")
    elif artifacts.weights_exists and actual_steps is None:
        status = "COMPLETE_UNVERIFIED"
        notes.append("Weights file exists, but parseable progress timesteps were not found.")
    elif scheduler.active_state in ACTIVE_PENDING_STATES:
        status = "ACTIVE_PENDING"
        notes.append(f"squeue reports {scheduler.active_state}.")
    elif scheduler.active_state in ACTIVE_RUNNING_STATES:
        status = "ACTIVE_STAGED"
        if heartbeat_age_hours is not None and heartbeat_age_hours > stale_hours:
            status = "ACTIVE_STALE"
            notes.append(f"Active in Slurm but no artifact heartbeat for {heartbeat_age_hours:.2f}h.")
        else:
            notes.append(f"squeue reports {scheduler.active_state}.")
    else:
        sacct_failed = scheduler.sacct_state in FAILED_STATES
        explicit_failure = False

        if actual_steps is not None and actual_steps >= expected_steps and not artifacts.weights_exists:
            explicit_failure = True
            failure_reason = "Expected timesteps were reached, but the weights file is missing."
        elif slurm_files and slurm_files.traceback_detected:
            explicit_failure = True
            failure_reason = "Traceback detected in Slurm output."
        elif sacct_failed:
            explicit_failure = True
            failure_reason = f"sacct reports {scheduler.sacct_state}."
        elif slurm_files and slurm_files.training_ended and not artifacts.weights_exists:
            explicit_failure = True
            failure_reason = "Training end marker found, but the weights file is missing."
        elif existing_failed_entries:
            explicit_failure = True
            failure_reason = "Run already present in failed_transfer_runs.jsonl."
        elif has_nonempty_err(slurm_files) and (slurm_files.error_lines or "Traceback" in slurm_files.err_text):
            explicit_failure = True
            failure_reason = "stderr contains error output."

        if explicit_failure:
            status = "FAILED"
            safe_to_rerun = True
        else:
            if has_any_artifact and actual_steps is not None and actual_steps < expected_steps:
                if heartbeat_age_hours is not None and heartbeat_age_hours > stale_hours:
                    status = "STALE_INACTIVE"
                    failure_reason = f"Run is incomplete ({actual_steps}/{expected_steps}) and stale."
                    safe_to_rerun = True
                else:
                    status = "INCOMPLETE_RECENT"
                    notes.append("Run is incomplete and not active, but the latest heartbeat is still recent.")
            elif has_any_artifact:
                if heartbeat_age_hours is not None and heartbeat_age_hours > stale_hours:
                    status = "STALE_INACTIVE"
                    failure_reason = "Run has artifacts but is incomplete and stale."
                    safe_to_rerun = True
                else:
                    status = "INCOMPLETE_RECENT"
                    notes.append("Run has partial artifacts but no definitive completion or failure marker.")
            else:
                status = "MISSING"
                safe_to_rerun = True
                notes.append("No Slurm outputs, progress.csv, or weights were found for this transfer run.")

    if status == "ACTIVE_STAGED":
        status = "ACTIVE_RUNNING"

    if status == "COMPLETE_UNVERIFIED":
        safe_to_rerun = False

    # Source-model prerequisite adjustment
    # EDITED:
    if not artifacts.source_weights_exists:
        if status in {"COMPLETE_VERIFIED", "COMPLETE_UNVERIFIED"}:
            notes.append(
                "The transfer output is complete, but the original source pretrained model is now missing."
            )
        elif status in {"ACTIVE_RUNNING", "ACTIVE_PENDING", "ACTIVE_STALE"}:
            notes.append(
                "The prerequisite source pretrained model is currently missing, "
                "but the transfer job is already active."
            )
            safe_to_rerun = False
        else:
            status = "BLOCKED_SOURCE_MISSING"
            safe_to_rerun = False
            failure_reason = (
                "Required source pretrained model is missing. "
                f"Expected: {artifacts.source_weights_path}"
            )
            notes.append(
                "Launch or recover the corresponding default-training source run before rerunning transfer."
            )

    return StatusRecord(
        run_spec=run_spec,
        slurm_files=slurm_files,
        artifacts=artifacts,
        scheduler=scheduler,
        status=status,
        safe_to_rerun=safe_to_rerun,
        expected_steps=expected_steps,
        actual_steps=actual_steps,
        steps_verified=steps_verified,
        last_heartbeat=last_heartbeat,
        heartbeat_age_hours=heartbeat_age_hours,
        failure_reason=failure_reason,
        notes=notes,
    )


# EDITED:
def build_failed_run_record(status_record: StatusRecord) -> dict:
    slurm_files = status_record.slurm_files
    return {
        "status": "failed",
        "detection_source": "check_status_transfer",
        "detected_at": now_utc().isoformat(),
        "classification": status_record.status,
        "reason": status_record.failure_reason,
        "run_name": status_record.run_spec.run_name,
        "algorithm": status_record.run_spec.algorithm,
        "load_set": status_record.run_spec.load_set,
        "set": status_record.run_spec.set_id,
        "seed": status_record.run_spec.seed,
        "device": status_record.run_spec.device,
        "num_robots": status_record.run_spec.num_robots,
        "source_run_name": status_record.run_spec.source_run_name,
        "expected_steps": status_record.expected_steps,
        "actual_steps": status_record.actual_steps,
        "slurm_script": str(status_record.run_spec.slurm_script),
        "task_id": status_record.run_spec.task_id,
        "slurm_job_name": status_record.run_spec.job_name,
        "slurm_job_id": scheduler_job_id(status_record),
        "slurm_array_task_id": status_record.run_spec.task_id,
        "sacct_state": status_record.scheduler.sacct_state,
        "sacct_exit_code": status_record.scheduler.sacct_exit_code,
        "active_state": status_record.scheduler.active_state,
        "progress_csv": str(status_record.artifacts.progress.path) if status_record.artifacts.progress.path else None,
        "weights_path": str(status_record.artifacts.weights_path),
        "source_weights_path": str(status_record.artifacts.source_weights_path),
        "slurm_out_path": str(slurm_files.out_path) if slurm_files and slurm_files.out_path else None,
        "slurm_err_path": str(slurm_files.err_path) if slurm_files and slurm_files.err_path else None,
        "error_lines": slurm_files.error_lines if slurm_files else [],
    }


# EDITED:
def print_expected_overview(
    script_specs: List[SlurmScriptSpec],
    expected_runs: List[RunSpec],
    args: argparse.Namespace,
    squeue_available: bool,
    sacct_available: bool,
) -> None:
    print("=" * 100)
    print("TRANSFER STATUS CHECK")
    print("=" * 100)
    print(f"Timestamp                 : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Repository root           : {args.repo_root}")
    print(f"Logs directory            : {args.logs_dir}")
    print(f"Slurm output directory    : {args.slurm_out_dir}")
    print(f"failed_transfer_runs.jsonl: {args.failed_jsonl_path}")
    print(f"Slurm scripts parsed      : {len(script_specs)}")
    print(f"Expected runs             : {len(expected_runs)}")
    if args.job_ids:
        print(f"Job IDs filter            : {', '.join(args.job_ids)}")
    print(f"Stale threshold (hours)   : {args.stale_hours}")
    print("=" * 100)

    print_status_section("Scheduler command availability")
    print(f"squeue available          : {'yes' if squeue_available else 'no'}")
    print(f"sacct available           : {'yes' if sacct_available else 'no'}")
    if not squeue_available:
        print("Note: active/pending detection will fall back to file heuristics only.")
    if not sacct_available:
        print("Note: completed/failed Slurm accounting states will not be available.")

    print_status_section("Parsed transfer Slurm scripts")
    for spec in script_specs:
        print(
            f"{spec.path}: job_name={spec.job_name}, array={spec.array_spec}, "
            f"algorithms={spec.algorithms}, load_set={spec.load_set}, "
            f"target_sets={spec.sets}, seeds={spec.seeds}, "
            f"steps={spec.steps}, num_robots={spec.num_robots}, device={spec.device}"
        )


# EDITED:
def print_source_model_overview(expected_runs: List[RunSpec]) -> None:
    unique_sources: Dict[str, Path] = {}
    for run_spec in expected_runs:
        source_weights_path = (
            REPO_ROOT
            / "logs"
            / "training_default_logs"
            / run_spec.source_run_name
            / "weights"
            / f"env{run_spec.load_set}_{run_spec.algorithm}.zip"
        )
        unique_sources[run_spec.source_run_name] = source_weights_path

    present = []
    missing = []
    for source_run_name, path in sorted(unique_sources.items()):
        if path.exists():
            present.append((source_run_name, path))
        else:
            missing.append((source_run_name, path))

    print_status_section("Source pretrained-model prerequisite overview")
    print(f"required_sources={len(unique_sources)} present={len(present)} missing={len(missing)}")
    if missing:
        print("Missing source models:")
        for source_run_name, path in missing:
            print(f"  - {source_run_name} -> {path}")


# EDITED:
def print_concise_summary_transfer(records: List[StatusRecord], failed_appends: int, job_ids_filter: Optional[set]) -> None:
    counts = Counter(record.status for record in records)
    safe_reruns = [record for record in records if record.safe_to_rerun]
    print_status_section("Concise summary")
    print(
        "expected={expected} complete_verified={complete} active_running={active_running} "
        "active_pending={active_pending} active_stale={active_stale} failed={failed} "
        "blocked_source_missing={blocked} stale_inactive={stale} missing={missing} "
        "complete_unverified={unverified} incomplete_recent={incomplete} "
        "safe_reruns={safe_reruns_count} failed_jsonl_appends={failed_appends}".format(
            expected=len(records),
            complete=counts.get("COMPLETE_VERIFIED", 0),
            active_running=counts.get("ACTIVE_RUNNING", 0),
            active_pending=counts.get("ACTIVE_PENDING", 0),
            active_stale=counts.get("ACTIVE_STALE", 0),
            failed=counts.get("FAILED", 0),
            blocked=counts.get("BLOCKED_SOURCE_MISSING", 0),
            stale=counts.get("STALE_INACTIVE", 0),
            missing=counts.get("MISSING", 0),
            unverified=counts.get("COMPLETE_UNVERIFIED", 0),
            incomplete=counts.get("INCOMPLETE_RECENT", 0),
            safe_reruns_count=len(safe_reruns),
            failed_appends=failed_appends,
        )
    )
    if job_ids_filter:
        print(f"job_ids_filter={','.join(sorted(job_ids_filter))}")


# EDITED:
def main() -> int:
    parser = argparse.ArgumentParser(description="Check transfer-learning status across Slurm outputs and transfer logs.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument(
        "--slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/transfer_all_3_robots.sh",
            "slurm_scripts/crossq_transfer_3_robots.sh",
        ],
        help="Transfer Slurm experiment scripts to parse.",
    )
    parser.add_argument("--logs-dir", type=Path, default=None, help="Override logs/training_transfer_logs.")
    parser.add_argument("--slurm-out-dir", type=Path, default=None, help="Override slurm_scripts/slurm_out.")
    parser.add_argument("--job-ids", nargs="*", default=[], help="Optional Slurm array job IDs to focus on.")
    parser.add_argument("--stale-hours", type=float, default=6.0, help="Heartbeat age above which a non-complete run is stale.")
    parser.add_argument("--failed-jsonl-name", default="failed_transfer_runs.jsonl", help="Root-level JSONL file for failed transfer runs.")
    parser.add_argument("--no-record-failed-jsonl", action="store_true", help="Do not append newly identified failed transfer runs to failed_transfer_runs.jsonl.")
    parser.add_argument("--nonzero-on-problems", action="store_true", help="Exit with code 1 when failed/stale/missing/unverified/blocked runs exist.")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.logs_dir = (args.logs_dir or (args.repo_root / "logs" / "training_transfer_logs")).resolve()
    args.slurm_out_dir = (args.slurm_out_dir or (args.repo_root / "slurm_scripts" / "slurm_out")).resolve()
    args.failed_jsonl_path = (args.repo_root / args.failed_jsonl_name).resolve()

    job_ids_filter = set(args.job_ids) if args.job_ids else None

    script_specs: List[SlurmScriptSpec] = []
    expected_runs: List[RunSpec] = []
    for script_relpath in args.slurm_scripts:
        script_path = (args.repo_root / script_relpath).resolve()
        spec = parse_slurm_script(script_path)
        script_specs.append(spec)
        expected_runs.extend(expand_expected_runs(spec))

    squeue_available = shutil.which("squeue") is not None
    sacct_available = shutil.which("sacct") is not None

    print_expected_overview(script_specs, expected_runs, args, squeue_available, sacct_available)
    print_source_model_overview(expected_runs)

    slurm_files_by_run_name = discover_slurm_files(args.slurm_out_dir, job_ids_filter)
    squeue_records = query_squeue(job_ids_filter)
    sacct_records = query_sacct(job_ids_filter)
    existing_failed_records = load_failed_runs_file(args.failed_jsonl_path)
    failed_runs_index: Dict[str, List[dict]] = defaultdict(list)
    for record in existing_failed_records:
        run_name = record.get("run_name")
        if run_name:
            failed_runs_index[str(run_name)].append(record)

    status_records: List[StatusRecord] = []
    for run_spec in expected_runs:
        artifacts = discover_run_artifacts(args.logs_dir, run_spec)
        slurm_candidates = slurm_files_by_run_name.get(run_spec.run_name, [])
        slurm_files = slurm_candidates[0] if slurm_candidates else None

        active_record = squeue_records.get((run_spec.job_name, run_spec.task_id))
        sacct_record = sacct_records.get((run_spec.job_name, run_spec.task_id))
        scheduler = SchedulerInfo(
            active_state=normalize_state(active_record["state"]) if active_record else None,
            active_job_id=active_record["job_id"] if active_record else None,
            sacct_state=normalize_state(sacct_record["state"]) if sacct_record else None,
            sacct_exit_code=sacct_record["exit_code"] if sacct_record else None,
            sacct_elapsed=sacct_record["elapsed"] if sacct_record else None,
        )

        status_record = classify_run(
            run_spec=run_spec,
            slurm_files=slurm_files,
            artifacts=artifacts,
            scheduler=scheduler,
            stale_hours=args.stale_hours,
            failed_runs_index=failed_runs_index,
        )
        status_records.append(status_record)

    print_summary_table(status_records)

    failed_records = [record for record in status_records if record.status == "FAILED"]
    blocked_records = [record for record in status_records if record.status == "BLOCKED_SOURCE_MISSING"]
    stale_records = [record for record in status_records if record.status == "STALE_INACTIVE"]
    active_stale_records = [record for record in status_records if record.status == "ACTIVE_STALE"]
    incomplete_recent_records = [record for record in status_records if record.status == "INCOMPLETE_RECENT"]
    missing_records = [record for record in status_records if record.status == "MISSING"]
    unverified_records = [record for record in status_records if record.status == "COMPLETE_UNVERIFIED"]
    active_running_records = [record for record in status_records if record.status == "ACTIVE_RUNNING"]
    active_pending_records = [record for record in status_records if record.status == "ACTIVE_PENDING"]

    print_detailed_runs("Blocked runs because the source pretrained model is missing", blocked_records)
    print_detailed_runs("Failed runs", failed_records)
    print_detailed_runs("Stale inactive runs", stale_records)
    print_detailed_runs("Active stale runs", active_stale_records)
    print_detailed_runs("Completed but not fully verified runs", unverified_records)
    print_compact_runs("Incomplete recent runs", incomplete_recent_records)
    print_compact_runs("Missing runs", missing_records)
    print_compact_runs("Active running runs", active_running_records)
    print_compact_runs("Active pending runs", active_pending_records)

    failed_appends = 0
    if not args.no_record_failed_jsonl:
        for record in failed_records:
            appended = append_jsonl_record(
                args.failed_jsonl_path,
                build_failed_run_record(record),
                dedupe_keys=["detection_source", "run_name", "slurm_job_id", "slurm_array_task_id", "reason"],
            )
            failed_appends += int(appended)

    rerun_candidates = [record for record in status_records if record.safe_to_rerun]
    grouped_reruns = group_safe_reruns(rerun_candidates)

    print_status_section("Safe rerun commands")
    if grouped_reruns:
        for script_path, task_ids in grouped_reruns.items():
            task_spec = compress_task_ids(task_ids)
            try:
                display_path = script_path.relative_to(args.repo_root)
            except ValueError:
                display_path = script_path
            print(f"sbatch --array={task_spec} {display_path}")
    else:
        print("No rerun commands are currently safe/ready to run.")

    print_concise_summary_transfer(status_records, failed_appends, job_ids_filter)

    problems = (
        failed_records
        or blocked_records
        or stale_records
        or active_stale_records
        or missing_records
        or unverified_records
    )
    if args.nonzero_on_problems and problems:
        return 1
    return 0


# EDITED:
if __name__ == "__main__":
    sys.exit(main())
