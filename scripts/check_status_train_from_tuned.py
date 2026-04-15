#!/usr/bin/env python3
# EDITED:
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
ACTIVE_PENDING_STATES = {"PENDING", "CONFIGURING", "SUSPENDED"}
ACTIVE_RUNNING_STATES = {"RUNNING", "COMPLETING", "STAGE_OUT", "RESIZING"}


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
    device: str
    run_name_template: str
    index_expr: str
    algorithm_index_expr: str
    set_index_expr: str
    seed_index_expr: str


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
    run_name: str


@dataclass
class SlurmFileRecord:
    job_name: Optional[str] = None
    job_id: Optional[str] = None
    task_id: Optional[int] = None
    out_path: Optional[Path] = None
    err_path: Optional[Path] = None
    out_text: str = ""
    err_text: str = ""
    out_mtime: Optional[datetime] = None
    err_mtime: Optional[datetime] = None
    parsed_run_name: Optional[str] = None
    parsed_algorithm: Optional[str] = None
    parsed_set: Optional[int] = None
    parsed_seed: Optional[int] = None
    parsed_device: Optional[str] = None
    training_started: bool = False
    training_ended: bool = False
    traceback_detected: bool = False
    error_lines: List[str] = field(default_factory=list)


@dataclass
class ProgressInfo:
    path: Optional[Path] = None
    last_timesteps: Optional[int] = None
    row_count: int = 0
    mtime: Optional[datetime] = None
    header: List[str] = field(default_factory=list)
    parse_error: Optional[str] = None


@dataclass
class RunArtifacts:
    run_home: Path
    progress: ProgressInfo
    weights_path: Path
    weights_exists: bool
    output_logs: List[Path]
    output_log_mtime: Optional[datetime]


@dataclass
class SchedulerInfo:
    active_state: Optional[str] = None
    active_job_id: Optional[str] = None
    sacct_state: Optional[str] = None
    sacct_exit_code: Optional[str] = None
    sacct_elapsed: Optional[str] = None


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


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def dt_from_epoch(timestamp: Optional[float]) -> Optional[datetime]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


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


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def append_jsonl_record(path: Path, record: dict, dedupe_keys: Optional[List[str]] = None) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json_ready(record)

    if dedupe_keys and path.exists():
        target_signature = tuple(normalized.get(key) for key in dedupe_keys)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_signature = tuple(existing.get(key) for key in dedupe_keys)
                if existing_signature == target_signature:
                    return False

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, sort_keys=True) + "\n")
    return True


def normalize_state(raw_state: Optional[str]) -> Optional[str]:
    if raw_state is None:
        return None
    state = raw_state.strip()
    if not state:
        return None
    return state.split()[0].rstrip("+")


def tail_lines(text: str, max_lines: int = 5) -> List[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def find_first_line_matching(text: str, patterns: Iterable[str]) -> Optional[str]:
    for line in text.splitlines():
        for pattern in patterns:
            if pattern in line:
                return line.strip()
    return None


def parse_slurm_array_indices(array_spec: str) -> List[int]:
    spec = array_spec.strip()
    if "%" in spec:
        spec = spec.split("%", 1)[0]

    indices: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if start <= end else -1
            indices.extend(list(range(start, end + step, step)))
        else:
            indices.append(int(chunk))
    return sorted(set(indices))


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


def parse_scalar_assignment(text: str, variable_name: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(variable_name)}=(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def parse_arithmetic_assignment(text: str, variable_name: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(variable_name)}=\$\(\((.+?)\)\)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def parse_run_name_template(text: str) -> Optional[str]:
    pattern = re.compile(r'^RUN_NAME="(.+)"$', re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


def parse_sbatch_directive(text: str, directive_name: str) -> Optional[str]:
    pattern = re.compile(rf"^#SBATCH\s+--{re.escape(directive_name)}(?:=(.+)|\s+(.+))$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip()


def parse_python_arg(text: str, arg_name: str) -> Optional[str]:
    pattern = re.compile(rf"--{re.escape(arg_name)}\s+([^\s\\]+)")
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def safe_eval_arithmetic(expr: str, variables: Dict[str, int]) -> int:
    sanitized = expr.replace("/", "//")
    if not re.fullmatch(r"[\w\s()+\-*%/]+", sanitized):
        raise ValueError(f"Unsupported arithmetic expression: {expr}")
    return int(eval(sanitized, {"__builtins__": {}}, dict(variables)))


def render_shell_template(template: str, variables: Dict[str, object]) -> str:
    def replace_braced(match: re.Match) -> str:
        key = match.group(1)
        return str(variables[key])

    rendered = re.sub(r"\$\{(\w+)\}", replace_braced, template)
    rendered = re.sub(r"\$(\w+)", replace_braced, rendered)
    return rendered


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
        device=device,
        run_name_template=run_name_template,
        index_expr=index_expr,
        algorithm_index_expr=algorithm_index_expr,
        set_index_expr=set_index_expr,
        seed_index_expr=seed_index_expr,
    )


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
            "set": set_id,
            "seed": seed,
        }
        run_name = render_shell_template(script_spec.run_name_template, render_vars)

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
                run_name=run_name,
            )
        )
    return runs


def parse_slurm_file_metadata(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None, None, None
    job_name, job_id, task_text = parts
    return job_name, job_id, parse_int(task_text)


def parse_slurm_text(out_text: str, err_text: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[int], Optional[int], Optional[str], bool, bool, bool, List[str]]:
    job_id = None
    task_id = None
    run_name = None
    algorithm = None
    set_id = None
    seed = None
    device = None
    training_started = "Training started on" in out_text
    training_ended = "Training ended on" in out_text
    traceback_detected = "Traceback (most recent call last):" in out_text or "Traceback (most recent call last):" in err_text
    error_lines: List[str] = []

    match = re.search(r"SLURM_JOB_ID=(\d+)", out_text)
    if match:
        job_id = match.group(1)
    match = re.search(r"SLURM_ARRAY_TASK_ID=(\d+)", out_text)
    if match:
        task_id = int(match.group(1))
    match = re.search(r"RUN_NAME=(\S+)", out_text)
    if match:
        run_name = match.group(1)
    match = re.search(r"ALG=(\S+)\s+SET=(\d+)\s+SEED=(\d+)\s+DEVICE=(\S+)", out_text)
    if match:
        algorithm = match.group(1)
        set_id = int(match.group(2))
        seed = int(match.group(3))
        device = match.group(4)

    for text in (err_text, out_text):
        for candidate in tail_lines(text, max_lines=20):
            if any(token in candidate for token in ["Traceback", "Error", "Exception", "Killed", "oom", "OOM"]):
                error_lines.append(candidate)

    deduped_error_lines = []
    seen = set()
    for line in error_lines:
        if line not in seen:
            seen.add(line)
            deduped_error_lines.append(line)

    return (
        job_id,
        run_name,
        task_id,
        algorithm,
        set_id,
        seed,
        device,
        training_started,
        training_ended,
        traceback_detected,
        deduped_error_lines[:10],
    )


def discover_slurm_files(slurm_out_dir: Path, job_ids_filter: Optional[set]) -> Dict[str, List[SlurmFileRecord]]:
    records_by_run_name: Dict[str, List[SlurmFileRecord]] = defaultdict(list)
    pair_map: Dict[Tuple[str, str, int], SlurmFileRecord] = {}

    if not slurm_out_dir.exists():
        return records_by_run_name

    for path in sorted(slurm_out_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in {".out", ".err"}:
            continue

        job_name, job_id, task_id = parse_slurm_file_metadata(path)
        if job_id is None or task_id is None:
            continue
        if job_ids_filter and job_id not in job_ids_filter:
            continue

        key = (job_name or "", job_id, task_id)
        record = pair_map.get(key)
        if record is None:
            record = SlurmFileRecord(job_name=job_name, job_id=job_id, task_id=task_id)
            pair_map[key] = record

        if path.suffix == ".out":
            record.out_path = path
            record.out_text = read_text(path)
            record.out_mtime = dt_from_epoch(path.stat().st_mtime)
        else:
            record.err_path = path
            record.err_text = read_text(path)
            record.err_mtime = dt_from_epoch(path.stat().st_mtime)

    for record in pair_map.values():
        (
            parsed_job_id,
            parsed_run_name,
            parsed_task_id,
            parsed_algorithm,
            parsed_set,
            parsed_seed,
            parsed_device,
            training_started,
            training_ended,
            traceback_detected,
            error_lines,
        ) = parse_slurm_text(record.out_text, record.err_text)

        if parsed_job_id is not None:
            record.job_id = parsed_job_id
        if parsed_task_id is not None:
            record.task_id = parsed_task_id
        record.parsed_run_name = parsed_run_name
        record.parsed_algorithm = parsed_algorithm
        record.parsed_set = parsed_set
        record.parsed_seed = parsed_seed
        record.parsed_device = parsed_device
        record.training_started = training_started
        record.training_ended = training_ended
        record.traceback_detected = traceback_detected
        record.error_lines = error_lines

        key_name = record.parsed_run_name
        if key_name:
            records_by_run_name[key_name].append(record)

    for run_name, records in records_by_run_name.items():
        records.sort(key=lambda record: (
            int(record.job_id) if record.job_id and record.job_id.isdigit() else -1,
            record.out_mtime or datetime.min.replace(tzinfo=timezone.utc),
            record.err_mtime or datetime.min.replace(tzinfo=timezone.utc),
        ), reverse=True)

    return records_by_run_name


def parse_progress_csv(path: Path) -> ProgressInfo:
    info = ProgressInfo(path=path, mtime=dt_from_epoch(path.stat().st_mtime))
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            info.header = reader.fieldnames or []
            timestep_key = None
            for candidate in info.header:
                if candidate == "time/total_timesteps":
                    timestep_key = candidate
                    break
            if timestep_key is None:
                for candidate in info.header:
                    if "timesteps" in candidate.lower():
                        timestep_key = candidate
                        break

            last_row = None
            for row in reader:
                if not row:
                    continue
                if all((value is None or str(value).strip() == "") for value in row.values()):
                    continue
                info.row_count += 1
                last_row = row

            if last_row and timestep_key and last_row.get(timestep_key) not in (None, ""):
                info.last_timesteps = int(float(last_row[timestep_key]))
    except Exception as exc:
        info.parse_error = str(exc)
    return info


def discover_run_artifacts(repo_root: Path, logs_dir: Path, run_spec: RunSpec) -> RunArtifacts:
    run_home = logs_dir / run_spec.run_name
    progress_files = sorted(run_home.glob("logs/**/progress.csv"))
    progress_info = ProgressInfo()
    if progress_files:
        latest_progress = max(progress_files, key=lambda candidate: candidate.stat().st_mtime)
        progress_info = parse_progress_csv(latest_progress)

    weights_path = run_home / "weights" / f"env{run_spec.set_id}_{run_spec.algorithm}.zip"
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
        output_logs=output_logs,
        output_log_mtime=output_log_mtime,
    )


def query_command(command: List[str]) -> Tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, ""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False, ""
    if completed.returncode != 0 and completed.stdout == "" and completed.stderr == "":
        return False, ""
    return True, completed.stdout


def parse_active_job_identifier(job_identifier: str) -> Tuple[Optional[str], Optional[int]]:
    token = job_identifier.strip()
    if "_" in token:
        job_id, task_text = token.split("_", 1)
        return job_id, parse_int(task_text)
    return token, None


def query_squeue(job_ids_filter: Optional[set]) -> Dict[Tuple[str, int], Dict[str, str]]:
    command = ["squeue", "-h", "-r", "-o", "%i|%T|%j|%M|%L|%R"]
    if job_ids_filter:
        command.extend(["-j", ",".join(sorted(job_ids_filter))])

    ok, stdout = query_command(command)
    records: Dict[Tuple[str, int], Dict[str, str]] = {}
    if not ok:
        return records

    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("|", 5)
        if len(parts) != 6:
            continue
        job_identifier, state, job_name, elapsed, time_left, reason = [part.strip() for part in parts]
        job_id, task_id = parse_active_job_identifier(job_identifier)
        if task_id is None:
            continue
        records[(job_name, task_id)] = {
            "job_id": job_id or "",
            "state": state,
            "elapsed": elapsed,
            "time_left": time_left,
            "reason": reason,
        }
    return records


def query_sacct(job_ids_filter: Optional[set]) -> Dict[Tuple[str, int], Dict[str, str]]:
    command = ["sacct", "-n", "-P", "-X", "--format=JobID,JobName,State,Elapsed,ExitCode"]
    if job_ids_filter:
        command.extend(["-j", ",".join(sorted(job_ids_filter))])

    ok, stdout = query_command(command)
    records: Dict[Tuple[str, int], Dict[str, str]] = {}
    if not ok:
        return records

    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("|")
        if len(parts) < 5:
            continue
        job_id_text, job_name, state, elapsed, exit_code = [part.strip() for part in parts[:5]]
        if not re.fullmatch(r"\d+_\d+", job_id_text):
            continue
        job_id, task_text = job_id_text.split("_", 1)
        task_id = parse_int(task_text)
        if task_id is None:
            continue
        records[(job_name, task_id)] = {
            "job_id": job_id,
            "state": normalize_state(state) or state,
            "elapsed": elapsed,
            "exit_code": exit_code,
        }
    return records


def load_failed_runs_file(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_heartbeat(*times: Optional[datetime]) -> Optional[datetime]:
    valid = [dt for dt in times if dt is not None]
    if not valid:
        return None
    return max(valid)


def has_nonempty_err(slurm_files: Optional[SlurmFileRecord]) -> bool:
    if slurm_files is None or slurm_files.err_path is None:
        return False
    return bool(slurm_files.err_text.strip())


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
            failure_reason = "Training end marker found, but weights file is missing."
        elif existing_failed_entries:
            explicit_failure = True
            failure_reason = "Run already present in failed_train_from_tuned_runs.jsonl."
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
                notes.append("No Slurm outputs, progress.csv, or weights were found for this run.")

    if status == "ACTIVE_STAGED":
        status = "ACTIVE_RUNNING"

    if status == "COMPLETE_UNVERIFIED":
        safe_to_rerun = False

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


def format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return "-"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def format_steps(actual: Optional[int], expected: int) -> str:
    if actual is None:
        return f"- / {expected}"
    pct = 100.0 * actual / expected if expected else 0.0
    return f"{actual} / {expected} ({pct:.1f}%)"


def build_failed_run_record(status_record: StatusRecord) -> dict:
    slurm_files = status_record.slurm_files
    return {
        "status": "failed",
        "detection_source": "check_status_train_from_tuned",
        "detected_at": now_utc().isoformat(),
        "classification": status_record.status,
        "reason": status_record.failure_reason,
        "run_name": status_record.run_spec.run_name,
        "algorithm": status_record.run_spec.algorithm,
        "set": status_record.run_spec.set_id,
        "seed": status_record.run_spec.seed,
        "device": status_record.run_spec.device,
        "num_robots": status_record.run_spec.num_robots,
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
        "slurm_out_path": str(slurm_files.out_path) if slurm_files and slurm_files.out_path else None,
        "slurm_err_path": str(slurm_files.err_path) if slurm_files and slurm_files.err_path else None,
        "error_lines": slurm_files.error_lines if slurm_files else [],
    }


def scheduler_job_id(status_record: StatusRecord) -> Optional[str]:
    if status_record.scheduler.active_job_id:
        return status_record.scheduler.active_job_id
    if status_record.slurm_files and status_record.slurm_files.job_id:
        return status_record.slurm_files.job_id
    return None


def format_run_header(status_record: StatusRecord) -> str:
    run = status_record.run_spec
    return f"{run.run_name} [{run.algorithm} set={run.set_id} seed={run.seed} task={run.task_id}]"


def print_status_section(title: str) -> None:
    print("")
    print(title)
    print("-" * len(title))


def print_detailed_runs(title: str, records: List[StatusRecord]) -> None:
    if not records:
        return
    print_status_section(title)
    for record in records:
        slurm_files = record.slurm_files
        scheduler = record.scheduler
        print(f"{record.status}: {format_run_header(record)}")
        print(f"  steps            : {format_steps(record.actual_steps, record.expected_steps)}")
        print(f"  safe_to_rerun    : {'YES' if record.safe_to_rerun else 'NO'}")
        print(f"  last_heartbeat   : {format_datetime(record.last_heartbeat)}")
        if record.heartbeat_age_hours is not None:
            print(f"  heartbeat_age_h  : {record.heartbeat_age_hours:.2f}")
        print(f"  weights          : {'present' if record.artifacts.weights_exists else 'missing'} -> {record.artifacts.weights_path}")
        print(f"  progress_csv     : {record.artifacts.progress.path or '-'}")
        print(f"  slurm_out        : {slurm_files.out_path if slurm_files and slurm_files.out_path else '-'}")
        print(f"  slurm_err        : {slurm_files.err_path if slurm_files and slurm_files.err_path else '-'}")
        print(f"  scheduler        : active={scheduler.active_state or '-'} sacct={scheduler.sacct_state or '-'} exit={scheduler.sacct_exit_code or '-'}")
        if record.failure_reason:
            print(f"  failure_reason   : {record.failure_reason}")
        if slurm_files and slurm_files.error_lines:
            print("  error_lines      :")
            for line in slurm_files.error_lines[:5]:
                print(f"    - {line}")
        if record.notes:
            print("  notes            :")
            for note in record.notes:
                print(f"    - {note}")
        print("")


def print_compact_runs(title: str, records: List[StatusRecord]) -> None:
    if not records:
        return
    print_status_section(title)
    print(f"{'status':<18} {'task':>4} {'run_name':<48} {'steps':<24} {'heartbeat_age_h':>15} {'rerun':>7}")
    print(f"{'-' * 18} {'-' * 4} {'-' * 48} {'-' * 24} {'-' * 15} {'-' * 7}")
    for record in records:
        heartbeat_age = "-" if record.heartbeat_age_hours is None else f"{record.heartbeat_age_hours:.2f}"
        print(
            f"{record.status:<18} {record.run_spec.task_id:>4} "
            f"{record.run_spec.run_name:<48.48} "
            f"{format_steps(record.actual_steps, record.expected_steps):<24} "
            f"{heartbeat_age:>15} "
            f"{'YES' if record.safe_to_rerun else 'NO':>7}"
        )


def group_safe_reruns(records: List[StatusRecord]) -> Dict[Path, List[int]]:
    grouped: Dict[Path, List[int]] = defaultdict(list)
    for record in records:
        if record.safe_to_rerun:
            grouped[record.run_spec.slurm_script].append(record.run_spec.task_id)
    for tasks in grouped.values():
        tasks.sort()
    return grouped


def compress_task_ids(task_ids: List[int]) -> str:
    if not task_ids:
        return ""
    task_ids = sorted(set(task_ids))
    ranges = []
    start = prev = task_ids[0]
    for task_id in task_ids[1:]:
        if task_id == prev + 1:
            prev = task_id
            continue
        ranges.append((start, prev))
        start = prev = task_id
    ranges.append((start, prev))

    chunks = []
    for start, end in ranges:
        if start == end:
            chunks.append(str(start))
        else:
            chunks.append(f"{start}-{end}")
    return ",".join(chunks)


def print_summary_table(records: List[StatusRecord]) -> None:
    counts = Counter(record.status for record in records)
    print_status_section("Run counts by status")
    for status in sorted(counts):
        print(f"{status:<20} {counts[status]:>4}")

    per_algorithm: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        per_algorithm[record.run_spec.algorithm][record.status] += 1

    print_status_section("Per-algorithm status counts")
    for algorithm in sorted(per_algorithm):
        pieces = [f"{status}={count}" for status, count in sorted(per_algorithm[algorithm].items())]
        print(f"{algorithm:<10} " + ", ".join(pieces))


def print_expected_overview(
    script_specs: List[SlurmScriptSpec],
    expected_runs: List[RunSpec],
    args: argparse.Namespace,
    squeue_available: bool,
    sacct_available: bool,
) -> None:
    print("=" * 100)
    print("TRAIN-FROM-TUNED STATUS CHECK")
    print("=" * 100)
    print(f"Timestamp                 : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Repository root           : {args.repo_root}")
    print(f"Logs directory            : {args.logs_dir}")
    print(f"Slurm output directory    : {args.slurm_out_dir}")
    print(f"failed_train_from_tuned_runs.jsonl         : {args.failed_jsonl_path}")
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

    print_status_section("Parsed Slurm scripts")
    for spec in script_specs:
        print(
            f"{spec.path}: job_name={spec.job_name}, array={spec.array_spec}, "
            f"algorithms={spec.algorithms}, sets={spec.sets}, seeds={spec.seeds}, "
            f"steps={spec.steps}, num_robots={spec.num_robots}, device={spec.device}"
        )


def print_concise_summary(records: List[StatusRecord], failed_appends: int, job_ids_filter: Optional[set]) -> None:
    counts = Counter(record.status for record in records)
    safe_reruns = [record for record in records if record.safe_to_rerun]
    print_status_section("Concise summary")
    print(
        "expected={expected} complete_verified={complete} active_running={active_running} "
        "active_pending={active_pending} active_stale={active_stale} failed={failed} "
        "stale_inactive={stale} missing={missing} complete_unverified={unverified} "
        "incomplete_recent={incomplete} safe_reruns={safe_reruns_count} failed_jsonl_appends={failed_appends}".format(
            expected=len(records),
            complete=counts.get("COMPLETE_VERIFIED", 0),
            active_running=counts.get("ACTIVE_RUNNING", 0),
            active_pending=counts.get("ACTIVE_PENDING", 0),
            active_stale=counts.get("ACTIVE_STALE", 0),
            failed=counts.get("FAILED", 0),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check train-from-tuned status across Slurm outputs and training logs.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument(
        "--slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/train_from_tuned_all_3_robots.sh",
            "slurm_scripts/crossq_from_tuned_3_robots.sh",
        ],
        help="Slurm experiment scripts to parse.",
    )
    parser.add_argument("--logs-dir", type=Path, default=None, help="Override logs/training_from_tuned_logs.")
    parser.add_argument("--slurm-out-dir", type=Path, default=None, help="Override slurm_scripts/slurm_out.")
    parser.add_argument("--job-ids", nargs="*", default=[], help="Optional Slurm array job IDs to focus on.")
    parser.add_argument("--stale-hours", type=float, default=6.0, help="Heartbeat age above which a non-complete run is stale.")
    parser.add_argument("--failed-jsonl-name", default="failed_train_from_tuned_runs.jsonl", help="Root-level JSONL file for failed runs.")
    parser.add_argument("--no-record-failed-jsonl", action="store_true", help="Do not append newly identified failed runs to failed_train_from_tuned_runs.jsonl.")
    parser.add_argument("--nonzero-on-problems", action="store_true", help="Exit with code 1 when failed/stale/missing/unverified runs exist.")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.logs_dir = (args.logs_dir or (args.repo_root / "logs" / "training_from_tuned_logs")).resolve()
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
        artifacts = discover_run_artifacts(args.repo_root, args.logs_dir, run_spec)
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
    stale_records = [record for record in status_records if record.status == "STALE_INACTIVE"]
    active_stale_records = [record for record in status_records if record.status == "ACTIVE_STALE"]
    incomplete_recent_records = [record for record in status_records if record.status == "INCOMPLETE_RECENT"]
    missing_records = [record for record in status_records if record.status == "MISSING"]
    unverified_records = [record for record in status_records if record.status == "COMPLETE_UNVERIFIED"]
    active_running_records = [record for record in status_records if record.status == "ACTIVE_RUNNING"]
    active_pending_records = [record for record in status_records if record.status == "ACTIVE_PENDING"]

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

    print_concise_summary(status_records, failed_appends, job_ids_filter)

    problems = (
        failed_records
        or stale_records
        or active_stale_records
        or missing_records
        or unverified_records
    )
    if args.nonzero_on_problems and problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
