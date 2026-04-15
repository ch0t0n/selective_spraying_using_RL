#!/usr/bin/env python3
# EDITED:
"""Check tuning status across Slurm outputs and Optuna tuning logs.

This mirrors the training status checker, but it verifies Optuna runs using
trial counts, tuning end markers, TensorBoard event files, and the saved reward
plot instead of final model weights.
"""

# EDITED:
import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# EDITED:
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
RUN_NAME_RE = re.compile(r"^(?P<algorithm>[A-Za-z0-9]+)_set(?P<set>\d+)_seed(?P<seed>-?\d+).*$")
TRIAL_STATE_RE = re.compile(r"Trial\s+(?P<trial>\d+)\s+(?P<state>finished|pruned|failed)\b", re.IGNORECASE)
VALUE_RE = re.compile(r"value:\s*(?P<value>[-+0-9.eE]+)")
PARAMS_RE = re.compile(r"parameters:\s*(?P<params>\{.*\})")
BEST_VALUE_RE = re.compile(r"Best mean reward:\s*(?P<value>[-+0-9.eE]+)")
N_TRIALS_RE = re.compile(r"n_trials=(?P<n_trials>\d+)")


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
    steps_per_trial: int
    n_trials: int
    num_robots: int
    num_envs: int
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
    steps_per_trial: int
    expected_trials: int
    num_robots: int
    num_envs: int
    run_name: str


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
    tuning_started: bool = False
    tuning_ended: bool = False
    objective_exception_count: int = 0
    traceback_detected: bool = False
    error_lines: List[str] = field(default_factory=list)
    trial_records: List[TrialRecord] = field(default_factory=list)
    observed_trials: int = 0
    finished_trials: int = 0
    pruned_trials: int = 0
    failed_trials: int = 0
    best_value: Optional[float] = None
    best_params: Optional[Dict[str, object]] = None
    mean_rewards_count: Optional[int] = None
    parsed_n_trials: Optional[int] = None


# EDITED:
@dataclass
class RunArtifacts:
    run_home: Path
    run_home_exists: bool
    tensorboard_files: List[Path]
    tensorboard_mtime: Optional[datetime]
    plot_path: Path
    plot_exists: bool
    plot_mtime: Optional[datetime]
    output_logs: List[Path]
    output_log_mtime: Optional[datetime]


# EDITED:
@dataclass
class SchedulerInfo:
    active_state: Optional[str] = None
    active_job_id: Optional[str] = None
    sacct_state: Optional[str] = None
    sacct_exit_code: Optional[str] = None
    sacct_elapsed: Optional[str] = None


# EDITED:
@dataclass
class StatusRecord:
    run_spec: RunSpec
    slurm_files: Optional[SlurmFileRecord]
    artifacts: RunArtifacts
    scheduler: SchedulerInfo
    status: str
    safe_to_rerun: bool
    expected_trials: int
    actual_trials: Optional[int]
    trials_verified: bool
    objective_exception_count: int
    finished_trials: int
    pruned_trials: int
    failed_trials: int
    best_value: Optional[float]
    last_heartbeat: Optional[datetime]
    heartbeat_age_hours: Optional[float]
    failure_reason: Optional[str]
    notes: List[str] = field(default_factory=list)


# EDITED:
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# EDITED:
def dt_from_epoch(timestamp: Optional[float]) -> Optional[datetime]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


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
        return float(text)
    except ValueError:
        return None


# EDITED:
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


# EDITED:
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


# EDITED:
def normalize_state(raw_state: Optional[str]) -> Optional[str]:
    if raw_state is None:
        return None
    state = raw_state.strip()
    if not state:
        return None
    return state.split()[0].rstrip("+")


# EDITED:
def tail_lines(text: str, max_lines: int = 5) -> List[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


# EDITED:
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
def parse_arithmetic_assignment(text: str, variable_name: str) -> Optional[str]:
    pattern = re.compile(rf"^{re.escape(variable_name)}=\$\(\((.+?)\)\)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


# EDITED:
def parse_run_name_template(text: str) -> Optional[str]:
    pattern = re.compile(r'^RUN_NAME="(.+)"$', re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


# EDITED:
def parse_sbatch_directive(text: str, directive_name: str) -> Optional[str]:
    pattern = re.compile(rf"^#SBATCH\s+--{re.escape(directive_name)}(?:=(.+)|\s+(.+))$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip()


# EDITED:
def parse_python_arg(text: str, arg_name: str) -> Optional[str]:
    pattern = re.compile(rf"--{re.escape(arg_name)}\s+([^\s\\]+)")
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


# EDITED:
def safe_eval_arithmetic(expr: str, variables: Dict[str, int]) -> int:
    sanitized = expr.replace("/", "//")
    if not re.fullmatch(r"[\w\s()+\-*%/]+", sanitized):
        raise ValueError(f"Unsupported arithmetic expression: {expr}")
    return int(eval(sanitized, {"__builtins__": {}}, dict(variables)))


# EDITED:
def render_shell_template(template: str, variables: Dict[str, object]) -> str:
    def replace_braced(match: re.Match) -> str:
        key = match.group(1)
        return str(variables[key])

    rendered = re.sub(r"\$\{(\w+)\}", replace_braced, template)
    rendered = re.sub(r"\$(\w+)", replace_braced, rendered)
    return rendered


# EDITED:
def parse_slurm_script(path: Path) -> SlurmScriptSpec:
    text = read_text(path)

    job_name = parse_sbatch_directive(text, "job-name")
    array_spec = parse_sbatch_directive(text, "array")
    algorithms = parse_shell_list(text, "algorithms")
    sets = [int(value) for value in parse_shell_list(text, "sets")]
    seeds_raw = parse_shell_list(text, "seed") or parse_shell_list(text, "seeds")
    seeds = [int(value) for value in seeds_raw]
    steps_per_trial = int(parse_scalar_assignment(text, "steps") or parse_python_arg(text, "steps") or 0)
    n_trials = int(parse_scalar_assignment(text, "n_trials") or parse_python_arg(text, "n_trials") or 0)
    num_robots = int(parse_scalar_assignment(text, "num_robots") or parse_python_arg(text, "num_robots") or 0)
    num_envs = int(parse_scalar_assignment(text, "num_envs") or parse_python_arg(text, "num_envs") or 0)
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
    if not steps_per_trial:
        missing.append("steps_per_trial")
    if not n_trials:
        missing.append("n_trials")
    if not num_robots:
        missing.append("num_robots")
    if not num_envs:
        missing.append("num_envs")
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
        steps_per_trial=steps_per_trial,
        n_trials=n_trials,
        num_robots=num_robots,
        num_envs=num_envs,
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
                steps_per_trial=script_spec.steps_per_trial,
                expected_trials=script_spec.n_trials,
                num_robots=script_spec.num_robots,
                num_envs=script_spec.num_envs,
                run_name=run_name,
            )
        )
    return runs


# EDITED:
def parse_slurm_file_metadata(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    stem = path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None, None, None
    job_name, job_id, task_text = parts
    return job_name, job_id, parse_int(task_text)


# EDITED:
def safe_literal_eval(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


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

        existing = records.get(trial_number)
        candidate = TrialRecord(
            trial_number=trial_number,
            state=state,
            value=value,
            params=params,
            raw_line=line.strip(),
        )

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
def parse_mean_rewards_count(text: str) -> Optional[int]:
    marker = "Mean rewards:"
    index = text.rfind(marker)
    if index == -1:
        return None
    line = text[index + len(marker):].strip()
    parsed = safe_literal_eval(line)
    if isinstance(parsed, list):
        return len(parsed)
    return None


# EDITED:
def parse_best_params(text: str) -> Optional[Dict[str, object]]:
    marker = "Best hyperparameters:"
    index = text.rfind(marker)
    if index == -1:
        return None
    line = text[index + len(marker):].strip().splitlines()[0].strip()
    parsed = safe_literal_eval(line)
    if isinstance(parsed, dict):
        return parsed
    return None


# EDITED:
def parse_best_value(text: str) -> Optional[float]:
    matches = list(BEST_VALUE_RE.finditer(text))
    if not matches:
        return None
    return parse_float(matches[-1].group("value"))


# EDITED:
def parse_runtime_n_trials(text: str) -> Optional[int]:
    matches = list(N_TRIALS_RE.finditer(text))
    if not matches:
        return None
    return parse_int(matches[-1].group("n_trials"))


# EDITED:
def parse_slurm_text(out_text: str, err_text: str) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[str],
    bool,
    bool,
    int,
    bool,
    List[str],
    List[TrialRecord],
    Optional[float],
    Optional[Dict[str, object]],
    Optional[int],
    Optional[int],
]:
    combined_text = "\n".join(part for part in [out_text, err_text] if part)

    job_id = None
    task_id = None
    run_name = None
    algorithm = None
    set_id = None
    seed = None
    device = None
    tuning_started = "Tuning started on" in combined_text
    tuning_ended = "Tuning ended on" in combined_text
    objective_exception_count = combined_text.count("Training failed with error:")
    traceback_detected = "Traceback (most recent call last):" in combined_text
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

    trial_records = parse_trial_records(combined_text)
    finished_trials = sum(1 for record in trial_records if record.state == "finished")
    pruned_trials = sum(1 for record in trial_records if record.state == "pruned")
    failed_trials = sum(1 for record in trial_records if record.state == "failed")
    observed_trials = len(trial_records)
    best_value = parse_best_value(combined_text)
    best_params = parse_best_params(combined_text)
    mean_rewards_count = parse_mean_rewards_count(combined_text)
    parsed_n_trials = parse_runtime_n_trials(combined_text)

    return (
        job_id,
        run_name,
        task_id,
        algorithm,
        set_id,
        seed,
        device,
        tuning_started,
        tuning_ended,
        objective_exception_count,
        traceback_detected,
        deduped_error_lines[:10],
        trial_records,
        best_value,
        best_params,
        mean_rewards_count,
        parsed_n_trials,
    )


# EDITED:
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
            tuning_started,
            tuning_ended,
            objective_exception_count,
            traceback_detected,
            error_lines,
            trial_records,
            best_value,
            best_params,
            mean_rewards_count,
            parsed_n_trials,
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
        record.tuning_started = tuning_started
        record.tuning_ended = tuning_ended
        record.objective_exception_count = objective_exception_count
        record.traceback_detected = traceback_detected
        record.error_lines = error_lines
        record.trial_records = trial_records
        record.observed_trials = len(trial_records)
        record.finished_trials = sum(1 for trial in trial_records if trial.state == "finished")
        record.pruned_trials = sum(1 for trial in trial_records if trial.state == "pruned")
        record.failed_trials = sum(1 for trial in trial_records if trial.state == "failed")
        record.best_value = best_value
        record.best_params = best_params
        record.mean_rewards_count = mean_rewards_count
        record.parsed_n_trials = parsed_n_trials

        key_name = record.parsed_run_name
        if key_name:
            records_by_run_name[key_name].append(record)

    for run_name, records in records_by_run_name.items():
        records.sort(
            key=lambda record: (
                int(record.job_id) if record.job_id and record.job_id.isdigit() else -1,
                record.out_mtime or datetime.min.replace(tzinfo=timezone.utc),
                record.err_mtime or datetime.min.replace(tzinfo=timezone.utc),
            ),
            reverse=True,
        )

    return records_by_run_name


# EDITED:
def discover_run_artifacts(logs_dir: Path, run_spec: RunSpec) -> RunArtifacts:
    run_home = logs_dir / run_spec.run_name
    tensorboard_files = sorted(run_home.glob("logs/**/events.out.tfevents*")) if run_home.exists() else []
    tensorboard_mtime = None
    if tensorboard_files:
        latest_tensorboard_file = max(tensorboard_files, key=lambda candidate: candidate.stat().st_mtime)
        tensorboard_mtime = dt_from_epoch(latest_tensorboard_file.stat().st_mtime)

    plot_path = run_home / "outputs" / f"{run_spec.algorithm}_set{run_spec.set_id}_optuna_rewards.png"
    plot_exists = plot_path.exists()
    plot_mtime = dt_from_epoch(plot_path.stat().st_mtime) if plot_exists else None

    output_logs = sorted(run_home.glob("outputs/*.log")) if run_home.exists() else []
    output_log_mtime = None
    if output_logs:
        latest_output_log = max(output_logs, key=lambda candidate: candidate.stat().st_mtime)
        output_log_mtime = dt_from_epoch(latest_output_log.stat().st_mtime)

    return RunArtifacts(
        run_home=run_home,
        run_home_exists=run_home.exists(),
        tensorboard_files=tensorboard_files,
        tensorboard_mtime=tensorboard_mtime,
        plot_path=plot_path,
        plot_exists=plot_exists,
        plot_mtime=plot_mtime,
        output_logs=output_logs,
        output_log_mtime=output_log_mtime,
    )


# EDITED:
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


# EDITED:
def parse_active_job_identifier(job_identifier: str) -> Tuple[Optional[str], Optional[int]]:
    token = job_identifier.strip()
    if "_" in token:
        job_id, task_text = token.split("_", 1)
        return job_id, parse_int(task_text)
    return token, None


# EDITED:
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


# EDITED:
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


# EDITED:
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


# EDITED:
def latest_heartbeat(*times: Optional[datetime]) -> Optional[datetime]:
    valid = [dt for dt in times if dt is not None]
    if not valid:
        return None
    return max(valid)


# EDITED:
def compute_actual_trials(slurm_files: Optional[SlurmFileRecord]) -> Optional[int]:
    if slurm_files is None:
        return None
    counts = [slurm_files.observed_trials]
    if slurm_files.mean_rewards_count is not None:
        counts.append(slurm_files.mean_rewards_count)
    counts = [count for count in counts if count is not None]
    if not counts:
        return None
    return max(counts)


# EDITED:
def scheduler_job_id(status_record: StatusRecord) -> Optional[str]:
    if status_record.scheduler.active_job_id:
        return status_record.scheduler.active_job_id
    if status_record.slurm_files and status_record.slurm_files.job_id:
        return status_record.slurm_files.job_id
    return None


# EDITED:
def classify_run(
    run_spec: RunSpec,
    slurm_files: Optional[SlurmFileRecord],
    artifacts: RunArtifacts,
    scheduler: SchedulerInfo,
    stale_hours: float,
    failed_runs_index: Dict[str, List[dict]],
) -> StatusRecord:
    actual_trials = compute_actual_trials(slurm_files)
    expected_trials = run_spec.expected_trials
    trials_verified = actual_trials is not None and actual_trials >= expected_trials
    objective_exception_count = slurm_files.objective_exception_count if slurm_files else 0
    finished_trials = slurm_files.finished_trials if slurm_files else 0
    pruned_trials = slurm_files.pruned_trials if slurm_files else 0
    failed_trials = slurm_files.failed_trials if slurm_files else 0
    best_value = slurm_files.best_value if slurm_files else None

    last_heartbeat = latest_heartbeat(
        artifacts.tensorboard_mtime,
        artifacts.plot_mtime,
        artifacts.output_log_mtime,
        slurm_files.out_mtime if slurm_files else None,
        slurm_files.err_mtime if slurm_files else None,
    )
    heartbeat_age_hours = None
    if last_heartbeat is not None:
        heartbeat_age_hours = (now_utc() - last_heartbeat).total_seconds() / 3600.0

    has_tuning_output = slurm_files is not None and (slurm_files.tuning_started or slurm_files.out_path is not None)
    has_any_artifact = (
        has_tuning_output
        or artifacts.run_home_exists
        or artifacts.plot_exists
        or bool(artifacts.tensorboard_files)
        or bool(artifacts.output_logs)
    )

    notes: List[str] = []
    failure_reason = None
    status = "UNKNOWN"
    safe_to_rerun = False

    existing_failed_entries = failed_runs_index.get(run_spec.run_name, [])
    tuning_ended = slurm_files.tuning_ended if slurm_files else False

    if objective_exception_count > 0:
        notes.append(f"{objective_exception_count} trial(s) raised caught exceptions inside the Optuna objective.")
    if slurm_files and slurm_files.parsed_n_trials is not None and slurm_files.parsed_n_trials != expected_trials:
        notes.append(
            f"Runtime n_trials={slurm_files.parsed_n_trials} differs from expected n_trials={expected_trials}."
        )

    if tuning_ended and trials_verified and artifacts.plot_exists:
        status = "COMPLETE_VERIFIED"
    elif tuning_ended and (artifacts.plot_exists or best_value is not None):
        status = "COMPLETE_UNVERIFIED"
        notes.append("Tuning end marker was found, but trial verification is incomplete.")
    elif scheduler.active_state in ACTIVE_PENDING_STATES:
        status = "ACTIVE_PENDING"
        notes.append(f"squeue reports {scheduler.active_state}.")
    elif scheduler.active_state in ACTIVE_RUNNING_STATES:
        status = "ACTIVE_RUNNING"
        if heartbeat_age_hours is not None and heartbeat_age_hours > stale_hours:
            status = "ACTIVE_STALE"
            notes.append(f"Active in Slurm but no artifact heartbeat for {heartbeat_age_hours:.2f}h.")
        else:
            notes.append(f"squeue reports {scheduler.active_state}.")
    else:
        sacct_failed = scheduler.sacct_state in FAILED_STATES
        slurm_completed = scheduler.sacct_state == "COMPLETED"
        explicit_failure = False

        if sacct_failed:
            explicit_failure = True
            failure_reason = f"sacct reports {scheduler.sacct_state}."
        elif existing_failed_entries:
            explicit_failure = True
            failure_reason = "Run already present in failed_tuning_runs.jsonl."
        elif slurm_completed and not tuning_ended:
            explicit_failure = True
            failure_reason = "Slurm job completed, but the tuning end marker was not found."
        elif slurm_completed and actual_trials is not None and actual_trials < expected_trials:
            explicit_failure = True
            failure_reason = (
                f"Slurm job completed, but only {actual_trials}/{expected_trials} trials were verified."
            )
        elif slurm_files and slurm_files.error_lines and not tuning_ended and not has_any_artifact:
            explicit_failure = True
            failure_reason = "Error output was found and no usable tuning artifacts were produced."

        if explicit_failure:
            status = "FAILED"
            safe_to_rerun = True
        else:
            if has_any_artifact and actual_trials is not None and actual_trials < expected_trials:
                if heartbeat_age_hours is not None and heartbeat_age_hours > stale_hours:
                    status = "STALE_INACTIVE"
                    failure_reason = (
                        f"Run is incomplete ({actual_trials}/{expected_trials} trials) and stale."
                    )
                    safe_to_rerun = True
                else:
                    status = "INCOMPLETE_RECENT"
                    notes.append("Run is incomplete and not active, but the latest heartbeat is still recent.")
            elif has_any_artifact:
                if heartbeat_age_hours is not None and heartbeat_age_hours > stale_hours:
                    status = "STALE_INACTIVE"
                    failure_reason = "Run has artifacts but no verified completion and is stale."
                    safe_to_rerun = True
                else:
                    status = "INCOMPLETE_RECENT"
                    notes.append("Run has partial artifacts but no definitive completion marker.")
            else:
                status = "MISSING"
                safe_to_rerun = True
                notes.append("No Slurm outputs, TensorBoard files, or reward plot were found for this run.")

    if status == "COMPLETE_UNVERIFIED":
        safe_to_rerun = False

    return StatusRecord(
        run_spec=run_spec,
        slurm_files=slurm_files,
        artifacts=artifacts,
        scheduler=scheduler,
        status=status,
        safe_to_rerun=safe_to_rerun,
        expected_trials=expected_trials,
        actual_trials=actual_trials,
        trials_verified=trials_verified,
        objective_exception_count=objective_exception_count,
        finished_trials=finished_trials,
        pruned_trials=pruned_trials,
        failed_trials=failed_trials,
        best_value=best_value,
        last_heartbeat=last_heartbeat,
        heartbeat_age_hours=heartbeat_age_hours,
        failure_reason=failure_reason,
        notes=notes,
    )


# EDITED:
def format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return "-"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# EDITED:
def format_trials(actual: Optional[int], expected: int) -> str:
    if actual is None:
        return f"- / {expected}"
    pct = 100.0 * actual / expected if expected else 0.0
    return f"{actual} / {expected} ({pct:.1f}%)"


# EDITED:
def build_failed_run_record(status_record: StatusRecord) -> dict:
    slurm_files = status_record.slurm_files
    return {
        "status": "failed",
        "detection_source": "check_status_tuning",
        "detected_at": now_utc().isoformat(),
        "classification": status_record.status,
        "reason": status_record.failure_reason,
        "run_name": status_record.run_spec.run_name,
        "algorithm": status_record.run_spec.algorithm,
        "set": status_record.run_spec.set_id,
        "seed": status_record.run_spec.seed,
        "device": status_record.run_spec.device,
        "num_robots": status_record.run_spec.num_robots,
        "num_envs": status_record.run_spec.num_envs,
        "steps_per_trial": status_record.run_spec.steps_per_trial,
        "expected_trials": status_record.expected_trials,
        "actual_trials": status_record.actual_trials,
        "objective_exception_count": status_record.objective_exception_count,
        "best_value": status_record.best_value,
        "slurm_script": str(status_record.run_spec.slurm_script),
        "task_id": status_record.run_spec.task_id,
        "slurm_job_name": status_record.run_spec.job_name,
        "slurm_job_id": scheduler_job_id(status_record),
        "slurm_array_task_id": status_record.run_spec.task_id,
        "sacct_state": status_record.scheduler.sacct_state,
        "sacct_exit_code": status_record.scheduler.sacct_exit_code,
        "active_state": status_record.scheduler.active_state,
        "run_home": str(status_record.artifacts.run_home),
        "plot_path": str(status_record.artifacts.plot_path),
        "slurm_out_path": str(slurm_files.out_path) if slurm_files and slurm_files.out_path else None,
        "slurm_err_path": str(slurm_files.err_path) if slurm_files and slurm_files.err_path else None,
        "error_lines": slurm_files.error_lines if slurm_files else [],
    }


# EDITED:
def format_run_header(status_record: StatusRecord) -> str:
    run = status_record.run_spec
    return f"{run.run_name} [{run.algorithm} set={run.set_id} seed={run.seed} task={run.task_id}]"


# EDITED:
def print_status_section(title: str) -> None:
    print("")
    print(title)
    print("-" * len(title))


# EDITED:
def print_detailed_runs(title: str, records: List[StatusRecord]) -> None:
    if not records:
        return
    print_status_section(title)
    for record in records:
        slurm_files = record.slurm_files
        scheduler = record.scheduler
        print(f"{record.status}: {format_run_header(record)}")
        print(f"  trials           : {format_trials(record.actual_trials, record.expected_trials)}")
        print(f"  trial_states     : finished={record.finished_trials} pruned={record.pruned_trials} failed={record.failed_trials}")
        print(f"  trial_exceptions : {record.objective_exception_count}")
        print(f"  safe_to_rerun    : {'YES' if record.safe_to_rerun else 'NO'}")
        print(f"  last_heartbeat   : {format_datetime(record.last_heartbeat)}")
        if record.heartbeat_age_hours is not None:
            print(f"  heartbeat_age_h  : {record.heartbeat_age_hours:.2f}")
        print(f"  best_value       : {record.best_value if record.best_value is not None else '-'}")
        print(f"  reward_plot      : {'present' if record.artifacts.plot_exists else 'missing'} -> {record.artifacts.plot_path}")
        print(f"  tensorboard_logs : {len(record.artifacts.tensorboard_files)} file(s)")
        print(f"  run_home         : {record.artifacts.run_home}")
        print(f"  slurm_out        : {slurm_files.out_path if slurm_files and slurm_files.out_path else '-'}")
        print(f"  slurm_err        : {slurm_files.err_path if slurm_files and slurm_files.err_path else '-'}")
        print(f"  scheduler        : active={scheduler.active_state or '-'} sacct={scheduler.sacct_state or '-'} exit={scheduler.sacct_exit_code or '-'}")
        if failure_reason := record.failure_reason:
            print(f"  failure_reason   : {failure_reason}")
        if slurm_files and slurm_files.error_lines:
            print("  error_lines      :")
            for line in slurm_files.error_lines[:5]:
                print(f"    - {line}")
        if record.notes:
            print("  notes            :")
            for note in record.notes:
                print(f"    - {note}")
        print("")


# EDITED:
def print_compact_runs(title: str, records: List[StatusRecord]) -> None:
    if not records:
        return
    print_status_section(title)
    print(f"{'status':<18} {'task':>4} {'run_name':<48} {'trials':<22} {'best_value':>12} {'rerun':>7}")
    print(f"{'-' * 18} {'-' * 4} {'-' * 48} {'-' * 22} {'-' * 12} {'-' * 7}")
    for record in records:
        best_value = "-" if record.best_value is None else f"{record.best_value:.4g}"
        print(
            f"{record.status:<18} {record.run_spec.task_id:>4} "
            f"{record.run_spec.run_name:<48.48} "
            f"{format_trials(record.actual_trials, record.expected_trials):<22} "
            f"{best_value:>12} "
            f"{'YES' if record.safe_to_rerun else 'NO':>7}"
        )


# EDITED:
def group_safe_reruns(records: List[StatusRecord]) -> Dict[Path, List[int]]:
    grouped: Dict[Path, List[int]] = defaultdict(list)
    for record in records:
        if record.safe_to_rerun:
            grouped[record.run_spec.slurm_script].append(record.run_spec.task_id)
    for tasks in grouped.values():
        tasks.sort()
    return grouped


# EDITED:
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


# EDITED:
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


# EDITED:
def print_expected_overview(
    script_specs: List[SlurmScriptSpec],
    expected_runs: List[RunSpec],
    args: argparse.Namespace,
    squeue_available: bool,
    sacct_available: bool,
) -> None:
    print("=" * 100)
    print("TUNING STATUS CHECK")
    print("=" * 100)
    print(f"Timestamp                 : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Repository root           : {args.repo_root}")
    print(f"Logs directory            : {args.logs_dir}")
    print(f"Slurm output directory    : {args.slurm_out_dir}")
    print(f"failed_tuning_runs.jsonl  : {args.failed_jsonl_path}")
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
            f"steps_per_trial={spec.steps_per_trial}, n_trials={spec.n_trials}, "
            f"num_robots={spec.num_robots}, num_envs={spec.num_envs}, device={spec.device}"
        )


# EDITED:
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


# EDITED:
def main() -> int:
    parser = argparse.ArgumentParser(description="Check tuning status across Slurm outputs and tuning logs.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent, help="Repository root.")
    parser.add_argument(
        "--slurm-scripts",
        nargs="*",
        default=[
            "slurm_scripts/train_random_all_3_robots.sh",
            "slurm_scripts/crossq_random_3_robots.sh",
        ],
        help="Slurm tuning scripts to parse.",
    )
    parser.add_argument("--logs-dir", type=Path, default=None, help="Override logs/training_tuning_logs.")
    parser.add_argument("--slurm-out-dir", type=Path, default=None, help="Override slurm_scripts/slurm_out.")
    parser.add_argument("--job-ids", nargs="*", default=[], help="Optional Slurm array job IDs to focus on.")
    parser.add_argument("--stale-hours", type=float, default=6.0, help="Heartbeat age above which a non-complete run is stale.")
    parser.add_argument("--failed-jsonl-name", default="failed_tuning_runs.jsonl", help="Root-level JSONL file for failed tuning runs.")
    parser.add_argument("--no-record-failed-jsonl", action="store_true", help="Do not append newly identified failed runs to failed_tuning_runs.jsonl.")
    parser.add_argument("--nonzero-on-problems", action="store_true", help="Exit with code 1 when failed/stale/missing/unverified runs exist.")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.logs_dir = (args.logs_dir or (args.repo_root / "logs" / "training_tuning_logs")).resolve()
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
    stale_records = [record for record in status_records if record.status == "STALE_INACTIVE"]
    active_stale_records = [record for record in status_records if record.status == "ACTIVE_STALE"]
    incomplete_recent_records = [record for record in status_records if record.status == "INCOMPLETE_RECENT"]
    missing_records = [record for record in status_records if record.status == "MISSING"]
    unverified_records = [record for record in status_records if record.status == "COMPLETE_UNVERIFIED"]
    active_running_records = [record for record in status_records if record.status == "ACTIVE_RUNNING"]
    active_pending_records = [record for record in status_records if record.status == "ACTIVE_PENDING"]
    trial_exception_records = [record for record in status_records if record.objective_exception_count > 0]

    print_detailed_runs("Failed runs", failed_records)
    print_detailed_runs("Stale inactive runs", stale_records)
    print_detailed_runs("Active stale runs", active_stale_records)
    print_detailed_runs("Completed but not fully verified runs", unverified_records)
    print_detailed_runs("Runs with caught trial exceptions", trial_exception_records)
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


# EDITED:
if __name__ == "__main__":
    sys.exit(main())
