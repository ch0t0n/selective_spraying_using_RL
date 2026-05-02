#!/usr/bin/env python3
"""
check_status_train_tune.py

Purpose
-------
Command-line-only status checker for the work immediately after submitting the
training/tuning jobs:

  checked as completed/status scopes
    Step 1 default training: CrossQ + other algorithms
    Step 3 Optuna tuning: all algorithms

  checked as readiness gates only
    Step 2 transfer learning can start when Step 1 env1 source checkpoints are good
    Step 4 tuned training can start when Step 3 tuned hyperparameters are good

  not checked as completed-output scopes
    transfer outputs, tuned-training outputs, ablation outputs, evaluation jobs,
    analysis jobs, plotting jobs, sensitivity jobs

The script prints everything to stdout. It does not create rerun scripts, JSON
reports, or any other files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Fixed grids used as fallbacks when the Slurm scripts cannot be parsed.
# ---------------------------------------------------------------------------

MAIN_OTHER_ALGS = ["A2C", "ARS", "PPO", "TQC", "TRPO"]
TUNE_ALGS = ["A2C", "ARS", "PPO", "TRPO", "CrossQ", "TQC"]
STEP4_OTHER_ALGS = ["A2C", "ARS", "PPO", "TQC", "TRPO"]
SEEDS = [0, 42, 123, 2024, 9999]
SETS = list(range(1, 11))
ROBOTS = [2, 3, 4, 5]
DEFAULT_STEPS = 2_000_000
DEFAULT_LOG_STEPS = 10_000
DEFAULT_N_EVAL_EPS = 5
DEFAULT_TUNE_STEPS = 2_000_000
DEFAULT_TUNE_TRIALS_PER_ALG = 50
DEFAULT_TUNE_SET = 1
DEFAULT_TUNE_NUM_ROBOTS = 3
DEFAULT_TUNE_SEED = 42
DEFAULT_PROJECT_ROOT_NAME = "selective_spraying_using_RL"

QUICK_GUIDE = f"""
CHECK_STATUS_TRAIN_TUNE QUICK GUIDE

Normal startup checked by default:
  Step 1: sbatch slurm/step1_crossq_default.sh
          sbatch slurm/step1_others_default.sh
  Step 3: sbatch slurm/step3_crossq_tune_hyperparameters.sh
          sbatch slurm/step3_others_tune_hyperparameters.sh

Commandments:
  Fast       python check_status_train_tune.py --fast
  Normal     python check_status_train_tune.py
  Done-check python check_status_train_tune.py --done
  Strict     python check_status_train_tune.py --done --strict
  Forensic   python check_status_train_tune.py --forensic

Core options:
  --project_root PATH        default: {DEFAULT_PROJECT_ROOT_NAME}; uses cwd automatically when already inside the repo
  --log_root PATH            default: logs
  --script_dir PATH          default: slurm
  --step3-mode split|combined|auto   default: split; combined is alternate, not normal

Depth options:
  --fast                     shallow/quick: squeue + file presence; skips sacct, log-content scans, zip/NPZ tests, Optuna replay
  --done                     alias for --assume-finished; missing work becomes actionable
  --strict                   alias for --strict-logs --strict-tensorboard
  --forensic                 done + strict + wider details + longer sacct window

Useful overrides:
  --no-squeue --no-sacct --sacct-days N --no-slurm-workdir-filter
  --allow-missing-ep-lengths --rerun-finished-bad
  --details-limit N --error-details-limit N --commands-limit N --rerun-chunk-size N
  --min-timestep-fraction F --min-eval-fraction F --min-model-bytes N --min-npz-bytes N
""".strip()

TRAIN_SCRIPTS = ["step1_crossq_default.sh", "step1_others_default.sh"]
TUNE_COMBINED_SCRIPT = "step3_tune_hyperparameters.sh"
TUNE_SPLIT_CROSSQ_SCRIPT = "step3_crossq_tune_hyperparameters.sh"
TUNE_SPLIT_OTHERS_SCRIPT = "step3_others_tune_hyperparameters.sh"
TUNE_CANDIDATE_SCRIPTS = [
    TUNE_COMBINED_SCRIPT,
    TUNE_SPLIT_CROSSQ_SCRIPT,
    TUNE_SPLIT_OTHERS_SCRIPT,
]
# Backwards-compatible alias. Do not use this to infer the selected Step 3 layout.
TUNE_SCRIPT = TUNE_COMBINED_SCRIPT
READINESS_SCRIPTS = [
    "step2_crossq_transfer.sh",
    "step2_others_transfer.sh",
    "step4_crossq_tuned.sh",
    "step4_others_tuned.sh",
]
ALL_SCRIPTS = TRAIN_SCRIPTS + TUNE_CANDIDATE_SCRIPTS + READINESS_SCRIPTS

HP_REQUIRED_KEYS = {
    "A2C": {"learning_rate", "gae_lambda", "vf_coef", "ent_coef", "max_grad_norm"},
    "ARS": {"learning_rate", "delta_std", "n_delta"},
    "PPO": {"learning_rate", "gae_lambda", "vf_coef", "ent_coef", "max_grad_norm", "clip_range", "n_epochs"},
    "TRPO": {"learning_rate", "gae_lambda", "target_kl", "cg_max_steps"},
    "CrossQ": {"learning_rate", "buffer_size", "batch_size"},
    "TQC": {"learning_rate", "buffer_size", "batch_size", "tau", "top_quantiles_to_drop_per_net"},
}

RUNNING_STATES = {"R", "RUNNING", "CG", "COMPLETING", "S", "SUSPENDED"}
PENDING_STATES = {"PD", "PENDING", "CF", "CONFIGURING"}
ACTIVE_STATES = RUNNING_STATES | PENDING_STATES
FATAL_SACCT_STATES = {
    "FAILED", "TIMEOUT", "CANCELLED", "CANCELLED+", "CANCELED", "CANCELED+",
    "OUT_OF_MEMORY", "OOM", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL",
    "DEADLINE", "REVOKED", "SPECIAL_EXIT",
}

FATAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Traceback \(most recent call last\)", re.I), "Python traceback"),
    (re.compile(r"FileNotFoundError|No such file or directory", re.I), "missing file/path"),
    (re.compile(r"ModuleNotFoundError|ImportError|cannot import name", re.I), "import/environment error"),
    (re.compile(r"AssertionError|KeyError|ValueError|TypeError|AttributeError", re.I), "Python exception"),
    (re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory|oom-kill|OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED", re.I), "out of memory / CUDA OOM"),
    (re.compile(r"\bKilled\b|oom_kill", re.I), "process killed"),
    (re.compile(r"Permission denied", re.I), "permission denied"),
    (re.compile(r"ERROR:\s*(Submit from the repository root|PYTHON_BIN is not executable|unknown experiment)", re.I), "job setup error"),
    (re.compile(r"slurmstepd: error|srun: error|sbatch: error", re.I), "Slurm step error"),
    (re.compile(r"DUE TO TIME LIMIT|TIMEOUT|time limit", re.I), "time limit"),
    (re.compile(r"CANCELLED|CANCELED|cancelled by|canceled by", re.I), "cancelled"),
    (re.compile(r"failed with exit code|exit code\s*[1-9]|exited with exit code [1-9]|\bFAILED\b|command not found", re.I), "nonzero exit/failure"),
]
WARN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"WARNING|UserWarning|DeprecationWarning|FutureWarning", re.I), "warning messages"),
    (re.compile(r"Gym has been unmaintained|cuDNN factory|cuBLAS factory|computation placer|TF-TRT|XLA|tensorflow", re.I), "library startup warnings"),
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlurmInfo:
    script_name: str
    path: Path | None
    job_name: str | None
    array_spec: str | None
    array_ids: tuple[int, ...]
    array_count: int | None
    arrays: dict[str, list[str]] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    output_glob: str | None = None
    error_glob: str | None = None

@dataclass(frozen=True)
class TrainSpec:
    group: str
    script_name: str
    array_id: int
    algorithm: str
    env_set: int
    num_robots: int
    seed: int
    version: str
    expected_steps: int
    log_steps: int
    n_eval_eps: int

    @property
    def tag(self) -> str:
        return f"{self.algorithm}_N{self.num_robots}_env{self.env_set}_seed{self.seed}"

    @property
    def final_model_name(self) -> str:
        return f"{self.algorithm}_N{self.num_robots}_env{self.env_set}.zip"

    @property
    def expected_eval_count(self) -> int:
        return self.expected_steps // self.log_steps if self.log_steps else 0

    @property
    def command_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.script_name, tuple())

@dataclass(frozen=True)
class TuneSpec:
    algorithm: str
    alg_idx: int
    script_name: str
    array_ids: tuple[int, ...]
    expected_trials: int
    storage_rel: str
    study_name: str
    output_json_rel: str
    tune_steps: int
    tune_set: int
    tune_num_robots: int
    tune_seed: int

    @property
    def command_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.script_name, tuple())

@dataclass(frozen=True)
class Step3Layout:
    mode: str                       # combined or split
    scripts: tuple[str, ...]
    reason: str
    combined_active: int = 0
    split_active: int = 0
    combined_logs: int = 0
    split_logs: int = 0
    warning: str = ""

@dataclass
class JobActivity:
    source: str
    job_id: str
    job_name: str
    array_id: int | None
    state: str
    elapsed: str = ""
    reason: str = ""
    exit_code: str = ""
    work_dir: str = ""

@dataclass
class LogFinding:
    path: str
    stream: str
    job_name: str | None
    job_id: str | None
    array_id: int | None
    severity: str
    reason: str
    snippet: str
    size: int
    mtime: float

@dataclass
class ArtifactResult:
    critical: list[str]
    warnings: list[str]
    metrics: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.critical

@dataclass
class TrainCheck:
    spec: TrainSpec
    state: str
    artifact: ArtifactResult
    active: list[JobActivity]
    sacct: list[JobActivity]
    logs: list[LogFinding]
    stdout_paths: list[str]
    fatal_evidence: list[str]

    @property
    def verified(self) -> bool:
        return self.state in {"verified_good", "verified_warn"}

@dataclass
class TuneCheck:
    spec: TuneSpec
    state: str
    issues: list[str]
    warnings: list[str]
    metrics: dict[str, Any]
    active: list[JobActivity]
    sacct: list[JobActivity]
    logs: list[LogFinding]
    stdout_paths: list[str]
    fatal_evidence: list[str]
    suggested_ids: list[int]

    @property
    def verified(self) -> bool:
        return self.state in {"verified_good", "verified_warn"}

# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def looks_like_project_root(path: Path) -> bool:
    return (path / "train.py").exists() and (path / "src").is_dir()


def resolve_project_root(value: str) -> Path:
    """Resolve the user-facing project root default without breaking cwd use.

    The normal repository directory is named selective_spraying_using_RL.  When
    users run from inside that repository, the default still resolves to cwd
    instead of cwd/selective_spraying_using_RL.
    """
    raw = (value or DEFAULT_PROJECT_ROOT_NAME).strip()
    p = Path(raw).expanduser()
    cwd = Path.cwd().expanduser()

    if raw == DEFAULT_PROJECT_ROOT_NAME and not p.is_absolute():
        candidates = []
        if looks_like_project_root(cwd):
            candidates.append(cwd)
        candidates.append(cwd / DEFAULT_PROJECT_ROOT_NAME)
        for parent in cwd.parents:
            if parent.name == DEFAULT_PROJECT_ROOT_NAME:
                candidates.append(parent)
            candidates.append(parent / DEFAULT_PROJECT_ROOT_NAME)
        for cand in candidates:
            try:
                if looks_like_project_root(cand):
                    return cand.resolve()
            except OSError:
                pass

    return p.resolve()


def pct(num: int | float, den: int | float) -> str:
    if not den:
        return "---"
    return f"{100.0 * float(num) / float(den):5.1f}%"


def human(n: Any) -> str:
    if n is None:
        return "---"
    if isinstance(n, str):
        return n
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def compact_ranges(values: Iterable[int]) -> str:
    nums = sorted(set(int(v) for v in values))
    if not nums:
        return ""
    parts: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def expand_ranges(expr: str) -> list[int]:
    expr = (expr or "").strip().split("%", 1)[0]
    ids: list[int] = []
    if not expr:
        return ids
    for part in expr.split(","):
        part = part.strip().strip("[]")
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = int(m.group(3) or 1)
            ids.extend(range(a, b + 1, step) if a <= b else range(a, b - 1, -step))
        else:
            try:
                ids.append(int(part))
            except ValueError:
                pass
    return sorted(set(ids))


def file_nonempty(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def read_text(path: Path, max_bytes: int = 200_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode(errors="replace")
    except Exception:
        return ""


def is_zip_ok(path: Path, min_bytes: int) -> tuple[bool, str]:
    if not file_nonempty(path, min_bytes=min_bytes):
        return False, "missing/too small"
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
        if bad is not None:
            return False, f"bad member {bad}"
        return True, "ok"
    except Exception as exc:
        return False, f"invalid zip: {exc}"


def print_table(headers: list[str], rows: list[list[Any]], *, max_col_width: int = 72) -> None:
    if not rows:
        print("  (none)")
        return
    str_headers = [str(h) for h in headers]
    str_rows = [[str(x) for x in row] for row in rows]
    widths: list[int] = []
    for i, h in enumerate(str_headers):
        w = len(h)
        for row in str_rows:
            if i < len(row):
                w = max(w, len(row[i]))
        widths.append(min(w, max_col_width))

    def cell(s: str, w: int) -> str:
        if len(s) > w:
            return s[: max(0, w - 1)] + "…"
        return s.ljust(w)

    print("  " + " | ".join(cell(str_headers[i], widths[i]) for i in range(len(widths))))
    print("  " + "-+-".join("-" * w for w in widths))
    for row in str_rows:
        print("  " + " | ".join(cell(row[i] if i < len(row) else "", widths[i]) for i in range(len(widths))))


def median_or_dash(vals: list[int | float]) -> str:
    if not vals:
        return "---"
    return human(median(vals))

# ---------------------------------------------------------------------------
# Slurm script parsing and expected run generation
# ---------------------------------------------------------------------------

def resolve_script(root: Path, script_name: str) -> Path | None:
    for p in (root / "slurm" / script_name, root / script_name):
        if p.exists():
            return p
    return None


def rel_or_abs(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def parse_bash_array(text: str, name: str) -> list[str]:
    m = re.search(rf"(?:^|\n)\s*{re.escape(name)}\s*=\s*\((.*?)\)", text, re.S)
    if not m:
        return []
    body = " ".join(line.split("#", 1)[0].strip() for line in m.group(1).splitlines())
    try:
        return [str(x) for x in shlex.split(body)]
    except Exception:
        return [x.strip().strip("'\"") for x in body.split() if x.strip()]


def parse_assignments(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            v = m.group(2).strip().strip("'\"")
            if not v.startswith("$("):
                out[m.group(1)] = v
    for flag in ["--steps", "--log_steps", "--n_eval_eps", "--tune_steps", "--n_trials"]:
        m = re.search(rf"{re.escape(flag)}\s+([0-9]+)", text)
        if m:
            out[flag] = m.group(1)
    return out


def parse_slurm(root: Path, script_name: str) -> SlurmInfo:
    path = resolve_script(root, script_name)
    if path is None:
        return SlurmInfo(script_name, None, None, None, tuple(), None)
    text = path.read_text(errors="replace")

    def sbatch_value(opt: str) -> str | None:
        m = re.search(rf"^\s*#SBATCH\s+{re.escape(opt)}(?:=|\s+)(\S+)", text, re.M)
        return m.group(1).strip() if m else None

    array_spec = sbatch_value("--array")
    array_ids = tuple(expand_ranges(array_spec or ""))
    job_name = sbatch_value("--job-name")
    out_pat = sbatch_value("--output")
    err_pat = sbatch_value("--error")

    def slurm_glob(pat: str | None) -> str | None:
        if not pat:
            return None
        g = pat.replace("%x", "*").replace("%A", "*").replace("%a", "*").replace("%j", "*")
        return str(rel_or_abs(root, g))

    arrays = {
        k: parse_bash_array(text, k)
        for k in ["algorithms", "devices", "sets", "robots", "seeds"]
    }
    arrays = {k: v for k, v in arrays.items() if v}
    return SlurmInfo(
        script_name=script_name,
        path=path,
        job_name=job_name,
        array_spec=array_spec,
        array_ids=array_ids,
        array_count=len(array_ids) if array_ids else None,
        arrays=arrays,
        assignments=parse_assignments(text),
        output_glob=slurm_glob(out_pat),
        error_glob=slurm_glob(err_pat),
    )


def as_ints(values: Sequence[str], default: list[int]) -> list[int]:
    out: list[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            pass
    return out or list(default)


def as_strs(values: Sequence[str], default: list[str]) -> list[str]:
    return [str(v) for v in values] if values else list(default)


def script_steps(info: SlurmInfo) -> tuple[int, int, int]:
    raw_steps = info.assignments.get("steps") or info.assignments.get("--steps")
    steps = parse_int(raw_steps, DEFAULT_STEPS) or DEFAULT_STEPS
    log_steps = parse_int(info.assignments.get("--log_steps"), DEFAULT_LOG_STEPS) or DEFAULT_LOG_STEPS
    n_eval_eps = parse_int(info.assignments.get("--n_eval_eps"), DEFAULT_N_EVAL_EPS) or DEFAULT_N_EVAL_EPS
    return steps, log_steps, n_eval_eps


def build_train_specs(infos: dict[str, SlurmInfo]) -> list[TrainSpec]:
    specs: list[TrainSpec] = []
    definitions = [
        ("step1_crossq_default.sh", "Step 1 CrossQ default", ["CrossQ"]),
        ("step1_others_default.sh", "Step 1 other algorithms default", MAIN_OTHER_ALGS),
    ]
    for script_name, group, fallback_algs in definitions:
        info = infos[script_name]
        algs = as_strs(info.arrays.get("algorithms", []), fallback_algs)
        sets = as_ints(info.arrays.get("sets", []), SETS)
        robots = as_ints(info.arrays.get("robots", []), ROBOTS)
        seeds = as_ints(info.arrays.get("seeds", []), SEEDS)
        steps, log_steps, n_eval_eps = script_steps(info)
        n_seeds, n_robots, n_sets = len(seeds), len(robots), len(sets)
        for alg_idx, alg in enumerate(algs):
            for set_idx, env_set in enumerate(sets):
                for robot_idx, n in enumerate(robots):
                    for seed_idx, seed in enumerate(seeds):
                        array_id = seed_idx + n_seeds * (robot_idx + n_robots * (set_idx + n_sets * alg_idx))
                        specs.append(TrainSpec(group, script_name, array_id, alg, env_set, n, seed, "main_default", steps, log_steps, n_eval_eps))
    return specs


def _tune_script_defaults(script_name: str) -> tuple[list[str], list[str]]:
    """Return fallback algorithm/device order for each supported Step 3 layout."""
    if script_name == TUNE_COMBINED_SCRIPT:
        return TUNE_ALGS, ["cpu", "cpu", "cpu", "cpu", "cuda", "cuda"]
    if script_name == TUNE_SPLIT_CROSSQ_SCRIPT:
        return ["CrossQ"], ["cuda"]
    if script_name == TUNE_SPLIT_OTHERS_SCRIPT:
        # This order matches step3_others_tune_hyperparameters.sh.
        return ["A2C", "ARS", "PPO", "TQC", "TRPO"], ["cpu", "cpu", "cpu", "cpu", "cpu"]
    return TUNE_ALGS, ["cpu"] * len(TUNE_ALGS)


def _count_active_for_job_names(active_idx: dict[tuple[str, int], list[JobActivity]], job_names: set[str]) -> int:
    return sum(len(rows) for (job_name, _aid), rows in active_idx.items() if job_name in job_names)


def _count_logs_for_job_names(paths: Sequence[Path], job_names: set[str]) -> tuple[int, float]:
    count = 0
    newest = 0.0
    for p in paths:
        job_name, _jid, _aid = parse_log_identity(p)
        if job_name in job_names:
            count += 1
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    return count, newest


def detect_step3_layout(
    infos: dict[str, SlurmInfo],
    active_idx: dict[tuple[str, int], list[JobActivity]],
    stdout_files: Sequence[Path],
    stderr_files: Sequence[Path],
    requested_mode: str,
) -> Step3Layout:
    """Detect whether Step 3 is being run as one combined script or two split scripts.

    Priority order:
      1. explicit --step3-mode
      2. live squeue evidence
      3. Slurm log evidence, with newest log directory winning if both exist
      4. repository script availability
      5. split fallback when ambiguous, because split is the normal repo workflow
    """
    combined_scripts = (TUNE_COMBINED_SCRIPT,)
    split_scripts = (TUNE_SPLIT_CROSSQ_SCRIPT, TUNE_SPLIT_OTHERS_SCRIPT)
    combined_jobs = {script_job_name(infos, TUNE_COMBINED_SCRIPT)}
    split_jobs = {script_job_name(infos, TUNE_SPLIT_CROSSQ_SCRIPT), script_job_name(infos, TUNE_SPLIT_OTHERS_SCRIPT)}

    combined_active = _count_active_for_job_names(active_idx, combined_jobs)
    split_active = _count_active_for_job_names(active_idx, split_jobs)
    combined_logs_out, combined_newest_out = _count_logs_for_job_names(stdout_files, combined_jobs)
    combined_logs_err, combined_newest_err = _count_logs_for_job_names(stderr_files, combined_jobs)
    split_logs_out, split_newest_out = _count_logs_for_job_names(stdout_files, split_jobs)
    split_logs_err, split_newest_err = _count_logs_for_job_names(stderr_files, split_jobs)
    combined_logs = combined_logs_out + combined_logs_err
    split_logs = split_logs_out + split_logs_err
    combined_newest = max(combined_newest_out, combined_newest_err)
    split_newest = max(split_newest_out, split_newest_err)

    if requested_mode == "combined":
        warning = ""
        if split_active or split_logs:
            warning = "split Step 3 evidence also exists; combined was selected explicitly"
        return Step3Layout("combined", combined_scripts, "forced by --step3-mode combined", combined_active, split_active, combined_logs, split_logs, warning)
    if requested_mode == "split":
        warning = ""
        if combined_active or combined_logs:
            warning = "combined Step 3 evidence also exists; split is the normal/default layout and was selected"
        return Step3Layout("split", split_scripts, "forced by --step3-mode split", combined_active, split_active, combined_logs, split_logs, warning)

    warning = ""
    if combined_active and not split_active:
        return Step3Layout("combined", combined_scripts, "auto: active s3_tune jobs found", combined_active, split_active, combined_logs, split_logs)
    if split_active and not combined_active:
        return Step3Layout("split", split_scripts, "auto: active split Step 3 jobs found", combined_active, split_active, combined_logs, split_logs)
    if combined_active and split_active:
        if split_active > combined_active:
            warning = "both combined and split Step 3 jobs are active; using split because it has more active tasks"
            return Step3Layout("split", split_scripts, "auto: mixed active evidence", combined_active, split_active, combined_logs, split_logs, warning)
        warning = "both combined and split Step 3 jobs are active; using combined because it has at least as many active tasks"
        return Step3Layout("combined", combined_scripts, "auto: mixed active evidence", combined_active, split_active, combined_logs, split_logs, warning)

    if combined_logs and not split_logs:
        return Step3Layout("combined", combined_scripts, "auto: combined Step 3 Slurm logs found", combined_active, split_active, combined_logs, split_logs)
    if split_logs and not combined_logs:
        return Step3Layout("split", split_scripts, "auto: split Step 3 Slurm logs found", combined_active, split_active, combined_logs, split_logs)
    if combined_logs and split_logs:
        if split_newest > combined_newest:
            warning = "both combined and split Step 3 logs exist; using split because its newest log is newer"
            return Step3Layout("split", split_scripts, "auto: mixed log evidence", combined_active, split_active, combined_logs, split_logs, warning)
        warning = "both combined and split Step 3 logs exist; using combined because its newest log is newer or tied"
        return Step3Layout("combined", combined_scripts, "auto: mixed log evidence", combined_active, split_active, combined_logs, split_logs, warning)

    combined_available = infos.get(TUNE_COMBINED_SCRIPT) is not None and infos[TUNE_COMBINED_SCRIPT].path is not None
    split_available = all(infos.get(s) is not None and infos[s].path is not None for s in split_scripts)
    if split_available and not combined_available:
        return Step3Layout("split", split_scripts, "auto: only split Step 3 scripts exist", combined_active, split_active, combined_logs, split_logs)
    if combined_available and not split_available:
        return Step3Layout("combined", combined_scripts, "auto: only combined Step 3 script exists", combined_active, split_active, combined_logs, split_logs)
    if split_available and combined_available:
        warning = "no active jobs or logs found for Step 3 and both layouts exist; defaulting to split because split is the normal repo workflow"
        return Step3Layout("split", split_scripts, "auto: ambiguous, both layouts available", combined_active, split_active, combined_logs, split_logs, warning)
    warning = "no Step 3 Slurm script layout found; using split fallback grid"
    return Step3Layout("split", split_scripts, "auto: no scripts found", combined_active, split_active, combined_logs, split_logs, warning)


def build_tune_specs_for_script(script_name: str, info: SlurmInfo) -> list[TuneSpec]:
    fallback_algs, fallback_devices = _tune_script_defaults(script_name)
    algs = as_strs(info.arrays.get("algorithms", []), fallback_algs)
    devices = as_strs(info.arrays.get("devices", []), fallback_devices)
    total = info.array_count or len(algs) * DEFAULT_TUNE_TRIALS_PER_ALG
    expected = max(1, total // max(1, len(algs)))
    tune_steps = parse_int(info.assignments.get("--tune_steps"), DEFAULT_TUNE_STEPS) or DEFAULT_TUNE_STEPS
    journal_dir = info.assignments.get("JOURNAL_DIR", "logs/optuna_studies")
    best_json = info.assignments.get("BEST_JSON", "logs/best_hyperparams.json")
    specs: list[TuneSpec] = []
    for idx, alg in enumerate(algs):
        ids = tuple(range(idx * expected, idx * expected + expected))
        specs.append(TuneSpec(
            alg, idx, script_name, ids, expected,
            str(Path(journal_dir) / f"{alg}_journal.log"),
            f"{alg}_tune", best_json, tune_steps,
            DEFAULT_TUNE_SET, DEFAULT_TUNE_NUM_ROBOTS, DEFAULT_TUNE_SEED,
        ))
    return specs


def build_tune_specs(infos: dict[str, SlurmInfo], layout: Step3Layout) -> list[TuneSpec]:
    specs: list[TuneSpec] = []
    for script_name in layout.scripts:
        info = infos.get(script_name)
        if info is None:
            info = SlurmInfo(script_name, None, None, None, tuple(), None)
        specs.extend(build_tune_specs_for_script(script_name, info))
    order = {alg: i for i, alg in enumerate(TUNE_ALGS)}
    return sorted(specs, key=lambda s: order.get(s.algorithm, 999))

# ---------------------------------------------------------------------------
# Scheduler and log evidence
# ---------------------------------------------------------------------------

def script_job_name(infos: dict[str, SlurmInfo], script: str) -> str:
    info = infos.get(script)
    return info.job_name if info and info.job_name else Path(script).stem


def parse_job_array_ids(job_id: str) -> list[int | None]:
    m = re.search(r"_\[([^\]]+)\]", job_id)
    if m:
        return [int(x) for x in expand_ranges(m.group(1))]
    m = re.search(r"_(\d+)(?:\.|$)", job_id)
    if m:
        return [int(m.group(1))]
    return [None]


def workdir_matches(root: Path, work_dir: str) -> bool:
    """Return True when a Slurm job WorkDir belongs to this project root.

    This prevents same-user, same-job-name arrays from another clone of the
    repository from being counted as active/failed work for the current clone.
    """
    work_dir = (work_dir or "").strip()
    if not work_dir:
        return False
    try:
        root_abs = root.expanduser().resolve()
        wd_abs = Path(work_dir).expanduser().resolve()
        return wd_abs == root_abs
    except Exception:
        return os.path.abspath(os.path.expanduser(work_dir)) == os.path.abspath(str(root))


def query_squeue(job_names: set[str], disabled: bool, project_root: Path | None = None) -> dict[tuple[str, int], list[JobActivity]]:
    if disabled or not shutil.which("squeue"):
        return {}
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    cmd = ["squeue", "-r", "--noheader", "--format=%i|%j|%t|%M|%R|%Z"]
    if user:
        cmd.extend(["-u", user])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=8)
    except Exception:
        cmd = ["squeue", "--noheader", "--format=%i|%j|%t|%M|%R|%Z"]
        if user:
            cmd.extend(["-u", user])
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=8)
        except Exception:
            return {}
    by_key: dict[tuple[str, int], list[JobActivity]] = defaultdict(list)
    for line in out.splitlines():
        parts = line.split("|", 5)
        if len(parts) < 5:
            continue
        if len(parts) == 5:
            jid, name, state, elapsed, reason = [p.strip() for p in parts]
            work_dir = ""
        else:
            jid, name, state, elapsed, reason, work_dir = [p.strip() for p in parts]
        if name not in job_names:
            continue
        if project_root is not None and not workdir_matches(project_root, work_dir):
            continue
        for aid in parse_job_array_ids(jid):
            if aid is not None:
                by_key[(name, aid)].append(JobActivity("squeue", jid, name, aid, state, elapsed, reason, work_dir=work_dir))
    return by_key


def query_sacct(job_names: set[str], days: int, disabled: bool, project_root: Path | None = None) -> dict[tuple[str, int], list[JobActivity]]:
    if disabled or days <= 0 or not shutil.which("sacct"):
        return {}
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    cmd = ["sacct", "-n", "-P", "-X", "-S", start, "-o", "JobIDRaw,JobName%80,State,ExitCode,Elapsed,Reason,WorkDir%300"]
    if user:
        cmd.extend(["-u", user])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return {}
    by_key: dict[tuple[str, int], list[JobActivity]] = defaultdict(list)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|", 6)]
        if len(parts) < 6:
            continue
        jid, name, state, exit_code, elapsed, reason = parts[:6]
        work_dir = parts[6] if len(parts) > 6 else ""
        if "." in jid or name not in job_names:
            continue
        if project_root is not None and not workdir_matches(project_root, work_dir):
            continue
        for aid in parse_job_array_ids(jid):
            if aid is not None:
                by_key[(name, aid)].append(JobActivity("sacct", jid, name, aid, state, elapsed, reason, exit_code, work_dir))
    return by_key


def parse_log_identity(path: Path) -> tuple[str | None, str | None, int | None]:
    stem = path.stem
    m = re.match(r"(?P<job>.+)_(?P<jid>\d+)_(?P<aid>\d+)$", stem)
    if m:
        return m.group("job"), m.group("jid"), int(m.group("aid"))
    m = re.match(r"(?P<job>.+)_(?P<jid>\d+)$", stem)
    if m:
        return m.group("job"), m.group("jid"), None
    return None, None, None


def snippet(text: str, pat: re.Pattern[str] | None = None, max_lines: int = 5) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    if pat is not None:
        for i, line in enumerate(lines):
            if pat.search(line):
                start = max(0, i - 2)
                end = min(len(lines), i + max_lines)
                return " | ".join(x.strip() for x in lines[start:end] if x.strip())[:600]
    return " | ".join(x.strip() for x in lines[-max_lines:] if x.strip())[:600]


def classify_log(path: Path, stream: str) -> LogFinding:
    size = path.stat().st_size if path.exists() else 0
    mtime = path.stat().st_mtime if path.exists() else 0.0
    job, jid, aid = parse_log_identity(path)
    if size == 0:
        return LogFinding(str(path), stream, job, jid, aid, "empty", "empty", "", size, mtime)
    text = read_text(path)
    for pat, reason in FATAL_PATTERNS:
        if pat.search(text):
            return LogFinding(str(path), stream, job, jid, aid, "fatal", reason, snippet(text, pat), size, mtime)
    for pat, reason in WARN_PATTERNS:
        if pat.search(text):
            return LogFinding(str(path), stream, job, jid, aid, "warn", reason, snippet(text, pat), size, mtime)
    return LogFinding(str(path), stream, job, jid, aid, "notice", "non-empty", snippet(text), size, mtime)


def collect_logs(infos: dict[str, SlurmInfo], scripts: Sequence[str], *, scan_contents: bool = True) -> tuple[list[Path], list[Path], list[LogFinding], dict[tuple[str, int], list[Path]], dict[tuple[str, int], list[LogFinding]]]:
    import glob
    stdout_files: set[Path] = set()
    stderr_files: set[Path] = set()
    for script in scripts:
        info = infos.get(script)
        if not info:
            continue
        if info.output_glob:
            stdout_files.update(Path(p) for p in glob.glob(info.output_glob))
        if info.error_glob:
            stderr_files.update(Path(p) for p in glob.glob(info.error_glob))

    findings: list[LogFinding] = []
    if scan_contents:
        for p in sorted(stdout_files):
            if p.is_file():
                findings.append(classify_log(p, "stdout"))
        for p in sorted(stderr_files):
            if p.is_file():
                findings.append(classify_log(p, "stderr"))

    stdout_index: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for p in sorted(stdout_files):
        job, _jid, aid = parse_log_identity(p)
        if job and aid is not None:
            stdout_index[(job, aid)].append(p)
    finding_index: dict[tuple[str, int], list[LogFinding]] = defaultdict(list)
    for f in findings:
        if f.job_name and f.array_id is not None:
            finding_index[(f.job_name, f.array_id)].append(f)
    return sorted(stdout_files), sorted(stderr_files), findings, stdout_index, finding_index

# ---------------------------------------------------------------------------
# Artifact inspection
# ---------------------------------------------------------------------------

def load_npz(path: Path, spec: TrainSpec, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    metrics: dict[str, Any] = {}
    issues: list[str] = []
    warnings: list[str] = []
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        issues.append(f"cannot verify evaluations.npz because numpy is unavailable: {exc}")
        return metrics, issues, warnings
    try:
        with np.load(path, allow_pickle=True) as d:
            keys = set(d.files)
            metrics["npz_keys"] = sorted(keys)
            required = {"timesteps", "results"}
            if not args.allow_missing_ep_lengths:
                required.add("ep_lengths")
            missing = sorted(required - keys)
            if missing:
                issues.append("evaluations.npz missing required key(s): " + ", ".join(missing))
                if "timesteps" in missing or "results" in missing:
                    return metrics, issues, warnings
            if args.allow_missing_ep_lengths and "ep_lengths" not in keys:
                warnings.append("evaluations.npz missing ep_lengths")
            timesteps = np.asarray(d["timesteps"]).reshape(-1)
            results = np.asarray(d["results"])
            metrics["eval_count"] = int(len(timesteps))
            metrics["last_timestep"] = int(timesteps[-1]) if len(timesteps) else 0
            metrics["results_shape"] = tuple(int(x) for x in results.shape)
            if len(timesteps) == 0:
                issues.append("evaluations.npz has zero evaluation rows")
            if results.ndim != 2:
                issues.append(f"results should be 2-D evaluations×episodes, got shape {results.shape}")
            elif results.shape[0] != len(timesteps):
                issues.append(f"results rows {results.shape[0]} do not match timesteps {len(timesteps)}")
            elif results.shape[1] != spec.n_eval_eps:
                warnings.append(f"results has {results.shape[1]} eval episodes per row, expected {spec.n_eval_eps}")
            if "ep_lengths" in keys:
                ep_lengths = np.asarray(d["ep_lengths"])
                metrics["ep_lengths_shape"] = tuple(int(x) for x in ep_lengths.shape)
                if results.ndim == 2 and ep_lengths.shape != results.shape:
                    warnings.append(f"ep_lengths shape {ep_lengths.shape} does not match results shape {results.shape}")
            min_evals = math.ceil(spec.expected_eval_count * args.min_eval_fraction)
            min_steps = math.ceil(spec.expected_steps * args.min_timestep_fraction)
            if len(timesteps) < min_evals:
                issues.append(f"only {len(timesteps)}/{spec.expected_eval_count} expected evaluation rows")
            if len(timesteps) and int(timesteps[-1]) < min_steps:
                issues.append(f"last eval timestep {int(timesteps[-1]):,} < expected {spec.expected_steps:,}")
            if results.size:
                finite = np.isfinite(results.astype(float).reshape(-1))
                metrics["reward_finite_count"] = int(finite.sum())
                metrics["reward_nonfinite_count"] = int((~finite).sum())
                if not finite.any():
                    issues.append("all evaluation rewards are NaN/Inf")
                elif (~finite).any():
                    warnings.append(f"{int((~finite).sum())} evaluation reward values are NaN/Inf")
    except Exception as exc:
        issues.append(f"could not load evaluations.npz: {exc}")
    return metrics, issues, warnings


def read_progress(path: Path, spec: TrainSpec, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
    metrics: dict[str, Any] = {}
    issues: list[str] = []
    warnings: list[str] = []
    if not file_nonempty(path):
        (issues if args.strict_logs else warnings).append("progress.csv missing or empty")
        return metrics, issues, warnings
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            last = None
            rows = 0
            for row in reader:
                last = row
                rows += 1
            metrics["progress_rows"] = rows
            metrics["progress_columns"] = list(reader.fieldnames or [])[:30]
        if not last:
            (issues if args.strict_logs else warnings).append("progress.csv has no rows")
            return metrics, issues, warnings
        for col in ("time/total_timesteps", "total_timesteps"):
            if col in last:
                total = parse_int(last.get(col), -1) or -1
                metrics["progress_last_total_timesteps"] = total
                if total < math.ceil(spec.expected_steps * args.min_timestep_fraction):
                    (issues if args.strict_logs else warnings).append(f"progress.csv last timestep {total:,} < expected {spec.expected_steps:,}")
                break
    except Exception as exc:
        (issues if args.strict_logs else warnings).append(f"could not read progress.csv: {exc}")
    return metrics, issues, warnings


def inspect_train_artifacts(spec: TrainSpec, log_root: Path, args: argparse.Namespace) -> ArtifactResult:
    run_dir = log_root / spec.version / spec.tag
    best = run_dir / "best_model" / "best_model.zip"
    npz = run_dir / "eval_logs" / "evaluations.npz"
    final = run_dir / spec.final_model_name
    progress = run_dir / "progress.csv"
    log_txt = run_dir / "log.txt"
    issues: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "best_model": str(best),
        "eval_npz": str(npz),
        "final_model": str(final),
        "progress_csv": str(progress),
        "log_txt": str(log_txt),
        "expected_steps": spec.expected_steps,
        "expected_eval_count": spec.expected_eval_count,
        "expected_eval_episodes": spec.n_eval_eps,
    }
    if not run_dir.exists():
        issues.append("run directory missing")
        return ArtifactResult(issues, warnings, metrics)

    if getattr(args, "fast", False):
        # Fast mode is intentionally shallow: it answers "what evidence is
        # readily visible right now?" without replaying/validating large files.
        for label, path, min_bytes in [
            ("best_model/best_model.zip", best, args.min_model_bytes),
            (f"final model {spec.final_model_name}", final, args.min_model_bytes),
            ("eval_logs/evaluations.npz", npz, args.min_npz_bytes),
        ]:
            if not file_nonempty(path, min_bytes):
                issues.append(f"{label} missing/too small")
            else:
                key = label.replace("/", "_").replace(".", "_").replace(" ", "_") + "_bytes"
                metrics[key] = path.stat().st_size
        metrics["progress_csv_exists"] = progress.exists()
        metrics["log_txt_exists"] = log_txt.exists()
        metrics["tensorboard_event_seen"] = any(run_dir.glob("events.out.tfevents.*"))
        warnings.append("fast mode: shallow file-presence check only; skipped zip/NPZ/progress/TensorBoard validation")
        return ArtifactResult(issues, warnings, metrics)

    ok, why = is_zip_ok(best, args.min_model_bytes)
    if not ok:
        issues.append(f"best_model/best_model.zip {why}")
    else:
        metrics["best_model_bytes"] = best.stat().st_size
    ok, why = is_zip_ok(final, args.min_model_bytes)
    if not ok:
        issues.append(f"final model {spec.final_model_name} {why}")
    else:
        metrics["final_model_bytes"] = final.stat().st_size
    if not file_nonempty(npz, args.min_npz_bytes):
        issues.append("eval_logs/evaluations.npz missing/too small")
    else:
        metrics["eval_npz_bytes"] = npz.stat().st_size
        m, c, w = load_npz(npz, spec, args)
        metrics.update(m); issues.extend(c); warnings.extend(w)
    m, c, w = read_progress(progress, spec, args)
    metrics.update(m); issues.extend(c); warnings.extend(w)
    if not file_nonempty(log_txt):
        (issues if args.strict_logs else warnings).append("log.txt missing or empty")
    else:
        metrics["log_txt_bytes"] = log_txt.stat().st_size
    tb = [p for p in run_dir.rglob("events.out.tfevents.*") if file_nonempty(p)] if run_dir.exists() else []
    metrics["tensorboard_event_count"] = len(tb)
    if not tb:
        (issues if args.strict_tensorboard else warnings).append("TensorBoard event file missing or empty")
    return ArtifactResult(issues, warnings, metrics)

# ---------------------------------------------------------------------------
# Per-run checks
# ---------------------------------------------------------------------------

def activity_state(active: list[JobActivity]) -> str | None:
    if not active:
        return None
    states = {a.state.upper() for a in active}
    if states and states <= PENDING_STATES:
        return "pending"
    if states & RUNNING_STATES:
        return "running"
    return "active"


def fatal_from(sacct: list[JobActivity], logs: list[LogFinding]) -> list[str]:
    evidence: list[str] = []
    for r in sacct:
        state0 = r.state.upper().split()[0]
        exit_bad = bool(r.exit_code and not r.exit_code.startswith("0:"))
        if state0 in FATAL_SACCT_STATES or exit_bad:
            evidence.append(f"sacct {r.job_id}: state={r.state} exit={r.exit_code or '?'} reason={r.reason or '?'}")
    for f in logs:
        if f.severity == "fatal":
            evidence.append(f"{f.stream} {Path(f.path).name}: {f.reason}: {f.snippet}")
    return evidence


def check_train(spec: TrainSpec, log_root: Path, infos: dict[str, SlurmInfo], active_idx: dict[tuple[str, int], list[JobActivity]], sacct_idx: dict[tuple[str, int], list[JobActivity]], stdout_idx: dict[tuple[str, int], list[Path]], log_idx: dict[tuple[str, int], list[LogFinding]], args: argparse.Namespace) -> TrainCheck:
    artifact = inspect_train_artifacts(spec, log_root, args)
    job = script_job_name(infos, spec.script_name)
    key = (job, spec.array_id)
    active = active_idx.get(key, [])
    sacct = sacct_idx.get(key, [])
    stdout_paths = [str(p) for p in stdout_idx.get(key, [])]
    logs = log_idx.get(key, [])
    fatal = fatal_from(sacct, logs)
    act_state = activity_state(active)
    started = bool(stdout_paths or logs or sacct or Path(str(artifact.metrics.get("run_dir"))).exists())
    if artifact.ok:
        state = "verified_warn" if (artifact.warnings or any(l.severity == "warn" for l in logs) or fatal) else "verified_good"
    elif act_state:
        state = act_state
    elif fatal:
        state = "confirmed_failed"
    elif started:
        state = "finished_bad"
    else:
        state = "missing_finished" if args.assume_finished else "not_started"
    return TrainCheck(spec, state, artifact, active, sacct, logs, stdout_paths, fatal)

# ---------------------------------------------------------------------------
# Tuning checks
# ---------------------------------------------------------------------------

def load_best_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"best_hyperparams.json missing: {path}"]
    if not file_nonempty(path):
        return {}, [f"best_hyperparams.json empty: {path}"]
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}, [f"best_hyperparams.json root is {type(data).__name__}, expected object"]
        return data, []
    except Exception as exc:
        return {}, [f"could not parse best_hyperparams.json: {exc}"]


def inspect_journal(path: Path, study_name: str, args: argparse.Namespace | None = None) -> tuple[dict[str, Any], list[str], list[str]]:
    metrics: dict[str, Any] = {}
    issues: list[str] = []
    warnings: list[str] = []
    if not file_nonempty(path):
        issues.append("journal log missing or empty")
        return metrics, issues, warnings
    metrics["journal_bytes"] = path.stat().st_size
    if args is not None and getattr(args, "fast", False):
        warnings.append("fast mode: journal exists, but Optuna replay/trial counts were skipped")
        return metrics, issues, warnings
    try:
        import optuna  # type: ignore
        from optuna.storages import JournalStorage  # type: ignore
        from optuna.storages.journal import JournalFileBackend  # type: ignore
        storage = JournalStorage(JournalFileBackend(str(path)))
        study = optuna.load_study(study_name=study_name, storage=storage)
        trials = list(study.trials)
        states = Counter(str(t.state).split(".")[-1] for t in trials)
        complete = [t for t in trials if str(t.state).endswith("COMPLETE")]
        finite: list[float] = []
        nonfinite = 0
        for t in complete:
            try:
                v = float(t.value) if t.value is not None else float("nan")
                if math.isfinite(v):
                    finite.append(v)
                else:
                    nonfinite += 1
            except Exception:
                nonfinite += 1
        metrics.update({
            "journal_trials_total": len(trials),
            "journal_state_counts": dict(states),
            "journal_trials_complete": len(complete),
            "journal_complete_finite_values": len(finite),
            "journal_complete_nonfinite_values": nonfinite,
            "journal_best_value": max(finite) if finite else None,
        })
        if nonfinite:
            warnings.append(f"{nonfinite} complete trial value(s) are NaN/Inf/-Inf")
        running_like = states.get("RUNNING", 0) + states.get("WAITING", 0)
        if running_like:
            warnings.append(f"journal contains {running_like} RUNNING/WAITING trial(s)")
    except Exception as exc:
        text = read_text(path, max_bytes=3_000_000)
        ids = {int(m.group(1)) for m in re.finditer(r'"trial_id"\s*:\s*(\d+)', text)}
        if ids:
            metrics["journal_trials_total"] = len(ids)
            warnings.append(f"Optuna import/load unavailable; fallback counted {len(ids)} unique trial_id values only: {exc}")
        else:
            issues.append(f"could not inspect Optuna journal: {exc}")
    return metrics, issues, warnings


def validate_best_for_alg(spec: TuneSpec, data: dict[str, Any], base_errors: list[str]) -> tuple[dict[str, Any], list[str], list[str]]:
    metrics: dict[str, Any] = {}
    issues = list(base_errors)
    warnings: list[str] = []
    if base_errors:
        return metrics, issues, warnings
    rec = data.get(spec.algorithm)
    if rec is None:
        issues.append(f"best_hyperparams.json missing algorithm entry: {spec.algorithm}")
        return metrics, issues, warnings
    if not isinstance(rec, dict):
        issues.append(f"best_hyperparams.json[{spec.algorithm}] is not an object")
        return metrics, issues, warnings
    params = rec.get("params")
    if not isinstance(params, dict):
        issues.append(f"best_hyperparams.json[{spec.algorithm}].params missing/not object")
    else:
        missing = sorted(HP_REQUIRED_KEYS.get(spec.algorithm, set()) - set(params))
        metrics["best_json_param_keys"] = sorted(params)
        if missing:
            issues.append(f"best_hyperparams.json[{spec.algorithm}].params missing HP(s): {', '.join(missing)}")
    try:
        iqm = float(rec.get("iqm"))
        metrics["best_json_iqm"] = iqm
        if not math.isfinite(iqm):
            issues.append(f"best_hyperparams.json[{spec.algorithm}].iqm is not finite")
    except Exception:
        issues.append(f"best_hyperparams.json[{spec.algorithm}].iqm missing/not numeric")
    ctx = rec.get("context")
    if isinstance(ctx, dict):
        metrics["best_json_context"] = ctx
        expected = {"set": spec.tune_set, "num_robots": spec.tune_num_robots, "tune_seed": spec.tune_seed, "tune_steps": spec.tune_steps}
        for k, v in expected.items():
            if ctx.get(k) != v:
                warnings.append(f"best_hyperparams.json[{spec.algorithm}].context.{k}={ctx.get(k)!r}, expected {v!r}")
    else:
        warnings.append(f"best_hyperparams.json[{spec.algorithm}].context missing")
    return metrics, issues, warnings


def check_tune(spec: TuneSpec, root: Path, infos: dict[str, SlurmInfo], active_idx: dict[tuple[str, int], list[JobActivity]], sacct_idx: dict[tuple[str, int], list[JobActivity]], stdout_idx: dict[tuple[str, int], list[Path]], log_idx: dict[tuple[str, int], list[LogFinding]], best_data: dict[str, Any], best_errors: list[str], args: argparse.Namespace) -> TuneCheck:
    storage = rel_or_abs(root, spec.storage_rel)
    best_path = rel_or_abs(root, spec.output_json_rel)
    metrics: dict[str, Any] = {"storage_path": str(storage), "best_json_path": str(best_path), "expected_trials": spec.expected_trials}
    issues: list[str] = []
    warnings: list[str] = []
    m, c, w = inspect_journal(storage, spec.study_name, args)
    metrics.update(m); issues.extend(c); warnings.extend(w)
    m, c, w = validate_best_for_alg(spec, best_data, best_errors)
    metrics.update(m); issues.extend(c); warnings.extend(w)

    complete = metrics.get("journal_trials_complete")
    finite = metrics.get("journal_complete_finite_values")
    total = metrics.get("journal_trials_total")
    if getattr(args, "fast", False):
        # In fast mode, journal existence + best_json validity is enough for a
        # shallow status.  Trial-count verification belongs to normal/done/strict.
        pass
    elif isinstance(complete, int):
        if complete < spec.expected_trials:
            issues.append(f"only {complete}/{spec.expected_trials} complete Optuna trials")
        if isinstance(finite, int) and finite < spec.expected_trials:
            issues.append(f"only {finite}/{spec.expected_trials} complete trials have finite objective values")
        if isinstance(finite, int) and finite > spec.expected_trials:
            warnings.append(f"{finite}/{spec.expected_trials} finite trials found; possible duplicate/over-budget tuning")
    elif isinstance(total, int):
        if total < spec.expected_trials:
            issues.append(f"only {total}/{spec.expected_trials} journal trials; exact COMPLETE count unavailable")
        else:
            warnings.append("journal trial count exists but COMPLETE/finite count unavailable without Optuna")
    else:
        issues.append(f"cannot confirm {spec.expected_trials} tuning trials")

    job = script_job_name(infos, spec.script_name)
    active: list[JobActivity] = []
    sacct: list[JobActivity] = []
    logs: list[LogFinding] = []
    stdout_paths: list[str] = []
    for aid in spec.array_ids:
        key = (job, aid)
        active.extend(active_idx.get(key, []))
        sacct.extend(sacct_idx.get(key, []))
        logs.extend(log_idx.get(key, []))
        stdout_paths.extend(str(p) for p in stdout_idx.get(key, []))
    fatal = fatal_from(sacct, logs)
    act_state = activity_state(active)
    started = bool(file_nonempty(storage) or stdout_paths or logs or sacct or best_path.exists())
    if not issues and not act_state:
        state = "verified_warn" if warnings or fatal else "verified_good"
    elif act_state:
        state = act_state
    elif fatal:
        state = "confirmed_failed"
    elif started:
        state = "finished_bad"
    else:
        state = "missing_finished" if args.assume_finished else "not_started"

    suggested: list[int] = []
    if state == "not_started":
        suggested = list(spec.array_ids)
    elif state in {"confirmed_failed", "missing_finished"} or (state == "finished_bad" and args.rerun_finished_bad):
        done = complete if isinstance(complete, int) else 0
        missing_n = max(1, spec.expected_trials - int(done)) if spec.expected_trials else 1
        suggested = list(spec.array_ids[:missing_n])
    return TuneCheck(spec, state, issues, warnings, metrics, active, sacct, logs, stdout_paths, fatal, suggested)

# ---------------------------------------------------------------------------
# Summaries and next action logic
# ---------------------------------------------------------------------------

def train_group_summary(checks: list[TrainCheck]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    by_group: dict[str, list[TrainCheck]] = defaultdict(list)
    for c in checks:
        by_group[c.spec.group].append(c)
    for group, cs in sorted(by_group.items()):
        cnt = Counter(c.state for c in cs)
        verified = cnt["verified_good"] + cnt["verified_warn"]
        active = cnt["running"] + cnt["pending"] + cnt["active"]
        bad = cnt["confirmed_failed"] + cnt["finished_bad"] + cnt["missing_finished"]
        warnings = sum(1 for c in cs if c.artifact.warnings or c.state == "verified_warn")
        evals = [int(c.artifact.metrics.get("eval_count", 0)) for c in cs if c.verified]
        last_ts = [int(c.artifact.metrics.get("last_timestep", 0)) for c in cs if c.verified]
        issues = Counter()
        for c in cs:
            for x in c.artifact.critical:
                issues[x.split(":", 1)[0]] += 1
            for x in c.fatal_evidence:
                issues["fatal evidence"] += 1
        top = "; ".join(f"{k} ({v})" for k, v in issues.most_common(2)) or "---"
        rows.append([group, len(cs), cnt["verified_good"], cnt["verified_warn"], active, bad, cnt["not_started"], warnings, pct(verified, len(cs)), median_or_dash(evals), median_or_dash(last_ts), top])
    return rows


def tune_summary(checks: list[TuneCheck]) -> list[list[Any]]:
    rows = []
    for c in checks:
        m = c.metrics
        active = len({a.array_id for a in c.active if a.array_id is not None})
        best = m.get("best_json_iqm", m.get("journal_best_value"))
        best_s = f"{best:.3f}" if isinstance(best, (float, int)) and math.isfinite(float(best)) else "---"
        issue = "; ".join((c.issues + c.warnings + c.fatal_evidence)[:2]) or "---"
        rows.append([c.spec.algorithm, c.spec.expected_trials, m.get("journal_trials_complete", "?"), m.get("journal_complete_finite_values", "?"), m.get("journal_trials_total", "?"), active, best_s, c.state, issue])
    return rows


def fatal_log_summary(findings: list[LogFinding], limit: int) -> tuple[list[list[Any]], list[list[Any]]]:
    relevant = [f for f in findings if f.severity in {"fatal", "warn"}]
    counts = Counter((f.severity, f.reason) for f in relevant)
    summary = [[sev, reason, count] for (sev, reason), count in counts.most_common()]
    fatal = [f for f in relevant if f.severity == "fatal"][:limit]
    details = [[Path(f.path).parent.name, Path(f.path).name, f.array_id if f.array_id is not None else "?", f.reason, f.snippet] for f in fatal]
    return summary, details


def build_commands(items: dict[tuple[str, tuple[str, ...]], list[int]], script_dir: str, limit: int, chunk_size: int) -> list[str]:
    commands: list[str] = []
    for (script, args), ids in sorted(items.items()):
        for chunk in (sorted(set(ids))[i:i + chunk_size] for i in range(0, len(set(ids)), chunk_size)):
            path = f"{script_dir.rstrip('/')}/{script}" if script_dir else script
            arg_s = " " + " ".join(shlex.quote(a) for a in args) if args else ""
            commands.append(f"sbatch --array={compact_ranges(chunk)} {path}{arg_s}")
    return commands[:limit]


def collect_action_items(train_checks: list[TrainCheck], tune_checks: list[TuneCheck], args: argparse.Namespace) -> tuple[dict, dict, dict]:
    rerun: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    investigate: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    launch: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    for c in train_checks:
        if c.state in {"confirmed_failed", "missing_finished"} or (c.state == "finished_bad" and args.rerun_finished_bad):
            rerun[c.spec.command_key].append(c.spec.array_id)
        elif c.state == "finished_bad":
            investigate[c.spec.command_key].append(c.spec.array_id)
        elif c.state == "not_started":
            launch[c.spec.command_key].append(c.spec.array_id)
    for c in tune_checks:
        if c.suggested_ids:
            if c.state in {"confirmed_failed", "missing_finished"} or (c.state == "finished_bad" and args.rerun_finished_bad):
                rerun[c.spec.command_key].extend(c.suggested_ids)
            elif c.state == "not_started":
                launch[c.spec.command_key].extend(c.suggested_ids)
        elif c.state == "finished_bad":
            investigate[c.spec.command_key].extend(c.spec.array_ids)
    return rerun, investigate, launch


def readiness(train_checks: list[TrainCheck], tune_checks: list[TuneCheck]) -> list[list[Any]]:
    # Step 2 uses Step 1 env1 source checkpoint for same alg/N/seed.
    source: dict[tuple[str, int, int], TrainCheck] = {}
    for c in train_checks:
        if c.spec.env_set == 1 and c.spec.version == "main_default":
            source[(c.spec.algorithm, c.spec.num_robots, c.spec.seed)] = c

    rows: list[list[Any]] = []
    for label, algs, target_jobs in [
        ("Step 2 CrossQ transfer", ["CrossQ"], 180),
        ("Step 2 other-alg transfer", MAIN_OTHER_ALGS, 900),
    ]:
        total_sources = len(algs) * len(ROBOTS) * len(SEEDS)
        ready = 0
        blockers = Counter()
        for alg in algs:
            for n in ROBOTS:
                for seed in SEEDS:
                    st = source.get((alg, n, seed))
                    if st and st.verified:
                        ready += 1
                    else:
                        blockers[st.state if st else "missing"] += 1
        target_ready = ready * len(range(2, 11))
        rows.append([label, f"{ready}/{total_sources} source checkpoints", f"{target_ready}/{target_jobs} target jobs unblocked", "YES" if ready == total_sources else "NO", "; ".join(f"{k}={v}" for k, v in blockers.items()) or "---"])

    tune_by_alg = {c.spec.algorithm: c for c in tune_checks}
    for label, algs, target_jobs in [
        ("Step 4 CrossQ tuned training", ["CrossQ"], 200),
        ("Step 4 other-alg tuned training", STEP4_OTHER_ALGS, 1000),
    ]:
        ready_algs = [alg for alg in algs if tune_by_alg.get(alg) and tune_by_alg[alg].verified]
        blockers = [f"{alg}:{tune_by_alg[alg].state if alg in tune_by_alg else 'missing'}" for alg in algs if alg not in ready_algs]
        rows.append([label, f"{len(ready_algs)}/{len(algs)} HP sets", f"{len(ready_algs) * 200}/{target_jobs} target jobs unblocked", "YES" if len(ready_algs) == len(algs) else "NO", "; ".join(blockers) or "---"])

    rows.append(["Ablation training", "no dependency on Step 1/3", "not checked here", "DEPENDENCY OK", "submit separately if desired"])
    rows.append(["Evaluation / analysis", "requires downstream outputs", "not checked here", "NO", "do not use this script as eval/analysis readiness"])
    return rows


def detailed_bad_rows(train_checks: list[TrainCheck], tune_checks: list[TuneCheck], limit: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    interesting_train = [c for c in train_checks if c.state in {"confirmed_failed", "finished_bad", "missing_finished", "running", "pending", "active"}]
    order = {"confirmed_failed": 0, "finished_bad": 1, "missing_finished": 2, "running": 3, "pending": 4, "active": 5}
    interesting_train.sort(key=lambda c: (order.get(c.state, 99), c.spec.script_name, c.spec.array_id))
    for c in interesting_train[:limit]:
        issue = "; ".join((c.fatal_evidence + c.artifact.critical + c.artifact.warnings)[:2]) or "---"
        progress = f"evals={c.artifact.metrics.get('eval_count', '---')}, ts={human(c.artifact.metrics.get('last_timestep'))}"
        rows.append(["train", c.state, c.spec.script_name, c.spec.array_id, c.spec.tag, progress, issue])
    remaining = limit - len(rows)
    if remaining > 0:
        interesting_tune = [c for c in tune_checks if c.state not in {"verified_good", "verified_warn", "not_started"}]
        for c in interesting_tune[:remaining]:
            issue = "; ".join((c.fatal_evidence + c.issues + c.warnings)[:2]) or "---"
            progress = f"complete={c.metrics.get('journal_trials_complete', '?')}, finite={c.metrics.get('journal_complete_finite_values', '?')}"
            rows.append(["tune", c.state, c.spec.script_name, compact_ranges(c.spec.array_ids), c.spec.algorithm, progress, issue])
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print-only status checker for Step 1 default training and split Step 3 tuning.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=QUICK_GUIDE,
    )
    p.add_argument("--guide", action="store_true", help="Print the concise command guide and exit.")
    p.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT_NAME,
                   help=f"Project root. Default: {DEFAULT_PROJECT_ROOT_NAME}; if already inside the repo, cwd is used.")
    p.add_argument("--log_root", default="logs")
    p.add_argument("--script_dir", default="slurm")

    # Simple depth presets.  The normal/default command is intentionally the
    # standard workflow: Step 1 + split Step 3, with workdir-safe Slurm filtering.
    p.add_argument("--fast", action="store_true",
                   help="Fast shallow check: squeue + file presence; skips sacct, log-content scans, zip/NPZ tests, and Optuna replay.")
    p.add_argument("--done", dest="assume_finished", action="store_true",
                   help="Alias for --assume-finished: missing work is bad/actionable.")
    p.add_argument("--assume-finished", dest="assume_finished", action="store_true",
                   help="Treat not-started/missing work as bad/actionable; useful after all jobs should be done.")
    p.add_argument("--strict", action="store_true",
                   help="Alias for --strict-logs --strict-tensorboard.")
    p.add_argument("--forensic", action="store_true",
                   help="Done + strict + wider detail limits + longer sacct history.")

    # Compatibility / deeper audit options.
    p.add_argument("--rerun-finished-bad", action="store_true", help="Include finished_bad runs in rerun commands. Default: investigate first.")
    p.add_argument("--allow-missing-ep-lengths", action="store_true", help="Warn instead of failing when evaluations.npz lacks ep_lengths.")
    p.add_argument("--strict-logs", action="store_true", help="Treat missing progress.csv/log.txt as bad, not warning.")
    p.add_argument("--strict-tensorboard", action="store_true", help="Treat missing TensorBoard event files as bad, not warning.")
    p.add_argument("--min-timestep-fraction", type=float, default=1.0)
    p.add_argument("--min-eval-fraction", type=float, default=1.0)
    p.add_argument("--min-model-bytes", type=int, default=1024)
    p.add_argument("--min-npz-bytes", type=int, default=512)
    p.add_argument("--details-limit", type=int, default=40)
    p.add_argument("--error-details-limit", type=int, default=25)
    p.add_argument("--commands-limit", type=int, default=30)
    p.add_argument("--rerun-chunk-size", type=int, default=1000)
    p.add_argument("--step3-mode", choices=["auto", "combined", "split"], default="split",
                   help="Step 3 Slurm layout. Default: split. Use combined only if you submitted step3_tune_hyperparameters.sh.")
    p.add_argument("--no-squeue", action="store_true")
    p.add_argument("--no-sacct", action="store_true")
    p.add_argument("--no-slurm-workdir-filter", action="store_true",
                   help="Do not restrict squeue/sacct records to jobs whose Slurm WorkDir matches --project_root. Use only if your cluster does not report WorkDir correctly.")
    p.add_argument("--sacct-days", type=int, default=14)

    args = p.parse_args()

    if args.guide:
        return args

    if args.strict:
        args.strict_logs = True
        args.strict_tensorboard = True
    if args.forensic:
        args.assume_finished = True
        args.strict_logs = True
        args.strict_tensorboard = True
        args.sacct_days = max(args.sacct_days, 60)
        args.details_limit = max(args.details_limit, 120)
        args.error_details_limit = max(args.error_details_limit, 80)
        args.commands_limit = max(args.commands_limit, 80)
    if args.fast:
        args.no_sacct = True
        # Keep squeue unless explicitly disabled; it is usually quick and is the
        # most useful live signal.  Keep missing-work semantics unchanged unless
        # the user also asks for --done.
        args.details_limit = min(args.details_limit, 25)
        args.error_details_limit = min(args.error_details_limit, 10)

    return args

def main() -> int:
    args = parse_args()
    if args.guide:
        print(QUICK_GUIDE)
        return 0

    started = time.time()
    root = resolve_project_root(args.project_root)
    log_root_arg = Path(args.log_root).expanduser()
    log_root = log_root_arg if log_root_arg.is_absolute() else root / log_root_arg

    infos = {name: parse_slurm(root, name) for name in ALL_SCRIPTS}
    train_specs = build_train_specs(infos)

    # Step 3 can be submitted either as one combined array or as two split
    # arrays.  Query/scan all candidate Step 3 job names first, then select the
    # active layout from current Slurm evidence, logs, or script availability.
    candidate_scripts = TRAIN_SCRIPTS + TUNE_CANDIDATE_SCRIPTS
    candidate_job_names = {script_job_name(infos, s) for s in candidate_scripts}
    slurm_project_root = None if args.no_slurm_workdir_filter else root
    active_idx = query_squeue(candidate_job_names, args.no_squeue, slurm_project_root)
    sacct_idx = query_sacct(candidate_job_names, args.sacct_days, args.no_sacct, slurm_project_root)
    stdout_all, stderr_all, findings_all, stdout_idx, log_idx = collect_logs(infos, candidate_scripts, scan_contents=not args.fast)

    step3_layout = detect_step3_layout(infos, active_idx, stdout_all, stderr_all, args.step3_mode)
    tune_specs = build_tune_specs(infos, step3_layout)
    checked_scripts = TRAIN_SCRIPTS + list(step3_layout.scripts)
    checked_job_names = {script_job_name(infos, s) for s in checked_scripts}
    stdout_files = [p for p in stdout_all if parse_log_identity(p)[0] in checked_job_names]
    stderr_files = [p for p in stderr_all if parse_log_identity(p)[0] in checked_job_names]
    all_findings = [f for f in findings_all if f.job_name in checked_job_names]

    print("\n" + "=" * 104)
    print(" train/tune status check — Step 1 default training + Step 3 tuning only")
    print("=" * 104)
    print(f"  Project root : {root}")
    print(f"  Log root     : {log_root}")
    print(f"  Slurm scope  : {'all same-name jobs for this user' if args.no_slurm_workdir_filter else 'jobs whose WorkDir matches project root'}")
    print("  Scope        : check submitted training/tuning jobs; print what finished, failed, is active, and what to do next")
    print("  No files     : this script prints only; it does not write rerun scripts or reports")
    print("  Proceed gates: Step 2 transfer and Step 4 tuned training readiness")
    print(f"  Step 3 mode  : {step3_layout.mode} ({step3_layout.reason})")
    if step3_layout.warning:
        print(f"  Step 3 note  : {step3_layout.warning}")
    if args.fast:
        print("  Mode         : --fast enabled (shallow, no sacct/log-content/zip/NPZ/Optuna replay)")
    if args.assume_finished:
        print("  Mode         : --done / --assume-finished enabled")
    if args.strict_logs or args.strict_tensorboard:
        strict_bits = []
        if args.strict_logs:
            strict_bits.append("logs")
        if args.strict_tensorboard:
            strict_bits.append("tensorboard")
        print(f"  Strict       : {', '.join(strict_bits)}")

    print("\nExpected train/tune work")
    rows = [
        ["Step 1 CrossQ default", "step1_crossq_default.sh", infos["step1_crossq_default.sh"].array_count or "?", 200],
        ["Step 1 other algorithms default", "step1_others_default.sh", infos["step1_others_default.sh"].array_count or "?", 1000],
    ]
    for script in step3_layout.scripts:
        script_specs = [t for t in tune_specs if t.script_name == script]
        if script == TUNE_COMBINED_SCRIPT:
            label = "Step 3 tuning (combined)"
        elif script == TUNE_SPLIT_CROSSQ_SCRIPT:
            label = "Step 3 tuning (CrossQ split)"
        elif script == TUNE_SPLIT_OTHERS_SCRIPT:
            label = "Step 3 tuning (other algs split)"
        else:
            label = "Step 3 tuning"
        rows.append([label, script, infos[script].array_count or "?", sum(t.expected_trials for t in script_specs)])
    print_table(["scope", "script", "Slurm array", "expected outputs/trials"], rows)

    print("\nScheduler/log evidence loaded")
    print_table(["source", "records", "meaning"], [
        ["squeue", sum(len(v) for v in active_idx.values()), "active/pending tasks currently visible" if active_idx else "none found or unavailable"],
        ["sacct", sum(len(v) for v in sacct_idx.values()), "skipped" if args.no_sacct else (f"last {args.sacct_days} day(s), if available" if sacct_idx else "none found or unavailable")],
        ["stdout", len(stdout_files), "Slurm .out files for checked scripts"],
        ["stderr", len(stderr_files), "Slurm .err files for checked scripts"],
        ["log scan", 0 if args.fast else len(all_findings), "skipped by --fast" if args.fast else "stdout/stderr content scanned for fatal/warn patterns"],
    ])

    train_checks = [check_train(s, log_root, infos, active_idx, sacct_idx, stdout_idx, log_idx, args) for s in train_specs]
    best_path = rel_or_abs(root, tune_specs[0].output_json_rel) if tune_specs else root / "logs/best_hyperparams.json"
    best_data, best_errors = load_best_json(best_path)
    tune_checks = [check_tune(s, root, infos, active_idx, sacct_idx, stdout_idx, log_idx, best_data, best_errors, args) for s in tune_specs]

    print("\nTraining status summary")
    print_table([
        "group", "expected", "good", "warn", "active", "bad", "not started", "warn runs", "verified %", "med evals", "med last ts", "top issue"
    ], train_group_summary(train_checks), max_col_width=64)

    print("\nTuning status summary")
    print_table(["alg", "expected", "complete", "finite", "journal", "active", "best IQM", "state", "issue/evidence"], tune_summary(tune_checks), max_col_width=72)

    print("\nSlurm error/warning summary")
    if args.fast:
        print("  Skipped by --fast. Use Normal/Done-check/Strict/Forensic to scan stdout/stderr contents.")
    else:
        err_summary, err_details = fatal_log_summary(all_findings, args.error_details_limit)
        print_table(["severity", "reason", "count"], err_summary, max_col_width=56)
        if err_details:
            print("\nFatal log details")
            print_table(["log dir", "file", "array", "reason", "snippet"], err_details, max_col_width=88)
        else:
            print("  No fatal log patterns found in checked stdout/stderr files.")

    print("\nReadiness to continue")
    print_table(["next step", "dependency coverage", "target coverage", "OK?", "blockers / note"], readiness(train_checks, tune_checks), max_col_width=80)

    print("\nDetailed active / failed / bad items")
    details = detailed_bad_rows(train_checks, tune_checks, args.details_limit)
    print_table(["kind", "state", "script", "array", "decoded run / alg", "progress", "issue/evidence"], details, max_col_width=80)

    rerun, investigate, launch = collect_action_items(train_checks, tune_checks, args)
    rerun_cmds = build_commands(rerun, args.script_dir, args.commands_limit, args.rerun_chunk_size)
    investigate_cmds = build_commands(investigate, args.script_dir, args.commands_limit, args.rerun_chunk_size)
    launch_cmds = build_commands(launch, args.script_dir, args.commands_limit, args.rerun_chunk_size)

    print("\nWhat to do next")
    if rerun_cmds:
        print("  Confirmed rerun commands:")
        for cmd in rerun_cmds:
            print("    " + cmd)
    else:
        print("  Confirmed rerun commands: none")
    if investigate_cmds:
        print("  Investigate before rerun (not automatically confirmed failures):")
        for cmd in investigate_cmds:
            print("    # inspect first, then optionally rerun:")
            print("    " + cmd)
    else:
        print("  Investigation-before-rerun commands: none")
    if launch_cmds:
        print("  Optional launch/continue commands for not-started work:")
        for cmd in launch_cmds:
            print("    " + cmd)
    else:
        print("  Optional launch/continue commands: none")

    # Concise final summary.
    train_state = Counter(c.state for c in train_checks)
    tune_state = Counter(c.state for c in tune_checks)
    train_verified = train_state["verified_good"] + train_state["verified_warn"]
    train_active = train_state["running"] + train_state["pending"] + train_state["active"]
    train_bad = train_state["confirmed_failed"] + train_state["finished_bad"] + train_state["missing_finished"]
    tune_verified = tune_state["verified_good"] + tune_state["verified_warn"]
    tune_active = tune_state["running"] + tune_state["pending"] + tune_state["active"]
    tune_bad = tune_state["confirmed_failed"] + tune_state["finished_bad"] + tune_state["missing_finished"]
    fatal_logs = sum(1 for f in all_findings if f.severity == "fatal")
    if train_verified == len(train_checks) and tune_verified == len(tune_checks) and fatal_logs == 0:
        if args.fast:
            overall = "FAST CHECK PASSED (SHALLOW; RUN NORMAL/DONE FOR PROOF)"
        else:
            overall = "READY FOR TRANSFER + TUNED TRAINING"
        code = 0
    elif train_active or tune_active:
        overall = "STILL RUNNING / PENDING"
        code = 1
    elif train_bad or tune_bad or fatal_logs:
        overall = "ATTENTION REQUIRED"
        code = 2
    else:
        overall = "INCOMPLETE / NOT STARTED"
        code = 2

    print("\n" + "=" * 104)
    print(" Concise final summary")
    print("=" * 104)
    print_table(["area", "verified", "active", "bad", "not started", "warnings"], [
        ["training", f"{train_verified}/{len(train_checks)}", train_active, train_bad, train_state["not_started"], train_state["verified_warn"]],
        ["tuning", f"{tune_verified}/{len(tune_checks)}", tune_active, tune_bad, tune_state["not_started"], tune_state["verified_warn"]],
    ])
    fatal_display = "skipped" if args.fast else f"{fatal_logs:,}"
    print(f"  Fatal log files : {fatal_display}")
    print(f"  Rerun commands  : {len(rerun_cmds):,}")
    print(f"  Overall         : {overall}")
    print(f"  Elapsed         : {time.time() - started:.1f}s")
    print("=" * 104 + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
