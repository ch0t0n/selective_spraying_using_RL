#!/usr/bin/env python3
"""
check_status_ablation_dr.py

Purpose
-------
Command-line-only status checker for the ablation/domain-randomization
training jobs:

  checked as completed/status scopes
    Step 5 reward ablation training
    Step 6 observation ablation training
    Step 7 uncertainty ablation training
    Step 8 domain-randomization training

  checked as readiness gates only
    reward-ablation evaluation can start when Step 5 training is good
    observation-ablation analysis can start when Step 6 training is good
    uncertainty cross-evaluation can start when Step 7 training is good
    DR evaluation / wind sweep / DR plots can start when needed Step 8 sources are good

  not checked as completed-output scopes
    eval_ablations.sh outputs, eval_wind_sweep.sh outputs, merged CSVs,
    analyze_results.py outputs, plot outputs, sensitivity jobs

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
# Fixed fallback workflow. Script parsing overrides these when possible.
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT_NAME = "selective_spraying_using_RL"
ALGORITHM = "CrossQ"
SEEDS = [0, 42, 123, 2024, 9999]
SETS = list(range(1, 11))
ROBOTS_DR = [2, 3, 4, 5]
N_ABLATION = 3
DEFAULT_STEPS = 2_000_000
DEFAULT_LOG_STEPS = 10_000
DEFAULT_N_EVAL_EPS = 5

ABLATION_SCRIPT = "step5_6_7_ablations.sh"
DR_SCRIPT = "step8_dr.sh"
ALL_SCRIPTS = [ABLATION_SCRIPT, DR_SCRIPT]
COMMAND_ORDER = {
    (ABLATION_SCRIPT, ("ablation_reward",)): 0,
    (ABLATION_SCRIPT, ("ablation_obs",)): 1,
    (ABLATION_SCRIPT, ("ablation_uncertainty",)): 2,
    (DR_SCRIPT, tuple()): 3,
}

ABLATION_EXPERIMENTS = {
    "ablation_reward": {
        "step": "Step 5 reward ablation",
        "array_name": "conditions",
        "conditions": ["full", "no_term", "no_spr", "no_path"],
        "echo_regex": re.compile(r"S5-ablation-reward\s*\|\s*condition=(\S+)\s*\|\s*set=(\d+)\s*\|\s*seed=(\d+)", re.I),
    },
    "ablation_obs": {
        "step": "Step 6 observation ablation",
        "array_name": "obs_modes",
        "conditions": ["full", "no_pos", "no_inf_hist", "pos_only"],
        "echo_regex": re.compile(r"S6-ablation-obs\s*\|\s*obs_mode=(\S+)\s*\|\s*set=(\d+)\s*\|\s*seed=(\d+)", re.I),
    },
    "ablation_uncertainty": {
        "step": "Step 7 uncertainty ablation",
        "array_name": "uncertainty_modes",
        "conditions": ["full", "wind_only", "act_only", "deterministic"],
        "echo_regex": re.compile(r"S7-ablation-uncertainty\s*\|\s*mode=(\S+)\s*\|\s*set=(\d+)\s*\|\s*seed=(\d+)", re.I),
    },
}

DR_MODES = ["none", "wind", "full"]
DR_ECHO_RE = re.compile(
    r"S8-DR\s*\|\s*dr_mode=(\S+)\s*\|\s*set=(\d+)\s*\|\s*robots=(\d+)\s*\|\s*seed=(\d+)",
    re.I,
)

QUICK_GUIDE = f"""
CHECK_STATUS_ABLATION_DR QUICK GUIDE

Normal workflow checked by default:
  Step 5: sbatch slurm/step5_6_7_ablations.sh ablation_reward
  Step 6: sbatch slurm/step5_6_7_ablations.sh ablation_obs
  Step 7: sbatch slurm/step5_6_7_ablations.sh ablation_uncertainty
  Step 8: sbatch slurm/step8_dr.sh

Commandments:
  Fast       python check_status_ablation_dr.py --fast
  Normal     python check_status_ablation_dr.py
  Done-check python check_status_ablation_dr.py --done
  Strict     python check_status_ablation_dr.py --done --strict
  Forensic   python check_status_ablation_dr.py --forensic

Core options:
  --project_root PATH        default: {DEFAULT_PROJECT_ROOT_NAME}; uses cwd automatically when already inside repo
  --log_root PATH            default: logs
  --results_dir PATH         default: logs/results
  --script_dir PATH          default: slurm
  --scope all|ablations|reward|obs|uncertainty|dr   default: all

Depth options:
  --fast                     shallow/quick: squeue + file presence; skips sacct, log-content scans, zip/NPZ/progress/TensorBoard validation
  --done                     alias for --assume-finished; missing work becomes actionable
  --strict                   alias for --strict-logs --strict-tensorboard --strict-episode-metrics
  --forensic                 done + strict + wider details + longer sacct window

Useful overrides:
  --no-squeue --no-sacct --sacct-days N --no-slurm-workdir-filter
  --allow-missing-ep-lengths --rerun-finished-bad
  --details-limit N --error-details-limit N --commands-limit N --rerun-chunk-size N
  --min-timestep-fraction F --min-eval-fraction F --min-model-bytes N --min-npz-bytes N
""".strip()

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
    (re.compile(r"Unknown reward_ablation|Unknown obs_mode|Unknown uncertainty_mode|Unknown dr_mode", re.I), "invalid ablation/DR mode"),
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
    partition: str | None = None
    gres: str | None = None
    gpus_per_node: str | None = None
    mem: str | None = None
    time_limit: str | None = None

@dataclass(frozen=True)
class RunSpec:
    group: str
    script_name: str
    script_arg: str | None
    array_id: int
    algorithm: str
    env_set: int
    num_robots: int
    seed: int
    experiment: str
    ablation: str
    version: str
    expected_steps: int
    log_steps: int
    n_eval_eps: int
    device: str

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
        return (self.script_name, (self.script_arg,) if self.script_arg else tuple())

    @property
    def log_key(self) -> tuple[str, str | None, int]:
        return (self.script_name, self.script_arg, self.array_id)

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
    script_name: str | None
    script_arg: str | None
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
class RunCheck:
    spec: RunSpec
    state: str
    artifact: ArtifactResult
    active: list[JobActivity]
    sacct: list[JobActivity]
    logs: list[LogFinding]
    stdout_paths: list[str]
    fatal_evidence: list[str]
    ambiguity_warnings: list[str] = field(default_factory=list)

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
    raw = (value or DEFAULT_PROJECT_ROOT_NAME).strip()
    p = Path(raw).expanduser()
    cwd = Path.cwd().expanduser()
    if raw == DEFAULT_PROJECT_ROOT_NAME and not p.is_absolute():
        candidates: list[Path] = []
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


def median_or_dash(vals: list[int | float]) -> str:
    if not vals:
        return "---"
    return human(median(vals))


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


def read_text_head(path: Path, max_bytes: int = 16_000) -> str:
    try:
        with path.open("rb") as f:
            return f.read(max_bytes).decode(errors="replace")
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

# ---------------------------------------------------------------------------
# Slurm parsing and expected run generation
# ---------------------------------------------------------------------------

def resolve_script(root: Path, script_name: str, script_dir: str = "slurm") -> Path | None:
    candidates = [root / script_dir / script_name, root / script_name]
    for p in candidates:
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
    for flag in ["--steps", "--log_steps", "--n_eval_eps", "--device"]:
        m = re.search(rf"{re.escape(flag)}\s+([^\s\\]+)", text)
        if m:
            out[flag] = m.group(1).strip().strip("'\"")
    return out


def parse_slurm(root: Path, script_name: str, script_dir: str) -> SlurmInfo:
    path = resolve_script(root, script_name, script_dir)
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

    array_names = ["seeds", "sets", "robots", "conditions", "obs_modes", "uncertainty_modes", "dr_modes"]
    arrays = {k: parse_bash_array(text, k) for k in array_names}
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
        partition=sbatch_value("--partition"),
        gres=sbatch_value("--gres"),
        gpus_per_node=sbatch_value("--gpus-per-node"),
        mem=sbatch_value("--mem"),
        time_limit=sbatch_value("--time"),
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


def script_steps(info: SlurmInfo) -> tuple[int, int, int, str]:
    raw_steps = info.assignments.get("steps") or info.assignments.get("--steps")
    steps = parse_int(raw_steps, DEFAULT_STEPS) or DEFAULT_STEPS
    log_steps = parse_int(info.assignments.get("--log_steps"), DEFAULT_LOG_STEPS) or DEFAULT_LOG_STEPS
    n_eval_eps = parse_int(info.assignments.get("--n_eval_eps"), DEFAULT_N_EVAL_EPS) or DEFAULT_N_EVAL_EPS
    device = info.assignments.get("--device", "cuda")
    return steps, log_steps, n_eval_eps, device


def build_ablation_specs(info: SlurmInfo) -> list[RunSpec]:
    seeds = as_ints(info.arrays.get("seeds", []), SEEDS)
    sets = as_ints(info.arrays.get("sets", []), SETS)
    steps, log_steps, n_eval_eps, device = script_steps(info)
    n_seeds = len(seeds)
    n_sets = len(sets)
    specs: list[RunSpec] = []
    for experiment, meta in ABLATION_EXPERIMENTS.items():
        conditions = as_strs(info.arrays.get(str(meta["array_name"]), []), list(meta["conditions"]))
        for cond_idx, cond in enumerate(conditions):
            for set_idx, env_set in enumerate(sets):
                for seed_idx, seed in enumerate(seeds):
                    array_id = seed_idx + n_seeds * (set_idx + n_sets * cond_idx)
                    version = f"{experiment}_{cond}"
                    specs.append(RunSpec(
                        group=str(meta["step"]),
                        script_name=ABLATION_SCRIPT,
                        script_arg=experiment,
                        array_id=array_id,
                        algorithm=ALGORITHM,
                        env_set=env_set,
                        num_robots=N_ABLATION,
                        seed=seed,
                        experiment=experiment,
                        ablation=cond,
                        version=version,
                        expected_steps=steps,
                        log_steps=log_steps,
                        n_eval_eps=n_eval_eps,
                        device=device,
                    ))
    return specs


def build_dr_specs(info: SlurmInfo) -> list[RunSpec]:
    dr_modes = as_strs(info.arrays.get("dr_modes", []), DR_MODES)
    sets = as_ints(info.arrays.get("sets", []), SETS)
    robots = as_ints(info.arrays.get("robots", []), ROBOTS_DR)
    seeds = as_ints(info.arrays.get("seeds", []), SEEDS)
    steps, log_steps, n_eval_eps, device = script_steps(info)
    n_seeds, n_robots, n_sets = len(seeds), len(robots), len(sets)
    specs: list[RunSpec] = []
    for dr_idx, dr_mode in enumerate(dr_modes):
        for set_idx, env_set in enumerate(sets):
            for robot_idx, n in enumerate(robots):
                for seed_idx, seed in enumerate(seeds):
                    array_id = seed_idx + n_seeds * (robot_idx + n_robots * (set_idx + n_sets * dr_idx))
                    specs.append(RunSpec(
                        group="Step 8 domain randomization",
                        script_name=DR_SCRIPT,
                        script_arg=None,
                        array_id=array_id,
                        algorithm=ALGORITHM,
                        env_set=env_set,
                        num_robots=n,
                        seed=seed,
                        experiment="dr",
                        ablation=dr_mode,
                        version=f"dr_{dr_mode}",
                        expected_steps=steps,
                        log_steps=log_steps,
                        n_eval_eps=n_eval_eps,
                        device=device,
                    ))
    return specs


def scope_filter(spec: RunSpec, scope: str) -> bool:
    aliases = {
        "reward": "ablation_reward",
        "obs": "ablation_obs",
        "uncertainty": "ablation_uncertainty",
    }
    scope = aliases.get(scope, scope)
    if scope == "all":
        return True
    if scope == "ablations":
        return spec.script_name == ABLATION_SCRIPT
    if scope == "dr":
        return spec.script_name == DR_SCRIPT
    return spec.experiment == scope


def expected_script_warnings(info: SlurmInfo, specs: list[RunSpec]) -> list[str]:
    warnings: list[str] = []
    if info.path is None:
        warnings.append(f"{info.script_name} not found; using fallback grid")
    if info.array_count is not None:
        max_id = max((s.array_id for s in specs if s.script_name == info.script_name), default=-1)
        expected_per_command = max_id + 1 if max_id >= 0 else 0
        if info.script_name == ABLATION_SCRIPT:
            # One declared array covers each ablation command separately, not all three together.
            if info.array_count != expected_per_command:
                warnings.append(f"{info.script_name} declares {info.array_count} array IDs, expected {expected_per_command} per ablation command")
        elif info.array_count != expected_per_command:
            warnings.append(f"{info.script_name} declares {info.array_count} array IDs, expected {expected_per_command}")
    return warnings

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


def base_job_id(job_id: str | None) -> str | None:
    if not job_id:
        return None
    return str(job_id).split("_", 1)[0].split(".", 1)[0]


def workdir_matches(root: Path, work_dir: str) -> bool:
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


def extract_log_key(path: Path, text: str, infos: dict[str, SlurmInfo]) -> tuple[str | None, str | None, int | None]:
    job_name, _jid, aid = parse_log_identity(path)
    if aid is None:
        # As a fallback, try to read explicit "job=N" from stdout.
        jm = re.search(r"\bjob=(\d+)", text)
        if jm:
            aid = int(jm.group(1))
    if job_name == script_job_name(infos, ABLATION_SCRIPT):
        for experiment, meta in ABLATION_EXPERIMENTS.items():
            if meta["echo_regex"].search(text):
                return ABLATION_SCRIPT, experiment, aid
    if job_name == script_job_name(infos, DR_SCRIPT) or DR_ECHO_RE.search(text):
        return DR_SCRIPT, None, aid
    return None, None, aid


def classify_log(path: Path, stream: str, infos: dict[str, SlurmInfo], stdout_marker_map: dict[tuple[str | None, str | None, int | None], tuple[str | None, str | None, int | None]], *, fast: bool = False) -> LogFinding:
    size = path.stat().st_size if path.exists() else 0
    mtime = path.stat().st_mtime if path.exists() else 0.0
    job, jid, aid = parse_log_identity(path)
    text = "" if size == 0 else (read_text_head(path) if fast else read_text(path))
    script, script_arg, key_aid = extract_log_key(path, text, infos) if text else (None, None, aid)
    if script is None:
        script, script_arg, key_aid = stdout_marker_map.get((job, jid, aid), (None, None, aid))
    severity = "empty" if size == 0 else "notice"
    reason = "empty" if size == 0 else "non-empty"
    snip = "" if size == 0 else snippet(text)
    if size != 0:
        for pat, why in FATAL_PATTERNS:
            if pat.search(text):
                severity, reason, snip = "fatal", why, snippet(text, pat)
                break
        else:
            for pat, why in WARN_PATTERNS:
                if pat.search(text):
                    severity, reason, snip = "warn", why, snippet(text, pat)
                    break
    return LogFinding(str(path), stream, job, jid, key_aid, script, script_arg, severity, reason, snip, size, mtime)


def collect_logs(
    infos: dict[str, SlurmInfo],
    scripts: Sequence[str],
    *,
    scan_contents: bool = True,
    fast: bool = False,
) -> tuple[
    list[Path],
    list[Path],
    list[LogFinding],
    dict[tuple[str, str | None, int], list[Path]],
    dict[tuple[str, str | None, int], list[LogFinding]],
    dict[tuple[str, str | None, int], set[str]],
]:
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

    stdout_marker_map: dict[tuple[str | None, str | None, int | None], tuple[str | None, str | None, int | None]] = {}
    if scan_contents:
        for p in sorted(stdout_files):
            if not p.is_file():
                continue
            job, jid, aid = parse_log_identity(p)
            text = read_text_head(p)
            script, script_arg, key_aid = extract_log_key(p, text, infos)
            if script is not None:
                stdout_marker_map[(job, jid, aid)] = (script, script_arg, key_aid)

    findings: list[LogFinding] = []
    if scan_contents:
        for p in sorted(stdout_files):
            if p.is_file():
                findings.append(classify_log(p, "stdout", infos, stdout_marker_map, fast=fast))
        for p in sorted(stderr_files):
            if p.is_file():
                findings.append(classify_log(p, "stderr", infos, stdout_marker_map, fast=fast))

    stdout_index: dict[tuple[str, str | None, int], list[Path]] = defaultdict(list)
    for p in sorted(stdout_files):
        job, jid, aid = parse_log_identity(p)
        marker = stdout_marker_map.get((job, jid, aid))
        if marker and marker[0] is not None and marker[2] is not None:
            stdout_index[(str(marker[0]), marker[1], int(marker[2]))].append(p)
        elif job == script_job_name(infos, DR_SCRIPT) and aid is not None:
            stdout_index[(DR_SCRIPT, None, aid)].append(p)

    finding_index: dict[tuple[str, str | None, int], list[LogFinding]] = defaultdict(list)
    jobids_index: dict[tuple[str, str | None, int], set[str]] = defaultdict(set)
    for f in findings:
        if f.script_name is not None and f.array_id is not None:
            key = (f.script_name, f.script_arg, int(f.array_id))
            finding_index[key].append(f)
            if f.job_id:
                bid = base_job_id(f.job_id)
                if bid:
                    jobids_index[key].add(bid)
    return sorted(stdout_files), sorted(stderr_files), findings, stdout_index, finding_index, jobids_index

# ---------------------------------------------------------------------------
# Artifact inspection
# ---------------------------------------------------------------------------

def load_npz(path: Path, spec: RunSpec, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
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


def read_progress(path: Path, spec: RunSpec, args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[str]]:
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


def inspect_run_artifacts(spec: RunSpec, log_root: Path, args: argparse.Namespace) -> ArtifactResult:
    run_dir = log_root / spec.version / spec.tag
    best = run_dir / "best_model" / "best_model.zip"
    npz = run_dir / "eval_logs" / "evaluations.npz"
    final = run_dir / spec.final_model_name
    progress = run_dir / "progress.csv"
    log_txt = run_dir / "log.txt"
    episode_metrics = run_dir / "episode_metrics.csv"
    issues: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "best_model": str(best),
        "eval_npz": str(npz),
        "final_model": str(final),
        "progress_csv": str(progress),
        "log_txt": str(log_txt),
        "episode_metrics_csv": str(episode_metrics),
        "expected_steps": spec.expected_steps,
        "expected_eval_count": spec.expected_eval_count,
        "expected_eval_episodes": spec.n_eval_eps,
    }
    if not run_dir.exists():
        issues.append("run directory missing")
        return ArtifactResult(issues, warnings, metrics)

    if getattr(args, "fast", False):
        for label, path, min_bytes in [
            ("best_model/best_model.zip", best, args.min_model_bytes),
            (f"final model {spec.final_model_name}", final, args.min_model_bytes),
            ("eval_logs/evaluations.npz", npz, args.min_npz_bytes),
        ]:
            if not file_nonempty(path, min_bytes):
                issues.append(f"{label} missing/too small")
            else:
                metrics[label.replace("/", "_").replace(".", "_").replace(" ", "_") + "_bytes"] = path.stat().st_size
        metrics["progress_csv_exists"] = progress.exists()
        metrics["log_txt_exists"] = log_txt.exists()
        metrics["episode_metrics_exists"] = episode_metrics.exists()
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
    if not file_nonempty(episode_metrics):
        (issues if args.strict_episode_metrics else warnings).append("episode_metrics.csv missing or empty")
    else:
        metrics["episode_metrics_bytes"] = episode_metrics.stat().st_size
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


def filter_sacct_for_spec(spec: RunSpec, sacct_rows: list[JobActivity], mapped_jobids: set[str]) -> list[JobActivity]:
    if spec.script_name != ABLATION_SCRIPT:
        return sacct_rows
    if not mapped_jobids:
        # step5_6_7 uses one Slurm job name for three different commands, so
        # unmarked accounting rows are ambiguous. Use logs/artifacts instead.
        return []
    return [r for r in sacct_rows if base_job_id(r.job_id) in mapped_jobids]


def check_run(
    spec: RunSpec,
    log_root: Path,
    infos: dict[str, SlurmInfo],
    active_idx: dict[tuple[str, int], list[JobActivity]],
    sacct_idx: dict[tuple[str, int], list[JobActivity]],
    stdout_idx: dict[tuple[str, str | None, int], list[Path]],
    log_idx: dict[tuple[str, str | None, int], list[LogFinding]],
    jobids_idx: dict[tuple[str, str | None, int], set[str]],
    args: argparse.Namespace,
) -> RunCheck:
    artifact = inspect_run_artifacts(spec, log_root, args)
    job = script_job_name(infos, spec.script_name)
    scheduler_key = (job, spec.array_id)
    log_key = spec.log_key
    active = active_idx.get(scheduler_key, [])
    all_sacct = sacct_idx.get(scheduler_key, [])
    mapped_jobids = jobids_idx.get(log_key, set())
    sacct = filter_sacct_for_spec(spec, all_sacct, mapped_jobids)
    stdout_paths = [str(p) for p in stdout_idx.get(log_key, [])]
    logs = log_idx.get(log_key, [])
    fatal = fatal_from(sacct, logs)
    act_state = activity_state(active)
    ambiguity_warnings: list[str] = []
    if spec.script_name == ABLATION_SCRIPT and active and not stdout_paths:
        ambiguity_warnings.append("active s5_6_7 job-name evidence is ambiguous until stdout identifies ablation_reward/obs/uncertainty")
    if spec.script_name == ABLATION_SCRIPT and all_sacct and not mapped_jobids and not artifact.ok:
        ambiguity_warnings.append("unmapped s5_6_7 sacct evidence exists for this array ID; script argument cannot be proven from accounting alone")
    started = bool(stdout_paths or logs or sacct or Path(str(artifact.metrics.get("run_dir"))).exists())
    if artifact.ok:
        state = "verified_warn" if (artifact.warnings or ambiguity_warnings or any(l.severity == "warn" for l in logs) or fatal) else "verified_good"
    elif act_state:
        state = act_state
    elif fatal:
        state = "confirmed_failed"
    elif started:
        state = "finished_bad"
    else:
        state = "missing_finished" if args.assume_finished else "not_started"
    return RunCheck(spec, state, artifact, active, sacct, logs, stdout_paths, fatal, ambiguity_warnings)

# ---------------------------------------------------------------------------
# Summaries and next-action logic
# ---------------------------------------------------------------------------

def group_summary(checks: list[RunCheck]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    by_group: dict[str, list[RunCheck]] = defaultdict(list)
    for c in checks:
        by_group[c.spec.group].append(c)
    order = [
        "Step 5 reward ablation",
        "Step 6 observation ablation",
        "Step 7 uncertainty ablation",
        "Step 8 domain randomization",
    ]
    for group in order:
        cs = by_group.get(group, [])
        if not cs:
            continue
        cnt = Counter(c.state for c in cs)
        verified = cnt["verified_good"] + cnt["verified_warn"]
        active = cnt["running"] + cnt["pending"] + cnt["active"]
        bad = cnt["confirmed_failed"] + cnt["finished_bad"] + cnt["missing_finished"]
        warnings = sum(1 for c in cs if c.artifact.warnings or c.ambiguity_warnings or c.state == "verified_warn")
        evals = [int(c.artifact.metrics.get("eval_count", 0)) for c in cs if c.verified]
        last_ts = [int(c.artifact.metrics.get("last_timestep", 0)) for c in cs if c.verified]
        issues = Counter()
        for c in cs:
            for x in c.artifact.critical:
                issues[x.split(":", 1)[0]] += 1
            for x in c.fatal_evidence:
                issues["fatal evidence"] += 1
            for x in c.ambiguity_warnings:
                issues["ambiguous Slurm evidence"] += 1
        top = "; ".join(f"{k} ({v})" for k, v in issues.most_common(2)) or "---"
        rows.append([group, len(cs), cnt["verified_good"], cnt["verified_warn"], active, bad, cnt["not_started"], warnings, pct(verified, len(cs)), median_or_dash(evals), median_or_dash(last_ts), top])
    return rows


def condition_summary(checks: list[RunCheck]) -> list[list[Any]]:
    by_key: dict[tuple[str, str], list[RunCheck]] = defaultdict(list)
    for c in checks:
        by_key[(c.spec.experiment, c.spec.ablation)].append(c)
    rows: list[list[Any]] = []
    order: list[tuple[str, str]] = []
    for exp in ABLATION_EXPERIMENTS:
        for cond in ABLATION_EXPERIMENTS[exp]["conditions"]:
            order.append((exp, cond))
    for dr in DR_MODES:
        order.append(("dr", dr))
    for exp, cond in order:
        cs = by_key.get((exp, cond), [])
        if not cs:
            continue
        cnt = Counter(c.state for c in cs)
        verified = cnt["verified_good"] + cnt["verified_warn"]
        rows.append([exp, cond, len(cs), verified, cnt["running"] + cnt["pending"] + cnt["active"], cnt["confirmed_failed"] + cnt["finished_bad"] + cnt["missing_finished"], cnt["not_started"], pct(verified, len(cs))])
    return rows


def fatal_log_summary(findings: list[LogFinding], limit: int) -> tuple[list[list[Any]], list[list[Any]]]:
    relevant = [f for f in findings if f.severity in {"fatal", "warn"}]
    counts = Counter((f.severity, f.reason) for f in relevant)
    summary = [[sev, reason, count] for (sev, reason), count in counts.most_common()]
    fatal = [f for f in relevant if f.severity == "fatal"][:limit]
    details = [[Path(f.path).parent.name, Path(f.path).name, f.script_arg or "dr", f.array_id if f.array_id is not None else "?", f.reason, f.snippet] for f in fatal]
    return summary, details


def build_commands(items: dict[tuple[str, tuple[str, ...]], list[int]], script_dir: str, limit: int, chunk_size: int) -> list[str]:
    commands: list[str] = []
    for (script, args), ids in sorted(items.items(), key=lambda kv: COMMAND_ORDER.get(kv[0], 999)):
        unique = sorted(set(ids))
        if not unique:
            continue
        path = f"{script_dir.rstrip('/')}/{script}" if script_dir else script
        arg_s = " " + " ".join(shlex.quote(a) for a in args) if args else ""
        full_ablation = script == ABLATION_SCRIPT and unique == list(range(0, 200))
        full_dr = script == DR_SCRIPT and unique == list(range(0, 600))
        if full_ablation or full_dr:
            commands.append(f"sbatch {path}{arg_s}")
            continue
        for i in range(0, len(unique), chunk_size):
            chunk = unique[i:i + chunk_size]
            commands.append(f"sbatch --array={compact_ranges(chunk)} {path}{arg_s}")
    return commands[:limit]


def collect_action_items(checks: list[RunCheck], args: argparse.Namespace) -> tuple[dict, dict, dict]:
    rerun: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    investigate: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    launch: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    for c in checks:
        key = c.spec.command_key
        if c.state in {"confirmed_failed", "missing_finished"} or (c.state == "finished_bad" and args.rerun_finished_bad):
            rerun[key].append(c.spec.array_id)
        elif c.state == "finished_bad":
            investigate[key].append(c.spec.array_id)
        elif c.state == "not_started":
            launch[key].append(c.spec.array_id)
    return rerun, investigate, launch


def gate_status(required: list[RunCheck]) -> tuple[str, str, int, int]:
    total = len(required)
    verified = sum(1 for c in required if c.verified)
    cnt = Counter(c.state for c in required)
    if total == 0:
        return "NO SPECS", "no matching expected runs", 0, 0
    if verified == total:
        warn = sum(1 for c in required if c.state == "verified_warn")
        return ("READY-WARN" if warn else "READY"), (f"{warn} warning run(s)" if warn else "all required runs verified"), verified, total
    active = cnt["running"] + cnt["pending"] + cnt["active"]
    bad = cnt["confirmed_failed"] + cnt["finished_bad"] + cnt["missing_finished"]
    if active:
        return "WAIT", f"{active} active, {bad} bad/missing, {total - verified} not verified", verified, total
    if bad:
        return "BLOCKED", f"{bad} failed/bad/missing, {total - verified} not verified", verified, total
    return "NOT STARTED", f"{cnt['not_started']} not started", verified, total


def readiness_rows(checks: list[RunCheck], script_dir: str, results_dir: str) -> list[list[Any]]:
    by_exp: dict[str, list[RunCheck]] = defaultdict(list)
    for c in checks:
        by_exp[c.spec.experiment].append(c)

    def g(label: str, subset: list[RunCheck], command: str, required_note: str) -> list[Any]:
        status, why, ok, total = gate_status(subset)
        return [label, f"{ok}/{total}", status, required_note, command, why]

    reward = by_exp.get("ablation_reward", [])
    obs = by_exp.get("ablation_obs", [])
    unc = by_exp.get("ablation_uncertainty", [])
    dr = by_exp.get("dr", [])
    wind_subset = [c for c in dr if c.spec.ablation in {"none", "full"} and c.spec.env_set == 1 and c.spec.num_robots == 3]
    dr_curves_subset = [c for c in dr if c.spec.num_robots == 3]

    return [
        g("Reward eval/Table 3", reward, f"sbatch --array=0-199 {script_dir}/eval_ablations.sh ablation_reward", "all reward-ablation training best_model.zip files"),
        g("Obs analysis/Table 4", obs, f"python analyze_results.py --log_root logs --results_dir {results_dir}", "all obs-ablation training evaluations.npz files; no eval job needed"),
        g("Uncertainty eval/Table 5", unc, f"sbatch --array=0-799 {script_dir}/eval_ablations.sh ablation_uncertainty", "all uncertainty-ablation training best_model.zip files"),
        g("DR eval/Table 7", dr, f"sbatch --array=0-599 {script_dir}/eval_ablations.sh dr", "all DR training best_model.zip files"),
        g("Wind sweep/Figure 4", wind_subset, f"sbatch {script_dir}/eval_wind_sweep.sh", "DR none+full, set=1, N=3, all seeds"),
        g("DR curves/Figure 5", dr_curves_subset, f"python plot_figures.py --log_root logs --results_dir {results_dir} --figures_dir figures", "all DR modes, N=3, all sets/seeds"),
    ]


def workflow_rows(infos: dict[str, SlurmInfo], specs: list[RunSpec], script_dir: str) -> list[list[Any]]:
    by_cmd: dict[tuple[str, tuple[str, ...]], list[RunSpec]] = defaultdict(list)
    for s in specs:
        by_cmd[s.command_key].append(s)
    rows: list[list[Any]] = []
    for key, ss in sorted(by_cmd.items(), key=lambda kv: COMMAND_ORDER.get(kv[0], 999)):
        script, argv = key
        info = infos[script]
        exp = argv[0] if argv else "dr"
        if exp == "dr":
            mode_order = ["none", "wind", "full"]
        else:
            mode_order = ["full", "no_term", "no_spr", "no_path", "no_pos", "no_inf_hist", "pos_only", "wind_only", "act_only", "deterministic"]
        conditions = ",".join(sorted({x.ablation for x in ss}, key=lambda x: mode_order.index(x) if x in mode_order else 99))
        sets = compact_ranges(s.env_set for s in ss)
        robots = compact_ranges(s.num_robots for s in ss)
        seeds = compact_ranges(s.seed for s in ss)
        args = f"--algorithm CrossQ --experiment {exp} --ablation <condition> --steps {ss[0].expected_steps} --device {ss[0].device}"
        cmd_path = f"{script_dir.rstrip('/')}/{script}" if script_dir else script
        cmd = f"sbatch {cmd_path}" + (f" {' '.join(argv)}" if argv else "")
        rows.append([cmd, script, info.array_spec or "?", len(ss), conditions, f"sets {sets}; N {robots}; seeds {seeds}", "train.py", args, "GPU" if ss[0].device == "cuda" else "CPU"])
    return rows


def print_sample_failures() -> None:
    rows = [
        ["Missing Slurm output dir", "sbatch error opening logs/slurm_outputs/...", "recoverable", "run bash slurm/beocat_prepare_dirs.sh; resubmit"],
        ["Wrong submit directory", "stderr: ERROR: Submit from the repository root", "recoverable", "cd repo root; resubmit"],
        ["Bad Python env", "PYTHON_BIN is not executable / ModuleNotFoundError", "recoverable", "fix slurm/beocat_env.sh or environment; rerun"],
        ["GPU/CUDA problem", "CUDA unavailable/OOM/CUBLAS error", "usually", "verify GPU node; rerun after resource fix"],
        ["Timeout/cancel/node fail", "sacct TIMEOUT/CANCELLED/NODE_FAIL", "yes", "rerun affected array IDs"],
        ["Partial training", "best_model exists but final zip/evaluations.npz missing", "yes", "treat as incomplete; rerun"],
        ["Corrupt artifact", "invalid zip/NPZ or too small file", "yes", "delete stale partial run dir; rerun"],
        ["NaN/Inf rewards", "evaluations.npz contains non-finite rewards", "maybe", "investigate instability; rerun only after review"],
        ["Ablation Slurm ambiguity", "same job name/array used for reward/obs/uncertainty", "n/a", "trust artifacts/stdout markers, not raw sacct alone"],
    ]
    print_table(["Failure", "How it appears", "Recoverable", "Action"], rows, max_col_width=44)

# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check status of Step 5-8 ablation/domain-randomization training jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT_NAME)
    p.add_argument("--log_root", default="logs")
    p.add_argument("--results_dir", default="logs/results")
    p.add_argument("--script_dir", default="slurm")
    p.add_argument("--scope", default="all", choices=["all", "ablations", "reward", "obs", "uncertainty", "dr", "ablation_reward", "ablation_obs", "ablation_uncertainty"])
    p.add_argument("--guide", action="store_true", help="Print the quick guide and exit")

    p.add_argument("--fast", action="store_true", help="Very fast shallow check: no sacct and no heavy artifact validation scans, and no heavy artifact validation")
    p.add_argument("--done", action="store_true", help="Alias for --assume-finished")
    p.add_argument("--strict", action="store_true", help="Alias for --strict-logs --strict-tensorboard --strict-episode-metrics")
    p.add_argument("--forensic", action="store_true", help="Done + strict + more details + longer sacct history")
    p.add_argument("--assume-finished", action="store_true", help="Treat missing work as missing_after_expected_completion rather than not_started")

    p.add_argument("--strict-logs", action="store_true")
    p.add_argument("--strict-tensorboard", action="store_true")
    p.add_argument("--strict-episode-metrics", action="store_true")
    p.add_argument("--allow-missing-ep-lengths", action="store_true")
    p.add_argument("--rerun-finished-bad", action="store_true")

    p.add_argument("--no-squeue", action="store_true")
    p.add_argument("--no-sacct", action="store_true")
    p.add_argument("--sacct-days", type=int, default=14)
    p.add_argument("--no-slurm-workdir-filter", action="store_true")

    p.add_argument("--details-limit", type=int, default=20)
    p.add_argument("--error-details-limit", type=int, default=12)
    p.add_argument("--commands-limit", type=int, default=20)
    p.add_argument("--rerun-chunk-size", type=int, default=600)

    p.add_argument("--min-timestep-fraction", type=float, default=1.0)
    p.add_argument("--min-eval-fraction", type=float, default=1.0)
    p.add_argument("--min-model-bytes", type=int, default=10_000)
    p.add_argument("--min-npz-bytes", type=int, default=1_000)
    args = p.parse_args(argv)

    if args.guide:
        print(QUICK_GUIDE)
        raise SystemExit(0)
    scope_aliases = {"reward": "ablation_reward", "obs": "ablation_obs", "uncertainty": "ablation_uncertainty"}
    args.scope = scope_aliases.get(args.scope, args.scope)
    if args.done:
        args.assume_finished = True
    if args.strict:
        args.strict_logs = True
        args.strict_tensorboard = True
        args.strict_episode_metrics = True
    if args.forensic:
        args.assume_finished = True
        args.strict_logs = True
        args.strict_tensorboard = True
        args.strict_episode_metrics = True
        args.sacct_days = max(args.sacct_days, 45)
        args.details_limit = max(args.details_limit, 60)
        args.error_details_limit = max(args.error_details_limit, 40)
        args.commands_limit = max(args.commands_limit, 80)
    if args.fast:
        args.no_sacct = True
        args.min_timestep_fraction = min(args.min_timestep_fraction, 1.0)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    root = resolve_project_root(args.project_root)
    log_root = rel_or_abs(root, args.log_root)
    results_dir = rel_or_abs(root, args.results_dir)
    script_dir = args.script_dir

    infos = {script: parse_slurm(root, script, script_dir) for script in ALL_SCRIPTS}
    all_specs = build_ablation_specs(infos[ABLATION_SCRIPT]) + build_dr_specs(infos[DR_SCRIPT])
    specs = [s for s in all_specs if scope_filter(s, args.scope)]

    job_names = {script_job_name(infos, s) for s in ALL_SCRIPTS}
    project_filter = None if args.no_slurm_workdir_filter else root

    stdout_files, stderr_files, findings, stdout_idx, log_idx, jobids_idx = collect_logs(
        infos,
        ALL_SCRIPTS,
        scan_contents=not args.fast,  # fast mode skips log-content reads; normal mode uses stdout markers to disambiguate shared ablation job names
    )
    active_idx = query_squeue(job_names, args.no_squeue, project_filter)
    sacct_idx = query_sacct(job_names, args.sacct_days, args.no_sacct, project_filter)

    checks = [check_run(s, log_root, infos, active_idx, sacct_idx, stdout_idx, log_idx, jobids_idx, args) for s in specs]

    print("\nCHECK_STATUS_ABLATION_DR")
    print("=" * 92)
    print(f"Project root : {root}")
    print(f"Log root     : {log_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Script dir   : {script_dir}")
    print(f"Scope        : {args.scope}")
    print(f"Mode         : {'fast' if args.fast else 'forensic' if args.forensic else 'strict' if args.strict else 'done-check' if args.assume_finished else 'normal'}")
    print(f"Slurm filter : {'disabled' if args.no_slurm_workdir_filter else 'WorkDir == project root'}")
    print(f"Squeue/sacct : {'off' if args.no_squeue else 'on'} / {'off' if args.no_sacct else 'on (' + str(args.sacct_days) + 'd)'}")

    print("\nWorkflow model")
    print("-" * 92)
    print_table(["Command", "Script", "Array", "Runs", "Conditions", "Grid", "Python", "Args", "Class"], workflow_rows(infos, specs, args.script_dir), max_col_width=50)

    script_warns: list[str] = []
    for script, info in infos.items():
        script_warns.extend(expected_script_warnings(info, all_specs))
    if script_warns:
        print("\nScript/grid warnings")
        print("-" * 92)
        for w in script_warns:
            print(f"  WARN: {w}")

    if ABLATION_SCRIPT in {s.script_name for s in specs}:
        print("\nImportant ablation-script difference")
        print("-" * 92)
        print("  step5_6_7_ablations.sh is submitted three times with different script arguments")
        print("  but all three submissions share the same Slurm job name and array IDs.")
        print("  This checker uses stdout markers and output artifacts to separate reward vs obs vs uncertainty.")
        print("  Raw squeue/sacct evidence for s5_6_7_ablations can be ambiguous before stdout is created.")

    print("\nTraining artifact coverage by stage")
    print("-" * 92)
    print_table(["Group", "Expected", "Good", "Warn", "Active", "Bad/missing", "Not started", "Warn runs", "% verified", "med evals", "med last ts", "Top issue"], group_summary(checks), max_col_width=56)

    print("\nCoverage by condition/mode")
    print("-" * 92)
    print_table(["Experiment", "Condition/mode", "Expected", "Verified", "Active", "Bad/missing", "Not started", "% verified"], condition_summary(checks), max_col_width=34)

    print("\nReadiness gates")
    print("-" * 92)
    print_table(["Gate", "Ready", "Status", "Requires", "Next command", "Reason"], readiness_rows(checks, script_dir, args.results_dir), max_col_width=60)

    fatal_summary, fatal_details = fatal_log_summary(findings, args.error_details_limit)
    print("\nSlurm/log warning summary")
    print("-" * 92)
    print_table(["Severity", "Reason", "Count"], fatal_summary, max_col_width=58)
    if fatal_details:
        print("\nFatal log examples")
        print("-" * 92)
        print_table(["Folder", "File", "Scope", "Array", "Reason", "Snippet"], fatal_details, max_col_width=54)

    # Detailed examples of bad/not-started work.
    bad_checks = [c for c in checks if c.state in {"confirmed_failed", "finished_bad", "missing_finished", "not_started"}]
    if bad_checks:
        print("\nExamples needing attention")
        print("-" * 92)
        detail_rows = []
        for c in bad_checks[:args.details_limit]:
            issues = "; ".join((c.artifact.critical + c.fatal_evidence + c.ambiguity_warnings)[:2]) or "---"
            detail_rows.append([c.state, c.spec.experiment, c.spec.ablation, c.spec.array_id, f"set={c.spec.env_set} N={c.spec.num_robots} seed={c.spec.seed}", issues])
        print_table(["State", "Experiment", "Mode", "Array", "Run", "Issue"], detail_rows, max_col_width=62)

    rerun, investigate, launch = collect_action_items(checks, args)
    if launch:
        print("\nSubmit commands for not-started work")
        print("-" * 92)
        for cmd in build_commands(launch, script_dir, args.commands_limit, args.rerun_chunk_size):
            print("  " + cmd)
    if rerun:
        print("\nRerun commands for failed/missing work")
        print("-" * 92)
        for cmd in build_commands(rerun, script_dir, args.commands_limit, args.rerun_chunk_size):
            print("  " + cmd)
    if investigate:
        print("\nInvestigate before rerun")
        print("-" * 92)
        for cmd in build_commands(investigate, script_dir, args.commands_limit, args.rerun_chunk_size):
            print("  " + cmd)

    print("\nWhat successful training proves")
    print("-" * 92)
    artifact_rows = [
        ["best_model/best_model.zip", "EvalCallback saved at least one usable checkpoint", "does not prove final save or full 2M steps", "valid zip, non-trivial size, optional SB3 load"],
        ["final CrossQ_N*_env*.zip", "model.save() ran after learn() returned", "does not prove evaluation quality", "valid zip and expected filename"],
        ["eval_logs/evaluations.npz", "EvalCallback recorded rewards/timesteps", "does not prove model loadability", "keys, row count, last timestep, finite rewards"],
        ["progress.csv/log.txt", "SB3 logger wrote training progress", "may be partial or missing despite model files", "readable and reaches expected timesteps"],
        ["episode_metrics.csv", "terminal episode metrics were captured", "not a substitute for EvalCallback NPZ", "non-empty; strict mode requires it"],
        ["TensorBoard events", "tensorboard logger wrote events", "does not prove completion", "non-empty event file; strict mode requires it"],
        ["Slurm stdout/stderr", "job mapping and runtime errors/warnings", "stdout existing does not prove success", "scan for fatal patterns and exact mode markers"],
    ]
    print_table(["Artifact", "Proves", "Does not prove", "Validate"], artifact_rows, max_col_width=42)

    print("\nCommon failure modes")
    print("-" * 92)
    print_sample_failures()

    cnt = Counter(c.state for c in checks)
    verified = cnt["verified_good"] + cnt["verified_warn"]
    bad = cnt["confirmed_failed"] + cnt["finished_bad"] + cnt["missing_finished"]
    active = cnt["running"] + cnt["pending"] + cnt["active"]
    elapsed = time.perf_counter() - started
    print("\nFinal status")
    print("-" * 92)
    print(f"  verified={verified}/{len(checks)} ({pct(verified, len(checks)).strip()}) | active={active} | bad/missing={bad} | not_started={cnt['not_started']} | elapsed={elapsed:.1f}s")
    if verified == len(checks):
        print("  OK: selected scope is verified. Review warnings if any before downstream evaluation/analysis.")
        return 0
    if active:
        print("  WAIT: selected scope still has active jobs or ambiguous active Slurm evidence.")
        return 2
    if bad:
        print("  BLOCKED: failed, corrupt, incomplete, or missing-after-finished runs exist.")
        return 3
    print("  NOT STARTED or no artifacts detected for selected scope.")
    return 4 if args.assume_finished else 0


if __name__ == "__main__":
    raise SystemExit(main())
