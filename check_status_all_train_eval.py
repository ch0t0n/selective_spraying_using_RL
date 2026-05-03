#!/usr/bin/env python3
"""
check_status_all_train_eval.py

Purpose
-------
Command-line-only final readiness checker for the repository steps that run
after training/evaluation jobs have finished:

  python merge_eval_results.py --results_dir logs/results
  python analyze_results.py --log_root logs --results_dir logs/results
  python plot_figures.py --log_root logs --results_dir logs/results --figures_dir figures
  python sensitivity_hp.py --write_latex_only --results_dir logs/results

This checker does not submit jobs, create rerun scripts, write JSON reports, or
modify results. It prints a concise readiness report to stdout.
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
# Normal project defaults
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT_NAME = "selective_spraying_using_RL"
DEFAULT_LOG_ROOT = "logs"
DEFAULT_RESULTS_DIR = "logs/results"
DEFAULT_FIGURES_DIR = "figures"
DEFAULT_SCRIPT_DIR = "slurm"

ALGORITHMS = ["A2C", "ARS", "PPO", "TQC", "TRPO", "CrossQ"]
OTHER_ALGORITHMS = ["A2C", "ARS", "PPO", "TQC", "TRPO"]
SEEDS = [0, 42, 123, 2024, 9999]
SETS = list(range(1, 11))
TRANSFER_SETS = list(range(2, 11))
ROBOTS = [2, 3, 4, 5]
ABLATION_N = 3

REWARD_CONDITIONS = ["full", "no_term", "no_spr", "no_path"]
OBS_MODES = ["full", "no_pos", "no_inf_hist", "pos_only"]
UNCERTAINTY_MODES = ["full", "wind_only", "act_only", "deterministic"]
DR_MODES = ["none", "wind", "full"]

EXPECTED = {
    "main_default_npz": len(ALGORITHMS) * len(ROBOTS) * len(SETS) * len(SEEDS),
    "main_transfer_npz": len(ALGORITHMS) * len(ROBOTS) * len(TRANSFER_SETS) * len(SEEDS),
    "main_tuned_npz": len(ALGORITHMS) * len(ROBOTS) * len(SETS) * len(SEEDS),
    "ablation_reward_npz": len(REWARD_CONDITIONS) * len(SETS) * len(SEEDS),
    "ablation_obs_npz": len(OBS_MODES) * len(SETS) * len(SEEDS),
    "ablation_uncertainty_npz": len(UNCERTAINTY_MODES) * len(SETS) * len(SEEDS),
    "dr_npz": len(DR_MODES) * len(ROBOTS) * len(SETS) * len(SEEDS),
    "dr_n3_npz": len(DR_MODES) * len(SETS) * len(SEEDS),
    "eval_ablation_reward": len(REWARD_CONDITIONS) * len(SETS) * len(SEEDS),
    "eval_ablation_uncertainty": len(UNCERTAINTY_MODES) * len(UNCERTAINTY_MODES) * len(SETS) * len(SEEDS),
    "eval_dr": len(DR_MODES) * len(ROBOTS) * len(SETS) * len(SEEDS),
    "eval_wind_sweep": 2 * 10 * len(SEEDS),
}

REQUIRED_EVAL_COLUMNS = {
    "algorithm", "experiment", "ablation", "hp_tag", "num_robots", "env_set", "seed",
    "eval_wind_min", "eval_wind_max", "eval_uncertainty_mode",
    "mean_reward", "std_reward", "max_reward", "iqm", "cvar_0.1",
    "mean_ep_length", "iqm_ep_length", "n_episodes", "elapsed_s",
}

HP_REQUIRED = {
    "A2C": ["learning_rate", "gae_lambda", "vf_coef", "ent_coef", "max_grad_norm"],
    "ARS": ["learning_rate", "delta_std", "n_delta"],
    "PPO": ["learning_rate", "gae_lambda", "vf_coef", "ent_coef", "max_grad_norm", "clip_range", "n_epochs"],
    "TRPO": ["learning_rate", "gae_lambda", "target_kl", "cg_max_steps"],
    "CrossQ": ["learning_rate", "buffer_size", "batch_size"],
    "TQC": ["learning_rate", "buffer_size", "batch_size", "tau", "top_quantiles_to_drop_per_net"],
}
EXPECTED_CV_ROWS = sum(len(v) for v in HP_REQUIRED.values())
EXPECTED_RAW_ROWS = EXPECTED_CV_ROWS * 7

# Upstream Slurm scripts/jobs that can still be writing files needed by the
# final Python commands. The checker parses job names from these scripts when
# available and falls back to the known job-name defaults.
TARGET_PY_SCRIPTS = [
    "merge_eval_results.py",
    "analyze_results.py",
    "plot_figures.py",
    "sensitivity_hp.py",
]

UPSTREAM_SCRIPTS = [
    "step1_crossq_default.sh",
    "step1_others_default.sh",
    "step2_crossq_transfer.sh",
    "step2_others_transfer.sh",
    "step3_crossq_tune_hyperparameters.sh",
    "step3_others_tune_hyperparameters.sh",
    "step3_tune_hyperparameters.sh",
    "step4_crossq_tuned.sh",
    "step4_others_tuned.sh",
    "step5_6_7_ablations.sh",
    "step8_dr.sh",
    "eval_ablations.sh",
    "eval_wind_sweep.sh",
    "step9_sensitivity_hp.sh",
]
FALLBACK_JOB_NAMES = {
    "step1_crossq_default.sh": "s1_crossq_default",
    "step1_others_default.sh": "s1_others_default",
    "step2_crossq_transfer.sh": "s2_crossq_transfer",
    "step2_others_transfer.sh": "s2_others_transfer",
    "step3_crossq_tune_hyperparameters.sh": "s3_crossq_tune",
    "step3_others_tune_hyperparameters.sh": "s3_others_tune",
    "step3_tune_hyperparameters.sh": "s3_tune",
    "step4_crossq_tuned.sh": "s4_crossq_tuned",
    "step4_others_tuned.sh": "s4_others_tuned",
    "step5_6_7_ablations.sh": "s5_6_7_ablations",
    "step8_dr.sh": "s8_dr",
    "eval_ablations.sh": "eval_ablation",
    "eval_wind_sweep.sh": "eval_wind_sweep",
    "step9_sensitivity_hp.sh": "s9_sensitivity_hp",
}

ANALYZE_OUTPUTS = [
    "main_default_summary.csv",
    "main_transfer_summary.csv",
    "main_tuned_summary.csv",
    "ablation_reward_agg.csv",
    "ablation_obs_agg.csv",
    "ablation_uncertainty_agg.csv",
    "dr_results_agg.csv",
    "main_default_latex_rows.txt",
    "main_transfer_latex_rows.txt",
    "main_tuned_latex_rows.txt",
    "main_combined_latex_rows.txt",
    "ablation_reward_latex_rows.txt",
    "ablation_obs_latex_rows.txt",
    "ablation_uncertainty_latex_rows.txt",
    "dr_results_latex_rows.txt",
    "combined_ablations_dr_latex_rows.txt",
]

FIGURE_OUTPUTS = [
    "default_learning_curves.png",
    "default_2_robots.png",
    "default_3_robots.png",
    "default_4_robots.png",
    "default_5_robots.png",
    "random_learning_curves.png",
    "random_2_robots.png",
    "random_3_robots.png",
    "random_4_robots.png",
    "random_5_robots.png",
    "transfer_learning_curves.png",
    "transfer_2_robots.png",
    "transfer_3_robots.png",
    "transfer_4_robots.png",
    "transfer_5_robots.png",
    "combined_learning_curves.png",
    "scalability.png",
    "wind_sensitivity.png",
    "dr_curves.png",
]

QUICK_GUIDE = f"""
CHECK_STATUS_ALL_TRAIN_EVAL QUICK GUIDE

Checks readiness for:
  python merge_eval_results.py --results_dir logs/results
  python analyze_results.py --log_root logs --results_dir logs/results
  python plot_figures.py --log_root logs --results_dir logs/results --figures_dir figures
  python sensitivity_hp.py --write_latex_only --results_dir logs/results

Commandments:
  Fast       python check_status_all_train_eval.py --fast
  Normal     python check_status_all_train_eval.py
  Done-check python check_status_all_train_eval.py --done
  Strict     python check_status_all_train_eval.py --done --strict
  Forensic   python check_status_all_train_eval.py --forensic

Core options:
  --project_root PATH        default: {DEFAULT_PROJECT_ROOT_NAME}; uses cwd automatically when already inside repo
  --log_root PATH            default: {DEFAULT_LOG_ROOT}
  --results_dir PATH         default: {DEFAULT_RESULTS_DIR}
  --figures_dir PATH         default: {DEFAULT_FIGURES_DIR}
  --script_dir PATH          default: {DEFAULT_SCRIPT_DIR}
  --scope all|merge|analyze|plot|sensitivity   default: all

Depth options:
  --fast                     quick readiness: squeue + file counts/sizes; skips sacct, log scans, CSV deep checks, NPZ loading
  --done                     alias for --assume-finished; missing final inputs are actionable blockers
  --strict                   require exact row counts, parse CSVs, validate NPZ keys, and check expected generated outputs if present
  --forensic                 done + strict + Slurm accounting/log scans + wider details

Useful overrides:
  --no-squeue --no-sacct --sacct-days N --no-slurm-workdir-filter
  --details-limit N --error-details-limit N --min-npz-bytes N --min-csv-bytes N
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
    (re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory|oom-kill|OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED", re.I), "out of memory / CUDA OOM"),
    (re.compile(r"\bKilled\b|oom_kill", re.I), "process killed"),
    (re.compile(r"Permission denied", re.I), "permission denied"),
    (re.compile(r"ERROR:\s*(Submit from the repository root|PYTHON_BIN is not executable|unknown experiment)", re.I), "job setup error"),
    (re.compile(r"slurmstepd: error|srun: error|sbatch: error", re.I), "Slurm step error"),
    (re.compile(r"DUE TO TIME LIMIT|TIMEOUT|time limit", re.I), "time limit"),
    (re.compile(r"CANCELLED|CANCELED|cancelled by|canceled by", re.I), "cancelled"),
    (re.compile(r"failed with exit code|exit code\s*[1-9]|exited with exit code [1-9]|\bFAILED\b|command not found", re.I), "nonzero exit/failure"),
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlurmInfo:
    script_name: str
    path: Path | None
    job_name: str
    array_spec: str | None = None
    output_glob: str | None = None
    error_glob: str | None = None

@dataclass(frozen=True)
class TrainNPZSpec:
    group: str
    version: str
    algorithm: str
    num_robots: int
    env_set: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.algorithm}_N{self.num_robots}_env{self.env_set}_seed{self.seed}"

@dataclass
class CheckResult:
    name: str
    expected: int | None = None
    found: int | None = None
    status: str = "not_started"
    critical: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.critical and self.status in {"verified_good", "verified_warn", "already_done", "ready"}

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

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def rel_or_abs(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


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


def print_table(headers: list[str], rows: list[list[Any]], *, max_col_width: int = 74) -> None:
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


def status_from_counts(expected: int, found: int, critical: list[str], warnings: list[str], assume_finished: bool) -> str:
    if critical:
        return "finished_bad" if found else ("missing_after_expected_completion" if assume_finished else "not_started")
    if found >= expected:
        return "verified_warn" if warnings else "verified_good"
    if found:
        return "finished_bad"
    return "missing_after_expected_completion" if assume_finished else "not_started"

# ---------------------------------------------------------------------------
# Slurm parsing and live job evidence
# ---------------------------------------------------------------------------

def parse_slurm_script(root: Path, script_dir: str, script_name: str) -> SlurmInfo:
    path_candidates = [root / script_dir / script_name, root / script_name]
    path = next((p for p in path_candidates if p.exists()), None)
    fallback_job = FALLBACK_JOB_NAMES.get(script_name, Path(script_name).stem)
    if path is None:
        return SlurmInfo(script_name=script_name, path=None, job_name=fallback_job)
    text = path.read_text(errors="replace")

    def sbatch_value(opt: str) -> str | None:
        m = re.search(rf"^\s*#SBATCH\s+{re.escape(opt)}(?:=|\s+)(\S+)", text, re.M)
        return m.group(1).strip() if m else None

    array_spec = sbatch_value("--array")
    out_pat = sbatch_value("--output")
    err_pat = sbatch_value("--error")
    job_name = sbatch_value("--job-name") or fallback_job

    def to_glob(pat: str | None) -> str | None:
        if not pat:
            return None
        g = pat.replace("%x", "*").replace("%A", "*").replace("%a", "*").replace("%j", "*")
        return str(rel_or_abs(root, g))

    return SlurmInfo(
        script_name=script_name,
        path=path,
        job_name=job_name,
        array_spec=array_spec,
        output_glob=to_glob(out_pat),
        error_glob=to_glob(err_pat),
    )


def parse_job_array_ids(job_id: str) -> list[int | None]:
    m = re.search(r"_\[([^\]]+)\]", job_id)
    if m:
        return [int(x) for x in expand_ranges(m.group(1))]
    m = re.search(r"_(\d+)(?:\.|$)", job_id)
    if m:
        return [int(m.group(1))]
    return [None]


def workdir_matches(root: Path, work_dir: str) -> bool:
    work_dir = (work_dir or "").strip()
    if not work_dir:
        return False
    try:
        return Path(work_dir).expanduser().resolve() == root.expanduser().resolve()
    except Exception:
        return os.path.abspath(os.path.expanduser(work_dir)) == os.path.abspath(str(root))


def query_squeue(job_names: set[str], disabled: bool, project_root: Path | None) -> list[JobActivity]:
    if disabled or not shutil.which("squeue"):
        return []
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
            return []
    rows: list[JobActivity] = []
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
            rows.append(JobActivity("squeue", jid, name, aid, state, elapsed, reason, work_dir=work_dir))
    return rows


def query_sacct(job_names: set[str], days: int, disabled: bool, project_root: Path | None) -> list[JobActivity]:
    if disabled or days <= 0 or not shutil.which("sacct"):
        return []
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    cmd = ["sacct", "-n", "-P", "-X", "-S", start, "-o", "JobIDRaw,JobName%80,State,ExitCode,Elapsed,Reason,WorkDir%300"]
    if user:
        cmd.extend(["-u", user])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return []
    rows: list[JobActivity] = []
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
            rows.append(JobActivity("sacct", jid, name, aid, state, elapsed, reason, exit_code, work_dir))
    return rows


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
    return LogFinding(str(path), stream, job, jid, aid, "notice", "non-empty", snippet(text), size, mtime)


def collect_log_findings(infos: dict[str, SlurmInfo], *, scan_contents: bool) -> list[LogFinding]:
    if not scan_contents:
        return []
    import glob
    findings: list[LogFinding] = []
    for info in infos.values():
        for glob_pat, stream in [(info.output_glob, "stdout"), (info.error_glob, "stderr")]:
            if not glob_pat:
                continue
            for raw in glob.glob(glob_pat):
                p = Path(raw)
                if p.is_file():
                    findings.append(classify_log(p, stream))
    return findings

# ---------------------------------------------------------------------------
# Expected file lists
# ---------------------------------------------------------------------------

def expected_npz_specs() -> dict[str, list[TrainNPZSpec]]:
    groups: dict[str, list[TrainNPZSpec]] = defaultdict(list)
    for version, group_name, sets in [
        ("main_default", "Step 1 default NPZs", SETS),
        ("main_transfer", "Step 2 transfer NPZs", TRANSFER_SETS),
        ("main_tuned", "Step 4 tuned NPZs", SETS),
    ]:
        for alg in ALGORITHMS:
            for n in ROBOTS:
                for s in sets:
                    for seed in SEEDS:
                        groups[group_name].append(TrainNPZSpec(group_name, version, alg, n, s, seed))

    for cond in REWARD_CONDITIONS:
        for s in SETS:
            for seed in SEEDS:
                groups["Step 5 reward-ablation training NPZs"].append(
                    TrainNPZSpec("Step 5 reward-ablation training NPZs", f"ablation_reward_{cond}", "CrossQ", ABLATION_N, s, seed)
                )
    for mode in OBS_MODES:
        for s in SETS:
            for seed in SEEDS:
                groups["Step 6 observation-ablation training NPZs"].append(
                    TrainNPZSpec("Step 6 observation-ablation training NPZs", f"ablation_obs_{mode}", "CrossQ", ABLATION_N, s, seed)
                )
    for mode in UNCERTAINTY_MODES:
        for s in SETS:
            for seed in SEEDS:
                groups["Step 7 uncertainty-ablation training NPZs"].append(
                    TrainNPZSpec("Step 7 uncertainty-ablation training NPZs", f"ablation_uncertainty_{mode}", "CrossQ", ABLATION_N, s, seed)
                )
    for mode in DR_MODES:
        for n in ROBOTS:
            for s in SETS:
                for seed in SEEDS:
                    groups["Step 8 DR training NPZs"].append(
                        TrainNPZSpec("Step 8 DR training NPZs", f"dr_{mode}", "CrossQ", n, s, seed)
                    )
    return dict(groups)


def npz_path(log_root: Path, spec: TrainNPZSpec) -> Path:
    return log_root / spec.version / spec.tag / "eval_logs" / "evaluations.npz"

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_npz_file(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        warnings.append(f"numpy unavailable; skipped NPZ load: {exc}")
        return issues, warnings, metrics
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
            metrics["keys"] = sorted(keys)
            missing = {"timesteps", "results"} - keys
            if missing:
                issues.append("missing key(s): " + ", ".join(sorted(missing)))
                return issues, warnings, metrics
            ts = np.asarray(data["timesteps"]).reshape(-1)
            results = np.asarray(data["results"])
            metrics["eval_rows"] = int(len(ts))
            metrics["last_timestep"] = int(ts[-1]) if len(ts) else 0
            if len(ts) == 0:
                issues.append("zero evaluation rows")
            if results.ndim != 2:
                issues.append(f"results has shape {results.shape}, expected 2-D")
            elif results.shape[0] != len(ts):
                issues.append(f"results rows {results.shape[0]} != timesteps rows {len(ts)}")
            if results.size:
                finite = np.isfinite(results.astype(float).reshape(-1))
                if not finite.any():
                    issues.append("all reward values are NaN/Inf")
                elif not finite.all():
                    warnings.append(f"{int((~finite).sum())} nonfinite reward value(s)")
            if "ep_lengths" not in keys:
                warnings.append("ep_lengths missing")
    except Exception as exc:
        issues.append(f"could not load: {exc}")
    return issues, warnings, metrics


def check_npz_group(name: str, specs: list[TrainNPZSpec], log_root: Path, args: argparse.Namespace) -> CheckResult:
    expected = len(specs)
    found = 0
    too_small = 0
    load_bad = 0
    warnings_count = 0
    examples_missing: list[str] = []
    examples_bad: list[str] = []
    total_bytes = 0

    for spec in specs:
        p = npz_path(log_root, spec)
        if not p.exists():
            if len(examples_missing) < args.details_limit:
                examples_missing.append(str(p))
            continue
        if not file_nonempty(p, args.min_npz_bytes):
            too_small += 1
            if len(examples_bad) < args.details_limit:
                examples_bad.append(f"too small: {p}")
            continue
        found += 1
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
        if args.strict and not args.fast:
            issues, warns, _metrics = validate_npz_file(p)
            if issues:
                load_bad += 1
                if len(examples_bad) < args.details_limit:
                    examples_bad.append(f"bad: {p}: {'; '.join(issues[:2])}")
            if warns:
                warnings_count += 1

    critical: list[str] = []
    warnings: list[str] = []
    if found < expected:
        critical.append(f"missing {expected - found}/{expected} evaluations.npz files")
    if too_small:
        critical.append(f"{too_small} evaluations.npz files are too small")
    if load_bad:
        critical.append(f"{load_bad} evaluations.npz files failed strict load/key checks")
    if warnings_count:
        warnings.append(f"{warnings_count} NPZ files had strict warnings")
    if examples_missing:
        warnings.append("missing examples: " + " | ".join(examples_missing[:3]))
    if examples_bad:
        warnings.append("bad examples: " + " | ".join(examples_bad[:3]))
    if args.fast:
        warnings.append("fast mode: checked presence/size only; skipped NPZ loading")

    return CheckResult(
        name=name,
        expected=expected,
        found=found,
        status=status_from_counts(expected, found, critical, warnings, args.assume_finished),
        critical=critical,
        warnings=warnings,
        metrics={"too_small": too_small, "strict_bad": load_bad, "bytes": total_bytes},
    )


def csv_line_count(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def read_csv_rows(path: Path, max_rows: int | None = None) -> tuple[list[dict[str, str]], list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                issues.append("missing header")
                return rows, issues, warnings
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    warnings.append(f"stopped after {max_rows} rows")
                    break
                rows.append({k: (v or "") for k, v in row.items()})
    except Exception as exc:
        issues.append(f"could not read CSV: {exc}")
    return rows, issues, warnings



def check_csv_file(
    name: str,
    path: Path,
    expected_rows: int,
    args: argparse.Namespace,
    *,
    required_columns: set[str] | None = None,
    exact: bool = True,
) -> CheckResult:
    critical: list[str] = []
    warnings: list[str] = []
    found_rows: int | None = None
    metrics: dict[str, Any] = {"path": str(path)}

    if not file_nonempty(path, args.min_csv_bytes):
        critical.append("CSV missing/empty/too small")
        return CheckResult(name, expected_rows, 0, "missing_after_expected_completion" if args.assume_finished else "not_started", critical, warnings, metrics)

    metrics["bytes"] = path.stat().st_size
    line_count = csv_line_count(path)
    if line_count is not None:
        metrics["line_count"] = line_count
        found_rows = max(0, line_count - 1)

    if args.fast:
        if found_rows is None:
            warnings.append("fast mode: file exists, row count unavailable")
            found_rows = 0
        if found_rows < expected_rows:
            critical.append(f"only {found_rows}/{expected_rows} data rows by line count")
        elif exact and found_rows > expected_rows:
            warnings.append(f"{found_rows}/{expected_rows} rows found; possible duplicates or stale rows")
        warnings.append("fast mode: skipped CSV parse/column/duplicate checks")
        return CheckResult(name, expected_rows, found_rows, status_from_counts(expected_rows, found_rows, critical, warnings, args.assume_finished), critical, warnings, metrics)

    rows, issues, warns = read_csv_rows(path)
    critical.extend(issues)
    warnings.extend(warns)
    if not issues:
        found_rows = len(rows)
        metrics["data_rows"] = found_rows
        if rows:
            columns = set(rows[0].keys())
            metrics["columns"] = sorted(columns)
            if required_columns:
                missing_cols = sorted(required_columns - columns)
                if missing_cols:
                    critical.append("missing column(s): " + ", ".join(missing_cols))
            if args.strict:
                key_fields = ["algorithm", "experiment", "ablation", "hp_tag", "num_robots", "env_set", "seed", "eval_wind_min", "eval_wind_max", "eval_uncertainty_mode"]
                if set(key_fields).issubset(columns):
                    keys = [tuple(r.get(k, "") for k in key_fields) for r in rows]
                    duplicate_count = len(keys) - len(set(keys))
                    if duplicate_count:
                        warnings.append(f"{duplicate_count} duplicate evaluation key row(s)")
                for col in ["mean_reward", "iqm", "mean_ep_length", "n_episodes"]:
                    if col in columns:
                        bad = 0
                        for r in rows:
                            try:
                                v = float(r.get(col, "nan"))
                                if not math.isfinite(v):
                                    bad += 1
                            except Exception:
                                bad += 1
                        if bad:
                            warnings.append(f"{bad} nonnumeric/nonfinite value(s) in {col}")
    if found_rows is None:
        found_rows = 0
    if found_rows < expected_rows:
        critical.append(f"only {found_rows}/{expected_rows} data rows")
    elif exact and found_rows > expected_rows:
        warnings.append(f"{found_rows}/{expected_rows} data rows found; possible duplicates, stale rows, or extra evals")
    return CheckResult(name, expected_rows, found_rows, status_from_counts(expected_rows, found_rows, critical, warnings, args.assume_finished), critical, warnings, metrics)


def check_tmp_csv_group(name: str, root: Path, pattern: str, expected_files: int, args: argparse.Namespace) -> CheckResult:
    files = sorted(root.rglob(pattern)) if root.exists() else []
    found = len(files)
    critical: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {"root": str(root), "pattern": pattern}
    too_small = 0
    empty_or_bad = 0
    rows_total = 0
    examples: list[str] = []

    if found < expected_files:
        critical.append(f"only {found}/{expected_files} per-job CSV files")
    elif found > expected_files:
        warnings.append(f"{found}/{expected_files} per-job CSV files found; possible reruns/stale extras")

    if not args.fast:
        for p in files:
            if not file_nonempty(p, args.min_csv_bytes):
                too_small += 1
                if len(examples) < args.details_limit:
                    examples.append(f"empty/too small: {p}")
                continue
            rows, issues, _warns = read_csv_rows(p)
            if issues or not rows:
                empty_or_bad += 1
                if len(examples) < args.details_limit:
                    examples.append(f"bad/no rows: {p}: {'; '.join(issues[:1])}")
                continue
            rows_total += len(rows)
            if len(rows) != 1:
                warnings.append(f"{p} has {len(rows)} rows; expected 1 per per-job CSV")
    else:
        for p in files:
            if not file_nonempty(p, args.min_csv_bytes):
                too_small += 1
        warnings.append("fast mode: counted per-job files/sizes only; skipped CSV parse")

    if too_small:
        critical.append(f"{too_small} per-job CSV file(s) missing/too small")
    if empty_or_bad:
        critical.append(f"{empty_or_bad} per-job CSV file(s) unreadable or empty")
    if examples:
        warnings.append("examples: " + " | ".join(examples[:3]))
    metrics.update({"too_small": too_small, "bad": empty_or_bad, "rows_total": rows_total})
    return CheckResult(name, expected_files, found, status_from_counts(expected_files, found, critical, warnings, args.assume_finished), critical, warnings, metrics)


def check_generated_outputs(name: str, root: Path, files: list[str], args: argparse.Namespace, *, required_now: bool = False) -> CheckResult:
    found = 0
    too_small = 0
    missing: list[str] = []
    for rel in files:
        p = root / rel
        if file_nonempty(p, 1):
            found += 1
        else:
            missing.append(str(p))
            if p.exists():
                too_small += 1
    critical: list[str] = []
    warnings: list[str] = []
    if required_now and found < len(files):
        critical.append(f"missing {len(files) - found}/{len(files)} expected output files")
    elif found < len(files):
        warnings.append(f"{len(files) - found}/{len(files)} outputs not generated yet")
    if too_small:
        critical.append(f"{too_small} output file(s) are empty")
    if missing and args.strict:
        warnings.append("missing examples: " + " | ".join(missing[:3]))
    status = status_from_counts(len(files), found, critical, warnings, args.assume_finished)
    if found == len(files) and not critical:
        status = "already_done" if not warnings else "verified_warn"
    return CheckResult(name, len(files), found, status, critical, warnings, {"missing": missing[:args.details_limit]})


def check_sensitivity_cv(results_dir: Path, args: argparse.Namespace) -> CheckResult:
    path = results_dir / "cv_table.csv"
    critical: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {"path": str(path), "expected_rows": EXPECTED_CV_ROWS}
    if not file_nonempty(path, args.min_csv_bytes):
        critical.append("cv_table.csv missing/empty/too small")
        return CheckResult("sensitivity cv_table.csv", EXPECTED_CV_ROWS, 0, "missing_after_expected_completion" if args.assume_finished else "not_started", critical, warnings, metrics)
    if args.fast:
        lc = csv_line_count(path)
        found = max(0, (lc or 1) - 1)
        if found < EXPECTED_CV_ROWS:
            critical.append(f"only {found}/{EXPECTED_CV_ROWS} rows by line count")
        warnings.append("fast mode: skipped sensitivity CSV coverage parse")
        return CheckResult("sensitivity cv_table.csv", EXPECTED_CV_ROWS, found, status_from_counts(EXPECTED_CV_ROWS, found, critical, warnings, args.assume_finished), critical, warnings, metrics)

    rows, issues, warns = read_csv_rows(path)
    critical.extend(issues)
    warnings.extend(warns)
    found = len(rows)
    columns = set(rows[0].keys()) if rows else set()
    missing_cols = sorted({"algorithm", "hp_name", "cv"} - columns)
    if missing_cols:
        critical.append("missing column(s): " + ", ".join(missing_cols))
    seen: Counter[tuple[str, str]] = Counter()
    bad_cv = 0
    for row in rows:
        alg = row.get("algorithm", "")
        hp = row.get("hp_name", "")
        seen[(alg, hp)] += 1
        try:
            cv = float(row.get("cv", "nan"))
            if not math.isfinite(cv):
                bad_cv += 1
        except Exception:
            bad_cv += 1
    missing_pairs = []
    for alg, hps in HP_REQUIRED.items():
        for hp in hps:
            if seen[(alg, hp)] == 0:
                missing_pairs.append(f"{alg}.{hp}")
    duplicates = sum(count - 1 for count in seen.values() if count > 1)
    if missing_pairs:
        critical.append(f"missing {len(missing_pairs)} required algorithm/HP row(s)")
        warnings.append("missing pairs: " + ", ".join(missing_pairs[:12]))
    if bad_cv:
        critical.append(f"{bad_cv} cv value(s) are missing/nonfinite")
    if duplicates:
        warnings.append(f"{duplicates} duplicate algorithm/HP row(s); sensitivity_hp.py will keep the last value per pair")
    raw = results_dir / "sensitivity_hp_raw.csv"
    if raw.exists():
        raw_lc = csv_line_count(raw)
        raw_rows = max(0, (raw_lc or 1) - 1)
        metrics["raw_rows"] = raw_rows
        if raw_rows < EXPECTED_RAW_ROWS:
            warnings.append(f"sensitivity_hp_raw.csv has {raw_rows}/{EXPECTED_RAW_ROWS} expected grid-point rows")
    else:
        warnings.append("sensitivity_hp_raw.csv not found; not required for --write_latex_only")
    status = status_from_counts(EXPECTED_CV_ROWS, len(seen), critical, warnings, args.assume_finished)
    return CheckResult("sensitivity cv_table.csv", EXPECTED_CV_ROWS, len(seen), status, critical, warnings, metrics)

# ---------------------------------------------------------------------------
# Readiness decisions
# ---------------------------------------------------------------------------

def result_ok(res: CheckResult) -> bool:
    return not res.critical and res.status in {"verified_good", "verified_warn", "already_done", "ready"}


def active_blockers(active: list[JobActivity]) -> tuple[int, int]:
    running = sum(1 for a in active if a.state.upper() in RUNNING_STATES)
    pending = sum(1 for a in active if a.state.upper() in PENDING_STATES)
    other = len(active) - running - pending
    return running + other, pending


def sacct_fatal_rows(rows: list[JobActivity]) -> list[JobActivity]:
    fatal = []
    for r in rows:
        state0 = r.state.upper().split()[0]
        exit_bad = bool(r.exit_code and not r.exit_code.startswith("0:"))
        if state0 in FATAL_SACCT_STATES or exit_bad:
            fatal.append(r)
    return fatal


def collect_input_checks(root: Path, log_root: Path, results_dir: Path, figures_dir: Path, args: argparse.Namespace) -> dict[str, CheckResult]:
    checks: dict[str, CheckResult] = {}
    for script_name in TARGET_PY_SCRIPTS:
        path = root / script_name
        found = 1 if file_nonempty(path, 1) else 0
        critical = [] if found else [f"target script missing/empty: {path}"]
        checks[f"target {script_name}"] = CheckResult(
            name=f"target {script_name}", expected=1, found=found,
            status=status_from_counts(1, found, critical, [], args.assume_finished),
            critical=critical, warnings=[], metrics={"path": str(path)},
        )
    npz_groups = expected_npz_specs()
    for name, specs in npz_groups.items():
        checks[name] = check_npz_group(name, specs, log_root, args)

    tmp = results_dir / "tmp"
    checks["tmp ablation_reward per-job CSVs"] = check_tmp_csv_group(
        "tmp ablation_reward per-job CSVs", tmp / "ablation_reward", "result_*.csv", EXPECTED["eval_ablation_reward"], args
    )
    checks["tmp ablation_uncertainty per-job CSVs"] = check_tmp_csv_group(
        "tmp ablation_uncertainty per-job CSVs", tmp / "ablation_uncertainty", "result_*.csv", EXPECTED["eval_ablation_uncertainty"], args
    )
    checks["tmp DR inDist per-job CSVs"] = check_tmp_csv_group(
        "tmp DR inDist per-job CSVs", tmp / "dr", "inDist_*.csv", EXPECTED["eval_dr"], args
    )
    checks["tmp DR OOD per-job CSVs"] = check_tmp_csv_group(
        "tmp DR OOD per-job CSVs", tmp / "dr", "OOD_*.csv", EXPECTED["eval_dr"], args
    )

    checks["merged ablation_reward.csv"] = check_csv_file(
        "merged ablation_reward.csv", results_dir / "ablation_reward.csv", EXPECTED["eval_ablation_reward"], args, required_columns=REQUIRED_EVAL_COLUMNS
    )
    checks["merged ablation_uncertainty.csv"] = check_csv_file(
        "merged ablation_uncertainty.csv", results_dir / "ablation_uncertainty.csv", EXPECTED["eval_ablation_uncertainty"], args, required_columns=REQUIRED_EVAL_COLUMNS
    )
    checks["merged dr_inDist.csv"] = check_csv_file(
        "merged dr_inDist.csv", results_dir / "dr_inDist.csv", EXPECTED["eval_dr"], args, required_columns=REQUIRED_EVAL_COLUMNS
    )
    checks["merged dr_OOD.csv"] = check_csv_file(
        "merged dr_OOD.csv", results_dir / "dr_OOD.csv", EXPECTED["eval_dr"], args, required_columns=REQUIRED_EVAL_COLUMNS
    )
    checks["wind_sweep.csv"] = check_csv_file(
        "wind_sweep.csv", results_dir / "wind_sweep.csv", EXPECTED["eval_wind_sweep"], args, required_columns=REQUIRED_EVAL_COLUMNS
    )

    checks["analyze_results.py outputs"] = check_generated_outputs("analyze_results.py outputs", results_dir, ANALYZE_OUTPUTS, args, required_now=False)
    checks["plot_figures.py outputs"] = check_generated_outputs("plot_figures.py outputs", figures_dir, FIGURE_OUTPUTS, args, required_now=False)
    checks["sensitivity cv_table.csv"] = check_sensitivity_cv(results_dir, args)
    checks["sensitivity_hp_latex_rows.txt"] = check_generated_outputs("sensitivity_hp_latex_rows.txt", results_dir, ["sensitivity_hp_latex_rows.txt"], args, required_now=False)
    return checks


def merge_sources_ok(checks: dict[str, CheckResult]) -> bool:
    return all(result_ok(checks[k]) for k in [
        "tmp ablation_reward per-job CSVs",
        "tmp ablation_uncertainty per-job CSVs",
        "tmp DR inDist per-job CSVs",
        "tmp DR OOD per-job CSVs",
    ])


def merged_csvs_ok(checks: dict[str, CheckResult]) -> bool:
    return all(result_ok(checks[k]) for k in [
        "merged ablation_reward.csv",
        "merged ablation_uncertainty.csv",
        "merged dr_inDist.csv",
        "merged dr_OOD.csv",
    ])


def main_npz_ok(checks: dict[str, CheckResult]) -> bool:
    return all(result_ok(checks[k]) for k in [
        "Step 1 default NPZs",
        "Step 2 transfer NPZs",
        "Step 4 tuned NPZs",
    ])


def analyze_npz_ok(checks: dict[str, CheckResult]) -> bool:
    return main_npz_ok(checks) and result_ok(checks["Step 6 observation-ablation training NPZs"])


def analysis_outputs_ok_for_plot(checks: dict[str, CheckResult], results_dir: Path) -> bool:
    # plot_figures.py can still run without some summaries but will skip/limit
    # figures; for readiness we require the outputs analyze_results.py is meant
    # to create, especially main_tuned_summary.csv for scalability.
    required = [
        results_dir / "main_tuned_summary.csv",
        results_dir / "main_default_summary.csv",
        results_dir / "main_transfer_summary.csv",
    ]
    return all(file_nonempty(p) for p in required)


def command_readiness(checks: dict[str, CheckResult], active: list[JobActivity], fatal_sacct: list[JobActivity], fatal_logs: list[LogFinding], results_dir: Path, args: argparse.Namespace) -> list[list[Any]]:
    running, pending = active_blockers(active)
    upstream_block = running or pending
    fatal_block = bool(fatal_sacct or fatal_logs)

    rows: list[list[Any]] = []

    if upstream_block:
        upstream_reason = f"WAIT: {running} running/active, {pending} pending upstream Slurm task(s)"
    elif fatal_block:
        upstream_reason = "CHECK: fatal Slurm/log evidence exists; inspect before trusting outputs"
    else:
        upstream_reason = "no active upstream Slurm jobs seen"

    merge_script_ok = result_ok(checks["target merge_eval_results.py"])
    merge_ok = merge_sources_ok(checks)
    if not merge_script_ok:
        merge_status = "BLOCK"
        merge_reason = checks["target merge_eval_results.py"].critical[0]
    elif upstream_block:
        merge_status = "WAIT"
        merge_reason = upstream_reason
    elif merge_ok:
        merge_status = "READY"
        merge_reason = "all per-job eval CSV sources are present"
    else:
        merge_status = "BLOCK"
        merge_reason = first_problem(checks, [
            "tmp ablation_reward per-job CSVs",
            "tmp ablation_uncertainty per-job CSVs",
            "tmp DR inDist per-job CSVs",
            "tmp DR OOD per-job CSVs",
        ])
    rows.append(["1", "merge_eval_results.py", merge_status, merge_reason])

    eval_csv_ready_now = merged_csvs_ok(checks)
    eval_csv_ready_after_merge = merge_ok
    analyze_inputs = analyze_npz_ok(checks) and (eval_csv_ready_now or eval_csv_ready_after_merge)
    wind_ok = result_ok(checks["wind_sweep.csv"])
    analyze_script_ok = result_ok(checks["target analyze_results.py"])
    if not analyze_script_ok:
        analyze_status = "BLOCK"
        analyze_reason = checks["target analyze_results.py"].critical[0]
    elif upstream_block:
        analyze_status = "WAIT"
        analyze_reason = upstream_reason
    elif analyze_inputs:
        analyze_status = "READY" if eval_csv_ready_now else "READY AFTER MERGE"
        analyze_reason = "training NPZs + evaluation CSVs are available" if eval_csv_ready_now else "training NPZs OK; run merge first to create aggregate CSVs"
    else:
        analyze_status = "BLOCK"
        analyze_reason = first_problem(checks, [
            "Step 1 default NPZs",
            "Step 2 transfer NPZs",
            "Step 4 tuned NPZs",
            "Step 6 observation-ablation training NPZs",
            "merged ablation_reward.csv" if not merge_ok else "tmp ablation_reward per-job CSVs",
            "merged ablation_uncertainty.csv" if not merge_ok else "tmp ablation_uncertainty per-job CSVs",
            "merged dr_inDist.csv" if not merge_ok else "tmp DR inDist per-job CSVs",
            "merged dr_OOD.csv" if not merge_ok else "tmp DR OOD per-job CSVs",
        ])
    rows.append(["2", "analyze_results.py", analyze_status, analyze_reason])

    plot_npz_ok = all(result_ok(checks[k]) for k in [
        "Step 1 default NPZs",
        "Step 2 transfer NPZs",
        "Step 4 tuned NPZs",
        "Step 8 DR training NPZs",
    ])
    analysis_ready_now = analysis_outputs_ok_for_plot(checks, results_dir)
    plot_script_ok = result_ok(checks["target plot_figures.py"])
    if not plot_script_ok:
        plot_status = "BLOCK"
        plot_reason = checks["target plot_figures.py"].critical[0]
    elif upstream_block:
        plot_status = "WAIT"
        plot_reason = upstream_reason
    elif plot_npz_ok and wind_ok and analysis_ready_now:
        plot_status = "READY"
        plot_reason = "learning-curve NPZs, DR NPZs, summaries, and wind_sweep.csv are present"
    elif plot_npz_ok and wind_ok and analyze_inputs:
        plot_status = "READY AFTER ANALYZE"
        plot_reason = "inputs OK; run analyze_results.py first for summary CSVs"
    else:
        plot_status = "BLOCK"
        plot_reason = first_problem(checks, [
            "Step 1 default NPZs",
            "Step 2 transfer NPZs",
            "Step 4 tuned NPZs",
            "Step 8 DR training NPZs",
            "wind_sweep.csv",
            "analyze_results.py outputs",
        ])
    rows.append(["3", "plot_figures.py", plot_status, plot_reason])

    sens_script_ok = result_ok(checks["target sensitivity_hp.py"])
    sens_ok = result_ok(checks["sensitivity cv_table.csv"])
    if not sens_script_ok:
        sens_status = "BLOCK"
        sens_reason = checks["target sensitivity_hp.py"].critical[0]
    elif upstream_block:
        sens_status = "WAIT"
        sens_reason = upstream_reason
    elif sens_ok:
        sens_status = "READY"
        sens_reason = "cv_table.csv has required algorithm/HP coverage"
    else:
        sens_status = "BLOCK"
        sens_reason = first_problem(checks, ["sensitivity cv_table.csv"])
    rows.append(["4", "sensitivity_hp.py --write_latex_only", sens_status, sens_reason])

    return rows


def first_problem(checks: dict[str, CheckResult], keys: list[str]) -> str:
    for k in keys:
        res = checks.get(k)
        if res is None:
            continue
        if res.critical:
            return f"{k}: {res.critical[0]}"
    for k in keys:
        res = checks.get(k)
        if res is None:
            continue
        if res.warnings and not result_ok(res):
            return f"{k}: {res.warnings[0]}"
    return "missing or incomplete required inputs"

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_checks(checks: dict[str, CheckResult], keys: list[str]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in keys:
        res = checks[key]
        issue = "; ".join((res.critical + res.warnings)[:2]) or "---"
        rows.append([res.name, human(res.expected), human(res.found), res.status, issue])
    return rows


def print_active_jobs(active: list[JobActivity], details_limit: int) -> None:
    if not active:
        print("  No active upstream Slurm jobs found.")
        return
    counts = Counter(a.state for a in active)
    print("  Active Slurm states:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    rows = []
    for a in active[:details_limit]:
        rows.append([a.job_name, a.job_id, a.array_id if a.array_id is not None else "?", a.state, a.elapsed, a.reason])
    print_table(["job", "job_id", "array", "state", "elapsed", "reason"], rows)


def print_fatal_evidence(fatal_sacct: list[JobActivity], fatal_logs: list[LogFinding], limit: int) -> None:
    rows: list[list[Any]] = []
    for r in fatal_sacct[:limit]:
        rows.append(["sacct", r.job_name, r.job_id, r.array_id if r.array_id is not None else "?", r.state, r.exit_code or "?", r.reason or "?"])
    remaining = max(0, limit - len(rows))
    for f in fatal_logs[:remaining]:
        rows.append([f.stream, f.job_name or "?", f.job_id or "?", f.array_id if f.array_id is not None else "?", f.reason, Path(f.path).name, f.snippet])
    print_table(["source", "job", "job_id", "array", "state/reason", "exit/file", "detail"], rows)


def command_block() -> str:
    return """
Recommended command sequence when READY:
  python merge_eval_results.py --results_dir logs/results
  python analyze_results.py --log_root logs --results_dir logs/results
  python plot_figures.py --log_root logs --results_dir logs/results --figures_dir figures
  python sensitivity_hp.py --write_latex_only --results_dir logs/results
""".strip()

# ---------------------------------------------------------------------------
# Arg parsing and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print-only readiness checker for merge/analyze/plot/sensitivity final commands.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=QUICK_GUIDE,
    )
    p.add_argument("--guide", action="store_true", help="Print the concise command guide and exit.")
    p.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT_NAME,
                   help=f"Project root. Default: {DEFAULT_PROJECT_ROOT_NAME}; if already inside the repo, cwd is used.")
    p.add_argument("--log_root", default=DEFAULT_LOG_ROOT)
    p.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--figures_dir", default=DEFAULT_FIGURES_DIR)
    p.add_argument("--script_dir", default=DEFAULT_SCRIPT_DIR)
    p.add_argument("--scope", choices=["all", "merge", "analyze", "plot", "sensitivity"], default="all")

    p.add_argument("--fast", action="store_true",
                   help="Fast shallow check: squeue + file counts/sizes; skips sacct, log scans, CSV deep checks, and NPZ loading.")
    p.add_argument("--done", dest="assume_finished", action="store_true",
                   help="Alias for --assume-finished: missing final inputs are blockers/actionable.")
    p.add_argument("--assume-finished", dest="assume_finished", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="Parse CSVs, validate NPZ keys, require exact row coverage, and report output coverage strictly.")
    p.add_argument("--forensic", action="store_true",
                   help="Done + strict + sacct/log fatal scan + wider details.")

    p.add_argument("--no-squeue", action="store_true")
    p.add_argument("--no-sacct", action="store_true")
    p.add_argument("--no-slurm-workdir-filter", action="store_true",
                   help="Do not restrict squeue/sacct records to this project WorkDir.")
    p.add_argument("--sacct-days", type=int, default=14)
    p.add_argument("--details-limit", type=int, default=30)
    p.add_argument("--error-details-limit", type=int, default=20)
    p.add_argument("--min-npz-bytes", type=int, default=512)
    p.add_argument("--min-csv-bytes", type=int, default=16)
    args = p.parse_args()

    if args.guide:
        return args
    if args.fast:
        args.no_sacct = True
        args.strict = False
    if args.forensic:
        args.assume_finished = True
        args.strict = True
        args.sacct_days = max(args.sacct_days, 60)
        args.details_limit = max(args.details_limit, 80)
        args.error_details_limit = max(args.error_details_limit, 60)
    return args


def main() -> int:
    args = parse_args()
    if args.guide:
        print(QUICK_GUIDE)
        return 0

    t0 = time.time()
    root = resolve_project_root(args.project_root)
    log_root = rel_or_abs(root, args.log_root)
    results_dir = rel_or_abs(root, args.results_dir)
    figures_dir = rel_or_abs(root, args.figures_dir)

    infos = {s: parse_slurm_script(root, args.script_dir, s) for s in UPSTREAM_SCRIPTS}
    job_names = {info.job_name for info in infos.values()}
    project_for_slurm = None if args.no_slurm_workdir_filter else root
    active = query_squeue(job_names, args.no_squeue, project_for_slurm)
    sacct_rows = query_sacct(job_names, args.sacct_days, args.no_sacct, project_for_slurm)
    fatal_sacct = sacct_fatal_rows(sacct_rows)
    log_findings = collect_log_findings(infos, scan_contents=(args.forensic and not args.fast))
    fatal_logs = [f for f in log_findings if f.severity == "fatal"]

    checks = collect_input_checks(root, log_root, results_dir, figures_dir, args)
    readiness_rows = command_readiness(checks, active, fatal_sacct, fatal_logs, results_dir, args)

    print("\nCHECK_STATUS_ALL_TRAIN_EVAL")
    print("=" * 80)
    print(f"Project root : {root}")
    print(f"Log root     : {log_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Figures dir  : {figures_dir}")
    print(f"Scope        : {args.scope}")
    print(f"Mode         : {'fast' if args.fast else 'forensic' if args.forensic else 'strict' if args.strict else 'normal'}")
    print(f"Elapsed      : {time.time() - t0:.1f}s")

    print("\nUpstream Slurm state")
    print("-" * 80)
    print_active_jobs(active, args.details_limit)
    if fatal_sacct or fatal_logs:
        print("\nFatal Slurm/log evidence")
        print("-" * 80)
        print_fatal_evidence(fatal_sacct, fatal_logs, args.error_details_limit)

    print("\nReadiness for final commands")
    print("-" * 80)
    rows_to_show = readiness_rows
    if args.scope != "all":
        scope_match = {
            "merge": "merge_eval_results.py",
            "analyze": "analyze_results.py",
            "plot": "plot_figures.py",
            "sensitivity": "sensitivity_hp.py --write_latex_only",
        }[args.scope]
        rows_to_show = [r for r in readiness_rows if r[1] == scope_match]
    print_table(["#", "command", "status", "reason"], rows_to_show)

    if args.scope == "all":
        print("\nTarget Python scripts")
        print("-" * 80)
        print_table(["item", "expected", "found", "status", "issue"], summarize_checks(checks, [
            "target merge_eval_results.py",
            "target analyze_results.py",
            "target plot_figures.py",
            "target sensitivity_hp.py",
        ]))

    if args.scope in {"all", "merge"}:
        print("\nMerge inputs / outputs")
        print("-" * 80)
        print_table(["item", "expected", "found", "status", "issue"], summarize_checks(checks, [
            "tmp ablation_reward per-job CSVs",
            "tmp ablation_uncertainty per-job CSVs",
            "tmp DR inDist per-job CSVs",
            "tmp DR OOD per-job CSVs",
            "merged ablation_reward.csv",
            "merged ablation_uncertainty.csv",
            "merged dr_inDist.csv",
            "merged dr_OOD.csv",
            "wind_sweep.csv",
        ]))

    if args.scope in {"all", "analyze", "plot"}:
        print("\nTraining NPZ coverage needed by analyze/plot")
        print("-" * 80)
        print_table(["item", "expected", "found", "status", "issue"], summarize_checks(checks, [
            "Step 1 default NPZs",
            "Step 2 transfer NPZs",
            "Step 4 tuned NPZs",
            "Step 5 reward-ablation training NPZs",
            "Step 6 observation-ablation training NPZs",
            "Step 7 uncertainty-ablation training NPZs",
            "Step 8 DR training NPZs",
        ]))

    if args.scope in {"all", "analyze", "plot"}:
        print("\nGenerated analysis / figure outputs")
        print("-" * 80)
        print_table(["item", "expected", "found", "status", "issue"], summarize_checks(checks, [
            "analyze_results.py outputs",
            "plot_figures.py outputs",
        ]))

    if args.scope in {"all", "sensitivity"}:
        print("\nSensitivity table readiness")
        print("-" * 80)
        print_table(["item", "expected", "found", "status", "issue"], summarize_checks(checks, [
            "sensitivity cv_table.csv",
            "sensitivity_hp_latex_rows.txt",
        ]))

    print("\n" + command_block())
    print("\nGuide: python check_status_all_train_eval.py --guide")

    # Exit code: 0 when every requested command is ready or ready-after-prior-step.
    bad_statuses = {"BLOCK", "WAIT"}
    requested_rows = rows_to_show
    if any(r[2] in bad_statuses for r in requested_rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
