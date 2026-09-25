"""
Flask API for the Scope 3 AI Auditor pipeline.

Wraps the four standalone pipeline/step_*.py CLI scripts as background
subprocesses (same Popen/env pattern as _original_app.py's
run_script_with_progress), and exposes job status + output artifacts
over HTTP for the React frontend to poll.

# ponytail: JOBS is a single in-memory dict shared by the whole process
# (one Flask instance, no per-session isolation) - fine for a single local
# user. Upgrade to per-session job scoping (e.g. keyed by a session cookie)
# if this is ever exposed to multiple concurrent users.
"""
import codecs
import ctypes
import hashlib
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import psutil
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.security import safe_join
from werkzeug.exceptions import BadRequest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Output/cwd handling: the step scripts do bare `pd.read_csv('waste_codes.csv')`
# / `pd.read_excel('regional_emission_factors.xlsx')` and write bare
# `generated_data/...`, `dashboards_<company>/...`, `evaluation_plots_.../...`
# - all resolved relative to the process cwd, and none of the argparse
# parsers expose a --data-dir/--out-dir flag. So the simplest robust fix is:
# run each script with cwd=SCRATCH_DIR, and copy the 3 data assets into that
# same cwd once so the bare relative reads find them. Imports of sibling
# modules (hardware_profiler, shared) still work regardless of cwd because
# Python puts the *script's own* directory on sys.path[0] when it's launched
# directly (and step_2 additionally appends its own dir explicitly).
#
# SCRATCH_DIR defaults to "<project_root>/output" rather than the OS temp dir:
# the project folder is synced by OneDrive across machines/platforms, while
# tempfile.gettempdir() is a different, unsynced, per-OS location (and isn't
# even the same path shape on Windows vs Mac). os.path.join keeps the path
# itself platform-correct either way.
SCRATCH_DIR = os.environ.get("SCOPE3_OUT") or os.path.join(BASE_DIR, "output")
APP_RUNS_DIR = os.path.join(SCRATCH_DIR, "app-runs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend", "dist")

DATA_ASSETS = ["waste_codes.csv", "regional_emission_factors.xlsx"]
PUBLIC_DEMO = os.path.isfile(os.path.join(BASE_DIR, "PUBLIC-DEMO-MODE"))

SCRIPTS = {
    "demo": os.path.join("..", "tools", "run_demo.py"),
    "generate": "step_1_generate_data.py",
    "verify": "step_2_verify_data.py",
    "accuracy": "step_3_accuracy_test.py",
    "materiality": "step_4_materiality.py",
}

# Friendly stage text per script, matched against streamed stdout lines.
# (Ported from _original_app.py's STAGE_DESCRIPTIONS; percent scaling is
# simplified here to "use tqdm's own %" instead of the original's per-company
# slot remapping - good enough for a progress bar + log tail.)
STAGE_DESCRIPTIONS = {
    "step_1_generate_data.py": [
        (r"Generating", "Generating deep-dive synthetic data..."),
        (r"Building Profile", "Building company profile with product recipes..."),
        (r"Generated \d+", "Monthly production records generated..."),
        (r"Data Generation Complete", "Data generation complete - files saved..."),
    ],
    "step_2_verify_data.py": [
        (r"Hardware Profile", "Detecting hardware capabilities..."),
        (r"Training Enhanced", "Training context-aware audit classifier..."),
        (r"GREENWASH", "Running mass-balance greenwashing detectors..."),
        (r"XGBoost|Regressor", "Training XGBoost regressors..."),
        (r"SHAP", "Generating SHAP feature importance explanations..."),
        (r"Precomputing Cache", "Precomputing physical series cache..."),
        (r"Targets|DAG order", "Discovering verification targets..."),
        (r"GUARDRAIL", "Applying physics-based mass-balance guardrails..."),
        (r"Benford", "Running Benford's Law verification..."),
        (r"Starting verification", "Starting AI verification pipeline..."),
    ],
    "step_3_accuracy_test.py": [
        (r"Loading CSV", "Loading pristine, messy and AI-corrected datasets..."),
        (r"Phase 2|Generating 6-Panel", "Generating 6-panel accuracy dashboards..."),
        (r"saved to", "Saving diagnostic plots to disk..."),
    ],
    "step_4_materiality.py": [
        (r"Aggregating illustrative emissions-intensity", "Aggregating illustrative emissions-intensity scores..."),
        (r"Loading CSV", "Loading datasets for materiality scoring..."),
        (r"ILLUSTRATIVE EMISSIONS INTENSITY REPORT", "Calculating illustrative emissions-intensity scores..."),
        (r"PRODUCTION EFFICIENCY", "Computing production efficiency metrics..."),
    ],
}

JOBS = {}
JOBS_LOCK = threading.Lock()
ACTIVE_JOB_ID = None
MAX_JOBS = 25
MAX_LOG_LINES = 300
MAX_LOG_LINE_BYTES = 16_384
MAX_BODY_BYTES = 65_536
MAX_JOB_DEADLINE_SECONDS = 4 * 60 * 60
MAX_APP_RUNS = 25
MAX_APP_RUN_BYTES = 2 * 1024**3


def _deadline_seconds(value):
    if value is None:
        return MAX_JOB_DEADLINE_SECONDS
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return MAX_JOB_DEADLINE_SECONDS
    return seconds if 1 <= seconds <= MAX_JOB_DEADLINE_SECONDS else MAX_JOB_DEADLINE_SECONDS


JOB_DEADLINE_SECONDS = _deadline_seconds(os.environ.get("SCOPE3_JOB_DEADLINE_SECONDS"))

ERROR_TYPES = {
    "fat_finger", "dropped_zero", "unit_conversion", "keyboard_mistype",
    "true_random_error", "scale_up_1000", "lbs_to_kg_confusion",
    "currency_confusion", "repeated_digits", "off_by_one_digit",
    "random_noise_high", "random_noise_low", "accidental_zero", "negative_value",
}
ALLOWED_KEYS = {
    "demo": set(),
    "generate": {"companies", "products", "ollama_model", "error_types", "error_rate", "seed"},
    "verify": {"companies", "mode", "scope", "hardware", "shap_mode", "ollama_model", "engine", "all", "source_job_id"},
    "accuracy": {"companies", "all", "source_job_id"},
    "materiality": {"companies", "source_job_id"},
}


def _prepare_scratch(root=SCRATCH_DIR):
    os.makedirs(root, exist_ok=True)
    for fname in DATA_ASSETS:
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root, fname))
        elif PUBLIC_DEMO and fname == "waste_codes.csv":
            raise FileNotFoundError("invented demo waste_codes.csv is missing")


def _source_job(step, body):
    expected = {"verify": "generate", "accuracy": "verify", "materiality": "accuracy"}.get(step)
    if not expected:
        return None
    job_id = body.get("source_job_id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError(f"{step} requires the completed {expected} job_id")
    job = JOBS.get(job_id)
    if not job or job.get("step") != expected or job.get("state") != "done":
        raise ValueError(f"source_job_id must identify a completed {expected} job")
    requested_companies = int(body.get("companies", 1))
    if requested_companies != job.get("companies"):
        raise ValueError(
            f"companies must match the source job ({job.get('companies')}); "
            "start a new generation job to change the company count"
        )
    project = Path(job["root"], "project 01")
    if not project.is_dir():
        raise ValueError("source job output is no longer available")
    recorded = job.get("project_manifest")
    try:
        current = _project_manifest(project)
    except (OSError, ValueError):
        current = None
    if not recorded or current != recorded:
        raise ValueError("source job content changed after completion")
    pristine = sorted((project / "generated company").glob("COMP_*_pristine.csv"))
    if len(pristine) != requested_companies or any(
        not path.with_name(path.name.replace("_pristine.csv", "_messy.csv")).is_file()
        for path in pristine
    ):
        raise ValueError("source job is missing its generated company pair")
    if expected in {"verify", "accuracy"} and any(
        not (project / "ai corrected" / path.name.replace("_pristine.csv", "_AI_corrected.csv")).is_file()
        for path in pristine
    ):
        raise ValueError("source job is missing corrected company output")
    return job


def _prepare_job_root(step, root, source_job):
    root.mkdir(parents=True)
    _prepare_scratch(str(root))
    if source_job:
        source = Path(source_job["root"], "project 01")
        recorded = source_job["project_manifest"]
        if _project_manifest(source) != recorded:
            raise ValueError("source job content changed before copy")
        shutil.copytree(source, root / "project 01")
        if _project_manifest(root / "project 01") != recorded:
            raise ValueError("source job content changed during copy")
    elif step == "demo" and not PUBLIC_DEMO:
        shutil.copytree(Path(DATA_DIR, "demo_project"), root / "project 01")


def _app_runs_usage():
    root = Path(APP_RUNS_DIR)
    if not root.is_dir():
        return 0, 0
    runs = [path for path in root.iterdir() if path.is_dir() and not path.is_symlink()]
    size = sum(path.stat().st_size for run in runs for path in run.rglob("*") if path.is_file() and not path.is_symlink())
    return len(runs), size


def _job_files(root):
    root = Path(root).resolve()
    files = {}
    if not root.is_dir():
        return files
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.name in DATA_ASSETS:
            continue
        if path.suffix.lower() not in {".csv", ".json", ".png", ".jpg", ".jpeg"}:
            continue
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
            stat = resolved.stat()
        except (OSError, ValueError):
            continue
        files[relative] = (stat.st_size, _sha256_file(resolved))
    return files


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_manifest(project):
    project = Path(project).resolve(strict=True)
    manifest = {}
    for path in sorted(project.rglob("*")):
        if path.is_symlink():
            raise ValueError("source job contains a link")
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(project).as_posix()
        manifest[relative] = _sha256_file(resolved)
    return manifest


def _freeze_artifacts(job):
    if "root" not in job:
        return
    after = _job_files(job["root"])
    before = job["artifact_before"]
    job["artifacts"] = {path: fingerprint for path, fingerprint in after.items() if before.get(path) != fingerprint}
    if job.get("state") == "done":
        try:
            job["project_manifest"] = _project_manifest(Path(job["root"], "project 01"))
        except (OSError, ValueError):
            with job["lock"]:
                job["state"] = "error"
                job["stage"] = "Completed process did not leave a valid project state."
                job["log"].append(job["stage"])


def _match_stage(script_name, line):
    for pattern, desc in STAGE_DESCRIPTIONS.get(script_name, []):
        if re.search(pattern, line, re.IGNORECASE):
            return desc
    return None


# ---------------------------------------------------------------------------
# PERCENT SCALING — ported line-for-line from _original_app.py's
# run_script_with_progress (its per-script/per-company tqdm remapping).
# The naive "just track tqdm's own raw %" approach breaks the moment a script
# runs its own tqdm loop more than once (once per company, or a Phase-1/
# Phase-2 split): each loop restarts at 0%, but since the reported percent is
# monotonic-max, the bar snaps to 100% on the FIRST loop and then sits there,
# frozen, while the job keeps running for real. Scaling each loop into its own
# slice of the overall 0-100% range is what makes the bar track true progress.
# ---------------------------------------------------------------------------
TQDM_RE = re.compile(r"(?:([\w().:\-\s]+?):\s*)?(\d{1,3})%\|")

_MAPPING_PRODUCT_RE = re.compile(r"Mapping product (\d+)/(\d+) for COMP_(\d+)")
_BUILD_PROFILE_RE = re.compile(r"Building Profile for COMP_(\d+)")
_GENERATED_RE = re.compile(r"Generated \d+ monthly.*for COMP_(\d+)")
_COMP_NUM_RE = re.compile(r"COMP_(\d+)")

_ACCURACY_TARGET_COLS = [
    "supplier_emissions_mtco2", "waste_emissions_mtco2", "utility_emissions_mtco2",
    "total_product_emissions_mtco2", "grid_elec_kwh", "non_grid_energy_mj", "water_use_m3",
]
_ACCURACY_TARGET_RE = re.compile(r"\b(" + "|".join(_ACCURACY_TARGET_COLS) + r")\b")


def _map_verify_pct(desc, pct, total_comps):
    if desc == "Pre-calculating Medians":
        return int(pct * 0.15)  # Phase 1: 0-15%
    if desc.startswith("Precomputing Cache (COMP_"):
        m = _COMP_NUM_RE.search(desc)
        if m:
            comp_idx = int(m.group(1)) - 1
            # Cache precomputation occupies the first 50% of the company slot
            comp_prog = (comp_idx / total_comps) * 100 + (pct / total_comps) * 0.5
            return 15 + int(comp_prog * 0.85)
        return pct
    if desc.startswith("Targets (COMP_"):
        m = _COMP_NUM_RE.search(desc)
        if m:
            comp_idx = int(m.group(1)) - 1
            # Target evaluation occupies the remaining 50% of the company slot
            comp_prog = (comp_idx / total_comps) * 100 + 50.0 / total_comps + (pct / total_comps) * 0.5
            return 15 + int(comp_prog * 0.85)
        return pct
    if desc == "Overall Progress":
        return 15 + int(pct * 0.85)
    return pct


def _map_accuracy_pct(desc, pct):
    if desc == "Loading CSVs":
        return int(pct * 0.2)  # Phase 1: 0-20%
    if desc == "Generating 6-Panel Dashboards":
        return 20 + int(pct * 0.8)  # Phase 2: 20-100%
    return pct


def _map_materiality_pct(desc, pct):
    if desc == "Loading CSVs":
        return 15 + int(pct * 0.25)  # 15-40%
    return pct


def _compute_progress(job, line):
    """Returns (updated_pct or None, display_text or None), mirroring the
    original's tqdm-based mapping first, then its text-only fallback markers."""
    script = job["script"]
    total_comps = max(1, job.get("companies", 1))

    m = TQDM_RE.search(line)
    if m:
        desc = (m.group(1) or "").strip()
        pct = int(m.group(2))
        if not (0 <= pct <= 100):
            return None, None
        if script == "step_2_verify_data.py":
            mapped = _map_verify_pct(desc, pct, total_comps)
        elif script == "step_3_accuracy_test.py":
            mapped = _map_accuracy_pct(desc, pct)
        elif script == "step_4_materiality.py":
            mapped = _map_materiality_pct(desc, pct)
        else:
            mapped = pct
        return max(0, min(100, mapped)), (desc or None)

    # Text-only fallback markers (no tqdm bar on this line)
    if script == "step_1_generate_data.py":
        mm = _MAPPING_PRODUCT_RE.search(line)
        if mm:
            p, total_p, comp_num = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
            comp_idx = comp_num - 1
            comp_prog = (p / total_p) * 80  # product mapping = 80% of the company slot
            pct = int((comp_idx / total_comps) * 100 + (comp_prog / total_comps))
            return pct, f"Generating data - company {comp_num}/{total_comps} (product {p}/{total_p})"
        mm = _BUILD_PROFILE_RE.search(line)
        if mm:
            idx = int(mm.group(1))
            pct = int(((idx - 1) / total_comps) * 100)
            return pct, f"Generating data - company {idx}/{total_comps} (building profile)"
        mm = _GENERATED_RE.search(line)
        if mm:
            idx = int(mm.group(1))
            pct = int(idx / total_comps * 100)
            return pct, f"Generating data - company {idx}/{total_comps} (records generated)"
    elif script == "step_3_accuracy_test.py":
        if "Phase 1: Loading CSVs" in line:
            return 10, "Phase 1: loading CSVs..."
        if "Phase 2: Generating 6-Panel Dashboards" in line:
            return 30, "Phase 2: generating dashboards..."
        if "diagnostic plots saved to" in line or "All 6-panel dashboards saved" in line:
            return 98, "Saving diagnostics to disk..."
        mm = _ACCURACY_TARGET_RE.search(line)
        if mm:
            idx = _ACCURACY_TARGET_COLS.index(mm.group(1)) + 1
            total = len(_ACCURACY_TARGET_COLS)
            return 30 + int(idx / total * 65), f"Generating dashboards - target {idx}/{total}"
    elif script == "step_4_materiality.py":
        if "Aggregating illustrative emissions-intensity" in line:
            return 15, "Aggregating illustrative emissions-intensity scores..."
        if "Total Companies Evaluated:" in line:
            return 35, "Calculating illustrative emissions-intensity scores..."
        if "PRODUCTION EFFICIENCY" in line:
            return 60, "Calculating production efficiency ratios..."
        if "MODEL OUTPUT - ILLUSTRATIVE" in line:
            return 80, "Analyzing score movements..."
        if "REPORTED EMISSIONS BELOW PRISTINE" in line:
            return 95, "Summarizing reported emissions below pristine aggregate..."

    return None, None


def _process_line(job, line):
    line = line.replace("\r", " ").replace("\n", " ")
    encoded = line.encode("utf-8", "replace")[:MAX_LOG_LINE_BYTES]
    line = encoded.decode("utf-8", "ignore")
    with job["lock"]:
        job["log"].append(line)
        pct, text = _compute_progress(job, line)
        if pct is not None and pct > job["percent"]:
            job["percent"] = pct
        if text:
            job["stage"] = text
        else:
            matched = _match_stage(job["script"], line)
            if matched:
                job["stage"] = matched


def _read_process(job):
    proc = job["proc"]
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    assert proc.stdout is not None
    while True:
        byte_char = proc.stdout.read(1)
        if not byte_char:
            break
        try:
            char = decoder.decode(byte_char)
        except Exception:
            char = ""
        if not char:
            continue
        if char in ("\r", "\n"):
            line = buffer.rstrip("\r\n")
            buffer = ""
            if line.strip():
                _process_line(job, line)
        else:
            buffer += char
            if len(buffer.encode("utf-8", "replace")) >= MAX_LOG_LINE_BYTES:
                _process_line(job, buffer)
                buffer = ""
    if buffer.strip():
        _process_line(job, buffer)
    proc.wait()
    with job["lock"]:
        if job["state"] == "running":
            job["state"] = "done" if proc.returncode == 0 else "error"
        if job["state"] == "done":
            job["percent"] = 100
        job["finished_at"] = time.time()


def _reader_thread(job):
    try:
        _read_process(job)
    except Exception as exc:
        with job["lock"]:
            if job["state"] == "running":
                job["state"] = "error"
                job["stage"] = "Output reader failed."
            job["log"].append(f"Output reader failed: {type(exc).__name__}")
            job["finished_at"] = time.time()
    finally:
        try:
            _stop_process_tree(job)
        finally:
            try:
                _freeze_artifacts(job)
                close_stdout = getattr(job["proc"].stdout, "close", None)
                if close_stdout:
                    close_stdout()
            finally:
                job["finished"].set()
                _release_job(job["id"])


def _release_job(job_id):
    global ACTIVE_JOB_ID
    with JOBS_LOCK:
        if ACTIVE_JOB_ID == job_id:
            ACTIVE_JOB_ID = None


if sys.platform.startswith("win"):
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMITS(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMITS), ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _BASIC_ACCOUNTING(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _KERNEL32.TerminateJobObject.restype = wintypes.BOOL
    _KERNEL32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    _KERNEL32.QueryInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _CREATE_SUSPENDED = 0x00000004
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _new_windows_job():
    handle = _KERNEL32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = _EXTENDED_LIMITS()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _KERNEL32.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.WinError(ctypes.get_last_error())
        _KERNEL32.CloseHandle(handle)
        raise error
    return handle


def _assign_windows_job(handle, pid):
    access = _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION
    process_handle = _KERNEL32.OpenProcess(access, False, pid)
    if not process_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not _KERNEL32.AssignProcessToJobObject(handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _KERNEL32.CloseHandle(process_handle)


def _windows_job_active(handle):
    accounting = _BASIC_ACCOUNTING()
    if not _KERNEL32.QueryInformationJobObject(handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
        raise ctypes.WinError(ctypes.get_last_error())
    return accounting.ActiveProcesses


def _stop_process_tree(job):
    """Stop the OS-owned job/process group, then wait before releasing admission."""
    proc = job.get("proc")
    if proc is None:
        return
    if sys.platform.startswith("win"):
        lock = job["job_handle_lock"]
        with lock:
            handle = job.get("job_handle")
            if not handle:
                return
            try:
                _KERNEL32.TerminateJobObject(handle, 1)
                deadline = time.monotonic() + 10
                while _windows_job_active(handle) and time.monotonic() < deadline:
                    time.sleep(0.02)
            finally:
                _KERNEL32.CloseHandle(handle)
                job["job_handle"] = None
        try:
            proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return
    pgid = job.get("process_group")
    if pgid is None:
        return
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
    try:
        proc.wait(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _close_unassigned_windows_job(handle):
    if handle:
        try:
            _KERNEL32.CloseHandle(handle)
        except Exception:
            pass


def _deadline_thread(job):
    while True:
        remaining = max(0, job["deadline_at"] - time.time())
        if job["finished"].wait(min(0.5, remaining)):
            return
        state = "timed_out"
        stage = "Stopped after reaching the job deadline."
        if remaining > 0:
            try:
                _count, run_bytes = _app_runs_usage()
                free_bytes = shutil.disk_usage(Path(SCRATCH_DIR).parent).free
            except OSError:
                state = "error"
                stage = "Stopped because the storage safety check failed."
            else:
                if run_bytes > MAX_APP_RUN_BYTES:
                    state = "error"
                    stage = "Stopped because app-owned run storage exceeded 2 GB."
                elif free_bytes < 512 * 1024**2:
                    state = "error"
                    stage = "Stopped because free disk space fell below 512 MB."
                else:
                    continue
        with job["lock"]:
            if job["state"] != "running":
                return
            job["state"] = state
            job["stage"] = stage
            job.setdefault("log", deque(maxlen=MAX_LOG_LINES)).append(stage)
        _stop_process_tree(job)
        return


def _trim_jobs_locked():
    finished = sorted(
        ((j.get("finished_at", 0), job_id) for job_id, j in JOBS.items()
         if job_id != ACTIVE_JOB_ID and j.get("state") != "running")
    )
    while len(JOBS) >= MAX_JOBS and finished:
        JOBS.pop(finished.pop(0)[1], None)


def _start_job(step_key, args, companies=1, source_job=None):
    global ACTIVE_JOB_ID
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        if ACTIVE_JOB_ID is not None:
            return None
        _trim_jobs_locked()
        ACTIVE_JOB_ID = job_id
    proc = None
    windows_job = None
    run_root = Path(APP_RUNS_DIR, job_id)
    try:
        run_count, run_bytes = _app_runs_usage()
        if run_count >= MAX_APP_RUNS or run_bytes >= MAX_APP_RUN_BYTES:
            raise RuntimeError("local run storage is full; archive or remove old output/app-runs folders")
        if shutil.disk_usage(Path(SCRATCH_DIR).parent).free < 512 * 1024**2:
            raise RuntimeError("less than 512 MB free; clear space before starting a run")
        _prepare_job_root(step_key, run_root, source_job)
        artifact_before = _job_files(run_root)
        script_name = SCRIPTS[step_key]
        script_path = os.path.join(PIPELINE_DIR, script_name)

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        popen_kwargs = {}
        if sys.platform.startswith("win"):
            windows_job = _new_windows_job()
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | _CREATE_SUSPENDED
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path] + args,
            cwd=run_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, env=env, **popen_kwargs,
        )
        if windows_job:
            _assign_windows_job(windows_job, proc.pid)
            psutil.Process(proc.pid).resume()
    except Exception:
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        _close_unassigned_windows_job(windows_job)
        _release_job(job_id)
        try:
            run_root.resolve().relative_to(Path(APP_RUNS_DIR).resolve())
            if run_root.is_dir() and not run_root.is_symlink():
                shutil.rmtree(run_root)
        except (OSError, ValueError):
            pass
        raise

    job = {
        "id": job_id,
        "proc": proc,
        "script": script_name,
        "step": step_key,
        "root": str(run_root),
        "artifact_before": artifact_before,
        "artifacts": {},
        "companies": max(1, int(companies)),
        "percent": 0,
        "stage": "Initializing...",
        "state": "running",
        "log": deque(maxlen=MAX_LOG_LINES),
        "lock": threading.Lock(),
        "started_at": time.time(),
        "deadline_at": time.time() + JOB_DEADLINE_SECONDS,
        "finished": threading.Event(),
        "job_handle": windows_job,
        "job_handle_lock": threading.Lock(),
        "process_group": None if sys.platform.startswith("win") else proc.pid,
    }
    try:
        with JOBS_LOCK:
            JOBS[job_id] = job
        threading.Thread(target=_reader_thread, args=(job,), daemon=True).start()
        threading.Thread(target=_deadline_thread, args=(job,), daemon=True).start()
    except Exception:
        with job["lock"]:
            job["state"] = "error"
        _stop_process_tree(job)
        _release_job(job_id)
        raise
    return job_id


def _get_ollama_models():
    """Mirror _original_app.py's _get_ollama_models(): list installed local
    Ollama models + pick a sensible default. Read-only, safe to call per-request."""
    try:
        if PIPELINE_DIR not in sys.path:
            sys.path.insert(0, PIPELINE_DIR)
        import hardware_profiler as hw
        models = hw.get_installed_ollama_models()
        best = hw.get_best_ollama_model(models)
        return models, best
    except Exception:
        return [], None


def _integer(body, key, default, low, high):
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer from {low} to {high}")
    return value


def _boolean(body, key, default=False):
    value = body.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _choice(body, key, allowed):
    value = body.get(key)
    if value is not None and (not isinstance(value, str) or value not in allowed):
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def _model(body):
    value = body.get("ollama_model")
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,100}", value):
        raise ValueError("ollama_model has an invalid format")
    return value


def _validate_request(step, body):
    unknown = set(body) - ALLOWED_KEYS[step]
    if unknown:
        raise ValueError("unknown parameter(s): " + ", ".join(sorted(unknown)))
    companies = 1 if step == "demo" else _integer(body, "companies", 3, 1, 3)
    args = [] if step == "demo" else ["--companies", str(companies)]
    if step == "generate":
        products = _integer(body, "products", 20, 1, 20)
        if companies * products > 60:
            raise ValueError("companies multiplied by products must not exceed 60")
        args += ["--products", str(products)]
        model = _model(body)
        if model:
            args += ["--ollama-model", model]
        error_types = body.get("error_types")
        if error_types is not None:
            if not isinstance(error_types, str):
                raise ValueError("error_types must be a string")
            names = {n.strip() for n in error_types.split(",") if n.strip()}
            if error_types not in {"all", "none", "random"} and (not names or not names <= ERROR_TYPES):
                raise ValueError("error_types contains an unknown value")
            args += ["--error-types", error_types]
        if "error_rate" in body:
            rate = body["error_rate"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("error_rate must be a finite number from 0 to 1")
            args += ["--error-rate", str(rate)]
        if "seed" in body:
            seed = _integer(body, "seed", 0, 0, 2**31 - 2)
            args += ["--seed", str(seed)]
    elif step == "verify":
        for key, allowed, flag in (
            ("mode", {"speed", "accuracy", "max_accuracy"}, "--mode"),
            ("scope", {"fast", "total"}, "--scope"),
            ("hardware", {"1", "2", "3"}, "--hardware"),
            ("shap_mode", {"researcher", "production"}, "--shap-mode"),
            ("engine", {"auto", "heuristic", "conditional", "full"}, "--engine"),
        ):
            value = _choice(body, key, allowed)
            if value:
                args += [flag, value]
        model = _model(body)
        if model:
            args += ["--ollama-model", model]
        if _boolean(body, "all"):
            args.append("--all")
    elif step == "accuracy" and _boolean(body, "all"):
        args.append("--all")
    return args, companies


def _same_origin():
    origin = request.headers.get("Origin")
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc == request.host


def _artifact_path(relpath, artifact_root=SCRATCH_DIR):
    if not relpath or "\\" in relpath or ":" in relpath:
        return None
    parts = Path(relpath).parts
    if any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
        return None
    if Path(relpath).suffix.lower() not in {".csv", ".json", ".png", ".jpg", ".jpeg"}:
        return None
    root = Path(artifact_root).resolve()
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() and resolved.name not in DATA_ASSETS else None


def _artifact_directory(parent, name, artifact_root=SCRATCH_DIR):
    try:
        root = Path(artifact_root).resolve()
        Path(parent, name).resolve(strict=True).relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.config["TRUSTED_HOSTS"] = ["127.0.0.1:5004", "localhost:5004", "127.0.0.1", "localhost"]

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": f"request body must not exceed {MAX_BODY_BYTES} bytes"}), 413

    @app.route("/api/scope3/<any(demo,generate,verify,accuracy,materiality):step>", methods=["POST", "OPTIONS"])
    def run_step(step):
        if request.method == "OPTIONS":
            return "", 204
        # A text/plain POST is a "simple request" and skips preflight entirely,
        # so requiring JSON is what actually stops a stray page starting a run.
        if not request.is_json:
            return jsonify({"error": "JSON body required"}), 415
        if not _same_origin():
            return jsonify({"error": "cross-origin request rejected"}), 403
        try:
            body = request.get_json(silent=False)
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            args, companies = _validate_request(step, body)
            with JOBS_LOCK:
                active_job_id = ACTIVE_JOB_ID
            if active_job_id:
                return jsonify({"error": "another pipeline job is active", "active_job_id": active_job_id}), 409
            source_job = _source_job(step, body)
        except (ValueError, BadRequest) as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            job_id = _start_job(step, args, companies=companies, source_job=source_job)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except (OSError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 507
        if job_id is None:
            return jsonify({"error": "another pipeline job is active", "active_job_id": ACTIVE_JOB_ID}), 409
        return jsonify({"job_id": job_id}), 202

    @app.route("/api/scope3/ollama-models")
    def ollama_models():
        models, best = _get_ollama_models()
        return jsonify({"models": models, "best": best})

    @app.route("/api/scope3/engine-profile")
    def engine_profile():
        # ponytail: same sys.path guard as _get_ollama_models() above - PIPELINE_DIR
        # isn't on sys.path until something puts it there, and this route can be
        # hit before ollama-models ever is.
        if PIPELINE_DIR not in sys.path:
            sys.path.insert(0, PIPELINE_DIR)
        from hardware_profiler import recommend_engine
        return jsonify(recommend_engine())

    @app.route("/api/scope3/status/<job_id>")
    def status(job_id):
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job_id"}), 404
        with job["lock"]:
            return jsonify(
                {
                    "state": job["state"],
                    "percent": job["percent"],
                    "current_stage": job["stage"],
                    "log_tail": list(job["log"])[-50:],
                    "started_at": job["started_at"],
                    "deadline_at": job["deadline_at"],
                    "step": job["step"],
                }
            )

    @app.route("/api/scope3/active")
    def active():
        with JOBS_LOCK:
            job_id = ACTIVE_JOB_ID
        job = JOBS.get(job_id) if job_id else None
        return jsonify({"active_job_id": job_id, "step": job.get("step") if job else None})

    @app.route("/api/scope3/cancel/<job_id>", methods=["POST"])
    def cancel(job_id):
        if not _same_origin():
            return jsonify({"error": "cross-origin request rejected"}), 403
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job_id"}), 404
        with job["lock"]:
            if job["state"] != "running":
                return jsonify({"error": "job is not running", "state": job["state"]}), 409
            job["state"] = "cancelled"
            job["stage"] = "Cancellation requested."
        _stop_process_tree(job)
        return jsonify({"job_id": job_id, "state": "cancelled"})

    @app.route("/api/scope3/artifacts")
    def artifacts():
        job_id = request.args.get("job_id", "")
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "known job_id required"}), 400
        results = []
        for rel in sorted(job.get("artifacts", ())):
            if _artifact_path(rel, job["root"]) is None:
                continue
            results.append({
                "path": rel,
                "type": "image" if rel.lower().endswith((".png", ".jpg", ".jpeg")) else Path(rel).suffix.lower()[1:],
            })
        return jsonify({"artifacts": sorted(results, key=lambda r: r["path"])})

    @app.route("/api/scope3/artifact/<job_id>/<path:relpath>")
    def artifact(job_id, relpath):
        job = JOBS.get(job_id)
        if not job or relpath not in job.get("artifacts", {}):
            return jsonify({"error": "artifact not found"}), 404
        path = _artifact_path(relpath, job["root"])
        if path is None:
            return jsonify({"error": "artifact not found"}), 404
        expected_size, expected_sha256 = job["artifacts"][relpath]
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha256:
            return jsonify({"error": "artifact changed after job completion"}), 409
        response = send_from_directory(path.parent, path.name, as_attachment=path.suffix.lower() == ".csv")
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    # ── The built frontend, served from this same process and port ────────
    # Registered last so it cannot shadow anything above, and it declines
    # /api/ itself so a typo in an API path 404s as an API path instead of
    # silently returning the HTML shell.
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def frontend(path):
        if path.startswith("api/"):
            return jsonify({"error": "unknown API route"}), 404
        if not os.path.isdir(FRONTEND_DIR):
            return (
                "<h1>Scope 3 Auditor</h1><p>The API is running, but the page has not "
                "been built. Run <code>npm --prefix frontend install</code> then "
                "<code>npm --prefix frontend run build</code>.</p>",
                200,
            )
        # A built single-page app owns its own routing, so a path that is not a
        # real file falls back to the shell.
        full = safe_join(FRONTEND_DIR, path) if path else None
        if full and os.path.isfile(full):
            return send_from_directory(FRONTEND_DIR, path)
        # ...but only for route-shaped paths. A missing .js or .css must 404
        # rather than fall back, or the browser parses the HTML shell as
        # JavaScript and reports a syntax error instead of the missing file.
        if os.path.splitext(path)[1]:
            return jsonify({"error": "not found"}), 404
        return send_from_directory(FRONTEND_DIR, "index.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=False)
