"""
Hardware Profiler Utility
─────────────────────────
Lightweight system probe that exports machine capabilities for
downstream compute-intensity decisions.

Exports
-------
TOTAL_RAM_GB      : float   – Total physical RAM in GiB.
AVAILABLE_RAM_GB  : float   – Currently free RAM in GiB.
CPU_CORES         : int     – Physical CPU core count.
GPU_PRESENT       : bool    – True if CUDA or Metal acceleration is usable.
MACHINE_TIER      : str     – 'high' | 'mid' | 'low'
TREE_METHOD       : str     – XGBoost tree_method to use.
DEVICE            : str     – XGBoost device string ('cuda', 'metal', or 'cpu').
"""

import psutil  # type: ignore
import numpy as np  # type: ignore
import xgboost as xgb  # type: ignore

# ── RAM & CPU ───────────────────────────────────────────────────────────
_mem = psutil.virtual_memory()
TOTAL_RAM_GB: float = round(_mem.total / (1024 ** 3), 2)
AVAILABLE_RAM_GB: float = round(_mem.available / (1024 ** 3), 2)
CPU_CORES: int = psutil.cpu_count(logical=False) or 1

# ── GPU probe (CUDA → Metal → CPU fallback) ────────────────────────────
GPU_PRESENT: bool = False
TREE_METHOD: str = 'hist'
DEVICE: str = 'cpu'

for _candidate_device in ('cuda', 'gpu'):
    try:
        _probe = xgb.XGBClassifier(
            n_estimators=2, tree_method='hist',
            device=_candidate_device, verbosity=0,
        )
        _X = np.random.rand(20, 2)
        _y = (_X[:, 0] > 0.5).astype(int)
        _probe.fit(_X, _y)
        GPU_PRESENT = True
        DEVICE = _candidate_device
        break
    except Exception:
        continue

# ── Machine tier ────────────────────────────────────────────────────────
if GPU_PRESENT and TOTAL_RAM_GB >= 16:
    MACHINE_TIER: str = 'high'
elif TOTAL_RAM_GB >= 8:
    MACHINE_TIER = 'mid'
else:
    MACHINE_TIER = 'low'

# ── Live Telemetry Check ────────────────────────────────────────────────
def get_live_metrics() -> dict:
    """Get dynamic, real-time memory, CPU, and GPU VRAM utilization."""
    metrics = {
        "available_ram": 0.0,
        "used_ram": 0.0,
        "ram_pct": 0.0,
        "cpu_pct": 0.0,
    }
    
    # 1. System RAM Telemetry
    try:
        mem = psutil.virtual_memory()
        metrics["available_ram"] = round(mem.available / (1024 ** 3), 2)
        metrics["used_ram"] = round((mem.total - mem.available) / (1024 ** 3), 2)
        metrics["ram_pct"] = mem.percent
    except Exception:
        pass
    
    # 2. CPU Telemetry (100ms sampling to avoid standard first-call 0.0% bug)
    try:
        metrics["cpu_pct"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        pass
    
    # 3. GPU VRAM Telemetry (Isolate entirely for CUDA setups)
    if GPU_PRESENT and DEVICE == 'cuda':
        try:
            import subprocess
            import sys
            kwargs = {}
            if sys.platform.startswith('win'):
                kwargs['creationflags'] = 0x08000000
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2,
                **kwargs
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                total = int(parts[0].strip())
                used = int(parts[1].strip())
                free = int(parts[2].strip())
                metrics.update({
                    "vram_total_mib": total,
                    "vram_used_mib": used,
                    "vram_free_mib": free,
                    "vram_pct": round((used / total) * 100, 1) if total > 0 else 0.0
                })
        except Exception:
            pass
            
    return metrics

# ── Summary on import ───────────────────────────────────────────────────
def print_profile() -> None:
    """Print a concise hardware summary to stdout."""
    print("+--- Hardware Profile -------------------------------+")
    print(f"|  Total RAM      : {TOTAL_RAM_GB:>8.2f} GiB")
    print(f"|  Available RAM  : {AVAILABLE_RAM_GB:>8.2f} GiB")
    print(f"|  CPU Cores      : {CPU_CORES:>8d}")
    gpu_label = 'METAL' if DEVICE == 'gpu' else DEVICE.upper()
    print(f"|  GPU Accel      : {gpu_label if GPU_PRESENT else 'None':>8s}")
    print(f"|  Machine Tier   : {MACHINE_TIER.upper():>8s}")
    print("+----------------------------------------------------+")


# ── Ollama Model Auto-Detection ──────────────────────────────────────────
def get_installed_ollama_models() -> list[str]:
    """Offline scanner for installed local Ollama models, falling back to API check.
    Cross-platform: Ollama stores models under ~/.ollama/models on macOS and Linux
    the same way it does on Windows (just via HOME instead of USERPROFILE), so the
    directory scan below already covers Mac - this just makes each candidate root
    resilient to being scanned independently, and adds expanduser("~") as one more
    fallback in case neither HOME nor USERPROFILE is set in the environment."""
    import os
    import sys

    candidate_roots = []

    # 1. Check OLLAMA_MODELS env var
    env_models = os.environ.get("OLLAMA_MODELS")
    if env_models:
        candidate_roots.append(env_models)

    # 2. Check default path on Windows and macOS/Linux
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidate_roots.append(os.path.join(user_profile, ".ollama", "models"))

    home_dir = os.environ.get("HOME")
    if home_dir:
        candidate_roots.append(os.path.join(home_dir, ".ollama", "models"))

    # Cross-platform fallback if neither HOME nor USERPROFILE was set
    try:
        expanded_home = os.path.expanduser("~")
        if expanded_home and expanded_home != "~":
            candidate_roots.append(os.path.join(expanded_home, ".ollama", "models"))
    except Exception:
        pass

    # Dedupe while preserving order (macOS/Linux commonly resolve to the same path twice)
    seen_roots = set()
    candidate_roots = [r for r in candidate_roots if not (r in seen_roots or seen_roots.add(r))]

    found_models = []
    for root in candidate_roots:
        try:
            manifest_dir = os.path.join(root, "manifests", "registry.ollama.ai", "library")
            if os.path.exists(manifest_dir) and os.path.isdir(manifest_dir):
                for model_name in os.listdir(manifest_dir):
                    model_path = os.path.join(manifest_dir, model_name)
                    if os.path.isdir(model_path):
                        for tag_name in os.listdir(model_path):
                            tag_path = os.path.join(model_path, tag_name)
                            if os.path.isfile(tag_path):
                                found_models.append(f"{model_name}:{tag_name}")
        except OSError:
            # One unreadable/permission-denied root shouldn't abort the whole scan
            continue

    # Fallback to local API query (non-blocking / fast timeout) if server is already running
    if not found_models:
        import urllib.request
        import json
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=1) as r:
                res = json.loads(r.read().decode('utf-8'))
                for m in res.get('models', []):
                    found_models.append(m['name'])
        except Exception:
            pass
            
    # Deduplicate and sort
    return sorted(list(set(found_models)))


def get_best_ollama_model(models: list[str]) -> str | None:
    """Choose the best model based on name ranking heuristic."""
    if not models:
        return None
    # Prioritized keywords for best-matching models
    for keyword in ['gemma4', 'gemma', 'llama3', 'llama', 'mistral', 'phi3']:
        for m in models:
            if keyword in m.lower():
                return m
    return models[0]


def recommend_engine() -> dict:
    """Recommend a verification engine based on MACHINE_TIER and PySR availability.
    Mirrors the resolution logic step_2_verify_data.py falls back to for --engine auto,
    but this is the authoritative source once wired in (see E1b below)."""
    import importlib.util
    # ponytail: find_spec, not `import pysr` - a real import triggers PySR's Julia
    # startup just to answer "is it installed", which is slow and has side effects.
    pysr_available = importlib.util.find_spec('pysr') is not None

    available = ['heuristic']
    if MACHINE_TIER in ('mid', 'high'):
        available.append('conditional')
    if pysr_available:
        available.append('full')

    if MACHINE_TIER == 'low':
        recommended = 'heuristic'
        reason = 'low-tier hardware — heuristic engine recommended'
    elif MACHINE_TIER == 'mid':
        recommended = 'conditional'
        reason = (
            'mid-tier hardware — conditional engine recommended, PySR not installed for full engine'
            if not pysr_available else
            'mid-tier hardware — conditional engine recommended (full engine needs high-tier hardware)'
        )
    else:  # high
        if pysr_available:
            recommended = 'full'
            reason = 'high-tier hardware with PySR installed'
        else:
            recommended = 'conditional'
            reason = 'high-tier hardware — conditional engine recommended, PySR not installed for full engine'

    return {'recommended': recommended, 'available': available, 'reason': reason}


if __name__ == '__main__':
    print_profile()

