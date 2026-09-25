#!/usr/bin/env python3
"""Install once, then run the local Scope 3 Auditor without network access."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
LOCK = ROOT / "requirements.txt"
MARKER = VENV / ".scope3-installed.json"
PORT = 5004
PYTHON_MINOR = (3, 12)
CORE_IMPORTS = ("flask", "matplotlib", "numpy", "openpyxl", "pandas", "PIL", "psutil", "scipy", "seaborn", "shap", "sklearn", "tqdm", "xgboost")


def venv_python(venv_dir: Path, platform: str = sys.platform) -> Path:
    return venv_dir / ("Scripts/python.exe" if platform.startswith("win") else "bin/python")


def lock_hash() -> str:
    return hashlib.sha256(LOCK.read_bytes()).hexdigest()


def uv_executable() -> str | None:
    configured = os.environ.get("SCOPE3_UV")
    local = ROOT / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
    return configured or shutil.which("uv") or (str(local) if local.exists() else None)


def environment_problem() -> str | None:
    py = venv_python(VENV)
    if not VENV.exists():
        return "The project environment is not installed."
    if not py.exists():
        return "The .venv was created on another operating system or is incomplete."
    try:
        marker = json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "The .venv was not created by this project's installer."
    if not isinstance(marker, dict):
        return "The .venv was not created by this project's installer."
    if marker.get("python") != "3.12" or marker.get("platform") != sys.platform:
        return "The .venv belongs to a different Python version or operating system."
    if interpreter_identity(py) != (sys.platform, "3.12"):
        return "The .venv interpreter is missing, damaged, or incompatible with this computer."
    if marker.get("lock_sha256") != lock_hash():
        return "The locked dependencies changed since this .venv was installed."
    return None


def print_remedy(problem: str) -> None:
    remedy = ("Run the installer again to apply the reviewed lockfile."
              if "locked dependencies changed" in problem
              else "Rename or remove .venv yourself, then run the installer again.")
    print(f"{problem}\n\n{remedy}", file=sys.stderr)


def interpreter_identity(py: Path) -> tuple[str, str] | None:
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import json,sys;print(json.dumps([sys.platform,sys.version_info[:2]]))"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        platform, version = json.loads(out)
        return platform, ".".join(map(str, version))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def install() -> int:
    if sys.version_info[:2] != PYTHON_MINOR:
        print("Installation requires Python 3.12.", file=sys.stderr)
        return 2
    if VENV.exists():
        problem = environment_problem()
        if problem and "locked dependencies changed" not in problem:
            print_remedy(problem)
            return 2
    uv = uv_executable()
    if not uv:
        print("uv is missing. Run install.ps1 on Windows or install.sh on macOS/Linux.", file=sys.stderr)
        return 2
    if not VENV.exists():
        rc = subprocess.call([uv, "venv", str(VENV), "--python", sys.executable])
        if rc:
            return rc
    py = venv_python(VENV)
    rc = subprocess.call([uv, "pip", "sync", str(LOCK), "--python", str(py), "--require-hashes"])
    if rc:
        return rc
    probe = ";".join(f"import {name}" for name in CORE_IMPORTS)
    rc = subprocess.call([str(py), "-c", probe])
    if rc:
        return rc
    MARKER.write_text(json.dumps({"lock_sha256": lock_hash(), "platform": sys.platform, "python": "3.12"}, indent=2) + "\n", encoding="utf-8")
    print("Installation complete. Normal starts will not install or use the network.")
    return 0


def serve_browse() -> int:
    import functools
    import http.server
    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").exists():
        print("frontend/dist is missing. Build the frontend first.", file=sys.stderr)
        return 1
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist))
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"Scope 3 Auditor: http://127.0.0.1:{PORT} (browse only)\nPress Ctrl+C to stop.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def run_server() -> int:
    problem = environment_problem()
    if problem:
        print_remedy(problem)
        return 2
    py = venv_python(VENV)
    if Path(sys.executable).resolve() != py.resolve():
        return subprocess.call([str(py), str(Path(__file__).resolve()), "--run"])
    for name in CORE_IMPORTS:
        try:
            __import__(name)
        except ImportError:
            print_remedy(f"The installed environment cannot import {name}.")
            return 2
    sys.path.insert(0, str(ROOT))
    from scope3_server import app
    print(f"Scope 3 Auditor: http://127.0.0.1:{PORT}\nPress Ctrl+C to stop.", flush=True)
    app.run(host="127.0.0.1", port=PORT, debug=False)
    return 0


def main() -> int:
    if "--browse" in sys.argv:
        return serve_browse()
    if "--install" in sys.argv:
        return install()
    return run_server()


if __name__ == "__main__":
    sys.exit(main())
