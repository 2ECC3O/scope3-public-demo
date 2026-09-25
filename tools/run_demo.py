"""Run the fixed bundled demonstration through the real verifier and scorer."""

import subprocess
import sys
import hashlib
import json
import platform
from pathlib import Path

from build_demo_results import build


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path.cwd()
PROJECT = RUN_ROOT / "project 01"


def runtime_manifest():
    public_demo = (ROOT / "PUBLIC-DEMO-MODE").is_file()
    paths = sorted((ROOT / "pipeline").glob("*.py"))
    paths += ([ROOT / "PUBLIC-DEMO-MODE", ROOT / "waste_codes.csv",
               ROOT / "data" / "waste_codes.csv", ROOT / "data" / "factors.db",
               ROOT / "requirements.txt"] if public_demo else [
               ROOT / "pipeline" / "emission_factors_tgo.csv",
               ROOT / "pipeline" / "waste_stoichiometry.csv",
               ROOT / "data" / "waste_codes.csv",
               ROOT / "data" / "regional_emission_factors.xlsx",
               ROOT / "data" / "factors.db", ROOT / "requirements.txt"])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing demonstration inputs: {missing}")
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main():
    common = [sys.executable, "-u"]
    public_demo = (ROOT / "PUBLIC-DEMO-MODE").is_file()
    if public_demo:
        # ponytail: generate fresh invented inputs in the same fixed project path;
        # the private frozen demonstration and its measurements stay untouched.
        subprocess.run(common + [str(ROOT / "pipeline" / "step_1_generate_data.py"),
            "--companies", "1", "--products", "3", "--seed", "20260913",
            "--error-rate", "0.15", "--error-types", "random",
            "--ollama-model", "placeholder", "--output-dir", str(PROJECT)],
            cwd=RUN_ROOT, check=True)
    verify = [str(ROOT / "pipeline" / "step_2_verify_data.py"),
        "--companies", "1", "--mode", "speed", "--scope", "fast", "--hardware", "1",
        "--shap-mode", "production", "--ollama-model", "placeholder", "--engine", "heuristic"]
    score = [str(ROOT / "pipeline" / "step_3_accuracy_test.py"), "--companies", "1"]
    code_data_sha256 = runtime_manifest()
    (RUN_ROOT / "run_manifest.json").write_text(json.dumps({
        "settings": {"companies": 1, "products": 3, "rows": 36, "seed": 20260913,
                     "verify": ["speed", "fast", "heuristic", "production"], "ollama": False},
        "code_data_sha256": code_data_sha256,
        "runtime": {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine()},
    }, indent=2) + "\n", encoding="utf-8")
    subprocess.run(common + verify, cwd=RUN_ROOT, check=True)
    subprocess.run(common + score, cwd=RUN_ROOT, check=True)
    if runtime_manifest() != code_data_sha256:
        raise RuntimeError("runtime code or data changed during the demonstration")
    build(PROJECT, RUN_ROOT / "demo-results.json")
    print("Demonstration complete: demo-results.json")


if __name__ == "__main__":
    main()
