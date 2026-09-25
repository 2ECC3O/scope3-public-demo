"""Build the public demonstration summary from real scorer output."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(project_dir, destination):
    generated = project_dir / "generated company"
    messy_path = generated / "COMP_001_messy.csv"
    pristine_path = generated / "COMP_001_pristine.csv"
    corrected_path = project_dir / "ai corrected" / "COMP_001_AI_corrected.csv"
    report_path = project_dir / "accuracy test" / "accuracy_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = project_dir.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with messy_path.open(encoding="utf-8-sig", newline="") as handle:
        messy = list(csv.DictReader(handle))
    with pristine_path.open(encoding="utf-8-sig", newline="") as handle:
        pristine = list(csv.DictReader(handle))
    with corrected_path.open(encoding="utf-8-sig", newline="") as handle:
        corrected = list(csv.DictReader(handle))

    expected_rows = int(manifest["settings"]["rows"])
    if not (len(messy) == len(pristine) == len(corrected) == expected_rows):
        raise ValueError(
            "demonstration inputs and corrected output must each contain "
            f"{expected_rows} rows"
        )
    identity = ("company_id", "product_id", "reporting_month")
    for index, rows in enumerate(zip(messy, pristine, corrected), start=2):
        if any(tuple(row.get(key) for key in identity) != tuple(rows[0].get(key) for key in identity) for row in rows[1:]):
            raise ValueError(f"row identity mismatch at CSV row {index}")

    def is_true_error(before, truth, field):
        try:
            p = float(truth[field])
            m = float(before[field])
        except (KeyError, TypeError, ValueError):
            return False
        return not bool(np.isclose(p, m, rtol=1e-5, equal_nan=True))

    examples = []
    for row_index, (before, truth, after) in enumerate(zip(messy, pristine, corrected)):
        for key, value in after.items():
            if not key.endswith("_anomaly"):
                continue
            field = key.removesuffix("_anomaly")
            flagged = after.get(f"{field}_error_type") == "Human Review Required" or after.get(f"{field}_review_flag") == "1"
            if not flagged or field not in report["columns"]:
                continue
            corrected_value = after.get(f"{field}_corrected", "")
            examples.append({
                "company": before["company_id"],
                "product": before["product_id"],
                "month": before["reporting_month"],
                "field": field,
                "reported_value": before.get(field, ""),
                "suggested_correction": corrected_value or None,
                "verdict": "true error" if is_true_error(before, truth, field) else "clean cell flagged",
                "reason": after.get(f"{field}_error_type", "Review required"),
                "row": row_index + 2,
            })

    goal = report["goal_metrics"]
    payload = {
        "label": "Bundled tiny synthetic demonstration",
        "provenance": {
            "code_data_sha256": manifest["code_data_sha256"],
            "code_data_manifest_sha256": hashlib.sha256(json.dumps(
                manifest["code_data_sha256"], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "runtime": manifest["runtime"],
            "settings": manifest["settings"],
            "source": ("fresh invented-data run/generated company"
                       if (Path(__file__).resolve().parents[1] / "PUBLIC-DEMO-MODE").is_file()
                       else "data/demo_project/generated company"),
            "source_sha256": _sha256(messy_path),
            "pristine_sha256": _sha256(pristine_path),
            "corrected_sha256": _sha256(corrected_path),
            "scorer_report_canonical_sha256": hashlib.sha256(json.dumps(
                {key: value for key, value in report.items() if key not in {"generated_at", "project"}},
                sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
            "scorer_hash_definition": "SHA-256 of sorted compact report JSON excluding generated_at and project",
        },
        "counts": {key: goal[key] for key in (
            "n_cells", "n_columns_evaluated", "true_errors", "tp", "fp", "tn", "fn",
            "flagged_total", "flagged_tp", "eligible_flagged_tp", "should_flag", "degraded",
        )},
        "rates_percent": {key: goal[key] for key in (
            "detection_accuracy", "detection_precision", "detection_recall", "tp_flag_rate",
            "fp_flag_rate", "silent_fail_rate", "degraded_rate", "verified_pct", "unverified_pct",
        )},
        "flag_metric_version": goal["flag_metric_version"],
        "row_reviews": {
            "rows": report["aggregate"]["row_identity_review_rows"],
            "events": report["aggregate"]["row_identity_review_events"],
            "total_rows": report["aggregate"]["row_identity_review_total_rows"],
            "explanation": "Checks found inconsistent values but could not identify a single responsible field. These counts are separate from cell-level scores.",
        },
        "examples": examples[:8],
        "example_definition": "First eight scorer-evaluated flagged numeric cells in row and column order",
        "limitations": "This fixed-schema synthetic run demonstrates execution and scoring. It is not customer-spreadsheet ingestion or independent real-world validation.",
    }
    payload["counts"]["clean_flags"] = goal["flagged_total"] - goal["flagged_tp"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_demo_results.py PROJECT_DIR DESTINATION")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
