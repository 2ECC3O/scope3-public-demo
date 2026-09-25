# Public runnable demonstration

This archive runs the research prototype with invented teaching inputs. The
material factor (2.0 kg CO2e/kg) and waste codes are not environmental
estimates. Generate, Verify, Accuracy and Materiality operate on synthetic
records. Bundled check generates a fresh fixed-seed 36-row input, then runs
the actual verifier and scorer. Private-run results are excluded.

Clone this repository or extract the archive. Install uv 0.12.13 from
https://docs.astral.sh/uv/getting-started/installation/. On Windows, from
the extracted folder set $env:SCOPE3_SOURCE=(Get-Location).Path, then run
powershell -ExecutionPolicy Bypass -File .\install.ps1. On macOS/Linux,
use SCOPE3_SOURCE="$PWD" ./install.sh. Start with
.venv/Scripts/python.exe start.py on Windows, or .venv/bin/python start.py
on macOS/Linux. Open http://127.0.0.1:5004 on the same computer.

The public inputs cover fewer materials and waste processes than the private
research set. Missing factors use deterministic invented fallback values;
missing waste links disable their checks. Do not use generated emissions as
real inventory estimates. See DATA-LICENSES.md and THIRD-PARTY-NOTICES/.

For an optional separate arithmetic exercise on federal Scope 3 tables, see
docs/plans/2026-09-20-federal-scope3-evaluation.md. Fetch the four hash-pinned
source files with .venv/Scripts/python.exe tools/fetch_federal_scope3.py,
then run .venv/Scripts/python.exe tools/review_inventory_tables.py
tools/federal_scope3_mapping.json --year 2023 (use the bin/python path on
macOS/Linux). This checks reported totals and percentages only; it does not
run the full web auditor on agency data. Raw files are fetched separately.
