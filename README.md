# Scope 3 Auditor — public demo

A local, interactive demonstration of a Scope 3 research prototype. It creates **invented supply-chain records**, checks them for anomalies, and shows the resulting diagnostics. You can explore the interface without installing Node.js or connecting to a hosted service.

**Start here:** install the demo, open <http://127.0.0.1:5004>, and click **Bundled check**. That check creates a fixed-seed, 36-row example and runs the real verifier and scorer. The first check may take a few minutes while the local engine starts.

> **Not for real emissions reporting.** The teaching factor (2.0 kg CO₂e/kg), fallback factors, waste codes, and generated records are illustrative, not validated environmental estimates. The public dataset is deliberately smaller than the private research set; missing waste links disable their checks. A successful demo is not evidence of accuracy on customer data.

## Before you start

- An internet connection is needed for the first install. Later starts use the local environment.
- Install **uv 0.12.13**; the included installers require that exact version and download Python 3.12 for you. You do not need to install Python, Node.js, or Ollama separately for the bundled check.
- Get the code with Git, or use GitHub's **Code → Download ZIP** and extract it. Run all commands from the folder containing `start.py`.
- The server listens only on `127.0.0.1:5004` on your own computer. Press **Ctrl+C** in the terminal to stop it.

### Windows (PowerShell) — tested

1. Open **PowerShell**. Install the required uv version using the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/):

   ```powershell
   powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/0.12.13/install.ps1 | iex"
   ```

   Open a new PowerShell window if `uv` is not recognised, then check:

   ```powershell
   uv --version
   ```

   It must print `uv 0.12.13`.

2. Download the repository and enter its folder:

   ```powershell
   git clone https://github.com/2ECC3O/scope3-public-demo.git
   cd scope3-public-demo
   ```

   If you downloaded a ZIP instead, extract it and use `cd` to enter the extracted folder.

3. Install dependencies and start the local demo:

   ```powershell
   $env:SCOPE3_SOURCE = (Get-Location).Path
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

   Keep this terminal open. The installer creates `.tools/` and `.venv/` inside the repository, then starts the server.

4. Open <http://127.0.0.1:5004> and click **Bundled check**. When it finishes, use the download links in that panel to inspect the output.

**Next time:** from the repository folder, run `.\.venv\Scripts\python.exe start.py`; installation is not repeated.

### macOS (Terminal) — not yet tested on a Mac

The macOS installer has passed a shell syntax check, but a complete macOS installation and interactive run have **not** been verified. The steps below are the intended path; if the locked Python dependencies fail to install, use the browse-only option below. Please report the exact error instead of treating an unverified run as a pass.

1. Open **Terminal**. Install the required uv version using the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/):

   ```sh
   curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
   ```

   Open a new Terminal window if `uv` is not found, then check that `uv --version` prints `uv 0.12.13`.

2. Download the repository and enter its folder:

   ```sh
   git clone https://github.com/2ECC3O/scope3-public-demo.git
   cd scope3-public-demo
   ```

   If you downloaded a ZIP instead, extract it and `cd` into the extracted folder.

3. Install dependencies and start the local demo:

   ```sh
   SCOPE3_SOURCE="$PWD" sh ./install.sh
   ```

   Keep Terminal open. If installation succeeds, open <http://127.0.0.1:5004> and click **Bundled check**.

4. If step 3 fails, you can still explore the prebuilt interface without the Python packages:

   ```sh
   SCOPE3_TIER=browse SCOPE3_SOURCE="$PWD" sh ./install.sh
   ```

   This is **browse-only**: the page opens at the same address, but Generate, Verify, and Bundled check need the full server and will not run.

**Next time, after a successful full install:** from the repository folder, run `.venv/bin/python start.py`.

## What to try after Bundled check

Use **Generate → Verify → Accuracy → Materiality** in that order. Generate makes fresh synthetic records; Verify checks them; Accuracy creates comparison charts; Materiality shows an illustrative score. Ollama is optional for waste-code assistance during Generate, and is **not** used by Bundled check or Verify.

Generated files stay on your computer. The demo is for exploring the workflow, not for calculating a real organisation's inventory. See [data and source limits](DATA-LICENSES.md) and [third-party notices](THIRD-PARTY-NOTICES/).

For a separate arithmetic exercise on published federal Scope 3 tables, see the [federal evaluation guide](docs/plans/2026-09-20-federal-scope3-evaluation.md). It checks reported totals and percentages; it does not run the full web auditor on those agency files.
