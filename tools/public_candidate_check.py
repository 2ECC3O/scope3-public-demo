"""Check the extracted public demo's invented inputs and local build integrity."""

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    assert (ROOT / "PUBLIC-DEMO-MODE").is_file()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = ROOT / name
        assert path.is_file() and path.resolve().is_relative_to(ROOT.resolve()), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    with sqlite3.connect(ROOT / "data" / "factors.db") as conn:
        rows = conn.execute("SELECT name, value, source_id FROM factor").fetchall()
        assert rows == [("Invented steel factor", "2.0", "invented")]
        assert conn.execute("SELECT COUNT(*) FROM waste_link").fetchone()[0] == 0
    assert (ROOT / "waste_codes.csv").read_bytes() == (ROOT / "data" / "waste_codes.csv").read_bytes()
    assert (ROOT / "frontend" / ".env.production").read_text().strip() == "VITE_PUBLIC_DEMO=1"
    assert json.loads((ROOT / "frontend" / "src" / "demo-results.json").read_text())["counts"]["n_cells"] == 0
    assert b"const REPO = '';" in (ROOT / "frontend" / "src" / "App.jsx").read_bytes()
    js = b"".join(path.read_bytes() for path in (ROOT / "frontend" / "dist" / "assets").glob("*.js"))
    assert b"36792" not in js and b"Bundled tiny synthetic demonstration" not in js
    assert b"github.com/2ECC3O" not in js
    print(f"PASS: {len(manifest)} manifest entries, invented factors, public-only frontend")


if __name__ == "__main__":
    main()
