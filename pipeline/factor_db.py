"""The SQLite store for the two reference datasets, replacing the standalone CSVs.

WHY THIS MODULE EXISTS
-----------------------
`emission_factors_tgo.csv` and `waste_stoichiometry.csv` are build artefacts: research
notes in docs/research/raw/*.md go through tools/build_ef_csv.py and tools/build_waste_kb.py
to become those two files, which pipeline/emission_factors.py and pipeline/waste_kb.py then
read. This module gives both a single database to write into and read from instead, without
touching either the builders or the readers yet (later waves rewire them).

THE SAFETY PROPERTY THIS FILE MUST NOT BREAK
----------------------------------------------
`emissions = activity x EF` holds exactly in this system, and step_1 and step_2 both import
the same factor tables. If a single number changes shape during this migration - a value
rounded, an empty string turned into NULL, whitespace stripped - an error detector fires on
every row of every company, including clean data. So every column here is TEXT, stored
byte-identical to the CSV cell, and the reader functions below must be indistinguishable
from `csv.DictReader` on the original file: same keys, same string values, same row order.
All parsing (including the em-dash-means-"no value" convention) stays in the two existing
reader modules, not here.

ponytail: three CREATE TABLEs, four writers, two readers, one describe. No ORM, no
migrations framework - the schema is small enough that "the CREATE TABLE IS the migration"
is the right amount of machinery.
"""
import csv
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DB_PATH = REPO_ROOT / "data" / "factors.db"          # NOT cwd-relative: the pipeline runs
                                                       # from output/, not from pipeline/

TGO_CSV = HERE / "emission_factors_tgo.csv"
KB_CSV = HERE / "waste_stoichiometry.csv"

# Column lists, taken verbatim from the CSV header rows, so the schema and the
# csv.DictReader-equivalence of the readers cannot drift apart.
FACTOR_COLUMNS = [
    "row_id", "name", "material_match", "is_commodity_grade", "category", "value", "unit",
    "vintage", "scope", "is_ipcc_default", "verdict", "status", "verify_report", "note",
    "tgo_source_doc", "source_url", "source_date", "confidence", "geography", "gwp_method",
]
WASTE_LINK_COLUMNS = [
    "row_id", "process", "input_material", "waste_code", "waste_code_desc", "ratio",
    "ratio_low", "ratio_high", "source_kind", "basis", "basis_qualifier", "obligation_class",
    "carries_discard_stream", "obligation_raw", "obligation", "corrected_ratio",
    "corrected_waste_code", "corrected_obligation", "verdict", "status", "verify_report",
    "note", "source", "source_url", "source_date", "confidence", "source_wave",
]
SOURCE_COLUMNS = ["source_id", "name", "url", "licence", "access_method", "retrieved_at"]

# Not sourced from a CSV - there is no research document to mirror. A row here records a
# judgement a human actually made (which competing row wins for a material), so it is
# written directly, one INSERT at a time, from wherever the judgement was made (today:
# tools/build_ef_csv.py's COMMODITY_GRADE table). Same TEXT-everywhere contract as the
# other two tables regardless.
#
# `region` keys the row alongside `material` (PRIMARY KEY is the pair). Sentinel '*'
# means "any region" - a wildcard row, not a fifth real region. The 9 pre-existing rows
# are region-agnostic TGO grade choices, so they are all wildcard rows. A row naming a
# real region (e.g. "Europe") only wins for that region; everyone else still falls
# through to the wildcard row for the same material, if one exists.
RESOLUTION_COLUMNS = ["material", "region", "row_id", "reason", "decided_on"]


def connect(path=None):
    """sqlite3.Connection with dict-like row access.

    path=None reads the CURRENT module-global DB_PATH, not a value frozen at def-time -
    a bare `path=DB_PATH` default would capture the Path object that existed when this
    module first loaded, so a later `factor_db.DB_PATH = other_path` (the swap-and-restore
    pattern this file's own demo() and emission_factors.py's collision demo both use)
    would be silently ignored by every no-arg connect() call. Latent bug found while
    building the collision-detection demo; fixed here since it would have quietly broken
    any test/tool that retargets DB_PATH to a real, existing alternate file.
    """
    if path is None:
        path = DB_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn):
    """CREATE TABLE IF NOT EXISTS x3. Safe to call on every startup."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS source (
            source_id TEXT PRIMARY KEY,
            {", ".join(c + " TEXT" for c in SOURCE_COLUMNS[1:])}
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS factor (
            {", ".join(c + " TEXT" for c in FACTOR_COLUMNS)},
            source_id TEXT REFERENCES source(source_id),
            PRIMARY KEY (source_id, row_id)
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS waste_link (
            pk INTEGER PRIMARY KEY AUTOINCREMENT,
            {", ".join(c + " TEXT" for c in WASTE_LINK_COLUMNS)},
            source_id TEXT REFERENCES source(source_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_waste_link_row_id ON waste_link(row_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS resolution (
            material TEXT,
            region TEXT,
            row_id TEXT,
            reason TEXT,
            decided_on TEXT,
            PRIMARY KEY (material, region)
        )
    """)
    # Migration for a DB built before a column was added: CREATE TABLE IF NOT EXISTS
    # above is a no-op on an existing table, so new FACTOR_COLUMNS need an explicit
    # ALTER. TEXT, no default - matches every other column's contract.
    existing = {r[1] for r in conn.execute("PRAGMA table_info(factor)")}
    for col in FACTOR_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE factor ADD COLUMN {col} TEXT")

    waste_existing = {r[1] for r in conn.execute("PRAGMA table_info(waste_link)")}
    for col in WASTE_LINK_COLUMNS:
        if col not in waste_existing:
            conn.execute(f"ALTER TABLE waste_link ADD COLUMN {col} TEXT")

    # `resolution` changed PRIMARY KEY (material -> material+region), which ALTER TABLE
    # cannot do - rebuild the table if an older DB still has the one-column key. Existing
    # rows get region='*' (wildcard): they were all region-agnostic TGO grade choices.
    res_cols = {r[1] for r in conn.execute("PRAGMA table_info(resolution)")}
    if "region" not in res_cols:
        conn.execute("ALTER TABLE resolution RENAME TO resolution_old")
        conn.execute("""
            CREATE TABLE resolution (
                material TEXT,
                region TEXT,
                row_id TEXT,
                reason TEXT,
                decided_on TEXT,
                PRIMARY KEY (material, region)
            )
        """)
        conn.execute(
            "INSERT INTO resolution (material, region, row_id, reason, decided_on) "
            "SELECT material, '*', row_id, reason, decided_on FROM resolution_old"
        )
        conn.execute("DROP TABLE resolution_old")
    conn.commit()


def replace_source(conn, source_id, name, url, licence, access_method, retrieved_at):
    """Upsert one source row (provenance for a dataset feeding the DB)."""
    conn.execute(
        "INSERT OR REPLACE INTO source (source_id, name, url, licence, access_method, "
        "retrieved_at) VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, name, url, licence, access_method, retrieved_at),
    )
    conn.commit()


def replace_factors(conn, source_id, rows):
    """Delete `source_id`'s rows from `factor`, insert `rows` (list[dict] keyed by
    FACTOR_COLUMNS) in order."""
    conn.execute("DELETE FROM factor WHERE source_id = ?", (source_id,))
    sql = (f"INSERT INTO factor ({', '.join(FACTOR_COLUMNS)}, source_id) "
           f"VALUES ({', '.join('?' for _ in FACTOR_COLUMNS)}, ?)")
    for row in rows:
        conn.execute(sql, [row.get(c, "") for c in FACTOR_COLUMNS] + [source_id])
    conn.commit()


def replace_waste_links(conn, source_id, rows):
    """Same contract as replace_factors, for `waste_link`. `pk` is autoincrement, never
    supplied by the caller."""
    conn.execute("DELETE FROM waste_link WHERE source_id = ?", (source_id,))
    sql = (f"INSERT INTO waste_link ({', '.join(WASTE_LINK_COLUMNS)}, source_id) "
           f"VALUES ({', '.join('?' for _ in WASTE_LINK_COLUMNS)}, ?)")
    for row in rows:
        conn.execute(sql, [row.get(c, "") for c in WASTE_LINK_COLUMNS] + [source_id])
    conn.commit()


def set_resolution(conn, material, row_id, reason, decided_on, region="*"):
    """Upsert the one resolution row for (`material`, `region`) - which competing row_id
    wins, and why. `region` defaults to '*' (wildcard, any region) - every existing
    caller passes only material/row_id/reason/decided_on and keeps writing wildcard rows.

    One row per (material, region) (PRIMARY KEY), not append-only: a later decision for
    the same pair replaces an earlier one rather than leaving two to disagree.
    """
    conn.execute(
        "INSERT OR REPLACE INTO resolution (material, region, row_id, reason, decided_on) "
        "VALUES (?, ?, ?, ?, ?)",
        (material, region, row_id, reason, decided_on),
    )
    conn.commit()


def resolution_rows(conn=None):
    """list[dict] keyed by RESOLUTION_COLUMNS. [] if the database file does not exist -
    same missing-file contract as factor_rows()/waste_link_rows()."""
    close = False
    if conn is None:
        if not Path(DB_PATH).exists():
            return []
        conn = connect()
        close = True
    try:
        cur = conn.execute(f"SELECT {', '.join(RESOLUTION_COLUMNS)} FROM resolution")
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            conn.close()


def factor_rows(conn=None):
    """list[dict], indistinguishable from csv.DictReader on emission_factors_tgo.csv.

    [] if the database file does not exist, so a stripped checkout still runs on the
    deterministic-hash fallback in emission_factors.py - same contract that module's
    load_tgo_material_efs() already has for the missing CSV.
    """
    close = False
    if conn is None:
        if not Path(DB_PATH).exists():
            return []
        conn = connect()
        close = True
    try:
        cur = conn.execute(f"SELECT {', '.join(FACTOR_COLUMNS)} FROM factor ORDER BY rowid")
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            conn.close()


def waste_link_rows(conn=None):
    """list[dict], indistinguishable from csv.DictReader on waste_stoichiometry.csv.

    [] if the database file does not exist - same contract as waste_kb.load_rows().
    """
    close = False
    if conn is None:
        if not Path(DB_PATH).exists():
            return []
        conn = connect()
        close = True
    try:
        cur = conn.execute(
            f"SELECT {', '.join(WASTE_LINK_COLUMNS)} FROM waste_link ORDER BY rowid")
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close:
            conn.close()


def describe_db(conn=None):
    """One line per table: row count and the distinct source ids present."""
    close = False
    if conn is None:
        if not Path(DB_PATH).exists():
            return f"[DB] no database at {DB_PATH}"
        conn = connect()
        close = True
    try:
        lines = []
        for table in ("source", "factor", "waste_link"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            id_col = "source_id"
            sources = [r[0] for r in conn.execute(
                f"SELECT DISTINCT {id_col} FROM {table} ORDER BY {id_col}")]
            lines.append(f"[DB] {table}: {n} rows, sources={sources}")
        n = conn.execute("SELECT COUNT(*) FROM resolution").fetchone()[0]
        mats = [r[0] for r in conn.execute("SELECT material FROM resolution ORDER BY material")]
        lines.append(f"[DB] resolution: {n} rows, materials={mats}")
        return "\n".join(lines)
    finally:
        if close:
            conn.close()


def demo():
    """Self-check: round-trip every real CSV through the DB and compare to
    csv.DictReader byte-for-byte. That equality IS the point of this file - the whole
    reason it stores TEXT everywhere is so this assertion holds."""
    import tempfile

    with TGO_CSV.open(encoding="utf-8") as fh:
        tgo_original = list(csv.DictReader(fh))
    with KB_CSV.open(encoding="utf-8") as fh:
        kb_original = list(csv.DictReader(fh))
    assert tgo_original and kb_original, "real CSVs are empty - nothing to round-trip"

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_db = tmp_dir / "factors_demo.db"

    conn = connect(tmp_db)
    init_schema(conn)
    replace_source(conn, "tgo", "TGO CFP Emission Factors", TGO_CSV.as_uri(),
                    "unknown", "csv", "2026-08-21")
    replace_source(conn, "waste_kb", "Waste stoichiometry KB", KB_CSV.as_uri(),
                    "unknown", "csv", "2026-08-21")
    replace_factors(conn, "tgo", tgo_original)
    replace_waste_links(conn, "waste_kb", kb_original)

    tgo_roundtrip = factor_rows(conn)
    kb_roundtrip = waste_link_rows(conn)

    assert len(tgo_roundtrip) == len(tgo_original), (
        f"factor round-trip changed row count: {len(tgo_original)} -> {len(tgo_roundtrip)}")
    assert tgo_roundtrip == tgo_original, (
        "factor_rows() is not byte-identical to csv.DictReader on "
        "emission_factors_tgo.csv - the migration would move real error-detector "
        "thresholds. See the module docstring.")

    assert len(kb_roundtrip) == len(kb_original), (
        f"waste_link round-trip changed row count: {len(kb_original)} -> {len(kb_roundtrip)}")
    assert kb_roundtrip == kb_original, (
        "waste_link_rows() is not byte-identical to csv.DictReader on "
        "waste_stoichiometry.csv - the migration would move real error-detector "
        "thresholds. See the module docstring.")

    # resolution: not CSV-sourced, so just round-trip a row through set_resolution/resolution_rows.
    assert resolution_rows(conn) == [], "fresh schema must start with an empty resolution table"
    set_resolution(conn, "Steel", "w4-1-tgo-material-ef:49", "test reason", "2026-08-22")
    got = resolution_rows(conn)
    assert got == [{"material": "Steel", "region": "*", "row_id": "w4-1-tgo-material-ef:49",
                     "reason": "test reason", "decided_on": "2026-08-22"}], got
    set_resolution(conn, "Steel", "w4-1-tgo-material-ef:99", "changed my mind", "2026-08-22")
    got = resolution_rows(conn)
    assert len(got) == 1 and got[0]["row_id"] == "w4-1-tgo-material-ef:99", (
        "set_resolution must upsert on (material, region), not append: " + repr(got))
    # A region-specific row for the same material is a DIFFERENT key (material, region) -
    # it must coexist with the wildcard row, not overwrite it.
    set_resolution(conn, "Steel", "w4-1-tgo-material-ef:77", "Europe pick", "2026-08-22",
                    region="Europe")
    got = resolution_rows(conn)
    assert len(got) == 2 and {g["region"] for g in got} == {"*", "Europe"}, (
        "a region-specific row must coexist with the wildcard row for the same "
        "material: " + repr(got))

    conn.close()

    # Missing-file contract: both readers must return [] rather than crash. DB_PATH
    # itself has no factors.db yet in this checkout, which already exercises this - but
    # assert it explicitly against a path we know is absent, not by accident of cwd state.
    missing = tmp_dir / "does_not_exist.db"
    global DB_PATH
    saved, DB_PATH = DB_PATH, missing
    try:
        assert factor_rows() == [], "factor_rows() must return [] for a missing DB file"
        assert waste_link_rows() == [], "waste_link_rows() must return [] for a missing DB file"
        assert resolution_rows() == [], "resolution_rows() must return [] for a missing DB file"
    finally:
        DB_PATH = saved

    conn = connect(tmp_db)
    print(describe_db(conn))
    conn.close()
    print("\nfactor_db demo: OK")


if __name__ == "__main__":
    demo()
