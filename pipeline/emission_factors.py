"""The single source of emission factors for step_1 and step_2.

WHY THIS MODULE EXISTS
----------------------
`get_deterministic_ef`, `RAW_MATERIALS` and `EMISSION_FACTORS` used to be defined
VERBATIM TWICE - step_1_generate_data.py:174/400-409 and step_2_verify_data.py:446/481-489.
That duplication is load-bearing in the worst way: `emissions = activity x EF` holds
exactly in this system, and identity 4 (`total_product_emissions_mtco2 == sum(7 em cols)`)
depends on it. If the two copies ever drift by so much as a rounding, every emissions
identity fires on every company including clean data - see progress.md 10.5.

So both steps now import from here. The two-sided rule (progress.md 0.3) is satisfied
structurally rather than by remembering to edit two files.

PROVENANCE
----------
Material factors come from `emission_factors_tgo.csv` (Thai National LCI / TGO CFP,
updated April 2026) where TGO publishes one. 16 of 45 materials do; the other 29 are
confirmed structural gaps, not search failures - Lithium and Rare_Earth_Elements have no
JRC BREF at all, semiconductors and wafers appear in none of TGO's 540 entries. Those 29
fall back to the original deterministic hash and are LOGGED, so coverage stays visible
instead of fiction passing as sourced (progress.md 11 Phase C).

Where TGO publishes several grades of one material, the CSV marks one `is_commodity_grade`
- see the COMMODITY_GRADE table in tools/build_ef_csv.py for which and why.

ponytail: a loader and two dicts. The grade decisions live in the builder next to the
research, not here, so this file has no judgement in it to go stale.
"""
import hashlib
from pathlib import Path

import factor_db

HERE = Path(__file__).resolve().parent

RAW_MATERIALS = [
    'Steel', 'Aluminum', 'Copper', 'Zinc', 'Brass', 'Titanium', 'Nickel', 'Iron_Ore',
    'PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS', 'Polyurethane', 'Nylon', 'Resin', 'Rubber', 'Silicone',
    'Sulfuric_Acid', 'Sodium_Hydroxide', 'Ammonia', 'Chlorine', 'Solvents_Organic', 'Solvents_Aqueous',
    'Catalyst_Precious', 'Catalyst_Base', 'Paints_Coatings', 'Adhesives', 'Dyes_Pigments',
    'Silicon_Wafers', 'PCB_Boards', 'Semiconductors', 'Lithium', 'Cobalt', 'Rare_Earth_Elements',
    'Cardboard', 'Paper', 'Wood_Pallets', 'Glass', 'Shrink_Wrap',
    'Textiles_Cotton', 'Textiles_Synthetic', 'Lubricating_Oils', 'Ceramics'
]

# Grid electricity, kgCO2e/kWh. Thailand is pinned to the TGO CFO Scope 2 factor by
# progress.md 10.8 decision 4 (2026-08-12): vintage 2022-2024 grid mix, location-based,
# generation + T&D, excluding upstream fuel supply. Justified as TGO-programme
# consistency, NOT as a general GHG Protocol argument - GHG Protocol would use a
# generation-only factor and report T&D under Scope 3 cat 3.
#
# The other four regions have no TGO equivalent and keep their original unsourced values.
GRID_EF_THAILAND_ROW = "w4-2-grid-freight-ef:69"
REGIONS = {
    'Thailand': 0.4750,        # TGO, pinned - see above
    'Vietnam': 0.72,           # unsourced, pre-existing
    'China': 0.58,             # unsourced, pre-existing
    'Europe': 0.25,            # unsourced, pre-existing
    'North America': 0.38,     # unsourced, pre-existing
}
if (HERE.parent / 'PUBLIC-DEMO-MODE').is_file():
    # ponytail: extracted public demo uses an invented grid input; private
    # research checkouts keep the pinned TGO value and frozen measurements.
    REGIONS['Thailand'] = 0.5

# No TGO source found for either. Left as-is and reported by describe_coverage().
UTILITY_EFS = {'Non_Grid_Energy_MJ': 0.07, 'Water_Use_m3': 0.3}

# Pre-existing hand-set waste-code factors, kept. TGO publishes 7 usable per-kg TREATMENT
# PATHWAY factors (sanitary landfill 0.7933, open dumping 1.0387, anaerobic digestion
# 0.1055, compost 0.3323, ...) but those are not per-EWC-code factors and cannot replace
# these: this schema has 857 codes and models treatment as a multiplier on top of a
# per-code factor. Wiring the TGO pathway factors in means restructuring that model, which
# belongs with the _assign_treatment rewrite in B-2B, not here. See progress.md 10.9.
WASTE_CODE_OVERRIDES = {'150110': 0.8, '160602': 1.2, '160215': 1.5,
                        '160213': 1.1, '080318': 0.9, '150202': 0.7}

# Categories 1/5/9/12: single definition. step_1 uses the nested structure to
# generate messy data; step_2's xlsx-parse fallback flattens the same numbers into
# its REG_EF_MAP key names (e.g. INDIRECT_SPEND_CATEGORIES['IT_Services']['ef_per_usd']
# == the fallback's Spend_IT_Services_EF). These used to be hand-duplicated between
# the two files - see reviews/.../2026-08-14-code-audit.md (C1b).

# CATEGORY 1 - kgCO2e per USD spent, and typical monthly spend range per product
INDIRECT_SPEND_CATEGORIES = {
    'IT_Services':     {'ef_per_usd': 0.15, 'spend_range': (5000, 50000)},
    'Consulting':      {'ef_per_usd': 0.10, 'spend_range': (2000, 30000)},
    'Packaging':       {'ef_per_usd': 0.25, 'spend_range': (3000, 40000)},
    'Office_Supplies': {'ef_per_usd': 0.08, 'spend_range': (1000, 15000)},
    'Logistics_Mgmt':  {'ef_per_usd': 0.20, 'spend_range': (4000, 35000)},
}

# CATEGORY 5 - emission multiplier applied ON TOP of the base waste EF, by treatment method
WASTE_TREATMENT_METHODS = {
    'Landfill':       {'ef_multiplier': 0.52, 'weight': 0.45},
    'Incineration':   {'ef_multiplier': 1.10, 'weight': 0.25},
    'Recycling':      {'ef_multiplier': 0.05, 'weight': 0.20},
    'Composting':     {'ef_multiplier': 0.03, 'weight': 0.10},
}

# CATEGORY 9 - kgCO2e per tonne-kilometre, and typical delivery distance range
TRANSPORT_MODES = {
    'Road':  {'ef_per_tkm': 0.062, 'dist_range': (50, 800)},
    'Rail':  {'ef_per_tkm': 0.022, 'dist_range': (200, 2000)},
    'Sea':   {'ef_per_tkm': 0.008, 'dist_range': (1000, 15000)},
    'Air':   {'ef_per_tkm': 0.602, 'dist_range': (500, 10000)},
}

# CATEGORY 12 - end-of-life disposal profile per region, and base EF per pathway
EOL_PROFILES = {
    'North America': {'Landfill': 0.70, 'Recycled': 0.15, 'Incinerated': 0.15},
    'Europe':        {'Landfill': 0.30, 'Recycled': 0.45, 'Incinerated': 0.25},
    'Thailand':      {'Landfill': 0.60, 'Recycled': 0.20, 'Incinerated': 0.20},
    'Vietnam':       {'Landfill': 0.65, 'Recycled': 0.18, 'Incinerated': 0.17},
    'China':         {'Landfill': 0.50, 'Recycled': 0.30, 'Incinerated': 0.20},
}
EOL_EF = {  # kgCO2e per kg disposed
    'Landfill': 0.46,
    'Recycled': 0.02,
    'Incinerated': 0.95,
}

_coverage = {}          # filled by build_emission_factors(), read by describe_coverage()


def get_deterministic_ef(name, min_val, max_val):
    """Generates mathematically consistent factors for any string."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return round(min_val + (h % 1000) / 1000.0 * (max_val - min_val), 2)


def load_material_efs(region=None):
    """material name -> (kgCO2e/kg, row_id, source name), for the chosen row only.

    Returns {} if the database is missing, so a stripped checkout still runs on hash
    factors. factor_db.factor_rows() is byte-identical to csv.DictReader on the CSV this
    used to read (same keys, same string values, same order - asserted in factor_db's own
    demo()), so every filter below is unchanged from the CSV-reading version.

    Renamed from load_tgo_material_efs(): `factor` holds only TGO rows today, but the
    schema is source-agnostic (see factor_db.py / plan.md "The EF Broker") and this is
    the loader a second source lands behind, not a TGO-specific one.

    Collision handling: `is_commodity_grade == "true"` used to guarantee at most one
    eligible row per material (build_ef_csv.py asserts that for TGO). That guarantee is
    TGO-internal - a second source publishing its own `is_commodity_grade == "true"` row
    for a material TGO already covers will not be caught by it. So collisions are detected
    explicitly below: >=2 eligible rows for the same material consult the `resolution`
    table (material, region, row_id, reason, decided_on); named -> use it, visibly;
    unnamed -> skip the material loudly rather than let dict-key overwrite pick one by
    table order. Same non-crashing shape as the `unit != "kg"` guard just above it.

    `region`: optional, defaults to None (wildcard-only - no caller passes a region
    today). When given, a resolution row named for that exact region wins; absent one,
    the wildcard ('*') row for the material wins instead; absent both, unchanged
    ambiguous-skip behaviour. See factor_db.py's RESOLUTION_COLUMNS comment for why '*'
    is the sentinel.
    """
    rows = factor_db.factor_rows()
    if not rows:
        return {}
    resolutions = {}  # material -> {region: row}, so an exact region and '*' can coexist
    for r in factor_db.resolution_rows():
        resolutions.setdefault(r["material"], {})[r.get("region") or "*"] = r
    eligible = {}  # material -> [(value, row_id, name), ...] - same shape as the return value
    for r in rows:
        mat = (r.get("material_match") or "").strip()
        if mat.lower() in ("", "none", "-"):
            continue
        if r.get("is_commodity_grade") != "true":
            continue          # a non-chosen grade of a material that has several
        if r.get("status", "") not in ("USE", "USE-WITH-CAVEAT"):
            continue          # never silently consume a refuted or volumetric row
        if r.get("unit") != "kg":
            # Structural guard, not just the editorial DO-NOT-USE-ON-MASS status:
            # a per-kg mass factor is the only unit this loader may hand out.
            # Skip + log (not assert) - a future non-kg row must not crash the
            # pipeline, only be excluded loudly.
            # See reviews/_project/process-audit/2026-08-14-code-audit.md (C8).
            print(f"[EF] skipping {r.get('row_id')} ({mat}): unit "
                  f"{r.get('unit')!r} is not a per-kg mass unit")
            continue
        try:
            val = float(r["value"])
        except (TypeError, ValueError):
            continue
        eligible.setdefault(mat, []).append((val, r["row_id"], r["name"]))

    out = {}
    for mat, candidates in eligible.items():
        if len(candidates) == 1:
            out[mat] = candidates[0]
            continue
        # Collision: two or more eligible rows claim the same material. Never let dict-key
        # overwrite silently pick the last one by table order (the hazard this loader used
        # to have - see the collision-detection task, no CSV/doc source, ask the owner).
        row_ids = sorted(c[1] for c in candidates)
        mat_res = resolutions.get(mat, {})
        # Exact region first, then the wildcard, then nothing - matches the docstring.
        res = (mat_res.get(region) if region else None) or mat_res.get("*")
        if res and res["row_id"] in row_ids:
            chosen = next(c for c in candidates if c[1] == res["row_id"])
            out[mat] = chosen
            print(f"[EF] {mat}: {len(candidates)} competing rows ({', '.join(row_ids)}) - "
                  f"resolution ({res.get('region') or '*'}) picks {res['row_id']}: "
                  f"{res['reason']}")
        else:
            print(f"[EF] AMBIGUOUS: {mat} has {len(candidates)} competing eligible rows "
                  f"({', '.join(row_ids)}) and no `resolution` row names a winner. Add one "
                  f"via factor_db.set_resolution(conn, {mat!r}, <row_id>, <reason>, <date>) "
                  f"to resolve this. Skipping {mat} - it falls back to the deterministic hash.")
            # Not added to `out` - loud skip, not a silent pick, not a crash.
    return out


def build_emission_factors(waste_codes_list):
    """The EMISSION_FACTORS dict both steps use. Identical by construction.

    TGO value where one exists for the material, deterministic hash otherwise.
    """
    tgo = load_material_efs()

    mats, sourced, fallback = {}, [], []
    for mat in RAW_MATERIALS:
        if mat in tgo:
            mats[mat] = tgo[mat][0]
            sourced.append(mat)
        else:
            mats[mat] = get_deterministic_ef(mat, 0.5, 15.0)
            fallback.append(mat)

    waste = {w: get_deterministic_ef(w, 0.1, 5.0) for w in waste_codes_list}
    for code, val in WASTE_CODE_OVERRIDES.items():
        if code in waste:
            waste[code] = val

    _coverage.clear()
    _coverage.update(sourced=sourced, fallback=fallback, tgo=tgo,
                     n_waste=len(waste), csv_found=bool(tgo))
    return {'mats': mats, 'waste': waste, 'utilities': dict(UTILITY_EFS)}


def describe_coverage():
    """One-line-per-fact summary. Both steps print this so a run records what it used."""
    if not _coverage:
        return "[EF] build_emission_factors() has not been called."
    s, f = _coverage["sourced"], _coverage["fallback"]
    public_demo = (HERE.parent / 'PUBLIC-DEMO-MODE').is_file()
    lines = [
        f"[EF] material factors: {len(s)}/{len(RAW_MATERIALS)} from {'invented demo input' if public_demo else 'TGO'} "
        f"({factor_db.DB_PATH.name}), {len(f)} on deterministic hash fallback.",
        f"[EF]   sourced : {', '.join(sorted(s)) or '(none)'}",
        f"[EF]   FALLBACK: {', '.join(sorted(f)) or '(none)'}",
        (f"[EF] grid Thailand invented demo input: {REGIONS['Thailand']} kgCO2e/kWh; other regions are project assumptions."
         if public_demo else f"[EF] grid Thailand pinned to {REGIONS['Thailand']} kgCO2e/kWh "
         f"(TGO CFO Scope 2, 2022-2024; row {GRID_EF_THAILAND_ROW}). Other 4 regions unsourced."),
        f"[EF] waste codes: {_coverage['n_waste']} on deterministic hash "
        f"(no per-EWC-code TGO factor exists); utilities unsourced.",
    ]
    if not _coverage["csv_found"]:
        lines.insert(0, f"[EF] WARNING: {factor_db.DB_PATH} not loaded - ALL factors are hash fiction.")
    return "\n".join(lines)


def demo():
    """Self-check: the two properties that break the pipeline if violated."""
    codes = ['150110', '160602', '999999']
    a = build_emission_factors(codes)
    b = build_emission_factors(codes)
    assert a == b, "EF construction is not deterministic - identities will not hold"

    tgo = load_material_efs()
    assert tgo, "TGO CSV loaded no rows - check emission_factors_tgo.csv"
    # One grade per material, or the loader picked arbitrarily from a contested set.
    assert len(tgo) == len({m for m in tgo}), "duplicate material in TGO map"
    # A silently empty/short database must fail loudly, not pass as "everything fell
    # back to hash".
    assert len(tgo) == 16, f"expected 16 TGO-sourced materials, got {len(tgo)}"
    assert a['mats']['Steel'] == 1.8578, f"Steel should be TGO hot-rolled coil, got {a['mats']['Steel']}"
    assert a['mats']['Aluminum'] == 11.9048, "Aluminum should be TGO billet"
    # Silicon_Wafers has no TGO entry and must fall back, not keep the old magic 12.0.
    assert 'Silicon_Wafers' not in tgo
    assert a['mats']['Silicon_Wafers'] == get_deterministic_ef('Silicon_Wafers', 0.5, 15.0)
    assert a['waste']['150110'] == 0.8, "override not applied"
    assert a['waste']['999999'] == get_deterministic_ef('999999', 0.1, 5.0)
    assert set(a['mats']) == set(RAW_MATERIALS), "material set drifted"

    # Categories 1/5/9/12 - single-source tables shared by step_1 and step_2.
    assert set(INDIRECT_SPEND_CATEGORIES) == {
        'IT_Services', 'Consulting', 'Packaging', 'Office_Supplies', 'Logistics_Mgmt'}
    assert set(WASTE_TREATMENT_METHODS) == {'Landfill', 'Incineration', 'Recycling', 'Composting'}
    assert abs(sum(v['weight'] for v in WASTE_TREATMENT_METHODS.values()) - 1.0) < 1e-9, \
        "WASTE_TREATMENT_METHODS weights must sum to 1"
    assert set(TRANSPORT_MODES) == {'Road', 'Rail', 'Sea', 'Air'}
    assert set(EOL_PROFILES) == set(REGIONS), "EOL_PROFILES regions must match REGIONS"
    for region, profile in EOL_PROFILES.items():
        assert set(profile) == {'Landfill', 'Recycled', 'Incinerated'}
        assert abs(sum(profile.values()) - 1.0) < 1e-9, \
            f"EOL_PROFILES['{region}'] fractions must sum to 1"
    assert set(EOL_EF) == {'Landfill', 'Recycled', 'Incinerated'}

    print(describe_coverage())
    _demo_collision()
    print("\nemission_factors demo: OK")


def _demo_collision():
    """Prove the collision path on a synthetic pair of colliding rows, since nothing in
    the real data collides today (is_commodity_grade already selects one TGO row per
    material) - so this is the only way the new code path in load_material_efs() ever
    runs. Three cases, in order: (1) no resolution row at all -> loud skip, neither
    value silently chosen; (2) a wildcard ('*') resolution row -> wins for a region that
    has no row of its own; (3) a region-specific resolution row -> wins over the
    wildcard for that one region.
    """
    import tempfile

    tmp_db = Path(tempfile.mkdtemp()) / "collision_demo.db"
    conn = factor_db.connect(tmp_db)
    factor_db.init_schema(conn)
    factor_db.replace_source(conn, "demo", "synthetic collision fixture", "", "", "manual",
                              "2026-08-22")
    base = {"category": "", "vintage": "", "scope": "", "is_ipcc_default": "", "verdict": "",
            "verify_report": "", "note": "", "tgo_source_doc": "", "source_url": "",
            "source_date": "", "confidence": "", "geography": "", "gwp_method": ""}
    rows = [
        dict(base, row_id="demo:1", name="Widget grade A", material_match="SyntheticWidget",
             is_commodity_grade="true", value="1.0", unit="kg", status="USE"),
        dict(base, row_id="demo:2", name="Widget grade B", material_match="SyntheticWidget",
             is_commodity_grade="true", value="2.0", unit="kg", status="USE"),
    ]
    factor_db.replace_factors(conn, "demo", rows)
    conn.close()

    saved_db_path = factor_db.DB_PATH
    factor_db.DB_PATH = tmp_db
    try:
        # Case 3 (checked first, since it needs zero resolution rows): no resolution row
        # named yet, for any region - must fail loud, not pick one silently.
        efs = load_material_efs(region="Europe")
        assert "SyntheticWidget" not in efs, (
            "an unresolved collision must not silently choose a value: " + repr(efs))

        # Name a wildcard winner (region defaults to '*'). It applies to any region that
        # has no row of its own - including a no-region caller (today's only real shape).
        conn = factor_db.connect(tmp_db)
        factor_db.set_resolution(conn, "SyntheticWidget", "demo:2", "demo pick, higher grade",
                                  "2026-08-22")
        conn.close()

        # Case 2: a region with no region-specific row falls back to the wildcard row.
        efs = load_material_efs(region="Thailand")
        assert efs.get("SyntheticWidget") == (2.0, "demo:2", "Widget grade B"), (
            "wildcard resolution must decide the collision when no region-specific row "
            "exists: " + repr(efs.get("SyntheticWidget")))
        efs = load_material_efs()  # no region at all -> same wildcard row
        assert efs.get("SyntheticWidget") == (2.0, "demo:2", "Widget grade B")

        # Name a Europe-specific winner - the OTHER row, so precedence is provable.
        conn = factor_db.connect(tmp_db)
        factor_db.set_resolution(conn, "SyntheticWidget", "demo:1", "Europe-specific pick",
                                  "2026-08-22", region="Europe")
        conn.close()

        # Case 1: a Europe company now gets the Europe-specific row, not the wildcard.
        efs = load_material_efs(region="Europe")
        assert efs.get("SyntheticWidget") == (1.0, "demo:1", "Widget grade A"), (
            "region-specific resolution must win over the wildcard for its own region: "
            + repr(efs.get("SyntheticWidget")))
        # A Thailand company still has no Thailand-specific row - still the wildcard.
        efs = load_material_efs(region="Thailand")
        assert efs.get("SyntheticWidget") == (2.0, "demo:2", "Widget grade B"), (
            "a region without its own row must keep falling back to the wildcard row: "
            + repr(efs.get("SyntheticWidget")))
    finally:
        factor_db.DB_PATH = saved_db_path
    print("[EF] _demo_collision: unresolved collision skipped loud, wildcard resolution "
          "picked for a region with no row of its own, region-specific resolution won "
          "over the wildcard for its own region - OK")


if __name__ == "__main__":
    demo()
