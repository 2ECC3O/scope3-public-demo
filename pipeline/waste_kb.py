"""The input->waste knowledge base, resolved onto this schema's column names.

Shared by step_2 (B-2A detectors) and step_1 (B-2B generation), so the two-sided rule
in progress.md 0.3 holds structurally rather than by remembering to edit both files -
same arrangement as emission_factors.py.

WHAT THIS IS FOR
----------------
Every waste reference in the engine today is derived from reported waste - the company's
own, or a pooled cross-company median. When a company suppresses or rescales its waste,
the reference moves with the numbers being audited. That single defect is behind the
greenwash detector's 141,600 false positives, the contaminated peer median and the
c12_eol misses (progress.md 10.0).

A stoichiometric link computes expected waste from MATERIAL INPUTS - an independent
channel. `steel_kg -> expected slag` does not care what the company reported for waste.
That is the property no current detector has.

TWO ARMS, AND THEY HAVE DIFFERENT ELIGIBILITY - progress.md 10.7 is the reason
-----------------------------------------------------------------------------
A waste code names TWO unrelated streams (8.9): the process residue, which `ratio`
predicts, and discarded unused stock, which it does not. So:

  magnitude_links()  - "expected = ratio x driver". Needs the reported figure to BE the
                       process stream, so it excludes any code that also carries a
                       discard stream, and any row whose denominator is not this
                       schema's denominator (see basis_qualifier below).
  absence_links()    - "a mandatory code is zero while its driving input is present".
                       Needs only `obligation_class`, not `ratio`, so it keeps the
                       discard-carrying codes - that is exactly what keeps them mandatory.

Applying the ratio where only the absence check is valid is the specific mistake 10.7
warns about, so the two lists are built separately and never merged.

BLINDNESS
---------
Published stoichiometric coefficients are domain knowledge a real auditor has, and this
file contains no pristine value and no injection mechanic - progress.md 0.3 and 2.
`ratio_low`/`ratio_high` are the SOURCE's published range, not a tuned threshold.

ponytail: a resolver and two filtered lists. All judgement lives in the CSV, which is
built from the research by tools/build_waste_kb.py.
"""
import re
from pathlib import Path

import factor_db
import shared

HERE = Path(__file__).resolve().parent
# The rows now come from factor_db (data/factors.db). KB_CSV is gone: naming the CSV
# in the coverage line while reading the DB would misreport provenance, which is the
# one thing this project's logging exists to get right.

USABLE_STATUS = ("USE", "USE-WITH-CAVEAT")

# Which obligation classes each arm may judge. Deliberately different (5g, 2026-08-13):
# the absence arm asks "is an ABSENCE suspicious?", which only `mandatory_process` can
# answer; the magnitude arm asks "does the published ratio predict a stream that IS
# reported?", which an `optional` or `route_conditional` row answers just as well.
# Conflating the two cost the magnitude arm 18 of 49 codes.
MAGNITUDE_OBLIGATION_CLASSES = {"mandatory_process", "route_conditional", "optional"}
ABSENCE_OBLIGATION_CLASSES = {"mandatory_process"}

# Relative half-width applied to a published POINT estimate (ratio_low == ratio_high)
# before it is used as a band. Lives here, not in either step, because step_1 generates
# inside this band and step_2 tests against it - the two-sided rule (progress.md 0.3).
# Applying it on one side only silently makes the generator and the auditor disagree
# about what "in range" means, which is exactly the bug this constant was moved to fix.
POINT_RATIO_SPREAD = 0.25


def ratio_band(link):
    """(low, mid, high) for a link, with point estimates widened by POINT_RATIO_SPREAD.

    Returns (None, None, None) when the link carries no usable ratio.
    """
    mid = link.get("ratio")
    if mid is None:
        return (None, None, None)
    kind = (link.get("source_kind") or "").strip()
    lo = link.get("ratio_low") if link.get("ratio_low") is not None else mid
    hi = link.get("ratio_high") if link.get("ratio_high") is not None else mid
    lo, hi = min(lo, hi), max(lo, hi)
    if kind == "upper_bound":
        lo, mid = 0.0, (hi / 2.0)
    elif kind == "physical_identity":
        lo = mid = hi = 1.0
    elif kind in ("benchmark_maximum", "observed_range", "summary_band"):
        pass
    elif kind == "point_model":
        lo, hi = lo * (1.0 - POINT_RATIO_SPREAD), hi * (1.0 + POINT_RATIO_SPREAD)
    else:
        return (None, None, None)
    return (lo, mid, hi)

_EWC = re.compile(r"(\d{2})\s*(\d{2})\s*(\d{2})")
_coverage = {}


def _clean_material(name):
    """'Resin (unsaturated polyester chemistry)' -> 'Resin'. The only mismatch against
    RAW_MATERIALS; every other input_material is already an exact match."""
    return re.sub(r"\(.*?\)", "", name or "").strip()


def resolve_codes(raw, known_codes):
    """Every code in `known_codes` that a research waste_code cell refers to.

    Handles the four shapes the research tables actually use:
      '10 02 02'                -> 100202
      '10 03 08*/10 03 09*'     -> both
      '07 02 08*/14*'           -> 070208, 070214  (second is abbreviated to the leaf)
      '10 12 01 (waste prep...' -> 100201, parenthetical ignored
    Returns [] for chapter-level codes ('16 11'), analyst placeholders and the deliberate
    'none - internally recirculated' sentinel, all of which are not columns in this schema.
    """
    s = re.sub(r"\[.*?\]", "", raw or "")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.split(r"[—–]|--", s)[0]
    out, prefix = [], None
    for part in s.split("/"):
        t = part.strip().rstrip("*").strip()
        if not t:
            continue
        m = _EWC.match(t)
        if m:
            code = "".join(m.groups())
            prefix = code[:4]
        else:
            d = t.replace(" ", "")
            if prefix and d.isdigit() and len(d) <= 2:
                code = prefix + d.zfill(2)      # '/14*' inherits the '0702' prefix
            else:
                continue
        if code in known_codes and code not in out:
            out.append(code)
    return out


def load_rows():
    """Rows from data/factors.db (waste_link table), byte-identical to what
    csv.DictReader used to return from waste_stoichiometry.csv - see factor_db's
    module docstring and demo() for the equivalence this relies on.

    ponytail: factor_db.waste_link_rows() already returns [] when the database file
    is absent, so this stays a one-line funnel. That [] must not raise: it is the
    stripped-checkout contract (no data/factors.db yet) that both arms below depend
    on to degrade to no-ops instead of crashing step_1/step_2 at import time.
    """
    return factor_db.waste_link_rows()


def _ratio(row, field):
    """Numeric ratio or None. corrected_ratio overrides ratio (progress.md 11 B-1a item 3);
    the research files use an em-dash for 'no value', which is not a number."""
    if field == "ratio":
        raw = row.get("corrected_ratio") or row.get("ratio")
    else:
        raw = row.get(field)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def build_links(known_codes, known_materials, role=None, process_contexts=()):
    """One entry per (row, resolved code). Both arms are filtered from this."""
    links = []
    for r in load_rows():
        if r.get("status") not in USABLE_STATUS:
            continue                     # never consume REFUTED or undecided rows
        mat = _clean_material(r.get("input_material"))
        if mat not in known_materials:
            continue
        codes = resolve_codes(r.get("corrected_waste_code") or r.get("waste_code"), known_codes)
        source_kind = (r.get("source_kind") or "").strip()
        process = (r.get("process") or "").strip()
        eligible = source_kind == "physical_identity" and role == "general_buyer"
        eligible |= process in set(process_contexts)
        if not eligible:
            continue
        for code in codes:
            lo, hi = _ratio(r, "ratio_low"), _ratio(r, "ratio_high")
            links.append({
                "row_id": r["row_id"],
                "material": mat,
                "material_col": f"c1_{mat.lower()}_kg",
                "code": code,
                "waste_col": f"waste_{code}_kg",
                "ratio": _ratio(r, "ratio"),
                "ratio_low": lo,
                "ratio_high": hi,
                "source_kind": source_kind,
                "process": process,
                "basis": r.get("basis", ""),
                "basis_qualifier": (r.get("basis_qualifier") or "").strip(),
                "obligation_class": r.get("obligation_class", ""),
                "carries_discard": r.get("carries_discard_stream") == "true",
                "status": r.get("status"),
            })
    return links


def magnitude_links(known_codes, known_materials, role=None, process_contexts=()):
    """Links where `expected = ratio x driver` predicts the REPORTED quantity.

    Eligibility is NOT the same question the absence arm asks. `obligation_class` answers
    "is an ABSENCE suspicious?" - it says nothing about whether the published ratio predicts
    the MAGNITUDE of a stream that is actually reported. An `optional` code a company does
    report should still match its coefficient, and a `route_conditional` code being reported
    at all means that process route was chosen. Restricting this arm to `mandatory_process`
    (as it was until 5g, 2026-08-13) conflated the two and cost 18 of 49 codes for no
    reason: the arm only ever judges rows where `reported > 0`, so admitting these links
    cannot fire on an absent stream.

    `not_waste` stays out - a sold by-product that never reaches a manifest is not a
    quantity this schema's waste column is even trying to hold.

    Three exclusions remain, each with a reason that is not a threshold choice:
      - carries_discard       : the code mixes process residue with discarded stock, so
                                the ratio predicts only part of what is reported (10.7).
      - no numeric ratio      : nothing to compute.
      - basis_qualifier set   : the denominator is not this schema's. 'per_kg_product
                                (liquid steel, LS)' is per kg of LIQUID STEEL, not of
                                finished product, and converting needs a process yield
                                that no source in the pass supplies (11 B-1a item 4).
                                Two of the 19 research corrections were exactly this slip.
    """
    out = []
    for L in build_links(known_codes, known_materials, role, process_contexts):
        if L["source_kind"] not in ("observed_range", "upper_bound",
                                    "physical_identity", "point_model"):
            continue
        if L["obligation_class"] not in MAGNITUDE_OBLIGATION_CLASSES:
            continue
        if L["carries_discard"]:
            continue
        if L["ratio"] is None:
            continue
        if L["basis_qualifier"]:
            continue
        if L["basis"] not in ("per_kg_input", "per_kg_product"):
            continue
        out.append(L)
    return out


def absence_links(known_codes, known_materials, role=None, process_contexts=()):
    """Links where a ZERO is suspicious if the driving input is present.

    Keeps the discard-carrying codes on purpose: a code covering both streams is still
    mandatory, and its absence is still suppression, even though the ratio cannot say
    how much (10.7). Needs no ratio.
    """
    return [L for L in build_links(known_codes, known_materials, role, process_contexts)
            if L["obligation_class"] in ABSENCE_OBLIGATION_CLASSES]


def driver_series(link, df):
    """The independent quantity this link's expectation is computed FROM.

    per_kg_input   -> the material input column, kg
    per_kg_product -> product mass in kg (the schema stores tonnes)
    Returns None when the driving column is absent from the frame.
    """
    if link["basis"] == "per_kg_input":
        if link.get("source_kind") == "physical_identity":
            return shared.audited_fate_driver(df, link["material"], "waste")
        col = link["material_col"]
        return df[col] if col in df.columns else None
    if link["basis"] == "per_kg_product":
        return shared.audited_product_mass(df)
    return None


def describe_coverage(known_codes, known_materials, role=None, process_contexts=()):
    mag = magnitude_links(known_codes, known_materials, role, process_contexts)
    abs_ = absence_links(known_codes, known_materials, role, process_contexts)
    all_ = build_links(known_codes, known_materials, role, process_contexts)
    _coverage.update(n_all=len(all_), n_mag=len(mag), n_abs=len(abs_))
    if not all_:
        return (f"[KB] WARNING: {factor_db.DB_PATH.name} produced no usable links - "
                f"stoichiometric detectors are inert.")
    return "\n".join([
        f"[KB] {factor_db.DB_PATH.name}: {len(all_)} usable (row x code) links over "
        f"{len({L['code'] for L in all_})} waste columns and "
        f"{len({L['material'] for L in all_})} materials.",
        f"[KB]   magnitude arm: {len(mag)} links, {len({L['code'] for L in mag})} columns "
        f"(ratio predicts the reported figure)",
        f"[KB]   absence arm  : {len(abs_)} links, {len({L['code'] for L in abs_})} columns "
        f"(mandatory; zero is suspicious when the input is present)",
    ])


def demo():
    """Self-check against the real schema. Asserts the things that silently break a detector."""
    import pandas as pd
    wdf = pd.read_csv(HERE.parent / "waste_codes.csv")
    codes = {str(c).replace(" ", "").strip() for c in wdf["Waste Code"].dropna()
             if str(c).strip() and str(c) != "nan"}
    import emission_factors as EF
    mats = set(EF.RAW_MATERIALS)

    # resolver shapes seen in the research tables
    assert resolve_codes("10 02 02", codes) == ["100202"]
    assert resolve_codes("10 03 08*/10 03 09*", codes) == ["100308", "100309"]
    assert resolve_codes("07 02 08*/14*", codes) == ["070208", "070214"]
    assert resolve_codes("16 11", codes) == []                    # chapter, not a leaf
    assert resolve_codes("none - internally recirculated", codes) == []

    # Catches a silently empty database: with an absent data/factors.db, load_rows()
    # legitimately returns [] and both arms below go inert without error - the exact
    # behaviour a stripped checkout needs. But a PRESENT, empty-of-rows database would
    # look identical to "the KB is legitimately empty" without this count pinned down.
    assert len(load_rows()) == 139, (
        f"load_rows() returned {len(load_rows())} rows, expected 139 - "
        f"data/factors.db is missing rows, not just missing")

    buyer_links = build_links(codes, mats, role="general_buyer")
    assert len(buyer_links) == 4 and all(
        L["source_kind"] == "physical_identity" for L in buyer_links), (
        "general-buyer coverage must be the four inbound packaging identities")
    declared_processes = {row.get("process", "") for row in load_rows()}
    links = build_links(codes, mats, process_contexts=declared_processes)
    assert links, "declared process links are not reaching the schema"
    mag = magnitude_links(codes, mats, process_contexts=declared_processes)
    ab = absence_links(codes, mats, process_contexts=declared_processes)

    # every arm's invariants
    assert all(L["status"] in USABLE_STATUS for L in links), "a non-usable row leaked in"
    assert all(L["ratio"] is not None and not L["carries_discard"] for L in mag)
    assert all(L["obligation_class"] in MAGNITUDE_OBLIGATION_CLASSES for L in mag)
    assert all(L["obligation_class"] in ABSENCE_OBLIGATION_CLASSES for L in ab)
    assert not any(L["basis_qualifier"] for L in mag), "a mismatched denominator leaked in"
    # After 5g the arms overlap rather than nest: magnitude admits optional and
    # route_conditional links the absence arm must not touch, and absence keeps the
    # discard-carrying links magnitude must not. Assert the invariant that still holds --
    # neither arm ever contains a link the other's own filters would reject outright.
    assert not any(L["carries_discard"] for L in mag)
    assert all(L["ratio"] is not None for L in mag)
    # the refuted rows must never appear
    refuted = {"w1-2-nonferrous:20", "w1-3-plastics:31", "w1-3-plastics:38"}
    assert not (refuted & {L["row_id"] for L in links}), "a REFUTED row leaked in"

    print(describe_coverage(codes, mats, role="general_buyer"))
    print("\nwaste_kb demo: OK")


if __name__ == "__main__":
    demo()
