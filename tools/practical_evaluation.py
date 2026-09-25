"""Review EPA 2023 reported gas totals with the verifier's existing identity rule.

Download the official archive named in docs/plans/2026-09-20-practical-evaluation.md,
extract ghgp_data_2023.xlsx, then pass its path. No data are downloaded here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import step_2_verify_data as verifier

SOURCE_SHA256 = 'e348830722f59de38640b6f3ddfc9b7d406f26c3c153ffbdd7192c2260114ac3'
TOTAL = 'total_reported_direct_emissions_mtco2e'
GASES = ('co2_mtco2e', 'ch4_mtco2e', 'n2o_mtco2e')
HEADERS = ('Facility Id', 'Total reported direct emissions',
           'CO2 emissions (non-biogenic) ', 'Methane (CH4) emissions ',
           'Nitrous Oxide (N2O) emissions ')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def rows_from_workbook(path):
    if sha256(path) != SOURCE_SHA256:
        raise ValueError('Workbook SHA-256 differs from pinned EPA 2023 file; see evaluation plan')
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if 'Direct Point Emitters' not in book:
            raise ValueError('Missing EPA Direct Point Emitters sheet')
        rows = book['Direct Point Emitters'].values
        first, dated, unit, headings = (next(rows) for _ in range(4))
        if '2023' not in str(first[0]) or '8/16/2024' not in str(dated[0]):
            raise ValueError('Unexpected EPA reporting year/date')
        if 'metric tons of carbon dioxide equivalent' not in str(unit[0]) or 'AR4' not in str(unit[0]):
            raise ValueError('Unexpected emissions units or GWP basis')
        if tuple(headings[i] for i in (0, 13, 14, 15, 16)) != HEADERS or headings[17] != 'HFC emissions' or headings[24] != 'Other GHGs (metric tons CO2e)':
            raise ValueError('EPA facility columns changed')
        selected = {0: [], 1: []}
        excluded = 0
        for values in rows:
            if len(selected[0]) == len(selected[1]) == 24:
                break
            fid, total, *gases = values[0], values[13], *values[14:17]
            # Other-gas blanks mean "not reported"; they are not certified zeros.
            eligible = (type(fid) is int and number(total) and total > 0
                        and all(number(x) for x in gases)
                        and all(x is None for x in values[17:25]))
            if not eligible:
                excluded += 1
                continue
            parity = fid % 2
            if len(selected[parity]) < 24:
                selected[parity].append((fid, float(total), *(float(x) for x in gases)))
        if len(selected[0]) != 24 or len(selected[1]) != 24:
            raise ValueError('Fewer than 24 eligible EPA facilities in a split')
        if {x[0] for x in selected[0]} & {x[0] for x in selected[1]}:
            raise ValueError('Facility ID appears in both splits')
        return selected, excluded
    finally:
        book.close()


def check(rows, split, variant):
    if len(rows) != 24 or any(len(row) != 5 or type(row[0]) is not int or
                              any(not number(value) for value in row[1:]) for row in rows):
        raise ValueError('Expected 24 facilities with finite, nonnegative numeric gas and total fields')
    df = pd.DataFrame(rows, columns=('facility_id', TOTAL, *GASES))
    expected = df[list(GASES)].sum(axis=1).to_numpy(dtype=float)
    original = df[TOTAL].to_numpy(dtype=float).copy()
    consistent = np.isclose(original, expected, rtol=1e-8, atol=0.001)
    if not consistent.all():
        raise ValueError(f'{split}: {int((~consistent).sum())} source rows have arithmetic discrepancies; unchanged rows cannot be labelled negatives')
    changed = np.zeros(len(df), dtype=bool)
    if variant == 'controlled':
        changed[[2, 10]] = True
        df.loc[changed, TOTAL] *= 10
    residual = df[TOTAL].to_numpy(dtype=float) - expected
    members = [(TOTAL, 1.0)] + [(gas, -1.0) for gas in GASES]
    # No invented product baseline: a violation is a row review, not cell blame.
    verifier._check_identity_violation(df, 'EPA total == sum(reported gases)',
                                       residual, expected, members, pd.DataFrame(), 'EPA_2023')
    reviews = df.get('row_identity_review_needed', pd.Series(0, index=df.index)).eq(1).to_numpy()
    result = df[['facility_id', TOTAL, *GASES]].copy()
    result.insert(0, 'split', split)
    result.insert(1, 'record', variant)
    result['source_total_mtco2e'] = original
    result['reported_gas_sum_mtco2e'] = expected
    result['residual_mtco2e'] = residual
    result['total_deliberately_changed'] = changed
    result['row_review_needed'] = reviews
    result['reported_component_identity_status_only'] = df[f'{TOTAL}_status'].to_numpy()
    result['other_gases'] = 'not reported; completeness unverified'
    result['interpretation'] = 'arithmetic identity only; emissions correctness unverified'
    return result, {'facilities': len(df), 'source_arithmetic_discrepancies': 0,
                    'changed_total_cells': int(changed.sum()),
                    'unchanged_total_cells': int((~changed).sum()),
                    'tp': int(np.sum(reviews & changed)), 'fp': int(np.sum(reviews & ~changed)),
                    'fn': int(np.sum(~reviews & changed)), 'tn': int(np.sum(~reviews & ~changed)),
                    'row_review_ids': df.loc[reviews, 'facility_id'].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/practical-evaluation/review-v1')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Output exists; use a fresh --output-dir: {args.output_dir}')
    selected, excluded = rows_from_workbook(args.workbook)
    results, records = {}, []
    for split, parity in (('development', 0), ('evaluation', 1)):
        results[split] = {}
        for variant in ('source', 'controlled'):
            table, metrics = check(selected[parity], split, variant)
            records.append(table)
            results[split][variant] = metrics
    # Fixed v1 result is a regression check, not a threshold calibration.
    for split in ('development', 'evaluation'):
        m = results[split]['controlled']
        assert (m['tp'], m['fp'], m['fn'], m['tn']) == (2, 0, 0, 22)
        assert results[split]['source']['row_review_ids'] == []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'source': 'EPA GHGRP 2023 direct-emitter facility summary',
              'source_url': 'https://www.epa.gov/system/files/other-files/2024-10/2023_data_summary_spreadsheets.zip',
              'workbook_sha256': sha256(args.workbook),
              'verifier_source_sha256': sha256(ROOT / 'pipeline/step_2_verify_data.py'),
              'adapter_sha256': sha256(Path(__file__)),
              'units': 'metric tonnes CO2e, IPCC AR4',
              'selection': 'first 24 eligible even and odd facility IDs in source order',
              'selected_facility_ids': {s: [row[0] for row in selected[p]] for s, p in (('development', 0), ('evaluation', 1))},
              'excluded_before_selection_complete': excluded,
              'scope': 'reported-component arithmetic only; other gases and all Scope 3 fields unverified',
              'results': results}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pd.concat(records, ignore_index=True).to_csv(args.output_dir / 'facility_reviews.csv', index=False)
    artifacts = ('summary.json', 'facility_reviews.csv')
    if not all((args.output_dir / name).is_file() and (args.output_dir / name).stat().st_size > 0 for name in artifacts):
        raise RuntimeError('Incomplete review output')
    (args.output_dir / 'PASS.json').write_text(json.dumps({name: sha256(args.output_dir / name) for name in artifacts}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output_dir': str(args.output_dir), 'results': results}, indent=2))


if __name__ == '__main__':
    main()







