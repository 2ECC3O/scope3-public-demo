"""Check arithmetic in two CSV breakdowns of the same reported inventory.

Usage: uv run python tools/review_inventory_tables.py mapping.json --year 2022
Mapping JSON contains unit, year-indexed files, and each table's id_column,
value_column, percent_column, and total_label. Paths are relative to mapping.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


NUMBER = re.compile(r'(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d+)?\Z')


def decimal(text, field, percent=False):
    if not isinstance(text, str):
        raise ValueError(f'{field}: missing number')
    text = text.strip()
    if percent and text.endswith('%'):
        text = text[:-1]
    if not NUMBER.fullmatch(text):
        raise ValueError(f'{field}: invalid number {text!r}')
    try:
        value = Decimal(text.replace(',', ''))
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f'{field}: invalid number {text!r}') from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f'{field}: expected finite nonnegative number')
    return value


def read_table(path, config):
    for precision in ('value_resolution_mtco2e', 'percent_resolution_points'):
        value = decimal(str(config[precision]), precision)
        if value <= 0:
            raise ValueError(f'{precision}: expected positive reporting precision')
    encoding = 'utf-16' if path.read_bytes()[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig'
    with path.open(encoding=encoding, newline='') as stream:
        sample = stream.read(4096)
        stream.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t;')
        reader = csv.DictReader(stream, dialect=dialect)
        columns = [config[key] for key in ('id_column', 'value_column', 'percent_column')]
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError(f'{path}: duplicate CSV header')
        if len(set(columns)) != 3 or not set(columns) <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: missing or repeated mapped columns')
        records = list(reader)
    if not records or any(None in row or any(row[c] is None for c in columns) for row in records):
        raise ValueError(f'{path}: empty or malformed record')
    seen, parts, total = set(), [], None
    for row in records:
        key = row[config['id_column']].strip()
        if not key or key in seen:
            raise ValueError(f'{path}: blank or duplicate ID {key!r}')
        seen.add(key)
        value = decimal(row[config['value_column']], f'{path}:{key}:value')
        pct = decimal(row[config['percent_column']], f'{path}:{key}:percent', percent=True)
        if (value % Decimal(config['value_resolution_mtco2e']) != 0 or
                pct % Decimal(config['percent_resolution_points']) != 0):
            raise ValueError(f'{path}:{key}: precision differs from mapping')
        if key == config['total_label']:
            total = (value, pct)
        else:
            parts.append((key, value, pct))
    if total is None or not parts or total[0] == 0:
        raise ValueError(f'{path}: expected positive total and components')
    if 'expected_ids' in config and seen != set(config['expected_ids']) | {config['total_label']}:
        raise ValueError(f'{path}: mapped component coverage differs from expected IDs')
    return total, parts


def review(path, config, change_id=None):
    total, parts = read_table(path, config)
    if total[1] != 100:
        raise ValueError(f'{path}: grand-total percentage is not 100')
    if change_id and change_id not in {key for key, _, _ in parts}:
        raise ValueError(f'{path}: controlled-change ID absent')
    if change_id:
        # ponytail: fixed large error tests the rule; small-error sensitivity is unmeasured.
        parts = [(key, value + Decimal(10000000) if key == change_id else value, pct)
                 for key, value, pct in parts]
    reported_total = total[0]
    residual = sum((value for _, value, _ in parts), Decimal(0)) - reported_total
    sum_limit = Decimal(len(parts) + 1) * Decimal(config['value_resolution_mtco2e']) / 2
    percent_limit = Decimal(config['percent_resolution_points']) / 2 + Decimal('0.000000001')
    percent_flags = [key for key, value, pct in parts
                     if abs(pct - 100 * value / reported_total) > percent_limit]
    return {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'component_count': len(parts), 'reported_total_mtco2e': str(reported_total),
            'component_sum_mtco2e': str(reported_total + residual),
            'sum_residual_mtco2e': str(residual), 'sum_rounding_limit_mtco2e': str(sum_limit),
            'sum_review': abs(residual) > sum_limit, 'percent_review_ids': percent_flags,
            'coverage_status': 'checked_expected_ids' if 'expected_ids' in config else 'unverified',
            'controlled_change_id': change_id}


def run(mapping_path, year, controlled=False):
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    if mapping['unit'] != 'metric tonnes CO2e':
        raise ValueError('Only explicit metric tonnes CO2e mapping is supported')
    year_config = mapping['years'][str(year)]
    results = {}
    for name in ('agency', 'category'):
        config = mapping['tables'][name]
        path = mapping_path.parent / year_config[name]
        change_id = mapping['controlled_change_ids'][name] if controlled else None
        results[name] = review(path, config, change_id)
    difference = abs(Decimal(results['agency']['reported_total_mtco2e']) -
                     Decimal(results['category']['reported_total_mtco2e']))
    cross_limit = (Decimal(mapping['tables']['agency']['value_resolution_mtco2e']) +
                   Decimal(mapping['tables']['category']['value_resolution_mtco2e'])) / 2
    return {'year': year, 'unit': mapping['unit'], 'controlled': controlled,
            'source': mapping['source'], 'tables': results,
            'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'mapping_sha256': hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            'cross_table_total_difference_mtco2e': str(difference),
            'cross_table_rounding_limit_mtco2e': str(cross_limit),
            'cross_table_review': difference > cross_limit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mapping', type=Path)
    parser.add_argument('--year', required=True)
    parser.add_argument('--controlled', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.mapping, args.year, args.controlled)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
