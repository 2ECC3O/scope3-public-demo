"""Development-only input boundary check, not a scored inventory case."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_inventory_tables import decimal, read_table


class TestInventoryImport(unittest.TestCase):
    def test_ambiguous_or_incomplete_csv_is_rejected(self):
        config = {'id_column': 'id', 'value_column': 'value', 'percent_column': 'percent',
                  'total_label': 'Total', 'expected_ids': ['A', 'B'],
                  'value_resolution_mtco2e': '1', 'percent_resolution_points': '0.1'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            for content in (
                'id,value,value,percent\nTotal,10,10,100\nA,10,10,100\n',
                'id,value,percent\nTotal,10,100\nA,10,100\n',
                'id,value,percent\nTotal,10,100\nA,10,100\nA,0,0\n',
                'id,value,percent\nTotal,10,100\nA,NaN,100\nB,0,0\n',
                'id,value,percent\nTotal,10,100\nA,10.5,100\nB,0,0\n',
                'id,value,percent\nTotal,10,100\nA,10,99.95\nB,0,0\n',
            ):
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_table(path, config)

    def test_numeric_grammar_and_field_unit(self):
        for bad in ('1,2', '12,34', '50%', '1e3', '=1+2', '', '-1'):
            with self.assertRaises(ValueError):
                decimal(bad, 'emissions')
        self.assertEqual(decimal('1,234', 'emissions'), 1234)
        self.assertEqual(decimal('50.0%', 'share', percent=True), 50)

    def test_nonfinite_reporting_precision_rejected(self):
        config = {'id_column': 'id', 'value_column': 'value', 'percent_column': 'percent',
                  'total_label': 'Total', 'value_resolution_mtco2e': 'Infinity',
                  'percent_resolution_points': '0.1'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            path.write_text('id,value,percent\nTotal,10,100\nA,10,100\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_table(path, config)


if __name__ == '__main__':
    unittest.main()
