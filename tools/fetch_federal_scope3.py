"""Fetch the four EDGI Zenodo CSVs required by federal_scope3_mapping.json."""
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'output/practical-evaluation/federal-agencies'
RECORD = 'https://zenodo.org/records/19476276/files/'
FILES = {
    '2022-agency.csv': ('2022_Data Table — L1 Agency.csv', '6aa5d32d6e77e764af9e89cf3511c7433fe0bc5ddb7fbbb7965048a93f1741cf'),
    '2022-category.csv': ('2022_Data Table — L1 Category.csv', 'efe0cf3fcd64de1fe75d39bbef60d2a5948fb093e041949fb97dff487a379b7b'),
    '2023-agency.csv': ('2023_Data Table — L1 Agency.csv', '555c94f246bbf0e84864470ed681d8a644fa97e28907a188dd4a69d47f6a395f'),
    '2023-category.csv': ('2023_Data Table — L1 Category.csv', 'b0eeb103b59c57ea88d0847bd4a7791fc26f784ed518f5f5063c8a6c87efa688'),
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for local, (remote, expected) in FILES.items():
        target = DEST / local
        if target.exists():
            if sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Existing file hash differs: {target}')
            continue
        with urlopen(RECORD + quote(remote) + '?download=1', timeout=30) as response:
            data = response.read()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f'Download hash differs: {remote}')
        with target.open('xb') as stream:
            stream.write(data)
    print(f'Four verified CSV files in {DEST}')


if __name__ == '__main__':
    main()
