import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import random
import argparse
import os
import hashlib
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shared
import emission_factors
import waste_kb

# Reconfigure stdout/stderr to utf-8 to avoid UnicodeEncodeError on Windows console
if sys.platform.startswith('win'):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            reconfigure(encoding='utf-8')


# ── LLM CONFIGURATION ───────────────────────────────────────────────────────
# Model name served by Ollama. Set to a real model (e.g. "gemma4:e4b", "llama3")
# to enable intelligent waste-code mapping. When set to "placeholder" or if the
# Ollama server is unreachable, the script falls back to deterministic MD5 hashing.
OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 180  # seconds per request

# How many waste codes from the CSV catalog to include in each LLM prompt.
# Increase for richer context (up to the full catalog), decrease for speed.
# Set to None to send ALL codes.
LLM_WASTE_CATALOG_SIZE = 50

# ── THE ERROR CATALOGUE ─────────────────────────────────────────────────────
# The 14 ways this generator corrupts a cell. Defined here rather than inside the
# injection loop so `--error-types` can be validated at parse time and listed with
# `--list-error-types`. step_2 must never import these: knowing which corruptions
# exist is exactly the ground truth it is walled off from (Handover.md §15).
ERROR_TYPES = {
    'fat_finger':          'a digit typed twice or a neighbouring key hit, giving roughly 10x or 0.1x',
    'dropped_zero':        'a trailing zero lost, giving 0.1x',
    'unit_conversion':     'tonnes entered where kilograms were wanted, or the reverse',
    'keyboard_mistype':    'two digits transposed',
    'true_random_error':   'the value replaced with an unrelated one',
    'scale_up_1000':       'grams read as kilograms, giving 1000x',
    'lbs_to_kg_confusion': 'pounds entered in a kilogram column',
    'currency_confusion':  'one currency entered in another currency column',
    'repeated_digits':     'a digit accidentally repeated',
    'off_by_one_digit':    'a single digit wrong',
    'random_noise_high':   'the value inflated by a small random amount',
    'random_noise_low':    'the value deflated by a small random amount',
    'accidental_zero':     'the cell zeroed, as if the stream were never reported',
    'negative_value':      'the sign flipped',
}

# What `--error-types random` selects: the seven that a careless typist produces.
# The other seven need a unit or currency mix-up, which is a different mistake.
RANDOM_ERROR_TYPES = ['fat_finger', 'dropped_zero', 'unit_conversion', 'keyboard_mistype',
                      'true_random_error', 'repeated_digits', 'off_by_one_digit']


def _error_rate(text):
    """A fraction of rows, so it has to sit in [0, 1]. Rejecting 30 for "30 percent"
    matters: `int(len(df) * 30)` reaches numpy as an impossible sample size and fails
    with a message about the sample being larger than the population, which sends you
    looking in the wrong place."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"{value} is out of range. Give a fraction between 0 and 1, so 0.30 for 30 percent")
    return value


def _error_type_list(text):
    """`all`, `none`, `random`, or a comma-separated list of names from ERROR_TYPES.

    An unknown name is an error rather than something to skip. The previous version
    filtered silently and fell back to fat_finger when nothing survived, so a typo
    produced a full run against a corruption set nobody asked for and said nothing."""
    key = text.strip().lower()
    if key in ('all', 'none', 'random', ''):
        return key or 'none'
    names = [n.strip() for n in text.split(',') if n.strip()]
    unknown = [n for n in names if n not in ERROR_TYPES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown error type(s): " + ", ".join(repr(n) for n in unknown) +
            "\nvalid names: " + ", ".join(sorted(ERROR_TYPES)) +
            "\nor use 'all', 'random', 'none'. Run --list-error-types for descriptions.")
    return ",".join(names)


class _ListErrorTypes(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        width = max(len(n) for n in ERROR_TYPES)
        print("Error types this generator can inject:\n")
        for name, description in ERROR_TYPES.items():
            marker = "*" if name in RANDOM_ERROR_TYPES else " "
            print(f"  {marker} {name:<{width}}  {description}")
        print("\n  * included by --error-types random (the default)")
        print("\n  --error-types all           every one of the 14")
        print("  --error-types none          clean data, no injection")
        print("  --error-types fat_finger,dropped_zero    just those two")
        parser.exit()


# ── CLI ARGUMENT PARSING ────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description='Multi-Product Scope 3 Data Generator (1 Company per Batch)')
_parser.add_argument('--list-error-types', nargs=0, action=_ListErrorTypes,
                     help='Print the 14 error types with descriptions and exit')
_parser.add_argument('--companies', type=int, default=1, help='Number of companies (batches) to generate')
_parser.add_argument('--products', type=int, default=100, help='Number of unique products per company')
_parser.add_argument('--llm-catalog-size', type=int, default=None,
                     help='Override LLM_WASTE_CATALOG_SIZE (number of waste codes per LLM prompt). '
                          'Use 0 or negative for ALL codes.')
_parser.add_argument('--ollama-model', type=str, default="placeholder",
                     help='Ollama model name to use (default: placeholder). Use "placeholder" to bypass LLM.')
_parser.add_argument('--seed', type=int, default=None, help='Seed for random number generators')
_parser.add_argument('--error-types', type=_error_type_list, default="random",
                     help='Comma-separated error type names, or "all", "random", "none". '
                          'See --list-error-types')
_parser.add_argument('--error-rate', type=_error_rate, default=0.15,
                     help='Fraction of rows to corrupt, between 0 and 1. 0.30 means 30 percent')
_parser.add_argument('--output-dir', type=str, default=None,
                     help='Use this project directory instead of allocating project NN')

NUM_COMPANIES = 1
NUM_PRODUCTS = 100
BASE_SEED = None
_args = None

# ── LOAD WASTE CODES FROM CSV ───────────────────────────────────────────────
try:
    waste_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        'waste_codes.csv'))
    # Clean the codes (remove spaces, drop NaNs from trailing empty commas)
    raw_codes = [str(x) for x in waste_df['Waste Code'].dropna()]
    THAI_WASTE_CODES_LIST = sorted(list(set([
        c.replace(' ', '').strip() for c in raw_codes if c.strip() and c != 'nan'
    ])))
    # Build a lookup of code -> description for the LLM prompt
    _waste_desc_map = {}
    for _, wrow in waste_df.dropna(subset=['Waste Code']).iterrows():
        code_clean = str(wrow['Waste Code']).replace(' ', '').strip()
        if code_clean and code_clean != 'nan':
            _waste_desc_map[code_clean] = str(wrow.get('Description', '')).strip()
    if __name__ == '__main__':
        print(f"Successfully loaded {len(THAI_WASTE_CODES_LIST)} waste codes from 'waste_codes.csv'.")
except FileNotFoundError:
    THAI_WASTE_CODES_LIST, _waste_desc_map = [], {}

# ── DEFINITIONS ─────────────────────────────────────────────────────────────
# RAW_MATERIALS, REGIONS and get_deterministic_ef live in emission_factors.py, which
# step_2 imports too. They used to be duplicated verbatim in both files; any drift
# between the copies breaks every emissions identity. See progress.md 0.3 and 10.5.
RAW_MATERIALS = emission_factors.RAW_MATERIALS
REGIONS = emission_factors.REGIONS                  # Thailand pinned to TGO 0.4750
get_deterministic_ef = emission_factors.get_deterministic_ef

SECTORS = ['Manufacturing', 'Technology', 'Consumer Goods', 'Energy', 'Healthcare']

# ── LLM WASTE MAPPING ───────────────────────────────────────────────────────
MATERIAL_KEYWORDS = {
    'Steel': ['steel', 'iron', 'metal', 'ferrous', 'machining', 'shaving', 'scrap'],
    'Aluminum': ['aluminum', 'aluminium', 'metal', 'non-ferrous', 'scrap', 'shaving'],
    'Copper': ['copper', 'metal', 'non-ferrous', 'wire', 'scrap', 'cable'],
    'Zinc': ['zinc', 'metal', 'non-ferrous', 'galvaniz'],
    'Brass': ['brass', 'metal', 'non-ferrous', 'alloy'],
    'Titanium': ['titanium', 'metal', 'non-ferrous', 'alloy'],
    'Nickel': ['nickel', 'metal', 'non-ferrous', 'alloy'],
    'Iron_Ore': ['iron', 'ore', 'mineral', 'mine', 'excavation'],
    'PET': ['plastic', 'pet', 'polyethylene', 'terephthalate', 'bottle', 'packaging'],
    'HDPE': ['plastic', 'hdpe', 'polyethylene', 'packaging'],
    'PVC': ['plastic', 'pvc', 'vinyl', 'pipe'],
    'LDPE': ['plastic', 'ldpe', 'polyethylene', 'packaging', 'film'],
    'PP': ['plastic', 'pp', 'polypropylene'],
    'PS': ['plastic', 'ps', 'polystyrene', 'foam'],
    'Polyurethane': ['plastic', 'polyurethane', 'foam', 'urethane'],
    'Nylon': ['plastic', 'nylon', 'polyamide', 'synthetic', 'fiber'],
    'Resin': ['resin', 'epoxy', 'polymer'],
    'Rubber': ['rubber', 'elastomer', 'tyre', 'tire'],
    'Silicone': ['silicone', 'silicon', 'sealant'],
    'Sulfuric_Acid': ['acid', 'sulfuric', 'chemical', 'ph', 'electrolyte'],
    'Sodium_Hydroxide': ['hydroxide', 'sodium', 'alkali', 'chemical', 'caustic', 'base'],
    'Ammonia': ['ammonia', 'chemical', 'nitrogen'],
    'Chlorine': ['chlorine', 'halogen', 'chemical', 'disinfect'],
    'Solvents_Organic': ['solvent', 'organic', 'degreas', 'thin', 'chemical'],
    'Solvents_Aqueous': ['solvent', 'aqueous', 'water', 'chemical'],
    'Catalyst_Precious': ['catalyst', 'platinum', 'palladium', 'gold', 'precious', 'metal'],
    'Catalyst_Base': ['catalyst', 'metal', 'spent'],
    'Paints_Coatings': ['paint', 'coating', 'varnish', 'lacquer', 'solvent'],
    'Adhesives': ['adhesive', 'glue', 'sealant', 'binder'],
    'Dyes_Pigments': ['dye', 'pigment', 'color', 'ink', 'paint'],
    'Silicon_Wafers': ['silicon', 'wafer', 'semiconductor', 'electronic', 'microelectronic'],
    'PCB_Boards': ['pcb', 'board', 'circuit', 'electronic', 'solder'],
    'Semiconductors': ['semiconductor', 'electronic', 'chip', 'component'],
    'Lithium': ['lithium', 'battery', 'cell', 'metal'],
    'Cobalt': ['cobalt', 'metal', 'battery'],
    'Rare_Earth_Elements': ['rare earth', 'element', 'metal', 'magnet'],
    'Cardboard': ['cardboard', 'paper', 'packaging', 'box'],
    'Paper': ['paper', 'packaging', 'shredded', 'office'],
    'Wood_Pallets': ['wood', 'pallet', 'timber', 'crate', 'packaging'],
    'Glass': ['glass', 'cullet', 'packaging'],
    'Shrink_Wrap': ['plastic', 'wrap', 'packaging', 'film'],
    'Textiles_Cotton': ['textile', 'cotton', 'fabric', 'fiber', 'rag', 'cloth'],
    'Textiles_Synthetic': ['textile', 'synthetic', 'fabric', 'fiber', 'polyester', 'nylon'],
    'Lubricating_Oils': ['oil', 'lubricant', 'engine', 'hydraulic', 'gear'],
    'Ceramics': ['ceramic', 'clay', 'brick', 'porcelain'],
}

# Cache for LLM responses — keyed by frozenset of material names per product
_LLM_WASTE_CACHE: dict = {}
_LLM_AVAILABLE: bool | None = None  # None = not tested yet


def _build_waste_catalog_subset(material_list: list[str]) -> list[tuple[str, str]]:
    """Build a prioritized subset of waste codes relevant to the given materials.

    Returns a list of (code, description) tuples, ranked by keyword relevance,
    and sized to LLM_WASTE_CATALOG_SIZE (or all codes if that is None).
    """
    # Collect all keywords for the given materials
    keywords = set()
    for m in material_list:
        keywords.add(m.lower())
        # Add sub-words if material contains underscores (e.g. Sulfuric_Acid -> sulfuric, acid)
        if '_' in m:
            for part in m.split('_'):
                keywords.add(part.lower())
        # Add keywords from our mapped dictionary
        if m in MATERIAL_KEYWORDS:
            for kw in MATERIAL_KEYWORDS[m]:
                keywords.add(kw.lower())

    scored_codes = []
    for code in THAI_WASTE_CODES_LIST:
        desc = _waste_desc_map.get(code, '').lower()
        score = 0
        for kw in keywords:
            if kw in desc:
                score += 1
        scored_codes.append((score, code, _waste_desc_map.get(code, '')))

    # Sort descending by score, then ascending by code for stability
    scored_codes.sort(key=lambda x: (-x[0], x[1]))

    # Extract just the (code, desc) tuples
    catalog_subset = [(code, desc) for _, code, desc in scored_codes]

    if LLM_WASTE_CATALOG_SIZE is not None:
        target = max(LLM_WASTE_CATALOG_SIZE, 10)
        return catalog_subset[:target]
    else:
        return catalog_subset


def _query_ollama(material_list: list[str], catalog_subset: list[tuple[str, str]]) -> list[str] | None:
    """Call the local Ollama API to map materials to waste codes.

    Returns a list of 2-4 validated waste code strings, or None on failure.
    """
    global _LLM_AVAILABLE

    # Fast-skip if we already know Ollama is down
    if _LLM_AVAILABLE is False:
        return None
    if OLLAMA_MODEL == "placeholder":
        _LLM_AVAILABLE = False
        return None

    catalog_text = "\n".join(f"  {code}: {desc}" for code, desc in catalog_subset[:200])

    prompt = (
        f"You are an industrial waste classification expert. "
        f"A manufacturing product uses these raw materials: {', '.join(material_list)}.\n\n"
        f"From the following waste code catalog, select exactly 2 to 4 waste codes that "
        f"would MOST LOGICALLY be generated during the manufacturing process using these materials. "
        f"Consider machining residues, chemical byproducts, packaging waste, and treatment sludges.\n\n"
        f"WASTE CODE CATALOG:\n{catalog_text}\n\n"
        f"Respond with ONLY a JSON array of waste code strings. Example: [\"120101\", \"150102\", \"070299\"]\n"
        f"Do NOT include any other text."
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 100}
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode('utf-8'))
            raw_text = body.get('response', '').strip()

            # Try to extract a JSON array from the response
            # Handle cases where LLM wraps in markdown code fences
            if '```' in raw_text:
                raw_text = raw_text.split('```')[1]
                if raw_text.startswith('json'):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            codes = json.loads(raw_text)
            if not isinstance(codes, list):
                return None

            # Validate: must be real codes from our catalog
            valid = [c.replace(' ', '').strip() for c in codes if isinstance(c, str)]
            valid = [c for c in valid if c in set(THAI_WASTE_CODES_LIST)]

            if 2 <= len(valid) <= 4:
                _LLM_AVAILABLE = True
                return valid
            elif len(valid) > 4:
                _LLM_AVAILABLE = True
                return valid[:4]
            elif len(valid) == 1:
                _LLM_AVAILABLE = True
                return valid  # accept 1 if that's all we got
            else:
                return None

    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        _LLM_AVAILABLE = False
        print(f"  [WARNING] Ollama not reachable at {OLLAMA_URL}. Using deterministic fallback for waste mapping.")
        return None
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        # LLM responded but output was unparseable
        return None


def _md5_fallback_waste_codes(material_list: list[str], k_min: int = 2, k_max: int = 4) -> list[str]:
    """Deterministic fallback: hash material names to pick waste codes from the prioritized subset."""
    catalog_subset = _build_waste_catalog_subset(material_list)
    top_candidates = [code for code, desc in catalog_subset[:15]]

    key = "|".join(sorted(material_list))
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    rng = random.Random(h)
    k = rng.randint(k_min, k_max)

    if len(top_candidates) >= k:
        return rng.sample(top_candidates, k=k)
    else:
        return rng.sample(THAI_WASTE_CODES_LIST, k=k)


def get_waste_codes_for_product(material_list: list[str]) -> list[str]:
    """Get waste codes for a product's material list, via LLM or fallback.

    Results are cached per unique material combination.
    """
    cache_key = tuple(sorted(material_list))
    if cache_key in _LLM_WASTE_CACHE:
        return _LLM_WASTE_CACHE[cache_key]

    catalog_subset = _build_waste_catalog_subset(material_list)
    codes = _query_ollama(material_list, catalog_subset)

    if codes is None:
        codes = _md5_fallback_waste_codes(material_list)

    _LLM_WASTE_CACHE[cache_key] = codes
    return codes


# ── EMISSION FACTORS ────────────────────────────────────────────────────────
# Built by the shared module so step_2 gets bit-identical values. TGO published factors
# where they exist (16/45 materials), deterministic hash elsewhere, logged either way.
#
# The four hardcoded overrides that used to sit here are gone: Steel 1.8 and Aluminum 8.5
# both disagreed with TGO (progress.md 10.6), Silicon_Wafers 12.0 had no source at all and
# now falls back to hash, and 'Plastic' was a NO-OP - it is not in RAW_MATERIALS, so that
# entry was never read by anything.
EMISSION_FACTORS = emission_factors.build_emission_factors(THAI_WASTE_CODES_LIST)
THAI_WASTE_CATALOG = EMISSION_FACTORS['waste']
if __name__ == '__main__':
    print(emission_factors.describe_coverage())

# ── CATEGORIES 1, 5, 9, 12 — shared with step_2's fallback dict ────────────
# Single definition now lives in emission_factors.py (was hand-duplicated, flattened,
# into step_2's REG_EF_MAP fallback dict). Re-exported here under the original names
# so every call site below is unchanged. See emission_factors.py and
# reviews/.../2026-08-14-code-audit.md (C1b).
INDIRECT_SPEND_CATEGORIES = emission_factors.INDIRECT_SPEND_CATEGORIES  # CATEGORY 1
WASTE_TREATMENT_METHODS = emission_factors.WASTE_TREATMENT_METHODS      # CATEGORY 5
_TREATMENT_NAMES = list(WASTE_TREATMENT_METHODS.keys())
_TREATMENT_WEIGHTS = [v['weight'] for v in WASTE_TREATMENT_METHODS.values()]
TRANSPORT_MODES = emission_factors.TRANSPORT_MODES                      # CATEGORY 9
EOL_PROFILES = emission_factors.EOL_PROFILES                            # CATEGORY 12
EOL_EF = emission_factors.EOL_EF                                        # CATEGORY 12

# ── HELPER ──────────────────────────────────────────────────────────────────
def transpose_digits(val: float) -> float:
    """Transpose two adjacent digits of a number to simulate keyboard mistypes."""
    val_str = f"{val:.4f}"
    digits = [c for c in val_str if c.isdigit()]
    if len(digits) >= 2:
        idx = random.randint(0, len(digits) - 2)
        digits[idx], digits[idx+1] = digits[idx+1], digits[idx]
        new_str = ""
        digit_idx = 0
        for c in val_str:
            if c.isdigit():
                new_str += digits[digit_idx]
                digit_idx += 1
            else:
                new_str += c
        try:
            return float(new_str)
        except ValueError:
            pass
    return val * 1.34  # fallback

def true_random_error(val: float) -> float:
    """Enigma scrambler: 3 passes of random ops using value as seed, plus digit scrambling."""
    try:
        current_val = float(val)
        for _ in range(3):
            rng = random.Random(current_val)
            op = rng.choice(['add', 'sub', 'mul', 'div'])
            rand_var = rng.uniform(0.1, 100.0)
            if op == 'add': current_val += rand_var
            elif op == 'sub': current_val -= rand_var
            elif op == 'mul': current_val *= rand_var
            elif op == 'div' and rand_var != 0: current_val /= rand_var

        val_str = f"{current_val:.4f}"
        peg_board = list("0123456789")
        random.Random(current_val).shuffle(peg_board)
        mapped = {str(i): peg_board[i] for i in range(10)}
        wheel_pos = random.Random(current_val).randint(0, 9)

        scrambled_str = ""
        for c in val_str:
            if c.isdigit():
                mapped_digit = int(mapped[c])
                shifted = (mapped_digit + wheel_pos) % 10
                scrambled_str += str(shifted)
            else:
                scrambled_str += c
        return float(scrambled_str)
    except Exception:
        return val * 1.55

def off_by_one_digit(val: float) -> float:
    """Add or subtract 1 from any random digit in the number."""
    val_str = f"{val:.4f}"
    digits = [(i, c) for i, c in enumerate(val_str) if c.isdigit()]
    if digits:
        idx, char = random.choice(digits)
        digit_val = int(char)
        new_digit = (digit_val + random.choice([-1, 1])) % 10
        new_str = val_str[:idx] + str(new_digit) + val_str[idx+1:]
        try:
            return float(new_str)
        except:
            pass
    return val

def repeated_digits_adv(val: float) -> float:
    """Either repeat a single digit (45 -> 445) or repeat the whole number (4545)."""
    val_str = str(val).rstrip('0').rstrip('.')
    if random.random() < 0.5:
        digits = [(i, c) for i, c in enumerate(val_str) if c.isdigit()]
        if digits:
            idx, char = random.choice(digits)
            new_str = val_str[:idx] + char + char + val_str[idx+1:]
            try:
                return float(new_str)
            except:
                pass
    else:
        digits_only = "".join(c for c in val_str if c.isdigit())
        if digits_only:
            new_digits = digits_only + digits_only
            dec_idx = val_str.find('.')
            if dec_idx != -1:
                decimals_count = len(val_str) - dec_idx - 1
                if decimals_count < len(new_digits):
                    new_str = new_digits[:-decimals_count] + '.' + new_digits[-decimals_count:]
                else:
                    new_str = new_digits
            else:
                new_str = new_digits
            try:
                return float(new_str)
            except:
                pass
    return val * 1.11

# ── B-2B / B-3 / INPUT FATE / TREATMENT FRACTIONS ───────────────────────────
# progress.md 0.9 and docs/plans/2026-08-13-phase-b-step1-rewrite.md.

_KB_CODES = set(THAI_WASTE_CODES_LIST)
_KB_MATS = set(RAW_MATERIALS)
_KB_ABSENCE = waste_kb.absence_links(_KB_CODES, _KB_MATS, role="general_buyer")
_KB_MAGNITUDE = waste_kb.magnitude_links(_KB_CODES, _KB_MATS, role="general_buyer")
# material -> [link], for the links that carry a usable ratio
_KB_BY_MAT = {}
for _L in _KB_MAGNITUDE:
    _KB_BY_MAT.setdefault(_L['material'], []).append(_L)

# 5f: links the ABSENCE arm checks but the magnitude arm cannot use -- mandatory codes
# whose research row carries no numeric ratio, or a basis_qualifier that makes the
# denominator wrong. Until 2026-08-13 these were never generated, so the material was
# present, the code was always absent, and step_2's absence arm exempted every candidate
# via `occurrence > 0`. That is why it had flagged 0 cells on every dataset ever
# (progress.md 0.11). A mandatory code with no published coefficient is still manifested
# in reality, so it gets generated -- just without pretending to a sourced magnitude.
_MAG_KEYS = {(_L['material'], _L['code']) for _L in _KB_MAGNITUDE}
_MAG_CODES = {_L['code'] for _L in _KB_MAGNITUDE}
_IDENTITY_CODES = {_L['code'] for _L in _KB_MAGNITUDE
                   if _L.get('source_kind') == 'physical_identity'}
_KB_ABS_ONLY_BY_MAT = {}
for _L in _KB_ABSENCE:
    if (_L['material'], _L['code']) not in _MAG_KEYS:
        _KB_ABS_ONLY_BY_MAT.setdefault(_L['material'], []).append(_L)
if __name__ == '__main__':
    print(f"[KB] absence-only links generated with a nominal ratio: "
          f"{sum(len(v) for v in _KB_ABS_ONLY_BY_MAT.values())} over "
          f"{len({L['code'] for v in _KB_ABS_ONLY_BY_MAT.values() for L in v})} codes")


def _dedicated_rng(*parts) -> random.Random:
    """A seeded RNG that does NOT consume the global `random` stream.

    Every generator change so far has perturbed the global stream and thereby changed
    which errors the injector produces -- that is why `--seed 1690137184` gave project 12
    zero transport_suppression where project 10 had 240, and why p12-vs-p10 was never a
    valid before/after (progress.md 0.10). Keeping new draws off the global stream is what
    lets a seed keep reproducing the same companies and the same injected errors across
    generator changes, so `step_1` work stays measurable.
    """
    return random.Random(int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16))


def _nominal_ratio(company_id, product_id, material, code) -> float:
    """Plausible magnitude for a mandatory stream with no published coefficient.

    Log-uniform across 1e-4 .. 1e-2 kg per kg of driver, which spans the low end of the
    sourced coefficients without claiming to be one. step_2's magnitude arm excludes these
    links by construction, so this number is never tested against a published range -- it
    exists only so the stream is present and its ABSENCE becomes meaningful.
    """
    r = _dedicated_rng(company_id, product_id, material, code, "nominal")
    return 10.0 ** r.uniform(-4.0, -2.0)
if __name__ == '__main__':
    print(waste_kb.describe_coverage(_KB_CODES, _KB_MATS, role="general_buyer"))

# Disposal routes the research pins. progress.md 10.8 decision 3: a sold or recirculated
# by-product is a TREATMENT SPLIT, not an absence - the stream is still generated and
# manifested, so `obligation` stays mandatory and the nuance lives here instead.
SOURCED_TREATMENT = {
    '100202': {'Recycling': 0.95, 'Landfill': 0.05},   # BF slag - worldsteel: near-100% into cement/aggregate
    '100210': {'Recycling': 0.90, 'Landfill': 0.10},   # mill scale - ~90% recirculated to sinter
    '110202': {'Landfill': 0.85, 'Recycling': 0.15},   # zinc hydrometallurgy sludge - predominantly landfilled
}

# Materials whose fate is not "mostly becomes product".
_PACKAGING_IN = {'Cardboard', 'Paper', 'Shrink_Wrap'}       # discarded on receipt, ratio-1.0 rows
_RETURNABLE_CAPABLE = {'Glass', 'Wood_Pallets'}             # decision 2: may leave INTACT


def _treatment_fractions(waste_code: str, company_id: str) -> dict:
    """Fraction of a code's mass going to each of the four methods. Sums to 1.

    Replaces the old md5 -> ONE winner, which made every code 100% landfilled or 100%
    recycled AND identical across every company (progress.md 10.9). Sourced routes are
    fixed; everything else gets a company-specific Dirichlet-ish draw around the global
    weights, so cross-company variation is real but reproducible from the seed.
    """
    if waste_code in SOURCED_TREATMENT:
        return dict(SOURCED_TREATMENT[waste_code])
    rng = random.Random(int(hashlib.md5(f"{company_id}|{waste_code}".encode()).hexdigest(), 16))
    draws = [rng.gammavariate(max(w * 6.0, 0.35), 1.0) for w in _TREATMENT_WEIGHTS]
    tot = sum(draws) or 1.0
    return {name: d / tot for name, d in zip(_TREATMENT_NAMES, draws)}


def _allocate_exact(total_kg: float, fractions: dict) -> dict:
    """Split `total_kg` across methods with the final part as the exact residual.

    Identity 1 (`sum(c5_*_kg) == sum(waste_*_kg)`) is checked by step_2 with a residual
    threshold near float dust. Identity waste can carry more than two decimals because it
    is consumption multiplied by a written fate, so cent-rounding would destroy either
    the packaging identity or the treatment identity.
    """
    items = list(fractions.items())
    parts, remaining = {}, float(total_kg)
    for index, (method, fraction) in enumerate(items):
        part = remaining if index == len(items) - 1 else float(total_kg) * fraction
        parts[method] = part
        remaining -= part
    return parts


def _unwind_c5(df_messy, mask, waste_code: str, mass_removed, company_id: str):
    """Take `mass_removed` back out of the c5 buckets in the SAME proportions the
    generator put it in, and return the treatment-emissions delta (mtCO2e) to subtract.

    The suppression injectors reduce `waste_{code}_kg`; the c5 columns have to follow, or
    identity 1 (`sum(c5_*) == sum(waste_*)`) fires on the injected rows for the wrong
    reason. A suppression is meant to be caught by the mass-balance and greenwash
    detectors, not by an accounting identity the injector itself broke.

    Now that treatment is a fraction across four methods rather than one hash-picked
    winner, the unwind has to be proportional too - subtracting the whole mass from a
    single bucket would drive it negative and break the identity in the other direction.
    """
    fracs = _treatment_fractions(waste_code, company_id)
    emissions_delta = 0.0
    for method, f in fracs.items():
        part = mass_removed * f
        df_messy.loc[mask, f'c5_{method.lower()}_kg'] -= part
        emissions_delta = emissions_delta + part * WASTE_TREATMENT_METHODS[method]['ef_multiplier']
    return emissions_delta / 1000.0


def _inject_random_errors(df_messy, target_cols, err_idx, active_error_types, error_summary):
    """Apply one independent accidental error per selected row."""
    for idx in err_idx:
        row_vals = df_messy.loc[idx, target_cols]
        active_cols = [c for c in target_cols
                       if isinstance(row_vals[c], (int, float)) and row_vals[c] > 0]
        if not active_cols:
            continue

        col = random.choice(active_cols)
        current_val = df_messy.at[idx, col]
        err_type = random.choice(active_error_types)
        error_summary[err_type] += 1

        if err_type == 'fat_finger':
            df_messy.at[idx, col] = current_val * 10
        elif err_type == 'dropped_zero':
            df_messy.at[idx, col] = current_val / 10
        elif err_type == 'unit_conversion':
            df_messy.at[idx, col] = current_val / 1000
        elif err_type == 'keyboard_mistype':
            df_messy.at[idx, col] = transpose_digits(current_val)
        elif err_type == 'true_random_error':
            df_messy.at[idx, col] = true_random_error(current_val)
        elif err_type == 'scale_up_1000':
            df_messy.at[idx, col] = current_val * 1000
        elif err_type == 'lbs_to_kg_confusion':
            df_messy.at[idx, col] = current_val * 2.20462
        elif err_type == 'currency_confusion':
            df_messy.at[idx, col] = current_val / 35.0
        elif err_type == 'repeated_digits':
            df_messy.at[idx, col] = repeated_digits_adv(current_val)
        elif err_type == 'off_by_one_digit':
            df_messy.at[idx, col] = off_by_one_digit(current_val)
        elif err_type == 'random_noise_high':
            df_messy.at[idx, col] = current_val * random.uniform(1.5, 2.0)
        elif err_type == 'random_noise_low':
            df_messy.at[idx, col] = current_val * random.uniform(0.1, 0.5)
        elif err_type == 'accidental_zero':
            df_messy.at[idx, col] = 0.0
        elif err_type == 'negative_value':
            df_messy.at[idx, col] = current_val * -1.0
        if df_messy.at[idx, col] != current_val:
            error_summary['random_error_rows_changed'] += 1


def _draw_input_fates(material: str, rng: random.Random) -> dict:
    """Fraction of an input that becomes product / leaves as waste / leaves intact.

    progress.md 10.8 decisions 1 and 2. Sums to 1, which gives step_2 a new exact
    identity across all 45 materials.
    """
    if material == 'Dyes_Pigments':
        # Decision 1: the fate split IS the fixation rate - what bonds to the fabric
        # versus what leaves unfixed. Textiles BREF Table 2.16 ranges are wide by dye
        # class, so this spans them rather than pinning one.
        fixation = rng.uniform(0.55, 0.95)
        return {'product': fixation, 'waste': 1.0 - fixation, 'intact': 0.0}
    if material in _PACKAGING_IN:
        # Inbound packaging is discarded on receipt; at steady state inbound mass equals
        # discarded mass. That is the physical argument that survived the failed Article
        # 6a citation (progress.md 10.9).
        w = rng.uniform(0.90, 1.0)
        return {'product': 0.0, 'waste': w, 'intact': 1.0 - w}
    if material in _RETURNABLE_CAPABLE:
        # Decision 2: a returnable/pooled system reports near-zero waste AND the schema
        # knows why, instead of the detector having to guess.
        if rng.random() < 0.5:
            i = rng.uniform(0.85, 0.98)
            return {'product': 0.0, 'waste': 1.0 - i, 'intact': i}
        w = rng.uniform(0.90, 1.0)
        return {'product': 0.0, 'waste': w, 'intact': 1.0 - w}
    y = rng.uniform(0.88, 0.99)                       # ordinary process yield
    return {'product': y, 'waste': 1.0 - y, 'intact': 0.0}


def _audited_identity_waste(row, links):
    """Aggregate exact identity waste from the fields that will be written."""
    frame = pd.DataFrame([row])
    totals = {}
    for link in links:
        contribution = waste_kb.driver_series(link, frame)
        if contribution is None or pd.isna(contribution.iloc[0]):
            raise ValueError(f"invalid audited identity driver for {link['material']}")
        totals[link['code']] = totals.get(link['code'], 0.0) + float(contribution.iloc[0])
    return totals


# One "project NN" folder per run - all companies generated in this invocation
# share it. Self-repairing: PROJECT_DIR is recreated by shared.ensure_dir() right
# before every write below, so a mid-run deletion just gets recreated in place.
def main():
    global _args, NUM_COMPANIES, NUM_PRODUCTS, OLLAMA_MODEL, LLM_WASTE_CATALOG_SIZE, BASE_SEED
    _args, _ = _parser.parse_known_args()
    NUM_COMPANIES, NUM_PRODUCTS = _args.companies, _args.products
    OLLAMA_MODEL = _args.ollama_model
    if _args.llm_catalog_size is not None:
        LLM_WASTE_CATALOG_SIZE = _args.llm_catalog_size if _args.llm_catalog_size > 0 else None
    BASE_SEED = (_args.seed if _args.seed is not None
                 else random.SystemRandom().randrange(2**31 - 1))
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    print(f"Run seed: {BASE_SEED}")
    if OLLAMA_MODEL != "placeholder":
        shared.ensure_ollama_running()
    print(f"Generating Deep-Dive Synthetic Data: {NUM_COMPANIES} Companies, {NUM_PRODUCTS} Products each...")
    if _args.output_dir:
        PROJECT_DIR = shared.prepare_fresh_output_dir(os.path.abspath(_args.output_dir))
        shared.ensure_dir(os.path.join(PROJECT_DIR, 'generated company'))
    else:
        PROJECT_DIR = shared.new_project_dir()
    GENERATED_DIR = os.path.join(PROJECT_DIR, 'generated company')
    expected_company_ids = [f"COMP_{i:03d}" for i in range(1, NUM_COMPANIES + 1)]
    completed_company_ids, pair_hashes = [], {}
    shared.write_generation_state(
        GENERATED_DIR, expected_company_ids, seed=BASE_SEED,
        requested_products=NUM_PRODUCTS)
    catalog_size_label = LLM_WASTE_CATALOG_SIZE if LLM_WASTE_CATALOG_SIZE else "ALL"
    print(f"Enforcing Sparse Matrix: {len(RAW_MATERIALS)} Materials + {len(THAI_WASTE_CODES_LIST)} Waste Codes (0.0 if unused).")
    print(f"LLM waste catalog size per prompt: {catalog_size_label} codes | Model: {OLLAMA_MODEL}")

    global_error_summary = {
        'seed': BASE_SEED, 'requested_error_rate': _args.error_rate,
        'requested_error_types': _args.error_types,
        'generated_rows': 0, 'requested_error_rows': 0, 'random_error_rows_changed': 0,
        'fat_finger': 0, 'dropped_zero': 0, 'unit_conversion': 0, 'keyboard_mistype': 0,
        'true_random_error': 0, 'scale_up_1000': 0, 'lbs_to_kg_confusion': 0, 'currency_confusion': 0,
        'repeated_digits': 0, 'off_by_one_digit': 0, 'random_noise_high': 0, 'random_noise_low': 0,
        'accidental_zero': 0, 'negative_value': 0,
        'intentional_zero_attempted': 0, 'intentional_zero': 0,
        'synthetic_fallback_waste_codes': 0,
        'waste_suppression': 0, 'transport_suppression': 0
    }

    # ── GENERATION LOOP ─────────────────────────────────────────────────────────
    for company_idx in range(1, NUM_COMPANIES + 1):
        company_id = f"COMP_{company_idx:03d}"
        company_region = random.choice(list(REGIONS.keys()))
        company_sector = random.choice(SECTORS)
        grid_ef = REGIONS[company_region]

        print(f"\nBuilding Profile for {company_id} (Region: {company_region}, Sector: {company_sector}, Grid EF: {grid_ef})...")

        np.random.seed(BASE_SEED + company_idx)
        random.seed(BASE_SEED + company_idx)

        # Company-level attributes (deterministic per seed)
        company_renewable_pct = round(random.uniform(0.0, 1.0), 4)

        # 1. Generate the "Formulas" (Recipes) for this company's products
        # --products is honored exactly (the old ±20% jitter made runs incomparable
        # and surprised users asking for a specific count).
        actual_products = NUM_PRODUCTS

        product_recipes = {}
        for p in range(1, actual_products + 1):
            product_id = f"PROD_{p:03d}"

            # Pick 3 to 12 random raw materials
            mats = random.sample(RAW_MATERIALS, k=random.randint(3, 12))
            mat_reqs = {}
            for mat in mats:
                m_idx = sum(ord(c) for c in mat)
                mat_reqs[mat] = round(0.1 + (m_idx % 100)/100.0 * 19.9, 2)

            # ── INPUT FATE (decisions 1 and 2) ──────────────────────────────
            # Drawn once per (company, product, material) so it is a stable property of the
            # production line, not monthly noise.
            fate_rng = random.Random(int(hashlib.md5(
                f"{company_id}|{product_id}|fate".encode()).hexdigest(), 16))
            fates = {mat: _draw_input_fates(mat, fate_rng) for mat in mats}

            # Product mass is now the yield term, not the whole input. Packaging received and
            # discarded, and returnable containers that leave intact, never became product.
            product_kg_per_unit = sum(mat_reqs[m] * fates[m]['product'] for m in mats)

            # ── B-2B: WASTE FROM STOICHIOMETRY, NOT A RANDOM DRAW ───────────
            # The codes a product reports are now the codes its MATERIALS actually produce,
            # via the same waste_kb module step_2 audits with. Before this, step_1 sampled
            # from all 857 with no reference to the recipe, so the KB's codes barely appeared
            # in the data and B-2A could judge under 7% of waste cells (progress.md 0.9).
            #
            # The per-(company, product) ratio is DRAWN from the source's published
            # [ratio_low, ratio_high] and never revealed; step_2 knows only the range. So the
            # detector can only catch errors that push outside the published band, and that
            # ceiling is real rather than designed away.
            ratio_rng = random.Random(int(hashlib.md5(
                f"{company_id}|{product_id}|stoich".encode()).hexdigest(), 16))
            waste_reqs, drawn_ratios, identity_links = {}, {}, []
            for mat in mats:
                for L in _KB_BY_MAT.get(mat, []):
                    # Band comes from waste_kb so the generator and the auditor agree on it.
                    # Point estimates are widened rather than generated exactly, which would
                    # hand step_2 a free win on those codes.
                    lo, _mid, hi = waste_kb.ratio_band(L)
                    if _mid is None:
                        continue
                    if L.get('source_kind') == 'physical_identity':
                        identity_links.append(L)
                        drawn_ratios[(mat, L['code'])] = 1.0
                        continue
                    r = ratio_rng.uniform(lo, hi)
                    driver = (mat_reqs[mat] if L['basis'] == 'per_kg_input'
                              else product_kg_per_unit)
                    if driver <= 0:
                        continue
                    waste_reqs[L['code']] = waste_reqs.get(L['code'], 0.0) + r * driver
                    drawn_ratios[(mat, L['code'])] = r

            # 5f: mandatory codes the absence arm checks but the magnitude arm cannot use.
            # Generated proportional to the same driver, so the input->waste relationship the
            # absence arm tests is structurally true, but with a nominal ratio rather than a
            # sourced one. Without these, occurrence is 0 on 18 columns and the absence arm is
            # permanently inert (progress.md 0.11).
            for mat in mats:
                for L in _KB_ABS_ONLY_BY_MAT.get(mat, []):
                    # Never add nominal mass to a code the MAGNITUDE arm also predicts. Two
                    # codes (070208, 070214) carry both a sourced link from one material and
                    # an absence-only link from another; adding unsourced mass on top of a
                    # sourced expectation makes step_2 under-predict and flag clean rows.
                    # Same generator/detector mismatch shape as 0.11, in the other direction.
                    if L['code'] in _MAG_CODES:
                        continue
                    driver = (mat_reqs[mat] * fates[mat]['waste']
                              if L.get('source_kind') == 'physical_identity'
                              else mat_reqs[mat] if L['basis'] == 'per_kg_input'
                              else product_kg_per_unit)
                    if driver <= 0:
                        continue
                    r = _nominal_ratio(company_id, product_id, mat, L['code'])
                    waste_reqs[L['code']] = waste_reqs.get(L['code'], 0.0) + r * driver

            # A real site also manifests codes no coefficient covers - 11 of 45 materials have
            # no stoichiometry rows at all. Keep the old mechanism for those so the schema does
            # not become trivially predictable, and so B-2A is measured against a world where
            # it does NOT explain everything.
            uncovered = [m for m in mats if m not in _KB_BY_MAT]
            if uncovered:
                for w in get_waste_codes_for_product(uncovered):
                    if w not in waste_reqs and w not in _IDENTITY_CODES:
                        w_idx = sum(ord(c) for c in w)
                        waste_reqs[w] = round(0.01 + (w_idx % 100) / 100.0 * 2.99, 3)
                        global_error_summary['synthetic_fallback_waste_codes'] += 1
            product_waste_codes = list(dict.fromkeys(
                [*waste_reqs, *(L['code'] for L in identity_links)]))

            # Throttled progress reporting (at most 20 prints per company)
            print_interval = max(1, actual_products // 20)
            if p % print_interval == 0 or p == actual_products:
                print(f"Mapping product {p}/{actual_products} for {company_id}...", flush=True)

            # Category 5 — a FRACTION across the four methods, not one hash-picked winner
            waste_treatments = {w: _treatment_fractions(w, company_id) for w in product_waste_codes}

            # ── B-3: opening stock per material ─────────────────────────────
            # Without a stock level the accumulation term has nowhere to live and no mass
            # balance in this system can ever close (progress.md 11 B-3).
            # Held as MONTHS OF COVER rather than an absolute mass, so stock scales with
            # throughput the way the rest of the schema does. An absolute figure would be
            # flat while consumption varies with production_units, and step_2's expectation
            # model - which works on per-unit intensities - would read that as a drifting
            # intensity and flag clean rows.
            stock_rng = random.Random(int(hashlib.md5(
                f"{company_id}|{product_id}|stock".encode()).hexdigest(), 16))
            stock_months_cover = {m: stock_rng.uniform(1.5, 6.0) for m in mats}

            util_reqs = {
                'Grid_Elec_kWh': round(random.uniform(5.0, 150.0), 1),
                'Non_Grid_Energy_MJ': round(random.uniform(0.0, 50.0), 1),
                'Water_Use_m3': round(random.uniform(0.1, 5.0), 2)
            }

            # Deterministic product price from recipe complexity
            product_unit_price = round(50 + len(mats) * 100 + sum(mat_reqs.values()) * 5, 2)

            # Category 9 — transport profile per product
            transport_mode = random.choice(list(TRANSPORT_MODES.keys()))
            dist_lo, dist_hi = TRANSPORT_MODES[transport_mode]['dist_range']
            transport_dist_km = round(random.uniform(dist_lo, dist_hi), 1)

            # Category 1 — indirect spend profile per product (subset of categories)
            num_spend_cats = random.randint(2, len(INDIRECT_SPEND_CATEGORIES))
            selected_spend_cats = random.sample(list(INDIRECT_SPEND_CATEGORIES.keys()), k=num_spend_cats)
            indirect_spend = {}
            for cat in selected_spend_cats:
                lo, hi = INDIRECT_SPEND_CATEGORIES[cat]['spend_range']
                indirect_spend[cat] = round(random.uniform(lo, hi), 2)

            product_recipes[product_id] = {
                'mats': mat_reqs,
                'fates': fates,
                'product_kg_per_unit': product_kg_per_unit,
                'stock_months_cover': stock_months_cover,
                'drawn_ratios': drawn_ratios,
                'identity_links': identity_links,
                'wastes': waste_reqs,
                'waste_treatments': waste_treatments,
                'utils': util_reqs,
                'base_volume': int(random.uniform(100, 5000)),
                'unit_price': product_unit_price,
                'transport_mode': transport_mode,
                'transport_dist_km': transport_dist_km,
                'indirect_spend': indirect_spend,
            }

        # 2. Simulate 12 Months of Production Data
        records = []
        for month in range(1, 13):
            seasonality = random.uniform(0.8, 1.2)

            for prod_id, recipe in product_recipes.items():
                prod_noise = random.uniform(0.9, 1.1)
                units = int(recipe['base_volume'] * seasonality * prod_noise)
                if units <= 0:
                    continue

                row = {
                    'company_id': company_id,
                    'region': company_region,
                    'sector': company_sector,
                    'reporting_month': month,
                    'product_id': prod_id,
                    'production_units': units,
                    # Placeholders — stamped with real values after the loop
                    'gen_total_revenue_usd': 0.0,
                    'headcount': 0,
                    'facility_sqft': 0,
                    'renewable_energy_pct': company_renewable_pct,
                }

                # --- EXHAUSTIVE SPARSE MATRIX INITIALIZATION ---
                # All waste codes and materials are forced into the DataFrame as 0.0
                for mat in RAW_MATERIALS:
                    m = mat.lower()
                    row[f'c1_{m}_kg'] = 0.0
                    # Input fate (decisions 1 and 2). The three sum to 1 for any material
                    # actually used, which is a new exact identity for step_2 across all 45.
                    row[f'c1_{m}_fate_product_pct'] = 0.0
                    row[f'c1_{m}_fate_waste_pct'] = 0.0
                    row[f'c1_{m}_fate_intact_pct'] = 0.0
                    # B-3 inventory. closing == opening + purchased - consumed, exactly.
                    row[f'c1_{m}_opening_kg'] = 0.0
                    row[f'c1_{m}_purchased_kg'] = 0.0
                    row[f'c1_{m}_closing_kg'] = 0.0
                for w in THAI_WASTE_CODES_LIST:
                    row[f'waste_{w}_kg'] = 0.0

                row['grid_elec_kwh'] = 0.0
                row['non_grid_energy_mj'] = 0.0
                row['water_use_m3'] = 0.0

                # Category 1 — Indirect spend sparse columns
                for cat in INDIRECT_SPEND_CATEGORIES:
                    row[f'c1_spend_{cat.lower()}_usd'] = 0.0
                row['c1_indirect_spend_emissions_mtco2'] = 0.0

                # Category 5 — Waste treatment sparse columns
                for method in WASTE_TREATMENT_METHODS:
                    row[f'c5_{method.lower()}_kg'] = 0.0
                row['c5_waste_treatment_emissions_mtco2'] = 0.0

                # Category 9 — Transport columns
                row['c9_transport_mode'] = ''
                row['c9_product_weight_tonnes'] = 0.0
                row['c9_distance_km'] = 0.0
                row['c9_tkm'] = 0.0
                row['c9_transport_emissions_mtco2'] = 0.0

                # Category 12 — End-of-life sparse columns
                for pathway in ['landfill', 'recycled', 'incinerated']:
                    row[f'c12_eol_{pathway}_kg'] = 0.0
                row['c12_eol_emissions_mtco2'] = 0.0

                # ── CATEGORY 1 — Physical materials (existing logic) ────────────
                supplier_emissions_kg = 0.0
                for mat, kg_per_unit in recipe['mats'].items():
                    m = mat.lower()
                    total_kg = units * kg_per_unit * random.uniform(0.98, 1.02)
                    consumed = round(total_kg, 2)
                    row[f'c1_{m}_kg'] = consumed
                    supplier_emissions_kg += total_kg * EMISSION_FACTORS['mats'][mat]

                    f = recipe['fates'][mat]
                    row[f'c1_{m}_fate_product_pct'] = round(f['product'], 4)
                    row[f'c1_{m}_fate_waste_pct'] = round(f['waste'], 4)
                    # The third is the exact residual, so the three always sum to 1.0 on the
                    # written row rather than to 0.9999 - step_2 checks this as an identity.
                    row[f'c1_{m}_fate_intact_pct'] = round(
                        1.0 - row[f'c1_{m}_fate_product_pct'] - row[f'c1_{m}_fate_waste_pct'], 4)

                    # B-3: closing == opening + purchased - consumed, exact to the cent.
                    opening = round(consumed * recipe['stock_months_cover'][mat]
                                    * random.uniform(0.90, 1.10), 2)
                    purchased = round(consumed * random.uniform(0.85, 1.15), 2)
                    row[f'c1_{m}_opening_kg'] = opening
                    row[f'c1_{m}_purchased_kg'] = purchased
                    row[f'c1_{m}_closing_kg'] = round(opening + purchased - consumed, 2)

                row['supplier_emissions_mtco2'] = round(supplier_emissions_kg / 1000, 4)

                # ── CATEGORY 1 — Indirect spend (EEIO) ─────────────────────────
                c1_indirect_emissions_kg = 0.0
                for cat, monthly_base in recipe['indirect_spend'].items():
                    monthly_spend = monthly_base * random.uniform(0.9, 1.1)
                    row[f'c1_spend_{cat.lower()}_usd'] = round(monthly_spend, 2)
                    c1_indirect_emissions_kg += monthly_spend * INDIRECT_SPEND_CATEGORIES[cat]['ef_per_usd']
                row['c1_indirect_spend_emissions_mtco2'] = round(c1_indirect_emissions_kg / 1000, 4)

                # ── CATEGORY 5 — Waste generation + treatment ──────────────────
                waste_emissions_kg = 0.0
                c5_treatment_emissions_kg = 0.0

                row_wastes = {}
                for w, kg_per_unit in recipe['wastes'].items():
                    # Dedicated RNG, not the global stream: the NUMBER of waste codes a
                    # product carries changes whenever the knowledge base or its filters
                    # change, so drawing this noise globally makes every later global draw --
                    # including the injector's -- shift with it. That is what destroyed
                    # seed comparability between projects 10 and 12. Keyed on the month so it
                    # still varies row to row.
                    total_waste = units * kg_per_unit * _dedicated_rng(
                        company_id, prod_id, month, w, "wnoise").uniform(0.95, 1.05)
                    reported_waste = round(total_waste, 2)
                    row_wastes[w] = reported_waste
                    waste_emissions_kg += total_waste * EMISSION_FACTORS['waste'][w]

                # Exact physical identities come from the values actually written above.
                # Multiple packaging materials may share one code, so their audited
                # consumed*fate_waste contributions are summed before publishing it.
                for code, contribution in _audited_identity_waste(
                        row, recipe['identity_links']).items():
                    row_wastes[code] = row_wastes.get(code, 0.0) + contribution

                for w, reported_waste in row_wastes.items():
                    row[f'waste_{w}_kg'] = reported_waste
                    if w in _IDENTITY_CODES:
                        waste_emissions_kg += reported_waste * EMISSION_FACTORS['waste'][w]

                    # Category 5: split across ALL four methods, not one winner. Allocated in
                    # integer cents so sum(c5_*) == sum(waste_*) stays exact - identity 1 is
                    # checked at float-dust tolerance and naive rounding would fire it on
                    # every clean row.
                    for method, part in _allocate_exact(reported_waste,
                                                        recipe['waste_treatments'][w]).items():
                        row[f'c5_{method.lower()}_kg'] += part
                        c5_treatment_emissions_kg += part * WASTE_TREATMENT_METHODS[method]['ef_multiplier']

                row['waste_emissions_mtco2'] = round(waste_emissions_kg / 1000, 4)
                row['c5_waste_treatment_emissions_mtco2'] = round(c5_treatment_emissions_kg / 1000, 4)

                # ── Utilities ───────────────────────────────────────────────────
                row['grid_elec_kwh'] = round(units * recipe['utils']['Grid_Elec_kWh'] * random.uniform(0.95, 1.05), 1)
                row['non_grid_energy_mj'] = round(units * recipe['utils']['Non_Grid_Energy_MJ'] * random.uniform(0.95, 1.05), 1)
                row['water_use_m3'] = round(units * recipe['utils']['Water_Use_m3'] * random.uniform(0.95, 1.05), 1)

                utility_emissions_kg = (
                    (row['grid_elec_kwh'] * grid_ef) +
                    (row['non_grid_energy_mj'] * EMISSION_FACTORS['utilities']['Non_Grid_Energy_MJ']) +
                    (row['water_use_m3'] * EMISSION_FACTORS['utilities']['Water_Use_m3'])
                )
                row['utility_emissions_mtco2'] = round(utility_emissions_kg / 1000, 4)

                # ── CATEGORY 9 — Downstream transport ──────────────────────────
                # Product mass is the YIELD, not the whole material input: packaging received
                # and discarded, and returnable containers that leave intact, never became
                # product. `product_mass_kg` was accumulated above as sum(input x fate_product).
                # This is the term a real mass balance needs, and it composes with the B-3
                # inventory columns to make `in = out + accumulation` closeable for the first
                # time in this system (progress.md 11 B-3).
                audited_mass = shared.audited_product_mass(pd.DataFrame([row]))
                if audited_mass is None or pd.isna(audited_mass.iloc[0]):
                    raise ValueError("generated row has no complete audited product-mass driver")
                product_mass_kg = float(audited_mass.iloc[0])
                product_weight_tonnes = product_mass_kg / 1000.0
                dist_km = round(recipe['transport_dist_km'] * random.uniform(0.95, 1.05), 1)
                tkm = product_weight_tonnes * dist_km
                ef_tkm = TRANSPORT_MODES[recipe['transport_mode']]['ef_per_tkm']

                row['c9_transport_mode'] = recipe['transport_mode']
                row['c9_product_weight_tonnes'] = product_weight_tonnes
                row['c9_distance_km'] = dist_km
                row['c9_tkm'] = tkm
                row['c9_transport_emissions_mtco2'] = round((tkm * ef_tkm) / 1000, 4)

                # ── CATEGORY 12 — End-of-life treatment ────────────────────────
                eol_profile = EOL_PROFILES[company_region]
                c12_emissions_kg = 0.0
                remaining_eol = product_mass_kg
                eol_items = list(eol_profile.items())
                for eol_index, (pathway, fraction) in enumerate(eol_items):
                    eol_mass_kg = (remaining_eol if eol_index == len(eol_items) - 1
                                   else product_mass_kg * fraction)
                    remaining_eol -= eol_mass_kg
                    pathway_key = pathway.lower()  # 'landfill', 'recycled', 'incinerated'
                    row[f'c12_eol_{pathway_key}_kg'] = eol_mass_kg
                    c12_emissions_kg += eol_mass_kg * EOL_EF[pathway]
                row['c12_eol_emissions_mtco2'] = round(c12_emissions_kg / 1000, 4)

                # ── TOTAL ROLLUP ────────────────────────────────────────────────
                row['total_product_emissions_mtco2'] = round(
                    row['supplier_emissions_mtco2'] +
                    row['c1_indirect_spend_emissions_mtco2'] +
                    row['waste_emissions_mtco2'] +
                    row['c5_waste_treatment_emissions_mtco2'] +
                    row['utility_emissions_mtco2'] +
                    row['c9_transport_emissions_mtco2'] +
                    row['c12_eol_emissions_mtco2'],
                    4
                )

                records.append(row)

        # ── Derive company-level aggregates from actual production ────────────
        total_revenue = 0.0
        total_annual_units = 0
        for r in records:
            prod_price = product_recipes[r['product_id']]['unit_price']
            total_revenue += r['production_units'] * prod_price * random.uniform(0.95, 1.05)
            total_annual_units += r['production_units']

        company_headcount = int(total_annual_units / random.uniform(500, 2000)) + 10
        company_sqft = int(company_headcount * random.uniform(200, 500))

        for r in records:
            r['gen_total_revenue_usd'] = round(total_revenue, 2)
            r['headcount'] = company_headcount
            r['facility_sqft'] = company_sqft

        df_pristine = pd.DataFrame(records)
        shared.ensure_dir(GENERATED_DIR)

        # 3. Inject Errors (The "Messy" Data)
        df_messy = df_pristine.copy()
        target_cols = [c for c in df_messy.columns if c.startswith('c1_') or c.startswith('waste_') or c.endswith('_mtco2')]
        target_cols.extend(['grid_elec_kwh', 'non_grid_energy_mj', 'water_use_m3'])
        # Include new numeric columns in the error injection pool
        target_cols.extend(['c9_tkm', 'c9_distance_km', 'c9_product_weight_tonnes'])
        for cat in INDIRECT_SPEND_CATEGORIES:
            target_cols.append(f'c1_spend_{cat.lower()}_usd')
        for method in WASTE_TREATMENT_METHODS:
            target_cols.append(f'c5_{method.lower()}_kg')
        for pathway in ['landfill', 'recycled', 'incinerated']:
            target_cols.append(f'c12_eol_{pathway}_kg')
        # De-duplicate (some columns may already be captured by the startswith filters)
        target_cols = list(dict.fromkeys(target_cols))

        # Accidental Errors
        err_idx = np.random.choice(df_messy.index, size=int(len(df_messy) * _args.error_rate), replace=False)
        global_error_summary['generated_rows'] += len(df_messy)
        global_error_summary['requested_error_rows'] += len(err_idx)

        # `_error_type_list` already rejected unknown names at parse time, so anything
        # arriving here is either a keyword or a validated list.
        all_error_types = list(ERROR_TYPES)
        if _args.error_types == 'all':
            active_error_types = all_error_types
        elif _args.error_types == 'none':
            active_error_types = []
        elif _args.error_types == 'random':
            active_error_types = list(RANDOM_ERROR_TYPES)
        else:
            active_error_types = [e.strip() for e in _args.error_types.split(',') if e.strip()]

        print(f"  [INJECT] {len(err_idx)} of {len(df_messy)} rows "
              f"({_args.error_rate:.0%}), error types: "
              f"{', '.join(active_error_types) if active_error_types else 'none'}")


        # ── Deliberate Under-reporting (Greenwashing) — Waste Suppression ────
        active_wastes = [col for col in df_messy.columns if col.startswith('waste_') and col.endswith('_kg') and df_messy[col].sum() > 0]

        if active_wastes:
            gw_waste_col = random.choice(active_wastes)
            suppression_factor = random.uniform(0.3, 0.6)
            gw_mask = df_messy[gw_waste_col] > 0
            global_error_summary['waste_suppression'] += int(gw_mask.sum())

            waste_code_only = gw_waste_col.replace('waste_', '').replace('_kg', '')
            ef = EMISSION_FACTORS['waste'][waste_code_only]

            original_waste_mass = df_messy.loc[gw_mask, gw_waste_col]
            suppressed_waste_mass = (original_waste_mass * suppression_factor).round(2)

            df_messy.loc[gw_mask, gw_waste_col] = suppressed_waste_mass

            mass_diff = original_waste_mass - suppressed_waste_mass
            emission_diff_mtco2 = (mass_diff * ef) / 1000.0
            df_messy.loc[gw_mask, 'waste_emissions_mtco2'] -= emission_diff_mtco2
            df_messy.loc[gw_mask, 'total_product_emissions_mtco2'] -= emission_diff_mtco2

            # Recalculate Category 5 treatment emissions for the suppressed waste code
            # Find which treatment method this waste code uses (from any product recipe)
            c5_treatment_diff_mtco2 = _unwind_c5(df_messy, gw_mask, waste_code_only,
                                                 mass_diff, company_id)
            df_messy.loc[gw_mask, 'c5_waste_treatment_emissions_mtco2'] -= c5_treatment_diff_mtco2
            df_messy.loc[gw_mask, 'total_product_emissions_mtco2'] -= c5_treatment_diff_mtco2

        # ── Zero-Emission Loophole Greenwashing ──────────────────────────────
        if random.random() < 0.20:
            active_cols = [c for c in shared.material_consumption_cols(df_messy.columns)
                           if df_messy[c].sum() > 0]
            if not active_cols:
                active_cols = [c for c in df_messy.columns if c.startswith('waste_') and c.endswith('_kg') and df_messy[c].sum() > 0]
            if active_cols:
                gw_zero_col = random.choice(active_cols)
                eligible_zero = df_messy[gw_zero_col] > 0
                global_error_summary['intentional_zero_attempted'] += int(eligible_zero.sum())
                gw_mask = eligible_zero & pd.Series(
                    [random.random() < 0.5 for _ in range(len(df_messy))], index=df_messy.index)
                if eligible_zero.any() and not gw_mask.any():
                    gw_mask.loc[eligible_zero.idxmax()] = True
                global_error_summary['intentional_zero'] += int(gw_mask.sum())

                # Zero out the physical mass in the messy file
                original_mass_for_zero = df_pristine.loc[gw_mask, gw_zero_col].copy()
                df_messy.loc[gw_mask, gw_zero_col] = 0.0

                # Deduct corresponding emissions from the summary emissions lines
                if gw_zero_col.startswith('waste_'):
                    waste_code = gw_zero_col.replace('waste_', '').replace('_kg', '')
                    ef = EMISSION_FACTORS['waste'][waste_code]
                    emission_diff = (original_mass_for_zero * ef) / 1000.0
                    df_messy.loc[gw_mask, 'waste_emissions_mtco2'] -= emission_diff
                    df_messy.loc[gw_mask, 'total_product_emissions_mtco2'] -= emission_diff
                    # Category 5 recalculation
                    c5_diff = _unwind_c5(df_messy, gw_mask, waste_code,
                                         original_mass_for_zero, company_id)
                    df_messy.loc[gw_mask, 'c5_waste_treatment_emissions_mtco2'] -= c5_diff
                    df_messy.loc[gw_mask, 'total_product_emissions_mtco2'] -= c5_diff

                elif gw_zero_col.startswith('c1_') and gw_zero_col.endswith('_kg'):
                    mat_name = gw_zero_col.replace('c1_', '').replace('_kg', '')
                    mat_key = next((k for k in EMISSION_FACTORS['mats'] if k.lower() == mat_name), None)
                    if mat_key:
                        ef = EMISSION_FACTORS['mats'][mat_key]
                        emission_diff = (original_mass_for_zero * ef) / 1000.0
                        df_messy.loc[gw_mask, 'supplier_emissions_mtco2'] -= emission_diff
                        df_messy.loc[gw_mask, 'total_product_emissions_mtco2'] -= emission_diff

                        # Recalculate Cat 9 and Cat 12 from audited product fate.
                        new_product_mass_kg = shared.audited_product_mass(df_messy)
                        new_weight_tonnes = new_product_mass_kg.loc[gw_mask] / 1000.0

                        old_weight = df_messy.loc[gw_mask, 'c9_product_weight_tonnes']
                        df_messy.loc[gw_mask, 'c9_product_weight_tonnes'] = new_weight_tonnes.round(4)

                        # Recalculate tkm and transport emissions
                        new_tkm = new_weight_tonnes * df_messy.loc[gw_mask, 'c9_distance_km']
                        df_messy.loc[gw_mask, 'c9_tkm'] = new_tkm.round(4)

                        # To recalculate transport emissions, we need the mode's EF
                        for mode_name, mode_info in TRANSPORT_MODES.items():
                            mode_mask = gw_mask & (df_messy['c9_transport_mode'] == mode_name)
                            if mode_mask.any():
                                new_transport_em = (df_messy.loc[mode_mask, 'c9_tkm'] * mode_info['ef_per_tkm']) / 1000.0
                                old_transport_em = df_messy.loc[mode_mask, 'c9_transport_emissions_mtco2']
                                transport_diff = old_transport_em - new_transport_em
                                df_messy.loc[mode_mask, 'c9_transport_emissions_mtco2'] = new_transport_em.round(4)
                                df_messy.loc[mode_mask, 'total_product_emissions_mtco2'] -= transport_diff

                        # Recalculate Cat 12 end-of-life from new weight
                        eol_profile = EOL_PROFILES[company_region]
                        old_c12 = df_messy.loc[gw_mask, 'c12_eol_emissions_mtco2'].copy()
                        new_c12_emissions_kg = pd.Series(0.0, index=df_messy.loc[gw_mask].index)
                        for pathway, fraction in eol_profile.items():
                            pathway_key = pathway.lower()
                            new_eol_mass = new_weight_tonnes * 1000.0 * fraction
                            df_messy.loc[gw_mask, f'c12_eol_{pathway_key}_kg'] = new_eol_mass.round(2)
                            new_c12_emissions_kg += new_eol_mass * EOL_EF[pathway]
                        df_messy.loc[gw_mask, 'c12_eol_emissions_mtco2'] = (new_c12_emissions_kg / 1000.0).round(4)
                        c12_diff = old_c12 - df_messy.loc[gw_mask, 'c12_eol_emissions_mtco2']
                        df_messy.loc[gw_mask, 'total_product_emissions_mtco2'] -= c12_diff

        # ── NEW: Transport Distance Greenwashing ─────────────────────────────
        if random.random() < 0.20:
            gw_transport_mask = df_messy['c9_distance_km'] > 0
            if gw_transport_mask.any():
                global_error_summary['transport_suppression'] += int(gw_transport_mask.sum())
                distance_suppression = random.uniform(0.3, 0.5)
                old_dist = df_messy.loc[gw_transport_mask, 'c9_distance_km'].copy()
                new_dist = old_dist * distance_suppression
                df_messy.loc[gw_transport_mask, 'c9_distance_km'] = new_dist.round(1)

                # Recalculate tkm
                new_tkm = df_messy.loc[gw_transport_mask, 'c9_product_weight_tonnes'] * new_dist
                old_tkm = df_messy.loc[gw_transport_mask, 'c9_tkm']
                df_messy.loc[gw_transport_mask, 'c9_tkm'] = new_tkm.round(4)

                # Recalculate transport emissions per mode
                for mode_name, mode_info in TRANSPORT_MODES.items():
                    mode_mask = gw_transport_mask & (df_messy['c9_transport_mode'] == mode_name)
                    if mode_mask.any():
                        new_em = (df_messy.loc[mode_mask, 'c9_tkm'] * mode_info['ef_per_tkm']) / 1000.0
                        old_em = df_messy.loc[mode_mask, 'c9_transport_emissions_mtco2']
                        df_messy.loc[mode_mask, 'c9_transport_emissions_mtco2'] = new_em.round(4)
                        df_messy.loc[mode_mask, 'total_product_emissions_mtco2'] -= (old_em - new_em)

        # Deliberate scenarios above propagate coherently from clean values. The one-cell
        # accidental error is applied last and is never copied into dependent fields.
        if active_error_types:
            _inject_random_errors(df_messy, target_cols, err_idx, active_error_types,
                                  global_error_summary)

        shared.ensure_dir(GENERATED_DIR)
        shared.atomic_write_csv(df_messy, os.path.join(GENERATED_DIR, f'{company_id}_messy.csv'), index=False)
        # Pristine is the completion marker for a generated pair, published only after messy.
        shared.atomic_write_csv(df_pristine, os.path.join(GENERATED_DIR, f'{company_id}_pristine.csv'), index=False)
        pair_hashes[company_id] = {
            'messy': shared.sha256_file(os.path.join(GENERATED_DIR, f'{company_id}_messy.csv')),
            'pristine': shared.sha256_file(os.path.join(GENERATED_DIR, f'{company_id}_pristine.csv')),
        }
        completed_company_ids.append(company_id)
        shared.write_generation_state(
            GENERATED_DIR, expected_company_ids, completed_company_ids, pair_hashes,
            seed=BASE_SEED, requested_products=NUM_PRODUCTS)
        print(f"  -> Generated {len(df_messy)} monthly product records for {company_id}. (Sparse matrix populated)")

    shared.ensure_dir(GENERATED_DIR)
    shared.atomic_write_json(global_error_summary,
                             os.path.join(GENERATED_DIR, 'injection_summary.json'), indent=4)
    shared.write_generation_state(
        GENERATED_DIR, expected_company_ids, completed_company_ids, pair_hashes,
        status='complete', seed=BASE_SEED, requested_products=NUM_PRODUCTS)
    print("\n--- Error Injection Summary ---")
    for k, v in global_error_summary.items():
        if not isinstance(v, (int, float)) or v > 0:
            print(f"  {k}: {v}")

    print(f"\nData Generation Complete. Files saved to '{GENERATED_DIR}/'.")


if __name__ == '__main__':
    main()
