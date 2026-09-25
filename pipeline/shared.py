import glob
import hashlib
import json
import os
import re
import tempfile
import urllib.request

PROJECT_RE = re.compile(r'project (\d+)')

OLLAMA_URL_TAGS = "http://127.0.0.1:11434/api/tags"


def ollama_is_up(timeout=2):
    try:
        with urllib.request.urlopen(urllib.request.Request(OLLAMA_URL_TAGS), timeout=timeout):
            return True
    except Exception:
        return False


def ensure_ollama_running(wait_s=30):
    """Report availability without starting or stopping the user's service."""
    if ollama_is_up():
        return True
    print("Ollama is unavailable; continuing with the deterministic fallback.")
    return False
def ensure_dir(path):
    """Self-repairing directory creation - recreates it if deleted mid-run."""
    os.makedirs(path, exist_ok=True)
    return path


def material_consumption_cols(columns):
    """Audited material-use columns, excluding opening/purchase/closing stock."""
    inventory = ("_opening_kg", "_purchased_kg", "_closing_kg")
    return [c for c in columns if c.startswith("c1_") and c.endswith("_kg")
            and not c.endswith(inventory)]


def is_flat_column(col):
    """Columns stored once per reporting row rather than per production unit."""
    return (col == "c9_distance_km" or col.startswith("c1_spend_")
            or col == "c1_indirect_spend_emissions_mtco2" or col.endswith("_pct"))


def audited_fate_driver(df, material, fate, value_getter=None):
    """Consumption multiplied by a valid, non-suspect audited fate fraction."""
    import numpy as np
    import pandas as pd

    base = f"c1_{material.lower()}"
    cols = [f"{base}_kg", *(f"{base}_fate_{name}_pct"
                             for name in ("product", "waste", "intact"))]
    if cols[0] not in df.columns:
        return None
    value_getter = value_getter or (lambda col: df[col])
    consumption = pd.to_numeric(value_getter(cols[0]), errors="coerce")
    if any(c not in df.columns for c in cols[1:]):
        valid_zero = consumption.eq(0) & np.isfinite(consumption)
        for col in cols:
            if f"{col}_anomaly" in df.columns:
                valid_zero &= pd.to_numeric(
                    df[f"{col}_anomaly"], errors="coerce").fillna(0).eq(0)
            if f"{col}_status" in df.columns:
                valid_zero &= df[f"{col}_status"].astype(str).ne("SUSPECT")
        return pd.Series(0.0, index=df.index).where(valid_zero)
    values = [consumption, *(pd.to_numeric(value_getter(c), errors="coerce")
                              for c in cols[1:])]
    finite = np.logical_and.reduce([np.isfinite(v) for v in values])
    fate_valid = np.logical_and.reduce([(v >= 0) & (v <= 1) for v in values[1:]])
    fate_valid &= np.isclose(values[1] + values[2] + values[3], 1.0, atol=1e-6)
    valid = finite & ((values[0] == 0) | fate_valid)
    for col in cols:
        if f"{col}_anomaly" in df.columns:
            valid &= pd.to_numeric(
                df[f"{col}_anomaly"], errors="coerce").fillna(0).eq(0)
        if f"{col}_status" in df.columns:
            valid &= df[f"{col}_status"].astype(str).ne("SUSPECT")
    return (values[0] * values[("product", "waste", "intact").index(fate) + 1]).where(valid)


def audited_product_mass(df, value_getter=None):
    """Product mass from audited consumption and product-fate fractions only."""
    parts = []
    for col in material_consumption_cols(df.columns):
        material = col[3:-3]
        part = audited_fate_driver(df, material, "product", value_getter)
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    total = parts[0].copy()
    for part in parts[1:]:
        total = total + part
    return total


def atomic_write_csv(df, path, **kwargs):
    """Write a complete CSV beside its destination, then publish it atomically."""
    path = os.fspath(path)
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(value, path, **kwargs):
    """Write complete JSON beside its destination, then publish it atomically."""
    path = os.fspath(path)
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                               dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


GENERATION_STATE_FILE = "generation_state.json"


def prepare_fresh_output_dir(path):
    """Create an output directory, refusing any pre-existing content."""
    path = os.fspath(path)
    if os.path.exists(path):
        if not os.path.isdir(path) or os.listdir(path):
            raise FileExistsError(f"generation output must be a fresh empty directory: {path}")
    else:
        os.makedirs(path)
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_generation_state(generated_dir, expected_ids, completed_ids=(), pair_hashes=None,
                           status="incomplete", **provenance):
    state = {
        "schema_version": 1,
        "status": status,
        "expected_company_ids": list(expected_ids),
        "completed_company_ids": list(completed_ids),
        "pair_hashes": pair_hashes or {},
        **provenance,
    }
    atomic_write_json(state, os.path.join(generated_dir, GENERATION_STATE_FILE), indent=2)
    return state


def generation_is_complete(project_dir):
    """Validate new manifests; projects without one remain legacy-compatible."""
    generated_dir = os.path.join(os.fspath(project_dir), "generated company")
    state_path = os.path.join(generated_dir, GENERATION_STATE_FILE)
    if not os.path.exists(state_path):
        return True
    try:
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
        expected = state["expected_company_ids"]
        completed = state["completed_company_ids"]
        hashes = state["pair_hashes"]
        if state.get("status") != "complete" or expected != completed:
            return False
        for company_id in expected:
            pair = hashes.get(company_id, {})
            messy = os.path.join(generated_dir, f"{company_id}_messy.csv")
            pristine = os.path.join(generated_dir, f"{company_id}_pristine.csv")
            # The verifier may authenticate the auditable input, but it must never
            # open or hash pristine truth. Its existence plus the generator-recorded
            # hash is bounded completion metadata; scoring reads pristine elsewhere.
            if (not os.path.isfile(messy) or pair.get("messy") != sha256_file(messy)
                    or not os.path.isfile(pristine)
                    or not isinstance(pair.get("pristine"), str)):
                return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _existing_project_numbers(output_root):
    numbers = []
    if os.path.isdir(output_root):
        for name in os.listdir(output_root):
            m = PROJECT_RE.fullmatch(name)
            if m and os.path.isdir(os.path.join(output_root, name)):
                numbers.append(int(m.group(1)))
    return numbers


def list_project_dirs(output_root='.'):
    """All existing 'project NN' dirs under output_root, sorted numerically."""
    numbers = sorted(_existing_project_numbers(output_root))
    return [os.path.join(output_root, f'project {n:02d}') for n in numbers]


def new_project_dir(output_root='.'):
    """Create and return the next sequential 'project NN/generated company' dir."""
    ensure_dir(output_root)
    next_n = max(_existing_project_numbers(output_root), default=0) + 1
    while True:
        project_dir = os.path.join(output_root, f'project {next_n:02d}')
        try:
            os.mkdir(project_dir)
            break
        except FileExistsError:
            next_n += 1
    ensure_dir(os.path.join(project_dir, 'generated company'))
    return project_dir


def _company_ids_in(dir_path, suffix):
    if not os.path.isdir(dir_path):
        return set()
    return {
        os.path.basename(p)[:-len(suffix)]
        for p in glob.glob(os.path.join(dir_path, f'COMP_*{suffix}'))
    }


def find_next_pending(output_root, source_subdir, source_suffix, dest_subdir, is_done_fn):
    """First project dir (in order) whose source_subdir has company files but
    isn't done yet per is_done_fn(project_dir). None if nothing is pending."""
    for project_dir in list_project_dirs(output_root):
        source = os.path.join(project_dir, source_subdir)
        if not _company_ids_in(source, source_suffix):
            continue
        if source_subdir == "generated company" and not generation_is_complete(project_dir):
            print(f"Skipping incomplete generated project: {project_dir}")
            continue
        if not is_done_fn(project_dir):
            return project_dir
    return None


def discover_company_files(company_dir, corrected_dir=None):
    """Find company file triplets: pristine/messy under company_dir, AI-corrected
    under corrected_dir (defaults to company_dir for backward compatibility)."""
    corrected_dir = corrected_dir or company_dir
    pattern = os.path.join(company_dir, 'COMP_*_pristine.csv')
    triplets = []
    for pristine_path in sorted(glob.glob(pattern)):
        company_id = os.path.basename(pristine_path).replace('_pristine.csv', '')
        messy_path = os.path.join(company_dir, f'{company_id}_messy.csv')
        corrected_path = os.path.join(corrected_dir, f'{company_id}_AI_corrected.csv')
        if os.path.exists(messy_path) and os.path.exists(corrected_path):
            triplets.append((company_id, pristine_path, messy_path, corrected_path))
    return triplets
