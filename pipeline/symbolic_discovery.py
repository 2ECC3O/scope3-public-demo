"""Phase D: symbolic emission-factor discovery via PySR (optional dependency).

Only exercised when the pipeline is run with `--engine full`. PySR shells out to a
Julia backend and is not in the base requirements (see requirements-full.txt), so
every public function here degrades to "PySR unavailable" instead of raising —
a `full`-less run never even imports this module's dependency, and an environment
without PySR installed gets the honest (default_efs, False) fallback.
"""
import json
import os

import numpy as np


def _try_import_pysr():
    try:
        from pysr import PySRRegressor
        return PySRRegressor
    except Exception:
        return None


# niterations scaled by MACHINE_TIER: this runs once per target-category per company
# inside the verification loop (not a standalone symbolic-regression job), so budgets
# are kept small — 10/25/50 trade search depth for wall-clock time across low/mid/high
# hardware tiers.
_NITERATIONS_BY_TIER = {'low': 10, 'mid': 25, 'high': 50}


def discover_emission_factors(X, y, x_cols, default_efs, machine_tier, company_id=None, target_name=None,
                               output_dir='generated_data'):
    """Fit a PySR symbolic regressor for y = f(X) and honestly compare it to the
    existing Ridge-derived linear combination (`default_efs`) on a held-out split.

    default_efs is double-duty: it's the Ridge coefficients dict to beat, and also
    the safe fallback returned unchanged whenever PySR is unavailable, fails, or
    does not win.

    Returns (efs_dict, won):
      - won=True  -> efs_dict is PySR-derived per-column coefficients (strictly
                     better holdout MAE than the Ridge combination).
      - won=False -> efs_dict is default_efs, unchanged. Caller should keep using
                     its existing Ridge/median logic.
    """
    PySRRegressor = _try_import_pysr()
    if PySRRegressor is None:
        return default_efs, False

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20 or X.ndim != 2 or X.shape[1] == 0 or X.shape[1] != len(x_cols):
        return default_efs, False

    try:
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        n_holdout = max(1, int(n * 0.2))
        holdout_idx, train_idx = idx[:n_holdout], idx[n_holdout:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

        model = PySRRegressor(
            niterations=_NITERATIONS_BY_TIER.get(machine_tier, 10),
            binary_operators=["+", "*"],
            unary_operators=[],
            elementwise_loss="L1DistLoss()",  # robust to outliers, matches Phase C's L1 philosophy
            parsimony=0.01,  # aggressive simplicity bias: EFs are linear-additive by
                              # construction, so we want PySR to prefer that shape over
                              # a lower-loss-but-overfit expression with extra terms
            random_state=42,
            deterministic=True,
            parallelism="serial",  # required by PySR for deterministic=True to be honored
            verbosity=0,
            progress=False,
        )
        model.fit(X_train, y_train, variable_names=list(x_cols))

        efs = _extract_coefficients(model, x_cols, X_train)
        if efs is None:
            # Couldn't safely linearize the discovered formula back to per-column
            # coefficients (caller reconstructs emissions as sum(col * coef)) —
            # don't force an incompatible formula shape onto that contract.
            return default_efs, False

        # Compare on the LINEARIZED coefficients, not the raw PySR formula: the
        # caller only ever uses `efs` (a linear combination) downstream, so the
        # holdout comparison must honestly reflect what "winning" will actually
        # reconstruct — a formula that fits well nonlinearly but linearizes to a
        # near-zero coefficient dict must not be allowed to win on the strength
        # of a prediction it doesn't hand back.
        efs_coefs = np.array([efs.get(c, 0.0) for c in x_cols], dtype=float)
        efs_pred = X_holdout @ efs_coefs
        efs_mae = float(np.mean(np.abs(y_holdout - efs_pred)))

        ridge_coefs = np.array([default_efs.get(c, 0.0) for c in x_cols], dtype=float)
        ridge_pred = X_holdout @ ridge_coefs
        ridge_mae = float(np.mean(np.abs(y_holdout - ridge_pred)))

        if not (efs_mae < ridge_mae):
            return default_efs, False

        if company_id is not None:
            _cache_formula(company_id, target_name or "unknown", str(model.sympy()), efs, output_dir=output_dir)

        return efs, True
    except Exception:
        return default_efs, False


def _extract_coefficients(model, x_cols, X_train):
    """Linearize the discovered (polynomial, since only +/* are allowed) formula
    around the training-column means to get an effective per-column coefficient —
    exact when the formula is a simple linear sum, an honest local approximation
    otherwise."""
    try:
        import sympy
        expr = model.sympy()
        symbols = sympy.symbols(list(x_cols))
        means = {s: float(np.mean(X_train[:, i])) for i, s in enumerate(symbols)}
        efs = {}
        for i, col in enumerate(x_cols):
            deriv = sympy.diff(expr, symbols[i])
            coef = float(deriv.subs(means))
            efs[col] = max(coef, 0.0)
        return efs
    except Exception:
        return None


def _cache_formula(company_id, target_name, equation_str, efs, output_dir='generated_data'):
    """Merge-write the discovered formula into {output_dir}/{company_id}_formulas.json,
    preserving entries from other target categories written earlier in the same run."""
    path = os.path.join(output_dir, f'{company_id}_formulas.json')
    try:
        os.makedirs(output_dir, exist_ok=True)
        data = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data[target_name] = {'formula': equation_str, 'coefficients': efs}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # caching is best-effort; never fail the caller over a write error


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.uniform(1, 10, n)
    x2 = rng.uniform(1, 10, n)
    noise = rng.normal(0, 0.1, n)
    y = 2 * x1 + 3 * x2 + noise
    X = np.column_stack([x1, x2])
    x_cols = ['x1', 'x2']
    default_efs = {'x1': 2.0, 'x2': 3.0}

    efs, won = discover_emission_factors(X, y, x_cols, default_efs, machine_tier='low', company_id=None)

    if not won:
        assert efs == default_efs, "unavailable/non-winning PySR must return defaults unchanged"
        print("PySR unavailable or did not win holdout comparison — returned defaults unchanged "
              "(expected in an environment without PySR installed).")
    else:
        assert set(efs.keys()) == set(x_cols), "winning result must cover every x_col"
        print(f"PySR won: {efs}")
    print("Self-check passed.")
