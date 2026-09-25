"""
Conditional models: RFOD-style (Random-Forest Outlier Detection) conditional engine.

Instead of predicting a target physical column from the thin generic FEATURES list
(13 company/product metadata columns), this fits a model that predicts a target
column from all OTHER active physical columns (materials, wastes, utilities, spend,
transport) — a much richer, target-specific predictor set. A companion RandomForest
gives per-row uncertainty (inter-tree std) so anomaly scores can be down-weighted
where the model itself is unsure, and a Predictive-Mean-Matching (PMM) imputer snaps
model predictions to real observed values for flagged/zero cells.
"""
import sys

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore

_EPS = 1e-6


def _mad_purify_mask(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Same MAD>k trim pattern used in step_2_verify_data.py. Returns a boolean
    keep-mask; if MAD is ~0 (degenerate/constant column) nothing is trimmed."""
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad > 1e-6:
        return np.abs(values - med) / mad <= k
    return np.ones(len(values), dtype=bool)


def fit_conditional(df: pd.DataFrame, target_col: str, predictor_cols: list,
                     tree_method: str, device: str, max_train_rows: int,
                     random_state: int = 42):
    """Fit an XGBoost (MAE) + RandomForest pair predicting `target_col` from
    `predictor_cols` on MAD-purified rows of df. Returns None when there's too
    little clean data to fit anything meaningful (caller falls back to the
    median estimator)."""
    predictor_cols = [c for c in predictor_cols if c in df.columns]
    if not predictor_cols:
        return None

    keep_mask = _mad_purify_mask(df[target_col].values)
    clean = df.loc[keep_mask, predictor_cols + [target_col]].dropna()
    if len(clean) < 30:
        return None

    if len(clean) > max_train_rows:
        clean = clean.sample(n=max_train_rows, random_state=random_state)

    X = clean[predictor_cols].values
    y = clean[target_col].values

    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        tree_method=tree_method,
        device=device,
        random_state=random_state,
    )
    xgb_model.fit(X, y)

    rf_model = RandomForestRegressor(
        # Inter-tree std is itself an estimate from n_estimators samples; 50 trees
        # gave jittery uncertainty. 300 is still cheap at a few hundred rows.
        n_estimators=300, max_depth=8, random_state=random_state, n_jobs=-1,
    )
    rf_model.fit(X, y)

    train_mae = float(np.mean(np.abs(xgb_model.predict(X) - y)))

    return {
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'predictor_cols': predictor_cols,
        'train_mae': train_mae,
    }


def predict_conditional(fitted: dict, df: pd.DataFrame):
    """Predict for ALL rows of df (zero-valued cells included). Returns
    (prediction, uncertainty) where uncertainty is the RF's per-row std across
    its trees, normalized by |prediction| so it's comparable across columns."""
    predictor_cols = fitted['predictor_cols']
    X = df[predictor_cols].values

    prediction = fitted['xgb_model'].predict(X)

    tree_preds = np.stack([t.predict(X) for t in fitted['rf_model'].estimators_], axis=0)
    tree_std = tree_preds.std(axis=0)
    uncertainty = tree_std / (np.abs(prediction) + _EPS)

    return prediction, uncertainty


def uncertainty_weighted_score(prediction, uncertainty, actual) -> np.ndarray:
    """Anomaly score per row: relative gap between actual and prediction,
    down-weighted where the RF's trees disagree (high uncertainty -> lower
    confidence in the flag). Works fine for actual == 0 rows."""
    prediction = np.asarray(prediction, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    actual = np.asarray(actual, dtype=float)

    raw_score = np.abs(actual - prediction) / (np.abs(prediction) + _EPS)
    weight = 1.0 - np.clip(uncertainty, 0.0, 1.0)
    return raw_score * weight


def impute_flagged(df: pd.DataFrame, flagged_mask, target_col: str, fitted: dict,
                    clean_reference_values) -> np.ndarray:
    """Impute flagged cells from the conditional model. Zero/missing cells use
    PMM (snap the prediction to the nearest observed CLEAN value — plausible for
    genuinely absent data). Non-zero flagged cells keep the RAW prediction: PMM
    by construction cannot return a value outside the observed range, so for
    magnitude errors (x1000, dropped zero) it snaps to the wrong donor instead
    of the out-of-range true value. Unflagged rows are untouched."""
    flagged_mask = np.asarray(flagged_mask, dtype=bool)
    original = df[target_col].values.astype(float)
    result = original.copy()
    if not flagged_mask.any():
        return result

    prediction, _ = predict_conditional(fitted, df)
    flagged_idx = np.where(flagged_mask)[0]
    result[flagged_idx] = prediction[flagged_idx]

    ref = np.asarray(clean_reference_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    zero_idx = flagged_idx[(original[flagged_idx] == 0) | np.isnan(original[flagged_idx])]
    if ref.size == 0 or zero_idx.size == 0:
        return result

    sorted_ref = np.sort(ref)
    preds_zero = prediction[zero_idx]
    pos = np.searchsorted(sorted_ref, preds_zero)
    pos_right = np.clip(pos, 0, len(sorted_ref) - 1)
    pos_left = np.clip(pos - 1, 0, len(sorted_ref) - 1)
    left = sorted_ref[pos_left]
    right = sorted_ref[pos_right]
    choose_left = np.abs(preds_zero - left) <= np.abs(right - preds_zero)
    result[zero_idx] = np.where(choose_left, left, right)
    return result


if __name__ == '__main__':
    # ponytail: minimal self-check, not a test suite — asserts the two planted
    # anomalies (10x spike, zeroed-out cell) land in the top 5% of anomaly scores.
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        'colA': rng.uniform(1, 10, n),
        'colB': rng.uniform(1, 10, n),
        'colC': rng.uniform(1, 10, n),
        'colD': rng.uniform(1, 10, n),
        'colE': rng.uniform(1, 10, n),
    })
    noise = rng.normal(0, 0.5, n)
    df['target'] = 2.0 * df['colA'] + 0.5 * df['colB'] + noise

    HIGH_ANOMALY_IDX = 10
    ZERO_ANOMALY_IDX = 20
    df.loc[HIGH_ANOMALY_IDX, 'target'] *= 10.0
    df.loc[ZERO_ANOMALY_IDX, ['colA', 'colB']] = [9.5, 9.5]  # implies a large target
    df.loc[ZERO_ANOMALY_IDX, 'target'] = 0.0

    predictor_cols = ['colA', 'colB', 'colC', 'colD', 'colE']

    try:
        fitted = fit_conditional(df, 'target', predictor_cols,
                                  tree_method='hist', device='cpu', max_train_rows=10_000)
        assert fitted is not None, "fit_conditional returned None on 500-row clean-ish data"

        prediction, uncertainty = predict_conditional(fitted, df)
        scores = uncertainty_weighted_score(prediction, uncertainty, df['target'].values)

        top_k = max(1, int(np.ceil(0.05 * n)))
        top_idx = set(np.argsort(scores)[-top_k:])

        assert HIGH_ANOMALY_IDX in top_idx, (
            f"10x spike anomaly (row {HIGH_ANOMALY_IDX}) not in top {top_k} scores "
            f"(score={scores[HIGH_ANOMALY_IDX]:.4f}, rank={n - 1 - np.argsort(scores).tolist().index(HIGH_ANOMALY_IDX)})"
        )
        assert ZERO_ANOMALY_IDX in top_idx, (
            f"zeroed-out anomaly (row {ZERO_ANOMALY_IDX}) not in top {top_k} scores "
            f"(score={scores[ZERO_ANOMALY_IDX]:.4f})"
        )

        flagged_mask = np.zeros(n, dtype=bool)
        flagged_mask[[HIGH_ANOMALY_IDX, ZERO_ANOMALY_IDX]] = True
        clean_ref = df.loc[~flagged_mask, 'target'].values
        imputed = impute_flagged(df, flagged_mask, 'target', fitted, clean_ref)
        assert imputed[ZERO_ANOMALY_IDX] > 1.0, (
            f"zero-cell PMM imputation should replace 0.0 with a plausible positive "
            f"value, got {imputed[ZERO_ANOMALY_IDX]}"
        )

        print(f"PASS: both planted anomalies scored in top {top_k}/{n} "
              f"(high-spike score={scores[HIGH_ANOMALY_IDX]:.3f}, "
              f"zero-cell score={scores[ZERO_ANOMALY_IDX]:.3f}, "
              f"zero-cell imputed value={imputed[ZERO_ANOMALY_IDX]:.3f})")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
