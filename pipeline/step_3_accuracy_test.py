import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib  # type: ignore
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.collections import LineCollection  # type: ignore
import seaborn as sns  # type: ignore
import os
import sys
import gc
import json
import argparse
from datetime import datetime, timezone
from tqdm import tqdm  # type: ignore
import shared
from shared import discover_company_files
import warnings
warnings.filterwarnings('ignore')

# --- CLI ARGUMENT PARSING ---
_parser = argparse.ArgumentParser(description='Scope 3 Accuracy Diagnostics (product-level schema)')
_parser.add_argument('--companies', type=int, default=None,
                      help='Number of companies to evaluate (default: every company with a '
                           'complete pristine/messy/corrected triplet)')
_parser.add_argument(
    '--all', action='store_true',
    help='Process every pending project (has AI-corrected data but no accuracy test yet) instead of just the next one.',
)
_args, _ = _parser.parse_known_args()

NUM_COMPANIES = _args.companies
PROCESS_ALL_PROJECTS = _args.all

# Maximum points rendered per scatter/KDE plot.
# Full data is ALWAYS used for metric calculations.
MAX_PLOT_POINTS = 5_000
# ---------------------------------------------------

# --- CONFIGURATION ---
TARGET_COLS = [
    'supplier_emissions_mtco2',
    'waste_emissions_mtco2',
    'utility_emissions_mtco2',
    'total_product_emissions_mtco2',
    'grid_elec_kwh',
    'non_grid_energy_mj',
    'water_use_m3',
]
# ---------------------


def _mem_mb() -> str:
    """Return current process RSS in MB for progress logging."""
    try:
        import psutil  # type: ignore
        return f"{psutil.Process().memory_info().rss / 1e6:.0f} MB"
    except ImportError:
        return "N/A"


def _downsample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def _within_tol(err, pristine_val):
    """F4: value-relative tolerance (2% of the pristine value) instead of a
    flat atol=1e-2. Flat atol was optimistic on mtco2 columns (~1e-3 scale,
    where any uncorrected error < 0.01 counted as "perfect") and overly
    strict on kg columns (~1e6 scale, where immaterial sub-0.002% wiggles
    counted as failures). Pure tolerance predicate only - see _is_restored /
    _is_immaterial below for what actually gets counted as "perfect".
    NOTE: report rows produced before this change used atol=1e-2 flat and
    are not directly comparable to rows produced after.
    """
    return err <= np.maximum(1e-9, 0.02 * np.abs(pristine_val))


def _is_restored(err_before, err_after, pristine_val):
    """F4b: "restored" requires the AI to have actually moved the cell (err_after
    < err_before) AND landed within tolerance - not just "ended up under 2%".
    Split from _is_immaterial after a diagnostic showed 493/493 newly-"perfect"
    cells in COMP_001 (project 15) were untouched (corrected == messy): counting
    them as restored inflated tp_flag_rate +7.35pts with zero detector improvement.
    """
    return _within_tol(err_after, pristine_val) & (err_after < err_before)


def _is_immaterial(err_before, pristine_val):
    """The injected corruption itself never exceeded tolerance - a blind auditor
    can't be expected to flag it, and nothing needed restoring."""
    return _within_tol(err_before, pristine_val)


def _column_family(col):
    """Column-name-prefix family used to group the FP/TP breakdown below.
    Order matters: 'c1_spend_' must be tested before 'c1_' or every spend
    column would be swallowed by the more general c1 family."""
    for prefix in ('waste_', 'c5_', 'c1_spend_', 'c1_', 'c9_', 'c12_eol_'):
        if col.startswith(prefix):
            return prefix.rstrip('_')
    if col.endswith('_mtco2'):
        return 'mtco2'
    return 'other'


def _eligible_flag_counts(flagged, true_error, restored, immaterial):
    """Raw flagged counts plus the v2 numerator/denominator population."""
    should_flag = true_error & ~restored & ~immaterial
    eligible_flagged = flagged & should_flag
    return {
        'flagged_total': int(flagged.sum()),
        'flagged_true_errors': int((flagged & true_error).sum()),
        'eligible_flagged_tp': int(eligible_flagged.sum()),
        'should_flag': int(should_flag.sum()),
        'tp_flag_rate': float(eligible_flagged.sum() / max(1, should_flag.sum())),
    }


def _fp_breakdown_rows(df, group_cols):
    """Collapse a long-form (error_type, confidence, column_family, is_fp)
    table into per-combination FP/TP counts + implied precision."""
    if df.empty:
        return []
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fp_n = int(g['is_fp'].sum())
        tp_n = len(g) - fp_n
        rec = {c: (float(k) if c == 'confidence' else k) for c, k in zip(group_cols, keys)}
        rec['fp'] = fp_n
        rec['tp'] = tp_n
        rec['precision'] = round(tp_n / max(1, fp_n + tp_n), 4)
        rows.append(rec)
    return rows


def _row_identity_review_counts(df):
    """Count row-level identity review separately from cell predictions."""
    needed_col = 'row_identity_review_needed'
    event_col = 'row_identity_review_event_count'
    if needed_col not in df.columns or event_col not in df.columns or df.empty:
        return 0, 0
    needed = pd.to_numeric(df[needed_col], errors='coerce').fillna(0).eq(1)
    events = pd.to_numeric(df[event_col], errors='coerce').fillna(0).clip(lower=0)
    return int(needed.sum()), int(events.where(needed, 0).sum())


def evaluate_global_dataset(project_dir, num_companies):
    # ------------------------------------------------------------------
    # PHASE 1: Load all data
    # ------------------------------------------------------------------
    print("--- Phase 1: Loading CSVs ---")
    triplets = discover_company_files(
        os.path.join(project_dir, 'generated company'),
        corrected_dir=os.path.join(project_dir, 'ai corrected'),
    )
    if num_companies:
        triplets = triplets[:num_companies]
    # F3 follow-up: report the actual number of triplets scored, not the raw
    # --companies flag (which is now None by default = "score everything") -
    # downstream readers of accuracy_report.json expect a real integer here.
    n_companies_scored = len(triplets)

    print(f"\n{'='*60}")
    print(f"  GLOBAL ACCURACY EVALUATION - {project_dir} - {n_companies_scored} Companies")
    print(f"  Process memory: {_mem_mb()}")
    print(f"  Plot subsample cap: {MAX_PLOT_POINTS:,} pts (metrics use FULL data)")
    print(f"{'='*60}\n")

    all_pristine = []
    all_messy = []
    all_corrected = []

    for company_id, p_path, m_path, c_path in tqdm(triplets, desc="Loading CSVs"):
        all_pristine.append(pd.read_csv(p_path))
        all_messy.append(pd.read_csv(m_path))
        all_corrected.append(pd.read_csv(c_path))
        print(f"  {company_id} loaded - process memory: {_mem_mb()}")

    if not all_pristine:
        print("Error: No data found to visualize.")
        return

    df_pristine = pd.concat(all_pristine, ignore_index=True)
    del all_pristine
    df_messy = pd.concat(all_messy, ignore_index=True)
    del all_messy
    df_corrected = pd.concat(all_corrected, ignore_index=True)
    del all_corrected
    gc.collect()

    total_rows = len(df_pristine)
    row_identity_review_rows, row_identity_review_events = _row_identity_review_counts(df_corrected)
    print(f"\n  Total rows loaded: {total_rows:,} - process memory: {_mem_mb()}")

    output_dir = os.path.join(project_dir, 'accuracy test')
    shared.ensure_dir(output_dir)
    sns.set_theme(style="whitegrid")

    # Join key for the new per-product schema
    join_keys = ['company_id', 'product_id', 'reporting_month']

    # ------------------------------------------------------------------
    # PHASE 2: Per-target evaluation
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Generating 6-Panel Dashboards ---")

    for target_col in tqdm(TARGET_COLS, desc="Generating 6-Panel Dashboards"):

        corrected_col = f'{target_col}_corrected'
        error_type_col = f'{target_col}_error_type'
        confidence_col = f'{target_col}_confidence'

        # Check columns exist
        if corrected_col not in df_corrected.columns:
            print(f"  [SKIP] {target_col}: missing corrected column")
            continue
        if target_col not in df_pristine.columns:
            print(f"  [SKIP] {target_col}: missing in pristine data")
            continue

        # Build evaluation DataFrame
        eval_cols_pristine = join_keys + [target_col]
        eval_cols_corrected = join_keys + [corrected_col]
        anomaly_col = f'{target_col}_anomaly'
        review_flag_col = f'{target_col}_review_flag'
        if error_type_col in df_corrected.columns:
            eval_cols_corrected.append(error_type_col)
        if confidence_col in df_corrected.columns:
            eval_cols_corrected.append(confidence_col)
        if anomaly_col in df_corrected.columns:
            eval_cols_corrected.append(anomaly_col)
        if review_flag_col in df_corrected.columns:
            eval_cols_corrected.append(review_flag_col)

        df_eval = df_pristine[eval_cols_pristine].rename(
            columns={target_col: 'pristine_val'}
        )
        pristine_n = len(df_eval)
        df_eval = df_eval.merge(
            df_messy[join_keys + [target_col]].rename(columns={target_col: 'messy_val'}),
            on=join_keys
        )
        # F2: inner merges silently drop rows on join-key mismatch - loud warning
        # instead of a quietly-shrunk (but plausible-looking) evaluation set.
        if len(df_eval) != pristine_n:
            print(f"  [WARN] {target_col}: pristine-messy merge dropped rows "
                  f"({pristine_n:,} -> {len(df_eval):,}) - join keys misaligned")
        before_corrected_n = len(df_eval)
        df_eval = df_eval.merge(
            df_corrected[eval_cols_corrected],
            on=join_keys
        )
        if len(df_eval) != before_corrected_n:
            print(f"  [WARN] {target_col}: +corrected merge dropped rows "
                  f"({before_corrected_n:,} -> {len(df_eval):,}) - join keys misaligned")

        rename_map = {corrected_col: 'corrected_val'}
        if error_type_col in df_eval.columns:
            rename_map[error_type_col] = 'ai_detected_error'
        if confidence_col in df_eval.columns:
            rename_map[confidence_col] = 'confidence_score'
        if anomaly_col in df_eval.columns:
            rename_map[anomaly_col] = 'ai_detected_anomaly'
        if review_flag_col in df_eval.columns:
            rename_map[review_flag_col] = 'review_flag'
        df_eval.rename(columns=rename_map, inplace=True)

        # Fill missing columns with defaults
        if 'ai_detected_error' not in df_eval.columns:
            df_eval['ai_detected_error'] = 'OK'
        if 'confidence_score' not in df_eval.columns:
            df_eval['confidence_score'] = 1.0
        if 'ai_detected_anomaly' not in df_eval.columns:
            df_eval['ai_detected_anomaly'] = 0
        if 'review_flag' not in df_eval.columns:
            df_eval['review_flag'] = 0

        # Isolate corrupted rows
        df_eval['is_true_error'] = ~np.isclose(
            df_eval['pristine_val'], df_eval['messy_val'],
            rtol=1e-5, equal_nan=True
        )

        # Calculate Anomaly Detection Accuracy on the full dataset
        actual_anomaly = df_eval['is_true_error'].values
        ai_anomaly = (df_eval['ai_detected_anomaly'].values == 1)

        tp = int((actual_anomaly & ai_anomaly).sum())
        fp = int((~actual_anomaly & ai_anomaly).sum())
        tn = int((~actual_anomaly & ~ai_anomaly).sum())
        fn = int((actual_anomaly & ~ai_anomaly).sum())

        total_pts = len(df_eval)
        detection_accuracy = (tp + tn) / max(1, total_pts) * 100
        detection_precision = tp / max(1, tp + fp) * 100
        detection_recall = tp / max(1, tp + fn) * 100

        df_eval['error_before'] = np.abs(df_eval['messy_val'] - df_eval['pristine_val'])
        df_eval['error_after'] = np.abs(df_eval['corrected_val'] - df_eval['pristine_val'])

        # Calculate human flagging metrics on full df_eval before deletion
        # F5: "flagged" = HRR error_type OR review_flag==1, matching the
        # goal-metrics definition below so aggregate/per-column/goal numbers
        # are internally consistent.
        is_flagged = (df_eval['ai_detected_error'].values == 'Human Review Required') | (
            df_eval['review_flag'].values == 1
        )
        total_flagged = int(is_flagged.sum())
        tp_flagged = int((is_flagged & df_eval['is_true_error']).sum())
        fp_flagged = total_flagged - tp_flagged

        # An anomaly should have been flagged if it had a true error, wasn't restored,
        # and wasn't immaterial to begin with (F4b: restored != immaterial, see
        # _is_restored/_is_immaterial - rows from before either redefinition
        # (flat atol=1e-2, then the unsplit relative-tolerance version) aren't comparable).
        restored = _is_restored(df_eval['error_before'].values, df_eval['error_after'].values,
                                 df_eval['pristine_val'].values)
        immaterial = _is_immaterial(df_eval['error_before'].values, df_eval['pristine_val'].values)
        should_have_flagged = int((df_eval['is_true_error'] & ~restored & ~immaterial).sum())
        fn_flagged = should_have_flagged - tp_flagged
        
        flagging_recall = (tp_flagged / max(1, should_have_flagged)) * 100
        flagging_precision = (tp_flagged / max(1, total_flagged)) * 100

        corrupted_data = df_eval[df_eval['is_true_error']].copy()
        del df_eval
        gc.collect()

        if len(corrupted_data) == 0:
            continue

        # Recovery and residuals on FULL corrupted dataset
        corrupted_data['recovery_score'] = 1 - (
            corrupted_data['error_after'] /
            corrupted_data['error_before'].replace(0, np.nan)
        )
        corrupted_data['recovery_score'] = corrupted_data['recovery_score'].fillna(0).clip(lower=-1, upper=1)

        corrupted_data['residual_pct'] = (
            (corrupted_data['corrected_val'] - corrupted_data['pristine_val']) /
            corrupted_data['pristine_val'].replace(0, 1e-5)
        ) * 100

        # ALL METRICS ON FULL DATA
        total_errors = len(corrupted_data)
        perfect_recoveries = int(_is_restored(
            corrupted_data['error_before'].values, corrupted_data['error_after'].values,
            corrupted_data['pristine_val'].values
        ).sum())
        degraded_count = int((corrupted_data['recovery_score'].values < 0).sum())
        avg_recovery = corrupted_data['recovery_score'].mean() * 100 if total_errors > 0 else 0

        overconfident_mask = (
            (corrupted_data['recovery_score'].values < 0.75) &
            (corrupted_data['confidence_score'].values >= 0.90)
        )
        overconfident_errors = int(overconfident_mask.sum())
        overconfident_pct = (overconfident_errors / total_errors) * 100 if total_errors > 0 else 0

        recovery_by_type = (
            corrupted_data.groupby('ai_detected_error')['recovery_score']
            .mean()
            .reset_index()
        )

        print(f"  {target_col}: {total_errors:,} corrupted rows - memory: {_mem_mb()}")

        # VISUALIZATION
        plot_sample = _downsample(corrupted_data, MAX_PLOT_POINTS)

        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(
            f'AI Risk & Accuracy Diagnostics: {target_col}\n'
            f'(Full dataset: {total_errors:,} corrupted rows - '
            f'plots show {len(plot_sample):,} sampled points)',
            fontsize=18, weight='bold', y=0.98
        )

        # --- PLOT 1: Trajectory ---
        ax1 = plt.subplot(3, 2, 1)
        eps = 1e-3
        plot_traj = plot_sample[['pristine_val', 'messy_val', 'corrected_val']].copy()
        plot_traj = plot_traj.clip(lower=eps)

        ax1.scatter(plot_traj['pristine_val'], plot_traj['pristine_val'],
                    c='blue', label='Pristine Truth', alpha=0.4, s=15, zorder=3)
        ax1.scatter(plot_traj['pristine_val'], plot_traj['messy_val'],
                    c='red', label='Reported', marker='x', alpha=0.5, s=20, zorder=2)
        ax1.scatter(plot_traj['pristine_val'], plot_traj['corrected_val'],
                    c='green', label='AI Corrected', marker='+', s=40, zorder=4)

        x_vals = plot_traj['pristine_val'].values
        y_messy = plot_traj['messy_val'].values
        y_corr = plot_traj['corrected_val'].values
        segments = np.array([
            [[x, ym], [x, yc]]
            for x, ym, yc in zip(x_vals, y_messy, y_corr)
        ])
        if len(segments) > 0:
            lc = LineCollection(segments, colors='gray', linestyles='--', alpha=0.12, linewidths=0.5)
            ax1.add_collection(lc)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('Correction Trajectory', fontsize=14)
        ax1.legend(fontsize=9)
        del plot_traj

        # --- PLOT 2: Error Density ---
        ax2 = plt.subplot(3, 2, 2)
        kde_sample = _downsample(corrupted_data, MAX_PLOT_POINTS)
        # Add a small epsilon (1e-5) so exact 0.0 remaining errors (perfect restorations) can render on log-scale
        sns.kdeplot(kde_sample['error_before'] + 1e-5, color='red', fill=True,
                    label='Error (Reported)', ax=ax2, log_scale=True, warn_singular=False)
        sns.kdeplot(kde_sample['error_after'] + 1e-5, color='green', fill=True,
                    label='Remaining Error', ax=ax2, log_scale=True, warn_singular=False)
        ax2.set_title('Shift in Error Magnitude', fontsize=14)
        ax2.legend()

        # --- PLOT 3: Recovery by Class ---
        ax3 = plt.subplot(3, 2, 3)
        sns.barplot(x='recovery_score', y='ai_detected_error',
                    data=recovery_by_type.sort_values(by='recovery_score', ascending=False),
                    palette='viridis', ax=ax3)
        ax3.set_title('Average Recovery by Archetype', fontsize=14)
        ax3.set_xlim(-0.2, 1.1)
        ax3.axvline(x=0, color='red', linestyle='--')

        # --- PLOT 4: Confidence vs Actual Accuracy ---
        ax4 = plt.subplot(3, 2, 4)
        plot_conf = _downsample(corrupted_data, MAX_PLOT_POINTS)
        sns.scatterplot(x='confidence_score', y='recovery_score',
                        hue='ai_detected_error', data=plot_conf,
                        alpha=0.5, s=30, ax=ax4)
        ax4.set_title('AI Confidence vs. Actual Recovery Accuracy', fontsize=14)
        ax4.set_xlabel('AI Confidence Score (Probability)')
        ax4.set_ylabel('True Recovery Score (1.0 = Perfect)')
        ax4.axhline(0, color='red', linestyle='--', label='Data Degraded (< 0)')
        ax4.legend(loc='lower left', fontsize=8)

        # --- PLOT 5: Overcorrection / Residual Analysis ---
        ax5 = plt.subplot(3, 2, 5)
        plot_res = _downsample(corrupted_data, MAX_PLOT_POINTS).copy()
        plot_res['residual_pct'] = plot_res['residual_pct'].clip(lower=-200, upper=200)

        sns.scatterplot(x='pristine_val', y='residual_pct',
                        hue='ai_detected_error', data=plot_res,
                        alpha=0.5, s=30, ax=ax5)
        ax5.axhline(0, color='black', linestyle='-', linewidth=2, label='Perfect Match')
        ax5.axhline(10, color='red', linestyle='--', alpha=0.5, label='+/- 10% Bounds')
        ax5.axhline(-10, color='red', linestyle='--', alpha=0.5)
        ax5.set_xscale('log')
        ax5.set_title('Overcorrection Tracker (AI Hallucinations)', fontsize=14)
        ax5.set_xlabel('True Pristine Value (Log Scale)')
        ax5.set_ylabel('Residual Error % (Corrected vs True)')
        ax5.legend(loc='lower right', fontsize=8)

        # --- PLOT 6: Statistical Risk Summary ---
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')

        # Human review flagging rates
        total_ai_detections = int((corrupted_data['ai_detected_error'].values != 'OK').sum())
        flagged_for_review = int((corrupted_data['ai_detected_error'].values == 'Human Review Required').sum())

        print(f"=================== {target_col} METRICS ===================")
        print(f"--- Core Performance ---")
        print(f"Total Corrupted Rows: {total_errors:,}")
        print(f"Perfectly Restored:   {perfect_recoveries:,} ({(perfect_recoveries/max(1, total_errors))*100:.1f}%)")
        print(f"Average Recovery:     {avg_recovery:.1f}%")
        print(f"")
        print(f"--- Anomaly Detection Metrics ---")
        metrics_text = (
            f"Detection Accuracy:   {detection_accuracy:.1f}%\n"
            f"Detection Precision:  {detection_precision:.1f}% / Recall: {detection_recall:.1f}%\n\n"
            f"--- Flagging & Human Review ---\n"
            f"Should Have Flagged:  {should_have_flagged:,} (errors not restored)\n"
            f"Actually Flagged:     {total_flagged:,} (TP: {tp_flagged:,} / FP: {fp_flagged:,})\n"
            f"Missed (Silent Fail): {fn_flagged:,}\n"
            f"Flagging Precision:   {flagging_precision:.1f}% / Recall: {flagging_recall:.1f}%\n\n"
            f"--- Algorithmic Risk Metrics ---\n"
            f"Data Degraded by AI:  {degraded_count:,} ({degraded_count/total_errors*100:.1f}%)\n"
            f"  *(AI intervened but made the error worse)*\n"
            f"Overconfident Errors: {overconfident_errors:,} ({overconfident_pct:.1f}%)\n"
            f"  *(AI was >=90% confident but scored <75% accuracy)*\n"
        )
        print(f"\n=================== {target_col} METRICS ===================")
        print(metrics_text.strip())
        print("==========================================================\n")

        ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', alpha=0.8))  # type: ignore[call-overload]
        ax6.set_title('Global Summary & Risk Profile', fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        shared.ensure_dir(output_dir)
        output_img = os.path.join(output_dir, f'GLOBAL_ACCURACY_{target_col}.png')
        plt.savefig(output_img, dpi=200)
        plt.close(fig)
        plt.close('all')

        # Free per-column memory
        del corrupted_data, plot_sample, plot_conf, plot_res, kde_sample
        gc.collect()

    # ------------------------------------------------------------------
    # PHASE 3: Full-coverage metrics on EVERY corrupted column + JSON report
    # The 7 TARGET_COLS above get dashboards; errors are injected across ~900
    # columns, so recovery on the actual injected cells is measured here.
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Full-coverage accuracy report (all corrupted columns) ---")
    dfp = df_pristine.sort_values(join_keys).reset_index(drop=True)
    dfm = df_messy.sort_values(join_keys).reset_index(drop=True)
    dfc = df_corrected.sort_values(join_keys).reset_index(drop=True)
    goal_metrics = None  # populated below; printed as a PASS/FAIL table at the very end
    if not (dfp[join_keys].equals(dfm[join_keys]) and dfp[join_keys].equals(dfc[join_keys])):
        # F2: the goal-metrics table below is the actual product of this script.
        # Silently skipping it and exiting 0 leaves a folder that looks like a
        # completed run but has no report - hard-fail instead.
        print("  [FAIL] pristine/messy/corrected rows do not align on join keys - "
              f"cannot build the full-coverage report (pristine={len(dfp):,} rows, "
              f"messy={len(dfm):,} rows, corrected={len(dfc):,} rows)")
        sys.exit(1)
    else:
        meta_cols = set(join_keys) | {'dataset_type'}
        per_column = {}
        agg = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'n_corrupted': 0, 'perfect': 0,
               'degraded': 0, 'overconfident': 0, 'flagged_tp': 0,
               'eligible_flagged_tp': 0, 'flagged_total': 0,
               'should_flag': 0, 'recovery_sum': 0.0,
               'material_n': 0, 'material_detected': 0, 'material_perfect': 0,
               'material_degraded': 0, 'material_overconfident': 0,
               'material_recovery_sum': 0.0, 'negligible_n': 0}
        for col in dfp.columns:
            if col in meta_cols or not pd.api.types.is_numeric_dtype(dfp[col]):
                continue
            if col not in dfm.columns:
                continue
            p = dfp[col].to_numpy(dtype=float)
            m = dfm[col].to_numpy(dtype=float)
            is_err = ~np.isclose(p, m, rtol=1e-5, equal_nan=True)
            if not is_err.any():
                continue
            corr_col = f'{col}_corrected'
            c = dfc[corr_col].to_numpy(dtype=float) if corr_col in dfc.columns else m.copy()
            anom = (dfc[f'{col}_anomaly'].to_numpy() == 1) if f'{col}_anomaly' in dfc.columns else np.zeros(len(dfp), dtype=bool)
            errty = dfc[f'{col}_error_type'].astype(str).to_numpy() if f'{col}_error_type' in dfc.columns else np.full(len(dfp), 'OK', dtype=object)
            conf = dfc[f'{col}_confidence'].to_numpy(dtype=float) if f'{col}_confidence' in dfc.columns else np.ones(len(dfp))

            tp = int((is_err & anom).sum()); fp = int((~is_err & anom).sum())
            tn = int((~is_err & ~anom).sum()); fn = int((is_err & ~anom).sum())
            err_before = np.abs(m - p)
            err_after = np.abs(c - p)
            restored_mask = _is_restored(err_before, err_after, p)
            immaterial_mask = _is_immaterial(err_before, p)
            perfect_mask = restored_mask[is_err]
            perfect = int(perfect_mask.sum())
            with np.errstate(divide='ignore', invalid='ignore'):
                recovery = 1 - err_after[is_err] / np.where(err_before[is_err] == 0, np.nan, err_before[is_err])
            recovery = np.clip(np.nan_to_num(recovery, nan=0.0), -1, 1)
            degraded = int((recovery < 0).sum())
            overconf = int(((recovery < 0.75) & (conf[is_err] >= 0.90)).sum())
            # F5: "flagged" = HRR OR review_flag==1, unified with the goal-metrics
            # definition further down (previously this block counted HRR only).
            review_col = f'{col}_review_flag'
            flagged = (errty == 'Human Review Required')
            if review_col in dfc.columns:
                flagged = flagged | (dfc[review_col].to_numpy() == 1)
            not_restored = is_err & ~restored_mask & ~immaterial_mask
            eligible_flagged = flagged & not_restored

            # Materiality split: sub-noise perturbations (<2% relative) are
            # undetectable by any blind verifier and pollute avg_recovery /
            # overconfident if left mixed in with the 10x fat-finger errors.
            rel_before = err_before / np.maximum(np.abs(p), 1e-9)
            material = is_err & (rel_before > 0.02)
            material_of_err = rel_before[is_err] > 0.02  # aligned to is_err-indexed arrays
            n_material = int(material_of_err.sum())
            if n_material > 0:
                material_recovery = recovery[material_of_err]
                material_perfect = int(perfect_mask[material_of_err].sum())
                material_degraded = int((material_recovery < 0).sum())
                material_overconf = int(((material_recovery < 0.75) & (conf[is_err][material_of_err] >= 0.90)).sum())
                material_recovery_sum = float(material_recovery.sum())
            else:
                material_perfect = 0
                material_degraded = 0
                material_overconf = 0
                material_recovery_sum = 0.0
            material_detected = int((material & anom).sum())

            per_column[col] = {
                'n_corrupted': int(is_err.sum()),
                'detection': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                              'precision': round(tp / max(1, tp + fp), 4),
                              'recall': round(tp / max(1, tp + fn), 4)},
                'perfect_restored': perfect,
                'avg_recovery': round(float(recovery.mean()), 4),
                'degraded_by_ai': degraded,
                'overconfident': overconf,
                'flagged_for_review': int(flagged.sum()),
                'flagged_true_errors': int((flagged & is_err).sum()),
                'eligible_flagged_tp': int(eligible_flagged.sum()),
                'should_have_flagged': int(not_restored.sum()),
                'flagging_recall': round(eligible_flagged.sum() / max(1, not_restored.sum()), 4),
                'flag_metric_version': 'v2 eligible remaining errors',
                'material_errors': {
                    'n': n_material,
                    'perfect_restored': material_perfect,
                    'avg_recovery': round(material_recovery_sum / max(1, n_material), 4) if n_material > 0 else 0.0,
                    'degraded': material_degraded,
                    'overconfident': material_overconf,
                    'detected': material_detected,
                },
                'n_negligible': int(is_err.sum()) - n_material,
            }
            agg['tp'] += tp; agg['fp'] += fp; agg['tn'] += tn; agg['fn'] += fn
            agg['n_corrupted'] += int(is_err.sum()); agg['perfect'] += perfect
            agg['degraded'] += degraded; agg['overconfident'] += overconf
            agg['flagged_tp'] += int((flagged & is_err).sum())
            agg['eligible_flagged_tp'] += int(eligible_flagged.sum())
            agg['flagged_total'] += int(flagged.sum())
            agg['should_flag'] += int(not_restored.sum())
            agg['recovery_sum'] += float(recovery.sum())
            agg['material_n'] += n_material
            agg['material_detected'] += material_detected
            agg['material_perfect'] += material_perfect
            agg['material_degraded'] += material_degraded
            agg['material_overconfident'] += material_overconf
            agg['material_recovery_sum'] += material_recovery_sum
            agg['negligible_n'] += int(is_err.sum()) - n_material

        # --------------------------------------------------------------
        # PHASE 3a: Project-wide goal metrics - EVERY numeric column that
        # has an AI `{col}_anomaly` column, not just the ones with at
        # least one injected error (so true negatives on clean columns
        # count too). Definitions mirror the per-column pass above.
        # --------------------------------------------------------------
        goal = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'true_err': 0, 'n_cells': 0,
                'flagged_total': 0, 'flagged_tp': 0, 'eligible_flagged_tp': 0, 'flagged_fp': 0,
                'should_flag': 0, 'silent_fail': 0, 'perfect': 0, 'degraded': 0,
                'n_columns': 0,
                # F4b: restored/immaterial split - immaterial and should_flag_incl_immaterial
                # let a reader reconstruct the stricter pre-split denominator (see goal_metrics).
                'immaterial': 0, 'should_flag_incl_immaterial': 0,
                # E3: coverage/blind-spot accounting, only populated for columns
                # that actually carry a `{col}_status` (E1 output); left at 0 and
                # reported as absent keys entirely on older runs with no status data.
                'status_columns': 0, 'status_n_cells': 0,
                'status_verified': 0, 'status_unverified': 0, 'status_suspect': 0,
                'errors_in_unverified': 0}
        # False-positive breakdown: which detector (by {col}_error_type string)
        # produced which flagged cells, cross-tabbed with its confidence stamp
        # and column family. Collected alongside the goal-metrics pass below so
        # it costs one extra small DataFrame per column, not a second full scan.
        company_arr = (
            dfc['company_id'].to_numpy() if 'company_id' in dfc.columns
            else np.full(len(dfp), 'UNKNOWN', dtype=object)
        )
        fp_records = []
        for col in dfp.columns:
            if col in meta_cols or not pd.api.types.is_numeric_dtype(dfp[col]):
                continue
            anomaly_col = f'{col}_anomaly'
            if col not in dfm.columns or anomaly_col not in dfc.columns:
                continue

            p = dfp[col].to_numpy(dtype=float)
            m = dfm[col].to_numpy(dtype=float)
            corr_col = f'{col}_corrected'
            c = dfc[corr_col].to_numpy(dtype=float) if corr_col in dfc.columns else m.copy()
            true_err = ~np.isclose(p, m, rtol=1e-5, equal_nan=True)
            ai_anom = (dfc[anomaly_col].to_numpy() == 1)

            errty_col = f'{col}_error_type'
            review_col = f'{col}_review_flag'
            errty_str = (
                dfc[errty_col].astype(str).to_numpy() if errty_col in dfc.columns
                else np.full(len(dfp), 'OK', dtype=object)
            )
            flagged = errty_str == 'Human Review Required'
            if review_col in dfc.columns:
                flagged = flagged | (dfc[review_col].to_numpy() == 1)

            if flagged.any():
                # Attribution caveat (see _fp_breakdown_rows / report note):
                # detectors never overwrite a non-'OK' error_type, so a cell
                # flagged by two mechanisms is credited here to whichever
                # fired first ("first-flagger" attribution), not both.
                conf_col_name = f'{col}_confidence'
                conf_arr = (
                    dfc[conf_col_name].to_numpy(dtype=float) if conf_col_name in dfc.columns
                    else np.ones(len(dfp))
                )
                idx = np.flatnonzero(flagged)
                fp_records.append(pd.DataFrame({
                    'company': company_arr[idx],
                    'error_type': errty_str[idx],
                    'confidence': np.round(conf_arr[idx], 2),
                    'column_family': _column_family(col),
                    'is_fp': ~true_err[idx],
                }))

            err_before = np.abs(m - p)
            err_after = np.abs(c - p)
            restored = _is_restored(err_before, err_after, p)
            immaterial = _is_immaterial(err_before, p)
            should_flag = true_err & ~restored & ~immaterial

            with np.errstate(divide='ignore', invalid='ignore'):
                recovery = 1 - err_after[true_err] / np.where(
                    err_before[true_err] == 0, np.nan, err_before[true_err]
                )
            recovery = np.clip(np.nan_to_num(recovery, nan=0.0), -1, 1)

            goal['tp'] += int((true_err & ai_anom).sum())
            goal['fp'] += int((~true_err & ai_anom).sum())
            goal['tn'] += int((~true_err & ~ai_anom).sum())
            goal['fn'] += int((true_err & ~ai_anom).sum())
            goal['true_err'] += int(true_err.sum())
            goal['n_cells'] += len(p)
            goal['flagged_total'] += int(flagged.sum())
            goal['flagged_tp'] += int((flagged & true_err).sum())
            goal['eligible_flagged_tp'] += int((flagged & should_flag).sum())
            goal['flagged_fp'] += int((flagged & ~true_err).sum())
            goal['should_flag'] += int(should_flag.sum())
            goal['silent_fail'] += int((should_flag & ~flagged).sum())
            goal['perfect'] += int((true_err & restored).sum())
            goal['immaterial'] += int((true_err & immaterial).sum())
            goal['should_flag_incl_immaterial'] += int((true_err & ~restored).sum())
            goal['degraded'] += int((recovery < 0).sum())
            goal['n_columns'] += 1

            # E3: read the E1 three-state status where the AI output actually
            # carries it (older AI-corrected CSVs won't have it -- handled by
            # 'status_columns' staying 0 and the keys being omitted below).
            status_col = f'{col}_status'
            if status_col in dfc.columns:
                status_arr = dfc[status_col].astype(str).to_numpy()
                goal['status_columns'] += 1
                goal['status_n_cells'] += len(status_arr)
                goal['status_verified'] += int((status_arr == 'VERIFIED').sum())
                goal['status_unverified'] += int((status_arr == 'UNVERIFIED').sum())
                goal['status_suspect'] += int((status_arr == 'SUSPECT').sum())
                goal['errors_in_unverified'] += int((true_err & (status_arr == 'UNVERIFIED')).sum())

            del p, m, c, true_err, ai_anom, flagged, err_before, err_after, restored, immaterial, should_flag, recovery

        gc.collect()

        # Build the FP/TP breakdown from the records gathered in the loop above.
        # NOTE (attribution limitation): grouping is by whatever {col}_error_type
        # string actually sits on the flagged cell. Because a detector never
        # overwrites a non-'OK' error_type, a cell touched by two mechanisms is
        # attributed to whichever ran first - this is "first-flagger" attribution,
        # not full attribution. Treat it as a lead for investigation, not a
        # precise causal split.
        fp_df = (
            pd.concat(fp_records, ignore_index=True) if fp_records
            else pd.DataFrame(columns=['company', 'error_type', 'confidence', 'column_family', 'is_fp'])
        )
        fp_breakdown = {
            'note': (
                "First-flagger attribution only: a detector never overwrites a "
                "non-'OK' {col}_error_type, so a cell flagged by two mechanisms "
                "is credited to whichever fired first, not both."
            ),
            'project_wide': _fp_breakdown_rows(fp_df, ['error_type', 'confidence', 'column_family']),
            'by_company': {
                str(company): _fp_breakdown_rows(g, ['error_type', 'confidence', 'column_family'])
                for company, g in fp_df.groupby('company', dropna=False)
            },
        }
        del fp_records, fp_df
        gc.collect()

        goal_metrics = {
            'n_cells': goal['n_cells'],
            'n_columns_evaluated': goal['n_columns'],
            'true_errors': goal['true_err'],
            'tp': goal['tp'], 'fp': goal['fp'], 'tn': goal['tn'], 'fn': goal['fn'],
            'flagged_total': goal['flagged_total'],
            'flagged_tp': goal['flagged_tp'],
            'eligible_flagged_tp': goal['eligible_flagged_tp'],
            'should_flag': goal['should_flag'],
            'perfect_restored': goal['perfect'],
            # F4b: transparency pair so old (pre-split) scoreboard rows can be
            # reconstructed - should_flag above already excludes immaterial errors.
            'immaterial_errors': goal['immaterial'],
            'should_flag_incl_immaterial': goal['should_flag_incl_immaterial'],
            'degraded': goal['degraded'],
            'detection_precision': round(goal['tp'] / max(1, goal['tp'] + goal['fp']) * 100, 2),
            'detection_recall': round(goal['tp'] / max(1, goal['tp'] + goal['fn']) * 100, 2),
            'detection_accuracy': round((goal['tp'] + goal['tn']) / max(1, goal['n_cells']) * 100, 2),
            'tp_flag_rate': round(goal['eligible_flagged_tp'] / max(1, goal['should_flag']) * 100, 2),
            'flag_metric_version': 'v2 eligible remaining errors',
            'fp_flag_rate': round(goal['flagged_fp'] / max(1, goal['flagged_total']) * 100, 2),
            'silent_fail_rate': round(goal['silent_fail'] / max(1, goal['should_flag']) * 100, 2),
            'degraded_rate': round(goal['degraded'] / max(1, goal['true_err']) * 100, 2),
        }

        # E3: coverage / blind-spot metrics -- only when this run's AI-corrected
        # data actually has `_status` columns (E1). Omitted entirely (not zeroed)
        # on older runs so a missing signal never masquerades as "100% verified".
        if goal['status_columns'] > 0:
            goal_metrics['verified_pct'] = round(goal['status_verified'] / max(1, goal['status_n_cells']) * 100, 2)
            goal_metrics['unverified_pct'] = round(goal['status_unverified'] / max(1, goal['status_n_cells']) * 100, 2)
            goal_metrics['suspect_pct'] = round(goal['status_suspect'] / max(1, goal['status_n_cells']) * 100, 2)
            goal_metrics['errors_in_unverified'] = goal['errors_in_unverified']
            goal_metrics['errors_in_unverified_pct'] = round(
                goal['errors_in_unverified'] / max(1, goal['true_err']) * 100, 2
            )

        aggregate = {
            'columns_with_errors': len(per_column),
            'total_corrupted_cells': agg['n_corrupted'],
            'row_identity_review_rows': row_identity_review_rows,
            'row_identity_review_events': row_identity_review_events,
            'row_identity_review_total_rows': total_rows,
            'detection_precision': round(agg['tp'] / max(1, agg['tp'] + agg['fp']), 4),
            'detection_recall': round(agg['tp'] / max(1, agg['tp'] + agg['fn']), 4),
            'perfect_restored': agg['perfect'],
            'avg_recovery': round(agg['recovery_sum'] / max(1, agg['n_corrupted']), 4),
            'degraded_by_ai': agg['degraded'],
            'overconfident': agg['overconfident'],
            'flagging_precision': round(agg['flagged_tp'] / max(1, agg['flagged_total']), 4),
            'eligible_flagged_tp': agg['eligible_flagged_tp'],
            'flagging_recall': round(agg['eligible_flagged_tp'] / max(1, agg['should_flag']), 4),
            'flag_metric_version': 'v2 eligible remaining errors',
            'material_errors': {
                'n': agg['material_n'],
                'detection_recall': round(agg['material_detected'] / max(1, agg['material_n']), 4),
                'perfect_restored': agg['material_perfect'],
                'avg_recovery': round(agg['material_recovery_sum'] / max(1, agg['material_n']), 4),
                'degraded': agg['material_degraded'],
                'overconfident': agg['material_overconfident'],
            },
            'negligible_errors_n': agg['negligible_n'],
        }
        report = {
            'project': str(project_dir),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'num_companies': n_companies_scored,
            'aggregate': aggregate,
            'goal_metrics': goal_metrics,
            'fp_breakdown': fp_breakdown,
            'columns': per_column,
        }
        report_path = os.path.join(output_dir, 'accuracy_report.json')
        shared.atomic_write_json(report, report_path, indent=2)
        print(f"  Machine-readable report written to {report_path}")
        print(f"  AGGREGATE ({aggregate['columns_with_errors']} corrupted columns, "
              f"{aggregate['total_corrupted_cells']} corrupted cells):")
        print(f"    Detection precision: {aggregate['detection_precision']*100:.1f}% / "
              f"recall: {aggregate['detection_recall']*100:.1f}%")
        print(f"    Avg recovery: {aggregate['avg_recovery']*100:.1f}% | perfect: {aggregate['perfect_restored']} | "
              f"degraded: {aggregate['degraded_by_ai']} | overconfident: {aggregate['overconfident']}")
        print(f"    Flagging precision: {aggregate['flagging_precision']*100:.1f}% / "
              f"recall: {aggregate['flagging_recall']*100:.1f}%")
        print(f"    Row identity review: {row_identity_review_rows}/{total_rows} rows, "
              f"{row_identity_review_events} identity events (outside cell metrics)")
        print(f"    Material errors (>2% off): {aggregate['material_errors']['n']} | "
              f"recall: {aggregate['material_errors']['detection_recall']*100:.1f}% | "
              f"avg recovery: {aggregate['material_errors']['avg_recovery']*100:.1f}% | "
              f"degraded: {aggregate['material_errors']['degraded']} | "
              f"overconfident: {aggregate['material_errors']['overconfident']}")

    if goal_metrics is not None:
        gm = goal_metrics
        print(f"\n=== GOAL METRICS (all {gm['n_columns_evaluated']} numeric columns, "
              f"{gm['n_cells']:,} cells) ===")
        for label, value, target, op in [
            ('detection accuracy', gm['detection_accuracy'], 95, '>'),
            ('TP flag', gm['tp_flag_rate'], 75, '>='),
            ('FP flag', gm['fp_flag_rate'], 5, '<'),
            ('silent fail', gm['silent_fail_rate'], 5, '<'),
            ('degraded', gm['degraded_rate'], 5, '<'),
        ]:
            passed = value >= target if op == '>=' else value < target
            print(f"  {label:<20} {value:>6.2f}%   target {op}{target}%    {'PASS' if passed else 'FAIL'}")

        # E3: coverage disclosure -- explicitly not a target, kept separate from
        # the PASS/FAIL table above so it can't be mistaken for one of the 5 goals.
        if 'verified_pct' in gm:
            print("  --- coverage (not a target) ---")
            print(f"  verified {gm['verified_pct']:.1f}% | unverified {gm['unverified_pct']:.1f}% | "
                  f"suspect {gm['suspect_pct']:.1f}%")
            print(f"  true errors sitting in UNVERIFIED cells: {gm['errors_in_unverified']:,} "
                  f"({gm['errors_in_unverified_pct']:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  All 6-panel dashboards saved to '{output_dir}'")
    print(f"  Final process memory: {_mem_mb()}")
    print(f"{'='*60}")

def _accuracy_done(project_dir):
    # F1: doneness = the JSON report exists, not just "the dir has something in
    # it" — dashboards (Phase 2) are written before accuracy_report.json (Phase
    # 3), so a crashed run used to leave a dir that looked done but wasn't.
    accuracy_dir = os.path.join(project_dir, 'accuracy test')
    return os.path.isfile(os.path.join(accuracy_dir, 'accuracy_report.json'))


def run_all():
    """Process the next pending project (has AI-corrected data but no accuracy
    test yet). With --all, keep going until no pending project remains."""
    while True:
        project_dir = shared.find_next_pending('.', 'ai corrected', '_AI_corrected.csv', 'accuracy test', _accuracy_done)
        if project_dir is None:
            print("No pending projects to accuracy-test.")
            return
        evaluate_global_dataset(project_dir, num_companies=NUM_COMPANIES)
        if not PROCESS_ALL_PROJECTS:
            remaining = shared.find_next_pending('.', 'ai corrected', '_AI_corrected.csv', 'accuracy test', _accuracy_done)
            if remaining:
                print(f"Done with {project_dir}. Another pending project ({remaining}) was found — "
                      f"run again, or pass --all to process every pending project in one run.")
            return


if __name__ == "__main__":
    run_all()
