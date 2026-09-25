import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import argparse
import glob
import os
from tqdm import tqdm  # type: ignore
import shared
from shared import discover_company_files
import warnings
warnings.filterwarnings('ignore')

# --- CLI ARGUMENT PARSING ---
_parser = argparse.ArgumentParser(description='Illustrative emissions-intensity scoring (product-level schema)')
_parser.add_argument('--companies', type=int, default=3, help='Number of companies to evaluate')
_args, _ = _parser.parse_known_args()

NUM_COMPANIES = _args.companies
# ---------------------------------------------------

# --- CONFIGURATION ---
EMISSION_COLS = [
    'supplier_emissions_mtco2',
    'waste_emissions_mtco2',
    'utility_emissions_mtco2',
]
# ---------------------


def calculate_global_esg_impact(num_companies):
    print(f"\n--- Aggregating illustrative emissions-intensity scores across {num_companies} companies ---")

    # Read-only report, no output artifact of its own to check for "done" - just
    # use the most recently completed project (highest-numbered with AI-corrected
    # data available).
    project_dirs = [p for p in shared.list_project_dirs('.')
                     if glob.glob(os.path.join(p, 'ai corrected', 'COMP_*_AI_corrected.csv'))]
    if not project_dirs:
        print("No project with AI-corrected data found.")
        return
    project_dir = project_dirs[-1]
    print(f"Using {project_dir}")

    triplets = discover_company_files(
        os.path.join(project_dir, 'generated company'),
        corrected_dir=os.path.join(project_dir, 'ai corrected'),
    )
    if num_companies:
        triplets = triplets[:num_companies]

    all_pristine = []
    all_messy = []
    all_corrected = []

    for company_id, p_path, m_path, c_path in tqdm(triplets, desc="Loading CSVs"):
        all_pristine.append(pd.read_csv(p_path))
        all_messy.append(pd.read_csv(m_path))
        all_corrected.append(pd.read_csv(c_path))

    if not all_pristine:
        print("No data found to evaluate.")
        return

    df_pristine = pd.concat(all_pristine, ignore_index=True)
    df_messy = pd.concat(all_messy, ignore_index=True)
    df_corrected = pd.concat(all_corrected, ignore_index=True)

    # Aggregate product-level rows to company level
    agg_cols = {'production_units': 'sum', 'gen_total_revenue_usd': 'first'}
    for col in EMISSION_COLS:
        agg_cols[col] = 'sum'

    df_p = df_pristine.groupby('company_id').agg(agg_cols).reset_index()
    df_m = df_messy.groupby('company_id').agg(agg_cols).reset_index()

    # For corrected, use the _corrected columns
    corr_agg = {'production_units': 'sum', 'gen_total_revenue_usd': 'first'}
    for col in EMISSION_COLS:
        corrected_col = f'{col}_corrected'
        if corrected_col in df_corrected.columns:
            corr_agg[corrected_col] = 'sum'
        else:
            corr_agg[col] = 'sum'
    df_c = df_corrected.groupby('company_id').agg(corr_agg).reset_index()

    # Build company-level evaluation DataFrame
    df = df_p[['company_id', 'gen_total_revenue_usd', 'production_units']].copy()

    df['true_emissions'] = sum(df_p[col] for col in EMISSION_COLS)
    df['reported_emissions'] = sum(df_m[col] for col in EMISSION_COLS)

    df['corrected_emissions'] = 0.0
    for col in EMISSION_COLS:
        corrected_col = f'{col}_corrected'
        if corrected_col in df_c.columns:
            df['corrected_emissions'] += df_c[corrected_col]
        elif col in df_c.columns:
            df['corrected_emissions'] += df_c[col]

    # --- DUAL INTENSITY SCORING ---
    # 1. Revenue-based intensity
    df['revenue_millions'] = df['gen_total_revenue_usd'] / 1_000_000
    df['revenue_millions'] = df['revenue_millions'].replace(0, 0.001)

    df['true_rev_intensity'] = df['true_emissions'] / df['revenue_millions']
    df['reported_rev_intensity'] = df['reported_emissions'] / df['revenue_millions']
    df['corrected_rev_intensity'] = df['corrected_emissions'] / df['revenue_millions']

    # 2. Production-based intensity (efficiency score)
    safe_prod = df['production_units'].replace(0, 1)
    df['true_prod_intensity'] = df['true_emissions'] / safe_prod
    df['reported_prod_intensity'] = df['reported_emissions'] / safe_prod
    df['corrected_prod_intensity'] = df['corrected_emissions'] / safe_prod

    # --- PROJECT 0.0 TO 5.0 TRANSFORM (revenue-based) ---
    global_mean_intensity = df['true_rev_intensity'].mean()

    def generate_ftse_score(intensity):
        if pd.isna(intensity): return 0.0
        score = 5.0 - ((intensity / global_mean_intensity) * 2.5)
        return round(max(0.0, min(5.0, score)), 1)  # type: ignore[call-overload]

    df['true_esg'] = df['true_rev_intensity'].apply(generate_ftse_score)
    df['reported_esg'] = df['reported_rev_intensity'].apply(generate_ftse_score)
    df['corrected_esg'] = df['corrected_rev_intensity'].apply(generate_ftse_score)

    df['is_over_reported'] = df['reported_emissions'] > (df['true_emissions'] * 1.01)
    df['is_under_reported'] = df['reported_emissions'] < (df['true_emissions'] * 0.99)

    impacted = df[df['is_over_reported'] | df['is_under_reported']]
    over_reporters = df[df['is_over_reported']]
    under_reporters = df[df['is_under_reported']]

    # --- SCORE MOVEMENT ANALYSIS ---
    df['score_movement'] = df['corrected_esg'] - df['reported_esg']
    scored_higher = (df['score_movement'] > 0).sum()
    scored_lower  = (df['score_movement'] < 0).sum()
    unchanged     = (df['score_movement'] == 0).sum()

    # --- PRODUCTION EFFICIENCY METRICS ---
    mean_true_eff = df['true_prod_intensity'].mean()
    mean_reported_eff = df['reported_prod_intensity'].mean()
    mean_corrected_eff = df['corrected_prod_intensity'].mean()

    print("\n=======================================================")
    print("      ILLUSTRATIVE EMISSIONS INTENSITY REPORT          ")
    print("=======================================================")
    print(f"Total Companies Evaluated:               {len(df):,}")
    print(f"Companies with >1% aggregate emissions discrepancy: {len(impacted):,}")
    print(f"  -> Reported aggregate >1% above pristine: {len(over_reporters):,}")
    print(f"  -> Reported aggregate >1% below pristine: {len(under_reporters):,}")
    print("-------------------------------------------------------")
    print(f"Average Pristine-Based Illustrative Score: {df['true_esg'].mean():.1f} / 5.0")
    print(f"Average Reported Illustrative Score:       {df['reported_esg'].mean():.1f} / 5.0")
    print(f"Average Model-Output Illustrative Score:   {df['corrected_esg'].mean():.1f} / 5.0")
    print("-------------------------------------------------------")
    print("PRODUCTION EFFICIENCY (mtCO2e per production unit):")
    print(f"  True Average:      {mean_true_eff:.6f}")
    print(f"  Reported Average:  {mean_reported_eff:.6f}")
    print(f"  Corrected Average: {mean_corrected_eff:.6f}")
    print("-------------------------------------------------------")
    print("MODEL OUTPUT - ILLUSTRATIVE SCORE MOVEMENT (ALL COMPANIES):")
    print(f"  Companies that scored HIGHER after correction:    {scored_higher:,}  ({scored_higher/len(df)*100:.1f}%)")
    print(f"  Companies that scored LOWER  after correction:    {scored_lower:,}  ({scored_lower/len(df)*100:.1f}%)")
    print(f"  Companies with NO CHANGE in score:                {unchanged:,}  ({unchanged/len(df)*100:.1f}%)")
    print("  (Higher score = lower revenue-based emissions intensity relative to this sample)")
    print("-------------------------------------------------------")

    if len(under_reporters) > 0:
        print("REPORTED EMISSIONS BELOW PRISTINE AGGREGATE BY >1%:")
        print(f"  Average reported illustrative score:     {under_reporters['reported_esg'].mean():.1f}")
        print(f"  Average model-output illustrative score: {under_reporters['corrected_esg'].mean():.1f}")
        print("-------------------------------------------------------")

    print("This project score is a capped 0-5 transform of revenue-based emissions intensity")
    print("relative to this synthetic sample's pristine mean; it is not an external rating.")
    print("Production-based intensity is reported separately; model output is not ground truth.")

if __name__ == "__main__":
    calculate_global_esg_impact(NUM_COMPANIES)
