#!/usr/bin/env python
"""
03_optimize_blend.py

Revised to:
1) Pass feature names to the classifier to avoid the "invalid feature names" warning.
2) Provide clearer, sectioned output with results for:
   - Best Cost Blend
   - Best Near-Limit Blend
   - Average Feasible Blend (approx)
   - A final summary of cost savings.
"""

import pandas as pd
import numpy as np
import joblib
import math

# ------------------------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------------------------
CLASSIFIER_MODEL_FILE = "models/model_inSpec_classifier.joblib"

COMPONENT_DATA_CSV    = "data/component_data_all.csv"

N_CANDIDATES    = 100000
TOTAL_BLEND_VOL = 70000.0
RANDOM_SEED     = 42

# SCENARIO
TARGET_DATE          = "1/1/2024"
TARGET_GASOLINE_TYPE = "regular"  # or "premium"

# Specs
RVP_SUMMER     = 7.0
RVP_WINTER     = 13.0
OCTANE_REGULAR = 83.0
OCTANE_PREMIUM = 88.0

# Components in the same order used for classifier training
COMPONENTS_LIST = [
    "Reformate",
    "LHCN/HHC",
    "LCN",
    "CYHEX/SAT C6",
    "Alkylate",
    "CC5/Dimate",
    "SAT C5",
    "NC4"
]

# The column names that the classifier was trained on, in the exact same order:
CLASSIFICATION_FEATURES = [
    "ReformateVol", "LHCN/HHCVol", "LCNVol", "CYHEX/SATC6Vol",
    "AlkylateVol", "CC5/DimateVol", "SATC5Vol", "NC4Vol",
    "GasolineTypeEncoded", "SeasonEncoded"
]

# ------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def determine_season(date_str):
    date = pd.to_datetime(date_str, format="%m/%d/%Y")
    m, d = date.month, date.day
    if (m > 5 and m < 10) or (m == 5 and d >= 1) or (m == 9 and d <= 30):
        return "summer"
    else:
        return "winter"

def load_daily_component_info(date_str, df_wide):
    row = df_wide.loc[df_wide["Date"] == date_str]
    if row.empty:
        raise ValueError(f"No data found for date {date_str} in CSV.")
    row = row.iloc[0]

    comp_dict = {}
    for comp in COMPONENTS_LIST:
        vol_col  = f"{comp} Tank Volume (BBL)"
        oct_col  = f"{comp} Octane"
        rvp_col  = f"{comp} RVP (PSI)"
        sulf_col = f"{comp} Sulfur (WPPM)"
        aro_col  = f"{comp} Aromatics (VOL %)"
        ben_col  = f"{comp} Benzene (VOL %)"
        ole_col  = f"{comp} Olefins (VOL %)"
        cst_col  = f"{comp} cost ($/bbl)"

        comp_dict[comp] = {
            "TankVolume": row.get(vol_col, np.nan),
            "Octane":     row.get(oct_col,  np.nan),
            "RVP":        row.get(rvp_col,  np.nan),
            "Sulfur":     row.get(sulf_col, np.nan),
            "Aromatics":  row.get(aro_col,  np.nan),
            "Benzene":    row.get(ben_col,  np.nan),
            "Olefins":    row.get(ole_col,  np.nan),
            "Cost":       row.get(cst_col,  np.nan),
        }
    return comp_dict

def make_classifier_input(volumes, gas_type, season):
    """
    Builds a single-row DataFrame with columns=CLASSIFICATION_FEATURES,
    matching the training column order. This avoids the "X does not have valid feature names" warning.
    """
    gas_type_enc = 0 if gas_type == "regular" else 1
    season_enc   = 0 if season == "winter" else 1

    data_dict = {
        "ReformateVol":  volumes[0],
        "LHCN/HHCVol":   volumes[1],
        "LCNVol":        volumes[2],
        "CYHEX/SATC6Vol":volumes[3],
        "AlkylateVol":   volumes[4],
        "CC5/DimateVol": volumes[5],
        "SATC5Vol":      volumes[6],
        "NC4Vol":        volumes[7],
        "GasolineTypeEncoded": gas_type_enc,
        "SeasonEncoded":       season_enc
    }
    # Make a single-row DataFrame
    X_df = pd.DataFrame([data_dict], columns=CLASSIFICATION_FEATURES)
    return X_df

def calculate_final_specs(volumes, comp_dict):
    """
    Direct linear mixing approximation for final specs & cost.
    """
    total_vol = volumes.sum()
    frac = volumes / total_vol if total_vol > 0 else np.zeros_like(volumes)

    final_octane     = 0.0
    final_rvp        = 0.0
    final_sulfur     = 0.0
    final_aromatics  = 0.0
    final_benzene    = 0.0
    final_olefins    = 0.0
    total_cost       = 0.0

    for idx, comp in enumerate(COMPONENTS_LIST):
        props = comp_dict[comp]
        w = frac[idx]

        octv = props["Octane"]    if pd.notnull(props["Octane"])    else 0
        rvpv = props["RVP"]       if pd.notnull(props["RVP"])       else 0
        sulf = props["Sulfur"]    if pd.notnull(props["Sulfur"])    else 0
        aro  = props["Aromatics"] if pd.notnull(props["Aromatics"]) else 0
        ben  = props["Benzene"]   if pd.notnull(props["Benzene"])   else 0
        ole  = props["Olefins"]   if pd.notnull(props["Olefins"])   else 0
        cst  = props["Cost"]      if pd.notnull(props["Cost"])      else 0

        final_octane    += w * octv
        final_rvp       += w * rvpv
        final_sulfur    += w * sulf
        final_aromatics += w * aro
        final_benzene   += w * ben
        final_olefins   += w * ole

        total_cost      += volumes[idx] * cst

    cost_per_bbl = total_cost / total_vol if total_vol > 0 else np.nan

    return {
        "Octane":     final_octane,
        "RVP":        final_rvp,
        "Sulfur":     final_sulfur,
        "Aromatics":  final_aromatics,
        "Benzene":    final_benzene,
        "Olefins":    final_olefins,
        "Cost":       cost_per_bbl
    }

# ------------------------------------------------------------------------------
# 3) MAIN
# ------------------------------------------------------------------------------
def main():
    np.random.seed(RANDOM_SEED)

    # 1) Load classifier
    inSpec_clf = joblib.load(CLASSIFIER_MODEL_FILE)

    # 2) Load day data
    df_wide    = pd.read_csv(COMPONENT_DATA_CSV)
    season_str = determine_season(TARGET_DATE)
    print(f"Target: date={TARGET_DATE}, gasType={TARGET_GASOLINE_TYPE}, season={season_str}")

    if season_str == "summer":
        rvp_limit = RVP_SUMMER
    else:
        rvp_limit = RVP_WINTER

    if TARGET_GASOLINE_TYPE == "regular":
        octane_limit = OCTANE_REGULAR
    else:
        octane_limit = OCTANE_PREMIUM

    # 3) Build component info
    day_components = load_daily_component_info(TARGET_DATE, df_wide)

    # Bookkeeping
    best_cost     = float("inf")
    best_cost_vol = None

    best_margin_dist = float("inf")
    best_margin_vol  = None

    feasible_vols   = []  # store volumes for all feasible in-spec
    feasible_costs  = []  # store cost for all feasible in-spec

    # 4) Random search
    for _ in range(N_CANDIDATES):
        volumes = np.random.rand(len(COMPONENTS_LIST))
        volumes = volumes / volumes.sum() * TOTAL_BLEND_VOL

        # leftover check
        leftover_ok = True
        for i, comp in enumerate(COMPONENTS_LIST):
            tv = day_components[comp]["TankVolume"]
            if pd.isnull(tv) or tv <= 0:
                leftover_ok = False
                break
            if volumes[i] > 0.75 * tv:
                leftover_ok = False
                break
        if not leftover_ok:
            continue

        # classifier check
        X_df = make_classifier_input(volumes, TARGET_GASOLINE_TYPE, season_str)
        pred_in_spec = inSpec_clf.predict(X_df)[0]
        if pred_in_spec == 0:
            continue

        # direct mixing
        specs = calculate_final_specs(volumes, day_components)
        if specs["Octane"] < octane_limit:
            continue
        if specs["RVP"] > rvp_limit:
            continue

        # truly feasible & in-spec
        final_cost = specs["Cost"]
        feasible_vols.append(volumes)
        feasible_costs.append(final_cost)

        # best-cost check
        if final_cost < best_cost:
            best_cost     = final_cost
            best_cost_vol = volumes.copy()

        # near-limit check
        oct_margin  = specs["Octane"] - octane_limit
        rvp_margin  = specs["RVP"]    - rvp_limit
        margin_dist = math.sqrt(oct_margin**2 + rvp_margin**2)

        if margin_dist < best_margin_dist:
            best_margin_dist = margin_dist
            best_margin_vol  = volumes.copy()

    # 5) Evaluate results
    feasible_count = len(feasible_costs)
    print(f"\nFound {feasible_count} feasible in-spec blends out of {N_CANDIDATES} candidates.")
    if feasible_count == 0:
        print("No feasible in-spec blend found! Exiting.")
        return

    avg_cost = np.mean(feasible_costs)
    # Build an "average volumes" array (component-wise mean) to see an approximate "average feasible blend"
    avg_volumes = np.mean(np.array(feasible_vols), axis=0)

    # 6) Compute final specs for each special blend
    # 6a) BEST COST BLEND
    best_cost_specs = calculate_final_specs(best_cost_vol, day_components)
    bc_oct_margin   = best_cost_specs["Octane"] - octane_limit
    bc_rvp_margin   = rvp_limit - best_cost_specs["RVP"]  # how much below RVP limit
    # 6b) BEST NEAR-LIMIT BLEND
    near_specs      = calculate_final_specs(best_margin_vol, day_components)
    near_cost       = near_specs["Cost"]
    nl_oct_margin   = near_specs["Octane"] - octane_limit
    nl_rvp_margin   = rvp_limit - near_specs["RVP"]
    # 6c) AVERAGE FEASIBLE BLEND (approx)
    avg_specs  = calculate_final_specs(avg_volumes, day_components)

    # But check leftover feasibility for the average volumes
    avg_leftover_ok = True
    for i, comp in enumerate(COMPONENTS_LIST):
        tv = day_components[comp]["TankVolume"]
        if pd.isnull(tv) or tv <= 0 or avg_volumes[i] > 0.75 * tv:
            avg_leftover_ok = False
            break

    # 7) Summaries
    # 7a) Print Best Cost Blend
    print("\n==================== BEST COST BLEND ====================")
    print("Volumes (BBL):")
    for comp, vol in zip(COMPONENTS_LIST, best_cost_vol):
        print(f"  {comp}: {vol:,.2f}")
    print(f"\nFinal Blend Specs (Direct Mixing):")
    print(f"  Cost:    ${best_cost_specs['Cost']:.2f} per bbl")
    print(f"  Octane:  {best_cost_specs['Octane']:.2f} (limit={octane_limit}) => margin=+{bc_oct_margin:.2f}")
    print(f"  RVP:     {best_cost_specs['RVP']:.2f} (limit={rvp_limit}) => margin=+{bc_rvp_margin:.2f} from limit")
    print(f"  Sulfur:  {best_cost_specs['Sulfur']:.2f} ppm")
    print(f"  Benzene: {best_cost_specs['Benzene']:.2f}%")
    print(f"  Aromatics: {best_cost_specs['Aromatics']:.2f}%")
    print(f"  Olefins: {best_cost_specs['Olefins']:.2f}%")

    # 7b) Print Best Near-Limit Blend
    print("\n================== BEST NEAR-LIMIT BLEND =================")
    print("Volumes (BBL):")
    for comp, vol in zip(COMPONENTS_LIST, best_margin_vol):
        print(f"  {comp}: {vol:,.2f}")
    print("\nFinal Blend Specs (Direct Mixing):")
    print(f"  Cost:    ${near_specs['Cost']:.2f} per bbl")
    print(f"  Octane:  {near_specs['Octane']:.2f} (limit={octane_limit}) => margin=+{nl_oct_margin:.2f}")
    print(f"  RVP:     {near_specs['RVP']:.2f} (limit={rvp_limit}) => margin=+{nl_rvp_margin:.2f} from limit")
    print(f"  Sulfur:  {near_specs['Sulfur']:.2f} ppm")
    print(f"  Benzene: {near_specs['Benzene']:.2f}%")
    print(f"  Aromatics: {near_specs['Aromatics']:.2f}%")
    print(f"  Olefins: {near_specs['Olefins']:.2f}%")

    # 7c) Print Approximate Average Feasible Blend
    print("\n================ AVERAGE FEASIBLE BLEND =================")
    print("Note: This is simply the arithmetic mean of volumes from all feasible blends.\n"
          "It may or may not itself be leftover-feasible (we check below).")
    for comp, vol in zip(COMPONENTS_LIST, avg_volumes):
        print(f"  {comp}: {vol:,.2f}")
    print(f"\nLeftover-Feasible? {'Yes' if avg_leftover_ok else 'No'}")
    print("Final Blend Specs (Direct Mixing):")
    print(f"  Cost:    ${avg_specs['Cost']:.2f} per bbl")
    print(f"  Octane:  {avg_specs['Octane']:.2f}")
    print(f"  RVP:     {avg_specs['RVP']:.2f}")
    print(f"  Sulfur:  {avg_specs['Sulfur']:.2f} ppm")
    print(f"  Benzene: {avg_specs['Benzene']:.2f}%")
    print(f"  Aromatics: {avg_specs['Aromatics']:.2f}%")
    print(f"  Olefins: {avg_specs['Olefins']:.2f}%")

    # 8) Final Summary & Savings
    # (A) best cost vs average feasible
    savings_vs_avg = ((avg_cost - best_cost) / avg_cost * 100) if avg_cost > 0 else float("nan")
    # (B) best cost vs near-limit
    if near_cost > 0:
        savings_vs_nearlimit = ((near_cost - best_cost) / near_cost * 100)
    else:
        savings_vs_nearlimit = float('nan')

    print("\n====================== SUMMARY ==========================")
    print(f"Total feasible blends found: {feasible_count}")
    print(f"Average feasible cost: ${avg_cost:.2f}")
    print(f"Best cost blend cost:  ${best_cost_specs['Cost']:.2f}")
    print(f"Near-limit blend cost: ${near_cost:.2f}")
    print(f"\nCost Savings of Best-Cost vs. Average Feasible:  {savings_vs_avg:.1f}%")
    print(f"Cost Savings of Best-Cost vs. Near-Limit Blend: {savings_vs_nearlimit:.1f}%")
    print("=========================================================")

if __name__ == "__main__":
    main()
