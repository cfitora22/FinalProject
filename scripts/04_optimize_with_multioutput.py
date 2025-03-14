#!/usr/bin/env python
"""
04_optimize_with_multioutput.py

Demonstrates using model_costmargin_regressor.joblib to speed up
blend selection, then confirming final specs with direct mixing.

Workflow:
1) Generate random volumes for N_CANDIDATES.
2) Enforce leftover rule. If fail => discard immediately.
3) Build X_row DataFrame for the multi-output regressor.
4) If X_row has any NaN => skip (can't predict).
5) Predict [Cost, OctMargin, RVPExcess]. If out-of-spec => skip (octMargin<0 or RVPExcess>0).
6) Keep the rest, sort by predicted cost, pick top K for final check.
7) For those top K, do actual direct mixing to confirm specs (octane≥limit, RVP≤limit, etc.).
8) Among the truly in-spec, pick best cost & best near-limit.
"""

import pandas as pd
import numpy as np
import math
import joblib

# -------------- CONFIG --------------
N_CANDIDATES    = 100000
TOP_K           = 2000
TOTAL_BLEND_VOL = 70000.0
RANDOM_SEED     = 42

TARGET_DATE          = "1/1/2024"
TARGET_GASOLINE_TYPE = "regular"

RVP_SUMMER     = 7.0
RVP_WINTER     = 13.0
OCTANE_REGULAR = 83.0
OCTANE_PREMIUM = 88.0

COMPONENTS = [
    "Reformate", 
    "LHCN/HHC", 
    "LCN", 
    "CYHEX/SAT C6",
    "Alkylate", 
    "CC5/Dimate", 
    "SAT C5", 
    "NC4"
]

# Features for the multi-output pipeline
FEATURE_NAMES = [
    "ReformateVol", "LHCN/HHCVol", "LCNVol", "CYHEX/SATC6Vol",
    "AlkylateVol", "CC5/DimateVol", "SATC5Vol", "NC4Vol",
    "GasolineTypeEncoded", "SeasonEncoded"
]

# -------------------------------------

def determine_season(date_str):
    """
    Return 'summer' if date is between May 1 and Sep 30 (inclusive),
    otherwise 'winter'.
    """
    date = pd.to_datetime(date_str, format="%m/%d/%Y")
    m, d = date.month, date.day
    if (m > 5 and m < 10) or (m == 5 and d >= 1) or (m == 9 and d <= 30):
        return "summer"
    else:
        return "winter"

def build_day_component_dict(row):
    """
    Build a dict: {component -> {TankVolume, Octane, RVP, Sulfur, Aromatics, Benzene, Olefins, Cost}}
    from a single row in wide format.
    """
    comp_dict = {}
    for comp in COMPONENTS:
        tv_col   = f"{comp} Tank Volume (BBL)"
        oct_col  = f"{comp} Octane"
        rvp_col  = f"{comp} RVP (PSI)"
        s_col    = f"{comp} Sulfur (WPPM)"
        aro_col  = f"{comp} Aromatics (VOL %)"
        ben_col  = f"{comp} Benzene (VOL %)"
        ole_col  = f"{comp} Olefins (VOL %)"
        cost_col = f"{comp} cost ($/bbl)"

        comp_dict[comp] = {
            "TankVolume": row.get(tv_col,  np.nan),
            "Octane":     row.get(oct_col, np.nan),
            "RVP":        row.get(rvp_col, np.nan),
            "Sulfur":     row.get(s_col,   np.nan),
            "Aromatics":  row.get(aro_col, np.nan),
            "Benzene":    row.get(ben_col, np.nan),
            "Olefins":    row.get(ole_col, np.nan),
            "Cost":       row.get(cost_col,np.nan),
        }
    return comp_dict

def direct_mixing_specs(volumes, comp_dict):
    """
    Actual linear blending calc: 
    Return dict with {Octane, RVP, Sulfur, Aromatics, Benzene, Olefins, Cost}
    """
    total_vol = volumes.sum()
    if total_vol <= 0:
        return {
            "Octane": 0, "RVP": 9999, "Sulfur": 9999, "Aromatics": 9999,
            "Benzene":9999, "Olefins":9999, "Cost":9999
        }
    frac = volumes / total_vol

    final_octane     = 0.0
    final_rvp        = 0.0
    final_sulfur     = 0.0
    final_aromatics  = 0.0
    final_benzene    = 0.0
    final_olefins    = 0.0
    total_cost       = 0.0

    for idx, comp in enumerate(COMPONENTS):
        p = comp_dict[comp]
        w = frac[idx]

        octv = p["Octane"]    if pd.notnull(p["Octane"])    else 0
        rvpv = p["RVP"]       if pd.notnull(p["RVP"])       else 0
        sulf = p["Sulfur"]    if pd.notnull(p["Sulfur"])    else 0
        aro  = p["Aromatics"] if pd.notnull(p["Aromatics"]) else 0
        ben  = p["Benzene"]   if pd.notnull(p["Benzene"])   else 0
        ole  = p["Olefins"]   if pd.notnull(p["Olefins"])   else 0
        cst  = p["Cost"]      if pd.notnull(p["Cost"])      else 0

        final_octane    += w * octv
        final_rvp       += w * rvpv
        final_sulfur    += w * sulf
        final_aromatics += w * aro
        final_benzene   += w * ben
        final_olefins   += w * ole
        total_cost      += volumes[idx] * cst

    cost_per_bbl = total_cost / total_vol
    return {
        "Octane":     final_octane,
        "RVP":        final_rvp,
        "Sulfur":     final_sulfur,
        "Aromatics":  final_aromatics,
        "Benzene":    final_benzene,
        "Olefins":    final_olefins,
        "Cost":       cost_per_bbl
    }

def main():
    np.random.seed(RANDOM_SEED)

    # 1) Load multi-output regressor
    mo_reg = joblib.load("models/model_costmargin_regressor.joblib")

    # 2) Load day data
    df_all = pd.read_csv("data/component_data_all.csv")
    row_sel = df_all.loc[df_all["Date"]==TARGET_DATE]
    if row_sel.empty:
        print(f"No data for {TARGET_DATE} in CSV.")
        return

    row_sel = row_sel.iloc[0]
    comp_dict = build_day_component_dict(row_sel)

    # 3) Derive season, RVP & octane limits
    season_str = determine_season(TARGET_DATE)
    if TARGET_GASOLINE_TYPE == "regular":
        octane_limit = OCTANE_REGULAR
    else:
        octane_limit = OCTANE_PREMIUM

    if season_str == "summer":
        rvp_limit = RVP_SUMMER
    else:
        rvp_limit = RVP_WINTER

    # Collect tank volumes for leftover check
    tank_volumes = []
    for comp in COMPONENTS:
        tv = comp_dict[comp]["TankVolume"]
        tank_volumes.append(tv)

    # We'll store feasible predictions
    pred_candidates = []

    # 4) Random generation
    for _ in range(N_CANDIDATES):
        volumes = np.random.rand(len(COMPONENTS))
        volumes = volumes / volumes.sum() * TOTAL_BLEND_VOL

        # leftover rule
        leftover_ok = True
        for i, tv in enumerate(tank_volumes):
            if pd.isnull(tv) or tv <= 0:
                leftover_ok = False
                break
            if volumes[i] > 0.75 * tv:
                leftover_ok = False
                break
        if not leftover_ok:
            continue

        # build single-row DF for mo_reg
        gas_type_enc = 0 if TARGET_GASOLINE_TYPE=="regular" else 1
        season_enc   = 0 if season_str=="winter" else 1

        data_dict = {
            "ReformateVol":    volumes[0],
            "LHCN/HHCVol":     volumes[1],
            "LCNVol":          volumes[2],
            "CYHEX/SATC6Vol":  volumes[3],
            "AlkylateVol":     volumes[4],
            "CC5/DimateVol":   volumes[5],
            "SATC5Vol":        volumes[6],
            "NC4Vol":          volumes[7],
            "GasolineTypeEncoded": gas_type_enc,
            "SeasonEncoded":       season_enc
        }
        X_row = pd.DataFrame([data_dict], columns=FEATURE_NAMES)

        # if any NaN => skip
        if X_row.isnull().any(axis=None):
            continue

        # predict => cost_pred, octMargin, rvpExcess
        cost_pred, oct_margin, rvp_excess = mo_reg.predict(X_row)[0]

        # skip if out-of-spec by margin
        if oct_margin < 0:
            continue
        if rvp_excess > 0:
            continue

        # keep
        pred_candidates.append((volumes, cost_pred, oct_margin, rvp_excess))

    print(f"After multi-output filter, we have {len(pred_candidates)} feasible predictions.")
    print(f"Taking top K={TOP_K} for final direct mixing check.\n")

    # 5) Sort pred_candidates by cost_pred ascending, pick top K
    pred_candidates.sort(key=lambda x: x[1])  # x=(volumes, cost_pred, oct_marg, rvp_marg)
    top_k_list = pred_candidates[:TOP_K]

    # 6) Final direct mixing
    final_blends = []
    for (vols, cost_pred, om, rm) in top_k_list:
        specs = direct_mixing_specs(vols, comp_dict)
        final_oct = specs["Octane"]
        final_rvp = specs["RVP"]
        final_ct  = specs["Cost"]

        # check real specs
        if final_oct < octane_limit:
            continue
        if final_rvp > rvp_limit:
            continue

        final_blends.append((vols, final_ct, final_oct, final_rvp))

    print(f"Final feasible after direct mixing: {len(final_blends)}")

    if len(final_blends)==0:
        print("No truly in-spec blend found.")
        return

    # 7) Among these final feasible blends, pick best cost & near-limit
    best_cost_val = float("inf")
    best_cost_vol = None

    best_margin_dist = float("inf")
    best_margin_vol  = None

    for (vols, cst, octv, rvpv) in final_blends:
        if cst < best_cost_val:
            best_cost_val = cst
            best_cost_vol = vols.copy()

        # near-limit distance
        oct_margin = octv - octane_limit  # how far above limit
        rvp_margin = rvpv - rvp_limit     # how far above limit
        dist = math.sqrt(oct_margin**2 + rvp_margin**2)
        if dist < best_margin_dist:
            best_margin_dist = dist
            best_margin_vol  = vols.copy()

    print(f"Best cost found: ${best_cost_val:.2f}")
    if best_cost_vol is not None:
        bc_specs = direct_mixing_specs(best_cost_vol, comp_dict)
        print("Best-Cost Blend: final octane={:.2f}, RVP={:.2f}, Cost={:.2f}".format(
            bc_specs["Octane"], bc_specs["RVP"], bc_specs["Cost"]))

    if best_margin_vol is not None:
        nm_specs = direct_mixing_specs(best_margin_vol, comp_dict)
        print("Near-Limit Blend: final octane={:.2f}, RVP={:.2f}, Cost={:.2f}".format(
            nm_specs["Octane"], nm_specs["RVP"], nm_specs["Cost"]))
    else:
        print("No near-limit candidate found?")

if __name__=="__main__":
    main()
