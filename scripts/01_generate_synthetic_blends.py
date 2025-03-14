#!/usr/bin/env python
"""
01_generate_synthetic_blends.py (Revised)

Generates a larger synthetic dataset (e.g., 100 random blends per day per gasoline type)
that incorporates the 25% leftover rule (discard any blend that exceeds 75% of tank volume).
- We only keep leftover-feasible blends in the final dataset.
- We keep all in-spec blends, but downsample out-of-spec blends to reduce class imbalance.
- We add margin columns for Octane and RVP (and optional extras) so we can later
  identify blends that are near limits.

"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1) CONFIGURATION
# ------------------------------------------------------------------------------
INPUT_CSV  = "data/component_data_all.csv"   # Must match your actual file name
OUTPUT_CSV = "data/blends_dataset.csv"

# Number of random blends per day per gas type
N_PER_DAY_PER_TYPE = 750

TOTAL_BLEND_VOL = 70000.0
RANDOM_SEED     = 42

# RVP limits by season
RVP_SUMMER = 7.0
RVP_WINTER = 13.0

# Octane limits by gasoline type
OCTANE_REGULAR = 83.0
OCTANE_PREMIUM = 88.0

# Other property constraints
MAX_SULFUR    = 8.50  # ppm
MAX_OLEFINS   = 8.0   # vol%
MAX_BENZENE   = 0.80  # vol%
MAX_AROMATICS = 28.0  # vol%

# Downsampling ratio for out-of-spec blends
# e.g., 0.25 means keep only 25% of out-of-spec examples
OUT_OF_SPEC_RATIO = 0.17

# Components in the CSV (columns)
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

# Property suffix mapping (adjust if your CSV headers differ):
PROPERTY_MAP = {
    "Tank Volume (BBL)":        "TankVolume",
    "Octane":                   "Octane",
    "RVP (PSI)":                "RVP",
    "Sulfur (WPPM)":            "Sulfur",
    "Aromatics (VOL %)":        "Aromatics",   # or (LV%) for certain components; adjust if needed
    "Benzene (VOL %)":          "Benzene",
    "Olefins (VOL %)":          "Olefins",     # or (LV%) for certain components; adjust if needed
    "cost ($/bbl)":             "Cost"
}

# ------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def determine_season(date_str):
    """
    Return "summer" if date is between May 1 and Sep 30 (inclusive),
    otherwise "winter".
    """
    date = pd.to_datetime(date_str, format="%m/%d/%Y")
    m, d = date.month, date.day
    if (m > 5 and m < 10) or (m == 5 and d >= 1) or (m == 9 and d <= 30):
        return "summer"
    else:
        return "winter"

def is_in_spec(final_octane, final_rvp, final_sulfur, final_aromatics,
               final_benzene, final_olefins, gas_type, season):
    """
    Check if blend meets all specs:
      - octane >= 83 (regular) or >= 88 (premium)
      - RVP <= 7 (summer) or <= 13 (winter)
      - sulfur <= 8.50 ppm, olefins <= 8%, benzene <= 0.80%, aromatics <= 28%
    """
    # Octane
    if gas_type == "regular" and final_octane < OCTANE_REGULAR:
        return 0
    if gas_type == "premium" and final_octane < OCTANE_PREMIUM:
        return 0

    # RVP
    if season == "summer" and final_rvp > RVP_SUMMER:
        return 0
    if season == "winter" and final_rvp > RVP_WINTER:
        return 0

    # Other constraints
    if final_sulfur > MAX_SULFUR:
        return 0
    if final_olefins > MAX_OLEFINS:
        return 0
    if final_benzene > MAX_BENZENE:
        return 0
    if final_aromatics > MAX_AROMATICS:
        return 0

    return 1

def compute_margin_columns(octane, gas_type, rvp, season):
    """
    Compute how close we are to the minimum Octane limit and maximum RVP limit.
    - For 'regular', min octane = 83, for 'premium', min octane = 88
    - For 'summer', max RVP = 7, for 'winter', max RVP = 13

    Returns a dict with margin info, e.g.:
      {
        "OctaneMargin": (final_octane - min_required_octane),   # negative => out of spec
        "RVPExcess":    (final_rvp    - max_allowed_rvp),       # positive => out of spec
      }
    """
    if gas_type == "regular":
        min_oct = OCTANE_REGULAR
    else:
        min_oct = OCTANE_PREMIUM

    if season == "summer":
        max_rvp = RVP_SUMMER
    else:
        max_rvp = RVP_WINTER

    return {
        "OctaneMargin": octane - min_oct,  # how much above the min octane we are
        "RVPExcess":    rvp - max_rvp      # how much above the max RVP we are
    }

# ------------------------------------------------------------------------------
# 3) MAIN LOGIC
# ------------------------------------------------------------------------------
def main():
    np.random.seed(RANDOM_SEED)

    df_wide = pd.read_csv(INPUT_CSV)
    df_wide["season"] = df_wide["Date"].apply(determine_season)

    # Build a dictionary: date_str -> {season, components: {prop -> value}}
    daily_data = {}
    for i, row in df_wide.iterrows():
        date_str = row["Date"]
        season_str = row["season"]

        # build component dict for this day
        day_dict = {}
        for comp in COMPONENTS:
            comp_props = {}
            for csv_suffix, prop_key in PROPERTY_MAP.items():
                col_name = f"{comp} {csv_suffix}"
                # If your CSV has any slight naming differences for some comps, handle them here
                comp_props[prop_key] = row.get(col_name, np.nan)

            day_dict[comp] = comp_props

        daily_data[date_str] = {
            "season": season_str,
            "components": day_dict
        }

    # Generate random blends and keep only leftover feasible
    blends_list = []
    for date_str, day_info in daily_data.items():
        season = day_info["season"]
        comp_data = day_info["components"]

        for gas_type in ["regular", "premium"]:
            for _ in range(N_PER_DAY_PER_TYPE):
                # A) Random volumes that sum to 70k
                volumes = np.random.rand(len(COMPONENTS))
                volumes = volumes / volumes.sum() * TOTAL_BLEND_VOL

                # B) Check leftover rule immediately: discard if not feasible
                leftover_ok = 1
                for idx, comp in enumerate(COMPONENTS):
                    tank_vol = comp_data[comp]["TankVolume"]
                    if pd.isnull(tank_vol) or tank_vol <= 0:
                        leftover_ok = 0
                        break
                    max_draw = 0.75 * tank_vol
                    if volumes[idx] > max_draw:
                        leftover_ok = 0
                        break

                if leftover_ok == 0:
                    continue  # skip this blend

                # C) Compute final properties by direct mixing (linear approximation)
                frac = volumes / volumes.sum()  # fraction
                final_octane    = 0.0
                final_rvp       = 0.0
                final_sulfur    = 0.0
                final_aromatics = 0.0
                final_benzene   = 0.0
                final_olefins   = 0.0
                total_cost      = 0.0

                for idx, comp in enumerate(COMPONENTS):
                    p = comp_data[comp]
                    w = frac[idx]

                    octane    = p["Octane"] if pd.notnull(p["Octane"]) else 0
                    rvp       = p["RVP"] if pd.notnull(p["RVP"]) else 0
                    sulfur    = p["Sulfur"] if pd.notnull(p["Sulfur"]) else 0
                    aromatics = p["Aromatics"] if pd.notnull(p["Aromatics"]) else 0
                    benzene   = p["Benzene"] if pd.notnull(p["Benzene"]) else 0
                    olefins   = p["Olefins"] if pd.notnull(p["Olefins"]) else 0
                    cost_bbl  = p["Cost"] if pd.notnull(p["Cost"]) else 0

                    final_octane    += w * octane
                    final_rvp       += w * rvp
                    final_sulfur    += w * sulfur
                    final_aromatics += w * aromatics
                    final_benzene   += w * benzene
                    final_olefins   += w * olefins
                    total_cost      += volumes[idx] * cost_bbl

                cost_per_bbl = total_cost / TOTAL_BLEND_VOL

                # D) in-spec label
                in_spec_flag = is_in_spec(
                    final_octane, final_rvp, final_sulfur,
                    final_aromatics, final_benzene, final_olefins,
                    gas_type, season
                )

                # E) Compute margin columns (helpful for near-limit optimization)
                margin_info = compute_margin_columns(
                    final_octane, gas_type,
                    final_rvp, season
                )
                # margin_info => dict with "OctaneMargin", "RVPExcess"

                blend_record = {
                    "Date": date_str,
                    "Season": season,
                    "GasolineType": gas_type,
                    # Volume columns
                    "ReformateVol":    volumes[0],
                    "LHCN/HHCVol":     volumes[1],
                    "LCNVol":          volumes[2],
                    "CYHEX/SATC6Vol":  volumes[3],
                    "AlkylateVol":     volumes[4],
                    "CC5/DimateVol":   volumes[5],
                    "SATC5Vol":        volumes[6],
                    "NC4Vol":          volumes[7],
                    # Final properties
                    "FinalOctane":     final_octane,
                    "FinalRVP":        final_rvp,
                    "FinalSulfur":     final_sulfur,
                    "FinalAromatics":  final_aromatics,
                    "FinalBenzene":    final_benzene,
                    "FinalOlefins":    final_olefins,
                    "Cost_per_bbl":    cost_per_bbl,
                    "InSpec":          in_spec_flag,
                    # Margin columns
                    "OctaneMargin":    margin_info["OctaneMargin"],
                    "RVPExcess":       margin_info["RVPExcess"]
                }
                blends_list.append(blend_record)

    df_blends = pd.DataFrame(blends_list)
    print(f"Generated {len(df_blends)} leftover-feasible blends before downsampling.")

    # ------------------------------------------------------------------------------
    # 4) Address Class Imbalance: Keep all in-spec, downsample out-of-spec
    # ------------------------------------------------------------------------------
    df_in_spec = df_blends[df_blends["InSpec"] == 1]
    df_out_spec = df_blends[df_blends["InSpec"] == 0]

    # Keep all in-spec
    # Sample a fraction of out-of-spec (if OUT_OF_SPEC_RATIO < 1.0)
    if OUT_OF_SPEC_RATIO < 1.0 and len(df_out_spec) > 0:
        df_out_spec_sampled = df_out_spec.sample(
            frac=OUT_OF_SPEC_RATIO,
            random_state=RANDOM_SEED
        )
    else:
        df_out_spec_sampled = df_out_spec

    df_final = pd.concat([df_in_spec, df_out_spec_sampled], ignore_index=True)
    df_final = df_final.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    print(f"Final dataset size after downsampling: {len(df_final)}")
    print(f"  # in-spec: {sum(df_final['InSpec'] == 1)}")
    print(f"  # out-of-spec: {sum(df_final['InSpec'] == 0)}")

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved final dataset to {OUTPUT_CSV}")
    print("Columns:", df_final.columns.tolist())

# ------------------------------------------------------------------------------
# 5) CLI ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
