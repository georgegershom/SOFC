#!/usr/bin/env python3
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Synthetic dataset generator for: 
# "The Influence of FDI on the Performance of Food Processing Firms in Lagos, Nigeria: 
#  The Moderating Role of the Nigerian Government Policy"
# Cross-sectional, 2024. Fabricated data for research prototyping only.

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

PRIMARY_DIR = os.path.join("/workspace", "data", "primary")
SECONDARY_DIR = os.path.join("/workspace", "data", "secondary")
INTEGRATED_DIR = os.path.join("/workspace", "data", "integrated")

N_FIRMS = 300
YEAR = 2024

SUBSECTORS = [
    "Bakery", "Dairy", "Beverages", "Meat Processing", "Grains & Cereals",
    "Fruits & Vegetables", "Confectionery", "Oils & Fats", "Seafood Processing", "Ready Meals"
]
OWNERSHIP_TYPES = ["Domestic", "Foreign", "Joint Venture"]


def clip_round(x, lo, hi, decimals=1):
    return np.round(np.clip(x, lo, hi), decimals)


def likert_from_base(base, sd, lo, hi):
    return clip_round(rng.normal(base, sd, size=N_FIRMS), lo, hi, decimals=1)


def generate_primary():
    # IDs and strata
    firm_ids = [f"FIRM_{i:03d}" for i in range(1, N_FIRMS + 1)]

    # Size: 200 SMEs, 100 Large
    size = np.array(["SME"] * 200 + ["Large"] * 100)
    rng.shuffle(size)

    # Employees
    employees = np.where(size == "SME",
                         rng.integers(10, 200, size=N_FIRMS),
                         rng.integers(200, 2001, size=N_FIRMS))

    # Assets (USD millions) - broader for Large
    assets = np.where(size == "SME",
                      rng.normal(3.0, 1.2, size=N_FIRMS),
                      rng.normal(25.0, 10.0, size=N_FIRMS))
    assets = clip_round(assets, 0.5, 120.0, decimals=2)

    # Age in years
    age_years = rng.integers(1, 51, size=N_FIRMS)

    # Subsector
    subsector = rng.choice(SUBSECTORS, size=N_FIRMS, replace=True)

    # Ownership mix (heavier Domestic)
    ownership = rng.choice(OWNERSHIP_TYPES, p=[0.7, 0.1, 0.2], size=N_FIRMS)

    # FDI presence: 1 if Foreign or JV else 0
    fdi_presence = ((ownership == "Foreign") | (ownership == "Joint Venture")).astype(int)

    # FDI equity share (%)
    fdi_equity = np.zeros(N_FIRMS)
    # JV: 10-90%; Foreign: 51-100%; Domestic: 0
    mask_jv = ownership == "Joint Venture"
    mask_foreign = ownership == "Foreign"
    fdi_equity[mask_jv] = rng.integers(10, 91, size=mask_jv.sum())
    fdi_equity[mask_foreign] = rng.integers(51, 101, size=mask_foreign.sum())

    # IoT use probability: higher for Large
    iot_use = rng.binomial(1, p=np.where(size == "Large", 0.55, 0.35), size=N_FIRMS)

    # Skilled labor ratio: higher for Large on average
    skilled_ratio = np.where(size == "Large",
                             rng.normal(0.62, 0.12, size=N_FIRMS),
                             rng.normal(0.48, 0.12, size=N_FIRMS))
    skilled_ratio = clip_round(skilled_ratio, 0.05, 0.95, decimals=2)

    # R&D spend % revenue
    rd_spend = np.where(size == "Large",
                        rng.normal(3.2, 1.2, size=N_FIRMS),
                        rng.normal(2.2, 1.1, size=N_FIRMS))
    rd_spend = clip_round(rd_spend, 0.0, 9.0, decimals=2)

    # Liquidity ratio
    liquidity = clip_round(rng.normal(1.6, 0.5, size=N_FIRMS), 0.4, 4.0, decimals=2)

    # Government policy perception (Likert 1-7)
    tax_incentives = clip_round(rng.normal(4.2, 1.0, size=N_FIRMS), 1.0, 7.0)
    regulatory_stability = clip_round(rng.normal(3.8, 1.0, size=N_FIRMS), 1.0, 7.0)
    infrastructure_support = clip_round(rng.normal(3.5, 1.1, size=N_FIRMS), 1.0, 7.0)
    # Higher means more corruption experienced
    corruption_experience = clip_round(rng.normal(4.3, 1.2, size=N_FIRMS), 1.0, 7.0)

    # Policy effectiveness index (1-7), invert corruption
    policy_effectiveness = (tax_incentives + regulatory_stability + infrastructure_support + (8 - corruption_experience)) / 4.0
    policy_effectiveness = clip_round(policy_effectiveness, 1.0, 7.0)

    # FDI constructs (Likert 1-5)
    knowledge_absorption = clip_round(2.9 + 0.7 * fdi_presence + 0.4 * (skilled_ratio - 0.5) * 2 + rng.normal(0, 0.5, N_FIRMS), 1.0, 5.0)
    firm_resources_likert = clip_round(2.8 + 0.6 * (skilled_ratio - 0.5) * 2 + 0.3 * iot_use + rng.normal(0, 0.5, N_FIRMS), 1.0, 5.0)
    innovation = clip_round(2.8 + 0.6 * fdi_presence + 0.35 * (rd_spend / 3.0) + 0.2 * iot_use + rng.normal(0, 0.5, N_FIRMS), 1.0, 5.0)
    task_performance = clip_round(2.9 + 0.25 * policy_effectiveness + 0.25 * firm_resources_likert + rng.normal(0, 0.45, N_FIRMS), 1.0, 5.0)

    # Performance outcomes
    # Policy moderation: boosts FDI -> performance linkage
    moderation = (policy_effectiveness - 4.0) * 0.7

    roi = 6.0 \
        + 2.4 * (knowledge_absorption - 3.0) \
        + 2.1 * (innovation - 3.0) \
        + 1.8 * (firm_resources_likert - 3.0) \
        + 2.0 * fdi_presence * (1.0 + moderation) \
        + rng.normal(0, 2.5, N_FIRMS)
    roi = clip_round(roi, -5.0, 45.0, decimals=2)

    roa = 3.2 \
        + 1.6 * (knowledge_absorption - 3.0) \
        + 1.3 * (innovation - 3.0) \
        + 1.2 * (firm_resources_likert - 3.0) \
        + 1.4 * fdi_presence * (1.0 + moderation) \
        + rng.normal(0, 1.5, N_FIRMS)
    roa = clip_round(roa, -3.0, 22.0, decimals=2)

    export_intensity = 8.0 \
        + 6.0 * fdi_presence \
        + 3.5 * (innovation - 3.0) \
        + 2.0 * (policy_effectiveness - 4.0) \
        + rng.normal(0, 5.0, N_FIRMS)
    export_intensity = clip_round(export_intensity, 0.0, 70.0, decimals=2)

    market_share = np.where(size == "SME",
                            clip_round(rng.normal(1.0, 0.8, N_FIRMS) + 0.8 * fdi_presence, 0.01, 8.0, 2),
                            clip_round(rng.normal(4.5, 3.0, N_FIRMS) + 1.2 * fdi_presence, 0.1, 22.0, 2))

    operational_efficiency = clip_round(52.0 \
        + 6.0 * (task_performance - 3.0) \
        + 5.0 * iot_use \
        + 3.0 * (policy_effectiveness - 4.0) \
        + rng.normal(0, 8.0, N_FIRMS), 20.0, 95.0, decimals=1)

    # Assemble primary dataframe
    primary = pd.DataFrame({
        "FirmID": firm_ids,
        "Year": YEAR,
        "Size": size,
        "Employees": employees,
        "Assets_USD_millions": assets,
        "Age_Years": age_years,
        "Subsector": subsector,
        "Ownership_Type": ownership,
        "FDI_Presence": fdi_presence,
        "FDI_Equity_Share_percent": fdi_equity,
        # FDI constructs (Likert 1-5)
        "Knowledge_Absorption_1_5": knowledge_absorption,
        "Task_Performance_1_5": task_performance,
        "Innovation_1_5": innovation,
        "Firm_Resources_1_5": firm_resources_likert,
        # Resources
        "Skilled_Labor_Ratio": skilled_ratio,
        "IoT_Use": iot_use,
        "RD_Spend_percent_of_Revenue": rd_spend,
        "Liquidity_Ratio": liquidity,
        # Government policy perception
        "Tax_Incentives_1_7": tax_incentives,
        "Regulatory_Stability_1_7": regulatory_stability,
        "Infrastructure_Support_1_7": infrastructure_support,
        "Corruption_Experience_1_7": corruption_experience,
        "Govt_Policy_Score_1_7": policy_effectiveness,
        # Outcomes (self-reported)
        "ROI_Self_percent": roi,
        "ROA_Self_percent": roa,
        "Export_Intensity_percent": export_intensity,
        "Market_Share_percent": market_share,
        "Operational_Efficiency_Index_0_100": operational_efficiency,
    })

    return primary


def generate_secondary_macro():
    # Synthetic macro/sectoral indicators for Nigeria/Lagos, 2024
    # Values are fabricated for demonstration only
    data = {
        "Year": [YEAR],
        "Country": ["Nigeria"],
        "Region": ["Lagos"],
        # FDI inflows (USD billions)
        "FDI_Inflow_Total_USD_billion": [3.6],
        "FDI_Inflow_Manufacturing_USD_billion": [0.95],
        "FDI_Inflow_FoodProcessing_USD_billion": [0.22],
        "FDI_Inflow_Lagos_USD_billion": [2.1],
        # FDI type shares
        "FDI_Type_Share_Greenfield_percent": [60.0],
        "FDI_Type_Share_MA_percent": [40.0],
        # Policy and institutional
        "Ease_of_Doing_Business_Score": [56.0],
        "EEG_Disbursement_USD_billion": [0.15],
        "Regulatory_Quality_Index": [-0.8],
        "Corruption_Perception_Index_0_100": [24],
        # Infrastructure
        "Power_Supply_Reliability_Hours_per_Day": [15.5],
        "Logistics_Quality_Index_1_5": [2.5],
        # Sectoral performance
        "Food_Sector_GDP_Contribution_percent": [4.1],
        "Sector_Employment_thousands": [820],
        "Sector_Output_Growth_percent": [3.2],
        "Sector_Export_Value_USD_billion": [0.78],
        # Origins (as a semicolon-separated string)
        "Top_Origin_Countries": ["Netherlands; United Kingdom; United States; China; South Africa"],
    }
    return pd.DataFrame(data)


def generate_secondary_firm_financials(primary: pd.DataFrame):
    # Generate verified financials (as if from CBN/NBS/Orbis) per firm
    # Introduce small measurement error relative to self-reported
    roi_verified = clip_round(primary["ROI_Self_percent"].to_numpy() + rng.normal(-0.3, 1.2, N_FIRMS), -6.0, 46.0, 2)
    roa_verified = clip_round(primary["ROA_Self_percent"].to_numpy() + rng.normal(-0.2, 0.9, N_FIRMS), -4.0, 23.0, 2)

    # Revenue and exports (USD millions) correlated with size and performance
    base_revenue = np.where(primary["Size"].to_numpy() == "Large",
                            rng.normal(45.0, 18.0, N_FIRMS),
                            rng.normal(6.5, 3.0, N_FIRMS))
    perf_bump = 0.08 * (primary["Operational_Efficiency_Index_0_100"].to_numpy() - 50.0) / 50.0
    revenue = clip_round(base_revenue * (1.0 + perf_bump), 0.8, 220.0, 2)

    # Export value consistent with export intensity
    export_value = clip_round(revenue * primary["Export_Intensity_percent"].to_numpy() / 100.0, 0.0, 120.0, 2)

    # Assets (should roughly align with primary assets but allow noise)
    assets_verified = clip_round(primary["Assets_USD_millions"].to_numpy() + rng.normal(0.0, 2.0, N_FIRMS), 0.4, 125.0, 2)

    secon = pd.DataFrame({
        "FirmID": primary["FirmID"],
        "Year": YEAR,
        "Revenue_USD_millions": revenue,
        "Assets_Verified_USD_millions": assets_verified,
        "Export_Value_USD_millions": export_value,
        "ROI_Verified_percent": roi_verified,
        "ROA_Verified_percent": roa_verified,
    })
    return secon


def integrate(primary: pd.DataFrame, secon_macro: pd.DataFrame, secon_fin: pd.DataFrame):
    # Merge firm-level with verified financials, then broadcast macro
    merged = primary.merge(secon_fin, on=["FirmID", "Year"], how="left")
    integrated = merged.merge(secon_macro, on=["Year"], how="left")

    # Helpful interaction terms for analysis
    integrated["FDI_x_Policy"] = integrated["FDI_Presence"] * integrated["Govt_Policy_Score_1_7"]
    integrated["Innovation_x_Policy"] = integrated["Innovation_1_5"] * integrated["Govt_Policy_Score_1_7"]

    return integrated


def ensure_dirs():
    os.makedirs(PRIMARY_DIR, exist_ok=True)
    os.makedirs(SECONDARY_DIR, exist_ok=True)
    os.makedirs(INTEGRATED_DIR, exist_ok=True)


def main():
    ensure_dirs()

    primary = generate_primary()
    secon_macro = generate_secondary_macro()
    secon_fin = generate_secondary_firm_financials(primary)
    integrated_df = integrate(primary, secon_macro, secon_fin)

    # Save CSVs
    primary_path = os.path.join(PRIMARY_DIR, "primary_firm_survey_2024.csv")
    secon_macro_path = os.path.join(SECONDARY_DIR, "nigeria_lagos_macro_sector_2024.csv")
    secon_fin_path = os.path.join(SECONDARY_DIR, "firm_financials_2024.csv")
    integrated_path = os.path.join(INTEGRATED_DIR, "firm_integrated_2024.csv")

    primary.to_csv(primary_path, index=False)
    secon_macro.to_csv(secon_macro_path, index=False)
    secon_fin.to_csv(secon_fin_path, index=False)
    integrated_df.to_csv(integrated_path, index=False)

    print("Generated:")
    print(primary_path)
    print(secon_macro_path)
    print(secon_fin_path)
    print(integrated_path)

if __name__ == "__main__":
    main()
