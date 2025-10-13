#!/usr/bin/env python3
import os
import random
import string
import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PRIMARY_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'primary')
SECONDARY_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'secondary')
INTEGRATED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'integrated')

N_POPULATION = 450
N_SAMPLE = 300
N_SME = 200
N_LARGE = 100

YEARS = [2022, 2023, 2024]

SUBSECTORS = [
    'Grain milling', 'Dairy processing', 'Meat processing', 'Fish processing',
    'Fruit & vegetable processing', 'Bakery products', 'Beverages', 'Confectionery'
]
OWNERSHIP_TYPES = ['Domestic', 'Foreign', 'Joint Venture']


@dataclass
class SamplingFrame:
    firm_id: str
    size: str
    subsector: str
    ownership: str
    founded_year: int
    employees: int
    assets_musd: float


def ensure_dirs():
    os.makedirs(PRIMARY_DIR, exist_ok=True)
    os.makedirs(SECONDARY_DIR, exist_ok=True)
    os.makedirs(INTEGRATED_DIR, exist_ok=True)


def generate_population_frame(n: int = N_POPULATION) -> pd.DataFrame:
    frames: List[SamplingFrame] = []
    # Construct 450 firms with plausible attributes
    for i in range(1, n + 1):
        firm_id = f"FIRM_{i:03d}"
        size = np.random.choice(['SME', 'Large'], p=[0.7, 0.3])
        subsector = np.random.choice(SUBSECTORS)
        ownership = np.random.choice(OWNERSHIP_TYPES, p=[0.7, 0.15, 0.15])
        founded_year = int(np.random.choice(np.arange(1975, 2021)))
        base_employees = np.random.randint(20, 200) if size == 'SME' else np.random.randint(200, 2000)
        # Adjust employees by subsector intensity
        intensity_factor = {
            'Grain milling': 1.0,
            'Dairy processing': 1.1,
            'Meat processing': 1.2,
            'Fish processing': 1.1,
            'Fruit & vegetable processing': 0.9,
            'Bakery products': 0.8,
            'Beverages': 1.3,
            'Confectionery': 0.9,
        }[subsector]
        employees = int(base_employees * intensity_factor)
        assets_musd = float(np.round(np.random.lognormal(mean=2.5 if size=='Large' else 1.7, sigma=0.6), 2))
        frames.append(SamplingFrame(
            firm_id, size, subsector, ownership, founded_year, employees, assets_musd
        ))
    df = pd.DataFrame([f.__dict__ for f in frames])
    return df


def stratified_sample(frame_df: pd.DataFrame, n_sme: int = N_SME, n_large: int = N_LARGE) -> pd.DataFrame:
    sme_df = frame_df[frame_df['size'] == 'SME'].sample(n=n_sme, random_state=RANDOM_SEED)
    large_df = frame_df[frame_df['size'] == 'Large'].sample(n=n_large, random_state=RANDOM_SEED)
    sample_df = pd.concat([sme_df, large_df], axis=0).sort_values('firm_id').reset_index(drop=True)
    return sample_df


def generate_primary_data(sample_df: pd.DataFrame) -> pd.DataFrame:
    # Latent constructs - Likert and continuous
    n = len(sample_df)
    # FDI presence: 0/1 with higher probability for Large and Foreign ownership
    base_prob = 0.45
    size_boost = sample_df['size'].map({'SME': -0.1, 'Large': 0.2})
    ownership_boost = sample_df['ownership'].map({'Domestic': -0.05, 'Foreign': 0.25, 'Joint Venture': 0.15})
    p_fdi = np.clip(base_prob + size_boost + ownership_boost, 0.05, 0.95)
    fdi_presence = (np.random.rand(n) < p_fdi).astype(int)

    # Knowledge absorption (1-5)
    knowledge_absorption = np.clip(np.random.normal(3.2 + 0.8*fdi_presence, 0.8, n), 1, 5)

    # Task performance (1-5)
    task_performance = np.clip(np.random.normal(3.3 + 0.5*fdi_presence, 0.7, n), 1, 5)

    # Innovation (1-5)
    innovation = np.clip(np.random.normal(3.0 + 0.7*fdi_presence, 0.9, n), 1, 5)

    # Firm resources
    skilled_labor_ratio = np.clip(np.random.beta(2 + fdi_presence, 3), 0, 1)
    iot_use = (np.random.rand(n) < np.clip(0.25 + 0.25*fdi_presence, 0.05, 0.9)).astype(int)
    rd_spend_ratio = np.clip(np.random.lognormal(mean=-2 + 0.6*fdi_presence, sigma=0.6, size=n), 0.0, 0.25)
    liquidity_ratio = np.clip(np.random.normal(1.3 + 0.2*fdi_presence, 0.4, n), 0.2, 3.0)

    # Government policy perception (1-7)
    tax_incentives = np.clip(np.random.normal(4.0 + 0.2*fdi_presence, 1.1, n), 1, 7)
    regulatory_stability = np.clip(np.random.normal(3.6 + 0.1*fdi_presence, 1.2, n), 1, 7)
    infrastructure_support = np.clip(np.random.normal(3.2 + 0.2*fdi_presence, 1.1, n), 1, 7)
    corruption_experience = np.clip(np.random.normal(4.5 - 0.3*fdi_presence, 1.2, n), 1, 7)  # higher is worse
    # Policy effectiveness index (composite where higher is better)
    policy_effectiveness = np.clip(
        0.3*tax_incentives + 0.25*regulatory_stability + 0.3*infrastructure_support + 0.15*(8 - corruption_experience),
        1, 7
    )

    # Firm performance
    roi = np.clip(np.random.normal(0.12 + 0.03*fdi_presence + 0.02*innovation/5, 0.05, n), -0.2, 0.6)
    roa = np.clip(np.random.normal(0.08 + 0.02*fdi_presence + 0.02*task_performance/5, 0.04, n), -0.1, 0.4)
    export_intensity = np.clip(np.random.beta(1 + 1.5*fdi_presence, 5), 0, 0.8)
    market_share = np.clip(np.random.lognormal(mean=-2.0 + 0.2*fdi_presence, sigma=0.8, size=n), 0.0005, 0.15)
    operational_efficiency = np.clip(np.random.normal(3.3 + 0.5*fdi_presence + 0.3*knowledge_absorption/5, 0.7, n), 1, 5)

    # Controls
    age = 2025 - sample_df['founded_year']
    export_intensity_ctrl = export_intensity  # for clarity in integrated dataset

    primary = pd.DataFrame({
        'FirmID': sample_df['firm_id'],
        'Year': np.random.choice(YEARS, size=n, p=[0.3,0.35,0.35]),
        'Size': sample_df['size'],
        'Subsector': sample_df['subsector'],
        'Ownership': sample_df['ownership'],
        'FDI_Presence': fdi_presence,
        'Knowledge_Absorption': np.round(knowledge_absorption, 2),
        'Task_Performance': np.round(task_performance, 2),
        'Innovation_Score': np.round(innovation, 2),
        'Skilled_Labor_Ratio': np.round(skilled_labor_ratio, 3),
        'IoT_Use': iot_use,
        'RD_Spend_Ratio': np.round(rd_spend_ratio, 3),
        'Liquidity_Ratio': np.round(liquidity_ratio, 2),
        'Tax_Incentives': np.round(tax_incentives, 2),
        'Regulatory_Stability': np.round(regulatory_stability, 2),
        'Infrastructure_Support': np.round(infrastructure_support, 2),
        'Corruption_Experience': np.round(corruption_experience, 2),
        'Policy_Effectiveness_Index': np.round(policy_effectiveness, 2),
        'ROI': np.round(roi, 4),
        'ROA': np.round(roa, 4),
        'Export_Intensity': np.round(export_intensity, 3),
        'Market_Share': np.round(market_share, 4),
        'Operational_Efficiency': np.round(operational_efficiency, 2),
        'Employees': sample_df['employees'],
        'Assets_MUSD': np.round(sample_df['assets_musd'], 2),
        'Age': age,
    })
    return primary


def generate_secondary_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Macro-level indices per year
    years = YEARS
    # FDI inflows to Lagos and Nigeria (fabricated but plausible magnitudes)
    fdi_inflows = pd.DataFrame({
        'Year': years,
        'FDI_Total_Nigeria_USD_Bn': [4.1, 3.8, 4.5],
        'FDI_Lagos_USD_Bn': [2.1, 1.9, 2.4],
        'FDI_Sector_Food_Bev_Tob_USD_Bn': [0.55, 0.52, 0.63],
        'FDI_Type_Greenfield_pct': [62, 58, 65],
        'FDI_Type_MA_pct': [38, 42, 35],
        'Top_Origin_Countries': ['UK, USA, China', 'USA, Netherlands, China', 'UK, South Africa, China']
    })

    # Government policy indices
    policy = pd.DataFrame({
        'Year': years,
        'Ease_Doing_Business_Index': [56.9, 58.1, 58.7],
        'Regulatory_Quality_Index': [-0.65, -0.61, -0.59],
        'Corruption_Perception_Index': [24, 24, 25],
        'EEG_Disbursement_USD_M': [120, 95, 140],
        'Tax_Incentive_Spend_USD_M': [310, 295, 330]
    })

    # Infrastructure and institutions
    infra = pd.DataFrame({
        'Year': years,
        'Power_Supply_Stability_Index': [38, 41, 44],
        'Logistics_Performance_Index': [2.31, 2.35, 2.4],
        'Port_Dwell_Time_Days': [19.0, 18.2, 17.5]
    })

    # Sectoral performance
    sector = pd.DataFrame({
        'Year': years,
        'Food_Sector_GDP_Contribution_USD_Bn': [48.2, 49.0, 50.5],
        'Food_Sector_Employment_Thousands': [2100, 2120, 2145],
        'Food_Sector_Output_Growth_pct': [2.1, 2.4, 2.6],
        'Food_Sector_Exports_USD_Bn': [1.1, 1.2, 1.35]
    })

    # Regional (Lagos-specific)
    regional = pd.DataFrame({
        'Year': years,
        'Lagos_GDP_USD_Bn': [130, 134, 140],
        'Lagos_Manufacturing_Employment_Thousands': [820, 830, 845]
    })

    return fdi_inflows, policy, infra, sector, regional


def merge_integrated(primary: pd.DataFrame,
                      fdi_inflows: pd.DataFrame,
                      policy: pd.DataFrame,
                      infra: pd.DataFrame,
                      sector: pd.DataFrame,
                      regional: pd.DataFrame) -> pd.DataFrame:
    integrated = primary.merge(fdi_inflows, on='Year', how='left')\
                        .merge(policy, on='Year', how='left')\
                        .merge(infra, on='Year', how='left')\
                        .merge(sector, on='Year', how='left')\
                        .merge(regional, on='Year', how='left')
    return integrated


def main():
    ensure_dirs()

    frame = generate_population_frame()
    frame.to_csv(os.path.join(PRIMARY_DIR, 'population_frame.csv'), index=False)

    sample_df = stratified_sample(frame)
    sample_df.to_csv(os.path.join(PRIMARY_DIR, 'sample_frame.csv'), index=False)

    primary = generate_primary_data(sample_df)
    primary.to_csv(os.path.join(PRIMARY_DIR, 'primary_firm_survey.csv'), index=False)

    fdi_inflows, policy, infra, sector, regional = generate_secondary_data()
    fdi_inflows.to_csv(os.path.join(SECONDARY_DIR, 'fdi_inflows.csv'), index=False)
    policy.to_csv(os.path.join(SECONDARY_DIR, 'policy_indices.csv'), index=False)
    infra.to_csv(os.path.join(SECONDARY_DIR, 'infrastructure.csv'), index=False)
    sector.to_csv(os.path.join(SECONDARY_DIR, 'sectoral_performance.csv'), index=False)
    regional.to_csv(os.path.join(SECONDARY_DIR, 'regional_lagos.csv'), index=False)

    integrated = merge_integrated(primary, fdi_inflows, policy, infra, sector, regional)
    integrated.to_csv(os.path.join(INTEGRATED_DIR, 'integrated_firm_year.csv'), index=False)

    # Write a simple codebook/README in JSON
    codebook = {
        'description': 'Synthetic dataset on FDI and performance of food processing firms in Lagos, Nigeria. Fabricated for research design and testing.',
        'primary_files': ['population_frame.csv', 'sample_frame.csv', 'primary_firm_survey.csv'],
        'secondary_files': ['fdi_inflows.csv', 'policy_indices.csv', 'infrastructure.csv', 'sectoral_performance.csv', 'regional_lagos.csv'],
        'integrated_files': ['integrated_firm_year.csv'],
        'notes': [
            'Values are synthetic and do not represent actual firms.',
            'Likert constructs approximate distributions from literature.',
            'Indices are on their usual scales where applicable, but fabricated.',
            'Firm identifiers anonymized as FirmID.'
        ]
    }
    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'codebook.json'), 'w') as f:
        json.dump(codebook, f, indent=2)

    print('Generated files:')
    for root in [PRIMARY_DIR, SECONDARY_DIR, INTEGRATED_DIR]:
        for fn in sorted(os.listdir(root)):
            print(os.path.join(root, fn))


if __name__ == '__main__':
    main()
