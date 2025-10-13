#!/usr/bin/env python3
import os
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATA_DIR = os.path.join("/workspace", "data")
OUTPUT_DIR = os.path.join("/workspace", "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ZONES = [
    "Ikeja",
    "Apapa",
    "Oshodi/Isolo",
    "Amuwo-Odofin",
    "Ikorodu",
    "Ibeju-Lekki",
]
ZONE_WEIGHTS = np.array([0.26, 0.17, 0.20, 0.16, 0.12, 0.09])
ZONE_WEIGHTS = ZONE_WEIGHTS / ZONE_WEIGHTS.sum()

REVENUE_BINS = [
    ("<50M",  10e6,  50e6),
    ("50M-500M", 50e6,  500e6),
    ("500M-5B", 500e6, 5e9),
    (">5B", 5e9, 15e9),
]


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def choose_with_weights(options: List[str], weights: List[float], size: int) -> List[str]:
    return list(np.random.choice(options, size=size, p=np.array(weights) / np.sum(weights)))


def make_sampling_frame(n_sme: int = 300, n_large: int = 150) -> pd.DataFrame:
    total = n_sme + n_large
    firm_codes = [f"FIRM{str(i+1).zfill(3)}" for i in range(total)]
    strata = ["SME"] * n_sme + ["Large"] * n_large

    # Assign zones with weights
    zones = list(np.random.choice(ZONES, size=total, p=ZONE_WEIGHTS))

    # Years of operation: SMEs generally younger than large
    years_sme = np.clip(np.random.normal(loc=9, scale=6, size=n_sme).round(), 1, 40)
    years_large = np.clip(np.random.normal(loc=18, scale=8, size=n_large).round(), 3, 60)
    years = np.concatenate([years_sme, years_large]).astype(int)

    # Employees by stratum
    employees_sme = np.clip(np.random.lognormal(mean=np.log(60), sigma=0.7, size=n_sme).round(), 5, 250)
    employees_large = np.clip(np.random.lognormal(mean=np.log(400), sigma=0.5, size=n_large).round(), 251, 2000)
    employees = np.concatenate([employees_sme, employees_large]).astype(int)

    # Revenue category by stratum
    rev_weights_sme = np.array([0.35, 0.45, 0.18, 0.02])
    rev_weights_large = np.array([0.02, 0.25, 0.48, 0.25])
    rev_cats = []
    for s in strata:
        if s == "SME":
            rev_cats.append(np.random.choice([b[0] for b in REVENUE_BINS], p=rev_weights_sme))
        else:
            rev_cats.append(np.random.choice([b[0] for b in REVENUE_BINS], p=rev_weights_large))

    # Ownership types
    ownership_opts = ["Local", "Foreign-owned", "Joint venture"]
    ownership_weights_sme = [0.75, 0.10, 0.15]
    ownership_weights_large = [0.45, 0.25, 0.30]
    ownership_types = [
        np.random.choice(ownership_opts, p=(ownership_weights_sme if s == "SME" else ownership_weights_large))
        for s in strata
    ]

    # FDI engagement correlated with ownership
    fdi_yes_prob = []
    for own in ownership_types:
        if own == "Local":
            fdi_yes_prob.append(0.25)
        elif own == "Foreign-owned":
            fdi_yes_prob.append(0.90)
        else:  # JV
            fdi_yes_prob.append(0.75)
    fdi_partnership = (np.random.rand(total) < np.array(fdi_yes_prob))

    # FDI types if engaged
    def draw_fdi_types(is_fdi: bool) -> Tuple[bool, bool, bool, bool]:
        if not is_fdi:
            return (False, False, False, False)
        # Weighted multi-select
        base_probs = np.array([0.55, 0.45, 0.60, 0.30])  # Equity, JV, Tech, Mgmt
        draws = np.random.rand(4) < base_probs
        if not draws.any():
            draws[np.argmax(base_probs)] = True
        return tuple(bool(x) for x in draws)

    fdi_types = [draw_fdi_types(flag) for flag in fdi_partnership]
    fdi_equity, fdi_jv, fdi_tech, fdi_mgmt = map(lambda idx: [t[idx] for t in fdi_types], range(4))

    # Years with FDI
    years_fdi = []
    for flag, s in zip(fdi_partnership, strata):
        if not flag:
            years_fdi.append(0)
        else:
            base = 1 if s == "SME" else 2
            years_fdi.append(int(np.clip(np.random.gamma(shape=3, scale=2) + base, 1, 25)))

    df = pd.DataFrame({
        "firm_code": firm_codes,
        "stratum": strata,
        "zone": zones,
        "years_operation": years,
        "employees": employees,
        "annual_revenue_cat": rev_cats,
        "ownership_type": ownership_types,
        "fdi_partnership": fdi_partnership,
        "fdi_equity": fdi_equity,
        "fdi_joint_venture": fdi_jv,
        "fdi_technology_transfer": fdi_tech,
        "fdi_management_contract": fdi_mgmt,
        "years_with_fdi": years_fdi,
        "registered_LCCI": True,
        "sector": "Food Processing (NACE 10)",
    })

    return df


def neyman_sample(frame: pd.DataFrame, n_total: int = 300) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Allocation: 200 SMEs, 100 Large (given)
    n_sme = 200
    n_large = 100

    sme_pool = frame[frame["stratum"] == "SME"].sample(frac=1, random_state=RANDOM_SEED)
    large_pool = frame[frame["stratum"] == "Large"].sample(frac=1, random_state=RANDOM_SEED)

    main_sme = sme_pool.head(n_sme)
    main_large = large_pool.head(n_large)
    main = pd.concat([main_sme, main_large], ignore_index=True)

    # Backups: 20% of 300 = 60 (40 SME, 20 Large)
    backup_sme = sme_pool.iloc[n_sme: n_sme + 40]
    backup_large = large_pool.iloc[n_large: n_large + 20]
    backup = pd.concat([backup_sme, backup_large], ignore_index=True)

    # Mark flags
    selection = frame[["firm_code", "stratum", "zone"]].copy()
    selection["is_main_sample"] = selection["firm_code"].isin(main["firm_code"]).astype(int)
    selection["is_backup"] = selection["firm_code"].isin(backup["firm_code"]).astype(int)

    # Responses: target 70% of 300 = 210; keep ratio 140/70
    responded_sme = set(main_sme.sample(n=140, random_state=RANDOM_SEED + 1)["firm_code"].tolist())
    responded_large = set(main_large.sample(n=70, random_state=RANDOM_SEED + 2)["firm_code"].tolist())
    responded = responded_sme.union(responded_large)

    selection["responded"] = selection["firm_code"].isin(responded).astype(int)

    return main.reset_index(drop=True), backup.reset_index(drop=True), selection.reset_index(drop=True)


def simulate_responses(frame: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    # Only for responded firms in main sample
    responded_codes = set(selection[(selection["is_main_sample"] == 1) & (selection["responded"] == 1)]["firm_code"].tolist())
    base = frame[frame["firm_code"].isin(responded_codes)].copy()

    n = len(base)

    # Dates within last 6 months
    start_date = datetime.now() - timedelta(days=180)
    dates = [start_date + timedelta(days=int(np.random.uniform(0, 180))) for _ in range(n)]

    # Section E - Firm Resources exogenous items
    # Skilled workforce percentage and training hours
    skilled_pct = np.clip(np.random.normal(loc=55, scale=18, size=n), 10, 95)
    training_hours = np.clip(np.random.normal(loc=24, scale=12, size=n), 2, 120)

    # Modern equipment use: more likely if FDI and large
    modern_equip_prob = logistic(
        -0.3 + 0.8 * base["fdi_partnership"].astype(int).values + 0.4 * (base["stratum"] == "Large").astype(int).values
    )
    modern_equip = (np.random.rand(n) < modern_equip_prob).astype(int)

    # Age of machinery: newer if modern equipment
    machinery_age = np.clip(
        np.where(modern_equip == 1, np.random.normal(6, 3, size=n), np.random.normal(12, 5, size=n)),
        1,
        30,
    )

    # Access to credit categorical
    credit_categories = ["Easy", "Moderate", "Difficult"]
    credit_probs = np.vstack([
        logistic(-0.2 + 0.4 * base["fdi_partnership"].astype(int).values),  # Easy
        0.5 * np.ones(n),                                                 # Moderate
        logistic(-0.6 - 0.2 * base["fdi_partnership"].astype(int).values),  # Difficult
    ])
    # Normalize columns to sum to 1
    credit_probs = credit_probs / credit_probs.sum(axis=0)
    credit_choices = [credit_categories[np.random.choice(3, p=credit_probs[:, i])] for i in range(n)]

    reinvest_rate = np.clip(np.random.normal(loc=18, scale=10, size=n) + 6 * base["fdi_partnership"].astype(int).values, 0, 60)

    # Government Policy perceptions latent and items (Section F)
    gp_latent = np.random.normal(loc=0.0, scale=1.0, size=n)

    def likert_from_latent(latent: np.ndarray, k: int = 5, bias: float = 0.0) -> np.ndarray:
        # Map latent to 1..k using thresholds
        z = latent + bias + np.random.normal(0, 0.6, size=len(latent))
        # Compute quantile thresholds for near-uniform then shift by latent
        bins = np.quantile(z, np.linspace(0, 1, k + 1))
        # Ensure unique bins
        bins = np.unique(bins)
        # Digitize into 1..k
        # If bins collapsed due to unique, fallback to simple transformation
        if len(bins) <= 2:
            scaled = np.clip(np.round(2.5 + z), 1, k)
            return scaled.astype(int)
        # numpy.digitize returns 1..k with right=False when using bins[1:-1]
        labels = np.digitize(z, bins[1:-1]) + 1
        return np.clip(labels, 1, k).astype(int)

    gp_tax = likert_from_latent(gp_latent + np.random.normal(0, 0.4, size=n), k=7)
    gp_reg = likert_from_latent(gp_latent + np.random.normal(0, 0.4, size=n), k=7)
    gp_infra = likert_from_latent(gp_latent + np.random.normal(0, 0.4, size=n), k=7)
    gp_permits = likert_from_latent(gp_latent + np.random.normal(0, 0.4, size=n), k=7)

    gp_index = (gp_tax + gp_reg + gp_infra) / 3.0
    gp_index = (gp_index - 1) / 6.0  # scale ~ 0..1

    # FDI latent intensity index (0..1)
    fdi_components = (
        1.0 * base["fdi_partnership"].astype(int).values +
        0.6 * base["fdi_equity"].astype(int).values +
        0.5 * base["fdi_joint_venture"].astype(int).values +
        0.7 * base["fdi_technology_transfer"].astype(int).values +
        0.4 * base["fdi_management_contract"].astype(int).values +
        0.05 * np.clip(base["years_with_fdi"].values, 0, 20)
    )
    fdi_latent = logistic(-1.0 + 0.25 * fdi_components)

    # Firm Resources latent index (0..1)
    credit_score = np.array([0.8 if c == "Easy" else (0.5 if c == "Moderate" else 0.2) for c in credit_choices])
    fr_latent = (
        0.35 * (skilled_pct / 100.0) +
        0.20 * logistic((training_hours - 20.0) / 10.0) +
        0.20 * modern_equip +
        0.10 * (1.0 - (machinery_age / 30.0)) +
        0.10 * credit_score +
        0.05 * (reinvest_rate / 100.0)
    )
    fr_latent = np.clip(fr_latent, 0.0, 1.0)

    # Construct latents per SEM
    ka_latent = 0.70 * fdi_latent + 0.20 * fr_latent + np.random.normal(0, 0.25, size=n)
    tp_latent = 0.50 * fdi_latent + 0.30 * fr_latent + np.random.normal(0, 0.30, size=n)
    inn_latent = 0.60 * fdi_latent + 0.20 * fr_latent + np.random.normal(0, 0.30, size=n)

    perf_latent = (
        0.35 * ka_latent + 0.25 * tp_latent + 0.25 * inn_latent + 0.20 * fr_latent +
        0.10 * (fdi_latent * gp_index) + 0.08 * (ka_latent * gp_index) + 0.05 * (tp_latent * gp_index) + 0.05 * (inn_latent * gp_index) +
        np.random.normal(0, 0.25, size=n)
    )

    # Map KA and TP to Likert (1..5)
    ka1 = likert_from_latent(ka_latent)
    ka2 = likert_from_latent(ka_latent + np.random.normal(0, 0.2, size=n))
    ka3 = likert_from_latent(ka_latent + np.random.normal(0, 0.2, size=n))
    ka4 = likert_from_latent(ka_latent + np.random.normal(0, 0.2, size=n))

    tp1 = likert_from_latent(tp_latent)
    tp2 = likert_from_latent(tp_latent + np.random.normal(0, 0.2, size=n))
    tp3 = likert_from_latent(tp_latent + np.random.normal(0, 0.2, size=n))
    tp4 = likert_from_latent(tp_latent + np.random.normal(0, 0.2, size=n))

    # Innovation indicators
    rd_intensity = np.clip(100 * logistic(-2.0 + 1.8 * inn_latent + np.random.normal(0, 0.6, size=n)), 0.2, 12.0)  # percent
    new_products = np.clip(np.random.poisson(lam=np.clip(2.0 + 3.0 * inn_latent, 0.2, 8.0)), 0, 20)

    proc_prob = logistic(-0.5 + 1.5 * inn_latent)
    proc_iot = (np.random.rand(n) < proc_prob).astype(int)
    proc_auto = (np.random.rand(n) < np.clip(proc_prob + 0.1, 0, 1)).astype(int)
    proc_qm = (np.random.rand(n) < np.clip(proc_prob + 0.2, 0, 1)).astype(int)
    proc_other = (np.random.rand(n) < 0.10).astype(int)
    proc_other_text = np.where(proc_other == 1, np.random.choice([
        "Lean Six Sigma",
        "ERP upgrade",
        "Solar retrofit",
        "Waste-to-energy",
        "Cold chain automation",
    ], size=n), "")

    # Performance metrics
    roi = np.clip(100 * logistic(-0.2 + 1.2 * perf_latent) + np.random.normal(0, 2.0, size=n), 0.0, 40.0)
    roa = np.clip(100 * logistic(-0.6 + 1.0 * perf_latent) + np.random.normal(0, 1.5, size=n), -5.0, 25.0)
    export_intensity = np.clip(100 * logistic(-1.0 + 1.0 * perf_latent + 0.5 * fdi_latent + 0.3 * inn_latent) + np.random.normal(0, 2.0, size=n), 0.0, 45.0)
    capacity_util = np.clip(50 + 45 * logistic(-0.3 + 1.1 * perf_latent + 0.2 * gp_index) + np.random.normal(0, 4.0, size=n), 35.0, 98.0)
    # Market share skewed, higher with size and performance
    size_factor = np.where(base["stratum"].values == "Large", 1.0, 0.6)
    market_share = np.clip(0.2 + 6.0 * logistic(-3.0 + 1.2 * perf_latent + 0.6 * size_factor) + np.random.normal(0, 0.6, size=n), 0.05, 20.0)

    # Section A
    # Date strings DD/MM/YYYY
    dates_str = [d.strftime("%d/%m/%Y") for d in dates]

    # Employee category derived from employees
    emp_cat = []
    for e in base["employees"].values:
        if e <= 50:
            emp_cat.append("1-50")
        elif e <= 250:
            emp_cat.append("51-250")
        elif e <= 500:
            emp_cat.append("251-500")
        else:
            emp_cat.append("500+")

    # Annual revenue category already available; keep it

    # Ownership consistent

    # Years with FDI already available

    # Years of operation plus jitter
    years_operation = base["years_operation"].astype(int).values

    # Build responses DataFrame
    responses = pd.DataFrame({
        # Identifiers
        "firm_code": base["firm_code"].values,
        "date": dates_str,
        "stratum": base["stratum"].values,
        "zone": base["zone"].values,

        # Section A: Firm Background
        "years_of_operation": years_operation,
        "num_employees": base["employees"].astype(int).values,
        "num_employees_cat": emp_cat,
        "annual_revenue_cat": base["annual_revenue_cat"].values,
        "ownership_type": base["ownership_type"].values,

        # FDI Engagement
        "fdi_partnership": base["fdi_partnership"].astype(int).values,
        "fdi_equity": base["fdi_equity"].astype(int).values,
        "fdi_joint_venture": base["fdi_joint_venture"].astype(int).values,
        "fdi_technology_transfer": base["fdi_technology_transfer"].astype(int).values,
        "fdi_management_contract": base["fdi_management_contract"].astype(int).values,
        "years_with_fdi": base["years_with_fdi"].astype(int).values,

        # Section B: Knowledge Absorption (1-5)
        "ka1": ka1,
        "ka2": ka2,
        "ka3": ka3,
        "ka4": ka4,

        # Section C: Task Performance (1-5)
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp4": tp4,

        # Section D: Innovation
        "rd_intensity_percent": rd_intensity.round(2),
        "new_products_3yrs": new_products.astype(int),
        "proc_iot": proc_iot,
        "proc_automation": proc_auto,
        "proc_quality_mgmt": proc_qm,
        "proc_other": proc_other,
        "proc_other_text": proc_other_text,

        # Section E: Firm Resources
        "skilled_workforce_percent": np.round(skilled_pct, 1),
        "training_hours_per_employee": np.round(training_hours, 0).astype(int),
        "modern_equipment": modern_equip,
        "machinery_age_years": np.round(machinery_age, 1),
        "access_to_credit": credit_choices,
        "reinvestment_rate_percent": np.round(reinvest_rate, 1),

        # Section F: Government Policy Perception (1-7)
        "gp_tax_incentives": gp_tax,
        "gp_regulatory_stability": gp_reg,
        "gp_infrastructure_support": gp_infra,
        "gp_ease_permits": gp_permits,

        # Section G: Performance Metrics
        "avg_roi_percent": np.round(roi, 2),
        "avg_roa_percent": np.round(roa, 2),
        "export_intensity_percent": np.round(export_intensity, 2),
        "capacity_utilization_percent": np.round(capacity_util, 1),
        "market_share_lagos_percent": np.round(market_share, 2),
    })

    # Create indices for SEM-ready dataset
    tp_index = responses[["tp1", "tp2", "tp3", "tp4"]].mean(axis=1)
    fr_index = (
        0.35 * (responses["skilled_workforce_percent"] / 100.0) +
        0.20 * logistic((responses["training_hours_per_employee"] - 20.0) / 10.0) +
        0.20 * responses["modern_equipment"] +
        0.10 * (1.0 - (responses["machinery_age_years"] / 30.0)) +
        0.10 * responses["access_to_credit"].map({"Easy": 0.8, "Moderate": 0.5, "Difficult": 0.2}).values +
        0.05 * (responses["reinvestment_rate_percent"] / 100.0)
    )
    fr_index = np.clip(fr_index, 0.0, 1.0)
    fdi_index = fdi_latent  # already 0..1
    gp_3item_index = (responses["gp_tax_incentives"] + responses["gp_regulatory_stability"] + responses["gp_infrastructure_support"]) / 3.0
    gp_3item_index = (gp_3item_index - 1) / 6.0

    perform_index = (
        0.35 * (responses[["ka1", "ka2", "ka3", "ka4"]].mean(axis=1) / 5.0) +
        0.25 * (tp_index / 5.0) +
        0.25 * (responses["rd_intensity_percent"] / 12.0) +
        0.15 * (responses["export_intensity_percent"] / 45.0)
    )
    perform_index = np.clip(perform_index, 0.0, 1.0)

    # Innovation item 3: process innovation adoption share
    inn3_proc_index = (responses[["proc_iot", "proc_automation", "proc_quality_mgmt"]].mean(axis=1))

    thesis_data = pd.DataFrame({
        "firm_code": responses["firm_code"],
        "ka1": responses["ka1"],
        "ka2": responses["ka2"],
        "ka3": responses["ka3"],
        "ka4": responses["ka4"],
        "inn1": np.round(responses["rd_intensity_percent"], 2),
        "inn2": responses["new_products_3yrs"].astype(int),
        "inn3": np.round(inn3_proc_index, 3),
        "gp1": responses["gp_tax_incentives"].astype(int),
        "gp2": responses["gp_regulatory_stability"].astype(int),
        "gp3": responses["gp_infrastructure_support"].astype(int),
        "TP": np.round(tp_index, 3),
        "FR": np.round(fr_index, 3),
        "FDI": np.round(fdi_index, 3),
        "PERFORM": np.round(perform_index, 3),
    })

    return responses.reset_index(drop=True), thesis_data.reset_index(drop=True)


def main():
    # 1) Sampling frame
    frame = make_sampling_frame()
    frame_path = os.path.join(DATA_DIR, "sampling_frame.csv")
    frame.to_csv(frame_path, index=False)

    # 2) Neyman sample and backups + response flags
    main_sample, backup, selection = neyman_sample(frame)
    selection_path = os.path.join(DATA_DIR, "sample_selection.csv")
    selection.to_csv(selection_path, index=False)

    # 3) Simulate responses for responded firms (210)
    responses, thesis_data = simulate_responses(frame, selection)
    responses_path = os.path.join(DATA_DIR, "survey_responses.csv")
    responses.to_csv(responses_path, index=False)

    thesis_data_path = os.path.join(DATA_DIR, "thesis_data.csv")
    thesis_data.to_csv(thesis_data_path, index=False)

    # 4) Zip outputs for download
    zip_path = os.path.join(OUTPUT_DIR, "fdi_synthetic_dataset.zip")
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.write(frame_path, arcname="sampling_frame.csv")
        zf.write(selection_path, arcname="sample_selection.csv")
        zf.write(responses_path, arcname="survey_responses.csv")
        zf.write(thesis_data_path, arcname="thesis_data.csv")

    print("Generated files:")
    print(f" - {frame_path}")
    print(f" - {selection_path}")
    print(f" - {responses_path}")
    print(f" - {thesis_data_path}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
