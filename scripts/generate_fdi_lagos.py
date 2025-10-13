#!/usr/bin/env python3

import argparse
import csv
import math
import os
import random
from typing import List, Dict


def clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def bounded_int_normal(mean: float, sd: float, lower: int, upper: int) -> int:
    # Box-Muller for normal, truncated by clamping
    u1 = random.random() or 1e-9
    u2 = random.random() or 1e-9
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    val = mean + sd * z
    return int(round(clamp(val, lower, upper)))


def bounded_float_normal(mean: float, sd: float, lower: float, upper: float) -> float:
    u1 = random.random() or 1e-9
    u2 = random.random() or 1e-9
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    val = mean + sd * z
    return float(clamp(val, lower, upper))


def logistic(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def choose_weighted(items: List[str], weights: List[float]) -> str:
    assert len(items) == len(weights)
    r = random.random() * sum(weights)
    cum = 0.0
    for item, w in zip(items, weights):
        cum += w
        if r <= cum:
            return item
    return items[-1]


def generate_record(idx: int) -> Dict[str, object]:
    firm_id = f"LAG-FP-{idx:04d}"

    subsectors = [
        "beverages",
        "bakery",
        "dairy",
        "meat",
        "fruits_veg",
        "grains_cereals",
        "oils_fats",
        "confectionery",
    ]
    subsector = choose_weighted(
        subsectors,
        [0.16, 0.18, 0.10, 0.08, 0.12, 0.18, 0.10, 0.08],
    )

    ownership_type = choose_weighted(["domestic", "foreign", "joint_venture"], [0.68, 0.14, 0.18])

    if ownership_type == "domestic":
        foreign_ownership_percent = 0.0
        fdi_presence = 1 if random.random() < 0.08 else 0  # tech/contract-based FDI without equity
    elif ownership_type == "foreign":
        foreign_ownership_percent = bounded_float_normal(80.0, 10.0, 50.0, 100.0)
        fdi_presence = 1
    else:  # joint venture
        foreign_ownership_percent = bounded_float_normal(45.0, 20.0, 10.0, 90.0)
        fdi_presence = 1

    years_since_fdi = 0 if fdi_presence == 0 else int(bounded_float_normal(6.0, 4.0, 1.0, 25.0))

    age_years = int(bounded_float_normal(15.0, 10.0, 1.0, 60.0))

    # Size and assets vary by subsector
    subsector_size_mu = {
        "beverages": (120, 1.8),
        "bakery": (70, 1.5),
        "dairy": (110, 1.6),
        "meat": (90, 1.5),
        "fruits_veg": (85, 1.4),
        "grains_cereals": (95, 1.6),
        "oils_fats": (130, 1.7),
        "confectionery": (60, 1.4),
    }
    size_mu, size_sd = subsector_size_mu[subsector]
    employees = int(bounded_float_normal(size_mu, size_sd * size_mu * 0.25, 5, 1500))

    assets_mu_by_size = max(10.0, employees * 0.25)  # million NGN
    assets_million_ngn = round(bounded_float_normal(assets_mu_by_size, assets_mu_by_size * 0.4, 2.0, 5000.0), 2)

    # Policy perceptions (1–7). Lagos typical mid but dispersed.
    policy_tax_incentives = bounded_int_normal(4.5, 1.4, 1, 7)
    policy_regulatory_stability = bounded_int_normal(4.2, 1.5, 1, 7)
    policy_infrastructure_support = bounded_int_normal(4.0, 1.6, 1, 7)
    policy_corruption_experience = bounded_int_normal(3.6, 1.7, 1, 7)  # higher=worse

    policy_effectiveness_index = round(
        (
            policy_tax_incentives
            + policy_regulatory_stability
            + policy_infrastructure_support
            + (8 - policy_corruption_experience)
        )
        / 4.0,
        2,
    )

    policy_incentive_received = 1 if random.random() < logistic((policy_tax_incentives - 4) * 0.8 + (ownership_type != "domestic") * 0.6) else 0

    # Human capital and technology
    skilled_labor_ratio = round(
        clamp(
            bounded_float_normal(0.42 + 0.12 * (1 if fdi_presence else 0), 0.15, 0.05, 0.9),
            0.05,
            0.9,
        ),
        2,
    )

    rd_spend_pct_revenue = round(
        clamp(
            bounded_float_normal(1.2 + 0.8 * (1 if fdi_presence else 0) + 0.2 * (policy_effectiveness_index - 4), 0.9, 0.0, 8.0),
            0.0,
            8.0,
        ),
        2,
    )

    iot_use_prob = logistic(-1.0 + 0.9 * (1 if fdi_presence else 0) + 0.3 * (policy_effectiveness_index - 4))
    iot_use = 1 if random.random() < iot_use_prob else 0

    liquidity_ratio = round(bounded_float_normal(1.5, 0.5, 0.5, 3.5), 2)

    # Latents for constructs
    absorptive_latent = (
        0.2 * (skilled_labor_ratio * 10)
        + 0.15 * (rd_spend_pct_revenue)
        + 0.5 * (1 if fdi_presence else 0)
        + 0.1 * (policy_effectiveness_index - 4)
        + random.uniform(-0.8, 0.8)
    )

    innovation_latent = (
        0.5 * (rd_spend_pct_revenue / 2.0)
        + 0.8 * iot_use
        + 0.2 * (policy_effectiveness_index - 4)
        + 0.2 * (1 if fdi_presence else 0)
        + random.uniform(-0.6, 0.6)
    )

    resources_latent = (
        0.4 * (skilled_labor_ratio * 10)
        + 0.3 * (liquidity_ratio)
        + 0.3 * (1 if iot_use else 0)
        + random.uniform(-0.7, 0.7)
    )

    task_perf_latent = (
        0.35 * absorptive_latent
        + 0.25 * resources_latent
        + 0.2 * (policy_effectiveness_index - 4)
        + random.uniform(-0.7, 0.7)
    )

    # Likert items from latents
    def latent_to_likert5(lat: float, center: float = 3.0) -> int:
        mean = clamp(center + lat * 0.35, 1.0, 5.0)
        return bounded_int_normal(mean, 1.0, 1, 5)

    ka_i1 = latent_to_likert5(absorptive_latent)
    ka_i2 = latent_to_likert5(absorptive_latent)
    ka_i3 = latent_to_likert5(absorptive_latent)
    ka_i4 = latent_to_likert5(absorptive_latent)
    knowledge_absorption_index = round((ka_i1 + ka_i2 + ka_i3 + ka_i4) / 4.0, 2)

    tp_i1 = latent_to_likert5(task_perf_latent)
    tp_i2 = latent_to_likert5(task_perf_latent)
    tp_i3 = latent_to_likert5(task_perf_latent)
    task_performance_index = round((tp_i1 + tp_i2 + tp_i3) / 3.0, 2)

    inn_i1 = latent_to_likert5(innovation_latent)
    inn_i2 = latent_to_likert5(innovation_latent)
    inn_i3 = latent_to_likert5(innovation_latent)
    innovation_index = round((inn_i1 + inn_i2 + inn_i3) / 3.0, 2)

    res_hr = latent_to_likert5(resources_latent)
    res_tech = latent_to_likert5(resources_latent)
    res_fin = latent_to_likert5(resources_latent)
    firm_resources_index = round((res_hr + res_tech + res_fin) / 3.0, 2)

    # Performance with moderation by policy
    # Base performance components
    perf_base = 6.0 + 0.02 * (employees ** 0.5) + 0.4 * (policy_effectiveness_index - 4)

    moderation_multiplier = 1.0 + 0.08 * (policy_effectiveness_index - 4)

    roi_pct = perf_base + moderation_multiplier * (
        2.4 * (knowledge_absorption_index - 3)
        + 2.0 * (innovation_index - 3)
        + 1.6 * (task_performance_index - 3)
        + 1.2 * (firm_resources_index - 3)
    ) + random.uniform(-2.0, 2.0)
    roi_pct = round(clamp(roi_pct, 2.0, 35.0), 2)

    roa_pct = 0.6 * roi_pct + random.uniform(-2.0, 2.0)
    roa_pct = round(clamp(roa_pct, 1.0, 20.0), 2)

    export_base = 3.0 + 6.0 * (1 if fdi_presence else 0) + 2.0 * (knowledge_absorption_index - 3)
    export_intensity_pct = export_base * moderation_multiplier + random.uniform(-3.0, 3.0)
    export_intensity_pct = round(clamp(export_intensity_pct, 0.0, 60.0), 2)

    eff_base = 50.0 + 6.0 * (task_performance_index - 3) + 4.0 * (firm_resources_index - 3)
    operational_efficiency_index = eff_base * moderation_multiplier + random.uniform(-8.0, 8.0)
    operational_efficiency_index = round(clamp(operational_efficiency_index, 20.0, 100.0), 2)

    # Market share depends on size, performance, subsector scale
    subsector_scale = {
        "beverages": 1.2,
        "bakery": 0.9,
        "dairy": 1.1,
        "meat": 1.0,
        "fruits_veg": 0.8,
        "grains_cereals": 1.0,
        "oils_fats": 1.1,
        "confectionery": 0.8,
    }[subsector]
    market_share_pct = (
        0.02 * (employees ** 0.6)
        + 0.06 * (roi_pct)
        + 0.05 * (export_intensity_pct / 2.0)
    ) * 0.2 * subsector_scale + random.uniform(-1.0, 1.0)
    market_share_pct = round(clamp(market_share_pct, 0.0, 20.0), 2)

    record = {
        "firm_id": firm_id,
        "subsector": subsector,
        "ownership_type": ownership_type,
        "foreign_ownership_percent": round(foreign_ownership_percent, 2),
        "years_since_fdi": years_since_fdi,
        "age_years": age_years,
        "employees": employees,
        "assets_million_ngn": assets_million_ngn,
        # Policy
        "policy_tax_incentives": policy_tax_incentives,
        "policy_regulatory_stability": policy_regulatory_stability,
        "policy_infrastructure_support": policy_infrastructure_support,
        "policy_corruption_experience": policy_corruption_experience,
        "policy_effectiveness_index": policy_effectiveness_index,
        "policy_incentive_received": policy_incentive_received,
        # Resources objective
        "skilled_labor_ratio": skilled_labor_ratio,
        "iot_use": iot_use,
        "rd_spend_pct_revenue": rd_spend_pct_revenue,
        "liquidity_ratio": liquidity_ratio,
        # Likert items and indices
        "ka_i1": ka_i1,
        "ka_i2": ka_i2,
        "ka_i3": ka_i3,
        "ka_i4": ka_i4,
        "knowledge_absorption_index": knowledge_absorption_index,
        "tp_i1": tp_i1,
        "tp_i2": tp_i2,
        "tp_i3": tp_i3,
        "task_performance_index": task_performance_index,
        "inn_i1": inn_i1,
        "inn_i2": inn_i2,
        "inn_i3": inn_i3,
        "innovation_index": innovation_index,
        "res_hr": res_hr,
        "res_tech": res_tech,
        "res_fin": res_fin,
        "firm_resources_index": firm_resources_index,
        # Performance
        "roi_pct": roi_pct,
        "roa_pct": roa_pct,
        "export_intensity_pct": export_intensity_pct,
        "market_share_pct": market_share_pct,
        "operational_efficiency_index": operational_efficiency_index,
        # FDI presence flag
        "fdi_presence": fdi_presence,
    }

    return record


def generate_dataset(n: int, seed: int) -> List[Dict[str, object]]:
    random.seed(seed)
    return [generate_record(i + 1) for i in range(n)]


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FDI-Lagos firm-level survey dataset")
    parser.add_argument("--n", type=int, default=500, help="Number of firms to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="/workspace/data/fdi_lagos_firm_survey.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = generate_dataset(args.n, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()
