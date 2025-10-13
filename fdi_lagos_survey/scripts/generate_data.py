#!/usr/bin/env python3
import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

N_SME = 200
N_LARGE = 100
N_TOTAL = N_SME + N_LARGE

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)

def rand_date(start: datetime, end: datetime) -> str:
    delta_days = (end - start).days
    d = start + timedelta(days=random.randint(0, delta_days))
    return d.strftime("%d/%m/%Y")

def clamp(x, low, high):
    return max(low, min(high, x))

def likert_from_latent(latent: float, min_v: int, max_v: int, sd: float = 0.9) -> int:
    noisy = latent + random.gauss(0, sd)
    val = round(noisy)
    return int(clamp(val, min_v, max_v))

def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

def choice_weighted(options, weights):
    total = sum(weights)
    r = random.random() * total
    upto = 0
    for opt, w in zip(options, weights):
        if upto + w >= r:
            return opt
        upto += w
    return options[-1]

def generate_records():
    records = []

    # Pre-build firm sizes list to enforce 200 SME, 100 large
    sizes = ["SME"] * N_SME + ["LARGE"] * N_LARGE
    random.shuffle(sizes)

    for i in range(N_TOTAL):
        firm_code = f"FIRM{(i+1):03d}"
        survey_date = rand_date(START_DATE, END_DATE)
        size = sizes[i]

        # Employee category and revenue by size
        if size == "SME":
            employees_cat = choice_weighted([
                "1-50", "51-250"
            ], [0.6, 0.4])
            annual_revenue_cat = choice_weighted([
                "<50M", "50M-500M", "500M-5B", ">5B"
            ], [0.45, 0.4, 0.13, 0.02])
            years_operation = clamp(int(random.gauss(12, 7)), 1, 50)
        else:
            employees_cat = choice_weighted([
                "251-500", "500+"
            ], [0.6, 0.4])
            annual_revenue_cat = choice_weighted([
                "<50M", "50M-500M", "500M-5B", ">5B"
            ], [0.02, 0.18, 0.5, 0.3])
            years_operation = clamp(int(random.gauss(22, 8)), 2, 60)

        # Ownership
        if size == "SME":
            ownership_type = choice_weighted(["Local", "Foreign-owned", "Joint venture"], [0.73, 0.07, 0.20])
        else:
            ownership_type = choice_weighted(["Local", "Foreign-owned", "Joint venture"], [0.45, 0.25, 0.30])

        # FDI partnership likelihood
        base_fdi_p = 0.25 if size == "SME" else 0.65
        if ownership_type == "Foreign-owned":
            base_fdi_p += 0.25
        elif ownership_type == "Joint venture":
            base_fdi_p += 0.20
        fdi_partnership = bernoulli(clamp(base_fdi_p, 0.05, 0.95))

        # FDI types (multi-select, but zero if no FDI)
        fdi_equity = fdi_joint_venture = fdi_tech_transfer = fdi_mgmt_contract = 0
        years_with_fdi = 0
        if fdi_partnership:
            years_with_fdi = clamp(int(random.gauss(6 if size == "SME" else 8, 4)), 1, min(20, years_operation))
            # Equity more common in large and foreign/JV
            fdi_equity = bernoulli(0.35 if size == "SME" else 0.6)
            fdi_joint_venture = 1 if ownership_type == "Joint venture" else bernoulli(0.15)
            fdi_tech_transfer = bernoulli(0.55)
            fdi_mgmt_contract = bernoulli(0.25)

        # Firm resources
        skilled_workforce_pct = clamp(random.gauss(52 if size == "SME" else 62, 12), 15, 95)
        training_hours = clamp(random.gauss(22 if size == "SME" else 30, 12), 0, 120)
        modern_equipment = bernoulli(0.55 if size == "SME" else 0.75)
        machinery_age = clamp(int(random.gauss(12 if size == "SME" else 9, 5)), 1, 28)
        access_credit = choice_weighted(["Easy", "Moderate", "Difficult"], [0.25 if size == "LARGE" else 0.12, 0.55, 0.20 if size == "LARGE" else 0.33])
        reinvestment_rate = clamp(random.gauss(16 if size == "SME" else 20, 8), 0, 50)

        # Firm resource composite (0..1)
        credit_norm = {"Easy": 1.0, "Moderate": 0.7, "Difficult": 0.4}[access_credit]
        skilled_norm = skilled_workforce_pct / 100.0
        training_norm = clamp(training_hours / 60.0, 0, 1)
        equipment_norm = 1.0 if modern_equipment else 0.0
        machine_norm = 1.0 - clamp(machinery_age / 20.0, 0, 1)
        reinvest_norm = clamp(reinvestment_rate / 35.0, 0, 1)
        fr_score = (
            0.25 * skilled_norm +
            0.20 * training_norm +
            0.15 * equipment_norm +
            0.15 * machine_norm +
            0.10 * credit_norm +
            0.15 * reinvest_norm
        )

        # Government policy perception (1..7)
        gp1_tax = clamp(int(random.gauss(4.0 if size == "SME" else 4.4, 1.2)), 1, 7)
        gp2_reg = clamp(int(random.gauss(3.9 if size == "SME" else 4.3, 1.2)), 1, 7)
        gp3_infra = clamp(int(random.gauss(3.6 if size == "SME" else 4.1, 1.3)), 1, 7)
        gp4_permits = clamp(int(random.gauss(3.4 if size == "SME" else 3.9, 1.3)), 1, 7)
        gp_score = (gp1_tax + gp2_reg + gp3_infra) / 3.0

        # Knowledge absorption latent
        ka_latent = 2.6 + 0.9 * fdi_partnership + 0.06 * years_with_fdi + 0.25 * training_norm + 0.15 * equipment_norm + 0.08 * gp_score
        ka1 = likert_from_latent(ka_latent + 0.2 * fdi_tech_transfer, 1, 5)
        ka2 = likert_from_latent(ka_latent + 0.1 * training_norm, 1, 5)
        ka3 = likert_from_latent(ka_latent + 0.2 * equipment_norm, 1, 5)
        ka4 = likert_from_latent(ka_latent + 0.15 * fdi_equity, 1, 5)
        ka_score = (ka1 + ka2 + ka3 + ka4) / 4.0

        # Innovation
        r_and_d_pct = clamp(random.gauss((1.6 if size == "SME" else 2.4) + 0.6 * fdi_partnership + 0.08 * gp_score + 0.2 * ka_score, 0.7), 0.0, 12.0)
        new_products = int(clamp(random.gauss(2.0 + 0.25 * r_and_d_pct + 0.8 * fdi_partnership, 1.8), 0, 20))
        iot = bernoulli(0.15 + 0.15 * fdi_partnership + 0.15 * (size == "LARGE"))
        automation = bernoulli(0.22 + 0.18 * fdi_partnership + 0.15 * (size == "LARGE") + 0.15 * equipment_norm)
        quality_mgmt = bernoulli(0.35 + 0.10 * fdi_partnership)
        process_other = bernoulli(0.08)
        inn1 = r_and_d_pct
        inn2 = new_products
        inn3 = iot + automation + quality_mgmt
        inn_score = ( (inn1 / 12.0) + (inn2 / 20.0) + (inn3 / 3.0) ) / 3.0 * 5.0  # scaled to ~1-5

        # Task performance latent
        tp_latent = 2.8 + 0.35 * (ka_score - 3.0) + 1.2 * (fr_score - 0.5) + 0.30 * (inn_score - 2.5) + 0.05 * gp_score
        tp1 = likert_from_latent(tp_latent + 0.15 * automation, 1, 5)
        tp2 = likert_from_latent(tp_latent + 0.15 * quality_mgmt, 1, 5)
        tp3 = likert_from_latent(tp_latent + 0.10 * iot, 1, 5)
        tp4 = likert_from_latent(tp_latent + 0.10 * (training_norm - 0.5), 1, 5)
        tp_score = (tp1 + tp2 + tp3 + tp4) / 4.0

        # Financial and operational performance
        roi = clamp(random.gauss(8 + 6 * (tp_score - 3.0) + 5 * (inn_score - 2.5)/2 + 3 * (fr_score - 0.5) + 1.5 * fdi_partnership + 0.5 * gp_score, 3.0), -5, 40)
        roa = clamp(random.gauss(5 + 4 * (tp_score - 3.0) + 3 * (inn_score - 2.5)/2 + 2 * (fr_score - 0.5) + 0.8 * fdi_partnership + 0.4 * gp_score, 2.5), -3, 25)
        export_intensity = clamp(random.gauss(7 + 10 * fdi_partnership + 3 * (inn_score - 2.5) + 1.0 * gp_score, 6.0), 0, 65)
        capacity_util = clamp(random.gauss(70 + 10 * (tp_score - 3.0) + 5 * (fr_score - 0.5), 8.0), 35, 98)
        market_share = clamp(random.gauss((4.5 if size == "SME" else 7.5) + 2 * (tp_score - 3.0) + 1.2 * fdi_partnership, 2.0), 0, 35)

        # Performance index (0..1 approx, then scale 0..100)
        perf_index = (
            0.25 * (roi / 40.0) +
            0.20 * (roa / 25.0) +
            0.20 * (export_intensity / 100.0) +
            0.20 * (capacity_util / 100.0) +
            0.15 * (market_share / 35.0)
        ) * 100.0
        perf_index = clamp(perf_index, 0, 100)

        # Map number of employees estimate midpoint for reference
        emp_mid = {
            "1-50": 35,
            "51-250": 150,
            "251-500": 350,
            "500+": 700,
        }[employees_cat]

        # Build record
        rec = {
            "firm_code": firm_code,
            "survey_date": survey_date,
            "years_operation": years_operation,
            "employees_cat": employees_cat,
            "employees_est": emp_mid,
            "annual_revenue_cat": annual_revenue_cat,
            "ownership_type": ownership_type,
            "fdi_partnership": "Yes" if fdi_partnership else "No",
            "FDI": fdi_partnership,
            "fdi_equity": fdi_equity,
            "fdi_joint_venture": fdi_joint_venture,
            "fdi_tech_transfer": fdi_tech_transfer,
            "fdi_mgmt_contract": fdi_mgmt_contract,
            "years_with_fdi": years_with_fdi,
            # Knowledge absorption items
            "ka1": ka1,
            "ka2": ka2,
            "ka3": ka3,
            "ka4": ka4,
            # Task performance items
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "tp4": tp4,
            # Innovation indicators
            "r_and_d_pct": round(r_and_d_pct, 2),
            "new_products_3yr": new_products,
            "proc_innov_iot": iot,
            "proc_innov_automation": automation,
            "proc_innov_quality": quality_mgmt,
            "proc_innov_other": process_other,
            "inn1": round(inn1, 2),
            "inn2": inn2,
            "inn3": inn3,
            # Firm resources
            "skilled_workforce_pct": round(skilled_workforce_pct, 1),
            "training_hours_per_employee": round(training_hours, 1),
            "modern_equipment": 1 if modern_equipment else 0,
            "machinery_age_years": machinery_age,
            "access_credit": access_credit,
            "reinvestment_rate_pct": round(reinvestment_rate, 1),
            # Government policy
            "gp1_tax_incentives": gp1_tax,
            "gp2_regulatory_stability": gp2_reg,
            "gp3_infrastructure_support": gp3_infra,
            "gp4_ease_of_permits": gp4_permits,
            # Performance metrics
            "avg_roi_pct": round(roi, 2),
            "avg_roa_pct": round(roa, 2),
            "export_intensity_pct": round(export_intensity, 2),
            "capacity_utilization_pct": round(capacity_util, 2),
            "market_share_lagos_pct": round(market_share, 2),
            # Composites
            "KA_score": round(ka_score, 3),
            "TP_score": round(tp_score, 3),
            "INN_score": round(inn_score, 3),
            "FR_score": round(fr_score, 3),
            "GP_score": round(gp_score, 3),
            "PERFORM_index": round(perf_index, 2),
        }
        records.append(rec)

    return records


def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_codebook() -> list:
    entries = []
    def add(var, label, vtype, values, section):
        entries.append({
            "variable": var,
            "label": label,
            "type": vtype,
            "allowed_values": values,
            "section": section,
        })

    # Section A
    add("firm_code", "Firm Code", "string", "FIRM001..FIRM300", "A")
    add("survey_date", "Survey date", "string", "DD/MM/YYYY", "A")
    add("years_operation", "Years of operation", "integer", ">=1", "A")
    add("employees_cat", "Employee count category", "categorical", "1-50|51-250|251-500|500+", "A")
    add("employees_est", "Employees midpoint estimate", "integer", "35/150/350/700", "A")
    add("annual_revenue_cat", "Annual revenue category (₦)", "categorical", "<50M|50M-500M|500M-5B|>5B", "A")
    add("ownership_type", "Ownership type", "categorical", "Local|Foreign-owned|Joint venture", "A")
    add("fdi_partnership", "Has FDI partnership", "categorical", "Yes|No", "A")
    add("FDI", "FDI presence (binary)", "integer", "0|1", "A")
    add("fdi_equity", "FDI type: Equity", "integer", "0|1", "A")
    add("fdi_joint_venture", "FDI type: Joint venture", "integer", "0|1", "A")
    add("fdi_tech_transfer", "FDI type: Technology transfer", "integer", "0|1", "A")
    add("fdi_mgmt_contract", "FDI type: Management contract", "integer", "0|1", "A")
    add("years_with_fdi", "Years with FDI partnership", "integer", ">=0", "A")

    # Section B (KA)
    for j, lab in enumerate([
        "Acquire technical manuals from FDI partners",
        "Staff receive training from foreign partners",
        "Adapt foreign technology to local needs",
        "Commercialize knowledge from FDI partnerships",
    ], start=1):
        add(f"ka{j}", lab, "integer", "1..5", "B")
    add("KA_score", "Knowledge absorption composite (mean of ka1..ka4)", "float", "1..5", "B")

    # Section C (TP)
    for j, lab in enumerate([
        "Production efficiency",
        "Quality control",
        "Order fulfillment time",
        "Employee productivity",
    ], start=1):
        add(f"tp{j}", lab, "integer", "1..5", "C")
    add("TP_score", "Task performance composite (mean of tp1..tp4)", "float", "1..5", "C")

    # Section D (Innovation)
    add("r_and_d_pct", "R&D spending as % of revenue", "float", "0..12", "D")
    add("new_products_3yr", "New products launched (past 3 years)", "integer", "0..20", "D")
    add("proc_innov_iot", "Process innovation: IoT systems", "integer", "0|1", "D")
    add("proc_innov_automation", "Process innovation: Automation", "integer", "0|1", "D")
    add("proc_innov_quality", "Process innovation: Quality management", "integer", "0|1", "D")
    add("proc_innov_other", "Process innovation: Other (unspecified)", "integer", "0|1", "D")
    add("inn1", "INN1 indicator (R&D intensity)", "float", "0..12", "D")
    add("inn2", "INN2 indicator (New product count)", "integer", "0..20", "D")
    add("inn3", "INN3 indicator (Process innovation adoption count)", "integer", "0..3", "D")
    add("INN_score", "Innovation composite (scaled)", "float", "~1..5", "D")

    # Section E (Resources)
    add("skilled_workforce_pct", "% of skilled workforce", "float", "0..100", "E")
    add("training_hours_per_employee", "Annual training hours per employee", "float", "0..120", "E")
    add("modern_equipment", "Use of modern equipment", "integer", "0|1", "E")
    add("machinery_age_years", "Age of primary machinery (years)", "integer", "1..28", "E")
    add("access_credit", "Access to credit", "categorical", "Easy|Moderate|Difficult", "E")
    add("reinvestment_rate_pct", "Reinvestment rate (%)", "float", "0..50", "E")
    add("FR_score", "Firm resources composite (0..1)", "float", "0..1", "E")

    # Section F (Government policy)
    add("gp1_tax_incentives", "Tax incentives effectiveness (1-7)", "integer", "1..7", "F")
    add("gp2_regulatory_stability", "Regulatory stability (1-7)", "integer", "1..7", "F")
    add("gp3_infrastructure_support", "Infrastructure support (1-7)", "integer", "1..7", "F")
    add("gp4_ease_of_permits", "Ease of obtaining permits (1-7)", "integer", "1..7", "F")
    add("GP_score", "Government policy composite (gp1..gp3 mean)", "float", "1..7", "F")

    # Section G (Performance)
    add("avg_roi_pct", "Average ROI (%) - past 3 years", "float", "-5..40", "G")
    add("avg_roa_pct", "Average ROA (%) - past 3 years", "float", "-3..25", "G")
    add("export_intensity_pct", "Export intensity (%)", "float", "0..65", "G")
    add("capacity_utilization_pct", "Production capacity utilization (%)", "float", "35..98", "G")
    add("market_share_lagos_pct", "Market share in Lagos (%)", "float", "0..35", "G")
    add("PERFORM_index", "Composite performance index (0-100)", "float", "0..100", "G")

    return entries


def main():
    # Generate
    records = generate_records()

    # Write data
    data_path = "/workspace/fdi_lagos_survey/data/survey_data.csv"
    fieldnames = list(records[0].keys())
    write_csv(data_path, records, fieldnames)

    # Write codebook
    codebook = build_codebook()
    cb_path = "/workspace/fdi_lagos_survey/docs/codebook.csv"
    with open(cb_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variable", "label", "type", "allowed_values", "section"])
        writer.writeheader()
        for row in codebook:
            writer.writerow(row)

    print(f"Wrote {len(records)} rows to {data_path}")
    print(f"Wrote codebook to {cb_path}")

if __name__ == "__main__":
    main()
