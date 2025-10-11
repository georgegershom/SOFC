#!/usr/bin/env python3
"""
Generate a fabricated SSA FinTech micro-foundation dataset with plausible ranges.

Outputs:
- data/processed/fintech_micro_foundation.csv
- data/processed/fintech_events.csv (regulatory actions and funding events)

Notes:
- This is purely synthetic and should be replaced with real data collection for research.
"""
from __future__ import annotations
import csv
import math
import os
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List, Dict, Tuple

random.seed(42)

COUNTRIES = [
    "Nigeria", "Kenya", "Ghana", "South Africa", "Uganda",
    "Tanzania", "Rwanda", "Ethiopia", "Côte d'Ivoire", "Senegal",
]

# Representative FinTech verticals in SSA
VERTICALS = [
    "Payments/Mobile Money", "Lending", "Savings/Wealth", "Remittances",
    "Insurtech", "Crypto/FX", "Agent Networks", "Merchant Acquiring",
]

# Example major players per country (mix of large and startups)
EXAMPLE_PLAYERS: Dict[str, List[str]] = {
    "Nigeria": ["Flutterwave", "Paystack", "OPay", "Paga", "Moniepoint"],
    "Kenya": ["M-Pesa", "Tala", "M-KOPA", "Airtel Money", "Kopo Kopo"],
    "Ghana": ["ExpressPay", "Zeepay", "MTN MoMo", "Hubtel", "AirtelTigo"],
    "South Africa": ["Yoco", "TymeBank", "Ozow", "PayFast", "Luno"],
    "Uganda": ["MTN MoMo", "Airtel Money", "Wave", "Chipper Cash", "SafeBoda"],
    "Tanzania": ["Tigo Pesa", "Vodacom M-Pesa", "Airtel Money", "Nala", "Selcom"],
    "Rwanda": ["MTN MoMo", "Airtel Money", "BK App", "YegoMoto", "Wave"],
    "Ethiopia": ["Telebirr", "HelloCash", "Amole", "Chapa", "ArifPay"],
    "Côte d'Ivoire": ["Orange Money", "MTN MoMo", "Wave", "Moov Money", "Coris Pay"],
    "Senegal": ["Wave", "Orange Money", "Free Money", "Wari", "Expresso"],
}

@dataclass
class FintechMonthly:
    as_of: date
    country: str
    firm: str
    vertical: str
    revenue_usd: float
    revenue_growth_pct: float
    net_income_usd: float
    burn_rate_usd: float
    funding_stage: str
    funding_total_usd: float
    active_users: int
    tx_volume_usd: float
    tx_count: int
    agents: int
    cac_usd: float
    churn_rate_pct: float
    regulatory_sanction: int
    distress_label: int


FUNDING_STAGES = ["Seed", "Series A", "Series B", "Series C", "Growth", "Private/Strategic"]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d = date(d.year + (1 if d.month == 12 else 0), 1 if d.month == 12 else d.month + 1, 1)


def make_baseline_scales(vertical: str) -> Dict[str, float]:
    # Rough order-of-magnitude baselines by vertical
    if vertical == "Payments/Mobile Money":
        return {
            "revenue": 8e6,
            "users": 2_000_000,
            "tx_volume": 5e8,
            "tx_count": 15_000_000,
            "agents": 80_000,
        }
    if vertical == "Lending":
        return {
            "revenue": 3e6,
            "users": 350_000,
            "tx_volume": 9e7,
            "tx_count": 450_000,
            "agents": 8_000,
        }
    if vertical == "Savings/Wealth":
        return {
            "revenue": 1.5e6,
            "users": 150_000,
            "tx_volume": 3.5e7,
            "tx_count": 180_000,
            "agents": 2_000,
        }
    if vertical == "Remittances":
        return {
            "revenue": 2.2e6,
            "users": 220_000,
            "tx_volume": 1.8e8,
            "tx_count": 700_000,
            "agents": 4,  # app-based
        }
    if vertical == "Insurtech":
        return {
            "revenue": 9e5,
            "users": 120_000,
            "tx_volume": 1.2e7,
            "tx_count": 90_000,
            "agents": 1_500,
        }
    if vertical == "Crypto/FX":
        return {
            "revenue": 2.8e6,
            "users": 280_000,
            "tx_volume": 2.5e8,
            "tx_count": 900_000,
            "agents": 500,
        }
    if vertical == "Agent Networks":
        return {
            "revenue": 2.5e6,
            "users": 600_000,
            "tx_volume": 1.6e8,
            "tx_count": 8_000_000,
            "agents": 120_000,
        }
    if vertical == "Merchant Acquiring":
        return {
            "revenue": 1.9e6,
            "users": 300_000,
            "tx_volume": 1.1e8,
            "tx_count": 1_800_000,
            "agents": 35_000,
        }
    return {"revenue": 1e6, "users": 100_000, "tx_volume": 5e7, "tx_count": 100_000, "agents": 1_000}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def simulate_firm_months(country: str, firm: str, vertical: str, start: date, end: date) -> List[FintechMonthly]:
    scales = make_baseline_scales(vertical)
    records: List[FintechMonthly] = []

    # Firm-specific latent traits
    quality = random.uniform(0.6, 1.4)  # higher = better unit economics
    growth_bias = random.uniform(0.9, 1.2)  # compounding growth tendency
    funding_stage = random.choice(FUNDING_STAGES)
    funding_total = random.uniform(2e6, 250e6) * (1.0 if funding_stage in {"Seed", "Series A"} else random.uniform(1.2, 3.5))

    # Initialize metrics
    revenue = random.uniform(0.6, 1.4) * scales["revenue"]
    users = int(random.uniform(0.6, 1.4) * scales["users"])
    tx_volume = random.uniform(0.6, 1.4) * scales["tx_volume"]
    tx_count = int(random.uniform(0.6, 1.4) * scales["tx_count"])
    agents = int(random.uniform(0.6, 1.4) * scales["agents"])

    cash_on_hand = random.uniform(6, 18) * 1e6

    regulatory_flag = 0
    distress_state = 0
    consecutive_downturn = 0

    prev_revenue = revenue
    prev_users = users

    for d in daterange(start, end):
        # Random quarterly shocks and noise
        season = 1.0 + 0.08 * math.sin(2 * math.pi * ((d.month % 12) / 12.0))
        macro_shock = random.gauss(0.0, 0.03)
        regulatory_shock = 0.0

        # Occasional regulatory sanction event (short-lived impact)
        regulatory_event = 1 if random.random() < 0.02 else 0
        if regulatory_event:
            regulatory_flag = 1
            regulatory_shock = -random.uniform(0.05, 0.2)
        else:
            regulatory_flag = 0

        # Growth and noise updates
        growth_factor = (1.0 + 0.02 * growth_bias + macro_shock + regulatory_shock)
        revenue = max(0.0, revenue * growth_factor * season * random.uniform(0.96, 1.04))
        users = max(0, int(users * (1.0 + 0.015 * growth_bias + macro_shock + regulatory_shock) * random.uniform(0.98, 1.03)))
        tx_volume = max(0.0, tx_volume * (1.0 + 0.018 * growth_bias + macro_shock + regulatory_shock) * season * random.uniform(0.97, 1.04))
        tx_count = max(0, int(tx_count * (1.0 + 0.01 * growth_bias + macro_shock + regulatory_shock) * random.uniform(0.98, 1.02)))
        agents = max(0, int(agents * (1.0 + 0.005 * growth_bias + random.gauss(0, 0.01))))

        # Unit economics
        take_rate = clamp(random.gauss(0.015 if vertical == "Payments/Mobile Money" else 0.02, 0.006), 0.004, 0.05)
        implied_revenue = take_rate * tx_volume
        revenue = 0.6 * revenue + 0.4 * implied_revenue

        gross_margin = clamp(0.4 * quality + random.gauss(0.15, 0.08), 0.1, 0.9)
        opex = clamp(0.6 * revenue * (1.2 - quality) + random.uniform(200_000, 2_000_000), 80_000, 8_000_000)
        net_income = revenue * gross_margin - opex

        # CAC and churn dynamics
        cac = clamp(random.gauss(2.5 if vertical == "Payments/Mobile Money" else 6.0, 2.0) * (1.2 - quality), 0.5, 25.0)
        churn_rate = clamp(random.gauss(0.012 if vertical == "Payments/Mobile Money" else 0.02, 0.01) * (1.3 - quality), 0.002, 0.15)

        # Burn rate when negative income
        burn = -min(0.0, net_income)
        cash_on_hand = max(0.0, cash_on_hand - burn + max(0.0, random.gauss(0, 0.15) * 1e6))

        # Funding cadence: occasional top-ups increase cash and funding_total
        if random.random() < 0.03:
            top_up = random.uniform(1e6, 35e6)
            cash_on_hand += top_up
            funding_total += top_up
            # Stage progression
            stage_idx = min(len(FUNDING_STAGES) - 1, FUNDING_STAGES.index(funding_stage) + (1 if random.random() < 0.5 else 0))
            funding_stage = FUNDING_STAGES[stage_idx]

        # Distress heuristic
        revenue_growth = (revenue - prev_revenue) / prev_revenue if prev_revenue > 0 else 0.0
        user_growth = (users - prev_users) / prev_users if prev_users > 0 else 0.0
        sharp_drop = (revenue_growth < -0.25) or (user_growth < -0.15)
        low_cash = cash_on_hand < 3e6 and burn > 0.5e6
        consecutive_downturn = consecutive_downturn + 1 if (revenue_growth < -0.05 and user_growth < -0.02) else 0

        distress_state = 1 if (sharp_drop or low_cash or consecutive_downturn >= 4 or regulatory_flag == 1 and random.random() < 0.3) else 0

        # Construct record
        record = FintechMonthly(
            as_of=d,
            country=country,
            firm=firm,
            vertical=vertical,
            revenue_usd=round(revenue, 2),
            revenue_growth_pct=round(100.0 * revenue_growth, 2),
            net_income_usd=round(net_income, 2),
            burn_rate_usd=round(burn, 2),
            funding_stage=funding_stage,
            funding_total_usd=round(funding_total, 2),
            active_users=int(users),
            tx_volume_usd=round(tx_volume, 2),
            tx_count=int(tx_count),
            agents=int(agents),
            cac_usd=round(cac, 2),
            churn_rate_pct=round(100.0 * churn_rate, 3),
            regulatory_sanction=int(regulatory_flag),
            distress_label=int(distress_state),
        )
        records.append(record)

        prev_revenue = revenue
        prev_users = users

    return records


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    start = date(2019, 1, 1)
    end = date(2024, 12, 1)

    firms: List[Tuple[str, str, str]] = []  # (country, firm, vertical)
    for c in COUNTRIES:
        players = EXAMPLE_PLAYERS.get(c, [])
        for firm in players:
            vertical = random.choice(VERTICALS)
            firms.append((c, firm, vertical))

    all_rows: List[Dict[str, object]] = []

    for country, firm, vertical in firms:
        monthly = simulate_firm_months(country, firm, vertical, start, end)
        for m in monthly:
            row = asdict(m)
            row["as_of"] = m.as_of.isoformat()
            all_rows.append(row)

    # Secondary events table (regulatory and funding events)
    events: List[Dict[str, object]] = []
    for row in all_rows:
        # Include regulatory events and notable funding events as separate table
        if row["regulatory_sanction"] == 1:
            events.append({
                "as_of": row["as_of"],
                "country": row["country"],
                "firm": row["firm"],
                "type": "regulatory_sanction",
                "details": "Reported regulatory action impacting operations",
            })
        if random.random() < 0.015:  # random funding news
            events.append({
                "as_of": row["as_of"],
                "country": row["country"],
                "firm": row["firm"],
                "type": "funding_round",
                "details": f"Raised USD {int(random.uniform(1, 50))}m",
            })

    fieldnames = [
        "as_of", "country", "firm", "vertical",
        "revenue_usd", "revenue_growth_pct", "net_income_usd", "burn_rate_usd",
        "funding_stage", "funding_total_usd",
        "active_users", "tx_volume_usd", "tx_count", "agents",
        "cac_usd", "churn_rate_pct",
        "regulatory_sanction", "distress_label",
    ]

    write_csv("data/processed/fintech_micro_foundation.csv", all_rows, fieldnames)

    event_fields = ["as_of", "country", "firm", "type", "details"]
    write_csv("data/processed/fintech_events.csv", events, event_fields)

    print(f"Rows written: {len(all_rows)} to data/processed/fintech_micro_foundation.csv")
    print(f"Events written: {len(events)} to data/processed/fintech_events.csv")


if __name__ == "__main__":
    main()
