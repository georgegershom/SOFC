#!/usr/bin/env python3
"""
Synthetic SSA FinTech micro dataset generator (Category 1: FinTech-Specific Data).

This script fabricates anonymized firm-month panel data for SSA FinTechs, including
financial performance, operational metrics, funding events, sanctions, downturns,
and distress outcomes suitable for early-warning modeling.

Notes:
- All data are synthetic and anonymized (Firm_###). No real firms are represented.
- Distributions are calibrated to be plausible for SSA contexts but are not factual.
- Outputs: monthly panel CSV/Parquet, firm-level CSV, and a data dictionary CSV.

Example:
  python scripts/generate_fintech_micro.py \
    --num-firms 120 --start 2018-01 --end 2025-06 \
    --output-dir data/fintech_micro --seed 42 --write-parquet
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional parquet support
HAS_PARQUET = False
try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:
    HAS_PARQUET = False


# ----------------------------- Configuration ----------------------------- #
COUNTRIES = [
    ("NG", "Nigeria", 0.20, 0.55, 0.55),  # weight, mobile penetration proxy, regulatory strictness
    ("KE", "Kenya", 0.12, 0.75, 0.50),
    ("GH", "Ghana", 0.08, 0.60, 0.55),
    ("ZA", "South Africa", 0.12, 0.80, 0.65),
    ("TZ", "Tanzania", 0.06, 0.55, 0.60),
    ("UG", "Uganda", 0.05, 0.55, 0.60),
    ("RW", "Rwanda", 0.04, 0.65, 0.50),
    ("SN", "Senegal", 0.05, 0.55, 0.55),
    ("CI", "Cote d'Ivoire", 0.04, 0.55, 0.55),
    ("CM", "Cameroon", 0.04, 0.50, 0.60),
    ("ZM", "Zambia", 0.04, 0.50, 0.60),
    ("ZW", "Zimbabwe", 0.03, 0.40, 0.70),
    ("MZ", "Mozambique", 0.03, 0.45, 0.60),
    ("BW", "Botswana", 0.04, 0.55, 0.50),
    ("ET", "Ethiopia", 0.06, 0.35, 0.70),
]

SEGMENTS = [
    ("mobile_money", 0.22),
    ("payments_gateway", 0.30),
    ("digital_lending", 0.23),
    ("savings_wallet", 0.10),
    ("insuretech", 0.08),
    ("wealth_investing", 0.07),
]

FUNDING_STAGES_ORDERED = [
    "bootstrapped",
    "pre_seed",
    "seed",
    "series_a",
    "series_b",
    "series_c_plus",
]

# Segment parameterization (plausible SSA ranges)
SEGMENT_PARAMS = {
    "mobile_money": {
        "avg_txn_value": (8, 40),  # USD per transaction
        "avg_txn_count_per_user": (2.5, 12.0),
        "take_rate": (0.006, 0.012),  # revenue share of volume
        "agents_base": (800, 25000),
        "agents_growth": (0.0, 0.02),  # monthly
        "cac": (1.0, 4.0),
        "churn": (0.01, 0.05),  # monthly
        "users_initial": (15_000, 2_200_000),
        "users_momentum": (0.002, 0.025),
    },
    "payments_gateway": {
        "avg_txn_value": (12, 120),
        "avg_txn_count_per_user": (1.0, 4.0),
        "take_rate": (0.004, 0.010),
        "agents_base": (0, 0),
        "agents_growth": (0.0, 0.0),
        "cac": (2.0, 8.0),
        "churn": (0.02, 0.08),
        "users_initial": (6_000, 600_000),
        "users_momentum": (0.000, 0.020),
    },
    "digital_lending": {
        "avg_txn_value": (40, 220),  # loan size proxy
        "avg_txn_count_per_user": (0.2, 0.8),
        "take_rate": (0.020, 0.050),  # interest and fees
        "agents_base": (0, 0),
        "agents_growth": (0.0, 0.0),
        "cac": (5.0, 18.0),
        "churn": (0.04, 0.12),
        "users_initial": (2_000, 220_000),
        "users_momentum": (0.000, 0.018),
    },
    "savings_wallet": {
        "avg_txn_value": (6, 50),
        "avg_txn_count_per_user": (1.0, 5.0),
        "take_rate": (0.004, 0.010),
        "agents_base": (0, 0),
        "agents_growth": (0.0, 0.0),
        "cac": (2.0, 6.0),
        "churn": (0.015, 0.06),
        "users_initial": (3_000, 400_000),
        "users_momentum": (0.000, 0.020),
    },
    "insuretech": {
        "avg_txn_value": (5, 30),
        "avg_txn_count_per_user": (0.3, 1.2),
        "take_rate": (0.070, 0.160),  # margin-like
        "agents_base": (0, 0),
        "agents_growth": (0.0, 0.0),
        "cac": (3.0, 10.0),
        "churn": (0.010, 0.05),
        "users_initial": (2_500, 150_000),
        "users_momentum": (0.000, 0.016),
    },
    "wealth_investing": {
        "avg_txn_value": (20, 240),
        "avg_txn_count_per_user": (0.3, 1.5),
        "take_rate": (0.003, 0.010),
        "agents_base": (0, 0),
        "agents_growth": (0.0, 0.0),
        "cac": (3.0, 12.0),
        "churn": (0.012, 0.05),
        "users_initial": (2_000, 180_000),
        "users_momentum": (0.000, 0.015),
    },
}

# Funding distributions in USD
FUNDING_DISTRIBUTIONS = {
    "pre_seed": (50_000, 800_000),
    "seed": (300_000, 3_000_000),
    "series_a": (2_000_000, 25_000_000),
    "series_b": (10_000_000, 80_000_000),
    "series_c_plus": (30_000_000, 200_000_000),
}


@dataclass
class Firm:
    firm_id: str
    country_iso: str
    country_name: str
    segment: str
    founding_year: int
    funding_stage: str
    base_scale: float  # proxy for initial size/momentum
    regulatory_strictness: float
    mobile_penetration: float


def softplus(x: float) -> float:
    return math.log1p(math.exp(x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def choose_weighted(items_with_weights: List[Tuple], rng: random.Random):
    items = [i[0] for i in items_with_weights]
    weights = [i[1] for i in items_with_weights]
    return rng.choices(items, weights=weights, k=1)[0]


def month_range(start: str, end: str) -> List[pd.Timestamp]:
    idx = pd.period_range(start=start, end=end, freq="M").to_timestamp()
    return list(idx)


def generate_firms(n: int, rng: random.Random) -> List[Firm]:
    country_choices = [(c[0], c[2]) for c in COUNTRIES]  # iso, weight
    firms: List[Firm] = []
    for i in range(n):
        iso = rng.choices([c[0] for c in COUNTRIES], weights=[c[2] for c in COUNTRIES], k=1)[0]
        country_tuple = next(c for c in COUNTRIES if c[0] == iso)
        country_name = country_tuple[1]
        mobile_penetration = country_tuple[3]
        regulatory_strictness = country_tuple[4]
        segment = choose_weighted(SEGMENTS, rng)

        founding_year = rng.randint(2010, 2024)
        # Stage depends on age
        age = max(0, 2025 - founding_year)
        if age < 2:
            funding_stage = rng.choices(
                ["bootstrapped", "pre_seed", "seed"], weights=[0.50, 0.30, 0.20], k=1
            )[0]
        elif age < 5:
            funding_stage = rng.choices(
                ["pre_seed", "seed", "series_a"], weights=[0.10, 0.55, 0.35], k=1
            )[0]
        else:
            funding_stage = rng.choices(
                ["seed", "series_a", "series_b", "series_c_plus"],
                weights=[0.20, 0.45, 0.25, 0.10],
                k=1,
            )[0]

        # Base scale approximates initial size; linked to stage and segment
        stage_scale = FUNDING_STAGES_ORDERED.index(funding_stage) + 1
        seg_params = SEGMENT_PARAMS[segment]
        users_base_lo, users_base_hi = seg_params["users_initial"]
        base_users = rng.uniform(math.sqrt(users_base_lo), math.sqrt(users_base_hi)) ** 2
        base_scale = base_users / 100_000.0

        firms.append(
            Firm(
                firm_id=f"FIRM_{i+1:03d}",
                country_iso=iso,
                country_name=country_name,
                segment=segment,
                founding_year=founding_year,
                funding_stage=funding_stage,
                base_scale=float(base_scale * (0.6 + 0.8 * rng.random())),
                regulatory_strictness=float(regulatory_strictness),
                mobile_penetration=float(mobile_penetration),
            )
        )
    return firms


def simulate_panel(
    firms: List[Firm], months: List[pd.Timestamp], seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    rows = []
    firm_rows = []

    for firm in firms:
        segp = SEGMENT_PARAMS[firm.segment]
        users_initial_lo, users_initial_hi = segp["users_initial"]
        users_momentum_lo, users_momentum_hi = segp["users_momentum"]
        avg_txn_value_lo, avg_txn_value_hi = segp["avg_txn_value"]
        avg_txn_count_lo, avg_txn_count_hi = segp["avg_txn_count_per_user"]
        take_rate_lo, take_rate_hi = segp["take_rate"]
        cac_lo, cac_hi = segp["cac"]
        churn_lo, churn_hi = segp["churn"]

        # Initial state
        active_users = py_rng.uniform(users_initial_lo, users_initial_hi) * (0.7 + 0.6 * firm.base_scale)
        avg_txn_value = py_rng.uniform(avg_txn_value_lo, avg_txn_value_hi)
        avg_txn_count = py_rng.uniform(avg_txn_count_lo, avg_txn_count_hi)
        take_rate = py_rng.uniform(take_rate_lo, take_rate_hi)
        agents_count = 0
        if firm.segment == "mobile_money":
            agents_base_lo, agents_base_hi = segp["agents_base"]
            agents_count = int(py_rng.uniform(agents_base_lo, agents_base_hi) * (0.7 + 0.6 * firm.base_scale))

        # Funding/finance state
        stage_index = FUNDING_STAGES_ORDERED.index(firm.funding_stage)
        cumulative_funding = 0.0
        if firm.funding_stage in FUNDING_DISTRIBUTIONS:
            lo, hi = FUNDING_DISTRIBUTIONS[firm.funding_stage]
            cumulative_funding += py_rng.uniform(lo, hi)
        cash_balance = cumulative_funding * (0.35 + 0.25 * py_rng.random())

        # Opex baseline influenced by scale and country
        opex_baseline = 50_000 * (0.4 + 1.8 * firm.base_scale) * (0.8 + 0.4 * (1.0 - firm.mobile_penetration))

        # Sanction state
        sanction_cooldown = 0
        sanction_severity = 0.0

        is_active = True
        had_distress = False
        distress_type = "none"
        distress_happened_on: datetime | None = None

        # Track for downturn detection
        revenue_history: List[float] = []
        users_history: List[float] = []

        for t, month in enumerate(months):
            # Stop realistic metrics after distress default/closure
            if not is_active:
                rows.append(
                    {
                        "firm_id": firm.firm_id,
                        "date": month.date(),
                        "country": firm.country_iso,
                        "segment": firm.segment,
                        "is_active": 0,
                        "revenue_usd": 0.0,
                        "net_income_usd": 0.0,
                        "burn_rate_usd": 0.0,
                        "active_users": 0,
                        "transaction_volume_usd": 0.0,
                        "transaction_count": 0,
                        "agents_count": 0 if firm.segment == "mobile_money" else np.nan,
                        "cac_usd": np.nan,
                        "churn_rate": np.nan,
                        "funding_round_amount_usd": 0.0,
                        "funding_stage": FUNDING_STAGES_ORDERED[min(stage_index, len(FUNDING_STAGES_ORDERED)-1)],
                        "cumulative_funding_usd": cumulative_funding,
                        "regulatory_sanction_flag": 0,
                        "sanction_severity": 0.0,
                        "downturn_flag": 0,
                        "distress_event": 1 if had_distress else 0,
                        "distress_type": distress_type,
                    }
                )
                continue

            # Growth dynamics
            momentum = py_rng.uniform(users_momentum_lo, users_momentum_hi)
            momentum *= (0.75 + 0.5 * firm.mobile_penetration)  # higher penetration supports growth
            momentum *= (0.9 + 0.2 * (1 - firm.regulatory_strictness))  # heavier regulation -> a bit slower
            # Shock term
            growth_shock = float(rng.normal(0.0, 0.01))
            # Apply sanctions drag if any
            sanction_drag = 1.0
            if sanction_cooldown > 0:
                sanction_drag = clamp(1.0 - 0.15 * sanction_severity, 0.55, 0.98)
                sanction_cooldown -= 1

            active_users = max(
                0.0,
                active_users * (1.0 + momentum + growth_shock) * sanction_drag
                - active_users * py_rng.uniform(churn_lo, churn_hi),
            )

            # Agents for mobile money
            if firm.segment == "mobile_money":
                agents_growth = py_rng.uniform(SEGMENT_PARAMS["mobile_money"]["agents_growth"][0], SEGMENT_PARAMS["mobile_money"]["agents_growth"][1])
                agents_growth *= sanction_drag
                agents_count = int(max(0, agents_count * (1.0 + agents_growth + float(rng.normal(0.0, 0.01)))))

            # Transactions and revenue
            txn_count = max(0.0, active_users * max(0.2, avg_txn_count + float(rng.normal(0.0, 0.15))))
            avg_value = max(1.0, avg_txn_value + float(rng.normal(0.0, 2.0)))
            txn_volume = txn_count * avg_value
            revenue = txn_volume * max(0.0005, take_rate + float(rng.normal(0.0, take_rate * 0.10)))

            # Opex and net income
            variable_cost = 0.20 * revenue  # processing and network costs
            scale_cost = 0.00004 * txn_volume  # scale-linked infra costs
            opex = opex_baseline * (1.0 + 0.07 * math.sin(t / 6.0)) + variable_cost + scale_cost
            net_income = revenue - opex
            burn_rate = float(-min(0.0, net_income))

            # CAC and churn observed
            cac = clamp(py_rng.uniform(cac_lo, cac_hi) * (1.0 + float(rng.normal(0.0, 0.10))), 0.5, 40.0)
            churn_rate = clamp(py_rng.uniform(churn_lo, churn_hi) * (1.0 + float(rng.normal(0.0, 0.15))), 0.002, 0.25)

            # Funding event hazard increases with growth and revenue, decreases with sanctions and high churn
            funding_round_amount = 0.0
            growth_recent = 0.0
            if t > 2:
                prev_users = users_history[-1] if users_history else active_users
                growth_recent = (active_users - prev_users) / max(1.0, prev_users)
            funding_hazard = clamp(
                0.04 + 0.25 * clamp(growth_recent, -0.2, 0.5) + 0.15 * math.tanh(revenue / 300_000.0) - 0.10 * churn_rate - 0.08 * sanction_severity,
                0.0,
                0.60,
            )
            if py_rng.random() < funding_hazard:
                # advance stage stochastically
                advance = 1 if py_rng.random() < 0.75 else 0
                stage_index = min(stage_index + advance, len(FUNDING_STAGES_ORDERED) - 1)
                new_stage = FUNDING_STAGES_ORDERED[stage_index]
                if new_stage in FUNDING_DISTRIBUTIONS:
                    lo, hi = FUNDING_DISTRIBUTIONS[new_stage]
                    funding_round_amount = float(py_rng.uniform(lo, hi) * (0.8 + 0.4 * py_rng.random()))
                    cumulative_funding += funding_round_amount
                    cash_balance += funding_round_amount * (0.80 + 0.15 * py_rng.random())

            # Cash dynamics
            cash_balance += max(0.0, net_income) - burn_rate
            # Keep non-negative
            cash_balance = max(0.0, cash_balance)
            runway_months = cash_balance / max(1.0, burn_rate)

            # Regulatory sanction hazard: higher in stricter regimes and lending/mobile segments
            base_sanction_hazard = 0.01 + 0.03 * firm.regulatory_strictness
            seg_risk = 0.02 if firm.segment in ("digital_lending", "mobile_money") else 0.0
            sanction_hazard = clamp(base_sanction_hazard + seg_risk - 0.01 * math.tanh(cac / 20.0), 0.002, 0.12)
            sanction_flag = 0
            if py_rng.random() < sanction_hazard:
                sanction_flag = 1
                sanction_severity = clamp(py_rng.uniform(0.25, 1.0), 0.25, 1.0)
                sanction_cooldown = int(rng.integers(2, 7))  # months of drag

            # Downturn detection: sustained drop vs 3-month SMA
            revenue_history.append(revenue)
            users_history.append(active_users)
            downturn_flag = 0
            if len(revenue_history) >= 6:
                sma3 = np.mean(revenue_history[-3:])
                sma3_prev = np.mean(revenue_history[-6:-3])
                if sma3_prev > 0 and (sma3 - sma3_prev) / sma3_prev <= -0.20:
                    # verify users also dropped
                    u_sma3 = np.mean(users_history[-3:])
                    u_sma3_prev = np.mean(users_history[-6:-3])
                    if u_sma3_prev > 0 and (u_sma3 - u_sma3_prev) / u_sma3_prev <= -0.10:
                        downturn_flag = 1

            # Distress hazard: function of runway, downturn, sanctions, churn, negative income
            distress_base = 0.002
            distress_hazard = distress_base
            if runway_months < 3:
                distress_hazard += 0.08 * (3 - runway_months) / 3.0
            if downturn_flag:
                distress_hazard += 0.06
            distress_hazard += 0.05 * sanction_severity
            distress_hazard += 0.06 * clamp(churn_rate - 0.06, 0.0, 0.20)
            if net_income < 0:
                distress_hazard += 0.03
            distress_hazard = clamp(distress_hazard, 0.001, 0.40)

            distress_event = 0
            distress_type_local = "none"
            if py_rng.random() < distress_hazard:
                distress_event = 1
                # Categorize type
                r = py_rng.random()
                if runway_months < 2 and net_income < 0 and (downturn_flag or sanction_severity > 0.5):
                    distress_type_local = "default_closure"
                elif sanction_flag and sanction_severity >= 0.6:
                    distress_type_local = "regulatory_sanction"
                elif downturn_flag:
                    distress_type_local = "sustained_downturn"
                else:
                    distress_type_local = "acquisition_duress"

                # If default/closure, deactivate from next month
                if distress_type_local == "default_closure":
                    is_active = False
                    had_distress = True
                    distress_type = distress_type_local
                    distress_happened_on = month

            rows.append(
                {
                    "firm_id": firm.firm_id,
                    "date": month.date(),
                    "country": firm.country_iso,
                    "segment": firm.segment,
                    "is_active": 1 if is_active else 0,
                    "revenue_usd": float(max(0.0, revenue)),
                    "net_income_usd": float(net_income),
                    "burn_rate_usd": float(burn_rate),
                    "active_users": int(active_users),
                    "transaction_volume_usd": float(max(0.0, txn_volume)),
                    "transaction_count": int(max(0, txn_count)),
                    "agents_count": int(agents_count) if firm.segment == "mobile_money" else np.nan,
                    "cac_usd": float(cac),
                    "churn_rate": float(churn_rate),
                    "funding_round_amount_usd": float(funding_round_amount),
                    "funding_stage": FUNDING_STAGES_ORDERED[min(stage_index, len(FUNDING_STAGES_ORDERED)-1)],
                    "cumulative_funding_usd": float(cumulative_funding),
                    "regulatory_sanction_flag": int(sanction_flag),
                    "sanction_severity": float(sanction_severity),
                    "downturn_flag": int(downturn_flag),
                    "distress_event": int(distress_event),
                    "distress_type": distress_type_local,
                }
            )

            # If distress but not default/closure, keep active but add drag
            if distress_event and is_active:
                # temporary drag
                sanction_severity = max(sanction_severity, 0.3)
                sanction_cooldown = max(sanction_cooldown, int(rng.integers(1, 4)))

        # Firm-level row
        firm_rows.append(
            {
                "firm_id": firm.firm_id,
                "country": firm.country_iso,
                "country_name": firm.country_name,
                "segment": firm.segment,
                "founding_year": firm.founding_year,
                "initial_funding_stage": firm.funding_stage,
                "regulatory_strictness": firm.regulatory_strictness,
                "mobile_penetration": firm.mobile_penetration,
                "base_scale": firm.base_scale,
            }
        )

    panel_df = pd.DataFrame(rows)

    # Add lead labels for early warning: t+3 and t+6 months
    panel_df = panel_df.sort_values(["firm_id", "date"]).reset_index(drop=True)
    panel_df["distress_event_t_plus_3m"] = (
        panel_df.groupby("firm_id")["distress_event"].shift(-3).fillna(0).astype(int)
    )
    panel_df["distress_event_t_plus_6m"] = (
        panel_df.groupby("firm_id")["distress_event"].shift(-6).fillna(0).astype(int)
    )

    firms_df = pd.DataFrame(firm_rows)

    return panel_df, firms_df


def build_data_dictionary() -> pd.DataFrame:
    cols = [
        ("firm_id", "string", "Entity ID (anonymized)", "key"),
        ("date", "date", "Month (period end)", "key"),
        ("country", "string", "ISO-2 country code", "feature"),
        ("segment", "string", "FinTech segment", "feature"),
        ("is_active", "int", "1 if firm operating, 0 after closure", "feature"),
        ("revenue_usd", "float", "Monthly revenue in USD", "feature"),
        ("net_income_usd", "float", "Monthly net income in USD", "feature"),
        ("burn_rate_usd", "float", "Monthly cash burn if loss (USD)", "feature"),
        ("active_users", "int", "Estimated active users/customers", "feature"),
        ("transaction_volume_usd", "float", "Monthly processed volume (USD)", "feature"),
        ("transaction_count", "int", "Monthly processed transactions count", "feature"),
        ("agents_count", "int", "Active agents (mobile money only)", "feature"),
        ("cac_usd", "float", "Customer acquisition cost (USD)", "feature"),
        ("churn_rate", "float", "Monthly churn rate (0-1)", "feature"),
        ("funding_round_amount_usd", "float", "Funding amount if round this month (USD)", "feature"),
        ("funding_stage", "string", "Current funding stage label", "feature"),
        ("cumulative_funding_usd", "float", "Cumulative funding to date (USD)", "feature"),
        ("regulatory_sanction_flag", "int", "1 if sanctioned this month", "feature"),
        ("sanction_severity", "float", "Sanction severity (0-1)", "feature"),
        ("downturn_flag", "int", "Sustained revenue/users downturn started", "feature"),
        ("distress_event", "int", "Any distress event this month (0/1)", "label"),
        ("distress_type", "string", "none | default_closure | regulatory_sanction | sustained_downturn | acquisition_duress", "label"),
        ("distress_event_t_plus_3m", "int", "Lead label: distress within 3 months", "label"),
        ("distress_event_t_plus_6m", "int", "Lead label: distress within 6 months", "label"),
    ]
    return pd.DataFrame(cols, columns=["column", "type", "description", "role"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic SSA FinTech Category 1 dataset")
    p.add_argument("--num-firms", type=int, default=120, help="Number of synthetic firms")
    p.add_argument("--start", type=str, default="2018-01", help="Start month YYYY-MM")
    p.add_argument("--end", type=str, default="2025-06", help="End month YYYY-MM")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--output-dir", type=str, default="data/fintech_micro", help="Output directory")
    p.add_argument("--write-parquet", action="store_true", help="Also write Parquet if pyarrow installed")
    return p.parse_args()


def main():
    args = parse_args()

    rng = random.Random(args.seed)
    firms = generate_firms(args.num_firms, rng)
    months = month_range(args.start, args.end)

    panel_df, firms_df = simulate_panel(firms, months, seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    panel_path = os.path.join(args.output_dir, "fintech_micro_monthly.csv")
    firms_path = os.path.join(args.output_dir, "fintech_micro_firms.csv")
    dict_path = os.path.join(args.output_dir, "fintech_micro_dictionary.csv")

    panel_df.to_csv(panel_path, index=False)
    firms_df.to_csv(firms_path, index=False)
    build_data_dictionary().to_csv(dict_path, index=False)

    if args.write_parquet and HAS_PARQUET:
        panel_parquet = os.path.join(args.output_dir, "fintech_micro_monthly.parquet")
        firms_parquet = os.path.join(args.output_dir, "fintech_micro_firms.parquet")
        panel_df.to_parquet(panel_parquet, index=False)
        firms_df.to_parquet(firms_parquet, index=False)

    # Simple console summary
    by_type = panel_df[panel_df["distress_event"] == 1]["distress_type"].value_counts().to_dict()
    print({"rows": len(panel_df), "firms": len(firms_df), "distress_events": by_type})


if __name__ == "__main__":
    main()
