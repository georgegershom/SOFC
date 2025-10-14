import random
from datetime import datetime, timedelta
from typing import Dict, Iterator, List

from ..utils import to_iso

MEASURES = [
    "insulation_roof",
    "insulation_wall",
    "window_double_glazing",
    "hvac_upgrade",
    "lighting_led",
    "solar_pv",
]


def generate_retrofit_measures(building_rows: List[Dict]) -> Iterator[Dict]:
    for b in building_rows:
        num = random.choice([0, 1, 2])
        retrofit_year = random.randint(2014, 2023)
        base_date = datetime(retrofit_year, random.randint(1, 12), random.randint(1, 28))
        savings_accum = 0.0
        for i in range(num):
            measure = random.choice(MEASURES)
            capex = {
                "insulation_roof": random.uniform(10000, 30000),
                "insulation_wall": random.uniform(20000, 60000),
                "window_double_glazing": random.uniform(15000, 50000),
                "hvac_upgrade": random.uniform(30000, 120000),
                "lighting_led": random.uniform(8000, 25000),
                "solar_pv": random.uniform(40000, 200000),
            }[measure]
            savings = {
                "insulation_roof": 0.06,
                "insulation_wall": 0.12,
                "window_double_glazing": 0.08,
                "hvac_upgrade": 0.15,
                "lighting_led": 0.05,
                "solar_pv": 0.18,
            }[measure] * random.uniform(0.8, 1.2)
            savings_accum = min(0.6, savings_accum + savings)
            yield {
                "building_id": b["building_id"],
                "retrofit_date": to_iso(base_date + timedelta(days=i * 60)),
                "measure": measure,
                "capex_usd": round(capex, 2),
                "estimated_savings_pct": round(savings_accum, 3),
            }
