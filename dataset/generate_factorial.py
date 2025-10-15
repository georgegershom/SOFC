from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

from sofc_schema import get_schema
from sampling import RangeSpec, numeric_levels, full_factorial


SYSTEM_KEYS = [
    "system.fuel_utilization",
    "system.oxidant_utilization",
    "system.current_density",
    "system.cell_voltage",
    "system.temperature",
    "system.pressure",
]


def _get_param(schema: Dict[str, Any], key: str) -> Dict[str, Any]:
    for p in schema["parameters"]:
        if p["key"] == key:
            return p
    raise KeyError(key)


def generate_factorial_system(out_csv: str, n_levels: int = 3):
    s = get_schema()
    params = [_get_param(s, k) for k in SYSTEM_KEYS]
    # Build levels per dim
    levels_per_dim: List[List[float]] = []
    for p in params:
        spec = RangeSpec(p["min"], p["max"], p.get("scale", "linear"))
        levels_per_dim.append(numeric_levels(spec, n_levels))

    combos = full_factorial(levels_per_dim)

    headers = [k.replace(".", "_") for k in SYSTEM_KEYS]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers + ["system_fuel_type", "system_steam_to_carbon"])  # fixed categorical
        for c in combos:
            row = list(c)
            # Fix categorical: H2 fuel
            fuel_type = "H2"
            s_c = 0.0
            w.writerow(row + [fuel_type, s_c])


if __name__ == "__main__":
    generate_factorial_system("/workspace/artifacts/inputs_factorial_system_6d.csv", n_levels=3)
