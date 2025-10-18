from typing import Dict, List
import numpy as np


def generate_xrd_phase_table(phases_wt_pct: Dict[str, float], rng: np.random.Generator) -> List[Dict[str, float]]:
    # Return a list of rows with minor noise applied to represent replicate variability
    rows: List[Dict[str, float]] = []
    noise_scale = 0.5
    for phase, wt in phases_wt_pct.items():
        wt_i = max(0.0, wt + rng.normal(0, noise_scale))
        rows.append({"phase": phase, "wt_percent": float(wt_i)})
    # Normalize to 100
    total = sum(r["wt_percent"] for r in rows)
    if total <= 0:
        total = 1.0
    for r in rows:
        r["wt_percent"] = 100.0 * r["wt_percent"] / total
    return rows
