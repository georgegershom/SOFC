from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

from sofc_schema import get_schema
from sampling import RangeSpec, latin_hypercube, full_factorial, numeric_levels


def _flatten_key_to_col(key: str) -> str:
    return key.replace(".", "_")


def _collect_parameters(schema: Dict[str, Any], fidelity: str) -> List[Dict[str, Any]]:
    return schema["by_fidelity"][fidelity]


def _split_numeric_categorical(params: List[Dict[str, Any]]):
    numeric, integer, categorical = [], [], []
    for p in params:
        if p["ptype"] == "categorical":
            categorical.append(p)
        elif p["ptype"] == "integer":
            integer.append(p)
        else:
            numeric.append(p)
    return numeric, integer, categorical


def _sample_numeric_block(n_samples: int, numeric_params: List[Dict[str, Any]], seed: int | None = None) -> List[List[float]]:
    ranges = [RangeSpec(p["min"], p["max"], p.get("scale", "linear")) for p in numeric_params]
    samples = latin_hypercube(n_samples, ranges, seed=seed)
    # Round values for integer params after merge step
    return samples


def _expand_categorical(samples: List[List[float]], numeric_params: List[Dict[str, Any]], integer_params: List[Dict[str, Any]], categorical_params: List[Dict[str, Any]], seed: int | None = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    random.seed(seed)
    rows: List[Dict[str, Any]] = []
    headers: List[str] = []
    # Map in order of numeric, integer, then categorical
    ordered_params = numeric_params + integer_params + categorical_params
    headers = [_flatten_key_to_col(p["key"]) for p in ordered_params]

    for s in samples:
        row: Dict[str, Any] = {}
        # numeric
        for idx, p in enumerate(numeric_params):
            row[_flatten_key_to_col(p["key"])] = s[idx]
        # integer: sample uniformly (not stratified to keep LHS independent)
        for p in integer_params:
            lo, hi = int(p["min"]), int(p["max"])  # inclusive bounds assumption
            row[_flatten_key_to_col(p["key"]) ] = random.randint(lo, hi)
        # categorical: choose uniformly
        for p in categorical_params:
            levels = p.get("levels") or []
            if len(levels) == 0:
                levels = ["level0"]
            row[_flatten_key_to_col(p["key"]) ] = random.choice(levels)
        rows.append(row)
    return rows, headers


def _write_csv(path: str, rows: List[Dict[str, Any]], headers: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def generate_fidelity_dataset(out_dir: str, fidelity: str, n_samples: int, seed: int | None = None) -> str:
    schema = get_schema()
    params = _collect_parameters(schema, fidelity)
    numeric_params, integer_params, categorical_params = _split_numeric_categorical(params)
    samples_num = _sample_numeric_block(n_samples, numeric_params, seed=seed)
    rows, headers = _expand_categorical(samples_num, numeric_params, integer_params, categorical_params, seed=seed)

    # Apply basic consistency constraints
    for r in rows:
        # If fuel_type == H2, set steam_to_carbon to 0
        ft = r.get("system_fuel_type")
        if ft == "H2":
            r["system_steam_to_carbon"] = 0.0
        # Ensure current density and voltage are roughly consistent (no hard physics here)
        jd = r.get("system_current_density")
        v = r.get("system_cell_voltage")
        if jd and v:
            # Clip unrealistic combos (very high J with very high V)
            if jd > 1.2 and v > 0.85:
                r["system_cell_voltage"] = max(0.60, v - 0.1)

    fname = os.path.join(out_dir, f"inputs_{fidelity}.csv")
    _write_csv(fname, rows, headers)
    return fname


def write_schema(out_dir: str):
    s = get_schema()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "schema.json"), "w") as f:
        json.dump(s, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SOFC multi-fidelity datasets")
    parser.add_argument("--out", default="/workspace/artifacts", help="Output directory")
    parser.add_argument("--lf", type=int, default=1000, help="LF sample size")
    parser.add_argument("--mf", type=int, default=500, help="MF sample size")
    parser.add_argument("--hf", type=int, default=200, help="HF sample size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out = args.out
    os.makedirs(out, exist_ok=True)
    write_schema(out)

    print("Generating LF/MF/HF input CSVs...")
    lf = generate_fidelity_dataset(out, "LF", args.lf, seed=args.seed)
    mf = generate_fidelity_dataset(out, "MF", args.mf, seed=args.seed)
    hf = generate_fidelity_dataset(out, "HF", args.hf, seed=args.seed)
    print("Wrote:", lf, mf, hf)


if __name__ == "__main__":
    main()
