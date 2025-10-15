#!/usr/bin/env python3
"""
Context Dataset Generator (Dataset 2: Process and Material Parameters)

Generates a reproducible CSV of process, geometry, and material parameters
for multilayer ceramic laminates (e.g., anode/electrolyte/cathode stacks),
intended to be concatenated with geometry/warpage features for residual
stress prediction models.

Outputs:
- CSV with one row per sample
- Gzipped CSV (.csv.gz)
- JSON schema describing fields, units, and typical ranges

Usage:
  python scripts/generate_context_dataset.py \
      --n 5000 \
      --seed 42 \
      --outdir data/context

Notes:
- Units are SI where practical. Thicknesses are in micrometers for convenience.
- Temperature is in degrees Celsius.
- Atmosphere oxygen partial pressure (pO2) is in Pascals.
- CTE is provided in 1e-6 per °C (microstrain/°C) to keep magnitudes readable.
- Young's modulus temperature dependence is encoded as fractional slope per °C
  so that E(T) ≈ E0 * (1 + slope_frac_per_C * (T - 25)). For ceramics,
  slope is typically negative.
- Creep parameters use a Norton law form: strain_rate = A * sigma^n * exp(-Q/RT)
  where Q is in kJ/mol and sigma in MPa; A accordingly carries 1/s/MPa^n units.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import random

DATASET_VERSION = "v1"
DATASET_NAME = "context_dataset"

AMBIENT_TEMP_C = 25.0
ATMOSPHERIC_PRESSURE_ATM = 1.0
AIR_PO2_PA = 0.21 * 101_325.0

# ------------------------------- RANGES ------------------------------------
# Geometric ranges
PLATE_LENGTH_MM_RANGE = (20.0, 200.0)
PLATE_WIDTH_MIN_MM = 10.0  # width sampled in [PLATE_WIDTH_MIN_MM, plate_length]
ANODE_THICKNESS_UM_RANGE = (200.0, 1200.0)
ELECTROLYTE_THICKNESS_UM_RANGE = (5.0, 50.0)
CATHODE_THICKNESS_UM_RANGE = (20.0, 150.0)
# Green dimensions: target overall linear shrinkage weighted by layer thicknesses
TOTAL_LINEAR_SHRINK_FRACTION_RANGE = (0.02, 0.20)  # used if layer-weighted gives extreme

# Material properties per layer
# Young's modulus at 25 C and temperature dependence
E0_GPA_RANGE = {
    "anode": (50.0, 200.0),
    "electrolyte": (150.0, 300.0),
    "cathode": (60.0, 180.0),
}
# Fractional slope per C (negative for decreasing E)
E_FRAC_SLOPE_PER_C_RANGE = (-6e-4, -1e-4)
# Coefficient of Thermal Expansion (1e-6 / C)
CTE_UE6_PER_C_RANGE = {
    "anode": (11.0, 14.0),
    "electrolyte": (9.0, 11.5),
    "cathode": (12.0, 15.0),
}
# Poisson's ratio
POISSON_RANGE = (0.20, 0.32)

# Sintering shrinkage parameters per layer
SHRINK_ONSET_C_RANGE = (650.0, 900.0)
SHRINK_RATE_PER_C_RANGE = (1e-4, 5e-3)  # approximate linearized rate param
FINAL_LINEAR_SHRINK_FRACTION_RANGE = {
    "anode": (0.06, 0.22),
    "electrolyte": (0.03, 0.12),
    "cathode": (0.05, 0.18),
}

# Creep (Norton law) parameters per layer
# A in 1/s/MPa^n (log-uniform), n unitless, Q in kJ/mol
CREEP_A_MIN = 1e-25
CREEP_A_MAX = 1e-15
CREEP_N_RANGE = (2.0, 5.0)
CREEP_Q_KJ_PER_MOL_RANGE = (200.0, 500.0)

# Process parameters
RAMP_RATE_C_PER_MIN_RANGE = (1.0, 10.0)
HOLD_TEMP_C_RANGE = (1000.0, 1500.0)
HOLD_TIME_MIN_RANGE = (30.0, 300.0)
COOL_RATE_C_PER_MIN_RANGE = (1.0, 10.0)
ATMOSPHERE_TYPES = ("air", "argon", "nitrogen", "forming_gas")
# pO2 ranges by atmosphere (log-uniform where appropriate)
PO2_PA_RANGE = {
    "air": (AIR_PO2_PA, AIR_PO2_PA),  # fixed near air
    "argon": (1e-1, 1e3),
    "nitrogen": (1e-1, 1e3),
    "forming_gas": (1e-7, 10.0),
}
HUMIDITY_PERCENT_RANGE = (0.0, 50.0)
PRESSURE_ATM_RANGE = (0.9, 1.1)

# ---------------------------- HELPERS --------------------------------------

def loguniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError("loguniform requires positive bounds")
    lo = math.log(low)
    hi = math.log(high)
    return math.exp(rng.uniform(lo, hi))


def sample_atmosphere(rng: random.Random) -> Dict[str, object]:
    atmos = rng.choices(ATMOSPHERE_TYPES, weights=[0.6, 0.15, 0.15, 0.10], k=1)[0]
    if PO2_PA_RANGE[atmos][0] == PO2_PA_RANGE[atmos][1]:
        pO2 = PO2_PA_RANGE[atmos][0]
    else:
        pO2 = loguniform(rng, PO2_PA_RANGE[atmos][0], PO2_PA_RANGE[atmos][1])
    humidity = rng.uniform(*HUMIDITY_PERCENT_RANGE)
    pressure_atm = rng.uniform(*PRESSURE_ATM_RANGE)
    return {
        "atmosphere_type": atmos,
        "atmosphere_pO2_Pa": pO2,
        "atmosphere_humidity_percent": humidity,
        "environment_pressure_atm": pressure_atm,
    }


def bounded(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sample_layer_properties(rng: random.Random, layer: str) -> Dict[str, float]:
    E0 = rng.uniform(*E0_GPA_RANGE[layer])
    slope_frac = rng.uniform(*E_FRAC_SLOPE_PER_C_RANGE)
    E_1000C = E0 * (1.0 + slope_frac * (1000.0 - AMBIENT_TEMP_C))
    E_1000C = max(E_1000C, 0.05 * E0)  # avoid negative/zero

    cte = rng.uniform(*CTE_UE6_PER_C_RANGE[layer])
    nu = rng.uniform(*POISSON_RANGE)

    shrink_onset = rng.uniform(*SHRINK_ONSET_C_RANGE)
    shrink_rate = rng.uniform(*SHRINK_RATE_PER_C_RANGE)
    shrink_final = rng.uniform(*FINAL_LINEAR_SHRINK_FRACTION_RANGE[layer])

    creep_A = loguniform(rng, CREEP_A_MIN, CREEP_A_MAX)
    creep_n = rng.uniform(*CREEP_N_RANGE)
    creep_Q = rng.uniform(*CREEP_Q_KJ_PER_MOL_RANGE)

    return {
        f"{layer}_E0_GPa": E0,
        f"{layer}_E_slope_frac_per_C": slope_frac,
        f"{layer}_E_1000C_GPa": E_1000C,
        f"{layer}_CTE_1e-6_per_C": cte,
        f"{layer}_nu": nu,
        f"{layer}_shrinkage_onset_C": shrink_onset,
        f"{layer}_shrinkage_rate_per_C": shrink_rate,
        f"{layer}_final_linear_shrinkage_fraction": shrink_final,
        f"{layer}_creep_A_1_per_s_MPa_pow_n": creep_A,
        f"{layer}_creep_n": creep_n,
        f"{layer}_creep_Q_kJ_per_mol": creep_Q,
    }


def build_schema() -> Dict[str, object]:
    # Build a machine-readable schema with units and descriptions.
    fields: List[Dict[str, object]] = []

    def add_field(name: str, ftype: str, units: str, desc: str, typical_range: Tuple[float, float] | None = None, allowed: List[str] | None = None, distribution: str | None = None):
        entry: Dict[str, object] = {
            "name": name,
            "type": ftype,
            "units": units,
            "description": desc,
        }
        if typical_range is not None:
            entry["typical_range"] = list(typical_range)
        if allowed is not None:
            entry["allowed_values"] = allowed
        if distribution is not None:
            entry["sampling_distribution"] = distribution
        fields.append(entry)

    # Core identifiers
    add_field("sample_id", "integer", "-", "Sequential sample identifier starting at 0")
    add_field("random_seed", "integer", "-", "Deterministic RNG seed used for this sample")
    add_field("doe_tag", "string", "-", "Dataset/DOE tag for traceability")
    add_field("generated_timestamp_utc", "string", "ISO-8601", "Generation timestamp (UTC)")

    # Geometry
    add_field("plate_length_mm", "number", "mm", "Final (sintered) plate length", PLATE_LENGTH_MM_RANGE, "uniform")
    add_field("plate_width_mm", "number", "mm", "Final (sintered) plate width (≤ length)", (PLATE_WIDTH_MIN_MM, PLATE_LENGTH_MM_RANGE[1]), "uniform_conditional")
    add_field("anode_thickness_um", "number", "µm", "Anode layer thickness", ANODE_THICKNESS_UM_RANGE, "uniform")
    add_field("electrolyte_thickness_um", "number", "µm", "Electrolyte layer thickness", ELECTROLYTE_THICKNESS_UM_RANGE, "uniform")
    add_field("cathode_thickness_um", "number", "µm", "Cathode layer thickness", CATHODE_THICKNESS_UM_RANGE, "uniform")
    add_field("green_length_mm", "number", "mm", "Initial green tape length before sintering")
    add_field("green_width_mm", "number", "mm", "Initial green tape width before sintering")
    add_field("green_density_frac_anode", "number", "-", "Green relative density (anode)", (0.30, 0.60), "uniform")
    add_field("green_density_frac_electrolyte", "number", "-", "Green relative density (electrolyte)", (0.35, 0.70), "uniform")
    add_field("green_density_frac_cathode", "number", "-", "Green relative density (cathode)", (0.30, 0.60), "uniform")

    # Materials per layer
    for layer in ("anode", "electrolyte", "cathode"):
        add_field(f"{layer}_E0_GPa", "number", "GPa", f"Young's modulus at {AMBIENT_TEMP_C:.0f} °C ({layer})", E0_GPA_RANGE[layer], "uniform")
        add_field(f"{layer}_E_slope_frac_per_C", "number", "1/°C", f"Fractional slope of E vs T ({layer}); typically negative", E_FRAC_SLOPE_PER_C_RANGE, "uniform")
        add_field(f"{layer}_E_1000C_GPa", "number", "GPa", f"Young's modulus at 1000 °C ({layer})")
        add_field(f"{layer}_CTE_1e-6_per_C", "number", "1e-6/°C", f"CTE ({layer})", CTE_UE6_PER_C_RANGE[layer], "uniform")
        add_field(f"{layer}_nu", "number", "-", f"Poisson's ratio ({layer})", POISSON_RANGE, "uniform")
        add_field(f"{layer}_shrinkage_onset_C", "number", "°C", f"Sintering shrinkage onset temperature ({layer})", SHRINK_ONSET_C_RANGE, "uniform")
        add_field(f"{layer}_shrinkage_rate_per_C", "number", "1/°C", f"Approx. linearized shrinkage rate parameter ({layer})", SHRINK_RATE_PER_C_RANGE, "uniform")
        add_field(f"{layer}_final_linear_shrinkage_fraction", "number", "-", f"Final linear shrinkage fraction ({layer})", FINAL_LINEAR_SHRINK_FRACTION_RANGE[layer], "uniform")
        add_field(f"{layer}_creep_A_1_per_s_MPa_pow_n", "number", "1/s/MPa^n", f"Norton creep prefactor A ({layer})", (CREEP_A_MIN, CREEP_A_MAX), "loguniform")
        add_field(f"{layer}_creep_n", "number", "-", f"Norton creep stress exponent n ({layer})", CREEP_N_RANGE, "uniform")
        add_field(f"{layer}_creep_Q_kJ_per_mol", "number", "kJ/mol", f"Norton creep activation energy Q ({layer})", CREEP_Q_KJ_PER_MOL_RANGE, "uniform")

    # Derived mismatches
    add_field("cte_mismatch_anode_electrolyte_1e-6_per_C", "number", "1e-6/°C", "|CTE_anode - CTE_electrolyte|")
    add_field("cte_mismatch_cathode_electrolyte_1e-6_per_C", "number", "1e-6/°C", "|CTE_cathode - CTE_electrolyte|")
    add_field("cte_mismatch_anode_cathode_1e-6_per_C", "number", "1e-6/°C", "|CTE_anode - CTE_cathode|")

    # Process params
    add_field("sinter_ramp_rate_C_per_min", "number", "°C/min", "Ramp-up rate to hold temperature", RAMP_RATE_C_PER_MIN_RANGE, "uniform")
    add_field("sinter_hold_temp_C", "number", "°C", "Peak hold temperature", HOLD_TEMP_C_RANGE, "uniform")
    add_field("sinter_hold_time_min", "number", "min", "Hold time at peak temperature", HOLD_TIME_MIN_RANGE, "uniform")
    add_field("cool_rate_C_per_min", "number", "°C/min", "Cooling rate after hold", COOL_RATE_C_PER_MIN_RANGE, "uniform")
    add_field("ramp_time_min", "number", "min", "Time to ramp from ambient to hold temp")
    add_field("cool_time_min", "number", "min", "Time to cool from hold temp to ambient")
    add_field("total_cycle_time_min", "number", "min", "Approximate total cycle time (ramp + hold + cool)")
    add_field("atmosphere_type", "string", "-", "Sintering atmosphere type", allowed=list(ATMOSPHERE_TYPES))
    add_field("atmosphere_pO2_Pa", "number", "Pa", "Oxygen partial pressure in furnace", (min(v[0] for v in PO2_PA_RANGE.values()), max(v[1] for v in PO2_PA_RANGE.values() if math.isfinite(v[1]))), "loguniform")
    add_field("atmosphere_humidity_percent", "number", "%", "Relative humidity during sintering", HUMIDITY_PERCENT_RANGE, "uniform")
    add_field("environment_pressure_atm", "number", "atm", "Ambient pressure in furnace", PRESSURE_ATM_RANGE, "uniform")

    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "description": "Process, geometry, and material context parameters for residual stress prediction",
        "units_convention": "SI (thickness in µm; CTE in 1e-6/°C)",
        "fields": fields,
    }


def sample_record(rng: random.Random, sample_id: int) -> Dict[str, object]:
    # Geometry
    length_mm = rng.uniform(*PLATE_LENGTH_MM_RANGE)
    width_mm = rng.uniform(PLATE_WIDTH_MIN_MM, length_mm)

    anode_t_um = rng.uniform(*ANODE_THICKNESS_UM_RANGE)
    elec_t_um = rng.uniform(*ELECTROLYTE_THICKNESS_UM_RANGE)
    cath_t_um = rng.uniform(*CATHODE_THICKNESS_UM_RANGE)

    # Green densities
    green_rho_an = rng.uniform(0.30, 0.60)
    green_rho_el = rng.uniform(0.35, 0.70)
    green_rho_ca = rng.uniform(0.30, 0.60)

    # Materials
    anode = sample_layer_properties(rng, "anode")
    electrolyte = sample_layer_properties(rng, "electrolyte")
    cathode = sample_layer_properties(rng, "cathode")

    # Derived CTE mismatches
    cte_mis_an_el = abs(anode["anode_CTE_1e-6_per_C"] - electrolyte["electrolyte_CTE_1e-6_per_C"])
    cte_mis_ca_el = abs(cathode["cathode_CTE_1e-6_per_C"] - electrolyte["electrolyte_CTE_1e-6_per_C"])
    cte_mis_an_ca = abs(anode["anode_CTE_1e-6_per_C"] - cathode["cathode_CTE_1e-6_per_C"])

    # Process
    ramp_rate = rng.uniform(*RAMP_RATE_C_PER_MIN_RANGE)
    hold_temp = rng.uniform(*HOLD_TEMP_C_RANGE)
    hold_time = rng.uniform(*HOLD_TIME_MIN_RANGE)
    cool_rate = rng.uniform(*COOL_RATE_C_PER_MIN_RANGE)

    ramp_time = (hold_temp - AMBIENT_TEMP_C) / ramp_rate
    cool_time = (hold_temp - AMBIENT_TEMP_C) / cool_rate
    total_time = ramp_time + hold_time + cool_time

    atmos = sample_atmosphere(rng)

    # Weighted final shrinkage for green -> sintered dimension estimate
    t_weights = [anode_t_um, elec_t_um, cath_t_um]
    shrink_finals = [
        anode["anode_final_linear_shrinkage_fraction"],
        electrolyte["electrolyte_final_linear_shrinkage_fraction"],
        cathode["cathode_final_linear_shrinkage_fraction"],
    ]
    if sum(t_weights) <= 0:
        total_shrink = rng.uniform(*TOTAL_LINEAR_SHRINK_FRACTION_RANGE)
    else:
        total_shrink = sum(ti * si for ti, si in zip(t_weights, shrink_finals)) / sum(t_weights)
        total_shrink = bounded(total_shrink, *TOTAL_LINEAR_SHRINK_FRACTION_RANGE)

    green_length = length_mm * (1.0 + total_shrink)
    green_width = width_mm * (1.0 + total_shrink)

    record: Dict[str, object] = {
        "sample_id": sample_id,
        "random_seed": rng.seed_value if hasattr(rng, "seed_value") else None,
        "doe_tag": f"{DATASET_NAME}_{DATASET_VERSION}",
        "generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),

        # Geometry
        "plate_length_mm": length_mm,
        "plate_width_mm": width_mm,
        "anode_thickness_um": anode_t_um,
        "electrolyte_thickness_um": elec_t_um,
        "cathode_thickness_um": cath_t_um,
        "green_length_mm": green_length,
        "green_width_mm": green_width,
        "green_density_frac_anode": green_rho_an,
        "green_density_frac_electrolyte": green_rho_el,
        "green_density_frac_cathode": green_rho_ca,

        # Derived mismatches
        "cte_mismatch_anode_electrolyte_1e-6_per_C": cte_mis_an_el,
        "cte_mismatch_cathode_electrolyte_1e-6_per_C": cte_mis_ca_el,
        "cte_mismatch_anode_cathode_1e-6_per_C": cte_mis_an_ca,

        # Process
        "sinter_ramp_rate_C_per_min": ramp_rate,
        "sinter_hold_temp_C": hold_temp,
        "sinter_hold_time_min": hold_time,
        "cool_rate_C_per_min": cool_rate,
        "ramp_time_min": ramp_time,
        "cool_time_min": cool_time,
        "total_cycle_time_min": total_time,
        **atmos,
    }

    # Merge per-layer props
    record.update(anode)
    record.update(electrolyte)
    record.update(cathode)

    return record


def infer_field_order(schema: Dict[str, object]) -> List[str]:
    # Use schema field order
    return [f["name"] for f in schema["fields"]]


def write_csv(records: List[Dict[str, object]], path_csv: str, field_order: List[str]) -> None:
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_gzip(src_csv: str, dest_gz: str) -> None:
    with open(src_csv, "rb") as fin, gzip.open(dest_gz, "wb") as fout:
        fout.writelines(fin)


def main():
    parser = argparse.ArgumentParser(description="Generate Context Dataset (process/material parameters)")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    parser.add_argument("--outdir", type=str, default="data/context", help="Output directory")
    parser.add_argument("--prefix", type=str, default=f"{DATASET_NAME}_{DATASET_VERSION}", help="Filename prefix")
    args = parser.parse_args()

    # Initialize RNG; create per-sample deterministic seeds for traceability
    global_rng = random.Random(args.seed)

    schema = build_schema()
    field_order = infer_field_order(schema)

    records: List[Dict[str, object]] = []
    for i in range(args.n):
        seed_i = global_rng.getrandbits(32)
        rng_i = random.Random(seed_i)
        # attach seed for record
        setattr(rng_i, "seed_value", int(seed_i))
        rec = sample_record(rng_i, i)
        records.append(rec)

    out_csv = os.path.join(args.outdir, f"{args.prefix}.csv")
    out_gz = os.path.join(args.outdir, f"{args.prefix}.csv.gz")
    out_schema = os.path.join(args.outdir, f"{args.prefix}_schema.json")

    write_csv(records, out_csv, field_order)
    write_gzip(out_csv, out_gz)

    with open(out_schema, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote CSV.GZ: {out_gz}")
    print(f"Wrote Schema: {out_schema}")


if __name__ == "__main__":
    main()
