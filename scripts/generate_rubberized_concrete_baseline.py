#!/usr/bin/env python3
"""
Generate a baseline dataset for:
Pillar 1: Material Characterization & Mixture Design (The "Before" State)
Topic: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

Outputs multiple CSV files under data/baseline/ with:
- mix_design.csv
- constituents.csv
- rubber_characterization.csv
- rubber_psd.csv
- rubber_tga.csv
- rubber_ftir.csv
- fresh_state.csv
- compressive_strength.csv
- split_tensile_strength.csv
- static_modulus.csv
- density.csv
- mip_psd.csv
- upv.csv
- metadata.json

All values are synthetic but domain-informed and internally consistent.
No external dependencies required (uses only Python stdlib).
"""
import csv
import json
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

BASE_DIR = "/workspace/data/baseline"
SEED = 20251017
random.seed(SEED)

# ---------- Utilities ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, headers: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def gaussian(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)


def logistic_cdf(x: float, x0: float, k: float) -> float:
    # 1 / (1 + exp(-(x-x0)/k))
    return 1.0 / (1.0 + math.exp(-(x - x0) / k))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------- Mixture Design ----------

def compute_mix_proportions_by_volume_replacement(
    fine_agg_mass_ctrl: float,
    fine_agg_sg: float,
    rubber_sg: float,
    replacement_pct: float,
) -> Tuple[float, float]:
    """
    Compute (fine_agg_mass, rubber_mass) for a given replacement percentage by VOLUME
    of fine aggregate. Assumes control fine aggregate mass corresponds to 1 m^3 batch.
    """
    # Convert masses to volumes using density = sg * 1000 kg/m3
    density_fine = fine_agg_sg * 1000.0
    density_rubber = rubber_sg * 1000.0

    vol_fine_ctrl = fine_agg_mass_ctrl / density_fine
    vol_replace = replacement_pct * vol_fine_ctrl

    # New fine aggregate volume and rubber volume
    vol_fine_new = vol_fine_ctrl - vol_replace
    vol_rubber = vol_replace

    fine_mass_new = vol_fine_new * density_fine
    rubber_mass = vol_rubber * density_rubber
    return fine_mass_new, rubber_mass


def generate_mix_design() -> List[Dict[str, object]]:
    # Control baseline (HPC-like)
    cement_type = "CEM I 52.5R"
    cement_sg = 3.15
    coarse_agg_type = "Crushed granite"
    coarse_agg_sg = 2.70
    coarse_agg_max_size_mm = 19
    fine_agg_type = "Natural river sand"
    fine_agg_sg = 2.65
    water_type = "Potable"
    sp_type = "PCE-based superplasticizer"

    # Baseline control mix per m3 (typical HPC-ish)
    cement_kg = 450.0
    water_kg = 150.0  # w/b ≈ 0.33
    sp_bwoc = 0.010  # 1% by weight of cement
    sp_kg = cement_kg * sp_bwoc
    coarse_kg = 1050.0
    fine_kg_ctrl = 700.0

    # Rubber properties (for replacement calc)
    rubber_sg = 1.10

    mixes = []
    # Define replacement by volume of fine aggregate
    replacements = [
        ("CTRL_0", 0.00),
        ("RUB_5", 0.05),
        ("RUB_10", 0.10),
        ("RUB_15", 0.15),
    ]

    for mix_id, repl in replacements:
        if repl == 0.0:
            fine_kg = fine_kg_ctrl
            rubber_kg = 0.0
        else:
            fine_kg, rubber_kg = compute_mix_proportions_by_volume_replacement(
                fine_agg_mass_ctrl=fine_kg_ctrl,
                fine_agg_sg=fine_agg_sg,
                rubber_sg=rubber_sg,
                replacement_pct=repl,
            )

        w_b = water_kg / cement_kg
        mixes.append({
            "mix_id": mix_id,
            "rubber_replacement_type": "fine_aggregate_volume",
            "rubber_replacement_pct": round(repl * 100.0, 2),
            "cement_type": cement_type,
            "cement_source": "Local supplier, EN 197-1 compliant",
            "cement_specific_gravity": cement_sg,
            "coarse_aggregate_type": coarse_agg_type,
            "coarse_aggregate_ssd_sg": coarse_agg_sg,
            "coarse_aggregate_max_size_mm": coarse_agg_max_size_mm,
            "fine_aggregate_type": fine_agg_type,
            "fine_aggregate_ssd_sg": fine_agg_sg,
            "fine_aggregate_fineness_modulus": 2.7,
            "water_type": water_type,
            "superplasticizer_type": sp_type,
            "superplasticizer_dosage_bwoc_pct": 100.0 * sp_bwoc,
            "curing_regime": "28 days in lime-saturated water at 23°C",
            "curing_days": 28,
            "water_binder_ratio": round(w_b, 3),
            "cement_kg_per_m3": round(cement_kg, 2),
            "water_kg_per_m3": round(water_kg, 2),
            "superplasticizer_kg_per_m3": round(sp_kg, 2),
            "coarse_aggregate_kg_per_m3": round(coarse_kg, 1),
            "fine_aggregate_kg_per_m3": round(fine_kg, 1),
            "rubber_kg_per_m3": round(rubber_kg, 3),
            "notes": "Rubber replaces fine aggregate by volume. Masses reflect SG differences.",
        })
    return mixes


def generate_constituents() -> List[Dict[str, object]]:
    rows = [
        {
            "material": "Cement",
            "type": "CEM I 52.5R",
            "source": "Local supplier, EN 197-1 compliant",
            "standard": "EN 197-1",
            "specific_gravity": 3.15,
            "notes": "Ordinary Portland cement, high early strength",
        },
        {
            "material": "Coarse aggregate",
            "type": "Crushed granite",
            "source": "Regional quarry",
            "standard": "EN 12620",
            "specific_gravity": 2.70,
            "notes": "Max size 19 mm, angular",
        },
        {
            "material": "Fine aggregate",
            "type": "Natural river sand",
            "source": "Local river source",
            "standard": "EN 12620",
            "specific_gravity": 2.65,
            "notes": "Zone II grading, FM≈2.7",
        },
        {
            "material": "Water",
            "type": "Potable",
            "source": "City supply",
            "standard": "EN 1008",
            "specific_gravity": 1.00,
            "notes": "Meets potable water requirements",
        },
        {
            "material": "Superplasticizer",
            "type": "PCE-based",
            "source": "Commercial admixture",
            "standard": "EN 934-2",
            "specific_gravity": 1.08,
            "notes": "High-range water reducer",
        },
        {
            "material": "Rubber aggregate",
            "type": "Crumb rubber (truck tires)",
            "source": "Post-consumer",
            "standard": "--",
            "specific_gravity": 1.10,
            "notes": "Particle size 1–4 mm",
        },
    ]
    return rows

# ---------- Rubber Characterization ----------

def generate_rubber_physical_characterization() -> Dict[str, object]:
    specific_gravity = round(random.uniform(1.07, 1.12), 3)
    water_absorption_pct = round(random.uniform(1.0, 2.5), 2)
    shore_a = int(round(random.uniform(60, 70)))
    return {
        "rubber_id": "CR_1_4mm",
        "source": "Post-consumer truck tires",
        "type": "Crumb rubber",
        "psd_range_mm": "1–4",
        "specific_gravity": specific_gravity,
        "water_absorption_pct": water_absorption_pct,
        "hardness_shore_a": shore_a,
        "chemical_characterization": "TGA & FTIR provided",
        "pre_treatment": "None",
        "pre_treatment_notes": "Baseline (no pre-treatment).",
    }


def generate_rubber_psd(num_samples: int = 1000) -> List[Dict[str, object]]:
    """
    Generate particle size distribution within 1–4 mm using a truncated lognormal.
    Output is binned into 7 bins for readability.
    """
    # Lognormal parameters (median ≈ 2.0 mm)
    mu = math.log(2.0)
    sigma = 0.30
    # Sample and truncate to [1, 4]
    sizes = []
    for _ in range(num_samples * 2):  # oversample to allow truncation
        x = random.lognormvariate(mu, sigma)
        if 1.0 <= x <= 4.0:
            sizes.append(x)
        if len(sizes) >= num_samples:
            break
    # Define bins
    bin_edges = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    counts = [0] * (len(bin_edges) - 1)
    for x in sizes:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                counts[i] += 1
                break
    total = sum(counts)
    rows = []
    for i in range(len(counts)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        pct = 100.0 * counts[i] / total if total > 0 else 0.0
        rows.append({
            "rubber_id": "CR_1_4mm",
            "bin_lower_mm": lo,
            "bin_upper_mm": hi,
            "percent_by_mass": round(pct, 2),
        })
    return rows


def generate_rubber_tga() -> List[Dict[str, object]]:
    """
    Synthetic TGA curve for crumb rubber: mass% vs temperature.
    Two-step decomposition: main losses around ~350°C and ~480°C; residual char ~8-12%.
    """
    rows = []
    # Temperature from 25 to 800 C, step 5 C
    temps = list(range(25, 801, 5))
    loss1 = random.uniform(48, 55)  # %
    loss2 = random.uniform(33, 40)  # %
    residual = 100.0 - (loss1 + loss2)
    x01 = random.uniform(330, 370)
    k1 = random.uniform(12, 18)
    x02 = random.uniform(460, 500)
    k2 = random.uniform(14, 22)

    prev_mass = None
    prev_temp = None
    for T in temps:
        mass = 100.0 - (loss1 * logistic_cdf(T, x01, k1) + loss2 * logistic_cdf(T, x02, k2))
        # Add small noise and clamp
        mass += random.uniform(-0.15, 0.15)
        mass = clamp(mass, residual - 0.5, 100.1)
        if prev_mass is None:
            dtg = 0.0
        else:
            dtg = (mass - prev_mass) / (T - prev_temp)
        rows.append({
            "rubber_id": "CR_1_4mm",
            "temperature_C": T,
            "mass_percent": round(mass, 3),
            "dtg_percent_per_C": round(dtg, 4),
        })
        prev_mass = mass
        prev_temp = T
    return rows


def generate_rubber_ftir() -> List[Dict[str, object]]:
    """
    Synthetic FTIR spectrum: wavenumber (cm^-1) vs absorbance (AU).
    Peaks typical of rubber:
      2918, 2850 (C–H), 1664 (C=C), 1450, 1375 (CH2/CH3), 970 (isoprene), 835 (out-of-plane).
    """
    rows = []
    # Wavenumber from 4000 down to 600 cm^-1, step -4
    wns = list(range(4000, 599, -4))
    # Gaussian peaks: (center, amplitude, sigma)
    peaks = [
        (2918, 0.9, 28),
        (2850, 0.7, 24),
        (1664, 0.6, 20),
        (1450, 0.8, 22),
        (1375, 0.6, 22),
        (970, 0.7, 18),
        (835, 0.5, 16),
    ]
    baseline = 0.05
    for wn in wns:
        absorb = baseline + random.uniform(-0.01, 0.01)
        for (c, a, s) in peaks:
            absorb += a * gaussian(wn, c, s)
        rows.append({
            "rubber_id": "CR_1_4mm",
            "wavenumber_cm_1": wn,
            "absorbance_AU": round(absorb, 4),
        })
    return rows

# ---------- Fresh State Properties ----------

def generate_fresh_state(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        # Baselines
        target_slump_flow = 620.0  # mm for control HPC
        base_air = 1.8  # % for control
        base_density = 2400.0  # kg/m3 approximate fresh density
        # Rubber effects: slump slightly decreases with replacement, air increases, density decreases
        slump_effect = -120.0 * repl_pct  # up to -18 mm per 1% replacement (approx total -18*?); tuned below
        air_effect = 6.0 * repl_pct  # up to +0.9% at 15%
        density_effect = -220.0 * repl_pct  # up to -33 kg/m3 at 15%
        mean_slump = target_slump_flow + slump_effect
        mean_air = base_air + air_effect
        mean_density = base_density + density_effect
        for rep in range(1, 4):
            slump = random.gauss(mean_slump, 15.0)
            air = clamp(random.gauss(mean_air, 0.3), 0.5, 8.0)
            density = random.gauss(mean_density, 12.0)
            rows.append({
                "mix_id": mix_id,
                "replicate": rep,
                "slump_flow_mm": round(slump, 1),
                "air_content_pct": round(air, 2),
                "fresh_density_kg_m3": round(density, 1),
                "ambient_temperature_C": 23.0,
            })
    return rows

# ---------- Mechanical & Physical Properties ----------

def generate_compressive_strength(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        # Control baseline strengths (cylinders or cubes normalized to MPa)
        fc28_ctrl = 62.0
        fc7_ctrl = 0.72 * fc28_ctrl
        # Rubber effect: strength reduction roughly linear with replacement
        reduction_factor_28 = 1.0 - 0.9 * repl_pct  # ~13.5% reduction at 15%
        reduction_factor_7 = 1.0 - 0.8 * repl_pct
        mean28 = fc28_ctrl * reduction_factor_28
        mean7 = fc7_ctrl * reduction_factor_7
        for age in (7, 28):
            mean = mean7 if age == 7 else mean28
            sd = 2.2 if age == 28 else 2.5
            for i in range(1, 4):
                value = random.gauss(mean, sd)
                rows.append({
                    "mix_id": mix_id,
                    "age_days": age,
                    "specimen_id": f"{mix_id}_fc_{age}d_{i}",
                    "compressive_strength_MPa": round(value, 2),
                })
    return rows


def generate_split_tensile(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        ft_ctrl = 4.3  # MPa at 28d
        reduction = 1.0 - 0.6 * repl_pct  # gentler reduction
        mean = ft_ctrl * reduction
        for i in range(1, 4):
            value = random.gauss(mean, 0.25)
            rows.append({
                "mix_id": mix_id,
                "age_days": 28,
                "specimen_id": f"{mix_id}_ft_28d_{i}",
                "split_tensile_MPa": round(value, 2),
            })
    return rows


def generate_static_modulus(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        E_ctrl = 40.0  # GPa
        reduction = 1.0 - 1.1 * repl_pct  # modulus more sensitive to rubber
        mean = E_ctrl * reduction
        for i in range(1, 4):
            value = random.gauss(mean, 1.2)
            rows.append({
                "mix_id": mix_id,
                "age_days": 28,
                "specimen_id": f"{mix_id}_E_28d_{i}",
                "static_modulus_GPa": round(value, 2),
            })
    return rows


def generate_density(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        density_ssd_ctrl = 2380.0  # kg/m3 at 28d
        density_od_ctrl = 2320.0
        # Rubber reduces density
        ssdeff = -160.0 * repl_pct
        odeff = -150.0 * repl_pct
        mean_ssd = density_ssd_ctrl + ssdeff
        mean_od = density_od_ctrl + odeff
        for age in (7, 28):
            for state, mean, sd in (("ssd", mean_ssd - 20.0*(28-age)/21.0, 7.5), ("oven_dry", mean_od - 20.0*(28-age)/21.0, 7.5)):
                for i in range(1, 4):
                    value = random.gauss(mean, sd)
                    rows.append({
                        "mix_id": mix_id,
                        "age_days": age,
                        "state": state,
                        "specimen_id": f"{mix_id}_dens_{state}_{age}d_{i}",
                        "density_kg_m3": round(value, 1),
                    })
    return rows


def generate_mip_psd(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Generate MIP pore size distribution:
    - pore_diameter_um: 0.01 to 10 um (log-spaced)
    - dV/dlogD mm^3/g shaped like a lognormal; integrate to cumulative and porosity
    Rubber tends to increase total porosity and shift pores to slightly larger sizes.
    """
    rows = []
    # Log-spaced diameters
    num_points = 120
    d_min = math.log10(0.01)
    d_max = math.log10(10.0)
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        # Control distribution parameters
        mu = math.log(0.12)  # in um, for lognormal in natural log domain
        sigma = 0.55
        # Rubber effect: shift mean to larger size, increase amplitude
        mu = math.log(math.exp(mu) * (1.0 + 0.25 * repl_pct))
        amplitude = 1.0 + 0.6 * repl_pct
        # Generate points
        diam_um = [10 ** (d_min + i * (d_max - d_min) / (num_points - 1)) for i in range(num_points)]
        # dV/dlogD synthetic
        dv_dlogd = []
        for d in diam_um:
            val = amplitude * gaussian(math.log(d), mu, sigma)
            # Small noise
            val *= (1.0 + random.uniform(-0.05, 0.05))
            dv_dlogd.append(max(val, 0.0))
        # Normalize to a plausible total porosity range (mm^3/g proxy)
        area = sum(dv_dlogd) * (math.log(diam_um[-1]) - math.log(diam_um[0])) / (num_points - 1)
        target_total_porosity_pct = 11.0 + 20.0 * repl_pct + random.uniform(-0.6, 0.6)  # %
        scale = target_total_porosity_pct / (area * 100.0) if area > 0 else 0.0
        dv_dlogd = [v * scale for v in dv_dlogd]
        # Cumulative
        cumulative = []
        c = 0.0
        for v in dv_dlogd:
            c += v
            cumulative.append(c)
        for d, v, cum in zip(diam_um, dv_dlogd, cumulative):
            rows.append({
                "mix_id": mix_id,
                "pore_diameter_um": round(d, 5),
                "dV_dlogD_mm3_per_g": round(v, 6),
                "cumulative_intrusion_mm3_per_g": round(cum, 6),
                "total_porosity_pct_estimate": round(target_total_porosity_pct, 2),
            })
    return rows


def generate_upv(mixes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for mix in mixes:
        mix_id = mix["mix_id"]
        repl_pct = mix["rubber_replacement_pct"] / 100.0
        upv_ctrl = 4.65  # km/s at 28d
        reduction = 1.0 - 0.7 * repl_pct
        mean = upv_ctrl * reduction
        for i in range(1, 4):
            value = random.gauss(mean, 0.06)
            rows.append({
                "mix_id": mix_id,
                "age_days": 28,
                "specimen_id": f"{mix_id}_UPV_28d_{i}",
                "upv_km_per_s": round(value, 3),
            })
    return rows

# ---------- Main ----------

def main() -> None:
    ensure_dir(BASE_DIR)

    mixes = generate_mix_design()
    constituents = generate_constituents()
    rubber_phys = generate_rubber_physical_characterization()
    rubber_psd = generate_rubber_psd()
    rubber_tga = generate_rubber_tga()
    rubber_ftir = generate_rubber_ftir()

    fresh = generate_fresh_state(mixes)
    comp = generate_compressive_strength(mixes)
    split = generate_split_tensile(mixes)
    modulus = generate_static_modulus(mixes)
    density = generate_density(mixes)
    mip = generate_mip_psd(mixes)
    upv = generate_upv(mixes)

    # Write CSVs
    write_csv(
        os.path.join(BASE_DIR, "mix_design.csv"),
        headers=list(mixes[0].keys()),
        rows=mixes,
    )

    write_csv(
        os.path.join(BASE_DIR, "constituents.csv"),
        headers=list(constituents[0].keys()),
        rows=constituents,
    )

    write_csv(
        os.path.join(BASE_DIR, "rubber_characterization.csv"),
        headers=list(rubber_phys.keys()),
        rows=[rubber_phys],
    )

    if rubber_psd:
        write_csv(
            os.path.join(BASE_DIR, "rubber_psd.csv"),
            headers=list(rubber_psd[0].keys()),
            rows=rubber_psd,
        )

    if rubber_tga:
        write_csv(
            os.path.join(BASE_DIR, "rubber_tga.csv"),
            headers=list(rubber_tga[0].keys()),
            rows=rubber_tga,
        )

    if rubber_ftir:
        write_csv(
            os.path.join(BASE_DIR, "rubber_ftir.csv"),
            headers=list(rubber_ftir[0].keys()),
            rows=rubber_ftir,
        )

    if fresh:
        write_csv(
            os.path.join(BASE_DIR, "fresh_state.csv"),
            headers=list(fresh[0].keys()),
            rows=fresh,
        )

    if comp:
        write_csv(
            os.path.join(BASE_DIR, "compressive_strength.csv"),
            headers=list(comp[0].keys()),
            rows=comp,
        )

    if split:
        write_csv(
            os.path.join(BASE_DIR, "split_tensile_strength.csv"),
            headers=list(split[0].keys()),
            rows=split,
        )

    if modulus:
        write_csv(
            os.path.join(BASE_DIR, "static_modulus.csv"),
            headers=list(modulus[0].keys()),
            rows=modulus,
        )

    if density:
        write_csv(
            os.path.join(BASE_DIR, "density.csv"),
            headers=list(density[0].keys()),
            rows=density,
        )

    if mip:
        write_csv(
            os.path.join(BASE_DIR, "mip_psd.csv"),
            headers=list(mip[0].keys()),
            rows=mip,
        )

    if upv:
        write_csv(
            os.path.join(BASE_DIR, "upv.csv"),
            headers=list(upv[0].keys()),
            rows=upv,
        )

    # Metadata
    metadata = {
        "topic": "Thermo-Mechanical Model - Fire-Resistant HP Rubberized Concrete",
        "pillar": "Pillar 1: Material Characterization & Mixture Design (Before State)",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": SEED,
        "mixes": [m["mix_id"] for m in mixes],
        "notes": "Synthetic dataset. Rubber replaces fine aggregate by volume (0,5,10,15%).",
    }
    with open(os.path.join(BASE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Baseline dataset generated under: {BASE_DIR}")


if __name__ == "__main__":
    main()
