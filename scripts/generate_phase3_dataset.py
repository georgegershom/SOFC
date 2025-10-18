#!/usr/bin/env python3
import os
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("numpy is required to run this generator. Install with: pip install numpy") from exc

# ------------------------------
# Dataset configuration
# ------------------------------
OUTPUT_DIR = os.path.join("data", "phase3")
VOLUME_DIR = os.path.join(OUTPUT_DIR, "volumes")
RNG = np.random.default_rng(42)
random.seed(42)

TEMPERATURES_C = [20, 200, 400, 600, 800]
REPLICATES = [1, 2, 3]
FOVS_PER_SEM_REP = 6
VOXEL_SIZE_UM = 5.0
VOLUME_SHAPE = (64, 64, 64)  # z, y, x

# Mix designs: rubber replacement levels of fine aggregate by volume
MIX_DESIGNS = [
    {"Mix_ID": "M_R0",  "rubber_vol_frac": 0.00, "w_c": 0.35},
    {"Mix_ID": "M_R10", "rubber_vol_frac": 0.10, "w_c": 0.35},
    {"Mix_ID": "M_R20", "rubber_vol_frac": 0.20, "w_c": 0.35},
    {"Mix_ID": "M_R30", "rubber_vol_frac": 0.30, "w_c": 0.35},
]

# ------------------------------
# Helper math
# ------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def logistic(x: float, x0: float, k: float) -> float:
    """Standard logistic curve in [0,1]."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


# ------------------------------
# Underlying thermo-chemical state model
# ------------------------------
@dataclass
class BaselineState:
    """Room-temperature baseline fractions by mass (not all phases enumerated)."""
    csh_mass_frac: float
    ch_mass_frac: float
    caco3_mass_frac: float
    quartz_mass_frac: float
    rubber_mass_frac: float
    free_water_mass_frac: float
    base_porosity_vol_frac: float


@dataclass
class TemperatureState:
    temperature_c: int
    csh_remaining: float        # fraction of baseline C-S-H remaining
    ch_remaining: float         # fraction of baseline CH remaining
    caco3_remaining: float      # fraction of baseline CaCO3 remaining
    rubber_remaining: float     # fraction of baseline rubber remaining (solid/gel)
    rubber_char_frac_of_rubber: float  # fraction of baseline rubber converted to char
    cao_formed_mass_frac: float # formed from CaCO3 decarbonation
    porosity_vol_frac: float
    crack_density_mm2: float
    itz_damage_index: float     # 0-1


@dataclass
class Sample:
    sample_id: str
    mix_id: str
    temperature_c: int
    replicate_id: int


# ------------------------------
# Physics-inspired evolution rules
# ------------------------------

def compute_baseline_state(rubber_vol_frac: float, w_c: float) -> BaselineState:
    # Approximate mass fractions at 20C
    # Increase porosity and free water slightly with higher w/c and rubber content
    base_porosity = 0.06 + 0.06 * rubber_vol_frac  # rubber introduces voids/weak ITZ
    free_water = 0.02 + 0.01 * (w_c - 0.35) + 0.01 * rubber_vol_frac

    # Assign baseline mass fractions (sum <= 1; unmodeled phases are the rest)
    quartz = 0.28
    csh = 0.34
    ch = 0.14
    caco3 = 0.10
    rubber_mass = 0.00 + 0.20 * rubber_vol_frac  # simple mapping vol->mass proxy

    return BaselineState(
        csh_mass_frac=csh,
        ch_mass_frac=ch,
        caco3_mass_frac=caco3,
        quartz_mass_frac=quartz,
        rubber_mass_frac=rubber_mass,
        free_water_mass_frac=free_water,
        base_porosity_vol_frac=base_porosity,
    )


def compute_temperature_state(baseline: BaselineState, temperature_c: int, rubber_vol_frac: float) -> TemperatureState:
    t = temperature_c

    # Rubber decomposition: strong between 200-400C. Residual char ~5% of baseline rubber.
    if t < 150:
        rubber_remaining = 1.0
        rubber_char_of_rubber = 0.0
    elif t < 300:
        rubber_remaining = 0.7
        rubber_char_of_rubber = 0.02
    elif t < 500:
        rubber_remaining = 0.15
        rubber_char_of_rubber = 0.05
    else:
        rubber_remaining = 0.0
        rubber_char_of_rubber = 0.05

    # CH dehydroxylation: 400-550C
    ch_remaining = 1.0 - logistic(t, x0=480.0, k=0.025)  # falls from ~1 to ~0
    ch_remaining = clamp(ch_remaining, 0.0, 1.0)

    # CaCO3 decarbonation: 650-800C -> CaO formation
    caco3_remaining = 1.0 - logistic(t, x0=740.0, k=0.03)
    caco3_remaining = clamp(caco3_remaining, 0.0, 1.0)

    # CaO formed mass fraction relative to baseline CaCO3 (stoichiometry: CaCO3 -> CaO + CO2)
    # mass(CaO)/mass(CaCO3) = 56/100 = 0.56
    cao_from_caco3 = (1.0 - caco3_remaining) * baseline.caco3_mass_frac * 0.56

    # C-S-H dehydration: progressive 200-800C
    # mild decay starting around 200C, stronger after 600C
    csh_remaining = 1.0 - 0.20 * logistic(t, x0=250.0, k=0.03) - 0.55 * logistic(t, x0=650.0, k=0.025)
    csh_remaining = clamp(csh_remaining, 0.15, 1.0)

    # Porosity increase: baseline + contributions from rubber gas evolution, CH water loss, CaCO3 CO2 release + thermal cracking
    # Rubber gas creation proportional to lost rubber (non-char)
    rubber_gas_frac = baseline.rubber_mass_frac * (1.0 - rubber_remaining - rubber_char_of_rubber)
    ch_water_release = baseline.ch_mass_frac * (1.0 - ch_remaining) * (18.0 / 74.0)  # water mass fraction
    caco3_co2_release = baseline.caco3_mass_frac * (1.0 - caco3_remaining) * (44.0 / 100.0)

    porosity = (
        baseline.base_porosity_vol_frac
        + 0.35 * rubber_gas_frac
        + 0.10 * ch_water_release
        + 0.15 * caco3_co2_release
        + 0.02 * logistic(t, x0=600.0, k=0.02)  # thermal cracking contribution
    )
    porosity = clamp(porosity, 0.05, 0.65)

    # Crack density (planar density per mm^2 in SEM): increases with T and rubber content
    crack_density = (
        0.5
        + 2.5 * logistic(t, x0=500.0, k=0.02)
        + 3.0 * rubber_vol_frac * logistic(t, x0=350.0, k=0.03)
    )

    # ITZ damage index 0-1, stronger with rubber content and temperature
    itz_damage = clamp(0.1 + 0.7 * logistic(t, x0=400.0, k=0.02) + 0.4 * rubber_vol_frac, 0.0, 1.0)

    return TemperatureState(
        temperature_c=t,
        csh_remaining=csh_remaining,
        ch_remaining=ch_remaining,
        caco3_remaining=caco3_remaining,
        rubber_remaining=rubber_remaining,
        rubber_char_frac_of_rubber=rubber_char_of_rubber,
        cao_formed_mass_frac=cao_from_caco3,
        porosity_vol_frac=porosity,
        crack_density_mm2=crack_density,
        itz_damage_index=itz_damage,
    )


# ------------------------------
# Metrics synthesis for each technique
# ------------------------------

def synthesize_tga_dta(baseline: BaselineState, ts: TemperatureState, noise: float = 0.003) -> Dict[str, float]:
    # Stage mass losses based on transitions
    # Stage 30-200C: free water + a bit of bound water from early C-S-H dehydration
    stage1 = baseline.free_water_mass_frac + 0.05 * baseline.csh_mass_frac * (1.0 - ts.csh_remaining)

    # Stage 200-400C: rubber decomposition (to gas, not char)
    rubber_gas_from_baseline = baseline.rubber_mass_frac * (1.0 - ts.rubber_remaining - ts.rubber_char_frac_of_rubber)
    stage2 = max(0.0, rubber_gas_from_baseline)

    # Stage 400-550C: CH dehydroxylation (water loss)
    stage3 = baseline.ch_mass_frac * (1.0 - ts.ch_remaining) * (18.0 / 74.0)

    # Stage 600-800C: CaCO3 decarbonation (CO2 release)
    stage4 = baseline.caco3_mass_frac * (1.0 - ts.caco3_remaining) * (44.0 / 100.0)

    # DTA peak temperatures (characteristic)
    peak1 = 120 + RNG.normal(0, 5)  # endotherm water loss
    peak2 = 350 + 30 * (baseline.rubber_mass_frac > 0) + RNG.normal(0, 8)  # rubber decomposition
    peak3 = 460 + RNG.normal(0, 10)  # CH
    peak4 = 720 + RNG.normal(0, 12)  # CaCO3

    # Add slight measurement noise
    def jitter(x: float) -> float:
        return max(0.0, x + RNG.normal(0, noise * max(1.0, x)))

    stage1 = jitter(stage1)
    stage2 = jitter(stage2)
    stage3 = jitter(stage3)
    stage4 = jitter(stage4)

    total_loss = stage1 + stage2 + stage3 + stage4

    return {
        "mass_loss_pct_30_200": 100.0 * stage1,
        "mass_loss_pct_200_400": 100.0 * stage2,
        "mass_loss_pct_400_550": 100.0 * stage3,
        "mass_loss_pct_600_800": 100.0 * stage4,
        "total_mass_loss_pct": 100.0 * total_loss,
        "DTA_peak_C_1": float(peak1),
        "DTA_peak_C_2": float(peak2),
        "DTA_peak_C_3": float(peak3),
        "DTA_peak_C_4": float(peak4),
    }


def synthesize_xrd(baseline: BaselineState, ts: TemperatureState, noise: float = 0.8) -> Dict[str, float]:
    # Compute mass fractions of key phases at T
    ch_mass = baseline.ch_mass_frac * ts.ch_remaining
    caco3_mass = baseline.caco3_mass_frac * ts.caco3_remaining
    cao_mass = ts.cao_formed_mass_frac
    quartz_mass = baseline.quartz_mass_frac
    char_mass = baseline.rubber_mass_frac * ts.rubber_char_frac_of_rubber

    # C-S-H amorphous fraction proxy: remaining C-S-H
    csh_mass = baseline.csh_mass_frac * ts.csh_remaining

    # Normalize crystalline phase weights to 100% for XRD quant (excluding amorphous C-S-H)
    phases = {
        "CH_wt_pct": ch_mass,
        "CaCO3_wt_pct": caco3_mass,
        "CaO_wt_pct": cao_mass,
        "Quartz_wt_pct": quartz_mass,
        "Char_wt_pct": char_mass,
    }
    total = sum(phases.values())
    if total <= 0:
        total = 1e-6
    for k in list(phases.keys()):
        phases[k] = 100.0 * phases[k] / total
        phases[k] = clamp(phases[k] + RNG.normal(0, noise), 0.0, 100.0)

    # Amorphous hump intensity proxy (a.u.) proportional to C-S-H mass
    amorphous_index = 40.0 * csh_mass + RNG.normal(0, 1.0)

    # Peak intensities (a.u.) for key reflections; scaled with phase amounts
    peak_ch_18 = 2.0 * phases["CH_wt_pct"] + RNG.normal(0, 1.0)
    peak_caco3_29 = 1.5 * phases["CaCO3_wt_pct"] + RNG.normal(0, 1.0)
    peak_quartz_26 = 1.4 * phases["Quartz_wt_pct"] + RNG.normal(0, 1.0)

    return {
        **phases,
        "Amorphous_index": amorphous_index,
        "peak_intensity_CH_18deg": peak_ch_18,
        "peak_intensity_CaCO3_29deg": peak_caco3_29,
        "peak_intensity_Quartz_26deg": peak_quartz_26,
    }


def synthesize_sem_fov_metrics(baseline: BaselineState, ts: TemperatureState, fov_index: int, rubber_vol_frac: float) -> Dict[str, float]:
    # Pore size distribution percentiles (um) increase with porosity and temperature
    base_scale = 5.0 + 30.0 * ts.porosity_vol_frac + 15.0 * rubber_vol_frac
    d50 = base_scale * (1.0 + 0.3 * logistic(ts.temperature_c, 600.0, 0.02))
    d10 = 0.4 * d50 + RNG.normal(0, 0.5)
    d90 = 1.8 * d50 + RNG.normal(0, 1.5)

    crack_density = ts.crack_density_mm2 * (1.0 + RNG.normal(0, 0.05))
    itz_damage = clamp(ts.itz_damage_index + RNG.normal(0, 0.03), 0.0, 1.0)

    interface_porosity_pct = clamp(100.0 * (baseline.base_porosity_vol_frac * (1.0 + 0.8 * rubber_vol_frac) * (1.0 + 0.6 * logistic(ts.temperature_c, 450.0, 0.03))), 0.5, 45.0)

    return {
        "FOV_ID": fov_index,
        "pore_D10_um": max(0.1, d10),
        "pore_D50_um": max(0.2, d50),
        "pore_D90_um": max(0.3, d90),
        "crack_density_per_mm2": max(0.0, crack_density),
        "itz_damage_index_0_1": itz_damage,
        "interface_porosity_pct": interface_porosity_pct,
    }


def _draw_random_spheres(shape: Tuple[int, int, int], centers_radii: List[Tuple[Tuple[int, int, int], int]], label: int, labels_vol: np.ndarray) -> None:
    zdim, ydim, xdim = shape
    for (cz, cy, cx), r in centers_radii:
        z0, z1 = max(0, cz - r), min(zdim, cz + r + 1)
        y0, y1 = max(0, cy - r), min(ydim, cy + r + 1)
        x0, x1 = max(0, cx - r), min(xdim, cx + r + 1)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        labels_vol[z0:z1, y0:y1, x0:x1][mask] = label


def _random_centers_for_fraction(shape: Tuple[int, int, int], target_vol_frac: float, r_range: Tuple[int, int], max_iter: int = 400) -> List[Tuple[Tuple[int, int, int], int]]:
    zdim, ydim, xdim = shape
    target_voxels = int(target_vol_frac * zdim * ydim * xdim)
    placed: List[Tuple[Tuple[int, int, int], int]] = []
    total = 0
    for _ in range(max_iter):
        if total >= target_voxels:
            break
        r = int(RNG.integers(r_range[0], r_range[1] + 1))
        cz = int(RNG.integers(r, zdim - r))
        cy = int(RNG.integers(r, ydim - r))
        cx = int(RNG.integers(r, xdim - r))
        placed.append(((cz, cy, cx), r))
        total += int((4.0 / 3.0) * math.pi * r ** 3)
    return placed


def generate_microstructure_volume(baseline: BaselineState, ts: TemperatureState, rubber_vol_frac: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    # Label meanings: 0 void, 1 matrix, 2 aggregate (quartz), 3 rubber, 4 crack, 5 char
    labels = np.ones(VOLUME_SHAPE, dtype=np.uint8)

    zdim, ydim, xdim = VOLUME_SHAPE

    # Aggregates (constant volume ~ quartz fraction proxy) -> label 2
    agg_vol_frac = 0.22
    agg_centers = _random_centers_for_fraction(VOLUME_SHAPE, agg_vol_frac, r_range=(3, 7))
    _draw_random_spheres(VOLUME_SHAPE, agg_centers, label=2, labels_vol=labels)

    # Rubber inclusions at 20C; reduced with temperature according to ts.rubber_remaining
    rubber_base_vol_frac = 0.08 * rubber_vol_frac  # map vol frac to microscale occupancy
    rubber_current_vol_frac = rubber_base_vol_frac * ts.rubber_remaining
    if rubber_current_vol_frac > 0.0:
        rubber_centers = _random_centers_for_fraction(VOLUME_SHAPE, rubber_current_vol_frac, r_range=(2, 4))
        _draw_random_spheres(VOLUME_SHAPE, rubber_centers, label=3, labels_vol=labels)

    # Pores/voids: baseline + gas evolution + cracking -> label 0
    target_porosity = ts.porosity_vol_frac
    # We already occupied aggregates/rubber, so add pores accordingly
    pores_centers = _random_centers_for_fraction(VOLUME_SHAPE, target_porosity, r_range=(1, 3))
    _draw_random_spheres(VOLUME_SHAPE, pores_centers, label=0, labels_vol=labels)

    # Cracks: generate a few random thin planes -> label 4
    crack_planes = int(1 + 4 * logistic(ts.temperature_c, 600.0, 0.02) + 6 * rubber_vol_frac)
    for _ in range(crack_planes):
        axis = int(RNG.integers(0, 3))
        if axis == 0:  # z-plane
            z = int(RNG.integers(0, zdim))
            thickness = int(1 + RNG.integers(0, 2))
            z0 = clamp(z - thickness // 2, 0, zdim - 1)
            z1 = clamp(z0 + thickness, 0, zdim)
            labels[int(z0):int(z1), :, :] = 4
        elif axis == 1:  # y-plane
            y = int(RNG.integers(0, ydim))
            thickness = int(1 + RNG.integers(0, 2))
            y0 = clamp(y - thickness // 2, 0, ydim - 1)
            y1 = clamp(y0 + thickness, 0, ydim)
            labels[:, int(y0):int(y1), :] = 4
        else:  # x-plane
            x = int(RNG.integers(0, xdim))
            thickness = int(1 + RNG.integers(0, 2))
            x0 = clamp(x - thickness // 2, 0, xdim - 1)
            x1 = clamp(x0 + thickness, 0, xdim)
            labels[:, :, int(x0):int(x1)] = 4

    # Char from rubber decomposition (dense small inclusions) -> label 5
    char_vol_frac = 0.2 * rubber_vol_frac * ts.rubber_char_frac_of_rubber
    if char_vol_frac > 0.0:
        char_centers = _random_centers_for_fraction(VOLUME_SHAPE, char_vol_frac, r_range=(1, 2))
        _draw_random_spheres(VOLUME_SHAPE, char_centers, label=5, labels_vol=labels)

    # Convert labels to grayscale CT-like intensities
    intensity_map = {
        0: (0.02, 0.03),   # void
        1: (0.45, 0.08),   # matrix
        2: (0.80, 0.07),   # aggregate (quartz)
        3: (0.30, 0.05),   # rubber
        4: (0.05, 0.02),   # crack (air)
        5: (0.35, 0.04),   # char
    }
    vol = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    for label_value, (mean_int, std_int) in intensity_map.items():
        mask = labels == label_value
        n = int(mask.sum())
        if n > 0:
            vol[mask] = RNG.normal(mean_int, std_int, size=n)
    vol = np.clip(vol, 0.0, 1.0)

    # Derive Micro-CT metrics from known placements
    porosity_pct = 100.0 * float(np.mean(labels == 0))
    cracks_pct = 100.0 * float(np.mean(labels == 4))
    crack_density_mm3 = 5.0 * cracks_pct  # simple proxy scaling

    # Pore radius proxy using input sphere radii statistics (approximate percentiles)
    # We cannot recover radii from volume; approximate from target porosity and temperature
    pore_d50_um = max(1.0, 20.0 * ts.porosity_vol_frac + 10.0 * logistic(ts.temperature_c, 600.0, 0.02))
    sphericity_mean = clamp(0.75 - 0.20 * logistic(ts.temperature_c, 700.0, 0.03) + 0.05 * (rubber_vol_frac > 0), 0.3, 0.9)
    connectivity_index = clamp(1.2 + 2.0 * logistic(ts.temperature_c, 550.0, 0.02) + 1.5 * rubber_vol_frac, 0.5, 5.0)

    metrics = {
        "voxel_size_um": VOXEL_SIZE_UM,
        "porosity_pct": porosity_pct,
        "crack_density_per_mm3": crack_density_mm3,
        "pore_D50_um": pore_d50_um,
        "sphericity_mean": sphericity_mean,
        "connectivity_index": connectivity_index,
    }

    return vol, labels, metrics


# ------------------------------
# Dataset assembly
# ------------------------------

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VOLUME_DIR, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ensure_dirs()

    samples_meta: List[Dict[str, object]] = []
    sem_rows: List[Dict[str, object]] = []
    xrd_rows: List[Dict[str, object]] = []
    tga_rows: List[Dict[str, object]] = []
    microct_rows: List[Dict[str, object]] = []

    for mix in MIX_DESIGNS:
        mix_id = mix["Mix_ID"]
        rubber_vol_frac = float(mix["rubber_vol_frac"])
        w_c = float(mix["w_c"])
        baseline = compute_baseline_state(rubber_vol_frac, w_c)

        for temp in TEMPERATURES_C:
            ts = compute_temperature_state(baseline, temp, rubber_vol_frac)

            for rep in REPLICATES:
                sample_id = f"C-28-{mix_id}-{temp}C-Furnace-R{rep}"
                samples_meta.append({
                    "Sample_ID": sample_id,
                    "Mix_ID": mix_id,
                    "Rubber_Vol_Frac": rubber_vol_frac,
                    "W_C": w_c,
                    "Temperature_C": temp,
                    "Replicate_ID": rep,
                })

                # XRD
                xrd = synthesize_xrd(baseline, ts, noise=0.9)
                xrd_rows.append({
                    "Sample_ID": sample_id,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp,
                    "Replicate_ID": rep,
                    **xrd,
                })

                # TGA/DTA
                tga = synthesize_tga_dta(baseline, ts, noise=0.004)
                tga_rows.append({
                    "Sample_ID": sample_id,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp,
                    "Replicate_ID": rep,
                    **tga,
                })

                # SEM (multiple FOVs)
                for fov in range(1, FOVS_PER_SEM_REP + 1):
                    sem_metrics = synthesize_sem_fov_metrics(baseline, ts, fov, rubber_vol_frac)
                    sem_rows.append({
                        "Sample_ID": sample_id,
                        "Mix_ID": mix_id,
                        "Temperature_C": temp,
                        "Replicate_ID": rep,
                        **sem_metrics,
                    })

                # Micro-CT volume and metrics (per replicate)
                vol, labels, mct_metrics = generate_microstructure_volume(baseline, ts, rubber_vol_frac)
                npz_path = os.path.join(VOLUME_DIR, f"{sample_id}.npz")
                meta = {
                    "Sample_ID": sample_id,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp,
                    "Replicate_ID": rep,
                    "label_map": {"void": 0, "matrix": 1, "aggregate_quartz": 2, "rubber": 3, "crack": 4, "char": 5},
                    "voxel_size_um": VOXEL_SIZE_UM,
                    "shape_zyx": list(vol.shape),
                }
                np.savez_compressed(npz_path, volume=vol.astype(np.float32), labels=labels.astype(np.uint8), meta=json.dumps(meta))

                microct_rows.append({
                    "Sample_ID": sample_id,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp,
                    "Replicate_ID": rep,
                    **mct_metrics,
                })

    # Write CSV files
    write_csv(os.path.join(OUTPUT_DIR, "samples.csv"), samples_meta, [
        "Sample_ID", "Mix_ID", "Rubber_Vol_Frac", "W_C", "Temperature_C", "Replicate_ID"
    ])

    write_csv(os.path.join(OUTPUT_DIR, "xrd.csv"), xrd_rows, [
        "Sample_ID", "Mix_ID", "Temperature_C", "Replicate_ID",
        "CH_wt_pct", "CaCO3_wt_pct", "Quartz_wt_pct", "CaO_wt_pct", "Char_wt_pct",
        "Amorphous_index", "peak_intensity_CH_18deg", "peak_intensity_CaCO3_29deg", "peak_intensity_Quartz_26deg",
    ])

    write_csv(os.path.join(OUTPUT_DIR, "tga_dta.csv"), tga_rows, [
        "Sample_ID", "Mix_ID", "Temperature_C", "Replicate_ID",
        "mass_loss_pct_30_200", "mass_loss_pct_200_400", "mass_loss_pct_400_550", "mass_loss_pct_600_800",
        "total_mass_loss_pct", "DTA_peak_C_1", "DTA_peak_C_2", "DTA_peak_C_3", "DTA_peak_C_4",
    ])

    write_csv(os.path.join(OUTPUT_DIR, "sem.csv"), sem_rows, [
        "Sample_ID", "Mix_ID", "Temperature_C", "Replicate_ID",
        "FOV_ID", "pore_D10_um", "pore_D50_um", "pore_D90_um",
        "crack_density_per_mm2", "itz_damage_index_0_1", "interface_porosity_pct",
    ])

    write_csv(os.path.join(OUTPUT_DIR, "microct.csv"), microct_rows, [
        "Sample_ID", "Mix_ID", "Temperature_C", "Replicate_ID",
        "voxel_size_um", "porosity_pct", "crack_density_per_mm3", "pore_D50_um", "sphericity_mean", "connectivity_index",
    ])

    # Consistency summary (basic cross-checks)
    # Compare XRD CH_wt_pct trend vs TGA stage3 mass loss trend; and porosity vs SEM pore_D50_um
    # Aggregate correlations across all entries (Pearson-like manual calc)
    def corr(xs: List[float], ys: List[float]) -> float:
        if not xs or len(xs) != len(ys):
            return float('nan')
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if x.std() == 0 or y.std() == 0:
            return float('nan')
        return float(np.corrcoef(x, y)[0, 1])

    # Map sample_id -> values
    ch_by_sample: Dict[str, float] = {r["Sample_ID"]: float(r["CH_wt_pct"]) for r in xrd_rows}
    tga_chloss_by_sample: Dict[str, float] = {r["Sample_ID"]: float(r["mass_loss_pct_400_550"]) for r in tga_rows}
    porosity_by_sample: Dict[str, float] = {r["Sample_ID"]: float(r["porosity_pct"]) for r in microct_rows}

    # Use SEM D50 average per sample
    sem_d50_by_sample: Dict[str, List[float]] = {}
    for r in sem_rows:
        sem_d50_by_sample.setdefault(r["Sample_ID"], []).append(float(r["pore_D50_um"]))
    sem_d50_mean_by_sample = {k: float(np.mean(v)) for k, v in sem_d50_by_sample.items()}

    samples_common = sorted(set(ch_by_sample.keys()) & set(tga_chloss_by_sample.keys()) & set(porosity_by_sample.keys()) & set(sem_d50_mean_by_sample.keys()))

    ch_vals = [ch_by_sample[s] for s in samples_common]
    tga_stage3_vals = [tga_chloss_by_sample[s] for s in samples_common]
    porosity_vals = [porosity_by_sample[s] for s in samples_common]
    sem_d50_vals = [sem_d50_mean_by_sample[s] for s in samples_common]

    summary = {
        "n_samples": len(samples_common),
        "corr_XRD_CH_vs_TGA_CH_stage": corr(ch_vals, tga_stage3_vals),
        "corr_MicroCT_porosity_vs_SEM_D50": corr(porosity_vals, sem_d50_vals),
        "notes": "Correlations should be negative for CH vs TGA_CH (more CH -> less CH mass loss) and positive for porosity vs SEM D50.",
    }

    with open(os.path.join(OUTPUT_DIR, "consistency_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated Phase 3 dataset in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
