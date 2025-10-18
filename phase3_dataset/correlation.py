from typing import Dict
import math
import numpy as np

from .config import TEMPERATURE_STEPS_C


def sigmoid(x: float, x0: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def compute_latent_state(mix: Dict[str, object], temp_c: int, rng: np.random.Generator) -> Dict[str, object]:
    rubber_vf = float(mix["rubber_vol_frac"])  # volume fraction rubber in fresh mix
    sf_f = float(mix["silica_fume_frac"])  # silica fume fraction (reduces CH)

    # Base phase fractions at ambient (normalized at 100%) including amorphous C-S-H
    # Note: These represent relative bulk phase inventory for XRD+amorphous estimate
    base = {
        "C-S-H (amorphous)": 58.0 + rng.normal(0, 1.5),
        "Portlandite (CH)": (18.0 - 10.0 * sf_f) + rng.normal(0, 0.8),
        "Calcite (CaCO3)": 6.0 + rng.normal(0, 0.5),
        "Quartz (SiO2)": 8.0 + rng.normal(0, 0.5),
        "Ettringite": 2.0 + rng.normal(0, 0.3),
        "Other crystalline": 3.0 + rng.normal(0, 0.5),
        "Rubber/Polymeric": rubber_vf * 10.0 + rng.normal(0, 0.4),  # scaled to wt%-like contribution
        "Char/Carbonaceous": 0.2 + rng.normal(0, 0.1),
    }

    # Temperature-dependent transformations
    t = temp_c

    # Dehydration of physically bound water (< 200C)
    dehydration = 0.04 * min(t, 200) / 200.0

    # Rubber softening/melt (around 200C) and pyrolysis (200-400C)
    melt = rubber_vf * 0.6 * sigmoid(t, 200, 0.05)
    pyrolysis = rubber_vf * 0.9 * max(0.0, (min(max(t, 200), 400) - 200) / 200.0)

    # Portlandite dehydroxylation (400-550C)
    ch_loss_frac = min(1.0, max(0.0, (t - 400) / 150.0))

    # C-S-H decalcification/structure collapse (600-800C)
    csh_loss_frac = 0.25 * max(0.0, (t - 600) / 200.0)

    # Calcite decarbonation (700-800C)
    caco3_loss_frac = 0.7 * max(0.0, (t - 700) / 100.0)

    phases = base.copy()

    # Apply transformations
    phases["Portlandite (CH)"] *= (1.0 - ch_loss_frac)
    phases["C-S-H (amorphous)"] *= (1.0 - csh_loss_frac)
    phases["Calcite (CaCO3)"] *= (1.0 - caco3_loss_frac)

    # Rubber transitions: decrease Rubber/Polymeric, increase Char at mid T, then gas/void
    rubber_poly = max(0.0, base["Rubber/Polymeric"] * (1.0 - pyrolysis))
    char_gain = base["Rubber/Polymeric"] * (0.4 * max(0.0, 1.0 - (t - 450) / 250.0))  # char remains until ~700C
    phases["Rubber/Polymeric"] = rubber_poly
    phases["Char/Carbonaceous"] = max(0.0, phases["Char/Carbonaceous"] + char_gain)

    # Normalize to 100 wt% equivalent for reporting
    total = sum(phases.values())
    if total <= 0:
        total = 1.0
    for k in list(phases.keys()):
        phases[k] = max(0.0, 100.0 * phases[k] / total)

    # Porosity evolution: base + effects from dehydration, rubber gas evolution, CH/CSH/CO2-related losses
    # NOTE: Remove stochastic noise to enforce monotonic increase with temperature for trend robustness
    base_porosity = 0.07 + 0.08 * rubber_vf  # HPRC slightly denser matrix; rubber increases base porosity
    porosity_gain = 0.0
    porosity_gain += 0.4 * dehydration  # water loss creates gel pores
    porosity_gain += 1.8 * pyrolysis  # gas channels from polymer pyrolysis
    porosity_gain += 0.6 * ch_loss_frac
    porosity_gain += 0.5 * csh_loss_frac
    porosity_gain += 0.5 * caco3_loss_frac
    # Small monotonicity bias to guarantee non-decreasing trend across 200->800C
    monotonic_bias = 0.003 * (t / 800.0)
    predicted_porosity = max(0.02, min(0.75, base_porosity + porosity_gain + monotonic_bias))

    # Crack factor: increases with temperature and rubber content (due to void coalescence)
    crack_factor = (0.2 * (t / 800.0) + 0.6 * rubber_vf * (t / 800.0))
    crack_factor = min(1.0, max(0.0, crack_factor + rng.normal(0, 0.03)))

    # ITZ damage index (0..1)
    itz_damage = min(1.0, max(0.0, 0.15 + 0.9 * (t / 800.0) + 0.8 * rubber_vf + rng.normal(0, 0.03)))

    # Melt phase fraction (rubber softening around 200C)
    melt_phase_fraction = min(1.0, max(0.0, melt + rng.normal(0, 0.02)))

    # TGA mass loss staging (fractions of initial mass)
    # Stage 1: <200C dehydration
    stage1 = 0.05 + 0.02 * (1 - rubber_vf)
    # Stage 2: 200-400C rubber pyrolysis
    stage2 = 0.25 * rubber_vf + 0.01
    # Stage 3: 400-550C CH dehydroxylation
    stage3 = 0.07 * (1.0 - sf_f / 0.15)
    # Stage 4: 600-800C decarbonation and C-S-H breakdown
    stage4 = 0.05 + 0.05 * (t >= 700)
    # Clip small negatives and renormalize for realism
    stages = np.array([stage1, stage2, stage3, stage4])
    stages = np.clip(stages + rng.normal(0, 0.002, size=4), 0.0, None)
    # total loss fraction at 800C
    total_loss_800 = stages.sum()

    return {
        "phases_wt_pct": phases,
        "predicted_porosity": predicted_porosity,
        "crack_factor": crack_factor,
        "itz_damage_index": itz_damage,
        "melt_phase_fraction": melt_phase_fraction,
        "tga_stage_fractions": stages.tolist(),
        "tga_total_loss_800": float(total_loss_800),
        "rubber_vol_frac": rubber_vf,
    }
