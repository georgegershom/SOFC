from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DATASET_ROOT = Path("/workspace/datasets/phase3_microstructural_chemical").resolve()
RANDOM_SEED = 42

# Mixture definitions (High-Performance Rubberized Concrete variants)
MIXTURES: List[Dict] = [
    {"code": "HPRC-0", "rubber_pct": 0},
    {"code": "HPRC-20", "rubber_pct": 20},
]

# Temperature exposures (°C)
TEMPERATURES_C: List[int] = [20, 200, 400, 600, 800]

# Replicates per mixture-temperature condition
REPLICATES_PER_CONDITION = 1

# Modality-specific generation parameters
SEM_IMAGES_PER_SPECIMEN = 3
SEM_IMAGE_SIZE = (512, 512)  # pixels (H, W)
SEM_PIXEL_SIZE_UM = 1.0      # micrometers per pixel (for crack length estimates)

MICRO_CT_VOLUME_SHAPE = (96, 96, 96)  # (Z, Y, X)

XRD_RANGE_DEG = (5.0, 60.0)  # 2θ degrees
XRD_STEP_DEG = 0.02

TGA_MAX_TEMP_C = 1000
TGA_STEP_C = 1


@dataclass(frozen=True)
class Specimen:
    specimen_id: str
    mixture_code: str
    rubber_pct: int
    temperature_C: int
    replicate: int


def enumerate_specimens() -> List[Specimen]:
    specimens: List[Specimen] = []
    for mix in MIXTURES:
        mix_code = mix["code"]
        rubber_pct = mix["rubber_pct"]
        for temp in TEMPERATURES_C:
            for rep in range(1, REPLICATES_PER_CONDITION + 1):
                specimen_id = f"{mix_code}_T{temp}_R{rep}"
                specimens.append(
                    Specimen(
                        specimen_id=specimen_id,
                        mixture_code=mix_code,
                        rubber_pct=rubber_pct,
                        temperature_C=temp,
                        replicate=rep,
                    )
                )
    return specimens


def model_parameters(specimen: Specimen) -> Dict[str, float]:
    """
    Returns physically-plausible parameters controlling synthetic signals across modalities.
    Parameters are scaled by temperature and rubber content to reflect degradation phenomena.
    """
    t = specimen.temperature_C
    r = specimen.rubber_pct / 100.0

    # Portlandite (CH) relative amount vs temperature (strong drop @ ~450°C)
    if t <= 100:
        ch = 1.0
    elif t <= 300:
        ch = 0.9
    elif t <= 500:
        # linear drop from 0.9 at 300°C to 0.2 at 500°C
        ch = 0.9 - 0.7 * ((t - 300) / 200)
    elif t <= 650:
        ch = 0.1
    else:
        ch = 0.02

    # Calcite (CaCO3) decarbonation above ~700-900°C
    if t < 650:
        caco3 = 0.25
    elif t < 800:
        caco3 = 0.15
    else:
        caco3 = 0.05

    # CaO formation after CH and CaCO3 decomposition
    if t < 500:
        cao = 0.0
    elif t < 700:
        cao = 0.2
    else:
        cao = 0.35

    # C-S-H amorphous hump intensity decreases with dehydration, especially >600°C
    if t <= 200:
        csh_hump = 1.0
    elif t <= 400:
        csh_hump = 0.85
    elif t <= 600:
        csh_hump = 0.65
    else:
        csh_hump = 0.45

    # Rubber pyrolysis mass fraction (zero for control), concentrated 300-500°C
    rubber_pyrolysis_fraction = 0.0 if r == 0 else 0.06 + 0.10 * r  # ~6-16%

    # Microstructural porosity/crack scaling with temperature and rubber
    base_porosity = 0.05 + 0.04 * r  # higher base porosity with rubber
    temp_porosity_boost = min(0.25, (t / 1000) * (0.25 + 0.20 * r))
    micro_porosity = min(0.6, base_porosity + temp_porosity_boost)

    base_crack = 0.01 + 0.01 * r
    temp_crack_boost = 0.0 if t < 200 else min(0.18, (t - 200) / 800 * (0.20 + 0.15 * r))
    crack_density = min(0.3, base_crack + temp_crack_boost)

    # SEM: ITZ degradation severity in [0, 1]
    itz_deg = min(1.0, 0.15 + 0.7 * (t / 800) + 0.3 * r)

    return {
        "ch_rel": float(max(0.0, ch)),
        "caco3_rel": float(max(0.0, caco3)),
        "cao_rel": float(max(0.0, cao)),
        "csh_hump": float(max(0.0, csh_hump)),
        "rubber_pyrolysis_fraction": float(max(0.0, rubber_pyrolysis_fraction)),
        "micro_porosity": float(min(0.6, max(0.0, micro_porosity))),
        "crack_density": float(min(0.3, max(0.0, crack_density))),
        "itz_deg": float(min(1.0, max(0.0, itz_deg))),
    }
