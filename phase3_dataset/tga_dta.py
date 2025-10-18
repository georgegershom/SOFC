from typing import Dict, List, Tuple
import numpy as np


def _logistic(x: np.ndarray, x0: float, k: float, amp: float) -> np.ndarray:
    return amp / (1.0 + np.exp(-k * (x - x0)))


def generate_tga_dta(latent: Dict[str, object], rng: np.random.Generator) -> Dict[str, object]:
    # Temperature grid
    T = np.arange(25.0, 801.0, 5.0)

    s1, s2, s3, s4 = latent["tga_stage_fractions"]  # type: ignore
    s1 = float(s1); s2 = float(s2); s3 = float(s3); s4 = float(s4)

    # Construct mass loss curve as cumulative logistic steps
    loss1 = _logistic(T, 120.0, 0.04, s1)
    loss2 = _logistic(T, 320.0, 0.06, s2)
    loss3 = _logistic(T, 480.0, 0.08, s3)
    loss4 = _logistic(T, 740.0, 0.10, s4)

    cumulative_loss = loss1 + loss2 + loss3 + loss4
    mass_fraction = np.clip(1.0 - cumulative_loss, 0.05, 1.0)

    # DTA peaks corresponding to stages (rough, synthetic)
    dta_peaks = [
        {"peak_C": 120.0 + rng.normal(0, 5.0), "enthalpy_a.u.": -0.8 * s1},  # dehydration endotherm
        {"peak_C": 330.0 + rng.normal(0, 8.0), "enthalpy_a.u.": -1.2 * s2},  # rubber pyrolysis endotherm
        {"peak_C": 480.0 + rng.normal(0, 6.0), "enthalpy_a.u.": -0.9 * s3},  # CH dehyd endotherm
        {"peak_C": 740.0 + rng.normal(0, 6.0), "enthalpy_a.u.": -0.7 * s4},  # decarbonation endotherm
    ]

    return {
        "T_C": T.tolist(),
        "mass_fraction": mass_fraction.tolist(),
        "dta_peaks": dta_peaks,
    }
