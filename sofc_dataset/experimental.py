from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import math
import numpy as np
import pandas as pd
from PIL import Image

from .eis import synthesize_eis


def synthesize_experimental_dataset(
    out_dir: Path,
    duration_hours: float = 24.0,
    sample_rate_hz: float = 1.0,
    eis_period_hours: float = 6.0,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    n_steps = int(duration_hours * 3600 * sample_rate_hz)
    t = np.arange(n_steps) / sample_rate_hz
    t_hours = t / 3600.0

    # Global signals: current, voltage, power, temps, flow
    I_A = 50.0 + 10.0 * np.sin(2 * math.pi * t_hours / 6.0) + rng.normal(scale=1.0, size=n_steps)
    V_V = 0.9 - 0.05 * (t_hours / duration_hours) + 0.01 * np.sin(2 * math.pi * t_hours / 12.0) + rng.normal(scale=0.005, size=n_steps)
    V_V = np.clip(V_V, 0.5, 1.1)
    P_W = I_A * V_V

    T_in_fuel_C = 700.0 + 10.0 * np.sin(2 * math.pi * t_hours / 3.0) + rng.normal(scale=0.5, size=n_steps)
    T_out_fuel_C = T_in_fuel_C + 15.0 + 3.0 * np.sin(2 * math.pi * t_hours / 2.0)

    T_in_air_C = 700.0 + 8.0 * np.cos(2 * math.pi * t_hours / 4.0) + rng.normal(scale=0.5, size=n_steps)
    T_out_air_C = T_in_air_C + 10.0 + 2.0 * np.cos(2 * math.pi * t_hours / 2.5)

    flow_fuel_sccm = 800.0 + 50.0 * np.sin(2 * math.pi * t_hours / 5.0) + rng.normal(scale=5.0, size=n_steps)
    flow_air_sccm = 2000.0 + 120.0 * np.cos(2 * math.pi * t_hours / 7.0) + rng.normal(scale=10.0, size=n_steps)

    df = pd.DataFrame({
        "time_s": t,
        "I_A": I_A,
        "V_V": V_V,
        "P_W": P_W,
        "T_in_fuel_C": T_in_fuel_C,
        "T_out_fuel_C": T_out_fuel_C,
        "T_in_air_C": T_in_air_C,
        "T_out_air_C": T_out_air_C,
        "flow_fuel_sccm": flow_fuel_sccm,
        "flow_air_sccm": flow_air_sccm,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "timeseries.csv", index=False)

    # EIS snapshots
    eis_dir = out_dir / "eis"
    eis_dir.mkdir(exist_ok=True)
    n_eis = max(1, int(duration_hours / eis_period_hours))
    for k in range(n_eis):
        th = k * eis_period_hours
        idx = min(n_steps - 1, int(th * 3600 * sample_rate_hz))
        eis = synthesize_eis(
            T_K= (T_in_fuel_C[idx] + 273.15),
            current_density_A_per_cm2= (I_A[idx] / 100.0),
            age_hours= th,
            seed= seed + k,
        )
        eis_df = pd.DataFrame({
            "frequency_Hz": eis["frequency_Hz"],
            "Zreal_ohm": eis["Zreal_ohm"],
            "Zimag_ohm": eis["Zimag_ohm"],
        })
        eis_df.to_csv(eis_dir / f"eis_{k:03d}.csv", index=False)

    # Thermal images (simulate a camera view of surface temperature)
    img_dir = out_dir / "thermal_images"
    img_dir.mkdir(exist_ok=True)
    n_imgs = min(12, max(3, n_eis * 2))
    for k in range(n_imgs):
        th = (k + 1) * duration_hours / (n_imgs + 1)
        base_temp = 700.0 + 20.0 * np.sin(2 * math.pi * th / duration_hours)
        img = _synthetic_thermal_image(256, 256, base_temp, rng)
        img.save(img_dir / f"thermal_{k:03d}.png")

    # Strain gauges (few points)
    sg_dir = out_dir / "strain_gauges"
    sg_dir.mkdir(exist_ok=True)
    n_gauges = 6
    locs = rng.uniform(0.1, 0.9, size=(n_gauges, 2))
    # simulate time-varying strain with thermal cycles
    sg = []
    for i in range(n_gauges):
        phase = rng.uniform(0, 2*np.pi)
        strain = 300e-6 + 60e-6 * np.sin(2 * math.pi * t_hours / 6.0 + phase) + rng.normal(scale=5e-6, size=n_steps)
        sg.append(strain)
    sg = np.stack(sg, axis=1)
    sg_df = pd.DataFrame(sg, columns=[f"gauge_{i}" for i in range(n_gauges)])
    sg_df.insert(0, "time_s", t)
    sg_df.to_csv(sg_dir / "strain_timeseries.csv", index=False)
    with open(sg_dir / "gauge_locations.json", "w") as f:
        json.dump({f"gauge_{i}": {"x_frac": float(locs[i,0]), "y_frac": float(locs[i,1])} for i in range(n_gauges)}, f, indent=2)

    # Acoustic emission events
    ae_dir = out_dir / "acoustic_emission"
    ae_dir.mkdir(exist_ok=True)
    lam0 = 1.0 / 3600.0  # base rate per second
    lam_t = lam0 * (1.0 + 3.0 * (t_hours / duration_hours))  # increase with time
    # thinning for inhomogeneous Poisson
    p = lam_t / lam_t.max()
    u = rng.uniform(size=n_steps)
    events_idx = np.where(u < p * 0.05)[0]  # scale overall rate
    ae_df = pd.DataFrame({
        "time_s": t[events_idx],
        "amplitude": rng.lognormal(mean=0.0, sigma=0.6, size=len(events_idx)),
        "center_freq_kHz": rng.uniform(50.0, 300.0, size=len(events_idx)),
        "duration_ms": rng.uniform(0.2, 5.0, size=len(events_idx)),
    })
    ae_df.to_csv(ae_dir / "ae_events.csv", index=False)


def _synthetic_thermal_image(width: int, height: int, base_temp_C: float, rng: np.random.Generator) -> Image.Image:
    x = np.linspace(0.0, 1.0, width)
    y = np.linspace(0.0, 1.0, height)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # smooth patterns
    field = base_temp_C + 5.0 * np.sin(2*np.pi*X) + 3.0 * np.cos(2*np.pi*Y) + 2.0 * np.sin(2*np.pi*(X+Y))
    field += rng.normal(scale=0.5, size=(width, height))
    # map to 8-bit
    fmin, fmax = field.min(), field.max()
    img = np.uint8(255 * (field - fmin) / (fmax - fmin + 1e-6))
    return Image.fromarray(img.T, mode="L")
