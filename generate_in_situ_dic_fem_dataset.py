#!/usr/bin/env python3
"""Generate a synthetic In-Situ Real-Time DIC-FEM synergized dataset.

This script fabricates a dataset inspired by the methodology described in
"A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time
Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted
Creep Activation".

It produces:
  * DIC real-time measurements (displacements, strains, quality metrics)
  * FEM simulation outputs (displacements, strains, stresses, internal states)
  * Residual fields between DIC and FEM results
  * Temperature profile log and a digital twin event log
  * Optional synthetic speckle images for qualitative inspection
  * Metadata and README files documenting the dataset structure

The dataset is written under `datasets/in_situ_real_time_dic_fem/`.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


try:  # Optional dependency for image generation
    from PIL import Image

    HAS_PIL = True
except Exception:  # pragma: no cover - environment specific
    HAS_PIL = False


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    coords: np.ndarray
    shape: Tuple[int, int]


@dataclass
class TimeStepData:
    time_index: int
    time_s: float
    temperature_k: float
    density: float
    dic: Dict[str, np.ndarray]
    fem: Dict[str, np.ndarray]
    residual: Dict[str, np.ndarray]
    summary: Dict[str, float]


def build_grid(size_mm: float = 40.0, nx: int = 25, ny: int = 25) -> Grid:
    half = size_mm / 2.0
    x_vals = np.linspace(-half, half, nx)
    y_vals = np.linspace(-half, half, ny)
    mesh_y, mesh_x = np.meshgrid(y_vals, x_vals, indexing="ij")
    coords = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    return Grid(x=mesh_x, y=mesh_y, coords=coords, shape=mesh_x.shape)


def temperature_profile(time_s: float, total_time_s: float) -> float:
    base_temp = 300.0
    peak_temp = 1673.0
    final_temp = 1273.0

    ramp_duration = total_time_s * 0.35
    soak_duration = total_time_s * 0.45
    cooldown_duration = total_time_s - ramp_duration - soak_duration

    if time_s <= ramp_duration:
        tau = time_s / ramp_duration
        return base_temp + (peak_temp - base_temp) * (tau ** 1.35)
    if time_s <= ramp_duration + soak_duration:
        tau = (time_s - ramp_duration) / soak_duration
        return peak_temp - 25.0 * tau
    tau = (time_s - ramp_duration - soak_duration) / max(cooldown_duration, 1.0)
    return peak_temp - 25.0 - (peak_temp - final_temp - 25.0) * (tau ** 1.15)


def relative_density(time_norm: float) -> float:
    base_density = 0.55
    max_gain = 0.42
    return base_density + max_gain / (1.0 + math.exp(-6.5 * (time_norm - 0.45)))


def elastic_modulus_mp(density: float, temperature_k: float) -> float:
    e_ref = 180_000.0  # MPa
    density_factor = (density / 0.97) ** 2.3
    thermal_softening = math.exp(-0.00055 * max(temperature_k - 300.0, 0.0))
    return e_ref * density_factor * thermal_softening


def shrinkage_alpha(density: float, temperature_k: float) -> float:
    dens_norm = (density - 0.55) / 0.42
    temp_norm = min(max((temperature_k - 300.0) / (1673.0 - 300.0), 0.0), 1.0)
    base = 2.5e-4
    variable = 3.8e-3 * (0.6 * dens_norm + 0.4 * (temp_norm ** 1.2))
    return base + variable


def compute_dic_fields(
    grid: Grid,
    shrink_alpha: float,
    density: float,
    temperature_k: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    anisotropy = 1.0 + 0.08 * np.sin(0.35 * grid.x / 10.0) * np.cos(0.4 * grid.y / 10.0)
    ux = -shrink_alpha * grid.x * anisotropy
    uy = -shrink_alpha * grid.y * (1.0 + 0.05 * np.cos(0.2 * grid.x / 10.0))
    radial = np.sqrt(grid.x ** 2 + grid.y ** 2)
    uz = 0.12 * shrink_alpha * np.exp(-(radial / 25.0) ** 2) * (1.0 + 0.1 * np.sin(radial / 6.0))

    eps_xx = -shrink_alpha * anisotropy * (1.0 + 0.04 * (grid.x / 20.0) ** 2)
    eps_yy = -shrink_alpha * (1.0 + 0.05 * (grid.y / 18.0) ** 2)
    eps_xy = 0.5 * shrink_alpha * np.sin(grid.x / 15.0) * np.sin(grid.y / 15.0)

    # Measurement noise increases with temperature
    noise_scale = 0.0025 + 0.0015 * ((temperature_k - 300.0) / (1673.0 - 300.0))
    ux += rng.normal(scale=noise_scale, size=ux.shape)
    uy += rng.normal(scale=noise_scale, size=uy.shape)
    uz += rng.normal(scale=noise_scale * 0.5, size=uz.shape)

    strain_noise = noise_scale * 0.15
    eps_xx += rng.normal(scale=strain_noise, size=eps_xx.shape)
    eps_yy += rng.normal(scale=strain_noise, size=eps_yy.shape)
    eps_xy += rng.normal(scale=strain_noise, size=eps_xy.shape)

    speckle_contrast = 0.72 + 0.12 * (1.0 - density) - 0.08 * ((temperature_k - 300.0) / 1400.0)
    tracking_quality = 0.9 - 0.15 * ((temperature_k - 300.0) / 1500.0) + 0.05 * density

    return {
        "ux_dic_mm": ux.ravel(),
        "uy_dic_mm": uy.ravel(),
        "uz_dic_mm": uz.ravel(),
        "exx_dic": eps_xx.ravel(),
        "eyy_dic": eps_yy.ravel(),
        "exy_dic": eps_xy.ravel(),
        "speckle_contrast": np.full(ux.size, speckle_contrast),
        "tracking_quality": np.clip(np.full(ux.size, tracking_quality), 0.0, 1.0),
    }


def compute_fem_fields(
    grid: Grid,
    shrink_alpha: float,
    density: float,
    temperature_k: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], float]:
    # FEM predictions are smoother than DIC measurements
    bias = 1.0 - 0.03 * np.cos(grid.x / 16.0) * np.cos(grid.y / 18.0)
    ux = -shrink_alpha * grid.x * bias
    uy = -shrink_alpha * grid.y * (1.0 + 0.02 * np.sin(grid.x / 20.0))
    radial = np.sqrt(grid.x ** 2 + grid.y ** 2)
    uz = 0.1 * shrink_alpha * np.exp(-(radial / 28.0) ** 2) * (1.0 + 0.06 * np.cos(radial / 8.0))

    eps_xx = -shrink_alpha * bias
    eps_yy = -shrink_alpha * (1.0 + 0.03 * np.cos(grid.y / 25.0))
    eps_xy = 0.45 * shrink_alpha * np.sin(grid.x / 18.0) * np.sin(grid.y / 18.0)

    e_mod = elastic_modulus_mp(density, temperature_k)
    nu_eff = 0.28 + 0.04 * (1.0 - density)

    sigma_xx = e_mod / (1.0 - nu_eff**2) * (eps_xx + nu_eff * eps_yy)
    sigma_yy = e_mod / (1.0 - nu_eff**2) * (eps_yy + nu_eff * eps_xx)
    sigma_xy = e_mod / (2.0 * (1.0 + nu_eff)) * eps_xy

    sigma_vm = np.sqrt(
        ((sigma_xx - sigma_yy) ** 2 + sigma_xx**2 + sigma_yy**2 + 6.0 * sigma_xy**2) / 2.0
    )

    plastic_strain = np.clip(0.0008 * (temperature_k - 1200.0) / 500.0, 0.0, 0.02)
    plastic_strain_field = plastic_strain * (0.6 + 0.4 * np.exp(-(radial / 20.0) ** 2))
    damage = np.clip(0.005 + 0.025 * (sigma_vm / max(e_mod * 0.002, 1.0)), 0.0, 0.15)
    density_field = density * (1.0 + 0.02 * np.exp(-(radial / 18.0) ** 2))

    return (
        {
            "ux_fem_mm": ux.ravel(),
            "uy_fem_mm": uy.ravel(),
            "uz_fem_mm": uz.ravel(),
            "exx_fem": eps_xx.ravel(),
            "eyy_fem": eps_yy.ravel(),
            "exy_fem": eps_xy.ravel(),
            "sigma_xx_mpa": sigma_xx.ravel(),
            "sigma_yy_mpa": sigma_yy.ravel(),
            "sigma_xy_mpa": sigma_xy.ravel(),
            "sigma_vm_mpa": sigma_vm.ravel(),
            "plastic_strain": plastic_strain_field.ravel(),
            "damage_parameter": np.clip(damage, 0.0, 0.5).ravel(),
            "relative_density": np.clip(density_field, 0.0, 1.0).ravel(),
        },
        float(np.max(sigma_vm)),
    )


def compute_residual_fields(
    dic_fields: Dict[str, np.ndarray], fem_fields: Dict[str, np.ndarray], e_mod_est: float
) -> Dict[str, np.ndarray]:
    dux = dic_fields["ux_dic_mm"] - fem_fields["ux_fem_mm"]
    duy = dic_fields["uy_dic_mm"] - fem_fields["uy_fem_mm"]
    duz = dic_fields["uz_dic_mm"] - fem_fields["uz_fem_mm"]
    dexx = dic_fields["exx_dic"] - fem_fields["exx_fem"]
    deyy = dic_fields["eyy_dic"] - fem_fields["eyy_fem"]
    dexy = dic_fields["exy_dic"] - fem_fields["exy_fem"]

    shear_mod = e_mod_est / (2.0 * (1.0 + 0.28))
    error_energy = 0.5 * (
        e_mod_est * (dexx**2 + deyy**2) + 2.0 * shear_mod * dexy**2
    )

    return {
        "dux_mm": dux,
        "duy_mm": duy,
        "duz_mm": duz,
        "dexx": dexx,
        "deyy": deyy,
        "dexy": dexy,
        "error_energy_density": error_energy,
    }


def generate_speckle_frame(
    base_pattern: np.ndarray,
    ux_field: np.ndarray,
    uy_field: np.ndarray,
    shape: Tuple[int, int],
    scale: float,
) -> np.ndarray:
    ux = ux_field.reshape(shape)
    uy = uy_field.reshape(shape)
    shift_x = int(np.round(np.mean(ux) * scale))
    shift_y = int(np.round(np.mean(uy) * scale))
    frame = np.roll(base_pattern, shift=shift_x, axis=1)
    frame = np.roll(frame, shift=shift_y, axis=0)
    modulation = 0.15 * np.tanh(ux / (np.max(np.abs(ux)) + 1e-6))
    modulation += 0.15 * np.tanh(uy / (np.max(np.abs(uy)) + 1e-6))

    # Upsample modulation field to match speckle resolution using bilinear interpolation
    target_shape = frame.shape
    src_y = np.linspace(0.0, 1.0, modulation.shape[0])
    src_x = np.linspace(0.0, 1.0, modulation.shape[1])
    tgt_y = np.linspace(0.0, 1.0, target_shape[0])
    tgt_x = np.linspace(0.0, 1.0, target_shape[1])

    temp = np.empty((modulation.shape[0], target_shape[1]))
    for i, row in enumerate(modulation):
        temp[i] = np.interp(tgt_x, src_x, row)

    resized = np.empty(target_shape)
    for j in range(temp.shape[1]):
        resized[:, j] = np.interp(tgt_y, src_y, temp[:, j])

    mod_frame = frame * (1.0 + resized)
    mod_frame = np.clip(mod_frame, 0.0, 1.0)
    return (mod_frame * 255).astype(np.uint8)


def controller_action(mean_abs_strain_residual: float, temperature_k: float) -> str:
    if mean_abs_strain_residual > 0.0025 and temperature_k > 1500.0:
        return "increase dwell to relieve stress"
    if mean_abs_strain_residual > 0.002:
        return "activate targeted creep heating"
    if mean_abs_strain_residual > 0.0012:
        return "adjust ramp rate"
    return "maintain schedule"


def generate_dataset(
    out_dir: Path,
    total_time_s: float = 3600.0,
    time_step_s: float = 60.0,
    grid_size_mm: float = 40.0,
    nx: int = 25,
    ny: int = 25,
    seed: int = 41,
) -> None:
    rng = np.random.default_rng(seed)
    grid = build_grid(size_mm=grid_size_mm, nx=nx, ny=ny)

    dic_dir = out_dir / "dic_stream"
    fem_dir = out_dir / "fem_simulation"
    residual_dir = out_dir / "residuals"
    log_dir = out_dir / "logs"

    for path in (dic_dir, fem_dir, residual_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    dic_csv = dic_dir / "dic_measurements.csv"
    fem_csv = fem_dir / "fem_predictions.csv"
    residual_csv = residual_dir / "residual_fields.csv"

    dic_headers = [
        "time_index",
        "time_s",
        "temperature_K",
        "x_mm",
        "y_mm",
        "ux_dic_mm",
        "uy_dic_mm",
        "uz_dic_mm",
        "exx_dic",
        "eyy_dic",
        "exy_dic",
        "speckle_contrast",
        "tracking_quality",
    ]

    fem_headers = [
        "time_index",
        "time_s",
        "temperature_K",
        "x_mm",
        "y_mm",
        "ux_fem_mm",
        "uy_fem_mm",
        "uz_fem_mm",
        "exx_fem",
        "eyy_fem",
        "exy_fem",
        "sigma_xx_mpa",
        "sigma_yy_mpa",
        "sigma_xy_mpa",
        "sigma_vm_mpa",
        "plastic_strain",
        "damage_parameter",
        "relative_density",
    ]

    residual_headers = [
        "time_index",
        "time_s",
        "temperature_K",
        "x_mm",
        "y_mm",
        "dux_mm",
        "duy_mm",
        "duz_mm",
        "dexx",
        "deyy",
        "dexy",
        "error_energy_density",
    ]

    dic_file = dic_csv.open("w", newline="")
    fem_file = fem_csv.open("w", newline="")
    residual_file = residual_csv.open("w", newline="")

    dic_writer = csv.writer(dic_file)
    fem_writer = csv.writer(fem_file)
    residual_writer = csv.writer(residual_file)

    dic_writer.writerow(dic_headers)
    fem_writer.writerow(fem_headers)
    residual_writer.writerow(residual_headers)

    temperature_profile_path = log_dir / "temperature_profile.csv"
    with temperature_profile_path.open("w", newline="") as temp_file:
        temp_writer = csv.writer(temp_file)
        temp_writer.writerow(["time_index", "time_s", "temperature_K"])

        time_indices = np.arange(0, int(total_time_s / time_step_s) + 1)

        digital_twin_log: List[Dict[str, float | str]] = []

        # Prepare speckle pattern if PIL is available
        if HAS_PIL:
            speckle_base = rng.uniform(0.0, 1.0, size=(256, 256))
            speckle_base = 0.5 * (speckle_base + np.roll(speckle_base, shift=3, axis=0))
            speckle_base = 0.5 * (speckle_base + np.roll(speckle_base, shift=5, axis=1))
            speckle_dir = dic_dir / "speckle_frames"
            speckle_dir.mkdir(exist_ok=True)
        else:
            speckle_base = None
            speckle_dir = None

        for idx, t_idx in enumerate(time_indices):
            time_s = float(t_idx * time_step_s)
            time_norm = time_s / total_time_s
            temp_k = temperature_profile(time_s, total_time_s)
            dens = relative_density(time_norm)
            shrink_alpha = shrinkage_alpha(dens, temp_k)

            dic_fields = compute_dic_fields(grid, shrink_alpha, dens, temp_k, rng)
            fem_fields, max_sigma_vm = compute_fem_fields(grid, shrink_alpha, dens, temp_k, rng)
            residual_fields = compute_residual_fields(
                dic_fields, fem_fields, elastic_modulus_mp(dens, temp_k)
            )

            temp_writer.writerow([idx, f"{time_s:.2f}", f"{temp_k:.2f}"])

            repeated = np.full(grid.coords.shape[0], idx)
            time_col = np.full(grid.coords.shape[0], time_s)
            temp_col = np.full(grid.coords.shape[0], temp_k)

            stacked_base = [
                repeated,
                time_col,
                temp_col,
                grid.coords[:, 0],
                grid.coords[:, 1],
            ]

            dic_stack = stacked_base + [dic_fields[k] for k in dic_headers[5:]]
            fem_stack = stacked_base + [fem_fields[k] for k in fem_headers[5:]]
            residual_stack = stacked_base + [residual_fields[k] for k in residual_headers[5:]]

            dic_rows = zip(*dic_stack)
            fem_rows = zip(*fem_stack)
            residual_rows = zip(*residual_stack)

            dic_writer.writerows(dic_rows)
            fem_writer.writerows(fem_rows)
            residual_writer.writerows(residual_rows)

            mean_abs_strain_residual = float(
                np.mean(np.abs(residual_fields["dexx"]))
            )
            max_disp_residual = float(
                np.max(np.sqrt(residual_fields["dux_mm"] ** 2 + residual_fields["duy_mm"] ** 2))
            )
            mean_energy = float(np.mean(residual_fields["error_energy_density"]))

            digital_twin_log.append(
                {
                    "time_index": int(idx),
                    "time_s": round(time_s, 2),
                    "temperature_K": round(temp_k, 2),
                    "relative_density_avg": round(float(np.mean(fem_fields["relative_density"])), 5),
                    "max_vm_stress_mpa": round(max_sigma_vm, 2),
                    "mean_abs_strain_residual": round(mean_abs_strain_residual, 5),
                    "max_displacement_residual_mm": round(max_disp_residual, 5),
                    "mean_error_energy_density": round(mean_energy, 5),
                    "controller_action": controller_action(mean_abs_strain_residual, temp_k),
                }
            )

            if HAS_PIL and speckle_base is not None and speckle_dir is not None:
                speckle_frame = generate_speckle_frame(
                    speckle_base,
                    dic_fields["ux_dic_mm"],
                    dic_fields["uy_dic_mm"],
                    grid.shape,
                    scale=50.0,
                )
                image = Image.fromarray(speckle_frame, mode="L")
                image.save(speckle_dir / f"frame_{idx:04d}.png")

        dic_file.close()
        fem_file.close()
        residual_file.close()

        digital_twin_path = log_dir / "digital_twin_log.json"
        digital_twin_path.write_text(
            json.dumps(
                {
                    "dataset": "In-Situ Real-Time DIC-FEM Synergized Dataset",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "time_step_s": time_step_s,
                    "entries": digital_twin_log,
                },
                indent=2,
            )
        )


def write_metadata(out_dir: Path, total_time_s: float, time_step_s: float, nx: int, ny: int) -> None:
    metadata = {
        "name": "In-Situ Real-Time DIC-FEM Synergized Dataset",
        "version": "1.0.0",
        "creation_timestamp": datetime.utcnow().isoformat() + "Z",
        "generator_script": "generate_in_situ_dic_fem_dataset.py",
        "description": (
            "Synthetic dataset emulating closed-loop DIC and FEM measurements during "
            "sintering of a solid oxide fuel cell half-cell. Includes synchronized "
            "displacement, strain, stress, and residual fields for digital twin analytics."
        ),
        "time": {
            "total_duration_s": total_time_s,
            "time_step_s": time_step_s,
            "num_steps": int(total_time_s / time_step_s) + 1,
        },
        "grid": {
            "points_x": nx,
            "points_y": ny,
            "extent_mm": 40.0,
            "coordinate_system": "Cartesian surface coordinates with origin at sample center",
        },
        "fields": {
            "dic_measurements": {
                "file": "dic_stream/dic_measurements.csv",
                "columns": {
                    "ux_dic_mm": "In-plane displacement along x (mm)",
                    "uy_dic_mm": "In-plane displacement along y (mm)",
                    "uz_dic_mm": "Out-of-plane displacement (mm)",
                    "exx_dic": "Normal strain xx from DIC",
                    "eyy_dic": "Normal strain yy from DIC",
                    "exy_dic": "Shear strain xy from DIC",
                    "speckle_contrast": "Speckle image contrast (0-1)",
                    "tracking_quality": "DIC correlation score (0-1)",
                },
            },
            "fem_predictions": {
                "file": "fem_simulation/fem_predictions.csv",
                "columns": {
                    "ux_fem_mm": "Predicted displacement x from FEM (mm)",
                    "uy_fem_mm": "Predicted displacement y from FEM (mm)",
                    "uz_fem_mm": "Predicted displacement z from FEM (mm)",
                    "sigma_vm_mpa": "Von Mises stress (MPa)",
                    "plastic_strain": "Accumulated plastic strain",
                    "damage_parameter": "Damage variable (0-1)",
                    "relative_density": "Local relative density",
                },
            },
            "residuals": {
                "file": "residuals/residual_fields.csv",
                "columns": {
                    "dux_mm": "DIC-FEM displacement residual x (mm)",
                    "duy_mm": "DIC-FEM displacement residual y (mm)",
                    "dexx": "DIC-FEM strain residual xx",
                    "error_energy_density": "Strain energy density of residuals",
                },
            },
        },
        "logs": {
            "temperature_profile": "logs/temperature_profile.csv",
            "digital_twin": "logs/digital_twin_log.json",
        },
        "notes": (
            "Values are numerically fabricated to preserve plausible trends while "
            "avoiding proprietary experimental data. Speckle frames are generated "
            "synthetically when Pillow is available."
        ),
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    readme_lines = [
        "# In-Situ Real-Time DIC-FEM Synergized Dataset\n",
        "\n",
        "????? `generate_in_situ_dic_fem_dataset.py` ?????????"
        "??DIC?FEM????????????????????????????\n",
        "\n",
        "## ????\n",
        "- `dic_stream/dic_measurements.csv`???????????????????\n",
        "- `dic_stream/speckle_frames/`??????????????????DIC???\n",
        "- `fem_simulation/fem_predictions.csv`?FEM??????????????????\n",
        "- `residuals/residual_fields.csv`?DIC?FEM???????????\n",
        "- `logs/temperature_profile.csv`??????-?????\n",
        "- `logs/digital_twin_log.json`???????????????\n",
        "- `metadata.json`??????????\n",
        "\n",
        "## ????\n",
        "- ?????{} s?????{} s?\n".format(int(time_step_s), int(total_time_s)),
        "- ?????{} ? {}??? 40 mm ? 40 mm ???\n".format(nx, ny),
        "- ?????????????????????????\n",
        "- ?????????????????????\n",
        "- ?????????????????????????\n",
        "\n",
        "## ??\n",
        "?????????????????????\n",
        "- *A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time "
        "Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted "
        "Creep Activation*.\n",
        "\n",
        "?????????????????????\n",
    ]

    readme_path = out_dir / "README.md"
    readme_path.write_text("".join(readme_lines))


def main() -> None:
    out_dir = Path("datasets/in_situ_real_time_dic_fem")
    total_time_s = 3600.0
    time_step_s = 60.0
    nx = 25
    ny = 25

    generate_dataset(
        out_dir=out_dir,
        total_time_s=total_time_s,
        time_step_s=time_step_s,
        nx=nx,
        ny=ny,
    )
    write_metadata(out_dir, total_time_s, time_step_s, nx, ny)


if __name__ == "__main__":
    main()
