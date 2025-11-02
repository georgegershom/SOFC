#!/usr/bin/env python3
"""???? DIC-FEM ?????????????

???????????????????SOFC???????????
?????????????DIC??????????FEM????
??????????????????????????????
?? NumPy ??? CSV/JSON ???

?????
    python scripts/generate_in_situ_dic_fem_dataset.py \
        --output datasets/in_situ_real_time_dic_fem
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridSpec:
    """?????"""

    size: Tuple[int, int]
    span_mm: Tuple[float, float]

    @property
    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.size
        width_mm, height_mm = self.span_mm
        x = np.linspace(-0.5 * width_mm, 0.5 * width_mm, nx)
        y = np.linspace(-0.5 * height_mm, 0.5 * height_mm, ny)
        return np.meshgrid(x, y)

    @property
    def spacing_mm(self) -> Tuple[float, float]:
        ny, nx = self.size
        width_mm, height_mm = self.span_mm
        dx = width_mm / (nx - 1)
        dy = height_mm / (ny - 1)
        return dx, dy


def build_time_and_temperature(time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """??????????????????????? 120 ???"""

    duration_s = 7200.0  # 2 ??
    time_s = np.linspace(0.0, duration_s, time_steps)
    # ???????->??->??
    key_times = np.array([0.0, 3600.0, 5400.0, duration_s])
    key_temps = np.array([25.0, 1300.0, 1300.0, 900.0])
    temperature_c = np.interp(time_s, key_times, key_temps)
    return time_s, temperature_c


def logistic_sintering_progress(temperature_c: np.ndarray) -> np.ndarray:
    """???????????0-1??"""

    centre = 950.0
    sharpness = 80.0
    progress = 1.0 / (1.0 + np.exp(-(temperature_c - centre) / sharpness))
    return np.clip(progress, 0.0, 1.0)


def generate_fields(
    time_s: np.ndarray,
    temperature_c: np.ndarray,
    grid: GridSpec,
    rng: np.random.Generator,
):
    """?? DIC ? FEM ??????????"""

    ny, nx = grid.size
    x_coords, y_coords = grid.coordinates
    x_norm = x_coords / (0.5 * grid.span_mm[0])
    y_norm = y_coords / (0.5 * grid.span_mm[1])
    r_sq = x_norm**2 + y_norm**2

    progress = logistic_sintering_progress(temperature_c)
    densification = 0.02 + 0.08 * progress**1.2
    creep_activation = 0.15 * progress + 0.02 * np.sin(0.5 * progress * np.pi)

    # ?? FEM ???
    shrink_x = densification[:, None, None] * (1.0 + 0.05 * y_norm)
    shrink_y = densification[:, None, None] * (1.0 - 0.05 * x_norm)
    warp_amplitude = 0.15 * progress + 0.02

    ux_fem = -shrink_x * x_coords
    uy_fem = -shrink_y * y_coords
    uz_fem = warp_amplitude[:, None, None] * (1.0 - r_sq) * np.exp(-r_sq)

    # ????????
    anisotropy = 0.03 * np.sin(np.pi * x_norm) * np.cos(np.pi * y_norm)

    # ??????????
    exx_fem = -densification[:, None, None] * (1.0 + anisotropy)
    eyy_fem = -densification[:, None, None] * (1.0 - anisotropy)
    exy_fem = 0.5 * creep_activation[:, None, None] * x_norm * y_norm

    # ?????????
    elasticity_e = 150e3  # MPa
    poisson = 0.28
    lam = elasticity_e * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = elasticity_e / (2 * (1 + poisson))
    trace_strain = exx_fem + eyy_fem
    sxx_fem = lam * trace_strain + 2 * mu * exx_fem
    syy_fem = lam * trace_strain + 2 * mu * eyy_fem
    sxy_fem = 2 * mu * exy_fem
    # Von Mises
    sigma_vm = np.sqrt(
        np.maximum(
            0.0,
            0.5 * ((sxx_fem - syy_fem) ** 2 + sxx_fem**2 + syy_fem**2) + 3.0 * sxy_fem**2,
        )
    )

    # ??????
    plastic_strain = 0.02 * progress[:, None, None] * (1.0 + 0.1 * r_sq)
    damage = np.clip(0.05 + 0.25 * progress[:, None, None] ** 1.3 + 0.05 * r_sq, 0.0, 0.95)
    relative_density = np.clip(0.55 + 0.4 * progress[:, None, None] - 0.02 * r_sq, 0.55, 0.99)

    # DIC ????????? & ??
    def noisy(field: np.ndarray, noise_scale: float, bias: float = 0.0) -> np.ndarray:
        return field + bias + rng.normal(scale=noise_scale, size=field.shape)

    ux_dic = noisy(ux_fem, noise_scale=0.0015)
    uy_dic = noisy(uy_fem, noise_scale=0.0015)
    uz_dic = noisy(uz_fem, noise_scale=0.0008, bias=0.0002 * (1.0 - progress)[:, None, None])
    exx_dic = noisy(exx_fem, noise_scale=2.5e-4)
    eyy_dic = noisy(eyy_fem, noise_scale=2.5e-4)
    exy_dic = noisy(exy_fem, noise_scale=1.5e-4)

    # ??
    residual_ux = ux_dic - ux_fem
    residual_uy = uy_dic - uy_fem
    residual_uz = uz_dic - uz_fem
    residual_exx = exx_dic - exx_fem
    residual_eyy = eyy_dic - eyy_fem
    residual_exy = exy_dic - exy_fem

    fields = {
        "fem": {
            "ux": ux_fem,
            "uy": uy_fem,
            "uz": uz_fem,
            "exx": exx_fem,
            "eyy": eyy_fem,
            "exy": exy_fem,
            "sxx": sxx_fem,
            "syy": syy_fem,
            "sxy": sxy_fem,
            "svm": sigma_vm,
            "plastic_strain": plastic_strain,
            "damage": damage,
            "relative_density": relative_density,
        },
        "dic": {
            "ux": ux_dic,
            "uy": uy_dic,
            "uz": uz_dic,
            "exx": exx_dic,
            "eyy": eyy_dic,
            "exy": exy_dic,
        },
        "residuals": {
            "ux": residual_ux,
            "uy": residual_uy,
            "uz": residual_uz,
            "exx": residual_exx,
            "eyy": residual_eyy,
            "exy": residual_exy,
        },
        "derived": {
            "progress": progress,
            "densification": densification,
            "creep_activation": creep_activation,
        },
        "grid": {
            "x_coords": x_coords,
            "y_coords": y_coords,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "r_sq": r_sq,
        },
    }

    return fields


def save_numpy_fields(output_dir: Path, time_s: np.ndarray, temperature_c: np.ndarray, fields):
    """?????????? NPZ?"""

    np.savez_compressed(
        output_dir / "dic" / "real_time_dic_fields.npz",
        time_s=time_s,
        temperature_c=temperature_c,
        x_coords=fields["grid"]["x_coords"],
        y_coords=fields["grid"]["y_coords"],
        ux=fields["dic"]["ux"],
        uy=fields["dic"]["uy"],
        uz=fields["dic"]["uz"],
        exx=fields["dic"]["exx"],
        eyy=fields["dic"]["eyy"],
        exy=fields["dic"]["exy"],
    )

    np.savez_compressed(
        output_dir / "fem" / "fem_predictions.npz",
        time_s=time_s,
        temperature_c=temperature_c,
        x_coords=fields["grid"]["x_coords"],
        y_coords=fields["grid"]["y_coords"],
        ux=fields["fem"]["ux"],
        uy=fields["fem"]["uy"],
        uz=fields["fem"]["uz"],
        exx=fields["fem"]["exx"],
        eyy=fields["fem"]["eyy"],
        exy=fields["fem"]["exy"],
        sxx=fields["fem"]["sxx"],
        syy=fields["fem"]["syy"],
        sxy=fields["fem"]["sxy"],
        svm=fields["fem"]["svm"],
        plastic_strain=fields["fem"]["plastic_strain"],
        damage=fields["fem"]["damage"],
        relative_density=fields["fem"]["relative_density"],
        densification=fields["derived"]["densification"],
        creep_activation=fields["derived"]["creep_activation"],
    )

    np.savez_compressed(
        output_dir / "residuals" / "field_residuals.npz",
        time_s=time_s,
        temperature_c=temperature_c,
        ux=fields["residuals"]["ux"],
        uy=fields["residuals"]["uy"],
        uz=fields["residuals"]["uz"],
        exx=fields["residuals"]["exx"],
        eyy=fields["residuals"]["eyy"],
        exy=fields["residuals"]["exy"],
    )


def save_temperature_profile(output_dir: Path, time_s: np.ndarray, temperature_c: np.ndarray) -> None:
    profile = pd.DataFrame({"time_s": time_s, "temperature_c": temperature_c})
    profile.to_csv(output_dir / "boundary_conditions" / "temperature_profile.csv", index=False)


def save_residual_statistics(output_dir: Path, residuals: dict) -> None:
    stats = []
    for name, array in residuals.items():
        stats.append(
            {
                "field": name,
                "mean": float(array.mean()),
                "std": float(array.std()),
                "max": float(array.max()),
                "min": float(array.min()),
                "rmse": float(np.sqrt(np.mean(array**2))),
            }
        )

    df = pd.DataFrame(stats)
    df.to_csv(output_dir / "residuals" / "residual_statistics.csv", index=False)


def save_metadata(output_dir: Path, grid: GridSpec, time_s: np.ndarray, temperature_c: np.ndarray, seed: int) -> None:
    metadata = {
        "dataset_name": "In-Situ Real-Time DIC-FEM Synergized Dataset (Synthetic)",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "generator_script": "scripts/generate_in_situ_dic_fem_dataset.py",
        "random_seed": seed,
        "time_steps": len(time_s),
        "frame_rate_hz": 1.0 / (time_s[1] - time_s[0]) if len(time_s) > 1 else None,
        "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0,
        "temperature_range_c": [float(temperature_c.min()), float(temperature_c.max())],
        "grid": {
            "size": {
                "ny": grid.size[0],
                "nx": grid.size[1],
            },
            "span_mm": {
                "y": grid.span_mm[1],
                "x": grid.span_mm[0],
            },
            "spacing_mm": {
                "dy": grid.spacing_mm[1],
                "dx": grid.spacing_mm[0],
            },
        },
        "fields": {
            "dic": ["ux", "uy", "uz", "exx", "eyy", "exy"],
            "fem": [
                "ux",
                "uy",
                "uz",
                "exx",
                "eyy",
                "exy",
                "sxx",
                "syy",
                "sxy",
                "svm",
                "plastic_strain",
                "damage",
                "relative_density",
                "densification",
                "creep_activation",
            ],
            "residuals": ["ux", "uy", "uz", "exx", "eyy", "exy"],
        },
        "experiment_context": {
            "furnace": "???????????",
            "camera": "???????? + ????????? 2048x2048",
            "speckle_pattern": "??????/???????",
            "notes": "???????????? DIC-FEM ????",
        },
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_digital_twin_log(output_dir: Path, time_s: np.ndarray, temperature_c: np.ndarray, fields) -> None:
    log = {
        "run_id": f"dic-fem-sim-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}",
        "time_span_s": [float(time_s[0]), float(time_s[-1])],
        "max_temperature_c": float(temperature_c.max()),
        "min_temperature_c": float(temperature_c.min()),
        "model_state": {
            "avg_relative_density": float(fields["fem"]["relative_density"].mean()),
            "max_plastic_strain": float(fields["fem"]["plastic_strain"].max()),
            "max_damage": float(fields["fem"]["damage"].max()),
        },
        "validation": {
            "residual_rms": {
                key: float(np.sqrt(np.mean(val**2))) for key, val in fields["residuals"].items()
            }
        },
        "notes": "?????DIC ????? FEM ???",
    }

    with open(output_dir / "logs" / "digital_twin_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def save_quicklook_csv(output_dir: Path, time_s: np.ndarray, temperature_c: np.ndarray, fields) -> None:
    """?????????????????????"""

    ny, nx = fields["dic"]["ux"].shape[1:]
    mid_idx = (ny // 2, nx // 2)
    edge_idx = (ny // 2, nx - 1)
    corner_idx = (0, 0)

    def extract_series(field: np.ndarray):
        return {
            "centre": field[:, mid_idx[0], mid_idx[1]],
            "edge": field[:, edge_idx[0], edge_idx[1]],
            "corner": field[:, corner_idx[0], corner_idx[1]],
        }

    ux_series = extract_series(fields["dic"]["ux"])
    uy_series = extract_series(fields["dic"]["uy"])
    uz_series = extract_series(fields["dic"]["uz"])

    df = pd.DataFrame(
        {
            "time_s": time_s,
            "temperature_c": temperature_c,
            "ux_centre_mm": ux_series["centre"],
            "ux_edge_mm": ux_series["edge"],
            "ux_corner_mm": ux_series["corner"],
            "uy_centre_mm": uy_series["centre"],
            "uy_edge_mm": uy_series["edge"],
            "uy_corner_mm": uy_series["corner"],
            "uz_centre_mm": uz_series["centre"],
            "uz_edge_mm": uz_series["edge"],
            "uz_corner_mm": uz_series["corner"],
        }
    )

    df.to_csv(output_dir / "quicklook" / "dic_displacement_timeseries.csv", index=False)


def ensure_directories(output_dir: Path) -> None:
    for sub in ["dic", "fem", "residuals", "logs", "boundary_conditions", "quicklook"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="???? DIC-FEM ?????")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/in_situ_real_time_dic_fem"),
        help="???????????datasets/in_situ_real_time_dic_fem?",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        metavar=("NY", "NX"),
        default=(48, 48),
        help="???? (NY NX)??? 48 48",
    )
    parser.add_argument(
        "--span-mm",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(20.0, 20.0),
        help="???????mm???? 20x20 mm",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=120,
        help="???????? 120 ??",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="??????? 2025?",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    grid = GridSpec(size=tuple(args.grid_size), span_mm=tuple(args.span_mm))
    time_s, temperature_c = build_time_and_temperature(args.time_steps)

    ensure_directories(args.output)

    fields = generate_fields(time_s, temperature_c, grid, rng)

    save_numpy_fields(args.output, time_s, temperature_c, fields)
    save_temperature_profile(args.output, time_s, temperature_c)
    save_residual_statistics(args.output, fields["residuals"])
    save_metadata(args.output, grid, time_s, temperature_c, args.seed)
    save_digital_twin_log(args.output, time_s, temperature_c, fields)
    save_quicklook_csv(args.output, time_s, temperature_c, fields)

    print(f"???????{args.output.resolve()}")


if __name__ == "__main__":
    main()
