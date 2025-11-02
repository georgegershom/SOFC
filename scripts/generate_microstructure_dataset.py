#!/usr/bin/env python3
"""Generate a synthetic microstructure-informed dataset for SOFC sintering studies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class LayerConfig:
    name: str
    material_id: int
    thickness: int
    base_porosity: float
    porosity_sigma: float
    roughness_amp: float


MATERIAL_LIBRARY = {
    "void": {"id": 0, "gray": 30},
    "ni_ysz": {"id": 1, "gray": 140},
    "ysz": {"id": 2, "gray": 210},
    "lsm": {"id": 3, "gray": 180},
    "crofer": {"id": 4, "gray": 110},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate microstructure-informed dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="??????",
    )
    parser.add_argument("--seed", type=int, default=42, help="????")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="?????????",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"?????: {path}. ?? --overwrite ???")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def smooth_field(field: np.ndarray, iterations: int = 4) -> np.ndarray:
    result = field.astype(np.float64)
    for _ in range(iterations):
        neighbors = (
            np.roll(result, 1, axis=0)
            + np.roll(result, -1, axis=0)
            + np.roll(result, 1, axis=1)
            + np.roll(result, -1, axis=1)
        )
        result = 0.5 * result + 0.125 * neighbors
    return result


def generate_interface_offsets(
    rng: np.random.Generator,
    xy_shape: Tuple[int, int],
    amplitude: float,
    correlation: int,
) -> np.ndarray:
    noise = rng.standard_normal(xy_shape)
    smoothed = smooth_field(noise, iterations=max(1, correlation))
    centered = smoothed - smoothed.mean()
    return amplitude * centered / (np.abs(centered).max() + 1e-8)


def build_layered_volume(
    rng: np.random.Generator,
    shape: Tuple[int, int, int],
    layers: List[LayerConfig],
    interface_correlation: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    nz, ny, nx = shape
    volume = np.full(shape, MATERIAL_LIBRARY["void"]["id"], dtype=np.uint8)
    interface_maps: Dict[str, np.ndarray] = {}

    z_grid = np.arange(nz, dtype=np.float64)[:, None, None]
    current_surface = np.zeros((ny, nx), dtype=np.float64)

    for idx, layer in enumerate(layers):
        offset = generate_interface_offsets(
            rng,
            xy_shape=(ny, nx),
            amplitude=layer.roughness_amp,
            correlation=interface_correlation,
        )
        thickness = layer.thickness + offset
        thickness = np.clip(thickness, 1.0, nz - current_surface - 1.0)
        top_surface = current_surface + thickness
        mask = (z_grid >= current_surface[None, :, :]) & (z_grid < top_surface[None, :, :])
        volume[mask] = layer.material_id
        interface_maps[f"{layer.name}_top"] = top_surface.astype(np.float32)
        current_surface = top_surface

        if idx == len(layers) - 1 and current_surface.max() < nz:
            volume[z_grid >= current_surface[None, :, :]] = layer.material_id

    return volume, interface_maps


def embed_porosity(
    rng: np.random.Generator,
    volume: np.ndarray,
    layers: List[LayerConfig],
    porosity_bias: float,
) -> np.ndarray:
    porosity = np.zeros_like(volume, dtype=np.float32)
    for layer in layers:
        mask = volume == layer.material_id
        if not np.any(mask):
            continue
        local = rng.normal(loc=layer.base_porosity + porosity_bias, scale=layer.porosity_sigma, size=mask.sum())
        porosity[mask] = np.clip(local, 0.0, 0.65)
    porosity[volume == MATERIAL_LIBRARY["void"]["id"]] = 1.0
    return porosity


def carve_delamination(
    rng: np.random.Generator,
    volume: np.ndarray,
    target_interface: np.ndarray,
    thickness: int,
) -> np.ndarray:
    nz, ny, nx = volume.shape
    defect = np.zeros_like(volume, dtype=np.uint8)
    center_y = rng.integers(low=ny // 4, high=3 * ny // 4)
    center_x = rng.integers(low=nx // 4, high=3 * nx // 4)
    radius_y = rng.integers(low=ny // 8, high=ny // 5)
    radius_x = rng.integers(low=nx // 8, high=nx // 5)
    yy, xx = np.mgrid[0:ny, 0:nx]
    elliptical_mask = (((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2) <= 1.0
    interface_z = np.clip(target_interface.astype(int), thickness, nz - thickness - 1)
    for y in range(ny):
        if not elliptical_mask[y].any():
            continue
        x_indices = np.where(elliptical_mask[y])[0]
        z_centers = interface_z[y, x_indices]
        for idx, x in enumerate(x_indices):
            zc = int(z_centers[idx])
            lower = max(0, zc - thickness // 2)
            upper = min(nz, zc + thickness // 2)
            defect[lower:upper, y, x] = 1
            volume[lower:upper, y, x] = MATERIAL_LIBRARY["void"]["id"]
    return defect


def carve_microcracks(
    rng: np.random.Generator,
    volume: np.ndarray,
    layer_id: int,
    count: int,
    length: int,
    aperture: int,
) -> np.ndarray:
    nz, ny, nx = volume.shape
    crack = np.zeros_like(volume, dtype=np.uint8)
    for _ in range(count):
        z0 = rng.integers(low=0, high=nz)
        y0 = rng.integers(low=0, high=ny)
        x0 = rng.integers(low=0, high=nx - length)
        for dx in range(length):
            for dy in range(-aperture, aperture + 1):
                yy = np.clip(y0 + dy, 0, ny - 1)
                xv = x0 + dx
                if volume[z0, yy, xv] == layer_id:
                    volume[z0, yy, xv] = MATERIAL_LIBRARY["void"]["id"]
                    crack[z0, yy, xv] = 1
    return crack


def save_pgm(path: Path, array: np.ndarray) -> None:
    arr = np.clip(array, 0, 255).astype(np.uint8)
    height, width = arr.shape
    with path.open("w", encoding="ascii") as fh:
        fh.write(f"P2\n{width} {height}\n255\n")
        flat = arr.reshape(-1)
        for idx, value in enumerate(flat):
            fh.write(str(int(value)))
            fh.write("\n" if (idx + 1) % width == 0 else " ")


def build_sem_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        slice_data = volume[index, :, :]
    elif axis == 1:
        slice_data = volume[:, index, :]
    else:
        slice_data = volume[:, :, index]
    intensity_map = np.zeros_like(slice_data, dtype=np.uint8)
    for material in MATERIAL_LIBRARY.values():
        intensity_map[slice_data == material["id"]] = material["gray"]
    return intensity_map


def export_sem_images(stage_dir: Path, volume: np.ndarray, stage: str) -> None:
    sem_dir = stage_dir / "sem"
    sem_dir.mkdir(parents=True, exist_ok=True)
    mid_z = volume.shape[0] // 2
    mid_y = volume.shape[1] // 2
    slices = {
        f"{stage}_cross_section_z{mid_z:03d}.pgm": (0, mid_z),
        f"{stage}_laminate_y{mid_y:03d}.pgm": (1, mid_y),
    }
    for name, (axis, idx) in slices.items():
        image = build_sem_slice(volume, axis=axis, index=idx)
        save_pgm(sem_dir / name, image)


def compute_element_mapping(volume: np.ndarray, mesh_shape: Tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = volume.shape
    mz, my, mx = mesh_shape
    block_z = nz // mz
    block_y = ny // my
    block_x = nx // mx
    tags = np.zeros((mz, my, mx), dtype=np.int16)

    for iz in range(mz):
        for iy in range(my):
            for ix in range(mx):
                sub = volume[
                    iz * block_z : (iz + 1) * block_z,
                    iy * block_y : (iy + 1) * block_y,
                    ix * block_x : (ix + 1) * block_x,
                ]
                unique, counts = np.unique(sub, return_counts=True)
                dominant = unique[np.argmax(counts)]
                tags[iz, iy, ix] = int(dominant)
    return tags


def export_structured_mesh(
    stage_dir: Path,
    tags: np.ndarray,
    domain_size: Tuple[float, float, float],
) -> None:
    mz, my, mx = tags.shape
    dz = domain_size[2] / mz
    dy = domain_size[1] / my
    dx = domain_size[0] / mx

    nodes_path = stage_dir / "mesh_nodes.csv"
    elements_path = stage_dir / "mesh_elements.csv"
    material_path = stage_dir / "mesh_material_tags.csv"

    with nodes_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["node_id", "x_mm", "y_mm", "z_mm"])
        node_id = 0
        for iz in range(mz + 1):
            for iy in range(my + 1):
                for ix in range(mx + 1):
                    x = ix * dx
                    y = iy * dy
                    z = iz * dz
                    writer.writerow([node_id, round(x, 6), round(y, 6), round(z, 6)])
                    node_id += 1

    def node_index(ix: int, iy: int, iz: int) -> int:
        return iz * (my + 1) * (mx + 1) + iy * (mx + 1) + ix

    with elements_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["element_id"] + [f"n{i}" for i in range(8)]
        writer.writerow(header)
        element_id = 0
        for iz in range(mz):
            for iy in range(my):
                for ix in range(mx):
                    n0 = node_index(ix, iy, iz)
                    n1 = node_index(ix + 1, iy, iz)
                    n2 = node_index(ix + 1, iy + 1, iz)
                    n3 = node_index(ix, iy + 1, iz)
                    n4 = node_index(ix, iy, iz + 1)
                    n5 = node_index(ix + 1, iy, iz + 1)
                    n6 = node_index(ix + 1, iy + 1, iz + 1)
                    n7 = node_index(ix, iy + 1, iz + 1)
                    writer.writerow([element_id, n0, n1, n2, n3, n4, n5, n6, n7])
                    element_id += 1

    with material_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["element_id", "material_id"])
        element_id = 0
        for iz in range(tags.shape[0]):
            for iy in range(tags.shape[1]):
                for ix in range(tags.shape[2]):
                    writer.writerow([element_id, int(tags[iz, iy, ix])])
                    element_id += 1


def export_material_database(target_dir: Path) -> None:
    data = {
        "materials": [
            {
                "name": "8YSZ Electrolyte",
                "id": MATERIAL_LIBRARY["ysz"]["id"],
                "density": 5900,
                "poisson_ratio": 0.23,
                "thermal_expansion": {
                    "300K": 10.0e-6,
                    "1073K": 10.5e-6,
                },
                "youngs_modulus": {
                    "300K": 200e9,
                    "1073K": 170e9,
                },
                "creep": {
                    "model": "Norton-Bailey",
                    "parameters": {
                        "B": 8.5e-12,
                        "n": 1.8,
                        "Q": 3.85e5,
                        "reference_temperature": 1573,
                    },
                },
            },
            {
                "name": "Ni-YSZ Anode",
                "id": MATERIAL_LIBRARY["ni_ysz"]["id"],
                "density": 4200,
                "poisson_ratio": 0.29,
                "thermal_expansion": {
                    "300K": 12.5e-6,
                    "1073K": 13.3e-6,
                },
                "youngs_modulus": {
                    "300K": 55e9,
                    "1073K": 29e9,
                },
            },
            {
                "name": "LSM Cathode",
                "id": MATERIAL_LIBRARY["lsm"]["id"],
                "density": 5200,
                "poisson_ratio": 0.25,
                "thermal_expansion": {
                    "300K": 11.5e-6,
                    "1073K": 12.0e-6,
                },
                "youngs_modulus": {
                    "300K": 45e9,
                    "1073K": 40e9,
                },
            },
            {
                "name": "Crofer 22 APU",
                "id": MATERIAL_LIBRARY["crofer"]["id"],
                "density": 7700,
                "poisson_ratio": 0.30,
                "thermal_expansion": {
                    "300K": 11.5e-6,
                    "1073K": 11.9e-6,
                },
                "youngs_modulus": {
                    "300K": 160e9,
                    "1073K": 140e9,
                },
            },
        ],
        "boundary_conditions": {
            "assembly_pressure_MPa": 0.2,
            "thermal_profile": {
                "sintering_cooldown": {
                    "start_K": 1623,
                    "end_K": 298,
                    "rate_K_per_min": 2,
                },
                "steady_operation": 1073,
                "thermal_cycle": {
                    "min_K": 298,
                    "max_K": 1073,
                    "ramp_K_per_min": 5,
                    "dwell_min": 120,
                },
            },
        },
    }
    target = target_dir / "constitutive_models.json"
    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    calibration = target_dir / "creep_calibration.csv"
    with calibration.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["temperature_K", "stress_MPa", "creep_rate_per_s"])
        for temp in (1373, 1473, 1573):
            for stress in (60, 80, 100, 120):
                rate = 8.5e-12 * (stress ** 1.8) * math.exp(-3.85e5 / (8.314 * temp))
                writer.writerow([temp, stress, f"{rate:.3e}"])


def export_post_processing(stage_dir: Path, volume: np.ndarray, porosity: np.ndarray, label: str) -> None:
    density = (1.0 - porosity) * 6.0
    mean_porosity = float(porosity.mean())
    void_fraction = float((volume == MATERIAL_LIBRARY["void"]["id"]).mean())
    stats = {
        "label": label,
        "mean_porosity": mean_porosity,
        "void_fraction": void_fraction,
        "microstructure_shape": list(volume.shape),
    }
    (stage_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    np.savez_compressed(stage_dir / "density_field.npz", density=density.astype(np.float32))


def export_volume(stage_dir: Path, volume: np.ndarray, porosity: np.ndarray, defects: Dict[str, np.ndarray]) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stage_dir / "microstructure.npz",
        phase=volume.astype(np.uint8),
        porosity=porosity.astype(np.float32),
        **{name: arr.astype(np.uint8) for name, arr in defects.items()},
    )


def export_eds(stage_dir: Path, volume: np.ndarray, prefix: str) -> None:
    eds_dir = stage_dir / "sem_eds"
    eds_dir.mkdir(parents=True, exist_ok=True)
    slice_idx = volume.shape[1] // 2
    slice_data = volume[:, slice_idx, :]

    channels = {
        "Ni": (slice_data == MATERIAL_LIBRARY["ni_ysz"]["id"]).astype(np.float32),
        "Zr": (slice_data == MATERIAL_LIBRARY["ysz"]["id"]).astype(np.float32),
        "La": (slice_data == MATERIAL_LIBRARY["lsm"]["id"]).astype(np.float32),
        "Fe": (slice_data == MATERIAL_LIBRARY["crofer"]["id"]).astype(np.float32),
    }
    total = sum(channels.values()) + 1e-6
    for key in channels:
        channels[key] = channels[key] / total
    np.savez_compressed(eds_dir / f"{prefix}_eds_maps.npz", **channels)


def export_displacement(stage_dir: Path, initial_interfaces: Dict[str, np.ndarray], sintered_interfaces: Dict[str, np.ndarray]) -> None:
    common_keys = set(initial_interfaces) & set(sintered_interfaces)
    displacement = {}
    for key in common_keys:
        displacement[key] = (sintered_interfaces[key] - initial_interfaces[key]).astype(np.float32)
    np.savez_compressed(stage_dir / "interface_displacement.npz", **displacement)


def export_readme(dataset_dir: Path) -> None:
    readme = dataset_dir / "documentation" / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        """# ?????????

?????????A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation??????????????????????????????????????????????????????

## ????

- `material_properties/`??????????????? JSON/CSV ???
- `initial_state/`???????? SEM??CT ????????????
- `post_sinter/`??????? SEM/EDS??CT ????????????
- `fem_model/`?? ?CT ????????????????????????
- `documentation/data_dictionary.md`???????????

## ????

1. ?? `python scripts/generate_microstructure_dataset.py` ?????????? `--overwrite` ????
2. ?? `microstructure_dataset.zip` ??????????
3. ?CT ????? `phase`??????? `porosity`??????????????? FEM ????????????
4. `interface_displacement.npz` ?????????????????????????????

""",
        encoding="utf-8",
    )


def export_data_dictionary(dataset_dir: Path) -> None:
    doc = dataset_dir / "documentation" / "data_dictionary.md"
    doc.write_text(
        """# ????

## ???? (`*.npz`)

- `phase`?uint8?0=???1=Ni-YSZ?2=8YSZ?3=LSM?4=Crofer?
- `porosity`?float32??????????
- `delamination`?uint8 ???1 ?????????
- `microcrack`?uint8 ???1 ????????
- `density`?float32????? (g/cm?)?

## FEM ??

- `mesh_nodes.csv`??? ID ????????? (x, y, z)?
- `mesh_elements.csv`??????????????????
- `mesh_material_tags.csv`??? ID ?????????

## ????

- `constitutive_models.json`??????? Norton-Bailey ?????????
- `creep_calibration.csv`???????????????????

## SEM/EDS

- `.pgm`?????????? SEM ?????
- `*_eds_maps.npz`????????Ni?Zr?La?Fe???????

""",
        encoding="utf-8",
    )


def summarize_to_json(dataset_dir: Path) -> None:
    summary = {
        "dataset": "microstructure_informed_SOFC",
        "version": 1,
        "files": {}
    }
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            summary["files"][str(path.relative_to(dataset_dir))] = {
                "size_bytes": path.stat().st_size
            }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_zip(dataset_dir: Path) -> None:
    archive_root = dataset_dir.parent / "microstructure_dataset"
    zip_path = dataset_dir.parent / "microstructure_dataset.zip"
    if zip_path.exists():
        zip_path.unlink()
    if archive_root.exists() and archive_root.is_dir():
        shutil.rmtree(archive_root)
    shutil.make_archive(str(archive_root), "zip", root_dir=dataset_dir)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ensure_empty_dir(args.output_dir, overwrite=args.overwrite)

    layers_unfired = [
        LayerConfig("anode", MATERIAL_LIBRARY["ni_ysz"]["id"], 28, 0.32, 0.05, 1.8),
        LayerConfig("electrolyte", MATERIAL_LIBRARY["ysz"]["id"], 6, 0.08, 0.02, 1.2),
        LayerConfig("cathode", MATERIAL_LIBRARY["lsm"]["id"], 4, 0.18, 0.04, 1.4),
        LayerConfig("interconnect", MATERIAL_LIBRARY["crofer"]["id"], 26, 0.05, 0.01, 1.0),
    ]

    layers_sintered = [
        LayerConfig("anode", MATERIAL_LIBRARY["ni_ysz"]["id"], 24, 0.18, 0.03, 1.2),
        LayerConfig("electrolyte", MATERIAL_LIBRARY["ysz"]["id"], 5, 0.05, 0.015, 0.9),
        LayerConfig("cathode", MATERIAL_LIBRARY["lsm"]["id"], 3, 0.12, 0.03, 1.0),
        LayerConfig("interconnect", MATERIAL_LIBRARY["crofer"]["id"], 22, 0.03, 0.008, 0.8),
    ]

    volume_shape = (64, 128, 128)

    initial_state_dir = args.output_dir / "initial_state"
    post_state_dir = args.output_dir / "post_sinter"

    volume_unfired, interfaces_unfired = build_layered_volume(
        rng, volume_shape, layers_unfired, interface_correlation=5
    )
    porosity_unfired = embed_porosity(rng, volume_unfired, layers_unfired, porosity_bias=0.0)
    delamination = carve_delamination(
        rng,
        volume_unfired,
        interfaces_unfired["electrolyte_top"],
        thickness=4,
    )
    microcrack = carve_microcracks(
        rng,
        volume_unfired,
        layer_id=MATERIAL_LIBRARY["ni_ysz"]["id"],
        count=5,
        length=18,
        aperture=1,
    )
    defects_unfired = {"delamination": delamination, "microcrack": microcrack}

    export_volume(initial_state_dir, volume_unfired, porosity_unfired, defects_unfired)
    export_sem_images(initial_state_dir, volume_unfired, "unfired")
    export_eds(initial_state_dir, volume_unfired, prefix="unfired")
    export_post_processing(initial_state_dir, volume_unfired, porosity_unfired, label="initial")

    volume_sintered, interfaces_sintered = build_layered_volume(
        rng, volume_shape, layers_sintered, interface_correlation=6
    )
    porosity_sintered = embed_porosity(rng, volume_sintered, layers_sintered, porosity_bias=-0.03)
    delamination_s = carve_delamination(
        rng,
        volume_sintered,
        interfaces_sintered["electrolyte_top"],
        thickness=3,
    )
    microcrack_s = carve_microcracks(
        rng,
        volume_sintered,
        layer_id=MATERIAL_LIBRARY["ysz"]["id"],
        count=7,
        length=24,
        aperture=2,
    )
    defects_sintered = {"delamination": delamination_s, "microcrack": microcrack_s}

    export_volume(post_state_dir, volume_sintered, porosity_sintered, defects_sintered)
    export_sem_images(post_state_dir, volume_sintered, "sintered")
    export_eds(post_state_dir, volume_sintered, prefix="sintered")
    export_post_processing(post_state_dir, volume_sintered, porosity_sintered, label="post_sinter")
    export_displacement(post_state_dir, interfaces_unfired, interfaces_sintered)

    tags = compute_element_mapping(volume_sintered, mesh_shape=(16, 24, 24))
    fem_dir = args.output_dir / "fem_model"
    fem_dir.mkdir(parents=True, exist_ok=True)
    export_structured_mesh(
        fem_dir,
        tags,
        domain_size=(100.0, 100.0, 2.5),
    )

    bc_path = fem_dir / "boundary_conditions.json"
    bc_path.write_text(
        json.dumps(
            {
                "symmetry": ["x=0", "y=0"],
                "pressure_MPa": 0.2,
                "thermal_cycle": [298, 1073, 298],
                "warpage_target_mm": 0.42,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    export_material_database(args.output_dir / "material_properties")
    export_readme(args.output_dir)
    export_data_dictionary(args.output_dir)
    summarize_to_json(args.output_dir)
    make_zip(args.output_dir)


if __name__ == "__main__":
    main()

