import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import math
import numpy as np

# Optional image/point cloud writing
try:
    import PIL.Image as Image  # type: ignore
except Exception:
    Image = None  # type: ignore

DATA_ROOT = Path("/workspace/data/experimental_validation")
SAMPLES_DIR = DATA_ROOT / "samples"
SCHEMAS_DIR = DATA_ROOT / "schemas"

RNG = np.random.default_rng(42)


@dataclass
class PlateParameters:
    anode_thickness_um: float
    electrolyte_thickness_um: float
    cathode_thickness_um: float
    sinter_cycle: str


@dataclass
class CurvatureInverse:
    R_mm: float  # radius of curvature in mm
    avg_stress_MPa: float
    method: str = "stoney_bilayer"


@dataclass
class LayerRemovalStep:
    removed_layer: str
    thickness_removed_um: float
    warp_point_cloud_path: str


@dataclass
class SampleMeasurementPaths:
    warp_point_cloud_path: str
    warp_height_map_path: str
    curvature_inverse: CurvatureInverse
    layer_removal_sequence: List[LayerRemovalStep]
    xrd_map_path: str
    raman_map_path: str


@dataclass
class Sample:
    id: str
    split: str
    parameters: PlateParameters
    measurements: SampleMeasurementPaths


# --- Synthetic physics-informed models ---
# These are simplified but attempt to preserve realistic trends.

def generate_parameter_grid(n_center: int = 10, n_extremes_each: int = 10) -> List[PlateParameters]:
    anode_range = (200.0, 1000.0)  # um
    electrolyte_range = (5.0, 30.0)  # um
    cathode_range = (20.0, 100.0)  # um
    sinter_cycles = [
        "fast_lowT",  # under-sintered
        "standard",   # nominal
        "slow_highT", # over-sintered
    ]

    params: List[PlateParameters] = []

    # Centers
    for _ in range(n_center):
        p = PlateParameters(
            anode_thickness_um=RNG.uniform(*anode_range),
            electrolyte_thickness_um=RNG.uniform(*electrolyte_range),
            cathode_thickness_um=RNG.uniform(*cathode_range),
            sinter_cycle=random.choice(sinter_cycles),
        )
        params.append(p)

    # Extremes
    extremes = [
        (anode_range[0], electrolyte_range[0], cathode_range[0]),
        (anode_range[1], electrolyte_range[1], cathode_range[1]),
        (anode_range[0], electrolyte_range[1], cathode_range[1]),
        (anode_range[1], electrolyte_range[0], cathode_range[0]),
    ]
    for _ in range(n_extremes_each):
        base = random.choice(extremes)
        # Jitter around extremes
        p = PlateParameters(
            anode_thickness_um=max(anode_range[0], min(anode_range[1], base[0] + RNG.normal(0, 20))),
            electrolyte_thickness_um=max(electrolyte_range[0], min(electrolyte_range[1], base[1] + RNG.normal(0, 1))),
            cathode_thickness_um=max(cathode_range[0], min(cathode_range[1], base[2] + RNG.normal(0, 3))),
            sinter_cycle=random.choice(sinter_cycles),
        )
        params.append(p)

    return params


def gaussian_surface(nx: int, ny: int, amp: float, lx: float, ly: float, tilt_x: float = 0.0, tilt_y: float = 0.0) -> np.ndarray:
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = amp * np.exp(-(X**2 / (2 * lx**2) + Y**2 / (2 * ly**2)))
    Z += tilt_x * X + tilt_y * Y
    return Z


def synthesize_warp_field(params: PlateParameters, nx: int = 256, ny: int = 256) -> np.ndarray:
    # Trend: thickness mismatch + sintering severity -> curvature amplitude
    mismatch = (params.cathode_thickness_um - params.anode_thickness_um) / 1000.0
    severities = {"fast_lowT": 0.6, "standard": 1.0, "slow_highT": 1.4}
    severity = severities[params.sinter_cycle]

    base_amp_um = 30.0 * (abs(mismatch) + 0.1) * severity
    corr_len_x = 0.6 + 0.2 * RNG.random()
    corr_len_y = 0.6 + 0.2 * RNG.random()

    tilt_x = RNG.normal(0, 2.0)
    tilt_y = RNG.normal(0, 2.0)
    Z = gaussian_surface(nx, ny, base_amp_um, corr_len_x, corr_len_y, tilt_x=tilt_x, tilt_y=tilt_y)

    # Add small-scale roughness
    rough = RNG.normal(0, 0.8, size=(nx, ny))
    Z = Z + rough

    return Z


def height_map_to_point_cloud(z: np.ndarray, pitch_mm: float = 0.05) -> np.ndarray:
    nx, ny = z.shape
    xs = np.arange(nx) * pitch_mm
    ys = np.arange(ny) * pitch_mm
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), z.ravel() / 1000.0], axis=1)  # convert um->mm
    return pts


def stoney_inverse_average_stress(R_mm: float, thickness_sub_um: float, E_sub_GPa: float, nu_sub: float, thickness_film_um: float) -> float:
    # Stoney formula (thin film on thick substrate approximation)
    # sigma_film = (E_s * t_s^2) / (6 * (1 - nu_s) * t_f * R)
    t_s = thickness_sub_um / 1000.0  # to mm
    t_f = thickness_film_um / 1000.0
    E_s = E_sub_GPa * 1000.0  # to MPa
    sigma = (E_s * t_s * t_s) / (6.0 * (1.0 - nu_sub) * t_f * R_mm)
    return float(sigma)


def curvature_from_warp(z_mm: np.ndarray, pitch_mm: float) -> float:
    # Fit a quadratic surface z = ax^2 + by^2 + cxy + dx + ey + f and derive curvature
    nx, ny = z_mm.shape
    xs = np.arange(nx) * pitch_mm
    ys = np.arange(ny) * pitch_mm
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    A = np.column_stack([
        (X**2).ravel(), (Y**2).ravel(), (X*Y).ravel(), X.ravel(), Y.ravel(), np.ones(nx*ny)
    ])
    b = z_mm.ravel()
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b2, c, d, e, f = coeffs
    # Principal curvatures approximations ~ 2a and 2b for small slopes
    kx = 2 * a
    ky = 2 * b2
    k_mean = (kx + ky) / 2.0
    if abs(k_mean) < 1e-9:
        return 1e9
    return 1.0 / abs(k_mean)


def synthesize_xrd_map(z_um: np.ndarray, params: PlateParameters) -> np.ndarray:
    # Stress proportional to curvature and local slope variant
    gradx, grady = np.gradient(z_um)
    local = 0.5 * gradx + 0.5 * grady
    base = np.mean(z_um) * 0.0 + np.std(z_um) * 0.2
    noise = RNG.normal(0, 5.0, size=z_um.shape)
    return base + local + noise


def synthesize_raman_map(z_um: np.ndarray, params: PlateParameters) -> np.ndarray:
    # Correlate with curvature magnitude with different noise pattern
    lap = (
        -4 * z_um + np.roll(z_um, 1, 0) + np.roll(z_um, -1, 0) + np.roll(z_um, 1, 1) + np.roll(z_um, -1, 1)
    )
    base = 0.1 * lap
    noise = RNG.normal(0, 3.0, size=z_um.shape)
    return base + noise


def save_height_map_png(z_um: np.ndarray, path: Path) -> None:
    if Image is None:
        np.save(path.with_suffix(".npy"), z_um)
        return
    # NumPy 2.0 removed ndarray.ptp(); use np.ptp instead
    z_norm = (z_um - z_um.min()) / (np.ptp(z_um) + 1e-9)
    img = (255.0 * z_norm).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_point_cloud_xyz(points_mm: np.ndarray, path: Path) -> None:
    with open(path, "w") as f:
        for x, y, z in points_mm:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def simulate_layer_removal(z_um: np.ndarray, params: PlateParameters, steps: List[Tuple[str, float]]) -> List[np.ndarray]:
    # Remove contribution from a given layer as a fraction of curvature
    results = []
    current = z_um.copy()
    for layer, thick_um in steps:
        factor = 0.15 + 0.7 * (thick_um / max(params.anode_thickness_um, 1.0))
        delta = gaussian_surface(*z_um.shape, amp=np.median(np.abs(current)) * factor, lx=0.7, ly=0.7)
        current = current - delta
        results.append(current.copy())
    return results


def fabricate_dataset(n_samples: int = 30, seed: int = 123) -> Dict:
    RNG = np.random.default_rng(seed)
    random.seed(seed)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    params_list = generate_parameter_grid(n_center=10, n_extremes_each=max(1, n_samples - 10))
    params_list = params_list[:n_samples]

    manifest = {
        "dataset_name": "Experimental Validation Dataset (Reality Check)",
        "version": "0.1.0",
        "description": "Synthetic-but-physics-informed fabrication of experimental warp and partial stress measurements for SOFC plates.",
        # Use timezone-aware UTC timestamp
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(params_list),
        "parameter_space": {
            "anode_thickness_um": [200, 1000],
            "electrolyte_thickness_um": [5, 30],
            "cathode_thickness_um": [20, 100],
            "sinter_cycles": ["fast_lowT", "standard", "slow_highT"],
        },
        "samples": [],
    }

    for i, params in enumerate(params_list):
        sample_id = f"EV{i:04d}"
        sample_dir = SAMPLES_DIR / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Warp field
        z_um = synthesize_warp_field(params)
        pitch_mm = 0.05
        R_mm = curvature_from_warp(z_um / 1000.0, pitch_mm=pitch_mm)

        # Stoney average stress for electrolyte on anode (illustrative numbers)
        E_anode_GPa = 200.0
        nu_anode = 0.28
        avg_stress_MPa = stoney_inverse_average_stress(
            R_mm=R_mm,
            thickness_sub_um=params.anode_thickness_um,
            E_sub_GPa=E_anode_GPa,
            nu_sub=nu_anode,
            thickness_film_um=params.electrolyte_thickness_um,
        )

        # Save height map and point cloud
        height_map_path = sample_dir / "warp_height_map.png"
        point_cloud_path = sample_dir / "warp_point_cloud.xyz"
        save_height_map_png(z_um, height_map_path)
        pts_mm = height_map_to_point_cloud(z_um, pitch_mm=pitch_mm)
        save_point_cloud_xyz(pts_mm, point_cloud_path)

        # Layer removal simulation
        steps_def = [("cathode", params.cathode_thickness_um * 0.5), ("cathode", params.cathode_thickness_um * 0.5), ("electrolyte", params.electrolyte_thickness_um)]
        lr_fields = simulate_layer_removal(z_um, params, steps_def)
        lr_paths: List[LayerRemovalStep] = []
        for j, (layer_name, thickness_um) in enumerate(steps_def):
            lr_pc_path = sample_dir / f"layer_removal_{j:02d}.xyz"
            lr_pts_mm = height_map_to_point_cloud(lr_fields[j], pitch_mm=pitch_mm)
            save_point_cloud_xyz(lr_pts_mm, lr_pc_path)
            lr_paths.append(LayerRemovalStep(removed_layer=layer_name, thickness_removed_um=thickness_um, warp_point_cloud_path=str(lr_pc_path.relative_to(DATA_ROOT))))

        # XRD and Raman maps
        xrd_map = synthesize_xrd_map(z_um, params)
        raman_map = synthesize_raman_map(z_um, params)
        np.save(sample_dir / "xrd_map.npy", xrd_map)
        np.save(sample_dir / "raman_map.npy", raman_map)

        sample_entry = {
            "id": sample_id,
            "split": random.choice(["train", "val", "test"]),
            "parameters": asdict(params),
            "measurements": {
                "warp_point_cloud_path": str((point_cloud_path).relative_to(DATA_ROOT)),
                "warp_height_map_path": str((height_map_path).relative_to(DATA_ROOT)),
                "curvature_inverse": asdict(CurvatureInverse(R_mm=R_mm, avg_stress_MPa=avg_stress_MPa)),
                "layer_removal_sequence": [asdict(step) for step in lr_paths],
                "xrd_map_path": str((sample_dir / "xrd_map.npy").relative_to(DATA_ROOT)),
                "raman_map_path": str((sample_dir / "raman_map.npy").relative_to(DATA_ROOT)),
            },
        }
        manifest["samples"].append(sample_entry)

    with open(DATA_ROOT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    fabricate_dataset(n_samples=30, seed=123)
    print("Fabricated Experimental Validation dataset at", DATA_ROOT)
