#!/usr/bin/env python3
import os
import json
import gzip
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import math
import random

# -----------------------------
# Configuration and Models
# -----------------------------

@dataclass
class Sensor:
    id: str
    type: str  # "thermocouple" or "ir_pixel"
    location: Tuple[float, float, float]  # x, y, z in mm relative to cell corner

@dataclass
class ScenarioSpec:
    name: str
    description: str
    duration_s: int
    sample_rate_hz: float
    spatial_grid_mm: Tuple[int, int]  # nx, ny for IR-like synthetic map
    thermocouples: List[Sensor]
    seed: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv_gz(path: str, header: List[str], rows: List[List[float]]) -> None:
    with gzip.open(path, mode='wt', newline='') as gz:
        writer = csv.writer(gz)
        writer.writerow(header)
        writer.writerows(rows)


def generate_ir_map(nx: int, ny: int, base_temp: float, gradients: Tuple[float, float], hotspot: Tuple[int, int, float]) -> List[List[float]]:
    gx, gy = gradients
    hx, hy, amp = hotspot
    data = []
    for j in range(ny):
        row = []
        for i in range(nx):
            dx = i - hx
            dy = j - hy
            r2 = dx*dx + dy*dy
            hotspot_term = amp * math.exp(-r2 / (2 * (max(nx, ny) * 0.08)**2))
            value = base_temp + gx * (i / max(1, nx-1)) + gy * (j / max(1, ny-1)) + hotspot_term
            row.append(value)
        data.append(row)
    return data


def add_gaussian_noise(value: float, sigma: float) -> float:
    return value + random.gauss(0.0, sigma)


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# -----------------------------
# Scenario Synthesizers
# -----------------------------

def synthesize_sintering(spec: ScenarioSpec, out_dir: str) -> Dict:
    random.seed(spec.seed)
    nx, ny = spec.spatial_grid_mm
    dt = 1.0 / spec.sample_rate_hz
    steps = int(spec.duration_s * spec.sample_rate_hz)

    # Temperature schedule (C): ramp to 1350C, hold, cool
    ramp_up_s = int(0.25 * spec.duration_s)
    hold_s = int(0.5 * spec.duration_s)
    ramp_down_s = spec.duration_s - ramp_up_s - hold_s

    def schedule(t: float) -> float:
        if t < ramp_up_s:
            return 25 + (1350 - 25) * (t / ramp_up_s)
        elif t < ramp_up_s + hold_s:
            return 1350 + 10 * math.sin(2 * math.pi * t / 600.0)
        else:
            tau = ramp_down_s
            return 1350 - (1350 - 25) * ((t - ramp_up_s - hold_s) / tau)

    # Spatial gradients increase during ramp and relax during cool
    gradients_x = 8.0  # C across nx
    gradients_y = -5.0 # C across ny

    # Outputs
    tc_rows = []
    ir_rows = []  # flattened nx*ny

    time_s = 0.0
    for k in range(steps):
        base = schedule(time_s)
        hotspot = (int(0.6*nx), int(0.4*ny), 20.0 * math.sin(2*math.pi*(k/600.0)))
        grid = generate_ir_map(nx, ny, base, (gradients_x, gradients_y), hotspot)
        flat_grid = [v for row in grid for v in row]
        flat_grid = [clip(add_gaussian_noise(v, 1.5), 20, 1400) for v in flat_grid]
        ir_rows.append([round(time_s, 2)] + [round(v, 2) for v in flat_grid])

        tc_values = []
        for s in spec.thermocouples:
            # Map sensor x,y to grid
            gx_i = clip(int((s.location[0] / (nx-1)) * (nx-1)), 0, nx-1)
            gy_j = clip(int((s.location[1] / (ny-1)) * (ny-1)), 0, ny-1)
            measured = grid[int(gy_j)][int(gx_i)]
            measured = add_gaussian_noise(measured, 0.8)
            tc_values.append(round(clip(measured, 20, 1400), 2))
        tc_rows.append([round(time_s, 2)] + tc_values)

        time_s += dt

    # Write files
    ensure_dir(out_dir)
    ir_header = ["time_s"] + [f"T_{i}_{j}_C" for j in range(ny) for i in range(nx)]
    tc_header = ["time_s"] + [s.id for s in spec.thermocouples]

    write_csv_gz(os.path.join(out_dir, "sintering_ir_map.csv.gz"), ir_header, ir_rows)
    write_csv_gz(os.path.join(out_dir, "sintering_thermocouples.csv.gz"), tc_header, tc_rows)

    return {
        "phase": "sintering_cofiring",
        "schedule": {
            "ramp_up_s": ramp_up_s,
            "hold_s": hold_s,
            "ramp_down_s": ramp_down_s,
            "peak_C": 1350
        },
        "spatial_grid": {"nx": nx, "ny": ny},
        "thermocouples": [asdict(s) for s in spec.thermocouples],
        "sample_rate_hz": spec.sample_rate_hz,
        "notes": "Synthetic data intended for residual stress initialization."
    }


def synthesize_startup_shutdown(spec: ScenarioSpec, out_dir: str) -> Dict:
    random.seed(spec.seed)
    nx, ny = spec.spatial_grid_mm
    dt = 1.0 / spec.sample_rate_hz
    steps = int(spec.duration_s * spec.sample_rate_hz)

    # Startup + shutdown over total duration with asymmetric ramps
    startup_s = int(0.35 * spec.duration_s)
    steady_s = int(0.3 * spec.duration_s)
    shutdown_s = spec.duration_s - startup_s - steady_s

    def schedule(t: float) -> float:
        if t < startup_s:
            # nonlinear warm-up
            x = t / startup_s
            return 25 + (800 - 25) * (x**0.6)
        elif t < startup_s + steady_s:
            return 800 + 15 * math.sin(2 * math.pi * t / 300.0)
        else:
            # faster cool-down
            x = (t - startup_s - steady_s) / shutdown_s
            return 800 - (800 - 25) * (x**0.9)

    # Thermal cycling stressors
    gradients_x = 20.0
    gradients_y = 15.0

    tc_rows = []
    ir_rows = []

    time_s = 0.0
    for k in range(steps):
        base = schedule(time_s)
        hotspot = (int(0.25*nx), int(0.75*ny), 35.0 * math.sin(2*math.pi*(k/180.0)))
        grid = generate_ir_map(nx, ny, base, (gradients_x, gradients_y), hotspot)
        flat_grid = [clip(add_gaussian_noise(v, 1.0), 20, 900) for row in grid for v in row]
        ir_rows.append([round(time_s, 2)] + [round(v, 2) for v in flat_grid])

        tc_values = []
        for s in spec.thermocouples:
            gx_i = clip(int((s.location[0] / (nx-1)) * (nx-1)), 0, nx-1)
            gy_j = clip(int((s.location[1] / (ny-1)) * (ny-1)), 0, ny-1)
            measured = grid[int(gy_j)][int(gx_i)]
            measured = add_gaussian_noise(measured, 0.6)
            tc_values.append(round(clip(measured, 20, 900), 2))
        tc_rows.append([round(time_s, 2)] + tc_values)

        time_s += dt

    ensure_dir(out_dir)
    ir_header = ["time_s"] + [f"T_{i}_{j}_C" for j in range(ny) for i in range(nx)]
    tc_header = ["time_s"] + [s.id for s in spec.thermocouples]

    write_csv_gz(os.path.join(out_dir, "startup_shutdown_ir_map.csv.gz"), ir_header, ir_rows)
    write_csv_gz(os.path.join(out_dir, "startup_shutdown_thermocouples.csv.gz"), tc_header, tc_rows)

    return {
        "phase": "startup_shutdown",
        "schedule": {
            "startup_s": startup_s,
            "steady_s": steady_s,
            "shutdown_s": shutdown_s,
            "max_operating_C": 800
        },
        "spatial_grid": {"nx": nx, "ny": ny},
        "thermocouples": [asdict(s) for s in spec.thermocouples],
        "sample_rate_hz": spec.sample_rate_hz,
        "notes": "Synthetic data for thermal cycling and delamination analysis."
    }


def synthesize_steady_state(spec: ScenarioSpec, out_dir: str) -> Dict:
    random.seed(spec.seed)
    nx, ny = spec.spatial_grid_mm
    dt = 1.0 / spec.sample_rate_hz
    steps = int(spec.duration_s * spec.sample_rate_hz)

    base_center = 750.0
    slow_drift_amp = 5.0

    def schedule(t: float) -> float:
        return base_center + slow_drift_amp * math.sin(2 * math.pi * t / 1200.0)

    gradients_x = 12.0
    gradients_y = -8.0

    tc_rows = []
    ir_rows = []

    time_s = 0.0
    for k in range(steps):
        base = schedule(time_s)
        hotspot = (int(0.5*nx), int(0.5*ny), 12.0 * math.sin(2*math.pi*(k/240.0)))
        grid = generate_ir_map(nx, ny, base, (gradients_x, gradients_y), hotspot)
        flat_grid = [clip(add_gaussian_noise(v, 0.6), 20, 900) for row in grid for v in row]
        ir_rows.append([round(time_s, 2)] + [round(v, 2) for v in flat_grid])

        tc_values = []
        for s in spec.thermocouples:
            gx_i = clip(int((s.location[0] / (nx-1)) * (nx-1)), 0, nx-1)
            gy_j = clip(int((s.location[1] / (ny-1)) * (ny-1)), 0, ny-1)
            measured = grid[int(gy_j)][int(gx_i)]
            measured = add_gaussian_noise(measured, 0.4)
            tc_values.append(round(clip(measured, 20, 900), 2))
        tc_rows.append([round(time_s, 2)] + tc_values)

        time_s += dt

    ensure_dir(out_dir)
    ir_header = ["time_s"] + [f"T_{i}_{j}_C" for j in range(ny) for i in range(nx)]
    tc_header = ["time_s"] + [s.id for s in spec.thermocouples]

    write_csv_gz(os.path.join(out_dir, "steady_state_ir_map.csv.gz"), ir_header, ir_rows)
    write_csv_gz(os.path.join(out_dir, "steady_state_thermocouples.csv.gz"), tc_header, tc_rows)

    return {
        "phase": "steady_state_operation",
        "spatial_grid": {"nx": nx, "ny": ny},
        "thermocouples": [asdict(s) for s in spec.thermocouples],
        "sample_rate_hz": spec.sample_rate_hz,
        "base_center_C": base_center,
        "notes": "Synthetic data emphasizing spatial gradients during steady load."
    }


# -----------------------------
# Entrypoint
# -----------------------------

def main() -> None:
    out_root = os.environ.get("OUT_DIR", "/workspace/data/sofc_thermal")

    # Define sensors (example positions across a 100x100 mm cell)
    thermocouples = [
        Sensor(id="TC_corner", type="thermocouple", location=(0.0, 0.0, 0.0)),
        Sensor(id="TC_edge_mid", type="thermocouple", location=(50.0, 0.0, 0.0)),
        Sensor(id="TC_center", type="thermocouple", location=(50.0, 50.0, 0.0)),
        Sensor(id="TC_edge_side", type="thermocouple", location=(0.0, 50.0, 0.0)),
        Sensor(id="TC_far_corner", type="thermocouple", location=(100.0, 100.0, 0.0)),
    ]

    # Common grid of 64x64 pixels
    grid = (64, 64)

    metadata = {"scenarios": []}

    # 1) Sintering & co-firing
    sintering_spec = ScenarioSpec(
        name="sintering_cofiring",
        description="Ramp-hold-cool to 1350C for residual stress init",
        duration_s=3*3600,  # 3 hours
        sample_rate_hz=0.2,  # every 5 seconds
        spatial_grid_mm=grid,
        thermocouples=thermocouples,
        seed=1234,
    )
    sintering_dir = os.path.join(out_root, "sintering")
    md1 = synthesize_sintering(sintering_spec, sintering_dir)
    metadata["scenarios"].append(md1)

    # 2) Startup & shutdown cycles
    startup_spec = ScenarioSpec(
        name="startup_shutdown",
        description="Warm-up, steady hold at 800C, cool-down",
        duration_s=2*3600,  # 2 hours
        sample_rate_hz=0.5,  # every 2 seconds
        spatial_grid_mm=grid,
        thermocouples=thermocouples,
        seed=5678,
    )
    startup_dir = os.path.join(out_root, "startup_shutdown")
    md2 = synthesize_startup_shutdown(startup_spec, startup_dir)
    metadata["scenarios"].append(md2)

    # 3) Steady-state operation
    steady_spec = ScenarioSpec(
        name="steady_state",
        description="Stable operation with mild drift and gradients",
        duration_s=1*3600,  # 1 hour
        sample_rate_hz=1.0,  # every 1 second
        spatial_grid_mm=grid,
        thermocouples=thermocouples,
        seed=9012,
    )
    steady_dir = os.path.join(out_root, "steady_state")
    md3 = synthesize_steady_state(steady_spec, steady_dir)
    metadata["scenarios"].append(md3)

    # Write metadata
    ensure_dir(out_root)
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
