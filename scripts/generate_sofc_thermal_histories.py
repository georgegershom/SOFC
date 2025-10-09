#!/usr/bin/env python3
"""
SOFC Thermal History Data Generator

Generates synthetic but physically plausible spatiotemporal temperature fields
for a planar SOFC cell for the following scenarios:
  - Sintering & co-firing (furnace): to estimate initial residual stresses
  - Start-up and shut-down cycles: thermal cycling drives delamination risk
  - Steady-state operation: cross-cell gradients during operation

Outputs per scenario:
  - temperature_field.csv: columns [time_s, x_mm, y_mm, temperature_C]
  - thermocouple_traces.csv: columns [sensor_id, x_mm, y_mm, time_s, temperature_C]
  - metadata.json: generation parameters for traceability

No external dependencies required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


@dataclass
class Grid:
    width_mm: float
    height_mm: float
    num_x: int
    num_y: int

    def xs(self) -> List[float]:
        return linspace(0.0, self.width_mm, self.num_x)

    def ys(self) -> List[float]:
        return linspace(0.0, self.height_mm, self.num_y)


@dataclass
class TimeProfile:
    total_duration_s: float
    num_steps: int
    knot_times_s: List[float]
    knot_temps_c: List[float]


@dataclass
class NoiseSpec:
    imaging_sigma_c: float
    thermocouple_sigma_c: float
    rng_seed: int


@dataclass
class ScenarioMetadata:
    scenario: str
    grid: Dict[str, float]
    time: Dict[str, float]
    temperature_units: str
    spatial_model: Dict[str, float]
    base_profile_knots: List[Tuple[float, float]]
    noise: Dict[str, float]
    thermocouples: List[Dict[str, float]]
    notes: str
    generator_version: str = "1.0.0"


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num < 2:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def piecewise_linear_temperature(t: float, knots: Sequence[Tuple[float, float]]) -> float:
    if not knots:
        return 0.0
    if t <= knots[0][0]:
        return knots[0][1]
    if t >= knots[-1][0]:
        return knots[-1][1]
    # Find segment
    for i in range(1, len(knots)):
        t0, y0 = knots[i - 1]
        t1, y1 = knots[i]
        if t0 <= t <= t1:
            if t1 == t0:
                return y1
            u = (t - t0) / (t1 - t0)
            return y0 * (1.0 - u) + y1 * u
    # Fallback
    return knots[-1][1]


def radial_edge_factor(x: float, y: float, grid: Grid) -> float:
    """Normalized 0 at center, ~1 near edges using elliptical radius."""
    cx = grid.width_mm / 2.0
    cy = grid.height_mm / 2.0
    rx = max(cx, 1e-9)
    ry = max(cy, 1e-9)
    nx = (x - cx) / rx
    ny = (y - cy) / ry
    r = math.sqrt(nx * nx + ny * ny)
    return min(1.0, r)


def normalized_x(x: float, grid: Grid) -> float:
    return 0.0 if grid.width_mm == 0 else x / grid.width_mm


def normalized_y(y: float, grid: Grid) -> float:
    return 0.0 if grid.height_mm == 0 else y / grid.height_mm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def default_grid(num_x: int, num_y: int) -> Grid:
    return Grid(width_mm=100.0, height_mm=100.0, num_x=num_x, num_y=num_y)


def build_sintering_profile(num_steps: int) -> TimeProfile:
    # Timeline: 0-3h ramp to 1350C, 2h soak, 4h controlled cool to 100C
    t0 = 0
    t_ramp_end = 3 * 3600
    t_soak_end = t_ramp_end + 2 * 3600
    t_cool_end = t_soak_end + 4 * 3600
    knots = [
        (t0, 25.0),
        (t_ramp_end, 1350.0),
        (t_soak_end, 1350.0),
        (t_cool_end, 100.0),
    ]
    return TimeProfile(total_duration_s=t_cool_end, num_steps=num_steps, knot_times_s=[t for (t, _) in knots], knot_temps_c=[T for (_, T) in knots])


def build_startup_shutdown_profile(num_steps: int, cycles: int = 3) -> TimeProfile:
    # Each cycle: 1h ramp to 800C, 0.5h hold, 1h cool to 100C
    knots: List[Tuple[float, float]] = []
    t = 0.0
    temp_ambient = 25.0
    for _ in range(cycles):
        # ramp up
        knots.append((t, temp_ambient))
        t += 1 * 3600
        knots.append((t, 800.0))
        # hold
        t += 0.5 * 3600
        knots.append((t, 800.0))
        # cool down
        t += 1 * 3600
        knots.append((t, 100.0))
        # brief idle between cycles
        t += 10 * 60
        knots.append((t, 100.0))
    # Remove potential duplicate sequential times by consolidating
    filtered: List[Tuple[float, float]] = []
    for time_s, temp_c in knots:
        if not filtered or time_s != filtered[-1][0]:
            filtered.append((time_s, temp_c))
        else:
            filtered[-1] = (time_s, temp_c)
    return TimeProfile(total_duration_s=filtered[-1][0], num_steps=num_steps, knot_times_s=[t for (t, _) in filtered], knot_temps_c=[T for (_, T) in filtered])


def build_steady_state_profile(num_steps: int, duration_s: float = 3600.0) -> TimeProfile:
    # 1 hour with mild sinusoidal fluctuation around 800C
    return TimeProfile(
        total_duration_s=duration_s,
        num_steps=num_steps,
        knot_times_s=[0.0, duration_s],
        knot_temps_c=[800.0, 800.0],
    )


def temperature_base_at(t: float, profile: TimeProfile) -> float:
    knots = list(zip(profile.knot_times_s, profile.knot_temps_c))
    base = piecewise_linear_temperature(t, knots)
    return base


def generate_thermocouple_positions(grid: Grid, num_sensors: int) -> List[Tuple[str, float, float]]:
    # Corner, center, mid-edges, plus random interior if needed
    candidates: List[Tuple[str, float, float]] = [
        ("TC_CENTER", grid.width_mm / 2.0, grid.height_mm / 2.0),
        ("TC_NW", 0.0, grid.height_mm),
        ("TC_NE", grid.width_mm, grid.height_mm),
        ("TC_SW", 0.0, 0.0),
        ("TC_SE", grid.width_mm, 0.0),
        ("TC_MID_W", 0.0, grid.height_mm / 2.0),
        ("TC_MID_E", grid.width_mm, grid.height_mm / 2.0),
        ("TC_MID_N", grid.width_mm / 2.0, grid.height_mm),
        ("TC_MID_S", grid.width_mm / 2.0, 0.0),
    ]
    if num_sensors <= len(candidates):
        return candidates[:num_sensors]
    # Add reproducible random interior points for extra sensors
    rng = random.Random(1337)
    positions = candidates[:]
    for i in range(len(candidates), num_sensors):
        label = f"TC_{i+1:02d}"
        positions.append((label, rng.uniform(5.0, grid.width_mm - 5.0), rng.uniform(5.0, grid.height_mm - 5.0)))
    return positions


def generate_sintering(
    output_root: str,
    grid: Grid,
    profile: TimeProfile,
    noise: NoiseSpec,
) -> None:
    scenario_name = "sintering_cofiring"
    scenario_dir = os.path.join(output_root, scenario_name)
    ensure_dir(scenario_dir)

    rng = random.Random(noise.rng_seed + 101)

    times = linspace(0.0, profile.total_duration_s, profile.num_steps)

    # Spatial model parameters
    max_edge_drop_c = 60.0  # Max cooler edge vs center at peak

    # Prepare writers
    field_path = os.path.join(scenario_dir, "temperature_field.csv")
    tc_path = os.path.join(scenario_dir, "thermocouple_traces.csv")

    # Thermocouple positions
    tc_positions = generate_thermocouple_positions(grid, num_sensors=8)

    def imaging_rows() -> Iterable[Tuple[float, float, float, float]]:
        for t in times:
            base = temperature_base_at(t, profile)
            edge_drop = min(max_edge_drop_c, 0.05 * max(base, 0.0))
            xs = grid.xs()
            ys = grid.ys()
            for y in ys:
                for x in xs:
                    r = radial_edge_factor(x, y, grid)
                    spatial = -edge_drop * (r ** 2)
                    noisy = base + spatial + rng.gauss(0.0, noise.imaging_sigma_c)
                    yield t, x, y, noisy

    def tc_rows() -> Iterable[Tuple[str, float, float, float, float]]:
        for t in times:
            base = temperature_base_at(t, profile)
            edge_drop = min(max_edge_drop_c, 0.05 * max(base, 0.0))
            for label, x, y in tc_positions:
                r = radial_edge_factor(x, y, grid)
                spatial = -edge_drop * (r ** 2)
                noisy = base + spatial + rng.gauss(0.0, noise.thermocouple_sigma_c)
                yield label, x, y, t, noisy

    write_csv(field_path, ("time_s", "x_mm", "y_mm", "temperature_C"), imaging_rows())
    write_csv(tc_path, ("sensor_id", "x_mm", "y_mm", "time_s", "temperature_C"), tc_rows())

    metadata = ScenarioMetadata(
        scenario=scenario_name,
        grid={"width_mm": grid.width_mm, "height_mm": grid.height_mm, "num_x": grid.num_x, "num_y": grid.num_y},
        time={"total_duration_s": profile.total_duration_s, "num_steps": profile.num_steps},
        temperature_units="C",
        spatial_model={"type": "center-hot furnace, cooler edges", "max_edge_drop_c": max_edge_drop_c},
        base_profile_knots=list(zip(profile.knot_times_s, profile.knot_temps_c)),
        noise={"imaging_sigma_c": noise.imaging_sigma_c, "thermocouple_sigma_c": noise.thermocouple_sigma_c, "rng_seed": noise.rng_seed + 101},
        thermocouples=[{"sensor_id": label, "x_mm": x, "y_mm": y} for (label, x, y) in tc_positions],
        notes="Synthetic furnace profile with soak; edges cooler than center due to radiative loss.",
    )
    write_json(os.path.join(scenario_dir, "metadata.json"), asdict(metadata))


def generate_startup_shutdown(
    output_root: str,
    grid: Grid,
    profile: TimeProfile,
    noise: NoiseSpec,
) -> None:
    scenario_name = "startup_shutdown_cycles"
    scenario_dir = os.path.join(output_root, scenario_name)
    ensure_dir(scenario_dir)

    rng = random.Random(noise.rng_seed + 202)

    times = linspace(0.0, profile.total_duration_s, profile.num_steps)

    # Spatial model parameters
    max_longitudinal_grad_c = 80.0  # peak delta across x during hot hold
    edge_drop_c = 30.0  # across y edges regardless of t

    field_path = os.path.join(scenario_dir, "temperature_field.csv")
    tc_path = os.path.join(scenario_dir, "thermocouple_traces.csv")

    tc_positions = generate_thermocouple_positions(grid, num_sensors=10)

    def imaging_rows() -> Iterable[Tuple[float, float, float, float]]:
        for t in times:
            base = temperature_base_at(t, profile)
            # Scale gradients with temperature above ambient
            hot_factor = max(0.0, (base - 100.0) / 700.0)  # 0 to ~1 between 100C and 800C
            xs = grid.xs()
            ys = grid.ys()
            for y in ys:
                for x in xs:
                    xn = normalized_x(x, grid)  # 0..1
                    # Longitudinal gradient centered to zero-mean
                    grad_x = (2.0 * xn - 1.0) * (max_longitudinal_grad_c * hot_factor)
                    # Edge cooling across y via radial factor but biased to y-edges
                    r = radial_edge_factor(x, y, grid)
                    edge_cool = -edge_drop_c * hot_factor * (r ** 2)
                    noisy = base + grad_x + edge_cool + rng.gauss(0.0, noise.imaging_sigma_c)
                    yield t, x, y, noisy

    def tc_rows() -> Iterable[Tuple[str, float, float, float, float]]:
        for t in times:
            base = temperature_base_at(t, profile)
            hot_factor = max(0.0, (base - 100.0) / 700.0)
            for label, x, y in tc_positions:
                xn = normalized_x(x, grid)
                grad_x = (2.0 * xn - 1.0) * (max_longitudinal_grad_c * hot_factor)
                r = radial_edge_factor(x, y, grid)
                edge_cool = -edge_drop_c * hot_factor * (r ** 2)
                noisy = base + grad_x + edge_cool + rng.gauss(0.0, noise.thermocouple_sigma_c)
                yield label, x, y, t, noisy

    write_csv(field_path, ("time_s", "x_mm", "y_mm", "temperature_C"), imaging_rows())
    write_csv(tc_path, ("sensor_id", "x_mm", "y_mm", "time_s", "temperature_C"), tc_rows())

    metadata = ScenarioMetadata(
        scenario=scenario_name,
        grid={"width_mm": grid.width_mm, "height_mm": grid.height_mm, "num_x": grid.num_x, "num_y": grid.num_y},
        time={"total_duration_s": profile.total_duration_s, "num_steps": profile.num_steps},
        temperature_units="C",
        spatial_model={
            "type": "longitudinal flow-induced gradient with y-edge cooling",
            "max_longitudinal_grad_c": max_longitudinal_grad_c,
            "edge_drop_c": edge_drop_c,
        },
        base_profile_knots=list(zip(profile.knot_times_s, profile.knot_temps_c)),
        noise={"imaging_sigma_c": noise.imaging_sigma_c, "thermocouple_sigma_c": noise.thermocouple_sigma_c, "rng_seed": noise.rng_seed + 202},
        thermocouples=[{"sensor_id": label, "x_mm": x, "y_mm": y} for (label, x, y) in tc_positions],
        notes="Startup/shutdown cycles with strong x-gradient at hot hold; edge cooling.",
    )
    write_json(os.path.join(scenario_dir, "metadata.json"), asdict(metadata))


def generate_steady_state(
    output_root: str,
    grid: Grid,
    profile: TimeProfile,
    noise: NoiseSpec,
) -> None:
    scenario_name = "steady_state_operation"
    scenario_dir = os.path.join(output_root, scenario_name)
    ensure_dir(scenario_dir)

    rng = random.Random(noise.rng_seed + 303)

    times = linspace(0.0, profile.total_duration_s, profile.num_steps)

    # Spatial model parameters
    longitudinal_grad_c = 100.0  # persistent delta across x at steady load
    edge_drop_c = 25.0
    fluctuation_amp_c = 10.0
    fluctuation_period_s = 15.0 * 60.0

    field_path = os.path.join(scenario_dir, "temperature_field.csv")
    tc_path = os.path.join(scenario_dir, "thermocouple_traces.csv")

    tc_positions = generate_thermocouple_positions(grid, num_sensors=8)

    def imaging_rows() -> Iterable[Tuple[float, float, float, float]]:
        for t in times:
            base = temperature_base_at(t, profile)
            fluct = fluctuation_amp_c * math.sin(2.0 * math.pi * t / fluctuation_period_s)
            xs = grid.xs()
            ys = grid.ys()
            for y in ys:
                for x in xs:
                    xn = normalized_x(x, grid)
                    grad_x = (2.0 * xn - 1.0) * longitudinal_grad_c
                    r = radial_edge_factor(x, y, grid)
                    edge_cool = -edge_drop_c * (r ** 2)
                    noisy = base + fluct + grad_x + edge_cool + rng.gauss(0.0, noise.imaging_sigma_c)
                    yield t, x, y, noisy

    def tc_rows() -> Iterable[Tuple[str, float, float, float, float]]:
        for t in times:
            base = temperature_base_at(t, profile)
            fluct = fluctuation_amp_c * math.sin(2.0 * math.pi * t / fluctuation_period_s)
            for label, x, y in tc_positions:
                xn = normalized_x(x, grid)
                grad_x = (2.0 * xn - 1.0) * longitudinal_grad_c
                r = radial_edge_factor(x, y, grid)
                edge_cool = -edge_drop_c * (r ** 2)
                noisy = base + fluct + grad_x + edge_cool + rng.gauss(0.0, noise.thermocouple_sigma_c)
                yield label, x, y, t, noisy

    write_csv(field_path, ("time_s", "x_mm", "y_mm", "temperature_C"), imaging_rows())
    write_csv(tc_path, ("sensor_id", "x_mm", "y_mm", "time_s", "temperature_C"), tc_rows())

    metadata = ScenarioMetadata(
        scenario=scenario_name,
        grid={"width_mm": grid.width_mm, "height_mm": grid.height_mm, "num_x": grid.num_x, "num_y": grid.num_y},
        time={"total_duration_s": profile.total_duration_s, "num_steps": profile.num_steps},
        temperature_units="C",
        spatial_model={
            "type": "persistent longitudinal gradient + edge cooling",
            "longitudinal_grad_c": longitudinal_grad_c,
            "edge_drop_c": edge_drop_c,
            "fluctuation_amp_c": fluctuation_amp_c,
            "fluctuation_period_s": fluctuation_period_s,
        },
        base_profile_knots=list(zip(profile.knot_times_s, profile.knot_temps_c)),
        noise={"imaging_sigma_c": noise.imaging_sigma_c, "thermocouple_sigma_c": noise.thermocouple_sigma_c, "rng_seed": noise.rng_seed + 303},
        thermocouples=[{"sensor_id": label, "x_mm": x, "y_mm": y} for (label, x, y) in tc_positions],
        notes="Steady operation near 800C with mild load-induced oscillation.",
    )
    write_json(os.path.join(scenario_dir, "metadata.json"), asdict(metadata))


def run_scenario(
    scenario: str,
    output_root: str,
    grid: Grid,
    noise: NoiseSpec,
    # Per-scenario knobs
    sintering_steps: int,
    startup_steps: int,
    steady_steps: int,
) -> None:
    if scenario == "sintering":
        profile = build_sintering_profile(sintering_steps)
        generate_sintering(output_root, grid, profile, noise)
    elif scenario == "startup_shutdown":
        profile = build_startup_shutdown_profile(startup_steps)
        generate_startup_shutdown(output_root, grid, profile, noise)
    elif scenario == "steady_state":
        profile = build_steady_state_profile(steady_steps)
        generate_steady_state(output_root, grid, profile, noise)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate SOFC thermal history datasets")
    parser.add_argument("--output-root", default=os.path.join(os.getcwd(), "data"), help="Output root directory")
    parser.add_argument("--grid-x", type=int, default=31, help="Number of grid points in x (width)")
    parser.add_argument("--grid-y", type=int, default=31, help="Number of grid points in y (height)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base")
    parser.add_argument("--imaging-noise", type=float, default=2.0, help="Imaging Gaussian noise sigma [C]")
    parser.add_argument("--tc-noise", type=float, default=0.5, help="Thermocouple Gaussian noise sigma [C]")

    parser.add_argument("--sintering-steps", type=int, default=120, help="Number of time steps for sintering")
    parser.add_argument("--startup-steps", type=int, default=150, help="Number of time steps for startup/shutdown")
    parser.add_argument("--steady-steps", type=int, default=60, help="Number of time steps for steady-state")

    parser.add_argument(
        "--scenario",
        choices=["sintering", "startup_shutdown", "steady_state", "all"],
        default="all",
        help="Which scenario to generate",
    )

    args = parser.parse_args(list(argv))

    grid = default_grid(args.grid_x, args.grid_y)
    noise = NoiseSpec(imaging_sigma_c=args.imaging_noise, thermocouple_sigma_c=args.tc_noise, rng_seed=args.seed)

    ensure_dir(args.output_root)

    if args.scenario == "all":
        run_scenario("sintering", args.output_root, grid, noise, args.sintering_steps, args.startup_steps, args.steady_steps)
        run_scenario("startup_shutdown", args.output_root, grid, noise, args.sintering_steps, args.startup_steps, args.steady_steps)
        run_scenario("steady_state", args.output_root, grid, noise, args.sintering_steps, args.startup_steps, args.steady_steps)
    else:
        run_scenario(args.scenario, args.output_root, grid, noise, args.sintering_steps, args.startup_steps, args.steady_steps)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
