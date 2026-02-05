#!/usr/bin/env python3
"""
Generate a synthetic core time-series pressure dataset.

This script fabricates high-frequency acoustic pressure data for
multiple test groups, creates metadata, figures, and a ZIP archive
containing all CSV files.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import zipfile

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1] / "core_time_series_pressure_data"
HF_DIR = BASE_DIR / "hf_acoustic"
FIG_DIR = BASE_DIR / "figures"
META_DIR = BASE_DIR / "metadata"
ARCHIVE_DIR = BASE_DIR / "csv_archives"


@dataclass(frozen=True)
class GroupConfig:
    group_id: str
    sample_rate_hz: int
    leak_location: str
    sensor_config: str
    stratification_level: str
    stratification_factor: float
    leak_amp: float
    alpha: float


@dataclass(frozen=True)
class Sensor:
    sensor_id: str
    position_m: float
    bank: str
    near_valve: str
    near_leak: str


LEAK_LOCATIONS: Dict[str, float] = {
    "A": 45.0,
    "D": 75.0,
    "E": 115.0,
}

VALVES: Dict[str, float] = {
    "V1": 30.0,
    "V2": 90.0,
}

SENSORS: List[Sensor] = [
    Sensor("PG1", 0.0, "A", "", ""),
    Sensor("PG2", 10.0, "A", "", ""),
    Sensor("PG3", 20.0, "A", "V1_before", ""),
    Sensor("PG4", 30.0, "A", "V1_after", ""),
    Sensor("PG5", 40.0, "B", "", "A_before"),
    Sensor("PG6", 50.0, "B", "", "A_after"),
    Sensor("PG7", 60.0, "B", "", ""),
    Sensor("PG8", 70.0, "B", "", "D_before"),
    Sensor("PG9", 80.0, "C", "V2_before", "D_after"),
    Sensor("PG10", 90.0, "C", "V2_after", ""),
    Sensor("PG11", 100.0, "C", "", ""),
    Sensor("PG12", 110.0, "D", "", "E_before"),
    Sensor("PG13", 120.0, "D", "", "E_after"),
    Sensor("PG14", 130.0, "D", "", ""),
]

GROUPS: List[GroupConfig] = [
    GroupConfig(
        group_id="01",
        sample_rate_hz=17060,
        leak_location="A",
        sensor_config="A,B",
        stratification_level="low",
        stratification_factor=0.10,
        leak_amp=1.25,
        alpha=0.020,
    ),
    GroupConfig(
        group_id="02",
        sample_rate_hz=10000,
        leak_location="D",
        sensor_config="A,B,C,D",
        stratification_level="medium",
        stratification_factor=0.18,
        leak_amp=1.15,
        alpha=0.025,
    ),
    GroupConfig(
        group_id="03",
        sample_rate_hz=10000,
        leak_location="E",
        sensor_config="A,B",
        stratification_level="high",
        stratification_factor=0.26,
        leak_amp=1.05,
        alpha=0.030,
    ),
    GroupConfig(
        group_id="04",
        sample_rate_hz=10000,
        leak_location="A",
        sensor_config="A,B,C,D",
        stratification_level="medium-high",
        stratification_factor=0.22,
        leak_amp=1.20,
        alpha=0.027,
    ),
]

CONDITIONS: List[Tuple[str, int, float]] = [
    ("baseline", 30, -30.0),
    ("transient", 30, 0.0),
    ("post", 10, 30.0),
]

EVENT_OPEN_SEC = 5.0
EVENT_CLOSE_SEC = 20.0


def ensure_dirs() -> None:
    for path in (HF_DIR, FIG_DIR, META_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        group_dir = HF_DIR / f"group{group.group_id}"
        for condition, _, _ in CONDITIONS:
            (group_dir / condition).mkdir(parents=True, exist_ok=True)


def write_metadata() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)

    with (META_DIR / "sensors.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sensor_id", "position_m", "bank", "near_valve", "near_leak"]
        )
        for sensor in SENSORS:
            writer.writerow(
                [
                    sensor.sensor_id,
                    f"{sensor.position_m:.1f}",
                    sensor.bank,
                    sensor.near_valve,
                    sensor.near_leak,
                ]
            )

    with (META_DIR / "leak_locations.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["leak_location", "position_m"])
        for name, pos in LEAK_LOCATIONS.items():
            writer.writerow([name, f"{pos:.1f}"])

    with (META_DIR / "valves.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["valve_id", "position_m"])
        for name, pos in VALVES.items():
            writer.writerow([name, f"{pos:.1f}"])

    with (META_DIR / "groups.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_id",
                "sample_rate_hz",
                "leak_location",
                "sensor_configuration",
                "stratification_level",
                "stratification_factor",
            ]
        )
        for group in GROUPS:
            writer.writerow(
                [
                    group.group_id,
                    group.sample_rate_hz,
                    group.leak_location,
                    group.sensor_config,
                    group.stratification_level,
                    f"{group.stratification_factor:.2f}",
                ]
            )

    with (META_DIR / "events.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "baseline_duration_sec",
                "transient_duration_sec",
                "post_duration_sec",
                "leak_open_sec",
                "leak_close_sec",
            ]
        )
        writer.writerow(
            [
                CONDITIONS[0][1],
                CONDITIONS[1][1],
                CONDITIONS[2][1],
                EVENT_OPEN_SEC,
                EVENT_CLOSE_SEC,
            ]
        )


def leak_envelope(t_abs: np.ndarray) -> np.ndarray:
    ramp_up = np.clip((t_abs - EVENT_OPEN_SEC) / 0.5, 0.0, 1.0)
    ramp_down = np.clip((EVENT_CLOSE_SEC - t_abs) / 0.5, 0.0, 1.0)
    return ramp_up * ramp_down


def ringdown_envelope(t_abs: np.ndarray) -> np.ndarray:
    tau = 3.0
    return np.where(
        t_abs >= EVENT_CLOSE_SEC, np.exp(-(t_abs - EVENT_CLOSE_SEC) / tau), 0.0
    )


def sensor_signal(
    t_abs: np.ndarray,
    sensor: Sensor,
    group: GroupConfig,
    leak_pos: float,
) -> np.ndarray:
    wave_speed = 1400.0
    base_speed = 1000.0
    distance = abs(sensor.position_m - leak_pos)
    max_distance = 130.0

    attenuation = math.exp(-group.alpha * distance)
    attenuation *= max(0.2, 1.0 - group.stratification_factor * (distance / max_distance))

    leak_delay = distance / wave_speed
    base_delay = sensor.position_m / base_speed

    t_leak = t_abs - leak_delay
    t_base = t_abs - base_delay

    base_amp = 0.35 * (1.0 - 0.002 * sensor.position_m)
    base_signal = base_amp * (
        np.sin(2 * math.pi * 60.0 * t_base + 0.2)
        + 0.3 * np.sin(2 * math.pi * 120.0 * t_base + 1.1)
    )

    strat_amp = 0.12 * group.stratification_factor * (1.0 + 0.003 * sensor.position_m)
    strat_signal = strat_amp * np.sin(2 * math.pi * 24.0 * t_abs + 0.8)

    noise_amp = 0.04 * (1.0 + 0.001 * sensor.position_m)
    noise_signal = noise_amp * (
        np.sin(2 * math.pi * 1513.0 * t_abs + 0.5)
        + 0.5 * np.sin(2 * math.pi * 1789.0 * t_abs + 1.3)
    )

    leak_wave = (
        np.sin(2 * math.pi * 820.0 * t_leak + 0.1)
        + 0.55 * np.sin(2 * math.pi * 1180.0 * t_leak + 2.2)
    )
    leak_signal = group.leak_amp * attenuation * leak_wave

    ring_wave = np.sin(2 * math.pi * 700.0 * t_leak + 1.7)
    ring_signal = 0.35 * group.leak_amp * attenuation * ring_wave

    envelope = leak_envelope(t_abs)
    ringdown = ringdown_envelope(t_abs)

    return base_signal + strat_signal + noise_signal + leak_signal * envelope + ring_signal * ringdown


def write_second_csv(
    path: Path,
    sample_rate_hz: int,
    t_offset: float,
    second_index: int,
    group: GroupConfig,
    leak_pos: float,
) -> np.ndarray:
    n = sample_rate_hz
    t_abs = t_offset + second_index + (np.arange(n) / sample_rate_hz)
    sample_index = np.arange(n, dtype=int)
    values = []
    for sensor in SENSORS:
        signal = sensor_signal(t_abs, sensor, group, leak_pos)
        values.append(np.round(signal, 4))
    data = np.column_stack([sample_index] + values)

    header = ",".join(["sample_index"] + [sensor.sensor_id for sensor in SENSORS])
    fmt = ["%d"] + ["%.4f"] * len(SENSORS)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmt)
    return values


def generate_dataset() -> Dict[str, Dict[str, np.ndarray]]:
    summary_rms: Dict[str, Dict[str, np.ndarray]] = {}
    for group in GROUPS:
        leak_pos = LEAK_LOCATIONS[group.leak_location]
        group_dir = HF_DIR / f"group{group.group_id}"
        summary_rms[group.group_id] = {}

        for condition, seconds, t_offset in CONDITIONS:
            condition_dir = group_dir / condition
            for sec in range(seconds):
                filename = condition_dir / f"sec_{sec:03d}.csv"
                sensor_values = write_second_csv(
                    filename,
                    group.sample_rate_hz,
                    t_offset,
                    sec,
                    group,
                    leak_pos,
                )

                if condition == "transient" and sec == 10:
                    rms_values = []
                    for values in sensor_values:
                        rms_values.append(math.sqrt(float(np.mean(values**2))))
                    summary_rms[group.group_id]["rms"] = np.array(rms_values)
    return summary_rms


def plot_timeseries(group_id: str, sensor_id: str) -> None:
    group = next(g for g in GROUPS if g.group_id == group_id)
    sensor_index = [s.sensor_id for s in SENSORS].index(sensor_id)

    sample_rate = group.sample_rate_hz
    t_series: List[float] = []
    values: List[float] = []
    downsample = max(1, sample_rate // 200)

    group_dir = HF_DIR / f"group{group_id}" / "transient"
    for sec in range(CONDITIONS[1][1]):
        file_path = group_dir / f"sec_{sec:03d}.csv"
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=range(0, 15))
        sample_index = data[:, 0]
        sensor_values = data[:, sensor_index + 1]
        sample_index = sample_index[::downsample]
        sensor_values = sensor_values[::downsample]
        t = sec + (sample_index / sample_rate)
        t_series.extend(t.tolist())
        values.extend(sensor_values.tolist())

    plt.figure(figsize=(10, 4))
    plt.plot(t_series, values, linewidth=0.8)
    plt.axvspan(EVENT_OPEN_SEC, EVENT_CLOSE_SEC, color="orange", alpha=0.15, label="Leak open")
    plt.title(f"Group {group_id} Transient Pressure (Sensor {sensor_id})")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"group{group_id}_timeseries_{sensor_id}.png", dpi=180)
    plt.close()


def plot_attenuation(summary_rms: Dict[str, Dict[str, np.ndarray]]) -> None:
    positions = np.array([sensor.position_m for sensor in SENSORS])
    plt.figure(figsize=(10, 5))

    for group in GROUPS:
        rms = summary_rms[group.group_id]["rms"]
        plt.plot(positions, rms, marker="o", linewidth=1.0, label=f"Group {group.group_id}")

    plt.title("Leak-Period RMS vs Sensor Position")
    plt.xlabel("Sensor position (m)")
    plt.ylabel("RMS Pressure (Pa)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attenuation_summary.png", dpi=180)
    plt.close()


def zip_csv_files() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    metadata_zip = ARCHIVE_DIR / "metadata_csv.zip"
    with zipfile.ZipFile(metadata_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in META_DIR.glob("*.csv"):
            arcname = file_path.relative_to(BASE_DIR)
            zipf.write(file_path, arcname.as_posix())

    for group in GROUPS:
        group_zip = ARCHIVE_DIR / f"hf_acoustic_group{group.group_id}.zip"
        group_dir = HF_DIR / f"group{group.group_id}"
        csv_files = list(group_dir.rglob("*.csv"))
        with zipfile.ZipFile(group_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for file_path in csv_files:
                arcname = file_path.relative_to(BASE_DIR)
                zipf.write(file_path, arcname.as_posix())


def main() -> None:
    ensure_dirs()
    write_metadata()
    summary_rms = generate_dataset()
    for group in GROUPS:
        plot_timeseries(group.group_id, "PG6")
    plot_attenuation(summary_rms)
    zip_csv_files()


if __name__ == "__main__":
    main()
