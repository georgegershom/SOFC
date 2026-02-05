#!/usr/bin/env python3
"""
Generate fabricated HF acoustic pressure dataset for leakage analysis.

Outputs:
  - data/csv/<group>/baseline/sec_XXX.csv
  - data/csv/<group>/event/sec_XXX.csv
  - data/metadata/*.csv
  - figures/*.png
  - data/hf_pressure_csv.zip
"""

from __future__ import annotations

import csv
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
META_DIR = DATA_DIR / "metadata"
FIG_DIR = ROOT / "figures"

BASELINE_SECONDS = 30
EVENT_SECONDS = 30
TOTAL_SECONDS = BASELINE_SECONDS + EVENT_SECONDS
LEAK_OPEN_S = 5.0
LEAK_CLOSE_S = 20.0
SPEED_OF_SOUND = 1480.0

SENSORS = [
    "PG1", "PG2", "PG3", "PG4", "PG5", "PG6", "PG7",
    "PG8", "PG9", "PG10", "PG11", "PG12", "PG13", "PG14",
]

SENSOR_POSITIONS_M = {
    "PG1": 0.0,
    "PG2": 1.0,
    "PG3": 2.0,
    "PG4": 3.0,
    "PG5": 4.0,
    "PG6": 5.0,
    "PG7": 6.5,
    "PG8": 8.0,
    "PG9": 9.0,
    "PG10": 10.0,
    "PG11": 11.0,
    "PG12": 12.0,
    "PG13": 13.0,
    "PG14": 14.0,
}

SENSOR_LAYER = {
    "PG1": "upper",
    "PG2": "upper",
    "PG3": "upper",
    "PG4": "upper",
    "PG5": "middle",
    "PG6": "middle",
    "PG7": "middle",
    "PG8": "middle",
    "PG9": "middle",
    "PG10": "middle",
    "PG11": "lower",
    "PG12": "lower",
    "PG13": "lower",
    "PG14": "lower",
}

LAYER_FACTOR = {
    "upper": 1.00,
    "middle": 0.92,
    "lower": 0.85,
}

LEAK_POSITIONS_M = {
    "A": 2.5,
    "D": 8.5,
    "E": 11.5,
}

GROUPS = [
    {"group_id": "group01", "sampling_rate_hz": 17060, "leak_location": "A", "sensor_config": "A,B"},
    {"group_id": "group02", "sampling_rate_hz": 10000, "leak_location": "D", "sensor_config": "A,B,C,D"},
    {"group_id": "group03", "sampling_rate_hz": 10000, "leak_location": "E", "sensor_config": "A,B"},
    {"group_id": "group04", "sampling_rate_hz": 10000, "leak_location": "A", "sensor_config": "A,B,C,D"},
]


def ensure_dirs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def leak_envelope(t: np.ndarray) -> np.ndarray:
    rise = 0.15
    fall = 0.15
    open_arg = np.clip((t - LEAK_OPEN_S) / rise, -60.0, 60.0)
    close_arg = np.clip((t - LEAK_CLOSE_S) / fall, -60.0, 60.0)
    open_step = 1.0 / (1.0 + np.exp(-open_arg))
    close_step = 1.0 / (1.0 + np.exp(-close_arg))
    env = open_step * (1.0 - close_step)
    return env.astype(np.float32)


def compute_attenuation(distance_m: float, layer: str) -> float:
    alpha = 0.12
    beta = 0.08
    layer_factor = LAYER_FACTOR[layer]
    return layer_factor * math.exp(-beta * distance_m) / (1.0 + alpha * distance_m)


def write_csv(path: Path, header: list[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.6f",
    )


def generate_group_signals(group: dict, seed: int) -> dict:
    fs = int(group["sampling_rate_hz"])
    total_samples = TOTAL_SECONDS * fs
    t = (np.arange(total_samples, dtype=np.float32) / fs) - BASELINE_SECONDS

    rng = np.random.default_rng(seed)
    base_background = rng.normal(0.0, 0.02, size=total_samples).astype(np.float32)
    pump_tone = (
        0.012 * np.sin(2.0 * np.pi * 40.0 * t)
        + 0.006 * np.sin(2.0 * np.pi * 120.0 * t)
    ).astype(np.float32)

    env = leak_envelope(t)
    broadband = rng.normal(0.0, 1.0, size=total_samples).astype(np.float32)
    hf_tone = (
        0.25 * np.sin(2.0 * np.pi * 1500.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 2300.0 * t)
    ).astype(np.float32)
    leak_base = env * (0.25 * broadband + hf_tone)

    signals = np.zeros((total_samples, len(SENSORS)), dtype=np.float32)
    leak_location = group["leak_location"]
    leak_position = LEAK_POSITIONS_M[leak_location]

    for idx, sensor in enumerate(SENSORS):
        distance = abs(SENSOR_POSITIONS_M[sensor] - leak_position)
        delay_s = distance / SPEED_OF_SOUND
        delay_samples = int(round(delay_s * fs))

        shifted = np.zeros_like(leak_base)
        if delay_samples > 0:
            shifted[delay_samples:] = leak_base[:-delay_samples]
        else:
            shifted = leak_base.copy()

        attenuation = compute_attenuation(distance, SENSOR_LAYER[sensor])
        sensor_noise = rng.normal(0.0, 0.006, size=total_samples).astype(np.float32)
        signals[:, idx] = base_background + pump_tone + attenuation * shifted + sensor_noise

    return {"fs": fs, "t": t, "signals": signals}


def save_group_csv(group: dict, t: np.ndarray, signals: np.ndarray) -> None:
    fs = int(group["sampling_rate_hz"])
    group_dir = CSV_DIR / group["group_id"]

    header = ["time_s"] + SENSORS

    for sec in range(BASELINE_SECONDS):
        start = sec * fs
        end = start + fs
        data = np.column_stack((t[start:end], signals[start:end]))
        path = group_dir / "baseline" / f"sec_{sec:03d}.csv"
        write_csv(path, header, data)

    for sec in range(EVENT_SECONDS):
        start = (BASELINE_SECONDS + sec) * fs
        end = start + fs
        data = np.column_stack((t[start:end], signals[start:end]))
        path = group_dir / "event" / f"sec_{sec:03d}.csv"
        write_csv(path, header, data)


def plot_group_figures(group: dict, t: np.ndarray, signals: np.ndarray) -> list[Path]:
    fs = int(group["sampling_rate_hz"])
    group_id = group["group_id"]

    event_start = BASELINE_SECONDS * fs
    event_end = event_start + EVENT_SECONDS * fs
    t_event = t[event_start:event_end]
    pg1_event = signals[event_start:event_end, 0]

    fig_paths: list[Path] = []

    plt.figure(figsize=(10, 4))
    plt.plot(t_event, pg1_event, linewidth=0.7)
    plt.axvline(LEAK_OPEN_S, color="red", linestyle="--", linewidth=1.0, label="Valve open")
    plt.axvline(LEAK_CLOSE_S, color="black", linestyle="--", linewidth=1.0, label="Valve close")
    plt.title(f"{group_id} PG1 Event Window")
    plt.xlabel("Time (s, relative to event start)")
    plt.ylabel("Pressure (Pa)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path_ts = FIG_DIR / f"{group_id}_pg1_timeseries.png"
    plt.savefig(path_ts, dpi=160)
    plt.close()
    fig_paths.append(path_ts)

    leak_mask = (t_event >= LEAK_OPEN_S + 1.0) & (t_event <= LEAK_CLOSE_S - 1.0)
    distances = []
    rms_values = []
    for idx, sensor in enumerate(SENSORS):
        distance = abs(SENSOR_POSITIONS_M[sensor] - LEAK_POSITIONS_M[group["leak_location"]])
        segment = signals[event_start:event_end, idx][leak_mask]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        distances.append(distance)
        rms_values.append(rms)

    plt.figure(figsize=(6, 4))
    plt.scatter(distances, rms_values, color="navy")
    plt.title(f"{group_id} Leak RMS vs Distance")
    plt.xlabel("Distance from leak (m)")
    plt.ylabel("RMS Pressure (Pa)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path_att = FIG_DIR / f"{group_id}_attenuation.png"
    plt.savefig(path_att, dpi=160)
    plt.close()
    fig_paths.append(path_att)

    return fig_paths


def write_metadata(rms_records: list[dict]) -> None:
    sensors_path = META_DIR / "sensors.csv"
    with sensors_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "position_m", "layer"])
        for sensor in SENSORS:
            writer.writerow([sensor, f"{SENSOR_POSITIONS_M[sensor]:.2f}", SENSOR_LAYER[sensor]])

    groups_path = META_DIR / "groups.csv"
    with groups_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "sampling_rate_hz", "leak_location", "sensor_config"])
        for group in GROUPS:
            writer.writerow([
                group["group_id"],
                group["sampling_rate_hz"],
                group["leak_location"],
                group["sensor_config"],
            ])

    leak_positions_path = META_DIR / "leak_positions.csv"
    with leak_positions_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["leak_location", "position_m"])
        for loc, pos in LEAK_POSITIONS_M.items():
            writer.writerow([loc, f"{pos:.2f}"])

    rms_path = META_DIR / "leak_rms_summary.csv"
    with rms_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "sensor", "distance_m", "rms_pressure_pa"])
        for record in rms_records:
            writer.writerow([
                record["group_id"],
                record["sensor"],
                f"{record['distance_m']:.3f}",
                f"{record['rms_pressure_pa']:.6f}",
            ])


def write_manifest() -> None:
    manifest_path = DATA_DIR / "dataset_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "bytes"])
        for path in sorted(DATA_DIR.rglob("*.csv")):
            rel = path.relative_to(ROOT)
            writer.writerow([str(rel), path.stat().st_size])


def build_zip_files() -> list[Path]:
    zip_paths: list[Path] = []

    for group in GROUPS:
        group_id = group["group_id"]
        zip_path = DATA_DIR / f"hf_pressure_csv_{group_id}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted((CSV_DIR / group_id).rglob("*.csv")):
                zf.write(path, path.relative_to(ROOT))
        zip_paths.append(zip_path)

    meta_zip = DATA_DIR / "hf_pressure_csv_metadata.zip"
    if meta_zip.exists():
        meta_zip.unlink()
    with zipfile.ZipFile(meta_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(META_DIR.rglob("*.csv")):
            zf.write(path, path.relative_to(ROOT))
        manifest = DATA_DIR / "dataset_manifest.csv"
        if manifest.exists():
            zf.write(manifest, manifest.relative_to(ROOT))
    zip_paths.append(meta_zip)
    return zip_paths


def main() -> None:
    ensure_dirs()
    rms_records: list[dict] = []
    fig_paths: list[Path] = []

    for idx, group in enumerate(GROUPS):
        result = generate_group_signals(group, seed=1000 + idx)
        save_group_csv(group, result["t"], result["signals"])
        fig_paths.extend(plot_group_figures(group, result["t"], result["signals"]))

        fs = result["fs"]
        event_start = BASELINE_SECONDS * fs
        event_end = event_start + EVENT_SECONDS * fs
        t_event = result["t"][event_start:event_end]
        leak_mask = (t_event >= LEAK_OPEN_S + 1.0) & (t_event <= LEAK_CLOSE_S - 1.0)
        for s_idx, sensor in enumerate(SENSORS):
            distance = abs(SENSOR_POSITIONS_M[sensor] - LEAK_POSITIONS_M[group["leak_location"]])
            segment = result["signals"][event_start:event_end, s_idx][leak_mask]
            rms = float(np.sqrt(np.mean(segment ** 2)))
            rms_records.append({
                "group_id": group["group_id"],
                "sensor": sensor,
                "distance_m": distance,
                "rms_pressure_pa": rms,
            })

    write_metadata(rms_records)
    write_manifest()
    zip_paths = build_zip_files()

    for zip_path in zip_paths:
        print(f"Generated CSV archive: {zip_path}")
    for fig_path in fig_paths:
        print(f"Generated figure: {fig_path}")


if __name__ == "__main__":
    main()
