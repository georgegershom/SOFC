#!/usr/bin/env python3
"""
Fabricated dataset generator: Core Time-Series Pressure Data (HF acoustic).

This script generates *synthetic* (fabricated) high-frequency acoustic pressure
time-series for 14 sensors (PG1–PG14), written as separate CSV files per second,
for four experimental groups (01–04). It also generates summary figures and a ZIP
containing the CSVs for easy download.

Event timeline (leak event run):
  - Baseline:   t ∈ [0, 5)   s
  - Leak open:  t = 5 s
  - Leak on:    t ∈ [5, 20)  s
  - Leak close: t = 20 s
  - Post-event: t ∈ [20, 30] s

Additionally, a 30-second baseline-only run is generated per group to satisfy
"30s before leak" baseline coverage.

Outputs (default under ./generated/):
  - generated/core_time_series_pressure_data/<group>/<run>/csv_per_second/sec_XXX.csv
  - generated/core_time_series_pressure_data/manifest_runs.csv
  - generated/core_time_series_pressure_data/sensors_layout.csv
  - generated/core_time_series_pressure_data/valves_layout.csv
  - generated/figures/*.png
  - generated/core_time_series_pressure_data_csv.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = e


SENSOR_IDS: List[str] = [f"PG{i}" for i in range(1, 15)]


@dataclass(frozen=True)
class Sensor:
    sensor_id: str
    position_m: float
    tag: str
    description: str


@dataclass(frozen=True)
class Valve:
    valve_id: str
    position_m: float
    description: str


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    fs_hz: int
    leak_location: str  # A, D, E
    sensor_configuration: str  # "A,B" or "A,B,C,D"
    stratification_index: float  # dimensionless [0..1], higher => stronger HF attenuation
    seed: int


def _ensure_matplotlib():
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate figures. "
            f"Import error was: {_MATPLOTLIB_IMPORT_ERROR!r}. "
            "Run: python3 -m pip install -r requirements.txt"
        )


def _write_csv(path: Path, header: List[str], rows: Iterable[Iterable[int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _save_per_second_csvs(
    out_dir: Path,
    fs_hz: int,
    duration_s: int,
    sensor_ids: List[str],
    counts_by_sensor: np.ndarray,  # shape: (n_sensors, duration_s * fs_hz)
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sensors = len(sensor_ids)
    assert counts_by_sensor.shape == (n_sensors, duration_s * fs_hz)

    header = ["sample_idx", *sensor_ids]
    sample_idx = np.arange(fs_hz, dtype=np.int32)

    # Write one CSV per second, matching the described experimental format.
    for sec in range(duration_s):
        start = sec * fs_hz
        end = (sec + 1) * fs_hz
        chunk = counts_by_sensor[:, start:end].T  # (fs_hz, n_sensors)

        arr = np.empty((fs_hz, 1 + n_sensors), dtype=np.int32)
        arr[:, 0] = sample_idx
        arr[:, 1:] = chunk

        sec_path = out_dir / f"sec_{sec:03d}.csv"
        np.savetxt(
            sec_path,
            arr,
            delimiter=",",
            fmt="%d",
            header=",".join(header),
            comments="",
        )


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _band_limit_fft(x: np.ndarray, fs_hz: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Band-limit signal using an FFT mask (real input, real output)."""
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X *= mask
    return np.fft.irfft(X, n=n)


def _colored_noise(rng: np.random.Generator, n: int, fs_hz: int, color: str = "pink") -> np.ndarray:
    """
    Generate colored noise (approx) via frequency shaping.
    - pink: 1/sqrt(f)
    - brown: 1/f
    """
    w = rng.normal(0.0, 1.0, size=n)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    # Avoid divide-by-zero at DC.
    f = np.maximum(freqs, 1.0)
    if color == "pink":
        shape = 1.0 / np.sqrt(f)
    elif color == "brown":
        shape = 1.0 / f
    else:
        shape = np.ones_like(f)

    W *= shape
    x = np.fft.irfft(W, n=n)
    x = (x - x.mean()) / (x.std() + 1e-12)
    return x


def _lowpass_one_pole(x: np.ndarray, fs_hz: int, fc_hz: float) -> np.ndarray:
    """Simple 1-pole IIR low-pass filter."""
    fc = float(max(1.0, min(fc_hz, 0.49 * fs_hz)))
    dt = 1.0 / fs_hz
    rc = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (rc + dt)

    y = np.empty_like(x)
    y0 = 0.0
    for i in range(x.size):
        y0 = y0 + alpha * (x[i] - y0)
        y[i] = y0
    return y


def _raised_cosine_envelope(t: np.ndarray, t_on: float, t_off: float, ramp_on: float, ramp_off: float) -> np.ndarray:
    env = np.zeros_like(t, dtype=np.float64)

    # Steady on region.
    on_start = t_on + ramp_on
    off_start = t_off - ramp_off
    env[(t >= on_start) & (t <= off_start)] = 1.0

    # Ramp up.
    m = (t >= t_on) & (t < on_start)
    if ramp_on > 0:
        x = (t[m] - t_on) / ramp_on
        env[m] = 0.5 - 0.5 * np.cos(np.pi * x)

    # Ramp down.
    m = (t > off_start) & (t <= t_off)
    if ramp_off > 0:
        x = (t[m] - off_start) / ramp_off
        env[m] = 0.5 + 0.5 * np.cos(np.pi * x)

    return env


def _transient_burst(t: np.ndarray, t0: float, f_hz: float, decay_s: float) -> np.ndarray:
    x = np.maximum(0.0, t - t0)
    return np.sin(2.0 * np.pi * f_hz * x) * np.exp(-x / max(1e-6, decay_s)) * (t >= t0)


def _path_crosses_point(a: float, b: float, x: float) -> bool:
    return (a - x) * (b - x) < 0.0


def _generate_run_counts(
    *,
    rng: np.random.Generator,
    sensors: List[Sensor],
    valves: List[Valve],
    fs_hz: int,
    duration_s: int,
    leak_pos_m: float,
    stratification_index: float,
    is_leak_event: bool,
    pressure_pa_per_count: float,
    leak_open_s: float = 5.0,
    leak_close_s: float = 20.0,
) -> np.ndarray:
    """
    Generate fabricated pressure time-series (int counts) for all sensors for one run.
    """
    n_total = duration_s * fs_hz
    t = np.arange(n_total, dtype=np.float64) / fs_hz

    # Background (stratified flow) acoustic field:
    # - pump/rotational tones
    # - broadband turbulence (colored noise)
    pump = (
        0.9 * np.sin(2.0 * np.pi * 50.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 100.0 * t + 0.7)
        + 0.15 * np.sin(2.0 * np.pi * 150.0 * t + 1.3)
    )
    turb = _colored_noise(rng, n_total, fs_hz, color="pink")
    bg_field = 2.5 * pump + 3.0 * turb + 0.7 * rng.normal(0.0, 1.0, size=n_total)

    # Leak source (only for leak-event runs).
    if is_leak_event:
        w = rng.normal(0.0, 1.0, size=n_total)
        leak_broad = _band_limit_fft(w, fs_hz, f_lo=200.0, f_hi=min(4500.0, 0.49 * fs_hz))
        leak_broad = (leak_broad - leak_broad.mean()) / (leak_broad.std() + 1e-12)

        # Quasi-tonal components (cavity/jet tones).
        tones = (
            0.6 * np.sin(2.0 * np.pi * 820.0 * t + 0.1)
            + 0.25 * np.sin(2.0 * np.pi * 1640.0 * t + 1.4)
            + 0.18 * np.sin(2.0 * np.pi * 2460.0 * t + 2.2)
        )

        # Amplitude modulation reflecting stratified interface effects.
        mod = 1.0 + (0.10 + 0.18 * stratification_index) * np.sin(2.0 * np.pi * 1.2 * t + 0.4)
        mod *= 1.0 + 0.05 * _colored_noise(rng, n_total, fs_hz, color="brown")

        env = _raised_cosine_envelope(t, t_on=leak_open_s, t_off=leak_close_s, ramp_on=0.18, ramp_off=0.30)

        open_burst = 2.0 * _transient_burst(t, leak_open_s, f_hz=1800.0, decay_s=0.06)
        close_burst = 1.7 * _transient_burst(t, leak_close_s, f_hz=2200.0, decay_s=0.05)

        leak_src = env * mod * (5.5 * leak_broad + 1.2 * tones) + open_burst + close_burst
    else:
        leak_src = np.zeros_like(t)

    # Propagation model parameters (fabricated but physically-inspired):
    c_eff = 900.0  # m/s effective wave speed in stratified two-phase core (lower than single-phase water)
    base_gamma = 0.010  # amplitude decay per meter (geometric + viscous)
    strat_gamma = 0.012 * stratification_index  # extra attenuation from stratification mechanisms

    counts = np.empty((len(sensors), n_total), dtype=np.int32)

    for si, s in enumerate(sensors):
        dist = abs(s.position_m - leak_pos_m)
        delay = int(round((dist / c_eff) * fs_hz))

        # Background varies by location (more near valves/leaks).
        local_amp = 1.0
        for v in valves:
            local_amp += 0.25 * math.exp(-abs(s.position_m - v.position_m) / 8.0)
        local_amp += 0.35 * math.exp(-abs(s.position_m - leak_pos_m) / 6.0)
        local_bg = local_amp * bg_field + (0.9 + 0.3 * stratification_index) * rng.normal(0.0, 1.0, size=n_total)

        # Leak component propagated to sensor.
        leak_at_sensor = np.zeros_like(leak_src)
        if is_leak_event and delay < n_total:
            leak_at_sensor[delay:] = leak_src[: n_total - delay]

            # Distance-dependent amplitude decay.
            amp = math.exp(-(base_gamma + strat_gamma) * dist)
            leak_at_sensor *= amp

            # Frequency-dependent attenuation (more severe with distance and stratification).
            # Stronger stratification => lower effective cutoff.
            fc = 4200.0 * math.exp(-dist * (0.010 + 0.020 * stratification_index))
            fc = max(250.0, min(fc, 0.45 * fs_hz))
            leak_at_sensor = _lowpass_one_pole(leak_at_sensor, fs_hz, fc_hz=fc)

            # Valve losses + weak echoes.
            valve_loss = 1.0
            for v in valves:
                if _path_crosses_point(leak_pos_m, s.position_m, v.position_m):
                    valve_loss *= 0.62

                    # Add a small echo (reflection at valve / discontinuity).
                    echo_delay = int(round((2.0 * abs(v.position_m - leak_pos_m) / c_eff) * fs_hz))
                    if echo_delay < n_total:
                        echo = np.zeros_like(leak_src)
                        echo[echo_delay:] = leak_src[: n_total - echo_delay]
                        echo *= 0.12 * math.exp(-(base_gamma + strat_gamma) * (dist + abs(v.position_m - leak_pos_m)))
                        echo = _lowpass_one_pole(echo, fs_hz, fc_hz=max(200.0, 0.8 * fc))
                        leak_at_sensor += echo

            leak_at_sensor *= valve_loss

        # Combine into pressure (Pa).
        # Scale factors chosen so baseline RMS is small and leak is clearly visible near leak location.
        pressure_pa = 4.0 * local_bg + 38.0 * leak_at_sensor

        # Quantize to int counts.
        counts[si, :] = np.round(pressure_pa / pressure_pa_per_count).astype(np.int32)

    return counts


def _welch_psd(x: np.ndarray, fs_hz: int, nperseg: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    n = x.size
    nperseg = int(min(nperseg, n))
    if nperseg < 256:
        nperseg = n
    step = nperseg // 2
    win = _hann(nperseg)
    scale = (win**2).sum()

    psd_acc = None
    k = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start : start + nperseg]
        seg = (seg - seg.mean()) * win
        X = np.fft.rfft(seg)
        P = (np.abs(X) ** 2) / (fs_hz * scale)
        if psd_acc is None:
            psd_acc = P
        else:
            psd_acc += P
        k += 1
    if psd_acc is None:
        seg = (x - x.mean()) * _hann(n)
        X = np.fft.rfft(seg)
        psd_acc = (np.abs(X) ** 2) / (fs_hz * (seg.size))
        k = 1
    psd = psd_acc / max(1, k)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs_hz)
    return freqs, psd


def _make_figures(
    *,
    out_fig_dir: Path,
    group: GroupSpec,
    sensors: List[Sensor],
    fs_hz: int,
    duration_s: int,
    leak_pos_m: float,
    pressure_pa_per_count: float,
    leak_counts: np.ndarray,  # (n_sensors, n_total)
    baseline_counts: np.ndarray,  # (n_sensors, n_total)
) -> List[Path]:
    _ensure_matplotlib()
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    n_total = duration_s * fs_hz
    t = np.arange(n_total, dtype=np.float64) / fs_hz

    def counts_to_pa(a: np.ndarray) -> np.ndarray:
        return a.astype(np.float64) * pressure_pa_per_count

    # 1) Waveforms around open/close transients for a few representative sensors.
    sensor_pick = ["PG3", "PG6", "PG10", "PG12"]
    pick_idx = [SENSOR_IDS.index(sid) for sid in sensor_pick]

    for center_s, tag in [(5.0, "open"), (20.0, "close")]:
        w0 = max(0.0, center_s - 0.25)
        w1 = min(duration_s, center_s + 0.35)
        i0 = int(w0 * fs_hz)
        i1 = int(w1 * fs_hz)
        tt = t[i0:i1]

        fig, ax = plt.subplots(len(pick_idx), 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Group {group.group_id}: Leak {tag} transient waveforms (fabricated)")
        for k, si in enumerate(pick_idx):
            y = counts_to_pa(leak_counts[si, i0:i1])
            ax[k].plot(tt, y, lw=0.8)
            ax[k].set_ylabel(f"{sensor_pick[k]}\nPa")
            ax[k].grid(True, alpha=0.25)
        ax[-1].set_xlabel("Time (s)")
        p = out_fig_dir / f"group_{group.group_id}_waveforms_{tag}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        created.append(p)

    # 2) PSD baseline vs leak (near leak sensor and far sensor).
    near_id = min(sensors, key=lambda s: abs(s.position_m - leak_pos_m)).sensor_id
    far_id = max(sensors, key=lambda s: abs(s.position_m - leak_pos_m)).sensor_id
    near_i = SENSOR_IDS.index(near_id)
    far_i = SENSOR_IDS.index(far_id)

    # Use steady windows: baseline [1..4]s, leak-on [10..15]s.
    def window(sec0: float, sec1: float) -> slice:
        return slice(int(sec0 * fs_hz), int(sec1 * fs_hz))

    b_win = window(1.0, 4.0)
    l_win = window(10.0, 15.0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Group {group.group_id}: PSD comparison (fabricated)")

    for j, (sid, si) in enumerate([(near_id, near_i), (far_id, far_i)]):
        fb, pb = _welch_psd(counts_to_pa(baseline_counts[si, b_win]), fs_hz)
        fl, pl = _welch_psd(counts_to_pa(leak_counts[si, l_win]), fs_hz)
        ax[j].semilogy(fb, pb + 1e-12, label="baseline-only run")
        ax[j].semilogy(fl, pl + 1e-12, label="leak-event run (10–15s)")
        ax[j].set_title(f"{sid}")
        ax[j].set_xlim(0, min(5000, fs_hz // 2))
        ax[j].set_xlabel("Frequency (Hz)")
        ax[j].grid(True, alpha=0.25)
        ax[j].legend()
    ax[0].set_ylabel("PSD (Pa²/Hz)")
    p = out_fig_dir / f"group_{group.group_id}_psd_baseline_vs_leak.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # 3) Attenuation curve: band energy (500–2000 Hz) vs distance.
    band_lo, band_hi = 500.0, 2000.0
    distances = []
    band_energy = []
    for si, s in enumerate(sensors):
        seg = counts_to_pa(leak_counts[si, l_win])
        seg = seg - seg.mean()
        n = seg.size
        win = _hann(n)
        X = np.fft.rfft(seg * win)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        P = (np.abs(X) ** 2)
        m = (freqs >= band_lo) & (freqs <= band_hi)
        e = float(P[m].sum() + 1e-12)
        distances.append(abs(s.position_m - leak_pos_m))
        band_energy.append(e)

    distances = np.asarray(distances, dtype=np.float64)
    band_energy = np.asarray(band_energy, dtype=np.float64)
    # Normalize for plotting.
    band_energy /= band_energy.max() if band_energy.max() > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(distances, band_energy, s=35)
    ax.set_title(
        f"Group {group.group_id}: Normalized band energy attenuation\n"
        f"Band {int(band_lo)}–{int(band_hi)} Hz, leak @ {group.leak_location} (fabricated)"
    )
    ax.set_xlabel("Distance to leak (m)")
    ax.set_ylabel("Normalized band energy (a.u.)")
    ax.grid(True, alpha=0.25)
    p = out_fig_dir / f"group_{group.group_id}_attenuation_band_energy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    return created


def _default_layout() -> Tuple[List[Sensor], List[Valve], Dict[str, float]]:
    sensors = [
        Sensor("PG1", 0.0, "upstream", "Upstream reference"),
        Sensor("PG2", 8.0, "upstream", "Upstream section"),
        Sensor("PG3", 16.0, "pre_valve_V1", "Before Valve V1"),
        Sensor("PG4", 26.0, "post_valve_V1", "After Valve V1"),
        Sensor("PG5", 34.0, "pre_leak_A", "Before leak point A"),
        Sensor("PG6", 38.0, "post_leak_A", "After leak point A"),
        Sensor("PG7", 50.0, "pre_leak_D", "Before leak point D"),
        Sensor("PG8", 54.0, "post_leak_D", "After leak point D"),
        Sensor("PG9", 70.0, "pre_valve_V2", "Before Valve V2"),
        Sensor("PG10", 74.0, "post_valve_V2", "After Valve V2"),
        Sensor("PG11", 88.0, "pre_leak_E", "Before leak point E"),
        Sensor("PG12", 92.0, "post_leak_E", "After leak point E"),
        Sensor("PG13", 110.0, "downstream", "Downstream section"),
        Sensor("PG14", 125.0, "downstream", "Downstream end"),
    ]
    valves = [
        Valve("V1", 24.0, "Upstream valve (flow control)"),
        Valve("V2", 72.0, "Midstream valve (isolation)"),
    ]
    leak_points = {"A": 36.0, "D": 52.0, "E": 90.0}
    return sensors, valves, leak_points


def _group_specs() -> List[GroupSpec]:
    return [
        GroupSpec("01", 17060, "A", "A,B", 0.25, seed=101),
        GroupSpec("02", 10000, "D", "A,B", 0.40, seed=202),
        GroupSpec("03", 10000, "E", "A,B,C,D", 0.55, seed=303),
        GroupSpec("04", 10000, "A", "A,B,C,D", 0.70, seed=404),
    ]


def _write_layout_files(out_root: Path, sensors: List[Sensor], valves: List[Valve]) -> None:
    _write_csv(
        out_root / "sensors_layout.csv",
        header=["sensor_id", "position_m", "tag", "description"],
        rows=((s.sensor_id, f"{s.position_m:.3f}", s.tag, s.description) for s in sensors),
    )
    _write_csv(
        out_root / "valves_layout.csv",
        header=["valve_id", "position_m", "description"],
        rows=((v.valve_id, f"{v.position_m:.3f}", v.description) for v in valves),
    )


def _write_manifest(out_root: Path, rows: List[Dict[str, object]]) -> None:
    out_path = out_root / "manifest_runs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _zip_csv_bundle(out_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as z:
        for p in sorted(out_root.rglob("*.csv")):
            z.write(p, arcname=str(p.relative_to(out_root)))
        # Include JSON group metadata as well (small, helpful).
        for p in sorted(out_root.rglob("*.json")):
            z.write(p, arcname=str(p.relative_to(out_root)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="generated/core_time_series_pressure_data",
        help="Output root directory (default: generated/core_time_series_pressure_data)",
    )
    ap.add_argument(
        "--duration-s",
        type=int,
        default=30,
        help="Duration per run in seconds (default: 30)",
    )
    ap.add_argument(
        "--pressure-pa-per-count",
        type=float,
        default=0.05,
        help="Quantization scale (Pa per integer count). Smaller => larger CSV values (default: 0.05)",
    )
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG figure generation (still creates CSVs + ZIP)",
    )
    ap.add_argument(
        "--zip-path",
        default="generated/core_time_series_pressure_data_csv.zip",
        help="ZIP output path (default: generated/core_time_series_pressure_data_csv.zip)",
    )
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    zip_path = Path(args.zip_path).resolve()
    duration_s = int(args.duration_s)
    pressure_pa_per_count = float(args.pressure_pa_per_count)

    sensors, valves, leak_points = _default_layout()
    groups = _group_specs()

    out_root.mkdir(parents=True, exist_ok=True)
    _write_layout_files(out_root, sensors, valves)

    manifest_rows: List[Dict[str, object]] = []
    fig_dir = out_root.parent / "figures"

    for g in groups:
        leak_pos = leak_points[g.leak_location]

        rng = np.random.default_rng(g.seed)

        group_dir = out_root / f"group_{g.group_id}"
        group_dir.mkdir(parents=True, exist_ok=True)

        # Baseline-only run: 30s normal operation.
        baseline_counts = _generate_run_counts(
            rng=rng,
            sensors=sensors,
            valves=valves,
            fs_hz=g.fs_hz,
            duration_s=duration_s,
            leak_pos_m=leak_pos,
            stratification_index=g.stratification_index,
            is_leak_event=False,
            pressure_pa_per_count=pressure_pa_per_count,
        )
        baseline_run_dir = group_dir / "run_baseline_only_30s" / "csv_per_second"
        _save_per_second_csvs(
            baseline_run_dir,
            fs_hz=g.fs_hz,
            duration_s=duration_s,
            sensor_ids=SENSOR_IDS,
            counts_by_sensor=baseline_counts,
        )

        manifest_rows.append(
            {
                "group_id": g.group_id,
                "run_id": "baseline_only_30s",
                "fs_hz": g.fs_hz,
                "duration_s": duration_s,
                "leak_location": g.leak_location,
                "leak_position_m": f"{leak_pos:.3f}",
                "sensor_configuration": g.sensor_configuration,
                "stratification_index": f"{g.stratification_index:.3f}",
                "event_open_s": "",
                "event_close_s": "",
                "notes": "Fabricated baseline-only run (no leak), used as '30s before leak' reference.",
            }
        )

        # Leak-event run: baseline + transient + post-event.
        leak_counts = _generate_run_counts(
            rng=rng,
            sensors=sensors,
            valves=valves,
            fs_hz=g.fs_hz,
            duration_s=duration_s,
            leak_pos_m=leak_pos,
            stratification_index=g.stratification_index,
            is_leak_event=True,
            pressure_pa_per_count=pressure_pa_per_count,
            leak_open_s=5.0,
            leak_close_s=20.0,
        )
        leak_run_dir = group_dir / "run_leak_event_30s" / "csv_per_second"
        _save_per_second_csvs(
            leak_run_dir,
            fs_hz=g.fs_hz,
            duration_s=duration_s,
            sensor_ids=SENSOR_IDS,
            counts_by_sensor=leak_counts,
        )

        manifest_rows.append(
            {
                "group_id": g.group_id,
                "run_id": "leak_event_30s",
                "fs_hz": g.fs_hz,
                "duration_s": duration_s,
                "leak_location": g.leak_location,
                "leak_position_m": f"{leak_pos:.3f}",
                "sensor_configuration": g.sensor_configuration,
                "stratification_index": f"{g.stratification_index:.3f}",
                "event_open_s": "5.0",
                "event_close_s": "20.0",
                "notes": "Fabricated leak event: valve opens at 5s, closes at 20s; record through 30s.",
            }
        )

        # Group metadata JSON.
        meta = {
            "group_id": g.group_id,
            "topic": "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics",
            "fabrication_notice": "This dataset is fully synthetic (fabricated) for analysis prototyping only.",
            "fs_hz": g.fs_hz,
            "duration_s": duration_s,
            "pressure_units": "Pa (quantized to integer counts in CSV)",
            "pressure_pa_per_count": pressure_pa_per_count,
            "sensors": [{"sensor_id": s.sensor_id, "position_m": s.position_m, "tag": s.tag} for s in sensors],
            "valves": [{"valve_id": v.valve_id, "position_m": v.position_m} for v in valves],
            "leak_location": g.leak_location,
            "leak_position_m": leak_pos,
            "sensor_configuration": g.sensor_configuration,
            "stratification_index": g.stratification_index,
            "leak_event_timeline_s": {
                "baseline": [0.0, 5.0],
                "open": 5.0,
                "leak_on": [5.0, 20.0],
                "close": 20.0,
                "post_event": [20.0, float(duration_s)],
            },
        }
        (group_dir / "group_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        if not args.no_figures:
            _make_figures(
                out_fig_dir=fig_dir,
                group=g,
                sensors=sensors,
                fs_hz=g.fs_hz,
                duration_s=duration_s,
                leak_pos_m=leak_pos,
                pressure_pa_per_count=pressure_pa_per_count,
                leak_counts=leak_counts,
                baseline_counts=baseline_counts,
            )

    _write_manifest(out_root, manifest_rows)
    _zip_csv_bundle(out_root, zip_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

