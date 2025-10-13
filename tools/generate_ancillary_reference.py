#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT, "data", "ancillary_reference")
np.random.seed(42)

# -------------------- Utility functions --------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


# -------------------- Physical constants --------------------
WATER_C = 1480.0  # m/s typical at ~20C
AIR_C = 343.0     # m/s typical at 20C
WATER_RHO = 998.0 # kg/m^3
AIR_RHO = 1.204   # kg/m^3


# -------------------- Single-phase baseline generation --------------------

def synth_acoustic_sweep(f_start: float, f_end: float, duration_s: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_s, 1.0/fs)
    k = (f_end - f_start) / duration_s
    phase = 2.0*np.pi*(f_start*t + 0.5*k*t**2)
    sweep = np.sin(phase)
    return t, sweep


def apply_attenuation(signal: np.ndarray, alpha_db_per_m: float, distance_m: float) -> np.ndarray:
    # Convert dB/m to linear amplitude factor: A_out = A_in * 10^(-alpha*L/20)
    factor = 10.0 ** (-(alpha_db_per_m * distance_m) / 20.0)
    return signal * factor


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(signal**2) + 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def generate_single_phase_baselines():
    out_root = os.path.join(DATA_ROOT, "single_phase_baseline")
    ensure_dir(os.path.join(out_root, "water"))
    ensure_dir(os.path.join(out_root, "air"))

    # Parameters
    fs = 20000.0
    duration = 2.0
    distances_m = [1.0, 2.0, 5.0]
    sweeps = [(200.0, 2000.0), (1000.0, 10000.0)]
    snr_choices = [20.0, 30.0, 40.0]

    rows: List[Dict] = []

    for medium in ["water", "air"]:
        c = WATER_C if medium == "water" else AIR_C
        rho = WATER_RHO if medium == "water" else AIR_RHO
        base_alpha = 0.002 if medium == "water" else 0.01  # dB/m nominal baseline

        for (f0, f1) in sweeps:
            t, s = synth_acoustic_sweep(f0, f1, duration, fs)
            for d in distances_m:
                for snr_db in snr_choices:
                    attenuated = apply_attenuation(s, base_alpha, d)
                    measured = add_noise(attenuated, snr_db)
                    df = pd.DataFrame({
                        "time_s": t,
                        "signal": measured,
                    })
                    fname = f"{medium}_sweep_{int(f0)}-{int(f1)}Hz_{d:.1f}m_SNR{int(snr_db)}.csv"
                    fpath = os.path.join(out_root, medium, fname)
                    write_csv(fpath, df)

                    rows.append({
                        "file": os.path.relpath(fpath, ROOT),
                        "medium": medium,
                        "fs_Hz": fs,
                        "duration_s": duration,
                        "distance_m": d,
                        "snr_db": snr_db,
                        "f_start_Hz": f0,
                        "f_end_Hz": f1,
                        "sound_speed_mps": c,
                        "density_kgpm3": rho,
                        "alpha_db_per_m": base_alpha,
                    })

    meta = pd.DataFrame(rows)
    write_csv(os.path.join(out_root, "baseline_index.csv"), meta)


# -------------------- Materials and geometry --------------------
@dataclass
class PipeSpec:
    material: str
    wall_thickness_m: float
    inner_diameter_m: float
    length_m: float
    roughness_m: float


def create_materials_geometry_tables():
    out_root = os.path.join(DATA_ROOT, "materials_geometry")
    ensure_dir(out_root)

    pipe_rows = [
        PipeSpec("Steel", wall_thickness_m=0.006, inner_diameter_m=0.050, length_m=10.0, roughness_m=4.5e-5),
        PipeSpec("PVC", wall_thickness_m=0.004, inner_diameter_m=0.063, length_m=8.0, roughness_m=1.5e-6),
        PipeSpec("Copper", wall_thickness_m=0.003, inner_diameter_m=0.025, length_m=5.0, roughness_m=1.5e-6),
    ]
    pipe_df = pd.DataFrame([asdict(p) for p in pipe_rows])
    write_csv(os.path.join(out_root, "pipes.csv"), pipe_df)

    sensors = [
        {
            "model": "Hydrophone-H1",
            "type": "Hydrophone",
            "sensitivity_V_per_Pa": 1.2e-3,
            "freq_response_Hz": "100-10000",
            "noise_floor_uPa_per_sqrtHz": 20.0,
        },
        {
            "model": "Microphone-M1",
            "type": "Microphone",
            "sensitivity_V_per_Pa": 3.5e-3,
            "freq_response_Hz": "20-20000",
            "noise_floor_uPa_per_sqrtHz": 35.0,
        },
    ]
    sensors_df = pd.DataFrame(sensors)
    write_csv(os.path.join(out_root, "sensors.csv"), sensors_df)


# -------------------- Signal processing --------------------

def _hann_transition(x: np.ndarray, left: float, right: float) -> np.ndarray:
    window = np.zeros_like(x)
    # Flat passband
    window[(x >= left) & (x <= right)] = 1.0
    # Smooth edges (10% of band width on each side where possible)
    bw = max(right - left, 1e-9)
    t = 0.1 * bw
    # Left transition
    l0, l1 = max(0.0, left - t), left
    m = (x >= l0) & (x < l1)
    if np.any(m):
        xi = (x[m] - l0) / (l1 - l0)
        window[m] = 0.5 - 0.5 * np.cos(np.pi * xi)
    # Right transition
    r0, r1 = right, right + t
    m = (x > r0) & (x <= r1)
    if np.any(m):
        xi = (x[m] - r0) / (r1 - r0)
        window[m] = 0.5 + 0.5 * np.cos(np.pi * xi)
    return np.clip(window, 0.0, 1.0)


def fourier_filter(signal: np.ndarray, fs: float, lowcut: float, highcut: float) -> np.ndarray:
    # FFT-domain bandpass with Hann transitions
    n = len(signal)
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    if lowcut <= 0.0 and highcut >= (fs / 2.0):
        mask = np.ones_like(freqs)
    else:
        left = max(0.0, lowcut)
        right = min(highcut, fs/2.0)
        if right <= left:
            mask = np.zeros_like(freqs)
        else:
            mask = _hann_transition(freqs, left, right)
    y = np.fft.irfft(spec * mask, n=n)
    return y.astype(float)


def wavelet_like_denoise(signal: np.ndarray) -> np.ndarray:
    # Spectral soft-threshold denoising (wavelet-inspired)
    n = len(signal)
    spec = np.fft.rfft(signal)
    mag = np.abs(spec)
    sigma = np.median(np.abs(mag - np.median(mag))) / 0.6745 + 1e-12
    uthresh = sigma * np.sqrt(2.0 * np.log(n))
    scale = np.maximum(0.0, 1.0 - (uthresh / (mag + 1e-12)))
    y = np.fft.irfft(spec * scale, n=n)
    return y.astype(float)


def cross_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lags = np.arange(-len(x) + 1, len(x))
    return lags, corr


def extract_features(signal: np.ndarray, fs: float) -> Dict:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    mag = np.abs(spectrum)
    total_power = np.sum(mag**2)
    if total_power <= 0:
        centroid = 0.0
    else:
        centroid = float(np.sum(freqs * mag**2) / total_power)
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    return {
        "spectral_centroid_Hz": centroid,
        "rms": rms,
        "peak": peak,
    }


def simulate_signals_and_process():
    raw_root = os.path.join(DATA_ROOT, "signal_processing", "raw_signals")
    filt_root = os.path.join(DATA_ROOT, "signal_processing", "filtered_signals")
    xcorr_root = os.path.join(DATA_ROOT, "signal_processing", "cross_correlation")
    feat_root = os.path.join(DATA_ROOT, "signal_processing", "features")
    ensure_dir(raw_root); ensure_dir(filt_root); ensure_dir(xcorr_root); ensure_dir(feat_root)

    fs = 20000.0
    duration = 1.0
    t = np.arange(0, duration, 1.0/fs)

    # Two sensors separated by 0.5 m; simulate TDE via delay using water speed
    separation_m = 0.5
    delay_s = separation_m / WATER_C

    f_sig = 2000.0
    x1 = np.sin(2*np.pi*f_sig*t)
    # Apply a delay to create x2
    delay_samples = int(round(delay_s * fs))
    x2 = np.concatenate([np.zeros(delay_samples), x1])[:len(x1)]

    # Add noise
    x1n = add_noise(x1, 30.0)
    x2n = add_noise(x2, 30.0)

    # Filters
    x1_bp = fourier_filter(x1n, fs, 500.0, 5000.0)
    x2_bp = fourier_filter(x2n, fs, 500.0, 5000.0)

    x1_wv = wavelet_like_denoise(x1n)
    x2_wv = wavelet_like_denoise(x2n)

    # Cross-correlation for TDE
    lags, r_x1x2 = cross_correlation(x1_bp, x2_bp)
    tau = lags / fs
    est_delay_idx = int(np.argmax(r_x1x2))
    est_delay_s = tau[est_delay_idx]

    # Write raw
    write_csv(os.path.join(raw_root, "sensor_x1.csv"), pd.DataFrame({"time_s": t, "signal": x1n}))
    write_csv(os.path.join(raw_root, "sensor_x2.csv"), pd.DataFrame({"time_s": t, "signal": x2n}))

    # Write filtered
    write_csv(os.path.join(filt_root, "x1_bandpass.csv"), pd.DataFrame({"time_s": t, "signal": x1_bp}))
    write_csv(os.path.join(filt_root, "x2_bandpass.csv"), pd.DataFrame({"time_s": t, "signal": x2_bp}))
    write_csv(os.path.join(filt_root, "x1_wavelet.csv"), pd.DataFrame({"time_s": t, "signal": x1_wv}))
    write_csv(os.path.join(filt_root, "x2_wavelet.csv"), pd.DataFrame({"time_s": t, "signal": x2_wv}))

    # Write cross-correlation
    write_csv(os.path.join(xcorr_root, "R_x1x2.csv"), pd.DataFrame({"tau_s": tau, "R": r_x1x2}))

    # Features for leak detection/location proxy
    features = {
        "x1_bandpass": extract_features(x1_bp, fs),
        "x2_bandpass": extract_features(x2_bp, fs),
        "x1_wavelet": extract_features(x1_wv, fs),
        "x2_wavelet": extract_features(x2_wv, fs),
        "estimated_delay_s": float(est_delay_s),
        "true_delay_s": float(delay_s),
        "sensor_separation_m": float(separation_m),
        "sound_speed_mps_assumed": float(WATER_C),
    }
    write_json(os.path.join(feat_root, "features.json"), features)


# -------------------- Published datasets manifest --------------------

def create_published_manifest():
    out_root = os.path.join(DATA_ROOT, "published_datasets")
    ensure_dir(out_root)

    manifest = {
        "note": "This manifest lists key references with URLs/DOIs for manual download (copyright-respecting).",
        "entries": [
            {
                "citation": "Li, X., et al. (2022). Acoustic propagation in stratified two-phase pipe flow...",
                "doi_or_url": "doi:10.XXXXX/li2022",
                "type": "sound speed & attenuation",
                "status": "reference-only",
            },
            {
                "citation": "Xue, Y., et al. (2022). Ultrasound in stratified gas-liquid flows...",
                "doi_or_url": "doi:10.XXXXX/xue2022",
                "type": "attenuation & interface effects",
                "status": "reference-only",
            },
            {
                "citation": "Dijk, E. (2005). Acoustic waves in multiphase flows (PhD Thesis).",
                "doi_or_url": "doi:10.XXXXX/dijk2005",
                "type": "foundational",
                "status": "reference-only",
            },
        ],
    }
    write_json(os.path.join(out_root, "manifest.json"), manifest)


# -------------------- Dataset catalog --------------------

def emit_dataset_catalog():
    catalog = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": os.path.relpath(DATA_ROOT, ROOT),
        "categories": {
            "single_phase_baseline": {
                "index_csv": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "single_phase_baseline", "baseline_index.csv"),
                "description": "Acoustic attenuation and sound speed baselines in single-phase media.",
            },
            "materials_geometry": {
                "pipes_csv": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "materials_geometry", "pipes.csv"),
                "sensors_csv": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "materials_geometry", "sensors.csv"),
            },
            "signal_processing": {
                "raw_signals": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "signal_processing", "raw_signals"),
                "filtered_signals": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "signal_processing", "filtered_signals"),
                "cross_correlation": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "signal_processing", "cross_correlation"),
                "features": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "signal_processing", "features", "features.json"),
            },
            "published_datasets": {
                "manifest": os.path.join(os.path.relpath(DATA_ROOT, ROOT), "published_datasets", "manifest.json"),
            }
        }
    }
    write_json(os.path.join(DATA_ROOT, "catalog.json"), catalog)


def main():
    ensure_dir(DATA_ROOT)
    generate_single_phase_baselines()
    create_materials_geometry_tables()
    simulate_signals_and_process()
    create_published_manifest()
    emit_dataset_catalog()
    print("Ancillary & Reference dataset generated at:", DATA_ROOT)


if __name__ == "__main__":
    main()
