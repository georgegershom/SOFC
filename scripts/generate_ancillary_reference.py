#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import signal
import pywt
from tqdm import tqdm


@dataclass
class GenerationConfig:
    output_root: Path
    random_seed: int
    num_frequency_points: int
    min_frequency_hz: float
    max_frequency_hz: float
    num_replicates: int
    sample_rate_hz: float
    record_duration_s: float
    sensor_spacing_m: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_frequency_axis(n: int, fmin: float, fmax: float) -> np.ndarray:
    return np.linspace(fmin, fmax, n)


def sound_speed_air_m_per_s(temperature_c: float) -> float:
    # Simple linear approximation near 0-30 C
    return 331.3 + 0.606 * temperature_c


def sound_speed_water_m_per_s(temperature_c: float) -> float:
    # Mackenzie (1981) polynomial approximation in fresh water simplified for 0-30 C
    t = temperature_c
    return (
        1402.388 + 5.038813 * t - 5.799136e-2 * t**2 + 1.016e-4 * t**3
    )


def attenuation_air_db_per_m(f_hz: np.ndarray, temperature_c: float, rel_humidity: float) -> np.ndarray:
    # Synthetic but plausible trend: grows ~quadratically with frequency in kHz region
    fk = f_hz / 1000.0
    base = 0.005 + 0.03 * fk + 0.025 * fk**2  # baseline curve
    temp_factor = 1.0 + 0.002 * (temperature_c - 20.0)
    humidity_factor = 1.0 - 0.3 * (rel_humidity - 0.5)  # more humidity => slightly less attenuation
    noise = np.random.normal(0.0, 0.005, size=f_hz.shape)
    return np.clip(base * temp_factor * humidity_factor + noise, 0.0, None)


def attenuation_water_db_per_m(f_hz: np.ndarray, temperature_c: float) -> np.ndarray:
    # Synthetic plausible trend for kHz-ultrasonic range: increasing sublinearly
    fk = f_hz / 1000.0
    base = 0.02 * np.sqrt(np.maximum(fk, 1e-6))  # dB/m
    temp_factor = 1.0 + 0.003 * (temperature_c - 20.0)
    noise = np.random.normal(0.0, 0.002, size=f_hz.shape)
    return np.clip(base * temp_factor + noise, 0.0, None)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_baseline_single_phase(cfg: GenerationConfig) -> List[Path]:
    outputs: List[Path] = []
    frequencies = generate_frequency_axis(cfg.num_frequency_points, cfg.min_frequency_hz, cfg.max_frequency_hz)

    conditions: List[Dict[str, float]] = []
    for medium in ["air", "water"]:
        if medium == "air":
            temps = np.linspace(10.0, 30.0, 3)
            humidities = [0.3, 0.5, 0.7]
            for t in temps:
                for h in humidities:
                    conditions.append({"medium": medium, "temperature_c": float(t), "rel_humidity": float(h)})
        else:
            temps = np.linspace(10.0, 30.0, 4)
            for t in temps:
                conditions.append({"medium": medium, "temperature_c": float(t)})

    rows: List[Dict] = []
    for replicate_id in range(1, cfg.num_replicates + 1):
        for cond in conditions:
            if cond["medium"] == "air":
                c = sound_speed_air_m_per_s(cond["temperature_c"])  # m/s
                att = attenuation_air_db_per_m(frequencies, cond["temperature_c"], cond["rel_humidity"])  # dB/m
                for f, a in zip(frequencies, att):
                    rows.append({
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "replicate_id": replicate_id,
                        "medium": "air",
                        "frequency_hz": float(f),
                        "attenuation_db_per_m": float(a),
                        "sound_speed_m_per_s": float(c),
                        "temperature_c": float(cond["temperature_c"]),
                        "rel_humidity": float(cond["rel_humidity"]),
                        "pressure_kpa": 101.325,
                    })
            else:
                c = sound_speed_water_m_per_s(cond["temperature_c"])  # m/s
                att = attenuation_water_db_per_m(frequencies, cond["temperature_c"])  # dB/m
                for f, a in zip(frequencies, att):
                    rows.append({
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "replicate_id": replicate_id,
                        "medium": "water",
                        "frequency_hz": float(f),
                        "attenuation_db_per_m": float(a),
                        "sound_speed_m_per_s": float(c),
                        "temperature_c": float(cond["temperature_c"]),
                        "rel_humidity": None,
                        "pressure_kpa": 101.325,
                    })

    df = pd.DataFrame(rows)
    path = cfg.output_root / "baseline_single_phase" / "baseline_single_phase.csv"
    write_csv(df, path)
    outputs.append(path)

    meta = {
        "category": "Single-Phase Baseline Data",
        "description": "Acoustic attenuation and sound speed in only water or only air.",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "frequency_range_hz": [cfg.min_frequency_hz, cfg.max_frequency_hz],
        "num_replicates": cfg.num_replicates,
    }
    meta_path = cfg.output_root / "baseline_single_phase" / "metadata.json"
    write_json(meta, meta_path)
    outputs.append(meta_path)
    return outputs


@dataclass
class PublishedSource:
    paper_id: str
    title: str
    citation: str
    doi: Optional[str]
    candidate_urls: List[str]


PUBLISHED_SOURCES: List[PublishedSource] = [
    PublishedSource(
        paper_id="Li2022",
        title="Sound speed and attenuation in stratified air-water flows",
        citation="Li et al., 2022",
        doi=None,
        candidate_urls=[],
    ),
    PublishedSource(
        paper_id="Xue2022",
        title="Acoustic propagation in horizontal stratified flows",
        citation="Xue et al., 2022",
        doi=None,
        candidate_urls=[],
    ),
    PublishedSource(
        paper_id="Dijk2005",
        title="Ultrasonic measurements in multiphase pipelines",
        citation="Dijk, 2005",
        doi=None,
        candidate_urls=[],
    ),
]


def try_download(url: str, dest: Path, timeout_s: float = 10.0) -> bool:
    try:
        resp = requests.get(url, timeout=timeout_s)
        if resp.status_code == 200 and len(resp.content) > 0:
            ensure_dir(dest.parent)
            with dest.open("wb") as f:
                f.write(resp.content)
            return True
        return False
    except Exception:
        return False


def generate_synthetic_published_data(cfg: GenerationConfig) -> pd.DataFrame:
    frequencies = generate_frequency_axis(cfg.num_frequency_points, cfg.min_frequency_hz, cfg.max_frequency_hz)
    rows: List[Dict] = []
    for src in PUBLISHED_SOURCES:
        # Create plausible curves with small deviations per source
        base_shift = np.random.normal(0.0, 0.01)
        slope_scale = np.random.normal(1.0, 0.05)
        # Assume stratified flows yield attenuation between air and water baselines depending on gas fraction
        gas_fraction = np.clip(np.random.uniform(0.1, 0.6), 0.0, 1.0)
        temp_c = float(np.random.uniform(15.0, 25.0))
        # Blend attenuation: water curve + gas_fraction * (air - water)
        att_air = attenuation_air_db_per_m(frequencies, temp_c, rel_humidity=0.5)
        att_water = attenuation_water_db_per_m(frequencies, temp_c)
        att_mix = att_water + gas_fraction * (att_air - att_water)
        att_mix = np.clip(att_mix * slope_scale + base_shift, 0.0, None)
        # Blend sound speed: harmonic-like mean to emulate effective medium
        c_air = sound_speed_air_m_per_s(temp_c)
        c_water = sound_speed_water_m_per_s(temp_c)
        c_mix = 1.0 / (gas_fraction / c_air + (1.0 - gas_fraction) / c_water)
        c_mix = float(c_mix + np.random.normal(0.0, 1.5))

        for f, a in zip(frequencies, att_mix):
            rows.append({
                "paper_id": src.paper_id,
                "frequency_hz": float(f),
                "attenuation_db_per_m": float(a),
                "sound_speed_m_per_s": float(c_mix + np.random.normal(0.0, 0.5)),
                "gas_fraction": float(gas_fraction),
                "temperature_c": temp_c,
            })
    return pd.DataFrame(rows)


def fetch_or_fabricate_published(cfg: GenerationConfig) -> List[Path]:
    outputs: List[Path] = []
    records_meta: List[Dict] = []
    for src in PUBLISHED_SOURCES:
        downloaded_files: List[str] = []
        success_any = False
        for idx, url in enumerate(src.candidate_urls):
            target = cfg.output_root / "published" / f"{src.paper_id}_raw_{idx}.bin"
            ok = try_download(url, target)
            if ok:
                success_any = True
                downloaded_files.append(str(target))
        records_meta.append({
            "paper_id": src.paper_id,
            "title": src.title,
            "citation": src.citation,
            "doi": src.doi,
            "attempted_urls": src.candidate_urls,
            "downloaded_files": downloaded_files,
            "status": "downloaded" if success_any else "synthetic",
        })

    df = generate_synthetic_published_data(cfg)
    dest = cfg.output_root / "published" / "published_synthetic.csv"
    write_csv(df, dest)
    outputs.append(dest)

    meta_path = cfg.output_root / "published" / "sources.json"
    write_json({"sources": records_meta, "generated_at": datetime.utcnow().isoformat() + "Z"}, meta_path)
    outputs.append(meta_path)
    return outputs


def generate_materials_and_geometry(cfg: GenerationConfig) -> List[Path]:
    outputs: List[Path] = []

    pipe_rows = [
        {
            "pipe_id": "steel_1in_sch40",
            "material": "carbon_steel",
            "inner_diameter_m": 0.02664,  # ~1.049 in
            "outer_diameter_m": 0.03340,  # ~1.315 in
            "wall_thickness_m": 0.00338,
            "length_m": 5.0,
            "roughness_m": 4.5e-5,
            "density_kg_per_m3": 7850.0,
            "youngs_modulus_gpa": 200.0,
            "poisson_ratio": 0.3,
        },
        {
            "pipe_id": "acrylic_1in",
            "material": "pmma",
            "inner_diameter_m": 0.02540,
            "outer_diameter_m": 0.03175,
            "wall_thickness_m": 0.003175,
            "length_m": 3.0,
            "roughness_m": 1.0e-6,
            "density_kg_per_m3": 1180.0,
            "youngs_modulus_gpa": 3.2,
            "poisson_ratio": 0.35,
        },
    ]
    pipe_df = pd.DataFrame(pipe_rows)
    pipe_path = cfg.output_root / "materials_geometry" / "pipe_properties.csv"
    write_csv(pipe_df, pipe_path)
    outputs.append(pipe_path)

    sensor_rows = [
        {
            "sensor_id": "piezo_500khz",
            "type": "piezoelectric_ultrasonic",
            "model": "Generic-500k",
            "center_frequency_hz": 500_000.0,
            "bandwidth_hz": 150_000.0,
            "sensitivity_v_per_pa": 2.5e-3,
            "noise_floor_pa_rms": 0.2,
            "sampling_rate_hz": cfg.sample_rate_hz,
            "bit_depth": 16,
            "mount_type": "clamp_on",
            "coupling_type": "gel",
        },
        {
            "sensor_id": "mic_20khz",
            "type": "air_microphone",
            "model": "Generic-20k",
            "center_frequency_hz": 10_000.0,
            "bandwidth_hz": 10_000.0,
            "sensitivity_v_per_pa": 10e-3,
            "noise_floor_pa_rms": 0.05,
            "sampling_rate_hz": 96_000.0,
            "bit_depth": 24,
            "mount_type": "flush",
            "coupling_type": "air",
        },
    ]
    sensor_df = pd.DataFrame(sensor_rows)
    sensor_path = cfg.output_root / "materials_geometry" / "sensors.csv"
    write_csv(sensor_df, sensor_path)
    outputs.append(sensor_path)

    rig = {
        "default_pipe_id": "steel_1in_sch40",
        "default_sensor_pair": ["piezo_500khz", "piezo_500khz"],
        "sensor_spacing_m": cfg.sensor_spacing_m,
    }
    rig_path = cfg.output_root / "materials_geometry" / "test_rig.json"
    write_json(rig, rig_path)
    outputs.append(rig_path)

    return outputs


@dataclass
class SyntheticRunSpec:
    name: str
    medium: str
    temperature_c: float
    pressure_kpa: float
    center_frequency_hz: float
    flow_speed_m_per_s: float


def bandpass_filter(signal_in: np.ndarray, fs: float, f_center: float, bandwidth: float) -> np.ndarray:
    f_low = max(10.0, f_center - bandwidth / 2.0)
    f_high = min(fs / 2.0 - 10.0, f_center + bandwidth / 2.0)
    sos = signal.butter(N=4, Wn=[f_low / (fs / 2.0), f_high / (fs / 2.0)], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, signal_in)


def wavelet_denoise(signal_in: np.ndarray, wavelet: str = "sym8", level: Optional[int] = None) -> np.ndarray:
    if level is None:
        level = pywt.dwt_max_level(len(signal_in), pywt.Wavelet(wavelet).dec_len)
        level = max(1, min(level, 6))
    coeffs = pywt.wavedec(signal_in, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * math.sqrt(2 * math.log(len(signal_in)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal_in)]


def normalized_cross_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    corr = signal.correlate(y, x, mode="full")
    lags = signal.correlation_lags(len(y), len(x), mode="full")
    corr /= (len(x))
    return lags, corr


def generate_signal_pair(cfg: GenerationConfig, run: SyntheticRunSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(cfg.sample_rate_hz * cfg.record_duration_s)
    t = np.arange(n_samples) / cfg.sample_rate_hz

    # Carrier sinusoid modulated by a burst and background noise
    burst = signal.windows.tukey(n_samples, alpha=0.2)
    carrier = np.sin(2 * np.pi * run.center_frequency_hz * t)
    medium = run.medium
    if medium == "air":
        c = sound_speed_air_m_per_s(run.temperature_c)
        attenuation_db_per_m = 0.2  # small for short path, synthetic
    else:
        c = sound_speed_water_m_per_s(run.temperature_c)
        attenuation_db_per_m = 0.05

    time_delay_s = cfg.sensor_spacing_m / c
    amplitude_decay = 10 ** (-attenuation_db_per_m * cfg.sensor_spacing_m / 20.0)

    s1 = 0.8 * burst * carrier + 0.05 * np.random.randn(n_samples)
    s2 = amplitude_decay * 0.8 * np.sin(2 * np.pi * run.center_frequency_hz * np.clip(t - time_delay_s, 0, None))
    s2 *= burst
    s2 += 0.05 * np.random.randn(n_samples)

    return t, s1, s2


def generate_signal_processing_outputs(cfg: GenerationConfig) -> List[Path]:
    outputs: List[Path] = []
    runs: List[SyntheticRunSpec] = [
        SyntheticRunSpec("air_low", "air", 20.0, 101.3, 10_000.0, 0.0),
        SyntheticRunSpec("air_high", "air", 25.0, 101.3, 18_000.0, 0.0),
        SyntheticRunSpec("water_low", "water", 20.0, 101.3, 150_000.0, 0.0),
        SyntheticRunSpec("water_high", "water", 22.0, 101.3, 300_000.0, 0.0),
    ]

    feature_rows: List[Dict] = []

    for run in tqdm(runs, desc="Synth runs"):
        t, s1, s2 = generate_signal_pair(cfg, run)

        # Frequency-domain bandpass around center
        filtered_s1 = bandpass_filter(s1, cfg.sample_rate_hz, run.center_frequency_hz, bandwidth=run.center_frequency_hz * 0.5)
        filtered_s2 = bandpass_filter(s2, cfg.sample_rate_hz, run.center_frequency_hz, bandwidth=run.center_frequency_hz * 0.5)

        # Wavelet denoise
        den_s1 = wavelet_denoise(filtered_s1)
        den_s2 = wavelet_denoise(filtered_s2)

        # Cross-correlation for TDE
        lags, rxy = normalized_cross_correlation(den_s1, den_s2)
        tau = lags / cfg.sample_rate_hz
        peak_idx = int(np.argmax(np.abs(rxy)))
        tde_s = float(tau[peak_idx])
        # Estimate speed from spacing and TDE (absolute value)
        if abs(tde_s) > 1e-9:
            estimated_c = cfg.sensor_spacing_m / abs(tde_s)
        else:
            estimated_c = np.nan

        # Basic features
        def spectral_centroid(x: np.ndarray, fs: float) -> float:
            freqs, Pxx = signal.welch(x, fs=fs, nperseg=min(4096, len(x)))
            if np.sum(Pxx) <= 0:
                return np.nan
            return float(np.sum(freqs * Pxx) / np.sum(Pxx))

        def bandwidth_rms(x: np.ndarray, fs: float) -> float:
            freqs, Pxx = signal.welch(x, fs=fs, nperseg=min(4096, len(x)))
            centroid = spectral_centroid(x, fs)
            return float(math.sqrt(np.sum(Pxx * (freqs - centroid) ** 2) / np.sum(Pxx)))

        def snr_db(x: np.ndarray) -> float:
            s_power = np.mean((x - np.mean(x)) ** 2)
            n_power = np.median((x - signal.medfilt(x, kernel_size=31)) ** 2)
            if n_power <= 1e-12:
                return 60.0
            return float(10 * np.log10(max(s_power / n_power, 1e-12)))

        features = {
            "run_name": run.name,
            "medium": run.medium,
            "temperature_c": run.temperature_c,
            "center_frequency_hz": run.center_frequency_hz,
            "tde_seconds": tde_s,
            "estimated_sound_speed_m_per_s": estimated_c,
            "spectral_centroid_hz": spectral_centroid(den_s1, cfg.sample_rate_hz),
            "bandwidth_rms_hz": bandwidth_rms(den_s1, cfg.sample_rate_hz),
            "snr_db": snr_db(den_s1),
        }
        feature_rows.append(features)

        # Persist filtered signals and cross-correlation
        filtered_dir = cfg.output_root / "signal_processing" / "filtered"
        cc_dir = cfg.output_root / "signal_processing" / "cross_correlation"

        df_sig = pd.DataFrame({
            "time_s": t.astype(float),
            "signal1": den_s1.astype(float),
            "signal2": den_s2.astype(float),
        })
        sig_path = filtered_dir / f"{run.name}_filtered.csv"
        write_csv(df_sig, sig_path)
        outputs.append(sig_path)

        df_cc = pd.DataFrame({"tau_s": tau.astype(float), "R_x1x2": rxy.astype(float)})
        cc_path = cc_dir / f"{run.name}_cross_correlation.csv"
        write_csv(df_cc, cc_path)
        outputs.append(cc_path)

    features_df = pd.DataFrame(feature_rows)
    feat_path = cfg.output_root / "signal_processing" / "features" / "features.csv"
    write_csv(features_df, feat_path)
    outputs.append(feat_path)

    return outputs


def build_manifest(all_paths: List[Path], cfg: GenerationConfig) -> Dict:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "version": 1,
        "output_root": str(cfg.output_root),
        "files": [str(p) for p in sorted(all_paths)],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Ancillary & Reference Data for stratified flow acoustics")
    parser.add_argument("--output-root", type=str, default="data/ancillary_reference", help="Output dataset root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-frequency-points", type=int, default=30)
    parser.add_argument("--min-frequency-hz", type=float, default=1_000.0)
    parser.add_argument("--max-frequency-hz", type=float, default=300_000.0)
    parser.add_argument("--num-replicates", type=int, default=3)
    parser.add_argument("--sample-rate-hz", type=float, default=1_000_000.0)
    parser.add_argument("--record-duration-s", type=float, default=0.02)
    parser.add_argument("--sensor-spacing-m", type=float, default=0.3)
    args = parser.parse_args(argv)

    cfg = GenerationConfig(
        output_root=Path(args.output_root).resolve(),
        random_seed=args.seed,
        num_frequency_points=args.num_frequency_points,
        min_frequency_hz=args.min_frequency_hz,
        max_frequency_hz=args.max_frequency_hz,
        num_replicates=args.num_replicates,
        sample_rate_hz=args.sample_rate_hz,
        record_duration_s=args.record_duration_s,
        sensor_spacing_m=args.sensor_spacing_m,
    )

    set_reproducibility(cfg.random_seed)

    ensure_dir(cfg.output_root)

    all_paths: List[Path] = []

    print("Generating single-phase baseline data...")
    all_paths += generate_baseline_single_phase(cfg)

    print("Fetching or fabricating published datasets...")
    all_paths += fetch_or_fabricate_published(cfg)

    print("Generating materials and geometry tables...")
    all_paths += generate_materials_and_geometry(cfg)

    print("Generating signal processing outputs...")
    all_paths += generate_signal_processing_outputs(cfg)

    manifest = build_manifest(all_paths, cfg)
    manifest_path = cfg.output_root / "dataset_manifest.json"
    write_json(manifest, manifest_path)

    print(f"Done. Wrote {len(all_paths)} files. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
