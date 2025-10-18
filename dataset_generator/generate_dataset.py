#!/usr/bin/env python3
import argparse
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import yaml
from PIL import Image, ImageDraw

np.random.seed(42)

# -----------------------------
# Domain helpers (physically-informed synthetic logic)
# -----------------------------

RNG = np.random.default_rng(42)

@dataclass
class MixDesign:
    mix_id: str
    rubber_pct_vol: float  # 0-40% typical
    w_c_ratio: float       # water-cement
    silica_fume_pct: float # binder replacement
    fiber_pct_vol: float   # optional fibers
    air_content_pct: float


def sample_mix_designs(n: int) -> List[MixDesign]:
    mixes: List[MixDesign] = []
    for i in range(n):
        rubber = float(RNG.uniform(0.0, 0.4))
        w_c = float(RNG.uniform(0.28, 0.45))
        sf = float(RNG.choice([0.0, 0.05, 0.08, 0.10]))
        fiber = float(RNG.choice([0.0, 0.003, 0.005, 0.007]))
        air = float(RNG.uniform(2.0, 6.0))
        mixes.append(MixDesign(
            mix_id=f"M{i+1:02d}",
            rubber_pct_vol=rubber,
            w_c_ratio=w_c,
            silica_fume_pct=sf,
            fiber_pct_vol=fiber,
            air_content_pct=air,
        ))
    return mixes


# Strength development with age (ambient) baseline
# Simple model: f_c(t) = f28 * (t/(4+t))^k, where f28 depends on mix

def predict_f28_from_mix(mix: MixDesign) -> float:
    base = 70.0  # MPa for high-performance baseline
    # penalties and bonuses
    penalty_rubber = 35.0 * mix.rubber_pct_vol  # rubber weakens compressive
    penalty_air = 0.8 * (mix.air_content_pct - 2.0)
    bonus_sf = 7.0 * mix.silica_fume_pct
    bonus_fiber = 8.0 * mix.fiber_pct_vol
    penalty_wc = 100.0 * (mix.w_c_ratio - 0.30)
    f28 = base - penalty_rubber - penalty_wc - penalty_air + bonus_sf + bonus_fiber
    noise = RNG.normal(0, 2.0)
    return max(20.0, f28 + noise)


def age_gain_factor(age_days: int, k: float = 0.6) -> float:
    t = float(age_days)
    return (t / (4.0 + t)) ** k


def tensile_from_compressive(fc: float, method: str) -> float:
    if method == 'split':
        return 0.56 * (fc ** 0.5)  # MPa, rough empirical
    else:
        return 0.62 * (fc ** 0.5)  # flexural tends to be a bit higher


def modulus_from_compressive(fc: float, density: float = 2400.0) -> float:
    # MPa, ACI-like: Ec = 0.043*w^1.5*sqrt(fc) with w in kg/m3
    return 0.043 * (density ** 1.5) * np.sqrt(fc) / 1000.0


def density_from_mix(mix: MixDesign) -> float:
    # Rubber lowers density; air increases reduction
    base = 2450.0
    delta_rubber = 400.0 * mix.rubber_pct_vol
    delta_air = 5.0 * (mix.air_content_pct - 2.0)
    noise = RNG.normal(0, 10.0)
    return max(2000.0, base - delta_rubber - delta_air + noise)


def upv_from_quality(fc: float, density: float) -> float:
    # km/s; higher with fc and density
    val = 3.0 + 0.0004 * fc + 0.0006 * (density - 2200.0)
    noise = RNG.normal(0, 0.05)
    return max(2.0, val + noise)


# High temperature degradation models

def residual_factor_temp(temp_c: float, rubber_pct: float, cooling: str) -> float:
    # Residual strength fraction after exposure and cooling
    # Piecewise-like degradation with harsher drop for quench
    base_curve = {
        23: 1.00,
        200: 0.95,
        400: 0.75,
        600: 0.45,
        800: 0.25,
    }
    temps = np.array(sorted(base_curve.keys()), dtype=float)
    vals = np.array([base_curve[t] for t in temps], dtype=float)
    rf = float(np.interp(temp_c, temps, vals))
    # Rubber can help at mid temps (crack control) but harm at high (softening)
    rf += 0.05 * np.clip(0.20 - abs(temp_c - 400.0) / 400.0, 0.0, 0.20) * (rubber_pct/0.4)
    # Cooling effect: quench harsher
    rf *= 0.92 if cooling == 'quench' else 1.0
    # Noise
    rf += RNG.normal(0, 0.02)
    return float(np.clip(rf, 0.05, 1.05))


def mass_loss_pct(temp_c: float, rubber_pct: float, cooling: str) -> float:
    base = 0.2 + 0.02 * (temp_c/100.0) ** 1.2
    base += 5.0 * rubber_pct  # rubber burns out
    base += 1.0 if cooling == 'quench' else 0.0
    noise = RNG.normal(0, 0.5)
    return float(np.clip(base + noise, 0.0, 25.0))


def upv_residual(upv_ambient: float, temp_c: float, cooling: str) -> float:
    drop = 0.0
    if temp_c >= 200:
        drop += 0.2
    if temp_c >= 400:
        drop += 0.6
    if temp_c >= 600:
        drop += 1.0
    if temp_c >= 800:
        drop += 1.2
    if cooling == 'quench':
        drop += 0.2
    noise = RNG.normal(0, 0.05)
    return float(max(0.8, upv_ambient - drop + noise))


def stress_strain_curve(fc: float, Ec: float, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    # Simplified Hognestad-like curve for concrete post-fire residual
    eps0 = fc / Ec  # strain at peak
    eps_u = 3.0 * eps0
    strain = np.linspace(0.0, eps_u, n_points)
    stress = np.where(
        strain <= eps0,
        Ec * strain * (1.0 - 0.5 * (strain / eps0)),
        fc * np.exp(-3.0 * (strain - eps0) / eps0)
    )
    return strain, stress


def poisson_ratio_from_mix(mix: MixDesign) -> float:
    base = 0.20
    return float(np.clip(base + 0.02 * mix.rubber_pct_vol + RNG.normal(0, 0.005), 0.15, 0.26))


def cte_curve(temp_max: float, pts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    temps = np.linspace(20, temp_max, pts)
    cte = 8.0e-6 + 2.0e-8 * (temps - 20)  # per C
    return temps, cte


def transient_thermal_strain_curve(temp_max: float, load_ratio: float, pts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    temps = np.linspace(20, temp_max, pts)
    # Free thermal expansion + transient creep under load
    free = 9.0e-6 * (temps - 20)
    creep = load_ratio * 2.0e-6 * ((temps - 20) / 100.0) ** 1.3
    noise = RNG.normal(0, 2.0e-6, size=pts)
    return temps, free + creep + noise


def pore_pressure_curve(temp_max: float, rubber_pct: float, pts: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    temps = np.linspace(20, temp_max, pts)
    # Peak around 250-300C, higher with rubber
    peak_temp = 260 + 30 * RNG.random()
    peak_pressure = 0.4 + 1.6 * rubber_pct + 0.1 * RNG.random()
    pressures = peak_pressure * np.exp(-0.5 * ((temps - peak_temp)/60.0)**2)
    pressures += 0.02 * RNG.random(size=pts)
    return temps, pressures


def residual_poisson_ratio(nu_ambient: float, residual_factor: float) -> float:
    # Damage typically reduces effective Poisson's ratio slightly
    nu = nu_ambient * (1.0 - 0.25 * (1.0 - residual_factor))
    return float(np.clip(nu, 0.10, 0.28))


def insitu_strength_factor(temp_c: float, rubber_pct: float) -> float:
    # In-situ (at-temperature) strength fraction; generally low at high T
    base_curve = {
        23: 1.00,
        200: 0.90,
        400: 0.65,
        600: 0.35,
        800: 0.18,
    }
    temps = np.array(sorted(base_curve.keys()), dtype=float)
    vals = np.array([base_curve[t] for t in temps], dtype=float)
    rf = float(np.interp(temp_c, temps, vals))
    # Rubber may provide modest benefit around 200-400C
    rf += 0.04 * np.clip(0.25 - abs(temp_c - 350.0) / 350.0, 0.0, 0.25) * (rubber_pct/0.4)
    rf += RNG.normal(0, 0.02)
    return float(np.clip(rf, 0.05, 1.0))


# -----------------------------
# Generator
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_plot(x, y, xlabel, ylabel, title, out_path):
    # Lightweight plotting using Pillow to avoid heavy deps
    width, height = 900, 600
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 50, 80
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Compute plot area
    x0, y0 = margin_left, margin_top
    x1, y1 = width - margin_right, height - margin_bottom

    # Axes
    draw.rectangle([(x0, y0), (x1, y1)], outline=(220, 220, 220))
    draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=2)
    draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)

    # Data scaling
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    def to_px(xi, yi):
        px = x0 + (xi - x_min) / (x_max - x_min) * (x1 - x0)
        py = y1 - (yi - y_min) / (y_max - y_min) * (y1 - y0)
        return (int(px), int(py))

    # Gridlines (5 ticks)
    for t in range(6):
        tx = x0 + t * (x1 - x0) / 5
        ty = y1 - t * (y1 - y0) / 5
        draw.line([(tx, y0), (tx, y1)], fill=(240, 240, 240))
        draw.line([(x0, ty), (x1, ty)], fill=(240, 240, 240))

    # Plot polyline
    points = [to_px(float(xi), float(yi)) for xi, yi in zip(x, y)]
    if len(points) > 1:
        draw.line(points, fill=(30, 100, 200), width=3)

    # Title and labels (simple)
    # Pillow's default font used to avoid extra dependencies
    draw.text((width // 2 - 200, 10), title[:60], fill=(0, 0, 0))
    draw.text((x0, height - margin_bottom + 30), xlabel[:40], fill=(0, 0, 0))
    draw.text((10, y0), ylabel[:40], fill=(0, 0, 0))

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_xy_csv(x, y, x_label: str, y_label: str, out_path: str):
    import csv
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([x_label, y_label])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])


def synth_image_crack(width: int, height: int, severity: float, out_path: str):
    img = Image.new('RGB', (width, height), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    # Draw random crack lines, severity controls thickness and count
    num_cracks = int(3 + severity * 7)
    for _ in range(num_cracks):
        x = int(RNG.uniform(0, width))
        y = 0
        points = [(x, y)]
        for step in range(20):
            x += int(RNG.normal(0, 20))
            y += int(height / 20)
            x = max(0, min(width - 1, x))
            points.append((x, y))
        thickness = max(1, int(1 + severity * RNG.uniform(1, 4)))
        draw.line(points, fill=(30, 30, 30), width=thickness)
    img.save(out_path)


def synth_spalling_mask(width: int, height: int, area_pct: float, out_path: str):
    # Create a grayscale mask roughly matching target coverage
    img = Image.new('L', (width, height), color=0)
    draw = ImageDraw.Draw(img)
    target_pixels = int(width * height * (area_pct / 100.0))
    drawn_pixels = 0
    # Draw random blotches until approximate coverage is reached
    while drawn_pixels < target_pixels:
        cx = int(RNG.uniform(0, width))
        cy = int(RNG.uniform(0, height))
        rx = int(RNG.uniform(20, 120))
        ry = int(RNG.uniform(15, 90))
        angle = RNG.uniform(0, 360)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=255)
        drawn_pixels = int(np.array(img).sum() / 255)
    # Save as PNG; white = spalled area
    img.convert('RGB').save(out_path)


def fabricate_dataset(config: Dict):
    out_dir = config['output_dir']
    ensure_dir(out_dir)

    meta = {
        'protocol': {
            'ages_days': config['ages_days'],
            'target_temperatures_c': config['target_temperatures_c'],
            'heating_rate_c_per_min': config['heating_rate_c_per_min'],
            'soak_time_min': config['soak_time_min'],
            'cooling_methods': config['cooling_methods'],
            'replicates_per_condition': config['replicates_per_condition'],
            'insitu_target_temperatures_c': config['insitu_target_temperatures_c'],
        },
        'disclaimer': 'Synthetic, physically-informed dataset for research & modeling prototyping.'
    }

    mixes = sample_mix_designs(config['n_mix_designs'])

    rows_ambient = []
    rows_residual = []
    rows_insitu = []
    rows_spalling = []

    ambient_dir = os.path.join(out_dir, 'ambient')
    residual_dir = os.path.join(out_dir, 'residual')
    insitu_dir = os.path.join(out_dir, 'insitu')
    images_dir = os.path.join(out_dir, 'images')
    ensure_dir(ambient_dir)
    ensure_dir(residual_dir)
    ensure_dir(insitu_dir)
    ensure_dir(images_dir)

    # Ambient tests per mix and age
    for mix in mixes:
        f28 = predict_f28_from_mix(mix)
        density = density_from_mix(mix)
        nu_amb = poisson_ratio_from_mix(mix)
        for age in config['ages_days']:
            fc = f28 * age_gain_factor(age)
            fc = float(max(10.0, fc + RNG.normal(0, 1.5)))
            method = 'flexural' if config.get('make_flexural_instead_of_splitting', False) else 'split'
            ft = tensile_from_compressive(fc, 'split' if method == 'split' else 'flex')
            Ec = modulus_from_compressive(fc, density)
            upv = upv_from_quality(fc, density)
            rows_ambient.append({
                'mix_id': mix.mix_id,
                'age_days': age,
                'rubber_pct_vol': mix.rubber_pct_vol,
                'w_c_ratio': mix.w_c_ratio,
                'silica_fume_pct': mix.silica_fume_pct,
                'fiber_pct_vol': mix.fiber_pct_vol,
                'air_content_pct': mix.air_content_pct,
                'density_kg_m3': density,
                'compressive_strength_mpa': fc,
                'tensile_method': method,
                'tensile_strength_mpa': ft,
                'static_modulus_gpa': Ec/1000.0,
                'upv_km_s': upv,
                'poisson_ratio': nu_amb,
            })

    # High-temperature residual tests
    for mix in mixes:
        f28 = predict_f28_from_mix(mix)
        density = density_from_mix(mix)
        nu_amb = poisson_ratio_from_mix(mix)
        method = 'flexural' if config.get('make_flexural_instead_of_splitting', False) else 'split'
        upv_amb = upv_from_quality(f28, density)
        for temp in config['target_temperatures_c']:
            for cooling in config['cooling_methods']:
                for rep in range(config['replicates_per_condition']):
                    rf = residual_factor_temp(temp, mix.rubber_pct_vol, cooling)
                    fc_res = max(2.0, f28 * rf)
                    ft_res = tensile_from_compressive(fc_res, 'split' if method == 'split' else 'flex')
                    Ec_res = modulus_from_compressive(fc_res, density)
                    upv_res = upv_residual(upv_amb, temp, cooling)
                    ml = mass_loss_pct(temp, mix.rubber_pct_vol, cooling)
                    nu_res = residual_poisson_ratio(nu_amb, rf)
                    rows_residual.append({
                        'mix_id': mix.mix_id,
                        'temp_c': temp,
                        'cooling': cooling,
                        'replicate': rep+1,
                        'rubber_pct_vol': mix.rubber_pct_vol,
                        'compressive_strength_residual_mpa': fc_res,
                        'tensile_method': method,
                        'tensile_strength_residual_mpa': ft_res,
                        'static_modulus_residual_gpa': Ec_res/1000.0,
                        'upv_residual_km_s': upv_res,
                        'mass_loss_pct': ml,
                        'poisson_ratio_residual': nu_res,
                    })
                    # Curves and images
                    if config.get('make_curves_png', True):
                        strain, stress = stress_strain_curve(fc_res, Ec_res)
                        curve_dir = os.path.join(residual_dir, mix.mix_id, f"{temp}C_{cooling}")
                        ensure_dir(curve_dir)
                        save_plot(strain, stress, 'Strain', 'Stress (MPa)', f'{mix.mix_id} {temp}C {cooling}', os.path.join(curve_dir, f'stress_strain_rep{rep+1}.png'))
                        # Also save CSV for the curve
                        save_xy_csv(strain, stress, 'strain', 'stress_MPa', os.path.join(curve_dir, f'stress_strain_rep{rep+1}.csv'))
                    if config.get('make_images', True):
                        sev = float(np.clip(1.0 - rf, 0.0, 1.0))
                        w, h = config.get('image_resolution', [1280, 960])
                        img_dir = os.path.join(images_dir, 'residual', mix.mix_id, f"{temp}C_{cooling}")
                        ensure_dir(img_dir)
                        synth_image_crack(w, h, sev, os.path.join(img_dir, f'specimen_rep{rep+1}.png'))
                        # Spalling mask image driven by a synthetic area estimate
                        # Use a quick estimate close to expected area; replicate variability via noise
                        sp_area = float(np.clip(2 + 0.03 * temp + (3 if cooling=='quench' else 0) + 15 * max(0, mix.rubber_pct_vol - 0.25) + RNG.normal(0, 2.0), 0, 100))
                        synth_spalling_mask(w, h, sp_area, os.path.join(img_dir, f'spalling_mask_rep{rep+1}.png'))

                    # Spalling quantitative metrics per replicate
                    depth_mm = max(0.0, -3 + 0.02 * temp + (5 if cooling=='quench' else 0) + 20 * max(0, mix.rubber_pct_vol - 0.2) + RNG.normal(0, 1.5))
                    area_pct = float(np.clip(2 + 0.03 * temp + (3 if cooling=='quench' else 0) + 15 * max(0, mix.rubber_pct_vol - 0.25) + RNG.normal(0, 2.0), 0, 100))
                    rows_spalling.append({
                        'mix_id': mix.mix_id,
                        'temp_c': temp,
                        'cooling': cooling,
                        'replicate': rep+1,
                        'spalling_depth_mm': depth_mm,
                        'spalling_area_pct': area_pct,
                    })

    # In-situ tests
    for mix in mixes:
        f28 = predict_f28_from_mix(mix)
        density = density_from_mix(mix)
        nu_amb = poisson_ratio_from_mix(mix)
        for temp in config['insitu_target_temperatures_c']:
            # Transient thermal strain under load
            T, eps = transient_thermal_strain_curve(temp, config.get('load_ratio_for_transient_strain', 0.2))
            curve_dir = os.path.join(insitu_dir, mix.mix_id, f"{temp}C")
            ensure_dir(curve_dir)
            save_plot(T, eps, 'Temperature (C)', 'Strain', f'{mix.mix_id} Transient Thermal Strain {temp}C', os.path.join(curve_dir, 'transient_strain.png'))
            save_xy_csv(T, eps, 'temperature_C', 'strain', os.path.join(curve_dir, 'transient_strain.csv'))
            # Thermal expansion (dilatometry)
            Tcte, cte = cte_curve(temp)
            save_plot(Tcte, cte, 'Temperature (C)', 'CTE (1/C)', f'{mix.mix_id} CTE up to {temp}C', os.path.join(curve_dir, 'cte.png'))
            save_xy_csv(Tcte, cte, 'temperature_C', 'cte_1_per_C', os.path.join(curve_dir, 'cte.csv'))
            # Pore pressure (optional)
            if config.get('include_pore_pressure', True):
                Tp, P = pore_pressure_curve(temp, mix.rubber_pct_vol)
                save_plot(Tp, P, 'Temperature (C)', 'Pore Pressure (MPa)', f'{mix.mix_id} Pore Pressure up to {temp}C', os.path.join(curve_dir, 'pore_pressure.png'))
                save_xy_csv(Tp, P, 'temperature_C', 'pore_pressure_MPa', os.path.join(curve_dir, 'pore_pressure.csv'))

            # In-situ properties at temperature
            rf_insitu = insitu_strength_factor(temp, mix.rubber_pct_vol)
            fc_insitu = max(1.0, f28 * rf_insitu)
            Ec_insitu = modulus_from_compressive(fc_insitu, density)
            nu_insitu = residual_poisson_ratio(nu_amb, rf_insitu)
            rows_insitu.append({
                'mix_id': mix.mix_id,
                'temp_c': temp,
                'insitu_compressive_strength_mpa': fc_insitu,
                'insitu_static_modulus_gpa': Ec_insitu/1000.0,
                'insitu_poisson_ratio': nu_insitu,
                'transient_strain_at_temp': float(eps[-1]),
                'cte_mean_1_per_c': float(np.mean(cte)),
                'load_ratio': config.get('load_ratio_for_transient_strain', 0.2),
            })

    # Spalling behavior summary per condition (synthetic metrics)
    for mix in mixes:
        for temp in config['target_temperatures_c']:
            for cooling in config['cooling_methods']:
                # Spalling depth increases with temp and quench; rubber can increase at high temp
                depth_mm = max(0.0, -3 + 0.02 * temp + (5 if cooling=='quench' else 0) + 20 * max(0, mix.rubber_pct_vol - 0.2) + RNG.normal(0, 1.5))
                area_pct = float(np.clip(2 + 0.03 * temp + (3 if cooling=='quench' else 0) + 15 * max(0, mix.rubber_pct_vol - 0.25) + RNG.normal(0, 1.0), 0, 100))
                rows_spalling.append({
                    'mix_id': mix.mix_id,
                    'temp_c': temp,
                    'cooling': cooling,
                    'spalling_depth_mm': depth_mm,
                    'spalling_area_pct': area_pct,
                })

    # Write CSVs without pandas
    import csv
    ambient_csv = os.path.join(ambient_dir, 'ambient_properties.csv')
    residual_csv = os.path.join(residual_dir, 'residual_properties.csv')
    spalling_csv = os.path.join(residual_dir, 'spalling_summary.csv')
    insitu_csv = os.path.join(insitu_dir, 'insitu_properties.csv')

    def write_rows(path: str, rows: List[Dict]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not rows:
            with open(path, 'w', newline='') as f:
                f.write('')
            return
        fieldnames = list(rows[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(ambient_csv, rows_ambient)
    write_rows(residual_csv, rows_residual)
    write_rows(spalling_csv, rows_spalling)
    write_rows(insitu_csv, rows_insitu)

    # Metadata
    meta_path = os.path.join(out_dir, 'metadata.yaml')
    with open(meta_path, 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    # Summary JSON for quick programmatic discovery
    summary = {
        'ambient_csv': os.path.relpath(ambient_csv, out_dir),
        'residual_csv': os.path.relpath(residual_csv, out_dir),
        'spalling_csv': os.path.relpath(spalling_csv, out_dir),
        'insitu_csv': os.path.relpath(insitu_csv, out_dir),
        'image_dirs': [
            os.path.relpath(os.path.join(images_dir, 'residual'), out_dir)
        ],
        'insitu_dir': os.path.relpath(insitu_dir, out_dir),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic HPRC fire dataset')
    parser.add_argument('--config', type=str, default='dataset_generator/config.yaml')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.out:
        config['output_dir'] = args.out
    if args.seed is not None:
        global RNG
        RNG = np.random.default_rng(args.seed)

    fabricate_dataset(config)

    # Zip the directory for convenience
    out_dir = config['output_dir']
    zip_path = os.path.join(os.path.dirname(out_dir), 'hprc_dataset')
    # create zip without extension, shutil will add it
    import shutil
    shutil.make_archive(zip_path, 'zip', out_dir)
    print(f"Dataset generated at: {out_dir}")
    print(f"Zipped at: {zip_path}.zip")


if __name__ == '__main__':
    main()
