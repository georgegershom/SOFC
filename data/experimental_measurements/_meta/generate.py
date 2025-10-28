#!/usr/bin/env python3
import csv
import json
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path('/workspace/data/experimental_measurements')
DIC_DIR = ROOT / 'DIC'
XRD_DIR = ROOT / 'XRD'
PM_DIR = ROOT / 'PostMortem'
META_DIR = ROOT / '_meta'

random.seed(42)


def ensure_dirs():
    for p in [
        DIC_DIR,
        XRD_DIR,
        PM_DIR,
        DIC_DIR / 'speckle',
        DIC_DIR / 'images',
        XRD_DIR / 'images',
        PM_DIR / 'images',
        META_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def generate_speckle_svg(path: Path, width=800, height=600, num_dots=2000, radius_range=(1, 3)):
    random.seed(hash(path) % (2**32))
    dots = []
    for _ in range(num_dots):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        r = random.uniform(*radius_range)
        shade = random.randint(20, 235)
        dots.append((x, y, r, shade))
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
    ]
    for x, y, r, s in dots:
        svg.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{r:.2f}' fill='rgb({s},{s},{s})' />")
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def synth_strain(x: float, y: float, t_norm: float, mode: str = 'sintering'):
    # x,y in [0,1], t_norm in [0,1]
    base = 0.002 * (1 + 0.5 * math.sin(2 * math.pi * (x + y + t_norm)))
    grad_x = 0.006 * (x - 0.5)
    grad_y = -0.004 * (y - 0.5)
    if mode == 'sintering':
        thermo = 0.008 * t_norm
    elif mode == 'cycling':
        thermo = 0.010 * math.sin(2 * math.pi * t_norm)
    else:  # startup_shutdown
        thermo = 0.009 * (1 - abs(2 * t_norm - 1))
    exx = base + grad_x + 0.5 * thermo
    eyy = base + grad_y + 0.4 * thermo
    exy = 0.2 * base + 0.1 * thermo * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)
    # Localized hotspot near interface band
    hotspot = 1 if (0.45 < y < 0.55 and exx + eyy > 0.02) else 0
    return exx, eyy, exy, hotspot


def generate_dic_case(case_name: str, n_time: int, temp_range, mode: str):
    grid_n = 25
    rows = []
    start_ts = datetime(2025, 1, 1, 12, 0, 0)
    for ti in range(n_time):
        t_norm = ti / (n_time - 1) if n_time > 1 else 0.0
        temperature = temp_range[0] + (temp_range[1] - temp_range[0]) * t_norm
        ts = start_ts + timedelta(seconds=5 * ti)
        for iy in range(grid_n):
            for ix in range(grid_n):
                x = ix / (grid_n - 1)
                y = iy / (grid_n - 1)
                exx, eyy, exy, hotspot = synth_strain(x, y, t_norm, mode)
                rows.append([
                    ts.isoformat(),
                    f"{temperature:.1f}",
                    ix,
                    iy,
                    f"{exx:.5f}",
                    f"{eyy:.5f}",
                    f"{exy:.5f}",
                    'interface' if 0.45 < y < 0.55 else 'bulk',
                    hotspot,
                ])
    header = ['timestamp_iso', 'temperature_C', 'x_px', 'y_px', 'exx', 'eyy', 'exy', 'region', 'hotspot_flag']
    out_csv = DIC_DIR / f"{case_name}_strain_map.csv"
    write_csv(out_csv, header, rows)


def generate_dic():
    generate_dic_case('sintering', n_time=8, temp_range=(1200, 1500), mode='sintering')
    generate_dic_case('thermal_cycling', n_time=12, temp_range=(200, 600), mode='cycling')
    generate_dic_case('startup_shutdown', n_time=6, temp_range=(25, 800), mode='startup')
    # hotspots summary
    summary = []
    for name in ['sintering', 'thermal_cycling', 'startup_shutdown']:
        path = DIC_DIR / f"{name}_strain_map.csv"
        hotspot_count = 0
        total = 0
        with open(path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                total += 1
                hotspot_count += int(row['hotspot_flag'])
        summary.append([name, total, hotspot_count, f"{(100 * hotspot_count / max(total, 1)):.2f}"])
    write_csv(DIC_DIR / 'hotspots_summary.csv', ['case', 'num_points', 'hotspot_points', 'hotspot_percent'], summary)
    # speckle images with timestamps
    for i in range(3):
        ts = datetime(2025, 1, 1, 12, 0, 0) + timedelta(seconds=10 * i)
        sp = DIC_DIR / 'speckle' / f"speckle_{ts.strftime('%Y%m%dT%H%M%S')}.svg"
        generate_speckle_svg(sp, width=640, height=480, num_dots=1500)


def generate_xrd():
    # Residual stress profiles across cross-section depths
    rows = []
    for depth_um in range(0, 501, 10):
        # Simulate compressive at surface transitioning to tensile
        sigma_xx = -180 * math.exp(-depth_um / 100) + 60 * (1 - math.exp(-depth_um / 200))
        sigma_yy = -120 * math.exp(-depth_um / 120) + 40 * (1 - math.exp(-depth_um / 250))
        sigma_xy = 10 * math.sin(depth_um / 80)
        rows.append([depth_um, f"{sigma_xx:.1f}", f"{sigma_yy:.1f}", f"{sigma_xy:.1f}"])
    write_csv(
        XRD_DIR / 'residual_stress_profile.csv',
        ['depth_um', 'sigma_xx_MPa', 'sigma_yy_MPa', 'sigma_xy_MPa'],
        rows,
    )

    # Lattice strain measurements under thermal load for phases
    temps = [25 + i * 50 for i in range(0, 17)]  # up to 825 C
    phases = ['YSZ', 'Ni-YSZ']
    rows = []
    for phase in phases:
        coef = 1.1e-5 if phase == 'YSZ' else 1.4e-5
        for T in temps:
            eps = coef * (T - 25)
            anis = 0.2e-5 * math.sin(T / 80)
            rows.append([phase, T, f"{(eps + anis):.6f}", f"{(eps - anis):.6f}"])
    write_csv(XRD_DIR / 'lattice_strain_vs_temp.csv', ['phase', 'temperature_C', 'epsilon_11', 'epsilon_22'], rows)

    # sin^2 psi peak shift data
    rows = []
    for i in range(0, 51):
        psi = -30 + i * (60 / 50)
        phi = 0
        sin2psi = math.sin(math.radians(psi)) ** 2
        stress = 50 + 120 * sin2psi  # linear dependence for demo
        d0 = 2.314
        d = d0 * (1 + stress * 1e-6)  # simplistic elastics
        try:
            two_theta = 2 * math.degrees(math.asin(1.5406 / (2 * d)))  # Cu Kalpha
        except ValueError:
            two_theta = float('nan')
        rows.append([
            f"{sin2psi:.4f}",
            f"{d:.5f}",
            f"{two_theta:.3f}",
            f"{psi:.2f}",
            f"{phi:.2f}",
            f"{stress:.1f}",
        ])
    write_csv(
        XRD_DIR / 'sin2psi_peak_shift.csv',
        ['sin2psi', 'd_spacing_A', 'two_theta_deg', 'psi_deg', 'phi_deg', 'stress_MPa_calc'],
        rows,
    )

    # Microcrack thresholds
    rows = [
        ['YSZ', '0.020', 'Onset near electrolyte/cathode interface'],
        ['Ni-YSZ', '0.015', 'Lower threshold due to porosity'],
        ['Cathode', '0.018', 'Affected by microstructural gradients'],
    ]
    write_csv(XRD_DIR / 'microcrack_thresholds.csv', ['material', 'epsilon_cr', 'notes'], rows)


def generate_sem_svg(path: Path, width=1024, height=768, num_cracks=120):
    random.seed(hash(path) % (2**32))
    lines = []
    for _ in range(num_cracks):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        length = random.uniform(20, 150)
        angle = random.uniform(0, math.pi)
        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle)
        stroke = random.uniform(0.8, 1.8)
        lines.append((x, y, x2, y2, stroke))
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='rgb(230,230,230)'/>",
    ]
    for x1, y1, x2, y2, st in lines:
        svg.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='black' stroke-width='{st:.2f}' stroke-linecap='round' opacity='0.9' />"
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def generate_postmortem():
    # SEM crack images and density summary
    density_rows = []
    for i, area_mm2 in enumerate([0.5, 0.75, 1.0]):
        img_path = PM_DIR / 'images' / f"SEM_{i + 1:02d}.svg"
        crack_count = random.randint(60, 180)
        generate_sem_svg(img_path, num_cracks=crack_count)
        density_rows.append([img_path.name, f"{area_mm2:.2f}", crack_count, f"{crack_count / area_mm2:.1f}"])
    write_csv(PM_DIR / 'crack_density.csv', ['image_id', 'area_mm2', 'crack_count', 'cracks_per_mm2'], density_rows)

    # EDS line scans across anode-electrolyte interface (0-500 um)
    rows = []
    for i in range(0, 201):
        pos = i * 2.5  # um
        # Smooth transition: Ni decreases, Zr increases, Y low constant, O complement
        Ni = max(0.0, 35 - 0.10 * pos + 2.0 * math.sin(pos / 40))
        Zr = min(60.0, 15 + 0.12 * pos + 1.0 * math.cos(pos / 50))
        Y = 7.0 + 0.3 * math.sin(pos / 60)
        O = max(0.0, 100 - (Ni + Zr + Y))
        rows.append([f"{pos:.1f}", f"{Ni:.2f}", f"{Zr:.2f}", f"{Y:.2f}", f"{O:.2f}"])
    write_csv(PM_DIR / 'eds_line_scan.csv', ['position_um', 'Ni_at_pct', 'Zr_at_pct', 'Y_at_pct', 'O_at_pct'], rows)

    # Nanoindentation modulus maps
    def modulus_field(mean: float, std: float, n: int = 20):
        out = []
        for gy in range(n):
            for gx in range(n):
                val = random.gauss(mean, std) + 2.0 * math.sin(gx / 4) - 1.5 * math.cos(gy / 5)
                out.append([gx, gy, f"{val:.2f}"])
        return out

    write_csv(PM_DIR / 'nanoindentation_modulus_map_YSZ.csv', ['grid_x', 'grid_y', 'E_GPa'], modulus_field(184.7, 6.0))
    write_csv(PM_DIR / 'nanoindentation_modulus_map_NiYSZ.csv', ['grid_x', 'grid_y', 'E_GPa'], modulus_field(109.8, 5.0))

    # Hardness and creep compliance
    rows = []
    for material, meanH, stdH in [('YSZ', 12.5, 0.8), ('Ni-YSZ', 6.8, 0.6)]:
        for gy in range(15):
            for gx in range(15):
                H = max(0.5, random.gauss(meanH, stdH))
                creep = 1.0 / (H + (2 if material == 'YSZ' else 1)) + 0.02 * random.random()
                rows.append([material, gx, gy, f"{H:.2f}", f"{creep:.4f}"])
    write_csv(PM_DIR / 'nanoindentation_hardness_creep.csv', ['material', 'grid_x', 'grid_y', 'hardness_GPa', 'creep_compliance_1GPa'], rows)


def update_manifest():
    manifest = {
        'name': 'SOFC Experimental Measurements (Synthetic)',
        'version': '0.1.0',
        'generated_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'components': ['DIC', 'XRD', 'PostMortem'],
        'files': [],
    }
    for base in [DIC_DIR, XRD_DIR, PM_DIR]:
        for p in base.rglob('*'):
            if p.is_file():
                rel = str(p.relative_to(ROOT))
                size = p.stat().st_size
                manifest['files'].append({'path': rel, 'bytes': size})
    (META_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def main():
    ensure_dirs()
    generate_dic()
    generate_xrd()
    generate_postmortem()
    update_manifest()
    print('Synthetic dataset generated at', ROOT)


if __name__ == '__main__':
    main()

