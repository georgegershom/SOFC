#!/usr/bin/env python3
import csv
import json
import math
import os
import random
from pathlib import Path

BASE = Path('/workspace/datasets/validation_analysis')
random.seed(42)

UNITS = {
    "length": "mm",
    "stress": "MPa",
    "strain": "-",
    "temperature": "C",
    "time": "s",
    "displacement": "mm",
    "angle": "deg",
}

META = {
    "dataset": "Validation & Analysis (fabricated)",
    "version": "1.0.0",
    "provenance": {
        "generator": "scripts/generate_validation_dataset.py",
        "random_seed": 42,
        "date": None,
    },
    "assumptions": [
        "Electrolyte ceramic, isotropic at macro-scale",
        "Linear thermoelasticity for FEM outputs",
        "Residual stresses arise from CTE mismatch during sintering cool-down",
        "Meso-scale RVE captures porosity & grain size distributions",
        "Micro-scale GB properties are simplified surrogates",
    ],
}


def ensure_dirs():
    for d in [
        BASE / 'macro', BASE / 'meso', BASE / 'micro', BASE / 'crack',
        BASE / 'simulation', BASE / 'collocation', BASE / 'docs']:
        d.mkdir(parents=True, exist_ok=True)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def gen_macro():
    # Material properties (plausible ceramic electrolyte values)
    rows = [[200000.0, 0.28, 10.5e-6, 5.6, 2.0, 0.75]]  # E[MPa], nu, CTE[1/C], density[g/cc], k[W/mK], cp[J/gK]
    write_csv(BASE / 'macro' / 'material_properties.csv',
              ['E_MPa', 'nu', 'CTE_1_per_C', 'density_g_per_cc', 'k_W_per_mK', 'cp_J_per_gK'], rows)

    # Cell geometry (mm)
    rows = [["cell_A", 50.0, 50.0, 0.5]]
    write_csv(BASE / 'macro' / 'cell_geometry.csv', ['cell_id', 'width_mm', 'height_mm', 'thickness_mm'], rows)

    # Sintering temperature profile (cooling)
    times = [i * 60 for i in range(0, 121)]  # 0..120 min in seconds
    T0 = 1200.0
    cool_rate = -3.5  # C per minute
    temps = [T0 + (t/60.0) * cool_rate + 5.0 * math.sin(2*math.pi*(t/3600.0)) for t in times]
    write_csv(BASE / 'macro' / 'sinter_profile.csv', ['time_s', 'T_C'], list(zip(times, temps)))


def gen_meso():
    # Microstructure stats
    rows = [["sample_A", 8.2, 0.06, 2.1]]  # mean grain size [um], porosity [-], std [um]
    write_csv(BASE / 'meso' / 'microstructure_stats.csv', ['sample_id', 'mean_grain_size_um', 'porosity', 'std_grain_size_um'], rows)

    # RVE voxel grid (small for demo)
    voxel_rows = []
    size = 12
    for x in range(size):
        for y in range(size):
            for z in range(size):
                por = max(0.0, random.gauss(0.06, 0.02))
                phase = 'pore' if random.random() < por else 'solid'
                voxel_rows.append([x, y, z, phase, round(por, 4)])
    write_csv(BASE / 'meso' / 'rve_voxels.csv', ['x', 'y', 'z', 'phase', 'local_porosity'], voxel_rows)


def gen_micro():
    # Grain boundary properties (fabricated)
    gb_rows = []
    for gb_id in range(1, 151):
        mis = round(random.uniform(2, 35), 2)  # deg
        cohesion = round(random.uniform(0.5, 1.5), 3)  # a.u.
        diff = f"{random.uniform(1e-18,5e-17):.2e}"
        gb_rows.append([gb_id, mis, cohesion, diff])
    write_csv(BASE / 'micro' / 'grain_boundary_props.csv', ['gb_id', 'misorientation_deg', 'cohesion_energy_au', 'diffusivity_m2_per_s'], gb_rows)

    # EBSD-like orientations for grains
    ebsd_rows = []
    for gid in range(1, 101):
        phi1, bigPhi, phi2 = [round(random.uniform(0, 360), 2) for _ in range(3)]
        ebsd_rows.append([gid, phi1, bigPhi, phi2])
    write_csv(BASE / 'micro' / 'ebsd_orientations.csv', ['grain_id', 'phi1_deg', 'Phi_deg', 'phi2_deg'], ebsd_rows)


def fabricate_residual_field(x, y, z):
    # A smooth residual stress field with edge tension and center compression
    r = math.sqrt((x-25.0)**2 + (y-25.0)**2)
    sxx = -80.0 + 0.15 * r  # MPa
    syy = -75.0 + 0.12 * r
    szz = -50.0 + 0.05 * r + 0.2 * z
    sxy = 5.0 * math.sin(0.1*x) * math.cos(0.1*y)
    syz = 3.0 * math.sin(0.1*y) * math.cos(0.1*z)
    sxz = 2.0 * math.sin(0.1*x) * math.cos(0.1*z)
    return sxx, syy, szz, sxy, syz, sxz


def fabricate_strain_from_stress(sxx, syy, szz, E=200000.0, nu=0.28):
    # Simplified isotropic Hooke's law (approximate)
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    # invert linear elastic matrix in simplified way (not exact here)
    exx = (sxx - nu*(syy+szz))/E
    eyy = (syy - nu*(sxx+szz))/E
    ezz = (szz - nu*(sxx+syy))/E
    exy = 1.0/(2*mu) * 0.0
    eyz = 1.0/(2*mu) * 0.0
    exz = 1.0/(2*mu) * 0.0
    return exx, eyy, ezz, exy, eyz, exz


def gen_crack_and_experimental():
    # Experimental residual stress (macro XRD-like at surface z=0)
    xrd_rows = []
    for x in range(0, 51, 5):
        for y in range(0, 51, 5):
            sxx, syy, szz, sxy, syz, sxz = fabricate_residual_field(x, y, 0.0)
            # add measurement noise
            sxxn = sxx + random.gauss(0, 3.0)
            syyn = syy + random.gauss(0, 3.0)
            szzn = szz + random.gauss(0, 3.0)
            xrd_rows.append([x, y, 0.0, round(sxxn,2), round(syyn,2), round(szzn,2)])
    write_csv(BASE / 'macro' / 'surface_residual_stress_xrd.csv', ['x_mm', 'y_mm', 'z_mm', 'sxx_MPa', 'syy_MPa', 'szz_MPa'], xrd_rows)

    # Meso-scale bulk stress via synchrotron-like (sparse through-thickness)
    syn_rows = []
    for z in [0.0, 0.1, 0.25, 0.4, 0.5]:
        for x in [5, 15, 25, 35, 45]:
            for y in [5, 25, 45]:
                sxx, syy, szz, sxy, syz, sxz = fabricate_residual_field(x, y, z)
                syn_rows.append([x, y, z, round(sxx,2), round(syy,2), round(szz,2)])
    write_csv(BASE / 'meso' / 'bulk_residual_stress_synchrotron.csv', ['x_mm', 'y_mm', 'z_mm', 'sxx_MPa', 'syy_MPa', 'szz_MPa'], syn_rows)

    # Crack events (critical load/temperature)
    crack_rows = []
    event_id = 1
    for loc in [(12.5, 12.5, 0.1), (25.0, 25.0, 0.25), (37.5, 37.5, 0.1)]:
        Tcrit = round(random.uniform(150, 300), 1)
        loadcrit = round(random.uniform(20, 60), 1)  # MPa equiv
        stage = random.choice(["initiation", "propagation"])
        crack_rows.append([event_id, *loc, Tcrit, loadcrit, stage])
        event_id += 1
    write_csv(BASE / 'crack' / 'crack_events.csv', ['event_id', 'x_mm', 'y_mm', 'z_mm', 'Tcrit_C', 'loadcrit_MPa', 'stage'], crack_rows)

    # SEM micro-cracks
    sem_rows = []
    for cid in range(1, 61):
        x = round(random.uniform(0, 50), 2)
        y = round(random.uniform(0, 50), 2)
        z = round(random.uniform(0.0, 0.5), 3)
        length = round(random.uniform(2.0, 50.0), 2)  # microns
        orientation = round(random.uniform(0.0, 180.0), 1)
        sem_rows.append([cid, x, y, z, length, orientation])
    write_csv(BASE / 'crack' / 'sem_microcracks.csv', ['crack_id', 'x_mm', 'y_mm', 'z_mm', 'length_um', 'orientation_deg'], sem_rows)


def gen_simulation_full_field():
    # Generate a coarse node grid with T, U, sigma, epsilon
    node_rows = []
    node_id = 1
    for x in range(0, 51, 5):
        for y in range(0, 51, 5):
            for z in [0.0, 0.1, 0.25, 0.4, 0.5]:
                T = 25.0 + 0.01*(x-25) + 0.02*(y-25) - 5.0*z
                ux = 0.001*(x-25) - 0.0005*(y-25)
                uy = 0.0008*(y-25)
                uz = -0.0003*z
                sxx, syy, szz, sxy, syz, sxz = fabricate_residual_field(x, y, z)
                exx, eyy, ezz, exy, eyz, exz = fabricate_strain_from_stress(sxx, syy, szz)
                node_rows.append([node_id, x, y, z, round(T,2), round(ux,5), round(uy,5), round(uz,5),
                                  round(sxx,2), round(syy,2), round(szz,2), round(sxy,2), round(syz,2), round(sxz,2),
                                  round(exx,6), round(eyy,6), round(ezz,6), round(exy,6), round(eyz,6), round(exz,6)])
                node_id += 1
    write_csv(BASE / 'simulation' / 'full_field_nodes.csv',
              ['node_id','x_mm','y_mm','z_mm','T_C','ux_mm','uy_mm','uz_mm','sxx_MPa','syy_MPa','szz_MPa','sxy_MPa','syz_MPa','sxz_MPa','exx','eyy','ezz','exy','eyz','exz'],
              node_rows)

    # Elements summary (von Mises convenience)
    elem_rows = []
    elem_id = 1
    for x in range(2, 49, 5):
        for y in range(2, 49, 5):
            for z in [0.05, 0.2, 0.35, 0.45]:
                sxx, syy, szz, sxy, syz, sxz = fabricate_residual_field(x, y, z)
                sv = math.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2) + 3*(sxy**2+syz**2+sxz**2))
                ev = abs(sv)/200000.0
                elem_rows.append([elem_id, x, y, z, round(sv,2), round(ev,6)])
                elem_id += 1
    write_csv(BASE / 'simulation' / 'full_field_elems.csv', ['elem_id','centroid_x_mm','centroid_y_mm','centroid_z_mm','s_vm_MPa','e_vm'], elem_rows)


def gen_collocation_subset():
    # Strategically select points: near corners, center, and random pores from voxels
    points = []
    point_id = 1
    for (x, y, z, tag) in [
        (0.0, 0.0, 0.0, 'corner'), (50.0, 0.0, 0.0, 'corner'), (0.0, 50.0, 0.0, 'corner'), (50.0, 50.0, 0.0, 'corner'),
        (25.0, 25.0, 0.25, 'center'), (25.0, 5.0, 0.1, 'edge'), (5.0, 25.0, 0.4, 'edge')
    ]:
        points.append([point_id, x, y, z, tag])
        point_id += 1

    # Add a small random set of pore-tagged points from meso voxels
    vox_path = BASE / 'meso' / 'rve_voxels.csv'
    pore_candidates = []
    if vox_path.exists():
        # Read CSV and randomly sample up to 10 pore voxels without pandas
        with open(vox_path, 'r', newline='') as vf:
            reader = csv.DictReader(vf)
            pores = [row for row in reader if row.get('phase') == 'pore']
        rnd = random.Random(42)
        rnd.shuffle(pores)
        for r in pores[:min(10, len(pores))]:
            # Map voxel coords (0..11) into mm (0..0.5 thickness, 0..50 in x,y for illustration)
            x_mm = (float(r['x']) / 11.0) * 50.0
            y_mm = (float(r['y']) / 11.0) * 50.0
            z_mm = (float(r['z']) / 11.0) * 0.5
            points.append([point_id, round(x_mm,2), round(y_mm,2), round(z_mm,3), 'pore'])
            point_id += 1

    write_csv(BASE / 'collocation' / 'points.csv', ['point_id','x_mm','y_mm','z_mm','tag'], points)

    # Collocation values by sampling simulated field
    values = []
    for pid, x, y, z, tag in points:
        T = 25.0 + 0.01*(x-25) + 0.02*(y-25) - 5.0*z
        ux = 0.001*(x-25) - 0.0005*(y-25)
        uy = 0.0008*(y-25)
        uz = -0.0003*z
        sxx, syy, szz, sxy, syz, sxz = fabricate_residual_field(x, y, z)
        exx, eyy, ezz, exy, eyz, exz = fabricate_strain_from_stress(sxx, syy, szz)
        values.append([pid, round(T,2), round(ux,5), round(uy,5), round(uz,5),
                       round(sxx,2), round(syy,2), round(szz,2), round(sxy,2), round(syz,2), round(sxz,2),
                       round(exx,6), round(eyy,6), round(ezz,6), round(exy,6), round(eyz,6), round(exz,6)])

    write_csv(BASE / 'collocation' / 'values.csv',
              ['point_id','T_C','ux_mm','uy_mm','uz_mm','sxx_MPa','syy_MPa','szz_MPa','sxy_MPa','syz_MPa','sxz_MPa','exx','eyy','ezz','exy','eyz','exz'],
              values)


def write_docs_and_meta():
    # README
    readme = f"""
    Validation & Analysis Dataset (Fabricated)

    Units:
    - length: {UNITS['length']}
    - stress: {UNITS['stress']}
    - strain: {UNITS['strain']}
    - temperature: {UNITS['temperature']}
    - time: {UNITS['time']}
    - displacement: {UNITS['displacement']}
    - angle: {UNITS['angle']}

    Files:
    - macro: material properties, cell geometry, sintering profile, surface residual stress (XRD-like)
    - meso: microstructure stats, RVE voxels, bulk residual stress (synchrotron-like)
    - micro: grain boundary properties, EBSD-like orientations
    - crack: crack events (critical T/load), SEM micro-cracks
    - simulation: full-field nodes and element summaries
    - collocation: selected points and their values

    Note: All data are synthetic but physically plausible and consistent across files.
    """.strip()
    (BASE / 'docs' / 'README.md').write_text(readme)

    META['provenance']['date'] = __import__('datetime').datetime.utcnow().isoformat() + 'Z'
    META['units'] = UNITS
    (BASE / 'docs' / 'metadata.json').write_text(json.dumps(META, indent=2))


def main():
    ensure_dirs()
    gen_macro()
    gen_meso()
    gen_micro()
    gen_crack_and_experimental()
    gen_simulation_full_field()
    gen_collocation_subset()
    write_docs_and_meta()
    print('Dataset generated at', BASE)

if __name__ == '__main__':
    main()
