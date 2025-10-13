import os
import json
import numpy as np
import h5py
from typing import Dict, Any, Tuple
from .utils import ensure_dir, latin_hypercube

# Synthetic multi-physics surrogate using analytical fields + PDE-inspired couplings.
# Not a full solver; captures correlations between operating/material parameters and fields.

def sample_params(n: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lh = latin_hypercube(n, d=12, rng=rng)
    # Map LHS [0,1] -> physical ranges
    params = {
        'current_density_A_per_cm2': 0.1 + lh[:,0] * (2.0 - 0.1),
        'fuel_utilization_frac': 0.4 + lh[:,1] * (0.9 - 0.4),
        'air_utilization_frac': 0.2 + lh[:,2] * (0.7 - 0.2),
        'inlet_fuel_temp_C': 500 + lh[:,3] * (800 - 500),
        'inlet_air_temp_C': 500 + lh[:,4] * (850 - 500),
        'fuel_H2_frac': 0.5 + lh[:,5] * (0.95 - 0.5),
        'fuel_H2O_frac': 0.0 + lh[:,6] * (0.4 - 0.0),
        'electrode_porosity': 0.25 + lh[:,7] * (0.45 - 0.25),
        'electrode_tortuosity': 2.0 + lh[:,8] * (5.0 - 2.0),
        'electrode_thickness_um': 10 + lh[:,9] * (80 - 10),
        'electrolyte_thickness_um': 5 + lh[:,10] * (30 - 5),
        'interconnect_young_GPa': 150 + lh[:,11] * (230 - 150),
    }
    # Normalize fuel fractions
    h2 = params['fuel_H2_frac']
    h2o = params['fuel_H2O_frac']
    co = np.maximum(0.0, 1.0 - (h2 + h2o)) * 0.3
    ch4 = np.maximum(0.0, 1.0 - (h2 + h2o + co))
    params['fuel_CO_frac'] = co
    params['fuel_CH4_frac'] = ch4
    return params

def generate_fields(grid: Tuple[int,int,int], p: Dict[str, float], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    nx, ny, nz = grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    jd = p['current_density_A_per_cm2']
    fu = p['fuel_utilization_frac']
    au = p['air_utilization_frac']
    Tf = p['inlet_fuel_temp_C'] + 273.15
    Ta = p['inlet_air_temp_C'] + 273.15
    por = p['electrode_porosity']
    tau = p['electrode_tortuosity']
    t_el = p['electrolyte_thickness_um'] * 1e-6
    E_ic = p['interconnect_young_GPa'] * 1e9

    # Temperature field: baseline + ohmic heating proportional to current density and electrolyte thickness
    T = 800 + 80 * jd * (1 + 0.5*np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)) + 0.2*(Tf-700) + 0.1*(Ta-700)
    T += 5*np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(0.05 + 0.1*fu))
    T += rng.normal(0, 0.5, size=T.shape)  # small sensor-like noise

    # Current density distribution: peak near fuel inlet (x=0), modulated by porosity/tortuosity
    i_dist = jd * (1.2 - 0.4*X) * (0.8 + 0.4*por) / (1 + 0.2*(tau-2))
    i_dist *= (1 - 0.1*np.sin(2*np.pi*Y)*np.sin(2*np.pi*Z))

    # Species: H2 decreases along x, H2O increases; O2 decreases along y
    H2 = np.clip( p['fuel_H2_frac'] * np.exp(-2*X*(0.6+0.8*fu)), 0, 1)
    H2O = np.clip( p['fuel_H2O_frac'] + 0.4*(1-np.exp(-2*X*(0.6+0.8*fu))), 0, 1)
    O2 = np.clip( 0.21 * np.exp(-1.5*Y*(0.5+0.5*au)), 0, 0.21)

    # Voltage: simplified Nernst - ohmic - activation
    ohmic = 0.2 * jd * (t_el/1e-5) / max(1e-3, por/(tau))
    activation = 0.05 * np.log1p(jd)
    V = 1.1 - ohmic - activation - 0.02*fu - 0.01*au + rng.normal(0, 0.002)

    # Displacement and strain/stress: thermal expansion + mech loading surrogate
    alpha = 1e-5  # thermal expansion coeff
    Ux = alpha*(T-800) * X
    Uy = alpha*(T-800) * Y
    Uz = 0.3*alpha*(T-800) * (1-Z)

    # Compute strains via gradients
    def grad(f, axis):
        return np.gradient(f, axis=axis, edge_order=2)
    exx = grad(Ux, 0)
    eyy = grad(Uy, 1)
    ezz = grad(Uz, 2)
    exy = 0.5*(grad(Ux,1)+grad(Uy,0))
    eyz = 0.5*(grad(Uy,2)+grad(Uz,1))
    ezx = 0.5*(grad(Uz,0)+grad(Ux,2))

    # Hooke's law (isotropic) surrogate for stress; use E varying with E_ic
    poisson = 0.3
    E = 200e9 * (E_ic/200e9)
    lam = (poisson*E)/((1+poisson)*(1-2*poisson))
    mu = E/(2*(1+poisson))
    trace = exx + eyy + ezz
    s_xx = 2*mu*exx + lam*trace
    s_yy = 2*mu*eyy + lam*trace
    s_zz = 2*mu*ezz + lam*trace
    s_xy = 2*mu*exy
    s_yz = 2*mu*eyz
    s_zx = 2*mu*ezx

    # Von Mises
    von = np.sqrt(0.5*((s_xx-s_yy)**2 + (s_yy-s_zz)**2 + (s_zz-s_xx)**2 + 6*(s_xy**2 + s_yz**2 + s_zx**2)))

    # Fracture metrics (surrogates)
    K_I = 1e6 * np.max(von) * (0.1 + 0.9*np.random.random())
    K_II = 0.7*K_I
    K_III = 0.5*K_I
    G = (K_I**2) * (1 - poisson**2) / E

    return {
        'T': T.astype(np.float32),
        'i': i_dist.astype(np.float32),
        'H2': H2.astype(np.float32),
        'H2O': H2O.astype(np.float32),
        'O2': O2.astype(np.float32),
        'Ux': Ux.astype(np.float32),
        'Uy': Uy.astype(np.float32),
        'Uz': Uz.astype(np.float32),
        's_xx': s_xx.astype(np.float32),
        's_yy': s_yy.astype(np.float32),
        's_zz': s_zz.astype(np.float32),
        's_xy': s_xy.astype(np.float32),
        's_yz': s_yz.astype(np.float32),
        's_zx': s_zx.astype(np.float32),
        'e_xx': exx.astype(np.float32),
        'e_yy': eyy.astype(np.float32),
        'e_zz': ezz.astype(np.float32),
        'e_xy': exy.astype(np.float32),
        'e_yz': eyz.astype(np.float32),
        'e_zx': ezx.astype(np.float32),
        'von_mises': von.astype(np.float32),
        'V': np.float32(V),
        'K_I': np.float32(K_I),
        'K_II': np.float32(K_II),
        'K_III': np.float32(K_III),
        'G': np.float32(G),
    }

def write_h5(path: str, fields: Dict[str, Any], attrs: Dict[str, Any]):
    with h5py.File(path, 'w') as f:
        for k, v in fields.items():
            f.create_dataset(k, data=v, compression='gzip')
        for k, v in attrs.items():
            f.attrs[k] = v

def generate_dataset1(out_dir: str, n_sims: int, seed: int, grid: Tuple[int,int,int]):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    params_all = sample_params(n_sims, seed)
    meta = []
    for idx in range(n_sims):
        p = {k: float(params_all[k][idx]) for k in params_all}
        fields = generate_fields(grid, p, rng)
        fname = os.path.join(out_dir, f'sim_{idx:04d}.h5')
        write_h5(fname, fields, p)
        meta.append({'file': os.path.basename(fname), **p})
    with open(os.path.join(out_dir, 'cases.json'), 'w') as fp:
        json.dump(meta, fp, indent=2)
    return {'count': n_sims, 'grid': grid, 'dir': out_dir}
