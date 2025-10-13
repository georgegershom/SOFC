import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from .utils import ensure_dir

def generate_eis(freqs, R0, Cdl, noise=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    w = 2*np.pi*freqs
    Z_re = R0 / (1 + (w*R0*Cdl)**2)
    Z_im = - (w*R0**2*Cdl) / (1 + (w*R0*Cdl)**2)
    Z_re += rng.normal(0, noise*R0, size=Z_re.shape)
    Z_im += rng.normal(0, noise*R0, size=Z_im.shape)
    return Z_re, Z_im

def generate_dataset2(out_dir: str, seed: int):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    # Global time-series
    T = 3600
    t = np.arange(T)
    I = 20 + 5*np.sin(2*np.pi*t/600) + rng.normal(0, 0.3, T)
    V = 0.8 - 0.02*np.sin(2*np.pi*t/900) - 0.00002*t + rng.normal(0, 0.002, T)
    P = I*V
    Tin = 700 + 10*np.sin(2*np.pi*t/1800)
    Fin = 1.0 + 0.1*np.sin(2*np.pi*t/1500)
    df = pd.DataFrame({'t': t, 'I': I, 'V': V, 'P': P, 'T_in': Tin, 'F_in': Fin})
    df.to_csv(os.path.join(out_dir, 'global_timeseries.csv'), index=False)

    # EIS every 600 s
    freqs = np.logspace(1, 5, 60)
    eis_list = []
    for k, ts in enumerate(range(0, T, 600)):
        R0 = 0.2 + 0.00005*ts
        Cdl = 1e-3*(1 + 0.2*np.sin(2*np.pi*ts/7200))
        zr, zi = generate_eis(freqs, R0, Cdl, rng=rng)
        np.savez(os.path.join(out_dir, f'eis_{k:03d}.npz'), freqs=freqs, Z_re=zr, Z_im=zi, t=ts)
        eis_list.append({'file': f'eis_{k:03d}.npz', 't': ts})

    # Thermal images (coarse) at a few timepoints
    imgs_meta = []
    for k, ts in enumerate([300, 1800, 3300]):
        img = 700 + 15*np.random.rand(64, 64)
        np.save(os.path.join(out_dir, f'thermal_{k:03d}.npy'), img)
        imgs_meta.append({'file': f'thermal_{k:03d}.npy', 't': ts})

    # Strain gauge points
    sg = pd.DataFrame({
        't': t[::10],
        'strain': 5e-4 + 1e-4*np.sin(2*np.pi*t[::10]/1200) + rng.normal(0, 2e-5, len(t[::10]))
    })
    sg.to_csv(os.path.join(out_dir, 'strain_gauge.csv'), index=False)

    # Acoustic emission events
    ae_events = []
    for ts in rng.choice(t, size=30, replace=False):
        ae_events.append({'t': int(ts), 'amplitude': float(np.abs(rng.normal(1.0, 0.3)))})
    with open(os.path.join(out_dir, 'ae_events.json'), 'w') as fp:
        json.dump(sorted(ae_events, key=lambda x: x['t']), fp, indent=2)

    return {
        'timeseries_csv': 'global_timeseries.csv',
        'eis_files': eis_list,
        'thermal_images': imgs_meta,
        'strain_gauge_csv': 'strain_gauge.csv',
        'ae_events_json': 'ae_events.json',
        'dir': out_dir,
    }
