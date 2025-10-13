import os
import json
import numpy as np
import pandas as pd
from .utils import ensure_dir

def generate_dataset3(out_dir: str, seed: int):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    T = 7200
    t = np.arange(T)
    I = 25 + 3*np.sin(2*np.pi*t/1200) + rng.normal(0, 0.2, T)
    V = 0.82 - 0.000015*t + 0.01*np.sin(2*np.pi*t/2400) + rng.normal(0, 0.002, T)
    Tin = 710 + 8*np.sin(2*np.pi*t/3600)
    Fin = 1.1 + 0.05*np.sin(2*np.pi*t/1800)
    df = pd.DataFrame({'t': t, 'I': I, 'V': V, 'T_in': Tin, 'F_in': Fin})
    df.to_csv(os.path.join(out_dir, 'realtime_1hz.csv'), index=False)

    # Periodic EIS
    freqs = np.logspace(1, 5, 80)
    eis_list = []
    for k, ts in enumerate(range(0, T, 3600)):
        R0 = 0.22 + 0.00004*ts
        Cdl = 0.9e-3*(1 + 0.2*np.sin(2*np.pi*ts/7200))
        w = 2*np.pi*freqs
        Z_re = R0 / (1 + (w*R0*Cdl)**2)
        Z_im = - (w*R0**2*Cdl) / (1 + (w*R0*Cdl)**2)
        np.savez(os.path.join(out_dir, f'eis_{k:03d}.npz'), freqs=freqs, Z_re=Z_re, Z_im=Z_im, t=ts)
        eis_list.append({'file': f'eis_{k:03d}.npz', 't': ts})

    # Periodic thermal images
    imgs_meta = []
    for k, ts in enumerate(range(0, T, 3600)):
        img = 705 + 10*np.random.rand(64, 64)
        np.save(os.path.join(out_dir, f'thermal_{k:03d}.npy'), img)
        imgs_meta.append({'file': f'thermal_{k:03d}.npy', 't': ts})

    # AE sparse events
    ae_events = []
    for ts in rng.choice(t, size=20, replace=False):
        ae_events.append({'t': int(ts), 'amplitude': float(np.abs(rng.normal(1.0, 0.25)))})
    with open(os.path.join(out_dir, 'ae_events.json'), 'w') as fp:
        json.dump(sorted(ae_events, key=lambda x: x['t']), fp, indent=2)

    return {
        'realtime_csv': 'realtime_1hz.csv',
        'eis_files': eis_list,
        'thermal_images': imgs_meta,
        'ae_events_json': 'ae_events.json',
        'dir': out_dir,
    }
