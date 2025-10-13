import numpy as np
from pathlib import Path
from ..utils import write_json


def generate_operational_profiles(output_dir: Path, seed: int, profiles: int) -> None:
    rng = np.random.default_rng(seed + 1)
    ops_dir = output_dir / 'operations'
    ops_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(profiles):
        t = np.linspace(0, 3600, 720)  # one hour at 5s cadence
        fuel_h2 = 0.7 + 0.05*np.sin(2*np.pi*t/900 + rng.uniform(0, np.pi))
        fuel_co = 0.1 + 0.02*np.sin(2*np.pi*t/1200 + rng.uniform(0, np.pi))
        fuel_ch4 = 0.2 - (fuel_h2 - 0.7) - (fuel_co - 0.1)
        flow_fuel_sccm = 1000 + 100*np.sin(2*np.pi*t/600)
        flow_air_sccm = 3000 + 200*np.sin(2*np.pi*t/750 + 0.3)
        tin_fuel_C = 700 + 10*np.sin(2*np.pi*t/1100)
        tin_air_C = 700 + 8*np.sin(2*np.pi*t/1300 + 0.2)
        current_density_Acm2 = 0.5 + 0.1*np.sin(2*np.pi*t/1000 + rng.uniform(0,1))
        data = np.column_stack([
            t, fuel_h2, fuel_co, fuel_ch4, flow_fuel_sccm, flow_air_sccm,
            tin_fuel_C, tin_air_C, current_density_Acm2
        ]).astype(np.float32)
        np.save(ops_dir / f'profile_{i:03d}.npy', data)
        items.append({'file': f'profile_{i:03d}.npy', 'duration_s': float(t[-1])})
    write_json(ops_dir / 'metadata.json', {'count': profiles, 'schema': ['t','H2','CO','CH4','fuel_sccm','air_sccm','Tin_fuel_C','Tin_air_C','J_Acm2'], 'items': items})
