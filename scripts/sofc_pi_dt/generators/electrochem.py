import numpy as np
from pathlib import Path
from ..utils import write_json


def synthetic_polarization(current_density):
    # Nernst-like OCV minus losses (ohmic + activation)
    ocv = 1.1
    ohmic_R = 0.2  # ohm*cm^2
    activation = 0.05*np.log1p(current_density*10)
    voltage = ocv - ohmic_R*current_density - activation
    return np.clip(voltage, 0.2, ocv)


def synthetic_eis(freqs, state_factor):
    # Simple semicircle with state-dependent radius
    R0 = 0.2 + 0.1*state_factor
    C = 0.5
    w = 2*np.pi*freqs
    Z = R0 / (1 + 1j*w*R0*C)
    return Z


def generate_electrochem(output_dir: Path, seed: int, profiles: int, eis_points: int) -> None:
    rng = np.random.default_rng(seed + 2)
    e_dir = output_dir / 'electrochem'
    e_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(profiles):
        J = np.linspace(0.05, 1.0, 30)
        V = synthetic_polarization(J)
        pol = np.column_stack([J, V]).astype(np.float32)
        np.save(e_dir / f'polarization_{i:03d}.npy', pol)

        freqs = np.logspace(5, -2, eis_points)
        state = rng.uniform(0, 1)
        Z = synthetic_eis(freqs, state)
        eis = np.column_stack([freqs, Z.real, Z.imag]).astype(np.float32)
        np.save(e_dir / f'eis_{i:03d}.npy', eis)

        items.append({'polarization': f'polarization_{i:03d}.npy', 'eis': f'eis_{i:03d}.npy'})
    write_json(e_dir / 'metadata.json', {'count': profiles, 'items': items, 'notes': 'EIS is synthetic semicircle; polarization includes ohmic+activation losses'})
