import numpy as np
from pathlib import Path
from ..utils import write_json


def generate_degradation(output_dir: Path, seed: int, hours: int) -> None:
    rng = np.random.default_rng(seed + 4)
    deg_dir = output_dir / 'degradation'
    deg_dir.mkdir(parents=True, exist_ok=True)

    # Accelerated aging: voltage, temperature, EIS proxy vs time
    t = np.arange(0, hours+1)
    voltage = 0.9 - 0.0007*t + 0.005*np.sin(2*np.pi*t/24)
    temperature = 730 + 0.5*np.sin(2*np.pi*t/12)
    eis_radius = 0.3 + 0.0004*t  # growing polarization resistance
    data = np.column_stack([t, voltage, temperature, eis_radius]).astype(np.float32)
    np.save(deg_dir / 'aging_timeseries.npy', data)

    # Event labels for specific degradation modes
    events = [
        {'hour': int(0.2*hours), 'mode': 'thermal_cycling', 'note': 'minor delamination onset'},
        {'hour': int(0.5*hours), 'mode': 'redox_cycle', 'note': 'anode reduction/oxidation event'},
        {'hour': int(0.8*hours), 'mode': 'contamination', 'note': 'cathode poisoning indicator'},
    ]
    write_json(deg_dir / 'events.json', {'events': events})
