from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import math


def synthesize_realtime_dataset(
    out_dir: Path,
    duration_hours: float = 6.0,
    sample_rate_hz: float = 1.0,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)
    n_steps = int(duration_hours * 3600 * sample_rate_hz)
    t = np.arange(n_steps) / sample_rate_hz
    t_hours = t / 3600.0

    # Simplified control-like trajectories
    I_A = 40.0 + 5.0 * np.sin(2 * math.pi * t_hours / 2.0) + rng.normal(scale=0.5, size=n_steps)
    T_in_C = 720.0 + 5.0 * np.cos(2 * math.pi * t_hours / 3.0) + rng.normal(scale=0.3, size=n_steps)
    F_in_sccm = 900.0 + 30.0 * np.sin(2 * math.pi * t_hours / 4.0) + rng.normal(scale=3.0, size=n_steps)
    # Voltage reacts to current and temperature with noise
    V_V = 0.95 - 0.004 * (I_A - I_A.mean()) + 0.0005 * (T_in_C - T_in_C.mean()) + rng.normal(scale=0.004, size=n_steps)
    V_V = np.clip(V_V, 0.6, 1.1)

    df = pd.DataFrame({
        "time_s": t,
        "I_A": I_A,
        "V_V": V_V,
        "T_in_C": T_in_C,
        "F_in_sccm": F_in_sccm,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "realtime_stream.csv", index=False)
