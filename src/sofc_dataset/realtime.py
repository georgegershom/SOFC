from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RealTimeConfig:
    duration_s: int = 900
    sample_rate_hz: int = 1
    periodic_eis_interval_s: int = 300
    periodic_thermal_interval_s: int = 300


class RealTimeDataGenerator:
    def __init__(self, cfg: RealTimeConfig | None = None, seed: int | None = 11):
        self.cfg = cfg or RealTimeConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, operating_point: Dict[str, float]):
        c = self.cfg
        n = c.duration_s * c.sample_rate_hz
        t = np.arange(n) / c.sample_rate_hz

        jd = float(operating_point.get("current_density_A_per_cm2", 0.8))
        t_in = float(operating_point.get("inlet_fuel_temp_C", 700.0))

        I = jd * 100.0 + 3.0 * np.sin(2 * np.pi * t / 300.0) + 0.5 * self.rng.normal(size=t.shape)
        V = 0.85 - 0.18 * jd**0.7 - 0.01 * np.sin(2 * np.pi * t / 600.0) + 0.01 * self.rng.normal(size=t.shape)
        T_in = t_in + 1.0 * np.sin(2 * np.pi * t / 600.0) + 0.3 * self.rng.normal(size=t.shape)

        eis_times = np.arange(0, c.duration_s + 1, c.periodic_eis_interval_s)
        thermal_times = np.arange(0, c.duration_s + 1, c.periodic_thermal_interval_s)

        return {
            "time_s": t.astype(np.float32),
            "I_A": I.astype(np.float32),
            "V_V": V.astype(np.float32),
            "T_in_C": T_in.astype(np.float32),
            "eis_capture_s": eis_times.astype(np.int32),
            "thermal_capture_s": thermal_times.astype(np.int32),
        }
