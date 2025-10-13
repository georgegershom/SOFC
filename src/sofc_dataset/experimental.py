from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class ExperimentalConfig:
    duration_s: int = 3600  # 1 hour
    sample_rate_hz: int = 1
    eis_points: int = 30
    thermal_image_size: Tuple[int, int] = (64, 64)
    num_thermocouples: int = 4
    num_strain_gauges: int = 4
    ae_rate_per_min: float = 1.0
    noise_level: float = 0.01


class ExperimentalDataGenerator:
    def __init__(self, cfg: ExperimentalConfig | None = None, seed: int | None = 7):
        self.cfg = cfg or ExperimentalConfig()
        self.rng = np.random.default_rng(seed)

    def generate(self, operating_point: Dict[str, float]) -> Dict[str, object]:
        c = self.cfg
        n = c.duration_s * c.sample_rate_hz
        t = np.arange(n) / c.sample_rate_hz

        jd = float(operating_point.get("current_density_A_per_cm2", 0.8))
        t_in = float(operating_point.get("inlet_fuel_temp_C", 700.0))
        flow_fuel = float(operating_point.get("fuel_utilization_pct", 70.0))
        flow_air = float(operating_point.get("air_utilization_pct", 30.0))

        # Global I, V, P with slow drifts and noise
        I = jd * 100.0 + 5.0 * np.sin(2 * np.pi * t / 900.0)
        V = 0.9 - 0.2 * jd**0.7 - 0.02 * np.sin(2 * np.pi * t / 1200.0)
        P = I * V
        for arr in (I, V, P):
            arr += c.noise_level * self.rng.normal(size=arr.shape)

        # Inlet/Outlet gas temperatures and flow rates
        T_in = t_in + 2.0 * np.sin(2 * np.pi * t / 1800.0) + self.rng.normal(0, 0.3, size=t.shape)
        T_out = T_in + 10.0 + 1.5 * np.sin(2 * np.pi * t / 2400.0) + self.rng.normal(0, 0.5, size=t.shape)
        F_fuel = flow_fuel + 0.5 * np.sin(2 * np.pi * t / 1500.0) + self.rng.normal(0, 0.2, size=t.shape)
        F_air = flow_air + 1.0 * np.sin(2 * np.pi * t / 2100.0) + self.rng.normal(0, 0.5, size=t.shape)

        # EIS: synthetic circle/arc in Nyquist with degradation drift
        freqs = np.logspace(2, -1, c.eis_points)
        R0 = 0.2 + 0.1 * jd
        Rt = R0 * (1.0 + 0.1 * (t[-1] / 86400.0))
        Z_re = R0 + (Rt - R0) * (freqs / freqs.max())
        Z_im = 1.0 / (2 * np.pi * freqs * 0.1)  # simple CPE-like
        Z_im *= 1.0 / (1.0 + (freqs / 10.0) ** 0.5)

        # Thermal image: 2D field with hot spot near center
        H, W = c.thermal_image_size
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)
        X, Y = np.meshgrid(x, y)
        hotspot = 15.0 * np.exp(-3.0 * (X**2 + Y**2))
        base = t_in - 30.0 + 2.0 * self.rng.normal(size=(H, W))
        thermal_image = base + hotspot

        # Thermocouples and strain gauges at fixed positions
        tc_positions = np.array([
            [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]
        ])
        sg_positions = np.array([
            [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]
        ])
        tc_values = np.array([
            thermal_image[int(py * (H - 1)), int(px * (W - 1))] + self.rng.normal(0, 0.5)
            for px, py in tc_positions
        ])
        strain_base = 1e-4 * jd
        sg_values = strain_base + 1e-5 * self.rng.normal(size=len(sg_positions))

        # Acoustic emission events: Poisson process with higher rate at higher jd
        lam = c.ae_rate_per_min * (1.0 + 2.0 * jd)
        num_minutes = max(1, c.duration_s // 60)
        events = []
        current_time = 0.0
        for _ in range(num_minutes * 2):
            # inter-arrival exponential in minutes
            dt_min = self.rng.exponential(1.0 / lam)
            current_time += dt_min
            if current_time * 60.0 > c.duration_s:
                break
            events.append({
                "t_s": float(current_time * 60.0),
                "amplitude": float(abs(self.rng.normal(0.0, 1.0)) * (0.5 + jd)),
                "freq_kHz": float(50.0 + 20.0 * self.rng.random()),
            })

        return {
            "time_s": t.astype(np.float32),
            "I_A": I.astype(np.float32),
            "V_V": V.astype(np.float32),
            "P_W": P.astype(np.float32),
            "T_in_C": T_in.astype(np.float32),
            "T_out_C": T_out.astype(np.float32),
            "F_fuel": F_fuel.astype(np.float32),
            "F_air": F_air.astype(np.float32),
            "EIS": {
                "frequency_Hz": freqs.astype(np.float32),
                "Z_re_ohm": Z_re.astype(np.float32),
                "Z_im_ohm": Z_im.astype(np.float32),
            },
            "thermal_image_C": thermal_image.astype(np.float32),
            "thermocouples_C": tc_values.astype(np.float32),
            "strain_gauges": sg_values.astype(np.float32),
            "AE_events": events,
        }
