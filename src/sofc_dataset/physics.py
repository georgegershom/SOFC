from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class Geometry:
    nx: int = 32
    ny: int = 32
    nz: int = 8
    length_x_m: float = 0.02  # 2 cm
    length_y_m: float = 0.02
    length_z_m: float = 0.001  # 1 mm stack slice


@dataclass
class PhysicsConfig:
    # Simple coefficients to shape synthetic fields
    heat_coeff: float = 1.0
    current_coeff: float = 1.0
    diffusion_coeff: float = 1.0
    mech_coeff: float = 1.0
    noise_level: float = 0.01


class SyntheticPhysicsGenerator:
    """
    Generates synthetic multi-physics fields meant to qualitatively emulate
    coupled electro-thermal-mechanical responses for SOFC-like geometries.

    This is NOT a real solver. It produces smooth 3D fields with dependencies
    on inputs to approximate plausible correlations for ML pretraining.
    """

    def __init__(self, geom: Geometry | None = None, cfg: PhysicsConfig | None = None, seed: int | None = 123):
        self.geom = geom or Geometry()
        self.cfg = cfg or PhysicsConfig()
        self.rng = np.random.default_rng(seed)
        self._make_grids()

    def _make_grids(self) -> None:
        g = self.geom
        self.x = np.linspace(0.0, g.length_x_m, g.nx)
        self.y = np.linspace(0.0, g.length_y_m, g.ny)
        self.z = np.linspace(0.0, g.length_z_m, g.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def _smooth_random_field(self, scale: float = 1.0) -> np.ndarray:
        g = self.geom
        base = self.rng.normal(0.0, 1.0, size=(g.nx, g.ny, g.nz))
        # Smooth via separable cosine filter in k-space approximation using FFT magnitudes
        f = np.fft.rfftn(base)
        # For rfftn, only the last axis is the reduced (real) spectrum; use full fftfreq for others
        kx = np.fft.fftfreq(g.nx)
        ky = np.fft.fftfreq(g.ny)
        kz = np.fft.rfftfreq(g.nz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        filt = np.exp(-((KX**2 + KY**2 + KZ**2) / (scale**2 + 1e-9)))
        f_sm = f * filt
        sm = np.fft.irfftn(f_sm, s=base.shape)
        sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-12)
        return sm

    def generate_fields(self, inputs: Dict[str, float]) -> Dict[str, np.ndarray | float]:
        g = self.geom
        cfg = self.cfg

        jd = float(inputs.get("current_density_A_per_cm2", 0.8))
        fuel_util = float(inputs.get("fuel_utilization_pct", 70.0)) / 100.0
        air_util = float(inputs.get("air_utilization_pct", 30.0)) / 100.0
        t_fuel = float(inputs.get("inlet_fuel_temp_C", 700.0))
        t_air = float(inputs.get("inlet_air_temp_C", 700.0))
        h2 = float(inputs.get("fuel_H2_pct", 80.0)) / 100.0
        h2o = float(inputs.get("fuel_H2O_pct", 20.0)) / 100.0
        co = float(inputs.get("fuel_CO_pct", 0.0)) / 100.0
        ch4 = float(inputs.get("fuel_CH4_pct", 0.0)) / 100.0
        an_por = float(inputs.get("anode_porosity", 0.35))
        ca_por = float(inputs.get("cathode_porosity", 0.35))
        an_tau = float(inputs.get("anode_tortuosity", 4.0))
        ca_tau = float(inputs.get("cathode_tortuosity", 4.0))
        el_thk = float(inputs.get("electrolyte_thickness_um", 15.0)) * 1e-6
        inter_E = float(inputs.get("interconnect_E_GPa", 200.0)) * 1e9
        inter_cte = float(inputs.get("interconnect_CTE_ppmK", 13.0)) * 1e-6
        init_crack = float(inputs.get("init_crack_length_mm", 0.1)) * 1e-3
        init_delam = float(inputs.get("init_delamination_mm", 0.1)) * 1e-3
        por_var = float(inputs.get("init_porosity_variance", 0.01))

        # Temperature field: baseline from inlet temps plus ohmic heating ~ jd^2
        T0 = 0.5 * (t_fuel + t_air)
        heating = cfg.heat_coeff * (jd**2) * (1.0 + 0.5 * (1 - fuel_util) + 0.3 * (1 - air_util))
        grad_x = 50.0 * (self.X / self.x[-1])
        grad_y = 30.0 * (self.Y / self.y[-1])
        T = T0 + heating + grad_x + grad_y
        T += 20.0 * self._smooth_random_field(scale=2.0) * (por_var + 0.05)

        # Current density distribution: peak near fuel inlet, reduced by tortuosity and electrolyte thickness
        inlet_profile = 1.0 - 0.5 * (self.X / self.x[-1])
        tau_effect = 1.0 / (1.0 + 0.1 * (an_tau + ca_tau))
        thickness_effect = 1.0 / (1.0 + 100.0 * el_thk)
        i_dist = cfg.current_coeff * jd * inlet_profile * tau_effect * thickness_effect
        i_dist *= (0.8 + 0.2 * self._smooth_random_field(scale=1.5))

        # Species: simple diffusive gradients from inlets and consumption with JD
        H2 = h2 * np.exp(-2.0 * (self.X / self.x[-1])) * np.maximum(0.1, 1.0 - 0.6 * jd)
        H2O = h2o + 0.2 * jd * (self.X / self.x[-1])
        O2 = (1.0 - air_util) * np.exp(-1.5 * (self.Y / self.y[-1]))
        for arr in (H2, H2O, O2):
            arr += 0.05 * self._smooth_random_field(scale=2.5)
            np.clip(arr, 0.0, None, out=arr)

        # Voltage: Nernst-like baseline minus ohmic and polarization drops
        V_open = 1.1 + 0.05 * math.log1p(h2 / (h2o + 1e-3))
        ohmic = 0.2 * jd * (1.0 + 2.0 * el_thk)
        pol = 0.15 * (jd**0.7) * (1.0 + 0.3 * (1 - fuel_util))
        V = float(max(0.2, V_open - ohmic - pol))

        # Mechanical fields: thermal expansion mismatch creates stresses
        delta_T = T - T0
        thermal_strain = inter_cte * delta_T
        # Baseline displacement proportional to thermal strain integrated across thickness
        Uz = cfg.mech_coeff * np.cumsum(thermal_strain, axis=2)
        Ux = 0.1 * Uz * (self.X / self.x[-1])
        Uy = 0.1 * Uz * (self.Y / self.y[-1])

        # Strain tensor: gradients of displacement (small strain assumption)
        def gradient(field: np.ndarray, axis: int, coords: np.ndarray) -> np.ndarray:
            return np.gradient(field, coords, axis=axis, edge_order=2)

        exx = gradient(Ux, 0, self.x)
        eyy = gradient(Uy, 1, self.y)
        ezz = gradient(Uz, 2, self.z)
        exy = 0.5 * (gradient(Ux, 1, self.y) + gradient(Uy, 0, self.x))
        eyz = 0.5 * (gradient(Uy, 2, self.z) + gradient(Uz, 1, self.y))
        ezx = 0.5 * (gradient(Uz, 0, self.x) + gradient(Ux, 2, self.z))

        # Stress via Hooke (isotropic), using effective E and nu
        nu = 0.3
        lam = inter_E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = inter_E / (2 * (1 + nu))
        tr = exx + eyy + ezz
        s_xx = lam * tr + 2 * mu * exx
        s_yy = lam * tr + 2 * mu * eyy
        s_zz = lam * tr + 2 * mu * ezz
        s_xy = 2 * mu * exy
        s_yz = 2 * mu * eyz
        s_zx = 2 * mu * ezx

        # Von Mises (derived)
        von_mises = np.sqrt(
            0.5 * ((s_xx - s_yy) ** 2 + (s_yy - s_zz) ** 2 + (s_zz - s_xx) ** 2)
            + 3 * (s_xy**2 + s_yz**2 + s_zx**2)
        )

        # Fracture metrics (synthetic): scale with stress gradients and initial defects
        grad_vm_x = np.gradient(von_mises, self.x, axis=0, edge_order=2)
        grad_vm_y = np.gradient(von_mises, self.y, axis=1, edge_order=2)
        grad_vm_z = np.gradient(von_mises, self.z, axis=2, edge_order=2)
        tip_intensity = np.sqrt(grad_vm_x**2 + grad_vm_y**2 + grad_vm_z**2)
        K_I = (init_crack + 1e-4) * np.max(tip_intensity)
        K_II = 0.7 * K_I
        K_III = 0.5 * K_I
        G = (K_I**2) / (inter_E + 1e-9)

        # Creep/damage accumulation proxy
        creep = 1e-12 * von_mises * (delta_T.clip(min=0) + 1.0)

        # Add a small noise to mimic numerical/meshing error
        noise = cfg.noise_level * self.rng.normal(0.0, 1.0, size=(g.nx, g.ny, g.nz))
        T = T + noise
        i_dist = i_dist * (1.0 + 0.01 * self.rng.normal(size=i_dist.shape))

        return {
            "coords": {
                "x": self.x,
                "y": self.y,
                "z": self.z,
            },
            "temperature": T.astype(np.float32),
            "current_density": i_dist.astype(np.float32),
            "species": {
                "H2": H2.astype(np.float32),
                "H2O": H2O.astype(np.float32),
                "O2": O2.astype(np.float32),
            },
            "voltage": float(V),
            "displacement": {
                "Ux": Ux.astype(np.float32),
                "Uy": Uy.astype(np.float32),
                "Uz": Uz.astype(np.float32),
            },
            "stress": {
                "s_xx": s_xx.astype(np.float32),
                "s_yy": s_yy.astype(np.float32),
                "s_zz": s_zz.astype(np.float32),
                "s_xy": s_xy.astype(np.float32),
                "s_yz": s_yz.astype(np.float32),
                "s_zx": s_zx.astype(np.float32),
                "von_mises": von_mises.astype(np.float32),
            },
            "strain": {
                "e_xx": exx.astype(np.float32),
                "e_yy": eyy.astype(np.float32),
                "e_zz": ezz.astype(np.float32),
                "e_xy": exy.astype(np.float32),
                "e_yz": eyz.astype(np.float32),
                "e_zx": ezx.astype(np.float32),
            },
            "fracture_metrics": {
                "K_I": float(K_I),
                "K_II": float(K_II),
                "K_III": float(K_III),
                "G": float(G),
            },
            "creep_damage": creep.astype(np.float32),
        }
