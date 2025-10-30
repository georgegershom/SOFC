from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np


FloatArray = np.ndarray


@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)


@dataclass
class LayerIndices:
    anode: Tuple[int, int]
    electrolyte: Tuple[int, int]
    cathode: Tuple[int, int]
    interconnect: Tuple[int, int]
    sealant: Tuple[int, int]


_DEFAULT_POISSON: Dict[str, float] = {
    "anode": 0.28,
    "electrolyte": 0.25,
    "cathode": 0.30,
    "interconnect": 0.29,
    "sealant": 0.22,
}


class FieldFabricator:
    def __init__(self, grid: GridSpec, seed: int | None = None) -> None:
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        # Coordinates (dimensionless 0..1)
        self.X, self.Y, self.Z = np.meshgrid(
            np.linspace(0.0, 1.0, grid.nx, dtype=np.float64),
            np.linspace(0.0, 1.0, grid.ny, dtype=np.float64),
            np.linspace(0.0, 1.0, grid.nz, dtype=np.float64),
            indexing="ij",
        )

    @staticmethod
    def _compute_layer_indices(params: Dict[str, Any], nz: int) -> LayerIndices:
        # thickness in microns -> fraction
        t_an = float(params["thickness_anode_um"]) / 1e6
        t_el = float(params["thickness_electrolyte_um"]) / 1e6
        t_ca = float(params["thickness_cathode_um"]) / 1e6
        t_ic = float(params["thickness_interconnect_um"]) / 1e6
        t_se = float(params["thickness_sealant_um"]) / 1e6
        total = t_an + t_el + t_ca + t_ic + t_se
        frac = np.array([t_an, t_el, t_ca, t_ic, t_se], dtype=np.float64) / max(total, 1e-12)
        k = np.round(np.cumsum(frac) * nz).astype(int)
        k0 = 0
        idx = []
        for ki in k:
            idx.append((k0, min(ki, nz)))
            k0 = min(ki, nz)
        # Ensure coverage to end
        if idx[-1][1] < nz:
            idx[-1] = (idx[-1][0], nz)
        return LayerIndices(*idx)  # type: ignore[arg-type]

    def _layer_mask(self, indices: Tuple[int, int]) -> FloatArray:
        m = np.zeros(self.grid.shape, dtype=np.float64)
        a, b = indices
        if a >= b:
            return m
        m[:, :, a:b] = 1.0
        return m

    def _smooth3d(self, A: FloatArray, kernel_size: int = 3, passes: int = 1) -> FloatArray:
        """3D box blur (kernel 3x3x3 or 5x5x5) using pad+shift; preserves shape."""
        K = int(max(1, kernel_size))
        if K not in (3, 5):
            K = 3
        pad = K // 2
        out = A.astype(np.float64, copy=True)
        for _ in range(int(max(1, passes))):
            padded = np.pad(out, ((pad, pad), (pad, pad), (pad, pad)), mode="reflect")
            acc = np.zeros_like(out, dtype=np.float64)
            for dx in range(-pad, pad + 1):
                xs = slice(pad + dx, pad + dx + out.shape[0])
                for dy in range(-pad, pad + 1):
                    ys = slice(pad + dy, pad + dy + out.shape[1])
                    for dz in range(-pad, pad + 1):
                        zs = slice(pad + dz, pad + dz + out.shape[2])
                        acc += padded[xs, ys, zs]
            out = acc / (K * K * K)
        return out.astype(np.float32, copy=False)

    def _square_wave(self, Y: FloatArray, pitch_cells: int, duty: float) -> FloatArray:
        duty = float(np.clip(duty, 0.05, 0.95))
        # Use a single z-slice to create a 2D mask constant along z
        Y2 = Y[:, :, 0]
        # Map Y in [0,1] to index space 0..ny-1
        yy = (Y2 * (self.grid.ny - 1)).astype(int)
        yy = np.clip(yy, 0, self.grid.ny - 1)
        # Channel mask repeating pattern with period `pitch_cells`
        pattern = (np.arange(self.grid.ny) % max(1, pitch_cells)) < int(np.round(pitch_cells * duty))
        mask_y = pattern[yy]
        return mask_y.astype(np.float32)

    def generate_fields(self, params: Dict[str, Any]) -> Dict[str, FloatArray]:
        nx, ny, nz = self.grid.shape
        li = self._compute_layer_indices(params, nz)

        # Layer masks
        m_an = self._layer_mask(li.anode)
        m_el = self._layer_mask(li.electrolyte)
        m_ca = self._layer_mask(li.cathode)
        m_ic = self._layer_mask(li.interconnect)
        m_se = self._layer_mask(li.sealant)

        # Material props (per layer, scalar -> 3D by mask)
        def layer_prop(key: str, default: float = 0.0) -> FloatArray:
            val = (
                float(params.get(f"anode_{key}", default)) * m_an
                + float(params.get(f"electrolyte_{key}", default)) * m_el
                + float(params.get(f"cathode_{key}", default)) * m_ca
                + float(params.get(f"interconnect_{key}", default)) * m_ic
                + float(params.get(f"sealant_{key}", default)) * m_se
            )
            return val.astype(np.float32)

        ionic_sigma = layer_prop("ionic_conductivity_Spm")
        electronic_sigma = layer_prop("electronic_conductivity_Spm")
        E_modulus = layer_prop("youngs_modulus_Pa")
        CTE = layer_prop("cte_per_K")

        # Poisson ratio map
        nu_map = (
            _DEFAULT_POISSON["anode"] * m_an
            + _DEFAULT_POISSON["electrolyte"] * m_el
            + _DEFAULT_POISSON["cathode"] * m_ca
            + _DEFAULT_POISSON["interconnect"] * m_ic
            + _DEFAULT_POISSON["sealant"] * m_se
        ).astype(np.float32)

        # Channel-land pattern along y
        channel_pitch_mm = float(params["channel_pitch_mm"])  # only used relatively
        land_frac = float(params["channel_land_fraction"])    # duty of square wave under channel
        # Calibrate pitch to cell count (heuristic): larger pitch => more cells per period
        pitch_cells = int(np.clip(np.round(channel_pitch_mm / 3.0 * ny), 3, max(3, ny // 2)))
        under_channel = self._square_wave(self.Y, pitch_cells=pitch_cells, duty=1.0 - land_frac)
        under_channel = under_channel[..., np.newaxis]  # broadcast in z

        # Operating conditions
        j0 = float(params["current_density_A_per_cm2"])  # base
        air_Tin = float(params["inlet_temp_air_K"]) 
        fuel_Tin = float(params["inlet_temp_fuel_K"]) 
        Tin = 0.5 * (air_Tin + fuel_Tin)

        # Current density distribution: decay along x, modulated by channels, peaking near electrolyte interfaces
        ax = 1.5 + 1.0 * self.rng.uniform()
        beta = 0.15 + 0.25 * self.rng.uniform()
        # z Gaussian around electrolyte center
        z_center = 0.5 * (li.anode[1] + li.anode[0] + li.electrolyte[1] + li.electrolyte[0]) / (2.0 * nz)
        sigma_z = max(1.0 / nz, (li.electrolyte[1] - li.electrolyte[0] + 2) / (3.0 * nz))
        gz = np.exp(-0.5 * ((self.Z - z_center) / sigma_z) ** 2)
        j_field = j0 * np.exp(-ax * self.X) * (1.0 + beta * (under_channel - 0.5)) * (0.8 + 0.2 * gz)
        j_field = j_field.astype(np.float32)

        # Overpotential: activation + ohmic (heuristic)
        i0_ref = 0.05 + 0.05 * self.rng.uniform()  # A/cm^2
        a_coeff = 0.06 + 0.04 * self.rng.uniform()  # V
        r_ohmic = (1.0 / (ionic_sigma + electronic_sigma + 1e-9)).astype(np.float32)
        overpotential = a_coeff * np.log1p(np.maximum(j_field, 1e-6) / i0_ref) + j_field * r_ohmic
        overpotential = overpotential.astype(np.float32)

        # Temperature field: inlet base + Joule + reaction heat, softly smoothed
        c1 = 25.0 + 25.0 * self.rng.uniform()  # K per A/cm^2
        c2 = 5.0 + 10.0 * self.rng.uniform()   # K per (A/cm^2)^2 scaled by resistivity
        T_field = Tin + c1 * j_field + c2 * (j_field ** 2) * r_ohmic
        T_field = self._smooth3d(T_field, kernel_size=3, passes=1)
        T_field = T_field.astype(np.float32)

        # Thermal strain and stress (very approximate isotropic)
        T_ref = 298.15
        eps_th = CTE * (T_field - T_ref)
        eps_th = eps_th.astype(np.float32)

        # Effective modulus for plane-strain-like approximation
        # sigma_diag ≈ E / (1 - nu) * (eps_th - mean_eps_th_layer)
        # layerwise mean
        layer_mean = (
            (eps_th * m_an).sum() / max(m_an.sum(), 1.0)
            + (eps_th * m_el).sum() / max(m_el.sum(), 1.0)
            + (eps_th * m_ca).sum() / max(m_ca.sum(), 1.0)
            + (eps_th * m_ic).sum() / max(m_ic.sum(), 1.0)
            + (eps_th * m_se).sum() / max(m_se.sum(), 1.0)
        ) / 5.0
        eps_dev = eps_th - float(layer_mean)
        coeff = (E_modulus / (1.0 - nu_map + 1e-6)).astype(np.float32)
        sigma_xx = coeff * eps_dev
        sigma_yy = coeff * eps_dev
        sigma_zz = coeff * eps_dev

        # Shear from temperature gradients (proxy)
        # sigma_shear ≈ k_s * grad(T)
        k_s = (0.1 + 0.2 * self.rng.uniform()) * 1e6
        # central differences
        dTx = np.gradient(T_field, axis=0)
        dTy = np.gradient(T_field, axis=1)
        dTz = np.gradient(T_field, axis=2)
        sigma_xy = (k_s * 0.5 * (dTx + dTy)).astype(np.float32)
        sigma_yz = (k_s * 0.5 * (dTy + dTz)).astype(np.float32)
        sigma_zx = (k_s * 0.5 * (dTz + dTx)).astype(np.float32)

        # Von Mises
        s_xx = sigma_xx; s_yy = sigma_yy; s_zz = sigma_zz
        s_xy = sigma_xy; s_yz = sigma_yz; s_zx = sigma_zx
        von_mises = np.sqrt(
            0.5 * (
                (s_xx - s_yy) ** 2 + (s_yy - s_zz) ** 2 + (s_zz - s_xx) ** 2
                + 6.0 * (s_xy ** 2 + s_yz ** 2 + s_zx ** 2)
            )
        ).astype(np.float32)

        # Strain tensor: thermal strain + small mechanical from stress/E
        mech_strain = (s_xx + s_yy + s_zz) / (3.0 * (E_modulus + 1e-6))
        eps_xx = (eps_th + mech_strain).astype(np.float32)
        eps_yy = (eps_th + mech_strain).astype(np.float32)
        eps_zz = (eps_th + mech_strain).astype(np.float32)
        eps_xy = (s_xy / (2.0 * (E_modulus + 1e-6))).astype(np.float32)
        eps_yz = (s_yz / (2.0 * (E_modulus + 1e-6))).astype(np.float32)
        eps_zx = (s_zx / (2.0 * (E_modulus + 1e-6))).astype(np.float32)

        # Displacement by integrating strain (coarse cumulative integration)
        dx = 1.0 / max(nx - 1, 1)
        dy = 1.0 / max(ny - 1, 1)
        dz = 1.0 / max(nz - 1, 1)
        ux = np.cumsum(eps_xx, axis=0) * dx
        uy = np.cumsum(eps_yy, axis=1) * dy
        uz = np.cumsum(eps_zz, axis=2) * dz
        u_mag = np.sqrt(ux**2 + uy**2 + uz**2).astype(np.float32)

        # Species fields: H2 consumed along x near anode/electrolyte; H2O increases
        j_mean = float(np.mean(j_field))
        k_consume = 1.0 + 2.5 * (j_mean / (j0 + 1e-6))
        # Focus near anode/electrolyte interface
        an_top = li.anode[1] / nz
        ga = np.exp(-0.5 * ((self.Z - an_top) / max(1.0 / nz, 0.03)) ** 2)
        h2 = np.clip(np.exp(-k_consume * self.X) * (0.6 + 0.4 * ga), 0.0, 1.2).astype(np.float32)
        h2o = np.clip(1.2 - h2 + 0.05 * self.rng.normal(size=self.grid.shape), 0.0, 1.5).astype(np.float32)

        # Pack outputs
        fields: Dict[str, FloatArray] = {
            "current_density_A_per_cm2": j_field.astype(np.float32),
            "overpotential_V": overpotential.astype(np.float32),
            "temperature_K": T_field.astype(np.float32),
            "stress_von_mises_Pa": von_mises.astype(np.float32),
            "stress_tensor_Pa": np.stack([s_xx, s_yy, s_zz, s_xy, s_yz, s_zx], axis=-1).astype(np.float32),
            "strain_tensor": np.stack([eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_zx], axis=-1).astype(np.float32),
            "displacement_m": np.stack([ux, uy, uz], axis=-1).astype(np.float32),
            "species_H2": h2.astype(np.float32),
            "species_H2O": h2o.astype(np.float32),
        }
        return fields


def params_to_jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out
