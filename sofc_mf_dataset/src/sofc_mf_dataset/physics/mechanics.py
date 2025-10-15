from __future__ import annotations
import numpy as np
from ..config import GeneratorConfig
from ..sampling import InputSample
from .utils import smooth_nd


def gradients(field: np.ndarray) -> tuple[np.ndarray, ...]:
    return np.gradient(field)


def elastic_modulus_T(T: np.ndarray, cfg: GeneratorConfig) -> np.ndarray:
    # E decreases with T (~10% per 100K)
    return cfg.E_ref * (0.9 ** ((T - cfg.T_ref) / 100.0))


def cte_T(T: np.ndarray, cfg: GeneratorConfig) -> np.ndarray:
    return cfg.alpha_th_ref * (1.05 ** ((T - cfg.T_ref) / 100.0))


def stress_strain_damage(sample: InputSample, cfg: GeneratorConfig, T2: np.ndarray, T3: np.ndarray, overp: dict, rng: np.random.Generator) -> dict:
    # Thermal strain proxy
    alpha2 = cte_T(T2, cfg)
    alpha3 = cte_T(T3, cfg)
    dT2 = T2 - cfg.T_ref
    dT3 = T3 - cfg.T_ref

    # Use T gradients as a proxy for constrained thermal stress
    gx2, gy2 = gradients(T2)
    gx3, gy3, gz3 = gradients(T3)

    E2 = elastic_modulus_T(T2, cfg)
    E3 = elastic_modulus_T(T3, cfg)

    # Stress components ~ E * alpha * grad(T)
    sx2 = E2 * alpha2 * gx2 * 1e-6
    sy2 = E2 * alpha2 * gy2 * 1e-6

    sx3 = E3 * alpha3 * gx3 * 1e-6
    sy3 = E3 * alpha3 * gy3 * 1e-6
    sz3 = E3 * alpha3 * gz3 * 1e-6

    # von Mises (2D plane stress approx)
    sigma_vm_2d = np.sqrt(sx2**2 + sy2**2 - sx2 * sy2)
    sigma_vm_3d = np.sqrt(0.5 * ((sx3 - sy3) ** 2 + (sy3 - sz3) ** 2 + (sz3 - sx3) ** 2))

    # Strain: elastic + creep
    eps_el_2d = sigma_vm_2d / (E2 + 1e-9)
    eps_el_3d = sigma_vm_3d / (E3 + 1e-9)

    t_hours = sample.cycles * 0.5  # assume 0.5 h per cycle
    R = cfg.R
    Tm2 = np.maximum(1.0, T2)
    Tm3 = np.maximum(1.0, T3)
    eps_cr_2d = cfg.creep_A * (t_hours ** cfg.creep_m) * np.exp(-cfg.creep_Q / (R * Tm2))
    eps_cr_3d = cfg.creep_A * (t_hours ** cfg.creep_m) * np.exp(-cfg.creep_Q / (R * Tm3))

    eps_tot_2d = eps_el_2d + eps_cr_2d
    eps_tot_3d = eps_el_3d + eps_cr_3d

    # Damage indicators
    D2 = 1.0 / (1.0 + np.exp(-(sigma_vm_2d - cfg.sigma_vm_thresh) / (0.15 * cfg.sigma_vm_thresh)))
    D3 = 1.0 / (1.0 + np.exp(-(sigma_vm_3d - cfg.sigma_vm_thresh) / (0.15 * cfg.sigma_vm_thresh)))
    D2_bin = (sigma_vm_2d > cfg.sigma_vm_thresh).astype(np.uint8)
    D3_bin = (sigma_vm_3d > cfg.sigma_vm_thresh).astype(np.uint8)

    # Interface delamination proxy at Z mid-plane: G = 0.5 * sigma^2 * h / E
    z_mid = T3.shape[2] // 2
    h_eff = (sample.electrolyte_thickness_um * 1e-6)  # m
    E_mid = E3[:, :, z_mid]
    sigma_mid = sigma_vm_3d[:, :, z_mid]
    G_mid = 0.5 * (sigma_mid ** 2) * h_eff / (E_mid + 1e-9)
    delam_prob = 1.0 / (1.0 + np.exp(-(G_mid - cfg.Gc_interface) / (0.2 * cfg.Gc_interface)))

    # Ni-coarsening indicator ~ f(T, time, |eta_act|)
    eta_act_3d = np.abs(overp["eta_act_3d"])  # V
    ni_coarsen = (Tm3 / cfg.T_ref) ** 2.0 * (1.0 + 0.5 * eta_act_3d) * (t_hours / 1000.0) ** 0.7

    # Volume-averaged LF proxies
    sigma_vm_avg = float(np.mean(sigma_vm_3d))
    strain_avg = float(np.mean(eps_tot_3d))

    # Time-to-failure (Weibull-like)
    beta1 = 1.0 / cfg.sigma_vm_thresh
    beta2 = 1.0 / 200.0  # per K
    beta3 = 0.5  # per V
    stress_term = beta1 * (sigma_vm_avg)
    temp_term = beta2 * max(0.0, float(np.mean(T3) - cfg.T_ref))
    overp_term = beta3 * float(np.mean(np.abs(overp["eta_act_3d"])) )
    accel = np.exp(stress_term + temp_term + overp_term)
    rng_state = rng.random()
    k = cfg.weibull_k
    lam = cfg.weibull_lambda_hours / accel
    ttf_hours = lam * (-np.log(1.0 - 0.999 * rng_state)) ** (1.0 / k)

    return {
        "sigma_vm_2d": sigma_vm_2d.astype(np.float32),
        "sigma_vm_3d": sigma_vm_3d.astype(np.float32),
        "strain_2d": eps_tot_2d.astype(np.float32),
        "strain_3d": eps_tot_3d.astype(np.float32),
        "damage_2d": D2.astype(np.float32),
        "damage_2d_binary": D2_bin,
        "damage_3d": D3.astype(np.float32),
        "damage_3d_binary": D3_bin,
        "delamination_mid_G": G_mid.astype(np.float32),
        "delamination_mid_prob": delam_prob.astype(np.float32),
        "ni_coarsening_3d": ni_coarsen.astype(np.float32),
        "sigma_vm_avg": np.array([sigma_vm_avg], dtype=np.float32),
        "strain_avg": np.array([strain_avg], dtype=np.float32),
        "time_to_failure_hours": np.array([float(ttf_hours)], dtype=np.float32),
    }
