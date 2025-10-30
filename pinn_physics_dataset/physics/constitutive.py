from typing import Tuple
import torch

def elasticity_matrix_isotropic(E: float, nu: float, plane_stress: bool) -> torch.Tensor:
    """各向同性线弹性刚度矩阵 D (2D)。返回形状 (3,3)。"""
    dtype = torch.get_default_dtype()
    E_t = torch.tensor(E, dtype=dtype)
    nu_t = torch.tensor(nu, dtype=dtype)
    if plane_stress:
        c = E_t / (1.0 - nu_t ** 2)
        D = c * torch.tensor([
            [1.0, nu_t, 0.0],
            [nu_t, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu_t) / 2.0],
        ], dtype=dtype)
    else:  # plane strain
        c = E_t / ((1.0 + nu_t) * (1.0 - 2.0 * nu_t))
        D = c * torch.tensor([
            [1.0 - nu_t, nu_t, 0.0],
            [nu_t, 1.0 - nu_t, 0.0],
            [0.0, 0.0, 0.5 - nu_t],
        ], dtype=dtype)
    return D


def thermal_elastic_stress_2d(
    strain_xx: torch.Tensor,
    strain_yy: torch.Tensor,
    strain_xy: torch.Tensor,
    T: torch.Tensor,
    T_ref: float,
    alpha_T: float,
    E: float,
    nu: float,
    plane_stress: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """σ = D : (ε - ε_th)，其中 ε_th = α (T - T_ref) [1, 1, 0]^T。
    输入张量形状均为 (N,1)。
    返回 σ_xx, σ_yy, σ_xy，形状 (N,1)。
    """
    D = elasticity_matrix_isotropic(E, nu, plane_stress)
    dtype = torch.get_default_dtype()
    alpha = torch.tensor(alpha_T, dtype=dtype)
    T_ref_t = torch.tensor(T_ref, dtype=dtype)

    eps_th_xx = alpha * (T - T_ref_t)
    eps_th_yy = alpha * (T - T_ref_t)
    eps_th_xy = torch.zeros_like(T)

    eps_mech = torch.cat([
        strain_xx - eps_th_xx,
        strain_yy - eps_th_yy,
        strain_xy - eps_th_xy,
    ], dim=1)  # (N,3)
    sigma = eps_mech @ D.T  # (N,3)
    sigma_xx = sigma[:, 0:1]
    sigma_yy = sigma[:, 1:1+1]
    sigma_xy = sigma[:, 2:2+1]
    return sigma_xx, sigma_yy, sigma_xy
