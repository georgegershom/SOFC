from typing import Dict, Tuple
import torch
from .constitutive import thermal_elastic_stress_2d


def _grad_scalar(f: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回标量场梯度 df/dx, df/dy，均形状 (N,1)。"""
    ones = torch.ones_like(f)
    dfdx = torch.autograd.grad(f, coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    dfdy = torch.autograd.grad(f, coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    return dfdx, dfdy


def _laplacian_scalar(f: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    dfdx, dfdy = _grad_scalar(f, coords)
    ones_x = torch.ones_like(dfdx)
    ones_y = torch.ones_like(dfdy)
    d2fdx2 = torch.autograd.grad(dfdx, coords, grad_outputs=ones_x, create_graph=True)[0][:, 0:1]
    d2fdy2 = torch.autograd.grad(dfdy, coords, grad_outputs=ones_y, create_graph=True)[0][:, 1:2]
    return d2fdx2 + d2fdy2


def _vec_divergence(sig_x: torch.Tensor, sig_y: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """div([sig_x, sig_y])。输入与返回 (N,1)。"""
    ones_x = torch.ones_like(sig_x)
    ones_y = torch.ones_like(sig_y)
    dsigx_dx = torch.autograd.grad(sig_x, coords, grad_outputs=ones_x, create_graph=True)[0][:, 0:1]
    dsigy_dy = torch.autograd.grad(sig_y, coords, grad_outputs=ones_y, create_graph=True)[0][:, 1:2]
    return dsigx_dx + dsigy_dy


def charge_conservation_residuals(
    coords: torch.Tensor,
    phi_s: torch.Tensor,
    phi_e: torch.Tensor,
    is_reactive_mask: torch.Tensor,
    sigma_s: float,
    kappa_e: float,
    a_area: float,
    j0: float,
    alpha_a: float,
    alpha_c: float,
    R: float,
    F: float,
    U_eq: float,
    T: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 (r_s, r_e, j) 形状均为 (N,1)：
    r_s = div(σ_s grad phi_s) + a F j
    r_e = div(κ_e grad phi_e) - a F j
    j   = Butler-Volmer 电流密度
    """
    dtype = torch.get_default_dtype()
    sigma_s_t = torch.tensor(sigma_s, dtype=dtype)
    kappa_e_t = torch.tensor(kappa_e, dtype=dtype)
    a_t = torch.tensor(a_area, dtype=dtype)
    j0_t = torch.tensor(j0, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)
    F_t = torch.tensor(F, dtype=dtype)
    alpha_a_t = torch.tensor(alpha_a, dtype=dtype)
    alpha_c_t = torch.tensor(alpha_c, dtype=dtype)
    U_eq_t = torch.tensor(U_eq, dtype=dtype)

    eta = phi_s - phi_e - U_eq_t
    j = j0_t * (torch.exp(alpha_a_t * F_t * eta / (R_t * T)) - torch.exp(-alpha_c_t * F_t * eta / (R_t * T)))
    j = j * is_reactive_mask  # 仅在反应区生效

    lap_phi_s = _laplacian_scalar(phi_s, coords)
    lap_phi_e = _laplacian_scalar(phi_e, coords)

    r_s = sigma_s_t * lap_phi_s + a_t * F_t * j
    r_e = kappa_e_t * lap_phi_e - a_t * F_t * j
    return r_s, r_e, j


def heat_equation_residual(
    coords: torch.Tensor,
    T: torch.Tensor,
    phi_s: torch.Tensor,
    phi_e: torch.Tensor,
    rho: float,
    cp: float,
    k: float,
    joule_scale: float,
    reaction_heat_scale: float,
    j: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    """ρ c ∂T/∂t - ∇·(k ∇T) - (q_J + q_rxn)=0"""
    dtype = torch.get_default_dtype()
    rho_t = torch.tensor(rho, dtype=dtype)
    cp_t = torch.tensor(cp, dtype=dtype)
    k_t = torch.tensor(k, dtype=dtype)
    js_t = torch.tensor(joule_scale, dtype=dtype)
    rs_t = torch.tensor(reaction_heat_scale, dtype=dtype)

    # 时间导数（如果无时间维，假定 t 是 coords 的第3列为0，梯度返回0）
    ones = torch.ones_like(T)
    dTdt = torch.autograd.grad(T, coords, grad_outputs=ones, create_graph=True)[0]
    if dTdt.shape[1] >= 3:
        dTdt = dTdt[:, 2:3]
    else:
        dTdt = torch.zeros_like(T)

    grad_phi_s_x, grad_phi_s_y = _grad_scalar(phi_s, coords)
    grad_phi_e_x, grad_phi_e_y = _grad_scalar(phi_e, coords)
    q_joule = js_t * (
        (grad_phi_s_x ** 2 + grad_phi_s_y ** 2) + (grad_phi_e_x ** 2 + grad_phi_e_y ** 2)
    )

    lap_T = _laplacian_scalar(T, coords)
    q_rxn = rs_t * j * eta

    r_T = rho_t * cp_t * dTdt - k_t * lap_T - (q_joule + q_rxn)
    return r_T


def momentum_residuals(
    coords: torch.Tensor,
    u: torch.Tensor,  # (N,2)
    T: torch.Tensor,
    T_ref: float,
    alpha_T: float,
    E: float,
    nu: float,
    plane_stress: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """∇·σ=0 的两个分量残差。"""
    ones = torch.ones_like(u[:, 0:1])
    dux_dx = torch.autograd.grad(u[:, 0:1], coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    dux_dy = torch.autograd.grad(u[:, 0:1], coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    duy_dx = torch.autograd.grad(u[:, 1:2], coords, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    duy_dy = torch.autograd.grad(u[:, 1:2], coords, grad_outputs=ones, create_graph=True)[0][:, 1:2]

    strain_xx = dux_dx
    strain_yy = duy_dy
    strain_xy = 0.5 * (dux_dy + duy_dx)

    sigma_xx, sigma_yy, sigma_xy = thermal_elastic_stress_2d(
        strain_xx, strain_yy, strain_xy, T, T_ref, alpha_T, E, nu, plane_stress
    )

    r_mom_x = _vec_divergence(sigma_xx, sigma_xy, coords)
    r_mom_y = _vec_divergence(sigma_xy, sigma_yy, coords)
    return r_mom_x, r_mom_y
