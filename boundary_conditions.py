"""
边界条件定义 - 电池系统中PINN需要的各种边界条件
包含热边界、机械边界、电边界条件
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import sympy as sp
from sympy import symbols, diff, simplify

class BoundaryConditions:
    """
    边界条件类 - 定义各种物理边界条件
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 边界条件参数
        self.bc_params = {
            # 热边界条件参数
            'h_convection': 10.0,      # 对流换热系数 W/(m²·K)
            'T_ambient': 298.0,        # 环境温度 K
            'q_heat_flux': 1000.0,     # 热流密度 W/m²
            
            # 机械边界条件参数
            'traction_x': 0.0,         # x方向牵引力 Pa
            'traction_y': 0.0,         # y方向牵引力 Pa
            'traction_z': 0.0,         # z方向牵引力 Pa
            'displacement_x': 0.0,     # x方向位移 m
            'displacement_y': 0.0,     # y方向位移 m
            'displacement_z': 0.0,     # z方向位移 m
            
            # 电边界条件参数
            'phi_potential': 0.0,      # 电位 V
            'current_density': 0.0,    # 电流密度 A/m²
            'resistance': 1e-3,        # 接触电阻 Ω·m²
        }
    
    def thermal_boundary_conditions(self, coordinates, T, T_grad, normal_vectors, bc_type='dirichlet'):
        """
        热边界条件
        
        Args:
            coordinates: 边界点坐标 [N, 4] (x, y, z, t)
            T: 温度场 [N, 1]
            T_grad: 温度梯度 [N, 3]
            normal_vectors: 法向量 [N, 3]
            bc_type: 边界条件类型
            
        Returns:
            residual: 边界条件残差 [N, 1]
        """
        if bc_type == 'dirichlet':
            # Dirichlet边界条件: T = T_boundary
            T_boundary = self.bc_params['T_ambient']
            residual = T - T_boundary
            
        elif bc_type == 'neumann':
            # Neumann边界条件: k ∂T/∂n = q_boundary
            k = 1.0  # 热导率
            q_boundary = self.bc_params['q_heat_flux']
            normal_grad = torch.sum(T_grad * normal_vectors, dim=1, keepdim=True)
            residual = k * normal_grad - q_boundary
            
        elif bc_type == 'convective':
            # 对流边界条件: k ∂T/∂n = h(T - T_ambient)
            k = 1.0  # 热导率
            h = self.bc_params['h_convection']
            T_ambient = self.bc_params['T_ambient']
            normal_grad = torch.sum(T_grad * normal_vectors, dim=1, keepdim=True)
            residual = k * normal_grad - h * (T - T_ambient)
            
        elif bc_type == 'insulated':
            # 绝热边界条件: ∂T/∂n = 0
            normal_grad = torch.sum(T_grad * normal_vectors, dim=1, keepdim=True)
            residual = normal_grad
            
        else:
            raise ValueError(f"未知的热边界条件类型: {bc_type}")
        
        return residual
    
    def mechanical_boundary_conditions(self, coordinates, u, stress, normal_vectors, bc_type='fixed'):
        """
        机械边界条件
        
        Args:
            coordinates: 边界点坐标 [N, 4]
            u: 位移场 [N, 3]
            stress: 应力张量 [N, 6] (σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz)
            normal_vectors: 法向量 [N, 3]
            bc_type: 边界条件类型
            
        Returns:
            residual: 边界条件残差 [N, 3]
        """
        if bc_type == 'fixed':
            # 固定边界条件: u = 0
            residual = u
            
        elif bc_type == 'free':
            # 自由边界条件: σ·n = 0
            # 将应力张量转换为矩阵形式
            stress_matrix = torch.zeros(u.shape[0], 3, 3, device=self.device)
            stress_matrix[:, 0, 0] = stress[:, 0]  # σ_xx
            stress_matrix[:, 1, 1] = stress[:, 1]  # σ_yy
            stress_matrix[:, 2, 2] = stress[:, 2]  # σ_zz
            stress_matrix[:, 0, 1] = stress_matrix[:, 1, 0] = stress[:, 3]  # σ_xy
            stress_matrix[:, 0, 2] = stress_matrix[:, 2, 0] = stress[:, 4]  # σ_xz
            stress_matrix[:, 1, 2] = stress_matrix[:, 2, 1] = stress[:, 5]  # σ_yz
            
            # 计算 σ·n
            traction = torch.bmm(stress_matrix, normal_vectors.unsqueeze(-1)).squeeze(-1)
            residual = traction
            
        elif bc_type == 'traction':
            # 牵引力边界条件: σ·n = t_boundary
            t_boundary = torch.tensor([
                self.bc_params['traction_x'],
                self.bc_params['traction_y'], 
                self.bc_params['traction_z']
            ], device=self.device).expand_as(u)
            
            # 计算 σ·n
            stress_matrix = torch.zeros(u.shape[0], 3, 3, device=self.device)
            stress_matrix[:, 0, 0] = stress[:, 0]
            stress_matrix[:, 1, 1] = stress[:, 1]
            stress_matrix[:, 2, 2] = stress[:, 2]
            stress_matrix[:, 0, 1] = stress_matrix[:, 1, 0] = stress[:, 3]
            stress_matrix[:, 0, 2] = stress_matrix[:, 2, 0] = stress[:, 4]
            stress_matrix[:, 1, 2] = stress_matrix[:, 2, 1] = stress[:, 5]
            
            traction = torch.bmm(stress_matrix, normal_vectors.unsqueeze(-1)).squeeze(-1)
            residual = traction - t_boundary
            
        elif bc_type == 'displacement':
            # 位移边界条件: u = u_boundary
            u_boundary = torch.tensor([
                self.bc_params['displacement_x'],
                self.bc_params['displacement_y'],
                self.bc_params['displacement_z']
            ], device=self.device).expand_as(u)
            residual = u - u_boundary
            
        else:
            raise ValueError(f"未知的机械边界条件类型: {bc_type}")
        
        return residual
    
    def electrical_boundary_conditions(self, coordinates, phi, phi_grad, normal_vectors, bc_type='potential'):
        """
        电边界条件
        
        Args:
            coordinates: 边界点坐标 [N, 4]
            phi: 电位场 [N, 1]
            phi_grad: 电位梯度 [N, 3]
            normal_vectors: 法向量 [N, 3]
            bc_type: 边界条件类型
            
        Returns:
            residual: 边界条件残差 [N, 1]
        """
        if bc_type == 'potential':
            # 电位边界条件: φ = φ_boundary
            phi_boundary = self.bc_params['phi_potential']
            residual = phi - phi_boundary
            
        elif bc_type == 'current':
            # 电流边界条件: i·n = I_boundary
            sigma = 1e6  # 电导率 S/m
            I_boundary = self.bc_params['current_density']
            current_density = -sigma * phi_grad
            normal_current = torch.sum(current_density * normal_vectors, dim=1, keepdim=True)
            residual = normal_current - I_boundary
            
        elif bc_type == 'insulated':
            # 绝缘边界条件: i·n = 0
            sigma = 1e6  # 电导率 S/m
            current_density = -sigma * phi_grad
            normal_current = torch.sum(current_density * normal_vectors, dim=1, keepdim=True)
            residual = normal_current
            
        elif bc_type == 'resistance':
            # 电阻边界条件: φ = R * i·n
            R = self.bc_params['resistance']
            sigma = 1e6  # 电导率 S/m
            current_density = -sigma * phi_grad
            normal_current = torch.sum(current_density * normal_vectors, dim=1, keepdim=True)
            residual = phi - R * normal_current
            
        else:
            raise ValueError(f"未知的电边界条件类型: {bc_type}")
    
    def coupled_boundary_conditions(self, coordinates, fields, normal_vectors, bc_type='thermal_electrical'):
        """
        耦合边界条件
        
        Args:
            coordinates: 边界点坐标 [N, 4]
            fields: 场变量字典 {'T': T, 'phi_e': phi_e, 'phi_s': phi_s, 'u': u}
            normal_vectors: 法向量 [N, 3]
            bc_type: 耦合边界条件类型
            
        Returns:
            residual: 边界条件残差
        """
        if bc_type == 'thermal_electrical':
            # 热电耦合边界条件
            T = fields['T']
            phi_e = fields['phi_e']
            phi_s = fields['phi_s']
            
            # 焦耳热边界条件: k ∂T/∂n = σ(∇φ)²
            k = 1.0
            sigma = 1e6
            T_grad = torch.autograd.grad(T, coordinates, grad_outputs=torch.ones_like(T),
                                       create_graph=True, retain_graph=True)[0][:, :3]
            phi_e_grad = torch.autograd.grad(phi_e, coordinates, grad_outputs=torch.ones_like(phi_e),
                                           create_graph=True, retain_graph=True)[0][:, :3]
            phi_s_grad = torch.autograd.grad(phi_s, coordinates, grad_outputs=torch.ones_like(phi_s),
                                           create_graph=True, retain_graph=True)[0][:, :3]
            
            normal_T_grad = torch.sum(T_grad * normal_vectors, dim=1, keepdim=True)
            joule_heating = sigma * (torch.sum(phi_e_grad**2, dim=1, keepdim=True) + 
                                   torch.sum(phi_s_grad**2, dim=1, keepdim=True))
            
            residual = k * normal_T_grad - joule_heating
            
        elif bc_type == 'thermo_mechanical':
            # 热机械耦合边界条件
            T = fields['T']
            u = fields['u']
            
            # 热膨胀边界条件
            alpha = 23e-6  # 热膨胀系数
            T_ref = 298.0  # 参考温度
            thermal_strain = alpha * (T - T_ref)
            
            # 简化的热膨胀位移边界条件
            residual = u - thermal_strain * coordinates[:, :3]
            
        else:
            raise ValueError(f"未知的耦合边界条件类型: {bc_type}")
        
        return residual
    
    def get_boundary_loss(self, model_outputs, coordinates, boundary_info):
        """
        计算边界条件损失
        
        Args:
            model_outputs: 神经网络输出
            coordinates: 坐标点
            boundary_info: 边界信息列表
                [{'type': 'thermal', 'bc_type': 'dirichlet', 'points': indices, 'normal': vectors}]
            
        Returns:
            total_loss: 总边界损失
            individual_losses: 各边界条件的损失
        """
        total_loss = 0.0
        individual_losses = {}
        
        for i, bc_info in enumerate(boundary_info):
            bc_type = bc_info['type']
            bc_subtype = bc_info['bc_type']
            points = bc_info['points']
            normal_vectors = bc_info['normal']
            
            # 提取边界点
            bc_coords = coordinates[points]
            bc_outputs = {key: value[points] for key, value in model_outputs.items()}
            
            if bc_type == 'thermal':
                T = bc_outputs['T']
                T_grad = torch.autograd.grad(T, bc_coords, grad_outputs=torch.ones_like(T),
                                           create_graph=True, retain_graph=True)[0][:, :3]
                residual = self.thermal_boundary_conditions(bc_coords, T, T_grad, normal_vectors, bc_subtype)
                
            elif bc_type == 'mechanical':
                u = bc_outputs['u']
                stress = bc_outputs['stress']
                residual = self.mechanical_boundary_conditions(bc_coords, u, stress, normal_vectors, bc_subtype)
                
            elif bc_type == 'electrical':
                phi = bc_outputs['phi_e']  # 或 phi_s
                phi_grad = torch.autograd.grad(phi, bc_coords, grad_outputs=torch.ones_like(phi),
                                             create_graph=True, retain_graph=True)[0][:, :3]
                residual = self.electrical_boundary_conditions(bc_coords, phi, phi_grad, normal_vectors, bc_subtype)
                
            elif bc_type == 'coupled':
                residual = self.coupled_boundary_conditions(bc_coords, bc_outputs, normal_vectors, bc_subtype)
                
            else:
                raise ValueError(f"未知的边界条件类型: {bc_type}")
            
            # 计算损失
            loss = torch.mean(residual**2)
            individual_losses[f'bc_{i}_{bc_type}_{bc_subtype}'] = loss
            total_loss += loss
        
        return total_loss, individual_losses

def create_boundary_mesh(domain_bounds, resolution=50):
    """
    创建边界网格
    
    Args:
        domain_bounds: 域边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max], [t_min, t_max]]
        resolution: 网格分辨率
        
    Returns:
        boundary_info: 边界信息
    """
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]
    z_min, z_max = domain_bounds[2]
    t_min, t_max = domain_bounds[3]
    
    boundary_info = []
    
    # 左边界 (x = x_min)
    x_left = np.full((resolution, resolution), x_min)
    y_left, z_left = np.meshgrid(np.linspace(y_min, y_max, resolution),
                                np.linspace(z_min, z_max, resolution), indexing='ij')
    t_left = np.random.uniform(t_min, t_max, (resolution, resolution))
    coords_left = np.stack([x_left.flatten(), y_left.flatten(), z_left.flatten(), t_left.flatten()], axis=1)
    normal_left = np.tile([-1, 0, 0], (coords_left.shape[0], 1))
    
    boundary_info.append({
        'type': 'thermal',
        'bc_type': 'dirichlet',
        'points': np.arange(coords_left.shape[0]),
        'normal': normal_left,
        'coordinates': coords_left
    })
    
    # 右边界 (x = x_max)
    x_right = np.full((resolution, resolution), x_max)
    y_right, z_right = np.meshgrid(np.linspace(y_min, y_max, resolution),
                                  np.linspace(z_min, z_max, resolution), indexing='ij')
    t_right = np.random.uniform(t_min, t_max, (resolution, resolution))
    coords_right = np.stack([x_right.flatten(), y_right.flatten(), z_right.flatten(), t_right.flatten()], axis=1)
    normal_right = np.tile([1, 0, 0], (coords_right.shape[0], 1))
    
    boundary_info.append({
        'type': 'thermal',
        'bc_type': 'convective',
        'points': np.arange(coords_left.shape[0], coords_left.shape[0] + coords_right.shape[0]),
        'normal': normal_right,
        'coordinates': coords_right
    })
    
    return boundary_info

def test_boundary_conditions():
    """
    测试边界条件
    """
    print("=== 测试边界条件 ===\n")
    
    # 创建边界条件实例
    bc = BoundaryConditions()
    
    # 测试数据
    N = 100
    coordinates = torch.randn(N, 4)
    T = torch.randn(N, 1) * 50 + 300
    T_grad = torch.randn(N, 3)
    u = torch.randn(N, 3) * 0.01
    stress = torch.randn(N, 6) * 1e6
    phi = torch.randn(N, 1)
    phi_grad = torch.randn(N, 3)
    normal_vectors = torch.randn(N, 3)
    normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)
    
    print("1. 热边界条件测试:")
    for bc_type in ['dirichlet', 'neumann', 'convective', 'insulated']:
        residual = bc.thermal_boundary_conditions(coordinates, T, T_grad, normal_vectors, bc_type)
        print(f"   {bc_type}: 残差形状 {residual.shape}, 范围 [{residual.min().item():.3f}, {residual.max().item():.3f}]")
    
    print("\n2. 机械边界条件测试:")
    for bc_type in ['fixed', 'free', 'traction', 'displacement']:
        residual = bc.mechanical_boundary_conditions(coordinates, u, stress, normal_vectors, bc_type)
        print(f"   {bc_type}: 残差形状 {residual.shape}, 范围 [{residual.min().item():.3f}, {residual.max().item():.3f}]")
    
    print("\n3. 电边界条件测试:")
    for bc_type in ['potential', 'current', 'insulated', 'resistance']:
        residual = bc.electrical_boundary_conditions(coordinates, phi, phi_grad, normal_vectors, bc_type)
        print(f"   {bc_type}: 残差形状 {residual.shape}, 范围 [{residual.min().item():.3f}, {residual.max().item():.3f}]")
    
    print("\n4. 边界网格生成测试:")
    domain_bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    boundary_info = create_boundary_mesh(domain_bounds, resolution=10)
    print(f"   生成边界数量: {len(boundary_info)}")
    for i, info in enumerate(boundary_info):
        print(f"   边界 {i}: {info['type']}_{info['bc_type']}, 点数: {len(info['points'])}")
    
    print("\n=== 边界条件测试完成 ===")

if __name__ == "__main__":
    test_boundary_conditions()