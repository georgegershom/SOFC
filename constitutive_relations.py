"""
本构关系实现 - 电池系统中PINN需要的各种本构关系
包含Hooke定律、Fourier定律、Butler-Volmer方程等
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import sympy as sp
from sympy import symbols, diff, simplify

class ConstitutiveRelations:
    """
    本构关系类 - 实现各种物理本构关系
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 材料参数
        self.material_params = {
            # 热学参数
            'k_thermal': 1.0,        # 热导率 W/(m·K)
            'rho': 2700.0,           # 密度 kg/m³
            'cp': 900.0,             # 比热容 J/(kg·K)
            'alpha_thermal': 23e-6,   # 热膨胀系数 1/K
            
            # 力学参数
            'E_young': 70e9,         # 杨氏模量 Pa
            'nu': 0.33,              # 泊松比
            'G': 26.3e9,             # 剪切模量 Pa
            'K': 70e9,               # 体积模量 Pa
            
            # 电学参数
            'sigma_electrical': 1e6, # 电导率 S/m
            'kappa_ion': 1e-3,       # 离子电导率 S/m
            'D_li': 1e-12,           # 锂离子扩散系数 m²/s
            
            # 电化学参数
            'F': 96485.0,            # 法拉第常数 C/mol
            'R': 8.314,              # 气体常数 J/(mol·K)
            'j0': 1e-3,              # 交换电流密度 A/m²
            'alpha': 0.5,            # 传递系数
        }
    
    def hooke_law(self, strain, temperature=None, concentration=None):
        """
        Hooke定律 - 应力-应变关系
        
        σ = C : ε + σ_thermal + σ_chemical
        
        Args:
            strain: 应变张量 [batch_size, 6] (ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz)
            temperature: 温度场 [batch_size, 1]
            concentration: 浓度场 [batch_size, 1]
            
        Returns:
            stress: 应力张量 [batch_size, 6]
        """
        batch_size = strain.shape[0]
        
        # 弹性常数矩阵 (各向同性材料)
        E = self.material_params['E_young']
        nu = self.material_params['nu']
        
        # 拉梅常数
        lambda_lame = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        
        # 构建弹性刚度矩阵 (6x6)
        C = torch.zeros(6, 6, device=self.device)
        
        # 对角项
        C[0, 0] = C[1, 1] = C[2, 2] = lambda_lame + 2 * mu
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        
        # 非对角项
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lambda_lame
        
        # 计算弹性应力
        stress_elastic = torch.matmul(strain, C.T)
        
        # 热应力
        if temperature is not None:
            alpha = self.material_params['alpha_thermal']
            thermal_strain = alpha * temperature
            stress_thermal = torch.zeros_like(stress_elastic)
            stress_thermal[:, 0] = stress_thermal[:, 1] = stress_thermal[:, 2] = -E * thermal_strain.squeeze()
            stress_elastic += stress_thermal
        
        # 化学应力 (浓度变化引起的应力)
        if concentration is not None:
            # 假设浓度变化引起的体积变化
            beta = 0.1  # 化学膨胀系数
            chemical_strain = beta * concentration
            stress_chemical = torch.zeros_like(stress_elastic)
            stress_chemical[:, 0] = stress_chemical[:, 1] = stress_chemical[:, 2] = -E * chemical_strain.squeeze()
            stress_elastic += stress_chemical
        
        return stress_elastic
    
    def fourier_law(self, temperature_grad):
        """
        Fourier定律 - 热传导
        
        q = -k ∇T
        
        Args:
            temperature_grad: 温度梯度 [batch_size, 3]
            
        Returns:
            heat_flux: 热流密度 [batch_size, 3]
        """
        k = self.material_params['k_thermal']
        heat_flux = -k * temperature_grad
        return heat_flux
    
    def butler_volmer(self, overpotential, temperature, concentration):
        """
        Butler-Volmer方程 - 电化学反应动力学
        
        j = j₀[exp(αFη/RT) - exp(-(1-α)Fη/RT)]
        
        Args:
            overpotential: 过电位 [batch_size, 1]
            temperature: 温度 [batch_size, 1]
            concentration: 浓度 [batch_size, 1]
            
        Returns:
            current_density: 电流密度 [batch_size, 1]
        """
        j0 = self.material_params['j0']
        alpha = self.material_params['alpha']
        F = self.material_params['F']
        R = self.material_params['R']
        
        # 浓度修正
        c_ref = 1.0  # 参考浓度
        j0_corrected = j0 * (concentration / c_ref) ** alpha
        
        # Butler-Volmer方程
        RT = R * temperature
        exponent1 = alpha * F * overpotential / RT
        exponent2 = -(1 - alpha) * F * overpotential / RT
        
        current_density = j0_corrected * (torch.exp(exponent1) - torch.exp(exponent2))
        
        return current_density
    
    def fick_law(self, concentration_grad, temperature=None):
        """
        Fick定律 - 质量扩散
        
        J = -D ∇c
        
        Args:
            concentration_grad: 浓度梯度 [batch_size, 3]
            temperature: 温度 [batch_size, 1] (用于温度相关的扩散系数)
            
        Returns:
            flux: 扩散通量 [batch_size, 3]
        """
        D = self.material_params['D_li']
        
        # 温度相关的扩散系数 (Arrhenius关系)
        if temperature is not None:
            Ea = 0.5  # 激活能 eV
            k_B = 8.617e-5  # 玻尔兹曼常数 eV/K
            T_ref = 298.0  # 参考温度 K
            D = D * torch.exp(-Ea / k_B * (1 / temperature - 1 / T_ref))
        
        flux = -D * concentration_grad
        return flux
    
    def ohm_law(self, potential_grad, conductivity=None):
        """
        Ohm定律 - 电传导
        
        i = -σ ∇φ
        
        Args:
            potential_grad: 电位梯度 [batch_size, 3]
            conductivity: 电导率 [batch_size, 1] (可选，用于非线性电导率)
            
        Returns:
            current_density: 电流密度 [batch_size, 3]
        """
        if conductivity is None:
            sigma = self.material_params['sigma_electrical']
        else:
            sigma = conductivity
        
        current_density = -sigma * potential_grad
        return current_density
    
    def nernst_equation(self, concentration, temperature, concentration_ref=1.0):
        """
        Nernst方程 - 平衡电位
        
        E = E₀ + (RT/nF) ln(c/c_ref)
        
        Args:
            concentration: 浓度 [batch_size, 1]
            temperature: 温度 [batch_size, 1]
            concentration_ref: 参考浓度
            
        Returns:
            potential: 平衡电位 [batch_size, 1]
        """
        R = self.material_params['R']
        F = self.material_params['F']
        n = 1  # 电子数
        
        E0 = 3.7  # 标准电极电位 V
        RT_nF = R * temperature / (n * F)
        
        potential = E0 + RT_nF * torch.log(concentration / concentration_ref + 1e-10)
        return potential
    
    def arrhenius_law(self, temperature, A, Ea, T_ref=298.0):
        """
        Arrhenius定律 - 温度相关的材料性质
        
        k(T) = A * exp(-Ea/(RT))
        
        Args:
            temperature: 温度 [batch_size, 1]
            A: 指前因子
            Ea: 激活能 J/mol
            T_ref: 参考温度 K
            
        Returns:
            property_value: 材料性质值 [batch_size, 1]
        """
        R = self.material_params['R']
        property_value = A * torch.exp(-Ea / (R * temperature))
        return property_value
    
    def get_effective_properties(self, porosity, tortuosity, base_property):
        """
        计算多孔介质中的有效性质
        
        k_eff = k * (ε/τ)^β
        
        Args:
            porosity: 孔隙率 [batch_size, 1]
            tortuosity: 弯曲度 [batch_size, 1]
            base_property: 基体性质 [batch_size, 1]
            
        Returns:
            effective_property: 有效性质 [batch_size, 1]
        """
        beta = 1.5  # Bruggeman指数
        effective_property = base_property * (porosity / tortuosity) ** beta
        return effective_property
    
    def stress_strain_energy(self, strain, stress):
        """
        计算应变能密度
        
        W = (1/2) σ : ε
        
        Args:
            strain: 应变张量 [batch_size, 6]
            stress: 应力张量 [batch_size, 6]
            
        Returns:
            energy_density: 应变能密度 [batch_size, 1]
        """
        energy_density = 0.5 * torch.sum(stress * strain, dim=1, keepdim=True)
        return energy_density
    
    def heat_capacity_temperature_dependent(self, temperature, c0, alpha_cp):
        """
        温度相关的比热容
        
        cp(T) = c0 * (1 + alpha_cp * T)
        
        Args:
            temperature: 温度 [batch_size, 1]
            c0: 参考比热容
            alpha_cp: 比热容温度系数
            
        Returns:
            cp: 比热容 [batch_size, 1]
        """
        cp = c0 * (1 + alpha_cp * temperature)
        return cp
    
    def thermal_conductivity_temperature_dependent(self, temperature, k0, alpha_k):
        """
        温度相关的热导率
        
        k(T) = k0 * (1 + alpha_k * T)
        
        Args:
            temperature: 温度 [batch_size, 1]
            k0: 参考热导率
            alpha_k: 热导率温度系数
            
        Returns:
            k: 热导率 [batch_size, 1]
        """
        k = k0 * (1 + alpha_k * temperature)
        return k

def test_constitutive_relations():
    """
    测试本构关系
    """
    print("=== 测试本构关系 ===\n")
    
    # 创建本构关系实例
    constitutive = ConstitutiveRelations()
    
    # 测试数据
    batch_size = 10
    strain = torch.randn(batch_size, 6) * 0.01  # 小应变
    temperature = torch.randn(batch_size, 1) * 50 + 300  # 温度 250-350K
    concentration = torch.rand(batch_size, 1) * 2.0  # 浓度 0-2
    overpotential = torch.randn(batch_size, 1) * 0.1  # 过电位 ±0.1V
    
    print("1. Hooke定律测试:")
    stress = constitutive.hooke_law(strain, temperature, concentration)
    print(f"   输入应变形状: {strain.shape}")
    print(f"   输出应力形状: {stress.shape}")
    print(f"   应力范围: [{stress.min().item():.3f}, {stress.max().item():.3f}] GPa")
    
    print("\n2. Fourier定律测试:")
    temp_grad = torch.randn(batch_size, 3)
    heat_flux = constitutive.fourier_law(temp_grad)
    print(f"   热流密度范围: [{heat_flux.min().item():.3f}, {heat_flux.max().item():.3f}] W/m²")
    
    print("\n3. Butler-Volmer方程测试:")
    current = constitutive.butler_volmer(overpotential, temperature, concentration)
    print(f"   电流密度范围: [{current.min().item():.6f}, {current.max().item():.6f}] A/m²")
    
    print("\n4. Fick定律测试:")
    conc_grad = torch.randn(batch_size, 3)
    flux = constitutive.fick_law(conc_grad, temperature)
    print(f"   扩散通量范围: [{flux.min().item():.12f}, {flux.max().item():.12f}] mol/(m²·s)")
    
    print("\n5. Nernst方程测试:")
    potential = constitutive.nernst_equation(concentration, temperature)
    print(f"   平衡电位范围: [{potential.min().item():.3f}, {potential.max().item():.3f}] V")
    
    print("\n=== 本构关系测试完成 ===")

if __name__ == "__main__":
    test_constitutive_relations()