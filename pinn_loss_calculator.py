"""
PINN物理损失函数计算器
整合物理方程、本构关系和边界条件，计算PINN训练所需的物理损失
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from physics_equations import PhysicsEquations
from constitutive_relations import ConstitutiveRelations
from boundary_conditions import BoundaryConditions

class PINNLossCalculator:
    """
    PINN物理损失函数计算器
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 初始化各个组件
        self.physics = PhysicsEquations(device)
        self.constitutive = ConstitutiveRelations(device)
        self.boundary = BoundaryConditions(device)
        
        # 损失权重
        self.loss_weights = {
            'pde': {
                'charge_ion': 1.0,
                'charge_electron': 1.0,
                'energy': 1.0,
                'momentum': 1.0
            },
            'boundary': {
                'thermal': 1.0,
                'mechanical': 1.0,
                'electrical': 1.0,
                'coupled': 1.0
            },
            'data': 1.0,
            'initial': 1.0
        }
        
        # 损失历史记录
        self.loss_history = {
            'total': [],
            'pde': [],
            'boundary': [],
            'data': [],
            'initial': []
        }
    
    def calculate_pde_loss(self, model_outputs, coordinates):
        """
        计算PDE损失
        
        Args:
            model_outputs: 神经网络输出字典
            coordinates: 坐标点 [N, 4]
            
        Returns:
            pde_loss: PDE总损失
            individual_losses: 各PDE的损失
        """
        # 计算PDE残差
        residuals = self.physics.get_pde_residuals(model_outputs, coordinates)
        
        # 计算各PDE损失
        individual_losses = {}
        pde_loss = 0.0
        
        for name, residual in residuals.items():
            loss = torch.mean(residual**2)
            individual_losses[name] = loss
            pde_loss += self.loss_weights['pde'].get(name, 1.0) * loss
        
        return pde_loss, individual_losses
    
    def calculate_boundary_loss(self, model_outputs, coordinates, boundary_info):
        """
        计算边界条件损失
        
        Args:
            model_outputs: 神经网络输出
            coordinates: 坐标点
            boundary_info: 边界信息
            
        Returns:
            boundary_loss: 边界总损失
            individual_losses: 各边界条件的损失
        """
        boundary_loss, individual_losses = self.boundary.get_boundary_loss(
            model_outputs, coordinates, boundary_info
        )
        
        return boundary_loss, individual_losses
    
    def calculate_data_loss(self, model_outputs, data_targets, data_indices):
        """
        计算数据损失 (如果有观测数据)
        
        Args:
            model_outputs: 神经网络输出
            data_targets: 观测数据目标值
            data_indices: 数据点索引
            
        Returns:
            data_loss: 数据损失
        """
        if data_targets is None or data_indices is None:
            return torch.tensor(0.0, device=self.device)
        
        data_loss = 0.0
        for field_name, target_values in data_targets.items():
            if field_name in model_outputs:
                predicted_values = model_outputs[field_name][data_indices]
                field_loss = torch.mean((predicted_values - target_values)**2)
                data_loss += field_loss
        
        return data_loss
    
    def calculate_initial_condition_loss(self, model_outputs, coordinates, initial_conditions):
        """
        计算初始条件损失
        
        Args:
            model_outputs: 神经网络输出
            coordinates: 坐标点
            initial_conditions: 初始条件字典
            
        Returns:
            initial_loss: 初始条件损失
        """
        if initial_conditions is None:
            return torch.tensor(0.0, device=self.device)
        
        # 找到t=0的点
        t_zero_mask = torch.abs(coordinates[:, 3]) < 1e-6
        if not torch.any(t_zero_mask):
            return torch.tensor(0.0, device=self.device)
        
        initial_coords = coordinates[t_zero_mask]
        initial_outputs = {key: value[t_zero_mask] for key, value in model_outputs.items()}
        
        initial_loss = 0.0
        for field_name, initial_value in initial_conditions.items():
            if field_name in initial_outputs:
                predicted_value = initial_outputs[field_name]
                field_loss = torch.mean((predicted_value - initial_value)**2)
                initial_loss += field_loss
        
        return initial_loss
    
    def calculate_total_loss(self, model_outputs, coordinates, boundary_info=None, 
                           data_targets=None, data_indices=None, initial_conditions=None):
        """
        计算总损失函数
        
        Args:
            model_outputs: 神经网络输出
            coordinates: 坐标点
            boundary_info: 边界信息
            data_targets: 观测数据
            data_indices: 数据点索引
            initial_conditions: 初始条件
            
        Returns:
            total_loss: 总损失
            loss_breakdown: 损失分解
        """
        loss_breakdown = {}
        
        # 1. PDE损失
        pde_loss, pde_individual = self.calculate_pde_loss(model_outputs, coordinates)
        loss_breakdown['pde'] = pde_loss
        loss_breakdown.update(pde_individual)
        
        # 2. 边界条件损失
        if boundary_info is not None:
            boundary_loss, boundary_individual = self.calculate_boundary_loss(
                model_outputs, coordinates, boundary_info
            )
            loss_breakdown['boundary'] = boundary_loss
            loss_breakdown.update(boundary_individual)
        else:
            boundary_loss = torch.tensor(0.0, device=self.device)
            loss_breakdown['boundary'] = boundary_loss
        
        # 3. 数据损失
        data_loss = self.calculate_data_loss(model_outputs, data_targets, data_indices)
        loss_breakdown['data'] = data_loss
        
        # 4. 初始条件损失
        initial_loss = self.calculate_initial_condition_loss(model_outputs, coordinates, initial_conditions)
        loss_breakdown['initial'] = initial_loss
        
        # 5. 总损失
        total_loss = (self.loss_weights['pde']['charge_ion'] * pde_loss +
                     self.loss_weights['boundary']['thermal'] * boundary_loss +
                     self.loss_weights['data'] * data_loss +
                     self.loss_weights['initial'] * initial_loss)
        
        loss_breakdown['total'] = total_loss
        
        # 记录损失历史
        self.loss_history['total'].append(total_loss.item())
        self.loss_history['pde'].append(pde_loss.item())
        self.loss_history['boundary'].append(boundary_loss.item())
        self.loss_history['data'].append(data_loss.item())
        self.loss_history['initial'].append(initial_loss.item())
        
        return total_loss, loss_breakdown
    
    def adaptive_loss_balancing(self, loss_breakdown, adaptation_rate=0.01):
        """
        自适应损失平衡
        
        Args:
            loss_breakdown: 损失分解
            adaptation_rate: 适应率
        """
        # 计算各损失的相对大小
        total_loss = loss_breakdown['total']
        
        if total_loss > 0:
            pde_ratio = loss_breakdown['pde'] / total_loss
            boundary_ratio = loss_breakdown['boundary'] / total_loss
            data_ratio = loss_breakdown['data'] / total_loss
            initial_ratio = loss_breakdown['initial'] / total_loss
            
            # 调整权重
            if pde_ratio < 0.1:  # PDE损失太小
                self.loss_weights['pde']['charge_ion'] *= (1 + adaptation_rate)
                self.loss_weights['pde']['charge_electron'] *= (1 + adaptation_rate)
                self.loss_weights['pde']['energy'] *= (1 + adaptation_rate)
                self.loss_weights['pde']['momentum'] *= (1 + adaptation_rate)
            
            if boundary_ratio < 0.1:  # 边界损失太小
                self.loss_weights['boundary']['thermal'] *= (1 + adaptation_rate)
                self.loss_weights['boundary']['mechanical'] *= (1 + adaptation_rate)
                self.loss_weights['boundary']['electrical'] *= (1 + adaptation_rate)
    
    def plot_loss_history(self, save_path=None):
        """
        绘制损失历史
        
        Args:
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history['total'], label='Total Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Total Loss History')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history['pde'], label='PDE Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('PDE Loss History')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_history['boundary'], label='Boundary Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Boundary Loss History')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.loss_history['data'], label='Data Loss')
        plt.plot(self.loss_history['initial'], label='Initial Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Data & Initial Loss History')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_loss_statistics(self):
        """
        获取损失统计信息
        
        Returns:
            stats: 损失统计字典
        """
        stats = {}
        
        for loss_type, history in self.loss_history.items():
            if len(history) > 0:
                stats[loss_type] = {
                    'initial': history[0],
                    'final': history[-1],
                    'min': min(history),
                    'max': max(history),
                    'mean': np.mean(history),
                    'std': np.std(history)
                }
        
        return stats

class PINNModel(nn.Module):
    """
    简单的PINN模型示例
    """
    
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=5):
        super(PINNModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        output = self.network(x)
        
        # 分离输出
        T = output[:, 0:1]  # 温度
        u = output[:, 1:4]  # 位移 (3D)
        phi_e = output[:, 4:5]  # 电解质电位
        phi_s = output[:, 4:5]  # 固体电极电位 (简化)
        c_li = torch.nn.functional.softplus(output[:, 4:5])  # 锂离子浓度 (确保为正)
        
        return {
            'T': T,
            'u': u,
            'phi_e': phi_e,
            'phi_s': phi_s,
            'c_li': c_li,
            'stress': torch.zeros_like(u)  # 简化，实际应通过本构关系计算
        }

def test_pinn_loss_calculator():
    """
    测试PINN损失计算器
    """
    print("=== 测试PINN损失计算器 ===\n")
    
    # 创建损失计算器
    loss_calculator = PINNLossCalculator()
    
    # 创建模型
    model = PINNModel()
    
    # 生成测试数据
    N = 1000
    coordinates = torch.randn(N, 4, requires_grad=True)
    
    # 前向传播
    model_outputs = model(coordinates)
    
    print("1. 模型输出形状:")
    for key, value in model_outputs.items():
        print(f"   {key}: {value.shape}")
    
    # 计算PDE损失
    print("\n2. PDE损失计算:")
    pde_loss, pde_individual = loss_calculator.calculate_pde_loss(model_outputs, coordinates)
    print(f"   总PDE损失: {pde_loss.item():.6f}")
    for name, loss in pde_individual.items():
        print(f"   {name}: {loss.item():.6f}")
    
    # 创建边界信息
    from boundary_conditions import create_boundary_mesh
    domain_bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    boundary_info = create_boundary_mesh(domain_bounds, resolution=10)
    
    # 计算边界损失
    print("\n3. 边界损失计算:")
    boundary_loss, boundary_individual = loss_calculator.calculate_boundary_loss(
        model_outputs, coordinates, boundary_info
    )
    print(f"   总边界损失: {boundary_loss.item():.6f}")
    for name, loss in boundary_individual.items():
        print(f"   {name}: {loss.item():.6f}")
    
    # 计算总损失
    print("\n4. 总损失计算:")
    total_loss, loss_breakdown = loss_calculator.calculate_total_loss(
        model_outputs, coordinates, boundary_info
    )
    print(f"   总损失: {total_loss.item():.6f}")
    for name, loss in loss_breakdown.items():
        print(f"   {name}: {loss.item():.6f}")
    
    # 损失统计
    print("\n5. 损失统计:")
    stats = loss_calculator.get_loss_statistics()
    for loss_type, stat in stats.items():
        print(f"   {loss_type}:")
        for key, value in stat.items():
            print(f"     {key}: {value:.6f}")
    
    print("\n=== PINN损失计算器测试完成 ===")

if __name__ == "__main__":
    test_pinn_loss_calculator()