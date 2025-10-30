"""
PINN物理信息神经网络使用示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinn_loss_calculator import PINNLossCalculator, PINNModel
from sample_dataset import SampleDatasetGenerator
from boundary_conditions import create_boundary_mesh

def train_pinn_model():
    """
    训练PINN模型示例
    """
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 加载数据集
    dataset_generator = SampleDatasetGenerator(device)
    dataset = dataset_generator.load_dataset('/workspace/datasets/pinn_training_dataset.h5')
    
    # 2. 创建模型
    model = PINNModel(input_dim=4, hidden_dim=64, output_dim=5).to(device)
    
    # 3. 创建损失计算器
    loss_calculator = PINNLossCalculator(device)
    
    # 4. 准备训练数据
    domain_coords = torch.tensor(dataset['coordinates']['domain'], 
                                device=device, requires_grad=True)
    boundary_info = create_boundary_mesh([[0, 1], [0, 1], [0, 1], [0, 1]], resolution=20)
    
    # 观测数据
    data_coords = torch.tensor(dataset['coordinates']['data'], 
                              device=device, requires_grad=True)
    data_targets = {
        'T': torch.tensor(dataset['solutions']['experimental']['T'], device=device),
        'phi_e': torch.tensor(dataset['solutions']['experimental']['phi_e'], device=device),
        'phi_s': torch.tensor(dataset['solutions']['experimental']['phi_s'], device=device),
        'c_li': torch.tensor(dataset['solutions']['experimental']['c_li'], device=device),
        'u': torch.tensor(dataset['solutions']['experimental']['u'], device=device)
    }
    data_indices = torch.arange(len(data_coords), device=device)
    
    # 5. 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. 训练循环
    n_epochs = 1000
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 前向传播
        model_outputs = model(domain_coords)
        
        # 计算损失
        total_loss, loss_breakdown = loss_calculator.calculate_total_loss(
            model_outputs, domain_coords, boundary_info, 
            data_targets, data_indices
        )
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 打印进度
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}")
            for name, loss in loss_breakdown.items():
                print(f"  {name}: {loss.item():.6f}")
    
    print("训练完成!")
    return model

if __name__ == "__main__":
    model = train_pinn_model()
