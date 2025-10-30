"""
运行PINN物理数据集生成和测试
整合所有组件，生成完整的物理信息神经网络数据集
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# 导入自定义模块
from physics_equations import PhysicsEquations
from constitutive_relations import ConstitutiveRelations
from boundary_conditions import BoundaryConditions, create_boundary_mesh
from pinn_loss_calculator import PINNLossCalculator, PINNModel
from sample_dataset import SampleDatasetGenerator

def main():
    """
    主函数 - 运行完整的PINN物理数据集生成和测试
    """
    print("=" * 60)
    print("PINN物理信息神经网络数据集生成器")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print()
    
    # 1. 测试物理方程
    print("1. 测试物理方程定义...")
    physics = PhysicsEquations(device)
    print("   ✓ 物理方程定义完成")
    
    # 2. 测试本构关系
    print("\n2. 测试本构关系...")
    constitutive = ConstitutiveRelations(device)
    print("   ✓ 本构关系定义完成")
    
    # 3. 测试边界条件
    print("\n3. 测试边界条件...")
    boundary = BoundaryConditions(device)
    print("   ✓ 边界条件定义完成")
    
    # 4. 测试PINN损失计算器
    print("\n4. 测试PINN损失计算器...")
    loss_calculator = PINNLossCalculator(device)
    print("   ✓ PINN损失计算器完成")
    
    # 5. 生成示例数据集
    print("\n5. 生成示例数据集...")
    dataset_generator = SampleDatasetGenerator(device)
    
    # 生成训练数据集
    training_dataset = dataset_generator.generate_training_dataset(
        n_points=10000, 
        n_boundary=2000, 
        n_data=1000
    )
    print(f"   ✓ 训练数据集生成完成")
    print(f"   - 域内点: {training_dataset['coordinates']['domain'].shape[0]}")
    print(f"   - 边界点: {training_dataset['coordinates']['boundary'].shape[0]}")
    print(f"   - 观测数据点: {training_dataset['coordinates']['data'].shape[0]}")
    
    # 6. 保存数据集
    print("\n6. 保存数据集...")
    os.makedirs('/workspace/datasets', exist_ok=True)
    
    # 保存为不同格式
    dataset_generator.save_dataset(training_dataset, '/workspace/datasets/pinn_training_dataset.h5')
    dataset_generator.save_dataset(training_dataset, '/workspace/datasets/pinn_training_dataset.npz')
    
    # 保存为JSON格式的元数据
    metadata = {
        'creation_time': datetime.now().isoformat(),
        'device': device,
        'dataset_info': training_dataset['metadata'],
        'physics_equations': {
            'charge_conservation': physics.charge_conservation(None, None, None, None),
            'energy_conservation': physics.energy_conservation(None, None, None, None, None),
            'momentum_conservation': physics.momentum_conservation(None, None, None),
        },
        'constitutive_laws': {
            'hooke_law': 'σ = C : ε + σ_thermal + σ_chemical',
            'fourier_law': 'q = -k ∇T',
            'butler_volmer': 'j = j₀[exp(αFη/RT) - exp(-(1-α)Fη/RT)]',
            'fick_law': 'J = -D ∇c',
            'ohm_law': 'i = -σ ∇φ'
        },
        'boundary_conditions': {
            'thermal': ['dirichlet', 'neumann', 'convective', 'insulated'],
            'mechanical': ['fixed', 'free', 'traction', 'displacement'],
            'electrical': ['potential', 'current', 'insulated', 'resistance'],
            'coupled': ['thermal_electrical', 'thermo_mechanical']
        }
    }
    
    with open('/workspace/datasets/dataset_metadata.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("   ✓ 数据集已保存:")
    print("     - /workspace/datasets/pinn_training_dataset.h5")
    print("     - /workspace/datasets/pinn_training_dataset.npz")
    print("     - /workspace/datasets/dataset_metadata.json")
    
    # 7. 测试PINN模型
    print("\n7. 测试PINN模型...")
    model = PINNModel(input_dim=4, hidden_dim=64, output_dim=5).to(device)
    
    # 生成测试坐标
    test_coords = torch.randn(1000, 4, device=device, requires_grad=True)
    
    # 前向传播
    model_outputs = model(test_coords)
    
    print("   ✓ PINN模型测试完成")
    print("   模型输出:")
    for key, value in model_outputs.items():
        print(f"     {key}: {value.shape}")
    
    # 8. 测试损失计算
    print("\n8. 测试损失计算...")
    
    # 简化测试 - 只测试PDE损失
    pde_loss, pde_individual = loss_calculator.calculate_pde_loss(model_outputs, test_coords)
    
    print("   ✓ PDE损失计算完成")
    print("   PDE损失分解:")
    print(f"     总PDE损失: {pde_loss.item():.6f}")
    for name, loss in pde_individual.items():
        print(f"     {name}: {loss.item():.6f}")
    
    # 9. 生成可视化
    print("\n9. 生成可视化...")
    os.makedirs('/workspace/visualizations', exist_ok=True)
    
    # 可视化数据集
    dataset_generator.visualize_dataset(
        training_dataset, 
        '/workspace/visualizations/dataset_visualization.png'
    )
    
    # 绘制损失历史
    loss_calculator.plot_loss_history('/workspace/visualizations/loss_history.png')
    
    print("   ✓ 可视化完成")
    print("   图像保存至: /workspace/visualizations/")
    
    # 10. 生成使用示例
    print("\n10. 生成使用示例...")
    generate_usage_example()
    
    # 11. 生成文档
    print("\n11. 生成文档...")
    generate_documentation()
    
    print("\n" + "=" * 60)
    print("PINN物理数据集生成完成!")
    print("=" * 60)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("生成的文件:")
    print("  - 物理方程定义: physics_equations.py")
    print("  - 本构关系: constitutive_relations.py") 
    print("  - 边界条件: boundary_conditions.py")
    print("  - PINN损失计算器: pinn_loss_calculator.py")
    print("  - 示例数据集: sample_dataset.py")
    print("  - 训练数据集: /workspace/datasets/")
    print("  - 可视化图像: /workspace/visualizations/")
    print("  - 使用示例: usage_example.py")
    print("  - 文档: README.md")

def generate_usage_example():
    """
    生成使用示例
    """
    usage_code = '''"""
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
'''
    
    with open('/workspace/usage_example.py', 'w', encoding='utf-8') as f:
        f.write(usage_code)
    
    print("   ✓ 使用示例已生成: usage_example.py")

def generate_documentation():
    """
    生成文档
    """
    doc_content = '''# PINN物理信息神经网络数据集

## 概述

本项目为物理信息神经网络(PINN)提供了完整的物理方程定义、本构关系和边界条件，特别针对电池系统中的多物理场耦合问题。

## 文件结构

```
/workspace/
├── physics_equations.py          # 物理方程定义
├── constitutive_relations.py     # 本构关系实现
├── boundary_conditions.py        # 边界条件定义
├── pinn_loss_calculator.py       # PINN损失计算器
├── sample_dataset.py             # 示例数据集生成器
├── run_pinn_physics_dataset.py   # 主运行文件
├── usage_example.py              # 使用示例
├── datasets/                     # 生成的数据集
│   ├── pinn_training_dataset.h5
│   ├── pinn_training_dataset.npz
│   └── dataset_metadata.json
└── visualizations/               # 可视化图像
    ├── dataset_visualization.png
    └── loss_history.png
```

## 物理方程

### 1. 电荷守恒方程
- 离子电流守恒: ∇·(κ_eff ∇φ_e) + ∇·(κ_D ∇ln(c_li)) = 0
- 电子电流守恒: ∇·(σ_eff ∇φ_s) - j = 0

### 2. 能量守恒方程
- 热方程: ρcp ∂T/∂t = ∇·(k ∇T) + Q_joule + Q_reaction + Q_mechanical
- 焦耳热: Q_joule = σ(∇φ)²
- 反应热: Q_reaction = j·η
- 机械功: Q_mechanical = αT ∇·σ

### 3. 动量守恒方程
- 机械平衡: ∇·σ = 0

## 本构关系

### 1. Hooke定律
- 应力-应变关系: σ = C : ε
- 热应力: σ_thermal = -αT E I

### 2. Fourier定律
- 热传导: q = -k ∇T

### 3. Butler-Volmer方程
- 电化学反应: j = j₀[exp(αFη/RT) - exp(-(1-α)Fη/RT)]

### 4. Fick定律
- 质量扩散: J = -D ∇c

### 5. Ohm定律
- 电传导: i = -σ ∇φ

## 边界条件

### 1. 热边界条件
- Dirichlet: T = T_boundary
- Neumann: k ∂T/∂n = q_boundary
- 对流: k ∂T/∂n = h(T - T_ambient)
- 绝热: ∂T/∂n = 0

### 2. 机械边界条件
- 固定: u = 0
- 自由: σ·n = 0
- 牵引力: σ·n = t_boundary
- 位移: u = u_boundary

### 3. 电边界条件
- 电位: φ = φ_boundary
- 电流: i·n = I_boundary
- 绝缘: i·n = 0
- 电阻: φ = R * i·n

## 使用方法

### 1. 基本使用

```python
from pinn_loss_calculator import PINNLossCalculator, PINNModel
from sample_dataset import SampleDatasetGenerator

# 创建模型
model = PINNModel(input_dim=4, hidden_dim=64, output_dim=5)

# 创建损失计算器
loss_calculator = PINNLossCalculator()

# 生成数据集
dataset_generator = SampleDatasetGenerator()
dataset = dataset_generator.generate_training_dataset()

# 训练模型
# ... 训练代码 ...
```

### 2. 自定义物理方程

```python
from physics_equations import PhysicsEquations

physics = PhysicsEquations()

# 获取物理方程
charge_eqs = physics.charge_conservation(phi_e, phi_s, c_li, T)
energy_eqs = physics.energy_conservation(T, phi_e, phi_s, c_li, stress)
momentum_eqs = physics.momentum_conservation(u, T, c_li)
```

### 3. 自定义本构关系

```python
from constitutive_relations import ConstitutiveRelations

constitutive = ConstitutiveRelations()

# 计算应力
stress = constitutive.hooke_law(strain, temperature, concentration)

# 计算热流
heat_flux = constitutive.fourier_law(temperature_grad)

# 计算电流密度
current = constitutive.butler_volmer(overpotential, temperature, concentration)
```

## 数据集格式

### HDF5格式 (.h5)
- 坐标数据: `/coordinates/domain`, `/coordinates/boundary`, `/coordinates/data`
- 解析解: `/solutions/analytical/T`, `/solutions/analytical/phi_e`, etc.
- 实验数据: `/solutions/experimental/T`, `/solutions/experimental/phi_e`, etc.
- 元数据: `/metadata/`

### NPZ格式 (.npz)
- 所有数据以扁平化数组形式存储
- 键名格式: `coords_domain`, `analytical_T`, `experimental_T`, etc.

## 参数说明

### 物理参数
- `k_thermal`: 热导率 (W/(m·K))
- `rho`: 密度 (kg/m³)
- `cp`: 比热容 (J/(kg·K))
- `E_young`: 杨氏模量 (Pa)
- `nu`: 泊松比
- `alpha_thermal`: 热膨胀系数 (1/K)
- `sigma_electrical`: 电导率 (S/m)
- `F`: 法拉第常数 (C/mol)
- `R`: 气体常数 (J/(mol·K))

### 训练参数
- `n_points`: 域内点数
- `n_boundary`: 边界点数
- `n_data`: 观测数据点数
- `hidden_dim`: 神经网络隐藏层维度
- `learning_rate`: 学习率

## 注意事项

1. 确保PyTorch版本兼容性
2. 根据具体问题调整物理参数
3. 边界条件需要根据实际问题设置
4. 损失权重可能需要根据收敛情况调整
5. 建议使用GPU加速训练

## 扩展

本项目提供了良好的模块化设计，可以轻松扩展：
- 添加新的物理方程
- 实现新的本构关系
- 定义新的边界条件
- 集成其他数值方法

## 引用

如果您在研究中使用了本项目，请引用相关论文和代码。
'''
    
    with open('/workspace/README.md', 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print("   ✓ 文档已生成: README.md")

if __name__ == "__main__":
    main()