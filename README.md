# PINN物理信息神经网络数据集

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
