# PINN物理方程使用指南
# Physics-Informed Neural Network - Physics Equations Usage Guide

## 概述 / Overview

本文档说明如何使用 `pinn_physics_equations.py` 模块来构建电池系统的物理信息神经网络（PINN）。

## 目录

1. [物理方程概述](#物理方程概述)
2. [快速开始](#快速开始)
3. [详细使用说明](#详细使用说明)
4. [方程说明](#方程说明)
5. [边界条件](#边界条件)
6. [参数配置](#参数配置)
7. [训练流程示例](#训练流程示例)

---

## 物理方程概述

### 1. 电荷守恒方程 (Charge Conservation)

**离子相电荷守恒:**
```
∇·(σ_i·∇φ_i) = -j_rxn
```

**电子相电荷守恒:**
```
∇·(σ_e·∇φ_e) = j_rxn
```

其中:
- `φ_i`: 离子相电位 (V)
- `φ_e`: 电子相（固相）电位 (V)
- `σ_i`: 离子电导率 (S/m)
- `σ_e`: 电子电导率 (S/m)
- `j_rxn`: 电化学反应电流密度 (A/m³)

### 2. 能量守恒方程 (Energy Conservation)

**热方程（含焦耳热和电化学反应热）:**
```
ρ·cp·∂T/∂t = ∇·(k·∇T) + Q_joule + Q_rxn
```

其中:
- `T`: 温度 (K)
- `ρ`: 密度 (kg/m³)
- `cp`: 比热容 (J/(kg·K))
- `k`: 热导率 (W/(m·K))
- `Q_joule`: 焦耳热 = σ_i·|∇φ_i|² + σ_e·|∇φ_e|² (W/m³)
- `Q_rxn`: 电化学反应热 (W/m³)

### 3. 线性动量守恒方程 (Linear Momentum Conservation)

**机械平衡方程（准静态）:**
```
∇·σ = 0
```

展开为三个方向:
```
∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z = 0  (X方向)
∂σ_yx/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z = 0  (Y方向)
∂σ_zx/∂x + ∂σ_zy/∂y + ∂σ_zz/∂z = 0  (Z方向)
```

其中:
- `σ_ij`: 应力张量分量 (Pa)
- `u, v, w`: X, Y, Z方向的位移 (m)

### 4. 本构定律 (Constitutive Laws)

**热弹性胡克定律 (Hooke's Law with Thermal Effects):**
```
σ = λ·tr(ε)·I + 2μ·ε - (3λ + 2μ)·α·ΔT·I
```

**傅里叶热传导定律 (Fourier's Law):**
```
q = -k·∇T
```

**Butler-Volmer电化学动力学方程:**
```
j = i_0 · [exp(α_a·F·η/(R·T)) - exp(-α_c·F·η/(R·T))]
```

其中:
- `λ, μ`: 拉梅常数
- `ε`: 应变张量
- `α`: 热膨胀系数 (1/K)
- `i_0`: 交换电流密度 (A/m²)
- `η`: 过电位 (V)

---

## 快速开始

### 安装依赖

```bash
pip install torch numpy
```

### 基本使用

```python
import torch
from pinn_physics_equations import BatteryPhysicsEquations, create_default_parameters

# 1. 创建物理参数
params = create_default_parameters()

# 2. 初始化物理方程对象
physics = BatteryPhysicsEquations(params)

# 3. 在训练循环中使用
# 假设你有一个神经网络 model
predictions = {
    'T': model_T(x, y, z, t),
    'phi_i': model_phi_i(x, y, z, t),
    'phi_e': model_phi_e(x, y, z, t),
    'u': model_u(x, y, z, t),
    'v': model_v(x, y, z, t),
    'w': model_w(x, y, z, t)
}

coords = {'x': x, 'y': y, 'z': z, 't': t}

# 4. 计算物理损失
physics_losses = physics.compute_total_physics_loss(predictions, coords)

# 5. 总损失 = 数据损失 + 物理损失
total_loss = data_loss + physics_losses['total']
```

---

## 详细使用说明

### 步骤1: 自定义物理参数

```python
# 根据你的电池材料修改参数
params = {
    # 电学性质
    'sigma_ionic': 1.0,          # 电解液离子电导率
    'sigma_electronic': 1e5,     # 电极电子电导率
    
    # 热学性质
    'k_thermal': 1.5,            # 热导率
    'rho': 2000.0,               # 密度
    'cp': 1000.0,                # 比热容
    
    # 机械性质
    'E': 10e9,                   # 杨氏模量 (10 GPa)
    'nu': 0.3,                   # 泊松比
    'alpha': 1e-5,               # 热膨胀系数
    
    # 电化学性质
    'i0': 1.0,                   # 交换电流密度
    'alpha_a': 0.5,              # 阳极传递系数
    'alpha_c': 0.5,              # 阴极传递系数
    
    # 物理常数
    'F': 96485.0,                # 法拉第常数
    'R': 8.314,                  # 气体常数
    
    # 参考条件
    'T_ref': 298.15,             # 参考温度 (25°C)
    'T_operating': 298.15        # 工作温度
}

physics = BatteryPhysicsEquations(params)
```

### 步骤2: 准备训练数据点

```python
# 创建配点（collocation points）用于计算物理损失
n_collocation = 10000

# 域内点
x_col = torch.rand(n_collocation, 1, requires_grad=True) * L_x
y_col = torch.rand(n_collocation, 1, requires_grad=True) * L_y
z_col = torch.rand(n_collocation, 1, requires_grad=True) * L_z
t_col = torch.rand(n_collocation, 1, requires_grad=True) * T_max

# 边界点
n_boundary = 1000
x_boundary = torch.zeros(n_boundary, 1, requires_grad=True)  # 例如 x=0 边界
# ... 为其他边界创建点
```

### 步骤3: 定义神经网络

```python
import torch.nn as nn

class BatteryPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, y, z, t):
        # 输入: [x, y, z, t]
        inputs = torch.cat([x, y, z, t], dim=1)
        
        # 前向传播
        h = inputs
        for i, layer in enumerate(self.layers[:-1]):
            h = torch.tanh(layer(h))
        h = self.layers[-1](h)
        
        # 输出: [T, phi_i, phi_e, u, v, w]
        return h

# 创建网络 (4输入 -> 6输出)
model = BatteryPINN([4, 64, 64, 64, 6])
```

### 步骤4: 定义损失函数

```python
def compute_loss(model, physics, x_data, y_data, x_col, coords_col):
    """
    计算总损失 = 数据损失 + 物理损失
    """
    # 数据损失（如果有实验数据）
    if x_data is not None:
        y_pred = model(*x_data)
        data_loss = torch.mean((y_pred - y_data)**2)
    else:
        data_loss = 0.0
    
    # 物理损失
    outputs = model(*x_col)
    predictions = {
        'T': outputs[:, 0:1],
        'phi_i': outputs[:, 1:2],
        'phi_e': outputs[:, 2:3],
        'u': outputs[:, 3:4],
        'v': outputs[:, 4:5],
        'w': outputs[:, 5:6]
    }
    
    physics_losses = physics.compute_total_physics_loss(predictions, coords_col)
    
    # 总损失（可以添加权重）
    lambda_data = 1.0
    lambda_physics = 1.0
    
    total_loss = lambda_data * data_loss + lambda_physics * physics_losses['total']
    
    return total_loss, data_loss, physics_losses
```

### 步骤5: 训练循环

```python
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练
n_epochs = 10000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # 计算损失
    total_loss, data_loss, physics_losses = compute_loss(
        model, physics, 
        x_data, y_data,
        [x_col, y_col, z_col, t_col],
        {'x': x_col, 'y': y_col, 'z': z_col, 't': t_col}
    )
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    # 打印进度
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{n_epochs}")
        print(f"  Total Loss: {total_loss.item():.6e}")
        print(f"  Data Loss: {data_loss:.6e}")
        print(f"  Physics Loss: {physics_losses['total'].item():.6e}")
        print(f"    - Charge (ionic): {physics_losses['charge_ionic'].item():.6e}")
        print(f"    - Charge (electronic): {physics_losses['charge_electronic'].item():.6e}")
        print(f"    - Energy: {physics_losses['energy'].item():.6e}")
        print(f"    - Momentum X: {physics_losses['momentum_x'].item():.6e}")
        print(f"    - Momentum Y: {physics_losses['momentum_y'].item():.6e}")
        print(f"    - Momentum Z: {physics_losses['momentum_z'].item():.6e}")
```

---

## 方程说明

### 电荷守恒方程的使用

```python
# 单独计算离子电荷守恒残差
residual_ionic = physics.charge_conservation_ionic(phi_i, x, y, z, t)

# 单独计算电子电荷守恒残差
residual_electronic = physics.charge_conservation_electronic(phi_e, x, y, z, t)

# 计算损失
loss_charge = torch.mean(residual_ionic**2) + torch.mean(residual_electronic**2)
```

### 能量守恒方程的使用

```python
# 计算能量守恒残差
residual_energy = physics.energy_conservation(T, phi_i, phi_e, x, y, z, t)

# 计算损失
loss_energy = torch.mean(residual_energy**2)
```

### 动量守恒方程的使用

```python
# 计算三个方向的动量守恒残差
residual_x = physics.momentum_conservation_x(u, v, w, T, x, y, z)
residual_y = physics.momentum_conservation_y(u, v, w, T, x, y, z)
residual_z = physics.momentum_conservation_z(u, v, w, T, x, y, z)

# 计算损失
loss_momentum = (torch.mean(residual_x**2) + 
                 torch.mean(residual_y**2) + 
                 torch.mean(residual_z**2))
```

---

## 边界条件

### 热边界条件

#### 1. 绝热边界（热绝缘）

```python
# 例如: 在 x=0 边界上绝热
n_x, n_y, n_z = 1.0, 0.0, 0.0  # 法向量

residual_bc = physics.thermal_boundary_insulated(T, n_x, n_y, n_z, x, y, z)
loss_bc = torch.mean(residual_bc**2)
```

#### 2. 对流边界

```python
# 例如: 在 x=L 边界上对流换热
T_ambient = 298.15  # 环境温度 (K)
h = 10.0            # 对流换热系数 (W/(m²·K))
n_x, n_y, n_z = 1.0, 0.0, 0.0

residual_bc = physics.thermal_boundary_convective(
    T, T_ambient, h, n_x, n_y, n_z, x, y, z
)
loss_bc = torch.mean(residual_bc**2)
```

#### 3. 固定温度边界

```python
# 例如: 在某边界上固定温度
T_fixed = 300.0  # K

residual_bc = physics.thermal_boundary_fixed(T, T_fixed)
loss_bc = torch.mean(residual_bc**2)
```

### 机械边界条件

#### 1. 固定位移（固定边界）

```python
# 例如: 在 x=0 边界上固定
residual_u, residual_v, residual_w = physics.mechanical_boundary_fixed(u, v, w)
loss_bc = (torch.mean(residual_u**2) + 
           torch.mean(residual_v**2) + 
           torch.mean(residual_w**2))
```

#### 2. 自由边界（应力自由）

```python
# 首先计算应力
strain = physics.compute_strain(u, v, w, x, y, z)
stress = physics.hookes_law_thermoelastic(strain, T, x, y, z)

# 然后应用自由边界条件
n_x, n_y, n_z = 1.0, 0.0, 0.0  # 法向量
traction_x, traction_y, traction_z = physics.mechanical_boundary_free(
    stress, n_x, n_y, n_z
)

loss_bc = (torch.mean(traction_x**2) + 
           torch.mean(traction_y**2) + 
           torch.mean(traction_z**2))
```

### 电学边界条件

#### 1. 施加电位

```python
# 例如: 正极 φ = 4.2V
phi_applied = 4.2
residual_bc = physics.electrical_boundary_applied_potential(phi_e, phi_applied)
loss_bc = torch.mean(residual_bc**2)
```

#### 2. 施加电流

```python
# 例如: 施加 1A 电流，边界面积 0.01 m²
I_applied = 1.0    # A
A = 0.01           # m²
n_x, n_y, n_z = 1.0, 0.0, 0.0

residual_bc = physics.electrical_boundary_applied_current(
    phi_e, I_applied, A, n_x, n_y, n_z, x, y, z
)
loss_bc = torch.mean(residual_bc**2)
```

---

## 参数配置

### 典型锂离子电池材料参数

#### 石墨负极 (Graphite Anode)

```python
params_anode = {
    'sigma_ionic': 0.1,        # 电解液中 (S/m)
    'sigma_electronic': 1e5,   # 石墨 (S/m)
    'k_thermal': 1.7,          # (W/(m·K))
    'rho': 2260.0,             # (kg/m³)
    'cp': 1437.4,              # (J/(kg·K))
    'E': 15e9,                 # 15 GPa
    'nu': 0.3,
    'alpha': 3e-6,             # (1/K)
    'i0': 0.5,                 # (A/m²)
}
```

#### 三元材料正极 (NMC Cathode)

```python
params_cathode = {
    'sigma_ionic': 0.1,
    'sigma_electronic': 1e3,   # NMC (S/m)
    'k_thermal': 1.58,         # (W/(m·K))
    'rho': 4750.0,             # (kg/m³)
    'cp': 1269.0,              # (J/(kg·K))
    'E': 10e9,                 # 10 GPa
    'nu': 0.3,
    'alpha': 1.5e-5,           # (1/K)
    'i0': 0.35,                # (A/m²)
}
```

#### 隔膜 (Separator)

```python
params_separator = {
    'sigma_ionic': 0.5,
    'sigma_electronic': 1e-10, # 绝缘体
    'k_thermal': 0.344,        # (W/(m·K))
    'rho': 1009.0,             # (kg/m³)
    'cp': 1978.0,              # (J/(kg·K))
    'E': 0.5e9,                # 0.5 GPa
    'nu': 0.3,
    'alpha': 1e-4,
}
```

---

## 训练流程示例

### 完整的训练脚本示例

```python
import torch
import torch.nn as nn
from pinn_physics_equations import BatteryPhysicsEquations, create_default_parameters
import matplotlib.pyplot as plt

# ==================== 1. 参数设置 ====================
# 几何参数
L_x, L_y, L_z = 0.1, 0.1, 0.01  # 电池尺寸 (m)
T_max = 3600.0                   # 最大时间 (s) - 1小时

# 物理参数
params = create_default_parameters()
physics = BatteryPhysicsEquations(params)

# 训练参数
n_collocation = 5000
n_boundary = 500
n_initial = 500
n_epochs = 5000
learning_rate = 1e-3

# ==================== 2. 生成训练点 ====================
# 域内配点
x_col = torch.rand(n_collocation, 1, requires_grad=True) * L_x
y_col = torch.rand(n_collocation, 1, requires_grad=True) * L_y
z_col = torch.rand(n_collocation, 1, requires_grad=True) * L_z
t_col = torch.rand(n_collocation, 1, requires_grad=True) * T_max

# 边界点 (x=0 边界为例)
x_bc = torch.zeros(n_boundary, 1, requires_grad=True)
y_bc = torch.rand(n_boundary, 1) * L_y
z_bc = torch.rand(n_boundary, 1) * L_z
t_bc = torch.rand(n_boundary, 1) * T_max

# 初始条件点 (t=0)
x_ic = torch.rand(n_initial, 1) * L_x
y_ic = torch.rand(n_initial, 1) * L_y
z_ic = torch.rand(n_initial, 1) * L_z
t_ic = torch.zeros(n_initial, 1, requires_grad=True)

# ==================== 3. 定义神经网络 ====================
class BatteryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 6)
        )
    
    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        outputs = self.net(inputs)
        return outputs

model = BatteryPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ==================== 4. 训练循环 ====================
history = {
    'total_loss': [],
    'physics_loss': [],
    'bc_loss': [],
    'ic_loss': []
}

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # 4.1 物理损失（PDE残差）
    outputs_col = model(x_col, y_col, z_col, t_col)
    predictions_col = {
        'T': outputs_col[:, 0:1],
        'phi_i': outputs_col[:, 1:2],
        'phi_e': outputs_col[:, 2:3],
        'u': outputs_col[:, 3:4],
        'v': outputs_col[:, 4:5],
        'w': outputs_col[:, 5:6]
    }
    coords_col = {'x': x_col, 'y': y_col, 'z': z_col, 't': t_col}
    
    physics_losses = physics.compute_total_physics_loss(predictions_col, coords_col)
    loss_physics = physics_losses['total']
    
    # 4.2 边界条件损失（例如：x=0绝热边界）
    outputs_bc = model(x_bc, y_bc, z_bc, t_bc)
    T_bc = outputs_bc[:, 0:1]
    
    # 绝热边界条件
    x_bc.requires_grad_(True)
    residual_bc = physics.thermal_boundary_insulated(T_bc, 1.0, 0.0, 0.0, x_bc, y_bc, z_bc)
    loss_bc = torch.mean(residual_bc**2)
    
    # 4.3 初始条件损失
    outputs_ic = model(x_ic, y_ic, z_ic, t_ic)
    T_ic_pred = outputs_ic[:, 0:1]
    T_ic_true = torch.ones_like(T_ic_pred) * 298.15  # 初始温度 298.15K
    loss_ic = torch.mean((T_ic_pred - T_ic_true)**2)
    
    # 4.4 总损失
    lambda_physics = 1.0
    lambda_bc = 1.0
    lambda_ic = 1.0
    
    total_loss = (lambda_physics * loss_physics + 
                  lambda_bc * loss_bc + 
                  lambda_ic * loss_ic)
    
    # 4.5 反向传播
    total_loss.backward()
    optimizer.step()
    
    # 4.6 记录历史
    history['total_loss'].append(total_loss.item())
    history['physics_loss'].append(loss_physics.item())
    history['bc_loss'].append(loss_bc.item())
    history['ic_loss'].append(loss_ic.item())
    
    # 4.7 打印进度
    if epoch % 100 == 0:
        print(f"\nEpoch {epoch}/{n_epochs}")
        print(f"  Total Loss:   {total_loss.item():.6e}")
        print(f"  Physics Loss: {loss_physics.item():.6e}")
        print(f"  BC Loss:      {loss_bc.item():.6e}")
        print(f"  IC Loss:      {loss_ic.item():.6e}")

# ==================== 5. 可视化训练历史 ====================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.semilogy(history['total_loss'])
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.semilogy(history['physics_loss'])
plt.xlabel('Epoch')
plt.ylabel('Physics Loss')
plt.title('Physics Loss (PDE Residual)')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.semilogy(history['bc_loss'], label='BC Loss')
plt.semilogy(history['ic_loss'], label='IC Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Boundary & Initial Condition Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("\n训练完成！损失曲线已保存到 'training_history.png'")

# ==================== 6. 保存模型 ====================
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history,
    'params': params
}, 'battery_pinn_model.pth')
print("模型已保存到 'battery_pinn_model.pth'")
```

---

## 总结

本模块提供了完整的物理方程定义，用于构建电池系统的物理信息神经网络（PINN）。主要特点：

✅ **完整的物理方程**: 包含电荷守恒、能量守恒、动量守恒

✅ **本构关系**: 胡克定律、傅里叶定律、Butler-Volmer方程

✅ **边界条件**: 热边界、机械边界、电学边界

✅ **自动微分**: 使用PyTorch自动计算PDE残差

✅ **模块化设计**: 易于扩展和自定义

✅ **详细文档**: 包含使用示例和参数说明

---

## 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Cai, S., Mao, Z., Wang, Z., Yin, M., & Karniadakis, G. E. (2022). Physics-informed neural networks (PINNs) for fluid mechanics: A review. *Acta Mechanica Sinica*, 37(12), 1727-1738.

3. Newman, J., & Thomas-Alyea, K. E. (2012). *Electrochemical systems* (3rd ed.). John Wiley & Sons.

4. Timoshenko, S. P., & Goodier, J. N. (1970). *Theory of elasticity* (3rd ed.). McGraw-Hill.

---

## 许可证

本代码仅供学习和研究使用。

---

**联系方式**: 如有问题，请提交Issue或Pull Request。
