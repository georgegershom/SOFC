# PINN物理方程数据集
# Physics-Informed Neural Network (PINN) Physics Equations Dataset

## 📋 项目概述

本项目提供了用于物理信息神经网络（PINN）的完整物理方程定义，专门针对锂离子电池系统的多物理场建模。这不是传统意义上的"数据集"，而是PINN必须遵守的数学物理方程的计算机实现。

## 🎯 核心内容

### 1. 守恒定律

#### 电荷守恒 (Charge Conservation)
- **离子相**: ∇·(σ_i·∇φ_i) = -j_rxn
- **电子相**: ∇·(σ_e·∇φ_e) = j_rxn
- 用于描述锂离子和电子在电池中的传输

#### 能量守恒 (Energy Conservation)
- **热方程**: ρ·cp·∂T/∂t = ∇·(k·∇T) + Q_joule + Q_rxn
- 包含焦耳热和电化学反应热源项
- 用于预测电池的温度分布

#### 线性动量守恒 (Linear Momentum Conservation)
- **机械平衡**: ∇·σ = 0
- 用于计算电池的应力-应变分布
- 考虑热膨胀效应

### 2. 本构定律 (Constitutive Laws)

#### 热弹性胡克定律 (Hooke's Law with Thermal Effects)
```
σ = λ·tr(ε)·I + 2μ·ε - (3λ + 2μ)·α·ΔT·I
```
- 描述应力-应变关系
- 考虑温度变化引起的热应力

#### 傅里叶热传导定律 (Fourier's Law)
```
q = -k·∇T
```
- 描述热量传输
- k为材料的热导率

#### Butler-Volmer电化学动力学方程
```
j = i_0 · [exp(α_a·F·η/(R·T)) - exp(-α_c·F·η/(R·T))]
```
- 描述电极-电解液界面的电化学反应
- i_0为交换电流密度，η为过电位

### 3. 边界条件

#### 热边界条件
- ✅ 绝热边界 (Insulated): -k·∇T·n = 0
- ✅ 对流边界 (Convective): -k·∇T·n = h·(T - T_ambient)
- ✅ 固定温度 (Fixed Temperature): T = T_fixed

#### 机械边界条件
- ✅ 固定位移 (Fixed Displacement): u = v = w = 0
- ✅ 自由边界 (Traction-Free): σ·n = 0
- ✅ 对称边界 (Symmetry): u_n = 0, σ_t = 0

#### 电学边界条件
- ✅ 施加电位 (Applied Potential): φ = φ_applied
- ✅ 施加电流 (Applied Current): -σ·∇φ·n = j_applied

## 📁 文件结构

```
/workspace/
├── pinn_physics_equations.py          # 主要Python模块（807行）
├── physics_parameters_config.json     # 参数配置文件
├── PINN_Physics_Usage_Guide.md        # 详细使用指南（705行）
├── requirements.txt                   # Python依赖
├── validate_physics_module.py         # 验证脚本
└── README_PINN_Physics.md            # 本文件
```

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
```

需要的包：
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

### 步骤2: 导入模块

```python
from pinn_physics_equations import BatteryPhysicsEquations, create_default_parameters

# 创建物理参数
params = create_default_parameters()

# 初始化物理方程对象
physics = BatteryPhysicsEquations(params)
```

### 步骤3: 在PINN训练中使用

```python
# 假设你的神经网络输出6个物理量
predictions = {
    'T': T_pred,          # 温度 (K)
    'phi_i': phi_i_pred,  # 离子电位 (V)
    'phi_e': phi_e_pred,  # 电子电位 (V)
    'u': u_pred,          # X位移 (m)
    'v': v_pred,          # Y位移 (m)
    'w': w_pred           # Z位移 (m)
}

coords = {'x': x, 'y': y, 'z': z, 't': t}

# 计算物理损失（PDE残差）
physics_losses = physics.compute_total_physics_loss(predictions, coords)

# 训练时的总损失
total_loss = data_loss + physics_losses['total']
```

## 📊 包含的材料参数

### 已配置的材料类型

| 材料 | 用途 | 电导率 | 热导率 | 杨氏模量 |
|------|------|--------|--------|----------|
| 石墨 (Graphite) | 负极 | 10⁵ S/m | 1.7 W/(m·K) | 15 GPa |
| 三元材料 (NMC) | 正极 | 10³ S/m | 1.58 W/(m·K) | 10 GPa |
| 磷酸铁锂 (LFP) | 正极 | 10² S/m | 1.48 W/(m·K) | 8 GPa |
| 隔膜 (Separator) | 隔膜 | 0.5 S/m | 0.344 W/(m·K) | 0.5 GPa |
| 电解液 (Electrolyte) | 电解液 | 1.0 S/m | 0.6 W/(m·K) | - |

所有参数可在 `physics_parameters_config.json` 中查看和修改。

## 🔬 如何使用

### 单独计算某个PDE的残差

```python
# 计算离子电荷守恒残差
residual = physics.charge_conservation_ionic(phi_i, x, y, z, t)
loss_charge = torch.mean(residual**2)

# 计算能量守恒残差
residual = physics.energy_conservation(T, phi_i, phi_e, x, y, z, t)
loss_energy = torch.mean(residual**2)

# 计算动量守恒残差
residual_x = physics.momentum_conservation_x(u, v, w, T, x, y, z)
loss_momentum = torch.mean(residual_x**2)
```

### 应用边界条件

```python
# 绝热边界（例如在x=0处）
n_x, n_y, n_z = 1.0, 0.0, 0.0  # 法向量
residual_bc = physics.thermal_boundary_insulated(T, n_x, n_y, n_z, x, y, z)
loss_bc = torch.mean(residual_bc**2)

# 对流边界
T_ambient = 298.15  # K
h = 10.0            # W/(m²·K)
residual_bc = physics.thermal_boundary_convective(
    T, T_ambient, h, n_x, n_y, n_z, x, y, z
)
```

## 📖 详细文档

- **使用指南**: 查看 `PINN_Physics_Usage_Guide.md` 获取完整的使用说明
- **参数配置**: 查看 `physics_parameters_config.json` 了解所有可配置参数
- **代码注释**: `pinn_physics_equations.py` 包含详细的中英文注释

## ✅ 验证

运行验证脚本检查模块完整性：

```bash
python3 validate_physics_module.py
```

验证结果：
```
✓ 物理方程模块结构完整
✓ 所有关键组件已实现
• 19 个物理方程和边界条件函数
• 完整的中文和英文文档
• 详细的使用指南
• JSON配置文件用于参数管理
```

## 🎓 理论背景

### 什么是物理信息神经网络（PINN）？

PINN是一种结合了物理定律和神经网络的深度学习方法：

1. **传统神经网络**: 纯数据驱动，不考虑物理定律
2. **PINN**: 在损失函数中加入物理方程的残差

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}
$$

其中：
- $\mathcal{L}_{\text{data}}$: 数据拟合损失（如果有实验数据）
- $\mathcal{L}_{\text{physics}}$: 物理方程残差损失

### 为什么使用PINN？

✅ **数据效率高**: 在数据稀缺时仍能给出物理上合理的预测

✅ **物理一致性**: 预测结果自动满足物理定律

✅ **多物理场耦合**: 能够同时求解多个耦合的PDE

✅ **网格无关**: 不需要传统有限元方法的网格划分

### 本项目的物理方程系统

我们实现了电池系统的三大守恒定律：

```
1. 电荷守恒 → 电位分布 (φ_i, φ_e)
2. 能量守恒 → 温度分布 (T)
3. 动量守恒 → 应力-应变分布 (u, v, w, σ)
```

这些方程是耦合的：
- 电化学反应产生热量（影响温度）
- 温度变化导致热膨胀（产生应力）
- 应力可能影响离子传输（影响电化学性能）

## 📈 典型应用场景

### 1. 电池热管理设计

```python
# 预测电池在快充条件下的温度分布
params['i0'] = 10.0  # 高电流密度
T_pred = pinn_model.predict_temperature(x, y, z, t)
```

### 2. 应力分析

```python
# 计算充放电过程中的机械应力
stress = physics.hookes_law_thermoelastic(strain, T, x, y, z)
max_stress = torch.max(stress['sigma_xx'])
```

### 3. 寿命预测

```python
# 结合应力和温度预测电池寿命
degradation_rate = f(max_stress, max_temperature)
```

### 4. 优化设计

```python
# 优化电池的几何设计以降低最高温度
# 使用PINN作为代理模型进行快速迭代
```

## 🔧 自定义和扩展

### 修改材料参数

编辑 `physics_parameters_config.json`:

```json
{
  "custom_material": {
    "electrical": {
      "sigma_ionic": 0.5,
      "sigma_electronic": 1e4
    },
    "thermal": {
      "k_thermal": 2.0,
      "rho": 2500.0,
      "cp": 1100.0
    }
  }
}
```

### 添加新的物理方程

在 `BatteryPhysicsEquations` 类中添加新方法：

```python
def my_custom_equation(self, field, x, y, z, t):
    """
    自定义物理方程
    """
    # 计算梯度
    field_x = self._compute_gradient(field, x)
    
    # 定义方程残差
    residual = ...  # 你的方程
    
    return residual
```

### 添加新的边界条件

```python
def my_custom_boundary(self, field, params):
    """
    自定义边界条件
    """
    residual = ...  # 你的边界条件
    return residual
```

## 📚 参考文献

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** 
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* 
   Journal of Computational Physics, 378, 686-707.

2. **Newman, J., & Thomas-Alyea, K. E. (2012).** 
   *Electrochemical systems* (3rd ed.). 
   John Wiley & Sons.

3. **Cai, S., Mao, Z., Wang, Z., Yin, M., & Karniadakis, G. E. (2022).** 
   *Physics-informed neural networks (PINNs) for fluid mechanics: A review.* 
   Acta Mechanica Sinica, 37(12), 1727-1738.

4. **Timoshenko, S. P., & Goodier, J. N. (1970).** 
   *Theory of elasticity* (3rd ed.). 
   McGraw-Hill.

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题，请提交Issue。

---

## 🎉 总结

本项目提供了：

| 组件 | 描述 | 数量 |
|------|------|------|
| 守恒方程 | 电荷、能量、动量 | 3类 |
| 本构定律 | 胡克、傅里叶、Butler-Volmer | 3个 |
| 边界条件 | 热、机械、电学 | 9种 |
| 材料参数 | 电极、隔膜、电解液 | 5种 |
| 代码行数 | 主模块 | 807行 |
| 文档页数 | 使用指南 | 705行 |

**适用于**: 
- 🔋 电池系统建模
- 🧠 PINN研究
- 📊 多物理场仿真
- 🎓 学术研究
- 🏭 工业应用

**关键特点**:
- ✅ 完整的物理方程实现
- ✅ 详细的中英文文档
- ✅ 真实的材料参数
- ✅ 易于扩展和自定义
- ✅ 经过验证的代码结构

---

**开始使用**: 

```bash
pip install -r requirements.txt
python3 validate_physics_module.py
```

然后查看 `PINN_Physics_Usage_Guide.md` 开始你的PINN之旅！ 🚀
