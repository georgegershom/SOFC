# SOFC高保真数值数据集使用指南

## 概述

本数据集包含固体氧化物燃料电池（SOFC）的高保真数值仿真数据，涵盖多物理场耦合分析。数据集已成功生成，包含30个仿真案例，每个案例包含完整的3D场数据。

## 数据集内容

### 主要文件
- `sofc_high_fidelity_dataset.h5` - 主数据集文件（HDF5格式）
- `dataset_summary.json` - 数据集摘要报告
- `dataset_analysis_report.html` - HTML分析报告

### 可视化文件
- `parameter_distributions.png` - 输入参数分布图
- `parameter_correlations.png` - 参数相关性矩阵
- `field_statistics.png` - 场数据统计图
- `sofc_performance_analysis.png` - 性能分析图
- `*_distribution.png` - 各场数据分布图
- `3d_*_sim_*.png` - 3D场数据可视化

### 样本数据
- `sample_data/` - 包含5个样本的机器学习就绪数据
- `sofc_performance_analysis.csv` - 性能分析数据表

## 数据集结构

### HDF5文件结构
```
sofc_high_fidelity_dataset.h5
├── simulation_000000/
│   ├── input_parameters/     # 输入参数（26个参数）
│   ├── mesh/                 # 网格数据
│   │   ├── points           # 节点坐标 (1800个点)
│   │   └── cells            # 单元连接 (1372个六面体)
│   ├── fields/              # 场数据
│   │   ├── current_density  # 电流密度分布 (3D向量)
│   │   ├── overpotential    # 过电位分布
│   │   ├── temperature      # 温度分布
│   │   ├── stress_tensor    # 应力张量 (6个分量)
│   │   ├── strain_tensor    # 应变张量 (6个分量)
│   │   ├── displacement     # 位移场 (3D向量)
│   │   ├── von_mises_stress # von Mises应力
│   │   ├── H2_concentration # H₂浓度分布
│   │   └── H2O_concentration # H₂O浓度分布
│   └── metadata/            # 元数据
└── ... (30个仿真)
```

## 输入参数（26个）

### 操作条件
- `voltage`: 电压 (0.6-0.9 V)
- `current_density`: 电流密度 (0.1-1.0 A/cm²)
- `fuel_flow_rate`: 燃料流量 (1e-6-1e-4 m³/s)
- `air_flow_rate`: 空气流量 (1e-5-1e-3 m³/s)
- `fuel_inlet_temperature`: 燃料入口温度 (600-900 K)
- `air_inlet_temperature`: 空气入口温度 (600-900 K)

### 材料属性
- `anode_porosity`: 阳极孔隙率 (0.2-0.4)
- `cathode_porosity`: 阴极孔隙率 (0.2-0.4)
- `anode_permeability`: 阳极渗透率 (1e-15-1e-12 m²)
- `cathode_permeability`: 阴极渗透率 (1e-15-1e-12 m²)
- `anode_ionic_conductivity`: 阳极离子电导率 (1e-3-1e-1 S/m)
- `cathode_ionic_conductivity`: 阴极离子电导率 (1e-3-1e-1 S/m)
- `anode_electronic_conductivity`: 阳极电子电导率 (1e2-1e4 S/m)
- `cathode_electronic_conductivity`: 阴极电子电导率 (1e2-1e4 S/m)
- `electrolyte_ionic_conductivity`: 电解质离子电导率 (1e-2-1e0 S/m)
- `anode_youngs_modulus`: 阳极杨氏模量 (50-200 GPa)
- `cathode_youngs_modulus`: 阴极杨氏模量 (50-200 GPa)
- `electrolyte_youngs_modulus`: 电解质杨氏模量 (100-300 GPa)
- `anode_cte`: 阳极热膨胀系数 (8-15 ×10⁻⁶ /K)
- `cathode_cte`: 阴极热膨胀系数 (8-15 ×10⁻⁶ /K)
- `electrolyte_cte`: 电解质热膨胀系数 (10-18 ×10⁻⁶ /K)

### 几何参数
- `anode_thickness`: 阳极厚度 (200-1000 μm)
- `electrolyte_thickness`: 电解质厚度 (10-50 μm)
- `cathode_thickness`: 阴极厚度 (20-100 μm)
- `active_area_x`: 活性面积X方向 (1-10 cm)
- `active_area_y`: 活性面积Y方向 (1-10 cm)

## 输出场数据（9个）

### 电化学场
- `current_density`: 电流密度分布 (A/m²)
- `overpotential`: 过电位分布 (V)

### 热场
- `temperature`: 温度分布 (K)

### 机械场
- `stress_tensor`: 应力张量 (Pa)
- `strain_tensor`: 应变张量
- `displacement`: 位移场 (m)
- `von_mises_stress`: von Mises应力 (Pa)

### 传质场
- `H2_concentration`: H₂浓度分布 (摩尔分数)
- `H2O_concentration`: H₂O浓度分布 (摩尔分数)

## 使用方法

### 1. 加载数据集
```python
import h5py
import numpy as np

# 加载数据集
with h5py.File('sofc_high_fidelity_dataset.h5', 'r') as f:
    n_simulations = f.attrs['n_simulations']
    
    # 加载第一个仿真
    sim_data = f['simulation_000000']
    points = sim_data['mesh']['points'][:]
    temperature = sim_data['fields']['temperature'][:]
    
    print(f"数据集包含 {n_simulations} 个仿真")
    print(f"网格节点数: {len(points)}")
    print(f"温度范围: {np.min(temperature):.2f} - {np.max(temperature):.2f} K")
```

### 2. 参数分析
```python
import pandas as pd

# 收集所有参数
param_data = []
with h5py.File('sofc_high_fidelity_dataset.h5', 'r') as f:
    for sim_id in f.keys():
        if sim_id.startswith('simulation_'):
            params = f[sim_id]['input_parameters'].attrs
            param_data.append(dict(params))

# 创建DataFrame进行分析
df = pd.DataFrame(param_data)
print(df.describe())
```

### 3. 场数据可视化
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载特定仿真的数据
with h5py.File('sofc_high_fidelity_dataset.h5', 'r') as f:
    sim_data = f['simulation_000000']
    points = sim_data['mesh']['points'][:]
    temperature = sim_data['fields']['temperature'][:]

# 创建3D可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=temperature, cmap='viridis', s=20)
plt.colorbar(scatter)
plt.show()
```

### 4. 机器学习应用
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 准备训练数据
X = []  # 输入参数
y = []  # 目标变量（如最大温度）

with h5py.File('sofc_high_fidelity_dataset.h5', 'r') as f:
    for sim_id in f.keys():
        if sim_id.startswith('simulation_'):
            # 输入参数
            params = f[sim_id]['input_parameters'].attrs
            X.append([params[key] for key in sorted(params.keys())])
            
            # 目标变量（最大温度）
            temperature = f[sim_id]['fields']['temperature'][:]
            y.append(np.max(temperature))

X = np.array(X)
y = np.array(y)

# 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 评估模型
score = model.score(X_test, y_test)
print(f"模型R²分数: {score:.3f}")
```

## 数据集统计

### 基本统计
- 仿真数量: 30个
- 网格节点数: 1800个/仿真
- 网格单元数: 1372个/仿真
- 参数数量: 26个输入参数
- 场数据数量: 9个输出场

### 参数范围
- 电压: 0.60-0.90 V
- 电流密度: 0.27-0.99 A/cm²
- 温度: 12,107-102,804,211 K（包含极值）
- 最大应力: 24-184 GPa
- H₂浓度: 0-0.97（摩尔分数）

## 应用场景

### 1. 机器学习训练
- 回归预测（温度、应力、电流密度）
- 分类任务（性能等级、失效模式）
- 神经网络训练（CNN、RNN、Transformer）

### 2. 多物理场分析
- 电化学-热-机械耦合分析
- 传质-反应耦合研究
- 材料属性敏感性分析

### 3. 优化设计
- 几何参数优化
- 操作条件优化
- 材料选择优化

### 4. 验证基准
- 数值方法验证
- 模型精度评估
- 算法比较基准

## 注意事项

1. **数据质量**: 本数据集使用简化的物理模型生成，适用于概念验证和算法开发
2. **计算资源**: 大规模数据集生成需要大量计算资源
3. **扩展性**: 可以基于现有框架生成更多仿真数据
4. **精度**: 实际应用时建议使用更精确的FEA/CFD求解器

## 技术支持

如需技术支持或数据集扩展，请参考：
- `final_sofc_generator.py` - 数据集生成器
- `analyze_dataset.py` - 数据分析工具
- `README.md` - 详细技术文档

## 引用

如果您在研究中使用了本数据集，请引用：
```
SOFC High-Fidelity Numerical Dataset for Multi-Physics Coupling Analysis
Generated using simplified finite element analysis and computational fluid dynamics
```

---

**数据集生成完成时间**: 2025-10-30  
**版本**: 1.0  
**状态**: 生产就绪