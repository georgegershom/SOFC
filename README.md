# SOFC高保真数值数据集生成器

这是一个用于生成固体氧化物燃料电池（SOFC）高保真数值数据集的Python工具，支持多物理场耦合仿真，包括电化学、热传导、机械应力和传质分析。

## 功能特性

- **多物理场耦合仿真**：电化学、热传导、机械应力、传质
- **参数化建模**：支持操作条件、材料属性、几何参数的变化
- **拉丁超立方采样**：高效的参数空间采样
- **并行计算**：支持多进程并行仿真
- **多种输出格式**：HDF5、VTK、NetCDF
- **可视化工具**：3D场数据可视化和统计分析

## 数据集内容

### 输入参数（特征）
- **操作条件**：电压/电流密度、空气/燃料流量、入口温度
- **材料属性**：孔隙率、渗透率、离子/电子电导率、杨氏模量、热膨胀系数
- **几何参数**：各层厚度、活性面积、流道设计

### 输出场（标签）
- **电化学场**：电流密度分布、过电位分布
- **热场**：温度分布 T(x,y,z)
- **应力/应变场**：应力张量、应变张量、位移场
- **物种场**：燃料物种浓度分布（H₂, H₂O）

## 安装和使用

### 1. 安装依赖

```bash
python install_dependencies.py
```

### 2. 快速测试

```bash
python quick_test.py
```

### 3. 生成完整数据集

```bash
python sofc_dataset_generator.py
```

### 4. 可视化数据集

```bash
python sofc_data_visualizer.py
```

## 文件结构

```
/workspace/
├── sofc_dataset_generator.py    # 主数据集生成器
├── sofc_data_visualizer.py     # 数据可视化工具
├── quick_test.py               # 快速测试脚本
├── install_dependencies.py     # 依赖安装脚本
├── requirements.txt            # Python依赖列表
└── README.md                   # 说明文档
```

## 配置参数

在 `sofc_dataset_generator.py` 的 `main()` 函数中可以修改以下参数：

```python
config = {
    'n_simulations': 100,  # 仿真数量
    'n_parallel': 4,       # 并行进程数
    'mesh_resolution': 50  # 网格分辨率
}
```

## 输出文件

- `sofc_high_fidelity_dataset.h5`：主数据集文件（HDF5格式）
- `dataset_summary.json`：数据集摘要报告
- `parameter_distributions.png`：参数分布图
- `correlation_matrix.png`：参数相关性矩阵
- `field_statistics.png`：场数据统计图
- `field_visualizations/`：3D场数据可视化图像
- `vtk_exports/`：VTK格式文件（用于ParaView）
- `visualization_report.html`：HTML可视化报告

## 数据集格式

数据集采用HDF5格式存储，结构如下：

```
sofc_high_fidelity_dataset.h5
├── simulation_000000/
│   ├── input_parameters/     # 输入参数
│   ├── mesh/                 # 网格数据
│   │   ├── points           # 节点坐标
│   │   └── cells            # 单元连接
│   ├── fields/              # 场数据
│   │   ├── current_density  # 电流密度
│   │   ├── overpotential    # 过电位
│   │   ├── temperature      # 温度
│   │   ├── stress_tensor    # 应力张量
│   │   ├── strain_tensor    # 应变张量
│   │   ├── displacement     # 位移
│   │   ├── von_mises_stress # von Mises应力
│   │   ├── H2_concentration # H₂浓度
│   │   └── H2O_concentration # H₂O浓度
│   └── metadata/            # 元数据
└── ...
```

## 使用示例

### 加载数据集

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

### 参数分析

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

## 注意事项

1. **计算资源**：大规模数据集生成需要大量计算资源和时间
2. **内存使用**：高分辨率网格会消耗大量内存
3. **并行处理**：建议根据CPU核心数调整并行进程数
4. **存储空间**：完整数据集可能需要几GB到几TB的存储空间

## 扩展功能

- 支持更多SOFC几何配置
- 集成高级FEA/CFD求解器（FEniCS、OpenFOAM）
- 添加更多物理场（多相流、化学反应）
- 支持GPU加速计算
- 集成机器学习预处理工具

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。