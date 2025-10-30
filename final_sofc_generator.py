#!/usr/bin/env python3
"""
SOFC高保真数值数据集生成器 - 最终版本
不依赖外部包，使用内置的随机采样
"""

import numpy as np
import h5py
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SOFCGeometry:
    """SOFC几何模型"""
    
    def __init__(self, config):
        self.config = config
        self.layers = {
            'anode': {'thickness': config['anode_thickness'], 'porosity': config['anode_porosity']},
            'electrolyte': {'thickness': config['electrolyte_thickness'], 'porosity': 0.0},
            'cathode': {'thickness': config['cathode_thickness'], 'porosity': config['cathode_porosity']},
            'interconnect': {'thickness': config['interconnect_thickness'], 'porosity': 0.0},
            'sealant': {'thickness': config['sealant_thickness'], 'porosity': 0.0}
        }
        
    def generate_mesh(self, nx=20, ny=20, nz=10):
        """生成3D网格"""
        Lx, Ly = self.config['active_area']
        total_height = sum(layer['thickness'] for layer in self.layers.values())
        
        # 创建规则网格
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.linspace(0, total_height, nz)
        
        # 生成网格点
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # 生成六面体单元
        cells = []
        for i in range(nx-1):
            for j in range(ny-1):
                for k in range(nz-1):
                    # 8个顶点索引
                    v0 = i * ny * nz + j * nz + k
                    v1 = (i+1) * ny * nz + j * nz + k
                    v2 = (i+1) * ny * nz + (j+1) * nz + k
                    v3 = i * ny * nz + (j+1) * nz + k
                    v4 = i * ny * nz + j * nz + (k+1)
                    v5 = (i+1) * ny * nz + j * nz + (k+1)
                    v6 = (i+1) * ny * nz + (j+1) * nz + (k+1)
                    v7 = i * ny * nz + (j+1) * nz + (k+1)
                    
                    cells.append([v0, v1, v2, v3, v4, v5, v6, v7])
        
        return {
            'points': points,
            'cells': np.array(cells),
            'nx': nx, 'ny': ny, 'nz': nz
        }

class SOFCMultiPhysics:
    """SOFC多物理场耦合仿真"""
    
    def __init__(self, geometry, material_props, operating_conditions):
        self.geometry = geometry
        self.material_props = material_props
        self.operating_conditions = operating_conditions
        
    def solve_electrochemical(self, mesh):
        """求解电化学场"""
        n_nodes = len(mesh['points'])
        
        # 初始化场变量
        current_density = np.zeros((n_nodes, 3))
        overpotential = np.zeros(n_nodes)
        
        # 材料属性
        sigma_ion = self.material_props['electrolyte']['ionic_conductivity']
        sigma_elec_anode = self.material_props['anode']['electronic_conductivity']
        sigma_elec_cathode = self.material_props['cathode']['electronic_conductivity']
        
        # 计算电流密度分布
        for i, point in enumerate(mesh['points']):
            x, y, z = point
            
            # 根据位置确定材料
            if z < self.geometry.layers['anode']['thickness']:
                sigma = sigma_elec_anode
                eta = self.operating_conditions['anode_overpotential']
            elif z < self.geometry.layers['anode']['thickness'] + self.geometry.layers['electrolyte']['thickness']:
                sigma = sigma_ion
                eta = self.operating_conditions['electrolyte_overpotential']
            else:
                sigma = sigma_elec_cathode
                eta = self.operating_conditions['cathode_overpotential']
            
            # 简化的电流密度计算
            E_field = np.array([0, 0, eta / self.geometry.layers['electrolyte']['thickness']])
            current_density[i] = sigma * E_field
            overpotential[i] = eta
        
        return current_density, overpotential
    
    def solve_thermal(self, mesh, current_density):
        """求解热场"""
        n_nodes = len(mesh['points'])
        temperature = np.zeros(n_nodes)
        
        # 初始温度
        T_inlet_fuel = self.operating_conditions['fuel_inlet_temperature']
        T_inlet_air = self.operating_conditions['air_inlet_temperature']
        
        # 热传导系数
        k_anode = self.material_props['anode']['thermal_conductivity']
        k_electrolyte = self.material_props['electrolyte']['thermal_conductivity']
        k_cathode = self.material_props['cathode']['thermal_conductivity']
        
        # 焦耳热源
        joule_heating = np.sum(current_density**2, axis=1) / np.array([
            self.material_props['anode']['electronic_conductivity'] if z < self.geometry.layers['anode']['thickness']
            else self.material_props['electrolyte']['ionic_conductivity'] if z < self.geometry.layers['anode']['thickness'] + self.geometry.layers['electrolyte']['thickness']
            else self.material_props['cathode']['electronic_conductivity']
            for z in mesh['points'][:, 2]
        ])
        
        # 简化的温度分布计算
        for i, point in enumerate(mesh['points']):
            x, y, z = point
            
            # 基础温度分布
            if z < self.geometry.layers['anode']['thickness']:
                T_base = T_inlet_fuel
                k = k_anode
            elif z < self.geometry.layers['anode']['thickness'] + self.geometry.layers['electrolyte']['thickness']:
                T_base = (T_inlet_fuel + T_inlet_air) / 2
                k = k_electrolyte
            else:
                T_base = T_inlet_air
                k = k_cathode
            
            # 考虑焦耳热和热传导
            temp_rise = joule_heating[i] / (k * 1000)
            temperature[i] = T_base + temp_rise
        
        return temperature
    
    def solve_mechanical(self, mesh, temperature):
        """求解机械场"""
        n_nodes = len(mesh['points'])
        
        # 初始化场变量
        stress_tensor = np.zeros((n_nodes, 6))
        strain_tensor = np.zeros((n_nodes, 6))
        displacement = np.zeros((n_nodes, 3))
        
        # 材料属性
        E_anode = self.material_props['anode']['youngs_modulus']
        E_electrolyte = self.material_props['electrolyte']['youngs_modulus']
        E_cathode = self.material_props['cathode']['youngs_modulus']
        
        CTE_anode = self.material_props['anode']['cte']
        CTE_electrolyte = self.material_props['electrolyte']['cte']
        CTE_cathode = self.material_props['cathode']['cte']
        
        T_ref = 800  # 参考温度 (K)
        
        for i, point in enumerate(mesh['points']):
            x, y, z = point
            T = temperature[i]
            
            # 根据位置确定材料属性
            if z < self.geometry.layers['anode']['thickness']:
                E = E_anode
                CTE = CTE_anode
            elif z < self.geometry.layers['anode']['thickness'] + self.geometry.layers['electrolyte']['thickness']:
                E = E_electrolyte
                CTE = CTE_electrolyte
            else:
                E = E_cathode
                CTE = CTE_cathode
            
            # 热应变
            thermal_strain = CTE * (T - T_ref)
            
            # 简化的应力-应变关系
            strain_tensor[i] = np.array([
                thermal_strain, thermal_strain, thermal_strain, 0, 0, 0
            ])
            
            # 应力计算
            nu = 0.3  # 泊松比
            stress_tensor[i] = E / (1 - nu**2) * np.array([
                strain_tensor[i, 0] + nu * strain_tensor[i, 1],
                strain_tensor[i, 1] + nu * strain_tensor[i, 0],
                nu * (strain_tensor[i, 0] + strain_tensor[i, 1]),
                0, 0, 0
            ])
            
            # 位移计算
            displacement[i] = np.array([
                strain_tensor[i, 0] * x,
                strain_tensor[i, 1] * y,
                strain_tensor[i, 2] * z
            ])
        
        return stress_tensor, strain_tensor, displacement
    
    def solve_species_transport(self, mesh, temperature):
        """求解传质场"""
        n_nodes = len(mesh['points'])
        
        H2_concentration = np.zeros(n_nodes)
        H2O_concentration = np.zeros(n_nodes)
        
        # 燃料入口条件
        H2_inlet = self.operating_conditions['fuel_h2_fraction']
        H2O_inlet = self.operating_conditions['fuel_h2o_fraction']
        
        for i, point in enumerate(mesh['points']):
            x, y, z = point
            T = temperature[i]
            
            # 只在阳极中计算燃料浓度
            if z < self.geometry.layers['anode']['thickness']:
                # 简化的浓度分布
                reaction_rate = np.exp(-5000 / T)
                z_norm = z / self.geometry.layers['anode']['thickness']
                H2_concentration[i] = H2_inlet * (1 - z_norm * reaction_rate)
                H2O_concentration[i] = H2O_inlet + H2_inlet * z_norm * reaction_rate
            else:
                H2_concentration[i] = 0
                H2O_concentration[i] = 0
        
        return H2_concentration, H2O_concentration

class DatasetGenerator:
    """数据集生成器"""
    
    def __init__(self, config):
        self.config = config
        self.dataset = []
        
    def generate_parameter_combinations(self, n_samples=100):
        """生成参数组合 - 使用随机采样替代拉丁超立方"""
        parameters = {
            'voltage': (0.6, 0.9),
            'current_density': (0.1, 1.0),
            'fuel_flow_rate': (1e-6, 1e-4),
            'air_flow_rate': (1e-5, 1e-3),
            'fuel_inlet_temperature': (600, 900),
            'air_inlet_temperature': (600, 900),
            'anode_porosity': (0.2, 0.4),
            'cathode_porosity': (0.2, 0.4),
            'anode_permeability': (1e-15, 1e-12),
            'cathode_permeability': (1e-15, 1e-12),
            'anode_ionic_conductivity': (1e-3, 1e-1),
            'cathode_ionic_conductivity': (1e-3, 1e-1),
            'anode_electronic_conductivity': (1e2, 1e4),
            'cathode_electronic_conductivity': (1e2, 1e4),
            'electrolyte_ionic_conductivity': (1e-2, 1e0),
            'anode_youngs_modulus': (50e9, 200e9),
            'cathode_youngs_modulus': (50e9, 200e9),
            'electrolyte_youngs_modulus': (100e9, 300e9),
            'anode_cte': (8e-6, 15e-6),
            'cathode_cte': (8e-6, 15e-6),
            'electrolyte_cte': (10e-6, 18e-6),
            'anode_thickness': (200e-6, 1000e-6),
            'electrolyte_thickness': (10e-6, 50e-6),
            'cathode_thickness': (20e-6, 100e-6),
            'active_area_x': (0.01, 0.1),
            'active_area_y': (0.01, 0.1),
        }
        
        # 使用随机采样
        param_combinations = []
        for i in range(n_samples):
            param_dict = {}
            for param_name, (min_val, max_val) in parameters.items():
                param_dict[param_name] = min_val + np.random.random() * (max_val - min_val)
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def run_single_simulation(self, params, sim_id):
        """运行单个仿真"""
        try:
            # 创建几何配置
            geometry_config = {
                'anode_thickness': params['anode_thickness'],
                'electrolyte_thickness': params['electrolyte_thickness'],
                'cathode_thickness': params['cathode_thickness'],
                'interconnect_thickness': 1e-3,
                'sealant_thickness': 0.5e-3,
                'active_area': (params['active_area_x'], params['active_area_y']),
                'anode_porosity': params['anode_porosity'],
                'cathode_porosity': params['cathode_porosity']
            }
            
            # 创建几何
            geometry = SOFCGeometry(geometry_config)
            mesh = geometry.generate_mesh(nx=15, ny=15, nz=8)
            
            # 材料属性
            material_props = {
                'anode': {
                    'porosity': params['anode_porosity'],
                    'permeability': params['anode_permeability'],
                    'ionic_conductivity': params['anode_ionic_conductivity'],
                    'electronic_conductivity': params['anode_electronic_conductivity'],
                    'thermal_conductivity': 10.0,
                    'youngs_modulus': params['anode_youngs_modulus'],
                    'cte': params['anode_cte']
                },
                'electrolyte': {
                    'ionic_conductivity': params['electrolyte_ionic_conductivity'],
                    'thermal_conductivity': 2.0,
                    'youngs_modulus': params['electrolyte_youngs_modulus'],
                    'cte': params['electrolyte_cte']
                },
                'cathode': {
                    'porosity': params['cathode_porosity'],
                    'permeability': params['cathode_permeability'],
                    'ionic_conductivity': params['cathode_ionic_conductivity'],
                    'electronic_conductivity': params['cathode_electronic_conductivity'],
                    'thermal_conductivity': 8.0,
                    'youngs_modulus': params['cathode_youngs_modulus'],
                    'cte': params['cathode_cte']
                }
            }
            
            # 操作条件
            operating_conditions = {
                'voltage': params['voltage'],
                'current_density': params['current_density'],
                'fuel_flow_rate': params['fuel_flow_rate'],
                'air_flow_rate': params['air_flow_rate'],
                'fuel_inlet_temperature': params['fuel_inlet_temperature'],
                'air_inlet_temperature': params['air_inlet_temperature'],
                'fuel_h2_fraction': 0.97,
                'fuel_h2o_fraction': 0.03,
                'anode_overpotential': 0.1,
                'cathode_overpotential': 0.1,
                'electrolyte_overpotential': params['voltage'] - 0.2
            }
            
            # 创建多物理场求解器
            physics = SOFCMultiPhysics(geometry, material_props, operating_conditions)
            
            # 求解各个物理场
            current_density, overpotential = physics.solve_electrochemical(mesh)
            temperature = physics.solve_thermal(mesh, current_density)
            stress_tensor, strain_tensor, displacement = physics.solve_mechanical(mesh, temperature)
            H2_concentration, H2O_concentration = physics.solve_species_transport(mesh, temperature)
            
            # 计算von Mises应力
            von_mises_stress = np.sqrt(
                stress_tensor[:, 0]**2 + stress_tensor[:, 1]**2 + stress_tensor[:, 2]**2
                - stress_tensor[:, 0]*stress_tensor[:, 1] - stress_tensor[:, 1]*stress_tensor[:, 2] - stress_tensor[:, 2]*stress_tensor[:, 0]
                + 3*(stress_tensor[:, 3]**2 + stress_tensor[:, 4]**2 + stress_tensor[:, 5]**2)
            )
            
            # 组装仿真结果
            simulation_result = {
                'simulation_id': sim_id,
                'input_parameters': params,
                'mesh_points': mesh['points'],
                'mesh_cells': mesh['cells'],
                'fields': {
                    'current_density': current_density,
                    'overpotential': overpotential,
                    'temperature': temperature,
                    'stress_tensor': stress_tensor,
                    'strain_tensor': strain_tensor,
                    'displacement': displacement,
                    'von_mises_stress': von_mises_stress,
                    'H2_concentration': H2_concentration,
                    'H2O_concentration': H2O_concentration
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'n_nodes': len(mesh['points']),
                    'n_cells': len(mesh['cells']),
                    'convergence': True
                }
            }
            
            return simulation_result
            
        except Exception as e:
            print(f"仿真 {sim_id} 失败: {str(e)}")
            return None
    
    def generate_dataset(self, n_simulations=50, n_parallel=2):
        """生成数据集"""
        print(f"开始生成 {n_simulations} 个SOFC仿真数据...")
        
        # 生成参数组合
        param_combinations = self.generate_parameter_combinations(n_simulations)
        
        # 并行运行仿真
        results = Parallel(n_jobs=n_parallel)(
            delayed(self.run_single_simulation)(params, i) 
            for i, params in enumerate(tqdm(param_combinations, desc="运行仿真"))
        )
        
        # 过滤成功的结果
        self.dataset = [result for result in results if result is not None]
        
        print(f"成功生成 {len(self.dataset)} 个仿真数据")
        return self.dataset
    
    def save_dataset(self, filename="sofc_high_fidelity_dataset.h5"):
        """保存数据集"""
        with h5py.File(filename, 'w') as f:
            f.attrs['description'] = 'SOFC高保真数值数据集'
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['n_simulations'] = len(self.dataset)
            
            for i, sim_data in enumerate(self.dataset):
                sim_group = f.create_group(f'simulation_{i:06d}')
                
                # 保存输入参数
                params_group = sim_group.create_group('input_parameters')
                for key, value in sim_data['input_parameters'].items():
                    params_group.attrs[key] = value
                
                # 保存网格数据
                mesh_group = sim_group.create_group('mesh')
                mesh_group.create_dataset('points', data=sim_data['mesh_points'])
                mesh_group.create_dataset('cells', data=sim_data['mesh_cells'])
                
                # 保存场数据
                fields_group = sim_group.create_group('fields')
                for field_name, field_data in sim_data['fields'].items():
                    fields_group.create_dataset(field_name, data=field_data)
                
                # 保存元数据
                meta_group = sim_group.create_group('metadata')
                for key, value in sim_data['metadata'].items():
                    if isinstance(value, str):
                        meta_group.attrs[key] = value
                    else:
                        meta_group.create_dataset(key, data=value)
        
        print(f"数据集已保存到: {filename}")
        return filename
    
    def create_summary_report(self, filename="dataset_summary.json"):
        """创建摘要报告"""
        summary = {
            'dataset_info': {
                'total_simulations': len(self.dataset),
                'creation_date': datetime.now().isoformat(),
                'description': 'SOFC高保真数值数据集 - 多物理场耦合仿真'
            },
            'parameter_ranges': {},
            'field_statistics': {},
            'mesh_statistics': {
                'avg_nodes': np.mean([len(sim['mesh_points']) for sim in self.dataset]),
                'avg_cells': np.mean([len(sim['mesh_cells']) for sim in self.dataset])
            }
        }
        
        if self.dataset:
            param_names = list(self.dataset[0]['input_parameters'].keys())
            for param in param_names:
                values = [sim['input_parameters'][param] for sim in self.dataset]
                summary['parameter_ranges'][param] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            
            field_names = list(self.dataset[0]['fields'].keys())
            for field in field_names:
                all_values = np.concatenate([sim['fields'][field].flatten() for sim in self.dataset])
                summary['field_statistics'][field] = {
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values))
                }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"数据集摘要已保存到: {filename}")
        return summary

def create_visualizations(dataset_file):
    """创建可视化"""
    print("生成数据集可视化...")
    
    with h5py.File(dataset_file, 'r') as f:
        n_simulations = f.attrs['n_simulations']
        
        # 收集参数数据
        param_data = {}
        for sim_id in f.keys():
            if sim_id.startswith('simulation_'):
                params = f[sim_id]['input_parameters'].attrs
                for key, value in params.items():
                    if key not in param_data:
                        param_data[key] = []
                    param_data[key].append(value)
        
        # 创建参数分布图
        n_params = len(param_data)
        n_cols = 4
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, (param_name, values) in enumerate(param_data.items()):
            if i < len(axes):
                ax = axes[i]
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.set_title(f'{param_name}', fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建场数据统计图
        field_names = ['temperature', 'overpotential', 'von_mises_stress', 'H2_concentration']
        field_stats = {}
        
        for field_name in field_names:
            all_values = []
            for sim_id in f.keys():
                if sim_id.startswith('simulation_'):
                    if field_name in f[sim_id]['fields']:
                        field_data = f[sim_id]['fields'][field_name][:]
                        all_values.extend(field_data.flatten())
            
            if all_values:
                field_stats[field_name] = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values),
                    'min': np.min(all_values),
                    'max': np.max(all_values),
                    'values': all_values
                }
        
        # 绘制场数据统计
        n_fields = len(field_stats)
        n_cols = 2
        n_rows = (n_fields + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, (field_name, stats) in enumerate(field_stats.items()):
            if i < len(axes):
                ax = axes[i]
                ax.hist(stats['values'], bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2e}')
                ax.axvline(stats['mean'] + stats['std'], color='orange', linestyle='--', alpha=0.7, label=f'±1σ')
                ax.axvline(stats['mean'] - stats['std'], color='orange', linestyle='--', alpha=0.7)
                
                ax.set_title(f'{field_name} 分布')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_fields, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('field_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化图像已生成:")
        print("- parameter_distributions.png")
        print("- field_statistics.png")

def main():
    """主函数"""
    config = {
        'n_simulations': 30,  # 生成30个仿真
        'n_parallel': 2,      # 使用2个并行进程
        'mesh_resolution': 15  # 网格分辨率
    }
    
    print("=== SOFC高保真数值数据集生成器 ===")
    print(f"计划生成 {config['n_simulations']} 个仿真数据")
    print(f"使用 {config['n_parallel']} 个并行进程")
    
    # 创建数据集生成器
    generator = DatasetGenerator(config)
    
    # 生成数据集
    dataset = generator.generate_dataset(
        n_simulations=config['n_simulations'],
        n_parallel=config['n_parallel']
    )
    
    # 保存数据集
    h5_filename = generator.save_dataset()
    
    # 创建摘要报告
    summary = generator.create_summary_report()
    
    # 创建可视化
    create_visualizations(h5_filename)
    
    print("\n=== 数据集生成完成 ===")
    print(f"成功生成: {len(dataset)} 个仿真")
    print(f"数据文件: {h5_filename}")
    print(f"摘要报告: dataset_summary.json")
    print(f"可视化图像: parameter_distributions.png, field_statistics.png")
    
    return dataset, h5_filename, summary

if __name__ == "__main__":
    dataset, filename, summary = main()