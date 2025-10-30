#!/usr/bin/env python3
"""
SOFC高保真数值数据集生成器
生成用于预训练和验证的多物理场耦合仿真数据
"""

import numpy as np
import h5py
import netCDF4 as nc
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyDOE2
import meshio
import gmsh
import pyvista as pv
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SOFCGeometry:
    """SOFC几何模型定义"""
    
    def __init__(self, config):
        self.config = config
        self.layers = {
            'anode': {'thickness': config['anode_thickness'], 'porosity': config['anode_porosity']},
            'electrolyte': {'thickness': config['electrolyte_thickness'], 'porosity': 0.0},
            'cathode': {'thickness': config['cathode_thickness'], 'porosity': config['cathode_porosity']},
            'interconnect': {'thickness': config['interconnect_thickness'], 'porosity': 0.0},
            'sealant': {'thickness': config['sealant_thickness'], 'porosity': 0.0}
        }
        
    def generate_mesh(self, resolution=50):
        """生成3D网格"""
        gmsh.initialize()
        gmsh.model.add("SOFC")
        
        # 几何参数
        Lx, Ly = self.config['active_area']
        total_height = sum(layer['thickness'] for layer in self.layers.values())
        
        # 创建几何体
        points = []
        lines = []
        surfaces = []
        volumes = []
        
        # 定义点
        z_pos = 0
        for i in range(6):  # 5层 + 1个顶部点
            points.append(gmsh.model.geo.addPoint(0, 0, z_pos))
            points.append(gmsh.model.geo.addPoint(Lx, 0, z_pos))
            points.append(gmsh.model.geo.addPoint(Lx, Ly, z_pos))
            points.append(gmsh.model.geo.addPoint(0, Ly, z_pos))
            z_pos += list(self.layers.values())[i]['thickness'] if i < 5 else 0
        
        # 创建线和面
        for layer_idx in range(6):
            base_idx = layer_idx * 4
            # 底面
            if layer_idx < 5:
                lines.append(gmsh.model.geo.addLine(base_idx, base_idx + 1))
                lines.append(gmsh.model.geo.addLine(base_idx + 1, base_idx + 2))
                lines.append(gmsh.model.geo.addLine(base_idx + 2, base_idx + 3))
                lines.append(gmsh.model.geo.addLine(base_idx + 3, base_idx))
                
                surface = gmsh.model.geo.addCurveLoop([lines[-4], lines[-3], lines[-2], lines[-1]])
                surfaces.append(gmsh.model.geo.addPlaneSurface([surface]))
        
        # 创建体积
        for layer_idx in range(5):
            bottom_surf = surfaces[layer_idx]
            top_surf = surfaces[layer_idx + 1]
            
            # 创建侧面
            side_lines = []
            for i in range(4):
                side_lines.append(gmsh.model.geo.addLine(
                    layer_idx * 4 + i, 
                    (layer_idx + 1) * 4 + i
                ))
            
            # 创建侧面
            side_surfaces = []
            for i in range(4):
                curve_loop = gmsh.model.geo.addCurveLoop([
                    side_lines[i],
                    lines[(layer_idx + 1) * 4 + i],
                    -side_lines[(i + 1) % 4],
                    -lines[layer_idx * 4 + i]
                ])
                side_surfaces.append(gmsh.model.geo.addPlaneSurface([curve_loop]))
            
            # 创建体积
            volume_loop = gmsh.model.geo.addSurfaceLoop([bottom_surf, top_surf] + side_surfaces)
            volumes.append(gmsh.model.geo.addVolume([volume_loop]))
        
        gmsh.model.geo.synchronize()
        
        # 设置网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
        
        # 生成网格
        gmsh.model.mesh.generate(3)
        
        # 保存网格
        mesh_file = f"sofc_mesh_res{resolution}.msh"
        gmsh.write(mesh_file)
        
        # 读取网格数据
        mesh = meshio.read(mesh_file)
        
        gmsh.finalize()
        
        return mesh

class SOFCMultiPhysics:
    """SOFC多物理场耦合仿真"""
    
    def __init__(self, geometry, material_props, operating_conditions):
        self.geometry = geometry
        self.material_props = material_props
        self.operating_conditions = operating_conditions
        
    def solve_electrochemical(self, mesh):
        """求解电化学场（电流密度、过电位分布）"""
        n_nodes = len(mesh.points)
        
        # 初始化场变量
        current_density = np.zeros((n_nodes, 3))  # 3D电流密度向量
        overpotential = np.zeros(n_nodes)
        
        # 简化的电化学模型
        # 基于Butler-Volmer方程和Ohm定律
        
        # 电解质中的离子传导
        sigma_ion = self.material_props['electrolyte']['ionic_conductivity']
        sigma_elec_anode = self.material_props['anode']['electronic_conductivity']
        sigma_elec_cathode = self.material_props['cathode']['electronic_conductivity']
        
        # 计算电流密度分布
        for i, point in enumerate(mesh.points):
            x, y, z = point
            
            # 根据位置确定材料
            if z < self.geometry.layers['anode']['thickness']:
                # 阳极
                sigma = sigma_elec_anode
                eta = self.operating_conditions['anode_overpotential']
            elif z < self.geometry.layers['anode']['thickness'] + self.geometry.layers['electrolyte']['thickness']:
                # 电解质
                sigma = sigma_ion
                eta = self.operating_conditions['electrolyte_overpotential']
            else:
                # 阴极
                sigma = sigma_elec_cathode
                eta = self.operating_conditions['cathode_overpotential']
            
            # 简化的电流密度计算
            E_field = np.array([0, 0, eta / self.geometry.layers['electrolyte']['thickness']])
            current_density[i] = sigma * E_field
            overpotential[i] = eta
        
        return current_density, overpotential
    
    def solve_thermal(self, mesh, current_density):
        """求解热场（温度分布）"""
        n_nodes = len(mesh.points)
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
            for z in mesh.points[:, 2]
        ])
        
        # 简化的温度分布计算
        for i, point in enumerate(mesh.points):
            x, y, z = point
            
            # 基础温度分布
            if z < self.geometry.layers['anode']['thickness']:
                # 阳极（燃料侧）
                T_base = T_inlet_fuel
                k = k_anode
            elif z < self.geometry.layers['anode']['thickness'] + self.geometry.layers['electrolyte']['thickness']:
                # 电解质
                T_base = (T_inlet_fuel + T_inlet_air) / 2
                k = k_electrolyte
            else:
                # 阴极（空气侧）
                T_base = T_inlet_air
                k = k_cathode
            
            # 考虑焦耳热和热传导
            temp_rise = joule_heating[i] / (k * 1000)  # 简化的热传导计算
            temperature[i] = T_base + temp_rise
        
        return temperature
    
    def solve_mechanical(self, mesh, temperature):
        """求解机械场（应力、应变、位移）"""
        n_nodes = len(mesh.points)
        
        # 初始化场变量
        stress_tensor = np.zeros((n_nodes, 6))  # [σxx, σyy, σzz, σxy, σxz, σyz]
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
        
        for i, point in enumerate(mesh.points):
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
            # 假设平面应力状态
            strain_tensor[i] = np.array([
                thermal_strain,  # εxx
                thermal_strain,  # εyy
                thermal_strain,  # εzz
                0, 0, 0  # 剪切应变
            ])
            
            # 应力计算（简化的胡克定律）
            nu = 0.3  # 泊松比
            stress_tensor[i] = E / (1 - nu**2) * np.array([
                strain_tensor[i, 0] + nu * strain_tensor[i, 1],
                strain_tensor[i, 1] + nu * strain_tensor[i, 0],
                nu * (strain_tensor[i, 0] + strain_tensor[i, 1]),
                0, 0, 0
            ])
            
            # 位移计算（简化）
            displacement[i] = np.array([
                strain_tensor[i, 0] * x,
                strain_tensor[i, 1] * y,
                strain_tensor[i, 2] * z
            ])
        
        return stress_tensor, strain_tensor, displacement
    
    def solve_species_transport(self, mesh, temperature):
        """求解传质场（燃料物种浓度分布）"""
        n_nodes = len(mesh.points)
        
        # 物种浓度
        H2_concentration = np.zeros(n_nodes)
        H2O_concentration = np.zeros(n_nodes)
        
        # 燃料入口条件
        H2_inlet = self.operating_conditions['fuel_h2_fraction']
        H2O_inlet = self.operating_conditions['fuel_h2o_fraction']
        
        # 扩散系数（温度相关）
        D_H2 = 1e-4 * np.exp(-1000 / temperature)  # 简化的阿伦尼乌斯关系
        D_H2O = 1e-4 * np.exp(-1200 / temperature)
        
        for i, point in enumerate(mesh.points):
            x, y, z = point
            T = temperature[i]
            
            # 只在阳极中计算燃料浓度
            if z < self.geometry.layers['anode']['thickness']:
                # 简化的浓度分布（考虑反应消耗）
                reaction_rate = np.exp(-5000 / T)  # 简化的反应速率
                
                # 浓度随位置变化
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
        
    def generate_parameter_combinations(self, n_samples=1000):
        """使用拉丁超立方采样生成参数组合"""
        
        # 定义参数范围和分布
        parameters = {
            # 操作条件
            'voltage': (0.6, 0.9),  # V
            'current_density': (0.1, 1.0),  # A/cm²
            'fuel_flow_rate': (1e-6, 1e-4),  # m³/s
            'air_flow_rate': (1e-5, 1e-3),  # m³/s
            'fuel_inlet_temperature': (600, 900),  # K
            'air_inlet_temperature': (600, 900),  # K
            
            # 材料属性
            'anode_porosity': (0.2, 0.4),
            'cathode_porosity': (0.2, 0.4),
            'anode_permeability': (1e-15, 1e-12),  # m²
            'cathode_permeability': (1e-15, 1e-12),  # m²
            'anode_ionic_conductivity': (1e-3, 1e-1),  # S/m
            'cathode_ionic_conductivity': (1e-3, 1e-1),  # S/m
            'anode_electronic_conductivity': (1e2, 1e4),  # S/m
            'cathode_electronic_conductivity': (1e2, 1e4),  # S/m
            'electrolyte_ionic_conductivity': (1e-2, 1e0),  # S/m
            'anode_youngs_modulus': (50e9, 200e9),  # Pa
            'cathode_youngs_modulus': (50e9, 200e9),  # Pa
            'electrolyte_youngs_modulus': (100e9, 300e9),  # Pa
            'anode_cte': (8e-6, 15e-6),  # 1/K
            'cathode_cte': (8e-6, 15e-6),  # 1/K
            'electrolyte_cte': (10e-6, 18e-6),  # 1/K
            
            # 几何参数
            'anode_thickness': (200e-6, 1000e-6),  # m
            'electrolyte_thickness': (10e-6, 50e-6),  # m
            'cathode_thickness': (20e-6, 100e-6),  # m
            'active_area_x': (0.01, 0.1),  # m
            'active_area_y': (0.01, 0.1),  # m
        }
        
        # 生成拉丁超立方样本
        lhs_samples = pyDOE2.lhs(len(parameters), samples=n_samples, criterion='maximin')
        
        # 转换为实际参数值
        param_combinations = []
        for i, sample in enumerate(lhs_samples):
            param_dict = {}
            for j, (param_name, (min_val, max_val)) in enumerate(parameters.items()):
                param_dict[param_name] = min_val + sample[j] * (max_val - min_val)
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
                'interconnect_thickness': 1e-3,  # 固定值
                'sealant_thickness': 0.5e-3,  # 固定值
                'active_area': (params['active_area_x'], params['active_area_y']),
                'anode_porosity': params['anode_porosity'],
                'cathode_porosity': params['cathode_porosity']
            }
            
            # 创建几何
            geometry = SOFCGeometry(geometry_config)
            mesh = geometry.generate_mesh()
            
            # 材料属性
            material_props = {
                'anode': {
                    'porosity': params['anode_porosity'],
                    'permeability': params['anode_permeability'],
                    'ionic_conductivity': params['anode_ionic_conductivity'],
                    'electronic_conductivity': params['anode_electronic_conductivity'],
                    'thermal_conductivity': 10.0,  # W/m/K
                    'youngs_modulus': params['anode_youngs_modulus'],
                    'cte': params['anode_cte']
                },
                'electrolyte': {
                    'ionic_conductivity': params['electrolyte_ionic_conductivity'],
                    'thermal_conductivity': 2.0,  # W/m/K
                    'youngs_modulus': params['electrolyte_youngs_modulus'],
                    'cte': params['electrolyte_cte']
                },
                'cathode': {
                    'porosity': params['cathode_porosity'],
                    'permeability': params['cathode_permeability'],
                    'ionic_conductivity': params['cathode_ionic_conductivity'],
                    'electronic_conductivity': params['cathode_electronic_conductivity'],
                    'thermal_conductivity': 8.0,  # W/m/K
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
                'mesh_points': mesh.points,
                'mesh_cells': mesh.cells,
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
                    'n_nodes': len(mesh.points),
                    'n_cells': len(mesh.cells),
                    'convergence': True
                }
            }
            
            return simulation_result
            
        except Exception as e:
            print(f"仿真 {sim_id} 失败: {str(e)}")
            return None
    
    def generate_dataset(self, n_simulations=100, n_parallel=4):
        """生成完整数据集"""
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
        """保存数据集为HDF5格式"""
        with h5py.File(filename, 'w') as f:
            # 创建主组
            f.attrs['description'] = 'SOFC高保真数值数据集'
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['n_simulations'] = len(self.dataset)
            
            # 保存每个仿真数据
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
        """创建数据集摘要报告"""
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
        
        # 计算参数范围
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
            
            # 计算场数据统计
            field_names = list(self.dataset[0]['fields'].keys())
            for field in field_names:
                all_values = np.concatenate([sim['fields'][field].flatten() for sim in self.dataset])
                summary['field_statistics'][field] = {
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values))
                }
        
        # 保存摘要
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"数据集摘要已保存到: {filename}")
        return summary

def main():
    """主函数"""
    # 配置参数
    config = {
        'n_simulations': 100,  # 仿真数量
        'n_parallel': 4,       # 并行进程数
        'mesh_resolution': 50  # 网格分辨率
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
    
    print("\n=== 数据集生成完成 ===")
    print(f"成功生成: {len(dataset)} 个仿真")
    print(f"数据文件: {h5_filename}")
    print(f"摘要报告: dataset_summary.json")
    
    return dataset, h5_filename, summary

if __name__ == "__main__":
    dataset, filename, summary = main()