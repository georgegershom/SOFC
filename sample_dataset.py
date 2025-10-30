"""
示例数据集生成器 - 为PINN训练生成各种测试数据集
包含解析解、数值解、实验数据模拟等
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import h5py
import json
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

class SampleDatasetGenerator:
    """
    示例数据集生成器
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 物理参数
        self.physical_params = {
            'Lx': 1.0,      # 域长度 x
            'Ly': 1.0,      # 域长度 y  
            'Lz': 1.0,      # 域长度 z
            'T_max': 1.0,   # 时间长度
            'T_ref': 298.0, # 参考温度 K
            'c_ref': 1.0,   # 参考浓度 mol/m³
        }
    
    def generate_analytical_solution(self, coordinates, problem_type='heat_conduction'):
        """
        生成解析解
        
        Args:
            coordinates: 坐标点 [N, 4] (x, y, z, t)
            problem_type: 问题类型
            
        Returns:
            analytical_solution: 解析解字典
        """
        x = coordinates[:, 0]
        y = coordinates[:, 1] 
        z = coordinates[:, 2]
        t = coordinates[:, 3]
        
        if problem_type == 'heat_conduction':
            # 热传导解析解: T = sin(πx)sin(πy)exp(-π²t)
            T = np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-np.pi**2 * t)
            phi_e = np.cos(np.pi * x) * np.cos(np.pi * y) * t
            phi_s = np.sin(np.pi * x) * np.sin(np.pi * y) * t
            c_li = 1.0 + 0.1 * np.sin(np.pi * x) * np.cos(np.pi * y)
            u = 0.01 * np.sin(np.pi * x) * np.cos(np.pi * y) * t
            
        elif problem_type == 'wave_equation':
            # 波动方程解析解: u = sin(πx)sin(πy)cos(πt)
            T = np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * t)
            phi_e = np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * t)
            phi_s = np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * t)
            c_li = 1.0 + 0.1 * np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * t)
            u = 0.01 * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * t)
            
        elif problem_type == 'battery_thermal':
            # 电池热问题解析解
            T = 298.0 + 50 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-0.1 * t)
            phi_e = 3.7 + 0.1 * np.cos(np.pi * x) * np.cos(np.pi * y) * t
            phi_s = 3.7 + 0.1 * np.sin(np.pi * x) * np.sin(np.pi * y) * t
            c_li = 1000.0 + 100 * np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-0.05 * t)
            u = 0.001 * np.sin(np.pi * x) * np.cos(np.pi * y) * t
            
        else:
            raise ValueError(f"未知的问题类型: {problem_type}")
        
        return {
            'T': T.reshape(-1, 1),
            'phi_e': phi_e.reshape(-1, 1),
            'phi_s': phi_s.reshape(-1, 1),
            'c_li': c_li.reshape(-1, 1),
            'u': np.column_stack([u, u*0.5, u*0.2])  # 3D位移
        }
    
    def generate_numerical_solution(self, coordinates, problem_type='heat_conduction'):
        """
        生成数值解 (使用有限差分法)
        
        Args:
            coordinates: 坐标点
            problem_type: 问题类型
            
        Returns:
            numerical_solution: 数值解字典
        """
        # 这里简化为解析解加噪声
        analytical = self.generate_analytical_solution(coordinates, problem_type)
        
        # 添加数值误差
        noise_level = 0.01
        numerical = {}
        for key, value in analytical.items():
            noise = np.random.normal(0, noise_level * np.std(value), value.shape)
            numerical[key] = value + noise
        
        return numerical
    
    def generate_experimental_data(self, coordinates, measurement_noise=0.05):
        """
        生成实验数据 (解析解加测量噪声)
        
        Args:
            coordinates: 坐标点
            measurement_noise: 测量噪声水平
            
        Returns:
            experimental_data: 实验数据字典
        """
        # 基于解析解生成
        analytical = self.generate_analytical_solution(coordinates, 'battery_thermal')
        
        experimental = {}
        for key, value in analytical.items():
            # 添加测量噪声
            noise = np.random.normal(0, measurement_noise * np.std(value), value.shape)
            experimental[key] = value + noise
            
            # 添加测量误差 (传感器精度限制)
            if key == 'T':
                experimental[key] = np.round(experimental[key], 1)  # 温度精度0.1K
            elif key in ['phi_e', 'phi_s']:
                experimental[key] = np.round(experimental[key], 3)  # 电位精度1mV
            elif key == 'c_li':
                experimental[key] = np.round(experimental[key], 0)  # 浓度精度1mol/m³
        
        return experimental
    
    def generate_training_dataset(self, n_points=10000, n_boundary=1000, n_data=500):
        """
        生成训练数据集
        
        Args:
            n_points: 域内点数
            n_boundary: 边界点数
            n_data: 观测数据点数
            
        Returns:
            dataset: 训练数据集
        """
        # 1. 域内点
        domain_points = np.random.uniform(0, 1, (n_points, 4))
        
        # 2. 边界点
        boundary_points = []
        # 左边界 (x=0)
        left_boundary = np.random.uniform(0, 1, (n_boundary//4, 4))
        left_boundary[:, 0] = 0
        boundary_points.append(left_boundary)
        
        # 右边界 (x=1)
        right_boundary = np.random.uniform(0, 1, (n_boundary//4, 4))
        right_boundary[:, 0] = 1
        boundary_points.append(right_boundary)
        
        # 下边界 (y=0)
        bottom_boundary = np.random.uniform(0, 1, (n_boundary//4, 4))
        bottom_boundary[:, 1] = 0
        boundary_points.append(bottom_boundary)
        
        # 上边界 (y=1)
        top_boundary = np.random.uniform(0, 1, (n_boundary//4, 4))
        top_boundary[:, 1] = 1
        boundary_points.append(top_boundary)
        
        boundary_points = np.vstack(boundary_points)
        
        # 3. 观测数据点 (从域内点中随机选择)
        data_indices = np.random.choice(n_points, n_data, replace=False)
        data_points = domain_points[data_indices]
        
        # 4. 生成解
        all_coordinates = np.vstack([domain_points, boundary_points, data_points])
        analytical_solution = self.generate_analytical_solution(all_coordinates, 'battery_thermal')
        experimental_data = self.generate_experimental_data(data_points)
        
        # 5. 组装数据集
        dataset = {
            'coordinates': {
                'domain': domain_points,
                'boundary': boundary_points,
                'data': data_points
            },
            'solutions': {
                'analytical': analytical_solution,
                'experimental': experimental_data
            },
            'data_indices': data_indices,
            'metadata': {
                'n_domain': n_points,
                'n_boundary': n_boundary,
                'n_data': n_data,
                'domain_bounds': [[0, 1], [0, 1], [0, 1], [0, 1]],
                'problem_type': 'battery_thermal'
            }
        }
        
        return dataset
    
    def generate_test_cases(self):
        """
        生成各种测试用例
        """
        test_cases = {}
        
        # 测试用例1: 简单热传导
        print("生成测试用例1: 简单热传导...")
        coords1 = np.random.uniform(0, 1, (1000, 4))
        test_cases['heat_conduction'] = {
            'coordinates': coords1,
            'solution': self.generate_analytical_solution(coords1, 'heat_conduction'),
            'problem_type': 'heat_conduction'
        }
        
        # 测试用例2: 波动方程
        print("生成测试用例2: 波动方程...")
        coords2 = np.random.uniform(0, 1, (1000, 4))
        test_cases['wave_equation'] = {
            'coordinates': coords2,
            'solution': self.generate_analytical_solution(coords2, 'wave_equation'),
            'problem_type': 'wave_equation'
        }
        
        # 测试用例3: 电池热问题
        print("生成测试用例3: 电池热问题...")
        coords3 = np.random.uniform(0, 1, (1000, 4))
        test_cases['battery_thermal'] = {
            'coordinates': coords3,
            'solution': self.generate_analytical_solution(coords3, 'battery_thermal'),
            'problem_type': 'battery_thermal'
        }
        
        # 测试用例4: 训练数据集
        print("生成测试用例4: 训练数据集...")
        test_cases['training_dataset'] = self.generate_training_dataset()
        
        return test_cases
    
    def save_dataset(self, dataset, filepath):
        """
        保存数据集到文件
        
        Args:
            dataset: 数据集
            filepath: 文件路径
        """
        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'w') as f:
                # 保存坐标
                f.create_group('coordinates')
                for key, value in dataset['coordinates'].items():
                    f['coordinates'].create_dataset(key, data=value)
                
                # 保存解
                f.create_group('solutions')
                for sol_type, sol_data in dataset['solutions'].items():
                    f['solutions'].create_group(sol_type)
                    for field, data in sol_data.items():
                        f['solutions'][sol_type].create_dataset(field, data=data)
                
                # 保存元数据
                f.create_group('metadata')
                for key, value in dataset['metadata'].items():
                    f['metadata'].attrs[key] = value
                
                # 保存数据索引
                if 'data_indices' in dataset:
                    f.create_dataset('data_indices', data=dataset['data_indices'])
        
        elif filepath.endswith('.npz'):
            # 展平所有数据
            save_data = {}
            for key, value in dataset['coordinates'].items():
                save_data[f'coords_{key}'] = value
            for sol_type, sol_data in dataset['solutions'].items():
                for field, data in sol_data.items():
                    save_data[f'{sol_type}_{field}'] = data
            if 'data_indices' in dataset:
                save_data['data_indices'] = dataset['data_indices']
            
            np.savez(filepath, **save_data)
        
        else:
            raise ValueError("不支持的文件格式，请使用.h5或.npz")
    
    def load_dataset(self, filepath):
        """
        从文件加载数据集
        
        Args:
            filepath: 文件路径
            
        Returns:
            dataset: 数据集
        """
        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                dataset = {}
                
                # 加载坐标
                dataset['coordinates'] = {}
                for key in f['coordinates'].keys():
                    dataset['coordinates'][key] = f['coordinates'][key][:]
                
                # 加载解
                dataset['solutions'] = {}
                for sol_type in f['solutions'].keys():
                    dataset['solutions'][sol_type] = {}
                    for field in f['solutions'][sol_type].keys():
                        dataset['solutions'][sol_type][field] = f['solutions'][sol_type][field][:]
                
                # 加载元数据
                dataset['metadata'] = dict(f['metadata'].attrs)
                
                # 加载数据索引
                if 'data_indices' in f:
                    dataset['data_indices'] = f['data_indices'][:]
        
        elif filepath.endswith('.npz'):
            data = np.load(filepath)
            dataset = {
                'coordinates': {},
                'solutions': {'analytical': {}, 'experimental': {}},
                'metadata': {}
            }
            
            # 解析数据
            for key, value in data.items():
                if key.startswith('coords_'):
                    dataset['coordinates'][key[7:]] = value
                elif key.startswith('analytical_'):
                    dataset['solutions']['analytical'][key[11:]] = value
                elif key.startswith('experimental_'):
                    dataset['solutions']['experimental'][key[13:]] = value
                elif key == 'data_indices':
                    dataset['data_indices'] = value
        
        return dataset
    
    def visualize_dataset(self, dataset, save_path=None):
        """
        可视化数据集
        
        Args:
            dataset: 数据集
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 提取数据
        coords = dataset['coordinates']['domain']
        analytical = dataset['solutions']['analytical']
        
        # 确保数据维度匹配
        n_points = min(coords.shape[0], analytical['T'].shape[0])
        coords = coords[:n_points]
        for key in analytical:
            analytical[key] = analytical[key][:n_points]
        
        # 温度场
        scatter1 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=analytical['T'].flatten(), 
                                    cmap='hot', s=1)
        axes[0, 0].set_title('Temperature Field')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # 电解质电位
        scatter2 = axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=analytical['phi_e'].flatten(), 
                                    cmap='viridis', s=1)
        axes[0, 1].set_title('Electrolyte Potential')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # 锂离子浓度
        scatter3 = axes[0, 2].scatter(coords[:, 0], coords[:, 1], c=analytical['c_li'].flatten(), 
                                    cmap='plasma', s=1)
        axes[0, 2].set_title('Li+ Concentration')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')
        plt.colorbar(scatter3, ax=axes[0, 2])
        
        # 时间演化
        t_unique = np.unique(coords[:, 3])
        if len(t_unique) > 1:
            t_indices = [np.where(coords[:, 3] == t)[0] for t in t_unique[:5]]
            for i, t_idx in enumerate(t_indices):
                if len(t_idx) > 0:
                    axes[1, 0].plot(coords[t_idx, 0], analytical['T'][t_idx].flatten(), 
                                   label=f't={t_unique[i]:.2f}')
            axes[1, 0].set_title('Temperature Evolution')
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('T')
            axes[1, 0].legend()
        
        # 数据分布
        axes[1, 1].hist(analytical['T'].flatten(), bins=50, alpha=0.7, label='Temperature')
        axes[1, 1].hist(analytical['phi_e'].flatten(), bins=50, alpha=0.7, label='Potential')
        axes[1, 1].set_title('Data Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # 边界点
        boundary_coords = dataset['coordinates']['boundary']
        axes[1, 2].scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                          c='red', s=1, label='Boundary Points')
        axes[1, 2].scatter(coords[:, 0], coords[:, 1], c='blue', s=1, alpha=0.3, label='Domain Points')
        axes[1, 2].set_title('Point Distribution')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('y')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def test_sample_dataset_generator():
    """
    测试示例数据集生成器
    """
    print("=== 测试示例数据集生成器 ===\n")
    
    # 创建生成器
    generator = SampleDatasetGenerator()
    
    # 生成测试用例
    print("1. 生成测试用例...")
    test_cases = generator.generate_test_cases()
    
    for case_name, case_data in test_cases.items():
        if case_name != 'training_dataset':
            print(f"   {case_name}: 坐标形状 {case_data['coordinates'].shape}")
            for field, data in case_data['solution'].items():
                print(f"     {field}: {data.shape}")
    
    # 生成训练数据集
    print("\n2. 生成训练数据集...")
    training_dataset = generator.generate_training_dataset(n_points=5000, n_boundary=500, n_data=200)
    print(f"   域内点: {training_dataset['coordinates']['domain'].shape[0]}")
    print(f"   边界点: {training_dataset['coordinates']['boundary'].shape[0]}")
    print(f"   数据点: {training_dataset['coordinates']['data'].shape[0]}")
    
    # 保存数据集
    print("\n3. 保存数据集...")
    generator.save_dataset(training_dataset, '/workspace/training_dataset.h5')
    generator.save_dataset(training_dataset, '/workspace/training_dataset.npz')
    print("   数据集已保存为 .h5 和 .npz 格式")
    
    # 加载数据集
    print("\n4. 加载数据集...")
    loaded_dataset = generator.load_dataset('/workspace/training_dataset.h5')
    print("   数据集加载成功")
    
    # 可视化数据集
    print("\n5. 可视化数据集...")
    generator.visualize_dataset(training_dataset, '/workspace/dataset_visualization.png')
    print("   可视化图像已保存")
    
    print("\n=== 示例数据集生成器测试完成 ===")

if __name__ == "__main__":
    test_sample_dataset_generator()