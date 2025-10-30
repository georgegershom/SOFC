#!/usr/bin/env python3
"""
SOFC数据集分析工具
用于加载、分析和可视化生成的高保真数值数据集
"""

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import json
import os

class SOFCDatasetAnalyzer:
    """SOFC数据集分析器"""
    
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.data = None
        self.load_dataset()
    
    def load_dataset(self):
        """加载数据集"""
        print(f"加载数据集: {self.dataset_file}")
        self.data = h5py.File(self.dataset_file, 'r')
        print(f"数据集包含 {self.data.attrs['n_simulations']} 个仿真")
    
    def get_dataset_info(self):
        """获取数据集基本信息"""
        info = {
            'total_simulations': self.data.attrs['n_simulations'],
            'creation_date': self.data.attrs['creation_date'],
            'description': self.data.attrs['description']
        }
        return info
    
    def load_parameters_dataframe(self):
        """加载参数数据为DataFrame"""
        param_data = []
        
        for sim_id in self.data.keys():
            if sim_id.startswith('simulation_'):
                params = self.data[sim_id]['input_parameters'].attrs
                param_dict = dict(params)
                param_data.append(param_dict)
        
        return pd.DataFrame(param_data)
    
    def load_field_data(self, field_name, sim_id=0):
        """加载特定仿真的场数据"""
        sim_key = f'simulation_{sim_id:06d}'
        if sim_key not in self.data:
            raise ValueError(f"仿真 {sim_id} 不存在")
        
        field_data = self.data[sim_key]['fields'][field_name][:]
        return field_data
    
    def plot_parameter_correlations(self, save_path="parameter_correlations.png"):
        """绘制参数相关性热力图"""
        print("生成参数相关性分析...")
        
        df = self.load_parameters_dataframe()
        
        # 计算相关性矩阵
        corr_matrix = df.corr()
        
        # 创建热力图
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('SOFC参数相关性矩阵', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"参数相关性图已保存到: {save_path}")
    
    def plot_field_distribution(self, field_name, save_path=None):
        """绘制场数据分布"""
        if save_path is None:
            save_path = f"{field_name}_distribution.png"
        
        print(f"生成 {field_name} 场数据分布图...")
        
        # 收集所有仿真的场数据
        all_values = []
        for sim_id in range(self.data.attrs['n_simulations']):
            sim_key = f'simulation_{sim_id:06d}'
            if field_name in self.data[sim_key]['fields']:
                field_data = self.data[sim_key]['fields'][field_name][:]
                all_values.extend(field_data.flatten())
        
        if not all_values:
            print(f"场 {field_name} 不存在")
            return
        
        all_values = np.array(all_values)
        
        # 创建分布图
        plt.figure(figsize=(10, 6))
        plt.hist(all_values, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(all_values), color='red', linestyle='--', 
                   label=f'均值: {np.mean(all_values):.2e}')
        plt.axvline(np.median(all_values), color='orange', linestyle='--', 
                   label=f'中位数: {np.median(all_values):.2e}')
        
        plt.title(f'{field_name} 场数据分布')
        plt.xlabel('值')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{field_name} 分布图已保存到: {save_path}")
    
    def plot_3d_field_visualization(self, sim_id=0, field_name='temperature', save_path=None):
        """创建3D场数据可视化"""
        if save_path is None:
            save_path = f"3d_{field_name}_sim_{sim_id:03d}.png"
        
        print(f"生成仿真 {sim_id} 的 {field_name} 3D可视化...")
        
        # 加载网格和场数据
        sim_key = f'simulation_{sim_id:06d}'
        points = self.data[sim_key]['mesh']['points'][:]
        field_data = self.data[sim_key]['fields'][field_name][:]
        
        # 创建3D散点图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D散点图
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=field_data, cmap='viridis', s=20, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'仿真 {sim_id}: {field_name} 分布')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"3D可视化已保存到: {save_path}")
    
    def analyze_performance_metrics(self):
        """分析性能指标"""
        print("分析SOFC性能指标...")
        
        performance_data = []
        
        for sim_id in range(self.data.attrs['n_simulations']):
            sim_key = f'simulation_{sim_id:06d}'
            
            # 获取输入参数
            params = self.data[sim_key]['input_parameters'].attrs
            
            # 获取场数据
            temperature = self.data[sim_key]['fields']['temperature'][:]
            current_density = self.data[sim_key]['fields']['current_density'][:]
            von_mises_stress = self.data[sim_key]['fields']['von_mises_stress'][:]
            
            # 计算性能指标
            max_temp = np.max(temperature)
            avg_temp = np.mean(temperature)
            max_stress = np.max(von_mises_stress)
            current_magnitude = np.linalg.norm(current_density, axis=1)
            max_current = np.max(current_magnitude)
            avg_current = np.mean(current_magnitude)
            
            performance_data.append({
                'simulation_id': sim_id,
                'voltage': params['voltage'],
                'current_density_input': params['current_density'],
                'max_temperature': max_temp,
                'avg_temperature': avg_temp,
                'max_stress': max_stress,
                'max_current_density': max_current,
                'avg_current_density': avg_current,
                'anode_thickness': params['anode_thickness'],
                'electrolyte_thickness': params['electrolyte_thickness'],
                'cathode_thickness': params['cathode_thickness']
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        # 保存性能分析结果
        df_performance.to_csv('sofc_performance_analysis.csv', index=False)
        print("性能分析结果已保存到: sofc_performance_analysis.csv")
        
        # 创建性能指标可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 电压 vs 最大温度
        axes[0, 0].scatter(df_performance['voltage'], df_performance['max_temperature'], alpha=0.7)
        axes[0, 0].set_xlabel('电压 (V)')
        axes[0, 0].set_ylabel('最大温度 (K)')
        axes[0, 0].set_title('电压 vs 最大温度')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 电流密度 vs 最大应力
        axes[0, 1].scatter(df_performance['current_density_input'], df_performance['max_stress'], alpha=0.7)
        axes[0, 1].set_xlabel('输入电流密度 (A/cm²)')
        axes[0, 1].set_ylabel('最大von Mises应力 (Pa)')
        axes[0, 1].set_title('电流密度 vs 最大应力')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 电解质厚度 vs 最大温度
        axes[1, 0].scatter(df_performance['electrolyte_thickness']*1e6, df_performance['max_temperature'], alpha=0.7)
        axes[1, 0].set_xlabel('电解质厚度 (μm)')
        axes[1, 0].set_ylabel('最大温度 (K)')
        axes[1, 0].set_title('电解质厚度 vs 最大温度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 温度分布
        axes[1, 1].hist(df_performance['max_temperature'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('最大温度 (K)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('最大温度分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sofc_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("性能分析图已保存到: sofc_performance_analysis.png")
        
        return df_performance
    
    def export_sample_data(self, n_samples=5, output_dir="sample_data"):
        """导出样本数据用于机器学习"""
        print(f"导出 {n_samples} 个样本数据...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(min(n_samples, self.data.attrs['n_simulations'])):
            sim_key = f'simulation_{i:06d}'
            
            # 创建样本数据目录
            sample_dir = f"{output_dir}/sample_{i:03d}"
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存输入参数
            params = self.data[sim_key]['input_parameters'].attrs
            with open(f"{sample_dir}/input_parameters.json", 'w') as f:
                json.dump(dict(params), f, indent=2)
            
            # 保存网格数据
            points = self.data[sim_key]['mesh']['points'][:]
            cells = self.data[sim_key]['mesh']['cells'][:]
            np.savetxt(f"{sample_dir}/mesh_points.txt", points)
            np.savetxt(f"{sample_dir}/mesh_cells.txt", cells, fmt='%d')
            
            # 保存场数据
            fields = self.data[sim_key]['fields']
            for field_name, field_data in fields.items():
                np.savetxt(f"{sample_dir}/{field_name}.txt", field_data)
            
            print(f"样本 {i} 已保存到: {sample_dir}")
    
    def generate_report(self, output_file="dataset_analysis_report.html"):
        """生成HTML分析报告"""
        print("生成HTML分析报告...")
        
        # 获取基本信息
        info = self.get_dataset_info()
        
        # 加载参数数据
        df_params = self.load_parameters_dataframe()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SOFC高保真数值数据集分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .image {{ text-align: center; margin: 20px 0; }}
                .image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SOFC高保真数值数据集分析报告</h1>
                <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>数据集文件: {self.dataset_file}</p>
                <p>仿真数量: {info['total_simulations']}</p>
                <p>创建日期: {info['creation_date']}</p>
            </div>
            
            <div class="section">
                <h2>数据集概览</h2>
                <div class="stats">
                    <p>本数据集包含SOFC（固体氧化物燃料电池）的高保真数值仿真数据，涵盖多物理场耦合分析：</p>
                    <ul>
                        <li>电化学场：电流密度分布、过电位分布</li>
                        <li>热场：温度分布</li>
                        <li>机械场：应力张量、应变张量、位移场</li>
                        <li>传质场：燃料物种浓度分布（H₂, H₂O）</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>参数统计</h2>
                <div class="stats">
                    <p>数据集包含 {len(df_params.columns)} 个输入参数，涵盖操作条件、材料属性和几何参数。</p>
                </div>
            </div>
            
            <div class="section">
                <h2>可视化分析</h2>
                <div class="image">
                    <img src="parameter_correlations.png" alt="参数相关性矩阵">
                    <p>参数相关性分析</p>
                </div>
            </div>
            
            <div class="section">
                <h2>性能分析</h2>
                <div class="image">
                    <img src="sofc_performance_analysis.png" alt="性能分析">
                    <p>SOFC性能指标分析</p>
                </div>
            </div>
            
            <div class="section">
                <h2>数据使用建议</h2>
                <div class="stats">
                    <p>本数据集可用于：</p>
                    <ul>
                        <li>机器学习模型训练（回归、分类）</li>
                        <li>神经网络架构设计</li>
                        <li>多物理场耦合分析</li>
                        <li>SOFC性能优化研究</li>
                        <li>材料属性敏感性分析</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML分析报告已保存到: {output_file}")
    
    def close(self):
        """关闭数据集文件"""
        if self.data:
            self.data.close()

def main():
    """主函数"""
    dataset_file = "sofc_high_fidelity_dataset.h5"
    
    if not os.path.exists(dataset_file):
        print(f"数据集文件 {dataset_file} 不存在")
        return
    
    print("=== SOFC数据集分析工具 ===")
    
    # 创建分析器
    analyzer = SOFCDatasetAnalyzer(dataset_file)
    
    try:
        # 执行各种分析
        analyzer.plot_parameter_correlations()
        
        # 分析主要场数据
        field_names = ['temperature', 'overpotential', 'von_mises_stress', 'H2_concentration']
        for field_name in field_names:
            analyzer.plot_field_distribution(field_name)
        
        # 创建3D可视化
        for sim_id in range(min(3, analyzer.data.attrs['n_simulations'])):
            analyzer.plot_3d_field_visualization(sim_id, 'temperature')
        
        # 性能分析
        performance_df = analyzer.analyze_performance_metrics()
        
        # 导出样本数据
        analyzer.export_sample_data(n_samples=5)
        
        # 生成HTML报告
        analyzer.generate_report()
        
        print("\n=== 分析完成 ===")
        print("生成的文件:")
        print("- parameter_correlations.png: 参数相关性矩阵")
        print("- *_distribution.png: 各场数据分布图")
        print("- 3d_*_sim_*.png: 3D场数据可视化")
        print("- sofc_performance_analysis.png: 性能分析图")
        print("- sofc_performance_analysis.csv: 性能分析数据")
        print("- sample_data/: 样本数据目录")
        print("- dataset_analysis_report.html: HTML分析报告")
        
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()