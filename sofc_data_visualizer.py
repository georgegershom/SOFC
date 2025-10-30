#!/usr/bin/env python3
"""
SOFC数据集可视化和分析工具
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import pandas as pd
import seaborn as sns
from scipy import stats
import json
import os

class SOFCDataVisualizer:
    """SOFC数据集可视化工具"""
    
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.data = None
        self.load_dataset()
    
    def load_dataset(self):
        """加载数据集"""
        print(f"加载数据集: {self.dataset_file}")
        self.data = h5py.File(self.dataset_file, 'r')
        print(f"数据集包含 {self.data.attrs['n_simulations']} 个仿真")
    
    def plot_parameter_distributions(self, save_path="parameter_distributions.png"):
        """绘制输入参数分布"""
        print("生成参数分布图...")
        
        # 收集所有参数
        param_data = {}
        for sim_id in self.data.keys():
            if sim_id.startswith('simulation_'):
                params = self.data[sim_id]['input_parameters'].attrs
                for key, value in params.items():
                    if key not in param_data:
                        param_data[key] = []
                    param_data[key].append(value)
        
        # 创建子图
        n_params = len(param_data)
        n_cols = 4
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, (param_name, values) in enumerate(param_data.items()):
            if i < len(axes):
                ax = axes[i]
                ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{param_name}', fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"参数分布图已保存到: {save_path}")
    
    def plot_field_visualization(self, sim_id=0, save_dir="field_visualizations"):
        """可视化特定仿真的场数据"""
        print(f"生成仿真 {sim_id} 的场数据可视化...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        sim_key = f'simulation_{sim_id:06d}'
        if sim_key not in self.data:
            print(f"仿真 {sim_id} 不存在")
            return
        
        sim_data = self.data[sim_key]
        points = sim_data['mesh']['points'][:]
        fields = sim_data['fields']
        
        # 创建PyVista网格
        mesh = pv.PolyData(points)
        
        # 为每个场创建可视化
        field_names = ['temperature', 'overpotential', 'von_mises_stress', 'H2_concentration']
        
        for field_name in field_names:
            if field_name in fields:
                field_data = fields[field_name][:]
                
                # 创建3D可视化
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(mesh, scalars=field_data, cmap='viridis', 
                               point_size=3, render_points_as_spheres=True)
                plotter.add_scalar_bar(title=field_name)
                plotter.set_background('white')
                
                # 保存图像
                filename = f"{save_dir}/sim_{sim_id:06d}_{field_name}.png"
                plotter.screenshot(filename)
                plotter.close()
                
                print(f"  {field_name} 可视化已保存到: {filename}")
    
    def plot_correlation_matrix(self, save_path="correlation_matrix.png"):
        """绘制参数相关性矩阵"""
        print("生成参数相关性矩阵...")
        
        # 收集参数数据
        param_data = {}
        for sim_id in self.data.keys():
            if sim_id.startswith('simulation_'):
                params = self.data[sim_id]['input_parameters'].attrs
                for key, value in params.items():
                    if key not in param_data:
                        param_data[key] = []
                    param_data[key].append(value)
        
        # 创建DataFrame
        df = pd.DataFrame(param_data)
        
        # 计算相关性矩阵
        corr_matrix = df.corr()
        
        # 绘制热力图
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('参数相关性矩阵')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关性矩阵已保存到: {save_path}")
    
    def plot_field_statistics(self, save_path="field_statistics.png"):
        """绘制场数据统计信息"""
        print("生成场数据统计图...")
        
        # 收集场数据统计
        field_stats = {}
        field_names = ['temperature', 'overpotential', 'von_mises_stress', 'H2_concentration', 'H2O_concentration']
        
        for field_name in field_names:
            all_values = []
            for sim_id in self.data.keys():
                if sim_id.startswith('simulation_'):
                    if field_name in self.data[sim_id]['fields']:
                        field_data = self.data[sim_id]['fields'][field_name][:]
                        all_values.extend(field_data.flatten())
            
            if all_values:
                field_stats[field_name] = {
                    'mean': np.mean(all_values),
                    'std': np.std(all_values),
                    'min': np.min(all_values),
                    'max': np.max(all_values),
                    'values': all_values
                }
        
        # 创建子图
        n_fields = len(field_stats)
        n_cols = 2
        n_rows = (n_fields + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, (field_name, stats) in enumerate(field_stats.items()):
            if i < len(axes):
                ax = axes[i]
                
                # 绘制直方图
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"场数据统计图已保存到: {save_path}")
    
    def create_interactive_3d_plot(self, sim_id=0, field_name='temperature'):
        """创建交互式3D可视化"""
        print(f"创建仿真 {sim_id} 的交互式3D可视化...")
        
        sim_key = f'simulation_{sim_id:06d}'
        if sim_key not in self.data:
            print(f"仿真 {sim_id} 不存在")
            return
        
        sim_data = self.data[sim_key]
        points = sim_data['mesh']['points'][:]
        
        if field_name not in sim_data['fields']:
            print(f"场 {field_name} 不存在")
            return
        
        field_data = sim_data['fields'][field_name][:]
        
        # 创建PyVista网格
        mesh = pv.PolyData(points)
        mesh[field_name] = field_data
        
        # 创建交互式绘图
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars=field_name, cmap='viridis', 
                        point_size=5, render_points_as_spheres=True)
        plotter.add_scalar_bar(title=field_name)
        plotter.set_background('white')
        plotter.show()
    
    def export_to_vtk(self, sim_id=0, output_dir="vtk_exports"):
        """导出为VTK格式用于ParaView可视化"""
        print(f"导出仿真 {sim_id} 为VTK格式...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        sim_key = f'simulation_{sim_id:06d}'
        if sim_key not in self.data:
            print(f"仿真 {sim_id} 不存在")
            return
        
        sim_data = self.data[sim_key]
        points = sim_data['mesh']['points'][:]
        fields = sim_data['fields']
        
        # 创建PyVista网格
        mesh = pv.PolyData(points)
        
        # 添加所有场数据
        for field_name, field_data in fields.items():
            mesh[field_name] = field_data[:]
        
        # 保存为VTK文件
        vtk_filename = f"{output_dir}/sim_{sim_id:06d}.vtk"
        mesh.save(vtk_filename)
        print(f"VTK文件已保存到: {vtk_filename}")
    
    def generate_report(self, output_file="visualization_report.html"):
        """生成HTML可视化报告"""
        print("生成HTML可视化报告...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SOFC高保真数值数据集可视化报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .image {{ text-align: center; margin: 20px 0; }}
                .image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SOFC高保真数值数据集可视化报告</h1>
                <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>数据集文件: {self.dataset_file}</p>
                <p>仿真数量: {self.data.attrs['n_simulations']}</p>
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
                <h2>参数分布</h2>
                <div class="image">
                    <img src="parameter_distributions.png" alt="参数分布图">
                </div>
            </div>
            
            <div class="section">
                <h2>参数相关性</h2>
                <div class="image">
                    <img src="correlation_matrix.png" alt="相关性矩阵">
                </div>
            </div>
            
            <div class="section">
                <h2>场数据统计</h2>
                <div class="image">
                    <img src="field_statistics.png" alt="场数据统计">
                </div>
            </div>
            
            <div class="section">
                <h2>3D场数据可视化</h2>
                <p>详细的3D场数据可视化请查看 field_visualizations/ 目录中的图像文件。</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存到: {output_file}")
    
    def close(self):
        """关闭数据集文件"""
        if self.data:
            self.data.close()

def main():
    """主函数"""
    dataset_file = "sofc_high_fidelity_dataset.h5"
    
    if not os.path.exists(dataset_file):
        print(f"数据集文件 {dataset_file} 不存在，请先运行数据集生成器")
        return
    
    print("=== SOFC数据集可视化工具 ===")
    
    # 创建可视化器
    visualizer = SOFCDataVisualizer(dataset_file)
    
    try:
        # 生成各种可视化
        visualizer.plot_parameter_distributions()
        visualizer.plot_correlation_matrix()
        visualizer.plot_field_statistics()
        
        # 生成前几个仿真的场数据可视化
        for sim_id in range(min(5, visualizer.data.attrs['n_simulations'])):
            visualizer.plot_field_visualization(sim_id)
            visualizer.export_to_vtk(sim_id)
        
        # 生成HTML报告
        visualizer.generate_report()
        
        print("\n=== 可视化完成 ===")
        print("生成的文件:")
        print("- parameter_distributions.png: 参数分布图")
        print("- correlation_matrix.png: 参数相关性矩阵")
        print("- field_statistics.png: 场数据统计图")
        print("- field_visualizations/: 3D场数据可视化")
        print("- vtk_exports/: VTK格式文件（用于ParaView）")
        print("- visualization_report.html: HTML可视化报告")
        
    finally:
        visualizer.close()

if __name__ == "__main__":
    main()