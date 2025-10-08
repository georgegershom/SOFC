#!/usr/bin/env python3
"""
FEM Validation Dataset Visualization and Analysis Script
Generates comprehensive plots and analysis for multi-scale validation data
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FEMDataVisualizer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.figures_path = self.data_path / 'figures'
        self.figures_path.mkdir(exist_ok=True)
        
    def load_json_data(self, file_path):
        """Load JSON data from file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def plot_residual_stress_profile(self):
        """Plot residual stress depth profile from synchrotron data"""
        data = self.load_json_data(self.data_path / 'residual_stress/experimental/synchrotron_xrd.json')
        measurements = data['subsurface_stress_profile']['stress_measurements']
        
        depths = [m['depth_um'] for m in measurements]
        sigma_xx = [m['sigma_xx_MPa'] for m in measurements]
        sigma_yy = [m['sigma_yy_MPa'] for m in measurements]
        von_mises = [m['von_mises_stress_MPa'] for m in measurements]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Principal stresses vs depth
        ax1.plot(depths, sigma_xx, 'o-', label='σ_xx', linewidth=2, markersize=8)
        ax1.plot(depths, sigma_yy, 's-', label='σ_yy', linewidth=2, markersize=8)
        ax1.set_xlabel('Depth (μm)', fontsize=12)
        ax1.set_ylabel('Stress (MPa)', fontsize=12)
        ax1.set_title('Residual Stress Profile vs Depth', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Von Mises stress vs depth
        ax2.plot(depths, von_mises, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Depth (μm)', fontsize=12)
        ax2.set_ylabel('Von Mises Stress (MPa)', fontsize=12)
        ax2.set_title('Von Mises Stress vs Depth', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'residual_stress_profile.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_crack_statistics(self):
        """Plot crack initiation and propagation statistics"""
        data = self.load_json_data(self.data_path / 'crack_analysis/experimental/sem_crack_analysis.json')
        cracks = data['micro_crack_observations']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # Crack length distribution
        lengths = [c['characteristics']['length_um'] for c in cracks]
        ax1.hist(lengths, bins=15, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Crack Length (μm)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Crack Length Distribution', fontsize=14, fontweight='bold')
        ax1.axvline(np.mean(lengths), color='r', linestyle='--', label=f'Mean: {np.mean(lengths):.1f} μm')
        ax1.legend()
        
        # Crack type pie chart
        types = [c['characteristics']['type'] for c in cracks]
        type_counts = pd.Series(types).value_counts()
        ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Crack Type Distribution', fontsize=14, fontweight='bold')
        
        # Stress concentration factor vs crack length
        scf = [c['stress_concentration_factor'] for c in cracks]
        ax3.scatter(lengths, scf, s=100, alpha=0.6, edgecolors='black')
        ax3.set_xlabel('Crack Length (μm)', fontsize=12)
        ax3.set_ylabel('Stress Concentration Factor', fontsize=12)
        ax3.set_title('SCF vs Crack Length', fontsize=14, fontweight='bold')
        z = np.polyfit(lengths, scf, 1)
        p = np.poly1d(z)
        ax3.plot(lengths, p(lengths), "r--", alpha=0.8, label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.2f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Initiation site distribution
        sites = [c['initiation_site']['feature'] for c in cracks]
        site_counts = pd.Series(sites).value_counts()
        ax4.bar(site_counts.index, site_counts.values, edgecolor='black')
        ax4.set_xlabel('Initiation Site', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.set_title('Crack Initiation Sites', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'crack_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_collocation_points_3d(self):
        """3D visualization of collocation points with stress field"""
        data = self.load_json_data(self.data_path / 'simulation_output/collocation_points.json')
        points = data['collocation_points']
        
        fig = plt.figure(figsize=(16, 6))
        
        # Extract coordinates and stresses
        x = [p['coordinates_mm'][0] for p in points]
        y = [p['coordinates_mm'][1] for p in points]
        z = [p['coordinates_mm'][2] for p in points]
        von_mises = [p['von_mises_stress_MPa'] for p in points]
        
        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(x, y, z, c=von_mises, cmap='jet', s=100, alpha=0.6, edgecolors='black')
        ax1.set_xlabel('X (mm)', fontsize=11)
        ax1.set_ylabel('Y (mm)', fontsize=11)
        ax1.set_zlabel('Z (mm)', fontsize=11)
        ax1.set_title('Collocation Points Distribution', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='Von Mises Stress (MPa)', shrink=0.8)
        
        # 2D projection - XY plane
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(x, y, c=von_mises, cmap='jet', s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('X (mm)', fontsize=11)
        ax2.set_ylabel('Y (mm)', fontsize=11)
        ax2.set_title('XY Projection', fontsize=13, fontweight='bold')
        plt.colorbar(scatter2, ax=ax2, label='Von Mises (MPa)')
        
        # Stress vs location type
        ax3 = fig.add_subplot(133)
        location_types = [p['location_type'] for p in points]
        df = pd.DataFrame({'location': location_types, 'stress': von_mises})
        df.boxplot(column='stress', by='location', ax=ax3)
        ax3.set_xlabel('Location Type', fontsize=11)
        ax3.set_ylabel('Von Mises Stress (MPa)', fontsize=11)
        ax3.set_title('Stress by Location Type', fontsize=13, fontweight='bold')
        plt.sca(ax3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'collocation_points_3d.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_multiscale_comparison(self):
        """Compare data across different scales"""
        macro_data = self.load_json_data(self.data_path / 'multi_scale_data/macro_scale_cell_data.json')
        meso_data = self.load_json_data(self.data_path / 'multi_scale_data/meso_scale_microstructure.json')
        micro_data = self.load_json_data(self.data_path / 'multi_scale_data/micro_scale_grain_boundary.json')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # Temperature-dependent CTE
        cte_data = macro_data['bulk_material_properties']['thermal']['coefficient_thermal_expansion']['temperature_dependence']
        temps = [d['temp_C'] for d in cte_data]
        cte_values = [d['CTE_per_K'] * 1e6 for d in cte_data]  # Convert to ppm/K
        ax1.plot(temps, cte_values, 'o-', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Temperature (°C)', fontsize=12)
        ax1.set_ylabel('CTE (ppm/K)', fontsize=12)
        ax1.set_title('Thermal Expansion Coefficient', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Grain size distribution
        grain_dist = meso_data['grain_structure']['grain_size_distribution']
        x_grain = np.linspace(grain_dist['min_um'], grain_dist['max_um'], 100)
        y_grain = stats.lognorm.pdf(x_grain, s=0.3, scale=grain_dist['mean_um'])
        ax2.plot(x_grain, y_grain, linewidth=2, label='Log-normal fit')
        ax2.axvline(grain_dist['mean_um'], color='r', linestyle='--', label=f"Mean: {grain_dist['mean_um']} μm")
        ax2.fill_between(x_grain, y_grain, alpha=0.3)
        ax2.set_xlabel('Grain Size (μm)', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('Grain Size Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Pore size distribution
        pore_dist = meso_data['porosity_data']['pore_size_distribution']
        x_pore = np.linspace(pore_dist['min_um'], pore_dist['max_um'], 100)
        y_pore = stats.lognorm.pdf(x_pore, s=0.5, scale=pore_dist['mean_um'])
        ax3.plot(x_pore, y_pore, linewidth=2, color='green', label='Log-normal fit')
        ax3.axvline(pore_dist['mean_um'], color='r', linestyle='--', label=f"Mean: {pore_dist['mean_um']} μm")
        ax3.fill_between(x_pore, y_pore, alpha=0.3, color='green')
        ax3.set_xlabel('Pore Size (μm)', fontsize=12)
        ax3.set_ylabel('Probability Density', fontsize=12)
        ax3.set_title('Pore Size Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Grain boundary types
        gb_types = micro_data['grain_boundary_structure']['gb_types']
        labels = [gb['type'] for gb in gb_types]
        fractions = [gb['fraction'] for gb in gb_types]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax4.pie(fractions, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax4.set_title('Grain Boundary Type Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_path / 'multiscale_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        report = []
        report.append("=" * 80)
        report.append("FEM VALIDATION DATASET SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: 2025-10-08\n")
        
        # Load all data files
        xrd_data = self.load_json_data(self.data_path / 'residual_stress/experimental/xrd_surface_stress.json')
        crack_data = self.load_json_data(self.data_path / 'crack_analysis/experimental/sem_crack_analysis.json')
        collocation_data = self.load_json_data(self.data_path / 'simulation_output/collocation_points.json')
        macro_data = self.load_json_data(self.data_path / 'multi_scale_data/macro_scale_cell_data.json')
        
        # Residual Stress Summary
        report.append("\n1. RESIDUAL STRESS MEASUREMENTS")
        report.append("-" * 40)
        avg_stress = xrd_data['surface_residual_stress']['average_stress']
        report.append(f"  Average σ_xx: {avg_stress['sigma_xx_MPa']:.1f} MPa")
        report.append(f"  Average σ_yy: {avg_stress['sigma_yy_MPa']:.1f} MPa")
        report.append(f"  Standard Deviation: {avg_stress['standard_deviation_MPa']:.1f} MPa")
        
        # Crack Analysis Summary
        report.append("\n2. CRACK ANALYSIS")
        report.append("-" * 40)
        stats = crack_data['statistical_analysis']
        report.append(f"  Total Cracks Observed: {stats['total_cracks_observed']}")
        report.append(f"  Crack Density: {stats['crack_density_per_mm2']:.1f} per mm²")
        report.append(f"  Average Crack Length: {stats['average_crack_length_um']:.1f} ± {stats['std_dev_length_um']:.1f} μm")
        report.append(f"  Intergranular: {stats['intergranular_percentage']}%")
        report.append(f"  Transgranular: {stats['transgranular_percentage']}%")
        
        # Collocation Points Summary
        report.append("\n3. COLLOCATION POINTS")
        report.append("-" * 40)
        stress_stats = collocation_data['statistical_summary']['stress_statistics']
        report.append(f"  Total Points: {len(collocation_data['collocation_points'])}")
        report.append(f"  Mean Von Mises Stress: {stress_stats['mean_von_mises_MPa']:.1f} ± {stress_stats['std_von_mises_MPa']:.1f} MPa")
        report.append(f"  Max Von Mises Stress: {stress_stats['max_von_mises_MPa']:.1f} MPa")
        
        # Material Properties Summary
        report.append("\n4. MATERIAL PROPERTIES")
        report.append("-" * 40)
        mech = macro_data['bulk_material_properties']['mechanical']
        thermal = macro_data['bulk_material_properties']['thermal']
        report.append(f"  Young's Modulus: {mech['youngs_modulus_GPa']} GPa")
        report.append(f"  Poisson's Ratio: {mech['poissons_ratio']}")
        report.append(f"  CTE: {thermal['coefficient_thermal_expansion']['value_per_K']*1e6:.1f} ppm/K")
        report.append(f"  Fracture Toughness: {mech['fracture_toughness_MPa_sqrt_m']} MPa√m")
        
        # Save report
        report.append("\n" + "=" * 80)
        report_text = "\n".join(report)
        
        with open(self.data_path / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text

def main():
    """Main execution function"""
    print("Starting FEM Validation Dataset Visualization...")
    
    # Initialize visualizer
    viz = FEMDataVisualizer('/workspace/fem_validation_dataset')
    
    # Generate all plots
    print("\n1. Plotting residual stress profiles...")
    viz.plot_residual_stress_profile()
    
    print("2. Plotting crack statistics...")
    viz.plot_crack_statistics()
    
    print("3. Plotting collocation points in 3D...")
    viz.plot_collocation_points_3d()
    
    print("4. Plotting multi-scale comparison...")
    viz.plot_multiscale_comparison()
    
    print("5. Generating summary report...")
    viz.generate_summary_report()
    
    print("\nVisualization complete! All figures saved in 'figures' directory.")
    print("Summary report saved as 'summary_report.txt'")

if __name__ == "__main__":
    main()