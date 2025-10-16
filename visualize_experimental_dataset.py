"""
Visualization Tools for SOFC Experimental Validation Dataset
=============================================================

This script provides comprehensive visualization and analysis tools
for the experimental validation dataset.
"""

import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExperimentalDatasetVisualizer:
    """
    Comprehensive visualization for experimental validation dataset.
    """
    
    def __init__(self, dataset_dir='experimental_validation_dataset'):
        """
        Initialize visualizer with dataset directory.
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = self.dataset_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading dataset from: {self.dataset_dir}")
        self.load_dataset()
        
    def load_dataset(self):
        """
        Load all dataset components.
        """
        # Load CSV files
        self.fab_params = pd.read_csv(self.dataset_dir / 'fabrication_parameters.csv')
        self.curv_stress = pd.read_csv(self.dataset_dir / 'curvature_stress_measurements.csv')
        self.xrd_stress = pd.read_csv(self.dataset_dir / 'xrd_stress_measurements.csv')
        self.raman_stress = pd.read_csv(self.dataset_dir / 'raman_stress_measurements.csv')
        
        # Load metadata
        with open(self.dataset_dir / 'dataset_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"  Loaded {len(self.fab_params)} samples")
        print(f"  Loaded {len(self.curv_stress)} curvature measurements")
        print(f"  Loaded {len(self.xrd_stress)} XRD points")
        print(f"  Loaded {len(self.raman_stress)} Raman points")
        
    def plot_parameter_space_coverage(self):
        """
        Visualize parameter space coverage (Latin Hypercube Sampling).
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fabrication Parameter Space Coverage', fontsize=16, fontweight='bold')
        
        # Key parameter pairs
        param_pairs = [
            ('anode_thickness_um', 'electrolyte_thickness_um', 'Anode vs Electrolyte Thickness'),
            ('electrolyte_thickness_um', 'cathode_thickness_um', 'Electrolyte vs Cathode Thickness'),
            ('anode_sinter_temp_C', 'anode_sinter_time_h', 'Anode Sintering Conditions'),
            ('electrolyte_sinter_temp_C', 'electrolyte_sinter_time_h', 'Electrolyte Sintering Conditions'),
            ('cathode_sinter_temp_C', 'cathode_sinter_time_h', 'Cathode Sintering Conditions'),
            ('total_thickness_um', 'cooling_rate_C_per_min', 'Total Thickness vs Cooling Rate')
        ]
        
        for idx, (param_x, param_y, title) in enumerate(param_pairs):
            ax = axes[idx // 3, idx % 3]
            
            scatter = ax.scatter(
                self.fab_params[param_x],
                self.fab_params[param_y],
                c=range(len(self.fab_params)),
                cmap='viridis',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            ax.set_xlabel(param_x.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel(param_y.replace('_', ' ').title(), fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'parameter_space_coverage.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def plot_warp_measurements_3d(self, n_samples=6):
        """
        Plot 3D surface maps of warp measurements for selected samples.
        """
        # Load warp data
        warp_file = self.dataset_dir / 'warp_measurements_3d.h5'
        
        with h5py.File(warp_file, 'r') as f:
            sample_ids = list(f.keys())[:n_samples]
            
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('3D Warp Measurements (White Light Interferometry)', 
                        fontsize=16, fontweight='bold')
            
            for idx, sample_id in enumerate(sample_ids):
                ax = fig.add_subplot(2, 3, idx+1, projection='3d')
                
                grp = f[sample_id]
                X = grp['X'][:]
                Y = grp['Y'][:]
                Z = grp['Z'][:]
                
                # Convert to mm for plotting
                X_mm = X * 1000
                Y_mm = Y * 1000
                
                # Surface plot
                surf = ax.plot_surface(
                    X_mm, Y_mm, Z,
                    cmap='RdYlBu_r',
                    linewidth=0,
                    antialiased=True,
                    alpha=0.9
                )
                
                ax.set_xlabel('X (mm)', fontsize=9)
                ax.set_ylabel('Y (mm)', fontsize=9)
                ax.set_zlabel('Warp (μm)', fontsize=9)
                ax.set_title(f'{sample_id}\nMax: {grp.attrs["max_warp_um"]:.2f} μm', 
                           fontsize=10, fontweight='bold')
                
                # Add colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Set viewing angle
                ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        output_file = self.output_dir / 'warp_measurements_3d.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def plot_warp_contours(self, n_samples=6):
        """
        Plot contour maps of warp measurements.
        """
        warp_file = self.dataset_dir / 'warp_measurements_3d.h5'
        
        with h5py.File(warp_file, 'r') as f:
            sample_ids = list(f.keys())[:n_samples]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Warp Contour Maps', fontsize=16, fontweight='bold')
            
            for idx, sample_id in enumerate(sample_ids):
                ax = axes[idx // 3, idx % 3]
                
                grp = f[sample_id]
                X = grp['X'][:] * 1000  # Convert to mm
                Y = grp['Y'][:] * 1000
                Z = grp['Z'][:]
                
                # Contour plot
                contour = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
                ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.3)
                
                ax.set_xlabel('X (mm)', fontsize=9)
                ax.set_ylabel('Y (mm)', fontsize=9)
                ax.set_title(f'{sample_id}\nRMS: {grp.attrs["rms_warp_um"]:.2f} μm', 
                           fontsize=10, fontweight='bold')
                ax.set_aspect('equal')
                
                # Colorbar
                plt.colorbar(contour, ax=ax, label='Warp (μm)')
        
        plt.tight_layout()
        output_file = self.output_dir / 'warp_contour_maps.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def plot_stress_measurements_comparison(self):
        """
        Compare stress measurements from different techniques.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Residual Stress Measurements - Multi-Technique Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 1. Curvature-based stress (all layers)
        ax = axes[0, 0]
        layers = ['anode', 'electrolyte', 'cathode']
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        
        for i, layer in enumerate(layers):
            stress_col = f'{layer}_stress_GPa'
            ax.hist(self.curv_stress[stress_col], bins=15, alpha=0.7, 
                   color=colors[i], label=layer.capitalize(), edgecolor='black')
        
        ax.set_xlabel('Stress (GPa)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Curvature-Based Stress (Stoney Method)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. XRD stress spatial distribution
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.xrd_stress['x_position_mm'],
            self.xrd_stress['y_position_mm'],
            c=self.xrd_stress['stress_xx_GPa'],
            cmap='RdYlBu_r',
            s=80,
            edgecolors='black',
            linewidth=0.5
        )
        ax.set_xlabel('X Position (mm)', fontsize=11)
        ax.set_ylabel('Y Position (mm)', fontsize=11)
        ax.set_title('XRD Stress Measurements (Surface)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Stress σxx (GPa)')
        
        # 3. Raman stress distribution
        ax = axes[1, 0]
        ax.hexbin(
            self.raman_stress['x_position_mm'],
            self.raman_stress['y_position_mm'],
            C=self.raman_stress['stress_estimate_GPa'],
            gridsize=15,
            cmap='viridis',
            edgecolors='black',
            linewidths=0.3
        )
        ax.set_xlabel('X Position (mm)', fontsize=11)
        ax.set_ylabel('Y Position (mm)', fontsize=11)
        ax.set_title('Raman Stress Estimates (Density Map)', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        # 4. Stress vs thickness ratio
        ax = axes[1, 1]
        
        # Merge with fabrication parameters
        merged = self.curv_stress.merge(self.fab_params, on='sample_id')
        
        ax.scatter(merged['electrolyte_anode_ratio'], 
                  merged['electrolyte_stress_GPa'],
                  c=merged['electrolyte_sinter_temp_C'],
                  cmap='plasma',
                  s=100,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=0.5)
        
        ax.set_xlabel('Electrolyte/Anode Thickness Ratio', fontsize=11)
        ax.set_ylabel('Electrolyte Stress (GPa)', fontsize=11)
        ax.set_title('Stress vs Thickness Ratio (Colored by Sinter Temp)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'stress_measurements_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def plot_layer_removal_profiles(self):
        """
        Plot through-thickness stress profiles from layer removal technique.
        """
        layer_file = self.dataset_dir / 'layer_removal_stress_profiles.h5'
        
        if not layer_file.exists():
            print("  Layer removal data not found, skipping...")
            return
        
        with h5py.File(layer_file, 'r') as f:
            sample_ids = list(f.keys())
            
            fig, axes = plt.subplots(3, 5, figsize=(20, 12))
            fig.suptitle('Through-Thickness Stress Profiles (Layer Removal Method)', 
                        fontsize=16, fontweight='bold')
            
            for idx, sample_id in enumerate(sample_ids):
                if idx >= 15:
                    break
                    
                ax = axes[idx // 5, idx % 5]
                
                grp = f[sample_id]
                z_pos = grp['z_positions_um'][:]
                stress = grp['stress_profile_GPa'][:]
                
                # Plot stress profile
                ax.plot(stress, z_pos, 'b-', linewidth=2, label='Measured')
                ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
                
                # Shade uncertainty
                uncertainty = grp.attrs['uncertainty_GPa']
                ax.fill_betweenx(z_pos, stress - uncertainty, stress + uncertainty, 
                                alpha=0.3, color='blue')
                
                ax.set_xlabel('Stress (GPa)', fontsize=8)
                ax.set_ylabel('Depth (μm)', fontsize=8)
                ax.set_title(sample_id, fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
        
        # Hide empty subplots
        for idx in range(len(sample_ids), 15):
            axes[idx // 5, idx % 5].axis('off')
        
        plt.tight_layout()
        output_file = self.output_dir / 'layer_removal_stress_profiles.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def plot_measurement_uncertainty_analysis(self):
        """
        Analyze and visualize measurement uncertainties across techniques.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Measurement Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # 1. Uncertainty by technique
        ax = axes[0, 0]
        
        techniques = ['Curvature', 'XRD', 'Raman']
        uncertainties = [
            self.curv_stress['uncertainty_percent'].mean(),
            (self.xrd_stress['uncertainty_GPa'] / self.xrd_stress['stress_xx_GPa'].abs()).mean() * 100,
            (self.raman_stress['uncertainty_GPa'] / self.raman_stress['stress_estimate_GPa'].abs()).mean() * 100
        ]
        colors = ['#3498DB', '#E74C3C', '#2ECC71']
        
        bars = ax.bar(techniques, uncertainties, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Average Uncertainty (%)', fontsize=11)
        ax.set_title('Measurement Uncertainty by Technique', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars, uncertainties):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. XRD uncertainty vs position
        ax = axes[0, 1]
        
        # Calculate distance from center
        center_x = self.xrd_stress['x_position_mm'].mean()
        center_y = self.xrd_stress['y_position_mm'].mean()
        self.xrd_stress['dist_from_center'] = np.sqrt(
            (self.xrd_stress['x_position_mm'] - center_x)**2 + 
            (self.xrd_stress['y_position_mm'] - center_y)**2
        )
        
        scatter = ax.scatter(
            self.xrd_stress['dist_from_center'],
            self.xrd_stress['uncertainty_GPa'],
            c=self.xrd_stress['stress_xx_GPa'],
            cmap='viridis',
            s=60,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        ax.set_xlabel('Distance from Center (mm)', fontsize=11)
        ax.set_ylabel('XRD Uncertainty (GPa)', fontsize=11)
        ax.set_title('XRD Uncertainty vs Measurement Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Stress (GPa)')
        
        # 3. Stress magnitude vs uncertainty
        ax = axes[1, 0]
        
        ax.scatter(self.xrd_stress['stress_xx_GPa'].abs(), 
                  self.xrd_stress['uncertainty_GPa'],
                  alpha=0.6, s=50, color='#E74C3C', label='XRD')
        ax.scatter(self.raman_stress['stress_estimate_GPa'].abs(),
                  self.raman_stress['uncertainty_GPa'],
                  alpha=0.6, s=50, color='#2ECC71', label='Raman')
        
        ax.set_xlabel('|Stress| (GPa)', fontsize=11)
        ax.set_ylabel('Absolute Uncertainty (GPa)', fontsize=11)
        ax.set_title('Stress Magnitude vs Measurement Uncertainty', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Measurement time analysis
        ax = axes[1, 1]
        
        # XRD measurement times
        ax.hist(self.xrd_stress['measurement_time_min'], bins=15, 
               alpha=0.7, color='#E74C3C', label='XRD', edgecolor='black')
        ax.hist(self.raman_stress['integration_time_s'] / 60, bins=15,
               alpha=0.7, color='#2ECC71', label='Raman', edgecolor='black')
        
        ax.set_xlabel('Measurement Time (minutes)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Measurement Time Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'measurement_uncertainty_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def plot_correlation_matrix(self):
        """
        Plot correlation matrix between fabrication parameters and stress.
        """
        # Merge curvature stress with fabrication parameters
        merged = self.curv_stress.merge(self.fab_params, on='sample_id')
        
        # Select key columns
        cols = [
            'anode_thickness_um',
            'electrolyte_thickness_um',
            'cathode_thickness_um',
            'anode_sinter_temp_C',
            'electrolyte_sinter_temp_C',
            'cathode_sinter_temp_C',
            'cooling_rate_C_per_min',
            'anode_stress_GPa',
            'electrolyte_stress_GPa',
            'cathode_stress_GPa'
        ]
        
        corr_matrix = merged[cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        ax.set_title('Correlation Matrix: Fabrication Parameters vs Residual Stress',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = self.output_dir / 'correlation_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
        
    def generate_all_visualizations(self):
        """
        Generate all visualization plots.
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        print("\n1. Parameter space coverage...")
        self.plot_parameter_space_coverage()
        
        print("\n2. 3D warp measurements...")
        self.plot_warp_measurements_3d()
        
        print("\n3. Warp contour maps...")
        self.plot_warp_contours()
        
        print("\n4. Stress measurements comparison...")
        self.plot_stress_measurements_comparison()
        
        print("\n5. Layer removal stress profiles...")
        self.plot_layer_removal_profiles()
        
        print("\n6. Measurement uncertainty analysis...")
        self.plot_measurement_uncertainty_analysis()
        
        print("\n7. Correlation matrix...")
        self.plot_correlation_matrix()
        
        print("\n" + "="*80)
        print(f"ALL VISUALIZATIONS SAVED TO: {self.output_dir}")
        print("="*80)


def main():
    """
    Main execution.
    """
    visualizer = ExperimentalDatasetVisualizer('experimental_validation_dataset')
    visualizer.generate_all_visualizations()
    
    print("\n✓ Visualization complete!")
    print("✓ Use these plots for:")
    print("  - Dataset quality assessment")
    print("  - Publication figures")
    print("  - Presentation materials")
    print("  - Model validation reports")


if __name__ == '__main__':
    main()
