#!/usr/bin/env python3
"""
Residual Analysis Script for FEM Model Validation
Performs comprehensive residual analysis between experimental and simulation data
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import seaborn as sns

class ResidualAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.results_path = self.data_path / 'residual_analysis_results'
        self.results_path.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load experimental and simulation data"""
        # Load experimental stress data
        with open(self.data_path / 'residual_stress/experimental/xrd_surface_stress.json', 'r') as f:
            self.xrd_data = json.load(f)
        
        with open(self.data_path / 'residual_stress/experimental/synchrotron_xrd.json', 'r') as f:
            self.synchrotron_data = json.load(f)
        
        # Load simulation data
        self.fem_data = pd.read_csv(self.data_path / 'simulation_output/fem_full_field_data.csv')
        
        with open(self.data_path / 'simulation_output/collocation_points.json', 'r') as f:
            self.collocation_data = json.load(f)
    
    def calculate_residuals(self):
        """Calculate residuals between experimental and simulation data"""
        residuals = []
        
        # Surface stress residuals (XRD vs FEM)
        xrd_measurements = self.xrd_data['surface_residual_stress']['measurement_locations']
        
        for xrd_point in xrd_measurements:
            # Find nearest FEM point
            x_xrd, y_xrd = xrd_point['x_mm'], xrd_point['y_mm']
            
            # Calculate distances to all FEM nodes
            distances = np.sqrt((self.fem_data['X_mm'] - x_xrd)**2 + 
                              (self.fem_data['Y_mm'] - y_xrd)**2)
            
            # Get nearest node
            nearest_idx = distances.idxmin()
            fem_point = self.fem_data.iloc[nearest_idx]
            
            # Calculate residuals
            residual = {
                'location': xrd_point['location_id'],
                'x_mm': x_xrd,
                'y_mm': y_xrd,
                'experimental_sigma_xx': xrd_point['sigma_xx_MPa'],
                'experimental_sigma_yy': xrd_point['sigma_yy_MPa'],
                'simulated_sigma_xx': fem_point['Stress_XX_MPa'],
                'simulated_sigma_yy': fem_point['Stress_YY_MPa'],
                'residual_sigma_xx': xrd_point['sigma_xx_MPa'] - fem_point['Stress_XX_MPa'],
                'residual_sigma_yy': xrd_point['sigma_yy_MPa'] - fem_point['Stress_YY_MPa'],
                'distance_to_fem_node': distances[nearest_idx]
            }
            residuals.append(residual)
        
        self.residuals_df = pd.DataFrame(residuals)
        return self.residuals_df
    
    def statistical_analysis(self):
        """Perform statistical analysis on residuals"""
        results = {}
        
        # Basic statistics
        results['mean_residual_xx'] = self.residuals_df['residual_sigma_xx'].mean()
        results['std_residual_xx'] = self.residuals_df['residual_sigma_xx'].std()
        results['mean_residual_yy'] = self.residuals_df['residual_sigma_yy'].mean()
        results['std_residual_yy'] = self.residuals_df['residual_sigma_yy'].std()
        
        # Error metrics
        results['rmse_xx'] = np.sqrt(mean_squared_error(
            self.residuals_df['experimental_sigma_xx'],
            self.residuals_df['simulated_sigma_xx']
        ))
        results['rmse_yy'] = np.sqrt(mean_squared_error(
            self.residuals_df['experimental_sigma_yy'],
            self.residuals_df['simulated_sigma_yy']
        ))
        
        results['mae_xx'] = mean_absolute_error(
            self.residuals_df['experimental_sigma_xx'],
            self.residuals_df['simulated_sigma_xx']
        )
        results['mae_yy'] = mean_absolute_error(
            self.residuals_df['experimental_sigma_yy'],
            self.residuals_df['simulated_sigma_yy']
        )
        
        # R-squared
        results['r2_xx'] = r2_score(
            self.residuals_df['experimental_sigma_xx'],
            self.residuals_df['simulated_sigma_xx']
        )
        results['r2_yy'] = r2_score(
            self.residuals_df['experimental_sigma_yy'],
            self.residuals_df['simulated_sigma_yy']
        )
        
        # Normality test
        _, results['normality_p_value_xx'] = stats.shapiro(self.residuals_df['residual_sigma_xx'])
        _, results['normality_p_value_yy'] = stats.shapiro(self.residuals_df['residual_sigma_yy'])
        
        return results
    
    def gaussian_process_surrogate(self):
        """Build Gaussian Process surrogate model for residual prediction"""
        # Prepare data from collocation points
        points = self.collocation_data['collocation_points']
        
        # Extract features and targets
        X = np.array([[p['coordinates_mm'][0], p['coordinates_mm'][1], p['coordinates_mm'][2]] 
                     for p in points])
        y_vm = np.array([p['von_mises_stress_MPa'] for p in points])
        
        # Define kernel
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        
        # Create and fit GP model
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        self.gp_model.fit(X, y_vm)
        
        # Make predictions on a grid
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        z_mean = X[:, 2].mean()
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                             np.linspace(y_min, y_max, 20))
        
        X_pred = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z_mean)]
        
        y_pred, y_std = self.gp_model.predict(X_pred, return_std=True)
        
        return {
            'X_train': X,
            'y_train': y_vm,
            'X_pred': X_pred,
            'y_pred': y_pred.reshape(xx.shape),
            'y_std': y_std.reshape(xx.shape),
            'xx': xx,
            'yy': yy,
            'score': self.gp_model.score(X, y_vm)
        }
    
    def identify_critical_regions(self):
        """Identify regions with high residuals for model refinement"""
        threshold = 2 * self.residuals_df['residual_sigma_xx'].std()
        
        critical_regions = []
        for _, row in self.residuals_df.iterrows():
            if abs(row['residual_sigma_xx']) > threshold or abs(row['residual_sigma_yy']) > threshold:
                critical_regions.append({
                    'location': row['location'],
                    'coordinates': (row['x_mm'], row['y_mm']),
                    'residual_magnitude': np.sqrt(row['residual_sigma_xx']**2 + row['residual_sigma_yy']**2),
                    'recommendation': 'Refine mesh or adjust material properties'
                })
        
        return critical_regions
    
    def plot_residual_analysis(self):
        """Generate comprehensive residual analysis plots"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Experimental vs Simulated scatter plot
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.residuals_df['experimental_sigma_xx'], 
                   self.residuals_df['simulated_sigma_xx'], 
                   alpha=0.7, s=100, edgecolors='black')
        
        # Add perfect prediction line
        min_val = min(self.residuals_df['experimental_sigma_xx'].min(), 
                     self.residuals_df['simulated_sigma_xx'].min())
        max_val = max(self.residuals_df['experimental_sigma_xx'].max(), 
                     self.residuals_df['simulated_sigma_xx'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax1.set_xlabel('Experimental σ_xx (MPa)', fontsize=11)
        ax1.set_ylabel('Simulated σ_xx (MPa)', fontsize=11)
        ax1.set_title('Experimental vs Simulated Stress', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residual histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(self.residuals_df['residual_sigma_xx'], bins=15, alpha=0.7, 
                edgecolor='black', label='σ_xx residuals')
        ax2.hist(self.residuals_df['residual_sigma_yy'], bins=15, alpha=0.7, 
                edgecolor='black', label='σ_yy residuals')
        ax2.set_xlabel('Residual (MPa)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot for normality
        ax3 = plt.subplot(2, 3, 3)
        stats.probplot(self.residuals_df['residual_sigma_xx'], dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (σ_xx residuals)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Spatial distribution of residuals
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(self.residuals_df['x_mm'], 
                            self.residuals_df['y_mm'],
                            c=self.residuals_df['residual_sigma_xx'],
                            cmap='RdBu_r', s=200, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=ax4, label='Residual σ_xx (MPa)')
        ax4.set_xlabel('X (mm)', fontsize=11)
        ax4.set_ylabel('Y (mm)', fontsize=11)
        ax4.set_title('Spatial Distribution of Residuals', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. GP surrogate model prediction
        gp_results = self.gaussian_process_surrogate()
        ax5 = plt.subplot(2, 3, 5)
        contour = ax5.contourf(gp_results['xx'], gp_results['yy'], 
                              gp_results['y_pred'], levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax5, label='Predicted Von Mises (MPa)')
        ax5.scatter(gp_results['X_train'][:, 0], gp_results['X_train'][:, 1], 
                   c='red', s=50, edgecolors='white', label='Training Points')
        ax5.set_xlabel('X (mm)', fontsize=11)
        ax5.set_ylabel('Y (mm)', fontsize=11)
        ax5.set_title(f'GP Surrogate Model (R²={gp_results["score"]:.3f})', 
                     fontsize=12, fontweight='bold')
        ax5.legend()
        
        # 6. Uncertainty quantification
        ax6 = plt.subplot(2, 3, 6)
        contour_std = ax6.contourf(gp_results['xx'], gp_results['yy'], 
                                   gp_results['y_std'], levels=20, cmap='plasma')
        plt.colorbar(contour_std, ax=ax6, label='Prediction Std Dev (MPa)')
        ax6.set_xlabel('X (mm)', fontsize=11)
        ax6.set_ylabel('Y (mm)', fontsize=11)
        ax6.set_title('GP Model Uncertainty', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_residual_report(self):
        """Generate comprehensive residual analysis report"""
        stats_results = self.statistical_analysis()
        critical_regions = self.identify_critical_regions()
        
        report = []
        report.append("=" * 80)
        report.append("RESIDUAL ANALYSIS REPORT")
        report.append("=" * 80)
        
        report.append("\n1. ERROR METRICS")
        report.append("-" * 40)
        report.append(f"  RMSE (σ_xx): {stats_results['rmse_xx']:.2f} MPa")
        report.append(f"  RMSE (σ_yy): {stats_results['rmse_yy']:.2f} MPa")
        report.append(f"  MAE (σ_xx): {stats_results['mae_xx']:.2f} MPa")
        report.append(f"  MAE (σ_yy): {stats_results['mae_yy']:.2f} MPa")
        report.append(f"  R² (σ_xx): {stats_results['r2_xx']:.4f}")
        report.append(f"  R² (σ_yy): {stats_results['r2_yy']:.4f}")
        
        report.append("\n2. RESIDUAL STATISTICS")
        report.append("-" * 40)
        report.append(f"  Mean Residual (σ_xx): {stats_results['mean_residual_xx']:.2f} MPa")
        report.append(f"  Std Dev (σ_xx): {stats_results['std_residual_xx']:.2f} MPa")
        report.append(f"  Mean Residual (σ_yy): {stats_results['mean_residual_yy']:.2f} MPa")
        report.append(f"  Std Dev (σ_yy): {stats_results['std_residual_yy']:.2f} MPa")
        
        report.append("\n3. NORMALITY TEST")
        report.append("-" * 40)
        report.append(f"  Shapiro-Wilk p-value (σ_xx): {stats_results['normality_p_value_xx']:.4f}")
        report.append(f"  Shapiro-Wilk p-value (σ_yy): {stats_results['normality_p_value_yy']:.4f}")
        
        if stats_results['normality_p_value_xx'] > 0.05:
            report.append("  → σ_xx residuals appear normally distributed")
        else:
            report.append("  → σ_xx residuals do NOT appear normally distributed")
        
        report.append("\n4. CRITICAL REGIONS FOR REFINEMENT")
        report.append("-" * 40)
        if critical_regions:
            for region in critical_regions:
                report.append(f"  Location: {region['location']}")
                report.append(f"    Coordinates: {region['coordinates']}")
                report.append(f"    Residual Magnitude: {region['residual_magnitude']:.2f} MPa")
                report.append(f"    Recommendation: {region['recommendation']}")
        else:
            report.append("  No critical regions identified (all residuals within 2σ)")
        
        report.append("\n5. MODEL VALIDATION SUMMARY")
        report.append("-" * 40)
        
        # Overall assessment
        if stats_results['r2_xx'] > 0.9 and stats_results['r2_yy'] > 0.9:
            report.append("  ✓ EXCELLENT: Model shows excellent agreement with experimental data")
        elif stats_results['r2_xx'] > 0.8 and stats_results['r2_yy'] > 0.8:
            report.append("  ✓ GOOD: Model shows good agreement with experimental data")
        elif stats_results['r2_xx'] > 0.7 and stats_results['r2_yy'] > 0.7:
            report.append("  ⚠ FAIR: Model shows fair agreement, consider refinement")
        else:
            report.append("  ✗ POOR: Model requires significant refinement")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(self.results_path / 'residual_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        # Save residuals to CSV
        self.residuals_df.to_csv(self.results_path / 'residuals.csv', index=False)
        
        # Save statistics to JSON
        with open(self.results_path / 'statistics.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        print(report_text)
        return report_text

def main():
    """Main execution function"""
    print("Starting Residual Analysis...")
    
    analyzer = ResidualAnalyzer('/workspace/fem_validation_dataset')
    
    print("\n1. Loading experimental and simulation data...")
    analyzer.load_data()
    
    print("2. Calculating residuals...")
    residuals = analyzer.calculate_residuals()
    print(f"   Calculated residuals for {len(residuals)} points")
    
    print("3. Performing statistical analysis...")
    stats = analyzer.statistical_analysis()
    print(f"   RMSE (σ_xx): {stats['rmse_xx']:.2f} MPa")
    print(f"   R² (σ_xx): {stats['r2_xx']:.4f}")
    
    print("4. Building Gaussian Process surrogate model...")
    gp_results = analyzer.gaussian_process_surrogate()
    print(f"   GP Model R²: {gp_results['score']:.4f}")
    
    print("5. Identifying critical regions...")
    critical = analyzer.identify_critical_regions()
    print(f"   Found {len(critical)} critical regions requiring refinement")
    
    print("6. Generating plots...")
    analyzer.plot_residual_analysis()
    
    print("7. Generating residual analysis report...")
    analyzer.generate_residual_report()
    
    print("\nResidual analysis complete!")
    print("Results saved in 'residual_analysis_results' directory")

if __name__ == "__main__":
    main()