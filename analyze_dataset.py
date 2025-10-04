#!/usr/bin/env python3
"""
Atomic-Scale Simulation Dataset Analysis and Visualization
Analyzes the generated dataset and creates visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """Analyze and visualize the atomic-scale simulation dataset"""
    
    def __init__(self, data_dir: str = 'atomic_simulation_dataset'):
        self.data_dir = data_dir
        self.dft_formation = None
        self.activation_barriers = None
        self.surface_energies = None
        self.gb_sliding = None
        self.disl_mobility = None
        self.ff_params = None
        
    def load_data(self):
        """Load all dataset files"""
        print("Loading dataset files...")
        
        # Load DFT data
        self.dft_formation = pd.read_csv(f'{self.data_dir}/dft_formation_energies.csv')
        self.activation_barriers = pd.read_csv(f'{self.data_dir}/activation_barriers.csv')
        self.surface_energies = pd.read_csv(f'{self.data_dir}/surface_energies.csv')
        
        # Load MD data
        self.gb_sliding = pd.read_csv(f'{self.data_dir}/md_data/grain_boundary_sliding.csv')
        self.disl_mobility = pd.read_csv(f'{self.data_dir}/md_data/dislocation_mobility.csv')
        self.ff_params = pd.read_csv(f'{self.data_dir}/md_data/force_field_parameters.csv')
        
        print("Data loaded successfully!")
        
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("Generating summary statistics...")
        
        summary = {
            'dft_formation_energies': {
                'count': len(self.dft_formation),
                'materials': self.dft_formation['material'].unique().tolist(),
                'defect_types': self.dft_formation['defect_type'].unique().tolist(),
                'energy_range_eV': [
                    float(self.dft_formation['formation_energy_eV'].min()),
                    float(self.dft_formation['formation_energy_eV'].max())
                ],
                'mean_energy_eV': float(self.dft_formation['formation_energy_eV'].mean()),
                'std_energy_eV': float(self.dft_formation['formation_energy_eV'].std())
            },
            'activation_barriers': {
                'count': len(self.activation_barriers),
                'mechanisms': self.activation_barriers['diffusion_mechanism'].unique().tolist(),
                'barrier_range_eV': [
                    float(self.activation_barriers['activation_barrier_eV'].min()),
                    float(self.activation_barriers['activation_barrier_eV'].max())
                ],
                'mean_barrier_eV': float(self.activation_barriers['activation_barrier_eV'].mean()),
                'std_barrier_eV': float(self.activation_barriers['activation_barrier_eV'].std())
            },
            'surface_energies': {
                'count': len(self.surface_energies),
                'miller_indices': self.surface_energies['miller_indices'].unique().tolist(),
                'energy_range_J_m2': [
                    float(self.surface_energies['surface_energy_J_m2'].min()),
                    float(self.surface_energies['surface_energy_J_m2'].max())
                ],
                'mean_energy_J_m2': float(self.surface_energies['surface_energy_J_m2'].mean()),
                'std_energy_J_m2': float(self.surface_energies['surface_energy_J_m2'].std())
            },
            'grain_boundary_sliding': {
                'count': len(self.gb_sliding),
                'gb_types': self.gb_sliding['gb_type'].unique().tolist(),
                'sliding_rate_range': [
                    float(self.gb_sliding['avg_sliding_rate_A_ps'].min()),
                    float(self.gb_sliding['avg_sliding_rate_A_ps'].max())
                ],
                'mean_sliding_rate': float(self.gb_sliding['avg_sliding_rate_A_ps'].mean())
            },
            'dislocation_mobility': {
                'count': len(self.disl_mobility),
                'dislocation_types': self.disl_mobility['dislocation_type'].unique().tolist(),
                'velocity_range': [
                    float(self.disl_mobility['avg_velocity_A_ps'].min()),
                    float(self.disl_mobility['avg_velocity_A_ps'].max())
                ],
                'mean_velocity': float(self.disl_mobility['avg_velocity_A_ps'].mean())
            }
        }
        
        # Save summary
        with open(f'{self.data_dir}/analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_dir = f'{self.data_dir}/figures'
        os.makedirs(fig_dir, exist_ok=True)
        
        # 1. Formation energies by material and defect type
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.dft_formation, x='material', y='formation_energy_eV', hue='defect_type')
        plt.title('Formation Energies by Material and Defect Type')
        plt.xlabel('Material')
        plt.ylabel('Formation Energy (eV)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/formation_energies_by_material.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Activation barriers vs temperature
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.activation_barriers['temperature_K'], 
                            self.activation_barriers['activation_barrier_eV'],
                            c=self.activation_barriers['applied_stress_MPa'],
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Applied Stress (MPa)')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Activation Barrier (eV)')
        plt.title('Activation Barriers vs Temperature (colored by stress)')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/activation_barriers_vs_temperature.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Surface energies by Miller indices
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.surface_energies, x='miller_indices', y='surface_energy_J_m2')
        plt.title('Surface Energy Distribution by Miller Indices')
        plt.xlabel('Miller Indices')
        plt.ylabel('Surface Energy (J/m²)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/surface_energies_by_miller.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Grain boundary sliding analysis
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.gb_sliding, x='applied_stress_MPa', y='avg_sliding_rate_A_ps', 
                       hue='material', style='gb_type')
        plt.xlabel('Applied Stress (MPa)')
        plt.ylabel('Average Sliding Rate (Å/ps)')
        plt.title('GB Sliding Rate vs Applied Stress')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.gb_sliding, x='gb_type', y='stress_exponent')
        plt.xlabel('Grain Boundary Type')
        plt.ylabel('Stress Exponent')
        plt.title('Stress Exponent by GB Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/grain_boundary_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Dislocation mobility analysis
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.disl_mobility, x='temperature_K', y='avg_velocity_A_ps',
                       hue='material', style='dislocation_type')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Average Velocity (Å/ps)')
        plt.title('Dislocation Velocity vs Temperature')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.disl_mobility, x='dislocation_type', y='base_mobility_m2_Pa_s')
        plt.xlabel('Dislocation Type')
        plt.ylabel('Base Mobility (m²/Pa·s)')
        plt.title('Base Mobility by Dislocation Type')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/dislocation_mobility_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Correlation matrix for DFT data
        plt.figure(figsize=(10, 8))
        numeric_cols = self.dft_formation.select_dtypes(include=[np.number]).columns
        corr_matrix = self.dft_formation[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix - DFT Formation Energy Data')
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/dft_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Material property comparison
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different properties
        properties = [
            ('formation_energy_eV', 'Formation Energy (eV)', self.dft_formation),
            ('activation_barrier_eV', 'Activation Barrier (eV)', self.activation_barriers),
            ('surface_energy_J_m2', 'Surface Energy (J/m²)', self.surface_energies),
            ('avg_sliding_rate_A_ps', 'GB Sliding Rate (Å/ps)', self.gb_sliding)
        ]
        
        for i, (prop, label, data) in enumerate(properties, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(data=data, x='material', y=prop)
            plt.title(f'{label} by Material')
            plt.xlabel('Material')
            plt.ylabel(label)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/material_property_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {fig_dir}/")
    
    def perform_ml_analysis(self):
        """Perform machine learning analysis on the dataset"""
        print("Performing ML analysis...")
        
        # Prepare data for ML analysis
        # Combine DFT and MD data for comprehensive analysis
        
        # 1. Feature importance analysis for formation energy prediction
        dft_features = ['temperature_K', 'defect_concentration', 'supercell_size', 'cutoff_energy_eV']
        dft_categorical = ['material', 'defect_type', 'crystal_structure', 'exchange_correlation']
        
        # Encode categorical variables
        dft_encoded = pd.get_dummies(self.dft_formation[dft_categorical])
        dft_numeric = self.dft_formation[dft_features]
        X_dft = pd.concat([dft_numeric, dft_encoded], axis=1)
        y_dft = self.dft_formation['formation_energy_eV']
        
        # Train random forest for feature importance
        rf_dft = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_dft.fit(X_dft, y_dft)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_dft.columns,
            'importance': rf_dft.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 2. PCA analysis on activation barriers
        barrier_features = ['temperature_K', 'applied_stress_MPa', 'grain_size_nm', 
                          'migration_path_length_A', 'coordination_number', 'elastic_modulus_GPa']
        
        scaler = StandardScaler()
        barrier_scaled = scaler.fit_transform(self.activation_barriers[barrier_features])
        
        pca = PCA(n_components=3)
        barrier_pca = pca.fit_transform(barrier_scaled)
        
        # Save ML analysis results
        ml_results = {
            'feature_importance_dft': feature_importance.to_dict('records'),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'pca_components': pca.components_.tolist(),
            'rf_score_dft': float(rf_dft.score(X_dft, y_dft))
        }
        
        with open(f'{self.data_dir}/ml_analysis_results.json', 'w') as f:
            json.dump(ml_results, f, indent=2)
        
        # Create ML visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Features for Formation Energy Prediction')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        plt.scatter(barrier_pca[:, 0], barrier_pca[:, 1], 
                   c=self.activation_barriers['activation_barrier_eV'], cmap='viridis')
        plt.colorbar(label='Activation Barrier (eV)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Activation Barrier Data')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/figures/ml_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return ml_results
    
    def generate_data_quality_report(self):
        """Generate a comprehensive data quality report"""
        print("Generating data quality report...")
        
        datasets = {
            'DFT Formation Energies': self.dft_formation,
            'Activation Barriers': self.activation_barriers,
            'Surface Energies': self.surface_energies,
            'Grain Boundary Sliding': self.gb_sliding,
            'Dislocation Mobility': self.disl_mobility,
            'Force Field Parameters': self.ff_params
        }
        
        quality_report = {}
        
        for name, df in datasets.items():
            report = {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'missing_values': int(df.isnull().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum()),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024**2)
            }
            
            # Check for outliers in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = int(outliers)
            
            report['outlier_counts'] = outlier_counts
            report['total_outliers'] = int(sum(outlier_counts.values()))
            
            quality_report[name] = report
        
        # Save quality report
        with open(f'{self.data_dir}/data_quality_report.json', 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        return quality_report

def main():
    """Main analysis function"""
    print("Starting comprehensive dataset analysis...")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Generate summary statistics
    summary = analyzer.generate_summary_statistics()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Perform ML analysis
    ml_results = analyzer.perform_ml_analysis()
    
    # Generate data quality report
    quality_report = analyzer.generate_data_quality_report()
    
    print("\nAnalysis complete!")
    print(f"Total samples across all datasets: {sum([summary[key]['count'] for key in summary.keys()])}")
    print("Generated files:")
    print("- analysis_summary.json")
    print("- ml_analysis_results.json")
    print("- data_quality_report.json")
    print("- figures/ (directory with visualizations)")

if __name__ == "__main__":
    main()