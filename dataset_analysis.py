"""
Dataset Analysis and Visualization Tools for Welding Inverse Design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class WeldingDatasetAnalyzer:
    """
    Comprehensive analysis tools for welding dataset
    """
    
    def __init__(self, dataset_path: str = None):
        if dataset_path:
            self.load_dataset(dataset_path)
        else:
            self.dataset = None
            
    def load_dataset(self, path: str):
        """Load dataset from file"""
        if path.endswith('.csv'):
            self.dataset = pd.read_csv(path)
        elif path.endswith('.parquet'):
            self.dataset = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
        print(f"Dataset loaded: {len(self.dataset)} samples, {len(self.dataset.columns)} features")
        
    def generate_analysis_report(self, output_path: str = 'analysis_report'):
        """Generate comprehensive analysis report"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Create figure for subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Data Distribution Overview
        ax1 = plt.subplot(5, 3, 1)
        tier_counts = self.dataset['data_tier'].value_counts()
        ax1.bar(tier_counts.index, tier_counts.values)
        ax1.set_title('Data Distribution by Tier')
        ax1.set_xlabel('Data Tier')
        ax1.set_ylabel('Count')
        
        # 2. Material Combination Distribution
        ax2 = plt.subplot(5, 3, 2)
        material_counts = self.dataset['material_combination'].value_counts()
        ax2.pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
        ax2.set_title('Material Combination Distribution')
        
        # 3. Quality Score Distribution
        ax3 = plt.subplot(5, 3, 3)
        ax3.hist(self.dataset['quality_score'], bins=50, edgecolor='black')
        ax3.set_title('Quality Score Distribution')
        ax3.set_xlabel('Quality Score')
        ax3.set_ylabel('Frequency')
        
        # 4. Heat Input vs Tensile Strength
        ax4 = plt.subplot(5, 3, 4)
        scatter = ax4.scatter(self.dataset['heat_input'], 
                            self.dataset['tensile_strength'],
                            c=self.dataset['quality_score'],
                            cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Heat Input (kJ/mm)')
        ax4.set_ylabel('Tensile Strength (N)')
        ax4.set_title('Heat Input vs Tensile Strength')
        plt.colorbar(scatter, ax=ax4, label='Quality Score')
        
        # 5. Power Density vs Penetration Depth
        ax5 = plt.subplot(5, 3, 5)
        ax5.scatter(self.dataset['power_density'], 
                   self.dataset['penetration_depth'],
                   alpha=0.5)
        ax5.set_xlabel('Power Density (W/mm²)')
        ax5.set_ylabel('Penetration Depth (mm)')
        ax5.set_title('Power Density vs Penetration')
        
        # 6. IMC Growth Analysis (for dissimilar metals)
        ax6 = plt.subplot(5, 3, 6)
        dissimilar = self.dataset[self.dataset['imc_thickness_post_aging'] > 0]
        if len(dissimilar) > 0:
            ax6.scatter(dissimilar['imc_thickness_post_aging'], 
                       dissimilar['cycles_to_failure'],
                       alpha=0.5)
            ax6.set_xlabel('IMC Thickness Post-Aging (μm)')
            ax6.set_ylabel('Cycles to Failure')
            ax6.set_title('IMC Growth vs Thermal Cycling Performance')
        
        # 7. Defect Impact on Strength
        ax7 = plt.subplot(5, 3, 7)
        defect_data = []
        labels = []
        if 'has_cracks' in self.dataset.columns:
            no_defects = self.dataset[(self.dataset['has_cracks'] == 0) & 
                                     (self.dataset['has_porosity'] == 0)]['tensile_strength']
            with_cracks = self.dataset[self.dataset['has_cracks'] == 1]['tensile_strength']
            with_porosity = self.dataset[self.dataset['has_porosity'] == 1]['tensile_strength']
            
            defect_data = [no_defects.dropna(), with_cracks.dropna(), with_porosity.dropna()]
            labels = ['No Defects', 'With Cracks', 'With Porosity']
            
            ax7.boxplot(defect_data, labels=labels)
            ax7.set_ylabel('Tensile Strength (N)')
            ax7.set_title('Defect Impact on Strength')
            ax7.tick_params(axis='x', rotation=45)
        
        # 8. Welding Speed vs Quality
        ax8 = plt.subplot(5, 3, 8)
        speed_bins = pd.cut(self.dataset['welding_speed'], bins=5)
        quality_by_speed = self.dataset.groupby(speed_bins)['quality_score'].mean()
        ax8.bar(range(len(quality_by_speed)), quality_by_speed.values)
        ax8.set_xlabel('Welding Speed Bins')
        ax8.set_ylabel('Average Quality Score')
        ax8.set_title('Welding Speed Impact on Quality')
        
        # 9. Resistance Performance
        ax9 = plt.subplot(5, 3, 9)
        ax9.scatter(self.dataset['contact_resistance'], 
                   self.dataset['resistance_increase_percent'],
                   alpha=0.5, c=self.dataset['data_tier'])
        ax9.set_xlabel('Initial Contact Resistance (μΩ)')
        ax9.set_ylabel('Resistance Increase After Aging (%)')
        ax9.set_title('Electrical Performance Degradation')
        
        # 10. Correlation Heatmap - Input Parameters
        ax10 = plt.subplot(5, 3, 10)
        input_cols = ['laser_power', 'welding_speed', 'pulse_frequency', 
                     'beam_spot_size', 'sheet_thickness', 'heat_input', 'power_density']
        input_corr = self.dataset[input_cols].corr()
        sns.heatmap(input_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax10, cbar_kws={'label': 'Correlation'})
        ax10.set_title('Input Parameter Correlations')
        
        # 11. Correlation Heatmap - Output Parameters
        ax11 = plt.subplot(5, 3, 11)
        output_cols = ['tensile_strength', 'contact_resistance', 'cycles_to_failure',
                      'strength_degradation_percent', 'quality_score']
        output_cols = [col for col in output_cols if col in self.dataset.columns]
        if output_cols:
            output_corr = self.dataset[output_cols].corr()
            sns.heatmap(output_corr, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, ax=ax11, cbar_kws={'label': 'Correlation'})
            ax11.set_title('Output Parameter Correlations')
        
        # 12. Parameter Importance for Quality
        ax12 = plt.subplot(5, 3, 12)
        # Calculate correlation with quality score
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        quality_correlations = {}
        for col in numeric_cols:
            if col != 'quality_score' and not col.endswith('_std'):
                corr = self.dataset[col].corr(self.dataset['quality_score'])
                if not np.isnan(corr):
                    quality_correlations[col] = abs(corr)
        
        # Sort and plot top 10
        top_features = sorted(quality_correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_features:
            features, importances = zip(*top_features)
            ax12.barh(range(len(features)), importances)
            ax12.set_yticks(range(len(features)))
            ax12.set_yticklabels(features)
            ax12.set_xlabel('Absolute Correlation with Quality Score')
            ax12.set_title('Top 10 Features for Quality Prediction')
        
        # 13. Thermal Cycling Performance
        ax13 = plt.subplot(5, 3, 13)
        ax13.scatter(self.dataset['cycles_to_failure'], 
                    self.dataset['strength_degradation_percent'],
                    alpha=0.5)
        ax13.set_xlabel('Cycles to Failure')
        ax13.set_ylabel('Strength Degradation (%)')
        ax13.set_title('Thermal Cycling Performance Trade-off')
        
        # 14. Process Window Visualization
        ax14 = plt.subplot(5, 3, 14)
        # Define "good" welds (quality_score > 0.7)
        good_welds = self.dataset[self.dataset['quality_score'] > 0.7]
        ax14.scatter(self.dataset['laser_power'], self.dataset['welding_speed'], 
                    alpha=0.3, label='All welds', c='gray')
        ax14.scatter(good_welds['laser_power'], good_welds['welding_speed'], 
                    alpha=0.6, label='Good welds (Q>0.7)', c='green')
        ax14.set_xlabel('Laser Power (W)')
        ax14.set_ylabel('Welding Speed (mm/s)')
        ax14.set_title('Process Window for Good Welds')
        ax14.legend()
        
        # 15. Data Completeness by Tier
        ax15 = plt.subplot(5, 3, 15)
        completeness_by_tier = {}
        for tier in [1, 2, 3]:
            tier_data = self.dataset[self.dataset['data_tier'] == tier]
            completeness = 1 - (tier_data.isnull().sum() / len(tier_data))
            completeness_by_tier[f'Tier {tier}'] = completeness.mean()
        
        ax15.bar(completeness_by_tier.keys(), completeness_by_tier.values())
        ax15.set_ylabel('Data Completeness (%)')
        ax15.set_title('Data Completeness by Tier')
        ax15.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(f'{output_path}/analysis_overview.png', dpi=150)
        plt.show()
        
        print(f"Analysis report saved to {output_path}/")
        
    def perform_pca_analysis(self, n_components: int = 3):
        """Perform PCA analysis on the dataset"""
        # Select numeric features
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not col.endswith('_std') and 
                       col not in ['global_id', 'weld_id', 'data_tier', 'year']]
        
        # Prepare data
        X = self.dataset[feature_cols].fillna(self.dataset[feature_cols].mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance
        axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Explained Variance')
        
        # 2D projection colored by quality score
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=self.dataset['quality_score'],
                                 cmap='viridis', alpha=0.5)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[1].set_title('PCA 2D Projection')
        plt.colorbar(scatter, ax=axes[1], label='Quality Score')
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=150)
        plt.show()
        
        # Print top contributing features for PC1
        pc1_features = pd.DataFrame({
            'feature': feature_cols,
            'pc1_weight': abs(pca.components_[0])
        }).sort_values('pc1_weight', ascending=False)
        
        print("\nTop 10 features contributing to PC1:")
        print(pc1_features.head(10))
        
        return pca, X_pca
    
    def analyze_extreme_temperature_performance(self):
        """Detailed analysis of extreme temperature performance"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Material-specific thermal cycling performance
        ax = axes[0, 0]
        materials = self.dataset['material_combination'].unique()
        cycle_data = []
        for mat in materials:
            mat_data = self.dataset[self.dataset['material_combination'] == mat]['cycles_to_failure']
            cycle_data.append(mat_data.dropna())
        
        if cycle_data:
            bp = ax.boxplot(cycle_data, labels=materials)
            ax.set_xlabel('Material Combination')
            ax.set_ylabel('Cycles to Failure')
            ax.set_title('Thermal Cycling by Material')
            ax.tick_params(axis='x', rotation=45)
        
        # 2. IMC growth impact
        ax = axes[0, 1]
        dissimilar = self.dataset[self.dataset['imc_thickness_post_aging'] > 0]
        if len(dissimilar) > 0:
            ax.hexbin(dissimilar['imc_thickness_post_aging'],
                     dissimilar['strength_degradation_percent'],
                     gridsize=20, cmap='YlOrRd')
            ax.set_xlabel('IMC Thickness Post-Aging (μm)')
            ax.set_ylabel('Strength Degradation (%)')
            ax.set_title('IMC Impact on Strength Degradation')
        
        # 3. Creep resistance analysis
        ax = axes[0, 2]
        ax.scatter(self.dataset['tensile_strength'],
                  self.dataset['creep_time_to_failure'],
                  alpha=0.5, c=self.dataset['quality_score'], cmap='viridis')
        ax.set_xlabel('Initial Tensile Strength (N)')
        ax.set_ylabel('Creep Time to Failure (hours)')
        ax.set_title('Creep Resistance vs Initial Strength')
        
        # 4. Grain evolution
        ax = axes[1, 0]
        grain_growth = (self.dataset['grain_size_post_aging'] - 
                       self.dataset['grain_size_initial']) / self.dataset['grain_size_initial'] * 100
        ax.hist(grain_growth.dropna(), bins=30, edgecolor='black')
        ax.set_xlabel('Grain Growth (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Grain Size Evolution During Aging')
        
        # 5. Performance retention map
        ax = axes[1, 1]
        retention = 100 - self.dataset['strength_degradation_percent']
        ax.scatter(self.dataset['heat_input'], retention,
                  c=self.dataset['cycles_to_failure'], cmap='coolwarm', alpha=0.6)
        ax.set_xlabel('Heat Input (kJ/mm)')
        ax.set_ylabel('Strength Retention (%)')
        ax.set_title('Heat Input vs Performance Retention')
        
        # 6. Electrical degradation
        ax = axes[1, 2]
        ax.scatter(self.dataset['contact_resistance'],
                  self.dataset['resistance_increase_percent'],
                  c=self.dataset['imc_thickness_post_aging'], cmap='plasma', alpha=0.6)
        ax.set_xlabel('Initial Contact Resistance (μΩ)')
        ax.set_ylabel('Resistance Increase (%)')
        ax.set_title('Electrical Performance Degradation')
        
        plt.tight_layout()
        plt.savefig('extreme_temperature_analysis.png', dpi=150)
        plt.show()
        
        # Statistical summary
        print("\n=== Extreme Temperature Performance Summary ===")
        print(f"Average cycles to failure: {self.dataset['cycles_to_failure'].mean():.0f}")
        print(f"Average strength degradation: {self.dataset['strength_degradation_percent'].mean():.1f}%")
        print(f"Average resistance increase: {self.dataset['resistance_increase_percent'].mean():.1f}%")
        
        # Best performing parameters
        top_10_percent = self.dataset.nlargest(int(len(self.dataset) * 0.1), 'cycles_to_failure')
        print("\n=== Optimal Parameters for Thermal Cycling ===")
        print(f"Average laser power: {top_10_percent['laser_power'].mean():.0f} W")
        print(f"Average welding speed: {top_10_percent['welding_speed'].mean():.1f} mm/s")
        print(f"Average heat input: {top_10_percent['heat_input'].mean():.2f} kJ/mm")
        
    def export_for_ml_training(self, output_path: str = 'ml_ready_data'):
        """Export dataset in ML-ready format"""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Define input and output features
        input_features = [
            'laser_power', 'welding_speed', 'pulse_frequency', 'pulse_duration',
            'beam_focus_position', 'beam_spot_size', 'clamping_pressure',
            'gas_flow_rate', 'sheet_thickness', 'overlap_distance',
            'heat_input', 'power_density'
        ]
        
        output_features = [
            'nugget_width', 'penetration_depth', 'haz_width',
            'tensile_strength', 'peel_strength', 'contact_resistance',
            'cycles_to_failure', 'strength_degradation_percent',
            'resistance_increase_percent', 'quality_score'
        ]
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(self.dataset, 
                                   columns=['material_combination', 'shield_gas_type', 'joint_type'])
        
        # Get updated input feature list
        input_cols = input_features + [col for col in df_encoded.columns if 
                                      col.startswith(('material_combination_', 
                                                    'shield_gas_type_', 
                                                    'joint_type_'))]
        
        # Prepare X and Y
        X = df_encoded[input_cols].fillna(df_encoded[input_cols].mean())
        Y = df_encoded[output_features].fillna(df_encoded[output_features].mean())
        
        # Split by data tier for proper train/val/test splitting
        tier1_mask = df_encoded['data_tier'] == 1
        tier2_mask = df_encoded['data_tier'] == 2
        tier3_mask = df_encoded['data_tier'] == 3
        
        # Use Tier 2 for training (large synthetic data)
        X_train = X[tier2_mask]
        Y_train = Y[tier2_mask]
        
        # Use Tier 3 for validation
        X_val = X[tier3_mask]
        Y_val = Y[tier3_mask]
        
        # Use Tier 1 for testing (high-quality experimental data)
        X_test = X[tier1_mask]
        Y_test = Y[tier1_mask]
        
        # Save datasets
        np.save(f'{output_path}/X_train.npy', X_train.values)
        np.save(f'{output_path}/Y_train.npy', Y_train.values)
        np.save(f'{output_path}/X_val.npy', X_val.values)
        np.save(f'{output_path}/Y_val.npy', Y_val.values)
        np.save(f'{output_path}/X_test.npy', X_test.values)
        np.save(f'{output_path}/Y_test.npy', Y_test.values)
        
        # Save feature names
        import json
        feature_names = {
            'input_features': list(X.columns),
            'output_features': list(Y.columns)
        }
        with open(f'{output_path}/feature_names.json', 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Save normalization parameters
        normalization_params = {
            'input_mean': X_train.mean().to_dict(),
            'input_std': X_train.std().to_dict(),
            'output_mean': Y_train.mean().to_dict(),
            'output_std': Y_train.std().to_dict()
        }
        with open(f'{output_path}/normalization_params.json', 'w') as f:
            json.dump(normalization_params, f, indent=2)
        
        print(f"\nML-ready data exported to {output_path}/")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input features: {len(X.columns)}")
        print(f"Output features: {len(Y.columns)}")
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == "__main__":
    # Load and analyze dataset
    analyzer = WeldingDatasetAnalyzer('welding_dataset/complete_dataset.csv')
    
    # Generate comprehensive analysis
    analyzer.generate_analysis_report('analysis_report')
    
    # Perform PCA analysis
    analyzer.perform_pca_analysis(n_components=5)
    
    # Analyze extreme temperature performance
    analyzer.analyze_extreme_temperature_performance()
    
    # Export ML-ready data
    analyzer.export_for_ml_training('ml_ready_data')