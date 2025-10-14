#!/usr/bin/env python3
"""
Welding Inverse Design Dataset - Usage Examples
===============================================

This script demonstrates how to use the welding inverse design dataset
for various machine learning applications including inverse design,
multi-objective optimization, and performance prediction.

Author: AI Assistant
Date: 2025-10-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')

class WeldingDatasetDemo:
    """
    Demonstration class for welding inverse design dataset usage.
    """
    
    def __init__(self, dataset_path='welding_inverse_design_master_dataset.csv'):
        """Initialize with dataset."""
        print("Loading welding inverse design dataset...")
        self.df = pd.read_csv(dataset_path)
        
        # Define column groups
        self.input_cols = [
            'laser_power_w', 'welding_speed_mm_s', 'pulse_frequency_hz', 
            'pulse_duration_ms', 'beam_focus_position_mm', 'beam_spot_size_um',
            'clamping_pressure_mpa', 'shield_gas_flow_rate_l_min', 
            'sheet_thickness_mm', 'overlap_distance_mm'
        ]
        
        self.categorical_cols = ['material_combination', 'joint_type']
        
        self.performance_cols = [
            'tensile_shear_strength_n', 'contact_resistance_micro_ohm',
            'cycles_to_failure', 'strength_degradation_percent'
        ]
        
        self.morphology_cols = [
            'nugget_width_mm', 'penetration_depth_mm', 'haz_width_mm'
        ]
        
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Data sources: {dict(self.df['data_source'].value_counts())}")
    
    def basic_data_exploration(self):
        """Perform basic data exploration and visualization."""
        print("\n=== BASIC DATA EXPLORATION ===")
        
        # Dataset overview
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data source distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        self.df['data_source'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Data Source Distribution')
        
        # Material combination distribution
        plt.subplot(1, 3, 2)
        self.df['material_combination'].value_counts().plot(kind='bar')
        plt.title('Material Combinations')
        plt.xticks(rotation=45)
        
        # Joint type distribution
        plt.subplot(1, 3, 3)
        self.df['joint_type'].value_counts().plot(kind='bar')
        plt.title('Joint Types')
        
        plt.tight_layout()
        plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Key parameter distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        key_params = ['laser_power_w', 'welding_speed_mm_s', 'tensile_shear_strength_n',
                     'contact_resistance_micro_ohm', 'cycles_to_failure', 'nugget_width_mm']
        
        for i, param in enumerate(key_params):
            ax = axes[i//3, i%3]
            self.df[param].hist(bins=50, ax=ax, alpha=0.7)
            ax.set_title(f'{param}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self):
        """Analyze correlations between parameters."""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select numerical columns for correlation
        numerical_cols = self.input_cols + self.performance_cols + self.morphology_cols
        corr_data = self.df[numerical_cols].dropna()
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strongest correlations with performance metrics
        print("Strongest correlations with performance metrics:")
        for perf_col in self.performance_cols:
            if perf_col in corr_matrix.columns:
                correlations = corr_matrix[perf_col][self.input_cols].abs().sort_values(ascending=False)
                print(f"\n{perf_col}:")
                for param, corr_val in correlations.head(3).items():
                    actual_corr = corr_matrix.loc[param, perf_col]
                    print(f"  {param}: {actual_corr:.3f}")
    
    def forward_modeling_example(self):
        """Demonstrate forward modeling: Parameters → Performance."""
        print("\n=== FORWARD MODELING EXAMPLE ===")
        print("Predicting performance from welding parameters...")
        
        # Prepare data
        feature_cols = self.input_cols.copy()
        
        # Encode categorical variables
        df_model = self.df.copy()
        le_material = LabelEncoder()
        le_joint = LabelEncoder()
        
        df_model['material_encoded'] = le_material.fit_transform(df_model['material_combination'])
        df_model['joint_encoded'] = le_joint.fit_transform(df_model['joint_type'])
        
        feature_cols.extend(['material_encoded', 'joint_encoded'])
        
        # Select complete cases
        target_col = 'tensile_shear_strength_n'
        complete_mask = df_model[feature_cols + [target_col]].notna().all(axis=1)
        df_complete = df_model[complete_mask]
        
        print(f"Complete cases for modeling: {len(df_complete)}")
        
        # Prepare features and target
        X = df_complete[feature_cols]
        y = df_complete[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"  R² Score: {r2:.3f}")
        print(f"  RMSE: {np.sqrt(mse):.2f} N")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Tensile Strength (N)')
        plt.ylabel('Predicted Tensile Strength (N)')
        plt.title(f'Forward Model: Actual vs Predicted (R² = {r2:.3f})')
        plt.tight_layout()
        plt.savefig('forward_model_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return model, scaler, le_material, le_joint
    
    def multi_objective_analysis(self):
        """Analyze multi-objective optimization scenarios."""
        print("\n=== MULTI-OBJECTIVE ANALYSIS ===")
        print("Analyzing trade-offs between conflicting objectives...")
        
        # Define objectives (some to maximize, some to minimize)
        objectives = {
            'tensile_shear_strength_n': 'maximize',      # Higher is better
            'contact_resistance_micro_ohm': 'minimize',   # Lower is better
            'cycles_to_failure': 'maximize',              # Higher is better
            'strength_degradation_percent': 'minimize'    # Lower is better
        }
        
        # Get complete data for all objectives
        complete_mask = self.df[list(objectives.keys())].notna().all(axis=1)
        df_complete = self.df[complete_mask]
        
        print(f"Samples with all objectives: {len(df_complete)}")
        
        # Normalize objectives (0-1 scale)
        df_norm = df_complete.copy()
        for obj, direction in objectives.items():
            if direction == 'maximize':
                df_norm[f'{obj}_norm'] = (df_complete[obj] - df_complete[obj].min()) / (df_complete[obj].max() - df_complete[obj].min())
            else:  # minimize
                df_norm[f'{obj}_norm'] = (df_complete[obj].max() - df_complete[obj]) / (df_complete[obj].max() - df_complete[obj].min())
        
        # Calculate composite score (equal weights)
        norm_cols = [f'{obj}_norm' for obj in objectives.keys()]
        df_norm['composite_score'] = df_norm[norm_cols].mean(axis=1)
        
        # Find Pareto-optimal solutions (simplified)
        top_performers = df_norm.nlargest(20, 'composite_score')
        
        print(f"\nTop 5 Multi-Objective Solutions:")
        for i, (idx, row) in enumerate(top_performers.head().iterrows()):
            print(f"\nSolution {i+1} (Weld ID: {row['weld_id']}):")
            print(f"  Composite Score: {row['composite_score']:.3f}")
            print(f"  Laser Power: {row['laser_power_w']:.0f} W")
            print(f"  Welding Speed: {row['welding_speed_mm_s']:.1f} mm/s")
            print(f"  Material: {row['material_combination']}")
            for obj in objectives.keys():
                print(f"  {obj}: {row[obj]:.2f}")
        
        # Visualize trade-offs
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Strength vs Resistance trade-off
        axes[0,0].scatter(df_complete['tensile_shear_strength_n'], 
                         df_complete['contact_resistance_micro_ohm'], 
                         alpha=0.6, c=df_norm['composite_score'], cmap='viridis')
        axes[0,0].set_xlabel('Tensile Strength (N)')
        axes[0,0].set_ylabel('Contact Resistance (µΩ)')
        axes[0,0].set_title('Strength vs Resistance')
        
        # Cycles vs Degradation trade-off
        axes[0,1].scatter(df_complete['cycles_to_failure'], 
                         df_complete['strength_degradation_percent'], 
                         alpha=0.6, c=df_norm['composite_score'], cmap='viridis')
        axes[0,1].set_xlabel('Cycles to Failure')
        axes[0,1].set_ylabel('Strength Degradation (%)')
        axes[0,1].set_title('Durability vs Degradation')
        
        # Parameter space of top solutions
        axes[1,0].scatter(df_complete['laser_power_w'], 
                         df_complete['welding_speed_mm_s'], 
                         alpha=0.3, label='All data')
        axes[1,0].scatter(top_performers['laser_power_w'], 
                         top_performers['welding_speed_mm_s'], 
                         color='red', s=50, label='Top solutions')
        axes[1,0].set_xlabel('Laser Power (W)')
        axes[1,0].set_ylabel('Welding Speed (mm/s)')
        axes[1,0].set_title('Optimal Parameter Space')
        axes[1,0].legend()
        
        # Composite score distribution
        axes[1,1].hist(df_norm['composite_score'], bins=30, alpha=0.7)
        axes[1,1].axvline(top_performers['composite_score'].min(), color='red', 
                         linestyle='--', label='Top 20 threshold')
        axes[1,1].set_xlabel('Composite Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Multi-Objective Score Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('multi_objective_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_performers
    
    def inverse_design_example(self):
        """Demonstrate inverse design approach."""
        print("\n=== INVERSE DESIGN EXAMPLE ===")
        print("Finding parameters for desired performance targets...")
        
        # Define target performance criteria
        targets = {
            'tensile_shear_strength_n': {'min': 2000, 'max': 5000},
            'contact_resistance_micro_ohm': {'min': 5, 'max': 50},
            'cycles_to_failure': {'min': 1000, 'max': 10000}
        }
        
        print("Target Performance Criteria:")
        for target, limits in targets.items():
            print(f"  {target}: {limits['min']} - {limits['max']}")
        
        # Find samples meeting all criteria
        df_filtered = self.df.copy()
        
        for target, limits in targets.items():
            if target in df_filtered.columns:
                mask = (df_filtered[target] >= limits['min']) & (df_filtered[target] <= limits['max'])
                df_filtered = df_filtered[mask]
        
        print(f"\nSamples meeting all criteria: {len(df_filtered)} ({len(df_filtered)/len(self.df)*100:.1f}%)")
        
        if len(df_filtered) > 0:
            # Analyze parameter distributions for successful cases
            print("\nRecommended Parameter Ranges:")
            
            param_recommendations = {}
            for param in self.input_cols:
                if param in df_filtered.columns and df_filtered[param].notna().sum() > 0:
                    param_data = df_filtered[param].dropna()
                    recommendations = {
                        'mean': param_data.mean(),
                        'std': param_data.std(),
                        'min': param_data.min(),
                        'max': param_data.max(),
                        'q25': param_data.quantile(0.25),
                        'q75': param_data.quantile(0.75)
                    }
                    param_recommendations[param] = recommendations
                    
                    print(f"  {param}:")
                    print(f"    Optimal range: {recommendations['q25']:.2f} - {recommendations['q75']:.2f}")
                    print(f"    Mean ± Std: {recommendations['mean']:.2f} ± {recommendations['std']:.2f}")
            
            # Material and joint type recommendations
            print(f"\nRecommended Material Combinations:")
            material_success = df_filtered['material_combination'].value_counts()
            for material, count in material_success.items():
                success_rate = count / self.df[self.df['material_combination'] == material].shape[0] * 100
                print(f"  {material}: {count} samples ({success_rate:.1f}% success rate)")
            
            print(f"\nRecommended Joint Types:")
            joint_success = df_filtered['joint_type'].value_counts()
            for joint, count in joint_success.items():
                success_rate = count / self.df[self.df['joint_type'] == joint].shape[0] * 100
                print(f"  {joint}: {count} samples ({success_rate:.1f}% success rate)")
            
            # Visualize parameter distributions for successful cases
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            key_params = ['laser_power_w', 'welding_speed_mm_s', 'beam_spot_size_um',
                         'sheet_thickness_mm', 'clamping_pressure_mpa', 'shield_gas_flow_rate_l_min']
            
            for i, param in enumerate(key_params):
                ax = axes[i//3, i%3]
                
                # Plot all data
                self.df[param].hist(bins=30, alpha=0.5, label='All data', ax=ax, density=True)
                
                # Plot successful cases
                if param in df_filtered.columns:
                    df_filtered[param].hist(bins=20, alpha=0.7, label='Target achieved', 
                                          ax=ax, color='red', density=True)
                
                ax.set_title(f'{param}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig('inverse_design_parameters.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return param_recommendations, df_filtered
        
        else:
            print("No samples found meeting all criteria. Consider relaxing constraints.")
            return None, None
    
    def data_tier_comparison(self):
        """Compare data quality and characteristics across tiers."""
        print("\n=== DATA TIER COMPARISON ===")
        
        # Compare key metrics across tiers
        comparison_metrics = ['tensile_shear_strength_n', 'contact_resistance_micro_ohm', 
                            'nugget_width_mm', 'cycles_to_failure']
        
        tier_stats = {}
        for source in self.df['data_source'].unique():
            source_data = self.df[self.df['data_source'] == source]
            stats = {}
            
            for metric in comparison_metrics:
                if metric in source_data.columns:
                    metric_data = source_data[metric].dropna()
                    if len(metric_data) > 0:
                        stats[metric] = {
                            'count': len(metric_data),
                            'mean': metric_data.mean(),
                            'std': metric_data.std(),
                            'completeness': len(metric_data) / len(source_data) * 100
                        }
            
            tier_stats[source] = stats
        
        # Print comparison
        for metric in comparison_metrics:
            print(f"\n{metric}:")
            for source, stats in tier_stats.items():
                if metric in stats:
                    s = stats[metric]
                    print(f"  {source}: μ={s['mean']:.2f}, σ={s['std']:.2f}, "
                          f"n={s['count']}, completeness={s['completeness']:.1f}%")
        
        # Visualize tier differences
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, metric in enumerate(comparison_metrics):
            ax = axes[i//2, i%2]
            
            for source in self.df['data_source'].unique():
                source_data = self.df[self.df['data_source'] == source][metric].dropna()
                if len(source_data) > 0:
                    ax.hist(source_data, bins=20, alpha=0.6, label=source, density=True)
            
            ax.set_title(f'{metric} by Data Source')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('data_tier_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return tier_stats

def main():
    """Main demonstration function."""
    print("=== WELDING INVERSE DESIGN DATASET DEMONSTRATION ===")
    
    # Initialize demo
    demo = WeldingDatasetDemo()
    
    # Run demonstrations
    print("\n1. Basic Data Exploration")
    demo.basic_data_exploration()
    
    print("\n2. Correlation Analysis")
    demo.correlation_analysis()
    
    print("\n3. Forward Modeling Example")
    model, scaler, le_material, le_joint = demo.forward_modeling_example()
    
    print("\n4. Multi-Objective Analysis")
    top_solutions = demo.multi_objective_analysis()
    
    print("\n5. Inverse Design Example")
    param_recommendations, successful_cases = demo.inverse_design_example()
    
    print("\n6. Data Tier Comparison")
    tier_stats = demo.data_tier_comparison()
    
    print("\n=== DEMONSTRATION COMPLETE ===")
    print("Generated visualization files:")
    print("- dataset_overview.png")
    print("- parameter_distributions.png")
    print("- correlation_matrix.png")
    print("- forward_model_predictions.png")
    print("- multi_objective_analysis.png")
    print("- inverse_design_parameters.png")
    print("- data_tier_comparison.png")
    
    return demo

if __name__ == "__main__":
    demo = main()