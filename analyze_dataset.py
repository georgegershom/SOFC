"""
Dataset Analysis and Visualization Script
=======================================

This script analyzes and visualizes the generated residual stress dataset.
"""

import sys
import os
sys.path.append(os.getcwd())

from dataset_visualizer import DatasetVisualizer
from advanced_fea_simulator import AdvancedFEASimulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main analysis function."""
    
    print("Residual Stress Dataset Analysis")
    print("=" * 50)
    
    # Load the largest dataset
    dataset_file = "residual_stress_dataset_10000.csv"
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found!")
        return
    
    print(f"Loading dataset: {dataset_file}")
    
    # Initialize visualizer
    visualizer = DatasetVisualizer(dataset_path=dataset_file)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {visualizer.df.shape}")
    print(f"Columns: {list(visualizer.df.columns)}")
    
    # Basic statistics
    print("\nDataset Statistics:")
    print("-" * 30)
    print(f"Number of samples: {len(visualizer.df)}")
    print(f"Number of features: {len(visualizer.df.columns)}")
    print(f"Missing values: {visualizer.df.isnull().sum().sum()}")
    
    # Stress statistics
    stress_cols = [col for col in visualizer.df.columns if 'stress' in col]
    print(f"\nStress Variables ({len(stress_cols)}):")
    for col in stress_cols:
        mean_stress = visualizer.df[col].mean()
        std_stress = visualizer.df[col].std()
        min_stress = visualizer.df[col].min()
        max_stress = visualizer.df[col].max()
        print(f"  {col}: {mean_stress/1e6:.2f} ± {std_stress/1e6:.2f} MPa (range: {min_stress/1e6:.2f} to {max_stress/1e6:.2f} MPa)")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    try:
        # Set matplotlib backend to avoid display issues
        plt.switch_backend('Agg')
        
        # Generate summary report
        print("1. Generating summary report...")
        report = visualizer.generate_summary_report('dataset_analysis')
        
        print("2. Plotting parameter distributions...")
        visualizer.plot_parameter_distributions('dataset_analysis')
        
        print("3. Plotting stress distributions...")
        visualizer.plot_stress_distributions('dataset_analysis')
        
        print("4. Plotting correlation matrix...")
        corr_matrix = visualizer.plot_correlation_matrix('dataset_analysis')
        
        print("5. Plotting parameter vs stress relationships...")
        visualizer.plot_parameter_vs_stress('dataset_analysis')
        
        print("6. Performing PCA analysis...")
        pca_results = visualizer.plot_pca_analysis('dataset_analysis')
        
        print("7. Creating interactive dashboard...")
        dashboard = visualizer.create_interactive_dashboard('dataset_analysis')
        
        print("\nAll visualizations completed successfully!")
        
        # Print key findings
        print("\nKey Dataset Characteristics:")
        print("-" * 40)
        
        # Find strongest correlations with stress
        for stress_col in stress_cols:
            if stress_col in corr_matrix.columns:
                correlations = corr_matrix[stress_col].abs().sort_values(ascending=False)
                correlations = correlations[~correlations.index.str.contains('stress|sample_id')]
                print(f"\nStrongest correlations with {stress_col}:")
                for param, corr in correlations.head(3).items():
                    print(f"  {param}: {corr:.3f}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDataset analysis completed!")


if __name__ == "__main__":
    main()