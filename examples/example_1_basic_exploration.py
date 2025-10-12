#!/usr/bin/env python3
"""
Example 1: Basic Data Exploration
Stratified Flow Attenuation Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def explore_frequency_domain_data():
    """Explore frequency domain attenuation data"""
    
    # Load data
    df = pd.read_csv('../stratified_flow_datasets/frequency_domain.csv')
    
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Attenuation vs Frequency
    for config in df['flow_configuration'].unique():
        config_data = df[df['flow_configuration'] == config]
        axes[0,0].loglog(config_data['frequency_hz'], 
                        config_data['attenuation_db_per_m'], 
                        'o', alpha=0.6, label=config, markersize=2)
    
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Attenuation (dB/m)')
    axes[0,0].set_title('Attenuation vs Frequency by Flow Configuration')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Gas fraction distribution
    axes[0,1].hist(df['gas_fraction'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Gas Fraction')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Gas Fraction Distribution')
    
    # Attenuation by flow configuration
    df.boxplot(column='attenuation_db_per_m', by='flow_configuration', ax=axes[1,0])
    axes[1,0].set_title('Attenuation Distribution by Flow Configuration')
    plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
    
    # Correlation heatmap
    numerical_cols = ['frequency_hz', 'attenuation_db_per_m', 'gas_fraction', 
                     'reynolds_number_gas', 'weber_number', 'temperature_c']
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('frequency_domain_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

if __name__ == "__main__":
    df = explore_frequency_domain_data()
    print("\n✅ Basic exploration completed!")
