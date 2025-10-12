#!/usr/bin/env python3
"""
Example 3: Physical Model Analysis and Comparison
Stratified Flow Attenuation Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_attenuation_models():
    """Analyze and compare different attenuation models"""
    
    # Load attenuation models data
    df = pd.read_csv('../stratified_flow_datasets/attenuation_models.csv')
    
    print("Physical Model Analysis")
    print("=" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Models included: {df['model_type'].unique()}")
    
    # Model performance comparison
    model_stats = df.groupby('model_type')['predicted_attenuation_db_per_m'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    print(f"\nModel Statistics:")
    print(model_stats)
    
    # Frequency dependence analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model predictions vs frequency
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]
        axes[0,0].loglog(model_data['frequency_hz'], 
                        model_data['predicted_attenuation_db_per_m'],
                        'o', alpha=0.6, label=model, markersize=3)
    
    axes[0,0].set_xlabel('Frequency (Hz)')
    axes[0,0].set_ylabel('Predicted Attenuation (dB/m)')
    axes[0,0].set_title('Model Predictions vs Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Model distribution comparison
    df.boxplot(column='predicted_attenuation_db_per_m', by='model_type', ax=axes[0,1])
    axes[0,1].set_title('Attenuation Predictions by Model Type')
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
    
    # Temperature dependence (for applicable models)
    temp_models = df.dropna(subset=['temperature_c'])
    for model in temp_models['model_type'].unique()[:3]:  # Top 3 models
        model_data = temp_models[temp_models['model_type'] == model]
        axes[1,0].scatter(model_data['temperature_c'], 
                         model_data['predicted_attenuation_db_per_m'],
                         alpha=0.6, label=model, s=20)
    
    axes[1,0].set_xlabel('Temperature (°C)')
    axes[1,0].set_ylabel('Predicted Attenuation (dB/m)')
    axes[1,0].set_title('Temperature Dependence (Selected Models)')
    axes[1,0].legend()
    
    # Model correlation analysis
    model_pivot = df.pivot_table(
        values='predicted_attenuation_db_per_m',
        index='frequency_hz',
        columns='model_type',
        aggfunc='mean'
    )
    
    corr_matrix = model_pivot.corr()
    im = axes[1,1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(corr_matrix.columns)))
    axes[1,1].set_yticks(range(len(corr_matrix.columns)))
    axes[1,1].set_xticklabels(corr_matrix.columns, rotation=45)
    axes[1,1].set_yticklabels(corr_matrix.columns)
    axes[1,1].set_title('Model Correlation Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('attenuation_models_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print(f"\nModel Correlation Analysis:")
    print(corr_matrix.round(3))
    
    # Frequency scaling analysis
    print(f"\nFrequency Scaling Analysis:")
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]
        if len(model_data) > 10:
            # Fit power law: attenuation ~ frequency^n
            log_freq = np.log10(model_data['frequency_hz'])
            log_atten = np.log10(model_data['predicted_attenuation_db_per_m'])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_freq, log_atten)
            
            print(f"  {model}: attenuation ∝ f^{slope:.2f} (R² = {r_value**2:.3f})")
    
    return df

if __name__ == "__main__":
    df = analyze_attenuation_models()
    print("\n✅ Physical model analysis completed!")
