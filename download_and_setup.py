#!/usr/bin/env python3
"""
Dataset Download and Setup Script
For Stratified Flow Attenuation Research
"""

import os
import zipfile
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import json

def create_dataset_metadata():
    """Create comprehensive metadata for the dataset"""
    
    metadata = {
        "dataset_info": {
            "title": "Stratified Flow Attenuation Dataset",
            "research_topic": "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics",
            "version": "1.0",
            "creation_date": "2025-10-12",
            "total_samples": 5800,
            "file_format": "CSV",
            "license": "Research and Educational Use"
        },
        "datasets": {
            "frequency_domain": {
                "filename": "frequency_domain.csv",
                "samples": 2000,
                "description": "Frequency-dependent attenuation measurements across different flow configurations",
                "key_features": [
                    "frequency_hz", "flow_configuration", "attenuation_db_per_m", 
                    "gas_fraction", "interface_roughness_mm", "reynolds_number_gas",
                    "weber_number", "froude_number", "temperature_c", "pressure_bar"
                ],
                "applications": [
                    "Frequency-dependent modeling", "Flow pattern classification",
                    "Interface scattering analysis", "Dimensionless correlations"
                ]
            },
            "experimental_conditions": {
                "filename": "experimental_conditions.csv",
                "samples": 500,
                "description": "Realistic experimental setup parameters and operating conditions",
                "key_features": [
                    "pipe_diameter_m", "pipe_length_m", "inclination_angle_deg",
                    "gas_superficial_velocity_ms", "liquid_superficial_velocity_ms",
                    "predicted_flow_pattern", "transducer_frequency_hz"
                ],
                "applications": [
                    "Experimental design", "Flow pattern mapping",
                    "Scaling analysis", "Sensor placement optimization"
                ]
            },
            "attenuation_models": {
                "filename": "attenuation_models.csv",
                "samples": 1500,
                "description": "Theoretical predictions from various physical attenuation models",
                "key_features": [
                    "model_type", "frequency_hz", "predicted_attenuation_db_per_m",
                    "particle_radius_m", "viscosity_pas", "roughness_m"
                ],
                "applications": [
                    "Model validation", "Physical mechanism identification",
                    "Parameter sensitivity", "Hybrid model development"
                ]
            },
            "multiphase_flow": {
                "filename": "multiphase_flow.csv",
                "samples": 1000,
                "description": "Comprehensive multiphase flow characterization with acoustic properties",
                "key_features": [
                    "flow_regime", "gas_fraction", "bubble_diameter_m",
                    "mixture_sound_speed_ms", "total_attenuation_db_per_m_at_1khz",
                    "interface_area_density_m2m3", "turbulent_kinetic_energy_m2s2"
                ],
                "applications": [
                    "Mixture property modeling", "Multi-mechanism analysis",
                    "Flow regime characterization", "Turbulence-acoustics coupling"
                ]
            },
            "time_series_features": {
                "filename": "time_series_features.csv",
                "samples": 800,
                "description": "Time-domain acoustic signal features for flow pattern recognition",
                "key_features": [
                    "flow_pattern", "rms_amplitude", "dominant_frequency_hz",
                    "spectral_centroid_hz", "zero_crossings_per_sec", "spectral_bandwidth_hz"
                ],
                "applications": [
                    "Real-time monitoring", "Pattern recognition",
                    "Signal processing", "Machine learning training"
                ]
            }
        },
        "physical_parameters": {
            "frequency_range_hz": [10, 100000],
            "temperature_range_c": [10, 80],
            "pressure_range_bar": [1, 50],
            "gas_fraction_range": [0.01, 0.99],
            "pipe_diameter_range_m": [0.025, 0.3],
            "flow_velocities_ms": [0.01, 20.0]
        },
        "flow_configurations": [
            "horizontal_stratified", "inclined_stratified", "wavy_interface",
            "slug_flow", "annular_flow", "dispersed_flow"
        ],
        "attenuation_models": [
            "rayleigh_scattering", "mie_scattering", "viscous_losses",
            "thermal_losses", "interface_scattering", "mode_conversion"
        ],
        "research_applications": [
            "PhD thesis research", "Acoustic flow measurement development",
            "Physical model validation", "Machine learning applications",
            "Industrial flow monitoring", "Sensor optimization studies"
        ]
    }
    
    return metadata

def setup_research_environment():
    """Setup complete research environment with examples"""
    
    print("Setting up research environment...")
    
    # Create example analysis scripts
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    
    # Example 1: Basic data exploration
    example1_code = '''#!/usr/bin/env python3
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
    print("\\nBasic Statistics:")
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
    print("\\n✅ Basic exploration completed!")
'''
    
    with open(os.path.join(examples_dir, 'example_1_basic_exploration.py'), 'w') as f:
        f.write(example1_code)
    
    # Example 2: Machine learning application
    example2_code = '''#!/usr/bin/env python3
"""
Example 2: Machine Learning for Flow Pattern Classification
Stratified Flow Attenuation Dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def flow_pattern_classification():
    """Train ML model for flow pattern classification"""
    
    # Load time series features data
    df = pd.read_csv('../stratified_flow_datasets/time_series_features.csv')
    
    print("Flow Pattern Classification Analysis")
    print("=" * 40)
    print(f"Dataset shape: {df.shape}")
    print(f"Flow patterns: {df['flow_pattern'].unique()}")
    print(f"Pattern distribution:")
    print(df['flow_pattern'].value_counts())
    
    # Prepare features and target
    feature_cols = ['rms_amplitude', 'dominant_frequency_hz', 'spectral_centroid_hz',
                   'zero_crossings_per_sec', 'spectral_bandwidth_hz', 
                   'mfcc_1', 'mfcc_2', 'mfcc_3']
    
    X = df[feature_cols]
    y = df['flow_pattern']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = rf_model.score(X_train_scaled, y_train)
    test_score = rf_model.score(X_test_scaled, y_test)
    
    print(f"\\nModel Performance:")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\\nFeature Importance:")
    print(feature_importance)
    
    # Predictions and detailed evaluation
    y_pred = rf_model.predict(X_test_scaled)
    
    print(f"\\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Feature importance plot
    axes[0].barh(feature_importance['feature'], feature_importance['importance'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Feature Importance for Flow Pattern Classification')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('flow_pattern_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, scaler

if __name__ == "__main__":
    model, scaler = flow_pattern_classification()
    print("\\n✅ Machine learning analysis completed!")
'''
    
    with open(os.path.join(examples_dir, 'example_2_ml_classification.py'), 'w') as f:
        f.write(example2_code)
    
    # Example 3: Physical model analysis
    example3_code = '''#!/usr/bin/env python3
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
    
    print(f"\\nModel Statistics:")
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
    print(f"\\nModel Correlation Analysis:")
    print(corr_matrix.round(3))
    
    # Frequency scaling analysis
    print(f"\\nFrequency Scaling Analysis:")
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
    print("\\n✅ Physical model analysis completed!")
'''
    
    with open(os.path.join(examples_dir, 'example_3_physical_models.py'), 'w') as f:
        f.write(example3_code)
    
    print(f"✅ Research environment setup completed!")
    print(f"📁 Examples directory: {examples_dir}")
    print(f"📝 Created 3 example analysis scripts")

def create_requirements_file():
    """Create requirements.txt for easy environment setup"""
    
    requirements = """# Stratified Flow Attenuation Dataset Requirements
# Core data science libraries
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0

# Machine learning
scikit-learn>=1.0.0

# Optional advanced libraries
plotly>=5.0.0
jupyter>=1.0.0
ipywidgets>=7.6.0

# For data processing
openpyxl>=3.0.0
xlsxwriter>=3.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Requirements file created: requirements.txt")

def main():
    """Main setup function"""
    
    print("Stratified Flow Attenuation Dataset Setup")
    print("=" * 50)
    
    # Check if datasets exist
    data_dir = 'stratified_flow_datasets'
    if not os.path.exists(data_dir):
        print("❌ Dataset directory not found!")
        print("Please run 'python3 generate_datasets.py' first to create the datasets.")
        return
    
    # Create metadata
    print("Creating dataset metadata...")
    metadata = create_dataset_metadata()
    
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Setup research environment
    setup_research_environment()
    
    # Create requirements file
    create_requirements_file()
    
    # Verify dataset integrity
    print("\\nVerifying dataset integrity...")
    
    total_samples = 0
    for filename in ['frequency_domain.csv', 'experimental_conditions.csv', 
                    'attenuation_models.csv', 'multiphase_flow.csv', 
                    'time_series_features.csv']:
        
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"✅ {filename}: {len(df)} samples, {len(df.columns)} features")
            total_samples += len(df)
        else:
            print(f"❌ {filename}: Not found!")
    
    print(f"\\n📊 Total verified samples: {total_samples}")
    
    # Final instructions
    print("\\n🎯 Setup Complete!")
    print("\\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Explore examples: cd examples && python example_1_basic_exploration.py")
    print("3. Read documentation: README.md")
    print("4. Check analysis results: stratified_flow_datasets/research_insights.md")
    
    print("\\n📚 Dataset ready for PhD research!")
    print("Topic: 'Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics'")

if __name__ == "__main__":
    main()