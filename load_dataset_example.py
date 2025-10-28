#!/usr/bin/env python3
"""
Example script for loading and using the sintering process dataset
Demonstrates how to load data for ANN and PINN model training
"""

import numpy as np
import pandas as pd
import h5py
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_training_data(csv_path='/workspace/sintering_training_data.csv'):
    """Load training data from CSV"""
    print("Loading training data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} training samples with {len(df.columns)} features")
    return df

def load_validation_data(csv_path='/workspace/sintering_validation_data.csv'):
    """Load validation data from CSV"""
    print("Loading validation data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} validation samples")
    return df

def load_hdf5_data(hdf5_path='/workspace/sintering_dataset.h5'):
    """Load spatial data from HDF5 file"""
    print("Loading spatial data from HDF5...")
    with h5py.File(hdf5_path, 'r') as f:
        # Load stress fields
        stress_fields = []
        for i in range(len(f['stress_fields'])):
            stress_fields.append(f['stress_fields'][f'field_{i}'][:])
        
        # Load coordinates
        coordinates = []
        for i in range(len(f['coordinates'])):
            coordinates.append(f['coordinates'][f'coords_{i}'][:])
        
        print(f"Loaded {len(stress_fields)} stress fields")
        return stress_fields, coordinates

def prepare_features_for_ann(df):
    """Prepare features for ANN training"""
    print("Preparing features for ANN...")
    
    # Separate input features and output labels
    input_features = [
        'temperature', 'cooling_rate', 'porosity', 'tec_mismatch',
        'material_E', 'material_nu', 'material_alpha', 'material_density',
        'stress_strain_ratio', 'thermal_gradient', 'porosity_stress_factor'
    ]
    
    output_labels = [
        'stress_hotspot_density', 'crack_risk', 'delamination_probability'
    ]
    
    X = df[input_features].values
    y = df[output_labels].values
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    print(f"Input features shape: {X_scaled.shape}")
    print(f"Output labels shape: {y_scaled.shape}")
    
    return X_scaled, y_scaled, scaler_X, scaler_y, input_features, output_labels

def prepare_data_for_pinn(df, stress_fields, coordinates):
    """Prepare data for PINN training with spatial information"""
    print("Preparing data for PINN...")
    
    # PINN requires spatial coordinates and field data
    pinn_data = []
    
    for i in range(len(df)):
        sample_data = {
            'coordinates': coordinates[i],
            'stress_field': stress_fields[i],
            'temperature': df.iloc[i]['temperature'],
            'cooling_rate': df.iloc[i]['cooling_rate'],
            'porosity': df.iloc[i]['porosity'],
            'material_properties': {
                'E': df.iloc[i]['material_E'],
                'nu': df.iloc[i]['material_nu'],
                'alpha': df.iloc[i]['material_alpha'],
                'density': df.iloc[i]['material_density']
            },
            'outputs': {
                'stress_hotspot_density': df.iloc[i]['stress_hotspot_density'],
                'crack_risk': df.iloc[i]['crack_risk'],
                'delamination_probability': df.iloc[i]['delamination_probability']
            }
        }
        pinn_data.append(sample_data)
    
    print(f"Prepared {len(pinn_data)} samples for PINN training")
    return pinn_data

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train-test split"""
    print("Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def demonstrate_usage():
    """Demonstrate how to use the dataset"""
    print("🧠 Sintering Process Dataset Usage Example")
    print("=" * 50)
    
    # Load data
    df = load_training_data()
    val_df = load_validation_data()
    stress_fields, coordinates = load_hdf5_data()
    
    # Prepare for ANN
    X, y, scaler_X, scaler_y, input_features, output_labels = prepare_features_for_ann(df)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Prepare for PINN
    pinn_data = prepare_data_for_pinn(df, stress_fields, coordinates)
    
    # Display sample statistics
    print("\n=== Dataset Statistics ===")
    print(f"Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
    print(f"Cooling rate range: {df['cooling_rate'].min():.1f} - {df['cooling_rate'].max():.1f} °C/min")
    print(f"Porosity range: {df['porosity'].min():.3f} - {df['porosity'].max():.3f}")
    
    print(f"\nOutput statistics:")
    print(f"Stress hotspot density: {df['stress_hotspot_density'].mean():.3f} ± {df['stress_hotspot_density'].std():.3f}")
    print(f"Crack risk: {df['crack_risk'].mean():.3f} ± {df['crack_risk'].std():.3f}")
    print(f"Delamination probability: {df['delamination_probability'].mean():.3f} ± {df['delamination_probability'].std():.3f}")
    
    print(f"\nMaterial distribution:")
    print(df['material'].value_counts())
    
    print("\n=== Ready for Model Training ===")
    print("ANN Features:", input_features)
    print("ANN Outputs:", output_labels)
    print("PINN Data: Spatial stress fields with coordinates")
    
    return {
        'ann_data': (X_train, X_test, y_train, y_test, scaler_X, scaler_y),
        'pinn_data': pinn_data,
        'validation_data': val_df,
        'feature_names': input_features,
        'output_names': output_labels
    }

def create_simple_model_example():
    """Create a simple example of how to use the data with a basic model"""
    print("\n=== Simple Model Example ===")
    
    # Load and prepare data
    df = load_training_data()
    X, y, scaler_X, scaler_y, input_features, output_labels = prepare_features_for_ann(df)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Simple linear regression example
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    print("Training simple linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Show feature importance
    feature_importance = np.abs(model.coef_).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': input_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    print(importance_df.head())
    
    return model, importance_df

if __name__ == "__main__":
    # Demonstrate dataset usage
    data_dict = demonstrate_usage()
    
    # Create simple model example
    model, importance = create_simple_model_example()
    
    print("\n✅ Dataset loading and usage demonstration complete!")
    print("The dataset is ready for ANN and PINN model training.")