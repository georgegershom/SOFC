#!/usr/bin/env python3
"""
Machine Learning Example using Atomic Simulation Dataset
Demonstrates how to use the dataset for training surrogate models
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Load and prepare data for machine learning"""
    # Load DFT data
    with open('atomic_simulation_data/dft_simulations.json', 'r') as f:
        dft_data = json.load(f)
    
    # Extract features and targets
    features = []
    targets_formation = []
    targets_activation = []
    targets_surface = []
    
    for sim in dft_data:
        # Features: temperature, pressure, lattice parameter, cutoff energy
        feature_vector = [
            sim['parameters']['temperature'],
            sim['parameters']['pressure'],
            sim['parameters']['lattice_parameter'],
            sim['parameters']['cutoff_energy'],
            # Add material as one-hot encoding
            1 if sim['material'] == 'Ni' else 0,
            1 if sim['material'] == 'Al' else 0,
            1 if sim['material'] == 'Fe' else 0,
            1 if sim['material'] == 'Cu' else 0,
        ]
        
        features.append(feature_vector)
        targets_formation.append(sim['results']['formation_energy'])
        targets_activation.append(sim['results']['activation_barrier'])
        targets_surface.append(sim['results']['surface_energy'])
    
    X = np.array(features)
    y_formation = np.array(targets_formation)
    y_activation = np.array(targets_activation)
    y_surface = np.array(targets_surface)
    
    return X, y_formation, y_activation, y_surface

def train_surrogate_models(X, y_formation, y_activation, y_surface):
    """Train surrogate models for different properties"""
    
    # Split data
    X_train, X_test, y_formation_train, y_formation_test = train_test_split(
        X, y_formation, test_size=0.2, random_state=42
    )
    
    # Train models
    models = {}
    
    # Formation energy model
    print("Training formation energy model...")
    model_formation = RandomForestRegressor(n_estimators=100, random_state=42)
    model_formation.fit(X_train, y_formation_train)
    y_formation_pred = model_formation.predict(X_test)
    
    models['formation_energy'] = {
        'model': model_formation,
        'y_test': y_formation_test,
        'y_pred': y_formation_pred,
        'mse': mean_squared_error(y_formation_test, y_formation_pred),
        'r2': r2_score(y_formation_test, y_formation_pred)
    }
    
    # Activation barrier model
    print("Training activation barrier model...")
    X_train, X_test, y_activation_train, y_activation_test = train_test_split(
        X, y_activation, test_size=0.2, random_state=42
    )
    
    model_activation = RandomForestRegressor(n_estimators=100, random_state=42)
    model_activation.fit(X_train, y_activation_train)
    y_activation_pred = model_activation.predict(X_test)
    
    models['activation_barrier'] = {
        'model': model_activation,
        'y_test': y_activation_test,
        'y_pred': y_activation_pred,
        'mse': mean_squared_error(y_activation_test, y_activation_pred),
        'r2': r2_score(y_activation_test, y_activation_pred)
    }
    
    # Surface energy model
    print("Training surface energy model...")
    X_train, X_test, y_surface_train, y_surface_test = train_test_split(
        X, y_surface, test_size=0.2, random_state=42
    )
    
    model_surface = RandomForestRegressor(n_estimators=100, random_state=42)
    model_surface.fit(X_train, y_surface_train)
    y_surface_pred = model_surface.predict(X_test)
    
    models['surface_energy'] = {
        'model': model_surface,
        'y_test': y_surface_test,
        'y_pred': y_surface_pred,
        'mse': mean_squared_error(y_surface_test, y_surface_pred),
        'r2': r2_score(y_surface_test, y_surface_pred)
    }
    
    return models

def evaluate_models(models):
    """Evaluate and visualize model performance"""
    print("\n=== Model Performance ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (property_name, model_data) in enumerate(models.items()):
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        mse = model_data['mse']
        r2 = model_data['r2']
        
        print(f"{property_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Plot predictions vs actual
        axes[i].scatter(y_test, y_pred, alpha=0.6)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{property_name.replace("_", " ").title()}\nR² = {r2:.3f}')
    
    plt.tight_layout()
    plt.savefig('ml_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(models):
    """Analyze feature importance across models"""
    feature_names = [
        'Temperature', 'Pressure', 'Lattice Parameter', 'Cutoff Energy',
        'Ni', 'Al', 'Fe', 'Cu'
    ]
    
    print("\n=== Feature Importance ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (property_name, model_data) in enumerate(models.items()):
        importance = model_data['model'].feature_importances_
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        
        print(f"\n{property_name}:")
        for j in sorted_idx:
            print(f"  {feature_names[j]}: {importance[j]:.3f}")
        
        # Plot feature importance
        axes[i].barh(range(len(feature_names)), importance[sorted_idx])
        axes[i].set_yticks(range(len(feature_names)))
        axes[i].set_yticklabels([feature_names[j] for j in sorted_idx])
        axes[i].set_xlabel('Feature Importance')
        axes[i].set_title(f'{property_name.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_new_material(models, temperature=800, pressure=1.0, material='Ni'):
    """Predict properties for a new material configuration"""
    
    # Material encoding
    material_encoding = {
        'Ni': [1, 0, 0, 0],
        'Al': [0, 1, 0, 0],
        'Fe': [0, 0, 1, 0],
        'Cu': [0, 0, 0, 1]
    }
    
    # Material properties
    lattice_params = {'Ni': 3.52, 'Al': 4.05, 'Fe': 2.87, 'Cu': 3.61}
    
    # Create feature vector
    feature_vector = [
        temperature,
        pressure,
        lattice_params[material],
        500.0,  # Default cutoff energy
    ] + material_encoding[material]
    
    X_new = np.array([feature_vector])
    
    print(f"\n=== Predictions for {material} at T={temperature}K, P={pressure}GPa ===")
    
    predictions = {}
    for property_name, model_data in models.items():
        pred = model_data['model'].predict(X_new)[0]
        predictions[property_name] = pred
        
        unit = 'eV/atom' if 'energy' in property_name else 'eV' if 'barrier' in property_name else 'J/m²'
        print(f"{property_name.replace('_', ' ').title()}: {pred:.3f} {unit}")
    
    return predictions

def main():
    """Main function demonstrating ML workflow"""
    print("Loading atomic simulation dataset for machine learning...")
    X, y_formation, y_activation, y_surface = load_and_prepare_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: Temperature, Pressure, Lattice Parameter, Cutoff Energy, Material (one-hot)")
    
    print("\nTraining surrogate models...")
    models = train_surrogate_models(X, y_formation, y_activation, y_surface)
    
    evaluate_models(models)
    analyze_feature_importance(models)
    
    # Example prediction
    predict_new_material(models, temperature=1000, pressure=2.0, material='Fe')
    
    print("\nMachine learning example complete!")
    print("Check 'ml_model_performance.png' and 'feature_importance.png' for visualizations.")

if __name__ == "__main__":
    main()