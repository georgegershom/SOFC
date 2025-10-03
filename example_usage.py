#!/usr/bin/env python3
"""
Example Usage of FEM Simulation Dataset
Demonstrates how to load and use the generated data for various applications
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset():
    """Load the complete FEM dataset"""
    print("🔄 Loading FEM dataset...")
    
    # Load mesh data
    nodes = pd.read_csv('fem_dataset/nodes.csv')
    elements = pd.read_csv('fem_dataset/elements.csv')
    
    # Load simulation results
    with open('fem_dataset/stress_distributions.json', 'r') as f:
        stress_data = json.load(f)
    
    with open('fem_dataset/damage_evolution.json', 'r') as f:
        damage_data = json.load(f)
    
    with open('fem_dataset/material_models.json', 'r') as f:
        materials = json.load(f)
    
    print("✅ Dataset loaded successfully!")
    return nodes, elements, stress_data, damage_data, materials

def example_1_basic_analysis():
    """Example 1: Basic data analysis and visualization"""
    print("\n" + "="*60)
    print("📊 EXAMPLE 1: Basic Data Analysis")
    print("="*60)
    
    nodes, elements, stress_data, damage_data, materials = load_dataset()
    
    # Basic statistics
    print(f"Dataset overview:")
    print(f"  • Nodes: {len(nodes):,}")
    print(f"  • Elements: {len(elements):,}")
    print(f"  • Materials: {len(materials)}")
    
    # Stress analysis
    von_mises = np.array(stress_data['von_mises']['values'])
    time = np.array(stress_data['von_mises']['time']) / 60  # Convert to minutes
    
    print(f"\nStress analysis:")
    print(f"  • Peak stress: {np.max(von_mises):.0f} MPa")
    print(f"  • Average final stress: {np.mean(von_mises[:, -1]):.0f} MPa")
    print(f"  • Stress range: {np.min(von_mises):.0f} - {np.max(von_mises):.0f} MPa")
    
    # Damage analysis
    damage = np.array(damage_data['damage_variable'])
    final_damage = damage[:, -1]
    
    print(f"\nDamage analysis:")
    print(f"  • Elements with damage: {np.sum(final_damage > 0):,}")
    print(f"  • Failed elements: {np.sum(final_damage >= 1.0):,}")
    print(f"  • Average damage: {np.mean(final_damage):.3f}")
    
    # Quick visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(time, np.mean(von_mises, axis=0), 'b-', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average Von Mises Stress (MPa)')
    plt.title('Stress Evolution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    damaged_fraction = np.sum(damage > 0, axis=0) / damage.shape[0]
    plt.plot(time, damaged_fraction * 100, 'r-', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Damaged Elements (%)')
    plt.title('Damage Progression')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fem_dataset/example_1_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Analysis plots saved as 'example_1_analysis.png'")

def example_2_machine_learning():
    """Example 2: Machine learning prediction of final stress"""
    print("\n" + "="*60)
    print("🤖 EXAMPLE 2: Machine Learning Prediction")
    print("="*60)
    
    nodes, elements, stress_data, damage_data, materials = load_dataset()
    
    # Prepare features and target
    von_mises = np.array(stress_data['von_mises']['values'])
    final_stress = von_mises[:, -1]  # Target: final stress
    
    # Features: element properties
    features = elements[['element_size', 'material_id', 'interface_refinement']].copy()
    
    # Add node-based features (element centroid approximation)
    element_coords = []
    for _, elem in elements.iterrows():
        # Simple approximation: use element ID to map to node region
        node_idx = elem['element_id'] % len(nodes)
        node = nodes.iloc[node_idx]
        element_coords.append([node['x'], node['y'], node['z']])
    
    coords_df = pd.DataFrame(element_coords, columns=['x', 'y', 'z'])
    features = pd.concat([features, coords_df], axis=1)
    
    # One-hot encode element types
    element_type_dummies = pd.get_dummies(elements['element_type'], prefix='type')
    features = pd.concat([features, element_type_dummies], axis=1)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, final_stress, test_size=0.2, random_state=42
    )
    
    # Train model
    print("\n🔄 Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Model performance:")
    print(f"  • R² score: {r2:.3f}")
    print(f"  • RMSE: {np.sqrt(mse):.0f} MPa")
    print(f"  • Mean absolute error: {np.mean(np.abs(y_test - y_pred)):.0f} MPa")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Top 5 most important features:")
    for i, (_, row) in enumerate(importance.head().iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Stress (MPa)')
    plt.ylabel('Predicted Stress (MPa)')
    plt.title(f'Stress Prediction (R² = {r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    top_features = importance.head(8)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fem_dataset/example_2_ml.png', dpi=300, bbox_inches='tight')
    print("🤖 ML results saved as 'example_2_ml.png'")

def example_3_failure_analysis():
    """Example 3: Failure mode analysis and prediction"""
    print("\n" + "="*60)
    print("💥 EXAMPLE 3: Failure Analysis")
    print("="*60)
    
    nodes, elements, stress_data, damage_data, materials = load_dataset()
    
    # Load failure predictions
    with open('fem_dataset/failure_predictions.json', 'r') as f:
        failure_data = json.load(f)
    
    # Analyze delamination
    delam_risk = np.array(failure_data['delamination']['risk_factor'])
    delam_initiated = np.array(failure_data['delamination']['initiated'])
    
    print(f"Delamination analysis:")
    print(f"  • Interface elements: {len(failure_data['delamination']['interface_elements'])}")
    print(f"  • Peak risk factor: {np.max(delam_risk):.2f}")
    print(f"  • Delaminated interfaces: {np.sum(delam_initiated[:, -1])}")
    
    # Analyze crack initiation
    crack_initiated = np.array(failure_data['crack_initiation']['initiated'])
    stress_amplitude = np.array(failure_data['crack_initiation']['stress_amplitude'])
    
    print(f"\nCrack initiation analysis:")
    print(f"  • Crack-prone elements: {len(failure_data['crack_initiation']['elements'])}")
    print(f"  • Cracked elements: {np.sum(crack_initiated[:, -1])}")
    print(f"  • Max stress amplitude: {np.max(stress_amplitude)/1e6:.0f} MPa")
    
    # Failure timeline
    time = np.array(failure_data['delamination']['time']) / 60
    delam_progression = np.sum(delam_initiated, axis=0) / delam_initiated.shape[0]
    crack_progression = np.sum(crack_initiated, axis=0) / crack_initiated.shape[0]
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(time, delam_progression * 100, 'r-', linewidth=2, label='Delamination')
    plt.plot(time, crack_progression * 100, 'b-', linewidth=2, label='Crack Initiation')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Failed Elements (%)')
    plt.title('Failure Mode Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(stress_amplitude / 1e6, bins=30, alpha=0.7, edgecolor='black')
    threshold = failure_data['crack_initiation']['parameters']['threshold_stress'] / 1e6
    plt.axvline(threshold, color='red', linestyle='--', 
               label=f'Threshold: {threshold:.0f} MPa')
    plt.xlabel('Stress Amplitude (MPa)')
    plt.ylabel('Number of Elements')
    plt.title('Stress Amplitude Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fem_dataset/example_3_failure.png', dpi=300, bbox_inches='tight')
    print("💥 Failure analysis saved as 'example_3_failure.png'")

def example_4_material_comparison():
    """Example 4: Material property comparison and analysis"""
    print("\n" + "="*60)
    print("🔬 EXAMPLE 4: Material Analysis")
    print("="*60)
    
    nodes, elements, stress_data, damage_data, materials = load_dataset()
    
    # Analyze stress by material
    von_mises = np.array(stress_data['von_mises']['values'])
    final_stress = von_mises[:, -1]
    
    material_stress = {}
    for mat_id in elements['material_id'].unique():
        mask = elements['material_id'] == mat_id
        mat_stress = final_stress[mask]
        material_stress[mat_id] = mat_stress
        
        # Find material name
        mat_name = "Unknown"
        for name, props in materials.items():
            if props['id'] == mat_id:
                mat_name = props['name']
                break
        
        print(f"Material {mat_id} ({mat_name}):")
        print(f"  • Elements: {np.sum(mask):,}")
        print(f"  • Avg stress: {np.mean(mat_stress):.0f} MPa")
        print(f"  • Max stress: {np.max(mat_stress):.0f} MPa")
        print(f"  • Std stress: {np.std(mat_stress):.0f} MPa")
    
    # Material properties comparison
    print(f"\nMaterial properties comparison:")
    prop_data = []
    for name, props in materials.items():
        if 'elastic' in props:
            prop_data.append({
                'Material': props['name'],
                'E (GPa)': props['elastic']['youngs_modulus'] / 1e9,
                'ν': props['elastic']['poissons_ratio'],
                'ρ (kg/m³)': props['elastic']['density']
            })
    
    prop_df = pd.DataFrame(prop_data)
    print(prop_df.to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(15, 4))
    
    # Stress distribution by material
    plt.subplot(1, 3, 1)
    mat_names = []
    stress_data_plot = []
    for mat_id, stress_vals in material_stress.items():
        mat_name = "Unknown"
        for name, props in materials.items():
            if props['id'] == mat_id:
                mat_name = props['name']
                break
        mat_names.append(f"Mat {mat_id}\n({mat_name})")
        stress_data_plot.append(stress_vals)
    
    plt.boxplot(stress_data_plot, labels=mat_names)
    plt.ylabel('Final Stress (MPa)')
    plt.title('Stress Distribution by Material')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Young's modulus comparison
    plt.subplot(1, 3, 2)
    if len(prop_df) > 0:
        plt.bar(range(len(prop_df)), prop_df['E (GPa)'])
        plt.xticks(range(len(prop_df)), prop_df['Material'], rotation=45)
        plt.ylabel('Young\'s Modulus (GPa)')
        plt.title('Material Stiffness')
        plt.grid(True, alpha=0.3)
    
    # Density comparison
    plt.subplot(1, 3, 3)
    if len(prop_df) > 0:
        plt.bar(range(len(prop_df)), prop_df['ρ (kg/m³)'])
        plt.xticks(range(len(prop_df)), prop_df['Material'], rotation=45)
        plt.ylabel('Density (kg/m³)')
        plt.title('Material Density')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fem_dataset/example_4_materials.png', dpi=300, bbox_inches='tight')
    print("🔬 Material analysis saved as 'example_4_materials.png'")

def main():
    """Run all examples"""
    print("🚀 FEM Dataset Usage Examples")
    print("="*80)
    
    try:
        example_1_basic_analysis()
        example_2_machine_learning()
        example_3_failure_analysis()
        example_4_material_comparison()
        
        print("\n" + "="*80)
        print("🎉 All examples completed successfully!")
        print("📁 Check the fem_dataset/ folder for generated plots")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()