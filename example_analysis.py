"""
Example Analysis Script for Sintering Dataset
Demonstrates common use cases and analysis workflows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_and_explore():
    """Load and perform basic exploration of the dataset."""
    print("="*80)
    print("EXAMPLE 1: LOADING AND BASIC EXPLORATION")
    print("="*80)
    
    df = pd.read_csv('sintering_process_microstructure_dataset.csv')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    print("\n--- First 3 samples ---")
    print(df.head(3))
    
    print("\n--- Data types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Missing values ---")
    print(f"Total missing: {df.isnull().sum().sum()}")
    
    return df

def find_optimal_conditions(df):
    """Find optimal sintering conditions for high density."""
    print("\n" + "="*80)
    print("EXAMPLE 2: FINDING OPTIMAL CONDITIONS FOR HIGH DENSITY")
    print("="*80)
    
    # Define target: relative density > 0.75
    high_density_threshold = 0.75
    high_density_samples = df[df['Final_Relative_Density'] > high_density_threshold]
    
    print(f"\nTarget: Final Relative Density > {high_density_threshold}")
    print(f"Samples meeting criteria: {len(high_density_samples)} / {len(df)} ({len(high_density_samples)/len(df)*100:.1f}%)")
    
    if len(high_density_samples) > 0:
        print("\n--- Average process parameters for high density ---")
        key_params = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa',
                     'Initial_Relative_Density', 'Ramp_Rate_C_per_min', 'Cooling_Rate_C_per_min']
        
        comparison = pd.DataFrame({
            'High Density (>0.75)': high_density_samples[key_params].mean(),
            'All Samples': df[key_params].mean(),
            'Difference': high_density_samples[key_params].mean() - df[key_params].mean()
        })
        print(comparison)
        
        print("\n--- Atmosphere distribution for high density ---")
        print(high_density_samples['Atmosphere'].value_counts())
    
    return high_density_samples

def predict_density(df):
    """Train a model to predict final density from process parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 3: MACHINE LEARNING - PREDICTING FINAL DENSITY")
    print("="*80)
    
    # Prepare features
    input_features = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa',
                     'Initial_Relative_Density', 'Ramp_Rate_C_per_min', 'Cooling_Rate_C_per_min']
    
    # One-hot encode atmosphere
    df_encoded = pd.get_dummies(df, columns=['Atmosphere'], prefix='Atm')
    atmosphere_cols = [col for col in df_encoded.columns if col.startswith('Atm_')]
    
    all_features = input_features + atmosphere_cols
    
    X = df_encoded[all_features]
    y = df_encoded['Final_Relative_Density']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {len(all_features)}")
    
    # Train Random Forest
    print("\n--- Training Random Forest Model ---")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\nModel Performance:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    
    # Feature importance
    print("\n--- Feature Importance (Top 10) ---")
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Cross-validation
    print("\n--- 5-Fold Cross-Validation ---")
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"CV R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return rf_model, X_test, y_test, y_pred_test

def predict_grain_size(df):
    """Train a model to predict grain size from process parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 4: PREDICTING GRAIN SIZE")
    print("="*80)
    
    # Features
    input_features = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa',
                     'Initial_Relative_Density', 'Thermal_Load_C_hours']
    
    X = df[input_features]
    y = df['Mean_Grain_Size_um']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting
    print("\n--- Training Gradient Boosting Model ---")
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                        max_depth=5, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_test = gb_model.predict(X_test_scaled)
    
    # Evaluate
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f} μm")
    
    # Feature importance
    print("\n--- Feature Importance ---")
    for feat, imp in zip(input_features, gb_model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    return gb_model, scaler

def sensitivity_analysis(df):
    """Perform sensitivity analysis on key parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 5: PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Calculate correlations with final density
    params = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa',
             'Initial_Relative_Density', 'Ramp_Rate_C_per_min']
    
    print("\n--- Correlation with Final Relative Density ---")
    correlations = []
    for param in params:
        corr = df[param].corr(df['Final_Relative_Density'])
        correlations.append((param, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for param, corr in correlations:
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"  {param:30s}: {corr:+.3f} {direction} ({strength})")
    
    # Calculate correlations with grain size
    print("\n--- Correlation with Mean Grain Size ---")
    correlations_gs = []
    for param in params:
        corr = df[param].corr(df['Mean_Grain_Size_um'])
        correlations_gs.append((param, corr))
    
    correlations_gs.sort(key=lambda x: abs(x[1]), reverse=True)
    for param, corr in correlations_gs:
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"  {param:30s}: {corr:+.3f} {direction} ({strength})")

def multi_objective_optimization(df):
    """Find conditions that balance high density and small grain size."""
    print("\n" + "="*80)
    print("EXAMPLE 6: MULTI-OBJECTIVE OPTIMIZATION")
    print("="*80)
    
    print("\nObjective: Maximize density while minimizing grain size")
    print("Target: Relative Density > 0.70 AND Grain Size < 1.5 μm")
    
    # Define multi-objective criteria
    optimal_samples = df[(df['Final_Relative_Density'] > 0.70) & 
                        (df['Mean_Grain_Size_um'] < 1.5)]
    
    print(f"\nSamples meeting criteria: {len(optimal_samples)} / {len(df)} ({len(optimal_samples)/len(df)*100:.1f}%)")
    
    if len(optimal_samples) > 0:
        print("\n--- Recommended Process Parameters ---")
        key_params = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa']
        
        print("\nParameter ranges:")
        for param in key_params:
            print(f"  {param}:")
            print(f"    Min: {optimal_samples[param].min():.1f}")
            print(f"    Mean: {optimal_samples[param].mean():.1f}")
            print(f"    Max: {optimal_samples[param].max():.1f}")
        
        print("\nRecommended atmosphere:")
        print(optimal_samples['Atmosphere'].value_counts())
        
        print("\n--- Expected Outcomes ---")
        print(f"  Final Relative Density: {optimal_samples['Final_Relative_Density'].mean():.3f} ± {optimal_samples['Final_Relative_Density'].std():.3f}")
        print(f"  Mean Grain Size: {optimal_samples['Mean_Grain_Size_um'].mean():.2f} ± {optimal_samples['Mean_Grain_Size_um'].std():.2f} μm")
        print(f"  Final Porosity: {optimal_samples['Final_Porosity_percent'].mean():.2f} ± {optimal_samples['Final_Porosity_percent'].std():.2f} %")
    else:
        print("\nNo samples meet both criteria. Consider relaxing constraints.")

def main():
    """Run all example analyses."""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "SINTERING DATASET - EXAMPLE ANALYSES" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    
    # Load data
    df = load_and_explore()
    
    # Example analyses
    find_optimal_conditions(df)
    predict_density(df)
    predict_grain_size(df)
    sensitivity_analysis(df)
    multi_objective_optimization(df)
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Modify the analysis scripts for your specific use case")
    print("  2. Try different machine learning algorithms")
    print("  3. Perform your own feature engineering")
    print("  4. Export results for further analysis")
    print("  5. Integrate with your simulation workflow")
    print("\n")

if __name__ == "__main__":
    main()