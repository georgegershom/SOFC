#!/usr/bin/env python3
"""
Machine Learning Examples for Stratified Flow Dataset

Demonstrates ML applications including:
1. Flow pattern classification
2. Void fraction prediction
3. Attenuation coefficient prediction

Author: Generated for PhD Research
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MLAnalysis:
    """Machine Learning analysis for stratified flow data."""
    
    def __init__(self, data_dir='stratified_flow_dataset'):
        """Initialize ML analysis."""
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        """Load datasets."""
        print("Loading data for ML analysis...")
        self.flow_data = pd.read_csv(
            os.path.join(self.data_dir, 'flow_regime_characterization.csv')
        )
        self.attenuation_data = pd.read_csv(
            os.path.join(self.data_dir, 'acoustic_attenuation_data.csv')
        )
        self.turbulence_data = pd.read_csv(
            os.path.join(self.data_dir, 'turbulence_shear_data.csv')
        )
        print(f"✓ Data loaded successfully\n")
    
    def flow_pattern_classification(self):
        """
        Task 1: Classify flow patterns using flow parameters.
        """
        print("="*70)
        print("TASK 1: FLOW PATTERN CLASSIFICATION")
        print("="*70 + "\n")
        
        # Prepare features
        features = ['U_SG', 'U_SL', 'void_fraction', 'wave_amplitude', 
                   'wave_frequency', 'temperature', 'pressure']
        X = self.flow_data[features]
        y = self.flow_data['flow_pattern']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        
        # Results
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=clf.classes_, yticklabels=clf.classes_)
        plt.title('Flow Pattern Classification - Confusion Matrix', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('ml_flow_pattern_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_flow_pattern_confusion_matrix.png")
        plt.show()
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'], color='steelblue', edgecolor='black')
        plt.xlabel('Feature Importance', fontweight='bold')
        plt.title('Feature Importance for Flow Pattern Classification', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('ml_feature_importance_classification.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_feature_importance_classification.png\n")
        plt.show()
        
        return clf, scaler
    
    def void_fraction_prediction(self):
        """
        Task 2: Predict void fraction from acoustic measurements.
        """
        print("="*70)
        print("TASK 2: VOID FRACTION PREDICTION FROM ACOUSTIC DATA")
        print("="*70 + "\n")
        
        # Merge datasets
        merged = self.attenuation_data.merge(
            self.flow_data[['experiment_id', 'void_fraction', 'U_SG', 'U_SL']], 
            on='experiment_id'
        )
        
        # Pivot to get attenuation at different frequencies as features
        pivot = merged.pivot_table(
            index='experiment_id', 
            columns='frequency', 
            values='attenuation_coefficient'
        ).reset_index()
        
        # Merge back with void fraction
        data = pivot.merge(
            self.flow_data[['experiment_id', 'void_fraction', 'U_SG', 'U_SL']], 
            on='experiment_id'
        )
        
        # Features: attenuation at different frequencies
        freq_cols = [100, 500, 1000, 2000, 5000, 10000]
        X = data[freq_cols + ['U_SG', 'U_SL']]
        y = data['void_fraction']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest regressor
        reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
        reg.fit(X_train, y_train)
        
        # Predictions
        y_pred = reg.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}\n")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, s=100, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('True Void Fraction', fontweight='bold', fontsize=12)
        plt.ylabel('Predicted Void Fraction', fontweight='bold', fontsize=12)
        plt.title(f'Void Fraction Prediction (R²={r2:.3f}, MAE={mae:.3f})', 
                 fontweight='bold', fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ml_void_fraction_prediction.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_void_fraction_prediction.png")
        plt.show()
        
        # Feature importance
        feature_names = [f'{f} Hz' for f in freq_cols] + ['U_SG', 'U_SL']
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'], color='coral', edgecolor='black')
        plt.xlabel('Feature Importance', fontweight='bold')
        plt.title('Feature Importance for Void Fraction Prediction', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('ml_feature_importance_void_fraction.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_feature_importance_void_fraction.png\n")
        plt.show()
        
        return reg
    
    def attenuation_prediction(self):
        """
        Task 3: Predict attenuation coefficient from flow parameters.
        """
        print("="*70)
        print("TASK 3: ATTENUATION COEFFICIENT PREDICTION")
        print("="*70 + "\n")
        
        # Merge datasets
        merged = self.attenuation_data.merge(
            self.flow_data[['experiment_id', 'U_SG', 'U_SL', 'void_fraction', 
                           'flow_pattern', 'wave_amplitude', 'temperature']], 
            on='experiment_id'
        )
        
        # Encode flow pattern
        merged['flow_pattern_encoded'] = (merged['flow_pattern'] == 'wavy_stratified').astype(int)
        
        # Features
        features = ['frequency', 'U_SG', 'U_SL', 'void_fraction', 
                   'flow_pattern_encoded', 'wave_amplitude', 'temperature']
        X = merged[features]
        y = merged['attenuation_coefficient']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest regressor
        reg = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=20)
        reg.fit(X_train, y_train)
        
        # Predictions
        y_pred = reg.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.6f} Np/m\n")
        
        # Plot predictions vs actual (log scale)
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, s=80, edgecolors='k', c=X_test['frequency'], cmap='viridis')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('True Attenuation Coefficient (Np/m)', fontweight='bold', fontsize=12)
        plt.ylabel('Predicted Attenuation Coefficient (Np/m)', fontweight='bold', fontsize=12)
        plt.title(f'Attenuation Prediction (R²={r2:.3f}, MAE={mae:.4f})', 
                 fontweight='bold', fontsize=13)
        plt.colorbar(label='Frequency (Hz)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ml_attenuation_prediction.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_attenuation_prediction.png")
        plt.show()
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'], color='lightgreen', edgecolor='black')
        plt.xlabel('Feature Importance', fontweight='bold')
        plt.title('Feature Importance for Attenuation Prediction', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('ml_feature_importance_attenuation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_feature_importance_attenuation.png\n")
        plt.show()
        
        return reg
    
    def mechanism_contribution_analysis(self):
        """
        Task 4: Analyze contribution of different attenuation mechanisms.
        """
        print("="*70)
        print("TASK 4: ATTENUATION MECHANISM CONTRIBUTION ANALYSIS")
        print("="*70 + "\n")
        
        # Calculate percentage contributions
        self.attenuation_data['total_atten'] = (
            self.attenuation_data['viscous_atten_contribution'] + 
            self.attenuation_data['scattering_atten_contribution'] + 
            self.attenuation_data['turbulence_atten_contribution']
        )
        
        self.attenuation_data['viscous_pct'] = (
            100 * self.attenuation_data['viscous_atten_contribution'] / 
            self.attenuation_data['total_atten']
        )
        self.attenuation_data['scattering_pct'] = (
            100 * self.attenuation_data['scattering_atten_contribution'] / 
            self.attenuation_data['total_atten']
        )
        self.attenuation_data['turbulence_pct'] = (
            100 * self.attenuation_data['turbulence_atten_contribution'] / 
            self.attenuation_data['total_atten']
        )
        
        # Group by frequency
        freq_analysis = self.attenuation_data.groupby('frequency').agg({
            'viscous_pct': 'mean',
            'scattering_pct': 'mean',
            'turbulence_pct': 'mean'
        })
        
        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 7))
        
        frequencies = freq_analysis.index
        width = 0.6
        
        ax.bar(frequencies, freq_analysis['viscous_pct'], width, 
               label='Viscous Absorption', color='steelblue')
        ax.bar(frequencies, freq_analysis['scattering_pct'], width,
               bottom=freq_analysis['viscous_pct'],
               label='Scattering', color='coral')
        ax.bar(frequencies, freq_analysis['turbulence_pct'], width,
               bottom=freq_analysis['viscous_pct'] + freq_analysis['scattering_pct'],
               label='Turbulence', color='lightgreen')
        
        ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Contribution (%)', fontweight='bold', fontsize=12)
        ax.set_title('Attenuation Mechanism Contributions by Frequency', 
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_xscale('log')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_mechanism_contributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: ml_mechanism_contributions.png\n")
        plt.show()
        
        print("Average mechanism contributions across all frequencies:")
        print(f"  Viscous: {freq_analysis['viscous_pct'].mean():.1f}%")
        print(f"  Scattering: {freq_analysis['scattering_pct'].mean():.1f}%")
        print(f"  Turbulence: {freq_analysis['turbulence_pct'].mean():.1f}%\n")
    
    def run_all_analyses(self):
        """Run all ML analyses."""
        print("\n" + "="*70)
        print("MACHINE LEARNING ANALYSIS FOR STRATIFIED FLOW DATASET")
        print("="*70 + "\n")
        
        # Task 1: Flow pattern classification
        clf, scaler = self.flow_pattern_classification()
        
        # Task 2: Void fraction prediction
        reg_void = self.void_fraction_prediction()
        
        # Task 3: Attenuation prediction
        reg_atten = self.attenuation_prediction()
        
        # Task 4: Mechanism analysis
        self.mechanism_contribution_analysis()
        
        print("="*70)
        print("ALL ML ANALYSES COMPLETE!")
        print("="*70 + "\n")
        
        print("Generated files:")
        print("  - ml_flow_pattern_confusion_matrix.png")
        print("  - ml_feature_importance_classification.png")
        print("  - ml_void_fraction_prediction.png")
        print("  - ml_feature_importance_void_fraction.png")
        print("  - ml_attenuation_prediction.png")
        print("  - ml_feature_importance_attenuation.png")
        print("  - ml_mechanism_contributions.png\n")


def main():
    """Main function."""
    analyzer = MLAnalysis()
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()
