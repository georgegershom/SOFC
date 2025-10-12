#!/usr/bin/env python3
"""
Flow Regime Classification Tools for Stratified Flow Dataset

This module provides machine learning tools for classifying flow regimes
based on experimental measurements and developing predictive models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class FlowRegimeClassifier:
    """
    Class for flow regime classification using machine learning
    """
    
    def __init__(self, data_path="../"):
        """Initialize with path to dataset"""
        self.data_path = data_path
        self.load_data()
        self.prepare_features()
    
    def load_data(self):
        """Load experimental data"""
        try:
            self.flow_data = pd.read_csv(f"{self.data_path}/experimental_data/flow_regime_characterization.csv")
            self.fluid_data = pd.read_csv(f"{self.data_path}/fluid_properties/fluid_conditions.csv")
            print("Data loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        # Merge flow and fluid data
        self.merged_data = self.flow_data.merge(self.fluid_data, on='experiment_id')
        
        # Define feature columns
        self.feature_columns = [
            'void_fraction_alpha',
            'superficial_gas_velocity_usg_ms',
            'superficial_liquid_velocity_usl_ms',
            'interface_height_mm',
            'wave_amplitude_mm',
            'gas_density_kg_m3',
            'liquid_density_kg_m3',
            'gas_viscosity_pa_s',
            'liquid_viscosity_pa_s',
            'temperature_c'
        ]
        
        # Calculate additional dimensionless parameters
        self.calculate_dimensionless_parameters()
        
        # Target variable
        self.target_column = 'flow_pattern'
        
        print(f"Features prepared: {len(self.feature_columns)} features")
        print(f"Flow patterns: {self.merged_data[self.target_column].unique()}")
    
    def calculate_dimensionless_parameters(self):
        """Calculate dimensionless parameters for flow regime classification"""
        g = 9.81  # gravity
        
        # Froude numbers
        self.merged_data['froude_gas'] = (
            self.merged_data['superficial_gas_velocity_usg_ms'] / 
            np.sqrt(g * self.merged_data['pipe_diameter_mm'] / 1000)
        )
        
        self.merged_data['froude_liquid'] = (
            self.merged_data['superficial_liquid_velocity_usl_ms'] / 
            np.sqrt(g * self.merged_data['pipe_diameter_mm'] / 1000)
        )
        
        # Reynolds numbers
        D = self.merged_data['pipe_diameter_mm'] / 1000  # convert to meters
        
        self.merged_data['reynolds_gas'] = (
            self.merged_data['gas_density_kg_m3'] * 
            self.merged_data['superficial_gas_velocity_usg_ms'] * D /
            self.merged_data['gas_viscosity_pa_s']
        )
        
        self.merged_data['reynolds_liquid'] = (
            self.merged_data['liquid_density_kg_m3'] * 
            self.merged_data['superficial_liquid_velocity_usl_ms'] * D /
            self.merged_data['liquid_viscosity_pa_s']
        )
        
        # Weber number
        self.merged_data['weber_number'] = (
            self.merged_data['liquid_density_kg_m3'] *
            (self.merged_data['superficial_gas_velocity_usg_ms'] - 
             self.merged_data['superficial_liquid_velocity_usl_ms'])**2 * D /
            self.merged_data['surface_tension_n_m']
        )
        
        # Add dimensionless parameters to feature list
        self.feature_columns.extend([
            'froude_gas', 'froude_liquid', 
            'reynolds_gas', 'reynolds_liquid', 
            'weber_number'
        ])
    
    def plot_flow_map(self):
        """Plot traditional flow regime map"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for flow patterns
        pattern_colors = {'smooth_stratified': 'blue', 'wavy_stratified': 'red'}
        
        for pattern in self.merged_data['flow_pattern'].unique():
            data = self.merged_data[self.merged_data['flow_pattern'] == pattern]
            ax.scatter(data['superficial_gas_velocity_usg_ms'],
                      data['superficial_liquid_velocity_usl_ms'],
                      c=pattern_colors[pattern], label=pattern.replace('_', ' ').title(),
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Superficial Gas Velocity (m/s)', fontsize=14)
        ax.set_ylabel('Superficial Liquid Velocity (m/s)', fontsize=14)
        ax.set_title('Flow Regime Map', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def train_classifiers(self):
        """Train multiple classifiers for flow regime prediction"""
        # Prepare data
        X = self.merged_data[self.feature_columns]
        y = self.merged_data[self.target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        self.trained_models = {}
        self.results = {}
        
        print("Training classifiers...")
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'SVM':
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
            # Store model
            self.trained_models[name] = clf
            
            # Evaluate
            accuracy = clf.score(X_test_scaled if name == 'SVM' else X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                clf, X_train_scaled if name == 'SVM' else X_train, y_train, cv=5
            )
            
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"Test Accuracy: {accuracy:.3f}")
            print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return X_test, y_test
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all classifiers"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Smooth', 'Wavy'],
                       yticklabels=['Smooth', 'Wavy'])
            
            axes[i].set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}', fontsize=14)
            axes[i].set_xlabel('Predicted', fontsize=12)
            axes[i].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest"""
        if 'Random Forest' not in self.trained_models:
            print("Random Forest model not trained yet!")
            return
        
        rf_model = self.trained_models['Random Forest']
        importances = rf_model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(feature_importance_df)), 
                      feature_importance_df['importance'])
        ax.set_yticks(range(len(feature_importance_df)))
        ax.set_yticklabels(feature_importance_df['feature'])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Feature Importance for Flow Regime Classification', fontsize=14)
        
        # Color bars by importance
        colors = plt.cm.viridis(feature_importance_df['importance'] / 
                               feature_importance_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def predict_flow_regime(self, usg, usl, alpha, model_name='Random Forest'):
        """
        Predict flow regime for given conditions
        
        Parameters:
        -----------
        usg : float
            Superficial gas velocity (m/s)
        usl : float
            Superficial liquid velocity (m/s)
        alpha : float
            Void fraction
        model_name : str
            Name of model to use for prediction
        """
        if model_name not in self.trained_models:
            print(f"Model {model_name} not available!")
            return None
        
        # Create feature vector (simplified - using average values for other features)
        avg_values = self.merged_data[self.feature_columns].mean()
        
        # Update with provided values
        feature_vector = avg_values.copy()
        feature_vector['superficial_gas_velocity_usg_ms'] = usg
        feature_vector['superficial_liquid_velocity_usl_ms'] = usl
        feature_vector['void_fraction_alpha'] = alpha
        
        # Recalculate dimensionless parameters
        g = 9.81
        D = 0.1016  # pipe diameter in meters
        
        feature_vector['froude_gas'] = usg / np.sqrt(g * D)
        feature_vector['froude_liquid'] = usl / np.sqrt(g * D)
        feature_vector['reynolds_gas'] = (feature_vector['gas_density_kg_m3'] * usg * D / 
                                        feature_vector['gas_viscosity_pa_s'])
        feature_vector['reynolds_liquid'] = (feature_vector['liquid_density_kg_m3'] * usl * D / 
                                           feature_vector['liquid_viscosity_pa_s'])
        feature_vector['weber_number'] = (feature_vector['liquid_density_kg_m3'] * 
                                        (usg - usl)**2 * D / 
                                        feature_vector['surface_tension_n_m'])
        
        # Make prediction
        model = self.trained_models[model_name]
        
        if model_name == 'SVM':
            feature_scaled = self.scaler.transform([feature_vector])
            prediction = model.predict(feature_scaled)[0]
            probability = model.decision_function(feature_scaled)[0]
        else:
            prediction = model.predict([feature_vector])[0]
            probability = model.predict_proba([feature_vector])[0].max()
        
        return prediction, probability

def main():
    """Main function to demonstrate classification capabilities"""
    classifier = FlowRegimeClassifier()
    
    print("=== Flow Regime Classification Analysis ===\n")
    
    # 1. Plot flow regime map
    print("1. Plotting flow regime map...")
    classifier.plot_flow_map()
    
    # 2. Train classifiers
    print("2. Training classifiers...")
    X_test, y_test = classifier.train_classifiers()
    
    # 3. Plot confusion matrices
    print("3. Plotting confusion matrices...")
    classifier.plot_confusion_matrices()
    
    # 4. Feature importance analysis
    print("4. Analyzing feature importance...")
    importance_df = classifier.feature_importance_analysis()
    print("\nTop 5 most important features:")
    print(importance_df.head())
    
    # 5. Example predictions
    print("\n5. Example predictions:")
    test_conditions = [
        (0.5, 1.2, 0.15),  # Low gas velocity
        (2.0, 0.8, 0.35),  # High gas velocity
        (1.0, 1.0, 0.25)   # Moderate conditions
    ]
    
    for usg, usl, alpha in test_conditions:
        prediction, confidence = classifier.predict_flow_regime(usg, usl, alpha)
        print(f"USG={usg} m/s, USL={usl} m/s, α={alpha}: {prediction} (confidence: {confidence:.3f})")
    
    print("\nClassification analysis complete!")

if __name__ == "__main__":
    main()