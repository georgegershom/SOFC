"""
Usage Examples for Residual Stress Dataset
=========================================

This file contains comprehensive examples of how to use the residual stress dataset
for various machine learning and analysis tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class ResidualStressMLExample:
    """
    Comprehensive machine learning examples for residual stress prediction.
    """
    
    def __init__(self, dataset_path='residual_stress_dataset_10000.csv'):
        """Load and prepare the dataset."""
        print("Loading residual stress dataset...")
        self.df = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        
        # Prepare features and targets
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare features and target variables."""
        
        # Define feature categories
        self.geometric_features = [col for col in self.df.columns 
                                 if any(x in col for x in ['length', 'width', 'thickness', 'density'])]
        
        self.material_features = [col for col in self.df.columns 
                                if any(x in col for x in ['youngs_modulus', 'cte', 'poisson', 'shrinkage', 'activation'])]
        
        self.process_features = [col for col in self.df.columns 
                               if any(x in col for x in ['temp', 'rate', 'time', 'pressure', 'atmosphere'])]
        
        # All input features
        self.input_features = self.geometric_features + self.material_features + self.process_features
        
        # Target variables (stress outputs)
        self.stress_targets = [col for col in self.df.columns 
                             if 'residual_stress_total' in col]
        
        self.von_mises_targets = [col for col in self.df.columns 
                                if 'von_mises_stress' in col]
        
        print(f"Input features: {len(self.input_features)}")
        print(f"Stress targets: {len(self.stress_targets)}")
        print(f"von Mises targets: {len(self.von_mises_targets)}")
        
        # Prepare feature matrix
        self.X = self.df[self.input_features].copy()
        
        # Handle any missing values
        self.X = self.X.fillna(self.X.mean())
        
    def example_1_basic_regression(self):
        """Example 1: Basic regression for single target prediction."""
        
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic Regression for Anode Stress Prediction")
        print("="*60)
        
        # Target: Anode residual stress
        target = 'anode_residual_stress_total'
        y = self.df[target].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for linear models, original for tree-based
            if 'Regression' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'model': model
            }
            
            print(f"  RMSE: {rmse/1e6:.2f} MPa")
            print(f"  MAE:  {mae/1e6:.2f} MPa")
            print(f"  R²:   {r2:.3f}")
        
        # Best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['R²']:.3f})")
        
        return results, best_model, scaler
    
    def example_2_multi_target_prediction(self):
        """Example 2: Multi-target prediction for all stress components."""
        
        print("\n" + "="*60)
        print("EXAMPLE 2: Multi-Target Prediction (All Stress Components)")
        print("="*60)
        
        # All stress targets (excluding electrolyte which is zero)
        targets = ['anode_residual_stress_total', 'cathode_residual_stress_total']
        Y = self.df[targets].values
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, Y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest for multi-target
        print("Training Multi-Target Random Forest...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train)
        
        # Predictions
        Y_pred = model.predict(X_test)
        
        # Calculate metrics for each target
        for i, target in enumerate(targets):
            y_true = Y_test[:, i]
            y_pred = Y_pred[:, i]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            print(f"\n{target}:")
            print(f"  RMSE: {rmse/1e6:.2f} MPa")
            print(f"  R²:   {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.input_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.3f}")
        
        return model, feature_importance
    
    def example_3_feature_engineering(self):
        """Example 3: Advanced feature engineering for better predictions."""
        
        print("\n" + "="*60)
        print("EXAMPLE 3: Advanced Feature Engineering")
        print("="*60)
        
        # Create engineered features
        X_engineered = self.X.copy()
        
        # CTE mismatch features
        X_engineered['cte_mismatch_anode_electrolyte'] = (
            self.df['anode_cte'] - self.df['electrolyte_cte']
        )
        X_engineered['cte_mismatch_cathode_electrolyte'] = (
            self.df['cathode_cte'] - self.df['electrolyte_cte']
        )
        
        # Shrinkage mismatch features
        X_engineered['shrinkage_mismatch_anode_electrolyte'] = (
            self.df['anode_sintering_shrinkage'] - self.df['electrolyte_sintering_shrinkage']
        )
        X_engineered['shrinkage_mismatch_cathode_electrolyte'] = (
            self.df['cathode_sintering_shrinkage'] - self.df['electrolyte_sintering_shrinkage']
        )
        
        # Geometric ratios
        X_engineered['aspect_ratio'] = self.df['plate_length'] / self.df['plate_width']
        X_engineered['anode_electrolyte_thickness_ratio'] = (
            self.df['anode_thickness'] / self.df['electrolyte_thickness']
        )
        X_engineered['cathode_electrolyte_thickness_ratio'] = (
            self.df['cathode_thickness'] / self.df['electrolyte_thickness']
        )
        
        # Total thickness
        X_engineered['total_thickness'] = (
            self.df['anode_thickness'] + 
            self.df['electrolyte_thickness'] + 
            self.df['cathode_thickness']
        )
        
        # Temperature-related features
        X_engineered['cooling_rate_ratio'] = (
            self.df['cooling_rate'] / self.df['heating_rate']
        )
        
        # Modulus mismatch
        X_engineered['modulus_mismatch_anode_electrolyte'] = (
            self.df['anode_youngs_modulus_rt'] - self.df['electrolyte_youngs_modulus_rt']
        )
        
        print(f"Original features: {len(self.input_features)}")
        print(f"Engineered features: {X_engineered.shape[1]}")
        print(f"New features added: {X_engineered.shape[1] - len(self.input_features)}")
        
        # Compare models with and without feature engineering
        target = 'anode_residual_stress_total'
        y = self.df[target].values
        
        results_comparison = {}
        
        for name, X_data in [('Original Features', self.X), ('Engineered Features', X_engineered)]:
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results_comparison[name] = {'RMSE': rmse, 'R²': r2}
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse/1e6:.2f} MPa")
            print(f"  R²:   {r2:.3f}")
        
        # Improvement
        improvement = (
            results_comparison['Engineered Features']['R²'] - 
            results_comparison['Original Features']['R²']
        )
        print(f"\nImprovement in R²: {improvement:.3f}")
        
        return X_engineered, results_comparison
    
    def example_4_stress_classification(self):
        """Example 4: Classification of high-stress conditions."""
        
        print("\n" + "="*60)
        print("EXAMPLE 4: Classification of High-Stress Conditions")
        print("="*60)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Create binary classification target (high stress vs low stress)
        stress_col = 'anode_von_mises_stress'
        stress_values = self.df[stress_col].values
        
        # Define threshold (e.g., 75th percentile)
        threshold = np.percentile(stress_values, 75)
        y_binary = (stress_values > threshold).astype(int)
        
        print(f"Threshold for high stress: {threshold/1e6:.2f} MPa")
        print(f"High stress samples: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Predictions
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:, 1]
        
        # Metrics
        print("\nClassification Results:")
        print(classification_report(y_test, y_pred, target_names=['Low Stress', 'High Stress']))
        
        # Feature importance for classification
        feature_importance = pd.DataFrame({
            'feature': self.input_features,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Features for High Stress Prediction:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']:<30} {row['importance']:.3f}")
        
        return classifier, feature_importance, threshold
    
    def example_5_uncertainty_quantification(self):
        """Example 5: Uncertainty quantification using ensemble methods."""
        
        print("\n" + "="*60)
        print("EXAMPLE 5: Uncertainty Quantification")
        print("="*60)
        
        from sklearn.ensemble import RandomForestRegressor
        
        target = 'anode_residual_stress_total'
        y = self.df[target].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y, test_size=0.2, random_state=42
        )
        
        # Train ensemble of models with different random states
        n_models = 10
        predictions = []
        
        print(f"Training ensemble of {n_models} models...")
        
        for i in range(n_models):
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=i, 
                bootstrap=True
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Calculate ensemble statistics
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals (assuming normal distribution)
        confidence_level = 0.95
        z_score = 1.96  # for 95% confidence
        
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
        r2 = r2_score(y_test, mean_pred)
        
        print(f"\nEnsemble Results:")
        print(f"  RMSE: {rmse/1e6:.2f} MPa")
        print(f"  R²:   {r2:.3f}")
        print(f"  Mean prediction uncertainty: {np.mean(std_pred)/1e6:.2f} MPa")
        
        # Coverage analysis (what fraction of true values fall within confidence intervals)
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        print(f"  {confidence_level*100:.0f}% confidence interval coverage: {coverage*100:.1f}%")
        
        return mean_pred, std_pred, lower_bound, upper_bound
    
    def example_6_process_optimization(self):
        """Example 6: Process optimization to minimize stress."""
        
        print("\n" + "="*60)
        print("EXAMPLE 6: Process Optimization")
        print("="*60)
        
        from scipy.optimize import minimize
        
        # Train a model first
        target = 'anode_von_mises_stress'
        y = self.df[target].values
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X, y)
        
        print(f"Model trained for {target}")
        print(f"Model R² score: {model.score(self.X, y):.3f}")
        
        # Define optimization problem
        # We'll optimize process parameters while keeping material/geometric fixed
        
        # Fixed parameters (use median values)
        fixed_params = {}
        for col in self.geometric_features + self.material_features:
            fixed_params[col] = self.df[col].median()
        
        # Process parameters to optimize
        process_bounds = {
            'max_sintering_temp': (1573, 1773),  # K
            'heating_rate': (1, 10),             # K/min
            'cooling_rate': (1, 5),              # K/min
            'hold_time': (1, 8),                 # hours
            'atmosphere_oxygen_partial_pressure': (1e-10, 0.21)  # atm
        }
        
        def objective_function(process_params):
            """Objective: minimize von Mises stress."""
            
            # Create full parameter vector
            full_params = fixed_params.copy()
            
            for i, param in enumerate(process_bounds.keys()):
                full_params[param] = process_params[i]
            
            # Convert to feature vector in correct order
            feature_vector = np.array([full_params[col] for col in self.input_features])
            
            # Predict stress
            predicted_stress = model.predict(feature_vector.reshape(1, -1))[0]
            
            return predicted_stress
        
        # Initial guess (median values)
        x0 = [self.df[param].median() for param in process_bounds.keys()]
        
        # Bounds for optimization
        bounds = list(process_bounds.values())
        
        print(f"\nOptimizing process parameters...")
        print(f"Initial stress prediction: {objective_function(x0)/1e6:.2f} MPa")
        
        # Optimize
        result = minimize(
            objective_function, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        if result.success:
            optimal_stress = result.fun
            optimal_params = result.x
            
            print(f"\nOptimization successful!")
            print(f"Optimized stress: {optimal_stress/1e6:.2f} MPa")
            print(f"Stress reduction: {(objective_function(x0) - optimal_stress)/1e6:.2f} MPa")
            
            print(f"\nOptimal process parameters:")
            for param, value in zip(process_bounds.keys(), optimal_params):
                if 'temp' in param:
                    print(f"  {param}: {value:.1f} K ({value-273:.1f} °C)")
                elif 'pressure' in param:
                    print(f"  {param}: {value:.2e} atm")
                else:
                    print(f"  {param}: {value:.2f}")
        else:
            print(f"Optimization failed: {result.message}")
        
        return result
    
    def run_all_examples(self):
        """Run all examples in sequence."""
        
        print("COMPREHENSIVE MACHINE LEARNING EXAMPLES")
        print("FOR RESIDUAL STRESS DATASET")
        print("="*80)
        
        # Example 1: Basic regression
        results_1, best_model, scaler = self.example_1_basic_regression()
        
        # Example 2: Multi-target prediction
        model_2, importance_2 = self.example_2_multi_target_prediction()
        
        # Example 3: Feature engineering
        X_eng, results_3 = self.example_3_feature_engineering()
        
        # Example 4: Classification
        classifier_4, importance_4, threshold_4 = self.example_4_stress_classification()
        
        # Example 5: Uncertainty quantification
        mean_pred, std_pred, lower, upper = self.example_5_uncertainty_quantification()
        
        # Example 6: Process optimization
        opt_result = self.example_6_process_optimization()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return {
            'basic_regression': (results_1, best_model, scaler),
            'multi_target': (model_2, importance_2),
            'feature_engineering': (X_eng, results_3),
            'classification': (classifier_4, importance_4, threshold_4),
            'uncertainty': (mean_pred, std_pred, lower, upper),
            'optimization': opt_result
        }


def main():
    """Main function to run examples."""
    
    # Check if dataset exists
    import os
    dataset_file = 'residual_stress_dataset_10000.csv'
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found!")
        print("Please run residual_stress_dataset_generator.py first to generate the dataset.")
        return
    
    # Initialize and run examples
    ml_examples = ResidualStressMLExample(dataset_file)
    
    # Run all examples
    results = ml_examples.run_all_examples()
    
    print(f"\nAll examples completed! Results saved in memory.")
    print(f"You can access individual results using the returned dictionary.")


if __name__ == "__main__":
    main()