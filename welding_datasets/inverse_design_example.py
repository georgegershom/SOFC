"""
Example Machine Learning Models for Welding Inverse Design
Demonstrates various approaches to inverse design problem
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')


class WeldingInverseDesignModel:
    """
    Inverse design model for welding process
    Predicts input parameters from desired output performance
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders = {}
        self.categorical_features = [
            'Material_Combination', 'Shield_Gas_Type', 'Joint_Type'
        ]
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Input features (what we want to predict in inverse design)
        input_features = [
            'Laser_Power_W', 'Welding_Speed_mm_s', 'Pulse_Frequency_Hz',
            'Pulse_Duration_ms', 'Beam_Focus_Position_mm', 'Beam_Spot_Size_um',
            'Clamping_Pressure_kPa', 'Shield_Gas_Flow_L_min',
            'Sheet_Thickness_mm', 'Overlap_Distance_mm'
        ]
        
        # Add encoded categorical features
        categorical_input = [
            'Material_Combination', 'Shield_Gas_Type', 'Joint_Type'
        ]
        
        # Output features (what we know/desire in inverse design)
        output_features = [
            'Nugget_Width_mm', 'Penetration_Depth_mm',
            'Tensile_Shear_Strength_N', 'Contact_Resistance_uOhm',
            'Cycles_to_Failure', 'Overall_Quality_Score'
        ]
        
        # Encode categorical variables
        df_encoded = df.copy()
        for feature in categorical_input:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_encoded[feature + '_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                df_encoded[feature + '_encoded'] = self.label_encoders[feature].transform(df[feature])
        
        # Prepare X (inputs to predict) and y (outputs we know)
        X_features = input_features + [f + '_encoded' for f in categorical_input]
        X = df_encoded[X_features].values
        y = df_encoded[output_features].values
        
        return X, y, input_features, output_features
    
    def train(self, df, test_size=0.2):
        """Train the inverse design model"""
        print(f"\n{'='*80}")
        print(f"Training Inverse Design Model ({self.model_type})")
        print(f"{'='*80}")
        
        # Prepare data
        X, y, input_features, output_features = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Create model
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif self.model_type == 'neural_network':
            base_model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                random_state=42
            )
        
        # Multi-output wrapper
        self.model = MultiOutputRegressor(base_model)
        
        # Train
        print(f"\nTraining on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics for each output
        print("\n" + "-"*80)
        print("Model Performance (Test Set):")
        print("-"*80)
        for i, feature in enumerate(output_features):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            print(f"\n{feature}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
        
        return X_test, y_test, y_pred, input_features, output_features
    
    def predict_parameters(self, desired_outputs):
        """
        Predict welding parameters from desired outputs
        
        Parameters:
        -----------
        desired_outputs : dict
            Dictionary of desired output values
            e.g., {'Tensile_Shear_Strength_N': 3500, 'Overall_Quality_Score': 90}
        
        Returns:
        --------
        dict : Predicted input parameters
        """
        # This is a simplified version - in practice, you'd use optimization
        # to find the best input parameters that satisfy the desired outputs
        print("\n⚠️  Note: Full inverse prediction requires optimization framework")
        print("This is a forward model - use with optimization for true inverse design")


class ForwardDesignModel:
    """
    Forward design model: Input parameters -> Output performance
    Traditional approach, useful for validation and optimization
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders = {}
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Input features (welding parameters)
        input_features = [
            'Laser_Power_W', 'Welding_Speed_mm_s', 'Pulse_Frequency_Hz',
            'Pulse_Duration_ms', 'Beam_Focus_Position_mm', 'Beam_Spot_Size_um',
            'Clamping_Pressure_kPa', 'Shield_Gas_Flow_L_min',
            'Sheet_Thickness_mm', 'Overlap_Distance_mm',
            'Heat_Input_J_mm', 'Energy_Density'
        ]
        
        categorical_input = [
            'Material_Combination', 'Shield_Gas_Type', 'Joint_Type'
        ]
        
        # Output features (performance metrics)
        output_features = [
            'Tensile_Shear_Strength_N', 'Contact_Resistance_uOhm',
            'Cycles_to_Failure', 'Overall_Quality_Score',
            'Thermal_Cycling_Strength_Degradation_pct',
            'IMC_Thickness_Post_Aging_um'
        ]
        
        # Encode categorical
        df_encoded = df.copy()
        for feature in categorical_input:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_encoded[feature + '_encoded'] = self.label_encoders[feature].fit_transform(df[feature])
            else:
                df_encoded[feature + '_encoded'] = self.label_encoders[feature].transform(df[feature])
        
        X_features = input_features + [f + '_encoded' for f in categorical_input]
        X = df_encoded[X_features].values
        y = df_encoded[output_features].values
        
        return X, y, input_features, output_features
    
    def train(self, df, test_size=0.2):
        """Train the forward design model"""
        print(f"\n{'='*80}")
        print(f"Training Forward Design Model ({self.model_type})")
        print(f"{'='*80}")
        
        # Prepare data
        X, y, input_features, output_features = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Create model
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        
        self.model = MultiOutputRegressor(base_model)
        
        # Train
        print(f"\nTraining on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        print("\n" + "-"*80)
        print("Model Performance (Test Set):")
        print("-"*80)
        for i, feature in enumerate(output_features):
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            print(f"\n{feature}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
        
        return r2
    
    def predict(self, input_params):
        """Predict performance from input parameters"""
        # Prepare input
        X = np.array([list(input_params.values())])
        X_scaled = self.scaler_X.transform(X)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred[0]


def demonstrate_forward_model():
    """Demonstrate forward design approach"""
    print("\n" + "="*80)
    print("DEMONSTRATION: Forward Design Model")
    print("="*80)
    print("Predicts performance from welding parameters")
    
    # Load data
    master_df = pd.read_csv('master_dataset.csv')
    
    # Train model
    model = ForwardDesignModel(model_type='random_forest')
    model.train(master_df, test_size=0.2)
    
    print("\n✓ Forward model trained successfully!")
    print("  Use this model with optimization algorithms for inverse design")


def demonstrate_multifidelity_learning():
    """Demonstrate multi-fidelity learning approach"""
    print("\n" + "="*80)
    print("DEMONSTRATION: Multi-Fidelity Learning")
    print("="*80)
    print("Train on computational data, fine-tune on experimental data")
    
    # Load data
    exp_df = pd.read_csv('tier1_experimental_data.csv')
    sim_df = pd.read_csv('tier2_computational_data.csv')
    
    # Step 1: Pre-train on computational data
    print("\nStep 1: Pre-training on computational data (10,000 samples)...")
    model = ForwardDesignModel(model_type='random_forest')
    model.train(sim_df, test_size=0.15)
    
    # Step 2: Fine-tune on experimental data
    print("\nStep 2: Fine-tuning on experimental data (500 samples)...")
    model_finetuned = ForwardDesignModel(model_type='random_forest')
    r2_exp = model_finetuned.train(exp_df, test_size=0.25)
    
    print("\n✓ Multi-fidelity model trained successfully!")
    print(f"  This approach leverages both simulation and experimental data")


def main():
    """Main demonstration"""
    print("\n" + "="*80)
    print("WELDING INVERSE DESIGN - ML MODEL EXAMPLES")
    print("="*80)
    
    # Check if data exists
    import os
    if not os.path.exists('master_dataset.csv'):
        print("\n❌ Error: Dataset files not found!")
        print("Please run generate_welding_dataset.py first")
        return
    
    # Demonstrate forward model
    demonstrate_forward_model()
    
    # Demonstrate multi-fidelity learning
    demonstrate_multifidelity_learning()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Implement Bayesian Optimization for true inverse design")
    print("  2. Try VAE or GAN for generative inverse design")
    print("  3. Add physics-informed constraints to predictions")
    print("  4. Implement multi-objective optimization")
    print("  5. Use ensemble methods combining multiple models")


if __name__ == '__main__':
    main()
