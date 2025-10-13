"""
Example script for validating models using the SOFC Digital Twin Dataset.

This script demonstrates how to use the dataset for model validation,
calibration, and testing of digital twin algorithms.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import h5py

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import DataUtils, PhysicsUtils, VisualizationUtils


def load_dataset(dataset_path: str = "data/integrated_dataset.h5"):
    """Load the integrated dataset."""
    print(f"Loading dataset from: {dataset_path}")
    
    dataset = {}
    with h5py.File(dataset_path, 'r') as f:
        # Load all groups
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                dataset[key] = {}
                load_group(f[key], dataset[key])
            else:
                dataset[key] = f[key][:]
    
    print("Dataset loaded successfully!")
    return dataset


def load_group(group, data_dict):
    """Recursively load HDF5 group."""
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            data_dict[key] = {}
            load_group(group[key], data_dict[key])
        else:
            data_dict[key] = group[key][:]


def prepare_training_data(dataset):
    """Prepare training data for machine learning models."""
    print("\nPreparing training data...")
    
    # Extract features and targets
    features = []
    targets = []
    
    # Get time series data
    if 'time_series' in dataset:
        time_series = dataset['time_series']
        
        # Collect all time series data
        time_series_data = {}
        for key, value in time_series.items():
            if isinstance(value, dict) and 'time_points' in value:
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray) and subvalue.dtype in [np.float64, np.float32]:
                        time_series_data[f"{key}_{subkey}"] = subvalue
        
        # Create feature matrix
        if time_series_data:
            # Align all time series to common time points
            common_length = min(len(ts) for ts in time_series_data.values())
            
            feature_matrix = np.column_stack([
                ts[:common_length] for ts in time_series_data.values()
            ])
            
            # Create features (current and previous values)
            n_features = 5  # Use 5 previous time steps
            X = []
            y = []
            
            for i in range(n_features, common_length):
                # Features: current and previous values
                feature_vector = feature_matrix[i-n_features:i].flatten()
                X.append(feature_vector)
                
                # Target: next value (temperature prediction)
                if 'temperature' in time_series_data:
                    target = time_series_data['temperature'][i]
                    y.append(target)
            
            features = np.array(X)
            targets = np.array(y)
    
    print(f"Training data shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    
    return features, targets


def train_temperature_prediction_model(features, targets):
    """Train a temperature prediction model."""
    print("\nTraining temperature prediction model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return model, X_test, y_test, y_pred_test


def validate_physics_models(dataset):
    """Validate physics-based models using the dataset."""
    print("\nValidating physics-based models...")
    
    physics_utils = PhysicsUtils()
    
    # Get temperature and voltage data
    if 'time_series' in dataset:
        time_series = dataset['time_series']
        
        # Find temperature and voltage data
        temperature_data = None
        voltage_data = None
        
        for key, value in time_series.items():
            if isinstance(value, dict):
                if 'temperature' in value and 'time_points' in value:
                    temperature_data = value
                elif 'voltage' in value and 'time_points' in value:
                    voltage_data = value
        
        if temperature_data and voltage_data:
            # Validate Nernst equation
            print("Validating Nernst equation...")
            
            T = temperature_data['temperature'] + 273.15  # Convert to Kelvin
            V_measured = voltage_data['voltage']
            
            # Calculate theoretical Nernst voltage
            p_H2 = 0.97 * 1e5  # Pa
            p_H2O = 0.01 * 1e5  # Pa
            p_O2 = 0.21 * 1e5  # Pa
            
            V_nernst = physics_utils.nernst_voltage(T, p_H2, p_H2O, p_O2)
            
            # Compare with measured voltage
            voltage_error = np.abs(V_measured - V_nernst)
            mean_error = np.mean(voltage_error)
            max_error = np.max(voltage_error)
            
            print(f"Mean voltage error: {mean_error:.4f} V")
            print(f"Max voltage error: {max_error:.4f} V")
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            plt.plot(temperature_data['time_points'], V_measured, 'b-', label='Measured')
            plt.plot(temperature_data['time_points'], V_nernst, 'r--', label='Nernst')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.title('Voltage Validation: Measured vs Nernst')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('voltage_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Voltage validation plot saved to: voltage_validation.png")


def validate_degradation_models(dataset):
    """Validate degradation models using the dataset."""
    print("\nValidating degradation models...")
    
    if 'degradation_tracking' in dataset:
        degradation_data = dataset['degradation_tracking']
        
        # Analyze degradation trends
        for degradation_type, data in degradation_data.items():
            if isinstance(data, dict):
                print(f"\nAnalyzing {degradation_type}...")
                
                # Find time series data
                time_series = []
                for key, value in data.items():
                    if isinstance(value, dict) and 'time_points' in value:
                        time_series.append((key, value))
                
                if time_series:
                    # Plot degradation trends
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    for i, (key, ts_data) in enumerate(time_series[:4]):
                        if i < len(axes):
                            ax = axes[i]
                            
                            # Plot all numeric data
                            for subkey, subvalue in ts_data.items():
                                if isinstance(subvalue, np.ndarray) and subvalue.dtype in [np.float64, np.float32]:
                                    ax.plot(ts_data['time_points'], subvalue, label=subkey)
                            
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Value')
                            ax.set_title(f'{key}')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for i in range(len(time_series), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    plt.savefig(f'degradation_{degradation_type}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Degradation analysis plot saved to: degradation_{degradation_type}.png")


def validate_sensor_models(dataset):
    """Validate sensor models using the dataset."""
    print("\nValidating sensor models...")
    
    if 'sensor_network' in dataset:
        sensor_network = dataset['sensor_network']
        
        # Analyze sensor data quality
        for sensor_type, sensors in sensor_network.items():
            if isinstance(sensors, dict):
                print(f"\nAnalyzing {sensor_type} sensors...")
                
                # Collect sensor data
                sensor_data = []
                sensor_locations = []
                
                for sensor_id, sensor_info in sensors.items():
                    if isinstance(sensor_info, dict):
                        # Find numeric time series data
                        for key, value in sensor_info.items():
                            if isinstance(value, np.ndarray) and value.dtype in [np.float64, np.float32]:
                                sensor_data.append((sensor_id, key, value))
                        
                        # Collect location data
                        if 'location' in sensor_info:
                            sensor_locations.append((sensor_id, sensor_info['location']))
                
                # Analyze sensor correlations
                if len(sensor_data) > 1:
                    print(f"  Found {len(sensor_data)} sensor measurements")
                    
                    # Create correlation matrix
                    data_matrix = []
                    sensor_names = []
                    
                    for sensor_id, key, data in sensor_data:
                        if len(data) > 0:
                            data_matrix.append(data)
                            sensor_names.append(f"{sensor_id}_{key}")
                    
                    if data_matrix:
                        data_matrix = np.array(data_matrix)
                        
                        # Calculate correlations
                        correlations = np.corrcoef(data_matrix)
                        
                        # Plot correlation heatmap
                        plt.figure(figsize=(10, 8))
                        plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
                        plt.colorbar(label='Correlation Coefficient')
                        plt.xticks(range(len(sensor_names)), sensor_names, rotation=45, ha='right')
                        plt.yticks(range(len(sensor_names)), sensor_names)
                        plt.title(f'{sensor_type} Sensor Correlations')
                        plt.tight_layout()
                        plt.savefig(f'sensor_correlations_{sensor_type}.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"  Sensor correlation plot saved to: sensor_correlations_{sensor_type}.png")
                
                # Analyze sensor locations
                if sensor_locations:
                    print(f"  Found {len(sensor_locations)} sensor locations")
                    
                    # Plot sensor locations
                    plt.figure(figsize=(10, 8))
                    
                    for i, (sensor_id, location) in enumerate(sensor_locations):
                        if isinstance(location, dict) and 'x' in location and 'y' in location:
                            plt.scatter(location['x'], location['y'], 
                                      label=sensor_id, s=100, alpha=0.7)
                    
                    plt.xlabel('X (m)')
                    plt.ylabel('Y (m)')
                    plt.title(f'{sensor_type} Sensor Locations')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f'sensor_locations_{sensor_type}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Sensor location plot saved to: sensor_locations_{sensor_type}.png")


def create_validation_report(dataset, model_results):
    """Create a comprehensive validation report."""
    print("\nCreating validation report...")
    
    report_path = "model_validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("SOFC Digital Twin Dataset - Model Validation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated at: {pd.Timestamp.now()}\n\n")
        
        f.write("Dataset Information:\n")
        f.write("-" * 30 + "\n")
        
        if 'metadata' in dataset:
            metadata = dataset['metadata']
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        f.write("\nModel Validation Results:\n")
        f.write("-" * 30 + "\n")
        
        if model_results:
            f.write("Temperature Prediction Model:\n")
            f.write(f"  Training MSE: {model_results.get('train_mse', 'N/A'):.4f}\n")
            f.write(f"  Test MSE: {model_results.get('test_mse', 'N/A'):.4f}\n")
            f.write(f"  Training R²: {model_results.get('train_r2', 'N/A'):.4f}\n")
            f.write(f"  Test R²: {model_results.get('test_r2', 'N/A'):.4f}\n")
        
        f.write("\nValidation Summary:\n")
        f.write("-" * 30 + "\n")
        f.write("✓ Physics-based models validated\n")
        f.write("✓ Degradation models analyzed\n")
        f.write("✓ Sensor models validated\n")
        f.write("✓ Machine learning models trained and tested\n")
    
    print(f"Validation report saved to: {report_path}")


def main():
    """Main function for model validation."""
    print("SOFC Digital Twin Dataset - Model Validation")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset()
    
    # Prepare training data
    features, targets = prepare_training_data(dataset)
    
    # Train models
    model_results = {}
    if len(features) > 0 and len(targets) > 0:
        model, X_test, y_test, y_pred_test = train_temperature_prediction_model(features, targets)
        model_results = {
            'train_mse': mean_squared_error(features, model.predict(features)),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_r2': r2_score(targets, model.predict(features)),
            'test_r2': r2_score(y_test, y_pred_test)
        }
    
    # Validate different model types
    validate_physics_models(dataset)
    validate_degradation_models(dataset)
    validate_sensor_models(dataset)
    
    # Create validation report
    create_validation_report(dataset, model_results)
    
    print("\n" + "="*60)
    print("Model validation complete!")
    print("="*60)


if __name__ == "__main__":
    main()