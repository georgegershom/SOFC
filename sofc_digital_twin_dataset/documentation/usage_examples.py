#!/usr/bin/env python3
"""
SOFC Digital Twin Dataset Usage Examples
=======================================

This script demonstrates how to load, process, and analyze the SOFC digital twin dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json

# Example 1: Loading and visualizing microstructural data
def load_microstructural_data(dataset_path):
    """Load and visualize 3D microstructural data"""
    
    # Load anode microstructure
    with h5py.File(f"{dataset_path}/1_materials_geometry/microstructural/anode_microstructure_3D.h5", 'r') as f:
        microstructure = f['microstructure'][:]
        voxel_size = f['voxel_size'][()]
        
    # Load effective properties
    with open(f"{dataset_path}/1_materials_geometry/microstructural/anode_properties.json", 'r') as f:
        properties = json.load(f)
    
    # Visualize cross-section
    mid_slice = microstructure.shape[2] // 2
    cross_section = microstructure[:, :, mid_slice]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cross_section, cmap='viridis')
    plt.title(f'Anode Microstructure (Porosity: {properties["porosity"]:.3f})')
    plt.colorbar(label='Phase ID')
    plt.show()
    
    return microstructure, properties

# Example 2: Analyzing electrochemical performance trends
def analyze_electrochemical_performance(dataset_path):
    """Analyze voltage degradation and EIS evolution"""
    
    # Load voltage data
    voltage_data = pd.read_csv(f"{dataset_path}/2_operational_electrochemical/performance_data/voltage_response.csv")
    
    # Load EIS data
    with h5py.File(f"{dataset_path}/2_operational_electrochemical/performance_data/eis_spectra.h5", 'r') as f:
        frequencies = f['frequencies_Hz'][:]
        Z_real = f['Z_real_ohm_cm2'][:]
        Z_imag = f['Z_imag_ohm_cm2'][:]
        time_points = f['time_points_hours'][:]
    
    # Plot voltage degradation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(voltage_data['time_hours'], voltage_data['cell_voltage_V'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Cell Voltage (V)')
    plt.title('Voltage Degradation')
    plt.grid(True)
    
    # Plot EIS evolution
    plt.subplot(1, 2, 2)
    for i, t in enumerate(time_points):
        plt.plot(Z_real[i], Z_imag[i], 'o-', label=f't = {t:.0f} h')
    plt.xlabel('Z_real (Ω·cm²)')
    plt.ylabel('-Z_imag (Ω·cm²)')
    plt.title('EIS Evolution')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return voltage_data

# Example 3: Processing temperature and stress field data
def analyze_thermal_mechanical_fields(dataset_path):
    """Analyze temperature and stress field evolution"""
    
    # Load temperature fields
    with h5py.File(f"{dataset_path}/3_thermo_structural/temperature_fields/temperature_fields_2D.h5", 'r') as f:
        temp_fields = f['temperature_fields_C'][:]
        x_coords = f['x_coordinates_m'][:]
        y_coords = f['y_coordinates_m'][:]
    
    # Load stress fields
    with h5py.File(f"{dataset_path}/3_thermo_structural/stress_strain/stress_strain_fields_2D.h5", 'r') as f:
        stress_fields = f['von_mises_stress_Pa'][:]
    
    # Load thermocouple data
    tc_data = pd.read_csv(f"{dataset_path}/3_thermo_structural/temperature_fields/thermocouple_data.csv")
    
    # Visualize latest fields
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Temperature field
    im1 = axes[0].contourf(x_coords*1000, y_coords*1000, temp_fields[-1], levels=20, cmap='hot')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_title('Temperature Field (°C)')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # Stress field
    im2 = axes[1].contourf(x_coords*1000, y_coords*1000, stress_fields[-1]/1e6, levels=20, cmap='plasma')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Y (mm)')
    axes[1].set_title('Von Mises Stress (MPa)')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    # Thermocouple evolution
    axes[2].plot(tc_data['time_hours'], tc_data['TC1_inlet'], label='Inlet')
    axes[2].plot(tc_data['time_hours'], tc_data['TC2_center'], label='Center')
    axes[2].plot(tc_data['time_hours'], tc_data['TC3_outlet'], label='Outlet')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].set_title('Thermocouple Readings')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return temp_fields, stress_fields, tc_data

# Example 4: Implementing data assimilation workflow
def implement_data_assimilation(dataset_path):
    """Demonstrate data assimilation workflow"""
    
    # Load assimilation data
    with h5py.File(f"{dataset_path}/5_synthesis_workflows/sample_assimilation_workflow.h5", 'r') as f:
        time_minutes = f['time_minutes'][:]
        true_temp = f['true_states/temperature_center_C'][:]
        measured_temp = f['measurements/thermocouple_1_C'][:]
        estimated_temp = f['estimated_states/temperature_center_C'][:]
        uncertainty = f['estimation_uncertainty/temperature_center_C'][:]
    
    # Simple Kalman filter implementation
    def kalman_filter(measurements, process_noise=0.01, measurement_noise=0.5):
        n = len(measurements)
        estimates = np.zeros(n)
        uncertainties = np.zeros(n)
        
        # Initialize
        estimates[0] = measurements[0]
        uncertainties[0] = 1.0
        
        for i in range(1, n):
            # Predict
            pred_estimate = estimates[i-1]
            pred_uncertainty = uncertainties[i-1] + process_noise
            
            # Update
            kalman_gain = pred_uncertainty / (pred_uncertainty + measurement_noise)
            estimates[i] = pred_estimate + kalman_gain * (measurements[i] - pred_estimate)
            uncertainties[i] = (1 - kalman_gain) * pred_uncertainty
        
        return estimates, uncertainties
    
    # Apply Kalman filter
    kf_estimates, kf_uncertainties = kalman_filter(measured_temp)
    
    # Visualize results
    time_hours = time_minutes / 60
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_hours, true_temp, 'b-', label='True', alpha=0.7)
    plt.plot(time_hours, measured_temp, 'r.', label='Measured', alpha=0.5, markersize=1)
    plt.plot(time_hours, estimated_temp, 'g-', label='Dataset Estimate', linewidth=2)
    plt.plot(time_hours, kf_estimates, 'm--', label='Custom KF', linewidth=2)
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature State Estimation Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_hours, uncertainty, 'g-', label='Dataset Uncertainty')
    plt.plot(time_hours, kf_uncertainties, 'm--', label='Custom KF Uncertainty')
    plt.xlabel('Time (hours)')
    plt.ylabel('Uncertainty (°C)')
    plt.title('Estimation Uncertainty')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return kf_estimates, kf_uncertainties

# Example 5: Training reduced-order models
def train_reduced_order_model(dataset_path):
    """Train a simple ROM using the dataset"""
    
    # Load ROM training data
    with h5py.File(f"{dataset_path}/5_synthesis_workflows/rom_training_data.h5", 'r') as f:
        # Parameters
        temperature = f['parameters/temperature_C'][:]
        current_density = f['parameters/current_density_A_cm2'][:]
        
        # Outputs
        voltage = f['outputs/cell_voltage_V'][:]
        power = f['outputs/power_density_W_cm2'][:]
    
    # Prepare training data
    X = np.column_stack([temperature, current_density])
    y = voltage
    
    # Simple polynomial ROM
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train model
    rom_model = LinearRegression()
    rom_model.fit(X_train_poly, y_train)
    
    # Evaluate
    y_pred = rom_model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"ROM Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.4f}")
    
    # Visualize predictions
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Voltage (V)')
    plt.ylabel('Predicted Voltage (V)')
    plt.title(f'ROM Predictions (R² = {r2:.4f})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Voltage (V)')
    plt.ylabel('Residuals (V)')
    plt.title('Residual Analysis')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return rom_model, poly

# Main execution
if __name__ == "__main__":
    # Set dataset path
    dataset_path = "sofc_digital_twin_dataset"
    
    print("SOFC Digital Twin Dataset Usage Examples")
    print("=" * 50)
    
    # Run examples
    print("\n1. Loading microstructural data...")
    microstructure, properties = load_microstructural_data(dataset_path)
    
    print("\n2. Analyzing electrochemical performance...")
    voltage_data = analyze_electrochemical_performance(dataset_path)
    
    print("\n3. Processing thermal-mechanical fields...")
    temp_fields, stress_fields, tc_data = analyze_thermal_mechanical_fields(dataset_path)
    
    print("\n4. Implementing data assimilation...")
    kf_estimates, kf_uncertainties = implement_data_assimilation(dataset_path)
    
    print("\n5. Training reduced-order model...")
    rom_model, poly = train_reduced_order_model(dataset_path)
    
    print("\nAll examples completed successfully!")
