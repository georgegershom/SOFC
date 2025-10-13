"""
Example script demonstrating digital twin functionality using the SOFC dataset.

This script shows how to use the dataset for digital twin applications including:
- Real-time monitoring
- Predictive maintenance
- Performance optimization
- Failure detection
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd

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


class SOFCDigitalTwin:
    """SOFC Digital Twin class for real-time monitoring and prediction."""
    
    def __init__(self, dataset):
        """Initialize the digital twin with historical data."""
        self.dataset = dataset
        self.sensor_data = {}
        self.performance_metrics = {}
        self.degradation_models = {}
        self.anomaly_detector = None
        self.performance_predictor = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models."""
        print("Initializing digital twin models...")
        
        # Prepare training data
        training_data = self._prepare_training_data()
        
        if training_data is not None:
            # Initialize anomaly detector
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.anomaly_detector.fit(training_data)
            
            # Initialize performance predictor
            self.performance_predictor = self._train_performance_predictor(training_data)
            
            print("Models initialized successfully!")
        else:
            print("Warning: Could not initialize models - insufficient training data")
    
    def _prepare_training_data(self):
        """Prepare training data for machine learning models."""
        # Collect sensor data from the dataset
        sensor_data = []
        
        if 'sensor_network' in self.dataset:
            for sensor_type, sensors in self.dataset['sensor_network'].items():
                if isinstance(sensors, dict):
                    for sensor_id, sensor_info in sensors.items():
                        if isinstance(sensor_info, dict):
                            # Find numeric time series data
                            for key, value in sensor_info.items():
                                if isinstance(value, np.ndarray) and value.dtype in [np.float64, np.float32]:
                                    if len(value) > 0:
                                        sensor_data.append(value)
        
        if sensor_data:
            # Align all sensor data to common length
            min_length = min(len(data) for data in sensor_data)
            aligned_data = np.column_stack([data[:min_length] for data in sensor_data])
            return aligned_data
        
        return None
    
    def _train_performance_predictor(self, training_data):
        """Train a performance prediction model."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create features and targets
        X = training_data[:-1]  # All but last time step
        y = training_data[1:]   # All but first time step
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        return model
    
    def update_sensor_data(self, new_sensor_data):
        """Update sensor data with new measurements."""
        self.sensor_data.update(new_sensor_data)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Check for anomalies
        anomalies = self._detect_anomalies()
        
        # Update degradation models
        self._update_degradation_models()
        
        return {
            'performance_metrics': self.performance_metrics,
            'anomalies': anomalies,
            'degradation_status': self._get_degradation_status()
        }
    
    def _update_performance_metrics(self):
        """Update performance metrics based on current sensor data."""
        # Calculate basic performance metrics
        if 'temperature' in self.sensor_data:
            temp = self.sensor_data['temperature']
            self.performance_metrics['temperature'] = {
                'current': temp,
                'mean': np.mean(temp) if isinstance(temp, np.ndarray) else temp,
                'std': np.std(temp) if isinstance(temp, np.ndarray) else 0,
                'min': np.min(temp) if isinstance(temp, np.ndarray) else temp,
                'max': np.max(temp) if isinstance(temp, np.ndarray) else temp
            }
        
        if 'voltage' in self.sensor_data:
            voltage = self.sensor_data['voltage']
            self.performance_metrics['voltage'] = {
                'current': voltage,
                'mean': np.mean(voltage) if isinstance(voltage, np.ndarray) else voltage,
                'efficiency': self._calculate_efficiency(voltage)
            }
        
        if 'current_density' in self.sensor_data:
            current = self.sensor_data['current_density']
            self.performance_metrics['current'] = {
                'current': current,
                'mean': np.mean(current) if isinstance(current, np.ndarray) else current,
                'power': self._calculate_power(voltage, current)
            }
    
    def _calculate_efficiency(self, voltage):
        """Calculate cell efficiency."""
        # Simplified efficiency calculation
        V_ocv = 1.1  # Open circuit voltage
        return voltage / V_ocv if V_ocv > 0 else 0
    
    def _calculate_power(self, voltage, current):
        """Calculate power output."""
        if isinstance(voltage, np.ndarray) and isinstance(current, np.ndarray):
            return voltage * current
        else:
            return voltage * current
    
    def _detect_anomalies(self):
        """Detect anomalies in sensor data."""
        if self.anomaly_detector is None:
            return []
        
        # Prepare current sensor data for anomaly detection
        current_data = []
        for key, value in self.sensor_data.items():
            if isinstance(value, np.ndarray) and value.dtype in [np.float64, np.float32]:
                current_data.append(value[-1] if len(value) > 0 else 0)
            elif isinstance(value, (int, float)):
                current_data.append(value)
        
        if current_data:
            current_data = np.array(current_data).reshape(1, -1)
            anomaly_score = self.anomaly_detector.decision_function(current_data)
            is_anomaly = self.anomaly_detector.predict(current_data)
            
            anomalies = []
            if is_anomaly[0] == -1:  # Anomaly detected
                anomalies.append({
                    'type': 'sensor_anomaly',
                    'score': anomaly_score[0],
                    'timestamp': pd.Timestamp.now(),
                    'description': 'Unusual sensor reading detected'
                })
            
            return anomalies
        
        return []
    
    def _update_degradation_models(self):
        """Update degradation models based on current data."""
        # Simplified degradation tracking
        if 'voltage' in self.sensor_data:
            voltage = self.sensor_data['voltage']
            if isinstance(voltage, np.ndarray):
                voltage = voltage[-1]
            
            # Track voltage degradation
            if 'voltage_degradation' not in self.degradation_models:
                self.degradation_models['voltage_degradation'] = []
            
            self.degradation_models['voltage_degradation'].append(voltage)
            
            # Keep only recent history
            if len(self.degradation_models['voltage_degradation']) > 100:
                self.degradation_models['voltage_degradation'] = self.degradation_models['voltage_degradation'][-100:]
    
    def _get_degradation_status(self):
        """Get current degradation status."""
        status = {
            'overall_health': 'good',
            'degradation_rate': 0.0,
            'remaining_life': 'unknown'
        }
        
        if 'voltage_degradation' in self.degradation_models:
            voltage_history = self.degradation_models['voltage_degradation']
            
            if len(voltage_history) > 10:
                # Calculate degradation rate
                recent_voltage = np.mean(voltage_history[-10:])
                initial_voltage = np.mean(voltage_history[:10])
                
                degradation_rate = (initial_voltage - recent_voltage) / initial_voltage
                status['degradation_rate'] = degradation_rate
                
                # Determine health status
                if degradation_rate > 0.1:
                    status['overall_health'] = 'poor'
                elif degradation_rate > 0.05:
                    status['overall_health'] = 'fair'
                else:
                    status['overall_health'] = 'good'
                
                # Estimate remaining life (simplified)
                if degradation_rate > 0:
                    remaining_life = (recent_voltage - 0.5) / degradation_rate  # Assume failure at 0.5V
                    status['remaining_life'] = f"{remaining_life:.1f} hours"
        
        return status
    
    def predict_performance(self, time_horizon: int = 10):
        """Predict performance over a time horizon."""
        if self.performance_predictor is None:
            return None
        
        # Prepare current state for prediction
        current_state = []
        for key, value in self.sensor_data.items():
            if isinstance(value, np.ndarray) and value.dtype in [np.float64, np.float32]:
                current_state.append(value[-1] if len(value) > 0 else 0)
            elif isinstance(value, (int, float)):
                current_state.append(value)
        
        if current_state:
            current_state = np.array(current_state).reshape(1, -1)
            
            # Make predictions
            predictions = []
            state = current_state.copy()
            
            for _ in range(time_horizon):
                pred = self.performance_predictor.predict(state)
                predictions.append(pred[0])
                state = pred.reshape(1, -1)
            
            return np.array(predictions)
        
        return None
    
    def get_recommendations(self):
        """Get operational recommendations based on current state."""
        recommendations = []
        
        # Check temperature
        if 'temperature' in self.performance_metrics:
            temp_metrics = self.performance_metrics['temperature']
            current_temp = temp_metrics['current']
            
            if current_temp > 800:
                recommendations.append({
                    'type': 'temperature',
                    'priority': 'high',
                    'message': 'Temperature too high - reduce load or increase cooling',
                    'action': 'Reduce current density or increase air flow'
                })
            elif current_temp < 600:
                recommendations.append({
                    'type': 'temperature',
                    'priority': 'medium',
                    'message': 'Temperature too low - increase load or reduce cooling',
                    'action': 'Increase current density or reduce air flow'
                })
        
        # Check voltage
        if 'voltage' in self.performance_metrics:
            voltage_metrics = self.performance_metrics['voltage']
            current_voltage = voltage_metrics['current']
            
            if current_voltage < 0.7:
                recommendations.append({
                    'type': 'voltage',
                    'priority': 'high',
                    'message': 'Voltage too low - check for degradation or contamination',
                    'action': 'Inspect cell for damage or perform maintenance'
                })
        
        # Check degradation
        degradation_status = self._get_degradation_status()
        if degradation_status['overall_health'] == 'poor':
            recommendations.append({
                'type': 'degradation',
                'priority': 'high',
                'message': 'High degradation rate detected',
                'action': 'Schedule maintenance or cell replacement'
            })
        
        return recommendations


def simulate_real_time_monitoring(dataset, duration: int = 100):
    """Simulate real-time monitoring using the dataset."""
    print(f"\nSimulating real-time monitoring for {duration} time steps...")
    
    # Initialize digital twin
    digital_twin = SOFCDigitalTwin(dataset)
    
    # Get sensor data from dataset
    sensor_data = {}
    if 'sensor_network' in dataset:
        for sensor_type, sensors in dataset['sensor_network'].items():
            if isinstance(sensors, dict):
                for sensor_id, sensor_info in sensors.items():
                    if isinstance(sensor_info, dict):
                        for key, value in sensor_info.items():
                            if isinstance(value, np.ndarray) and value.dtype in [np.float64, np.float32]:
                                sensor_data[f"{sensor_id}_{key}"] = value
    
    # Simulate real-time updates
    monitoring_results = []
    
    for i in range(min(duration, len(list(sensor_data.values())[0]) if sensor_data else 0)):
        # Get current sensor readings
        current_readings = {}
        for key, value in sensor_data.items():
            if isinstance(value, np.ndarray) and len(value) > i:
                current_readings[key] = value[i]
        
        # Update digital twin
        update_result = digital_twin.update_sensor_data(current_readings)
        
        # Store results
        monitoring_results.append({
            'time_step': i,
            'sensor_data': current_readings,
            'performance_metrics': update_result['performance_metrics'],
            'anomalies': update_result['anomalies'],
            'degradation_status': update_result['degradation_status']
        })
        
        # Print status every 20 time steps
        if i % 20 == 0:
            print(f"  Time step {i}: {len(update_result['anomalies'])} anomalies, "
                  f"health: {update_result['degradation_status']['overall_health']}")
    
    return monitoring_results, digital_twin


def create_monitoring_dashboard(monitoring_results, output_dir="monitoring_dashboard"):
    """Create a monitoring dashboard."""
    print(f"\nCreating monitoring dashboard in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    time_steps = [result['time_step'] for result in monitoring_results]
    
    # Plot performance metrics
    if monitoring_results:
        # Temperature plot
        temperatures = []
        for result in monitoring_results:
            if 'temperature' in result['performance_metrics']:
                temp_metrics = result['performance_metrics']['temperature']
                temperatures.append(temp_metrics.get('current', 0))
            else:
                temperatures.append(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, temperatures, 'b-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.title('Real-time Temperature Monitoring')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'temperature_monitoring.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Voltage plot
        voltages = []
        for result in monitoring_results:
            if 'voltage' in result['performance_metrics']:
                voltage_metrics = result['performance_metrics']['voltage']
                voltages.append(voltage_metrics.get('current', 0))
            else:
                voltages.append(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, voltages, 'r-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Voltage (V)')
        plt.title('Real-time Voltage Monitoring')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'voltage_monitoring.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Anomaly detection plot
        anomaly_counts = [len(result['anomalies']) for result in monitoring_results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, anomaly_counts, 'g-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Number of Anomalies')
        plt.title('Real-time Anomaly Detection')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'anomaly_detection.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Degradation status plot
        health_status = [result['degradation_status']['overall_health'] for result in monitoring_results]
        health_numeric = [1 if h == 'good' else 0.5 if h == 'fair' else 0 for h in health_status]
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, health_numeric, 'm-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Health Status (1=good, 0.5=fair, 0=poor)')
        plt.title('Real-time Health Monitoring')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'health_monitoring.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Monitoring dashboard saved to: {output_dir}")


def demonstrate_predictive_maintenance(digital_twin):
    """Demonstrate predictive maintenance capabilities."""
    print("\nDemonstrating predictive maintenance...")
    
    # Get current recommendations
    recommendations = digital_twin.get_recommendations()
    
    print("Current Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['message']}")
        print(f"     Action: {rec['action']}")
    
    # Make performance predictions
    predictions = digital_twin.predict_performance(time_horizon=20)
    
    if predictions is not None:
        print(f"\nPerformance Prediction (next 20 time steps):")
        print(f"  Predicted values shape: {predictions.shape}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(predictions)), predictions[:, 0], 'b-', linewidth=2, label='Predicted')
        plt.xlabel('Time Steps Ahead')
        plt.ylabel('Predicted Value')
        plt.title('Performance Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('performance_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Performance prediction plot saved to: performance_prediction.png")
    
    return recommendations, predictions


def main():
    """Main function for digital twin demonstration."""
    print("SOFC Digital Twin Demonstration")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset()
    
    # Simulate real-time monitoring
    monitoring_results, digital_twin = simulate_real_time_monitoring(dataset, duration=100)
    
    # Create monitoring dashboard
    create_monitoring_dashboard(monitoring_results)
    
    # Demonstrate predictive maintenance
    recommendations, predictions = demonstrate_predictive_maintenance(digital_twin)
    
    # Print summary
    print("\n" + "="*60)
    print("Digital Twin Demonstration Summary")
    print("="*60)
    print(f"Monitoring duration: {len(monitoring_results)} time steps")
    print(f"Total anomalies detected: {sum(len(result['anomalies']) for result in monitoring_results)}")
    print(f"Recommendations generated: {len(recommendations)}")
    print(f"Performance predictions: {'Yes' if predictions is not None else 'No'}")
    print("="*60)


if __name__ == "__main__":
    main()