"""
Dataset 3 Generator: Real-Time Monitoring Data
Generates adaptive-scale real-time monitoring data for digital twin operation
"""

import numpy as np
import pandas as pd
import h5py
import yaml
import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import queue
from dataclasses import dataclass

@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    high_freq_sampling: float = 1.0  # Hz
    low_freq_sampling: float = 1/3600  # Hz (every hour)
    eis_interval: float = 24 * 3600  # seconds (daily)
    thermal_interval: float = 6 * 3600  # seconds (every 6 hours)
    ae_monitoring: bool = True
    buffer_size: int = 1000

class Dataset3Generator:
    """
    Generates Dataset 3: Real-time monitoring data
    
    This simulates the data stream that would feed the adaptive digital twin
    during actual operation, including:
    - High-frequency operational data (I, V, T, F)
    - Low-frequency high-value updates (EIS, thermal imaging)
    - Event-driven data (acoustic emission)
    - Adaptive sampling based on system state
    """
    
    def __init__(self, config_path: str):
        """Initialize real-time data generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.monitoring_config = MonitoringConfig()
        
        # Create output directory
        self.output_dir = "datasets/dataset3_realtime"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data buffers
        self.high_freq_buffer = queue.Queue(maxsize=self.monitoring_config.buffer_size)
        self.low_freq_buffer = queue.Queue(maxsize=100)
        self.event_buffer = queue.Queue(maxsize=500)
        
        # Simulation state
        self.simulation_time = 0.0  # hours
        self.is_running = False
        self.degradation_state = 0.0  # 0 = new, 1 = fully degraded
        
        # Base operating conditions
        self.base_conditions = {
            'current_density': 0.8,  # A/cm²
            'voltage': 0.75,         # V
            'temp_fuel': 800,        # °C
            'temp_air': 750,         # °C
            'flow_fuel': 100,        # sccm
            'flow_air': 500          # sccm
        }
        
    def update_degradation_state(self):
        """Update system degradation state based on simulation time"""
        # Simple linear degradation model
        max_time = 8760  # hours (1 year)
        self.degradation_state = min(self.simulation_time / max_time, 1.0)
        
    def generate_high_frequency_data(self) -> Dict:
        """Generate high-frequency operational data point"""
        self.update_degradation_state()
        
        # Add realistic variations and degradation effects
        noise_scale = 0.02
        
        # Current density with load variations
        current_density = (self.base_conditions['current_density'] * 
                          (1 + 0.1 * np.sin(2*np.pi*self.simulation_time/24) +
                           noise_scale * np.random.normal()))
        
        # Voltage with degradation
        voltage_degradation = -50e-6 * self.simulation_time  # V/hour
        voltage = (self.base_conditions['voltage'] + voltage_degradation +
                  noise_scale * 0.02 * np.random.normal())
        
        # Power
        power = current_density * voltage
        
        # Temperatures with thermal dynamics
        temp_fuel = (self.base_conditions['temp_fuel'] + 
                    10 * np.sin(2*np.pi*self.simulation_time/12) +
                    5 * np.random.normal())
        
        temp_air = (self.base_conditions['temp_air'] + 
                   8 * np.sin(2*np.pi*self.simulation_time/12) +
                   3 * np.random.normal())
        
        # Flow rates with control variations
        flow_fuel = (self.base_conditions['flow_fuel'] + 
                    2 * np.random.normal())
        flow_air = (self.base_conditions['flow_air'] + 
                   10 * np.random.normal())
        
        # Additional derived quantities
        fuel_utilization = 0.8 + 0.1 * np.sin(2*np.pi*self.simulation_time/48)
        air_utilization = 0.2 + 0.05 * np.sin(2*np.pi*self.simulation_time/36)
        
        data_point = {
            'timestamp': datetime.now(),
            'simulation_time_hours': self.simulation_time,
            'current_density': current_density,
            'voltage': voltage,
            'power': power,
            'temp_fuel_inlet': temp_fuel,
            'temp_air_inlet': temp_air,
            'flow_fuel': flow_fuel,
            'flow_air': flow_air,
            'fuel_utilization': fuel_utilization,
            'air_utilization': air_utilization,
            'degradation_state': self.degradation_state
        }
        
        return data_point
    
    def generate_eis_measurement(self) -> Dict:
        """Generate EIS measurement (low frequency, high value)"""
        print(f"Performing EIS measurement at t={self.simulation_time:.1f}h")
        
        # Frequency range
        frequencies = np.logspace(-2, 5, 50)  # 0.01 Hz to 100 kHz
        
        # Impedance parameters evolve with degradation
        R_ohmic = 0.15 * (1 + 0.5 * self.degradation_state)
        R_act = 0.10 * (1 + 1.0 * self.degradation_state)
        R_conc = 0.05 * (1 + 2.0 * self.degradation_state)
        
        # Capacitances
        C_dl = 0.01 * (1 - 0.2 * self.degradation_state)  # Decreases with degradation
        C_chem = 0.1 * (1 - 0.3 * self.degradation_state)
        
        Z_real, Z_imag = [], []
        
        for f in frequencies:
            omega = 2 * np.pi * f
            
            # Equivalent circuit model
            Z_act = R_act / (1 + 1j * omega * R_act * C_dl)
            Z_conc = R_conc / (1 + 1j * omega * R_conc * C_chem)
            Z_total = R_ohmic + Z_act + Z_conc
            
            # Add measurement noise
            noise_real = 0.005 * np.random.normal()
            noise_imag = 0.005 * np.random.normal()
            
            Z_real.append(Z_total.real + noise_real)
            Z_imag.append(-Z_total.imag + noise_imag)
        
        eis_data = {
            'timestamp': datetime.now(),
            'simulation_time_hours': self.simulation_time,
            'frequencies': frequencies.tolist(),
            'Z_real': Z_real,
            'Z_imag': Z_imag,
            'R_ohmic': R_ohmic,
            'R_activation': R_act,
            'R_concentration': R_conc,
            'degradation_state': self.degradation_state
        }
        
        return eis_data
    
    def generate_thermal_image(self) -> Dict:
        """Generate thermal imaging data"""
        print(f"Capturing thermal image at t={self.simulation_time:.1f}h")
        
        # Image dimensions
        width, height = 64, 48
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Base temperature pattern
        base_temp = 800 + 50 * self.degradation_state  # Temperature rises with degradation
        temp_variation = 30
        
        # Normal temperature distribution
        temperature_map = (base_temp + 
                          temp_variation * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.1) +
                          5 * np.random.normal(0, 1, (height, width)))
        
        # Add degradation-induced hot spots
        if self.degradation_state > 0.3:
            # Hot spot development
            hotspot_intensity = (self.degradation_state - 0.3) * 50
            hotspot_x, hotspot_y = 0.3, 0.7
            hotspot = hotspot_intensity * np.exp(-((X-hotspot_x)**2 + (Y-hotspot_y)**2) / 0.03)
            temperature_map += hotspot
        
        if self.degradation_state > 0.6:
            # Second hot spot
            hotspot2_intensity = (self.degradation_state - 0.6) * 40
            hotspot2_x, hotspot2_y = 0.7, 0.3
            hotspot2 = hotspot2_intensity * np.exp(-((X-hotspot2_x)**2 + (Y-hotspot2_y)**2) / 0.04)
            temperature_map += hotspot2
        
        thermal_data = {
            'timestamp': datetime.now(),
            'simulation_time_hours': self.simulation_time,
            'temperature_map': temperature_map.tolist(),
            'max_temperature': float(np.max(temperature_map)),
            'min_temperature': float(np.min(temperature_map)),
            'mean_temperature': float(np.mean(temperature_map)),
            'temperature_std': float(np.std(temperature_map)),
            'hotspot_detected': self.degradation_state > 0.3,
            'degradation_state': self.degradation_state
        }
        
        return thermal_data
    
    def check_acoustic_emission(self) -> Optional[Dict]:
        """Check for acoustic emission events"""
        # Event probability increases with degradation
        base_probability = 0.0001  # per time step
        degradation_factor = 1 + 10 * self.degradation_state
        event_probability = base_probability * degradation_factor
        
        if np.random.random() < event_probability:
            # Generate AE event
            amplitude = 30 + 30 * np.random.random() + 20 * self.degradation_state
            duration = 50 + 200 * np.random.random()
            energy = amplitude * duration / 100
            frequency_peak = 100 + 300 * np.random.random()
            
            # Event classification based on amplitude and degradation state
            if amplitude > 60:
                event_type = 'major_crack_propagation'
                severity = 'high'
            elif amplitude > 45:
                event_type = 'crack_propagation'
                severity = 'medium'
            else:
                event_type = 'micro_crack'
                severity = 'low'
            
            ae_event = {
                'timestamp': datetime.now(),
                'simulation_time_hours': self.simulation_time,
                'amplitude_db': amplitude,
                'duration_us': duration,
                'energy': energy,
                'frequency_peak_khz': frequency_peak,
                'event_type': event_type,
                'severity': severity,
                'x_location': np.random.random(),
                'y_location': np.random.random(),
                'degradation_state': self.degradation_state
            }
            
            print(f"AE Event detected: {event_type} (amplitude: {amplitude:.1f} dB)")
            return ae_event
        
        return None
    
    def adaptive_sampling_decision(self, recent_data: List[Dict]) -> Dict:
        """
        Make adaptive sampling decisions based on system state
        """
        if len(recent_data) < 10:
            return {'action': 'continue', 'reason': 'insufficient_data'}
        
        # Analyze recent trends
        voltages = [d['voltage'] for d in recent_data[-10:]]
        voltage_trend = np.polyfit(range(len(voltages)), voltages, 1)[0]
        voltage_std = np.std(voltages)
        
        temperatures = [d['temp_fuel_inlet'] for d in recent_data[-10:]]
        temp_std = np.std(temperatures)
        
        # Decision logic
        decisions = {
            'sampling_frequency': self.monitoring_config.high_freq_sampling,
            'trigger_eis': False,
            'trigger_thermal': False,
            'alert_level': 'normal'
        }
        
        # Voltage degradation detection
        if voltage_trend < -1e-4:  # Rapid voltage drop
            decisions['sampling_frequency'] *= 2  # Increase sampling
            decisions['trigger_eis'] = True
            decisions['alert_level'] = 'warning'
            
        # High voltage variability
        if voltage_std > 0.02:
            decisions['sampling_frequency'] *= 1.5
            decisions['trigger_thermal'] = True
            decisions['alert_level'] = 'caution'
        
        # Temperature instability
        if temp_std > 15:
            decisions['trigger_thermal'] = True
            decisions['alert_level'] = 'caution'
        
        # High degradation state
        if self.degradation_state > 0.7:
            decisions['sampling_frequency'] *= 3
            decisions['trigger_eis'] = True
            decisions['trigger_thermal'] = True
            decisions['alert_level'] = 'critical'
        
        return decisions
    
    def simulate_realtime_operation(self, duration_hours: float = 168):  # 1 week default
        """
        Simulate real-time operation with adaptive monitoring
        """
        print(f"Starting real-time simulation for {duration_hours} hours...")
        
        self.is_running = True
        start_time = time.time()
        
        # Data storage
        high_freq_data = []
        low_freq_data = []
        events_data = []
        
        # Timing variables
        last_eis_time = 0
        last_thermal_time = 0
        
        # Simulation loop
        time_step = 1.0 / self.monitoring_config.high_freq_sampling  # seconds
        
        while self.simulation_time < duration_hours and self.is_running:
            # Generate high-frequency data
            data_point = self.generate_high_frequency_data()
            high_freq_data.append(data_point)
            
            # Check for acoustic emission
            ae_event = self.check_acoustic_emission()
            if ae_event:
                events_data.append(ae_event)
            
            # Adaptive sampling decisions
            if len(high_freq_data) > 10:
                decisions = self.adaptive_sampling_decision(high_freq_data[-10:])
                
                # Trigger EIS if needed
                if (decisions['trigger_eis'] or 
                    (self.simulation_time - last_eis_time) >= 24):
                    eis_data = self.generate_eis_measurement()
                    low_freq_data.append(('eis', eis_data))
                    last_eis_time = self.simulation_time
                
                # Trigger thermal imaging if needed
                if (decisions['trigger_thermal'] or 
                    (self.simulation_time - last_thermal_time) >= 6):
                    thermal_data = self.generate_thermal_image()
                    low_freq_data.append(('thermal', thermal_data))
                    last_thermal_time = self.simulation_time
            
            # Update simulation time
            self.simulation_time += time_step / 3600  # Convert to hours
            
            # Sleep to simulate real-time (optional, for demonstration)
            # time.sleep(0.01)  # Uncomment for real-time simulation
            
            # Progress update
            if len(high_freq_data) % 100 == 0:
                print(f"Simulation progress: {self.simulation_time:.1f}h "
                      f"({len(high_freq_data)} data points, "
                      f"{len(events_data)} AE events)")
        
        # Save data
        self.save_realtime_data(high_freq_data, low_freq_data, events_data)
        
        print(f"Real-time simulation completed!")
        print(f"Generated {len(high_freq_data)} high-frequency data points")
        print(f"Generated {len(low_freq_data)} low-frequency measurements")
        print(f"Detected {len(events_data)} acoustic emission events")
        
        return {
            'high_freq_data': high_freq_data,
            'low_freq_data': low_freq_data,
            'events_data': events_data
        }
    
    def save_realtime_data(self, high_freq_data: List[Dict], 
                          low_freq_data: List[Tuple], 
                          events_data: List[Dict]):
        """Save real-time monitoring data"""
        
        # Convert high-frequency data to DataFrame
        df_high_freq = pd.DataFrame(high_freq_data)
        df_high_freq.to_csv(
            os.path.join(self.output_dir, 'realtime_operational_data.csv'),
            index=False
        )
        
        # Save events data
        if events_data:
            df_events = pd.DataFrame(events_data)
            df_events.to_csv(
                os.path.join(self.output_dir, 'acoustic_emission_events.csv'),
                index=False
            )
        
        # Save low-frequency data in HDF5
        with h5py.File(os.path.join(self.output_dir, 'realtime_monitoring.h5'), 'w') as f:
            # EIS measurements
            eis_group = f.create_group('eis_measurements')
            eis_count = 0
            
            # Thermal measurements
            thermal_group = f.create_group('thermal_measurements')
            thermal_count = 0
            
            for data_type, data in low_freq_data:
                if data_type == 'eis':
                    measurement_group = eis_group.create_group(f'measurement_{eis_count:03d}')
                    for key, value in data.items():
                        if key != 'timestamp':  # Skip timestamp for HDF5
                            measurement_group.create_dataset(key, data=value)
                    eis_count += 1
                    
                elif data_type == 'thermal':
                    image_group = thermal_group.create_group(f'image_{thermal_count:03d}')
                    for key, value in data.items():
                        if key != 'timestamp':
                            image_group.create_dataset(key, data=value)
                    thermal_count += 1
        
        # Save summary
        summary = {
            'dataset_info': {
                'name': 'Dataset 3 - Real-Time Monitoring Data',
                'simulation_duration_hours': self.simulation_time,
                'generation_timestamp': datetime.now().isoformat(),
                'final_degradation_state': self.degradation_state
            },
            'data_statistics': {
                'high_frequency_points': len(high_freq_data),
                'eis_measurements': sum(1 for dt, _ in low_freq_data if dt == 'eis'),
                'thermal_images': sum(1 for dt, _ in low_freq_data if dt == 'thermal'),
                'acoustic_events': len(events_data)
            },
            'monitoring_config': {
                'high_freq_sampling_hz': self.monitoring_config.high_freq_sampling,
                'eis_interval_hours': self.monitoring_config.eis_interval / 3600,
                'thermal_interval_hours': self.monitoring_config.thermal_interval / 3600
            }
        }
        
        with open(os.path.join(self.output_dir, 'dataset3_summary.yaml'), 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)

def main():
    """Main function to generate Dataset 3"""
    config_path = "config/simulation_config.yaml"
    
    # Create generator
    generator = Dataset3Generator(config_path)
    
    # Run real-time simulation
    results = generator.simulate_realtime_operation(duration_hours=168)  # 1 week
    
    print("\nDataset 3 generation completed!")

if __name__ == "__main__":
    main()