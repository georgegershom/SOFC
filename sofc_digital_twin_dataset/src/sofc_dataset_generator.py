"""
SOFC Digital Twin Dataset Generator
Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring
"""

import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import qmc
from datetime import datetime, timedelta
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SOFCPhysicsSimulator:
    """
    High-fidelity physics-based simulation for SOFC cells
    Simulates electrochemical, thermal, and structural responses
    """
    
    def __init__(self, grid_size=(50, 50, 10)):
        """
        Initialize the SOFC simulator
        
        Parameters:
        -----------
        grid_size : tuple
            3D grid dimensions (nx, ny, nz) for spatial fields
        """
        self.nx, self.ny, self.nz = grid_size
        self.setup_grid()
        self.setup_material_properties()
        
    def setup_grid(self):
        """Setup 3D computational grid"""
        # Physical dimensions of SOFC cell (in mm)
        self.Lx = 100  # Length
        self.Ly = 100  # Width
        self.Lz = 2.5  # Thickness (anode + electrolyte + cathode)
        
        # Grid points
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.z = np.linspace(0, self.Lz, self.nz)
        
        # 3D meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
    def setup_material_properties(self):
        """Define material properties for SOFC components"""
        self.materials = {
            'anode': {
                'thickness': 0.5,  # mm
                'porosity': 0.35,
                'tortuosity': 3.0,
                'ionic_conductivity': 2.0,  # S/m
                'electronic_conductivity': 3000,  # S/m
                'thermal_conductivity': 2.0,  # W/(m·K)
                'young_modulus': 50e9,  # Pa
                'poisson_ratio': 0.25,
                'CTE': 12.5e-6,  # 1/K
            },
            'electrolyte': {
                'thickness': 0.01,  # mm
                'ionic_conductivity': 10.0,  # S/m
                'thermal_conductivity': 2.2,  # W/(m·K)
                'young_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'CTE': 10.5e-6,  # 1/K
            },
            'cathode': {
                'thickness': 0.05,  # mm
                'porosity': 0.4,
                'tortuosity': 3.5,
                'ionic_conductivity': 1.5,  # S/m
                'electronic_conductivity': 100,  # S/m
                'thermal_conductivity': 1.5,  # W/(m·K)
                'young_modulus': 40e9,  # Pa
                'poisson_ratio': 0.25,
                'CTE': 13.0e-6,  # 1/K
            },
            'interconnect': {
                'thermal_conductivity': 25.0,  # W/(m·K)
                'young_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'CTE': 11.5e-6,  # 1/K
            }
        }
        
    def electrochemical_model(self, current_density, fuel_utilization, 
                             air_utilization, T_fuel, T_air, fuel_composition):
        """
        Simulate electrochemical fields
        
        Returns:
        --------
        dict : Containing voltage, current density distribution, species concentrations
        """
        # Operating temperature field (simplified)
        T_op = 800 + 50 * np.sin(np.pi * self.X / self.Lx) * np.cos(np.pi * self.Y / self.Ly)
        T_op += (T_fuel - 800) * np.exp(-self.Z / self.Lz)
        
        # Current density distribution (non-uniform due to fuel/air utilization)
        i_local = current_density * (1 + 0.3 * np.sin(2*np.pi * self.X / self.Lx) * 
                                    np.cos(2*np.pi * self.Y / self.Ly))
        i_local *= (1 - fuel_utilization * self.X / self.Lx)
        i_local *= (1 - air_utilization * self.Y / self.Ly)
        
        # Nernst voltage
        E_nernst = 1.253 - 2.4516e-4 * T_op  # Simplified
        
        # Activation overpotential (Butler-Volmer)
        eta_act = 0.05 + 0.1 * i_local / 10000  # Simplified
        
        # Ohmic overpotential
        eta_ohmic = i_local * (self.Lz * 1e-3) / self.materials['electrolyte']['ionic_conductivity']
        
        # Concentration overpotential
        eta_conc = 0.05 * (fuel_utilization + air_utilization)
        
        # Cell voltage
        V_cell = np.mean(E_nernst - eta_act - eta_ohmic - eta_conc)
        
        # Species concentrations (simplified distributions)
        H2_conc = fuel_composition['H2'] * (1 - fuel_utilization * self.X / self.Lx)
        H2O_conc = fuel_composition['H2O'] + fuel_utilization * fuel_composition['H2'] * self.X / self.Lx
        O2_conc = 0.21 * (1 - air_utilization * self.Y / self.Ly)
        
        return {
            'temperature': T_op,
            'current_density': i_local,
            'voltage': V_cell,
            'H2_concentration': H2_conc,
            'H2O_concentration': H2O_conc,
            'O2_concentration': O2_conc,
            'activation_overpotential': eta_act,
            'ohmic_overpotential': eta_ohmic,
            'concentration_overpotential': eta_conc
        }
    
    def thermal_model(self, electro_fields):
        """
        Simulate thermal fields based on electrochemical results
        
        Returns:
        --------
        dict : Temperature distribution and heat fluxes
        """
        T = electro_fields['temperature']
        i = electro_fields['current_density']
        
        # Heat generation due to ohmic losses
        q_ohmic = i**2 * self.Lz * 1e-3 / self.materials['electrolyte']['ionic_conductivity']
        
        # Heat generation due to activation losses
        q_act = i * electro_fields['activation_overpotential']
        
        # Total heat generation
        q_total = q_ohmic + q_act
        
        # Temperature field with heat generation (simplified heat equation solution)
        T_field = T + q_total / (self.materials['electrolyte']['thermal_conductivity'] * 1000)
        
        # Add thermal gradients
        T_field += 50 * np.exp(-((self.X - self.Lx/2)**2 + (self.Y - self.Ly/2)**2) / (self.Lx**2/4))
        
        # Heat flux vectors
        dT_dx = np.gradient(T_field, axis=0)
        dT_dy = np.gradient(T_field, axis=1)
        dT_dz = np.gradient(T_field, axis=2)
        
        q_x = -self.materials['electrolyte']['thermal_conductivity'] * dT_dx
        q_y = -self.materials['electrolyte']['thermal_conductivity'] * dT_dy
        q_z = -self.materials['electrolyte']['thermal_conductivity'] * dT_dz
        
        return {
            'temperature_field': T_field,
            'heat_generation': q_total,
            'heat_flux_x': q_x,
            'heat_flux_y': q_y,
            'heat_flux_z': q_z
        }
    
    def structural_model(self, thermal_fields, degradation_params):
        """
        Simulate thermo-structural fields
        
        Returns:
        --------
        dict : Stress, strain, displacement fields and failure metrics
        """
        T = thermal_fields['temperature_field']
        T_ref = 25  # Reference temperature (°C)
        
        # Material properties (using electrolyte as representative)
        E = self.materials['electrolyte']['young_modulus']
        nu = self.materials['electrolyte']['poisson_ratio']
        alpha = self.materials['electrolyte']['CTE']
        
        # Thermal strain
        epsilon_thermal = alpha * (T - T_ref)
        
        # Displacement field (simplified)
        u_x = alpha * (T - T_ref) * self.X
        u_y = alpha * (T - T_ref) * self.Y
        u_z = alpha * (T - T_ref) * self.Z * 0.1  # Constrained in z-direction
        
        # Add deformation due to degradation
        if degradation_params['crack_length'] > 0:
            crack_effect = degradation_params['crack_length'] * np.exp(
                -((self.X - self.Lx/2)**2 + (self.Y - self.Ly/2)**2) / (self.Lx**2/16)
            )
            u_x += crack_effect * 0.01
            u_y += crack_effect * 0.01
        
        # Strain tensor components
        epsilon_xx = np.gradient(u_x, axis=0) / (self.Lx / self.nx)
        epsilon_yy = np.gradient(u_y, axis=1) / (self.Ly / self.ny)
        epsilon_zz = np.gradient(u_z, axis=2) / (self.Lz / self.nz)
        epsilon_xy = 0.5 * (np.gradient(u_x, axis=1) / (self.Ly / self.ny) + 
                           np.gradient(u_y, axis=0) / (self.Lx / self.nx))
        epsilon_yz = 0.5 * (np.gradient(u_y, axis=2) / (self.Lz / self.nz) + 
                           np.gradient(u_z, axis=1) / (self.Ly / self.ny))
        epsilon_xz = 0.5 * (np.gradient(u_x, axis=2) / (self.Lz / self.nz) + 
                           np.gradient(u_z, axis=0) / (self.Lx / self.nx))
        
        # Stress tensor (using generalized Hooke's law)
        lambda_lame = E * nu / ((1 + nu) * (1 - 2*nu))
        mu_lame = E / (2 * (1 + nu))
        
        epsilon_vol = epsilon_xx + epsilon_yy + epsilon_zz
        
        sigma_xx = lambda_lame * epsilon_vol + 2 * mu_lame * epsilon_xx
        sigma_yy = lambda_lame * epsilon_vol + 2 * mu_lame * epsilon_yy
        sigma_zz = lambda_lame * epsilon_vol + 2 * mu_lame * epsilon_zz
        sigma_xy = 2 * mu_lame * epsilon_xy
        sigma_yz = 2 * mu_lame * epsilon_yz
        sigma_xz = 2 * mu_lame * epsilon_xz
        
        # Von Mises stress
        sigma_vm = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + 
                                  (sigma_yy - sigma_zz)**2 + 
                                  (sigma_zz - sigma_xx)**2 + 
                                  6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)))
        
        # Principal stresses (simplified - maximum principal stress)
        sigma_principal = np.maximum(sigma_xx, np.maximum(sigma_yy, sigma_zz))
        
        # Failure metrics
        # Stress intensity factors (simplified Mode I)
        if degradation_params['crack_length'] > 0:
            K_I = sigma_principal.max() * np.sqrt(np.pi * degradation_params['crack_length'] * 1e-3)
        else:
            K_I = 0
        
        # Strain energy release rate
        G = K_I**2 / E if K_I > 0 else 0
        
        # Creep damage accumulation (simplified Larson-Miller parameter)
        LMP = T.mean() * (20 + np.log10(degradation_params['time_hours']))
        creep_damage = LMP / 25000  # Normalized damage
        
        return {
            'displacement_x': u_x,
            'displacement_y': u_y,
            'displacement_z': u_z,
            'strain_xx': epsilon_xx,
            'strain_yy': epsilon_yy,
            'strain_zz': epsilon_zz,
            'strain_xy': epsilon_xy,
            'strain_yz': epsilon_yz,
            'strain_xz': epsilon_xz,
            'stress_xx': sigma_xx,
            'stress_yy': sigma_yy,
            'stress_zz': sigma_zz,
            'stress_xy': sigma_xy,
            'stress_yz': sigma_yz,
            'stress_xz': sigma_xz,
            'von_mises_stress': sigma_vm,
            'principal_stress_max': sigma_principal,
            'stress_intensity_factor_I': K_I,
            'strain_energy_release_rate': G,
            'creep_damage': creep_damage
        }
    
    def simulate(self, operating_conditions, material_params, degradation_params):
        """
        Run complete multi-physics simulation
        
        Parameters:
        -----------
        operating_conditions : dict
            Current density, fuel/air utilization, temperatures, composition
        material_params : dict
            Material property modifications
        degradation_params : dict
            Degradation state parameters
        
        Returns:
        --------
        dict : Complete simulation results
        """
        # Update material properties if provided
        if material_params:
            for component, props in material_params.items():
                if component in self.materials:
                    self.materials[component].update(props)
        
        # Run electrochemical model
        electro_fields = self.electrochemical_model(
            operating_conditions['current_density'],
            operating_conditions['fuel_utilization'],
            operating_conditions['air_utilization'],
            operating_conditions['fuel_temperature'],
            operating_conditions['air_temperature'],
            operating_conditions['fuel_composition']
        )
        
        # Run thermal model
        thermal_fields = self.thermal_model(electro_fields)
        
        # Run structural model
        structural_fields = self.structural_model(thermal_fields, degradation_params)
        
        # Combine all results
        results = {
            'operating_conditions': operating_conditions,
            'material_parameters': material_params,
            'degradation_parameters': degradation_params,
            'electrochemical': electro_fields,
            'thermal': thermal_fields,
            'structural': structural_fields,
            'timestamp': datetime.now().isoformat()
        }
        
        return results


class ExperimentalDataGenerator:
    """
    Generate synthetic experimental data that mimics real SOFC test rig measurements
    """
    
    def __init__(self):
        self.setup_sensor_characteristics()
        
    def setup_sensor_characteristics(self):
        """Define sensor noise and measurement characteristics"""
        self.sensor_noise = {
            'voltage': 0.001,  # V
            'current': 0.01,  # A
            'temperature': 0.5,  # °C
            'flow_rate': 0.02,  # relative
            'strain_gauge': 1e-6,  # strain
            'thermal_imaging': 2.0  # °C
        }
        
    def generate_operational_data(self, duration_hours=1000, sampling_rate=1.0):
        """
        Generate time-series operational data
        
        Parameters:
        -----------
        duration_hours : float
            Duration of experiment in hours
        sampling_rate : float
            Sampling frequency in Hz
        
        Returns:
        --------
        pd.DataFrame : Time-series data
        """
        n_samples = int(duration_hours * 3600 * sampling_rate)
        time = np.arange(n_samples) / sampling_rate / 3600  # Hours
        
        # Simulate varying load profile
        load_profile = 0.5 + 0.3 * np.sin(2 * np.pi * time / 100) + \
                      0.1 * np.sin(2 * np.pi * time / 10)
        
        # Current and voltage with degradation
        degradation_factor = 1 - 0.0001 * time  # Slow degradation
        current = 50 * load_profile + np.random.normal(0, self.sensor_noise['current'], n_samples)
        voltage = (0.8 - 0.1 * load_profile) * degradation_factor + \
                 np.random.normal(0, self.sensor_noise['voltage'], n_samples)
        
        # Temperatures
        T_inlet_fuel = 700 + 50 * load_profile + \
                      np.random.normal(0, self.sensor_noise['temperature'], n_samples)
        T_inlet_air = 650 + 50 * load_profile + \
                     np.random.normal(0, self.sensor_noise['temperature'], n_samples)
        T_outlet = 800 + 50 * load_profile + \
                  np.random.normal(0, self.sensor_noise['temperature'], n_samples)
        
        # Flow rates
        fuel_flow = 10 * (1 + 0.5 * load_profile) + \
                   np.random.normal(0, self.sensor_noise['flow_rate'], n_samples)
        air_flow = 100 * (1 + 0.5 * load_profile) + \
                  np.random.normal(0, self.sensor_noise['flow_rate'], n_samples)
        
        data = pd.DataFrame({
            'time_hours': time,
            'current_A': current,
            'voltage_V': voltage,
            'power_W': current * voltage,
            'T_inlet_fuel_C': T_inlet_fuel,
            'T_inlet_air_C': T_inlet_air,
            'T_outlet_C': T_outlet,
            'fuel_flow_slpm': fuel_flow,
            'air_flow_slpm': air_flow,
            'efficiency': voltage / 1.253 * degradation_factor
        })
        
        return data
    
    def generate_eis_data(self, frequencies=None, n_measurements=50):
        """
        Generate Electrochemical Impedance Spectroscopy data
        
        Parameters:
        -----------
        frequencies : np.array
            Frequency points for EIS (Hz)
        n_measurements : int
            Number of EIS measurements over time
        
        Returns:
        --------
        dict : EIS data
        """
        if frequencies is None:
            frequencies = np.logspace(-2, 5, 50)  # 0.01 Hz to 100 kHz
        
        eis_data = []
        
        for i in range(n_measurements):
            # Simulate impedance spectrum with degradation
            degradation = 1 + 0.01 * i
            
            # Ohmic resistance
            R_ohm = 0.1 * degradation
            
            # Charge transfer resistance
            R_ct = 0.2 * degradation
            
            # Warburg impedance (simplified)
            sigma_w = 0.05 * degradation
            
            # Calculate complex impedance
            omega = 2 * np.pi * frequencies
            Z_real = R_ohm + R_ct / (1 + (omega * R_ct * 1e-5)**2)
            Z_imag = -R_ct * omega * 1e-5 / (1 + (omega * R_ct * 1e-5)**2) - \
                    sigma_w / np.sqrt(omega)
            
            # Add noise
            Z_real += np.random.normal(0, 0.001, len(frequencies))
            Z_imag += np.random.normal(0, 0.001, len(frequencies))
            
            eis_data.append({
                'measurement_id': i,
                'time_hours': i * 24,  # Daily measurements
                'frequencies_Hz': frequencies,
                'Z_real_ohm': Z_real,
                'Z_imag_ohm': Z_imag,
                'R_ohmic': R_ohm,
                'R_charge_transfer': R_ct
            })
        
        return eis_data
    
    def generate_thermal_images(self, n_images=100, image_size=(64, 64)):
        """
        Generate synthetic thermal camera images
        
        Parameters:
        -----------
        n_images : int
            Number of thermal images
        image_size : tuple
            Image dimensions
        
        Returns:
        --------
        np.array : Thermal images
        """
        images = []
        
        for i in range(n_images):
            # Base temperature distribution
            x = np.linspace(0, 1, image_size[0])
            y = np.linspace(0, 1, image_size[1])
            X, Y = np.meshgrid(x, y)
            
            # Temperature field with hot spots
            T_base = 750 + 50 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
            
            # Add hot spots (potential failure points)
            n_hotspots = np.random.randint(1, 4)
            for _ in range(n_hotspots):
                x_center = np.random.rand()
                y_center = np.random.rand()
                intensity = np.random.uniform(20, 50)
                sigma = np.random.uniform(0.05, 0.15)
                
                hotspot = intensity * np.exp(-((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma**2))
                T_base += hotspot
            
            # Add noise
            T_base += np.random.normal(0, self.sensor_noise['thermal_imaging'], image_size)
            
            # Add time-dependent degradation pattern
            degradation_pattern = 10 * (i / n_images) * (X + Y) / 2
            T_base += degradation_pattern
            
            images.append(T_base)
        
        return np.array(images)
    
    def generate_strain_gauge_data(self, n_sensors=8, duration_hours=1000, sampling_rate=0.1):
        """
        Generate strain gauge sensor data
        
        Parameters:
        -----------
        n_sensors : int
            Number of strain gauge sensors
        duration_hours : float
            Duration in hours
        sampling_rate : float
            Sampling rate in Hz
        
        Returns:
        --------
        pd.DataFrame : Strain gauge data
        """
        n_samples = int(duration_hours * 3600 * sampling_rate)
        time = np.arange(n_samples) / sampling_rate / 3600
        
        data = {'time_hours': time}
        
        for sensor_id in range(n_sensors):
            # Base strain with thermal cycling
            thermal_strain = 100e-6 * np.sin(2 * np.pi * time / 24)  # Daily thermal cycling
            
            # Mechanical strain from pressure differential
            mechanical_strain = 50e-6 * (1 + 0.2 * np.sin(2 * np.pi * time / 100))
            
            # Creep strain (increasing over time)
            creep_strain = 10e-6 * time / duration_hours
            
            # Total strain with noise
            total_strain = thermal_strain + mechanical_strain + creep_strain + \
                          np.random.normal(0, self.sensor_noise['strain_gauge'], n_samples)
            
            data[f'strain_sensor_{sensor_id}_mu_strain'] = total_strain * 1e6  # Convert to microstrain
        
        return pd.DataFrame(data)
    
    def generate_acoustic_emission_events(self, duration_hours=1000):
        """
        Generate acoustic emission event data (crack formation/propagation)
        
        Parameters:
        -----------
        duration_hours : float
            Duration in hours
        
        Returns:
        --------
        pd.DataFrame : AE event data
        """
        # Generate random crack events with increasing frequency over time
        n_events = np.random.poisson(0.01 * duration_hours)
        
        events = []
        for i in range(n_events):
            # Event time (more likely later in life)
            t = duration_hours * (0.5 + 0.5 * np.random.beta(2, 1))
            
            # Event characteristics
            amplitude = np.random.lognormal(3, 0.5)  # dB
            duration_us = np.random.lognormal(2, 0.5)  # microseconds
            energy = amplitude * duration_us / 1000  # Arbitrary units
            
            # Location (assuming linear array of sensors)
            location = np.random.uniform(0, 100)  # mm along cell
            
            events.append({
                'event_id': i,
                'time_hours': t,
                'amplitude_dB': amplitude,
                'duration_us': duration_us,
                'energy': energy,
                'location_mm': location,
                'event_type': np.random.choice(['crack_initiation', 'crack_propagation', 'delamination'])
            })
        
        return pd.DataFrame(events)


class AdaptiveMonitoringDataGenerator:
    """
    Generate real-time monitoring data streams for adaptive digital twin
    """
    
    def __init__(self):
        self.current_state = {
            'voltage': 0.75,
            'current': 50.0,
            'temperature': 800.0,
            'degradation': 0.0
        }
        
    def generate_realtime_stream(self, duration_seconds=3600, frequency_hz=1.0):
        """
        Generate real-time operational data stream
        
        Parameters:
        -----------
        duration_seconds : int
            Duration of stream in seconds
        frequency_hz : float
            Update frequency in Hz
        
        Returns:
        --------
        list : Stream data packets
        """
        n_samples = int(duration_seconds * frequency_hz)
        dt = 1.0 / frequency_hz
        
        stream_data = []
        
        for i in range(n_samples):
            t = i * dt
            
            # Update state with dynamics
            self.current_state['current'] += np.random.normal(0, 0.1)
            self.current_state['voltage'] -= 1e-6 * dt  # Slow degradation
            self.current_state['temperature'] += np.random.normal(0, 0.5)
            self.current_state['degradation'] += 1e-7 * dt
            
            # Add disturbances
            if np.random.rand() < 0.01:  # 1% chance of disturbance
                self.current_state['current'] += np.random.normal(0, 5)
            
            packet = {
                'timestamp': datetime.now() + timedelta(seconds=t),
                'voltage_V': self.current_state['voltage'] + np.random.normal(0, 0.001),
                'current_A': self.current_state['current'] + np.random.normal(0, 0.01),
                'temperature_C': self.current_state['temperature'] + np.random.normal(0, 0.5),
                'fuel_flow_slpm': 10.0 + np.random.normal(0, 0.1),
                'air_flow_slpm': 100.0 + np.random.normal(0, 1.0),
                'power_W': self.current_state['voltage'] * self.current_state['current']
            }
            
            stream_data.append(packet)
        
        return stream_data
    
    def generate_adaptive_triggers(self, monitoring_data):
        """
        Generate triggers for adaptive model updates
        
        Parameters:
        -----------
        monitoring_data : pd.DataFrame
            Monitoring data
        
        Returns:
        --------
        list : Trigger events
        """
        triggers = []
        
        # Voltage drop trigger
        voltage_baseline = monitoring_data['voltage_V'].iloc[0]
        voltage_drops = monitoring_data[monitoring_data['voltage_V'] < 0.95 * voltage_baseline]
        
        for idx in voltage_drops.index:
            triggers.append({
                'time': monitoring_data.loc[idx, 'time_hours'],
                'trigger_type': 'voltage_drop',
                'severity': 'medium',
                'action': 'update_degradation_model'
            })
        
        # Temperature excursion trigger
        temp_excursions = monitoring_data[monitoring_data['T_outlet_C'] > 850]
        
        for idx in temp_excursions.index:
            triggers.append({
                'time': monitoring_data.loc[idx, 'time_hours'],
                'trigger_type': 'temperature_excursion',
                'severity': 'high',
                'action': 'high_fidelity_thermal_analysis'
            })
        
        # Efficiency degradation trigger
        if 'efficiency' in monitoring_data.columns:
            eff_baseline = monitoring_data['efficiency'].iloc[:100].mean()
            eff_degraded = monitoring_data[monitoring_data['efficiency'] < 0.9 * eff_baseline]
            
            for idx in eff_degraded.index:
                triggers.append({
                    'time': monitoring_data.loc[idx, 'time_hours'],
                    'trigger_type': 'efficiency_degradation',
                    'severity': 'low',
                    'action': 'recalibrate_model'
                })
        
        return triggers


def generate_complete_dataset(output_dir='../data'):
    """
    Generate the complete multi-fidelity dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SOFC Digital Twin Dataset Generation")
    print("=" * 80)
    
    # Initialize generators
    print("\n[1/6] Initializing simulators...")
    physics_sim = SOFCPhysicsSimulator(grid_size=(30, 30, 8))
    exp_gen = ExperimentalDataGenerator()
    monitor_gen = AdaptiveMonitoringDataGenerator()
    
    # 1. Generate high-fidelity simulation data
    print("\n[2/6] Generating high-fidelity physics simulation data...")
    
    # Design of Experiments using Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=7)
    n_simulations = 500
    samples = sampler.random(n=n_simulations)
    
    # Parameter ranges
    param_ranges = {
        'current_density': (1000, 10000),  # A/m²
        'fuel_utilization': (0.3, 0.85),
        'air_utilization': (0.2, 0.5),
        'fuel_temperature': (600, 800),  # °C
        'air_temperature': (500, 700),  # °C
        'crack_length': (0, 5),  # mm
        'porosity_change': (0.9, 1.2)  # relative
    }
    
    simulation_results = []
    
    with h5py.File(output_path / 'simulation' / 'high_fidelity_simulations.h5', 'w') as hf:
        for i in tqdm(range(n_simulations), desc="Running simulations"):
            # Scale samples to parameter ranges
            current_density = samples[i, 0] * (param_ranges['current_density'][1] - 
                                              param_ranges['current_density'][0]) + \
                             param_ranges['current_density'][0]
            fuel_util = samples[i, 1] * (param_ranges['fuel_utilization'][1] - 
                                        param_ranges['fuel_utilization'][0]) + \
                       param_ranges['fuel_utilization'][0]
            air_util = samples[i, 2] * (param_ranges['air_utilization'][1] - 
                                       param_ranges['air_utilization'][0]) + \
                      param_ranges['air_utilization'][0]
            T_fuel = samples[i, 3] * (param_ranges['fuel_temperature'][1] - 
                                     param_ranges['fuel_temperature'][0]) + \
                    param_ranges['fuel_temperature'][0]
            T_air = samples[i, 4] * (param_ranges['air_temperature'][1] - 
                                   param_ranges['air_temperature'][0]) + \
                   param_ranges['air_temperature'][0]
            crack_len = samples[i, 5] * (param_ranges['crack_length'][1] - 
                                        param_ranges['crack_length'][0]) + \
                       param_ranges['crack_length'][0]
            porosity = samples[i, 6] * (param_ranges['porosity_change'][1] - 
                                       param_ranges['porosity_change'][0]) + \
                      param_ranges['porosity_change'][0]
            
            # Setup simulation parameters
            operating_conditions = {
                'current_density': current_density,
                'fuel_utilization': fuel_util,
                'air_utilization': air_util,
                'fuel_temperature': T_fuel,
                'air_temperature': T_air,
                'fuel_composition': {
                    'H2': 0.89,
                    'H2O': 0.11,
                    'CO': 0.0,
                    'CO2': 0.0,
                    'CH4': 0.0
                }
            }
            
            material_params = {
                'anode': {'porosity': 0.35 * porosity},
                'cathode': {'porosity': 0.4 * porosity}
            }
            
            degradation_params = {
                'crack_length': crack_len,
                'time_hours': np.random.uniform(0, 10000)
            }
            
            # Run simulation
            results = physics_sim.simulate(operating_conditions, material_params, degradation_params)
            
            # Store in HDF5
            grp = hf.create_group(f'simulation_{i:04d}')
            
            # Store input parameters
            grp.attrs['current_density'] = current_density
            grp.attrs['fuel_utilization'] = fuel_util
            grp.attrs['air_utilization'] = air_util
            grp.attrs['fuel_temperature'] = T_fuel
            grp.attrs['air_temperature'] = T_air
            grp.attrs['crack_length'] = crack_len
            grp.attrs['porosity_factor'] = porosity
            
            # Store field data
            grp.create_dataset('temperature', data=results['thermal']['temperature_field'], 
                             compression='gzip')
            grp.create_dataset('current_density', data=results['electrochemical']['current_density'],
                             compression='gzip')
            grp.create_dataset('von_mises_stress', data=results['structural']['von_mises_stress'],
                             compression='gzip')
            grp.create_dataset('displacement_x', data=results['structural']['displacement_x'],
                             compression='gzip')
            grp.create_dataset('displacement_y', data=results['structural']['displacement_y'],
                             compression='gzip')
            grp.create_dataset('displacement_z', data=results['structural']['displacement_z'],
                             compression='gzip')
            
            # Store scalar outputs
            grp.attrs['voltage'] = results['electrochemical']['voltage']
            grp.attrs['max_stress'] = np.max(results['structural']['von_mises_stress'])
            grp.attrs['creep_damage'] = results['structural']['creep_damage']
    
    print(f"  Generated {n_simulations} multi-physics simulations")
    
    # 2. Generate experimental data
    print("\n[3/6] Generating experimental validation data...")
    
    # Operational time-series
    op_data = exp_gen.generate_operational_data(duration_hours=2000, sampling_rate=0.1)
    op_data.to_csv(output_path / 'experimental' / 'operational_data.csv', index=False)
    print(f"  Generated {len(op_data)} operational data points")
    
    # EIS data
    eis_data = exp_gen.generate_eis_data(n_measurements=100)
    with open(output_path / 'experimental' / 'eis_data.json', 'w') as f:
        json.dump(eis_data, f, default=str, indent=2)
    print(f"  Generated {len(eis_data)} EIS measurements")
    
    # Thermal images
    thermal_images = exp_gen.generate_thermal_images(n_images=200, image_size=(64, 64))
    np.save(output_path / 'experimental' / 'thermal_images.npy', thermal_images)
    print(f"  Generated {len(thermal_images)} thermal images")
    
    # Strain gauge data
    strain_data = exp_gen.generate_strain_gauge_data(n_sensors=8, duration_hours=2000)
    strain_data.to_csv(output_path / 'experimental' / 'strain_gauge_data.csv', index=False)
    print(f"  Generated strain gauge data from 8 sensors")
    
    # Acoustic emission events
    ae_events = exp_gen.generate_acoustic_emission_events(duration_hours=2000)
    ae_events.to_csv(output_path / 'experimental' / 'acoustic_emission_events.csv', index=False)
    print(f"  Generated {len(ae_events)} acoustic emission events")
    
    # 3. Generate monitoring data
    print("\n[4/6] Generating adaptive monitoring data...")
    
    # Real-time stream
    stream_data = monitor_gen.generate_realtime_stream(duration_seconds=7200, frequency_hz=1.0)
    stream_df = pd.DataFrame(stream_data)
    stream_df.to_csv(output_path / 'monitoring' / 'realtime_stream.csv', index=False)
    print(f"  Generated {len(stream_data)} real-time data packets")
    
    # Adaptive triggers
    triggers = monitor_gen.generate_adaptive_triggers(op_data)
    trigger_df = pd.DataFrame(triggers)
    if not trigger_df.empty:
        trigger_df.to_csv(output_path / 'monitoring' / 'adaptive_triggers.csv', index=False)
        print(f"  Generated {len(triggers)} adaptive trigger events")
    
    # 4. Generate metadata
    print("\n[5/6] Creating metadata...")
    
    metadata = {
        'dataset_name': 'SOFC Digital Twin Multi-Fidelity Dataset',
        'version': '1.0',
        'creation_date': datetime.now().isoformat(),
        'description': 'Multi-physics, multi-fidelity dataset for adaptive-scale SOFC digital twin',
        'data_sources': {
            'simulation': {
                'type': 'High-fidelity physics simulation',
                'n_samples': n_simulations,
                'grid_size': [30, 30, 8],
                'physics': ['electrochemical', 'thermal', 'structural']
            },
            'experimental': {
                'type': 'Synthetic experimental data',
                'duration_hours': 2000,
                'measurements': ['voltage', 'current', 'temperature', 'EIS', 'thermal_imaging', 
                               'strain_gauge', 'acoustic_emission']
            },
            'monitoring': {
                'type': 'Real-time monitoring streams',
                'frequency_hz': 1.0,
                'duration_seconds': 7200
            }
        },
        'parameter_ranges': param_ranges,
        'units': {
            'current_density': 'A/m²',
            'temperature': '°C',
            'stress': 'Pa',
            'strain': 'dimensionless',
            'displacement': 'mm',
            'voltage': 'V',
            'current': 'A'
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 5. Generate data splits for ML training
    print("\n[6/6] Creating train/validation/test splits...")
    
    # Create indices for data splitting
    indices = np.arange(n_simulations)
    np.random.shuffle(indices)
    
    train_size = int(0.7 * n_simulations)
    val_size = int(0.15 * n_simulations)
    
    splits = {
        'train': indices[:train_size].tolist(),
        'validation': indices[train_size:train_size+val_size].tolist(),
        'test': indices[train_size+val_size:].tolist()
    }
    
    with open(output_path / 'data_splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Validation: {len(splits['validation'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    print("\n" + "=" * 80)
    print("Dataset generation completed successfully!")
    print(f"Total dataset size: ~{calculate_dataset_size(output_path):.2f} MB")
    print("=" * 80)
    
    return output_path


def calculate_dataset_size(path):
    """Calculate total size of generated dataset in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)


if __name__ == "__main__":
    import os
    
    # Create output directory structure
    output_dir = Path("../data")
    for subdir in ['simulation', 'experimental', 'monitoring']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Generate the complete dataset
    dataset_path = generate_complete_dataset(output_dir)
    
    print(f"\nDataset saved to: {dataset_path.absolute()}")
    print("\nNext steps:")
    print("1. Use the visualization tools to explore the generated data")
    print("2. Train your Physics-Informed Neural Network using the simulation data")
    print("3. Validate with the experimental data")
    print("4. Test the adaptive monitoring system with the real-time streams")