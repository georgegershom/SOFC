"""
Generate Material and Geometric Properties Dataset
Essential for accurate CFD setup and understanding sensor limitations
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime

class MaterialPropertiesGenerator:
    def __init__(self):
        # Common pipe materials and their acoustic properties
        self.pipe_materials = {
            'PVC': {
                'density_kg_m3': 1380,
                'youngs_modulus_GPa': 3.4,
                'poisson_ratio': 0.4,
                'sound_speed_m_s': 2370,
                'thermal_conductivity_W_mK': 0.19,
                'roughness_mm': 0.0015,
                'damping_coefficient': 0.02,
                'max_pressure_bar': 10,
                'temperature_range_C': [-10, 60]
            },
            'Steel_Carbon': {
                'density_kg_m3': 7850,
                'youngs_modulus_GPa': 210,
                'poisson_ratio': 0.3,
                'sound_speed_m_s': 5960,
                'thermal_conductivity_W_mK': 45,
                'roughness_mm': 0.045,
                'damping_coefficient': 0.001,
                'max_pressure_bar': 100,
                'temperature_range_C': [-50, 500]
            },
            'Steel_Stainless': {
                'density_kg_m3': 8000,
                'youngs_modulus_GPa': 195,
                'poisson_ratio': 0.3,
                'sound_speed_m_s': 5790,
                'thermal_conductivity_W_mK': 16,
                'roughness_mm': 0.015,
                'damping_coefficient': 0.001,
                'max_pressure_bar': 150,
                'temperature_range_C': [-200, 800]
            },
            'Copper': {
                'density_kg_m3': 8960,
                'youngs_modulus_GPa': 120,
                'poisson_ratio': 0.34,
                'sound_speed_m_s': 4760,
                'thermal_conductivity_W_mK': 401,
                'roughness_mm': 0.0015,
                'damping_coefficient': 0.003,
                'max_pressure_bar': 50,
                'temperature_range_C': [-200, 200]
            },
            'Aluminum': {
                'density_kg_m3': 2700,
                'youngs_modulus_GPa': 70,
                'poisson_ratio': 0.33,
                'sound_speed_m_s': 6420,
                'thermal_conductivity_W_mK': 237,
                'roughness_mm': 0.002,
                'damping_coefficient': 0.002,
                'max_pressure_bar': 30,
                'temperature_range_C': [-200, 300]
            },
            'HDPE': {
                'density_kg_m3': 950,
                'youngs_modulus_GPa': 0.8,
                'poisson_ratio': 0.46,
                'sound_speed_m_s': 1950,
                'thermal_conductivity_W_mK': 0.48,
                'roughness_mm': 0.007,
                'damping_coefficient': 0.05,
                'max_pressure_bar': 8,
                'temperature_range_C': [-50, 80]
            },
            'Glass': {
                'density_kg_m3': 2500,
                'youngs_modulus_GPa': 70,
                'poisson_ratio': 0.22,
                'sound_speed_m_s': 5640,
                'thermal_conductivity_W_mK': 1.05,
                'roughness_mm': 0.001,
                'damping_coefficient': 0.005,
                'max_pressure_bar': 5,
                'temperature_range_C': [-50, 300]
            },
            'Concrete': {
                'density_kg_m3': 2300,
                'youngs_modulus_GPa': 30,
                'poisson_ratio': 0.2,
                'sound_speed_m_s': 3800,
                'thermal_conductivity_W_mK': 1.7,
                'roughness_mm': 1.5,
                'damping_coefficient': 0.03,
                'max_pressure_bar': 20,
                'temperature_range_C': [-30, 100]
            }
        }
        
        # Standard pipe geometries (ISO/DIN standards)
        self.standard_pipes = {
            'DN15': {'nominal_diameter_mm': 15, 'outer_diameter_mm': 21.3, 'wall_thickness_mm': [2.0, 2.8, 3.2]},
            'DN20': {'nominal_diameter_mm': 20, 'outer_diameter_mm': 26.9, 'wall_thickness_mm': [2.0, 2.3, 2.9]},
            'DN25': {'nominal_diameter_mm': 25, 'outer_diameter_mm': 33.7, 'wall_thickness_mm': [2.3, 2.9, 3.6]},
            'DN32': {'nominal_diameter_mm': 32, 'outer_diameter_mm': 42.4, 'wall_thickness_mm': [2.6, 3.2, 4.0]},
            'DN40': {'nominal_diameter_mm': 40, 'outer_diameter_mm': 48.3, 'wall_thickness_mm': [2.6, 3.2, 4.0]},
            'DN50': {'nominal_diameter_mm': 50, 'outer_diameter_mm': 60.3, 'wall_thickness_mm': [2.9, 3.6, 4.5]},
            'DN65': {'nominal_diameter_mm': 65, 'outer_diameter_mm': 76.1, 'wall_thickness_mm': [2.9, 3.6, 5.0]},
            'DN80': {'nominal_diameter_mm': 80, 'outer_diameter_mm': 88.9, 'wall_thickness_mm': [3.2, 4.0, 5.6]},
            'DN100': {'nominal_diameter_mm': 100, 'outer_diameter_mm': 114.3, 'wall_thickness_mm': [3.6, 4.5, 6.3]},
            'DN125': {'nominal_diameter_mm': 125, 'outer_diameter_mm': 141.3, 'wall_thickness_mm': [4.0, 5.0, 7.1]},
            'DN150': {'nominal_diameter_mm': 150, 'outer_diameter_mm': 168.3, 'wall_thickness_mm': [4.5, 5.6, 8.0]},
            'DN200': {'nominal_diameter_mm': 200, 'outer_diameter_mm': 219.1, 'wall_thickness_mm': [5.0, 6.3, 10.0]},
            'DN250': {'nominal_diameter_mm': 250, 'outer_diameter_mm': 273.0, 'wall_thickness_mm': [5.6, 7.1, 12.5]},
            'DN300': {'nominal_diameter_mm': 300, 'outer_diameter_mm': 323.9, 'wall_thickness_mm': [5.9, 8.0, 14.2]}
        }
        
        # Acoustic sensor specifications
        self.sensor_specs = {
            'Ultrasonic_Transit_Time': {
                'model': 'UFM-530',
                'manufacturer': 'Generic Industrial',
                'frequency_kHz': 1000,
                'frequency_range_kHz': [500, 2000],
                'beam_angle_deg': 3,
                'sensitivity_dB': -65,
                'max_temperature_C': 150,
                'max_pressure_bar': 40,
                'accuracy_percent': 0.5,
                'response_time_ms': 250,
                'pipe_diameter_range_mm': [15, 3000],
                'transducer_type': 'Piezoelectric',
                'mounting': 'Clamp-on',
                'output_signal': '4-20mA',
                'min_straight_pipe_diameters': 10
            },
            'Ultrasonic_Doppler': {
                'model': 'DFM-4000',
                'manufacturer': 'FlowTech Systems',
                'frequency_kHz': 640,
                'frequency_range_kHz': [200, 1000],
                'beam_angle_deg': 5,
                'sensitivity_dB': -70,
                'max_temperature_C': 100,
                'max_pressure_bar': 20,
                'accuracy_percent': 2.0,
                'response_time_ms': 500,
                'pipe_diameter_range_mm': [25, 1200],
                'transducer_type': 'Piezoelectric',
                'mounting': 'Clamp-on',
                'output_signal': 'Modbus RTU',
                'min_particle_size_micron': 100
            },
            'Acoustic_Emission': {
                'model': 'AE-1000',
                'manufacturer': 'AcousticMonitor Inc',
                'frequency_kHz': 150,
                'frequency_range_kHz': [20, 1000],
                'beam_angle_deg': 'Omnidirectional',
                'sensitivity_dB': -80,
                'max_temperature_C': 200,
                'max_pressure_bar': 50,
                'accuracy_percent': 5.0,
                'response_time_ms': 10,
                'pipe_diameter_range_mm': [10, 5000],
                'transducer_type': 'Piezoelectric',
                'mounting': 'Magnetic/Adhesive',
                'output_signal': 'Analog Voltage',
                'dynamic_range_dB': 100
            },
            'Sonar_Array': {
                'model': 'SONARtrac VF-100',
                'manufacturer': 'CiDRA Precision',
                'frequency_kHz': 100,
                'frequency_range_kHz': [10, 200],
                'beam_angle_deg': 'Array',
                'sensitivity_dB': -60,
                'max_temperature_C': 80,
                'max_pressure_bar': 100,
                'accuracy_percent': 1.0,
                'response_time_ms': 100,
                'pipe_diameter_range_mm': [50, 1500],
                'transducer_type': 'PVDF Array',
                'mounting': 'Wrap-around',
                'output_signal': 'Ethernet/IP',
                'array_elements': 32,
                'spatial_resolution_mm': 10
            },
            'Hydrophone': {
                'model': 'HYD-8000',
                'manufacturer': 'Underwater Acoustics',
                'frequency_kHz': 50,
                'frequency_range_kHz': [0.01, 100],
                'beam_angle_deg': 'Omnidirectional',
                'sensitivity_dB': -165,
                'max_temperature_C': 50,
                'max_pressure_bar': 200,
                'accuracy_percent': 3.0,
                'response_time_ms': 1,
                'pipe_diameter_range_mm': [25, 500],
                'transducer_type': 'Hydrophone',
                'mounting': 'Insertion',
                'output_signal': 'BNC Analog',
                'equivalent_noise_dB': 40
            },
            'Accelerometer': {
                'model': 'ACC-352C',
                'manufacturer': 'PCB Piezotronics',
                'frequency_kHz': 10,
                'frequency_range_kHz': [0.001, 30],
                'beam_angle_deg': 'N/A',
                'sensitivity_mV_g': 100,
                'max_temperature_C': 120,
                'max_pressure_bar': 'Atmospheric',
                'accuracy_percent': 1.0,
                'response_time_ms': 0.1,
                'pipe_diameter_range_mm': [1, 10000],  # Works with any size
                'transducer_type': 'ICP Accelerometer',
                'mounting': 'Stud/Adhesive',
                'output_signal': 'IEPE',
                'measurement_range_g': 50
            }
        }
    
    def generate_pipe_configurations(self):
        """Generate comprehensive pipe configuration dataset"""
        data_list = []
        
        for pipe_size, geometry in self.standard_pipes.items():
            for material_name, material_props in self.pipe_materials.items():
                for thickness in geometry['wall_thickness_mm']:
                    # Calculate inner diameter
                    inner_diameter = geometry['outer_diameter_mm'] - 2 * thickness
                    
                    # Calculate cross-sectional areas
                    area_outer = np.pi * (geometry['outer_diameter_mm']/1000)**2 / 4
                    area_inner = np.pi * (inner_diameter/1000)**2 / 4
                    area_wall = area_outer - area_inner
                    
                    # Calculate moment of inertia
                    I = np.pi * ((geometry['outer_diameter_mm']/1000)**4 - (inner_diameter/1000)**4) / 64
                    
                    # Natural frequency (first mode - simply supported pipe)
                    L = 6.0  # Assume 6m standard length
                    E = material_props['youngs_modulus_GPa'] * 1e9
                    rho = material_props['density_kg_m3']
                    A = area_wall
                    f_natural = (np.pi/2) * np.sqrt(E*I/(rho*A*L**4))
                    
                    # Ring frequency (breathing mode)
                    R = inner_diameter / 2000  # Convert to meters
                    c_material = material_props['sound_speed_m_s']
                    f_ring = c_material / (2 * np.pi * R)
                    
                    # Critical flow velocity (for vibration)
                    v_critical = 0.4 * c_material * thickness / geometry['outer_diameter_mm']
                    
                    # Acoustic impedance of pipe wall
                    Z_wall = material_props['density_kg_m3'] * material_props['sound_speed_m_s']
                    
                    # Transmission loss (simplified)
                    # Based on mass law at 1 kHz
                    f_test = 1000  # Hz
                    mass_per_area = material_props['density_kg_m3'] * thickness / 1000
                    TL_1kHz = 20 * np.log10(mass_per_area * 2 * np.pi * f_test / (2 * 1.2 * 343))
                    
                    data_list.append({
                        'pipe_size': pipe_size,
                        'material': material_name,
                        'outer_diameter_mm': geometry['outer_diameter_mm'],
                        'wall_thickness_mm': thickness,
                        'inner_diameter_mm': inner_diameter,
                        'length_m': L,
                        'cross_section_area_m2': area_inner,
                        'wall_area_m2': area_wall,
                        'moment_of_inertia_m4': I,
                        'material_density_kg_m3': material_props['density_kg_m3'],
                        'youngs_modulus_GPa': material_props['youngs_modulus_GPa'],
                        'poisson_ratio': material_props['poisson_ratio'],
                        'sound_speed_material_m_s': material_props['sound_speed_m_s'],
                        'thermal_conductivity_W_mK': material_props['thermal_conductivity_W_mK'],
                        'surface_roughness_mm': material_props['roughness_mm'],
                        'damping_coefficient': material_props['damping_coefficient'],
                        'natural_frequency_Hz': f_natural,
                        'ring_frequency_Hz': f_ring,
                        'critical_velocity_m_s': v_critical,
                        'acoustic_impedance_Pa_s_m': Z_wall,
                        'transmission_loss_1kHz_dB': TL_1kHz,
                        'max_pressure_bar': material_props['max_pressure_bar'],
                        'min_temperature_C': material_props['temperature_range_C'][0],
                        'max_temperature_C': material_props['temperature_range_C'][1],
                        'weight_per_meter_kg': area_wall * material_props['density_kg_m3'],
                        'volume_per_meter_m3': area_inner
                    })
        
        return pd.DataFrame(data_list)
    
    def generate_sensor_compatibility_matrix(self):
        """Generate sensor-pipe compatibility and performance matrix"""
        data_list = []
        
        for sensor_name, sensor in self.sensor_specs.items():
            for pipe_size, geometry in self.standard_pipes.items():
                for material_name in self.pipe_materials.keys():
                    # Check diameter compatibility
                    pipe_od = geometry['outer_diameter_mm']
                    # Handle special case for sensors that work with any diameter
                    if sensor['pipe_diameter_range_mm'] == 'Any':
                        diameter_compatible = True
                    else:
                        diameter_compatible = (
                            pipe_od >= sensor['pipe_diameter_range_mm'][0] and 
                            pipe_od <= sensor['pipe_diameter_range_mm'][1]
                        )
                    
                    # Check temperature compatibility
                    material_temp_max = self.pipe_materials[material_name]['temperature_range_C'][1]
                    temp_compatible = sensor['max_temperature_C'] >= material_temp_max
                    
                    # Calculate expected performance
                    if diameter_compatible:
                        # Estimate measurement uncertainty based on pipe size
                        size_factor = 1 + 0.5 * (pipe_od - 50) / 1000  # Increases with size
                        freq_factor = sensor['frequency_kHz'] / 1000  # Normalized frequency
                        material_factor = 1.5 if material_name in ['Concrete', 'HDPE'] else 1.0
                        
                        total_uncertainty = sensor['accuracy_percent'] * size_factor * material_factor
                        
                        # Signal attenuation through pipe wall (for clamp-on sensors)
                        if sensor['mounting'] == 'Clamp-on':
                            wall_thickness = geometry['wall_thickness_mm'][1]  # Use medium thickness
                            Z_material = self.pipe_materials[material_name]['density_kg_m3'] * \
                                       self.pipe_materials[material_name]['sound_speed_m_s']
                            Z_water = 1.48e6  # Acoustic impedance of water
                            
                            # Transmission coefficient
                            T = 4 * Z_material * Z_water / (Z_material + Z_water)**2
                            signal_loss_dB = -10 * np.log10(T)
                        else:
                            signal_loss_dB = 0
                        
                        # Estimate SNR
                        base_snr = 40  # dB
                        snr = base_snr - signal_loss_dB - 10 * np.log10(size_factor)
                        
                        # Maximum measurable flow velocity (based on sensor type)
                        if 'Transit_Time' in sensor_name:
                            max_velocity = 10.0  # m/s
                        elif 'Doppler' in sensor_name:
                            max_velocity = 5.0  # m/s (needs particles)
                        else:
                            max_velocity = 15.0  # m/s
                        
                        performance_score = min(100, 100 * np.exp(-total_uncertainty/10) * (snr/40))
                    else:
                        total_uncertainty = None
                        signal_loss_dB = None
                        snr = None
                        max_velocity = None
                        performance_score = 0
                    
                    data_list.append({
                        'sensor_type': sensor_name,
                        'sensor_model': sensor['model'],
                        'pipe_size': pipe_size,
                        'pipe_material': material_name,
                        'pipe_outer_diameter_mm': pipe_od,
                        'diameter_compatible': diameter_compatible,
                        'temperature_compatible': temp_compatible,
                        'sensor_frequency_kHz': sensor['frequency_kHz'],
                        'sensor_mounting': sensor['mounting'],
                        'measurement_uncertainty_percent': total_uncertainty,
                        'signal_loss_dB': signal_loss_dB,
                        'signal_to_noise_ratio_dB': snr,
                        'max_measurable_velocity_m_s': max_velocity,
                        'performance_score': performance_score,
                        'min_straight_pipe_required_diameters': sensor.get('min_straight_pipe_diameters', 5),
                        'response_time_ms': sensor['response_time_ms'],
                        'output_signal': sensor['output_signal']
                    })
        
        return pd.DataFrame(data_list)
    
    def generate_boundary_conditions(self):
        """Generate boundary conditions for CFD simulations"""
        data_list = []
        
        # Common flow conditions
        flow_conditions = [
            {'name': 'Low_Flow', 'velocity_m_s': 0.5, 'reynolds': 25000, 'turbulent': True},
            {'name': 'Medium_Flow', 'velocity_m_s': 1.5, 'reynolds': 75000, 'turbulent': True},
            {'name': 'High_Flow', 'velocity_m_s': 3.0, 'reynolds': 150000, 'turbulent': True},
            {'name': 'Laminar', 'velocity_m_s': 0.05, 'reynolds': 2000, 'turbulent': False},
            {'name': 'Transitional', 'velocity_m_s': 0.1, 'reynolds': 3000, 'turbulent': False}
        ]
        
        # Inlet boundary conditions
        for condition in flow_conditions:
            for pipe_size in ['DN50', 'DN100', 'DN150']:
                geometry = self.standard_pipes[pipe_size]
                inner_d = geometry['outer_diameter_mm'] - 2*geometry['wall_thickness_mm'][1]
                
                # Turbulence parameters
                if condition['turbulent']:
                    # Turbulent intensity based on Reynolds number
                    I = 0.16 * condition['reynolds']**(-1/8)
                    # Turbulent length scale
                    l_t = 0.07 * inner_d / 1000  # 7% of diameter
                    # Turbulent kinetic energy
                    k = 1.5 * (condition['velocity_m_s'] * I)**2
                    # Dissipation rate
                    C_mu = 0.09
                    epsilon = C_mu**0.75 * k**1.5 / l_t
                    # Specific dissipation rate
                    omega = epsilon / (C_mu * k)
                else:
                    I = 0.01
                    k = 1.5 * (condition['velocity_m_s'] * I)**2
                    epsilon = 0.001
                    omega = 1.0
                
                # Wall boundary conditions for different materials
                for material in ['PVC', 'Steel_Carbon', 'Concrete']:
                    roughness = self.pipe_materials[material]['roughness_mm'] / 1000
                    
                    # Wall function parameters
                    y_plus_target = 30  # Target y+ for wall functions
                    nu = 1e-6  # Kinematic viscosity of water
                    u_tau = condition['velocity_m_s'] * np.sqrt(0.008)  # Friction velocity estimate
                    first_cell_height = y_plus_target * nu / u_tau
                    
                    data_list.append({
                        'condition_name': condition['name'],
                        'pipe_size': pipe_size,
                        'pipe_material': material,
                        'inner_diameter_mm': inner_d,
                        # Inlet conditions
                        'inlet_velocity_m_s': condition['velocity_m_s'],
                        'inlet_turbulent_intensity': I,
                        'inlet_turbulent_length_scale_m': l_t,
                        'inlet_turbulent_kinetic_energy_m2_s2': k,
                        'inlet_dissipation_rate_m2_s3': epsilon,
                        'inlet_specific_dissipation_rate_1_s': omega,
                        'reynolds_number': condition['reynolds'],
                        # Wall conditions
                        'wall_roughness_m': roughness,
                        'wall_roughness_constant': 0.5,  # For wall functions
                        'first_cell_height_m': first_cell_height,
                        'y_plus_target': y_plus_target,
                        # Outlet conditions
                        'outlet_pressure_gauge_Pa': 0,
                        'outlet_backflow_turbulent_intensity': 0.05,
                        'outlet_backflow_length_scale_m': 0.1 * inner_d / 1000,
                        # Numerical settings
                        'recommended_time_step_s': 0.001 * inner_d / (1000 * condition['velocity_m_s']),
                        'recommended_mesh_size_m': inner_d / 100000,  # 100 cells across diameter
                        'courant_number_max': 1.0,
                        # Reference values
                        'reference_pressure_Pa': 101325,
                        'reference_temperature_K': 293.15,
                        'reference_density_kg_m3': 998.2
                    })
        
        return pd.DataFrame(data_list)
    
    def generate_calibration_standards(self):
        """Generate calibration standards and reference data"""
        data_list = []
        
        # ISO/ASME standards for flow measurement
        standards = [
            {
                'standard': 'ISO 17089-1:2019',
                'description': 'Ultrasonic meters for gas',
                'accuracy_class': 0.5,
                'calibration_interval_months': 12,
                'test_points': [0.1, 0.2, 0.4, 0.7, 1.0],  # Fraction of max flow
                'required_straight_pipe_upstream': 20,
                'required_straight_pipe_downstream': 10
            },
            {
                'standard': 'ASME MFC-5M-1985',
                'description': 'Transit-time ultrasonic flowmeters',
                'accuracy_class': 1.0,
                'calibration_interval_months': 24,
                'test_points': [0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                'required_straight_pipe_upstream': 10,
                'required_straight_pipe_downstream': 5
            },
            {
                'standard': 'ISO 12242:2012',
                'description': 'Ultrasonic transit-time meters for liquid',
                'accuracy_class': 0.3,
                'calibration_interval_months': 12,
                'test_points': [0.04, 0.1, 0.4, 0.7, 1.0],
                'required_straight_pipe_upstream': 15,
                'required_straight_pipe_downstream': 5
            }
        ]
        
        # Generate calibration data for each standard
        for std in standards:
            for test_flow_fraction in std['test_points']:
                for pipe_size in ['DN50', 'DN100', 'DN200']:
                    geometry = self.standard_pipes[pipe_size]
                    inner_d = (geometry['outer_diameter_mm'] - 2*geometry['wall_thickness_mm'][1]) / 1000
                    
                    # Calculate flow parameters
                    area = np.pi * inner_d**2 / 4
                    max_velocity = 5.0  # m/s typical max
                    test_velocity = max_velocity * test_flow_fraction
                    volumetric_flow = area * test_velocity  # m³/s
                    
                    # Reference measurements (simulated high-accuracy reference)
                    ref_velocity = test_velocity * (1 + np.random.normal(0, 0.0001))  # 0.01% uncertainty
                    ref_flow = area * ref_velocity
                    
                    # Measurement requirements
                    max_error = std['accuracy_class'] * test_flow_fraction / 100
                    
                    # Environmental conditions for calibration
                    for temperature in [15, 20, 25, 30]:
                        # Water properties at temperature
                        water_density = 999.1 - 0.0875 * (temperature - 4)**1.68
                        water_viscosity = 1.002e-3 * np.exp(-0.0248 * (temperature - 20))
                        water_sound_speed = 1402.385 + 5.03830*temperature - 5.81090e-2*temperature**2
                        
                        # Calculate Reynolds number
                        reynolds = water_density * test_velocity * inner_d / water_viscosity
                        
                        data_list.append({
                            'standard': std['standard'],
                            'standard_description': std['description'],
                            'pipe_size': pipe_size,
                            'inner_diameter_m': inner_d,
                            'test_flow_fraction': test_flow_fraction,
                            'test_velocity_m_s': test_velocity,
                            'volumetric_flow_m3_s': volumetric_flow,
                            'volumetric_flow_m3_h': volumetric_flow * 3600,
                            'reference_velocity_m_s': ref_velocity,
                            'reference_flow_m3_s': ref_flow,
                            'max_allowable_error_percent': std['accuracy_class'],
                            'temperature_C': temperature,
                            'water_density_kg_m3': water_density,
                            'water_viscosity_Pa_s': water_viscosity,
                            'water_sound_speed_m_s': water_sound_speed,
                            'reynolds_number': reynolds,
                            'flow_regime': 'Turbulent' if reynolds > 4000 else 'Transitional' if reynolds > 2300 else 'Laminar',
                            'straight_pipe_upstream_diameters': std['required_straight_pipe_upstream'],
                            'straight_pipe_downstream_diameters': std['required_straight_pipe_downstream'],
                            'calibration_interval_months': std['calibration_interval_months'],
                            'k_factor': 1 / (area * 1000),  # Pulses per liter
                            'meter_factor': ref_flow / volumetric_flow  # Correction factor
                        })
        
        return pd.DataFrame(data_list)
    
    def save_all_datasets(self):
        """Generate and save all material property datasets"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("Generating pipe configurations dataset...")
        pipe_configs = self.generate_pipe_configurations()
        pipe_configs.to_csv('pipe_configurations.csv', index=False)
        
        print("Generating sensor compatibility matrix...")
        sensor_matrix = self.generate_sensor_compatibility_matrix()
        sensor_matrix.to_csv('sensor_compatibility.csv', index=False)
        
        print("Generating CFD boundary conditions...")
        boundary_conditions = self.generate_boundary_conditions()
        boundary_conditions.to_csv('cfd_boundary_conditions.csv', index=False)
        
        print("Generating calibration standards...")
        calibration_data = self.generate_calibration_standards()
        calibration_data.to_csv('calibration_standards.csv', index=False)
        
        # Save material properties as JSON for easy lookup
        with open('material_properties.json', 'w') as f:
            json.dump(self.pipe_materials, f, indent=2)
        
        # Save sensor specifications
        with open('sensor_specifications.json', 'w') as f:
            json.dump(self.sensor_specs, f, indent=2)
        
        # Save pipe standards
        with open('pipe_standards.json', 'w') as f:
            json.dump(self.standard_pipes, f, indent=2)
        
        # Create summary
        summary = {
            'generation_timestamp': timestamp,
            'datasets': {
                'pipe_configurations': {
                    'file': 'pipe_configurations.csv',
                    'records': len(pipe_configs),
                    'parameters': list(pipe_configs.columns)[:10]  # First 10 columns
                },
                'sensor_compatibility': {
                    'file': 'sensor_compatibility.csv',
                    'records': len(sensor_matrix),
                    'sensor_types': len(self.sensor_specs),
                    'pipe_sizes': len(self.standard_pipes),
                    'materials': len(self.pipe_materials)
                },
                'boundary_conditions': {
                    'file': 'cfd_boundary_conditions.csv',
                    'records': len(boundary_conditions),
                    'flow_conditions': 5,
                    'pipe_sizes_tested': 3
                },
                'calibration_standards': {
                    'file': 'calibration_standards.csv',
                    'records': len(calibration_data),
                    'standards': 3,
                    'test_temperatures': 4
                }
            },
            'reference_files': {
                'materials': 'material_properties.json',
                'sensors': 'sensor_specifications.json',
                'pipes': 'pipe_standards.json'
            },
            'total_configurations': len(pipe_configs),
            'total_data_points': len(pipe_configs) + len(sensor_matrix) + len(boundary_conditions) + len(calibration_data)
        }
        
        with open('material_properties_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nMaterial properties datasets saved with timestamp: {timestamp}")
        return summary

if __name__ == "__main__":
    generator = MaterialPropertiesGenerator()
    summary = generator.save_all_datasets()
    print("\nGeneration complete!")
    print(f"Total data points generated: {summary['total_data_points']}")