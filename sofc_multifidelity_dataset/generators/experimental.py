"""
Experimental Data Simulator for SOFC
Generates realistic experimental-like data with noise, measurement artifacts, and degradation patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import h5py
import json
from datetime import datetime
import sys
import os
from scipy import signal, ndimage, interpolate
from scipy.stats import norm, lognorm
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sofc_physics import SOFCOperatingConditions
from generators.high_fidelity import HighFidelitySOFCModel


class ExperimentalDataSimulator:
    """Simulate realistic experimental measurements with noise and artifacts"""
    
    def __init__(self, config: dict):
        self.config = config
        self.exp_config = config['dataset']['experimental']
        self.hf_model = HighFidelitySOFCModel(config)
        
    def add_measurement_noise(self, value: float, noise_level: float = 0.02,
                            noise_type: str = 'gaussian') -> float:
        """Add realistic measurement noise"""
        if noise_type == 'gaussian':
            return value * (1 + np.random.normal(0, noise_level))
        elif noise_type == 'lognormal':
            return value * np.random.lognormal(0, noise_level)
        elif noise_type == 'uniform':
            return value * (1 + np.random.uniform(-noise_level, noise_level))
        else:
            return value
    
    def simulate_iv_curve(self, conditions: SOFCOperatingConditions) -> Dict:
        """Simulate I-V curve measurement"""
        
        # Current density sweep
        i_points = np.linspace(0, 10000, 50)  # 0 to 1 A/cm²
        v_points = []
        p_points = []
        
        for i_density in i_points:
            # Update conditions
            cond_temp = SOFCOperatingConditions(
                temperature=conditions.temperature,
                pressure=conditions.pressure,
                current_density=i_density,
                fuel_utilization=conditions.fuel_utilization,
                air_utilization=conditions.air_utilization,
                inlet_fuel_composition=conditions.inlet_fuel_composition,
                inlet_air_composition=conditions.inlet_air_composition,
                time=conditions.time
            )
            
            # Calculate voltage (simplified from HF model)
            from models.sofc_physics import ElectrochemicalModel
            electrochem = ElectrochemicalModel(self.config)
            V_cell, overpotentials = electrochem.cell_voltage(cond_temp)
            
            # Add experimental noise
            V_measured = self.add_measurement_noise(V_cell, 0.005, 'gaussian')
            
            # Add systematic errors (e.g., contact resistance)
            contact_resistance = 0.1e-4  # 0.1 mΩ·cm²
            V_measured -= i_density * contact_resistance
            
            v_points.append(max(0, V_measured))
            p_points.append(V_measured * i_density)
        
        # Add oscilloscope-like noise
        v_points = np.array(v_points)
        noise = 0.001 * np.sin(2 * np.pi * 50 * np.linspace(0, 1, len(v_points)))  # 50 Hz noise
        v_points += noise
        
        return {
            'current_density': i_points,
            'voltage': v_points,
            'power_density': np.array(p_points),
            'ocv': v_points[0],  # Open circuit voltage
            'max_power': np.max(p_points),
            'max_power_current': i_points[np.argmax(p_points)]
        }
    
    def simulate_eis(self, conditions: SOFCOperatingConditions) -> Dict:
        """Simulate Electrochemical Impedance Spectroscopy"""
        
        # Frequency range (0.01 Hz to 100 kHz)
        frequencies = np.logspace(-2, 5, 100)
        
        # Equivalent circuit parameters (RQ-RQ-RW model)
        R_ohm = 0.1  # Ohmic resistance
        
        # High frequency arc (charge transfer)
        R_ct = 0.2
        Q_ct = 1e-3
        n_ct = 0.85
        
        # Low frequency arc (mass transport)
        R_mt = 0.3
        Q_mt = 1e-2
        n_mt = 0.7
        
        # Warburg element
        W = 0.1
        
        # Temperature dependence
        T_factor = np.exp(-10000 / (8.314 * conditions.temperature))
        R_ct *= T_factor
        R_mt *= T_factor
        
        # Calculate impedance
        omega = 2 * np.pi * frequencies
        
        # CPE impedance: Z_CPE = 1 / (Q * (jω)^n)
        Z_ct = R_ct / (1 + (1j * omega * R_ct * Q_ct)**n_ct)
        Z_mt = R_mt / (1 + (1j * omega * R_mt * Q_mt)**n_mt)
        
        # Warburg impedance
        Z_w = W / np.sqrt(1j * omega)
        
        # Total impedance
        Z_total = R_ohm + Z_ct + Z_mt + Z_w
        
        # Add measurement noise
        noise_real = np.random.normal(0, 0.001, len(frequencies))
        noise_imag = np.random.normal(0, 0.001, len(frequencies))
        
        Z_real = np.real(Z_total) + noise_real
        Z_imag = np.imag(Z_total) + noise_imag
        
        # Add artifacts (e.g., inductance at high frequency)
        L = 1e-7  # Small inductance
        Z_imag += omega * L
        
        return {
            'frequency': frequencies,
            'Z_real': Z_real,
            'Z_imag': Z_imag,
            'Z_magnitude': np.abs(Z_total),
            'Z_phase': np.angle(Z_total, deg=True),
            'R_total': R_ohm + R_ct + R_mt,
            'characteristic_frequency': 1 / (2 * np.pi * R_ct * Q_ct)
        }
    
    def simulate_thermography(self, conditions: SOFCOperatingConditions) -> Dict:
        """Simulate infrared thermography measurement"""
        
        # Generate temperature field (simplified 2D)
        nx, ny = 256, 256  # IR camera resolution
        
        # Base temperature field from model
        from models.sofc_physics import ThermalModel, ElectrochemicalModel
        thermal = ThermalModel(self.config)
        electrochem = ElectrochemicalModel(self.config)
        
        V_cell, overpotentials = electrochem.cell_voltage(conditions)
        T_field = thermal.temperature_distribution_2D(
            conditions, V_cell, overpotentials['E_nernst'], (nx, ny)
        )
        
        # Add realistic IR camera effects
        
        # 1. Emissivity variations
        emissivity = 0.85 + 0.1 * np.random.randn(nx, ny)
        T_apparent = T_field * emissivity
        
        # 2. Spatial resolution limits (blur)
        T_apparent = ndimage.gaussian_filter(T_apparent, sigma=2)
        
        # 3. Thermal noise (NETD ~ 20 mK)
        NETD = 0.02  # K
        T_measured = T_apparent + np.random.normal(0, NETD, (nx, ny))
        
        # 4. Bad pixels
        n_bad_pixels = int(0.001 * nx * ny)
        bad_x = np.random.randint(0, nx, n_bad_pixels)
        bad_y = np.random.randint(0, ny, n_bad_pixels)
        T_measured[bad_x, bad_y] = 0
        
        # 5. Reflection artifacts
        reflection_pattern = 10 * np.sin(2 * np.pi * np.linspace(0, 1, nx)).reshape(-1, 1)
        T_measured += reflection_pattern
        
        # 6. Vignetting (darker at edges)
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        vignette = 1 - 0.3 * (X**2 + Y**2)
        T_measured *= vignette
        
        # Statistics
        T_stats = {
            'T_max': np.max(T_measured),
            'T_min': np.min(T_measured[T_measured > 0]),
            'T_mean': np.mean(T_measured[T_measured > 0]),
            'T_std': np.std(T_measured[T_measured > 0]),
            'hot_spots': np.sum(T_measured > conditions.temperature + 50),
            'gradient_max': np.max(np.gradient(T_measured)[0])
        }
        
        return {
            'temperature_field': T_measured,
            'resolution': (nx, ny),
            'stats': T_stats
        }
    
    def simulate_sem_images(self, conditions: SOFCOperatingConditions,
                           degradation_level: float = 0.0) -> Dict:
        """Simulate SEM/FIB images showing microstructural degradation"""
        
        # Image size (typical SEM resolution)
        img_size = (1024, 1024)
        
        # Generate microstructure
        microstructure = np.zeros(img_size)
        
        # Ni particles (bright in SEM)
        n_particles = 500
        for _ in range(n_particles):
            x, y = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
            
            # Particle size increases with degradation (coarsening)
            base_radius = 5
            radius = base_radius * (1 + 2 * degradation_level)
            
            # Draw particle
            yy, xx = np.ogrid[-x:img_size[0]-x, -y:img_size[1]-y]
            mask = xx*xx + yy*yy <= radius*radius
            microstructure[mask] = 255
        
        # YSZ phase (medium gray)
        ysz_mask = microstructure == 0
        microstructure[ysz_mask] = 128 + 30 * np.random.randn(np.sum(ysz_mask))
        
        # Pores (dark)
        n_pores = 200
        for _ in range(n_pores):
            x, y = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
            radius = np.random.randint(3, 10)
            yy, xx = np.ogrid[-x:img_size[0]-x, -y:img_size[1]-y]
            mask = xx*xx + yy*yy <= radius*radius
            microstructure[mask] = 0
        
        # Add cracks with degradation
        if degradation_level > 0.3:
            n_cracks = int(10 * degradation_level)
            for _ in range(n_cracks):
                # Random crack path
                start = (np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1]))
                length = np.random.randint(50, 200)
                angle = np.random.uniform(0, 2*np.pi)
                
                for l in range(length):
                    x = int(start[0] + l * np.cos(angle))
                    y = int(start[1] + l * np.sin(angle))
                    if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
                        microstructure[x:x+2, y:y+2] = 0
                    angle += np.random.normal(0, 0.1)  # Crack wandering
        
        # Add SEM imaging artifacts
        # 1. Gaussian noise
        noise = np.random.normal(0, 5, img_size)
        microstructure += noise
        
        # 2. Charging effects (bright edges)
        edges = cv2.Canny(microstructure.astype(np.uint8), 50, 150)
        microstructure += edges * 20
        
        # 3. Beam damage (slight blur over time)
        if degradation_level > 0:
            microstructure = ndimage.gaussian_filter(microstructure, sigma=degradation_level)
        
        # Clip to valid range
        microstructure = np.clip(microstructure, 0, 255)
        
        # Calculate microstructural parameters
        ni_fraction = np.sum(microstructure > 200) / microstructure.size
        pore_fraction = np.sum(microstructure < 50) / microstructure.size
        
        # Particle size distribution (simplified)
        from scipy import ndimage
        labeled, n_features = ndimage.label(microstructure > 200)
        particle_sizes = []
        for i in range(1, n_features + 1):
            particle_sizes.append(np.sum(labeled == i))
        
        if particle_sizes:
            avg_particle_size = np.mean(particle_sizes)
        else:
            avg_particle_size = 0
        
        return {
            'image': microstructure,
            'resolution': img_size,
            'ni_fraction': ni_fraction,
            'porosity': pore_fraction,
            'avg_particle_size': avg_particle_size,
            'n_cracks': int(10 * degradation_level) if degradation_level > 0.3 else 0,
            'degradation_visual': degradation_level
        }
    
    def simulate_xrd_stress(self, stress_level: float) -> Dict:
        """Simulate X-ray diffraction stress measurement"""
        
        # 2θ angles
        two_theta = np.linspace(20, 80, 500)
        
        # YSZ peaks (simplified)
        peak_positions = [30.2, 35.1, 50.3, 59.8, 62.7]  # degrees
        peak_intensities = [100, 20, 45, 35, 25]
        
        # Stress-induced peak shift
        strain = stress_level / 200e9  # Approximate Young's modulus
        peak_shift = strain * 0.5  # Simplified relationship
        
        # Generate diffraction pattern
        intensity = np.zeros_like(two_theta)
        
        for pos, inten in zip(peak_positions, peak_intensities):
            # Shifted position due to stress
            shifted_pos = pos * (1 + peak_shift)
            
            # Peak profile (Voigt function approximated as Gaussian)
            sigma = 0.1 + 0.05 * stress_level / 100e6  # Peak broadening with stress
            peak = inten * np.exp(-(two_theta - shifted_pos)**2 / (2 * sigma**2))
            intensity += peak
        
        # Add background
        background = 10 + 5 * np.exp(-two_theta / 30)
        intensity += background
        
        # Add noise
        intensity += np.random.normal(0, 2, len(two_theta))
        intensity = np.maximum(intensity, 0)
        
        # Calculate stress from peak shift (simplified)
        measured_stress = peak_shift * 200e9  # Back-calculation
        
        return {
            'two_theta': two_theta,
            'intensity': intensity,
            'peak_positions': np.array(peak_positions) * (1 + peak_shift),
            'measured_stress': measured_stress,
            'peak_broadening': sigma
        }
    
    def generate_experimental_dataset(self, n_samples: int = 20) -> Tuple[pd.DataFrame, Dict]:
        """Generate complete experimental-like dataset"""
        
        print(f"Generating {n_samples} experimental samples...")
        
        # Generate varied operating conditions
        conditions_list = []
        results_list = []
        experimental_data = {
            'iv_curves': [],
            'eis_spectra': [],
            'thermography': [],
            'sem_images': [],
            'xrd_patterns': []
        }
        
        for i in tqdm(range(n_samples), desc="Experimental samples"):
            # Vary conditions realistically
            T = 1073 + np.random.normal(0, 20)  # Temperature variations
            p = 101325 * (1 + np.random.normal(0, 0.02))  # Pressure variations
            i_density = 5000 + np.random.normal(0, 500)  # Current variations
            
            # Degradation increases with sample number (aging study)
            degradation = i / n_samples
            time = degradation * 10000  # hours
            
            # Fuel composition variations (realistic)
            h2_fraction = 0.97 - 0.1 * degradation + np.random.normal(0, 0.01)
            h2_fraction = np.clip(h2_fraction, 0.7, 0.97)
            
            conditions = SOFCOperatingConditions(
                temperature=T,
                pressure=p,
                current_density=i_density,
                fuel_utilization=0.75 + np.random.normal(0, 0.05),
                air_utilization=0.2 + np.random.normal(0, 0.02),
                inlet_fuel_composition={'H2': h2_fraction, 'H2O': 1-h2_fraction},
                inlet_air_composition={'O2': 0.21, 'N2': 0.79},
                time=time
            )
            
            # Simulate measurements
            iv_data = self.simulate_iv_curve(conditions)
            eis_data = self.simulate_eis(conditions)
            thermo_data = self.simulate_thermography(conditions)
            sem_data = self.simulate_sem_images(conditions, degradation)
            xrd_data = self.simulate_xrd_stress(100e6 * (1 + degradation))
            
            # Store experimental data
            experimental_data['iv_curves'].append(iv_data)
            experimental_data['eis_spectra'].append(eis_data)
            experimental_data['thermography'].append(thermo_data)
            experimental_data['sem_images'].append(sem_data)
            experimental_data['xrd_patterns'].append(xrd_data)
            
            # Compile scalar results
            results_list.append({
                'sample_id': i,
                'temperature': T,
                'pressure': p,
                'current_density': i_density,
                'time_hours': time,
                'degradation_level': degradation,
                
                # IV curve metrics
                'ocv': iv_data['ocv'],
                'max_power': iv_data['max_power'],
                'voltage_at_0.5A': iv_data['voltage'][25],  # At 0.5 A/cm²
                
                # EIS metrics
                'R_total': eis_data['R_total'],
                'char_frequency': eis_data['characteristic_frequency'],
                
                # Thermography metrics
                'T_max': thermo_data['stats']['T_max'],
                'T_gradient': thermo_data['stats']['gradient_max'],
                
                # SEM metrics
                'porosity': sem_data['porosity'],
                'particle_size': sem_data['avg_particle_size'],
                'n_cracks': sem_data['n_cracks'],
                
                # XRD metrics
                'measured_stress': xrd_data['measured_stress'],
                'peak_broadening': xrd_data['peak_broadening']
            })
        
        df = pd.DataFrame(results_list)
        
        return df, experimental_data
    
    def save_experimental_dataset(self, df: pd.DataFrame, exp_data: Dict,
                                 filename: str = "experimental_dataset.h5"):
        """Save experimental dataset"""
        
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save scalar data
            scalar_group = f.create_group('scalar_data')
            for col in df.columns:
                scalar_group.create_dataset(col, data=df[col].values,
                                          compression='gzip')
            
            # Save experimental measurements
            exp_group = f.create_group('experimental_measurements')
            
            # IV curves
            iv_group = exp_group.create_group('iv_curves')
            for i, iv_data in enumerate(exp_data['iv_curves']):
                sample_group = iv_group.create_group(f'sample_{i:03d}')
                for key, value in iv_data.items():
                    if isinstance(value, np.ndarray):
                        sample_group.create_dataset(key, data=value)
                    else:
                        sample_group.attrs[key] = value
            
            # EIS spectra
            eis_group = exp_group.create_group('eis_spectra')
            for i, eis_data in enumerate(exp_data['eis_spectra']):
                sample_group = eis_group.create_group(f'sample_{i:03d}')
                for key, value in eis_data.items():
                    if isinstance(value, np.ndarray):
                        sample_group.create_dataset(key, data=value)
                    else:
                        sample_group.attrs[key] = value
            
            # Thermography
            thermo_group = exp_group.create_group('thermography')
            for i, thermo_data in enumerate(exp_data['thermography']):
                sample_group = thermo_group.create_group(f'sample_{i:03d}')
                sample_group.create_dataset('temperature_field', 
                                          data=thermo_data['temperature_field'],
                                          compression='gzip')
                sample_group.attrs['resolution'] = thermo_data['resolution']
                for key, value in thermo_data['stats'].items():
                    sample_group.attrs[key] = value
            
            # SEM images
            sem_group = exp_group.create_group('sem_images')
            for i, sem_data in enumerate(exp_data['sem_images']):
                sample_group = sem_group.create_group(f'sample_{i:03d}')
                sample_group.create_dataset('image', data=sem_data['image'],
                                          compression='gzip')
                for key, value in sem_data.items():
                    if key != 'image':
                        sample_group.attrs[key] = value
            
            # XRD patterns
            xrd_group = exp_group.create_group('xrd_patterns')
            for i, xrd_data in enumerate(exp_data['xrd_patterns']):
                sample_group = xrd_group.create_group(f'sample_{i:03d}')
                for key, value in xrd_data.items():
                    if isinstance(value, np.ndarray):
                        sample_group.create_dataset(key, data=value)
                    else:
                        sample_group.attrs[key] = value
            
            # Metadata
            f.attrs['n_samples'] = len(df)
            f.attrs['generation_date'] = datetime.now().isoformat()
            f.attrs['measurement_techniques'] = list(exp_data.keys())
        
        print(f"Experimental dataset saved to {filepath}")
        
        # Save summary as CSV
        csv_path = filepath.replace('.h5', '_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to {csv_path}")
        
        return filepath


def generate_experimental_dataset(config_path: str = 'config.yaml', n_samples: int = None):
    """Main function to generate experimental dataset"""
    
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if n_samples is None:
        n_samples = config['dataset']['experimental']['n_samples']
    
    # Create simulator
    simulator = ExperimentalDataSimulator(config)
    
    # Generate dataset
    df, exp_data = simulator.generate_experimental_dataset(n_samples)
    
    # Print statistics
    print("\nExperimental Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nMeasurement summary:")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    print("\nExperimental techniques included:")
    for technique in exp_data.keys():
        print(f"  - {technique}: {len(exp_data[technique])} measurements")
    
    # Save dataset
    filepath = simulator.save_experimental_dataset(df, exp_data)
    
    return df, exp_data, filepath


if __name__ == "__main__":
    # Generate experimental dataset
    df, exp_data, filepath = generate_experimental_dataset(n_samples=10)
    print(f"\nExperimental dataset generation complete! File saved at: {filepath}")