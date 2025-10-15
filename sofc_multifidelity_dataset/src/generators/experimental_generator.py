"""Experimental data generator for SOFC with realistic noise and artifacts."""

import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import time
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.physics_utils import SOFCPhysics, MaterialProperties, DegradationModels
from src.utils.data_utils import DatasetManager


class ExperimentalDataGenerator:
    """
    Generate synthetic experimental SOFC data with realistic noise,
    measurement artifacts, and uncertainty quantification.
    """
    
    def __init__(self, config: Dict):
        """Initialize generator with configuration."""
        self.config = config
        self.physics = SOFCPhysics(config)
        self.materials = MaterialProperties()
        self.degradation = DegradationModels()
        
        # Extract parameters
        self.op_ranges = config['operating_conditions']
        self.noise_params = config['dataset']['experimental']['measurement_noise']
        
        # Define experimental measurement points
        self.iv_points = 20  # Number of I-V curve points
        self.eis_frequencies = np.logspace(-1, 5, 50)  # 0.1 Hz to 100 kHz
        self.thermography_resolution = (64, 64)  # IR camera resolution
        
    def generate_sample(self, sample_id: int, seed: Optional[int] = None) -> Dict:
        """Generate single experimental sample with realistic measurements."""
        if seed is not None:
            np.random.seed(seed + sample_id)
        
        start_time = time.time()
        
        # Sample experimental conditions
        inputs = self._sample_experimental_inputs()
        
        # Generate experimental measurements
        outputs = self._generate_experimental_outputs(inputs)
        
        # Add metadata
        metadata = {
            'sample_id': f'EXP_{sample_id:06d}',
            'timestamp': time.time(),
            'computation_time': time.time() - start_time,
            'convergence_flag': True,
            'notes': 'Synthetic experimental data with realistic noise'
        }
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'metadata': metadata
        }
    
    def _sample_experimental_inputs(self) -> Dict:
        """Sample realistic experimental conditions."""
        inputs = {}
        
        # Controlled operating conditions
        # Temperature typically well-controlled
        T_nominal = np.random.choice([973, 1023, 1073, 1123])  # Discrete setpoints
        inputs['temperature'] = T_nominal + np.random.normal(0, 2)  # ±2K uncertainty
        
        # Current/voltage controlled by electronic load
        inputs['current_density'] = np.random.uniform(0, 8000)  # 0-0.8 A/cm²
        
        # Flow rates (controlled by MFCs)
        inputs['fuel_utilization'] = np.random.uniform(0.4, 0.8)
        inputs['air_utilization'] = np.random.uniform(0.15, 0.3)
        
        # Pressure (usually atmospheric)
        inputs['pressure'] = 101325 + np.random.normal(0, 1000)  # ±1 kPa variation
        
        # Gas composition (controlled by mixing)
        fuel_comp = np.zeros(6)
        # Typical test conditions
        test_type = np.random.choice(['h2_h2o', 'reformed', 'syngas'])
        
        if test_type == 'h2_h2o':
            fuel_comp[0] = 0.97
            fuel_comp[1] = 0.03
        elif test_type == 'reformed':
            fuel_comp[0] = 0.70
            fuel_comp[1] = 0.20
            fuel_comp[2] = 0.05
            fuel_comp[3] = 0.03
            fuel_comp[4] = 0.02
        else:  # syngas
            fuel_comp[0] = 0.40
            fuel_comp[1] = 0.10
            fuel_comp[2] = 0.30
            fuel_comp[3] = 0.15
            fuel_comp[5] = 0.05
        
        # Add measurement uncertainty to composition
        fuel_comp += np.random.normal(0, 0.005, 6)
        fuel_comp = np.clip(fuel_comp, 0, 1)
        fuel_comp /= fuel_comp.sum()  # Renormalize
        
        inputs['fuel_composition'] = fuel_comp
        inputs['air_composition'] = np.array([0.21, 0.79])
        
        # Cell dimensions (measured pre-test)
        inputs['anode_thickness'] = self.config['geometry']['anode_thickness'] * \
                                   (1 + np.random.normal(0, 0.02))
        inputs['electrolyte_thickness'] = self.config['geometry']['electrolyte_thickness'] * \
                                         (1 + np.random.normal(0, 0.01))
        inputs['cathode_thickness'] = self.config['geometry']['cathode_thickness'] * \
                                     (1 + np.random.normal(0, 0.02))
        
        # Material properties (from characterization)
        inputs['anode_porosity'] = np.random.beta(10, 10) * 0.2 + 0.2  # 20-40%
        inputs['cathode_porosity'] = np.random.beta(10, 10) * 0.2 + 0.2
        
        # Test duration and cycling
        inputs['operating_time'] = np.random.gamma(2, 500)  # Hours
        inputs['thermal_cycles'] = np.random.poisson(5)
        inputs['redox_cycles'] = np.random.choice([0, 0, 0, 1, 2])  # Mostly 0
        
        return inputs
    
    def _generate_experimental_outputs(self, inputs: Dict) -> Dict:
        """Generate experimental measurements with realistic characteristics."""
        outputs = {}
        
        # 1. I-V Curve measurement
        iv_data, iv_uncertainty = self._measure_iv_curve(inputs)
        outputs['iv_curve'] = iv_data
        outputs['iv_uncertainty'] = iv_uncertainty
        
        # 2. EIS measurement
        eis_data, eis_uncertainty = self._measure_eis(inputs)
        outputs['eis_data'] = eis_data
        outputs['eis_uncertainty'] = eis_uncertainty
        
        # 3. Temperature field from IR thermography
        temp_map, temp_uncertainty = self._measure_temperature_field(inputs)
        outputs['temperature_map'] = temp_map
        outputs['temperature_uncertainty'] = temp_uncertainty
        
        # 4. Microstructure parameters from SEM/FIB
        microstructure_params = self._measure_microstructure(inputs)
        outputs['microstructure_params'] = microstructure_params
        
        return outputs
    
    def _measure_iv_curve(self, inputs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate I-V curve measurement with noise."""
        T = inputs['temperature']
        p = inputs['pressure']
        
        # Current density sweep points
        i_sweep = np.linspace(0, 10000, self.iv_points)  # 0 to 1 A/cm²
        
        iv_data = np.zeros((self.iv_points, 2))
        iv_uncertainty = np.zeros((self.iv_points, 2))
        
        for idx, i in enumerate(i_sweep):
            # Calculate theoretical voltage
            pH2 = p * inputs['fuel_composition'][0]
            pH2O = p * inputs['fuel_composition'][1]
            pO2 = p * inputs['air_composition'][0]
            
            E_nernst = self.physics.nernst_voltage(T, pH2, pH2O, pO2)
            
            # Simplified overpotentials
            i0_a = 1000 * np.exp(-60000 / (8.314 * T))
            i0_c = 100 * np.exp(-120000 / (8.314 * T))
            
            if i > 0:
                eta_act = (8.314 * T) / (2 * 96485) * (np.log(i / i0_a) + np.log(i / i0_c))
                eta_ohm = i * 2e-5  # Area-specific resistance
                
                # Concentration overpotential
                i_lim = 15000  # Limiting current density
                if i < 0.95 * i_lim:
                    eta_conc = (8.314 * T) / (2 * 96485) * np.log(1 - i / i_lim)
                else:
                    eta_conc = 0.3
                
                V = E_nernst - eta_act - eta_ohm - abs(eta_conc)
            else:
                V = E_nernst
            
            # Add degradation effects
            if inputs['operating_time'] > 0:
                deg_rate = self.degradation.voltage_degradation_rate(
                    inputs['operating_time'], T, i,
                    {'A': 0.1, 'E_act': 50000, 'current_exp': 0.5, 'time_exp': 0.25}
                )
                V *= (1 - deg_rate / 100)
            
            # Add measurement noise
            # Current measurement (electronic load)
            i_noise = np.random.normal(0, 10)  # ±10 A/m² precision
            i_measured = i + i_noise
            
            # Voltage measurement (multimeter)
            v_noise = np.random.normal(0, 0.001)  # ±1 mV precision
            v_drift = 0.0001 * idx  # Systematic drift during measurement
            v_measured = max(0, V + v_noise + v_drift)
            
            # AC ripple from power electronics
            v_ripple = 0.002 * np.sin(2 * np.pi * 100 * idx / self.iv_points)
            v_measured += v_ripple
            
            iv_data[idx, 0] = i_measured
            iv_data[idx, 1] = v_measured
            
            # Uncertainties (Type A and B combined)
            iv_uncertainty[idx, 0] = np.sqrt(10**2 + (0.001 * i)**2)  # Current
            iv_uncertainty[idx, 1] = np.sqrt(0.001**2 + (0.001 * V)**2)  # Voltage
        
        return iv_data, iv_uncertainty
    
    def _measure_eis(self, inputs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate EIS measurement with realistic impedance response."""
        T = inputs['temperature']
        
        eis_data = np.zeros((len(self.eis_frequencies), 2))
        eis_uncertainty = np.zeros((len(self.eis_frequencies), 2))
        
        # Equivalent circuit parameters
        R_ohm = 0.1  # Ohm*cm²
        R_pol = 0.3  # Polarization resistance
        C_dl = 0.01  # Double layer capacitance
        
        # Warburg element for diffusion
        sigma_w = 0.5  # Warburg coefficient
        
        for idx, freq in enumerate(self.eis_frequencies):
            omega = 2 * np.pi * freq
            
            # Ohmic resistance (high frequency)
            Z_ohm = R_ohm
            
            # Charge transfer impedance (RQ element)
            # Using constant phase element (CPE)
            n = 0.85  # CPE exponent (0.8-1 for real electrodes)
            Q = C_dl
            Z_ct = R_pol / (1 + (1j * omega * R_pol * Q) ** n)
            
            # Warburg impedance (low frequency)
            Z_w = sigma_w * (1 - 1j) / np.sqrt(omega)
            
            # Total impedance
            Z_total = Z_ohm + Z_ct + Z_w / (1 + 0.1j * omega)  # Modified Warburg
            
            # Add temperature dependence
            Z_total *= np.exp(-(T - 1073) / 500)
            
            # Add degradation effects
            if inputs['operating_time'] > 0:
                deg_factor = 1 + 0.0001 * inputs['operating_time']
                Z_total *= deg_factor
            
            # Add measurement noise
            # Magnitude noise (relative)
            mag_noise = np.random.normal(0, 0.002 * abs(Z_total))
            # Phase noise (degrees)
            phase_noise = np.random.normal(0, 0.5) * np.pi / 180
            
            Z_measured = Z_total + mag_noise * np.exp(1j * phase_noise)
            
            # Cable/connection artifacts at high frequency
            if freq > 10000:
                L_cable = 1e-7  # Cable inductance
                Z_measured += 1j * omega * L_cable
            
            eis_data[idx, 0] = Z_measured.real
            eis_data[idx, 1] = -Z_measured.imag  # Nyquist convention
            
            # Uncertainties
            eis_uncertainty[idx, 0] = 0.002 * abs(Z_measured.real) + 0.0001
            eis_uncertainty[idx, 1] = 0.002 * abs(Z_measured.imag) + 0.0001
        
        return eis_data, eis_uncertainty
    
    def _measure_temperature_field(self, inputs: Dict) -> Tuple[np.ndarray, float]:
        """Simulate IR thermography measurement."""
        T_inlet = inputs['temperature']
        i_avg = inputs['current_density']
        
        # Generate base temperature field
        nx, ny = self.thermography_resolution
        T_field = np.ones((nx, ny)) * T_inlet
        
        # Add spatial temperature gradients
        # Flow direction gradient
        x_grad = np.linspace(0, 1, nx)
        for i in range(nx):
            T_field[i, :] += 30 * x_grad[i] * (i_avg / 5000)  # Temperature rise
        
        # Channel/rib pattern (periodic)
        for j in range(ny):
            if (j // 8) % 2 == 0:  # Channel regions
                T_field[:, j] -= 5  # Cooler under channels
            else:  # Rib regions
                T_field[:, j] += 3  # Hotter under ribs
        
        # Add hot spots (defects)
        n_hotspots = np.random.poisson(2)
        for _ in range(n_hotspots):
            x_hs = np.random.randint(nx // 4, 3 * nx // 4)
            y_hs = np.random.randint(ny // 4, 3 * ny // 4)
            
            # Gaussian hot spot
            for i in range(nx):
                for j in range(ny):
                    dist = np.sqrt((i - x_hs)**2 + (j - y_hs)**2)
                    T_field[i, j] += 20 * np.exp(-dist**2 / 50)
        
        # IR camera characteristics
        # Spatial resolution limits
        T_field = gaussian_filter(T_field, sigma=1.5)
        
        # Emissivity corrections (assumed uniform but has variations)
        emissivity_field = 0.85 + np.random.normal(0, 0.02, (nx, ny))
        T_apparent = T_field * emissivity_field
        
        # Sensor noise
        # NETD (Noise Equivalent Temperature Difference)
        netd = 0.05  # 50 mK for good IR camera
        T_noise = np.random.normal(0, netd, (nx, ny))
        T_measured = T_apparent + T_noise
        
        # Bad pixels (dead/stuck)
        n_bad_pixels = np.random.poisson(5)
        for _ in range(n_bad_pixels):
            i_bad = np.random.randint(nx)
            j_bad = np.random.randint(ny)
            T_measured[i_bad, j_bad] = np.random.choice([273, T_inlet])  # Stuck cold or at T_inlet
        
        # Reflections (from surroundings)
        reflection_pattern = 10 * np.sin(2 * np.pi * np.arange(nx)[:, None] / 20) * \
                           np.sin(2 * np.pi * np.arange(ny)[None, :] / 20)
        T_measured += reflection_pattern * 0.1
        
        # Calibration drift
        T_measured += np.random.normal(0, 2)  # ±2K calibration uncertainty
        
        # Overall measurement uncertainty
        uncertainty = np.sqrt(netd**2 + 2**2 + (0.02 * T_inlet)**2)
        
        return T_measured, uncertainty
    
    def _measure_microstructure(self, inputs: Dict) -> np.ndarray:
        """Simulate microstructure characterization from SEM/FIB."""
        params = np.zeros(10)
        
        # Base microstructure parameters
        t_hours = inputs['operating_time']
        T_avg = inputs['temperature']
        
        # 1. Ni particle size (μm) - increases with time/temperature
        initial_ni_size = 1.0
        if t_hours > 0:
            coarsening = 0.1 * np.sqrt(t_hours / 1000) * np.exp((T_avg - 1073) / 200)
            ni_size = initial_ni_size * (1 + coarsening)
        else:
            ni_size = initial_ni_size
        
        # Add measurement uncertainty (image analysis)
        params[0] = ni_size + np.random.normal(0, 0.05)
        
        # 2. YSZ particle size (μm) - relatively stable
        params[1] = 0.5 + np.random.normal(0, 0.02)
        
        # 3. Porosity (%) - from image analysis
        true_porosity = inputs['anode_porosity']
        # Stereological correction and uncertainty
        params[2] = true_porosity * 100 * (1 + np.random.normal(0, 0.05))
        
        # 4. TPB density (m/m³) - decreases with coarsening
        tpb_initial = 1e12
        if t_hours > 0:
            tpb_loss = 0.3 * (ni_size / initial_ni_size - 1)**2
            tpb_density = tpb_initial * (1 - tpb_loss)
        else:
            tpb_density = tpb_initial
        params[3] = tpb_density * (1 + np.random.normal(0, 0.1))
        
        # 5. Pore size (μm)
        pore_size = 2.0 * (1 + 0.1 * np.random.randn())
        params[4] = pore_size + np.random.normal(0, 0.1)
        
        # 6. Tortuosity - from simulation/correlation
        tortuosity = 3.0 / true_porosity
        params[5] = tortuosity + np.random.normal(0, 0.2)
        
        # 7. Ni connectivity (%)
        connectivity = 95 - 5 * (ni_size / initial_ni_size - 1)**2
        params[6] = connectivity + np.random.normal(0, 2)
        
        # 8. Surface roughness (nm)
        roughness = 500 * (1 + t_hours / 10000)
        params[7] = roughness + np.random.normal(0, 50)
        
        # 9. Crack density (#/mm²)
        if inputs['thermal_cycles'] > 0:
            crack_density = 0.1 * inputs['thermal_cycles']**0.5
        else:
            crack_density = 0
        params[8] = max(0, crack_density + np.random.normal(0, 0.02))
        
        # 10. Interface delamination (%)
        if inputs['thermal_cycles'] > 10 or inputs['redox_cycles'] > 2:
            delamination = 5 * np.sqrt(inputs['thermal_cycles'] / 100 + inputs['redox_cycles'] / 10)
        else:
            delamination = 0
        params[9] = max(0, min(100, delamination + np.random.normal(0, 1)))
        
        return params
    
    def add_systematic_errors(self, data: Dict) -> Dict:
        """Add systematic measurement errors and artifacts."""
        # Temperature controller offset
        if 'temperature' in data['inputs']:
            data['inputs']['temperature'] += np.random.normal(0, 1)  # Systematic offset
        
        # Flow meter calibration error
        if 'fuel_utilization' in data['inputs']:
            cal_error = 1 + np.random.normal(0, 0.02)  # 2% calibration error
            data['inputs']['fuel_utilization'] *= cal_error
        
        # Pressure gauge drift
        if 'pressure' in data['inputs']:
            drift = 100 * np.random.randn()  # Pa
            data['inputs']['pressure'] += drift
        
        return data
    
    def add_environmental_effects(self, data: Dict, time_of_day: float = None) -> Dict:
        """Add environmental influences on measurements."""
        if time_of_day is None:
            time_of_day = np.random.uniform(0, 24)
        
        # Room temperature variation (affects IR measurements)
        room_temp_variation = 5 * np.sin(2 * np.pi * time_of_day / 24)
        
        if 'temperature_map' in data['outputs']:
            # Edge cooling from room temperature
            nx, ny = data['outputs']['temperature_map'].shape
            for i in [0, 1, nx-2, nx-1]:
                data['outputs']['temperature_map'][i, :] -= room_temp_variation * 0.1
            for j in [0, 1, ny-2, ny-1]:
                data['outputs']['temperature_map'][:, j] -= room_temp_variation * 0.1
        
        # Electrical grid noise (50/60 Hz)
        if 'eis_data' in data['outputs']:
            # Add harmonic at power line frequency
            grid_freq = np.random.choice([50, 60])  # Hz
            if grid_freq in self.eis_frequencies:
                idx = np.argmin(np.abs(self.eis_frequencies - grid_freq))
                data['outputs']['eis_data'][idx, 1] *= 1.5  # Artifact in imaginary part
        
        # Vibration effects (affects high-frequency EIS)
        if 'eis_data' in data['outputs']:
            high_freq_mask = self.eis_frequencies > 1000
            vibration_noise = np.random.normal(0, 0.001, sum(high_freq_mask))
            data['outputs']['eis_data'][high_freq_mask, 0] += vibration_noise
        
        return data
    
    def simulate_sensor_drift(self, data: Dict, test_duration: float) -> Dict:
        """Simulate sensor drift over long tests."""
        # Thermocouple drift
        tc_drift_rate = 0.1  # K/hour
        temp_drift = tc_drift_rate * test_duration
        
        if 'temperature' in data['inputs']:
            data['inputs']['temperature'] += temp_drift
        
        # Pressure sensor drift
        p_drift_rate = 10  # Pa/hour
        p_drift = p_drift_rate * test_duration
        
        if 'pressure' in data['inputs']:
            data['inputs']['pressure'] += p_drift
        
        # Electronic load drift (affects I-V curves)
        if 'iv_curve' in data['outputs']:
            i_drift = 0.01 * test_duration  # % per hour
            data['outputs']['iv_curve'][:, 0] *= (1 + i_drift / 100)
        
        return data
    
    def generate_dataset(self, n_samples: int, output_path: str = None) -> None:
        """Generate complete experimental dataset."""
        print(f"\nGenerating {n_samples} experimental samples...")
        
        # Initialize data manager
        if output_path is None:
            output_path = "./data"
        
        manager = DatasetManager(output_path)
        
        # Create HDF5 file structure
        # Spatial dims for experimental data (IR thermography)
        spatial_dims = self.thermography_resolution
        f = manager.create_hdf5_structure('experimental', n_samples, 
                                         spatial_dims, self.config)
        
        try:
            # Generate samples
            for idx in tqdm(range(n_samples), desc="Experimental Generation"):
                # Base sample
                sample = self.generate_sample(idx)
                
                # Add systematic errors
                sample = self.add_systematic_errors(sample)
                
                # Add environmental effects
                time_of_day = 8 + idx * 0.5  # Simulating tests throughout the day
                sample = self.add_environmental_effects(sample, time_of_day)
                
                # Add sensor drift for long tests
                test_duration = sample['inputs']['operating_time']
                if test_duration > 10:
                    sample = self.simulate_sensor_drift(sample, test_duration)
                
                # Write to file
                manager.write_sample(f, idx, sample)
                
                # Flush periodically
                if idx % 5 == 0:
                    f.flush()
            
            print(f"Successfully generated {n_samples} experimental samples")
            
        finally:
            f.close()
        
        # Validate dataset
        validation_report = manager.validate_dataset('experimental')
        print("\nDataset validation report:")
        print(f"  - File size: {validation_report['file_size_mb']:.2f} MB")
        print(f"  - Samples: {validation_report['n_samples']}")
        
        return validation_report


def main():
    """Main function for standalone execution."""
    import yaml
    
    # Load configuration
    config_path = "../../config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = ExperimentalDataGenerator(config)
    
    # Generate test dataset
    generator.generate_dataset(5)


if __name__ == "__main__":
    main()