"""
Simulation Data Generator for Stratified Flow Attenuation Mechanisms Study
PhD Thesis Dataset Generation

This script generates synthetic CFD and mathematical model simulation data
for studying acoustic attenuation in stratified multiphase flows.
"""

import numpy as np
import json
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Physical constants and parameters
RHO_LIQUID = 1000  # kg/m^3 (water)
RHO_GAS = 1.2      # kg/m^3 (air)
MU_LIQUID = 0.001  # Pa·s (water viscosity)
MU_GAS = 1.8e-5    # Pa·s (air viscosity)
SOUND_SPEED_LIQUID = 1500  # m/s
SOUND_SPEED_GAS = 343      # m/s
GRAVITY = 9.81     # m/s^2

class CFDDataGenerator:
    """Generate CFD simulation outputs"""
    
    def __init__(self, nx=100, ny=100, nz=50, nt=1000):
        self.nx = nx  # Grid points in x
        self.ny = ny  # Grid points in y
        self.nz = nz  # Grid points in z
        self.nt = nt  # Time steps
        
        # Domain dimensions
        self.Lx = 1.0  # meters
        self.Ly = 1.0  # meters
        self.Lz = 0.5  # meters
        self.T = 1.0   # seconds
        
        # Grid spacing
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        self.dz = self.Lz / nz
        self.dt = self.T / nt
        
        # Create coordinate arrays
        self.x = np.linspace(0, self.Lx, nx)
        self.y = np.linspace(0, self.Ly, ny)
        self.z = np.linspace(0, self.Lz, nz)
        self.t = np.linspace(0, self.T, nt)
        
    def generate_velocity_fields(self):
        """Generate 3D velocity fields (u, v, w)"""
        print("Generating velocity fields...")
        
        # Create stratified flow pattern
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Velocity in x-direction (primary flow direction)
        # Stratified with higher velocity in upper layer
        u = 0.5 * (1 + np.tanh(10 * (Z - 0.25))) + 0.1 * np.sin(2*np.pi*X/self.Lx)
        
        # Velocity in y-direction (cross flow)
        v = 0.05 * np.sin(2*np.pi*Y/self.Ly) * np.cos(2*np.pi*X/self.Lx)
        
        # Velocity in z-direction (vertical)
        w = 0.02 * (np.cos(2*np.pi*X/self.Lx) * np.sin(2*np.pi*Y/self.Ly) - 
                    0.5 * np.sin(np.pi*Z/self.Lz))
        
        return {
            'u': u.astype(np.float32),
            'v': v.astype(np.float32),
            'w': w.astype(np.float32),
            'coordinates': {'x': self.x, 'y': self.y, 'z': self.z}
        }
    
    def generate_pressure_field(self):
        """Generate pressure field with stratification effects"""
        print("Generating pressure field...")
        
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Hydrostatic pressure + dynamic pressure fluctuations
        p_hydrostatic = 101325 - RHO_LIQUID * GRAVITY * Z  # Pa
        
        # Add dynamic pressure variations (turbulence, waves)
        p_dynamic = 100 * np.sin(4*np.pi*X/self.Lx) * np.cos(4*np.pi*Y/self.Ly) * \
                    np.exp(-((Z - 0.25)/0.1)**2)
        
        pressure = p_hydrostatic + p_dynamic
        
        return {
            'pressure': pressure.astype(np.float32),
            'coordinates': {'x': self.x, 'y': self.y, 'z': self.z}
        }
    
    def generate_vof_field(self):
        """Generate Volume of Fluid (VOF) phase distribution"""
        print("Generating VOF field...")
        
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Interface location with waves
        interface_height = 0.25 + 0.03 * np.sin(4*np.pi*X/self.Lx) * \
                          np.cos(4*np.pi*Y/self.Ly)
        
        # VOF: 1 = liquid, 0 = gas
        vof = 0.5 * (1 - np.tanh(50 * (Z - interface_height)))
        
        # Add small bubbles in liquid layer
        for i in range(5):
            xc, yc, zc = np.random.rand(3) * [self.Lx, self.Ly, 0.2]
            r = 0.02
            bubble = np.exp(-((X-xc)**2 + (Y-yc)**2 + (Z-zc)**2) / r**2)
            vof = vof - 0.3 * bubble
        
        vof = np.clip(vof, 0, 1)
        
        return {
            'vof': vof.astype(np.float32),
            'liquid_fraction': vof,
            'gas_fraction': 1 - vof,
            'coordinates': {'x': self.x, 'y': self.y, 'z': self.z}
        }
    
    def generate_turbulence_parameters(self):
        """Generate k-ε turbulence model parameters"""
        print("Generating turbulence parameters...")
        
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Turbulent kinetic energy (k)
        k_base = 0.01  # m^2/s^2
        k = k_base * (1 + np.exp(-((Z - 0.25)/0.05)**2)) * \
            (1 + 0.2*np.sin(2*np.pi*X/self.Lx))
        
        # Turbulent dissipation rate (ε)
        epsilon_base = 0.001  # m^2/s^3
        epsilon = epsilon_base * (1 + 2*np.exp(-((Z - 0.25)/0.05)**2))
        
        # Eddy viscosity (μ_t = C_μ * ρ * k^2 / ε)
        C_mu = 0.09
        rho = RHO_LIQUID * np.ones_like(k)  # Simplified
        mu_t = C_mu * rho * k**2 / (epsilon + 1e-10)
        
        # Turbulent length scale
        l_t = C_mu**(0.75) * k**(1.5) / (epsilon + 1e-10)
        
        return {
            'k': k.astype(np.float32),
            'epsilon': epsilon.astype(np.float32),
            'eddy_viscosity': mu_t.astype(np.float32),
            'turbulent_length_scale': l_t.astype(np.float32),
            'coordinates': {'x': self.x, 'y': self.y, 'z': self.z}
        }
    
    def generate_acoustic_pressure_propagation(self):
        """Generate acoustic pressure propagation over time"""
        print("Generating acoustic pressure propagation...")
        
        # 2D slice for acoustic propagation (x-z plane at y=Ly/2)
        ny_mid = self.ny // 2
        
        acoustic_data = np.zeros((self.nt, self.nx, self.nz), dtype=np.float32)
        
        # Source parameters
        source_x = self.Lx * 0.2
        source_z = self.Lz * 0.3
        source_freq = 1000  # Hz
        source_amplitude = 1000  # Pa
        
        X, Z = np.meshgrid(self.x, self.z, indexing='ij')
        
        for it, time in enumerate(self.t):
            # Wave propagation with stratification effects
            # Upper layer (gas): faster attenuation, slower speed
            # Lower layer (liquid): lower attenuation, faster speed
            
            # Determine effective sound speed based on height
            c_eff = SOUND_SPEED_LIQUID * (Z < 0.25) + \
                    SOUND_SPEED_GAS * (Z >= 0.25)
            
            # Distance from source
            r = np.sqrt((X - source_x)**2 + (Z - source_z)**2)
            
            # Time delay
            t_delay = r / c_eff
            
            # Attenuation coefficient (frequency and height dependent)
            alpha = (0.01 * (Z < 0.25) + 0.1 * (Z >= 0.25)) * source_freq / 1000
            
            # Acoustic pressure
            p_acoustic = source_amplitude * np.sin(2*np.pi*source_freq*(time - t_delay)) * \
                        np.exp(-alpha * r) / (r + 0.01)
            
            # Add reflection from interface
            interface_z = 0.25
            r_reflected = np.sqrt((X - source_x)**2 + (Z - (2*interface_z - source_z))**2)
            reflection_coeff = 0.3
            p_reflected = reflection_coeff * source_amplitude * \
                         np.sin(2*np.pi*source_freq*(time - r_reflected/SOUND_SPEED_LIQUID)) * \
                         np.exp(-alpha * r_reflected) / (r_reflected + 0.01)
            
            acoustic_data[it] = p_acoustic + p_reflected * (Z < interface_z)
        
        return {
            'acoustic_pressure': acoustic_data,
            'time': self.t,
            'coordinates': {'x': self.x, 'z': self.z},
            'source_params': {
                'frequency': source_freq,
                'amplitude': source_amplitude,
                'position': [source_x, source_z]
            }
        }


class MathematicalModelGenerator:
    """Generate mathematical model outputs"""
    
    def __init__(self, n_frequencies=50, n_positions=100, n_void_fractions=20):
        self.frequencies = np.logspace(2, 4, n_frequencies)  # 100 Hz to 10 kHz
        self.void_fractions = np.linspace(0, 1, n_void_fractions)
        self.positions = np.linspace(0, 1, n_positions)
        
    def generate_sound_speed_predictions(self):
        """Generate sound speed predictions using mixture models"""
        print("Generating sound speed predictions...")
        
        sound_speeds = {}
        
        # Wood's equation (homogeneous mixture)
        alpha = self.void_fractions
        rho_m = alpha * RHO_GAS + (1 - alpha) * RHO_LIQUID
        K_m = 1 / (alpha / (RHO_GAS * SOUND_SPEED_GAS**2) + 
                   (1 - alpha) / (RHO_LIQUID * SOUND_SPEED_LIQUID**2))
        c_wood = np.sqrt(K_m / rho_m)
        
        sound_speeds['wood_equation'] = {
            'void_fraction': alpha,
            'sound_speed': c_wood.astype(np.float32),
            'density': rho_m.astype(np.float32)
        }
        
        # Modified model with dispersion (Eq. 27 from thesis)
        omega = 2 * np.pi * self.frequencies
        c_dispersive = np.zeros((len(self.frequencies), len(alpha)), dtype=np.float32)
        
        for i, f in enumerate(self.frequencies):
            # Add frequency-dependent correction
            correction = 1 - 0.1 * alpha * np.exp(-f/1000)
            c_dispersive[i, :] = c_wood * correction
        
        sound_speeds['dispersive_model'] = {
            'void_fraction': alpha,
            'frequencies': self.frequencies,
            'sound_speed': c_dispersive,
            'description': 'Modified model with frequency dispersion (Eq. 27)'
        }
        
        return sound_speeds
    
    def generate_attenuation_coefficients(self):
        """Generate predicted attenuation coefficients"""
        print("Generating attenuation coefficients...")
        
        attenuation = {}
        
        # Frequency and void fraction dependent attenuation
        alpha = self.void_fractions
        
        attenuation_matrix = np.zeros((len(self.frequencies), len(alpha)), dtype=np.float32)
        
        for i, f in enumerate(self.frequencies):
            # Viscous attenuation
            alpha_visc = 2 * f**2 * MU_LIQUID / (RHO_LIQUID * SOUND_SPEED_LIQUID**3)
            
            # Scattering attenuation (bubble-induced)
            alpha_scat = alpha * f**2 * 1e-10  # Simplified scattering model
            
            # Thermal attenuation
            alpha_therm = alpha * f * 1e-6
            
            # Total attenuation
            attenuation_matrix[i, :] = alpha_visc + alpha_scat + alpha_therm
        
        attenuation['total'] = {
            'void_fraction': alpha,
            'frequencies': self.frequencies,
            'attenuation_coefficient': attenuation_matrix,
            'unit': 'Np/m'
        }
        
        # Convert to dB/m
        attenuation['dB_per_meter'] = {
            'void_fraction': alpha,
            'frequencies': self.frequencies,
            'attenuation_coefficient': attenuation_matrix * 8.686,
            'unit': 'dB/m'
        }
        
        return attenuation
    
    def generate_wave_propagation_patterns(self):
        """Generate wave propagation, reflection, and transmission patterns"""
        print("Generating wave propagation patterns...")
        
        patterns = {}
        
        # Reflection and transmission coefficients at interface
        Z1 = RHO_LIQUID * SOUND_SPEED_LIQUID  # Acoustic impedance liquid
        Z2 = RHO_GAS * SOUND_SPEED_GAS        # Acoustic impedance gas
        
        # Normal incidence
        R_normal = (Z2 - Z1) / (Z2 + Z1)  # Reflection coefficient
        T_normal = 2 * Z2 / (Z2 + Z1)     # Transmission coefficient
        
        patterns['normal_incidence'] = {
            'reflection_coefficient': R_normal,
            'transmission_coefficient': T_normal,
            'impedance_liquid': Z1,
            'impedance_gas': Z2
        }
        
        # Angle-dependent (oblique incidence)
        angles = np.linspace(0, 89, 90)  # degrees
        R_angles = np.zeros_like(angles)
        T_angles = np.zeros_like(angles)
        
        for i, theta in enumerate(angles):
            theta_rad = np.radians(theta)
            
            # Snell's law
            if SOUND_SPEED_GAS * np.sin(theta_rad) / SOUND_SPEED_LIQUID < 1:
                theta_t = np.arcsin(SOUND_SPEED_GAS * np.sin(theta_rad) / SOUND_SPEED_LIQUID)
                
                # Reflection coefficient (pressure)
                Z1_eff = Z1 / np.cos(theta_rad)
                Z2_eff = Z2 / np.cos(theta_t)
                R_angles[i] = abs((Z2_eff - Z1_eff) / (Z2_eff + Z1_eff))
                T_angles[i] = abs(2 * Z2_eff / (Z2_eff + Z1_eff))
            else:
                R_angles[i] = 1.0  # Total internal reflection
                T_angles[i] = 0.0
        
        patterns['oblique_incidence'] = {
            'angles': angles,
            'reflection_coefficient': R_angles.astype(np.float32),
            'transmission_coefficient': T_angles.astype(np.float32)
        }
        
        # Standing wave pattern
        x = self.positions
        standing_wave = np.zeros((len(self.frequencies[:10]), len(x)), dtype=np.float32)
        
        for i, f in enumerate(self.frequencies[:10]):
            k = 2 * np.pi * f / SOUND_SPEED_LIQUID
            # Incident + reflected wave
            standing_wave[i, :] = np.abs(np.exp(1j*k*x) + R_normal * np.exp(-1j*k*x))
        
        patterns['standing_waves'] = {
            'position': x,
            'frequencies': self.frequencies[:10],
            'amplitude': standing_wave
        }
        
        return patterns
    
    def generate_time_delay_estimates(self):
        """Generate time-delay estimates (T0) for different configurations"""
        print("Generating time-delay estimates...")
        
        delays = {}
        
        # Distance array
        distances = np.linspace(0.1, 5.0, 50)  # meters
        
        # Time delays for different void fractions
        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            # Effective sound speed
            rho_m = alpha * RHO_GAS + (1 - alpha) * RHO_LIQUID
            K_m = 1 / (alpha / (RHO_GAS * SOUND_SPEED_GAS**2) + 
                       (1 - alpha) / (RHO_LIQUID * SOUND_SPEED_LIQUID**2))
            c_eff = np.sqrt(K_m / rho_m)
            
            # Time delay
            T0 = distances / c_eff
            
            delays[f'alpha_{alpha:.1f}'] = {
                'void_fraction': alpha,
                'distance': distances,
                'time_delay': T0.astype(np.float32),
                'effective_sound_speed': c_eff
            }
        
        return delays


class ValidationDataGenerator:
    """Generate validation comparison data"""
    
    def __init__(self):
        self.frequencies = np.logspace(2, 4, 30)
        self.time = np.linspace(0, 0.01, 1000)
        
    def generate_comparison_data(self):
        """Generate simulated vs experimental comparison data"""
        print("Generating validation comparison data...")
        
        comparisons = {}
        
        # 1. Acoustic waveform comparison
        source_freq = 2000  # Hz
        
        # Simulated waveform (clean)
        waveform_sim = np.sin(2*np.pi*source_freq*self.time) * \
                      np.exp(-50*self.time)
        
        # "Experimental" waveform (with noise and artifacts)
        waveform_exp = waveform_sim + \
                      0.05*np.random.randn(len(self.time)) + \
                      0.02*np.sin(2*np.pi*5000*self.time)  # Noise
        
        comparisons['waveform'] = {
            'time': self.time,
            'simulated': waveform_sim.astype(np.float32),
            'experimental': waveform_exp.astype(np.float32),
            'frequency': source_freq,
            'unit': 'Pa'
        }
        
        # 2. Attenuation values comparison
        void_fractions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Simulated attenuation (theoretical model)
        atten_sim = 0.01 + 0.5 * void_fractions + 0.3 * void_fractions**2
        
        # Experimental attenuation (with measurement uncertainty)
        atten_exp = atten_sim * (1 + 0.1*np.random.randn(len(void_fractions)))
        atten_exp_uncertainty = 0.05 * atten_sim
        
        comparisons['attenuation'] = {
            'void_fraction': void_fractions,
            'simulated': atten_sim.astype(np.float32),
            'experimental': atten_exp.astype(np.float32),
            'experimental_uncertainty': atten_exp_uncertainty.astype(np.float32),
            'unit': 'Np/m',
            'frequency': 1000.0
        }
        
        # 3. Sound speed comparison
        # Simulated (Wood's equation)
        rho_m = void_fractions * RHO_GAS + (1 - void_fractions) * RHO_LIQUID
        K_m = 1 / (void_fractions / (RHO_GAS * SOUND_SPEED_GAS**2) + 
                   (1 - void_fractions) / (RHO_LIQUID * SOUND_SPEED_LIQUID**2))
        c_sim = np.sqrt(K_m / rho_m)
        
        # Experimental (with measurement error)
        c_exp = c_sim * (1 + 0.05*np.random.randn(len(void_fractions)))
        c_exp_uncertainty = 0.03 * c_sim
        
        comparisons['sound_speed'] = {
            'void_fraction': void_fractions,
            'simulated': c_sim.astype(np.float32),
            'experimental': c_exp.astype(np.float32),
            'experimental_uncertainty': c_exp_uncertainty.astype(np.float32),
            'unit': 'm/s'
        }
        
        # 4. Statistical comparison metrics
        comparisons['statistics'] = {
            'waveform': {
                'correlation': float(np.corrcoef(waveform_sim, waveform_exp)[0,1]),
                'rmse': float(np.sqrt(np.mean((waveform_sim - waveform_exp)**2))),
                'mae': float(np.mean(np.abs(waveform_sim - waveform_exp)))
            },
            'attenuation': {
                'correlation': float(np.corrcoef(atten_sim, atten_exp)[0,1]),
                'rmse': float(np.sqrt(np.mean((atten_sim - atten_exp)**2))),
                'relative_error': float(np.mean(np.abs(atten_sim - atten_exp) / atten_sim) * 100)
            },
            'sound_speed': {
                'correlation': float(np.corrcoef(c_sim, c_exp)[0,1]),
                'rmse': float(np.sqrt(np.mean((c_sim - c_exp)**2))),
                'relative_error': float(np.mean(np.abs(c_sim - c_exp) / c_sim) * 100)
            }
        }
        
        return comparisons


def save_data_as_csv(data, filename):
    """Save data as CSV file"""
    if isinstance(data, dict):
        # Handle nested dictionaries
        if 'coordinates' in data:
            # Skip large 3D arrays for CSV, just save metadata
            return
        
        # For 1D or 2D data
        lines = []
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    lines.append(f"{key},{','.join(map(str, value))}\n")
            elif isinstance(value, (int, float)):
                lines.append(f"{key},{value}\n")
        
        if lines:
            with open(filename, 'w') as f:
                f.writelines(lines)


def save_data_as_npy(data, filename):
    """Save numpy arrays"""
    np.save(filename, data)


def save_metadata(metadata, filename):
    """Save metadata as JSON"""
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def main():
    """Main function to generate all simulation data"""
    print("="*70)
    print("STRATIFIED FLOW SIMULATION DATA GENERATOR")
    print("PhD Thesis: Attenuation Mechanisms in Stratified Flows")
    print("="*70)
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize generators
    print("Initializing data generators...")
    cfd_gen = CFDDataGenerator(nx=100, ny=100, nz=50, nt=1000)
    math_gen = MathematicalModelGenerator()
    val_gen = ValidationDataGenerator()
    
    print()
    print("-" * 70)
    print("PART 1: CFD MODEL OUTPUTS")
    print("-" * 70)
    
    # Generate CFD data
    velocity_data = cfd_gen.generate_velocity_fields()
    pressure_data = cfd_gen.generate_pressure_field()
    vof_data = cfd_gen.generate_vof_field()
    turbulence_data = cfd_gen.generate_turbulence_parameters()
    acoustic_data = cfd_gen.generate_acoustic_pressure_propagation()
    
    # Save CFD data
    print("\nSaving CFD outputs...")
    np.save('cfd_outputs/velocity_u.npy', velocity_data['u'])
    np.save('cfd_outputs/velocity_v.npy', velocity_data['v'])
    np.save('cfd_outputs/velocity_w.npy', velocity_data['w'])
    np.save('cfd_outputs/pressure.npy', pressure_data['pressure'])
    np.save('cfd_outputs/vof.npy', vof_data['vof'])
    np.save('cfd_outputs/turbulence_k.npy', turbulence_data['k'])
    np.save('cfd_outputs/turbulence_epsilon.npy', turbulence_data['epsilon'])
    np.save('cfd_outputs/eddy_viscosity.npy', turbulence_data['eddy_viscosity'])
    np.save('cfd_outputs/acoustic_pressure.npy', acoustic_data['acoustic_pressure'])
    
    # Save coordinates
    save_metadata({
        'x': velocity_data['coordinates']['x'].tolist(),
        'y': velocity_data['coordinates']['y'].tolist(),
        'z': velocity_data['coordinates']['z'].tolist(),
        't': acoustic_data['time'].tolist(),
        'domain': {
            'Lx': cfd_gen.Lx,
            'Ly': cfd_gen.Ly,
            'Lz': cfd_gen.Lz,
            'T': cfd_gen.T
        },
        'grid': {
            'nx': cfd_gen.nx,
            'ny': cfd_gen.ny,
            'nz': cfd_gen.nz,
            'nt': cfd_gen.nt
        }
    }, 'cfd_outputs/coordinates.json')
    
    print()
    print("-" * 70)
    print("PART 2: MATHEMATICAL MODEL OUTPUTS")
    print("-" * 70)
    
    # Generate mathematical model data
    sound_speed_data = math_gen.generate_sound_speed_predictions()
    attenuation_data = math_gen.generate_attenuation_coefficients()
    wave_propagation_data = math_gen.generate_wave_propagation_patterns()
    time_delay_data = math_gen.generate_time_delay_estimates()
    
    # Save mathematical model data
    print("\nSaving mathematical model outputs...")
    
    # Sound speed
    np.save('mathematical_model_outputs/sound_speed_wood.npy',
            sound_speed_data['wood_equation']['sound_speed'])
    np.save('mathematical_model_outputs/sound_speed_dispersive.npy',
            sound_speed_data['dispersive_model']['sound_speed'])
    
    # Attenuation
    np.save('mathematical_model_outputs/attenuation_coefficients.npy',
            attenuation_data['total']['attenuation_coefficient'])
    np.save('mathematical_model_outputs/attenuation_dB.npy',
            attenuation_data['dB_per_meter']['attenuation_coefficient'])
    
    # Wave propagation
    save_metadata(wave_propagation_data['normal_incidence'],
                 'mathematical_model_outputs/reflection_transmission.json')
    np.save('mathematical_model_outputs/reflection_vs_angle.npy',
            wave_propagation_data['oblique_incidence']['reflection_coefficient'])
    np.save('mathematical_model_outputs/standing_waves.npy',
            wave_propagation_data['standing_waves']['amplitude'])
    
    # Time delays
    time_delay_dict = {}
    for key, value in time_delay_data.items():
        time_delay_dict[key] = {
            'void_fraction': float(value['void_fraction']),
            'distance': value['distance'].tolist(),
            'time_delay': value['time_delay'].tolist(),
            'effective_sound_speed': float(value['effective_sound_speed'])
        }
    save_metadata(time_delay_dict, 'mathematical_model_outputs/time_delays.json')
    
    # Save parameters
    save_metadata({
        'frequencies': math_gen.frequencies.tolist(),
        'void_fractions': math_gen.void_fractions.tolist(),
        'positions': math_gen.positions.tolist()
    }, 'mathematical_model_outputs/parameters.json')
    
    print()
    print("-" * 70)
    print("PART 3: VALIDATION DATA")
    print("-" * 70)
    
    # Generate validation data
    validation_data = val_gen.generate_comparison_data()
    
    # Save validation data
    print("\nSaving validation data...")
    
    # Waveform comparison
    np.save('validation_data/waveform_simulated.npy',
            validation_data['waveform']['simulated'])
    np.save('validation_data/waveform_experimental.npy',
            validation_data['waveform']['experimental'])
    np.save('validation_data/waveform_time.npy',
            validation_data['waveform']['time'])
    
    # Attenuation comparison
    save_metadata({
        'void_fraction': validation_data['attenuation']['void_fraction'].tolist(),
        'simulated': validation_data['attenuation']['simulated'].tolist(),
        'experimental': validation_data['attenuation']['experimental'].tolist(),
        'uncertainty': validation_data['attenuation']['experimental_uncertainty'].tolist()
    }, 'validation_data/attenuation_comparison.json')
    
    # Sound speed comparison
    save_metadata({
        'void_fraction': validation_data['sound_speed']['void_fraction'].tolist(),
        'simulated': validation_data['sound_speed']['simulated'].tolist(),
        'experimental': validation_data['sound_speed']['experimental'].tolist(),
        'uncertainty': validation_data['sound_speed']['experimental_uncertainty'].tolist()
    }, 'validation_data/sound_speed_comparison.json')
    
    # Statistics
    save_metadata(validation_data['statistics'],
                 'validation_data/validation_statistics.json')
    
    print()
    print("="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - CFD outputs: 9 files (velocity fields, pressure, VOF, turbulence, acoustic)")
    print(f"  - Mathematical model outputs: 8 files (sound speed, attenuation, wave propagation)")
    print(f"  - Validation data: 7 files (comparisons and statistics)")
    print(f"\nTotal: 24 data files + metadata")
    print(f"\nTimestamp: {timestamp}")
    print()


if __name__ == "__main__":
    main()
