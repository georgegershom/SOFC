"""
Generate synthetic synchrotron X-ray diffraction data for SOFC creep analysis.
This simulates XRD patterns, phase identification, and residual stress/strain mapping.
"""

import numpy as np
import h5py
from scipy import signal, interpolate
from scipy.special import voigt_profile
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class SynchrotronDiffractionGenerator:
    """Generate synthetic synchrotron X-ray diffraction data."""
    
    def __init__(self, beam_energy=80, wavelength=None):
        """
        Initialize the diffraction generator.
        
        Parameters:
        -----------
        beam_energy : float
            X-ray beam energy in keV
        wavelength : float
            X-ray wavelength in Angstroms (calculated from energy if not provided)
        """
        self.beam_energy = beam_energy  # keV
        
        if wavelength is None:
            # Calculate wavelength from energy: λ = hc/E
            h_c = 12.398  # keV·Å
            self.wavelength = h_c / beam_energy  # Angstroms
        else:
            self.wavelength = wavelength
        
        # Define material phases and their properties
        self.phases = {
            'ferrite': {
                'structure': 'bcc',
                'lattice_param': 2.866,  # Angstroms
                'peaks': self._calculate_bcc_peaks(2.866),
                'intensity_factor': 1.0
            },
            'austenite': {
                'structure': 'fcc', 
                'lattice_param': 3.589,  # Angstroms
                'peaks': self._calculate_fcc_peaks(3.589),
                'intensity_factor': 0.8
            },
            'chromium_oxide': {
                'structure': 'hexagonal',
                'lattice_param': {'a': 4.958, 'c': 13.594},  # Cr2O3
                'peaks': self._calculate_hex_peaks(4.958, 13.594),
                'intensity_factor': 0.6
            },
            'carbide': {
                'structure': 'orthorhombic',
                'lattice_param': {'a': 4.52, 'b': 5.08, 'c': 6.73},  # Cr23C6
                'peaks': self._calculate_ortho_peaks(4.52, 5.08, 6.73),
                'intensity_factor': 0.7
            }
        }
    
    def _calculate_bcc_peaks(self, a):
        """Calculate BCC diffraction peaks."""
        # BCC allowed reflections: h+k+l = even
        peaks = []
        for h in range(5):
            for k in range(5):
                for l in range(5):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    if (h + k + l) % 2 == 0:
                        d_spacing = a / np.sqrt(h**2 + k**2 + l**2)
                        two_theta = 2 * np.arcsin(self.wavelength / (2 * d_spacing))
                        two_theta_deg = np.degrees(two_theta)
                        
                        if 10 < two_theta_deg < 120:  # Reasonable 2θ range
                            # Calculate structure factor (simplified)
                            intensity = self._calculate_intensity(h, k, l, 'bcc')
                            peaks.append({
                                'hkl': (h, k, l),
                                'd_spacing': d_spacing,
                                '2theta': two_theta_deg,
                                'intensity': intensity
                            })
        
        return sorted(peaks, key=lambda x: x['2theta'])
    
    def _calculate_fcc_peaks(self, a):
        """Calculate FCC diffraction peaks."""
        # FCC allowed reflections: all even or all odd
        peaks = []
        for h in range(5):
            for k in range(5):
                for l in range(5):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    # Check if all even or all odd
                    if (h % 2 == k % 2 == l % 2):
                        d_spacing = a / np.sqrt(h**2 + k**2 + l**2)
                        two_theta = 2 * np.arcsin(self.wavelength / (2 * d_spacing))
                        two_theta_deg = np.degrees(two_theta)
                        
                        if 10 < two_theta_deg < 120:
                            intensity = self._calculate_intensity(h, k, l, 'fcc')
                            peaks.append({
                                'hkl': (h, k, l),
                                'd_spacing': d_spacing,
                                '2theta': two_theta_deg,
                                'intensity': intensity
                            })
        
        return sorted(peaks, key=lambda x: x['2theta'])
    
    def _calculate_hex_peaks(self, a, c):
        """Calculate hexagonal diffraction peaks."""
        peaks = []
        for h in range(-3, 4):
            for k in range(-3, 4):
                for l in range(5):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    # Hexagonal d-spacing formula
                    d_inv_sq = (4/3) * ((h**2 + h*k + k**2) / a**2) + (l**2 / c**2)
                    if d_inv_sq > 0:
                        d_spacing = 1 / np.sqrt(d_inv_sq)
                        
                        if d_spacing < self.wavelength / 2:
                            continue
                            
                        two_theta = 2 * np.arcsin(self.wavelength / (2 * d_spacing))
                        two_theta_deg = np.degrees(two_theta)
                        
                        if 10 < two_theta_deg < 120:
                            intensity = self._calculate_intensity(h, k, l, 'hex')
                            peaks.append({
                                'hkl': (h, k, l),
                                'd_spacing': d_spacing,
                                '2theta': two_theta_deg,
                                'intensity': intensity
                            })
        
        return sorted(peaks, key=lambda x: x['2theta'])
    
    def _calculate_ortho_peaks(self, a, b, c):
        """Calculate orthorhombic diffraction peaks."""
        peaks = []
        for h in range(5):
            for k in range(5):
                for l in range(5):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    d_inv_sq = (h/a)**2 + (k/b)**2 + (l/c)**2
                    d_spacing = 1 / np.sqrt(d_inv_sq)
                    
                    if d_spacing < self.wavelength / 2:
                        continue
                        
                    two_theta = 2 * np.arcsin(self.wavelength / (2 * d_spacing))
                    two_theta_deg = np.degrees(two_theta)
                    
                    if 10 < two_theta_deg < 120:
                        intensity = self._calculate_intensity(h, k, l, 'ortho')
                        peaks.append({
                            'hkl': (h, k, l),
                            'd_spacing': d_spacing,
                            '2theta': two_theta_deg,
                            'intensity': intensity
                        })
        
        return sorted(peaks, key=lambda x: x['2theta'])
    
    def _calculate_intensity(self, h, k, l, structure):
        """Calculate relative intensity based on structure factor."""
        # Simplified intensity calculation
        multiplicity = self._get_multiplicity(h, k, l, structure)
        lorentz_pol = 1 / np.sin(np.radians(45))**2  # Simplified
        
        # Temperature factor (Debye-Waller)
        B = 0.5  # Thermal parameter
        sin_theta = np.sin(np.radians(45))
        temp_factor = np.exp(-B * (sin_theta / self.wavelength)**2)
        
        # Structure factor (simplified - would need atomic positions for accuracy)
        F_squared = (h**2 + k**2 + l**2) * 10  # Very simplified
        
        intensity = multiplicity * lorentz_pol * temp_factor * F_squared
        
        # Normalize
        return min(intensity / 100, 100)
    
    def _get_multiplicity(self, h, k, l, structure):
        """Get multiplicity of reflection."""
        # Count equivalent reflections
        unique = len(set([abs(h), abs(k), abs(l)]))
        
        if structure in ['bcc', 'fcc']:
            if unique == 1:
                return 8 if h != 0 else 6
            elif unique == 2:
                return 24 if 0 not in [h, k, l] else 12
            else:
                return 48
        else:
            # Simplified for other structures
            return 12
    
    def generate_powder_pattern(self, phase_composition, two_theta_range=(10, 120), 
                               step_size=0.02, strain=0, preferential_orientation=None):
        """
        Generate a powder diffraction pattern.
        
        Parameters:
        -----------
        phase_composition : dict
            Phase fractions, e.g., {'ferrite': 0.7, 'austenite': 0.2, 'carbide': 0.1}
        two_theta_range : tuple
            2θ range in degrees
        step_size : float
            Step size in degrees
        strain : float
            Uniform strain (positive = tensile, negative = compressive)
        preferential_orientation : dict
            Texture information
            
        Returns:
        --------
        pattern : dict
            Diffraction pattern data
        """
        two_theta = np.arange(two_theta_range[0], two_theta_range[1], step_size)
        intensity = np.zeros_like(two_theta)
        
        # Add peaks for each phase
        for phase_name, fraction in phase_composition.items():
            if phase_name not in self.phases:
                continue
                
            phase = self.phases[phase_name]
            
            for peak in phase['peaks']:
                # Apply strain shift
                peak_position = peak['2theta'] * (1 + strain)
                
                # Apply preferential orientation if specified
                intensity_mod = 1.0
                if preferential_orientation and phase_name in preferential_orientation:
                    hkl = peak['hkl']
                    pref_dir = preferential_orientation[phase_name]
                    # Simple dot product for texture effect
                    alignment = abs(np.dot(hkl, pref_dir) / (np.linalg.norm(hkl) * np.linalg.norm(pref_dir)))
                    intensity_mod = 0.5 + 1.5 * alignment
                
                # Calculate peak shape (pseudo-Voigt)
                sigma = 0.1 + 0.001 * peak_position  # Peak broadening with angle
                gamma = 0.05
                
                # Voigt profile (convolution of Gaussian and Lorentzian)
                peak_shape = voigt_profile(two_theta - peak_position, sigma, gamma)
                
                # Add to pattern
                intensity += (fraction * phase['intensity_factor'] * 
                            peak['intensity'] * intensity_mod * peak_shape)
        
        # Add background
        background = 50 + 10 * np.random.random(len(two_theta))
        background = signal.savgol_filter(background, 51, 3)  # Smooth background
        intensity += background
        
        # Add noise
        noise = np.random.normal(0, np.sqrt(intensity), len(intensity))
        intensity += noise
        intensity[intensity < 0] = 0
        
        return {
            '2theta': two_theta,
            'intensity': intensity,
            'phase_composition': phase_composition,
            'strain': strain
        }
    
    def generate_strain_map(self, grid_size=(50, 50), max_strain=0.005,
                           stress_field=None, temperature=700):
        """
        Generate 2D residual strain/stress maps.
        
        Parameters:
        -----------
        grid_size : tuple
            Size of the strain map grid
        max_strain : float
            Maximum strain value
        stress_field : ndarray
            Applied stress field (if None, generates random)
        temperature : float
            Temperature in Celsius
            
        Returns:
        --------
        strain_maps : dict
            Strain tensor components and derived quantities
        """
        print(f"Generating strain/stress maps ({grid_size[0]}x{grid_size[1]})...")
        
        # Generate or use provided stress field
        if stress_field is None:
            # Create realistic stress field with concentrations
            x = np.linspace(-1, 1, grid_size[0])
            y = np.linspace(-1, 1, grid_size[1])
            X, Y = np.meshgrid(x, y)
            
            # Base stress field with gradient
            stress_xx = 100 * (1 + 0.3 * X)  # MPa
            stress_yy = 80 * (1 + 0.2 * Y)
            stress_xy = 20 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            
            # Add stress concentrations (e.g., around defects)
            num_concentrations = 5
            for _ in range(num_concentrations):
                cx, cy = np.random.rand(2) * 2 - 1
                r = np.sqrt((X - cx)**2 + (Y - cy)**2)
                concentration = 50 * np.exp(-10 * r**2)
                stress_xx += concentration
                stress_yy += concentration * 0.8
        else:
            stress_xx = stress_field[0]
            stress_yy = stress_field[1]
            stress_xy = stress_field[2] if len(stress_field) > 2 else np.zeros(grid_size)
        
        # Calculate strain from stress (simplified Hooke's law)
        E = 200e3  # Young's modulus in MPa (200 GPa)
        nu = 0.3  # Poisson's ratio
        
        # Account for temperature effects
        alpha = 12e-6  # Thermal expansion coefficient (1/°C)
        thermal_strain = alpha * (temperature - 25)
        
        # Elastic strain components
        strain_xx = (stress_xx - nu * stress_yy) / E + thermal_strain
        strain_yy = (stress_yy - nu * stress_xx) / E + thermal_strain
        strain_xy = stress_xy / (E / (2 * (1 + nu)))
        
        # Add creep strain component (time-dependent)
        A = 1e-10 * np.exp(-temperature/100)
        n = 5
        von_mises = np.sqrt(stress_xx**2 + stress_yy**2 - stress_xx*stress_yy + 3*stress_xy**2)
        creep_strain = A * (von_mises ** n) * 1000  # Arbitrary time factor
        
        strain_xx += creep_strain
        strain_yy += creep_strain * 0.5  # Anisotropic creep
        
        # Calculate derived quantities
        strain_maps = {
            'strain_xx': strain_xx,
            'strain_yy': strain_yy,
            'strain_xy': strain_xy,
            'stress_xx': stress_xx,
            'stress_yy': stress_yy,
            'stress_xy': stress_xy,
            'von_mises_stress': von_mises,
            'hydrostatic_strain': (strain_xx + strain_yy) / 2,
            'max_shear_strain': np.sqrt((strain_xx - strain_yy)**2 / 4 + strain_xy**2),
            'temperature': temperature,
            'grid_size': grid_size
        }
        
        return strain_maps
    
    def simulate_in_situ_diffraction(self, time_hours=1000, time_steps=10,
                                    initial_composition=None, temperature=700, stress=100):
        """
        Simulate time-resolved in-situ diffraction during creep.
        
        Parameters:
        -----------
        time_hours : float
            Total test duration
        time_steps : int
            Number of time points
        initial_composition : dict
            Initial phase composition
        temperature : float
            Temperature in Celsius
        stress : float
            Applied stress in MPa
            
        Returns:
        --------
        time_series_data : dict
            Time-resolved diffraction data
        """
        print(f"Simulating in-situ diffraction: {time_steps} steps over {time_hours} hours")
        
        if initial_composition is None:
            initial_composition = {
                'ferrite': 0.75,
                'austenite': 0.15,
                'carbide': 0.08,
                'chromium_oxide': 0.02
            }
        
        time_points = np.linspace(0, time_hours, time_steps)
        patterns = []
        strain_maps_series = []
        phase_evolution = {phase: [] for phase in initial_composition.keys()}
        
        for i, t in enumerate(time_points):
            print(f"  Time point {i+1}/{time_steps}: t={t:.1f} hours")
            
            # Evolve phase composition (oxidation, phase transformation)
            composition = initial_composition.copy()
            
            # Oxidation increases with time
            oxidation_rate = 0.00001 * temperature/700 * np.sqrt(t + 1)
            composition['chromium_oxide'] = min(
                initial_composition['chromium_oxide'] + oxidation_rate,
                0.15
            )
            
            # Normalize remaining phases
            remaining = 1 - composition['chromium_oxide']
            for phase in ['ferrite', 'austenite', 'carbide']:
                composition[phase] = initial_composition[phase] * remaining / (1 - initial_composition['chromium_oxide'])
            
            # Calculate strain evolution
            creep_strain = 1e-5 * stress * (t / 100) ** 0.3  # Simplified creep law
            
            # Generate diffraction pattern
            pattern = self.generate_powder_pattern(
                composition,
                strain=creep_strain,
                preferential_orientation={'ferrite': [1, 1, 0]} if t > time_hours/2 else None
            )
            patterns.append(pattern)
            
            # Generate strain map
            strain_map = self.generate_strain_map(
                grid_size=(30, 30),
                max_strain=creep_strain,
                temperature=temperature
            )
            strain_maps_series.append(strain_map)
            
            # Record phase evolution
            for phase, fraction in composition.items():
                phase_evolution[phase].append(fraction)
        
        time_series_data = {
            'time_hours': time_points,
            'patterns': patterns,
            'strain_maps': strain_maps_series,
            'phase_evolution': phase_evolution,
            'test_conditions': {
                'temperature': temperature,
                'stress': stress,
                'initial_composition': initial_composition
            }
        }
        
        return time_series_data
    
    def save_diffraction_data(self, data, filename):
        """Save diffraction data to HDF5 file."""
        print(f"Saving diffraction data to {filename}...")
        
        with h5py.File(filename, 'w') as f:
            # Save metadata
            f.attrs['wavelength'] = self.wavelength
            f.attrs['beam_energy'] = self.beam_energy
            f.attrs['timestamp'] = datetime.now().isoformat()
            
            # Save patterns if present
            if 'patterns' in data:
                patterns_grp = f.create_group('patterns')
                for i, pattern in enumerate(data['patterns']):
                    pattern_grp = patterns_grp.create_group(f'pattern_{i:03d}')
                    pattern_grp.create_dataset('2theta', data=pattern['2theta'])
                    pattern_grp.create_dataset('intensity', data=pattern['intensity'])
                    pattern_grp.attrs['phase_composition'] = json.dumps(pattern['phase_composition'])
                    pattern_grp.attrs['strain'] = pattern['strain']
            
            # Save strain maps if present
            if 'strain_maps' in data:
                strain_grp = f.create_group('strain_maps')
                for i, strain_map in enumerate(data['strain_maps']):
                    map_grp = strain_grp.create_group(f'map_{i:03d}')
                    for key, value in strain_map.items():
                        if isinstance(value, np.ndarray):
                            map_grp.create_dataset(key, data=value, compression='gzip')
                        else:
                            map_grp.attrs[key] = value
            
            # Save phase evolution if present
            if 'phase_evolution' in data:
                phase_grp = f.create_group('phase_evolution')
                for phase, evolution in data['phase_evolution'].items():
                    phase_grp.create_dataset(phase, data=evolution)
            
            # Save time array if present
            if 'time_hours' in data:
                f.create_dataset('time_hours', data=data['time_hours'])
            
            # Save test conditions
            if 'test_conditions' in data:
                f.attrs['test_conditions'] = json.dumps(data['test_conditions'])
        
        print(f"  Saved diffraction data")


def main():
    """Generate complete synthetic diffraction dataset."""
    
    # Initialize generator
    gen = SynchrotronDiffractionGenerator(beam_energy=80)  # 80 keV synchrotron beam
    
    print(f"\n=== Synchrotron XRD Data Generation ===")
    print(f"Wavelength: {gen.wavelength:.4f} Å")
    
    # Generate initial powder pattern
    print("\n--- Generating Initial Powder Pattern ---")
    initial_pattern = gen.generate_powder_pattern(
        phase_composition={
            'ferrite': 0.75,
            'austenite': 0.15,
            'carbide': 0.08,
            'chromium_oxide': 0.02
        },
        strain=0
    )
    
    os.makedirs('synchrotron_data/diffraction/initial', exist_ok=True)
    gen.save_diffraction_data(
        {'patterns': [initial_pattern]},
        'synchrotron_data/diffraction/initial/initial_pattern.h5'
    )
    
    # Generate initial strain maps
    print("\n--- Generating Initial Strain/Stress Maps ---")
    initial_strain = gen.generate_strain_map(
        grid_size=(50, 50),
        temperature=25  # Room temperature
    )
    
    gen.save_diffraction_data(
        {'strain_maps': [initial_strain]},
        'synchrotron_data/diffraction/initial/initial_strain_map.h5'
    )
    
    # Generate time-resolved in-situ data for different conditions
    print("\n--- Generating In-Situ Time Series Data ---")
    
    test_conditions = [
        {'temperature': 600, 'stress': 150, 'time_hours': 500},
        {'temperature': 700, 'stress': 100, 'time_hours': 1000},
        {'temperature': 800, 'stress': 75, 'time_hours': 2000}
    ]
    
    os.makedirs('synchrotron_data/diffraction/time_series', exist_ok=True)
    
    for condition in test_conditions:
        print(f"\nCondition: T={condition['temperature']}°C, σ={condition['stress']}MPa")
        
        time_series = gen.simulate_in_situ_diffraction(
            time_hours=condition['time_hours'],
            time_steps=10,
            temperature=condition['temperature'],
            stress=condition['stress']
        )
        
        filename = (f"synchrotron_data/diffraction/time_series/"
                   f"insitu_T{condition['temperature']}_S{condition['stress']}.h5")
        gen.save_diffraction_data(time_series, filename)
    
    print("\n=== Diffraction Data Generation Complete ===")
    print("Data saved in synchrotron_data/diffraction/")


if __name__ == '__main__':
    main()