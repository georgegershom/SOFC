"""
Data Loader Helper Functions
Convenient functions for loading and accessing the stratified flow simulation data
"""

import numpy as np
import json
import os

class StratifiedFlowData:
    """
    Main class for loading and accessing stratified flow simulation data.
    
    Example usage:
        data = StratifiedFlowData()
        velocity = data.load_velocity_field()
        attenuation = data.load_attenuation()
        validation = data.load_validation_data()
    """
    
    def __init__(self, base_path='.'):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        base_path : str
            Base path to the data directory (default: current directory)
        """
        self.base_path = base_path
        self.cfd_path = os.path.join(base_path, 'cfd_outputs')
        self.math_path = os.path.join(base_path, 'mathematical_model_outputs')
        self.val_path = os.path.join(base_path, 'validation_data')
        
        # Load coordinates once
        self.coords = self._load_coordinates()
        self.params = self._load_parameters()
    
    def _load_coordinates(self):
        """Load coordinate system"""
        coord_file = os.path.join(self.cfd_path, 'coordinates.json')
        with open(coord_file, 'r') as f:
            coords = json.load(f)
        
        # Convert lists to numpy arrays
        coords['x'] = np.array(coords['x'])
        coords['y'] = np.array(coords['y'])
        coords['z'] = np.array(coords['z'])
        coords['t'] = np.array(coords['t'])
        
        return coords
    
    def _load_parameters(self):
        """Load mathematical model parameters"""
        param_file = os.path.join(self.math_path, 'parameters.json')
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        # Convert lists to numpy arrays
        params['frequencies'] = np.array(params['frequencies'])
        params['void_fractions'] = np.array(params['void_fractions'])
        params['positions'] = np.array(params['positions'])
        
        return params
    
    # ========== CFD Data Loading ==========
    
    def load_velocity_field(self, component='all'):
        """
        Load velocity field data.
        
        Parameters:
        -----------
        component : str
            'u', 'v', 'w', or 'all' (default: 'all')
        
        Returns:
        --------
        dict or ndarray
            If 'all': dictionary with 'u', 'v', 'w' components and coordinates
            Otherwise: single velocity component array
        """
        if component == 'all':
            return {
                'u': np.load(os.path.join(self.cfd_path, 'velocity_u.npy')),
                'v': np.load(os.path.join(self.cfd_path, 'velocity_v.npy')),
                'w': np.load(os.path.join(self.cfd_path, 'velocity_w.npy')),
                'coordinates': self.coords
            }
        elif component in ['u', 'v', 'w']:
            return np.load(os.path.join(self.cfd_path, f'velocity_{component}.npy'))
        else:
            raise ValueError("component must be 'u', 'v', 'w', or 'all'")
    
    def load_pressure(self):
        """Load pressure field"""
        return {
            'pressure': np.load(os.path.join(self.cfd_path, 'pressure.npy')),
            'coordinates': self.coords
        }
    
    def load_vof(self):
        """Load Volume of Fluid (phase distribution)"""
        return {
            'vof': np.load(os.path.join(self.cfd_path, 'vof.npy')),
            'coordinates': self.coords
        }
    
    def load_turbulence(self):
        """Load turbulence parameters (k, epsilon, eddy viscosity)"""
        return {
            'k': np.load(os.path.join(self.cfd_path, 'turbulence_k.npy')),
            'epsilon': np.load(os.path.join(self.cfd_path, 'turbulence_epsilon.npy')),
            'eddy_viscosity': np.load(os.path.join(self.cfd_path, 'eddy_viscosity.npy')),
            'coordinates': self.coords
        }
    
    def load_acoustic_pressure(self, mmap=False):
        """
        Load acoustic pressure propagation data.
        
        Parameters:
        -----------
        mmap : bool
            If True, use memory mapping (doesn't load entire file into RAM)
            Useful for large files. Default: False
        
        Returns:
        --------
        dict
            Dictionary with acoustic pressure array and metadata
        """
        if mmap:
            acoustic = np.load(os.path.join(self.cfd_path, 'acoustic_pressure.npy'), 
                             mmap_mode='r')
        else:
            acoustic = np.load(os.path.join(self.cfd_path, 'acoustic_pressure.npy'))
        
        return {
            'acoustic_pressure': acoustic,
            'time': self.coords['t'],
            'x': self.coords['x'],
            'z': self.coords['z']
        }
    
    # ========== Mathematical Model Data Loading ==========
    
    def load_sound_speed(self, model='all'):
        """
        Load sound speed predictions.
        
        Parameters:
        -----------
        model : str
            'wood', 'dispersive', or 'all' (default: 'all')
        
        Returns:
        --------
        dict or ndarray
        """
        if model == 'all':
            return {
                'wood': np.load(os.path.join(self.math_path, 'sound_speed_wood.npy')),
                'dispersive': np.load(os.path.join(self.math_path, 'sound_speed_dispersive.npy')),
                'void_fractions': self.params['void_fractions'],
                'frequencies': self.params['frequencies']
            }
        elif model == 'wood':
            return {
                'sound_speed': np.load(os.path.join(self.math_path, 'sound_speed_wood.npy')),
                'void_fractions': self.params['void_fractions']
            }
        elif model == 'dispersive':
            return {
                'sound_speed': np.load(os.path.join(self.math_path, 'sound_speed_dispersive.npy')),
                'void_fractions': self.params['void_fractions'],
                'frequencies': self.params['frequencies']
            }
        else:
            raise ValueError("model must be 'wood', 'dispersive', or 'all'")
    
    def load_attenuation(self, unit='dB'):
        """
        Load attenuation coefficients.
        
        Parameters:
        -----------
        unit : str
            'dB' for dB/m or 'Np' for Nepers/m (default: 'dB')
        
        Returns:
        --------
        dict
        """
        if unit == 'dB':
            filename = 'attenuation_dB.npy'
            unit_str = 'dB/m'
        elif unit == 'Np':
            filename = 'attenuation_coefficients.npy'
            unit_str = 'Np/m'
        else:
            raise ValueError("unit must be 'dB' or 'Np'")
        
        return {
            'attenuation': np.load(os.path.join(self.math_path, filename)),
            'frequencies': self.params['frequencies'],
            'void_fractions': self.params['void_fractions'],
            'unit': unit_str
        }
    
    def load_wave_propagation(self):
        """Load wave propagation patterns (reflection, transmission, standing waves)"""
        # Load JSON data
        with open(os.path.join(self.math_path, 'reflection_transmission.json'), 'r') as f:
            refl_trans = json.load(f)
        
        return {
            'normal_incidence': refl_trans,
            'reflection_vs_angle': np.load(os.path.join(self.math_path, 'reflection_vs_angle.npy')),
            'standing_waves': np.load(os.path.join(self.math_path, 'standing_waves.npy')),
            'angles': np.linspace(0, 89, 90),
            'positions': self.params['positions'],
            'frequencies': self.params['frequencies'][:10]  # First 10 frequencies
        }
    
    def load_time_delays(self):
        """Load time-delay estimates"""
        with open(os.path.join(self.math_path, 'time_delays.json'), 'r') as f:
            delays = json.load(f)
        
        # Convert lists to numpy arrays
        for key in delays:
            delays[key]['distance'] = np.array(delays[key]['distance'])
            delays[key]['time_delay'] = np.array(delays[key]['time_delay'])
        
        return delays
    
    # ========== Validation Data Loading ==========
    
    def load_validation_data(self):
        """Load all validation comparison data"""
        # Waveform data
        waveform = {
            'simulated': np.load(os.path.join(self.val_path, 'waveform_simulated.npy')),
            'experimental': np.load(os.path.join(self.val_path, 'waveform_experimental.npy')),
            'time': np.load(os.path.join(self.val_path, 'waveform_time.npy'))
        }
        
        # Attenuation comparison
        with open(os.path.join(self.val_path, 'attenuation_comparison.json'), 'r') as f:
            atten_comp = json.load(f)
            for key in ['void_fraction', 'simulated', 'experimental', 'uncertainty']:
                atten_comp[key] = np.array(atten_comp[key])
        
        # Sound speed comparison
        with open(os.path.join(self.val_path, 'sound_speed_comparison.json'), 'r') as f:
            speed_comp = json.load(f)
            for key in ['void_fraction', 'simulated', 'experimental', 'uncertainty']:
                speed_comp[key] = np.array(speed_comp[key])
        
        # Statistics
        with open(os.path.join(self.val_path, 'validation_statistics.json'), 'r') as f:
            statistics = json.load(f)
        
        return {
            'waveform': waveform,
            'attenuation': atten_comp,
            'sound_speed': speed_comp,
            'statistics': statistics
        }
    
    # ========== Utility Functions ==========
    
    def get_slice(self, data, axis='y', index=None):
        """
        Extract a 2D slice from 3D data.
        
        Parameters:
        -----------
        data : ndarray
            3D array with shape (nx, ny, nz)
        axis : str
            'x', 'y', or 'z' (default: 'y')
        index : int or None
            Index along the axis. If None, uses middle (default: None)
        
        Returns:
        --------
        ndarray
            2D slice
        """
        if data.ndim != 3:
            raise ValueError("Data must be 3D array")
        
        nx, ny, nz = data.shape
        
        if axis == 'x':
            idx = index if index is not None else nx // 2
            return data[idx, :, :]
        elif axis == 'y':
            idx = index if index is not None else ny // 2
            return data[:, idx, :]
        elif axis == 'z':
            idx = index if index is not None else nz // 2
            return data[:, :, idx]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
    
    def get_profile(self, data, direction='z', x_idx=None, y_idx=None):
        """
        Extract a 1D profile from 3D data.
        
        Parameters:
        -----------
        data : ndarray
            3D array with shape (nx, ny, nz)
        direction : str
            'x', 'y', or 'z' (default: 'z')
        x_idx, y_idx : int or None
            Indices for other dimensions. If None, uses middle
        
        Returns:
        --------
        ndarray
            1D profile
        """
        if data.ndim != 3:
            raise ValueError("Data must be 3D array")
        
        nx, ny, nz = data.shape
        xi = x_idx if x_idx is not None else nx // 2
        yi = y_idx if y_idx is not None else ny // 2
        
        if direction == 'x':
            return data[:, yi, nz // 2]
        elif direction == 'y':
            return data[xi, :, nz // 2]
        elif direction == 'z':
            return data[xi, yi, :]
        else:
            raise ValueError("direction must be 'x', 'y', or 'z'")
    
    def summary(self):
        """Print summary of available data"""
        print("=" * 70)
        print("STRATIFIED FLOW SIMULATION DATA SUMMARY")
        print("=" * 70)
        print()
        
        print("CFD DATA:")
        print("-" * 70)
        print(f"  Domain: {self.coords['domain']['Lx']} × {self.coords['domain']['Ly']} × {self.coords['domain']['Lz']} m")
        print(f"  Grid: {self.coords['grid']['nx']} × {self.coords['grid']['ny']} × {self.coords['grid']['nz']}")
        print(f"  Time steps: {self.coords['grid']['nt']}")
        print()
        print("  Available fields:")
        print("    - Velocity (u, v, w)")
        print("    - Pressure")
        print("    - Volume of Fluid (VOF)")
        print("    - Turbulence (k, ε, μ_t)")
        print("    - Acoustic pressure propagation")
        print()
        
        print("MATHEMATICAL MODELS:")
        print("-" * 70)
        print(f"  Frequencies: {len(self.params['frequencies'])} points ({self.params['frequencies'][0]:.0f} - {self.params['frequencies'][-1]:.0f} Hz)")
        print(f"  Void fractions: {len(self.params['void_fractions'])} points (0.0 - 1.0)")
        print()
        print("  Available models:")
        print("    - Sound speed (Wood's equation, dispersive model)")
        print("    - Attenuation coefficients")
        print("    - Wave propagation (reflection, transmission)")
        print("    - Time delays")
        print()
        
        print("VALIDATION DATA:")
        print("-" * 70)
        print("  Available comparisons:")
        print("    - Acoustic waveforms")
        print("    - Attenuation coefficients")
        print("    - Sound speed")
        print("    - Statistical metrics")
        print()
        print("=" * 70)


# ========== Convenience Functions ==========

def quick_load_velocity():
    """Quick load velocity field"""
    data = StratifiedFlowData()
    return data.load_velocity_field()

def quick_load_acoustic():
    """Quick load acoustic pressure"""
    data = StratifiedFlowData()
    return data.load_acoustic_pressure(mmap=True)

def quick_load_attenuation():
    """Quick load attenuation data"""
    data = StratifiedFlowData()
    return data.load_attenuation(unit='dB')

def quick_load_validation():
    """Quick load validation data"""
    data = StratifiedFlowData()
    return data.load_validation_data()


# ========== Example Usage ==========

if __name__ == "__main__":
    # Create data loader
    data = StratifiedFlowData()
    
    # Print summary
    data.summary()
    
    # Example 1: Load velocity field
    print("\nExample 1: Loading velocity field...")
    velocity = data.load_velocity_field()
    print(f"  u shape: {velocity['u'].shape}")
    print(f"  v shape: {velocity['v'].shape}")
    print(f"  w shape: {velocity['w'].shape}")
    
    # Example 2: Load and slice pressure
    print("\nExample 2: Loading pressure and extracting slice...")
    pressure_data = data.load_pressure()
    pressure_slice = data.get_slice(pressure_data['pressure'], axis='y')
    print(f"  Pressure slice shape: {pressure_slice.shape}")
    
    # Example 3: Load attenuation
    print("\nExample 3: Loading attenuation coefficients...")
    atten = data.load_attenuation(unit='dB')
    print(f"  Attenuation shape: {atten['attenuation'].shape}")
    print(f"  Frequency range: {atten['frequencies'][0]:.0f} - {atten['frequencies'][-1]:.0f} Hz")
    
    # Example 4: Load validation data
    print("\nExample 4: Loading validation data...")
    validation = data.load_validation_data()
    print(f"  Waveform correlation: {validation['statistics']['waveform']['correlation']:.4f}")
    print(f"  Sound speed RMSE: {validation['statistics']['sound_speed']['rmse']:.2f} m/s")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
