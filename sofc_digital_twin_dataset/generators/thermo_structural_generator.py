"""
Thermo-structural field data generator for SOFC digital twin.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Tuple, List
from .base_generator import BaseGenerator


class ThermoStructuralGenerator(BaseGenerator):
    """
    Generator for thermo-structural field data including:
    - Temperature field data (thermocouples, IR camera)
    - Stress and strain data (strain gauges, DIC)
    """
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        super().__init__(config_path)
        self.sensors = self.config['sensors']
        self.materials = self.config['materials']
        self.dimensions = self.config['dimensions']
        self.time_duration = self.config['generation']['time_duration']
        self.time_step = self.config['generation']['time_step']
        self.time_points = np.arange(0, self.time_duration, self.time_step)
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete thermo-structural dataset."""
        print("Generating thermo-structural field data...")
        
        data = {
            'temperature_fields': self._generate_temperature_fields(),
            'stress_strain': self._generate_stress_strain_data(),
            'thermal_gradients': self._generate_thermal_gradients(),
            'mechanical_response': self._generate_mechanical_response()
        }
        
        return data
    
    def _generate_temperature_fields(self) -> Dict[str, Any]:
        """Generate temperature field data from various sensors."""
        print("  Generating temperature field data...")
        
        # Generate 2D temperature field over time
        temp_field_2d = self._generate_2d_temperature_field()
        
        # Extract thermocouple data
        thermocouple_data = self._extract_thermocouple_data(temp_field_2d)
        
        # Generate IR camera data
        ir_camera_data = self._generate_ir_camera_data(temp_field_2d)
        
        # Generate 3D temperature field
        temp_field_3d = self._generate_3d_temperature_field()
        
        return {
            'temperature_field_2d': temp_field_2d,
            'temperature_field_3d': temp_field_3d,
            'thermocouple_data': thermocouple_data,
            'ir_camera_data': ir_camera_data,
            'time_points': self.time_points
        }
    
    def _generate_2d_temperature_field(self) -> np.ndarray:
        """Generate 2D temperature field over time."""
        nx, ny = 50, 50  # Spatial resolution
        nt = len(self.time_points)
        
        # Create coordinate grids
        x = np.linspace(0, 0.1, nx)  # 10 cm
        y = np.linspace(0, 0.1, ny)  # 10 cm
        X, Y = np.meshgrid(x, y)
        
        # Initialize temperature field
        temp_field = np.zeros((nt, nx, ny))
        
        # Base temperature
        base_temp = 750  # °C
        
        for t_idx, t in enumerate(self.time_points):
            # Create temperature distribution with hot spots and gradients
            temp = base_temp + 50 * np.sin(2 * np.pi * t / 3600)  # 1 hour cycle
            
            # Add spatial variations
            # Hot spot near center
            hot_spot_x, hot_spot_y = 0.05, 0.05
            hot_spot_temp = 100 * np.exp(-((X - hot_spot_x)**2 + (Y - hot_spot_y)**2) / 0.01)
            
            # Inlet/outlet temperature gradient
            inlet_gradient = 20 * (1 - X / 0.1)
            outlet_gradient = 15 * (X / 0.1)
            
            # Random thermal fluctuations
            noise = 5 * np.random.normal(0, 1, (nx, ny))
            
            # Combine all effects
            temp_field[t_idx] = temp + hot_spot_temp + inlet_gradient + outlet_gradient + noise
            
            # Apply smoothing
            temp_field[t_idx] = gaussian_filter(temp_field[t_idx], sigma=1.0)
        
        return temp_field
    
    def _extract_thermocouple_data(self, temp_field_2d: np.ndarray) -> Dict[str, Any]:
        """Extract thermocouple data from temperature field."""
        nt, nx, ny = temp_field_2d.shape
        n_thermocouples = self.sensors['thermocouples']['count']
        
        # Define thermocouple locations (strategic positions)
        locations = self._get_thermocouple_locations(nx, ny, n_thermocouples)
        
        thermocouple_data = {}
        for i, (x_idx, y_idx) in enumerate(locations):
            # Extract temperature time series
            temp_series = temp_field_2d[:, x_idx, y_idx]
            
            # Add sensor noise
            noise_std = self.sensors['thermocouples']['accuracy']
            temp_series += np.random.normal(0, noise_std, nt)
            
            thermocouple_data[f'TC_{i+1:02d}'] = {
                'temperature': temp_series,
                'location': (x_idx, y_idx),
                'coordinates': (x_idx * 0.1 / nx, y_idx * 0.1 / ny),
                'time_points': self.time_points
            }
        
        return thermocouple_data
    
    def _get_thermocouple_locations(self, nx: int, ny: int, n_thermocouples: int) -> List[Tuple[int, int]]:
        """Get strategic thermocouple locations."""
        locations = []
        
        # Corner locations
        locations.extend([(0, 0), (nx-1, 0), (0, ny-1), (nx-1, ny-1)])
        
        # Edge centers
        locations.extend([(nx//2, 0), (nx//2, ny-1), (0, ny//2), (nx-1, ny//2)])
        
        # Center region
        locations.extend([(nx//2, ny//2), (nx//3, ny//3), (2*nx//3, 2*ny//3)])
        
        # Additional random locations
        while len(locations) < n_thermocouples:
            x = np.random.randint(0, nx)
            y = np.random.randint(0, ny)
            if (x, y) not in locations:
                locations.append((x, y))
        
        return locations[:n_thermocouples]
    
    def _generate_ir_camera_data(self, temp_field_2d: np.ndarray) -> Dict[str, Any]:
        """Generate IR camera data."""
        nt, nx, ny = temp_field_2d.shape
        
        # Downsample to IR camera resolution
        ir_resolution = self.sensors['ir_camera']['resolution']
        ir_nx, ir_ny = ir_resolution
        
        # Resample temperature field
        ir_temp_field = np.zeros((nt, ir_nx, ir_ny))
        
        for t_idx in range(nt):
            # Downsample using interpolation
            x_orig = np.linspace(0, 1, nx)
            y_orig = np.linspace(0, 1, ny)
            x_new = np.linspace(0, 1, ir_nx)
            y_new = np.linspace(0, 1, ir_ny)
            
            X_orig, Y_orig = np.meshgrid(x_orig, y_orig)
            X_new, Y_new = np.meshgrid(x_new, y_new)
            
            points = np.column_stack([X_orig.ravel(), Y_orig.ravel()])
            values = temp_field_2d[t_idx].ravel()
            new_points = np.column_stack([X_new.ravel(), Y_new.ravel()])
            
            ir_temp_field[t_idx] = griddata(points, values, new_points, method='cubic').reshape(ir_nx, ir_ny)
        
        # Add IR camera noise
        noise_std = self.sensors['ir_camera']['accuracy']
        ir_temp_field += np.random.normal(0, noise_std, ir_temp_field.shape)
        
        return {
            'temperature_field': ir_temp_field,
            'resolution': ir_resolution,
            'temperature_range': self.sensors['ir_camera']['temperature_range'],
            'time_points': self.time_points[::10]  # Lower sampling rate
        }
    
    def _generate_3d_temperature_field(self) -> np.ndarray:
        """Generate 3D temperature field."""
        nx, ny, nz = 20, 20, 10  # 3D resolution
        nt = len(self.time_points)
        
        # Create coordinate grids
        x = np.linspace(0, 0.1, nx)
        y = np.linspace(0, 0.1, ny)
        z = np.linspace(0, 0.01, nz)  # 1 cm thickness
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        temp_field_3d = np.zeros((nt, nx, ny, nz))
        
        for t_idx, t in enumerate(self.time_points):
            # Base temperature with time variation
            base_temp = 750 + 30 * np.sin(2 * np.pi * t / 1800)  # 30 min cycle
            
            # Through-thickness temperature gradient
            thickness_gradient = 20 * (Z / 0.01)
            
            # Hot spot in center
            hot_spot = 50 * np.exp(-((X - 0.05)**2 + (Y - 0.05)**2 + (Z - 0.005)**2) / 0.001)
            
            # Random fluctuations
            noise = 3 * np.random.normal(0, 1, (nx, ny, nz))
            
            temp_field_3d[t_idx] = base_temp + thickness_gradient + hot_spot + noise
        
        return temp_field_3d
    
    def _generate_stress_strain_data(self) -> Dict[str, Any]:
        """Generate stress and strain data."""
        print("  Generating stress and strain data...")
        
        # Generate strain gauge data
        strain_gauge_data = self._generate_strain_gauge_data()
        
        # Generate DIC data
        dic_data = self._generate_dic_data()
        
        # Generate stress field
        stress_field = self._generate_stress_field()
        
        return {
            'strain_gauge_data': strain_gauge_data,
            'dic_data': dic_data,
            'stress_field': stress_field,
            'time_points': self.time_points
        }
    
    def _generate_strain_gauge_data(self) -> Dict[str, Any]:
        """Generate strain gauge measurements."""
        n_gauges = self.sensors['strain_gauges']['count']
        nt = len(self.time_points)
        
        strain_data = {}
        
        for i in range(n_gauges):
            # Generate strain time series
            # Thermal expansion strain
            thermal_strain = 12e-6 * (self.time_points / 3600)  # Linear increase
            
            # Mechanical strain (sinusoidal)
            mechanical_strain = 100e-6 * np.sin(2 * np.pi * self.time_points / 1800)
            
            # Random fluctuations
            noise = 10e-6 * np.random.normal(0, 1, nt)
            
            total_strain = thermal_strain + mechanical_strain + noise
            
            # Add sensor noise
            sensor_noise = self.sensors['strain_gauges']['accuracy']
            total_strain += np.random.normal(0, sensor_noise, nt)
            
            strain_data[f'strain_gauge_{i+1:02d}'] = {
                'strain': total_strain,
                'thermal_strain': thermal_strain,
                'mechanical_strain': mechanical_strain,
                'time_points': self.time_points
            }
        
        return strain_data
    
    def _generate_dic_data(self) -> Dict[str, Any]:
        """Generate Digital Image Correlation data."""
        # Generate 2D displacement field
        nx, ny = 30, 30
        nt = len(self.time_points)
        
        # Displacement components
        u_field = np.zeros((nt, nx, ny))  # x-displacement
        v_field = np.zeros((nt, nx, ny))  # y-displacement
        
        for t_idx, t in enumerate(self.time_points):
            # Create coordinate grids
            x = np.linspace(0, 0.1, nx)
            y = np.linspace(0, 0.1, ny)
            X, Y = np.meshgrid(x, y)
            
            # Thermal expansion displacement
            alpha = 12e-6  # Thermal expansion coefficient
            delta_T = 50 * np.sin(2 * np.pi * t / 3600)
            
            u_field[t_idx] = alpha * delta_T * X
            v_field[t_idx] = alpha * delta_T * Y
            
            # Add mechanical deformation
            u_field[t_idx] += 1e-6 * np.sin(2 * np.pi * X / 0.05) * np.cos(2 * np.pi * t / 1800)
            v_field[t_idx] += 1e-6 * np.cos(2 * np.pi * Y / 0.05) * np.sin(2 * np.pi * t / 1800)
            
            # Add noise
            u_field[t_idx] += 0.1e-6 * np.random.normal(0, 1, (nx, ny))
            v_field[t_idx] += 0.1e-6 * np.random.normal(0, 1, (nx, ny))
        
        # Calculate strain fields
        strain_xx = np.gradient(u_field, axis=1)
        strain_yy = np.gradient(v_field, axis=2)
        strain_xy = 0.5 * (np.gradient(u_field, axis=2) + np.gradient(v_field, axis=1))
        
        return {
            'displacement_u': u_field,
            'displacement_v': v_field,
            'strain_xx': strain_xx,
            'strain_yy': strain_yy,
            'strain_xy': strain_xy,
            'coordinates': (x, y),
            'time_points': self.time_points
        }
    
    def _generate_stress_field(self) -> np.ndarray:
        """Generate stress field from temperature and strain data."""
        # This is a simplified stress calculation
        # In practice, this would involve solving the thermo-mechanical equations
        
        nx, ny = 30, 30
        nt = len(self.time_points)
        
        stress_field = np.zeros((nt, nx, ny, 3))  # xx, yy, xy components
        
        for t_idx in range(nt):
            # Generate temperature field
            x = np.linspace(0, 0.1, nx)
            y = np.linspace(0, 0.1, ny)
            X, Y = np.meshgrid(x, y)
            
            # Temperature field
            T = 750 + 50 * np.sin(2 * np.pi * self.time_points[t_idx] / 3600)
            T_field = T + 20 * np.exp(-((X - 0.05)**2 + (Y - 0.05)**2) / 0.01)
            
            # Thermal stress (simplified)
            E = 200e9  # Young's modulus
            alpha = 12e-6  # Thermal expansion coefficient
            nu = 0.3  # Poisson's ratio
            
            # Thermal strain
            thermal_strain = alpha * (T_field - 750)
            
            # Stress (plane stress assumption)
            stress_field[t_idx, :, :, 0] = E / (1 - nu**2) * thermal_strain  # xx
            stress_field[t_idx, :, :, 1] = E / (1 - nu**2) * thermal_strain  # yy
            stress_field[t_idx, :, :, 2] = np.zeros_like(thermal_strain)  # xy
        
        return stress_field
    
    def _generate_thermal_gradients(self) -> Dict[str, Any]:
        """Generate thermal gradient data."""
        temp_field_2d = self._generate_2d_temperature_field()
        
        # Calculate gradients
        grad_x = np.gradient(temp_field_2d, axis=1)
        grad_y = np.gradient(temp_field_2d, axis=2)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'gradient_x': grad_x,
            'gradient_y': grad_y,
            'gradient_magnitude': grad_magnitude,
            'time_points': self.time_points
        }
    
    def _generate_mechanical_response(self) -> Dict[str, Any]:
        """Generate mechanical response data."""
        # Generate displacement and stress data
        strain_data = self._generate_strain_gauge_data()
        stress_field = self._generate_stress_field()
        
        # Calculate mechanical response metrics
        max_stress = np.max(stress_field, axis=(1, 2, 3))
        mean_stress = np.mean(stress_field, axis=(1, 2, 3))
        
        return {
            'max_stress': max_stress,
            'mean_stress': mean_stress,
            'strain_energy': np.sum(stress_field**2, axis=(1, 2, 3)),
            'time_points': self.time_points
        }
    
    def save_temperature_data(self, data: Dict[str, Any]) -> str:
        """Save temperature data to file."""
        return self.save_data(data, 'thermo_structural/temperature_data.h5')
    
    def save_stress_strain_data(self, data: Dict[str, Any]) -> str:
        """Save stress/strain data to file."""
        return self.save_data(data, 'thermo_structural/stress_strain_data.h5')