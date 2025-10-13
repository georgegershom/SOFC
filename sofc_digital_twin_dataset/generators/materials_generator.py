"""
Materials and geometry data generator for SOFC digital twin.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Dict, Any, Tuple
from .base_generator import BaseGenerator


class MaterialsGenerator(BaseGenerator):
    """
    Generator for materials and geometry data including:
    - 3D microstructural data (tomography images)
    - Macro-scale geometry and assembly data
    - Material properties
    """
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        super().__init__(config_path)
        self.materials = self.config['materials']
        self.dimensions = self.config['dimensions']
    
    def generate(self) -> Dict[str, Any]:
        """Generate complete materials and geometry dataset."""
        print("Generating materials and geometry data...")
        
        data = {
            'microstructural': self._generate_microstructural_data(),
            'macro_scale': self._generate_macro_scale_data(),
            'material_properties': self._generate_material_properties()
        }
        
        return data
    
    def _generate_microstructural_data(self) -> Dict[str, Any]:
        """Generate 3D microstructural data (tomography images)."""
        print("  Generating 3D microstructural data...")
        
        resolution = self.dimensions['micro_scale']['resolution']
        physical_size = self.dimensions['micro_scale']['physical_size']
        
        # Generate 3D microstructure for each component
        microstructures = {}
        
        for component in ['anode', 'electrolyte', 'cathode']:
            print(f"    Generating {component} microstructure...")
            
            # Generate 3D binary microstructure
            microstructure_3d = self._generate_3d_microstructure(
                resolution, component
            )
            
            # Calculate microstructural properties
            properties = self._calculate_microstructural_properties(
                microstructure_3d, component
            )
            
            microstructures[component] = {
                'voxel_data': microstructure_3d,
                'resolution': resolution,
                'physical_size': physical_size,
                'voxel_size': physical_size[0] / resolution[0],
                'properties': properties
            }
        
        return microstructures
    
    def _generate_3d_microstructure(self, resolution: Tuple[int, int, int], 
                                  component: str) -> np.ndarray:
        """Generate 3D binary microstructure using random sphere packing."""
        nx, ny, nz = resolution
        
        # Get material properties
        material = self.materials[component]
        porosity = material['porosity']
        
        # Initialize with solid phase (0 = solid, 1 = pore)
        microstructure = np.zeros((nx, ny, nz), dtype=np.uint8)
        
        # Generate random sphere packing for porosity
        n_spheres = int(porosity * nx * ny * nz / (4/3 * np.pi * 3**3))  # Approximate
        
        for _ in range(n_spheres):
            # Random center position
            cx = np.random.randint(3, nx-3)
            cy = np.random.randint(3, ny-3)
            cz = np.random.randint(3, nz-3)
            
            # Random radius
            radius = np.random.uniform(1, 4)
            
            # Create sphere
            x, y, z = np.ogrid[:nx, :ny, :nz]
            mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2
            microstructure[mask] = 1
        
        # Apply morphological operations for more realistic structure
        microstructure = ndimage.binary_opening(microstructure, structure=np.ones((2,2,2)))
        microstructure = ndimage.binary_closing(microstructure, structure=np.ones((1,1,1)))
        
        return microstructure.astype(np.uint8)
    
    def _calculate_microstructural_properties(self, microstructure: np.ndarray, 
                                            component: str) -> Dict[str, float]:
        """Calculate microstructural properties from 3D data."""
        porosity = np.sum(microstructure) / microstructure.size
        
        # Calculate tortuosity using random walk method
        tortuosity = self._calculate_tortuosity(microstructure)
        
        # Calculate specific surface area
        surface_area = self._calculate_surface_area(microstructure)
        
        # Calculate pore size distribution
        pore_sizes = self._calculate_pore_size_distribution(microstructure)
        
        return {
            'porosity': float(porosity),
            'tortuosity': float(tortuosity),
            'specific_surface_area': float(surface_area),
            'mean_pore_size': float(np.mean(pore_sizes)),
            'pore_size_std': float(np.std(pore_sizes)),
            'pore_size_distribution': pore_sizes
        }
    
    def _calculate_tortuosity(self, microstructure: np.ndarray) -> float:
        """Calculate tortuosity using random walk method."""
        # Simplified tortuosity calculation
        # In practice, this would involve more sophisticated algorithms
        porosity = np.sum(microstructure) / microstructure.size
        return 1.0 + 0.5 * (1 - porosity) / porosity
    
    def _calculate_surface_area(self, microstructure: np.ndarray) -> float:
        """Calculate specific surface area."""
        # Count surface voxels (solid-pore interfaces)
        surface_voxels = 0
        for i in range(1, microstructure.shape[0]-1):
            for j in range(1, microstructure.shape[1]-1):
                for k in range(1, microstructure.shape[2]-1):
                    if microstructure[i,j,k] == 0:  # Solid voxel
                        # Check if any neighbor is pore
                        neighbors = microstructure[i-1:i+2, j-1:j+2, k-1:k+2]
                        if np.any(neighbors == 1):
                            surface_voxels += 1
        
        voxel_volume = (self.dimensions['micro_scale']['physical_size'][0] / 
                       self.dimensions['micro_scale']['resolution'][0])**3
        return surface_voxels * voxel_volume**(2/3)
    
    def _calculate_pore_size_distribution(self, microstructure: np.ndarray) -> np.ndarray:
        """Calculate pore size distribution."""
        # Label connected components
        labeled, num_features = ndimage.label(microstructure)
        
        pore_sizes = []
        for i in range(1, num_features + 1):
            pore_volume = np.sum(labeled == i)
            pore_sizes.append(pore_volume)
        
        return np.array(pore_sizes)
    
    def _generate_macro_scale_data(self) -> Dict[str, Any]:
        """Generate macro-scale geometry and assembly data."""
        print("  Generating macro-scale geometry data...")
        
        cell_dims = self.dimensions['macro_scale']['cell_dimensions']
        stack_height = self.dimensions['macro_scale']['stack_height']
        interconnect_thickness = self.dimensions['macro_scale']['interconnect_thickness']
        
        # Generate CAD-like geometry data
        geometry = {
            'single_cell': {
                'dimensions': cell_dims,
                'anode_thickness': cell_dims[2] * 0.4,
                'electrolyte_thickness': cell_dims[2] * 0.1,
                'cathode_thickness': cell_dims[2] * 0.5,
                'active_area': cell_dims[0] * cell_dims[1] * 0.8  # 80% active area
            },
            'interconnect': {
                'dimensions': [cell_dims[0], cell_dims[1], interconnect_thickness],
                'channel_depth': 0.5e-3,  # 0.5 mm
                'channel_width': 1.0e-3,  # 1.0 mm
                'rib_width': 1.0e-3,      # 1.0 mm
                'material': 'Crofer22APU'
            },
            'stack_assembly': {
                'cell_count': int(stack_height / (cell_dims[2] + interconnect_thickness)),
                'total_height': stack_height,
                'seal_thickness': 0.1e-3,  # 0.1 mm
                'manifold_diameter': 5e-3   # 5 mm
            }
        }
        
        return geometry
    
    def _generate_material_properties(self) -> Dict[str, Any]:
        """Generate material properties data."""
        print("  Generating material properties data...")
        
        properties = {}
        
        for component, material in self.materials.items():
            # Add temperature-dependent properties
            temperatures = np.linspace(600, 800, 21)  # 600-800°C
            
            properties[component] = {
                'base_properties': material,
                'temperature_dependent': {
                    'temperatures': temperatures,
                    'thermal_conductivity': self._temperature_dependent_property(
                        material['thermal_conductivity'], temperatures, 'thermal_cond'
                    ),
                    'electrical_conductivity': self._temperature_dependent_property(
                        material['electrical_conductivity'], temperatures, 'electrical_cond'
                    ),
                    'youngs_modulus': self._temperature_dependent_property(
                        material['youngs_modulus'], temperatures, 'youngs_modulus'
                    ),
                    'thermal_expansion': self._temperature_dependent_property(
                        material['thermal_expansion'], temperatures, 'thermal_expansion'
                    )
                }
            }
        
        return properties
    
    def _temperature_dependent_property(self, base_value: float, temperatures: np.ndarray, 
                                      property_type: str) -> np.ndarray:
        """Generate temperature-dependent material properties."""
        T = temperatures + 273.15  # Convert to Kelvin
        
        if property_type == 'thermal_cond':
            # Thermal conductivity typically decreases with temperature
            return base_value * (T[0] / T) ** 0.5
        elif property_type == 'electrical_cond':
            # Electrical conductivity follows Arrhenius behavior
            E_a = 0.1  # eV (activation energy)
            k_B = 8.617e-5  # eV/K
            return base_value * np.exp(-E_a / (k_B * T))
        elif property_type == 'youngs_modulus':
            # Young's modulus decreases with temperature
            return base_value * (1 - 0.1 * (T - T[0]) / T[0])
        elif property_type == 'thermal_expansion':
            # Thermal expansion coefficient increases with temperature
            return base_value * (1 + 0.2 * (T - T[0]) / T[0])
        else:
            return np.full_like(temperatures, base_value)
    
    def generate_tomography_images(self, component: str, num_slices: int = 10) -> np.ndarray:
        """Generate 2D tomography slice images."""
        microstructure = self._generate_3d_microstructure(
            self.dimensions['micro_scale']['resolution'], component
        )
        
        # Select slices along z-direction
        z_indices = np.linspace(0, microstructure.shape[2]-1, num_slices, dtype=int)
        slices = microstructure[:, :, z_indices]
        
        # Convert to 8-bit images for visualization
        images = []
        for slice_data in slices:
            # Normalize and convert to 8-bit
            img = (slice_data * 255).astype(np.uint8)
            images.append(img)
        
        return np.array(images)
    
    def save_microstructural_data(self, data: Dict[str, Any]) -> str:
        """Save microstructural data to file."""
        return self.save_data(data, 'materials_geometry/microstructural_data.h5')
    
    def save_macro_scale_data(self, data: Dict[str, Any]) -> str:
        """Save macro-scale data to file."""
        return self.save_data(data, 'materials_geometry/macro_scale_data.h5')