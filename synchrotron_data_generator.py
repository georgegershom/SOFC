"""
Synthetic Synchrotron X-ray Data Generator for SOFC Creep Deformation Studies

This module generates realistic synthetic 4D (3D + time) synchrotron X-ray tomography 
and diffraction data for Solid Oxide Fuel Cell (SOFC) creep deformation analysis.
"""

import numpy as np
import h5py
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import morphology, measure
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MaterialProperties:
    """Material properties for SOFC components"""
    alloy_composition: Dict[str, float]  # Element percentages
    grain_size_mean: float  # micrometers
    grain_size_std: float   # micrometers
    initial_porosity: float  # fraction
    elastic_modulus: float  # GPa
    poisson_ratio: float
    thermal_expansion_coeff: float  # 1/K
    creep_exponent: float
    activation_energy: float  # kJ/mol

@dataclass
class OperationalParameters:
    """Operational conditions during experiment"""
    temperature: float  # Celsius
    mechanical_stress: float  # MPa
    time_points: List[float]  # hours
    atmosphere: str
    heating_rate: float  # K/min
    cooling_rate: float  # K/min

@dataclass
class SampleGeometry:
    """Sample dimensions and geometry"""
    length: float  # mm
    width: float   # mm
    thickness: float  # mm
    shape: str  # "rectangular", "cylindrical", etc.
    volume: float  # mm³

class SynchrotronDataGenerator:
    """
    Main class for generating synthetic synchrotron X-ray experimental data
    """
    
    def __init__(self, 
                 voxel_size: float = 0.5,  # micrometers
                 image_dimensions: Tuple[int, int, int] = (512, 512, 256),
                 seed: Optional[int] = None):
        """
        Initialize the data generator
        
        Args:
            voxel_size: Size of each voxel in micrometers
            image_dimensions: (x, y, z) dimensions of 3D images
            seed: Random seed for reproducibility
        """
        self.voxel_size = voxel_size
        self.dimensions = image_dimensions
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize data storage
        self.tomography_data = {}
        self.diffraction_data = {}
        self.metadata = {}
        
    def generate_initial_microstructure(self, 
                                      material_props: MaterialProperties,
                                      sample_geom: SampleGeometry) -> np.ndarray:
        """
        Generate initial 3D microstructure with grains, grain boundaries, and porosity
        
        Returns:
            3D array representing initial microstructure
        """
        print("Generating initial microstructure...")
        
        # Create base material matrix
        microstructure = np.ones(self.dimensions, dtype=np.uint16)
        
        # Generate grain structure using Voronoi-like approach
        n_grains = int(np.prod(self.dimensions) / 
                      (4/3 * np.pi * (material_props.grain_size_mean / self.voxel_size)**3))
        
        # Random grain centers
        grain_centers = np.random.rand(n_grains, 3) * np.array(self.dimensions)
        
        # Assign each voxel to nearest grain
        coords = np.mgrid[0:self.dimensions[0], 
                         0:self.dimensions[1], 
                         0:self.dimensions[2]].reshape(3, -1).T
        
        # Use chunked processing for large arrays
        chunk_size = 100000
        grain_ids = np.zeros(coords.shape[0], dtype=np.uint16)
        
        for i in range(0, coords.shape[0], chunk_size):
            end_idx = min(i + chunk_size, coords.shape[0])
            chunk_coords = coords[i:end_idx]
            distances = cdist(chunk_coords, grain_centers)
            grain_ids[i:end_idx] = np.argmin(distances, axis=1) + 1
            
        microstructure = grain_ids.reshape(self.dimensions)
        
        # Add grain boundaries (transition zones between grains)
        grain_boundaries = self._create_grain_boundaries(microstructure)
        
        # Add initial porosity
        initial_pores = self._create_initial_porosity(material_props.initial_porosity)
        
        # Combine structures (0=pore, 1=grain_boundary, 2+=grain_interior)
        microstructure[grain_boundaries] = 1
        microstructure[initial_pores] = 0
        
        return microstructure
    
    def _create_grain_boundaries(self, grain_structure: np.ndarray, 
                               boundary_width: int = 2) -> np.ndarray:
        """Create grain boundaries between different grains"""
        boundaries = np.zeros_like(grain_structure, dtype=bool)
        
        # Find boundaries using gradient
        for axis in range(3):
            grad = np.gradient(grain_structure.astype(float), axis=axis)
            boundaries |= np.abs(grad) > 0.1
            
        # Dilate boundaries to create realistic width
        if boundary_width > 1:
            boundaries = morphology.binary_dilation(boundaries, 
                                                  morphology.ball(boundary_width))
        
        return boundaries
    
    def _create_initial_porosity(self, porosity_fraction: float) -> np.ndarray:
        """Create initial pore distribution"""
        # Random pore locations
        n_pores = int(porosity_fraction * np.prod(self.dimensions) / 100)
        pore_locations = np.random.randint(0, min(self.dimensions), size=(n_pores, 3))
        
        pores = np.zeros(self.dimensions, dtype=bool)
        
        # Create pores with various sizes
        for loc in pore_locations:
            pore_size = np.random.exponential(2) + 1
            if pore_size > 10:
                pore_size = 10
                
            # Create spherical pores
            pore_size_int = int(pore_size)
            y, x, z = np.ogrid[-pore_size_int:pore_size_int+1, 
                              -pore_size_int:pore_size_int+1, 
                              -pore_size_int:pore_size_int+1]
            mask = x*x + y*y + z*z <= pore_size_int*pore_size_int
            
            # Apply pore to microstructure
            x_start = max(0, loc[0] - pore_size_int)
            x_end = min(self.dimensions[0], loc[0] + pore_size_int + 1)
            y_start = max(0, loc[1] - pore_size_int)
            y_end = min(self.dimensions[1], loc[1] + pore_size_int + 1)
            z_start = max(0, loc[2] - pore_size_int)
            z_end = min(self.dimensions[2], loc[2] + pore_size_int + 1)
            
            mask_crop = mask[max(0, pore_size_int-loc[0]):mask.shape[0]-(max(0, loc[0]+pore_size_int+1-self.dimensions[0])),
                            max(0, pore_size_int-loc[1]):mask.shape[1]-(max(0, loc[1]+pore_size_int+1-self.dimensions[1])),
                            max(0, pore_size_int-loc[2]):mask.shape[2]-(max(0, loc[2]+pore_size_int+1-self.dimensions[2]))]
            
            pores[x_start:x_end, y_start:y_end, z_start:z_end] |= mask_crop
            
        return pores
    
    def simulate_creep_evolution(self, 
                               initial_structure: np.ndarray,
                               material_props: MaterialProperties,
                               op_params: OperationalParameters) -> Dict[float, np.ndarray]:
        """
        Simulate microstructural evolution during creep deformation
        
        Returns:
            Dictionary mapping time points to 3D microstructures
        """
        print("Simulating creep evolution over time...")
        
        evolution_data = {}
        current_structure = initial_structure.copy()
        
        for i, time_point in enumerate(op_params.time_points):
            print(f"  Processing time point {i+1}/{len(op_params.time_points)}: {time_point:.1f} hours")
            
            if i == 0:
                # Initial state
                evolution_data[time_point] = current_structure.copy()
                continue
            
            # Calculate time step
            dt = time_point - op_params.time_points[i-1]
            
            # Apply creep deformation mechanisms
            current_structure = self._apply_creep_mechanisms(
                current_structure, dt, material_props, op_params)
            
            evolution_data[time_point] = current_structure.copy()
            
        return evolution_data
    
    def _apply_creep_mechanisms(self, 
                              structure: np.ndarray, 
                              dt: float,
                              material_props: MaterialProperties,
                              op_params: OperationalParameters) -> np.ndarray:
        """Apply various creep deformation mechanisms"""
        
        # Create stress concentration map
        stress_map = self._calculate_stress_concentrations(structure, op_params.mechanical_stress)
        
        # 1. Cavity nucleation and growth
        structure = self._cavity_nucleation_growth(structure, stress_map, dt, material_props, op_params)
        
        # 2. Crack propagation
        structure = self._crack_propagation(structure, stress_map, dt)
        
        # 3. Grain boundary sliding (subtle grain shape changes)
        structure = self._grain_boundary_sliding(structure, dt, material_props)
        
        return structure
    
    def _calculate_stress_concentrations(self, structure: np.ndarray, applied_stress: float) -> np.ndarray:
        """Calculate local stress concentrations based on microstructure"""
        
        # Identify stress concentrators (pores, grain boundaries)
        pores = (structure == 0)
        grain_boundaries = (structure == 1)
        
        # Base stress field
        stress_field = np.full_like(structure, applied_stress, dtype=float)
        
        # Increase stress around pores and grain boundaries
        for feature in [pores, grain_boundaries]:
            if np.any(feature):
                # Distance transform to find proximity to features
                distance = ndimage.distance_transform_edt(~feature)
                stress_concentration = 1.0 + 2.0 * np.exp(-distance / 5.0)
                stress_field *= stress_concentration
        
        return stress_field
    
    def _cavity_nucleation_growth(self, 
                                structure: np.ndarray, 
                                stress_map: np.ndarray,
                                dt: float,
                                material_props: MaterialProperties,
                                op_params: OperationalParameters) -> np.ndarray:
        """Simulate cavity nucleation and growth"""
        
        # Nucleation probability based on stress and grain boundaries
        grain_boundaries = (structure == 1)
        nucleation_prob = 0.001 * dt * (stress_map / op_params.mechanical_stress)**2
        nucleation_prob[~grain_boundaries] *= 0.1  # Lower probability in grain interiors
        
        # Nucleate new cavities
        new_cavities = np.random.random(structure.shape) < nucleation_prob
        structure[new_cavities] = 0
        
        # Grow existing cavities
        existing_cavities = (structure == 0)
        if np.any(existing_cavities):
            # Growth rate proportional to stress
            growth_rate = 0.1 * dt * (stress_map / op_params.mechanical_stress)
            
            # Probabilistic growth
            growth_prob = growth_rate / 10.0  # Normalize
            growth_mask = np.random.random(structure.shape) < growth_prob
            
            # Dilate cavities where growth occurs
            growth_regions = existing_cavities & growth_mask
            if np.any(growth_regions):
                grown_cavities = morphology.binary_dilation(growth_regions, morphology.ball(1))
                structure[grown_cavities] = 0
        
        return structure
    
    def _crack_propagation(self, structure: np.ndarray, stress_map: np.ndarray, dt: float) -> np.ndarray:
        """Simulate crack propagation along high-stress paths"""
        
        # Find existing cracks (connected pore regions)
        pores = (structure == 0)
        labeled_pores = measure.label(pores)
        
        for region_id in range(1, labeled_pores.max() + 1):
            region = (labeled_pores == region_id)
            region_size = np.sum(region)
            
            # Only propagate larger cracks
            if region_size > 50:
                # Find crack tips (boundary of pore region)
                dilated = morphology.binary_dilation(region, morphology.ball(1))
                crack_tips = dilated & ~region
                
                # Propagation probability based on local stress
                local_stress = stress_map[crack_tips]
                if len(local_stress) > 0:
                    prop_prob = 0.01 * dt * (local_stress / np.mean(local_stress))
                    propagate = np.random.random(len(local_stress)) < prop_prob
                    
                    # Apply propagation
                    tip_coords = np.where(crack_tips)
                    for i, should_prop in enumerate(propagate):
                        if should_prop:
                            x, y, z = tip_coords[0][i], tip_coords[1][i], tip_coords[2][i]
                            structure[x, y, z] = 0
        
        return structure
    
    def _grain_boundary_sliding(self, structure: np.ndarray, dt: float, material_props: MaterialProperties) -> np.ndarray:
        """Simulate subtle grain boundary sliding effects"""
        
        # This is a simplified representation - in reality, this would involve
        # complex grain rotation and boundary migration
        
        grain_boundaries = (structure == 1)
        
        # Small probability of boundary migration
        migration_prob = 0.001 * dt
        migrate_mask = np.random.random(structure.shape) < migration_prob
        migrate_boundaries = grain_boundaries & migrate_mask
        
        if np.any(migrate_boundaries):
            # Slightly erode and dilate boundaries to simulate migration
            eroded = morphology.binary_erosion(migrate_boundaries, morphology.ball(1))
            dilated = morphology.binary_dilation(migrate_boundaries, morphology.ball(1))
            
            # Update structure
            structure[eroded] = 2  # Convert to grain interior
            structure[dilated & (structure > 1)] = 1  # Convert to grain boundary
        
        return structure
    
    def generate_xrd_data(self, 
                         microstructure: np.ndarray,
                         material_props: MaterialProperties,
                         op_params: OperationalParameters) -> Dict:
        """
        Generate synthetic X-ray diffraction data including phase identification
        and residual stress/strain mapping
        """
        print("Generating X-ray diffraction data...")
        
        xrd_data = {
            'phases': {},
            'strain_map': {},
            'stress_map': {},
            'diffraction_patterns': {}
        }
        
        # Define material phases based on composition
        phases = self._define_material_phases(material_props)
        
        # Generate strain and stress maps
        strain_map = self._calculate_strain_field(microstructure, op_params)
        stress_map = self._calculate_stress_field(strain_map, material_props)
        
        # Generate diffraction patterns for each phase
        diffraction_patterns = {}
        for phase_name, phase_info in phases.items():
            pattern = self._simulate_diffraction_pattern(
                phase_info, strain_map, microstructure)
            diffraction_patterns[phase_name] = pattern
        
        xrd_data['phases'] = phases
        xrd_data['strain_map'] = strain_map
        xrd_data['stress_map'] = stress_map
        xrd_data['diffraction_patterns'] = diffraction_patterns
        
        return xrd_data
    
    def _define_material_phases(self, material_props: MaterialProperties) -> Dict:
        """Define crystalline phases based on material composition"""
        
        phases = {}
        
        # Common SOFC interconnect phases
        if 'Cr' in material_props.alloy_composition:
            cr_content = material_props.alloy_composition['Cr']
            if cr_content > 15:  # Ferritic stainless steel
                phases['Ferrite'] = {
                    'crystal_system': 'cubic',
                    'lattice_parameter': 2.87,  # Angstroms
                    'space_group': 'Im-3m',
                    'density': 7.87,  # g/cm³
                    'elastic_modulus': 200,  # GPa
                    'volume_fraction': 0.85
                }
        
        if 'Fe' in material_props.alloy_composition:
            phases['Iron_oxide'] = {
                'crystal_system': 'cubic',
                'lattice_parameter': 8.39,
                'space_group': 'Fd-3m',
                'density': 5.17,
                'elastic_modulus': 230,
                'volume_fraction': 0.10
            }
        
        # Chromium oxide (common oxidation product)
        phases['Chromium_oxide'] = {
            'crystal_system': 'hexagonal',
            'lattice_parameter': 4.96,
            'space_group': 'R-3c',
            'density': 5.22,
            'elastic_modulus': 280,
            'volume_fraction': 0.05
        }
        
        return phases
    
    def _calculate_strain_field(self, microstructure: np.ndarray, op_params: OperationalParameters) -> np.ndarray:
        """Calculate 3D strain field based on microstructure and loading"""
        
        # Initialize strain field
        strain_field = np.zeros(microstructure.shape + (6,))  # 6 strain components (xx,yy,zz,xy,xz,yz)
        
        # Applied strain from mechanical loading
        applied_strain = op_params.mechanical_stress / 200000  # Assuming E=200 GPa
        
        # Base uniform strain
        strain_field[:, :, :, 0] = applied_strain  # εxx
        strain_field[:, :, :, 1] = -0.3 * applied_strain  # εyy (Poisson effect)
        strain_field[:, :, :, 2] = -0.3 * applied_strain  # εzz (Poisson effect)
        
        # Add strain concentrations around defects
        pores = (microstructure == 0)
        grain_boundaries = (microstructure == 1)
        
        # Strain concentration around pores
        if np.any(pores):
            distance_to_pores = ndimage.distance_transform_edt(~pores)
            strain_concentration = 1.0 + 2.0 * np.exp(-distance_to_pores / 3.0)
            
            for i in range(3):  # Apply to normal strain components
                strain_field[:, :, :, i] *= strain_concentration
        
        # Additional strain at grain boundaries
        if np.any(grain_boundaries):
            strain_field[grain_boundaries, 0] *= 1.5  # Increased strain at boundaries
            
        # Add thermal strain
        thermal_strain = 0.00001 * (op_params.temperature - 20)  # Reference temp 20°C
        strain_field[:, :, :, :3] += thermal_strain
        
        return strain_field
    
    def _calculate_stress_field(self, strain_field: np.ndarray, material_props: MaterialProperties) -> np.ndarray:
        """Calculate stress field from strain using elastic constants"""
        
        E = material_props.elastic_modulus * 1000  # Convert to MPa
        nu = material_props.poisson_ratio
        
        # Elastic constants for isotropic material
        lambda_lame = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        
        stress_field = np.zeros_like(strain_field)
        
        # Calculate stress components using Hooke's law
        for i in range(strain_field.shape[0]):
            for j in range(strain_field.shape[1]):
                for k in range(strain_field.shape[2]):
                    strain = strain_field[i, j, k, :]
                    
                    # Normal stresses
                    trace = strain[0] + strain[1] + strain[2]
                    stress_field[i, j, k, 0] = lambda_lame * trace + 2 * mu * strain[0]  # σxx
                    stress_field[i, j, k, 1] = lambda_lame * trace + 2 * mu * strain[1]  # σyy
                    stress_field[i, j, k, 2] = lambda_lame * trace + 2 * mu * strain[2]  # σzz
                    
                    # Shear stresses
                    stress_field[i, j, k, 3] = 2 * mu * strain[3]  # τxy
                    stress_field[i, j, k, 4] = 2 * mu * strain[4]  # τxz
                    stress_field[i, j, k, 5] = 2 * mu * strain[5]  # τyz
        
        return stress_field
    
    def _simulate_diffraction_pattern(self, phase_info: Dict, strain_map: np.ndarray, microstructure: np.ndarray) -> Dict:
        """Simulate X-ray diffraction pattern for a specific phase"""
        
        # Common diffraction peaks for the phase
        if phase_info['crystal_system'] == 'cubic':
            # Generate cubic reflections
            reflections = [(1,1,0), (2,0,0), (2,2,0), (3,1,0), (2,2,2)]
        elif phase_info['crystal_system'] == 'hexagonal':
            reflections = [(1,0,0), (0,0,2), (1,0,1), (1,0,2), (1,1,0)]
        else:
            reflections = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)]
        
        pattern = {
            'two_theta': [],
            'intensity': [],
            'peak_width': [],
            'strain_shift': []
        }
        
        a = phase_info['lattice_parameter'] / 10.0  # Convert Angstroms to nm
        wavelength = 0.154056  # Cu Kα radiation (nm)
        
        for h, k, l in reflections:
            # Calculate d-spacing
            if phase_info['crystal_system'] == 'cubic':
                d_spacing = a / np.sqrt(h**2 + k**2 + l**2)
            elif phase_info['crystal_system'] == 'hexagonal':
                c = a * 1.633  # Typical c/a ratio for hexagonal
                d_spacing = 1 / np.sqrt(4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)
            else:
                d_spacing = a / np.sqrt(h**2 + k**2 + l**2)
            
            # Calculate 2θ using Bragg's law
            sin_theta = wavelength / (2 * d_spacing)
            if sin_theta <= 1:
                theta = np.arcsin(sin_theta)
                two_theta = 2 * theta * 180 / np.pi
                
                # Calculate intensity (simplified structure factor)
                intensity = 1000 * phase_info['volume_fraction'] * np.exp(-0.1 * (h**2 + k**2 + l**2))
                
                # Peak broadening due to strain
                avg_strain = np.mean(strain_map[:, :, :, 0])  # Use xx strain component
                peak_width = 0.1 + abs(avg_strain) * 1000  # Strain broadening
                
                # Peak shift due to strain
                strain_shift = -two_theta * avg_strain
                
                pattern['two_theta'].append(two_theta + strain_shift)
                pattern['intensity'].append(intensity)
                pattern['peak_width'].append(peak_width)
                pattern['strain_shift'].append(strain_shift)
        
        return pattern
    
    def generate_complete_dataset(self,
                                material_props: MaterialProperties,
                                op_params: OperationalParameters,
                                sample_geom: SampleGeometry,
                                output_dir: str = "synchrotron_data") -> Dict:
        """
        Generate complete synthetic synchrotron dataset
        
        Returns:
            Dictionary containing all generated data and metadata
        """
        print("=== Generating Complete Synchrotron Dataset ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate initial microstructure
        initial_structure = self.generate_initial_microstructure(material_props, sample_geom)
        
        # Simulate creep evolution
        evolution_data = self.simulate_creep_evolution(initial_structure, material_props, op_params)
        
        # Generate XRD data for each time point
        xrd_evolution = {}
        for time_point, structure in evolution_data.items():
            xrd_data = self.generate_xrd_data(structure, material_props, op_params)
            xrd_evolution[time_point] = xrd_data
        
        # Compile complete dataset
        complete_dataset = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'material_properties': asdict(material_props),
                'operational_parameters': asdict(op_params),
                'sample_geometry': asdict(sample_geom),
                'voxel_size_um': self.voxel_size,
                'image_dimensions': self.dimensions,
                'data_type': '4D_synchrotron_xray',
                'experiment_type': 'SOFC_creep_deformation'
            },
            'tomography_4d': evolution_data,
            'xrd_evolution': xrd_evolution,
            'analysis_metrics': self._calculate_analysis_metrics(evolution_data)
        }
        
        # Save dataset
        self._save_dataset(complete_dataset, output_dir)
        
        return complete_dataset
    
    def _calculate_analysis_metrics(self, evolution_data: Dict[float, np.ndarray]) -> Dict:
        """Calculate key analysis metrics from the evolution data"""
        
        metrics = {
            'porosity_evolution': {},
            'crack_density_evolution': {},
            'connectivity_evolution': {},
            'damage_evolution': {}
        }
        
        for time_point, structure in evolution_data.items():
            # Porosity calculation
            porosity = np.sum(structure == 0) / structure.size
            metrics['porosity_evolution'][time_point] = porosity
            
            # Crack density (connected pore regions)
            pores = (structure == 0)
            labeled_pores = measure.label(pores)
            n_cracks = labeled_pores.max()
            crack_density = n_cracks / structure.size * 1e6  # per mm³
            metrics['crack_density_evolution'][time_point] = crack_density
            
            # Connectivity (Euler characteristic)
            if np.any(pores):
                euler_char = measure.euler_number(pores)
                metrics['connectivity_evolution'][time_point] = euler_char
            else:
                metrics['connectivity_evolution'][time_point] = 0
            
            # Overall damage parameter
            damage = 1 - np.sum(structure > 1) / structure.size
            metrics['damage_evolution'][time_point] = damage
        
        return metrics
    
    def _save_dataset(self, dataset: Dict, output_dir: str):
        """Save the complete dataset in multiple formats"""
        
        print("Saving dataset...")
        
        # Save metadata as JSON
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)
        
        # Save tomography data as HDF5
        with h5py.File(os.path.join(output_dir, 'tomography_4d.h5'), 'w') as f:
            for time_point, structure in dataset['tomography_4d'].items():
                f.create_dataset(f'time_{time_point:.1f}h', data=structure, compression='gzip')
        
        # Save XRD data
        with h5py.File(os.path.join(output_dir, 'xrd_data.h5'), 'w') as f:
            for time_point, xrd_data in dataset['xrd_evolution'].items():
                time_group = f.create_group(f'time_{time_point:.1f}h')
                
                # Save strain and stress maps
                time_group.create_dataset('strain_map', data=xrd_data['strain_map'], compression='gzip')
                time_group.create_dataset('stress_map', data=xrd_data['stress_map'], compression='gzip')
                
                # Save diffraction patterns
                patterns_group = time_group.create_group('diffraction_patterns')
                for phase_name, pattern in xrd_data['diffraction_patterns'].items():
                    phase_group = patterns_group.create_group(phase_name)
                    for key, values in pattern.items():
                        phase_group.create_dataset(key, data=np.array(values))
        
        # Save analysis metrics
        with open(os.path.join(output_dir, 'analysis_metrics.json'), 'w') as f:
            json.dump(dataset['analysis_metrics'], f, indent=2)
        
        print(f"Dataset saved to: {output_dir}")
        
        # Generate summary report
        self._generate_summary_report(dataset, output_dir)
    
    def _generate_summary_report(self, dataset: Dict, output_dir: str):
        """Generate a summary report of the generated dataset"""
        
        report_path = os.path.join(output_dir, 'dataset_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("SYNTHETIC SYNCHROTRON X-RAY DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("GENERATION DETAILS:\n")
            f.write(f"Generated: {dataset['metadata']['generation_timestamp']}\n")
            f.write(f"Data Type: {dataset['metadata']['data_type']}\n")
            f.write(f"Experiment: {dataset['metadata']['experiment_type']}\n\n")
            
            f.write("SAMPLE SPECIFICATIONS:\n")
            sample_geom = dataset['metadata']['sample_geometry']
            f.write(f"Dimensions: {sample_geom['length']} x {sample_geom['width']} x {sample_geom['thickness']} mm\n")
            f.write(f"Volume: {sample_geom['volume']:.2f} mm³\n")
            f.write(f"Shape: {sample_geom['shape']}\n\n")
            
            f.write("MATERIAL PROPERTIES:\n")
            mat_props = dataset['metadata']['material_properties']
            f.write(f"Alloy Composition: {mat_props['alloy_composition']}\n")
            f.write(f"Mean Grain Size: {mat_props['grain_size_mean']} μm\n")
            f.write(f"Initial Porosity: {mat_props['initial_porosity']:.3f}\n")
            f.write(f"Elastic Modulus: {mat_props['elastic_modulus']} GPa\n\n")
            
            f.write("OPERATIONAL CONDITIONS:\n")
            op_params = dataset['metadata']['operational_parameters']
            f.write(f"Temperature: {op_params['temperature']}°C\n")
            f.write(f"Mechanical Stress: {op_params['mechanical_stress']} MPa\n")
            f.write(f"Time Points: {len(op_params['time_points'])} measurements\n")
            f.write(f"Duration: {max(op_params['time_points'])} hours\n\n")
            
            f.write("IMAGING PARAMETERS:\n")
            f.write(f"Voxel Size: {dataset['metadata']['voxel_size_um']} μm\n")
            f.write(f"Image Dimensions: {dataset['metadata']['image_dimensions']}\n")
            f.write(f"Total Voxels: {np.prod(dataset['metadata']['image_dimensions']):,}\n\n")
            
            f.write("DAMAGE EVOLUTION SUMMARY:\n")
            metrics = dataset['analysis_metrics']
            initial_porosity = list(metrics['porosity_evolution'].values())[0]
            final_porosity = list(metrics['porosity_evolution'].values())[-1]
            f.write(f"Initial Porosity: {initial_porosity:.4f}\n")
            f.write(f"Final Porosity: {final_porosity:.4f}\n")
            f.write(f"Porosity Increase: {(final_porosity - initial_porosity):.4f}\n")
            
            initial_damage = list(metrics['damage_evolution'].values())[0]
            final_damage = list(metrics['damage_evolution'].values())[-1]
            f.write(f"Initial Damage: {initial_damage:.4f}\n")
            f.write(f"Final Damage: {final_damage:.4f}\n")
            f.write(f"Damage Progression: {(final_damage - initial_damage):.4f}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("- metadata.json: Experimental metadata and parameters\n")
            f.write("- tomography_4d.h5: 4D tomography data (3D + time)\n")
            f.write("- xrd_data.h5: X-ray diffraction patterns and strain/stress maps\n")
            f.write("- analysis_metrics.json: Quantitative analysis results\n")
            f.write("- dataset_summary.txt: This summary report\n")
        
        print(f"Summary report saved to: {report_path}")