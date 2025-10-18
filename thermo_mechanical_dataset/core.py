"""
Core module for thermo-mechanical dataset generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from datetime import datetime
from scipy import interpolate
from scipy.stats import norm
import warnings

class ThermoMechanicalDataset:
    """
    Main class for generating comprehensive thermo-mechanical modeling datasets
    with multi-physics coupling and temperature-dependent properties.
    """
    
    # Mix identifications
    MIX_IDS = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    
    # Temperature range (°C)
    TEMP_MIN = 20
    TEMP_MAX = 800
    TEMP_CRITICAL_POINTS = [20, 100, 200, 300, 400, 500, 600, 700, 800]
    
    # Data types
    DATA_TYPES = ['Calibration', 'Validation']
    
    # Property types
    PROPERTY_TYPES = ['Thermal', 'Mechanical', 'Transport', 'Deformation']
    
    def __init__(self, 
                 output_dir: str = './output',
                 seed: int = 42,
                 uncertainty_level: float = 0.05):
        """
        Initialize the dataset generator
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        seed : int
            Random seed for reproducibility
        uncertainty_level : float
            Coefficient of variation for stochastic bounds (default 5%)
        """
        self.output_dir = output_dir
        self.seed = seed
        self.uncertainty_level = uncertainty_level
        
        np.random.seed(seed)
        
        # Create output directory structure
        self._create_output_structure()
        
        # Initialize property generators
        self.thermal_props = None
        self.mechanical_props = None
        self.transport_props = None
        
        # Storage for generated datasets
        self.calibration_data = {}
        self.validation_data = {}
        
        # Multi-scale linking parameters
        self.microstructure_params = self._initialize_microstructure()
        
    def _create_output_structure(self):
        """Create directory structure for output"""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, 'calibration'),
            os.path.join(self.output_dir, 'validation'),
            os.path.join(self.output_dir, 'fea_inputs'),
            os.path.join(self.output_dir, 'fea_inputs', 'abaqus'),
            os.path.join(self.output_dir, 'fea_inputs', 'ansys'),
            os.path.join(self.output_dir, 'fea_inputs', 'comsol'),
            os.path.join(self.output_dir, 'visualizations'),
            os.path.join(self.output_dir, 'documentation')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_microstructure(self) -> Dict:
        """
        Initialize microstructural parameters that inform macro-scale properties
        Based on Phase 3 microstructural analysis
        """
        microstructure = {}
        
        for mix_id in self.MIX_IDS:
            # Extract rubber content
            if mix_id == 'C':
                rubber_content = 0
                rubber_size = 'none'
            else:
                rubber_content = int(mix_id[1:].split('S')[0].split('L')[0])
                rubber_size = 'small' if 'S' in mix_id else 'large'
            
            microstructure[mix_id] = {
                'rubber_content_percent': rubber_content,
                'rubber_size': rubber_size,
                'initial_porosity': 0.02 + rubber_content * 0.002,  # Increases with rubber
                'pore_connectivity': 0.15 + rubber_content * 0.01,
                'interfacial_transition_zone': 50 + rubber_content * 5,  # μm
                'aggregate_volume_fraction': 0.70 - rubber_content * 0.015,
                'cement_paste_fraction': 0.30 - rubber_content * 0.005 + rubber_content * 0.02,
                'air_void_content': 0.02 + rubber_content * 0.003,
                'rubber_particle_size': {
                    'small': {'mean': 2.0, 'std': 0.5},  # mm
                    'large': {'mean': 8.0, 'std': 2.0},  # mm
                    'none': {'mean': 0, 'std': 0}
                }[rubber_size]
            }
        
        return microstructure
    
    def generate_temperature_array(self, n_points: int = 100, 
                                 include_critical: bool = True) -> np.ndarray:
        """
        Generate temperature array for property evaluation
        
        Parameters:
        -----------
        n_points : int
            Number of temperature points
        include_critical : bool
            Whether to include critical temperature points
        
        Returns:
        --------
        np.ndarray : Temperature values in °C
        """
        if include_critical:
            # Ensure critical points are included
            temps = np.unique(np.concatenate([
                self.TEMP_CRITICAL_POINTS,
                np.linspace(self.TEMP_MIN, self.TEMP_MAX, n_points)
            ]))
        else:
            temps = np.linspace(self.TEMP_MIN, self.TEMP_MAX, n_points)
        
        return np.sort(temps)
    
    def apply_stochastic_variation(self, 
                                  base_value: Union[float, np.ndarray],
                                  cv: Optional[float] = None,
                                  n_samples: int = 1) -> np.ndarray:
        """
        Apply stochastic variation to base values
        
        Parameters:
        -----------
        base_value : float or array
            Base deterministic value(s)
        cv : float
            Coefficient of variation (uses class default if None)
        n_samples : int
            Number of samples to generate
        
        Returns:
        --------
        np.ndarray : Values with stochastic variation
        """
        if cv is None:
            cv = self.uncertainty_level
        
        base_array = np.atleast_1d(base_value)
        std_dev = base_array * cv
        
        if n_samples == 1:
            return base_array
        
        samples = np.zeros((n_samples, len(base_array)))
        for i in range(n_samples):
            samples[i] = np.random.normal(base_array, std_dev)
        
        return samples
    
    def generate_complete_dataset(self, 
                                 calibration_ratio: float = 0.7,
                                 n_validation_sets: int = 5):
        """
        Generate complete calibration and validation datasets
        
        Parameters:
        -----------
        calibration_ratio : float
            Fraction of data for calibration (rest for validation)
        n_validation_sets : int
            Number of independent validation datasets
        """
        print("=" * 80)
        print("THERMO-MECHANICAL DATASET GENERATION")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Output directory: {self.output_dir}")
        print(f"Uncertainty level: {self.uncertainty_level*100:.1f}%")
        print(f"Calibration/Validation ratio: {calibration_ratio:.1%}/{(1-calibration_ratio):.1%}")
        print(f"Number of validation sets: {n_validation_sets}")
        print("=" * 80)
        
        # Import property generators
        from .thermal_properties import ThermalPropertyGenerator
        from .mechanical_properties import MechanicalPropertyGenerator
        from .transport_properties import TransportPropertyGenerator
        
        # Initialize generators with microstructure data
        self.thermal_props = ThermalPropertyGenerator(self.microstructure_params)
        self.mechanical_props = MechanicalPropertyGenerator(self.microstructure_params)
        self.transport_props = TransportPropertyGenerator(self.microstructure_params)
        
        # Generate calibration data
        print("\nGenerating calibration datasets...")
        self.calibration_data = self._generate_dataset_type('Calibration')
        
        # Generate validation datasets
        print("\nGenerating validation datasets...")
        self.validation_data = {}
        for i in range(n_validation_sets):
            print(f"  Validation set {i+1}/{n_validation_sets}")
            # Change seed for each validation set
            np.random.seed(self.seed + i + 1000)
            self.validation_data[f'Set_{i+1}'] = self._generate_dataset_type('Validation')
        
        # Reset seed
        np.random.seed(self.seed)
        
        # Export datasets
        self._export_all_datasets()
        
        print("\n" + "=" * 80)
        print("DATASET GENERATION COMPLETE")
        print("=" * 80)
        
        return self.calibration_data, self.validation_data
    
    def _generate_dataset_type(self, data_type: str) -> Dict:
        """Generate dataset for specific type (Calibration/Validation)"""
        dataset = {}
        
        for mix_id in self.MIX_IDS:
            print(f"  Processing mix: {mix_id}")
            
            dataset[mix_id] = {
                'thermal': self.thermal_props.generate(mix_id, data_type),
                'mechanical': self.mechanical_props.generate(mix_id, data_type),
                'transport': self.transport_props.generate(mix_id, data_type),
                'coupling': self._generate_coupling_parameters(mix_id, data_type),
                'metadata': self._generate_metadata(mix_id, data_type)
            }
        
        return dataset
    
    def _generate_coupling_parameters(self, mix_id: str, data_type: str) -> Dict:
        """Generate multi-physics coupling parameters"""
        micro = self.microstructure_params[mix_id]
        
        # Temperature array
        temps = self.generate_temperature_array(n_points=50)
        
        # Thermal-mechanical coupling
        thermal_strain_coeff = 8e-6 + micro['rubber_content_percent'] * 0.5e-6
        thermal_strain = thermal_strain_coeff * (temps - 20)
        
        # Poro-mechanical coupling (Biot coefficient)
        biot_coeff = 0.6 + 0.3 * np.exp(-0.005 * temps)
        biot_coeff *= (1 + micro['pore_connectivity'])
        
        # Thermal-transport coupling
        moisture_diffusivity_factor = np.exp(0.05 * (temps - 20) / 100)
        
        coupling = {
            'temperature': temps.tolist(),
            'thermal_expansion_coefficient': self.apply_stochastic_variation(thermal_strain_coeff).tolist(),
            'thermal_strain': thermal_strain.tolist(),
            'biot_coefficient': biot_coeff.tolist(),
            'biot_modulus': (2e9 / (1 + micro['porosity'] * 10)).tolist(),
            'moisture_diffusivity_factor': moisture_diffusivity_factor.tolist(),
            'thermo_hydro_coupling': 0.15 + micro['rubber_content_percent'] * 0.01,
            'creep_activation_energy': 45000 - micro['rubber_content_percent'] * 500,  # J/mol
        }
        
        return coupling
    
    def _generate_metadata(self, mix_id: str, data_type: str) -> Dict:
        """Generate metadata for dataset"""
        return {
            'mix_id': mix_id,
            'data_type': data_type,
            'generation_timestamp': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'temperature_range': [self.TEMP_MIN, self.TEMP_MAX],
            'uncertainty_level': self.uncertainty_level,
            'microstructure_params': self.microstructure_params[mix_id],
            'units': {
                'temperature': 'Celsius',
                'thermal_conductivity': 'W/(m·K)',
                'specific_heat': 'J/(kg·K)',
                'density': 'kg/m³',
                'elastic_modulus': 'GPa',
                'compressive_strength': 'MPa',
                'tensile_strength': 'MPa',
                'poisson_ratio': 'dimensionless',
                'permeability': 'm²',
                'porosity': 'fraction'
            }
        }
    
    def _export_all_datasets(self):
        """Export all generated datasets"""
        print("\nExporting datasets...")
        
        # Export calibration data
        self._export_dataset(self.calibration_data, 'calibration')
        
        # Export validation data
        for set_name, data in self.validation_data.items():
            self._export_dataset(data, f'validation/{set_name}')
        
        # Generate FEA input files
        self._generate_fea_inputs()
        
        # Generate summary statistics
        self._generate_summary_statistics()
    
    def _export_dataset(self, data: Dict, subdir: str):
        """Export dataset to various formats"""
        base_path = os.path.join(self.output_dir, subdir)
        os.makedirs(base_path, exist_ok=True)
        
        # Export as JSON
        json_path = os.path.join(base_path, 'complete_dataset.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Export as CSV for each mix and property type
        for mix_id in data.keys():
            mix_path = os.path.join(base_path, mix_id)
            os.makedirs(mix_path, exist_ok=True)
            
            for prop_type in ['thermal', 'mechanical', 'transport']:
                if prop_type in data[mix_id]:
                    df = pd.DataFrame(data[mix_id][prop_type])
                    csv_path = os.path.join(mix_path, f'{prop_type}_properties.csv')
                    df.to_csv(csv_path, index=False)
    
    def _generate_fea_inputs(self):
        """Generate FEA software input files"""
        from .fea_exporters import ABAQUSExporter, ANSYSExporter, COMSOLExporter
        
        print("\nGenerating FEA input files...")
        
        # ABAQUS
        abaqus = ABAQUSExporter(self.calibration_data)
        abaqus.export(os.path.join(self.output_dir, 'fea_inputs', 'abaqus'))
        
        # ANSYS
        ansys = ANSYSExporter(self.calibration_data)
        ansys.export(os.path.join(self.output_dir, 'fea_inputs', 'ansys'))
        
        # COMSOL
        comsol = COMSOLExporter(self.calibration_data)
        comsol.export(os.path.join(self.output_dir, 'fea_inputs', 'comsol'))
    
    def _generate_summary_statistics(self):
        """Generate summary statistics for all datasets"""
        summary = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'seed': self.seed,
                'uncertainty_level': self.uncertainty_level,
                'n_mixes': len(self.MIX_IDS),
                'n_validation_sets': len(self.validation_data),
                'temperature_range': [self.TEMP_MIN, self.TEMP_MAX]
            },
            'calibration_statistics': self._compute_statistics(self.calibration_data),
            'validation_statistics': {}
        }
        
        for set_name, data in self.validation_data.items():
            summary['validation_statistics'][set_name] = self._compute_statistics(data)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'dataset_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _compute_statistics(self, data: Dict) -> Dict:
        """Compute statistics for a dataset"""
        stats = {}
        
        for mix_id in data.keys():
            mix_stats = {}
            
            for prop_type in ['thermal', 'mechanical', 'transport']:
                if prop_type in data[mix_id]:
                    prop_data = data[mix_id][prop_type]
                    
                    # Compute statistics for each property
                    if isinstance(prop_data, dict):
                        mix_stats[prop_type] = {
                            key: {
                                'mean': float(np.mean(values)) if isinstance(values, list) else values,
                                'std': float(np.std(values)) if isinstance(values, list) else 0,
                                'min': float(np.min(values)) if isinstance(values, list) else values,
                                'max': float(np.max(values)) if isinstance(values, list) else values
                            }
                            for key, values in prop_data.items()
                            if isinstance(values, (list, float, int))
                        }
            
            stats[mix_id] = mix_stats
        
        return stats