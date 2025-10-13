"""
Main dataset generator class for SOFC digital twin datasets.
"""

import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

from ..physics_simulators.electrochemical import ElectrochemicalSimulator
from ..physics_simulators.thermal import ThermalSimulator
from ..physics_simulators.structural import StructuralSimulator
from ..data_generators.high_fidelity import HighFidelityDataGenerator
from ..data_generators.experimental import ExperimentalDataGenerator
from ..data_generators.monitoring import MonitoringDataGenerator
from ..degradation_models.crack_propagation import CrackPropagationModel
from ..degradation_models.material_aging import MaterialAgingModel

class SOFCDatasetGenerator:
    """
    Main class for generating comprehensive SOFC digital twin datasets.
    
    This class orchestrates the generation of multi-fidelity datasets including:
    - High-fidelity physics-based simulation data
    - Experimental validation data
    - Real-time monitoring data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SOFC dataset generator.
        
        Args:
            config: Configuration dictionary for dataset generation parameters
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Initialize physics simulators
        self.electrochemical_sim = ElectrochemicalSimulator(self.config.get('electrochemical', {}))
        self.thermal_sim = ThermalSimulator(self.config.get('thermal', {}))
        self.structural_sim = StructuralSimulator(self.config.get('structural', {}))
        
        # Initialize data generators
        self.high_fidelity_gen = HighFidelityDataGenerator(
            electrochemical_sim=self.electrochemical_sim,
            thermal_sim=self.thermal_sim,
            structural_sim=self.structural_sim,
            config=self.config.get('high_fidelity', {})
        )
        
        self.experimental_gen = ExperimentalDataGenerator(
            config=self.config.get('experimental', {})
        )
        
        self.monitoring_gen = MonitoringDataGenerator(
            config=self.config.get('monitoring', {})
        )
        
        # Initialize degradation models
        self.crack_model = CrackPropagationModel(self.config.get('crack_propagation', {}))
        self.aging_model = MaterialAgingModel(self.config.get('material_aging', {}))
        
    def _default_config(self) -> Dict:
        """Return default configuration parameters."""
        return {
            'electrochemical': {
                'fuel_composition': {'H2': 0.7, 'H2O': 0.3, 'CO': 0.0, 'CH4': 0.0},
                'air_composition': {'O2': 0.21, 'N2': 0.79},
                'operating_pressure': 1.0e5,  # Pa
                'reference_temperature': 1073.15,  # K
            },
            'thermal': {
                'ambient_temperature': 298.15,  # K
                'convection_coefficient': 10.0,  # W/m²K
                'radiation_emissivity': 0.8,
            },
            'structural': {
                'material_properties': {
                    'anode': {'E': 200e9, 'nu': 0.3, 'alpha': 12e-6, 'density': 3000},
                    'electrolyte': {'E': 200e9, 'nu': 0.3, 'alpha': 10e-6, 'density': 6000},
                    'cathode': {'E': 200e9, 'nu': 0.3, 'alpha': 12e-6, 'density': 3000},
                    'interconnect': {'E': 200e9, 'nu': 0.3, 'alpha': 12e-6, 'density': 8000},
                }
            },
            'high_fidelity': {
                'mesh_resolution': [50, 50, 20],  # x, y, z resolution
                'time_steps': 100,
                'convergence_tolerance': 1e-6,
            },
            'experimental': {
                'noise_level': 0.05,  # 5% noise
                'sampling_frequency': 1.0,  # Hz
                'measurement_uncertainty': 0.02,  # 2% uncertainty
            },
            'monitoring': {
                'high_freq_sampling': 1.0,  # Hz
                'low_freq_sampling': 1/3600,  # Hz (hourly)
                'data_assimilation_window': 3600,  # seconds
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the dataset generator."""
        logger = logging.getLogger('SOFCDatasetGenerator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_high_fidelity_data(
        self,
        n_samples: int = 1000,
        operating_conditions_range: Optional[Dict] = None,
        material_properties_range: Optional[Dict] = None,
        degradation_states: Optional[List[Dict]] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate high-fidelity physics-based simulation data.
        
        Args:
            n_samples: Number of simulation samples to generate
            operating_conditions_range: Range of operating conditions to sweep
            material_properties_range: Range of material properties to vary
            degradation_states: List of degradation states to simulate
            output_file: Optional HDF5 file to save data
            
        Returns:
            Dictionary containing the generated dataset
        """
        self.logger.info(f"Generating {n_samples} high-fidelity simulation samples...")
        
        # Generate parameter combinations using Latin Hypercube Sampling
        param_combinations = self._generate_parameter_combinations(
            n_samples, operating_conditions_range, material_properties_range
        )
        
        # Generate simulation data
        dataset = self.high_fidelity_gen.generate_dataset(
            param_combinations, degradation_states
        )
        
        # Save to file if specified
        if output_file:
            self._save_dataset(dataset, output_file, 'high_fidelity')
            self.logger.info(f"High-fidelity data saved to {output_file}")
        
        return dataset
    
    def generate_experimental_data(
        self,
        test_duration_hours: float = 100.0,
        operating_profile: Optional[Dict] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate experimental validation data.
        
        Args:
            test_duration_hours: Duration of experimental test in hours
            operating_profile: Operating profile for the test
            output_file: Optional HDF5 file to save data
            
        Returns:
            Dictionary containing the experimental dataset
        """
        self.logger.info(f"Generating experimental data for {test_duration_hours} hours...")
        
        dataset = self.experimental_gen.generate_dataset(
            test_duration_hours, operating_profile
        )
        
        if output_file:
            self._save_dataset(dataset, output_file, 'experimental')
            self.logger.info(f"Experimental data saved to {output_file}")
        
        return dataset
    
    def generate_monitoring_data(
        self,
        duration_hours: float = 24.0,
        high_freq_sampling: float = 1.0,
        low_freq_sampling: float = 1/3600.0,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate real-time monitoring data.
        
        Args:
            duration_hours: Duration of monitoring in hours
            high_freq_sampling: High-frequency sampling rate (Hz)
            low_freq_sampling: Low-frequency sampling rate (Hz)
            output_file: Optional HDF5 file to save data
            
        Returns:
            Dictionary containing the monitoring dataset
        """
        self.logger.info(f"Generating monitoring data for {duration_hours} hours...")
        
        dataset = self.monitoring_gen.generate_dataset(
            duration_hours, high_freq_sampling, low_freq_sampling
        )
        
        if output_file:
            self._save_dataset(dataset, output_file, 'monitoring')
            self.logger.info(f"Monitoring data saved to {output_file}")
        
        return dataset
    
    def generate_complete_dataset(
        self,
        high_fidelity_samples: int = 1000,
        experimental_duration: float = 100.0,
        monitoring_duration: float = 24.0,
        output_dir: str = "sofc_datasets"
    ) -> Dict[str, str]:
        """
        Generate complete dataset including all three types.
        
        Args:
            high_fidelity_samples: Number of high-fidelity simulation samples
            experimental_duration: Experimental test duration in hours
            monitoring_duration: Monitoring duration in hours
            output_dir: Directory to save all datasets
            
        Returns:
            Dictionary with paths to generated dataset files
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info("Generating complete SOFC digital twin dataset...")
        
        # Generate high-fidelity data
        hf_file = os.path.join(output_dir, f"high_fidelity_{timestamp}.h5")
        hf_data = self.generate_high_fidelity_data(
            n_samples=high_fidelity_samples,
            output_file=hf_file
        )
        
        # Generate experimental data
        exp_file = os.path.join(output_dir, f"experimental_{timestamp}.h5")
        exp_data = self.generate_experimental_data(
            test_duration_hours=experimental_duration,
            output_file=exp_file
        )
        
        # Generate monitoring data
        mon_file = os.path.join(output_dir, f"monitoring_{timestamp}.h5")
        mon_data = self.generate_monitoring_data(
            duration_hours=monitoring_duration,
            output_file=mon_file
        )
        
        # Generate metadata file
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        self._generate_metadata(metadata_file, hf_data, exp_data, mon_data)
        
        self.logger.info(f"Complete dataset generated in {output_dir}")
        
        return {
            'high_fidelity': hf_file,
            'experimental': exp_file,
            'monitoring': mon_file,
            'metadata': metadata_file
        }
    
    def _generate_parameter_combinations(
        self,
        n_samples: int,
        operating_conditions_range: Optional[Dict],
        material_properties_range: Optional[Dict]
    ) -> List[Dict]:
        """Generate parameter combinations using Latin Hypercube Sampling."""
        from scipy.stats import qmc
        
        # Default parameter ranges
        default_operating = {
            'current_density': (0.1, 1.0),  # A/cm²
            'fuel_utilization': (0.6, 0.9),  # %
            'air_utilization': (0.1, 0.3),  # %
            'inlet_fuel_temp': (973.15, 1173.15),  # K
            'inlet_air_temp': (973.15, 1173.15),  # K
        }
        
        default_material = {
            'anode_porosity': (0.2, 0.4),
            'cathode_porosity': (0.2, 0.4),
            'electrolyte_thickness': (5e-6, 20e-6),  # m
            'anode_thickness': (200e-6, 800e-6),  # m
            'cathode_thickness': (20e-6, 100e-6),  # m
        }
        
        operating_range = operating_conditions_range or default_operating
        material_range = material_properties_range or default_material
        
        # Combine all parameters
        all_params = {**operating_range, **material_range}
        param_names = list(all_params.keys())
        param_ranges = list(all_params.values())
        
        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=len(param_names))
        samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        param_combinations = []
        for sample in samples:
            combination = {}
            for i, (name, (min_val, max_val)) in enumerate(zip(param_names, param_ranges)):
                combination[name] = min_val + sample[i] * (max_val - min_val)
            param_combinations.append(combination)
        
        return param_combinations
    
    def _save_dataset(self, dataset: Dict[str, Any], filename: str, dataset_type: str):
        """Save dataset to HDF5 file."""
        with h5py.File(filename, 'w') as f:
            # Add metadata
            f.attrs['dataset_type'] = dataset_type
            f.attrs['generation_time'] = datetime.now().isoformat()
            f.attrs['generator_version'] = "1.0.0"
            
            # Save data arrays
            for key, value in dataset.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, dict):
                    group = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            group.create_dataset(subkey, data=subvalue, compression='gzip')
                        else:
                            group.attrs[subkey] = subvalue
                else:
                    f.attrs[key] = value
    
    def _generate_metadata(self, filename: str, hf_data: Dict, exp_data: Dict, mon_data: Dict):
        """Generate metadata file for the complete dataset."""
        import json
        
        metadata = {
            'dataset_info': {
                'generation_time': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'description': 'SOFC Digital Twin Multi-Fidelity Dataset'
            },
            'high_fidelity': {
                'n_samples': len(hf_data.get('parameters', [])),
                'spatial_resolution': hf_data.get('spatial_resolution', []),
                'time_steps': hf_data.get('time_steps', 0),
                'variables': list(hf_data.keys())
            },
            'experimental': {
                'duration_hours': exp_data.get('duration_hours', 0),
                'sampling_frequency': exp_data.get('sampling_frequency', 0),
                'variables': list(exp_data.keys())
            },
            'monitoring': {
                'duration_hours': mon_data.get('duration_hours', 0),
                'high_freq_sampling': mon_data.get('high_freq_sampling', 0),
                'low_freq_sampling': mon_data.get('low_freq_sampling', 0),
                'variables': list(mon_data.keys())
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)