#!/usr/bin/env python3
"""
SOFC Residual Stress Dataset Generator
=====================================

This module generates a comprehensive dataset for residual stress prediction in 
Solid Oxide Fuel Cells (SOFCs) using FEA simulations. The dataset includes:

1. Geometric parameters (plate dimensions, layer thicknesses, green density)
2. Material properties (Young's modulus, CTE, Poisson's ratio, sintering parameters)
3. Process parameters (sintering temperature profiles, atmosphere)
4. Simulation results (stress distributions, strain fields, fracture risk metrics)

The generator uses Design of Experiments (DOE) to systematically vary parameters
and FEA simulations to compute the resulting stress states.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
import h5py
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from pyDOE2 import lhs, fullfact, pbdesign
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GeometricParameters:
    """Geometric parameters for SOFC cell design"""
    plate_length: float = 100.0  # mm
    plate_width: float = 100.0   # mm
    anode_thickness: float = 0.3  # mm
    electrolyte_thickness: float = 0.15  # mm
    cathode_thickness: float = 0.05  # mm
    interconnect_thickness: float = 2.0  # mm
    green_density_anode: float = 0.6  # relative density
    green_density_electrolyte: float = 0.55  # relative density
    green_density_cathode: float = 0.6  # relative density
    green_density_interconnect: float = 0.7  # relative density

@dataclass
class MaterialProperties:
    """Material properties for each SOFC layer"""
    # Anode (Ni-YSZ)
    anode_E_25C: float = 55.0  # GPa
    anode_E_800C: float = 29.0  # GPa
    anode_nu: float = 0.29
    anode_CTE_25C: float = 12.5e-6  # 1/K
    anode_CTE_800C: float = 13.3e-6  # 1/K
    anode_k: float = 6.0  # W/m·K
    anode_rho: float = 6500.0  # kg/m³
    
    # Electrolyte (8YSZ)
    electrolyte_E_25C: float = 200.0  # GPa
    electrolyte_E_800C: float = 170.0  # GPa
    electrolyte_nu: float = 0.23
    electrolyte_CTE_25C: float = 10.0e-6  # 1/K
    electrolyte_CTE_800C: float = 10.5e-6  # 1/K
    electrolyte_k: float = 2.1  # W/m·K
    electrolyte_rho: float = 5900.0  # kg/m³
    
    # Cathode (LSM-YSZ)
    cathode_E_25C: float = 45.0  # GPa
    cathode_E_800C: float = 40.0  # GPa
    cathode_nu: float = 0.25
    cathode_CTE_25C: float = 11.5e-6  # 1/K
    cathode_CTE_800C: float = 12.0e-6  # 1/K
    cathode_k: float = 3.5  # W/m·K
    cathode_rho: float = 6200.0  # kg/m³
    
    # Interconnect (Crofer 22 APU)
    interconnect_E_25C: float = 160.0  # GPa
    interconnect_E_800C: float = 140.0  # GPa
    interconnect_nu: float = 0.30
    interconnect_CTE_25C: float = 11.5e-6  # 1/K
    interconnect_CTE_800C: float = 11.9e-6  # 1/K
    interconnect_k: float = 25.0  # W/m·K
    interconnect_rho: float = 7800.0  # kg/m³

@dataclass
class SinteringParameters:
    """Sintering process parameters"""
    # Temperature profile
    max_temperature: float = 1350.0  # °C
    heating_rate: float = 2.0  # °C/min
    cooling_rate: float = 2.0  # °C/min
    hold_time: float = 120.0  # minutes
    
    # Atmosphere
    atmosphere_type: str = "air"  # air, nitrogen, argon
    oxygen_partial_pressure: float = 0.21  # atm
    
    # Sintering kinetics
    anode_shrinkage_rate: float = 0.15  # per hour
    electrolyte_shrinkage_rate: float = 0.12  # per hour
    cathode_shrinkage_rate: float = 0.18  # per hour
    interconnect_shrinkage_rate: float = 0.08  # per hour
    
    # Onset temperatures
    anode_onset_temp: float = 1000.0  # °C
    electrolyte_onset_temp: float = 1100.0  # °C
    cathode_onset_temp: float = 950.0  # °C
    interconnect_onset_temp: float = 1200.0  # °C

@dataclass
class CreepParameters:
    """Creep behavior parameters for viscoelastic modeling"""
    # Norton-Bailey creep law: ė_cr = B * σ^n * exp(-Q/RT)
    anode_B: float = 1.2e-10  # s⁻¹ MPa⁻ⁿ
    anode_n: float = 2.1
    anode_Q: float = 420.0  # kJ/mol
    
    electrolyte_B: float = 8.5e-12  # s⁻¹ MPa⁻ⁿ
    electrolyte_n: float = 1.8
    electrolyte_Q: float = 385.0  # kJ/mol
    
    cathode_B: float = 2.1e-11  # s⁻¹ MPa⁻ⁿ
    cathode_n: float = 1.9
    cathode_Q: float = 400.0  # kJ/mol
    
    interconnect_B: float = 5.0e-8  # s⁻¹ MPa⁻ⁿ
    interconnect_n: float = 3.2
    interconnect_Q: float = 280.0  # kJ/mol

class SOFCDatasetGenerator:
    """
    Main class for generating SOFC residual stress datasets using FEA simulations
    """
    
    def __init__(self, output_dir: str = "sofc_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize parameter ranges for DOE
        self.param_ranges = self._define_parameter_ranges()
        
        # Initialize simulation results storage
        self.simulation_results = []
        
        logger.info(f"SOFC Dataset Generator initialized. Output directory: {self.output_dir}")
    
    def _define_parameter_ranges(self) -> Dict:
        """Define parameter ranges for Design of Experiments"""
        return {
            # Geometric parameters
            'plate_length': (80.0, 120.0),  # mm
            'plate_width': (80.0, 120.0),   # mm
            'anode_thickness': (0.2, 0.5),   # mm
            'electrolyte_thickness': (0.1, 0.2),  # mm
            'cathode_thickness': (0.03, 0.08),    # mm
            'interconnect_thickness': (1.5, 3.0), # mm
            
            # Green density variations
            'green_density_anode': (0.5, 0.7),
            'green_density_electrolyte': (0.45, 0.65),
            'green_density_cathode': (0.5, 0.7),
            'green_density_interconnect': (0.6, 0.8),
            
            # Material property variations (Young's modulus)
            'anode_E_25C': (45.0, 65.0),     # GPa
            'electrolyte_E_25C': (180.0, 220.0),  # GPa
            'cathode_E_25C': (35.0, 55.0),   # GPa
            'interconnect_E_25C': (140.0, 180.0), # GPa
            
            # CTE variations
            'anode_CTE_25C': (11.5e-6, 13.5e-6),  # 1/K
            'electrolyte_CTE_25C': (9.5e-6, 11.0e-6),  # 1/K
            'cathode_CTE_25C': (10.5e-6, 12.5e-6),  # 1/K
            'interconnect_CTE_25C': (10.5e-6, 12.5e-6),  # 1/K
            
            # Process parameters
            'max_temperature': (1300.0, 1400.0),  # °C
            'heating_rate': (1.0, 5.0),          # °C/min
            'cooling_rate': (1.0, 5.0),          # °C/min
            'hold_time': (60.0, 240.0),          # minutes
            
            # Sintering kinetics
            'anode_shrinkage_rate': (0.10, 0.20),
            'electrolyte_shrinkage_rate': (0.08, 0.16),
            'cathode_shrinkage_rate': (0.12, 0.24),
            'interconnect_shrinkage_rate': (0.05, 0.12),
            
            # Creep parameters
            'electrolyte_creep_B': (1e-12, 1e-11),  # s⁻¹ MPa⁻ⁿ
            'electrolyte_creep_n': (1.5, 2.2),
            'electrolyte_creep_Q': (350.0, 420.0),  # kJ/mol
        }
    
    def generate_doe_samples(self, n_samples: int = 1000, method: str = 'lhs') -> pd.DataFrame:
        """
        Generate Design of Experiments samples using Latin Hypercube Sampling
        
        Args:
            n_samples: Number of samples to generate
            method: Sampling method ('lhs', 'fullfact', 'pbdesign')
        
        Returns:
            DataFrame with parameter combinations
        """
        param_names = list(self.param_ranges.keys())
        param_ranges = list(self.param_ranges.values())
        
        if method == 'lhs':
            # Latin Hypercube Sampling
            samples = lhs(len(param_names), samples=n_samples, criterion='maximin')
        elif method == 'fullfact':
            # Full factorial design (for small parameter spaces)
            levels = [3] * len(param_names)  # 3 levels per parameter
            samples = fullfact(levels)
            samples = samples / (np.array(levels) - 1)  # Normalize to [0,1]
        elif method == 'pbdesign':
            # Plackett-Burman design
            samples = pbdesign(len(param_names))
            samples = (samples + 1) / 2  # Normalize to [0,1]
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Scale samples to parameter ranges
        scaled_samples = np.zeros_like(samples)
        for i, (param_min, param_max) in enumerate(param_ranges):
            scaled_samples[:, i] = param_min + samples[:, i] * (param_max - param_min)
        
        # Create DataFrame
        df = pd.DataFrame(scaled_samples, columns=param_names)
        
        # Add sample ID
        df['sample_id'] = range(len(df))
        
        logger.info(f"Generated {len(df)} DOE samples using {method} method")
        return df
    
    def run_fea_simulation(self, params: Dict) -> Dict:
        """
        Run FEA simulation for given parameters
        
        This is a simplified implementation. In practice, this would interface
        with FEniCS, COMSOL, or other FEA software.
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            Dictionary containing simulation results
        """
        # Extract parameters
        geom = GeometricParameters(
            plate_length=params['plate_length'],
            plate_width=params['plate_width'],
            anode_thickness=params['anode_thickness'],
            electrolyte_thickness=params['electrolyte_thickness'],
            cathode_thickness=params['cathode_thickness'],
            interconnect_thickness=params['interconnect_thickness'],
            green_density_anode=params['green_density_anode'],
            green_density_electrolyte=params['green_density_electrolyte'],
            green_density_cathode=params['green_density_cathode'],
            green_density_interconnect=params['green_density_interconnect']
        )
        
        mat_props = MaterialProperties(
            anode_E_25C=params['anode_E_25C'],
            electrolyte_E_25C=params['electrolyte_E_25C'],
            cathode_E_25C=params['cathode_E_25C'],
            interconnect_E_25C=params['interconnect_E_25C'],
            anode_CTE_25C=params['anode_CTE_25C'],
            electrolyte_CTE_25C=params['electrolyte_CTE_25C'],
            cathode_CTE_25C=params['cathode_CTE_25C'],
            interconnect_CTE_25C=params['interconnect_CTE_25C']
        )
        
        sinter_params = SinteringParameters(
            max_temperature=params['max_temperature'],
            heating_rate=params['heating_rate'],
            cooling_rate=params['cooling_rate'],
            hold_time=params['hold_time'],
            anode_shrinkage_rate=params['anode_shrinkage_rate'],
            electrolyte_shrinkage_rate=params['electrolyte_shrinkage_rate'],
            cathode_shrinkage_rate=params['cathode_shrinkage_rate'],
            interconnect_shrinkage_rate=params['interconnect_shrinkage_rate']
        )
        
        creep_params = CreepParameters(
            electrolyte_B=params['electrolyte_creep_B'],
            electrolyte_n=params['electrolyte_creep_n'],
            electrolyte_Q=params['electrolyte_creep_Q']
        )
        
        # Simulate FEA results (simplified analytical model)
        results = self._simulate_stress_analytically(geom, mat_props, sinter_params, creep_params)
        
        return results
    
    def _simulate_stress_analytically(self, geom: GeometricParameters, 
                                    mat_props: MaterialProperties,
                                    sinter_params: SinteringParameters,
                                    creep_params: CreepParameters) -> Dict:
        """
        Simplified analytical stress simulation
        
        This replaces actual FEA with analytical calculations for demonstration.
        In practice, this would call FEniCS or other FEA software.
        """
        # Calculate CTE mismatch stresses
        cte_mismatch_anode = mat_props.anode_CTE_25C - mat_props.electrolyte_CTE_25C
        cte_mismatch_cathode = mat_props.cathode_CTE_25C - mat_props.electrolyte_CTE_25C
        cte_mismatch_interconnect = mat_props.interconnect_CTE_25C - mat_props.electrolyte_CTE_25C
        
        # Temperature change from sintering to room temperature
        delta_T = sinter_params.max_temperature - 25.0
        
        # Calculate thermal stresses (simplified)
        # σ = E * α * ΔT for constrained thermal expansion
        stress_anode = mat_props.anode_E_25C * cte_mismatch_anode * delta_T
        stress_cathode = mat_props.cathode_E_25C * cte_mismatch_cathode * delta_T
        stress_interconnect = mat_props.interconnect_E_25C * cte_mismatch_interconnect * delta_T
        
        # Calculate residual stress in electrolyte (simplified)
        # This is a very simplified model - real FEA would be much more complex
        electrolyte_residual_stress = (
            (stress_anode * geom.anode_thickness + 
             stress_cathode * geom.cathode_thickness + 
             stress_interconnect * geom.interconnect_thickness) / 
            geom.electrolyte_thickness
        )
        
        # Add geometric effects
        aspect_ratio = geom.plate_length / geom.plate_width
        geometric_factor = 1.0 + 0.1 * (aspect_ratio - 1.0)
        
        # Add sintering effects
        sintering_factor = (
            sinter_params.anode_shrinkage_rate * geom.green_density_anode +
            sinter_params.electrolyte_shrinkage_rate * geom.green_density_electrolyte +
            sinter_params.cathode_shrinkage_rate * geom.green_density_cathode
        ) / 3.0
        
        # Calculate final stress components
        max_principal_stress = abs(electrolyte_residual_stress) * geometric_factor * (1.0 + sintering_factor)
        von_mises_stress = max_principal_stress * 0.8  # Simplified relationship
        shear_stress = max_principal_stress * 0.3
        
        # Calculate creep relaxation (simplified)
        # This would be much more complex in real FEA
        creep_relaxation_factor = 1.0 - 0.2 * np.exp(-creep_params.electrolyte_Q / (8.314 * 1073))  # 800°C
        
        # Apply creep relaxation
        max_principal_stress_relaxed = max_principal_stress * creep_relaxation_factor
        von_mises_stress_relaxed = von_mises_stress * creep_relaxation_factor
        
        # Calculate fracture risk metrics
        characteristic_strength = 165.0  # MPa for 8YSZ
        safety_factor_elastic = characteristic_strength / max_principal_stress
        safety_factor_viscoelastic = characteristic_strength / max_principal_stress_relaxed
        
        # Calculate strain components
        elastic_strain = max_principal_stress / mat_props.electrolyte_E_25C
        creep_strain = elastic_strain * (1.0 - creep_relaxation_factor)
        total_strain = elastic_strain + creep_strain
        
        # Calculate stress concentration factors
        stress_concentration_factor = 1.0 + 0.5 * (geom.electrolyte_thickness / geom.plate_length)
        
        # Compile results
        results = {
            # Input parameters
            'sample_id': params.get('sample_id', 0),
            
            # Geometric parameters
            'plate_length': geom.plate_length,
            'plate_width': geom.plate_width,
            'anode_thickness': geom.anode_thickness,
            'electrolyte_thickness': geom.electrolyte_thickness,
            'cathode_thickness': geom.cathode_thickness,
            'interconnect_thickness': geom.interconnect_thickness,
            'green_density_anode': geom.green_density_anode,
            'green_density_electrolyte': geom.green_density_electrolyte,
            'green_density_cathode': geom.green_density_cathode,
            'green_density_interconnect': geom.green_density_interconnect,
            
            # Material properties
            'anode_E_25C': mat_props.anode_E_25C,
            'electrolyte_E_25C': mat_props.electrolyte_E_25C,
            'cathode_E_25C': mat_props.cathode_E_25C,
            'interconnect_E_25C': mat_props.interconnect_E_25C,
            'anode_CTE_25C': mat_props.anode_CTE_25C,
            'electrolyte_CTE_25C': mat_props.electrolyte_CTE_25C,
            'cathode_CTE_25C': mat_props.cathode_CTE_25C,
            'interconnect_CTE_25C': mat_props.interconnect_CTE_25C,
            
            # Process parameters
            'max_temperature': sinter_params.max_temperature,
            'heating_rate': sinter_params.heating_rate,
            'cooling_rate': sinter_params.cooling_rate,
            'hold_time': sinter_params.hold_time,
            'anode_shrinkage_rate': sinter_params.anode_shrinkage_rate,
            'electrolyte_shrinkage_rate': sinter_params.electrolyte_shrinkage_rate,
            'cathode_shrinkage_rate': sinter_params.cathode_shrinkage_rate,
            'interconnect_shrinkage_rate': sinter_params.interconnect_shrinkage_rate,
            
            # Creep parameters
            'electrolyte_creep_B': creep_params.electrolyte_B,
            'electrolyte_creep_n': creep_params.electrolyte_n,
            'electrolyte_creep_Q': creep_params.electrolyte_Q,
            
            # Simulation results
            'max_principal_stress_elastic': max_principal_stress,
            'max_principal_stress_viscoelastic': max_principal_stress_relaxed,
            'von_mises_stress_elastic': von_mises_stress,
            'von_mises_stress_viscoelastic': von_mises_stress_relaxed,
            'shear_stress': shear_stress,
            'elastic_strain': elastic_strain,
            'creep_strain': creep_strain,
            'total_strain': total_strain,
            'stress_concentration_factor': stress_concentration_factor,
            'safety_factor_elastic': safety_factor_elastic,
            'safety_factor_viscoelastic': safety_factor_viscoelastic,
            'fracture_risk_elastic': 1.0 / safety_factor_elastic if safety_factor_elastic > 0 else 1.0,
            'fracture_risk_viscoelastic': 1.0 / safety_factor_viscoelastic if safety_factor_viscoelastic > 0 else 1.0,
            'creep_relaxation_factor': creep_relaxation_factor,
            
            # CTE mismatch calculations
            'cte_mismatch_anode': cte_mismatch_anode,
            'cte_mismatch_cathode': cte_mismatch_cathode,
            'cte_mismatch_interconnect': cte_mismatch_interconnect,
            'delta_temperature': delta_T,
        }
        
        return results
    
    def generate_dataset(self, n_samples: int = 1000, method: str = 'lhs', 
                        save_results: bool = True) -> pd.DataFrame:
        """
        Generate complete dataset with FEA simulations
        
        Args:
            n_samples: Number of samples to generate
            method: DOE sampling method
            save_results: Whether to save results to files
            
        Returns:
            Complete dataset DataFrame
        """
        logger.info(f"Generating dataset with {n_samples} samples using {method} method")
        
        # Generate DOE samples
        doe_samples = self.generate_doe_samples(n_samples, method)
        
        # Run simulations
        results = []
        for idx, row in tqdm(doe_samples.iterrows(), total=len(doe_samples), 
                           desc="Running FEA simulations"):
            try:
                sim_result = self.run_fea_simulation(row.to_dict())
                results.append(sim_result)
            except Exception as e:
                logger.error(f"Simulation failed for sample {idx}: {e}")
                continue
        
        # Convert to DataFrame
        dataset = pd.DataFrame(results)
        
        if save_results:
            self._save_dataset(dataset)
        
        logger.info(f"Dataset generation complete. Generated {len(dataset)} samples")
        return dataset
    
    def _save_dataset(self, dataset: pd.DataFrame):
        """Save dataset to various formats"""
        # Save as CSV
        csv_path = self.output_dir / "sofc_residual_stress_dataset.csv"
        dataset.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")
        
        # Save as HDF5 for efficient storage
        h5_path = self.output_dir / "sofc_residual_stress_dataset.h5"
        with h5py.File(h5_path, 'w') as f:
            for col in dataset.columns:
                f.create_dataset(col, data=dataset[col].values)
        logger.info(f"Dataset saved to {h5_path}")
        
        # Save as JSON for metadata
        metadata = {
            'n_samples': len(dataset),
            'parameters': list(dataset.columns),
            'parameter_ranges': self.param_ranges,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'description': 'SOFC Residual Stress Prediction Dataset'
        }
        
        json_path = self.output_dir / "dataset_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {json_path}")
    
    def create_visualizations(self, dataset: pd.DataFrame):
        """Create comprehensive visualizations of the dataset"""
        logger.info("Creating dataset visualizations")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Parameter distribution plots
        self._plot_parameter_distributions(dataset)
        
        # 2. Stress correlation analysis
        self._plot_stress_correlations(dataset)
        
        # 3. Fracture risk analysis
        self._plot_fracture_risk_analysis(dataset)
        
        # 4. Process parameter effects
        self._plot_process_parameter_effects(dataset)
        
        # 5. Material property effects
        self._plot_material_property_effects(dataset)
        
        logger.info("Visualizations complete")
    
    def _plot_parameter_distributions(self, dataset: pd.DataFrame):
        """Plot distributions of input parameters"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        # Select key parameters for visualization
        key_params = [
            'plate_length', 'plate_width', 'anode_thickness', 'electrolyte_thickness',
            'cathode_thickness', 'interconnect_thickness', 'anode_E_25C', 'electrolyte_E_25C',
            'cathode_E_25C', 'interconnect_E_25C', 'anode_CTE_25C', 'electrolyte_CTE_25C',
            'cathode_CTE_25C', 'interconnect_CTE_25C', 'max_temperature', 'heating_rate'
        ]
        
        for i, param in enumerate(key_params):
            if i < len(axes) and param in dataset.columns:
                axes[i].hist(dataset[param], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{param} Distribution')
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(key_params), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stress_correlations(self, dataset: pd.DataFrame):
        """Plot stress correlation heatmap"""
        stress_cols = [col for col in dataset.columns if 'stress' in col.lower()]
        
        if len(stress_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = dataset[stress_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('Stress Component Correlations')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'stress_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_fracture_risk_analysis(self, dataset: pd.DataFrame):
        """Plot fracture risk analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Safety factor comparison
        axes[0, 0].scatter(dataset['safety_factor_elastic'], dataset['safety_factor_viscoelastic'], 
                          alpha=0.6, s=20)
        axes[0, 0].plot([0, 3], [0, 3], 'r--', label='Equal safety factors')
        axes[0, 0].set_xlabel('Elastic Safety Factor')
        axes[0, 0].set_ylabel('Viscoelastic Safety Factor')
        axes[0, 0].set_title('Safety Factor Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Fracture risk distribution
        axes[0, 1].hist(dataset['fracture_risk_elastic'], bins=30, alpha=0.7, 
                       label='Elastic', color='red')
        axes[0, 1].hist(dataset['fracture_risk_viscoelastic'], bins=30, alpha=0.7, 
                       label='Viscoelastic', color='blue')
        axes[0, 1].set_xlabel('Fracture Risk')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Fracture Risk Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Stress vs thickness relationship
        axes[1, 0].scatter(dataset['electrolyte_thickness'], dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[1, 0].scatter(dataset['electrolyte_thickness'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[1, 0].set_xlabel('Electrolyte Thickness (mm)')
        axes[1, 0].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 0].set_title('Stress vs Thickness')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # CTE mismatch effects
        cte_mismatch = (dataset['anode_CTE_25C'] - dataset['electrolyte_CTE_25C']) * 1e6
        axes[1, 1].scatter(cte_mismatch, dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[1, 1].scatter(cte_mismatch, dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[1, 1].set_xlabel('CTE Mismatch (ppm/K)')
        axes[1, 1].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 1].set_title('CTE Mismatch Effects')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fracture_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_process_parameter_effects(self, dataset: pd.DataFrame):
        """Plot effects of process parameters on stress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature effects
        axes[0, 0].scatter(dataset['max_temperature'], dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[0, 0].scatter(dataset['max_temperature'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[0, 0].set_xlabel('Max Temperature (°C)')
        axes[0, 0].set_ylabel('Max Principal Stress (MPa)')
        axes[0, 0].set_title('Temperature Effects')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cooling rate effects
        axes[0, 1].scatter(dataset['cooling_rate'], dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[0, 1].scatter(dataset['cooling_rate'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[0, 1].set_xlabel('Cooling Rate (°C/min)')
        axes[0, 1].set_ylabel('Max Principal Stress (MPa)')
        axes[0, 1].set_title('Cooling Rate Effects')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hold time effects
        axes[1, 0].scatter(dataset['hold_time'], dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[1, 0].scatter(dataset['hold_time'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[1, 0].set_xlabel('Hold Time (min)')
        axes[1, 0].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 0].set_title('Hold Time Effects')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Shrinkage rate effects
        avg_shrinkage = (dataset['anode_shrinkage_rate'] + dataset['electrolyte_shrinkage_rate'] + 
                        dataset['cathode_shrinkage_rate']) / 3
        axes[1, 1].scatter(avg_shrinkage, dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[1, 1].scatter(avg_shrinkage, dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[1, 1].set_xlabel('Average Shrinkage Rate')
        axes[1, 1].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 1].set_title('Shrinkage Rate Effects')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'process_parameter_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_material_property_effects(self, dataset: pd.DataFrame):
        """Plot effects of material properties on stress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Young's modulus effects
        axes[0, 0].scatter(dataset['electrolyte_E_25C'], dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[0, 0].scatter(dataset['electrolyte_E_25C'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[0, 0].set_xlabel('Electrolyte E (GPa)')
        axes[0, 0].set_ylabel('Max Principal Stress (MPa)')
        axes[0, 0].set_title('Young\'s Modulus Effects')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # CTE effects
        axes[0, 1].scatter(dataset['electrolyte_CTE_25C'] * 1e6, dataset['max_principal_stress_elastic'], 
                          alpha=0.6, s=20, label='Elastic')
        axes[0, 1].scatter(dataset['electrolyte_CTE_25C'] * 1e6, dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20, label='Viscoelastic')
        axes[0, 1].set_xlabel('Electrolyte CTE (ppm/K)')
        axes[0, 1].set_ylabel('Max Principal Stress (MPa)')
        axes[0, 1].set_title('CTE Effects')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Creep parameter effects
        axes[1, 0].scatter(dataset['electrolyte_creep_B'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Creep Parameter B (s⁻¹ MPa⁻ⁿ)')
        axes[1, 0].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 0].set_title('Creep Parameter B Effects')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Creep activation energy effects
        axes[1, 1].scatter(dataset['electrolyte_creep_Q'], dataset['max_principal_stress_viscoelastic'], 
                          alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Creep Activation Energy (kJ/mol)')
        axes[1, 1].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 1].set_title('Creep Activation Energy Effects')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'material_property_effects.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate the SOFC dataset"""
    logger.info("Starting SOFC Residual Stress Dataset Generation")
    
    # Initialize generator
    generator = SOFCDatasetGenerator(output_dir="sofc_dataset")
    
    # Generate dataset
    dataset = generator.generate_dataset(n_samples=2000, method='lhs', save_results=True)
    
    # Create visualizations
    generator.create_visualizations(dataset)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SOFC RESIDUAL STRESS DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"Parameters: {len(dataset.columns)}")
    print(f"Output directory: {generator.output_dir}")
    
    print("\nKey Statistics:")
    print(f"Max Principal Stress (Elastic): {dataset['max_principal_stress_elastic'].mean():.1f} ± {dataset['max_principal_stress_elastic'].std():.1f} MPa")
    print(f"Max Principal Stress (Viscoelastic): {dataset['max_principal_stress_viscoelastic'].mean():.1f} ± {dataset['max_principal_stress_viscoelastic'].std():.1f} MPa")
    print(f"Safety Factor (Elastic): {dataset['safety_factor_elastic'].mean():.2f} ± {dataset['safety_factor_elastic'].std():.2f}")
    print(f"Safety Factor (Viscoelastic): {dataset['safety_factor_viscoelastic'].mean():.2f} ± {dataset['safety_factor_viscoelastic'].std():.2f}")
    
    print(f"\nFracture Risk (Elastic): {dataset['fracture_risk_elastic'].mean():.3f} ± {dataset['fracture_risk_elastic'].std():.3f}")
    print(f"Fracture Risk (Viscoelastic): {dataset['fracture_risk_viscoelastic'].mean():.3f} ± {dataset['fracture_risk_viscoelastic'].std():.3f}")
    
    print(f"\nCreep Relaxation Factor: {dataset['creep_relaxation_factor'].mean():.3f} ± {dataset['creep_relaxation_factor'].std():.3f}")
    
    print("\nDataset generation complete!")
    print(f"Files saved in: {generator.output_dir}")

if __name__ == "__main__":
    main()