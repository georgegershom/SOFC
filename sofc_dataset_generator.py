"""
SOFC Warpage and Residual Stress Dataset Generator
===================================================

This script generates a synthetic "Ground Truth" dataset for ML-augmented 
inverse modeling of residual stress from warped SOFC plates.

The dataset simulates:
1. Manufacturing parameter variations (DOE)
2. Resulting warp fields (3D surface deformations)
3. Corresponding residual stress fields (3D stress tensors)

This synthetic data approximates what would be obtained from high-fidelity 
coupled thermo-mechanical FEA simulations.
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
from dataclasses import dataclass, asdict
from scipy.interpolate import RBFInterpolator
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ManufacturingParameters:
    """Manufacturing parameters that affect warp and stress."""
    # Sintering parameters
    peak_sintering_temp: float  # °C (1200-1600)
    heating_rate: float  # °C/min (1-10)
    cooling_rate: float  # °C/min (1-10)
    dwell_time: float  # hours (1-8)
    
    # Material parameters
    anode_thickness: float  # μm (300-800)
    electrolyte_thickness: float  # μm (5-30)
    cathode_thickness: float  # μm (20-80)
    
    # Composition variations (affects CTE mismatch)
    anode_ni_content: float  # wt% (40-60)
    electrolyte_ysz_dopant: float  # mol% Y2O3 (6-10)
    cathode_porosity: float  # % (20-45)
    
    # Green body properties
    green_density: float  # % theoretical (50-65)
    binder_content: float  # wt% (1-5)
    
    # Geometry
    plate_length: float  # mm (50-150)
    plate_width: float  # mm (50-150)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to normalized array for ML."""
        return np.array(list(asdict(self).values()))


class SOFCPhysicsSimulator:
    """
    Physics-informed synthetic data generator.
    Simulates the relationship between manufacturing parameters and resulting
    warp/stress fields using simplified physics-based models.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
        # Physical constants
        self.ROOM_TEMP = 25  # °C
        
        # Material property ranges (temperature dependent)
        self.material_props = {
            'anode': {
                'cte': 12.5e-6,  # K^-1 (NiO-YSZ)
                'youngs_modulus': 150e9,  # Pa at room temp
                'poisson': 0.3,
                'yield_strength': 200e6,  # Pa
            },
            'electrolyte': {
                'cte': 10.5e-6,  # K^-1 (YSZ)
                'youngs_modulus': 200e9,  # Pa
                'poisson': 0.32,
                'yield_strength': 300e6,  # Pa
            },
            'cathode': {
                'cte': 11.5e-6,  # K^-1 (LSM)
                'youngs_modulus': 80e9,  # Pa
                'poisson': 0.28,
                'yield_strength': 150e6,  # Pa
            }
        }
    
    def create_mesh_grid(self, length: float, width: float, 
                        nx: int = 50, ny: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D mesh grid for the SOFC plate."""
        x = np.linspace(0, length, nx)
        y = np.linspace(0, width, ny)
        return np.meshgrid(x, y)
    
    def compute_thermal_stress(self, params: ManufacturingParameters, 
                              x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute thermal stress based on CTE mismatch and thermal history.
        Simplified analytical model approximating FEA results.
        """
        # Temperature drop from sintering
        delta_T = params.peak_sintering_temp - self.ROOM_TEMP
        
        # CTE mismatch between layers
        cte_anode = self.material_props['anode']['cte'] * (1 + 0.1 * (params.anode_ni_content - 50) / 10)
        cte_electrolyte = self.material_props['electrolyte']['cte'] * (1 + 0.05 * (params.electrolyte_ysz_dopant - 8) / 2)
        cte_cathode = self.material_props['cathode']['cte'] * (1 - 0.1 * params.cathode_porosity / 35)
        
        # Effective CTE mismatch
        delta_cte_1 = cte_anode - cte_electrolyte
        delta_cte_2 = cte_cathode - cte_electrolyte
        
        # Thickness ratio effects (bending moment)
        t_a = params.anode_thickness * 1e-6  # Convert to m
        t_e = params.electrolyte_thickness * 1e-6
        t_c = params.cathode_thickness * 1e-6
        total_thickness = t_a + t_e + t_c
        
        # Neutral axis offset
        E_a = self.material_props['anode']['youngs_modulus']
        E_e = self.material_props['electrolyte']['youngs_modulus']
        E_c = self.material_props['cathode']['youngs_modulus']
        
        # Weighted average position
        z_a = t_a / 2
        z_e = t_a + t_e / 2
        z_c = t_a + t_e + t_c / 2
        
        neutral_axis = (E_a * t_a * z_a + E_e * t_e * z_e + E_c * t_c * z_c) / \
                      (E_a * t_a + E_e * t_e + E_c * t_c)
        
        # Thermal stress magnitude (simplified)
        # In-plane stresses
        sigma_thermal = E_e / (1 - self.material_props['electrolyte']['poisson']) * \
                       (delta_cte_1 + delta_cte_2) * delta_T / 2
        
        # Cooling rate effects (internal gradients)
        cooling_stress_factor = 1 + 0.2 * (params.cooling_rate - 5.5) / 4.5
        
        # Heating rate effects (residual green body stress)
        heating_stress_factor = 1 + 0.1 * (params.heating_rate - 5.5) / 4.5
        
        sigma_base = sigma_thermal * cooling_stress_factor * heating_stress_factor
        
        return {
            'sigma_base': sigma_base,
            'delta_cte': (delta_cte_1 + delta_cte_2) / 2,
            'neutral_axis': neutral_axis,
            'total_thickness': total_thickness
        }
    
    def generate_warp_field(self, params: ManufacturingParameters, 
                           nx: int = 50, ny: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate 3D warp field (surface deformation).
        Returns height maps for top and bottom surfaces.
        """
        # Create mesh
        X, Y = self.create_mesh_grid(params.plate_length * 1e-3, 
                                     params.plate_width * 1e-3, nx, ny)
        
        # Compute thermal stress parameters
        stress_params = self.compute_thermal_stress(params, X, Y)
        
        # Curvature from thermal stress (plate bending theory)
        # κ = M / (E * I) where M is bending moment
        
        # Effective modulus
        E_eff = self.material_props['electrolyte']['youngs_modulus']
        nu = self.material_props['electrolyte']['poisson']
        D = E_eff * stress_params['total_thickness']**3 / (12 * (1 - nu**2))
        
        # Bending moment from CTE mismatch
        M_thermal = stress_params['sigma_base'] * stress_params['total_thickness']**2 / 6
        
        # Curvature (simplified, assumes isotropic)
        kappa_x = M_thermal / D * (1 + 0.3 * np.random.randn())
        kappa_y = M_thermal / D * (1 + 0.3 * np.random.randn())
        
        # Add edge effects and non-uniformities
        L_x = params.plate_length * 1e-3
        L_y = params.plate_width * 1e-3
        
        # Normalized coordinates
        x_norm = X / L_x - 0.5
        y_norm = Y / L_y - 0.5
        
        # Base warp shape (saddle + dome)
        warp_base = kappa_x * x_norm**2 * L_x**2 + kappa_y * y_norm**2 * L_y**2
        
        # Add sintering inhomogeneity effects
        n_modes = 5
        warp_noise = np.zeros_like(X)
        for i in range(1, n_modes + 1):
            amplitude = 1e-4 / i**2 * (params.green_density - 57.5) / 7.5
            warp_noise += amplitude * np.sin(i * np.pi * x_norm) * np.cos(i * np.pi * y_norm)
        
        # Edge boundary effects (constrained during sintering)
        edge_x = np.exp(-10 * x_norm**2)
        edge_y = np.exp(-10 * y_norm**2)
        edge_constraint = 1 - 0.3 * (edge_x + edge_y)
        
        # Final warp field
        warp = (warp_base + warp_noise) * edge_constraint
        
        # Scale by plate size and material properties
        warp *= 1e3  # Convert to mm
        
        # Top and bottom surfaces (symmetric about neutral axis)
        t_total = stress_params['total_thickness']
        warp_top = warp * (t_total / 2) / t_total
        warp_bottom = -warp * (t_total / 2) / t_total
        
        return {
            'X': X,  # m
            'Y': Y,  # m
            'warp_top': warp_top,  # mm
            'warp_bottom': warp_bottom,  # mm
            'warp_mean': warp,  # mm
        }
    
    def generate_stress_field(self, params: ManufacturingParameters,
                            warp_data: Dict[str, np.ndarray],
                            nz: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate 3D residual stress field throughout the volume.
        Returns stress tensor components at each voxel.
        """
        nx, ny = warp_data['X'].shape
        
        # Create 3D grid
        t_a = params.anode_thickness * 1e-6
        t_e = params.electrolyte_thickness * 1e-6
        t_c = params.cathode_thickness * 1e-6
        t_total = t_a + t_e + t_c
        
        z = np.linspace(0, t_total, nz)
        
        # Initialize stress components
        sigma_xx = np.zeros((nx, ny, nz))
        sigma_yy = np.zeros((nx, ny, nz))
        sigma_zz = np.zeros((nx, ny, nz))
        sigma_xy = np.zeros((nx, ny, nz))
        sigma_yz = np.zeros((nx, ny, nz))
        sigma_xz = np.zeros((nx, ny, nz))
        
        # Compute base thermal stress
        stress_params = self.compute_thermal_stress(params, 
                                                    warp_data['X'], 
                                                    warp_data['Y'])
        
        # Through-thickness stress distribution
        for k, z_coord in enumerate(z):
            # Determine which layer we're in
            if z_coord < t_a:
                layer = 'anode'
                layer_progress = z_coord / t_a
            elif z_coord < t_a + t_e:
                layer = 'electrolyte'
                layer_progress = (z_coord - t_a) / t_e
            else:
                layer = 'cathode'
                layer_progress = (z_coord - t_a - t_e) / t_c
            
            E = self.material_props[layer]['youngs_modulus']
            nu = self.material_props[layer]['poisson']
            
            # Distance from neutral axis
            z_from_neutral = z_coord - stress_params['neutral_axis']
            
            # Bending stress (linear through thickness)
            sigma_bending = stress_params['sigma_base'] * z_from_neutral / (t_total / 2)
            
            # Membrane stress (uniform through thickness)
            sigma_membrane = stress_params['sigma_base'] * 0.3
            
            # In-plane stresses
            sigma_xx[:, :, k] = sigma_bending + sigma_membrane
            sigma_yy[:, :, k] = sigma_bending * 0.8 + sigma_membrane  # Slight anisotropy
            
            # Out-of-plane stress (much smaller)
            sigma_zz[:, :, k] = sigma_membrane * 0.1
            
            # Shear stresses (from non-uniform cooling)
            X_norm = warp_data['X'] / (params.plate_length * 1e-3) - 0.5
            Y_norm = warp_data['Y'] / (params.plate_width * 1e-3) - 0.5
            
            sigma_xy[:, :, k] = stress_params['sigma_base'] * 0.1 * X_norm * Y_norm
            sigma_xz[:, :, k] = stress_params['sigma_base'] * 0.05 * X_norm * z_from_neutral / t_total
            sigma_yz[:, :, k] = stress_params['sigma_base'] * 0.05 * Y_norm * z_from_neutral / t_total
            
            # Add layer interface effects
            if layer == 'electrolyte':
                # Higher stress concentration in electrolyte
                interface_factor = 1.5
                sigma_xx[:, :, k] *= interface_factor
                sigma_yy[:, :, k] *= interface_factor
            
            # Add some realistic noise
            noise_level = 0.05 * stress_params['sigma_base']
            sigma_xx[:, :, k] += noise_level * np.random.randn(nx, ny)
            sigma_yy[:, :, k] += noise_level * np.random.randn(nx, ny)
        
        return {
            'sigma_xx': sigma_xx,  # Pa
            'sigma_yy': sigma_yy,  # Pa
            'sigma_zz': sigma_zz,  # Pa
            'sigma_xy': sigma_xy,  # Pa
            'sigma_yz': sigma_yz,  # Pa
            'sigma_xz': sigma_xz,  # Pa
            'z_coords': z,  # m
        }


class DOEGenerator:
    """Design of Experiments generator for manufacturing parameters."""
    
    @staticmethod
    def latin_hypercube_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
        """Generate Latin Hypercube samples in [0, 1]^n_dims."""
        np.random.seed(seed)
        samples = np.zeros((n_samples, n_dims))
        
        for i in range(n_dims):
            samples[:, i] = (np.random.permutation(n_samples) + 
                           np.random.rand(n_samples)) / n_samples
        
        return samples
    
    @staticmethod
    def generate_doe_matrix(n_samples: int = 1000, seed: int = 42) -> List[ManufacturingParameters]:
        """
        Generate DOE matrix using Latin Hypercube Sampling.
        Ensures good coverage of the parameter space.
        """
        # Parameter ranges
        param_ranges = {
            'peak_sintering_temp': (1200, 1600),
            'heating_rate': (1, 10),
            'cooling_rate': (1, 10),
            'dwell_time': (1, 8),
            'anode_thickness': (300, 800),
            'electrolyte_thickness': (5, 30),
            'cathode_thickness': (20, 80),
            'anode_ni_content': (40, 60),
            'electrolyte_ysz_dopant': (6, 10),
            'cathode_porosity': (20, 45),
            'green_density': (50, 65),
            'binder_content': (1, 5),
            'plate_length': (50, 150),
            'plate_width': (50, 150),
        }
        
        n_dims = len(param_ranges)
        
        # Generate LHS samples
        lhs_samples = DOEGenerator.latin_hypercube_sampling(n_samples, n_dims, seed)
        
        # Scale to parameter ranges
        param_list = []
        param_names = list(param_ranges.keys())
        
        for sample in lhs_samples:
            params_dict = {}
            for i, name in enumerate(param_names):
                min_val, max_val = param_ranges[name]
                params_dict[name] = min_val + sample[i] * (max_val - min_val)
            
            param_list.append(ManufacturingParameters(**params_dict))
        
        return param_list


class DatasetGenerator:
    """Main dataset generator orchestrator."""
    
    def __init__(self, output_dir: str = "sofc_dataset", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.simulator = SOFCPhysicsSimulator(seed=seed)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "warp_fields").mkdir(exist_ok=True)
        (self.output_dir / "stress_fields").mkdir(exist_ok=True)
        (self.output_dir / "parameters").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
    
    def generate_dataset(self, n_samples: int = 1000, 
                        nx: int = 50, ny: int = 50, nz: int = 10,
                        save_format: str = 'npz') -> Dict:
        """
        Generate complete dataset.
        
        Parameters:
        -----------
        n_samples: Number of DOE samples to generate
        nx, ny: Spatial resolution for warp field (2D)
        nz: Through-thickness resolution for stress field (3D)
        save_format: 'npz' or 'pickle' or 'both'
        """
        print(f"Generating SOFC Dataset with {n_samples} samples...")
        print(f"Spatial resolution: {nx}x{ny}x{nz}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        # Generate DOE matrix
        print("Generating DOE matrix...")
        doe_matrix = DOEGenerator.generate_doe_matrix(n_samples, self.seed)
        
        # Generate data for each sample
        dataset_metadata = {
            'n_samples': n_samples,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'seed': self.seed,
            'samples': []
        }
        
        for i, params in enumerate(doe_matrix):
            if (i + 1) % 100 == 0:
                print(f"Processing sample {i + 1}/{n_samples}...")
            
            # Generate warp field
            warp_data = self.simulator.generate_warp_field(params, nx, ny)
            
            # Generate stress field
            stress_data = self.simulator.generate_stress_field(params, warp_data, nz)
            
            # Save data
            sample_id = f"sample_{i:05d}"
            
            # Save warp field
            warp_file = self.output_dir / "warp_fields" / f"{sample_id}_warp.npz"
            np.savez_compressed(
                warp_file,
                X=warp_data['X'],
                Y=warp_data['Y'],
                warp_top=warp_data['warp_top'],
                warp_bottom=warp_data['warp_bottom'],
                warp_mean=warp_data['warp_mean']
            )
            
            # Save stress field
            stress_file = self.output_dir / "stress_fields" / f"{sample_id}_stress.npz"
            np.savez_compressed(
                stress_file,
                sigma_xx=stress_data['sigma_xx'],
                sigma_yy=stress_data['sigma_yy'],
                sigma_zz=stress_data['sigma_zz'],
                sigma_xy=stress_data['sigma_xy'],
                sigma_yz=stress_data['sigma_yz'],
                sigma_xz=stress_data['sigma_xz'],
                z_coords=stress_data['z_coords']
            )
            
            # Save parameters
            param_file = self.output_dir / "parameters" / f"{sample_id}_params.json"
            with open(param_file, 'w') as f:
                json.dump(params.to_dict(), f, indent=2)
            
            # Update metadata
            dataset_metadata['samples'].append({
                'sample_id': sample_id,
                'warp_file': str(warp_file.relative_to(self.output_dir)),
                'stress_file': str(stress_file.relative_to(self.output_dir)),
                'param_file': str(param_file.relative_to(self.output_dir)),
                'parameters': params.to_dict()
            })
        
        # Save master metadata
        metadata_file = self.output_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print("-" * 60)
        print(f"Dataset generation complete!")
        print(f"Total samples: {n_samples}")
        print(f"Metadata saved to: {metadata_file}")
        
        return dataset_metadata
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for the dataset."""
        metadata_file = self.output_dir / "metadata" / "dataset_metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Compute statistics
        all_params = np.array([s['parameters'] for s in metadata['samples']])
        
        # Load a few samples to get stress/warp statistics
        n_samples_to_check = min(100, len(metadata['samples']))
        warp_magnitudes = []
        stress_magnitudes = []
        
        for i in range(n_samples_to_check):
            sample = metadata['samples'][i]
            
            # Load warp data
            warp_file = self.output_dir / sample['warp_file']
            warp_data = np.load(warp_file)
            warp_magnitudes.append(np.max(np.abs(warp_data['warp_mean'])))
            
            # Load stress data
            stress_file = self.output_dir / sample['stress_file']
            stress_data = np.load(stress_file)
            # Von Mises stress
            sigma_vm = np.sqrt(0.5 * (
                (stress_data['sigma_xx'] - stress_data['sigma_yy'])**2 +
                (stress_data['sigma_yy'] - stress_data['sigma_zz'])**2 +
                (stress_data['sigma_zz'] - stress_data['sigma_xx'])**2 +
                6 * (stress_data['sigma_xy']**2 + 
                     stress_data['sigma_yz']**2 + 
                     stress_data['sigma_xz']**2)
            ))
            stress_magnitudes.append(np.max(sigma_vm) / 1e6)  # Convert to MPa
        
        summary = {
            'n_samples': len(metadata['samples']),
            'warp_statistics': {
                'mean_max_warp_mm': float(np.mean(warp_magnitudes)),
                'std_max_warp_mm': float(np.std(warp_magnitudes)),
                'min_max_warp_mm': float(np.min(warp_magnitudes)),
                'max_max_warp_mm': float(np.max(warp_magnitudes)),
            },
            'stress_statistics': {
                'mean_max_stress_MPa': float(np.mean(stress_magnitudes)),
                'std_max_stress_MPa': float(np.std(stress_magnitudes)),
                'min_max_stress_MPa': float(np.min(stress_magnitudes)),
                'max_max_stress_MPa': float(np.max(stress_magnitudes)),
            }
        }
        
        # Save summary
        summary_file = self.output_dir / "metadata" / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def main():
    """Main entry point for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic SOFC warp and stress dataset'
    )
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to generate (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='sofc_dataset',
                       help='Output directory (default: sofc_dataset)')
    parser.add_argument('--nx', type=int, default=50,
                       help='X resolution for warp field (default: 50)')
    parser.add_argument('--ny', type=int, default=50,
                       help='Y resolution for warp field (default: 50)')
    parser.add_argument('--nz', type=int, default=10,
                       help='Z resolution for stress field (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = DatasetGenerator(output_dir=args.output_dir, seed=args.seed)
    
    # Generate dataset
    metadata = generator.generate_dataset(
        n_samples=args.n_samples,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz
    )
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary = generator.generate_summary_statistics()
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary['n_samples']}")
    print(f"\nWarp Statistics (mm):")
    print(f"  Mean max warp: {summary['warp_statistics']['mean_max_warp_mm']:.4f}")
    print(f"  Std max warp:  {summary['warp_statistics']['std_max_warp_mm']:.4f}")
    print(f"  Range: [{summary['warp_statistics']['min_max_warp_mm']:.4f}, "
          f"{summary['warp_statistics']['max_max_warp_mm']:.4f}]")
    print(f"\nStress Statistics (MPa):")
    print(f"  Mean max stress: {summary['stress_statistics']['mean_max_stress_MPa']:.2f}")
    print(f"  Std max stress:  {summary['stress_statistics']['std_max_stress_MPa']:.2f}")
    print(f"  Range: [{summary['stress_statistics']['min_max_stress_MPa']:.2f}, "
          f"{summary['stress_statistics']['max_max_stress_MPa']:.2f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
