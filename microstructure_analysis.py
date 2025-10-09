"""
Advanced Analysis and Visualization Tools for 3D SOFC Microstructure Data

This module provides comprehensive analysis capabilities including:
- Phase connectivity analysis
- Interface characterization
- Pore network analysis
- Mechanical property estimation
- Interactive visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.filters import gaussian
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import h5py
import pyvista as pv
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MicrostructureAnalyzer:
    """
    Comprehensive analyzer for 3D SOFC microstructure data.
    """
    
    def __init__(self, microstructure: np.ndarray, voxel_size: float = 0.1):
        """
        Initialize analyzer with microstructure data.
        
        Parameters:
        -----------
        microstructure : np.ndarray
            3D array with phase labels
        voxel_size : float
            Size of each voxel in micrometers
        """
        self.microstructure = microstructure
        self.voxel_size = voxel_size
        self.resolution = microstructure.shape
        
        # Phase labels
        self.PORE = 0
        self.NI = 1
        self.YSZ_ANODE = 2
        self.YSZ_ELECTROLYTE = 3
        self.INTERLAYER = 4
        
        self.phase_names = {
            self.PORE: 'Pore',
            self.NI: 'Ni',
            self.YSZ_ANODE: 'YSZ_Anode',
            self.YSZ_ELECTROLYTE: 'YSZ_Electrolyte',
            self.INTERLAYER: 'Interlayer'
        }
        
        # Analysis results cache
        self._analysis_cache = {}
    
    def analyze_phase_connectivity(self) -> Dict[str, Dict]:
        """
        Analyze connectivity of each phase using 3D labeling.
        """
        print("Analyzing phase connectivity...")
        
        connectivity_results = {}
        
        for phase_id, phase_name in self.phase_names.items():
            if phase_id == self.PORE:
                continue  # Skip pore connectivity for now
            
            phase_mask = (self.microstructure == phase_id)
            
            if not np.any(phase_mask):
                connectivity_results[phase_name] = {
                    'n_components': 0,
                    'largest_component_size': 0,
                    'connectivity_percentage': 0.0
                }
                continue
            
            # Label connected components
            labeled, n_components = measure.label(phase_mask, return_num=True)
            
            # Calculate component sizes
            component_sizes = []
            for i in range(1, n_components + 1):
                size = np.sum(labeled == i)
                component_sizes.append(size)
            
            largest_component_size = max(component_sizes) if component_sizes else 0
            total_phase_voxels = np.sum(phase_mask)
            connectivity_percentage = (largest_component_size / total_phase_voxels) * 100
            
            connectivity_results[phase_name] = {
                'n_components': n_components,
                'largest_component_size': largest_component_size,
                'connectivity_percentage': connectivity_percentage,
                'component_sizes': component_sizes
            }
        
        self._analysis_cache['connectivity'] = connectivity_results
        return connectivity_results
    
    def analyze_pore_network(self) -> Dict[str, float]:
        """
        Analyze pore network properties including tortuosity and connectivity.
        """
        print("Analyzing pore network...")
        
        pore_mask = (self.microstructure == self.PORE)
        
        if not np.any(pore_mask):
            return {'tortuosity': 0, 'connectivity': 0, 'pore_size_distribution': []}
        
        # Calculate pore size distribution
        labeled_pores, n_pores = measure.label(pore_mask, return_num=True)
        pore_sizes = []
        
        for i in range(1, n_pores + 1):
            pore_size = np.sum(labeled_pores == i)
            pore_sizes.append(pore_size * (self.voxel_size ** 3))  # Convert to μm³
        
        # Calculate tortuosity (simplified approach)
        tortuosity = self._calculate_tortuosity(pore_mask)
        
        # Calculate connectivity
        connectivity = self._calculate_pore_connectivity(pore_mask)
        
        pore_network_results = {
            'tortuosity': tortuosity,
            'connectivity': connectivity,
            'n_pores': n_pores,
            'pore_size_distribution': pore_sizes,
            'mean_pore_size': np.mean(pore_sizes) if pore_sizes else 0,
            'std_pore_size': np.std(pore_sizes) if pore_sizes else 0
        }
        
        self._analysis_cache['pore_network'] = pore_network_results
        return pore_network_results
    
    def _calculate_tortuosity(self, pore_mask: np.ndarray) -> float:
        """Calculate tortuosity of pore network."""
        # Simplified tortuosity calculation
        # In practice, this would involve more sophisticated path finding
        
        # Find pore paths in z-direction
        z_paths = []
        for x in range(self.resolution[0]):
            for y in range(self.resolution[1]):
                z_profile = pore_mask[x, y, :]
                if np.any(z_profile):
                    # Find continuous pore segments
                    diff = np.diff(np.concatenate(([False], z_profile, [False])).astype(int))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        if end - start > 1:  # Minimum path length
                            z_paths.append(end - start)
        
        if not z_paths:
            return 0
        
        # Tortuosity is the ratio of actual path length to straight-line distance
        straight_line_distance = self.resolution[2]
        actual_path_length = np.mean(z_paths)
        
        return actual_path_length / straight_line_distance if straight_line_distance > 0 else 0
    
    def _calculate_pore_connectivity(self, pore_mask: np.ndarray) -> float:
        """Calculate pore connectivity percentage."""
        # Label connected pore components
        labeled, n_components = measure.label(pore_mask, return_num=True)
        
        if n_components == 0:
            return 0
        
        # Calculate size of each component
        component_sizes = []
        for i in range(1, n_components + 1):
            size = np.sum(labeled == i)
            component_sizes.append(size)
        
        # Connectivity is the percentage of pores in the largest connected component
        largest_component_size = max(component_sizes)
        total_pore_voxels = np.sum(pore_mask)
        
        return (largest_component_size / total_pore_voxels) * 100 if total_pore_voxels > 0 else 0
    
    def analyze_interface_properties(self) -> Dict[str, Dict]:
        """
        Analyze interface properties between different phases.
        """
        print("Analyzing interface properties...")
        
        interface_results = {}
        
        # Anode/Electrolyte interface
        anode_mask = (self.microstructure == self.NI) | (self.microstructure == self.YSZ_ANODE)
        electrolyte_mask = (self.microstructure == self.YSZ_ELECTROLYTE)
        
        if np.any(anode_mask) and np.any(electrolyte_mask):
            interface_area = self._calculate_interface_area(anode_mask, electrolyte_mask)
            interface_roughness = self._calculate_interface_roughness(anode_mask, electrolyte_mask)
            
            interface_results['Anode_Electrolyte'] = {
                'area_um2': interface_area,
                'roughness_um': interface_roughness,
                'area_fraction': interface_area / (self.resolution[0] * self.resolution[1] * (self.voxel_size ** 2))
            }
        
        # Pore/Solid interfaces
        pore_mask = (self.microstructure == self.PORE)
        solid_mask = ~pore_mask
        
        if np.any(pore_mask) and np.any(solid_mask):
            interface_area = self._calculate_interface_area(pore_mask, solid_mask)
            
            interface_results['Pore_Solid'] = {
                'area_um2': interface_area,
                'area_fraction': interface_area / (self.resolution[0] * self.resolution[1] * (self.voxel_size ** 2))
            }
        
        self._analysis_cache['interfaces'] = interface_results
        return interface_results
    
    def _calculate_interface_area(self, phase1_mask: np.ndarray, phase2_mask: np.ndarray) -> float:
        """Calculate interface area between two phases."""
        # Find interface voxels
        phase1_dilated = morphology.binary_dilation(phase1_mask, morphology.ball(1))
        phase2_dilated = morphology.binary_dilation(phase2_mask, morphology.ball(1))
        
        interface_mask = phase1_dilated & phase2_dilated
        interface_voxels = np.sum(interface_mask)
        
        # Convert to area (assuming interface is roughly perpendicular to z-axis)
        return interface_voxels * (self.voxel_size ** 2)
    
    def _calculate_interface_roughness(self, phase1_mask: np.ndarray, phase2_mask: np.ndarray) -> float:
        """Calculate interface roughness."""
        # Find interface points
        phase1_dilated = morphology.binary_dilation(phase1_mask, morphology.ball(1))
        phase2_dilated = morphology.binary_dilation(phase2_mask, morphology.ball(1))
        
        interface_mask = phase1_dilated & phase2_dilated
        
        # Calculate roughness as standard deviation of interface height
        interface_points = np.where(interface_mask)
        if len(interface_points[0]) == 0:
            return 0
        
        # For simplicity, use z-coordinate as height
        z_coords = interface_points[2]
        roughness = np.std(z_coords) * self.voxel_size
        
        return roughness
    
    def estimate_mechanical_properties(self) -> Dict[str, float]:
        """
        Estimate mechanical properties based on microstructure.
        """
        print("Estimating mechanical properties...")
        
        # Calculate volume fractions
        total_voxels = np.prod(self.resolution)
        volume_fractions = {}
        
        for phase_id, phase_name in self.phase_names.items():
            count = np.sum(self.microstructure == phase_id)
            volume_fractions[phase_name] = count / total_voxels
        
        # Material properties (typical values)
        material_properties = {
            'Pore': {'E': 0, 'nu': 0, 'density': 0},
            'Ni': {'E': 200e9, 'nu': 0.31, 'density': 8.9e3},  # Pa, kg/m³
            'YSZ_Anode': {'E': 200e9, 'nu': 0.31, 'density': 6.1e3},
            'YSZ_Electrolyte': {'E': 200e9, 'nu': 0.31, 'density': 6.1e3},
            'Interlayer': {'E': 150e9, 'nu': 0.32, 'density': 5.5e3}
        }
        
        # Rule of mixtures for effective properties
        effective_E = 0
        effective_nu = 0
        effective_density = 0
        
        for phase_name, vf in volume_fractions.items():
            if phase_name in material_properties:
                props = material_properties[phase_name]
                effective_E += vf * props['E']
                effective_nu += vf * props['nu']
                effective_density += vf * props['density']
        
        # Calculate additional properties
        porosity = volume_fractions.get('Pore', 0)
        
        mechanical_properties = {
            'effective_youngs_modulus_Pa': effective_E,
            'effective_poisson_ratio': effective_nu,
            'effective_density_kg_m3': effective_density,
            'porosity': porosity,
            'relative_density': 1 - porosity
        }
        
        self._analysis_cache['mechanical'] = mechanical_properties
        return mechanical_properties
    
    def create_comprehensive_report(self) -> pd.DataFrame:
        """Create comprehensive analysis report."""
        print("Creating comprehensive analysis report...")
        
        # Run all analyses
        connectivity = self.analyze_phase_connectivity()
        pore_network = self.analyze_pore_network()
        interfaces = self.analyze_interface_properties()
        mechanical = self.estimate_mechanical_properties()
        
        # Create report
        report_data = []
        
        # Phase information
        total_voxels = np.prod(self.resolution)
        for phase_id, phase_name in self.phase_names.items():
            count = np.sum(self.microstructure == phase_id)
            volume_fraction = count / total_voxels
            
            report_data.append({
                'Category': 'Phase_Properties',
                'Property': f'{phase_name}_Volume_Fraction',
                'Value': volume_fraction,
                'Unit': 'dimensionless'
            })
            
            report_data.append({
                'Category': 'Phase_Properties',
                'Property': f'{phase_name}_Count',
                'Value': count,
                'Unit': 'voxels'
            })
        
        # Connectivity information
        for phase_name, props in connectivity.items():
            report_data.append({
                'Category': 'Connectivity',
                'Property': f'{phase_name}_Components',
                'Value': props['n_components'],
                'Unit': 'count'
            })
            
            report_data.append({
                'Category': 'Connectivity',
                'Property': f'{phase_name}_Connectivity_Percentage',
                'Value': props['connectivity_percentage'],
                'Unit': '%'
            })
        
        # Pore network information
        report_data.append({
            'Category': 'Pore_Network',
            'Property': 'Tortuosity',
            'Value': pore_network['tortuosity'],
            'Unit': 'dimensionless'
        })
        
        report_data.append({
            'Category': 'Pore_Network',
            'Property': 'Connectivity',
            'Value': pore_network['connectivity'],
            'Unit': '%'
        })
        
        report_data.append({
            'Category': 'Pore_Network',
            'Property': 'Mean_Pore_Size',
            'Value': pore_network['mean_pore_size'],
            'Unit': 'μm³'
        })
        
        # Interface information
        for interface_name, props in interfaces.items():
            report_data.append({
                'Category': 'Interfaces',
                'Property': f'{interface_name}_Area',
                'Value': props['area_um2'],
                'Unit': 'μm²'
            })
        
        # Mechanical properties
        for prop_name, value in mechanical.items():
            unit = 'Pa' if 'modulus' in prop_name else 'kg/m³' if 'density' in prop_name else 'dimensionless'
            report_data.append({
                'Category': 'Mechanical',
                'Property': prop_name,
                'Value': value,
                'Unit': unit
            })
        
        return pd.DataFrame(report_data)
    
    def create_interactive_dashboard(self) -> dash.Dash:
        """Create interactive dashboard for microstructure analysis."""
        print("Creating interactive dashboard...")
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("SOFC Microstructure Analysis Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Phase Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="phase-distribution")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Connectivity Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="connectivity-plot")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Pore Network Properties"),
                        dbc.CardBody([
                            dcc.Graph(id="pore-network-plot")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Mechanical Properties"),
                        dbc.CardBody([
                            dcc.Graph(id="mechanical-properties")
                        ])
                    ])
                ], width=6)
            ])
        ])
        
        @app.callback(
            [Output("phase-distribution", "figure"),
             Output("connectivity-plot", "figure"),
             Output("pore-network-plot", "figure"),
             Output("mechanical-properties", "figure")],
            [Input("phase-distribution", "id")]
        )
        def update_plots(_):
            # Phase distribution pie chart
            total_voxels = np.prod(self.resolution)
            phase_counts = {}
            for phase_id, phase_name in self.phase_names.items():
                count = np.sum(self.microstructure == phase_id)
                phase_counts[phase_name] = count
            
            phase_fig = px.pie(
                values=list(phase_counts.values()),
                names=list(phase_counts.keys()),
                title="Phase Distribution"
            )
            
            # Connectivity bar chart
            connectivity = self.analyze_phase_connectivity()
            connectivity_fig = px.bar(
                x=list(connectivity.keys()),
                y=[props['connectivity_percentage'] for props in connectivity.values()],
                title="Phase Connectivity"
            )
            connectivity_fig.update_layout(xaxis_title="Phase", yaxis_title="Connectivity (%)")
            
            # Pore network analysis
            pore_network = self.analyze_pore_network()
            pore_fig = go.Figure()
            pore_fig.add_trace(go.Histogram(
                x=pore_network['pore_size_distribution'],
                name="Pore Size Distribution"
            ))
            pore_fig.update_layout(
                title="Pore Size Distribution",
                xaxis_title="Pore Size (μm³)",
                yaxis_title="Count"
            )
            
            # Mechanical properties
            mechanical = self.estimate_mechanical_properties()
            mech_fig = px.bar(
                x=list(mechanical.keys()),
                y=list(mechanical.values()),
                title="Mechanical Properties"
            )
            mech_fig.update_layout(xaxis_title="Property", yaxis_title="Value")
            
            return phase_fig, connectivity_fig, pore_fig, mech_fig
        
        return app
    
    def save_analysis_results(self, filename: str):
        """Save analysis results to file."""
        print(f"Saving analysis results to {filename}...")
        
        report = self.create_comprehensive_report()
        report.to_csv(filename, index=False)
        
        # Also save as HDF5 for more detailed data
        h5_filename = filename.replace('.csv', '.h5')
        with h5py.File(h5_filename, 'w') as f:
            # Save microstructure
            f.create_dataset('microstructure', data=self.microstructure, compression='gzip')
            
            # Save analysis results
            analysis_group = f.create_group('analysis')
            
            if 'connectivity' in self._analysis_cache:
                conn_group = analysis_group.create_group('connectivity')
                for phase, props in self._analysis_cache['connectivity'].items():
                    phase_group = conn_group.create_group(phase)
                    for key, value in props.items():
                        if isinstance(value, list):
                            phase_group.create_dataset(key, data=value)
                        else:
                            phase_group.attrs[key] = value
            
            if 'pore_network' in self._analysis_cache:
                pore_group = analysis_group.create_group('pore_network')
                for key, value in self._analysis_cache['pore_network'].items():
                    if isinstance(value, list):
                        pore_group.create_dataset(key, data=value)
                    else:
                        pore_group.attrs[key] = value
            
            if 'interfaces' in self._analysis_cache:
                interface_group = analysis_group.create_group('interfaces')
                for interface, props in self._analysis_cache['interfaces'].items():
                    int_group = interface_group.create_group(interface)
                    for key, value in props.items():
                        int_group.attrs[key] = value
            
            if 'mechanical' in self._analysis_cache:
                mech_group = analysis_group.create_group('mechanical')
                for key, value in self._analysis_cache['mechanical'].items():
                    mech_group.attrs[key] = value


def main():
    """Main function for analysis."""
    print("Starting microstructure analysis...")
    
    # Load microstructure data
    try:
        with h5py.File('output/sofc_microstructure.h5', 'r') as f:
            microstructure = f['microstructure'][:]
            voxel_size = f.attrs['voxel_size_um']
    except FileNotFoundError:
        print("Microstructure data not found. Please run microstructure_generator.py first.")
        return
    
    # Create analyzer
    analyzer = MicrostructureAnalyzer(microstructure, voxel_size)
    
    # Run comprehensive analysis
    print("Running comprehensive analysis...")
    report = analyzer.create_comprehensive_report()
    
    # Save results
    analyzer.save_analysis_results('output/microstructure_analysis.csv')
    
    # Create and save plots
    print("Creating visualization plots...")
    
    # Phase distribution
    plt.figure(figsize=(10, 6))
    total_voxels = np.prod(microstructure.shape)
    phase_counts = {}
    phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
    
    for phase_id, phase_name in phase_names.items():
        count = np.sum(microstructure == phase_id)
        phase_counts[phase_name] = count / total_voxels
    
    plt.pie(phase_counts.values(), labels=phase_counts.keys(), autopct='%1.1f%%')
    plt.title('Phase Distribution')
    plt.savefig('output/phase_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Connectivity analysis
    connectivity = analyzer.analyze_phase_connectivity()
    plt.figure(figsize=(10, 6))
    phases = list(connectivity.keys())
    connectivity_values = [props['connectivity_percentage'] for props in connectivity.values()]
    
    plt.bar(phases, connectivity_values)
    plt.title('Phase Connectivity')
    plt.xlabel('Phase')
    plt.ylabel('Connectivity (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/connectivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("Analysis results saved to 'output/' directory:")
    print("  - microstructure_analysis.csv (Detailed report)")
    print("  - microstructure_analysis.h5 (HDF5 format)")
    print("  - phase_distribution.png (Phase distribution plot)")
    print("  - connectivity_analysis.png (Connectivity plot)")


if __name__ == "__main__":
    main()