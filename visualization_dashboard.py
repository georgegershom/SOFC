"""
Interactive Visualization Dashboard for SOFC Microstructure Data

This module provides comprehensive visualization capabilities including:
- 3D interactive plots
- Cross-sectional analysis
- Phase distribution visualization
- Interface analysis
- Real-time parameter adjustment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pyvista as pv
import h5py
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MicrostructureVisualizer:
    """
    Comprehensive visualizer for 3D SOFC microstructure data.
    """
    
    def __init__(self, microstructure: np.ndarray, voxel_size: float = 0.1):
        """
        Initialize visualizer.
        
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
        
        # Phase labels and colors
        self.phase_labels = {
            0: 'Pore',
            1: 'Ni',
            2: 'YSZ_Anode',
            3: 'YSZ_Electrolyte',
            4: 'Interlayer'
        }
        
        self.phase_colors = {
            0: 'white',
            1: 'gold',
            2: 'lightblue',
            3: 'darkblue',
            4: 'red'
        }
        
        self.plotly_colors = {
            0: 'rgba(255, 255, 255, 0.8)',
            1: 'rgba(255, 215, 0, 0.8)',
            2: 'rgba(173, 216, 230, 0.8)',
            3: 'rgba(0, 0, 139, 0.8)',
            4: 'rgba(255, 0, 0, 0.8)'
        }
    
    def create_3d_plotly_visualization(self, 
                                     max_voxels: int = 50000,
                                     opacity: float = 0.7) -> go.Figure:
        """
        Create 3D Plotly visualization of microstructure.
        
        Parameters:
        -----------
        max_voxels : int
            Maximum number of voxels to display
        opacity : float
            Opacity of voxels (0-1)
        
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        print("Creating 3D Plotly visualization...")
        
        # Subsample if too many voxels
        if np.prod(self.resolution) > max_voxels:
            step = int(np.ceil(np.prod(self.resolution) / max_voxels) ** (1/3))
            microstructure_sub = self.microstructure[::step, ::step, ::step]
            voxel_size_sub = self.voxel_size * step
        else:
            microstructure_sub = self.microstructure
            voxel_size_sub = self.voxel_size
        
        fig = go.Figure()
        
        # Add each phase as separate trace
        for phase_id, phase_name in self.phase_labels.items():
            phase_mask = (microstructure_sub == phase_id)
            
            if not np.any(phase_mask):
                continue
            
            # Get coordinates of phase voxels
            coords = np.where(phase_mask)
            x_coords = coords[0] * voxel_size_sub
            y_coords = coords[1] * voxel_size_sub
            z_coords = coords[2] * voxel_size_sub
            
            # Add scatter3d trace
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.plotly_colors[phase_id],
                    opacity=opacity
                ),
                name=phase_name,
                text=[phase_name] * len(x_coords),
                hovertemplate='<b>%{text}</b><br>' +
                            'X: %{x:.2f} μm<br>' +
                            'Y: %{y:.2f} μm<br>' +
                            'Z: %{z:.2f} μm<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='3D SOFC Microstructure Visualization',
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_cross_section_plots(self, 
                                 z_slices: List[int] = None,
                                 n_slices: int = 4) -> go.Figure:
        """
        Create cross-sectional plots at different z-positions.
        
        Parameters:
        -----------
        z_slices : List[int], optional
            Specific z-slices to plot
        n_slices : int
            Number of slices if z_slices not provided
        
        Returns:
        --------
        go.Figure
            Plotly figure with subplots
        """
        print("Creating cross-sectional plots...")
        
        if z_slices is None:
            z_slices = np.linspace(0, self.resolution[2]-1, n_slices, dtype=int)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'Z = {z * self.voxel_size:.1f} μm' for z in z_slices],
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, z in enumerate(z_slices):
            if i >= 4:  # Limit to 4 subplots
                break
            
            # Get slice data
            slice_data = self.microstructure[:, :, z]
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=slice_data,
                colorscale='viridis',
                showscale=False,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Phase: %{z}<extra></extra>'
            )
            
            fig.add_trace(heatmap, row=positions[i][0], col=positions[i][1])
        
        # Update layout
        fig.update_layout(
            title='Cross-Sectional Views of SOFC Microstructure',
            height=600,
            width=800
        )
        
        return fig
    
    def create_phase_distribution_plots(self) -> go.Figure:
        """
        Create phase distribution plots.
        
        Returns:
        --------
        go.Figure
            Plotly figure with phase distribution
        """
        print("Creating phase distribution plots...")
        
        # Calculate phase counts and fractions
        total_voxels = np.prod(self.resolution)
        phase_counts = {}
        phase_fractions = {}
        
        for phase_id, phase_name in self.phase_labels.items():
            count = np.sum(self.microstructure == phase_id)
            phase_counts[phase_name] = count
            phase_fractions[phase_name] = count / total_voxels
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Phase Distribution (Count)', 'Phase Distribution (Fraction)'],
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=list(phase_counts.keys()),
                y=list(phase_counts.values()),
                name='Count',
                marker_color=[self.plotly_colors[i] for i in range(len(phase_counts))]
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=list(phase_fractions.keys()),
                values=list(phase_fractions.values()),
                name='Fraction',
                marker_colors=[self.plotly_colors[i] for i in range(len(phase_fractions))]
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Phase Distribution Analysis',
            height=400,
            width=800
        )
        
        return fig
    
    def create_interface_analysis_plot(self) -> go.Figure:
        """
        Create interface analysis plot.
        
        Returns:
        --------
        go.Figure
            Plotly figure with interface analysis
        """
        print("Creating interface analysis plot...")
        
        # Find interfaces
        anode_mask = (self.microstructure == 1) | (self.microstructure == 2)
        electrolyte_mask = (self.microstructure == 3)
        
        # Calculate interface area
        from skimage import morphology
        anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(1))
        electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(1))
        interface_mask = anode_dilated & electrolyte_dilated
        
        interface_area = np.sum(interface_mask) * (self.voxel_size ** 2)
        
        # Create visualization
        fig = go.Figure()
        
        # Add interface points
        interface_coords = np.where(interface_mask)
        if len(interface_coords[0]) > 0:
            fig.add_trace(go.Scatter3d(
                x=interface_coords[0] * self.voxel_size,
                y=interface_coords[1] * self.voxel_size,
                z=interface_coords[2] * self.voxel_size,
                mode='markers',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.8
                ),
                name='Interface',
                hovertemplate='Interface Point<br>X: %{x:.2f} μm<br>Y: %{y:.2f} μm<br>Z: %{z:.2f} μm<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Interface Analysis (Area: {interface_area:.2f} μm²)',
            scene=dict(
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                zaxis_title='Z (μm)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_interactive_dashboard(self) -> dash.Dash:
        """
        Create interactive dashboard.
        
        Returns:
        --------
        dash.Dash
            Interactive dashboard app
        """
        print("Creating interactive dashboard...")
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("SOFC Microstructure Analysis Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Visualization Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Max Voxels:"),
                                    dcc.Slider(
                                        id='max-voxels-slider',
                                        min=1000,
                                        max=100000,
                                        step=1000,
                                        value=50000,
                                        marks={i: str(i) for i in [1000, 25000, 50000, 75000, 100000]}
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Opacity:"),
                                    dcc.Slider(
                                        id='opacity-slider',
                                        min=0.1,
                                        max=1.0,
                                        step=0.1,
                                        value=0.7,
                                        marks={i/10: f'{i/10:.1f}' for i in range(1, 11)}
                                    )
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Z-Slice:"),
                                    dcc.Slider(
                                        id='z-slice-slider',
                                        min=0,
                                        max=self.resolution[2]-1,
                                        step=1,
                                        value=self.resolution[2]//2,
                                        marks={i: str(i) for i in range(0, self.resolution[2], self.resolution[2]//4)}
                                    )
                                ], width=12)
                            ])
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Microstructure Information"),
                        dbc.CardBody([
                            html.P(f"Resolution: {self.resolution[0]} × {self.resolution[1]} × {self.resolution[2]}"),
                            html.P(f"Voxel Size: {self.voxel_size} μm"),
                            html.P(f"Total Volume: {np.prod(self.resolution) * (self.voxel_size**3):.2f} μm³"),
                            html.P(f"Total Voxels: {np.prod(self.resolution):,}")
                        ])
                    ])
                ], width=8)
            ], className="mb-4"),
            
            # Main visualization
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("3D Microstructure Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="3d-plot")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Cross-sections and analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cross-Sectional Views"),
                        dbc.CardBody([
                            dcc.Graph(id="cross-section-plot")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Phase Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="phase-distribution-plot")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Interface analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Interface Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="interface-plot")
                        ])
                    ])
                ], width=12)
            ])
        ])
        
        # Callbacks
        @app.callback(
            Output("3d-plot", "figure"),
            [Input("max-voxels-slider", "value"),
             Input("opacity-slider", "value")]
        )
        def update_3d_plot(max_voxels, opacity):
            return self.create_3d_plotly_visualization(max_voxels, opacity)
        
        @app.callback(
            Output("cross-section-plot", "figure"),
            [Input("z-slice-slider", "value")]
        )
        def update_cross_section(z_slice):
            z_slices = [z_slice - 10, z_slice - 5, z_slice, z_slice + 5]
            z_slices = [max(0, min(z, self.resolution[2]-1)) for z in z_slices]
            return self.create_cross_section_plots(z_slices)
        
        @app.callback(
            Output("phase-distribution-plot", "figure"),
            [Input("3d-plot", "id")]
        )
        def update_phase_distribution(_):
            return self.create_phase_distribution_plots()
        
        @app.callback(
            Output("interface-plot", "figure"),
            [Input("3d-plot", "id")]
        )
        def update_interface_plot(_):
            return self.create_interface_analysis_plot()
        
        return app
    
    def save_static_plots(self, output_dir: str = 'output/plots'):
        """
        Save static plots to files.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for plots
        """
        print(f"Saving static plots to {output_dir}...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 3D visualization
        fig_3d = self.create_3d_plotly_visualization()
        fig_3d.write_html(f"{output_dir}/3d_visualization.html")
        fig_3d.write_image(f"{output_dir}/3d_visualization.png", width=1200, height=900)
        
        # Cross-sections
        fig_cross = self.create_cross_section_plots()
        fig_cross.write_html(f"{output_dir}/cross_sections.html")
        fig_cross.write_image(f"{output_dir}/cross_sections.png", width=1200, height=900)
        
        # Phase distribution
        fig_phase = self.create_phase_distribution_plots()
        fig_phase.write_html(f"{output_dir}/phase_distribution.html")
        fig_phase.write_image(f"{output_dir}/phase_distribution.png", width=1200, height=600)
        
        # Interface analysis
        fig_interface = self.create_interface_analysis_plot()
        fig_interface.write_html(f"{output_dir}/interface_analysis.html")
        fig_interface.write_image(f"{output_dir}/interface_analysis.png", width=1200, height=900)
        
        print(f"Static plots saved to {output_dir}/")


def main():
    """Main function for visualization."""
    print("Starting visualization...")
    
    # Load microstructure data
    try:
        with h5py.File('output/sofc_microstructure.h5', 'r') as f:
            microstructure = f['microstructure'][:]
            voxel_size = f.attrs['voxel_size_um']
    except FileNotFoundError:
        print("Microstructure data not found. Please run microstructure_generator.py first.")
        return
    
    # Create visualizer
    visualizer = MicrostructureVisualizer(microstructure, voxel_size)
    
    # Save static plots
    visualizer.save_static_plots()
    
    # Create and run interactive dashboard
    print("Starting interactive dashboard...")
    app = visualizer.create_interactive_dashboard()
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE")
    print("="*50)
    print("Static plots saved to 'output/plots/' directory")
    print("Interactive dashboard available at: http://127.0.0.1:8050")
    print("Run 'python visualization_dashboard.py' to start the dashboard")
    
    # Uncomment to run dashboard automatically
    # app.run_server(debug=True, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    main()