"""
Advanced Visualization Utilities for SOFC Digital Twin Data
Provides specialized visualization tools for multi-physics field data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

class SOFCVisualizer:
    """
    Specialized visualizer for SOFC digital twin data
    """
    
    def __init__(self):
        """Initialize visualizer with custom color schemes"""
        
        # Custom colormaps for different physics fields
        self.temp_colormap = LinearSegmentedColormap.from_list(
            'temperature', ['blue', 'cyan', 'yellow', 'red', 'darkred']
        )
        
        self.stress_colormap = LinearSegmentedColormap.from_list(
            'stress', ['green', 'yellow', 'orange', 'red', 'darkred']
        )
        
        self.current_colormap = LinearSegmentedColormap.from_list(
            'current', ['white', 'lightblue', 'blue', 'darkblue', 'black']
        )
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_3d_field(self, field_data: np.ndarray, field_name: str, 
                      geometry: Dict, slice_plane: str = 'z', 
                      slice_index: Optional[int] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D field data as 2D slice with proper scaling
        """
        
        if slice_index is None:
            slice_index = field_data.shape[2] // 2  # Middle slice
        
        # Extract 2D slice
        if slice_plane == 'z':
            field_slice = field_data[:, :, slice_index]
            x_coords = np.linspace(0, geometry['length'], field_data.shape[0])
            y_coords = np.linspace(0, geometry['width'], field_data.shape[1])
            xlabel, ylabel = 'Length (m)', 'Width (m)'
        elif slice_plane == 'y':
            field_slice = field_data[:, slice_index, :]
            x_coords = np.linspace(0, geometry['length'], field_data.shape[0])
            y_coords = np.linspace(0, geometry['height'], field_data.shape[2])
            xlabel, ylabel = 'Length (m)', 'Height (m)'
        else:  # x plane
            field_slice = field_data[slice_index, :, :]
            x_coords = np.linspace(0, geometry['width'], field_data.shape[1])
            y_coords = np.linspace(0, geometry['height'], field_data.shape[2])
            xlabel, ylabel = 'Width (m)', 'Height (m)'
        
        # Choose appropriate colormap
        if 'temperature' in field_name.lower():
            cmap = self.temp_colormap
            units = '°C' if np.max(field_slice) > 100 else 'K'
        elif 'stress' in field_name.lower():
            cmap = self.stress_colormap
            units = 'Pa'
        elif 'current' in field_name.lower():
            cmap = self.current_colormap
            units = 'A/m²'
        else:
            cmap = 'viridis'
            units = ''
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create meshgrid for proper scaling
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Plot field
        im = ax.contourf(X, Y, field_slice.T, levels=50, cmap=cmap)
        
        # Add contour lines
        contours = ax.contour(X, Y, field_slice.T, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{field_name} ({units})', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{field_name} - {slice_plane.upper()} plane (slice {slice_index})')
        
        # Add geometry annotations
        self._add_geometry_annotations(ax, geometry, slice_plane)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _add_geometry_annotations(self, ax, geometry: Dict, slice_plane: str):
        """Add geometry annotations to field plots"""
        
        # Add component boundaries (simplified SOFC geometry)
        if slice_plane == 'z':
            # Anode, electrolyte, cathode layers
            width = geometry['width']
            height = geometry['height']
            
            # Electrolyte (center)
            electrolyte_thickness = 20e-6  # 20 μm
            center_y = width / 2
            
            rect = patches.Rectangle(
                (0, center_y - electrolyte_thickness/2),
                geometry['length'], electrolyte_thickness,
                linewidth=2, edgecolor='white', facecolor='none',
                linestyle='--', alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(geometry['length']*0.02, center_y, 'Electrolyte', 
                   color='white', fontweight='bold', fontsize=8)
            
            # Anode (bottom)
            ax.text(geometry['length']*0.02, center_y - width*0.3, 'Anode', 
                   color='white', fontweight='bold', fontsize=10)
            
            # Cathode (top)
            ax.text(geometry['length']*0.02, center_y + width*0.3, 'Cathode', 
                   color='white', fontweight='bold', fontsize=10)
    
    def plot_multi_physics_comparison(self, simulation_data: Dict, 
                                    geometry: Dict,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create multi-physics comparison plot showing temperature, stress, and current density
        """
        
        fields = simulation_data.get('fields', {})
        
        # Select middle slice for visualization
        slice_idx = geometry['nz'] // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Temperature field
        if 'temperature' in fields:
            temp_field = fields['temperature'][:, :, slice_idx]
            im1 = axes[0, 0].imshow(temp_field.T, cmap=self.temp_colormap, 
                                  aspect='auto', origin='lower')
            axes[0, 0].set_title('Temperature Distribution')
            plt.colorbar(im1, ax=axes[0, 0], label='Temperature (K)')
        
        # Von Mises stress
        if 'von_mises_stress' in fields:
            stress_field = fields['von_mises_stress'][:, :, slice_idx]
            im2 = axes[0, 1].imshow(stress_field.T, cmap=self.stress_colormap, 
                                  aspect='auto', origin='lower')
            axes[0, 1].set_title('Von Mises Stress')
            plt.colorbar(im2, ax=axes[0, 1], label='Stress (Pa)')
        
        # Current density
        if 'current_density' in fields:
            current_field = fields['current_density'][:, :, slice_idx]
            im3 = axes[1, 0].imshow(current_field.T, cmap=self.current_colormap, 
                                  aspect='auto', origin='lower')
            axes[1, 0].set_title('Current Density Distribution')
            plt.colorbar(im3, ax=axes[1, 0], label='Current Density (A/m²)')
        
        # Displacement magnitude
        if all(field in fields for field in ['displacement_x', 'displacement_y', 'displacement_z']):
            disp_x = fields['displacement_x'][:, :, slice_idx]
            disp_y = fields['displacement_y'][:, :, slice_idx]
            disp_z = fields['displacement_z'][:, :, slice_idx]
            disp_mag = np.sqrt(disp_x**2 + disp_y**2 + disp_z**2)
            
            im4 = axes[1, 1].imshow(disp_mag.T, cmap='plasma', 
                                  aspect='auto', origin='lower')
            axes[1, 1].set_title('Displacement Magnitude')
            plt.colorbar(im4, ax=axes[1, 1], label='Displacement (m)')
        
        # Add common formatting
        for ax in axes.flat:
            ax.set_xlabel('X direction')
            ax.set_ylabel('Y direction')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_3d_plot(self, field_data: np.ndarray, field_name: str,
                                 geometry: Dict) -> go.Figure:
        """
        Create interactive 3D visualization using Plotly
        """
        
        # Create coordinate grids
        x = np.linspace(0, geometry['length'], field_data.shape[0])
        y = np.linspace(0, geometry['width'], field_data.shape[1])
        z = np.linspace(0, geometry['height'], field_data.shape[2])
        
        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten arrays for plotting
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        values_flat = field_data.flatten()
        
        # Create 3D scatter plot with color mapping
        fig = go.Figure(data=go.Scatter3d(
            x=x_flat[::10],  # Subsample for performance
            y=y_flat[::10],
            z=z_flat[::10],
            mode='markers',
            marker=dict(
                size=3,
                color=values_flat[::10],
                colorscale='Viridis',
                colorbar=dict(title=field_name),
                opacity=0.6
            ),
            text=[f'{field_name}: {val:.2f}' for val in values_flat[::10]],
            hovertemplate='X: %{x:.2e}<br>Y: %{y:.2e}<br>Z: %{z:.2e}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D {field_name} Distribution',
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Width (m)',
                zaxis_title='Height (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_eis_nyquist(self, eis_data: List[Dict], 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot EIS Nyquist diagrams showing impedance evolution
        """
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color map for time evolution
        colors = plt.cm.viridis(np.linspace(0, 1, len(eis_data)))
        
        for i, measurement in enumerate(eis_data):
            Z_real = measurement['Z_real']
            Z_imag = measurement['Z_imag']
            test_time = measurement.get('simulation_time_hours', i)
            
            ax.plot(Z_real, Z_imag, 'o-', color=colors[i], 
                   label=f't = {test_time:.1f}h', markersize=4, alpha=0.7)
        
        ax.set_xlabel('Real Impedance (Ω)')
        ax.set_ylabel('-Imaginary Impedance (Ω)')
        ax.set_title('EIS Nyquist Plot - Degradation Evolution')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add colorbar for time evolution
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=0, vmax=len(eis_data)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Measurement Number', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_thermal_evolution(self, thermal_data: List[Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot thermal imaging evolution over time
        """
        
        n_images = min(len(thermal_data), 9)  # Show up to 9 images
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(n_images):
            thermal_map = np.array(thermal_data[i]['temperature_map'])
            test_time = thermal_data[i].get('simulation_time_hours', i)
            
            im = axes[i].imshow(thermal_map, cmap=self.temp_colormap, 
                              aspect='auto', origin='lower')
            axes[i].set_title(f't = {test_time:.1f}h\nMax: {np.max(thermal_map):.1f}°C')
            axes[i].set_xlabel('X position')
            axes[i].set_ylabel('Y position')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Thermal Image Evolution', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_degradation_metrics(self, operational_data: Dict,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot key degradation metrics over time
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'operational' in operational_data:
            df = operational_data['operational']
            
            # Voltage degradation
            axes[0, 0].plot(df['time_hours'], df['voltage'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Time (hours)')
            axes[0, 0].set_ylabel('Voltage (V)')
            axes[0, 0].set_title('Voltage Degradation')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Power output
            axes[0, 1].plot(df['time_hours'], df['power'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Power (W/cm²)')
            axes[0, 1].set_title('Power Output')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Temperature evolution
            axes[1, 0].plot(df['time_hours'], df['temp_fuel_inlet'], 'g-', 
                          label='Fuel Inlet', linewidth=2)
            axes[1, 0].plot(df['time_hours'], df['temp_air_inlet'], 'b-', 
                          label='Air Inlet', linewidth=2)
            axes[1, 0].set_xlabel('Time (hours)')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].set_title('Temperature Evolution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Current density
            axes[1, 1].plot(df['time_hours'], df['current_density'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Time (hours)')
            axes[1, 1].set_ylabel('Current Density (A/cm²)')
            axes[1, 1].set_title('Current Density')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(self, simulation_data: Dict, 
                                  experimental_data: Dict,
                                  realtime_data: Dict,
                                  output_dir: str = "visualization_report"):
        """
        Create comprehensive visualization report
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating comprehensive visualization report...")
        
        # 1. Multi-physics field plots
        if 'fields' in simulation_data:
            geometry = simulation_data.get('geometry', {
                'length': 10e-3, 'width': 10e-3, 'height': 2e-3,
                'nx': 50, 'ny': 30, 'nz': 20
            })
            
            fig1 = self.plot_multi_physics_comparison(
                simulation_data, geometry,
                save_path=os.path.join(output_dir, 'multi_physics_fields.png')
            )
            plt.close(fig1)
        
        # 2. EIS evolution plots
        if 'eis' in experimental_data and experimental_data['eis']:
            fig2 = self.plot_eis_nyquist(
                experimental_data['eis'],
                save_path=os.path.join(output_dir, 'eis_evolution.png')
            )
            plt.close(fig2)
        
        # 3. Thermal evolution
        if 'thermal' in experimental_data and experimental_data['thermal']:
            fig3 = self.plot_thermal_evolution(
                experimental_data['thermal'],
                save_path=os.path.join(output_dir, 'thermal_evolution.png')
            )
            plt.close(fig3)
        
        # 4. Degradation metrics
        if realtime_data:
            fig4 = self.plot_degradation_metrics(
                realtime_data,
                save_path=os.path.join(output_dir, 'degradation_metrics.png')
            )
            plt.close(fig4)
        
        print(f"Visualization report saved to: {output_dir}")

def main():
    """Demonstrate visualization capabilities"""
    visualizer = SOFCVisualizer()
    
    # Create sample data for demonstration
    geometry = {
        'length': 10e-3, 'width': 10e-3, 'height': 2e-3,
        'nx': 50, 'ny': 30, 'nz': 20
    }
    
    # Generate sample field data
    nx, ny, nz = geometry['nx'], geometry['ny'], geometry['nz']
    
    # Sample temperature field
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    temperature = 800 + 100 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.1)
    
    # Create visualization
    fig = visualizer.plot_3d_field(
        temperature, 'Temperature', geometry,
        save_path='sample_temperature_field.png'
    )
    plt.show()
    
    print("Sample visualization created: sample_temperature_field.png")

if __name__ == "__main__":
    main()