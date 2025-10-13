"""
Visualization utilities for SOFC digital twin dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class VisualizationUtils:
    """Utility class for data visualization."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualization utilities."""
        plt.style.use(style)
        sns.set_palette("husl")
    
    @staticmethod
    def plot_temperature_field(temperature_data: Dict[str, Any], 
                             time_index: int = 0, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot 2D temperature field."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'temperature_field_2d' in temperature_data:
            temp_field = temperature_data['temperature_field_2d'][time_index]
            im = ax.imshow(temp_field, cmap='hot', origin='lower')
            ax.set_title(f'Temperature Field at t={time_index}')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            plt.colorbar(im, ax=ax, label='Temperature (°C)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_3d_temperature_field(temperature_data: Dict[str, Any], 
                                 time_index: int = 0,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot 3D temperature field."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if 'temperature_field_3d' in temperature_data:
            temp_field = temperature_data['temperature_field_3d'][time_index]
            
            # Create coordinate grids
            nx, ny, nz = temp_field.shape
            x = np.linspace(0, 0.1, nx)
            y = np.linspace(0, 0.1, ny)
            z = np.linspace(0, 0.01, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # Plot isosurfaces
            ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                      c=temp_field.flatten(), cmap='hot', s=1)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'3D Temperature Field at t={time_index}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_thermocouple_data(thermocouple_data: Dict[str, Any], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot thermocouple data over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for tc_id, tc_data in thermocouple_data.items():
            if 'temperature' in tc_data and 'time_points' in tc_data:
                ax.plot(tc_data['time_points'], tc_data['temperature'], 
                       label=tc_id, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Thermocouple Temperature Measurements')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_strain_data(strain_data: Dict[str, Any], 
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot strain gauge data over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for sg_id, sg_data in strain_data.items():
            if 'strain' in sg_data and 'time_points' in sg_data:
                ax.plot(sg_data['time_points'], sg_data['strain'] * 1e6, 
                       label=sg_id, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Strain (με)')
        ax.set_title('Strain Gauge Measurements')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_voltage_current_data(voltage_current_data: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot voltage and current data."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        time_points = voltage_current_data['time_points']
        
        # Voltage
        ax1.plot(time_points, voltage_current_data['voltage'], 'b-', linewidth=2)
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Cell Voltage')
        ax1.grid(True, alpha=0.3)
        
        # Current density
        ax2.plot(time_points, voltage_current_data['current_density'], 'r-', linewidth=2)
        ax2.set_ylabel('Current Density (A/cm²)')
        ax2.set_title('Current Density')
        ax2.grid(True, alpha=0.3)
        
        # Power
        ax3.plot(time_points, voltage_current_data['power'], 'g-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power Density (W/cm²)')
        ax3.set_title('Power Density')
        ax3.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_eis_spectra(eis_data: Dict[str, Any], 
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot EIS spectra."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for condition, data in eis_data.items():
            if 'frequencies' in data and 'real_impedance' in data:
                # Nyquist plot
                ax1.plot(data['real_impedance'], -data['imaginary_impedance'], 
                        'o-', label=condition, markersize=4)
                
                # Bode plot
                ax2.loglog(data['frequencies'], data['magnitude'], 
                          'o-', label=condition, markersize=4)
        
        ax1.set_xlabel('Real Impedance (Ω·cm²)')
        ax1.set_ylabel('-Imaginary Impedance (Ω·cm²)')
        ax1.set_title('Nyquist Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('|Z| (Ω·cm²)')
        ax2.set_title('Bode Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_degradation_data(degradation_data: Dict[str, Any], 
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot degradation data."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        time_points = degradation_data.get('time_points', np.arange(1000))
        
        # Voltage degradation
        if 'voltage_degradation' in degradation_data:
            axes[0, 0].plot(time_points, degradation_data['voltage_degradation'], 'b-', linewidth=2)
            axes[0, 0].set_ylabel('Voltage (V)')
            axes[0, 0].set_title('Voltage Degradation')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Resistance increase
        if 'resistance_increase' in degradation_data:
            axes[0, 1].plot(time_points, degradation_data['resistance_increase'], 'r-', linewidth=2)
            axes[0, 1].set_ylabel('Resistance Increase')
            axes[0, 1].set_title('Resistance Increase')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Porosity change
        if 'porosity_change' in degradation_data:
            axes[1, 0].plot(time_points, degradation_data['porosity_change'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Porosity Change')
            axes[1, 0].set_title('Porosity Change')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Anode damage
        if 'anode_damage' in degradation_data:
            axes[1, 1].plot(time_points, degradation_data['anode_damage'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Anode Damage')
            axes[1, 1].set_title('Anode Damage')
            axes[1, 1].grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_microstructure(microstructure_data: Dict[str, Any], 
                           component: str = 'anode',
                           slice_index: int = 0,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot microstructure slice."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if component in microstructure_data:
            microstructure = microstructure_data[component]['voxel_data']
            
            # Select slice
            if len(microstructure.shape) == 3:
                slice_data = microstructure[:, :, slice_index]
            else:
                slice_data = microstructure
            
            # Plot binary microstructure
            im = ax.imshow(slice_data, cmap='gray', origin='lower')
            ax.set_title(f'{component.capitalize()} Microstructure (Slice {slice_index})')
            ax.set_xlabel('X (voxels)')
            ax.set_ylabel('Y (voxels)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Phase (0=Solid, 1=Pore)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_sensor_locations(sensor_data: Dict[str, Any], 
                             sensor_type: str = 'thermocouple',
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot sensor locations on SOFC cell."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw SOFC cell outline
        cell_rect = patches.Rectangle((0, 0), 0.1, 0.1, linewidth=2, 
                                    edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(cell_rect)
        
        # Plot sensor locations
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (sensor_id, sensor_info) in enumerate(sensor_data.items()):
            if 'location' in sensor_info:
                loc = sensor_info['location']
                x, y = loc['x'], loc['y']
                
                ax.scatter(x, y, c=colors[i % len(colors)], s=100, 
                          label=sensor_id, edgecolors='black', linewidth=1)
                
                # Add sensor ID text
                ax.annotate(sensor_id, (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlim(-0.01, 0.11)
        ax.set_ylim(-0.01, 0.11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{sensor_type.capitalize()} Sensor Locations')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_interactive_dashboard(data: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Temperature', 'Voltage', 'Current', 'Strain', 'Power', 'EIS'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add temperature data
        if 'thermocouple_data' in data:
            for tc_id, tc_data in data['thermocouple_data'].items():
                if 'temperature' in tc_data and 'time_points' in tc_data:
                    fig.add_trace(
                        go.Scatter(x=tc_data['time_points'], y=tc_data['temperature'],
                                 mode='lines', name=tc_id),
                        row=1, col=1
                    )
        
        # Add voltage data
        if 'voltage_current_data' in data:
            vc_data = data['voltage_current_data']
            fig.add_trace(
                go.Scatter(x=vc_data['time_points'], y=vc_data['voltage'],
                          mode='lines', name='Voltage'),
                row=1, col=2
            )
        
        # Add current data
        if 'voltage_current_data' in data:
            vc_data = data['voltage_current_data']
            fig.add_trace(
                go.Scatter(x=vc_data['time_points'], y=vc_data['current_density'],
                          mode='lines', name='Current Density'),
                row=2, col=1
            )
        
        # Add strain data
        if 'strain_gauge_data' in data:
            for sg_id, sg_data in data['strain_gauge_data'].items():
                if 'strain' in sg_data and 'time_points' in sg_data:
                    fig.add_trace(
                        go.Scatter(x=sg_data['time_points'], y=sg_data['strain'] * 1e6,
                                 mode='lines', name=sg_id),
                        row=2, col=2
                    )
        
        # Add power data
        if 'voltage_current_data' in data:
            vc_data = data['voltage_current_data']
            fig.add_trace(
                go.Scatter(x=vc_data['time_points'], y=vc_data['power'],
                          mode='lines', name='Power'),
                row=3, col=1
            )
        
        # Add EIS data
        if 'electrochemical_response' in data and 'eis_spectra' in data['electrochemical_response']:
            eis_data = data['electrochemical_response']['eis_spectra']
            for condition, eis in eis_data.items():
                if 'frequencies' in eis and 'real_impedance' in eis:
                    fig.add_trace(
                        go.Scatter(x=eis['real_impedance'], y=-eis['imaginary_impedance'],
                                 mode='markers+lines', name=condition),
                        row=3, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title="SOFC Digital Twin Dashboard",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def plot_heatmap(data: np.ndarray, 
                    xlabel: str = 'X', 
                    ylabel: str = 'Y',
                    title: str = 'Heatmap',
                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot 2D heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot correlation matrix heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title('Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_distribution(data: np.ndarray, 
                         bins: int = 50,
                         title: str = 'Distribution',
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot data distribution histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig