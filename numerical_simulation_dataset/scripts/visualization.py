"""
Visualization tools for numerical simulation data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SimulationVisualizer:
    """Visualize numerical simulation results"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.colormap = 'viridis'
        
    def load_simulation_data(self, sim_id: str, data_type: str) -> Dict:
        """Load simulation data from HDF5 file"""
        
        file_path = self.data_path / f'output_data/{data_type}/{sim_id}_{data_type.split("_")[0]}.h5'
        
        data = {}
        with h5py.File(file_path, 'r') as f:
            def load_group(group, data_dict):
                for key in group.keys():
                    if isinstance(group[key], h5py.Group):
                        data_dict[key] = {}
                        load_group(group[key], data_dict[key])
                    else:
                        data_dict[key] = group[key][:]
                        
            load_group(f, data)
            
        return data
    
    def plot_stress_contour(self, sim_id: str, time_step: int = 0, 
                          stress_type: str = 'von_mises',
                          slice_axis: str = 'z', slice_index: int = None):
        """Plot 2D stress contour at specific slice"""
        
        # Load stress data
        stress_data = self.load_simulation_data(sim_id, 'stress_fields')
        time_key = f't_{time_step}'
        
        if time_key not in stress_data:
            raise ValueError(f"Time step {time_step} not found in data")
        
        stress_field = stress_data[time_key][stress_type]
        coords = stress_data[time_key]['coordinates']
        
        # Get slice
        if slice_axis == 'z':
            if slice_index is None:
                slice_index = stress_field.shape[2] // 2
            data_slice = stress_field[:, :, slice_index]
            x_coords = coords['x'][:, :, slice_index]
            y_coords = coords['y'][:, :, slice_index]
            xlabel, ylabel = 'X (mm)', 'Y (mm)'
        elif slice_axis == 'y':
            if slice_index is None:
                slice_index = stress_field.shape[1] // 2
            data_slice = stress_field[:, slice_index, :]
            x_coords = coords['x'][:, slice_index, :]
            y_coords = coords['z'][:, slice_index, :]
            xlabel, ylabel = 'X (mm)', 'Z (mm)'
        else:  # x
            if slice_index is None:
                slice_index = stress_field.shape[0] // 2
            data_slice = stress_field[slice_index, :, :]
            x_coords = coords['y'][slice_index, :, :]
            y_coords = coords['z'][slice_index, :, :]
            xlabel, ylabel = 'Y (mm)', 'Z (mm)'
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(x_coords, y_coords, data_slice, 
                             levels=20, cmap=self.colormap)
        plt.colorbar(contour, ax=ax, label=f'{stress_type} Stress (MPa)')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{stress_type} Stress Distribution\n' +
                    f'Simulation: {sim_id}, Time Step: {time_step}, {slice_axis}={slice_index}')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_3d_stress_isosurface(self, sim_id: str, time_step: int = 0,
                                 stress_type: str = 'von_mises',
                                 threshold_percentile: float = 75):
        """Plot 3D isosurface of stress using plotly"""
        
        # Load stress data
        stress_data = self.load_simulation_data(sim_id, 'stress_fields')
        time_key = f't_{time_step}'
        
        stress_field = stress_data[time_key][stress_type]
        coords = stress_data[time_key]['coordinates']
        
        # Calculate threshold
        threshold = np.percentile(stress_field, threshold_percentile)
        
        # Create isosurface
        fig = go.Figure(data=go.Isosurface(
            x=coords['x'].flatten(),
            y=coords['y'].flatten(),
            z=coords['z'].flatten(),
            value=stress_field.flatten(),
            isomin=threshold,
            isomax=stress_field.max(),
            surface_count=5,
            colorscale='Viridis',
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            colorbar=dict(title=f'{stress_type} Stress (MPa)')
        ))
        
        fig.update_layout(
            title=f'3D Stress Isosurface - {sim_id}, t={time_step}',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def plot_damage_evolution(self, sim_id: str):
        """Plot damage evolution over time"""
        
        # Load damage data
        damage_data = self.load_simulation_data(sim_id, 'damage_evolution')
        
        # Extract time series data
        times = []
        mean_damage = []
        max_damage = []
        critical_elements = []
        
        for t in range(len(damage_data)):
            time_key = f't_{t}'
            if time_key in damage_data:
                times.append(t)
                mean_damage.append(damage_data[time_key]['mean_damage'])
                max_damage.append(damage_data[time_key]['max_damage'])
                critical_elements.append(damage_data[time_key]['critical_elements'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Damage Evolution', 'Critical Elements'),
            vertical_spacing=0.15
        )
        
        # Damage evolution
        fig.add_trace(
            go.Scatter(x=times, y=mean_damage, name='Mean Damage',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=times, y=max_damage, name='Max Damage',
                      line=dict(color='red', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Critical elements
        fig.add_trace(
            go.Bar(x=times, y=critical_elements, name='Critical Elements',
                  marker_color='orange'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='Time Step', row=2, col=1)
        fig.update_yaxes(title_text='Damage Variable', row=1, col=1)
        fig.update_yaxes(title_text='Number of Elements', row=2, col=1)
        
        fig.update_layout(
            title=f'Damage Evolution - Simulation {sim_id}',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_temperature_history(self, sim_id: str, point_coords: Optional[Tuple] = None):
        """Plot temperature history at specific point or average"""
        
        # Load temperature data
        temp_data = self.load_simulation_data(sim_id, 'temperature_distributions')
        
        times = []
        temps = []
        max_temps = []
        min_temps = []
        
        for t in range(len(temp_data)):
            time_key = f't_{t}'
            if time_key in temp_data:
                times.append(t)
                
                if point_coords:
                    # Get temperature at specific point
                    i, j, k = point_coords
                    temp_field = temp_data[time_key]['temperature']
                    temps.append(temp_field[i, j, k])
                else:
                    # Use average temperature
                    temps.append(temp_data[time_key]['mean_temp'])
                
                max_temps.append(temp_data[time_key]['max_temp'])
                min_temps.append(temp_data[time_key]['min_temp'])
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times, y=temps,
            name='Temperature' if not point_coords else f'T at {point_coords}',
            line=dict(color='red', width=2)
        ))
        
        # Add min/max envelope
        fig.add_trace(go.Scatter(
            x=times, y=max_temps,
            name='Max T',
            line=dict(color='orange', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=min_temps,
            name='Min T',
            line=dict(color='blue', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title=f'Temperature Evolution - Simulation {sim_id}',
            xaxis_title='Time Step',
            yaxis_title='Temperature (°C)',
            height=500
        )
        
        return fig
    
    def plot_failure_prediction_map(self, sim_id: str, time_step: int = -1):
        """Plot failure prediction map"""
        
        # Load failure data
        failure_data = self.load_simulation_data(sim_id, 'failure_predictions')
        
        if time_step == -1:
            # Use last time step
            time_step = len(failure_data) - 1
        
        time_key = f't_{time_step}'
        
        delamination = failure_data[time_key]['delamination_risk']
        crack_prob = failure_data[time_key]['crack_probability']
        
        # Get middle slice for visualization
        z_mid = delamination.shape[2] // 2
        
        # Create subplot with delamination and crack probability
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Delamination risk
        im1 = ax1.imshow(delamination[:, :, z_mid], cmap='hot', aspect='auto')
        ax1.set_title('Delamination Risk')
        ax1.set_xlabel('Y index')
        ax1.set_ylabel('X index')
        plt.colorbar(im1, ax=ax1, label='Risk (0-1)')
        
        # Crack probability
        im2 = ax2.imshow(crack_prob[:, :, z_mid], cmap='hot', aspect='auto')
        ax2.set_title('Crack Initiation Probability')
        ax2.set_xlabel('Y index')
        ax2.set_ylabel('X index')
        plt.colorbar(im2, ax=ax2, label='Probability (0-1)')
        
        fig.suptitle(f'Failure Predictions - {sim_id}, Time Step: {time_step}')
        plt.tight_layout()
        
        return fig
    
    def create_animation(self, sim_id: str, data_type: str = 'stress',
                        output_file: str = None):
        """Create animation of field evolution over time"""
        
        import matplotlib.animation as animation
        
        # Load data
        if data_type == 'stress':
            data = self.load_simulation_data(sim_id, 'stress_fields')
            field_name = 'von_mises'
            label = 'Von Mises Stress (MPa)'
        elif data_type == 'damage':
            data = self.load_simulation_data(sim_id, 'damage_evolution')
            field_name = 'damage'
            label = 'Damage Variable'
        else:
            data = self.load_simulation_data(sim_id, 'temperature_distributions')
            field_name = 'temperature'
            label = 'Temperature (°C)'
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get data range for consistent colorbar
        all_values = []
        for time_key in data.keys():
            if time_key.startswith('t_'):
                all_values.append(data[time_key][field_name][:, :, data[time_key][field_name].shape[2]//2])
        
        vmin = np.min(all_values)
        vmax = np.max(all_values)
        
        # Initial plot
        im = ax.imshow(all_values[0], cmap=self.colormap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(f'{field_name} - Time: 0')
        
        def animate(frame):
            time_key = f't_{frame}'
            if time_key in data:
                field = data[time_key][field_name]
                z_mid = field.shape[2] // 2
                im.set_array(field[:, :, z_mid])
                ax.set_title(f'{field_name} - Time: {frame}')
            return [im]
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(data), interval=100, blit=True
        )
        
        if output_file:
            anim.save(output_file, writer='pillow', fps=10)
        
        return anim
    
    def generate_summary_report(self, sim_id: str, output_path: str = None):
        """Generate comprehensive summary report with multiple plots"""
        
        from matplotlib.backends.backend_pdf import PdfPages
        
        if output_path is None:
            output_path = self.data_path / f'reports/{sim_id}_report.pdf'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with PdfPages(output_path) as pdf:
            # Page 1: Stress contours at different time steps
            fig = plt.figure(figsize=(15, 10))
            for i, t in enumerate([0, 10, 25, 49]):
                ax = fig.add_subplot(2, 2, i+1)
                try:
                    stress_data = self.load_simulation_data(sim_id, 'stress_fields')
                    time_key = f't_{t}'
                    if time_key in stress_data:
                        stress = stress_data[time_key]['von_mises']
                        z_mid = stress.shape[2] // 2
                        im = ax.imshow(stress[:, :, z_mid], cmap='viridis')
                        ax.set_title(f'Von Mises Stress at t={t}')
                        plt.colorbar(im, ax=ax)
                except:
                    ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            
            fig.suptitle(f'Stress Evolution - {sim_id}', fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Page 2: Damage evolution
            try:
                fig = self.plot_damage_evolution(sim_id)
                pdf.savefig(fig)
                plt.close()
            except:
                pass
            
            # Page 3: Temperature history
            try:
                fig = self.plot_temperature_history(sim_id)
                pdf.savefig(fig)
                plt.close()
            except:
                pass
            
            # Page 4: Failure predictions
            try:
                fig = self.plot_failure_prediction_map(sim_id)
                pdf.savefig(fig)
                plt.close()
            except:
                pass
        
        print(f"Report saved to {output_path}")

def plot_parameter_sweep_results(data_path: str):
    """Plot parameter sweep analysis results"""
    
    # Load parameter sweep data
    df = pd.read_csv(Path(data_path) / 'summary_statistics/parameter_sweep.csv')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Stress vs Heating Rate',
            'Damage vs Mesh Refinement',
            'Stress by Load Type',
            'Damage Model Comparison'
        )
    )
    
    # Stress vs heating rate
    fig.add_trace(
        go.Scatter(x=df['heating_rate'], y=df['max_stress'],
                  mode='markers', marker=dict(size=8),
                  name='Max Stress'),
        row=1, col=1
    )
    
    # Damage vs mesh refinement
    for refinement in df['mesh_refinement'].unique():
        mask = df['mesh_refinement'] == refinement
        fig.add_trace(
            go.Box(y=df[mask]['max_damage'], name=f'Refinement {refinement}'),
            row=1, col=2
        )
    
    # Stress by load type
    for load_type in df['load_type'].unique():
        mask = df['load_type'] == load_type
        fig.add_trace(
            go.Box(y=df[mask]['max_stress'], name=load_type),
            row=2, col=1
        )
    
    # Damage model comparison
    for model in df['damage_model'].unique():
        mask = df['damage_model'] == model
        fig.add_trace(
            go.Box(y=df[mask]['max_damage'], name=model),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title='Parameter Sweep Analysis')
    
    return fig