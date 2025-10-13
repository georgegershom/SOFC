"""
Visualization and Analysis Tools for SOFC Digital Twin Dataset
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from scipy import signal
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SOFCDataVisualizer:
    """
    Comprehensive visualization tools for SOFC digital twin dataset
    """
    
    def __init__(self, data_path):
        """
        Initialize visualizer with dataset path
        
        Parameters:
        -----------
        data_path : str or Path
            Path to the dataset directory
        """
        self.data_path = Path(data_path)
        self.load_metadata()
        
    def load_metadata(self):
        """Load dataset metadata"""
        with open(self.data_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
    
    def visualize_simulation_fields(self, sim_id=0, save_fig=False):
        """
        Visualize 3D field data from simulation
        
        Parameters:
        -----------
        sim_id : int
            Simulation ID to visualize
        save_fig : bool
            Save figure to file
        """
        with h5py.File(self.data_path / 'simulation' / 'high_fidelity_simulations.h5', 'r') as hf:
            sim_group = hf[f'simulation_{sim_id:04d}']
            
            # Load field data
            temperature = sim_group['temperature'][:]
            stress = sim_group['von_mises_stress'][:]
            current_density = sim_group['current_density'][:]
            
            # Get parameters
            params = dict(sim_group.attrs)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Temperature field - XY plane at mid Z
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(temperature[:, :, temperature.shape[2]//2].T, 
                        cmap='hot', aspect='auto')
        ax1.set_title('Temperature Field (Mid-plane)')
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        plt.colorbar(im1, ax=ax1, label='Temperature [°C]')
        
        # Von Mises stress - XY plane at mid Z
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(stress[:, :, stress.shape[2]//2].T / 1e6, 
                        cmap='plasma', aspect='auto')
        ax2.set_title('Von Mises Stress (Mid-plane)')
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        plt.colorbar(im2, ax=ax2, label='Stress [MPa]')
        
        # Current density - XY plane at mid Z
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(current_density[:, :, current_density.shape[2]//2].T, 
                        cmap='viridis', aspect='auto')
        ax3.set_title('Current Density (Mid-plane)')
        ax3.set_xlabel('X [mm]')
        ax3.set_ylabel('Y [mm]')
        plt.colorbar(im3, ax=ax3, label='Current Density [A/m²]')
        
        # Temperature profile along X
        ax4 = fig.add_subplot(2, 3, 4)
        mid_y = temperature.shape[1] // 2
        mid_z = temperature.shape[2] // 2
        ax4.plot(temperature[:, mid_y, mid_z])
        ax4.set_title('Temperature Profile (X-direction)')
        ax4.set_xlabel('X Position [mm]')
        ax4.set_ylabel('Temperature [°C]')
        ax4.grid(True, alpha=0.3)
        
        # Stress profile along X
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(stress[:, mid_y, mid_z] / 1e6)
        ax5.set_title('Stress Profile (X-direction)')
        ax5.set_xlabel('X Position [mm]')
        ax5.set_ylabel('Von Mises Stress [MPa]')
        ax5.grid(True, alpha=0.3)
        
        # Parameter summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        param_text = f"""Simulation Parameters:
        
Current Density: {params['current_density']:.0f} A/m²
Fuel Utilization: {params['fuel_utilization']:.2f}
Air Utilization: {params['air_utilization']:.2f}
Fuel Temperature: {params['fuel_temperature']:.0f} °C
Air Temperature: {params['air_temperature']:.0f} °C
Crack Length: {params['crack_length']:.2f} mm
Voltage: {params['voltage']:.3f} V
Max Stress: {params['max_stress']/1e6:.1f} MPa
Creep Damage: {params['creep_damage']:.4f}"""
        
        ax6.text(0.1, 0.5, param_text, fontsize=11, 
                verticalalignment='center', family='monospace')
        ax6.set_title('Simulation Parameters & Results')
        
        plt.suptitle(f'SOFC Multi-Physics Simulation #{sim_id:04d}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'simulation_{sim_id:04d}.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def create_3d_field_visualization(self, sim_id=0, field='temperature'):
        """
        Create interactive 3D visualization of field data
        
        Parameters:
        -----------
        sim_id : int
            Simulation ID
        field : str
            Field to visualize ('temperature', 'stress', 'current_density')
        """
        with h5py.File(self.data_path / 'simulation' / 'high_fidelity_simulations.h5', 'r') as hf:
            sim_group = hf[f'simulation_{sim_id:04d}']
            
            if field == 'temperature':
                data = sim_group['temperature'][:]
                colorscale = 'Hot'
                title = 'Temperature Field [°C]'
            elif field == 'stress':
                data = sim_group['von_mises_stress'][:] / 1e6  # Convert to MPa
                colorscale = 'Plasma'
                title = 'Von Mises Stress [MPa]'
            elif field == 'current_density':
                data = sim_group['current_density'][:]
                colorscale = 'Viridis'
                title = 'Current Density [A/m²]'
            else:
                raise ValueError(f"Unknown field: {field}")
        
        # Create 3D volume plot
        X, Y, Z = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=data.min(),
            isomax=data.max(),
            opacity=0.3,
            surface_count=15,
            colorscale=colorscale,
            caps=dict(x_show=True, y_show=True, z_show=True)
        ))
        
        fig.update_layout(
            title=f'{title} - Simulation #{sim_id:04d}',
            scene=dict(
                xaxis_title='X [grid points]',
                yaxis_title='Y [grid points]',
                zaxis_title='Z [grid points]',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700
        )
        
        fig.show()
    
    def analyze_experimental_data(self):
        """
        Analyze and visualize experimental data
        """
        # Load operational data
        op_data = pd.read_csv(self.data_path / 'experimental' / 'operational_data.csv')
        
        # Load strain gauge data
        strain_data = pd.read_csv(self.data_path / 'experimental' / 'strain_gauge_data.csv')
        
        # Load AE events
        ae_events = pd.read_csv(self.data_path / 'experimental' / 'acoustic_emission_events.csv')
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Voltage Degradation', 'Power Output', 
                          'Temperature Evolution', 'Efficiency Trend',
                          'Strain Evolution', 'Acoustic Emission Events'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                  [{'secondary_y': False}, {'secondary_y': False}],
                  [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Voltage degradation
        fig.add_trace(
            go.Scatter(x=op_data['time_hours'], y=op_data['voltage_V'],
                      mode='lines', name='Voltage', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Power output
        fig.add_trace(
            go.Scatter(x=op_data['time_hours'], y=op_data['power_W'],
                      mode='lines', name='Power', line=dict(color='red')),
            row=1, col=2
        )
        
        # Temperature evolution
        fig.add_trace(
            go.Scatter(x=op_data['time_hours'], y=op_data['T_outlet_C'],
                      mode='lines', name='Outlet Temp', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=op_data['time_hours'], y=op_data['T_inlet_fuel_C'],
                      mode='lines', name='Fuel Inlet', line=dict(color='green', dash='dash')),
            row=2, col=1
        )
        
        # Efficiency trend
        fig.add_trace(
            go.Scatter(x=op_data['time_hours'], y=op_data['efficiency'],
                      mode='lines', name='Efficiency', line=dict(color='purple')),
            row=2, col=2
        )
        
        # Strain evolution (multiple sensors)
        strain_cols = [col for col in strain_data.columns if 'strain_sensor' in col]
        for col in strain_cols[:3]:  # Show first 3 sensors
            fig.add_trace(
                go.Scatter(x=strain_data['time_hours'], y=strain_data[col],
                          mode='lines', name=col.split('_')[2]),
                row=3, col=1
            )
        
        # Acoustic emission events
        if not ae_events.empty:
            fig.add_trace(
                go.Scatter(x=ae_events['time_hours'], y=ae_events['amplitude_dB'],
                          mode='markers', name='AE Events',
                          marker=dict(size=ae_events['energy']/ae_events['energy'].max()*20,
                                    color=ae_events['time_hours'],
                                    colorscale='Viridis',
                                    showscale=True)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Time [hours]", row=3, col=1)
        fig.update_xaxes(title_text="Time [hours]", row=3, col=2)
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_yaxes(title_text="Power [W]", row=1, col=2)
        fig.update_yaxes(title_text="Temperature [°C]", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency [-]", row=2, col=2)
        fig.update_yaxes(title_text="Strain [μɛ]", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude [dB]", row=3, col=2)
        
        fig.update_layout(
            title="SOFC Experimental Data Analysis",
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.show()
    
    def visualize_eis_data(self):
        """
        Visualize Electrochemical Impedance Spectroscopy data
        """
        with open(self.data_path / 'experimental' / 'eis_data.json', 'r') as f:
            eis_data = json.load(f)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Nyquist Plot Evolution', 'Resistance Degradation')
        )
        
        # Nyquist plots for selected measurements
        measurements_to_plot = [0, len(eis_data)//4, len(eis_data)//2, 
                              3*len(eis_data)//4, len(eis_data)-1]
        
        colors = px.colors.sequential.Plasma
        
        for i, idx in enumerate(measurements_to_plot):
            if idx < len(eis_data):
                measurement = eis_data[idx]
                fig.add_trace(
                    go.Scatter(x=measurement['Z_real_ohm'], 
                             y=-np.array(measurement['Z_imag_ohm']),
                             mode='lines+markers',
                             name=f"t={measurement['time_hours']:.0f}h",
                             line=dict(color=colors[i*2]),
                             marker=dict(size=4)),
                    row=1, col=1
                )
        
        # Resistance evolution
        times = [m['time_hours'] for m in eis_data]
        R_ohmic = [m['R_ohmic'] for m in eis_data]
        R_ct = [m['R_charge_transfer'] for m in eis_data]
        
        fig.add_trace(
            go.Scatter(x=times, y=R_ohmic, mode='lines',
                      name='R_ohmic', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=times, y=R_ct, mode='lines',
                      name='R_charge_transfer', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Z_real [Ω]", row=1, col=1)
        fig.update_yaxes(title_text="-Z_imag [Ω]", row=1, col=1)
        fig.update_xaxes(title_text="Time [hours]", row=1, col=2)
        fig.update_yaxes(title_text="Resistance [Ω]", row=1, col=2)
        
        fig.update_layout(
            title="EIS Data Analysis",
            height=500,
            showlegend=True
        )
        
        fig.show()
    
    def visualize_thermal_images(self, n_images=6):
        """
        Visualize thermal camera images
        """
        thermal_images = np.load(self.data_path / 'experimental' / 'thermal_images.npy')
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        indices = np.linspace(0, len(thermal_images)-1, n_images, dtype=int)
        
        for i, idx in enumerate(indices):
            im = axes[i].imshow(thermal_images[idx], cmap='hot', aspect='auto')
            axes[i].set_title(f'Time Step {idx}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.suptitle('Thermal Camera Images - Surface Temperature Evolution', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def analyze_monitoring_streams(self):
        """
        Analyze real-time monitoring data streams
        """
        # Load stream data
        stream_data = pd.read_csv(self.data_path / 'monitoring' / 'realtime_stream.csv')
        
        # Load triggers if available
        trigger_file = self.data_path / 'monitoring' / 'adaptive_triggers.csv'
        if trigger_file.exists():
            triggers = pd.read_csv(trigger_file)
        else:
            triggers = pd.DataFrame()
        
        # Create figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Real-time Voltage Monitoring', 
                          'Power Output', 
                          'Temperature Monitoring'),
            shared_xaxes=True
        )
        
        # Convert timestamp to seconds
        stream_data['time_seconds'] = pd.to_datetime(stream_data['timestamp'])
        stream_data['time_seconds'] = (stream_data['time_seconds'] - 
                                      stream_data['time_seconds'].iloc[0]).dt.total_seconds()
        
        # Voltage monitoring
        fig.add_trace(
            go.Scatter(x=stream_data['time_seconds'], y=stream_data['voltage_V'],
                      mode='lines', name='Voltage', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Power monitoring
        fig.add_trace(
            go.Scatter(x=stream_data['time_seconds'], y=stream_data['power_W'],
                      mode='lines', name='Power', line=dict(color='red', width=1)),
            row=2, col=1
        )
        
        # Temperature monitoring
        fig.add_trace(
            go.Scatter(x=stream_data['time_seconds'], y=stream_data['temperature_C'],
                      mode='lines', name='Temperature', line=dict(color='orange', width=1)),
            row=3, col=1
        )
        
        # Add trigger events if available
        if not triggers.empty:
            for _, trigger in triggers.iterrows():
                trigger_time = trigger['time'] * 3600  # Convert to seconds
                if trigger_time <= stream_data['time_seconds'].max():
                    for row in range(1, 4):
                        fig.add_vline(x=trigger_time, row=row, col=1,
                                    line_dash="dash", line_color="gray",
                                    annotation_text=trigger['trigger_type'][:10])
        
        fig.update_xaxes(title_text="Time [seconds]", row=3, col=1)
        fig.update_yaxes(title_text="Voltage [V]", row=1, col=1)
        fig.update_yaxes(title_text="Power [W]", row=2, col=1)
        fig.update_yaxes(title_text="Temperature [°C]", row=3, col=1)
        
        fig.update_layout(
            title="Real-time Monitoring Stream Analysis",
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.show()
    
    def generate_statistical_summary(self):
        """
        Generate statistical summary of the entire dataset
        """
        print("=" * 80)
        print("SOFC Digital Twin Dataset - Statistical Summary")
        print("=" * 80)
        
        # Simulation data statistics
        print("\n1. SIMULATION DATA STATISTICS")
        print("-" * 40)
        
        with h5py.File(self.data_path / 'simulation' / 'high_fidelity_simulations.h5', 'r') as hf:
            n_sims = len(hf.keys())
            
            voltages = []
            max_stresses = []
            creep_damages = []
            
            for key in hf.keys():
                sim = hf[key]
                voltages.append(sim.attrs['voltage'])
                max_stresses.append(sim.attrs['max_stress'] / 1e6)  # MPa
                creep_damages.append(sim.attrs['creep_damage'])
            
            print(f"Number of simulations: {n_sims}")
            print(f"Voltage range: {np.min(voltages):.3f} - {np.max(voltages):.3f} V")
            print(f"Mean voltage: {np.mean(voltages):.3f} ± {np.std(voltages):.3f} V")
            print(f"Max stress range: {np.min(max_stresses):.1f} - {np.max(max_stresses):.1f} MPa")
            print(f"Mean max stress: {np.mean(max_stresses):.1f} ± {np.std(max_stresses):.1f} MPa")
            print(f"Creep damage range: {np.min(creep_damages):.4f} - {np.max(creep_damages):.4f}")
        
        # Experimental data statistics
        print("\n2. EXPERIMENTAL DATA STATISTICS")
        print("-" * 40)
        
        op_data = pd.read_csv(self.data_path / 'experimental' / 'operational_data.csv')
        print(f"Operational data points: {len(op_data)}")
        print(f"Duration: {op_data['time_hours'].max():.1f} hours")
        print(f"Voltage degradation: {op_data['voltage_V'].iloc[0]:.3f} → "
              f"{op_data['voltage_V'].iloc[-1]:.3f} V")
        print(f"Mean power: {op_data['power_W'].mean():.1f} ± {op_data['power_W'].std():.1f} W")
        
        # Thermal images
        thermal_images = np.load(self.data_path / 'experimental' / 'thermal_images.npy')
        print(f"Thermal images: {len(thermal_images)} images of size {thermal_images[0].shape}")
        print(f"Temperature range: {thermal_images.min():.1f} - {thermal_images.max():.1f} °C")
        
        # AE events
        ae_events = pd.read_csv(self.data_path / 'experimental' / 'acoustic_emission_events.csv')
        if not ae_events.empty:
            print(f"Acoustic emission events: {len(ae_events)}")
            print(f"Event types: {ae_events['event_type'].value_counts().to_dict()}")
        
        # Monitoring data statistics
        print("\n3. MONITORING DATA STATISTICS")
        print("-" * 40)
        
        stream_data = pd.read_csv(self.data_path / 'monitoring' / 'realtime_stream.csv')
        print(f"Real-time stream samples: {len(stream_data)}")
        print(f"Stream duration: {len(stream_data)} seconds")
        print(f"Mean update frequency: 1.0 Hz")
        
        # Data splits
        print("\n4. DATA SPLITS")
        print("-" * 40)
        
        with open(self.data_path / 'data_splits.json', 'r') as f:
            splits = json.load(f)
        
        print(f"Training samples: {len(splits['train'])}")
        print(f"Validation samples: {len(splits['validation'])}")
        print(f"Test samples: {len(splits['test'])}")
        
        print("\n" + "=" * 80)


def main():
    """
    Main function to run all visualizations
    """
    # Assuming data is in ../data directory
    data_path = Path("../data")
    
    if not data_path.exists():
        print(f"Data directory not found at {data_path}")
        print("Please run the dataset generator first.")
        return
    
    # Initialize visualizer
    viz = SOFCDataVisualizer(data_path)
    
    print("SOFC Digital Twin Dataset Visualization")
    print("=" * 50)
    
    # Run visualizations
    print("\n1. Generating statistical summary...")
    viz.generate_statistical_summary()
    
    print("\n2. Visualizing simulation fields...")
    viz.visualize_simulation_fields(sim_id=0)
    
    print("\n3. Creating 3D field visualization...")
    viz.create_3d_field_visualization(sim_id=0, field='temperature')
    
    print("\n4. Analyzing experimental data...")
    viz.analyze_experimental_data()
    
    print("\n5. Visualizing EIS data...")
    viz.visualize_eis_data()
    
    print("\n6. Displaying thermal images...")
    viz.visualize_thermal_images()
    
    print("\n7. Analyzing monitoring streams...")
    viz.analyze_monitoring_streams()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()