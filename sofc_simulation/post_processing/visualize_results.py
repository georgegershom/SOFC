#!/usr/bin/env python
"""
Visualization script for SOFC simulation results
Creates plots and contour maps from Abaqus ODB data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd
from odbAccess import *
from abaqusConstants import *
import os

class SOFCVisualizer:
    """Visualization tools for SOFC analysis results"""
    
    def __init__(self, odb_path):
        """Initialize visualizer with ODB file"""
        self.odb = openOdb(path=odb_path, readOnly=True)
        self.assembly = self.odb.rootAssembly
        self.instance = self.assembly.instances['SOFC_CELL-1']
        
        # Layer colors for visualization
        self.layer_colors = {
            'anode': '#FF6B6B',
            'electrolyte': '#4ECDC4',
            'cathode': '#45B7D1',
            'interconnect': '#96CEB4'
        }
        
        # Create output directory
        self.output_dir = 'visualization_results'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_temperature_evolution(self, step_name='Heat_Transfer'):
        """Plot temperature evolution at different locations"""
        
        step = self.odb.steps[step_name]
        
        # Define monitoring points
        monitor_points = {
            'Bottom (Y=0)': 0.0,
            'Anode-Electrolyte': 0.0004,
            'Electrolyte-Cathode': 0.0005,
            'Cathode-Interconnect': 0.0009,
            'Top (Y=1mm)': 0.001
        }
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Temperature vs time
        for location, y_coord in monitor_points.items():
            temps = []
            times = []
            
            for frame in step.frames:
                time = frame.frameValue / 60.0  # Convert to minutes
                temp_field = frame.fieldOutputs['NT']
                
                # Find node closest to monitoring point
                min_dist = float('inf')
                temp_val = None
                
                for value in temp_field.values:
                    node = self.instance.nodes[value.nodeLabel-1]
                    if abs(node.coordinates[1] - y_coord) < min_dist:
                        min_dist = abs(node.coordinates[1] - y_coord)
                        temp_val = value.data - 273.15  # Convert to Celsius
                
                if temp_val is not None:
                    temps.append(temp_val)
                    times.append(time)
            
            ax1.plot(times, temps, label=location, linewidth=2)
        
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.set_title('Temperature Evolution at Different Locations', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Temperature gradient through thickness at peak temperature
        last_frame = step.frames[-1]
        temp_field = last_frame.fieldOutputs['NT']
        
        y_coords = []
        temps_final = []
        
        for value in temp_field.values:
            node = self.instance.nodes[value.nodeLabel-1]
            if abs(node.coordinates[0] - 0.005) < 0.0001:  # Center line
                y_coords.append(node.coordinates[1] * 1000)  # Convert to mm
                temps_final.append(value.data - 273.15)
        
        # Sort by y-coordinate
        sorted_data = sorted(zip(y_coords, temps_final))
        y_coords, temps_final = zip(*sorted_data)
        
        ax2.plot(y_coords, temps_final, 'b-', linewidth=2)
        
        # Add layer boundaries
        for y in [0.4, 0.5, 0.9]:
            ax2.axvline(x=y, color='gray', linestyle='--', alpha=0.5)
        
        # Add layer labels
        ax2.text(0.2, min(temps_final) + 5, 'Anode', ha='center', fontsize=10)
        ax2.text(0.45, min(temps_final) + 5, 'Elyte', ha='center', fontsize=10)
        ax2.text(0.7, min(temps_final) + 5, 'Cathode', ha='center', fontsize=10)
        ax2.text(0.95, min(temps_final) + 5, 'IC', ha='center', fontsize=10)
        
        ax2.set_xlabel('Thickness (mm)', fontsize=12)
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('Temperature Distribution Through Thickness (Final)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temperature_evolution.png'), dpi=150)
        plt.show()
    
    def plot_stress_distribution(self, step_name='Thermo_Mechanical', frame_idx=-1):
        """Plot stress distribution contours"""
        
        step = self.odb.steps[step_name]
        frame = step.frames[frame_idx]
        
        # Get stress field
        stress_field = frame.fieldOutputs['S']
        mises_field = frame.fieldOutputs['MISES']
        
        # Create mesh grid
        x_coords = []
        y_coords = []
        mises_values = []
        s11_values = []
        s22_values = []
        s12_values = []
        
        for value in mises_field.values:
            element = self.instance.elements[value.elementLabel-1]
            nodes = [self.instance.nodes[n-1] for n in element.connectivity]
            
            # Get element centroid
            centroid_x = np.mean([node.coordinates[0] for node in nodes]) * 1000  # mm
            centroid_y = np.mean([node.coordinates[1] for node in nodes]) * 1000  # mm
            
            x_coords.append(centroid_x)
            y_coords.append(centroid_y)
            mises_values.append(value.data / 1e6)  # Convert to MPa
        
        for value in stress_field.values:
            s11_values.append(value.data[0] / 1e6)  # S11 in MPa
            s22_values.append(value.data[1] / 1e6)  # S22 in MPa
            s12_values.append(value.data[2] / 1e6)  # S12 in MPa
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Define stress components
        stress_components = [
            (mises_values, 'Von Mises Stress (MPa)', 'hot'),
            (s11_values, 'S11 - Normal Stress X (MPa)', 'RdBu_r'),
            (s22_values, 'S22 - Normal Stress Y (MPa)', 'RdBu_r'),
            (s12_values, 'S12 - Shear Stress (MPa)', 'PiYG')
        ]
        
        for ax, (values, title, cmap) in zip(axes.flat, stress_components):
            # Create scatter plot with interpolation
            scatter = ax.tricontourf(x_coords, y_coords, values, levels=50, cmap=cmap)
            
            # Add layer boundaries
            for y in [0.4, 0.5, 0.9]:
                ax.axhline(y=y, color='black', linestyle='-', linewidth=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(title.split('(')[1].strip(')'), fontsize=10)
            
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
        
        plt.suptitle(f'Stress Distribution at t = {frame.frameValue:.1f} s', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stress_distribution.png'), dpi=150)
        plt.show()
    
    def plot_strain_evolution(self, step_name='Thermo_Mechanical'):
        """Plot strain evolution over time"""
        
        step = self.odb.steps[step_name]
        
        # Track strain metrics over time
        time_data = []
        max_peeq = []  # Plastic strain
        max_ceeq = []  # Creep strain
        max_total = []  # Total strain
        
        for frame in step.frames:
            time = frame.frameValue / 60.0  # Convert to minutes
            
            # Get strain fields
            if 'PEEQ' in frame.fieldOutputs:
                peeq_field = frame.fieldOutputs['PEEQ']
                max_peeq_val = max([v.data for v in peeq_field.values])
            else:
                max_peeq_val = 0.0
            
            if 'CEEQ' in frame.fieldOutputs:
                ceeq_field = frame.fieldOutputs['CEEQ']
                max_ceeq_val = max([v.data for v in ceeq_field.values])
            else:
                max_ceeq_val = 0.0
            
            if 'E' in frame.fieldOutputs:
                strain_field = frame.fieldOutputs['E']
                # Calculate equivalent strain
                max_eeq = 0.0
                for value in strain_field.values:
                    e11, e22, e12 = value.data[0], value.data[1], value.data[2]
                    eeq = np.sqrt(2/3 * (e11**2 + e22**2 + 2*e12**2))
                    max_eeq = max(max_eeq, eeq)
                max_total.append(max_eeq)
            else:
                max_total.append(max_peeq_val + max_ceeq_val)
            
            time_data.append(time)
            max_peeq.append(max_peeq_val)
            max_ceeq.append(max_ceeq_val)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot strain components
        ax1.plot(time_data, np.array(max_peeq)*100, 'r-', label='Plastic Strain', linewidth=2)
        ax1.plot(time_data, np.array(max_ceeq)*100, 'b-', label='Creep Strain', linewidth=2)
        ax1.plot(time_data, np.array(max_total)*100, 'k--', label='Total Strain', linewidth=2)
        
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('Maximum Strain (%)', fontsize=12)
        ax1.set_title('Strain Evolution During Thermal Cycle', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot strain rate
        if len(time_data) > 1:
            strain_rate_peeq = np.diff(max_peeq) / np.diff(np.array(time_data)*60)  # per second
            strain_rate_ceeq = np.diff(max_ceeq) / np.diff(np.array(time_data)*60)
            
            ax2.semilogy(time_data[1:], np.abs(strain_rate_peeq), 'r-', label='Plastic Strain Rate', linewidth=2)
            ax2.semilogy(time_data[1:], np.abs(strain_rate_ceeq), 'b-', label='Creep Strain Rate', linewidth=2)
            
            ax2.set_xlabel('Time (minutes)', fontsize=12)
            ax2.set_ylabel('Strain Rate (1/s)', fontsize=12)
            ax2.set_title('Strain Rate Evolution', fontsize=14, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strain_evolution.png'), dpi=150)
        plt.show()
    
    def plot_interface_analysis(self, step_name='Thermo_Mechanical'):
        """Analyze and plot interface behavior"""
        
        step = self.odb.steps[step_name]
        
        # Interface positions
        interfaces = {
            'Anode-Electrolyte': 0.0004,
            'Electrolyte-Cathode': 0.0005,
            'Cathode-Interconnect': 0.0009
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (interface_name, y_pos) in enumerate(interfaces.items()):
            ax_idx = idx if idx < 2 else idx - 1
            ax = axes.flat[ax_idx]
            
            # Collect interface data
            x_positions = []
            shear_stress = []
            normal_stress = []
            
            last_frame = step.frames[-1]
            stress_field = last_frame.fieldOutputs['S']
            
            for value in stress_field.values:
                element = self.instance.elements[value.elementLabel-1]
                nodes = [self.instance.nodes[n-1] for n in element.connectivity]
                centroid_y = np.mean([node.coordinates[1] for node in nodes])
                
                # Check if near interface
                if abs(centroid_y - y_pos) < 0.00002:  # Within 20 microns
                    centroid_x = np.mean([node.coordinates[0] for node in nodes]) * 1000  # mm
                    x_positions.append(centroid_x)
                    shear_stress.append(abs(value.data[2]) / 1e6)  # S12 in MPa
                    normal_stress.append(value.data[1] / 1e6)  # S22 in MPa
            
            if x_positions:
                # Sort by x position
                sorted_data = sorted(zip(x_positions, shear_stress, normal_stress))
                x_positions, shear_stress, normal_stress = zip(*sorted_data)
                
                # Plot shear stress
                ax.plot(x_positions, shear_stress, 'r-', label='Shear |S12|', linewidth=2)
                ax.plot(x_positions, normal_stress, 'b--', label='Normal S22', linewidth=2)
                
                # Add critical threshold
                if interface_name == 'Anode-Electrolyte':
                    ax.axhline(y=25, color='red', linestyle=':', label='Critical Shear')
                elif interface_name == 'Electrolyte-Cathode':
                    ax.axhline(y=20, color='red', linestyle=':', label='Critical Shear')
                elif interface_name == 'Cathode-Interconnect':
                    ax.axhline(y=30, color='red', linestyle=':', label='Critical Shear')
                
                ax.set_xlabel('Position X (mm)', fontsize=10)
                ax.set_ylabel('Stress (MPa)', fontsize=10)
                ax.set_title(f'{interface_name} Interface', fontsize=12, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
        
        # Use the fourth subplot for summary
        ax = axes.flat[3]
        
        # Summary bar chart
        max_shears = []
        interface_names = []
        critical_values = [25, 20, 30]
        
        for (interface_name, y_pos), crit_val in zip(interfaces.items(), critical_values):
            interface_names.append(interface_name.replace('-', '\n'))
            
            # Find maximum shear at this interface
            max_shear = 0.0
            for value in stress_field.values:
                element = self.instance.elements[value.elementLabel-1]
                nodes = [self.instance.nodes[n-1] for n in element.connectivity]
                centroid_y = np.mean([node.coordinates[1] for node in nodes])
                
                if abs(centroid_y - y_pos) < 0.00002:
                    max_shear = max(max_shear, abs(value.data[2]) / 1e6)
            
            max_shears.append(max_shear)
        
        x_pos = np.arange(len(interface_names))
        bars = ax.bar(x_pos, max_shears, color=['green' if s < c else 'red' 
                                                 for s, c in zip(max_shears, critical_values)])
        
        # Add critical thresholds
        for i, (crit, name) in enumerate(zip(critical_values, interface_names)):
            ax.plot([i-0.4, i+0.4], [crit, crit], 'k--', linewidth=2)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(interface_names)
        ax.set_ylabel('Maximum Shear Stress (MPa)', fontsize=10)
        ax.set_title('Interface Integrity Summary', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Interface Stress Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'interface_analysis.png'), dpi=150)
        plt.show()
    
    def export_data_to_csv(self):
        """Export key data to CSV files for further analysis"""
        
        print("Exporting data to CSV files...")
        
        # Export nodal coordinates
        nodes_data = []
        for node in self.instance.nodes:
            nodes_data.append({
                'node_id': node.label,
                'x': node.coordinates[0],
                'y': node.coordinates[1],
                'z': node.coordinates[2] if len(node.coordinates) > 2 else 0
            })
        
        df_nodes = pd.DataFrame(nodes_data)
        df_nodes.to_csv(os.path.join(self.output_dir, 'nodes.csv'), index=False)
        
        # Export element connectivity
        elements_data = []
        for element in self.instance.elements:
            elements_data.append({
                'element_id': element.label,
                'type': element.type,
                'nodes': ','.join(map(str, element.connectivity))
            })
        
        df_elements = pd.DataFrame(elements_data)
        df_elements.to_csv(os.path.join(self.output_dir, 'elements.csv'), index=False)
        
        # Export final stress state
        step = self.odb.steps['Thermo_Mechanical']
        last_frame = step.frames[-1]
        
        stress_field = last_frame.fieldOutputs['S']
        stress_data = []
        
        for value in stress_field.values:
            stress_data.append({
                'element_id': value.elementLabel,
                'S11': value.data[0],
                'S22': value.data[1],
                'S12': value.data[2],
                'Mises': np.sqrt(0.5 * ((value.data[0]-value.data[1])**2 + 
                                       value.data[1]**2 + 
                                       value.data[0]**2 + 
                                       6*value.data[2]**2))
            })
        
        df_stress = pd.DataFrame(stress_data)
        df_stress.to_csv(os.path.join(self.output_dir, 'final_stress.csv'), index=False)
        
        print(f"Data exported to {self.output_dir}/")
    
    def close(self):
        """Close ODB file"""
        self.odb.close()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SOFC Results Visualization')
    parser.add_argument('odb_file', help='Path to Abaqus ODB file')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--temp', action='store_true', help='Plot temperature evolution')
    parser.add_argument('--stress', action='store_true', help='Plot stress distribution')
    parser.add_argument('--strain', action='store_true', help='Plot strain evolution')
    parser.add_argument('--interface', action='store_true', help='Plot interface analysis')
    parser.add_argument('--export', action='store_true', help='Export data to CSV')
    
    args = parser.parse_args()
    
    # Check if ODB file exists
    if not os.path.exists(args.odb_file):
        print(f"Error: ODB file '{args.odb_file}' not found")
        return
    
    # Create visualizer
    viz = SOFCVisualizer(args.odb_file)
    
    # Generate requested plots
    if args.all or args.temp:
        print("Generating temperature plots...")
        viz.plot_temperature_evolution()
    
    if args.all or args.stress:
        print("Generating stress plots...")
        viz.plot_stress_distribution()
    
    if args.all or args.strain:
        print("Generating strain plots...")
        viz.plot_strain_evolution()
    
    if args.all or args.interface:
        print("Generating interface plots...")
        viz.plot_interface_analysis()
    
    if args.all or args.export:
        print("Exporting data...")
        viz.export_data_to_csv()
    
    viz.close()
    print("Visualization complete!")


if __name__ == '__main__':
    main()