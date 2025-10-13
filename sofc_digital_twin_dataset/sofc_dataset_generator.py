"""
Main SOFC Digital Twin Dataset Generator

This script orchestrates the generation of the complete dataset for
Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural 
Integrity Monitoring.
"""

import os
import sys
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add generators to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'generators'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from generators import (
    MaterialsGenerator, 
    OperationalGenerator, 
    ThermoStructuralGenerator,
    DegradationGenerator,
    SensorGenerator
)
from utils import DataUtils, PhysicsUtils, VisualizationUtils


class SOFCDatasetGenerator:
    """
    Main class for generating the complete SOFC digital twin dataset.
    
    This class orchestrates the generation of all dataset components:
    - Materials & Geometry Data
    - Operational & Electrochemical Data
    - Thermo-Structural Field Data
    - Degradation & Failure Mode Data
    - Synthetic Sensor Data
    """
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        """
        Initialize the dataset generator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = "data"
        self._ensure_output_dirs()
        
        # Initialize generators
        self.materials_generator = MaterialsGenerator(config_path)
        self.operational_generator = OperationalGenerator(config_path)
        self.thermo_structural_generator = ThermoStructuralGenerator(config_path)
        self.degradation_generator = DegradationGenerator(config_path)
        self.sensor_generator = SensorGenerator(config_path)
        
        # Initialize utilities
        self.data_utils = DataUtils()
        self.physics_utils = PhysicsUtils()
        self.viz_utils = VisualizationUtils()
        
        # Dataset storage
        self.dataset = {}
        self.metadata = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/materials_geometry", exist_ok=True)
        os.makedirs(f"{self.output_dir}/operational_electrochemical", exist_ok=True)
        os.makedirs(f"{self.output_dir}/thermo_structural", exist_ok=True)
        os.makedirs(f"{self.output_dir}/degradation_failure", exist_ok=True)
        os.makedirs(f"{self.output_dir}/synthetic_sensors", exist_ok=True)
        os.makedirs(f"{self.output_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
    
    def generate_complete_dataset(self, 
                                generate_visualizations: bool = True,
                                generate_reports: bool = True) -> Dict[str, Any]:
        """
        Generate the complete SOFC digital twin dataset.
        
        Args:
            generate_visualizations: Whether to generate visualization plots
            generate_reports: Whether to generate summary reports
        
        Returns:
            Complete dataset dictionary
        """
        print("=" * 80)
        print("SOFC Digital Twin Dataset Generation")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Generate each dataset component
        print("1. Generating Materials & Geometry Data...")
        self.dataset['materials_geometry'] = self.materials_generator.generate()
        self._save_component_data('materials_geometry', self.dataset['materials_geometry'])
        
        print("\n2. Generating Operational & Electrochemical Data...")
        self.dataset['operational_electrochemical'] = self.operational_generator.generate()
        self._save_component_data('operational_electrochemical', self.dataset['operational_electrochemical'])
        
        print("\n3. Generating Thermo-Structural Field Data...")
        self.dataset['thermo_structural'] = self.thermo_structural_generator.generate()
        self._save_component_data('thermo_structural', self.dataset['thermo_structural'])
        
        print("\n4. Generating Degradation & Failure Mode Data...")
        self.dataset['degradation_failure'] = self.degradation_generator.generate()
        self._save_component_data('degradation_failure', self.dataset['degradation_failure'])
        
        print("\n5. Generating Synthetic Sensor Data...")
        self.dataset['synthetic_sensors'] = self.sensor_generator.generate()
        self._save_component_data('synthetic_sensors', self.dataset['synthetic_sensors'])
        
        # Generate integrated dataset
        print("\n6. Integrating Dataset Components...")
        self._integrate_dataset()
        
        # Generate visualizations
        if generate_visualizations:
            print("\n7. Generating Visualizations...")
            self._generate_visualizations()
        
        # Generate reports
        if generate_reports:
            print("\n8. Generating Reports...")
            self._generate_reports()
        
        # Save complete dataset
        print("\n9. Saving Complete Dataset...")
        self._save_complete_dataset()
        
        print("\n" + "=" * 80)
        print("Dataset Generation Complete!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        
        return self.dataset
    
    def _save_component_data(self, component_name: str, data: Dict[str, Any]):
        """Save individual component data."""
        filepath = os.path.join(self.output_dir, f"{component_name}/{component_name}_data.h5")
        
        with h5py.File(filepath, 'w') as f:
            self._save_dict_to_h5(data, f)
            
            # Add metadata
            f.attrs['component'] = component_name
            f.attrs['generated_at'] = datetime.now().isoformat()
            f.attrs['generator_version'] = '1.0.0'
        
        print(f"  Saved {component_name} data to: {filepath}")
    
    def _save_dict_to_h5(self, data: Dict[str, Any], group: h5py.Group):
        """Recursively save dictionary to HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_dict_to_h5(value, subgroup)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip')
            else:
                group.attrs[key] = value
    
    def _integrate_dataset(self):
        """Integrate all dataset components into a unified format."""
        print("  Integrating dataset components...")
        
        # Create integrated dataset structure
        self.dataset['integrated'] = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'config': self.config,
                'components': list(self.dataset.keys())
            },
            'time_series': self._create_time_series_data(),
            'spatial_data': self._create_spatial_data(),
            'sensor_network': self._create_sensor_network_data(),
            'degradation_tracking': self._create_degradation_tracking_data()
        }
        
        # Validate integrated dataset
        self._validate_integrated_dataset()
        
        print("  Dataset integration complete.")
    
    def _create_time_series_data(self) -> Dict[str, Any]:
        """Create integrated time series data."""
        time_series = {}
        
        # Get time points from first available component
        time_points = None
        for component in self.dataset.values():
            if isinstance(component, dict) and 'time_points' in component:
                time_points = component['time_points']
                break
        
        if time_points is None:
            time_points = np.arange(0, self.config['generation']['time_duration'], 
                                  self.config['generation']['time_step'])
        
        time_series['time_points'] = time_points
        
        # Collect all time series data
        for component_name, component_data in self.dataset.items():
            if isinstance(component_data, dict):
                for key, value in component_data.items():
                    if isinstance(value, dict) and 'time_points' in value:
                        # This is a time series
                        time_series[f"{component_name}_{key}"] = value
        
        return time_series
    
    def _create_spatial_data(self) -> Dict[str, Any]:
        """Create integrated spatial data."""
        spatial_data = {}
        
        # Collect spatial data from thermo-structural component
        if 'thermo_structural' in self.dataset:
            thermo_data = self.dataset['thermo_structural']
            
            if 'temperature_fields' in thermo_data:
                spatial_data['temperature_field_2d'] = thermo_data['temperature_fields']['temperature_field_2d']
                spatial_data['temperature_field_3d'] = thermo_data['temperature_fields']['temperature_field_3d']
            
            if 'stress_strain' in thermo_data:
                spatial_data['stress_field'] = thermo_data['stress_strain']['stress_field']
        
        # Collect spatial data from materials component
        if 'materials_geometry' in self.dataset:
            materials_data = self.dataset['materials_geometry']
            
            if 'microstructural' in materials_data:
                spatial_data['microstructures'] = materials_data['microstructural']
        
        return spatial_data
    
    def _create_sensor_network_data(self) -> Dict[str, Any]:
        """Create integrated sensor network data."""
        sensor_network = {}
        
        # Collect sensor data
        if 'synthetic_sensors' in self.dataset:
            sensor_data = self.dataset['synthetic_sensors']
            
            for sensor_type, sensors in sensor_data.items():
                if isinstance(sensors, dict):
                    sensor_network[sensor_type] = sensors
        
        return sensor_network
    
    def _create_degradation_tracking_data(self) -> Dict[str, Any]:
        """Create integrated degradation tracking data."""
        degradation_tracking = {}
        
        # Collect degradation data
        if 'degradation_failure' in self.dataset:
            degradation_data = self.dataset['degradation_failure']
            
            for degradation_type, data in degradation_data.items():
                if isinstance(data, dict):
                    degradation_tracking[degradation_type] = data
        
        return degradation_tracking
    
    def _validate_integrated_dataset(self):
        """Validate the integrated dataset."""
        print("  Validating integrated dataset...")
        
        # Check for required components
        required_components = ['materials_geometry', 'operational_electrochemical', 
                              'thermo_structural', 'degradation_failure', 'synthetic_sensors']
        
        for component in required_components:
            if component not in self.dataset:
                print(f"  Warning: Missing component: {component}")
        
        # Validate data quality
        quality_metrics = self.data_utils.validate_data_quality(self.dataset['integrated'])
        
        # Check for invalid values
        invalid_count = 0
        for component, metrics in quality_metrics.items():
            if metrics['missing_values'] > 0 or metrics['infinite_values'] > 0:
                invalid_count += 1
                print(f"  Warning: Invalid values found in {component}")
        
        if invalid_count == 0:
            print("  Dataset validation passed.")
        else:
            print(f"  Dataset validation completed with {invalid_count} warnings.")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        print("  Generating visualization plots...")
        
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        # Temperature field plots
        if 'thermo_structural' in self.dataset:
            thermo_data = self.dataset['thermo_structural']
            
            if 'temperature_fields' in thermo_data:
                # 2D temperature field
                fig = self.viz_utils.plot_temperature_field(
                    thermo_data['temperature_fields'], 
                    time_index=0
                )
                fig.savefig(os.path.join(viz_dir, 'temperature_field_2d.png'))
                plt.close(fig)
                
                # 3D temperature field
                fig = self.viz_utils.plot_3d_temperature_field(
                    thermo_data['temperature_fields'], 
                    time_index=0
                )
                fig.savefig(os.path.join(viz_dir, 'temperature_field_3d.png'))
                plt.close(fig)
        
        # Thermocouple data plots
        if 'synthetic_sensors' in self.dataset:
            sensor_data = self.dataset['synthetic_sensors']
            
            if 'thermocouples' in sensor_data:
                fig = self.viz_utils.plot_thermocouple_data(sensor_data['thermocouples'])
                fig.savefig(os.path.join(viz_dir, 'thermocouple_data.png'))
                plt.close(fig)
            
            if 'strain_gauges' in sensor_data:
                fig = self.viz_utils.plot_strain_data(sensor_data['strain_gauges'])
                fig.savefig(os.path.join(viz_dir, 'strain_data.png'))
                plt.close(fig)
            
            if 'voltage_current' in sensor_data:
                fig = self.viz_utils.plot_voltage_current_data(sensor_data['voltage_current'])
                fig.savefig(os.path.join(viz_dir, 'voltage_current_data.png'))
                plt.close(fig)
        
        # EIS spectra plots
        if 'operational_electrochemical' in self.dataset:
            op_data = self.dataset['operational_electrochemical']
            
            if 'electrochemical_response' in op_data and 'eis_spectra' in op_data['electrochemical_response']:
                fig = self.viz_utils.plot_eis_spectra(op_data['electrochemical_response']['eis_spectra'])
                fig.savefig(os.path.join(viz_dir, 'eis_spectra.png'))
                plt.close(fig)
        
        # Degradation plots
        if 'degradation_failure' in self.dataset:
            deg_data = self.dataset['degradation_failure']
            
            if 'accelerated_aging' in deg_data:
                for aging_type, aging_data in deg_data['accelerated_aging'].items():
                    fig = self.viz_utils.plot_degradation_data(aging_data)
                    fig.suptitle(f'{aging_type.replace("_", " ").title()} Degradation')
                    fig.savefig(os.path.join(viz_dir, f'degradation_{aging_type}.png'))
                    plt.close(fig)
        
        # Microstructure plots
        if 'materials_geometry' in self.dataset:
            materials_data = self.dataset['materials_geometry']
            
            if 'microstructural' in materials_data:
                for component, micro_data in materials_data['microstructural'].items():
                    fig = self.viz_utils.plot_microstructure(
                        materials_data['microstructural'], 
                        component=component
                    )
                    fig.savefig(os.path.join(viz_dir, f'microstructure_{component}.png'))
                    plt.close(fig)
        
        # Sensor location plots
        if 'synthetic_sensors' in self.dataset:
            sensor_data = self.dataset['synthetic_sensors']
            
            if 'thermocouples' in sensor_data:
                fig = self.viz_utils.plot_sensor_locations(sensor_data['thermocouples'], 'thermocouple')
                fig.savefig(os.path.join(viz_dir, 'thermocouple_locations.png'))
                plt.close(fig)
            
            if 'strain_gauges' in sensor_data:
                fig = self.viz_utils.plot_sensor_locations(sensor_data['strain_gauges'], 'strain_gauge')
                fig.savefig(os.path.join(viz_dir, 'strain_gauge_locations.png'))
                plt.close(fig)
        
        # Interactive dashboard
        if 'integrated' in self.dataset:
            dashboard = self.viz_utils.create_interactive_dashboard(self.dataset)
            dashboard.write_html(os.path.join(viz_dir, 'interactive_dashboard.html'))
        
        print(f"  Visualizations saved to: {viz_dir}")
    
    def _generate_reports(self):
        """Generate summary reports."""
        print("  Generating summary reports...")
        
        reports_dir = os.path.join(self.output_dir, 'reports')
        
        # Dataset summary report
        self._generate_dataset_summary_report(reports_dir)
        
        # Data quality report
        self._generate_data_quality_report(reports_dir)
        
        # Sensor network report
        self._generate_sensor_network_report(reports_dir)
        
        # Degradation analysis report
        self._generate_degradation_analysis_report(reports_dir)
        
        print(f"  Reports saved to: {reports_dir}")
    
    def _generate_dataset_summary_report(self, reports_dir: str):
        """Generate dataset summary report."""
        report_path = os.path.join(reports_dir, 'dataset_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("SOFC Digital Twin Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config_path}\n\n")
            
            f.write("Dataset Components:\n")
            for component, data in self.dataset.items():
                f.write(f"  - {component}: {type(data).__name__}\n")
            
            f.write("\nData Statistics:\n")
            summary = self.data_utils.create_data_summary(self.dataset['integrated'])
            for key, value in summary.items():
                f.write(f"  - {key}: {value}\n")
    
    def _generate_data_quality_report(self, reports_dir: str):
        """Generate data quality report."""
        report_path = os.path.join(reports_dir, 'data_quality.txt')
        
        with open(report_path, 'w') as f:
            f.write("Data Quality Report\n")
            f.write("=" * 30 + "\n\n")
            
            quality_metrics = self.data_utils.validate_data_quality(self.dataset['integrated'])
            
            for component, metrics in quality_metrics.items():
                f.write(f"{component}:\n")
                for metric, value in metrics.items():
                    f.write(f"  - {metric}: {value}\n")
                f.write("\n")
    
    def _generate_sensor_network_report(self, reports_dir: str):
        """Generate sensor network report."""
        report_path = os.path.join(reports_dir, 'sensor_network.txt')
        
        with open(report_path, 'w') as f:
            f.write("Sensor Network Report\n")
            f.write("=" * 30 + "\n\n")
            
            if 'synthetic_sensors' in self.dataset:
                sensor_data = self.dataset['synthetic_sensors']
                
                for sensor_type, sensors in sensor_data.items():
                    f.write(f"{sensor_type}:\n")
                    if isinstance(sensors, dict):
                        f.write(f"  - Count: {len(sensors)}\n")
                        for sensor_id, sensor_info in sensors.items():
                            if 'location' in sensor_info:
                                loc = sensor_info['location']
                                f.write(f"    - {sensor_id}: {loc}\n")
                    f.write("\n")
    
    def _generate_degradation_analysis_report(self, reports_dir: str):
        """Generate degradation analysis report."""
        report_path = os.path.join(reports_dir, 'degradation_analysis.txt')
        
        with open(report_path, 'w') as f:
            f.write("Degradation Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            
            if 'degradation_failure' in self.dataset:
                deg_data = self.dataset['degradation_failure']
                
                for degradation_type, data in deg_data.items():
                    f.write(f"{degradation_type}:\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                f.write(f"  - {key}: {len(value)} items\n")
                            else:
                                f.write(f"  - {key}: {type(value).__name__}\n")
                    f.write("\n")
    
    def _save_complete_dataset(self):
        """Save the complete dataset."""
        # Save integrated dataset
        integrated_path = os.path.join(self.output_dir, 'integrated_dataset.h5')
        
        with h5py.File(integrated_path, 'w') as f:
            self._save_dict_to_h5(self.dataset['integrated'], f)
            
            # Add metadata
            f.attrs['dataset_name'] = 'SOFC Digital Twin Dataset'
            f.attrs['version'] = '1.0.0'
            f.attrs['generated_at'] = datetime.now().isoformat()
            f.attrs['description'] = 'Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring'
        
        print(f"  Complete dataset saved to: {integrated_path}")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'dataset_metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
        
        print(f"  Metadata saved to: {metadata_path}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the generated dataset."""
        info = {
            'components': list(self.dataset.keys()),
            'total_size': self._calculate_dataset_size(),
            'generation_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        return info
    
    def _calculate_dataset_size(self) -> str:
        """Calculate total dataset size."""
        total_size = 0
        
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
        
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024:
                return f"{total_size:.2f} {unit}"
            total_size /= 1024
        
        return f"{total_size:.2f} TB"


def main():
    """Main function to generate the complete dataset."""
    # Initialize generator
    generator = SOFCDatasetGenerator()
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(
        generate_visualizations=True,
        generate_reports=True
    )
    
    # Print dataset information
    info = generator.get_dataset_info()
    print("\nDataset Information:")
    print(f"  Components: {info['components']}")
    print(f"  Total Size: {info['total_size']}")
    print(f"  Generation Time: {info['generation_time']}")
    
    return dataset


if __name__ == "__main__":
    dataset = main()