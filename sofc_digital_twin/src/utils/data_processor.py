"""
Data Processing Utilities for SOFC Digital Twin Datasets
Provides tools for loading, processing, and analyzing the generated datasets
"""

import numpy as np
import pandas as pd
import h5py
import yaml
import os
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class SOFCDataProcessor:
    """
    Comprehensive data processor for SOFC digital twin datasets
    """
    
    def __init__(self, datasets_dir: str = "datasets"):
        """Initialize with datasets directory"""
        self.datasets_dir = datasets_dir
        self.dataset1_dir = os.path.join(datasets_dir, "dataset1_physics_simulation")
        self.dataset2_dir = os.path.join(datasets_dir, "dataset2_experimental")
        self.dataset3_dir = os.path.join(datasets_dir, "dataset3_realtime")
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def load_dataset1_summary(self) -> Dict:
        """Load Dataset 1 summary information"""
        summary_file = os.path.join(self.dataset1_dir, 'dataset_summary.yaml')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_dataset1_simulation(self, sim_id: int) -> Dict:
        """Load a specific simulation from Dataset 1"""
        filename = os.path.join(self.dataset1_dir, f"simulation_{sim_id:06d}.h5")
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Simulation file not found: {filename}")
        
        data = {}
        with h5py.File(filename, 'r') as f:
            # Load field data
            if 'fields' in f:
                data['fields'] = {}
                for field_name in f['fields'].keys():
                    data['fields'][field_name] = f['fields'][field_name][:]
            
            # Load parameters
            if 'parameters' in f:
                data['parameters'] = {}
                for key in f['parameters'].attrs.keys():
                    data['parameters'][key] = f['parameters'].attrs[key]
                
                # Load operating conditions
                if 'operating_conditions' in f['parameters']:
                    data['operating_conditions'] = {}
                    op_group = f['parameters']['operating_conditions']
                    for key in op_group.attrs.keys():
                        data['operating_conditions'][key] = op_group.attrs[key]
                    
                    # Load fuel composition
                    if 'fuel_composition' in op_group:
                        data['fuel_composition'] = {}
                        fuel_group = op_group['fuel_composition']
                        for key in fuel_group.attrs.keys():
                            data['fuel_composition'][key] = fuel_group.attrs[key]
                
                # Load material properties
                if 'material_properties' in f['parameters']:
                    data['material_properties'] = {}
                    mat_group = f['parameters']['material_properties']
                    for key in mat_group.attrs.keys():
                        data['material_properties'][key] = mat_group.attrs[key]
            
            # Load derived quantities
            if 'derived_quantities' in f:
                data['derived_quantities'] = {}
                for key in f['derived_quantities'].attrs.keys():
                    data['derived_quantities'][key] = f['derived_quantities'].attrs[key]
            
            # Load metadata
            if 'metadata' in f:
                data['metadata'] = {}
                for key in f['metadata'].attrs.keys():
                    data['metadata'][key] = f['metadata'].attrs[key]
        
        return data
    
    def load_dataset1_batch(self, sim_ids: Optional[List[int]] = None) -> List[Dict]:
        """Load multiple simulations from Dataset 1"""
        if sim_ids is None:
            # Load all available simulations
            sim_files = [f for f in os.listdir(self.dataset1_dir) if f.startswith('simulation_')]
            sim_ids = [int(f.split('_')[1].split('.')[0]) for f in sim_files]
        
        simulations = []
        for sim_id in sim_ids:
            try:
                sim_data = self.load_dataset1_simulation(sim_id)
                simulations.append(sim_data)
            except FileNotFoundError:
                print(f"Warning: Simulation {sim_id} not found, skipping...")
                continue
        
        return simulations
    
    def load_dataset2_operational(self) -> pd.DataFrame:
        """Load Dataset 2 operational data"""
        filename = os.path.join(self.dataset2_dir, 'operational_data.csv')
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return pd.DataFrame()
    
    def load_dataset2_strain(self) -> pd.DataFrame:
        """Load Dataset 2 strain gauge data"""
        filename = os.path.join(self.dataset2_dir, 'strain_gauge_data.csv')
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return pd.DataFrame()
    
    def load_dataset2_experimental(self) -> Dict:
        """Load Dataset 2 experimental data (EIS, thermal, etc.)"""
        filename = os.path.join(self.dataset2_dir, 'experimental_data.h5')
        
        if not os.path.exists(filename):
            return {}
        
        data = {}
        with h5py.File(filename, 'r') as f:
            # Load EIS data
            if 'eis_data' in f:
                data['eis'] = []
                eis_group = f['eis_data']
                for measurement_name in sorted(eis_group.keys()):
                    measurement = {}
                    measurement_group = eis_group[measurement_name]
                    for key in measurement_group.keys():
                        measurement[key] = measurement_group[key][:]
                    data['eis'].append(measurement)
            
            # Load thermal imaging data
            if 'thermal_imaging' in f:
                data['thermal'] = []
                thermal_group = f['thermal_imaging']
                for image_name in sorted(thermal_group.keys()):
                    image_data = {}
                    image_group = thermal_group[image_name]
                    for key in image_group.keys():
                        image_data[key] = image_group[key][:]
                    data['thermal'].append(image_data)
        
        return data
    
    def load_dataset3_realtime(self) -> Dict:
        """Load Dataset 3 real-time monitoring data"""
        data = {}
        
        # Load operational data
        op_file = os.path.join(self.dataset3_dir, 'realtime_operational_data.csv')
        if os.path.exists(op_file):
            data['operational'] = pd.read_csv(op_file)
        
        # Load events data
        events_file = os.path.join(self.dataset3_dir, 'acoustic_emission_events.csv')
        if os.path.exists(events_file):
            data['events'] = pd.read_csv(events_file)
        
        # Load monitoring data
        monitoring_file = os.path.join(self.dataset3_dir, 'realtime_monitoring.h5')
        if os.path.exists(monitoring_file):
            with h5py.File(monitoring_file, 'r') as f:
                # Load EIS measurements
                if 'eis_measurements' in f:
                    data['eis'] = []
                    eis_group = f['eis_measurements']
                    for measurement_name in sorted(eis_group.keys()):
                        measurement = {}
                        measurement_group = eis_group[measurement_name]
                        for key in measurement_group.keys():
                            measurement[key] = measurement_group[key][:]
                        data['eis'].append(measurement)
                
                # Load thermal measurements
                if 'thermal_measurements' in f:
                    data['thermal'] = []
                    thermal_group = f['thermal_measurements']
                    for image_name in sorted(thermal_group.keys()):
                        image_data = {}
                        image_group = thermal_group[image_name]
                        for key in image_group.keys():
                            image_data[key] = image_group[key][:]
                        data['thermal'].append(image_data)
        
        return data
    
    def extract_features_dataset1(self, simulations: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and targets from Dataset 1 for ML training
        """
        features = []
        targets = []
        
        for sim in simulations:
            # Input features (operating conditions + material properties)
            feature_vector = []
            
            # Operating conditions
            if 'operating_conditions' in sim:
                op_cond = sim['operating_conditions']
                feature_vector.extend([
                    op_cond.get('current_density', 0),
                    op_cond.get('fuel_utilization', 0),
                    op_cond.get('air_utilization', 0),
                    op_cond.get('inlet_fuel_temperature', 0),
                    op_cond.get('inlet_air_temperature', 0)
                ])
            
            # Fuel composition
            if 'fuel_composition' in sim:
                fuel_comp = sim['fuel_composition']
                feature_vector.extend([
                    fuel_comp.get('h2_percentage', 0),
                    fuel_comp.get('h2o_percentage', 0),
                    fuel_comp.get('co_percentage', 0),
                    fuel_comp.get('ch4_percentage', 0)
                ])
            
            # Material properties
            if 'material_properties' in sim:
                mat_props = sim['material_properties']
                feature_vector.extend([
                    mat_props.get('electrode_porosity', 0),
                    mat_props.get('electrode_tortuosity', 0),
                    mat_props.get('anode_conductivity', 0),
                    mat_props.get('cathode_conductivity', 0),
                    mat_props.get('electrolyte_thickness', 0),
                    mat_props.get('electrode_thickness', 0),
                    mat_props.get('initial_crack_length', 0),
                    mat_props.get('porosity_degradation', 0)
                ])
            
            # Target values (derived quantities + key field statistics)
            target_vector = []
            
            if 'derived_quantities' in sim:
                derived = sim['derived_quantities']
                target_vector.extend([
                    derived.get('cell_voltage', 0),
                    derived.get('max_temperature', 0),
                    derived.get('max_von_mises_stress', 0)
                ])
            
            # Add field statistics as targets
            if 'fields' in sim:
                fields = sim['fields']
                
                # Temperature statistics
                if 'temperature' in fields:
                    temp_field = fields['temperature']
                    target_vector.extend([
                        np.mean(temp_field),
                        np.std(temp_field),
                        np.max(temp_field),
                        np.min(temp_field)
                    ])
                
                # Stress statistics
                if 'von_mises_stress' in fields:
                    stress_field = fields['von_mises_stress']
                    target_vector.extend([
                        np.mean(stress_field),
                        np.std(stress_field),
                        np.max(stress_field),
                        np.percentile(stress_field, 95)
                    ])
            
            if len(feature_vector) > 0 and len(target_vector) > 0:
                features.append(feature_vector)
                targets.append(target_vector)
        
        return np.array(features), np.array(targets)
    
    def analyze_dataset1_statistics(self, simulations: List[Dict]) -> Dict:
        """Analyze statistical properties of Dataset 1"""
        
        features, targets = self.extract_features_dataset1(simulations)
        
        # Feature names
        feature_names = [
            'current_density', 'fuel_utilization', 'air_utilization',
            'inlet_fuel_temp', 'inlet_air_temp', 'h2_percentage',
            'h2o_percentage', 'co_percentage', 'ch4_percentage',
            'electrode_porosity', 'electrode_tortuosity', 'anode_conductivity',
            'cathode_conductivity', 'electrolyte_thickness', 'electrode_thickness',
            'initial_crack_length', 'porosity_degradation'
        ]
        
        target_names = [
            'cell_voltage', 'max_temperature', 'max_von_mises_stress',
            'temp_mean', 'temp_std', 'temp_max', 'temp_min',
            'stress_mean', 'stress_std', 'stress_max', 'stress_95th'
        ]
        
        # Calculate statistics
        stats_dict = {
            'n_simulations': len(simulations),
            'n_features': features.shape[1] if len(features) > 0 else 0,
            'n_targets': targets.shape[1] if len(targets) > 0 else 0,
            'feature_statistics': {},
            'target_statistics': {},
            'correlations': {}
        }
        
        if len(features) > 0:
            # Feature statistics
            for i, name in enumerate(feature_names[:features.shape[1]]):
                stats_dict['feature_statistics'][name] = {
                    'mean': float(np.mean(features[:, i])),
                    'std': float(np.std(features[:, i])),
                    'min': float(np.min(features[:, i])),
                    'max': float(np.max(features[:, i])),
                    'median': float(np.median(features[:, i]))
                }
            
            # Target statistics
            for i, name in enumerate(target_names[:targets.shape[1]]):
                stats_dict['target_statistics'][name] = {
                    'mean': float(np.mean(targets[:, i])),
                    'std': float(np.std(targets[:, i])),
                    'min': float(np.min(targets[:, i])),
                    'max': float(np.max(targets[:, i])),
                    'median': float(np.median(targets[:, i]))
                }
            
            # Correlation analysis
            if features.shape[1] > 1 and targets.shape[1] > 1:
                feature_corr = np.corrcoef(features.T)
                target_corr = np.corrcoef(targets.T)
                
                stats_dict['correlations']['feature_correlation_matrix'] = feature_corr.tolist()
                stats_dict['correlations']['target_correlation_matrix'] = target_corr.tolist()
        
        return stats_dict
    
    def create_dataset1_visualizations(self, simulations: List[Dict], output_dir: str = "visualizations"):
        """Create comprehensive visualizations for Dataset 1"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        features, targets = self.extract_features_dataset1(simulations)
        
        if len(features) == 0:
            print("No data available for visualization")
            return
        
        # 1. Feature distribution plots
        fig, axes = plt.subplots(3, 6, figsize=(24, 15))
        axes = axes.flatten()
        
        feature_names = [
            'Current Density', 'Fuel Utilization', 'Air Utilization',
            'Inlet Fuel Temp', 'Inlet Air Temp', 'H₂ %',
            'H₂O %', 'CO %', 'CH₄ %', 'Electrode Porosity',
            'Electrode Tortuosity', 'Anode Conductivity',
            'Cathode Conductivity', 'Electrolyte Thickness', 'Electrode Thickness',
            'Initial Crack Length', 'Porosity Degradation'
        ]
        
        for i in range(min(len(feature_names), features.shape[1])):
            axes[i].hist(features[:, i], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(feature_names[i])
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(features.shape[1], len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset1_feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Target distribution plots
        target_names = [
            'Cell Voltage', 'Max Temperature', 'Max von Mises Stress',
            'Temp Mean', 'Temp Std', 'Temp Max', 'Temp Min',
            'Stress Mean', 'Stress Std', 'Stress Max', 'Stress 95th'
        ]
        
        n_targets = min(len(target_names), targets.shape[1])
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i in range(n_targets):
            axes[i].hist(targets[:, i], bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[i].set_title(target_names[i])
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_targets, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset1_target_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        if features.shape[1] > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = np.corrcoef(features.T)
            sns.heatmap(corr_matrix, 
                       xticklabels=feature_names[:features.shape[1]], 
                       yticklabels=feature_names[:features.shape[1]],
                       annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dataset1_feature_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. PCA analysis
        if features.shape[0] > 10 and features.shape[1] > 2:
            pca = PCA(n_components=min(10, features.shape[1]))
            features_pca = pca.fit_transform(self.feature_scaler.fit_transform(features))
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(features_pca[:, 0], features_pca[:, 1], alpha=0.6)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA: First Two Components')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            plt.plot(range(len(cumsum_var)), cumsum_var, 'o-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dataset1_pca_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Dataset 1 visualizations saved to: {output_dir}")
    
    def create_interactive_dashboard(self, simulations: List[Dict]) -> go.Figure:
        """Create interactive Plotly dashboard for dataset exploration"""
        
        features, targets = self.extract_features_dataset1(simulations)
        
        if len(features) == 0:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Space (PCA)', 'Voltage vs Temperature', 
                           'Stress vs Current Density', 'Parameter Sensitivity'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # PCA plot
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(self.feature_scaler.fit_transform(features))
            
            fig.add_trace(
                go.Scatter(x=features_pca[:, 0], y=features_pca[:, 1],
                          mode='markers', name='Simulations',
                          marker=dict(color=targets[:, 0], colorscale='Viridis', 
                                    colorbar=dict(title="Cell Voltage")),
                          text=[f"Sim {i}" for i in range(len(features_pca))]),
                row=1, col=1
            )
        
        # Voltage vs Temperature
        if targets.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(x=targets[:, 1], y=targets[:, 0],
                          mode='markers', name='V vs T',
                          marker=dict(color='red', size=8)),
                row=1, col=2
            )
        
        # Stress vs Current Density
        if targets.shape[1] >= 3 and features.shape[1] >= 1:
            fig.add_trace(
                go.Scatter(x=features[:, 0], y=targets[:, 2],
                          mode='markers', name='Stress vs I',
                          marker=dict(color='blue', size=8)),
                row=2, col=1
            )
        
        # Parameter sensitivity (correlation with voltage)
        if targets.shape[1] >= 1:
            correlations = [np.corrcoef(features[:, i], targets[:, 0])[0, 1] 
                           for i in range(features.shape[1])]
            feature_names = ['Current', 'Fuel Util', 'Air Util', 'T_fuel', 'T_air'][:len(correlations)]
            
            fig.add_trace(
                go.Bar(x=feature_names, y=correlations,
                      name='Correlation with Voltage'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="SOFC Digital Twin Dataset Dashboard")
        return fig

def main():
    """Demonstrate data processing capabilities"""
    processor = SOFCDataProcessor()
    
    # Load and analyze Dataset 1 (if available)
    try:
        summary = processor.load_dataset1_summary()
        print("Dataset 1 Summary:")
        print(f"Total simulations: {summary.get('statistics', {}).get('total_simulations', 'N/A')}")
        
        # Load first few simulations for demonstration
        simulations = processor.load_dataset1_batch(sim_ids=[0, 1, 2])
        if simulations:
            stats = processor.analyze_dataset1_statistics(simulations)
            print(f"Loaded {len(simulations)} simulations")
            print(f"Features: {stats['n_features']}, Targets: {stats['n_targets']}")
            
            # Create visualizations
            processor.create_dataset1_visualizations(simulations)
            
    except Exception as e:
        print(f"Dataset 1 analysis failed: {e}")
    
    # Load Dataset 2 (if available)
    try:
        operational_data = processor.load_dataset2_operational()
        if not operational_data.empty:
            print(f"\nDataset 2 Operational Data: {len(operational_data)} points")
            
    except Exception as e:
        print(f"Dataset 2 analysis failed: {e}")
    
    # Load Dataset 3 (if available)
    try:
        realtime_data = processor.load_dataset3_realtime()
        if realtime_data:
            print(f"\nDataset 3 Real-time Data loaded")
            
    except Exception as e:
        print(f"Dataset 3 analysis failed: {e}")

if __name__ == "__main__":
    main()