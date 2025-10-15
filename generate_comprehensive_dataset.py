#!/usr/bin/env python3
"""
Comprehensive SOFC Residual Stress Dataset Generator
===================================================

This script generates a comprehensive dataset for residual stress prediction in SOFCs
by combining:

1. Design of Experiments (DOE) sampling
2. Analytical stress calculations (fast, for large datasets)
3. FEniCS FEA simulations (accurate, for validation subset)
4. Data augmentation and synthesis
5. Machine learning ready format

The dataset includes all parameters specified in the requirements:
- Geometric parameters (plate dimensions, layer thicknesses, green density)
- Material properties (Young's modulus, CTE, Poisson's ratio, sintering parameters)
- Process parameters (sintering temperature profiles, atmosphere)
- Simulation results (stress distributions, strain fields, fracture risk metrics)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import h5py
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# Import our custom modules
from sofc_dataset_generator import SOFCDatasetGenerator, GeometricParameters, MaterialProperties, SinteringParameters, CreepParameters
from fenics_simulation import SOFCFEASimulator

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSOFCDatasetGenerator:
    """
    Comprehensive dataset generator that combines analytical and FEA approaches
    """
    
    def __init__(self, output_dir: str = "comprehensive_sofc_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize generators
        self.analytical_generator = SOFCDatasetGenerator(output_dir / "analytical")
        self.fenics_simulator = SOFCFEASimulator()
        
        # Dataset storage
        self.analytical_dataset = None
        self.fenics_dataset = None
        self.combined_dataset = None
        
        # ML models
        self.stress_predictor = None
        self.fracture_risk_predictor = None
        
        logger.info(f"Comprehensive SOFC Dataset Generator initialized. Output: {self.output_dir}")
    
    def generate_analytical_dataset(self, n_samples: int = 5000, method: str = 'lhs') -> pd.DataFrame:
        """
        Generate large analytical dataset for initial exploration
        
        Args:
            n_samples: Number of samples to generate
            method: DOE sampling method
        
        Returns:
            Analytical dataset DataFrame
        """
        logger.info(f"Generating analytical dataset with {n_samples} samples")
        
        # Generate analytical dataset
        self.analytical_dataset = self.analytical_generator.generate_dataset(
            n_samples=n_samples, 
            method=method, 
            save_results=True
        )
        
        logger.info(f"Analytical dataset generated: {len(self.analytical_dataset)} samples")
        return self.analytical_dataset
    
    def generate_fenics_validation_dataset(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate FEA validation dataset using FEniCS
        
        Args:
            n_samples: Number of FEA samples to generate
        
        Returns:
            FEA validation dataset DataFrame
        """
        logger.info(f"Generating FEA validation dataset with {n_samples} samples")
        
        # Sample from analytical dataset for FEA validation
        if self.analytical_dataset is None:
            raise ValueError("Analytical dataset must be generated first")
        
        # Select diverse samples for FEA validation
        validation_samples = self._select_validation_samples(n_samples)
        
        # Run FEA simulations
        fenics_results = []
        for idx, sample in tqdm(validation_samples.iterrows(), 
                               total=len(validation_samples), 
                               desc="Running FEA simulations"):
            try:
                fenics_result = self._run_fenics_simulation(sample)
                fenics_results.append(fenics_result)
            except Exception as e:
                logger.error(f"FEA simulation failed for sample {idx}: {e}")
                continue
        
        self.fenics_dataset = pd.DataFrame(fenics_results)
        
        # Save FEA dataset
        self.fenics_dataset.to_csv(self.output_dir / "fenics_validation_dataset.csv", index=False)
        
        logger.info(f"FEA validation dataset generated: {len(self.fenics_dataset)} samples")
        return self.fenics_dataset
    
    def _select_validation_samples(self, n_samples: int) -> pd.DataFrame:
        """Select diverse samples for FEA validation"""
        # Use stratified sampling to ensure diversity
        from sklearn.cluster import KMeans
        
        # Select key parameters for clustering
        key_params = [
            'plate_length', 'plate_width', 'electrolyte_thickness',
            'electrolyte_E_25C', 'electrolyte_CTE_25C', 'max_temperature'
        ]
        
        X = self.analytical_dataset[key_params].values
        
        # Normalize for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster samples
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Select one sample from each cluster
        selected_indices = []
        for cluster_id in range(n_samples):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                # Select sample closest to cluster center
                cluster_samples = self.analytical_dataset[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                distances = np.linalg.norm(
                    scaler.transform(cluster_samples[key_params].values) - cluster_center, 
                    axis=1
                )
                closest_idx = cluster_samples.index[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return self.analytical_dataset.loc[selected_indices]
    
    def _run_fenics_simulation(self, sample: pd.Series) -> Dict:
        """Run FEA simulation for a single sample"""
        # Extract parameters
        dimensions = {
            'plate_length': sample['plate_length'],
            'plate_width': sample['plate_width'],
            'anode_thickness': sample['anode_thickness'],
            'electrolyte_thickness': sample['electrolyte_thickness'],
            'cathode_thickness': sample['cathode_thickness'],
            'interconnect_thickness': sample['interconnect_thickness']
        }
        
        # Create mesh
        mesh_data = self.fenics_simulator.create_sofc_mesh(dimensions, refinement_level=1)
        
        # Set material properties
        material_properties = {
            'anode': {
                'youngs_modulus': sample['anode_E_25C'] * 1e9,  # Convert GPa to Pa
                'poisson_ratio': 0.29,
                'cte': sample['anode_CTE_25C'],
                'thermal_conductivity': 6.0
            },
            'electrolyte': {
                'youngs_modulus': sample['electrolyte_E_25C'] * 1e9,
                'poisson_ratio': 0.23,
                'cte': sample['electrolyte_CTE_25C'],
                'thermal_conductivity': 2.1
            },
            'cathode': {
                'youngs_modulus': sample['cathode_E_25C'] * 1e9,
                'poisson_ratio': 0.25,
                'cte': sample['cathode_CTE_25C'],
                'thermal_conductivity': 3.5
            },
            'interconnect': {
                'youngs_modulus': sample['interconnect_E_25C'] * 1e9,
                'poisson_ratio': 0.30,
                'cte': sample['interconnect_CTE_25C'],
                'thermal_conductivity': 25.0
            }
        }
        
        self.fenics_simulator.set_material_properties(material_properties)
        
        # Thermal boundary conditions
        thermal_bc = {
            'temperature': sample['max_temperature'],
            'heat_generation': 1.5e6
        }
        
        # Solve thermal analysis
        temperature_field = self.fenics_simulator.solve_thermal_analysis(thermal_bc)
        
        # Solve mechanical analysis
        displacement_field, stress_field = self.fenics_simulator.solve_mechanical_analysis(
            temperature_field, constitutive_model='elastic'
        )
        
        # Calculate stress metrics
        stress_metrics = self.fenics_simulator.calculate_stress_metrics(stress_field)
        
        # Assess fracture risk
        fracture_assessment = self.fenics_simulator.assess_fracture_risk(stress_metrics)
        
        # Compile results
        result = sample.to_dict()
        result.update({
            'fenics_von_mises_max': stress_metrics['von_mises_max'] / 1e6,  # Convert to MPa
            'fenics_max_principal_max': stress_metrics['max_principal_max'] / 1e6,
            'fenics_shear_stress_max': stress_metrics['shear_stress_max'] / 1e6,
            'fenics_safety_factor': fracture_assessment['safety_factor_principal'],
            'fenics_fracture_risk': fracture_assessment['fracture_risk_principal'],
            'fenics_risk_category': fracture_assessment['risk_category']
        })
        
        return result
    
    def combine_datasets(self) -> pd.DataFrame:
        """Combine analytical and FEA datasets"""
        logger.info("Combining analytical and FEA datasets")
        
        if self.analytical_dataset is None or self.fenics_dataset is None:
            raise ValueError("Both analytical and FEA datasets must be generated first")
        
        # Merge datasets on sample_id
        self.combined_dataset = pd.merge(
            self.analytical_dataset, 
            self.fenics_dataset[['sample_id', 'fenics_von_mises_max', 'fenics_max_principal_max', 
                               'fenics_shear_stress_max', 'fenics_safety_factor', 
                               'fenics_fracture_risk', 'fenics_risk_category']],
            on='sample_id',
            how='left'
        )
        
        # Calculate differences between analytical and FEA results
        self.combined_dataset['von_mises_error'] = (
            self.combined_dataset['fenics_von_mises_max'] - 
            self.combined_dataset['von_mises_stress_elastic']
        )
        self.combined_dataset['max_principal_error'] = (
            self.combined_dataset['fenics_max_principal_max'] - 
            self.combined_dataset['max_principal_stress_elastic']
        )
        self.combined_dataset['safety_factor_error'] = (
            self.combined_dataset['fenics_safety_factor'] - 
            self.combined_dataset['safety_factor_elastic']
        )
        
        # Save combined dataset
        self.combined_dataset.to_csv(self.output_dir / "combined_sofc_dataset.csv", index=False)
        
        logger.info(f"Combined dataset created: {len(self.combined_dataset)} samples")
        return self.combined_dataset
    
    def train_ml_models(self) -> Dict:
        """Train machine learning models for stress prediction"""
        logger.info("Training machine learning models")
        
        if self.combined_dataset is None:
            raise ValueError("Combined dataset must be created first")
        
        # Prepare features and targets
        feature_columns = [
            'plate_length', 'plate_width', 'anode_thickness', 'electrolyte_thickness',
            'cathode_thickness', 'interconnect_thickness', 'green_density_anode',
            'green_density_electrolyte', 'green_density_cathode', 'green_density_interconnect',
            'anode_E_25C', 'electrolyte_E_25C', 'cathode_E_25C', 'interconnect_E_25C',
            'anode_CTE_25C', 'electrolyte_CTE_25C', 'cathode_CTE_25C', 'interconnect_CTE_25C',
            'max_temperature', 'heating_rate', 'cooling_rate', 'hold_time',
            'anode_shrinkage_rate', 'electrolyte_shrinkage_rate', 'cathode_shrinkage_rate',
            'interconnect_shrinkage_rate', 'electrolyte_creep_B', 'electrolyte_creep_n',
            'electrolyte_creep_Q'
        ]
        
        X = self.combined_dataset[feature_columns].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.combined_dataset['max_principal_stress_elastic'], 
            test_size=0.2, random_state=42
        )
        
        # Train stress prediction model
        self.stress_predictor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.stress_predictor.fit(X_train, y_train)
        
        # Evaluate stress prediction
        y_pred = self.stress_predictor.predict(X_test)
        stress_mse = mean_squared_error(y_test, y_pred)
        stress_r2 = r2_score(y_test, y_pred)
        
        # Train fracture risk prediction model
        X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
            X, self.combined_dataset['safety_factor_elastic'], 
            test_size=0.2, random_state=42
        )
        
        self.fracture_risk_predictor = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.fracture_risk_predictor.fit(X_train_risk, y_train_risk)
        
        # Evaluate fracture risk prediction
        y_pred_risk = self.fracture_risk_predictor.predict(X_test_risk)
        risk_mse = mean_squared_error(y_test_risk, y_pred_risk)
        risk_r2 = r2_score(y_test_risk, y_pred_risk)
        
        # Save models
        joblib.dump(self.stress_predictor, self.output_dir / "stress_predictor.pkl")
        joblib.dump(self.fracture_risk_predictor, self.output_dir / "fracture_risk_predictor.pkl")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.stress_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_results = {
            'stress_predictor': {
                'mse': stress_mse,
                'r2': stress_r2,
                'feature_importance': feature_importance
            },
            'fracture_risk_predictor': {
                'mse': risk_mse,
                'r2': risk_r2
            }
        }
        
        logger.info(f"ML models trained - Stress R²: {stress_r2:.3f}, Risk R²: {risk_r2:.3f}")
        return model_results
    
    def create_interactive_dashboard(self):
        """Create interactive dashboard for dataset exploration"""
        logger.info("Creating interactive dashboard")
        
        if self.combined_dataset is None:
            raise ValueError("Combined dataset must be created first")
        
        # Initialize Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("SOFC Residual Stress Dataset Explorer", className="text-center mb-4"),
                    html.P("Interactive exploration of the comprehensive SOFC residual stress dataset", 
                          className="text-center text-muted")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Dataset Statistics"),
                        dbc.CardBody([
                            html.P(f"Total Samples: {len(self.combined_dataset):,}"),
                            html.P(f"Analytical Samples: {len(self.analytical_dataset):,}"),
                            html.P(f"FEA Validation Samples: {len(self.fenics_dataset):,}"),
                            html.P(f"Features: {len([col for col in self.combined_dataset.columns if col not in ['sample_id']]):,}")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Key Metrics"),
                        dbc.CardBody([
                            html.P(f"Max Principal Stress: {self.combined_dataset['max_principal_stress_elastic'].mean():.1f} ± {self.combined_dataset['max_principal_stress_elastic'].std():.1f} MPa"),
                            html.P(f"Safety Factor: {self.combined_dataset['safety_factor_elastic'].mean():.2f} ± {self.combined_dataset['safety_factor_elastic'].std():.2f}"),
                            html.P(f"Fracture Risk: {self.combined_dataset['fracture_risk_elastic'].mean():.3f} ± {self.combined_dataset['fracture_risk_elastic'].std():.3f}")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='stress-distribution',
                        figure=self._create_stress_distribution_plot()
                    )
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(
                        id='parameter-correlation',
                        figure=self._create_parameter_correlation_plot()
                    )
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='fenics-validation',
                        figure=self._create_fenics_validation_plot()
                    )
                ], width=12)
            ])
        ])
        
        # Save dashboard
        app.run_server(debug=False, port=8050)
        
        logger.info("Interactive dashboard created")
    
    def _create_stress_distribution_plot(self):
        """Create stress distribution plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Max Principal Stress', 'Von Mises Stress', 
                          'Safety Factor', 'Fracture Risk'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Max Principal Stress
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['max_principal_stress_elastic'], 
                        name='Elastic', opacity=0.7, nbinsx=30),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['max_principal_stress_viscoelastic'], 
                        name='Viscoelastic', opacity=0.7, nbinsx=30),
            row=1, col=1
        )
        
        # Von Mises Stress
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['von_mises_stress_elastic'], 
                        name='Elastic', opacity=0.7, nbinsx=30, showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['von_mises_stress_viscoelastic'], 
                        name='Viscoelastic', opacity=0.7, nbinsx=30, showlegend=False),
            row=1, col=2
        )
        
        # Safety Factor
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['safety_factor_elastic'], 
                        name='Elastic', opacity=0.7, nbinsx=30, showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['safety_factor_viscoelastic'], 
                        name='Viscoelastic', opacity=0.7, nbinsx=30, showlegend=False),
            row=2, col=1
        )
        
        # Fracture Risk
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['fracture_risk_elastic'], 
                        name='Elastic', opacity=0.7, nbinsx=30, showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=self.combined_dataset['fracture_risk_viscoelastic'], 
                        name='Viscoelastic', opacity=0.7, nbinsx=30, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Stress Distribution Analysis")
        return fig
    
    def _create_parameter_correlation_plot(self):
        """Create parameter correlation heatmap"""
        # Select key parameters for correlation
        key_params = [
            'electrolyte_thickness', 'electrolyte_E_25C', 'electrolyte_CTE_25C',
            'max_temperature', 'max_principal_stress_elastic', 'safety_factor_elastic'
        ]
        
        correlation_matrix = self.combined_dataset[key_params].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(title="Parameter Correlation Matrix")
        return fig
    
    def _create_fenics_validation_plot(self):
        """Create FEA validation plot"""
        # Filter samples with FEA results
        fenics_data = self.combined_dataset.dropna(subset=['fenics_max_principal_max'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Analytical vs FEA Max Principal Stress', 
                          'Analytical vs FEA Safety Factor')
        )
        
        # Max Principal Stress comparison
        fig.add_trace(
            go.Scatter(
                x=fenics_data['max_principal_stress_elastic'],
                y=fenics_data['fenics_max_principal_max'],
                mode='markers',
                name='Max Principal Stress',
                marker=dict(size=8, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Add perfect correlation line
        min_val = min(fenics_data['max_principal_stress_elastic'].min(), 
                     fenics_data['fenics_max_principal_max'].min())
        max_val = max(fenics_data['max_principal_stress_elastic'].max(), 
                     fenics_data['fenics_max_principal_max'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Correlation',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )
        
        # Safety Factor comparison
        fig.add_trace(
            go.Scatter(
                x=fenics_data['safety_factor_elastic'],
                y=fenics_data['fenics_safety_factor'],
                mode='markers',
                name='Safety Factor',
                marker=dict(size=8, opacity=0.6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add perfect correlation line
        min_val = min(fenics_data['safety_factor_elastic'].min(), 
                     fenics_data['fenics_safety_factor'].min())
        max_val = max(fenics_data['safety_factor_elastic'].max(), 
                     fenics_data['fenics_safety_factor'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Correlation',
                line=dict(dash='dash', color='red'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Analytical Prediction", row=1, col=1)
        fig.update_yaxes(title_text="FEA Result", row=1, col=1)
        fig.update_xaxes(title_text="Analytical Prediction", row=1, col=2)
        fig.update_yaxes(title_text="FEA Result", row=1, col=2)
        
        fig.update_layout(height=500, title_text="FEA Validation Analysis")
        return fig
    
    def generate_comprehensive_report(self):
        """Generate comprehensive dataset report"""
        logger.info("Generating comprehensive dataset report")
        
        report = {
            'dataset_summary': {
                'total_samples': len(self.combined_dataset),
                'analytical_samples': len(self.analytical_dataset),
                'fenics_samples': len(self.fenics_dataset),
                'features': len([col for col in self.combined_dataset.columns if col not in ['sample_id']])
            },
            'parameter_ranges': self.analytical_generator.param_ranges,
            'stress_statistics': {
                'max_principal_elastic': {
                    'mean': float(self.combined_dataset['max_principal_stress_elastic'].mean()),
                    'std': float(self.combined_dataset['max_principal_stress_elastic'].std()),
                    'min': float(self.combined_dataset['max_principal_stress_elastic'].min()),
                    'max': float(self.combined_dataset['max_principal_stress_elastic'].max())
                },
                'max_principal_viscoelastic': {
                    'mean': float(self.combined_dataset['max_principal_stress_viscoelastic'].mean()),
                    'std': float(self.combined_dataset['max_principal_stress_viscoelastic'].std()),
                    'min': float(self.combined_dataset['max_principal_stress_viscoelastic'].min()),
                    'max': float(self.combined_dataset['max_principal_stress_viscoelastic'].max())
                }
            },
            'fracture_risk_statistics': {
                'safety_factor_elastic': {
                    'mean': float(self.combined_dataset['safety_factor_elastic'].mean()),
                    'std': float(self.combined_dataset['safety_factor_elastic'].std()),
                    'min': float(self.combined_dataset['safety_factor_elastic'].min()),
                    'max': float(self.combined_dataset['safety_factor_elastic'].max())
                },
                'fracture_risk_elastic': {
                    'mean': float(self.combined_dataset['fracture_risk_elastic'].mean()),
                    'std': float(self.combined_dataset['fracture_risk_elastic'].std()),
                    'min': float(self.combined_dataset['fracture_risk_elastic'].min()),
                    'max': float(self.combined_dataset['fracture_risk_elastic'].max())
                }
            },
            'fenics_validation': {
                'correlation_max_principal': float(
                    self.combined_dataset[['max_principal_stress_elastic', 'fenics_max_principal_max']]
                    .corr().iloc[0, 1]
                ) if 'fenics_max_principal_max' in self.combined_dataset.columns else None,
                'correlation_safety_factor': float(
                    self.combined_dataset[['safety_factor_elastic', 'fenics_safety_factor']]
                    .corr().iloc[0, 1]
                ) if 'fenics_safety_factor' in self.combined_dataset.columns else None
            }
        }
        
        # Save report
        with open(self.output_dir / "dataset_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        self._create_markdown_report(report)
        
        logger.info("Comprehensive report generated")
        return report
    
    def _create_markdown_report(self, report: Dict):
        """Create markdown report"""
        md_content = f"""# SOFC Residual Stress Dataset Report

## Dataset Summary

- **Total Samples**: {report['dataset_summary']['total_samples']:,}
- **Analytical Samples**: {report['dataset_summary']['analytical_samples']:,}
- **FEA Validation Samples**: {report['dataset_summary']['fenics_samples']:,}
- **Features**: {report['dataset_summary']['features']:,}

## Stress Statistics

### Max Principal Stress (Elastic Model)
- Mean: {report['stress_statistics']['max_principal_elastic']['mean']:.1f} MPa
- Std: {report['stress_statistics']['max_principal_elastic']['std']:.1f} MPa
- Range: {report['stress_statistics']['max_principal_elastic']['min']:.1f} - {report['stress_statistics']['max_principal_elastic']['max']:.1f} MPa

### Max Principal Stress (Viscoelastic Model)
- Mean: {report['stress_statistics']['max_principal_viscoelastic']['mean']:.1f} MPa
- Std: {report['stress_statistics']['max_principal_viscoelastic']['std']:.1f} MPa
- Range: {report['stress_statistics']['max_principal_viscoelastic']['min']:.1f} - {report['stress_statistics']['max_principal_viscoelastic']['max']:.1f} MPa

## Fracture Risk Statistics

### Safety Factor (Elastic Model)
- Mean: {report['fracture_risk_statistics']['safety_factor_elastic']['mean']:.2f}
- Std: {report['fracture_risk_statistics']['safety_factor_elastic']['std']:.2f}
- Range: {report['fracture_risk_statistics']['safety_factor_elastic']['min']:.2f} - {report['fracture_risk_statistics']['safety_factor_elastic']['max']:.2f}

### Fracture Risk (Elastic Model)
- Mean: {report['fracture_risk_statistics']['fracture_risk_elastic']['mean']:.3f}
- Std: {report['fracture_risk_statistics']['fracture_risk_elastic']['std']:.3f}
- Range: {report['fracture_risk_statistics']['fracture_risk_elastic']['min']:.3f} - {report['fracture_risk_statistics']['fracture_risk_elastic']['max']:.3f}

## FEA Validation

### Correlation with FEA Results
- Max Principal Stress: {report['fenics_validation']['correlation_max_principal']:.3f}
- Safety Factor: {report['fenics_validation']['correlation_safety_factor']:.3f}

## Parameter Ranges

The dataset includes the following parameter ranges:

### Geometric Parameters
- Plate Length: {report['parameter_ranges']['plate_length'][0]} - {report['parameter_ranges']['plate_length'][1]} mm
- Plate Width: {report['parameter_ranges']['plate_width'][0]} - {report['parameter_ranges']['plate_width'][1]} mm
- Electrolyte Thickness: {report['parameter_ranges']['electrolyte_thickness'][0]} - {report['parameter_ranges']['electrolyte_thickness'][1]} mm

### Material Properties
- Electrolyte Young's Modulus: {report['parameter_ranges']['electrolyte_E_25C'][0]} - {report['parameter_ranges']['electrolyte_E_25C'][1]} GPa
- Electrolyte CTE: {report['parameter_ranges']['electrolyte_CTE_25C'][0]*1e6:.1f} - {report['parameter_ranges']['electrolyte_CTE_25C'][1]*1e6:.1f} ppm/K

### Process Parameters
- Max Temperature: {report['parameter_ranges']['max_temperature'][0]} - {report['parameter_ranges']['max_temperature'][1]} °C
- Heating Rate: {report['parameter_ranges']['heating_rate'][0]} - {report['parameter_ranges']['heating_rate'][1]} °C/min

## Usage

This dataset is designed for machine learning applications in SOFC residual stress prediction. It includes:

1. **Input Features**: Geometric, material, and process parameters
2. **Target Variables**: Stress components, safety factors, fracture risk metrics
3. **Validation Data**: FEA simulation results for model validation
4. **Multiple Models**: Both elastic and viscoelastic constitutive models

The dataset can be used for:
- Regression modeling for stress prediction
- Classification for fracture risk assessment
- Sensitivity analysis of design parameters
- Optimization of SOFC designs
"""
        
        with open(self.output_dir / "dataset_report.md", 'w') as f:
            f.write(md_content)

def main():
    """Main function to generate comprehensive SOFC dataset"""
    logger.info("Starting comprehensive SOFC dataset generation")
    
    # Initialize generator
    generator = ComprehensiveSOFCDatasetGenerator()
    
    # Generate analytical dataset (large, fast)
    logger.info("Step 1: Generating analytical dataset")
    analytical_dataset = generator.generate_analytical_dataset(n_samples=5000, method='lhs')
    
    # Generate FEA validation dataset (small, accurate)
    logger.info("Step 2: Generating FEA validation dataset")
    fenics_dataset = generator.generate_fenics_validation_dataset(n_samples=50)
    
    # Combine datasets
    logger.info("Step 3: Combining datasets")
    combined_dataset = generator.combine_datasets()
    
    # Train ML models
    logger.info("Step 4: Training ML models")
    model_results = generator.train_ml_models()
    
    # Generate report
    logger.info("Step 5: Generating comprehensive report")
    report = generator.generate_comprehensive_report()
    
    # Create visualizations
    logger.info("Step 6: Creating visualizations")
    generator.analytical_generator.create_visualizations(analytical_dataset)
    
    # Print final summary
    print("\n" + "="*80)
    print("COMPREHENSIVE SOFC RESIDUAL STRESS DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Total samples: {len(combined_dataset):,}")
    print(f"Analytical samples: {len(analytical_dataset):,}")
    print(f"FEA validation samples: {len(fenics_dataset):,}")
    print(f"Features: {len([col for col in combined_dataset.columns if col not in ['sample_id']]):,}")
    print(f"Output directory: {generator.output_dir}")
    
    print(f"\nKey Statistics:")
    print(f"Max Principal Stress (Elastic): {combined_dataset['max_principal_stress_elastic'].mean():.1f} ± {combined_dataset['max_principal_stress_elastic'].std():.1f} MPa")
    print(f"Max Principal Stress (Viscoelastic): {combined_dataset['max_principal_stress_viscoelastic'].mean():.1f} ± {combined_dataset['max_principal_stress_viscoelastic'].std():.1f} MPa")
    print(f"Safety Factor (Elastic): {combined_dataset['safety_factor_elastic'].mean():.2f} ± {combined_dataset['safety_factor_elastic'].std():.2f}")
    print(f"Safety Factor (Viscoelastic): {combined_dataset['safety_factor_viscoelastic'].mean():.2f} ± {combined_dataset['safety_factor_viscoelastic'].std():.2f}")
    
    print(f"\nFEA Validation:")
    if 'fenics_max_principal_max' in combined_dataset.columns:
        fenics_data = combined_dataset.dropna(subset=['fenics_max_principal_max'])
        correlation = fenics_data[['max_principal_stress_elastic', 'fenics_max_principal_max']].corr().iloc[0, 1]
        print(f"Analytical vs FEA correlation: {correlation:.3f}")
    
    print(f"\nML Model Performance:")
    print(f"Stress Prediction R²: {model_results['stress_predictor']['r2']:.3f}")
    print(f"Fracture Risk Prediction R²: {model_results['fracture_risk_predictor']['r2']:.3f}")
    
    print(f"\nFiles generated:")
    print(f"- Combined dataset: {generator.output_dir}/combined_sofc_dataset.csv")
    print(f"- Analytical dataset: {generator.output_dir}/analytical/sofc_residual_stress_dataset.csv")
    print(f"- FEA validation: {generator.output_dir}/fenics_validation_dataset.csv")
    print(f"- Dataset report: {generator.output_dir}/dataset_report.md")
    print(f"- ML models: {generator.output_dir}/stress_predictor.pkl, {generator.output_dir}/fracture_risk_predictor.pkl")
    
    print(f"\nDataset generation complete! 🎉")
    
    # Optionally create interactive dashboard
    # generator.create_interactive_dashboard()

if __name__ == "__main__":
    main()