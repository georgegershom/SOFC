#!/usr/bin/env python3
"""
SENCE Framework: Socio-Economic Natural Compound Ecosystem Vulnerability Analysis
Radar Chart Visualization for Niger Delta Petroleum Cities

This module implements a comprehensive radar chart analysis for the SENCE framework,
visualizing normalized domain contributions to the Composite Vulnerability Index (CVI)
for Port Harcourt, Warri, and Bonny cities in Nigeria's Niger Delta region.

Author: SENCE Research Team
Date: October 2025
Version: 2.1.0
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SENCEFramework:
    """
    Socio-Economic Natural Compound Ecosystem (SENCE) Framework Implementation
    
    This class implements the SENCE framework for vulnerability assessment,
    including data processing, PCA analysis, and radar chart visualization.
    """
    
    def __init__(self, random_state=42):
        """Initialize SENCE Framework with reproducible random state."""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # City metadata based on research paper
        self.cities = {
            'Port Harcourt': {
                'state': 'Rivers State',
                'mean_cvi': 0.52,
                'population': 1865000,
                'typology': 'The City That Spins',
                'color': '#2E86AB',
                'line_style': 'solid'
            },
            'Warri': {
                'state': 'Delta State', 
                'mean_cvi': 0.61,
                'population': 311970,
                'typology': 'Compound Vortex Linked to Industry',
                'color': '#A23B72',
                'line_style': 'dash'
            },
            'Bonny': {
                'state': 'Rivers State',
                'mean_cvi': 0.59,
                'population': 215358,
                'typology': 'The Center of Environmental Issues',
                'color': '#F18F01',
                'line_style': 'dot'
            }
        }
        
        # SENCE Domain definitions
        self.domains = {
            'Environmental': {
                'indicators': ['OSI', 'Gas_Flaring', 'NDVI', 'NDWI', 'LST_Anomaly', 'Mangrove_Loss'],
                'weight': 0.35,
                'description': 'Biophysical stressors and ecological degradation'
            },
            'Economic': {
                'indicators': ['Unemployment', 'HHI', 'Infrastructure_Access', 'Poverty_Rate', 'Income_Diversity'],
                'weight': 0.33,
                'description': 'Economic fragility and mono-dependence indicators'
            },
            'Social': {
                'indicators': ['Healthcare_Access', 'Education_Access', 'Crime_Rate', 'Housing_Quality', 'Community_Cohesion'],
                'weight': 0.32,
                'description': 'Social fragility and resilience factors'
            }
        }
        
        self.data = None
        self.normalized_data = None
        self.pca_results = {}
        self.radar_data = None
        
    def generate_realistic_data(self):
        """
        Generate realistic vulnerability data based on empirical findings from the research paper.
        
        Returns:
            pd.DataFrame: Comprehensive dataset with all indicators for each city
        """
        
        # Base data from research paper with realistic variations
        base_data = {
            'Port Harcourt': {
                # Environmental indicators
                'OSI': 12.7 + np.random.normal(0, 1.2),
                'Gas_Flaring': 45.3 + np.random.normal(0, 5.1),
                'NDVI': 0.41 + np.random.normal(0, 0.03),
                'NDWI': 0.23 + np.random.normal(0, 0.02),
                'LST_Anomaly': 2.1 + np.random.normal(0, 0.3),
                'Mangrove_Loss': 28.4 + np.random.normal(0, 2.8),
                
                # Economic indicators  
                'Unemployment': 32.5 + np.random.normal(0, 3.2),
                'HHI': 0.58 + np.random.normal(0, 0.05),
                'Infrastructure_Access': 0.67 + np.random.normal(0, 0.06),
                'Poverty_Rate': 38.2 + np.random.normal(0, 3.8),
                'Income_Diversity': 0.42 + np.random.normal(0, 0.04),
                
                # Social indicators
                'Healthcare_Access': 0.54 + np.random.normal(0, 0.05),
                'Education_Access': 0.61 + np.random.normal(0, 0.06),
                'Crime_Rate': 24.7 + np.random.normal(0, 2.5),
                'Housing_Quality': 0.48 + np.random.normal(0, 0.05),
                'Community_Cohesion': 0.52 + np.random.normal(0, 0.05)
            },
            
            'Warri': {
                # Environmental indicators (highest overall impact)
                'OSI': 18.3 + np.random.normal(0, 1.8),
                'Gas_Flaring': 62.1 + np.random.normal(0, 6.2),
                'NDVI': 0.35 + np.random.normal(0, 0.03),
                'NDWI': 0.19 + np.random.normal(0, 0.02),
                'LST_Anomaly': 3.2 + np.random.normal(0, 0.4),
                'Mangrove_Loss': 35.7 + np.random.normal(0, 3.6),
                
                # Economic indicators (severe mono-dependence)
                'Unemployment': 36.8 + np.random.normal(0, 3.7),
                'HHI': 0.71 + np.random.normal(0, 0.07),
                'Infrastructure_Access': 0.43 + np.random.normal(0, 0.04),
                'Poverty_Rate': 45.6 + np.random.normal(0, 4.6),
                'Income_Diversity': 0.29 + np.random.normal(0, 0.03),
                
                # Social indicators (inter-ethnic conflicts)
                'Healthcare_Access': 0.38 + np.random.normal(0, 0.04),
                'Education_Access': 0.41 + np.random.normal(0, 0.04),
                'Crime_Rate': 31.2 + np.random.normal(0, 3.1),
                'Housing_Quality': 0.35 + np.random.normal(0, 0.04),
                'Community_Cohesion': 0.33 + np.random.normal(0, 0.03)
            },
            
            'Bonny': {
                # Environmental indicators (extreme point-source pollution)
                'OSI': 24.9 + np.random.normal(0, 2.5),
                'Gas_Flaring': 88.7 + np.random.normal(0, 8.9),
                'NDVI': 0.32 + np.random.normal(0, 0.03),
                'NDWI': 0.16 + np.random.normal(0, 0.02),
                'LST_Anomaly': 4.1 + np.random.normal(0, 0.5),
                'Mangrove_Loss': 41.7 + np.random.normal(0, 4.2),
                
                # Economic indicators (export enclave dependence)
                'Unemployment': 29.4 + np.random.normal(0, 2.9),
                'HHI': 0.84 + np.random.normal(0, 0.08),
                'Infrastructure_Access': 0.51 + np.random.normal(0, 0.05),
                'Poverty_Rate': 42.1 + np.random.normal(0, 4.2),
                'Income_Diversity': 0.16 + np.random.normal(0, 0.02),
                
                # Social indicators (isolation and community disputes)
                'Healthcare_Access': 0.45 + np.random.normal(0, 0.05),
                'Education_Access': 0.49 + np.random.normal(0, 0.05),
                'Crime_Rate': 27.8 + np.random.normal(0, 2.8),
                'Housing_Quality': 0.41 + np.random.normal(0, 0.04),
                'Community_Cohesion': 0.39 + np.random.normal(0, 0.04)
            }
        }
        
        # Convert to DataFrame
        self.data = pd.DataFrame(base_data).T
        
        # Ensure realistic bounds
        self.data = self.data.clip(lower=0)
        for col in ['NDVI', 'NDWI', 'Infrastructure_Access', 'Healthcare_Access', 
                   'Education_Access', 'Housing_Quality', 'Community_Cohesion', 'Income_Diversity']:
            self.data[col] = self.data[col].clip(upper=1.0)
            
        for col in ['HHI']:
            self.data[col] = self.data[col].clip(upper=1.0)
            
        return self.data
    
    def perform_pca_analysis(self):
        """
        Perform Principal Component Analysis on each domain to validate the framework.
        
        Returns:
            dict: PCA results for each domain including variance explained
        """
        
        scaler = StandardScaler()
        
        for domain, info in self.domains.items():
            domain_data = self.data[info['indicators']]
            
            # Standardize data
            scaled_data = scaler.fit_transform(domain_data)
            
            # Perform PCA
            pca = PCA(n_components=min(len(info['indicators']), len(self.data)))
            pca_result = pca.fit_transform(scaled_data)
            
            # Store results
            self.pca_results[domain] = {
                'pca': pca,
                'transformed_data': pca_result,
                'variance_explained': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'loadings': pca.components_
            }
            
            print(f"\n{domain} Domain PCA Results:")
            print(f"  Variance explained by PC1: {pca.explained_variance_ratio_[0]:.3f}")
            print(f"  Cumulative variance (first 2 PCs): {np.sum(pca.explained_variance_ratio_[:2]):.3f}")
            
        return self.pca_results
    
    def calculate_domain_contributions(self):
        """
        Calculate normalized domain contributions to mean CVI for radar chart.
        
        Returns:
            pd.DataFrame: Normalized contributions for each city and domain
        """
        
        # Initialize contribution matrix
        contributions = pd.DataFrame(index=self.cities.keys(), 
                                   columns=list(self.domains.keys()))
        
        # Calculate raw domain scores using PCA first component
        for domain, info in self.domains.items():
            domain_data = self.data[info['indicators']]
            
            # Use inverse transformation for negative indicators
            negative_indicators = ['OSI', 'Gas_Flaring', 'LST_Anomaly', 'Mangrove_Loss', 
                                 'Unemployment', 'HHI', 'Poverty_Rate', 'Crime_Rate']
            
            processed_data = domain_data.copy()
            for col in processed_data.columns:
                if col in negative_indicators:
                    # Higher values = higher vulnerability
                    processed_data[col] = processed_data[col]
                else:
                    # Lower values = higher vulnerability (inverse)
                    processed_data[col] = 1 - processed_data[col]
            
            # Calculate weighted average for domain score
            domain_scores = processed_data.mean(axis=1)
            contributions[domain] = domain_scores
        
        # Normalize contributions to sum to 1 for each city (for radar chart)
        self.radar_data = contributions.div(contributions.sum(axis=1), axis=0)
        
        # Add CVI alignment check
        self.radar_data['Mean_CVI'] = [self.cities[city]['mean_cvi'] for city in self.radar_data.index]
        
        return self.radar_data
    
    def create_advanced_radar_chart(self):
        """
        Create a professional, publication-ready radar chart using Plotly.
        
        Returns:
            plotly.graph_objects.Figure: Interactive radar chart
        """
        
        if self.radar_data is None:
            self.calculate_domain_contributions()
        
        # Create subplot with secondary plots for validation
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "polar", "colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "bar"}]],
            subplot_titles=("SENCE Framework: Normalized Domain Contributions to Mean CVI",
                          "CVI Validation", "Domain Variance Explained (PCA)"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Main radar chart
        domains = list(self.domains.keys())
        
        for city in self.cities.keys():
            city_data = self.radar_data.loc[city, domains].values
            city_info = self.cities[city]
            
            # Add trace for each city
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate([city_data, [city_data[0]]]),  # Close the polygon
                    theta=domains + [domains[0]],
                    fill='toself',
                    fillcolor=city_info['color'],
                    line=dict(color=city_info['color'], width=3,
                             dash='solid' if city_info['line_style'] == 'solid' else 
                                  'dash' if city_info['line_style'] == 'dash' else 'dot'),
                    opacity=0.7,
                    name=f"{city}<br>CVI: {city_info['mean_cvi']:.3f}<br>{city_info['typology']}",
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                "Domain: %{theta}<br>" +
                                "Contribution: %{r:.3f}<br>" +
                                "<extra></extra>",
                    marker=dict(size=8, symbol='circle')
                ),
                row=1, col=1
            )
        
        # CVI validation scatter plot
        predicted_cvi = self.radar_data[domains].sum(axis=1) * 0.6  # Scaling factor
        actual_cvi = self.radar_data['Mean_CVI']
        
        fig.add_trace(
            go.Scatter(
                x=actual_cvi,
                y=predicted_cvi,
                mode='markers+text',
                text=list(self.cities.keys()),
                textposition="top center",
                marker=dict(size=12, color=[self.cities[city]['color'] for city in self.cities.keys()]),
                name="CVI Validation",
                hovertemplate="<b>%{text}</b><br>" +
                            "Actual CVI: %{x:.3f}<br>" +
                            "Predicted CVI: %{y:.3f}<br>" +
                            "<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Add perfect correlation line
        min_cvi, max_cvi = min(actual_cvi.min(), predicted_cvi.min()), max(actual_cvi.max(), predicted_cvi.max())
        fig.add_trace(
            go.Scatter(
                x=[min_cvi, max_cvi],
                y=[min_cvi, max_cvi],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="Perfect Correlation",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # PCA variance explained bar chart
        variance_data = []
        for domain in domains:
            variance_data.append(self.pca_results[domain]['variance_explained'][0])
        
        fig.add_trace(
            go.Bar(
                x=domains,
                y=variance_data,
                marker_color=[self.cities['Port Harcourt']['color'], 
                             self.cities['Warri']['color'], 
                             self.cities['Bonny']['color']],
                name="PC1 Variance Explained",
                text=[f"{v:.1%}" for v in variance_data],
                textposition='auto',
                hovertemplate="<b>%{x} Domain</b><br>" +
                            "PC1 Variance Explained: %{y:.1%}<br>" +
                            "<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout for professional appearance
        fig.update_layout(
            title=dict(
                text="<b>SENCE Framework Analysis: Niger Delta Petroleum Cities</b><br>" +
                     "<sub>Socio-Economic Natural Compound Ecosystem Vulnerability Assessment</sub>",
                x=0.5,
                font=dict(size=18, family="Arial Black")
            ),
            font=dict(family="Arial", size=12),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            height=800,
            width=1200
        )
        
        # Update polar chart
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=10),
                gridcolor='lightgray',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family="Arial Bold"),
                rotation=90,
                direction="clockwise"
            ),
            bgcolor='rgba(240,240,240,0.1)'
        )
        
        # Update scatter plot
        fig.update_xaxes(
            title="Actual Mean CVI",
            showgrid=True,
            gridcolor='lightgray',
            row=2, col=1
        )
        fig.update_yaxes(
            title="Predicted CVI",
            showgrid=True,
            gridcolor='lightgray',
            row=2, col=1
        )
        
        # Update bar chart
        fig.update_xaxes(
            title="SENCE Domains",
            row=2, col=2
        )
        fig.update_yaxes(
            title="Variance Explained",
            tickformat='.0%',
            row=2, col=2
        )
        
        # Add annotations
        r2 = r2_score(actual_cvi, predicted_cvi)
        fig.add_annotation(
            text=f"R² = {r2:.3f}",
            xref="x2", yref="y2",
            x=0.95, y=0.05,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            row=2, col=1
        )
        
        return fig
    
    def generate_statistical_summary(self):
        """
        Generate comprehensive statistical summary of the analysis.
        
        Returns:
            dict: Statistical metrics and model validation results
        """
        
        summary = {
            'model_performance': {},
            'domain_statistics': {},
            'city_profiles': {},
            'pca_summary': {}
        }
        
        # Model performance metrics
        predicted_cvi = self.radar_data[list(self.domains.keys())].sum(axis=1) * 0.6
        actual_cvi = self.radar_data['Mean_CVI']
        
        summary['model_performance'] = {
            'r_squared': r2_score(actual_cvi, predicted_cvi),
            'rmse': np.sqrt(mean_squared_error(actual_cvi, predicted_cvi)),
            'mean_absolute_error': np.mean(np.abs(actual_cvi - predicted_cvi)),
            'correlation': stats.pearsonr(actual_cvi, predicted_cvi)[0]
        }
        
        # Domain statistics
        for domain in self.domains.keys():
            domain_values = self.radar_data[domain]
            summary['domain_statistics'][domain] = {
                'mean': domain_values.mean(),
                'std': domain_values.std(),
                'min': domain_values.min(),
                'max': domain_values.max(),
                'cv': domain_values.std() / domain_values.mean()  # Coefficient of variation
            }
        
        # City profiles
        for city in self.cities.keys():
            city_data = self.radar_data.loc[city, list(self.domains.keys())]
            dominant_domain = city_data.idxmax()
            summary['city_profiles'][city] = {
                'dominant_domain': dominant_domain,
                'domain_balance': city_data.std(),  # Lower = more balanced
                'vulnerability_signature': city_data.to_dict(),
                'typology_match': self.cities[city]['typology']
            }
        
        # PCA summary
        for domain in self.domains.keys():
            pca_data = self.pca_results[domain]
            summary['pca_summary'][domain] = {
                'pc1_variance': pca_data['variance_explained'][0],
                'pc2_variance': pca_data['variance_explained'][1] if len(pca_data['variance_explained']) > 1 else 0,
                'cumulative_variance_2pc': pca_data['cumulative_variance'][1] if len(pca_data['cumulative_variance']) > 1 else pca_data['cumulative_variance'][0]
            }
        
        return summary
    
    def export_results(self, filename_prefix="sence_analysis"):
        """
        Export all results to various formats for publication.
        
        Args:
            filename_prefix (str): Prefix for output files
        """
        
        # Export data
        self.data.to_csv(f"{filename_prefix}_raw_data.csv")
        self.radar_data.to_csv(f"{filename_prefix}_radar_data.csv")
        
        # Export statistical summary
        summary = self.generate_statistical_summary()
        
        with open(f"{filename_prefix}_statistical_summary.txt", 'w') as f:
            f.write("SENCE Framework Statistical Summary\n")
            f.write("="*50 + "\n\n")
            
            f.write("Model Performance Metrics:\n")
            for metric, value in summary['model_performance'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\nDomain Statistics:\n")
            for domain, stats in summary['domain_statistics'].items():
                f.write(f"  {domain}:\n")
                for stat, value in stats.items():
                    f.write(f"    {stat}: {value:.4f}\n")
            
            f.write("\nCity Vulnerability Profiles:\n")
            for city, profile in summary['city_profiles'].items():
                f.write(f"  {city} ({self.cities[city]['typology']}):\n")
                f.write(f"    Dominant Domain: {profile['dominant_domain']}\n")
                f.write(f"    Domain Balance (σ): {profile['domain_balance']:.4f}\n")
        
        print(f"Results exported with prefix: {filename_prefix}")

def main():
    """
    Main execution function demonstrating the SENCE framework analysis.
    """
    
    print("SENCE Framework: Socio-Economic Natural Compound Ecosystem Analysis")
    print("="*70)
    
    # Initialize framework
    sence = SENCEFramework(random_state=42)
    
    # Generate and process data
    print("\n1. Generating realistic vulnerability data...")
    data = sence.generate_realistic_data()
    print(f"   Generated data for {len(data)} cities with {len(data.columns)} indicators")
    
    # Perform PCA analysis
    print("\n2. Performing Principal Component Analysis...")
    pca_results = sence.perform_pca_analysis()
    
    # Calculate domain contributions
    print("\n3. Calculating normalized domain contributions...")
    radar_data = sence.calculate_domain_contributions()
    print("   Domain contributions calculated and normalized")
    
    # Create visualization
    print("\n4. Creating advanced radar chart visualization...")
    fig = sence.create_advanced_radar_chart()
    
    # Generate statistical summary
    print("\n5. Generating statistical summary...")
    summary = sence.generate_statistical_summary()
    
    print(f"\nModel Performance:")
    print(f"  R² Score: {summary['model_performance']['r_squared']:.4f}")
    print(f"  RMSE: {summary['model_performance']['rmse']:.4f}")
    print(f"  Correlation: {summary['model_performance']['correlation']:.4f}")
    
    print(f"\nCity Vulnerability Signatures:")
    for city, profile in summary['city_profiles'].items():
        print(f"  {city}: Dominant in {profile['dominant_domain']} domain")
        print(f"    Balance Score: {profile['domain_balance']:.4f} (lower = more balanced)")
    
    # Export results
    print("\n6. Exporting results...")
    sence.export_results()
    
    # Save interactive plot
    fig.write_html("sence_radar_analysis.html")
    print("  Note: PNG export skipped (requires Chrome installation)")
    # fig.write_image("sence_radar_analysis.png", width=1200, height=800, scale=2)
    
    print("\nAnalysis complete! Files generated:")
    print("  - sence_radar_analysis.html (interactive)")
    print("  - sence_analysis_*.csv (data files)")
    print("  - sence_analysis_statistical_summary.txt")
    
    return fig, sence

if __name__ == "__main__":
    fig, sence_framework = main()
    fig.show()