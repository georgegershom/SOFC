#!/usr/bin/env python3
"""
Advanced Visualization Script for Fire-Resistant Rubberized Concrete Dataset
Creates interactive and publication-ready plots for research presentation
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizer:
    def __init__(self, data_path="../"):
        """Initialize the visualizer with data path"""
        self.data_path = Path(data_path)
        self.load_processed_data()
    
    def load_processed_data(self):
        """Load processed CSV data"""
        try:
            self.df_combined = pd.read_csv(self.data_path / '04_analysis_scripts/combined_dataset.csv')
            print("✓ Processed data loaded successfully")
        except FileNotFoundError:
            print("Error: Run data_analysis.py first to generate processed data")
            return
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Slump vs Rubber Content', 'Air Content vs Rubber Content',
                          'Unit Weight vs Rubber Content', 'Fresh Properties Radar'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "polar"}]]
        )
        
        # Color mapping for rubber content
        colors = px.colors.sequential.Viridis
        
        # 1. Slump vs Rubber Content
        fig.add_trace(
            go.Scatter(
                x=self.df_combined['rubber_content'],
                y=self.df_combined['slump'],
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color=self.df_combined['rubber_content'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Rubber Content (%)", x=0.45)
                ),
                name='Slump',
                hovertemplate='<b>%{text}</b><br>' +
                             'Rubber Content: %{x}%<br>' +
                             'Slump: %{y} mm<extra></extra>',
                text=self.df_combined['mix_id']
            ),
            row=1, col=1
        )
        
        # 2. Air Content vs Rubber Content
        fig.add_trace(
            go.Scatter(
                x=self.df_combined['rubber_content'],
                y=self.df_combined['air_content'],
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color=self.df_combined['rubber_content'],
                    colorscale='Plasma',
                    showscale=False
                ),
                name='Air Content',
                hovertemplate='<b>%{text}</b><br>' +
                             'Rubber Content: %{x}%<br>' +
                             'Air Content: %{y}%<extra></extra>',
                text=self.df_combined['mix_id']
            ),
            row=1, col=2
        )
        
        # 3. Unit Weight vs Rubber Content
        fig.add_trace(
            go.Scatter(
                x=self.df_combined['rubber_content'],
                y=self.df_combined['unit_weight'],
                mode='markers+lines',
                marker=dict(
                    size=10,
                    color=self.df_combined['rubber_content'],
                    colorscale='Coolwarm',
                    showscale=False
                ),
                name='Unit Weight',
                hovertemplate='<b>%{text}</b><br>' +
                             'Rubber Content: %{x}%<br>' +
                             'Unit Weight: %{y} kg/m³<extra></extra>',
                text=self.df_combined['mix_id']
            ),
            row=2, col=1
        )
        
        # 4. Radar chart for fresh properties (normalized)
        # Normalize properties for radar chart
        properties = ['slump', 'air_content', 'unit_weight', 'fresh_temperature']
        normalized_data = {}
        
        for prop in properties:
            min_val = self.df_combined[prop].min()
            max_val = self.df_combined[prop].max()
            normalized_data[prop] = (self.df_combined[prop] - min_val) / (max_val - min_val)
        
        # Add radar traces for different rubber contents
        rubber_levels = [0, 10, 20, 30]
        colors_radar = ['red', 'blue', 'green', 'orange']
        
        for i, rubber_pct in enumerate(rubber_levels):
            if rubber_pct in self.df_combined['rubber_content'].values:
                idx = self.df_combined[self.df_combined['rubber_content'] == rubber_pct].index[0]
                values = [normalized_data[prop].iloc[idx] for prop in properties]
                values.append(values[0])  # Close the radar chart
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=properties + [properties[0]],
                        fill='toself',
                        name=f'{rubber_pct}% Rubber',
                        line_color=colors_radar[i]
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Fire-Resistant Rubberized Concrete - Interactive Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Rubber Content (%)", row=1, col=1)
        fig.update_yaxes(title_text="Slump (mm)", row=1, col=1)
        
        fig.update_xaxes(title_text="Rubber Content (%)", row=1, col=2)
        fig.update_yaxes(title_text="Air Content (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Rubber Content (%)", row=2, col=1)
        fig.update_yaxes(title_text="Unit Weight (kg/m³)", row=2, col=1)
        
        # Save interactive plot
        fig.write_html(self.data_path / '04_analysis_scripts/interactive_dashboard.html')
        print("✓ Interactive dashboard saved as 'interactive_dashboard.html'")
        
        return fig
    
    def create_publication_plots(self):
        """Create publication-ready plots with professional styling"""
        print("Creating publication-ready plots...")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'font.family': 'serif'
        })
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fire-Resistant Rubberized Concrete: Fresh Properties Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Slump vs Rubber Content with confidence intervals
        ax1 = axes[0, 0]
        
        # Group by rubber content and calculate statistics
        rubber_groups = self.df_combined.groupby('rubber_content')['slump']
        rubber_means = rubber_groups.mean()
        rubber_stds = rubber_groups.std().fillna(0)
        
        ax1.errorbar(rubber_means.index, rubber_means.values, yerr=rubber_stds.values,
                    fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                    color='#2E86AB', ecolor='#A23B72', alpha=0.8)
        
        # Add trend line
        z = np.polyfit(self.df_combined['rubber_content'], self.df_combined['slump'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 30, 100)
        ax1.plot(x_trend, p(x_trend), '--', color='#F18F01', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Rubber Content (%)')
        ax1.set_ylabel('Slump (mm)')
        ax1.set_title('(a) Slump vs Rubber Content', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add R² annotation
        r_squared = np.corrcoef(self.df_combined['rubber_content'], self.df_combined['slump'])[0,1]**2
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Air Content vs Rubber Content
        ax2 = axes[0, 1]
        
        rubber_groups_air = self.df_combined.groupby('rubber_content')['air_content']
        air_means = rubber_groups_air.mean()
        air_stds = rubber_groups_air.std().fillna(0)
        
        ax2.errorbar(air_means.index, air_means.values, yerr=air_stds.values,
                    fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                    color='#C73E1D', ecolor='#592E83', alpha=0.8)
        
        # Add trend line
        z = np.polyfit(self.df_combined['rubber_content'], self.df_combined['air_content'], 1)
        p = np.poly1d(z)
        ax2.plot(x_trend, p(x_trend), '--', color='#F18F01', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Rubber Content (%)')
        ax2.set_ylabel('Air Content (%)')
        ax2.set_title('(b) Air Content vs Rubber Content', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add R² annotation
        r_squared = np.corrcoef(self.df_combined['rubber_content'], self.df_combined['air_content'])[0,1]**2
        ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. Unit Weight vs Rubber Content
        ax3 = axes[0, 2]
        
        rubber_groups_weight = self.df_combined.groupby('rubber_content')['unit_weight']
        weight_means = rubber_groups_weight.mean()
        weight_stds = rubber_groups_weight.std().fillna(0)
        
        ax3.errorbar(weight_means.index, weight_means.values, yerr=weight_stds.values,
                    fmt='^-', capsize=5, capthick=2, linewidth=2, markersize=8,
                    color='#3F7CAC', ecolor='#95190C', alpha=0.8)
        
        # Add trend line
        z = np.polyfit(self.df_combined['rubber_content'], self.df_combined['unit_weight'], 1)
        p = np.poly1d(z)
        ax3.plot(x_trend, p(x_trend), '--', color='#F18F01', linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Rubber Content (%)')
        ax3.set_ylabel('Unit Weight (kg/m³)')
        ax3.set_title('(c) Unit Weight vs Rubber Content', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add R² annotation
        r_squared = np.corrcoef(self.df_combined['rubber_content'], self.df_combined['unit_weight'])[0,1]**2
        ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. Properties by rubber size (box plot)
        ax4 = axes[1, 0]
        
        # Filter out N/A rubber sizes
        size_data = self.df_combined[self.df_combined['rubber_size'] != 'N/A']
        
        if not size_data.empty:
            box_data = []
            labels = []
            for size in size_data['rubber_size'].unique():
                box_data.append(size_data[size_data['rubber_size'] == size]['slump'].values)
                labels.append(size)
            
            bp = ax4.boxplot(box_data, labels=labels, patch_artist=True, 
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
        
        ax4.set_xlabel('Rubber Size')
        ax4.set_ylabel('Slump (mm)')
        ax4.set_title('(d) Slump Distribution by Rubber Size', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        # 5. Correlation heatmap
        ax5 = axes[1, 1]
        
        corr_data = self.df_combined[['rubber_content', 'slump', 'air_content', 
                                   'unit_weight', 'fresh_temperature']].corr()
        
        im = ax5.imshow(corr_data, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                text = ax5.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax5.set_xticks(range(len(corr_data.columns)))
        ax5.set_yticks(range(len(corr_data.columns)))
        ax5.set_xticklabels([col.replace('_', ' ').title() for col in corr_data.columns], rotation=45)
        ax5.set_yticklabels([col.replace('_', ' ').title() for col in corr_data.columns])
        ax5.set_title('(e) Fresh Properties Correlation Matrix', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # 6. Multi-property comparison
        ax6 = axes[1, 2]
        
        # Normalize properties for comparison
        props_norm = {}
        properties = ['slump', 'air_content', 'unit_weight']
        
        for prop in properties:
            min_val = self.df_combined[prop].min()
            max_val = self.df_combined[prop].max()
            props_norm[prop] = (self.df_combined[prop] - min_val) / (max_val - min_val)
        
        # Plot normalized properties vs rubber content
        for i, prop in enumerate(properties):
            ax6.plot(self.df_combined['rubber_content'], props_norm[prop], 
                    'o-', label=prop.replace('_', ' ').title(), 
                    linewidth=2, markersize=6, alpha=0.8)
        
        ax6.set_xlabel('Rubber Content (%)')
        ax6.set_ylabel('Normalized Property Value')
        ax6.set_title('(f) Normalized Fresh Properties Comparison', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_path / '04_analysis_scripts/publication_plots.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.data_path / '04_analysis_scripts/publication_plots.pdf', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
        print("✓ Publication plots saved as 'publication_plots.png' and 'publication_plots.pdf'")
        
        plt.show()
    
    def create_3d_visualization(self):
        """Create 3D visualization of the data"""
        print("Creating 3D visualization...")
        
        fig = go.Figure()
        
        # Create 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=self.df_combined['rubber_content'],
            y=self.df_combined['slump'],
            z=self.df_combined['air_content'],
            mode='markers+text',
            marker=dict(
                size=self.df_combined['unit_weight']/100,  # Size based on unit weight
                color=self.df_combined['rubber_content'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Rubber Content (%)")
            ),
            text=self.df_combined['mix_id'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Rubber Content: %{x}%<br>' +
                         'Slump: %{y} mm<br>' +
                         'Air Content: %{z}%<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Visualization: Rubber Content vs Fresh Properties',
            scene=dict(
                xaxis_title='Rubber Content (%)',
                yaxis_title='Slump (mm)',
                zaxis_title='Air Content (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        fig.write_html(self.data_path / '04_analysis_scripts/3d_visualization.html')
        print("✓ 3D visualization saved as '3d_visualization.html'")
        
        return fig
    
    def create_animated_plot(self):
        """Create animated plot showing property evolution with rubber content"""
        print("Creating animated plot...")
        
        # Prepare data for animation
        animation_data = []
        
        for rubber_pct in sorted(self.df_combined['rubber_content'].unique()):
            subset = self.df_combined[self.df_combined['rubber_content'] == rubber_pct]
            for _, row in subset.iterrows():
                animation_data.append({
                    'rubber_content': rubber_pct,
                    'slump': row['slump'],
                    'air_content': row['air_content'],
                    'unit_weight': row['unit_weight'],
                    'mix_id': row['mix_id']
                })
        
        df_anim = pd.DataFrame(animation_data)
        
        # Create animated scatter plot
        fig = px.scatter(
            df_anim, 
            x="slump", 
            y="air_content",
            size="unit_weight",
            color="rubber_content",
            hover_name="mix_id",
            animation_frame="rubber_content",
            animation_group="mix_id",
            size_max=20,
            range_x=[80, 200],
            range_y=[4, 11],
            color_continuous_scale='Viridis',
            title="Animated View: Fresh Properties Evolution with Rubber Content"
        )
        
        fig.update_layout(
            xaxis_title="Slump (mm)",
            yaxis_title="Air Content (%)",
            width=800,
            height=600
        )
        
        fig.write_html(self.data_path / '04_analysis_scripts/animated_plot.html')
        print("✓ Animated plot saved as 'animated_plot.html'")
        
        return fig
    
    def run_advanced_visualization(self):
        """Run all advanced visualization functions"""
        print("ADVANCED VISUALIZATION SUITE")
        print("=" * 50)
        print("Fire-Resistant Rubberized Concrete Dataset")
        print("=" * 50)
        
        self.create_interactive_dashboard()
        self.create_publication_plots()
        self.create_3d_visualization()
        self.create_animated_plot()
        
        print("\\n" + "="*50)
        print("ADVANCED VISUALIZATION COMPLETE")
        print("="*50)
        print("All visualizations have been created and saved.")
        print("Check the 04_analysis_scripts directory for output files.")


if __name__ == "__main__":
    # Run the advanced visualization suite
    visualizer = AdvancedVisualizer()
    visualizer.run_advanced_visualization()