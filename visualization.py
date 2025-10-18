"""
Visualization tools for rubberized concrete experimental dataset
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ConcreteDataVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.setup_style()
    
    def setup_style(self):
        """Set up plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_strength_development(self, save_path: str = None):
        """Plot strength development with curing age"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strength Development with Curing Age', fontsize=16, fontweight='bold')
        
        # Compressive strength
        ax1 = axes[0, 0]
        for mix in self.data['mix_design'].unique():
            mix_data = self.data[(self.data['mix_design'] == mix) & 
                               (self.data['test_type'] == 'ambient')]
            if not mix_data.empty:
                strength_by_age = mix_data.groupby('curing_age_days')['compressive_strength_mpa'].mean()
                ax1.plot(strength_by_age.index, strength_by_age.values, 
                        marker='o', linewidth=2, label=mix)
        ax1.set_xlabel('Curing Age (days)')
        ax1.set_ylabel('Compressive Strength (MPa)')
        ax1.set_title('Compressive Strength Development')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Tensile strength
        ax2 = axes[0, 1]
        for mix in self.data['mix_design'].unique():
            mix_data = self.data[(self.data['mix_design'] == mix) & 
                               (self.data['test_type'] == 'ambient')]
            if not mix_data.empty:
                strength_by_age = mix_data.groupby('curing_age_days')['tensile_strength_mpa'].mean()
                ax2.plot(strength_by_age.index, strength_by_age.values, 
                        marker='s', linewidth=2, label=mix)
        ax2.set_xlabel('Curing Age (days)')
        ax2.set_ylabel('Tensile Strength (MPa)')
        ax2.set_title('Tensile Strength Development')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Modulus of elasticity
        ax3 = axes[1, 0]
        for mix in self.data['mix_design'].unique():
            mix_data = self.data[(self.data['mix_design'] == mix) & 
                               (self.data['test_type'] == 'ambient')]
            if not mix_data.empty:
                modulus_by_age = mix_data.groupby('curing_age_days')['modulus_elasticity_mpa'].mean()
                ax3.plot(modulus_by_age.index, modulus_by_age.values, 
                        marker='^', linewidth=2, label=mix)
        ax3.set_xlabel('Curing Age (days)')
        ax3.set_ylabel('Modulus of Elasticity (MPa)')
        ax3.set_title('Modulus of Elasticity Development')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Density
        ax4 = axes[1, 1]
        for mix in self.data['mix_design'].unique():
            mix_data = self.data[(self.data['mix_design'] == mix) & 
                               (self.data['test_type'] == 'ambient')]
            if not mix_data.empty:
                density_by_age = mix_data.groupby('curing_age_days')['density_kg_m3'].mean()
                ax4.plot(density_by_age.index, density_by_age.values, 
                        marker='d', linewidth=2, label=mix)
        ax4.set_xlabel('Curing Age (days)')
        ax4.set_ylabel('Density (kg/m³)')
        ax4.set_title('Density Development')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temperature_effects(self, save_path: str = None):
        """Plot temperature effects on material properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temperature Effects on Material Properties', fontsize=16, fontweight='bold')
        
        # Filter thermal exposure data
        thermal_data = self.data[self.data['test_type'] == 'thermal_exposure']
        
        # Compressive strength vs temperature
        ax1 = axes[0, 0]
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    strength_by_temp = mix_cooling_data.groupby('temperature_c')['compressive_strength_mpa'].mean()
                    ax1.plot(strength_by_temp.index, strength_by_temp.values, 
                            marker='o', linewidth=2, 
                            label=f'{mix} ({cooling})')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Compressive Strength (MPa)')
        ax1.set_title('Compressive Strength vs Temperature')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Tensile strength vs temperature
        ax2 = axes[0, 1]
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    strength_by_temp = mix_cooling_data.groupby('temperature_c')['tensile_strength_mpa'].mean()
                    ax2.plot(strength_by_temp.index, strength_by_temp.values, 
                            marker='s', linewidth=2, 
                            label=f'{mix} ({cooling})')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Tensile Strength (MPa)')
        ax2.set_title('Tensile Strength vs Temperature')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Modulus of elasticity vs temperature
        ax3 = axes[1, 0]
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    modulus_by_temp = mix_cooling_data.groupby('temperature_c')['modulus_elasticity_mpa'].mean()
                    ax3.plot(modulus_by_temp.index, modulus_by_temp.values, 
                            marker='^', linewidth=2, 
                            label=f'{mix} ({cooling})')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Modulus of Elasticity (MPa)')
        ax3.set_title('Modulus of Elasticity vs Temperature')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Mass loss vs temperature
        ax4 = axes[1, 1]
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    mass_loss_by_temp = mix_cooling_data.groupby('temperature_c')['mass_loss_percent'].mean()
                    ax4.plot(mass_loss_by_temp.index, mass_loss_by_temp.values, 
                            marker='d', linewidth=2, 
                            label=f'{mix} ({cooling})')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Mass Loss (%)')
        ax4.set_title('Mass Loss vs Temperature')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rubber_content_effects(self, save_path: str = None):
        """Plot effects of rubber content on material properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rubber Content Effects on Material Properties', fontsize=16, fontweight='bold')
        
        # Filter ambient data for 28-day strength
        ambient_28d = self.data[(self.data['test_type'] == 'ambient') & 
                               (self.data['curing_age_days'] == 28)]
        
        # Compressive strength vs rubber content
        ax1 = axes[0, 0]
        strength_by_rubber = ambient_28d.groupby('rubber_content_percent')['compressive_strength_mpa'].mean()
        ax1.plot(strength_by_rubber.index, strength_by_rubber.values, 
                marker='o', linewidth=3, markersize=8, color='red')
        ax1.set_xlabel('Rubber Content (%)')
        ax1.set_ylabel('Compressive Strength (MPa)')
        ax1.set_title('Compressive Strength vs Rubber Content')
        ax1.grid(True, alpha=0.3)
        
        # Tensile strength vs rubber content
        ax2 = axes[0, 1]
        tensile_by_rubber = ambient_28d.groupby('rubber_content_percent')['tensile_strength_mpa'].mean()
        ax2.plot(tensile_by_rubber.index, tensile_by_rubber.values, 
                marker='s', linewidth=3, markersize=8, color='blue')
        ax2.set_xlabel('Rubber Content (%)')
        ax2.set_ylabel('Tensile Strength (MPa)')
        ax2.set_title('Tensile Strength vs Rubber Content')
        ax2.grid(True, alpha=0.3)
        
        # Modulus of elasticity vs rubber content
        ax3 = axes[1, 0]
        modulus_by_rubber = ambient_28d.groupby('rubber_content_percent')['modulus_elasticity_mpa'].mean()
        ax3.plot(modulus_by_rubber.index, modulus_by_rubber.values, 
                marker='^', linewidth=3, markersize=8, color='green')
        ax3.set_xlabel('Rubber Content (%)')
        ax3.set_ylabel('Modulus of Elasticity (MPa)')
        ax3.set_title('Modulus of Elasticity vs Rubber Content')
        ax3.grid(True, alpha=0.3)
        
        # Density vs rubber content
        ax4 = axes[1, 1]
        density_by_rubber = ambient_28d.groupby('rubber_content_percent')['density_kg_m3'].mean()
        ax4.plot(density_by_rubber.index, density_by_rubber.values, 
                marker='d', linewidth=3, markersize=8, color='orange')
        ax4.set_xlabel('Rubber Content (%)')
        ax4.set_ylabel('Density (kg/m³)')
        ax4.set_title('Density vs Rubber Content')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_spalling_behavior(self, save_path: str = None):
        """Plot spalling behavior vs temperature and rubber content"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Spalling Behavior Analysis', fontsize=16, fontweight='bold')
        
        thermal_data = self.data[self.data['test_type'] == 'thermal_exposure']
        
        # Spalling depth vs temperature
        ax1 = axes[0]
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    spalling_by_temp = mix_cooling_data.groupby('temperature_c')['spalling_depth_mm'].mean()
                    ax1.plot(spalling_by_temp.index, spalling_by_temp.values, 
                            marker='o', linewidth=2, 
                            label=f'{mix} ({cooling})')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Spalling Depth (mm)')
        ax1.set_title('Spalling Depth vs Temperature')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Spalling depth vs rubber content at 800°C
        ax2 = axes[1]
        temp_800_data = thermal_data[thermal_data['temperature_c'] == 800]
        for cooling in temp_800_data['cooling_regime'].unique():
            cooling_data = temp_800_data[temp_800_data['cooling_regime'] == cooling]
            spalling_by_rubber = cooling_data.groupby('rubber_content_percent')['spalling_depth_mm'].mean()
            ax2.plot(spalling_by_rubber.index, spalling_by_rubber.values, 
                    marker='s', linewidth=3, markersize=8, 
                    label=f'800°C ({cooling})')
        ax2.set_xlabel('Rubber Content (%)')
        ax2.set_ylabel('Spalling Depth (mm)')
        ax2.set_title('Spalling Depth vs Rubber Content (800°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_insitu_properties(self, save_path: str = None):
        """Plot in-situ thermal properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('In-Situ Thermal Properties', fontsize=16, fontweight='bold')
        
        insitu_data = self.data[self.data['test_type'] == 'insitu_thermal']
        
        # In-situ compressive strength
        ax1 = axes[0, 0]
        for mix in insitu_data['mix_design'].unique():
            mix_data = insitu_data[insitu_data['mix_design'] == mix]
            if not mix_data.empty:
                strength_by_temp = mix_data.groupby('temperature_c')['compressive_strength_mpa'].mean()
                ax1.plot(strength_by_temp.index, strength_by_temp.values, 
                        marker='o', linewidth=2, label=mix)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('In-Situ Compressive Strength (MPa)')
        ax1.set_title('In-Situ Compressive Strength vs Temperature')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Thermal strain
        ax2 = axes[0, 1]
        for mix in insitu_data['mix_design'].unique():
            mix_data = insitu_data[insitu_data['mix_design'] == mix]
            if not mix_data.empty:
                strain_by_temp = mix_data.groupby('temperature_c')['thermal_strain_microstrain'].mean()
                ax2.plot(strain_by_temp.index, strain_by_temp.values, 
                        marker='s', linewidth=2, label=mix)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Thermal Strain (μstrain)')
        ax2.set_title('Thermal Strain vs Temperature')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Pore pressure
        ax3 = axes[1, 0]
        for mix in insitu_data['mix_design'].unique():
            mix_data = insitu_data[insitu_data['mix_design'] == mix]
            if not mix_data.empty:
                pressure_by_temp = mix_data.groupby('temperature_c')['pore_pressure_mpa'].mean()
                ax3.plot(pressure_by_temp.index, pressure_by_temp.values, 
                        marker='^', linewidth=2, label=mix)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Pore Pressure (MPa)')
        ax3.set_title('Pore Pressure vs Temperature')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Thermal expansion coefficient
        ax4 = axes[1, 1]
        thermal_data = self.data[self.data['test_type'] == 'thermal_exposure']
        for mix in thermal_data['mix_design'].unique():
            mix_data = thermal_data[thermal_data['mix_design'] == mix]
            if not mix_data.empty:
                cte_by_temp = mix_data.groupby('temperature_c')['thermal_expansion_coeff'].mean()
                ax4.plot(cte_by_temp.index, cte_by_temp.values * 1e6, 
                        marker='d', linewidth=2, label=mix)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Thermal Expansion Coefficient (×10⁻⁶/°C)')
        ax4.set_title('Thermal Expansion Coefficient vs Temperature')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, save_path: str = None):
        """Create an interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Compressive Strength vs Temperature', 'Tensile Strength vs Temperature',
                          'Modulus of Elasticity vs Temperature', 'Mass Loss vs Temperature',
                          'Spalling Depth vs Temperature', 'Thermal Strain vs Temperature'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Get data
        thermal_data = self.data[self.data['test_type'] == 'thermal_exposure']
        insitu_data = self.data[self.data['test_type'] == 'insitu_thermal']
        
        # Plot 1: Compressive strength vs temperature
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    strength_by_temp = mix_cooling_data.groupby('temperature_c')['compressive_strength_mpa'].mean()
                    fig.add_trace(
                        go.Scatter(x=strength_by_temp.index, y=strength_by_temp.values,
                                 mode='lines+markers', name=f'{mix} ({cooling})',
                                 line=dict(width=2)),
                        row=1, col=1
                    )
        
        # Plot 2: Tensile strength vs temperature
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    strength_by_temp = mix_cooling_data.groupby('temperature_c')['tensile_strength_mpa'].mean()
                    fig.add_trace(
                        go.Scatter(x=strength_by_temp.index, y=strength_by_temp.values,
                                 mode='lines+markers', name=f'{mix} ({cooling})',
                                 line=dict(width=2), showlegend=False),
                        row=1, col=2
                    )
        
        # Plot 3: Modulus of elasticity vs temperature
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    modulus_by_temp = mix_cooling_data.groupby('temperature_c')['modulus_elasticity_mpa'].mean()
                    fig.add_trace(
                        go.Scatter(x=modulus_by_temp.index, y=modulus_by_temp.values,
                                 mode='lines+markers', name=f'{mix} ({cooling})',
                                 line=dict(width=2), showlegend=False),
                        row=2, col=1
                    )
        
        # Plot 4: Mass loss vs temperature
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    mass_loss_by_temp = mix_cooling_data.groupby('temperature_c')['mass_loss_percent'].mean()
                    fig.add_trace(
                        go.Scatter(x=mass_loss_by_temp.index, y=mass_loss_by_temp.values,
                                 mode='lines+markers', name=f'{mix} ({cooling})',
                                 line=dict(width=2), showlegend=False),
                        row=2, col=2
                    )
        
        # Plot 5: Spalling depth vs temperature
        for mix in thermal_data['mix_design'].unique():
            for cooling in thermal_data['cooling_regime'].unique():
                mix_cooling_data = thermal_data[(thermal_data['mix_design'] == mix) & 
                                              (thermal_data['cooling_regime'] == cooling)]
                if not mix_cooling_data.empty:
                    spalling_by_temp = mix_cooling_data.groupby('temperature_c')['spalling_depth_mm'].mean()
                    fig.add_trace(
                        go.Scatter(x=spalling_by_temp.index, y=spalling_by_temp.values,
                                 mode='lines+markers', name=f'{mix} ({cooling})',
                                 line=dict(width=2), showlegend=False),
                        row=3, col=1
                    )
        
        # Plot 6: Thermal strain vs temperature
        for mix in insitu_data['mix_design'].unique():
            mix_data = insitu_data[insitu_data['mix_design'] == mix]
            if not mix_data.empty:
                strain_by_temp = mix_data.groupby('temperature_c')['thermal_strain_microstrain'].mean()
                fig.add_trace(
                    go.Scatter(x=strain_by_temp.index, y=strain_by_temp.values,
                             mode='lines+markers', name=f'{mix}',
                             line=dict(width=2), showlegend=False),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Rubberized Concrete Experimental Dataset - Interactive Dashboard",
            title_x=0.5,
            height=1200,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Compressive Strength (MPa)", row=1, col=1)
        fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Tensile Strength (MPa)", row=1, col=2)
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="Modulus of Elasticity (MPa)", row=2, col=1)
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=2)
        fig.update_yaxes(title_text="Mass Loss (%)", row=2, col=2)
        fig.update_xaxes(title_text="Temperature (°C)", row=3, col=1)
        fig.update_yaxes(title_text="Spalling Depth (mm)", row=3, col=1)
        fig.update_xaxes(title_text="Temperature (°C)", row=3, col=2)
        fig.update_yaxes(title_text="Thermal Strain (μstrain)", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for the dataset"""
        summary_stats = []
        
        for test_type in self.data['test_type'].unique():
            type_data = self.data[self.data['test_type'] == test_type]
            
            for mix in type_data['mix_design'].unique():
                mix_data = type_data[type_data['mix_design'] == mix]
                
                stats = {
                    'test_type': test_type,
                    'mix_design': mix,
                    'rubber_content_percent': mix_data['rubber_content_percent'].iloc[0],
                    'n_specimens': len(mix_data),
                    'compressive_strength_mean': mix_data['compressive_strength_mpa'].mean(),
                    'compressive_strength_std': mix_data['compressive_strength_mpa'].std(),
                    'tensile_strength_mean': mix_data['tensile_strength_mpa'].mean(),
                    'tensile_strength_std': mix_data['tensile_strength_mpa'].std(),
                    'modulus_elasticity_mean': mix_data['modulus_elasticity_mpa'].mean(),
                    'modulus_elasticity_std': mix_data['modulus_elasticity_mpa'].std(),
                    'density_mean': mix_data['density_kg_m3'].mean(),
                    'density_std': mix_data['density_kg_m3'].std(),
                    'upv_mean': mix_data['upv_m_s'].mean(),
                    'upv_std': mix_data['upv_m_s'].std(),
                    'mass_loss_mean': mix_data['mass_loss_percent'].mean(),
                    'mass_loss_std': mix_data['mass_loss_percent'].std(),
                    'spalling_depth_mean': mix_data['spalling_depth_mm'].mean(),
                    'spalling_depth_std': mix_data['spalling_depth_mm'].std()
                }
                
                summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)