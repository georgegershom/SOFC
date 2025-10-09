#!/usr/bin/env python3
"""
SOFC Thermal Analysis and Visualization
Comprehensive analysis of thermal history data including:
- Temperature profile visualization
- Thermal stress analysis
- Delamination risk assessment
- Performance degradation modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SOFCThermalAnalyzer:
    def __init__(self):
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def analyze_sintering_data(self, sintering_data):
        """Analyze sintering temperature profiles"""
        print("Analyzing sintering data...")
        
        # Temperature statistics
        temp_stats = sintering_data.groupby('time')['temperature'].agg(['mean', 'std', 'min', 'max'])
        
        # Create comprehensive sintering analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SOFC Sintering Process Analysis', fontsize=16, fontweight='bold')
        
        # 1. Temperature vs Time
        axes[0, 0].plot(temp_stats.index/3600, temp_stats['mean'], 'b-', linewidth=2, label='Mean Temperature')
        axes[0, 0].fill_between(temp_stats.index/3600, 
                               temp_stats['mean'] - temp_stats['std'],
                               temp_stats['mean'] + temp_stats['std'],
                               alpha=0.3, color='blue', label='±1σ')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].set_title('Temperature Profile During Sintering')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spatial temperature distribution at peak temperature
        peak_time = sintering_data.loc[sintering_data['temperature'].idxmax(), 'time']
        peak_data = sintering_data[sintering_data['time'] == peak_time]
        
        if len(peak_data) > 0:
            # Create spatial grid
            x_unique = sorted(peak_data['x_position'].unique())
            y_unique = sorted(peak_data['y_position'].unique())
            X, Y = np.meshgrid(x_unique, y_unique)
            
            # Interpolate temperature data
            Z = griddata((peak_data['x_position'], peak_data['y_position']),
                        peak_data['temperature'], (X, Y), method='cubic')
            
            im = axes[0, 1].contourf(X*1000, Y*1000, Z, levels=20, cmap='hot')
            axes[0, 1].set_xlabel('X Position (mm)')
            axes[0, 1].set_ylabel('Y Position (mm)')
            axes[0, 1].set_title(f'Temperature Distribution at Peak (t={peak_time/3600:.1f}h)')
            plt.colorbar(im, ax=axes[0, 1], label='Temperature (°C)')
        
        # 3. Temperature gradient analysis
        grad_magnitude = np.sqrt(sintering_data['gradient_x']**2 + sintering_data['gradient_y']**2)
        axes[1, 0].hist(grad_magnitude, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Temperature Gradient Magnitude (°C/m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Temperature Gradient Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Temperature uniformity over time
        uniformity = temp_stats['std'] / temp_stats['mean'] * 100
        axes[1, 1].plot(temp_stats.index/3600, uniformity, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Temperature Non-uniformity (%)')
        axes[1, 1].set_title('Temperature Uniformity During Sintering')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sofc_sintering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return temp_stats
    
    def analyze_thermal_cycling(self, cycling_data):
        """Analyze thermal cycling effects"""
        print("Analyzing thermal cycling data...")
        
        # Cycle-by-cycle analysis
        cycle_stats = cycling_data.groupby('cycle_number').agg({
            'temperature': ['mean', 'std', 'min', 'max'],
            'temp_range': 'mean',
            'max_temp': 'mean',
            'min_temp': 'mean'
        }).round(2)
        
        # Create cycling analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SOFC Thermal Cycling Analysis', fontsize=16, fontweight='bold')
        
        # 1. Temperature evolution over cycles
        cycle_means = cycling_data.groupby('cycle_number')['temperature'].mean()
        cycle_stds = cycling_data.groupby('cycle_number')['temperature'].std()
        
        axes[0, 0].errorbar(cycle_means.index, cycle_means.values, 
                           yerr=cycle_stds.values, fmt='o-', capsize=5, capthick=2)
        axes[0, 0].set_xlabel('Cycle Number')
        axes[0, 0].set_ylabel('Mean Temperature (°C)')
        axes[0, 0].set_title('Temperature Evolution Over Cycles')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Temperature range per cycle
        temp_ranges = cycling_data.groupby('cycle_number')['temp_range'].mean()
        axes[0, 1].plot(temp_ranges.index, temp_ranges.values, 'ro-', linewidth=2)
        axes[0, 1].set_xlabel('Cycle Number')
        axes[0, 1].set_ylabel('Temperature Range (°C)')
        axes[0, 1].set_title('Temperature Range Per Cycle')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Phase analysis
        phase_data = cycling_data.groupby(['cycle_number', 'phase'])['temperature'].mean().unstack()
        phase_data.plot(kind='line', ax=axes[1, 0], marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Cycle Number')
        axes[1, 0].set_ylabel('Mean Temperature (°C)')
        axes[1, 0].set_title('Temperature by Phase')
        axes[1, 0].legend(title='Phase')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Delamination risk assessment
        # Higher temperature ranges and gradients increase delamination risk
        delamination_risk = temp_ranges * 0.1 + cycle_means * 0.001  # Simplified risk metric
        axes[1, 1].plot(delamination_risk.index, delamination_risk.values, 'r-', linewidth=2)
        axes[1, 1].fill_between(delamination_risk.index, 0, delamination_risk.values, 
                               alpha=0.3, color='red')
        axes[1, 1].set_xlabel('Cycle Number')
        axes[1, 1].set_ylabel('Delamination Risk Index')
        axes[1, 1].set_title('Cumulative Delamination Risk')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sofc_thermal_cycling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cycle_stats
    
    def analyze_steady_state(self, steady_state_data):
        """Analyze steady-state temperature gradients"""
        print("Analyzing steady-state data...")
        
        # Layer-wise analysis
        layer_stats = steady_state_data.groupby('layer')['temperature'].agg(['mean', 'std', 'min', 'max'])
        
        # Create 3D visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 3D temperature distribution
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Sample data for 3D plot (every 5th point for clarity)
        sample_data = steady_state_data[::5]
        
        scatter = ax1.scatter(sample_data['x_position']*1000, 
                             sample_data['y_position']*1000,
                             sample_data['z_position']*1e6,
                             c=sample_data['temperature'],
                             cmap='hot', s=20)
        ax1.set_xlabel('X Position (mm)')
        ax1.set_ylabel('Y Position (mm)')
        ax1.set_zlabel('Z Position (μm)')
        ax1.set_title('3D Temperature Distribution')
        plt.colorbar(scatter, ax=ax1, label='Temperature (°C)')
        
        # Temperature profiles through thickness
        ax2 = fig.add_subplot(222)
        for layer in steady_state_data['layer'].unique():
            layer_data = steady_state_data[steady_state_data['layer'] == layer]
            z_pos = layer_data['z_position'].mean()
            temp_mean = layer_data.groupby('z_position')['temperature'].mean()
            ax2.plot(temp_mean.values, temp_mean.index*1e6, 'o-', label=layer, linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Z Position (μm)')
        ax2.set_title('Temperature Profile Through Thickness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Temperature distribution by layer
        ax3 = fig.add_subplot(223)
        steady_state_data.boxplot(column='temperature', by='layer', ax=ax3)
        ax3.set_title('Temperature Distribution by Layer')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Temperature (°C)')
        
        # Heat flux analysis (simplified)
        ax4 = fig.add_subplot(224)
        # Calculate approximate heat flux based on temperature gradients
        for layer in steady_state_data['layer'].unique():
            layer_data = steady_state_data[steady_state_data['layer'] == layer]
            temp_grad = layer_data.groupby('z_position')['temperature'].std()
            ax4.plot(temp_grad.values, temp_grad.index*1e6, 'o-', label=layer, linewidth=2)
        ax4.set_xlabel('Temperature Gradient (°C)')
        ax4.set_ylabel('Z Position (μm)')
        ax4.set_title('Temperature Gradient Through Thickness')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sofc_steady_state_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return layer_stats
    
    def analyze_residual_stresses(self, stress_data):
        """Analyze residual stress patterns"""
        print("Analyzing residual stress data...")
        
        # Stress statistics by layer
        stress_stats = stress_data.groupby('layer')['thermal_stress'].agg(['mean', 'std', 'min', 'max'])
        
        # Create stress analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SOFC Residual Stress Analysis', fontsize=16, fontweight='bold')
        
        # 1. Stress distribution by layer
        stress_data.boxplot(column='thermal_stress', by='layer', ax=axes[0, 0])
        axes[0, 0].set_title('Thermal Stress Distribution by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Thermal Stress (Pa)')
        
        # 2. Stress vs Temperature
        for layer in stress_data['layer'].unique():
            layer_data = stress_data[stress_data['layer'] == layer]
            axes[0, 1].scatter(layer_data['temperature'], layer_data['thermal_stress'], 
                              label=layer, alpha=0.6, s=10)
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Thermal Stress (Pa)')
        axes[0, 1].set_title('Stress vs Temperature')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stress evolution over time
        time_stress = stress_data.groupby('time')['thermal_stress'].mean()
        axes[1, 0].plot(time_stress.index/3600, time_stress.values/1e6, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Mean Thermal Stress (MPa)')
        axes[1, 0].set_title('Stress Evolution During Sintering')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Stress concentration analysis
        stress_data['stress_magnitude'] = np.abs(stress_data['thermal_stress'])
        high_stress = stress_data[stress_data['stress_magnitude'] > stress_data['stress_magnitude'].quantile(0.9)]
        
        axes[1, 1].scatter(high_stress['x_position']*1000, high_stress['y_position']*1000,
                          c=high_stress['thermal_stress']/1e6, cmap='RdBu_r', s=50)
        axes[1, 1].set_xlabel('X Position (mm)')
        axes[1, 1].set_ylabel('Y Position (mm)')
        axes[1, 1].set_title('High Stress Regions (>90th percentile)')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('sofc_residual_stress_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stress_stats
    
    def generate_interactive_plots(self, sintering_data, cycling_data, steady_state_data):
        """Generate interactive Plotly visualizations"""
        print("Generating interactive visualizations...")
        
        # Interactive sintering temperature profile
        fig1 = go.Figure()
        
        # Add temperature trace
        temp_stats = sintering_data.groupby('time')['temperature'].agg(['mean', 'std'])
        fig1.add_trace(go.Scatter(
            x=temp_stats.index/3600,
            y=temp_stats['mean'],
            mode='lines',
            name='Mean Temperature',
            line=dict(color='blue', width=3)
        ))
        
        # Add error bars
        fig1.add_trace(go.Scatter(
            x=temp_stats.index/3600,
            y=temp_stats['mean'] + temp_stats['std'],
            mode='lines',
            name='+1σ',
            line=dict(color='blue', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig1.add_trace(go.Scatter(
            x=temp_stats.index/3600,
            y=temp_stats['mean'] - temp_stats['std'],
            mode='lines',
            name='-1σ',
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            showlegend=False
        ))
        
        fig1.update_layout(
            title='SOFC Sintering Temperature Profile',
            xaxis_title='Time (hours)',
            yaxis_title='Temperature (°C)',
            hovermode='x unified'
        )
        
        fig1.write_html('sofc_sintering_interactive.html')
        
        # Interactive 3D steady-state visualization
        fig2 = go.Figure(data=go.Scatter3d(
            x=steady_state_data['x_position']*1000,
            y=steady_state_data['y_position']*1000,
            z=steady_state_data['z_position']*1e6,
            mode='markers',
            marker=dict(
                size=3,
                color=steady_state_data['temperature'],
                colorscale='hot',
                opacity=0.8,
                colorbar=dict(title="Temperature (°C)")
            ),
            text=steady_state_data['layer'],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f} mm<br>' +
                         'Y: %{y:.2f} mm<br>' +
                         'Z: %{z:.2f} μm<br>' +
                         'Temperature: %{marker.color:.1f} °C<br>' +
                         '<extra></extra>'
        ))
        
        fig2.update_layout(
            title='SOFC 3D Temperature Distribution (Steady-State)',
            scene=dict(
                xaxis_title='X Position (mm)',
                yaxis_title='Y Position (mm)',
                zaxis_title='Z Position (μm)'
            )
        )
        
        fig2.write_html('sofc_steady_state_3d.html')
        
        print("Interactive plots saved as HTML files")
    
    def generate_comprehensive_report(self, sintering_data, cycling_data, steady_state_data, stress_data):
        """Generate comprehensive analysis report"""
        print("Generating comprehensive report...")
        
        report = f"""
# SOFC Thermal History Data Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents comprehensive thermal analysis of SOFC operation including sintering, thermal cycling, and steady-state operation.

## 1. Sintering Process Analysis
- **Total Data Points**: {len(sintering_data):,}
- **Temperature Range**: {sintering_data['temperature'].min():.1f}°C to {sintering_data['temperature'].max():.1f}°C
- **Process Duration**: {sintering_data['time'].max()/3600:.1f} hours
- **Peak Temperature**: {sintering_data['temperature'].max():.1f}°C
- **Temperature Uniformity**: {sintering_data.groupby('time')['temperature'].std().mean():.1f}°C standard deviation

## 2. Thermal Cycling Analysis
- **Total Cycles**: {cycling_data['cycle_number'].max()}
- **Total Data Points**: {len(cycling_data):,}
- **Temperature Range**: {cycling_data['temperature'].min():.1f}°C to {cycling_data['temperature'].max():.1f}°C
- **Average Temperature Range per Cycle**: {cycling_data.groupby('cycle_number')['temp_range'].mean().mean():.1f}°C
- **Maximum Temperature Range**: {cycling_data['temp_range'].max():.1f}°C

## 3. Steady-State Operation Analysis
- **Total Data Points**: {len(steady_state_data):,}
- **Temperature Range**: {steady_state_data['temperature'].min():.1f}°C to {steady_state_data['temperature'].max():.1f}°C
- **Layers Analyzed**: {', '.join(steady_state_data['layer'].unique())}

### Layer-wise Temperature Statistics:
"""
        
        layer_stats = steady_state_data.groupby('layer')['temperature'].agg(['mean', 'std', 'min', 'max'])
        for layer, stats in layer_stats.iterrows():
            report += f"- **{layer.capitalize()}**: Mean={stats['mean']:.1f}°C, Std={stats['std']:.1f}°C, Range={stats['min']:.1f}-{stats['max']:.1f}°C\n"
        
        report += f"""
## 4. Residual Stress Analysis
- **Total Data Points**: {len(stress_data):,}
- **Stress Range**: {stress_data['thermal_stress'].min()/1e6:.1f} to {stress_data['thermal_stress'].max()/1e6:.1f} MPa
- **Strain Range**: {stress_data['thermal_strain'].min()*1e6:.1f} to {stress_data['thermal_strain'].max()*1e6:.1f} με

### Layer-wise Stress Statistics:
"""
        
        stress_stats = stress_data.groupby('layer')['thermal_stress'].agg(['mean', 'std', 'min', 'max'])
        for layer, stats in stress_stats.iterrows():
            report += f"- **{layer.capitalize()}**: Mean={stats['mean']/1e6:.1f} MPa, Std={stats['std']/1e6:.1f} MPa, Range={stats['min']/1e6:.1f}-{stats['max']/1e6:.1f} MPa\n"
        
        report += f"""
## 5. Key Findings and Recommendations

### Sintering Process:
- Temperature uniformity during sintering is critical for minimizing residual stresses
- Peak temperature of {sintering_data['temperature'].max():.1f}°C achieved with {sintering_data.groupby('time')['temperature'].std().mean():.1f}°C standard deviation
- Spatial temperature gradients may lead to non-uniform sintering and residual stress

### Thermal Cycling:
- {cycling_data['cycle_number'].max()} thermal cycles analyzed
- Average temperature range of {cycling_data.groupby('cycle_number')['temp_range'].mean().mean():.1f}°C per cycle
- Thermal cycling is a major contributor to delamination risk

### Steady-State Operation:
- Temperature gradients across the cell thickness are significant
- {steady_state_data['layer'].nunique()} distinct layers with different thermal properties
- Current density effects on temperature distribution are evident

### Residual Stresses:
- Maximum stress of {stress_data['thermal_stress'].max()/1e6:.1f} MPa observed
- Stress distribution varies significantly across layers
- Thermal expansion mismatch between layers contributes to stress development

## 6. Data Files Generated
- `sofc_sintering_data.csv`: Sintering temperature profiles
- `sofc_thermal_cycling_data.csv`: Thermal cycling data
- `sofc_steady_state_data.csv`: Steady-state temperature gradients
- `sofc_residual_stress_data.csv`: Residual stress calculations
- `sofc_thermal_data_summary.json`: Summary statistics

## 7. Visualization Files
- `sofc_sintering_analysis.png`: Sintering process analysis
- `sofc_thermal_cycling_analysis.png`: Thermal cycling analysis
- `sofc_steady_state_analysis.png`: Steady-state analysis
- `sofc_residual_stress_analysis.png`: Residual stress analysis
- `sofc_sintering_interactive.html`: Interactive sintering plot
- `sofc_steady_state_3d.html`: Interactive 3D visualization

---
*Report generated by SOFC Thermal Analysis System*
"""
        
        with open('sofc_thermal_analysis_report.md', 'w') as f:
            f.write(report)
        
        print("Comprehensive report saved as 'sofc_thermal_analysis_report.md'")
        return report

def main():
    """Main analysis function"""
    print("SOFC Thermal Analysis System")
    print("=" * 50)
    
    # Load data
    try:
        sintering_data = pd.read_csv('sofc_sintering_data.csv')
        cycling_data = pd.read_csv('sofc_thermal_cycling_data.csv')
        steady_state_data = pd.read_csv('sofc_steady_state_data.csv')
        stress_data = pd.read_csv('sofc_residual_stress_data.csv')
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Data files not found. Please run the data generator first.")
        return
    
    # Initialize analyzer
    analyzer = SOFCThermalAnalyzer()
    
    # Perform analyses
    sintering_stats = analyzer.analyze_sintering_data(sintering_data)
    cycling_stats = analyzer.analyze_thermal_cycling(cycling_data)
    steady_state_stats = analyzer.analyze_steady_state(steady_state_data)
    stress_stats = analyzer.analyze_residual_stresses(stress_data)
    
    # Generate interactive plots
    analyzer.generate_interactive_plots(sintering_data, cycling_data, steady_state_data)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(sintering_data, cycling_data, steady_state_data, stress_data)
    
    print("\nAnalysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main()