#!/usr/bin/env python3
"""
Dataset Analyzer and Visualization Tool
For Stratified Flow Attenuation Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class DatasetAnalyzer:
    """
    Comprehensive analyzer for stratified flow datasets
    """
    
    def __init__(self, data_dir='stratified_flow_datasets'):
        """Initialize analyzer with dataset directory"""
        self.data_dir = data_dir
        self.datasets = {}
        self.load_datasets()
    
    def load_datasets(self):
        """Load all CSV datasets"""
        print("Loading datasets...")
        
        dataset_files = [
            'frequency_domain.csv',
            'experimental_conditions.csv', 
            'attenuation_models.csv',
            'multiphase_flow.csv',
            'time_series_features.csv'
        ]
        
        for file in dataset_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                dataset_name = file.replace('.csv', '')
                self.datasets[dataset_name] = pd.read_csv(file_path)
                print(f"✅ Loaded {dataset_name}: {len(self.datasets[dataset_name])} samples")
    
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary"""
        print("\nGenerating statistical summary...")
        
        summary_path = os.path.join(self.data_dir, 'statistical_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("STATISTICAL SUMMARY OF STRATIFIED FLOW DATASETS\n")
            f.write("=" * 60 + "\n\n")
            
            for dataset_name, df in self.datasets.items():
                f.write(f"{dataset_name.upper()} DATASET\n")
                f.write("-" * 40 + "\n")
                f.write(f"Shape: {df.shape}\n")
                f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n\n")
                
                # Numerical columns statistics
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    f.write("NUMERICAL FEATURES:\n")
                    stats_df = df[numerical_cols].describe()
                    f.write(stats_df.to_string())
                    f.write("\n\n")
                
                # Categorical columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    f.write("CATEGORICAL FEATURES:\n")
                    for col in categorical_cols:
                        f.write(f"{col}: {df[col].nunique()} unique values\n")
                        value_counts = df[col].value_counts().head(5)
                        for val, count in value_counts.items():
                            f.write(f"  {val}: {count} ({count/len(df)*100:.1f}%)\n")
                    f.write("\n")
                
                # Missing values
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    f.write("MISSING VALUES:\n")
                    for col, count in missing.items():
                        f.write(f"  {col}: {count} ({count/len(df)*100:.1f}%)\n")
                else:
                    f.write("No missing values detected.\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"📊 Statistical summary saved to: {summary_path}")
        return summary_path
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create visualization directory
        viz_dir = os.path.join(self.data_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. Frequency Domain Analysis
        if 'frequency_domain' in self.datasets:
            self._plot_frequency_domain_analysis(viz_dir)
        
        # 2. Flow Pattern Distribution
        if 'experimental_conditions' in self.datasets:
            self._plot_flow_pattern_analysis(viz_dir)
        
        # 3. Attenuation Models Comparison
        if 'attenuation_models' in self.datasets:
            self._plot_attenuation_models(viz_dir)
        
        # 4. Multiphase Flow Characteristics
        if 'multiphase_flow' in self.datasets:
            self._plot_multiphase_characteristics(viz_dir)
        
        # 5. Time Series Features
        if 'time_series_features' in self.datasets:
            self._plot_time_series_features(viz_dir)
        
        print(f"📈 Visualizations saved to: {viz_dir}")
        return viz_dir
    
    def _plot_frequency_domain_analysis(self, viz_dir):
        """Plot frequency domain analysis"""
        df = self.datasets['frequency_domain']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Frequency Domain Attenuation Analysis', fontsize=16, fontweight='bold')
        
        # Attenuation vs Frequency by Flow Configuration
        ax1 = axes[0, 0]
        for config in df['flow_configuration'].unique():
            config_data = df[df['flow_configuration'] == config]
            ax1.loglog(config_data['frequency_hz'], config_data['attenuation_db_per_m'], 
                      'o', alpha=0.6, label=config, markersize=3)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Attenuation (dB/m)')
        ax1.set_title('Attenuation vs Frequency by Flow Configuration')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gas Fraction vs Attenuation
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['gas_fraction'], df['attenuation_db_per_m'], 
                             c=df['frequency_hz'], cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('Gas Fraction')
        ax2.set_ylabel('Attenuation (dB/m)')
        ax2.set_title('Attenuation vs Gas Fraction (colored by frequency)')
        plt.colorbar(scatter, ax=ax2, label='Frequency (Hz)')
        
        # Attenuation Distribution by Flow Configuration
        ax3 = axes[1, 0]
        df.boxplot(column='attenuation_db_per_m', by='flow_configuration', ax=ax3)
        ax3.set_xlabel('Flow Configuration')
        ax3.set_ylabel('Attenuation (dB/m)')
        ax3.set_title('Attenuation Distribution by Flow Configuration')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Correlation Heatmap
        ax4 = axes[1, 1]
        numerical_cols = ['frequency_hz', 'attenuation_db_per_m', 'gas_fraction', 
                         'reynolds_number_gas', 'weber_number', 'temperature_c']
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'frequency_domain_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_flow_pattern_analysis(self, viz_dir):
        """Plot flow pattern analysis"""
        df = self.datasets['experimental_conditions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Flow Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Flow Pattern Distribution
        ax1 = axes[0, 0]
        flow_counts = df['predicted_flow_pattern'].value_counts()
        ax1.pie(flow_counts.values, labels=flow_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Flow Pattern Distribution')
        
        # Velocity Map
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['gas_superficial_velocity_ms'], df['liquid_superficial_velocity_ms'],
                             c=df['predicted_flow_pattern'].astype('category').cat.codes, 
                             cmap='tab10', alpha=0.7, s=30)
        ax2.set_xlabel('Gas Superficial Velocity (m/s)')
        ax2.set_ylabel('Liquid Superficial Velocity (m/s)')
        ax2.set_title('Flow Pattern Map')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Add flow pattern legend
        patterns = df['predicted_flow_pattern'].unique()
        for i, pattern in enumerate(patterns):
            ax2.scatter([], [], c=plt.cm.tab10(i), label=pattern, s=50)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Pipe Diameter Distribution
        ax3 = axes[1, 0]
        ax3.hist(df['pipe_diameter_m'], bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Pipe Diameter (m)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Pipe Diameter Distribution')
        
        # Temperature vs Pressure
        ax4 = axes[1, 1]
        ax4.scatter(df['temperature_c'], df['pressure_bar'], alpha=0.6, s=20)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Pressure (bar)')
        ax4.set_title('Operating Conditions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'flow_pattern_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attenuation_models(self, viz_dir):
        """Plot attenuation models comparison"""
        df = self.datasets['attenuation_models']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attenuation Models Comparison', fontsize=16, fontweight='bold')
        
        # Model Performance by Type
        ax1 = axes[0, 0]
        df.boxplot(column='predicted_attenuation_db_per_m', by='model_type', ax=ax1)
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Predicted Attenuation (dB/m)')
        ax1.set_title('Attenuation Predictions by Model Type')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Frequency Response
        ax2 = axes[0, 1]
        for model in df['model_type'].unique():
            model_data = df[df['model_type'] == model]
            ax2.loglog(model_data['frequency_hz'], model_data['predicted_attenuation_db_per_m'],
                      'o', alpha=0.6, label=model, markersize=3)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Predicted Attenuation (dB/m)')
        ax2.set_title('Model Predictions vs Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Temperature Dependence
        ax3 = axes[1, 0]
        for model in df['model_type'].unique()[:3]:  # Show top 3 models
            model_data = df[df['model_type'] == model]
            ax3.scatter(model_data['temperature_c'], model_data['predicted_attenuation_db_per_m'],
                       alpha=0.6, label=model, s=20)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Predicted Attenuation (dB/m)')
        ax3.set_title('Temperature Dependence (Selected Models)')
        ax3.legend()
        
        # Model Distribution
        ax4 = axes[1, 1]
        model_counts = df['model_type'].value_counts()
        ax4.bar(range(len(model_counts)), model_counts.values)
        ax4.set_xticks(range(len(model_counts)))
        ax4.set_xticklabels(model_counts.index, rotation=45)
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Sample Distribution by Model Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'attenuation_models.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_multiphase_characteristics(self, viz_dir):
        """Plot multiphase flow characteristics"""
        df = self.datasets['multiphase_flow']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multiphase Flow Characteristics', fontsize=16, fontweight='bold')
        
        # Flow Regime Distribution
        ax1 = axes[0, 0]
        regime_counts = df['flow_regime'].value_counts()
        ax1.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Flow Regime Distribution')
        
        # Gas Fraction vs Sound Speed
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['gas_fraction'], df['mixture_sound_speed_ms'],
                             c=df['flow_regime'].astype('category').cat.codes,
                             cmap='tab10', alpha=0.7, s=30)
        ax2.set_xlabel('Gas Fraction')
        ax2.set_ylabel('Mixture Sound Speed (m/s)')
        ax2.set_title('Sound Speed vs Gas Fraction')
        
        # Attenuation Components
        ax3 = axes[1, 0]
        components = ['scattering_component_db_per_m', 'viscous_component_db_per_m', 
                     'interface_component_db_per_m']
        component_data = df[components].mean()
        ax3.bar(range(len(component_data)), component_data.values)
        ax3.set_xticks(range(len(component_data)))
        ax3.set_xticklabels(['Scattering', 'Viscous', 'Interface'], rotation=45)
        ax3.set_ylabel('Average Attenuation (dB/m)')
        ax3.set_title('Attenuation Mechanism Contributions')
        
        # Total Attenuation vs Gas Fraction
        ax4 = axes[1, 1]
        for regime in df['flow_regime'].unique():
            regime_data = df[df['flow_regime'] == regime]
            ax4.scatter(regime_data['gas_fraction'], regime_data['total_attenuation_db_per_m_at_1khz'],
                       alpha=0.7, label=regime, s=30)
        ax4.set_xlabel('Gas Fraction')
        ax4.set_ylabel('Total Attenuation at 1kHz (dB/m)')
        ax4.set_title('Attenuation vs Gas Fraction by Flow Regime')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'multiphase_characteristics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_features(self, viz_dir):
        """Plot time series features analysis"""
        df = self.datasets['time_series_features']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Time Series Acoustic Features', fontsize=16, fontweight='bold')
        
        # RMS Amplitude by Flow Pattern
        ax1 = axes[0, 0]
        df.boxplot(column='rms_amplitude', by='flow_pattern', ax=ax1)
        ax1.set_xlabel('Flow Pattern')
        ax1.set_ylabel('RMS Amplitude')
        ax1.set_title('RMS Amplitude Distribution by Flow Pattern')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Spectral Features
        ax2 = axes[0, 1]
        ax2.scatter(df['dominant_frequency_hz'], df['spectral_centroid_hz'],
                   c=df['flow_pattern'].astype('category').cat.codes,
                   cmap='tab10', alpha=0.7, s=30)
        ax2.set_xlabel('Dominant Frequency (Hz)')
        ax2.set_ylabel('Spectral Centroid (Hz)')
        ax2.set_title('Spectral Features by Flow Pattern')
        
        # Feature Correlation
        ax3 = axes[1, 0]
        feature_cols = ['rms_amplitude', 'dominant_frequency_hz', 'spectral_centroid_hz',
                       'zero_crossings_per_sec', 'spectral_bandwidth_hz']
        corr_matrix = df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Feature Correlation Matrix')
        
        # Zero Crossings Distribution
        ax4 = axes[1, 1]
        for pattern in df['flow_pattern'].unique():
            pattern_data = df[df['flow_pattern'] == pattern]
            ax4.hist(pattern_data['zero_crossings_per_sec'], alpha=0.6, 
                    label=pattern, bins=20, density=True)
        ax4.set_xlabel('Zero Crossings per Second')
        ax4.set_ylabel('Density')
        ax4.set_title('Zero Crossings Rate Distribution')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'time_series_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_research_insights(self):
        """Generate research insights and recommendations"""
        print("Generating research insights...")
        
        insights_path = os.path.join(self.data_dir, 'research_insights.md')
        
        with open(insights_path, 'w') as f:
            f.write("# Research Insights: Stratified Flow Attenuation Mechanisms\n\n")
            f.write("## Dataset Overview\n\n")
            
            total_samples = sum(len(df) for df in self.datasets.values())
            f.write(f"- **Total Samples**: {total_samples:,}\n")
            f.write(f"- **Datasets Generated**: {len(self.datasets)}\n")
            f.write(f"- **Research Focus**: Beyond Single Phase Leakage Acoustics\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Frequency Domain Insights
            if 'frequency_domain' in self.datasets:
                df_freq = self.datasets['frequency_domain']
                f.write("### Frequency Domain Analysis\n\n")
                f.write(f"- **Frequency Range**: {df_freq['frequency_hz'].min():.0f} - {df_freq['frequency_hz'].max():.0f} Hz\n")
                f.write(f"- **Attenuation Range**: {df_freq['attenuation_db_per_m'].min():.2f} - {df_freq['attenuation_db_per_m'].max():.2f} dB/m\n")
                
                # Flow configuration with highest attenuation
                max_atten_config = df_freq.groupby('flow_configuration')['attenuation_db_per_m'].mean().idxmax()
                f.write(f"- **Highest Attenuation Configuration**: {max_atten_config}\n")
                
                # Gas fraction correlation
                gas_corr = df_freq['gas_fraction'].corr(df_freq['attenuation_db_per_m'])
                f.write(f"- **Gas Fraction-Attenuation Correlation**: {gas_corr:.3f}\n\n")
            
            # Multiphase Flow Insights
            if 'multiphase_flow' in self.datasets:
                df_multi = self.datasets['multiphase_flow']
                f.write("### Multiphase Flow Characteristics\n\n")
                
                # Dominant flow regime
                dominant_regime = df_multi['flow_regime'].mode()[0]
                f.write(f"- **Most Common Flow Regime**: {dominant_regime}\n")
                
                # Sound speed variation
                sound_speed_range = df_multi['mixture_sound_speed_ms'].max() - df_multi['mixture_sound_speed_ms'].min()
                f.write(f"- **Sound Speed Variation**: {sound_speed_range:.0f} m/s\n")
                
                # Attenuation mechanism dominance
                components = ['scattering_component_db_per_m', 'viscous_component_db_per_m', 'interface_component_db_per_m']
                dominant_mechanism = df_multi[components].mean().idxmax().replace('_component_db_per_m', '')
                f.write(f"- **Dominant Attenuation Mechanism**: {dominant_mechanism}\n\n")
            
            f.write("## Research Recommendations\n\n")
            f.write("### Experimental Priorities\n\n")
            f.write("1. **High-Frequency Measurements**: Focus on frequencies >10 kHz for enhanced sensitivity\n")
            f.write("2. **Multi-Modal Sensing**: Combine acoustic with optical/electrical measurements\n")
            f.write("3. **Interface Characterization**: Detailed roughness and wave amplitude measurements\n")
            f.write("4. **Temperature Effects**: Systematic study of thermal influence on attenuation\n\n")
            
            f.write("### Theoretical Development\n\n")
            f.write("1. **Multi-Scale Modeling**: Bridge molecular to continuum scales\n")
            f.write("2. **Machine Learning Integration**: Use datasets for pattern recognition\n")
            f.write("3. **Uncertainty Quantification**: Develop probabilistic attenuation models\n")
            f.write("4. **Real-Time Applications**: Fast algorithms for online monitoring\n\n")
            
            f.write("### Future Work\n\n")
            f.write("1. **Validation Experiments**: Compare predictions with controlled lab experiments\n")
            f.write("2. **Industrial Applications**: Test in realistic pipeline conditions\n")
            f.write("3. **Advanced Materials**: Study of complex fluid mixtures\n")
            f.write("4. **Sensor Development**: Optimize transducer design for stratified flows\n")
        
        print(f"🔬 Research insights saved to: {insights_path}")
        return insights_path
    
    def run_complete_analysis(self):
        """Run complete dataset analysis"""
        print("Running complete dataset analysis...")
        
        # Generate statistical summary
        summary_path = self.generate_statistical_summary()
        
        # Create visualizations
        viz_dir = self.create_visualizations()
        
        # Generate research insights
        insights_path = self.generate_research_insights()
        
        print(f"\n🎯 Complete analysis finished!")
        print(f"📊 Statistical summary: {summary_path}")
        print(f"📈 Visualizations: {viz_dir}")
        print(f"🔬 Research insights: {insights_path}")
        
        return {
            'summary': summary_path,
            'visualizations': viz_dir,
            'insights': insights_path
        }

def main():
    """Main function"""
    print("Dataset Analysis for Stratified Flow Attenuation Research")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print(f"\n✅ Analysis complete! Check the results:")
    for key, path in results.items():
        print(f"   {key}: {path}")

if __name__ == "__main__":
    main()