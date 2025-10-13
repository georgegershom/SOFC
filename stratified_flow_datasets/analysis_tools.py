#!/usr/bin/env python3
"""
Analysis Tools for Stratified Flow Attenuation Datasets
========================================================

This module provides tools for loading, analyzing, and visualizing
the stratified flow attenuation datasets.

Author: PhD Research Project
Date: January 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class StratifiedFlowAnalyzer:
    """Main class for analyzing stratified flow datasets."""
    
    def __init__(self, data_path="./"):
        """Initialize the analyzer with dataset path."""
        self.data_path = Path(data_path)
        self.datasets = {}
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """Load dataset metadata."""
        try:
            with open(self.data_path / "metadata.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Metadata file not found. Using default structure.")
            return {}
    
    def load_dataset(self, category, filename=None):
        """
        Load a specific dataset.
        
        Parameters:
        -----------
        category : str
            Dataset category ('single_phase_baseline', 'published_datasets', 
                            'material_properties', 'signal_processing')
        filename : str, optional
            Specific filename to load. If None, loads all files in category.
        """
        category_map = {
            'single_phase_baseline': '01_single_phase_baseline',
            'published_datasets': '02_published_datasets',
            'material_properties': '03_material_geometric_properties',
            'signal_processing': '04_signal_processing'
        }
        
        folder_path = self.data_path / category_map[category]
        
        if filename:
            file_path = folder_path / filename
            if file_path.exists():
                self.datasets[f"{category}_{filename}"] = pd.read_csv(file_path)
                print(f"Loaded {filename} from {category}")
            else:
                print(f"File {filename} not found in {category}")
        else:
            # Load all CSV files in the category
            for csv_file in folder_path.glob("*.csv"):
                dataset_key = f"{category}_{csv_file.name}"
                self.datasets[dataset_key] = pd.read_csv(csv_file)
                print(f"Loaded {csv_file.name} from {category}")
    
    def load_all_datasets(self):
        """Load all available datasets."""
        categories = ['single_phase_baseline', 'published_datasets', 
                     'material_properties', 'signal_processing']
        for category in categories:
            self.load_dataset(category)
    
    def get_dataset_summary(self):
        """Get summary statistics for all loaded datasets."""
        summary = {}
        for name, df in self.datasets.items():
            summary[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'missing_values': df.isnull().sum().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
        return pd.DataFrame(summary).T
    
    def plot_baseline_comparison(self):
        """Plot comparison between water and air baseline measurements."""
        if 'single_phase_baseline_water_baseline_data.csv' not in self.datasets:
            print("Water baseline data not loaded")
            return
        if 'single_phase_baseline_air_baseline_data.csv' not in self.datasets:
            print("Air baseline data not loaded")
            return
            
        water_data = self.datasets['single_phase_baseline_water_baseline_data.csv']
        air_data = self.datasets['single_phase_baseline_air_baseline_data.csv']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sound speed comparison
        for temp in water_data['temperature_c'].unique():
            water_temp = water_data[water_data['temperature_c'] == temp]
            ax1.semilogx(water_temp['frequency_hz'], water_temp['sound_speed_ms'], 
                        label=f'Water {temp}°C', marker='o', markersize=4)
        
        for temp in air_data['temperature_c'].unique():
            air_temp = air_data[air_data['temperature_c'] == temp]
            ax1.semilogx(air_temp['frequency_hz'], air_temp['sound_speed_ms'], 
                        label=f'Air {temp}°C', marker='s', markersize=4, linestyle='--')
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Sound Speed (m/s)')
        ax1.set_title('Sound Speed vs Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Attenuation comparison
        for temp in water_data['temperature_c'].unique():
            water_temp = water_data[water_data['temperature_c'] == temp]
            ax2.loglog(water_temp['frequency_hz'], water_temp['attenuation_db_m'], 
                      label=f'Water {temp}°C', marker='o', markersize=4)
        
        for temp in air_data['temperature_c'].unique():
            air_temp = air_data[air_data['temperature_c'] == temp]
            ax2.loglog(air_temp['frequency_hz'], air_temp['attenuation_db_m'], 
                      label=f'Air {temp}°C', marker='s', markersize=4, linestyle='--')
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Attenuation (dB/m)')
        ax2.set_title('Attenuation vs Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Temperature effects on sound speed
        water_1khz = water_data[water_data['frequency_hz'] == 1000]
        air_1khz = air_data[air_data['frequency_hz'] == 1000]
        
        ax3.plot(water_1khz['temperature_c'], water_1khz['sound_speed_ms'], 
                'bo-', label='Water', markersize=8)
        ax3.plot(air_1khz['temperature_c'], air_1khz['sound_speed_ms'], 
                'rs--', label='Air', markersize=8)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Sound Speed (m/s)')
        ax3.set_title('Temperature Effect on Sound Speed (1 kHz)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Density vs sound speed
        ax4.scatter(water_data['density_kg_m3'], water_data['sound_speed_ms'], 
                   c=water_data['temperature_c'], cmap='viridis', 
                   label='Water', alpha=0.7, s=50)
        ax4.scatter(air_data['density_kg_m3'], air_data['sound_speed_ms'], 
                   c=air_data['temperature_c'], cmap='plasma', 
                   label='Air', alpha=0.7, s=50, marker='s')
        ax4.set_xlabel('Density (kg/m³)')
        ax4.set_ylabel('Sound Speed (m/s)')
        ax4.set_title('Density vs Sound Speed')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_published_data_comparison(self):
        """Compare data from different published sources."""
        published_files = [k for k in self.datasets.keys() if 'published_datasets' in k]
        
        if len(published_files) == 0:
            print("No published datasets loaded")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green']
        markers = ['o', 's', '^']
        
        for i, dataset_key in enumerate(published_files):
            df = self.datasets[dataset_key]
            source = dataset_key.split('_')[-2]  # Extract source name
            
            # Sound speed vs void fraction
            if 'void_fraction' in df.columns and 'sound_speed_ms' in df.columns:
                ax1.scatter(df['void_fraction'], df['sound_speed_ms'], 
                           c=colors[i], marker=markers[i], label=source, 
                           alpha=0.7, s=50)
        
        ax1.set_xlabel('Void Fraction')
        ax1.set_ylabel('Sound Speed (m/s)')
        ax1.set_title('Sound Speed vs Void Fraction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Attenuation vs frequency for different sources
        for i, dataset_key in enumerate(published_files):
            df = self.datasets[dataset_key]
            source = dataset_key.split('_')[-2]
            
            if 'frequency_hz' in df.columns and 'attenuation_db_m' in df.columns:
                # Group by frequency and plot mean values
                freq_groups = df.groupby('frequency_hz')['attenuation_db_m'].mean()
                ax2.loglog(freq_groups.index, freq_groups.values, 
                          color=colors[i], marker=markers[i], label=source, 
                          markersize=6, linewidth=2)
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Attenuation (dB/m)')
        ax2.set_title('Attenuation vs Frequency (Published Data)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Flow regime distribution
        flow_regimes = []
        sources = []
        for i, dataset_key in enumerate(published_files):
            df = self.datasets[dataset_key]
            source = dataset_key.split('_')[-2]
            
            if 'flow_regime' in df.columns:
                regimes = df['flow_regime'].value_counts()
                for regime, count in regimes.items():
                    flow_regimes.extend([regime] * count)
                    sources.extend([source] * count)
            elif 'flow_pattern' in df.columns:
                regimes = df['flow_pattern'].value_counts()
                for regime, count in regimes.items():
                    flow_regimes.extend([regime] * count)
                    sources.extend([source] * count)
        
        if flow_regimes:
            regime_df = pd.DataFrame({'regime': flow_regimes, 'source': sources})
            regime_counts = regime_df.groupby(['regime', 'source']).size().unstack(fill_value=0)
            regime_counts.plot(kind='bar', ax=ax3, color=colors[:len(regime_counts.columns)])
            ax3.set_xlabel('Flow Regime')
            ax3.set_ylabel('Count')
            ax3.set_title('Flow Regime Distribution by Source')
            ax3.legend(title='Source')
            ax3.tick_params(axis='x', rotation=45)
        
        # Pipe diameter effects
        diameters = []
        sound_speeds = []
        source_labels = []
        
        for i, dataset_key in enumerate(published_files):
            df = self.datasets[dataset_key]
            source = dataset_key.split('_')[-2]
            
            diameter_col = None
            if 'pipe_diameter_mm' in df.columns:
                diameter_col = 'pipe_diameter_mm'
            elif 'pipe_inner_diameter_mm' in df.columns:
                diameter_col = 'pipe_inner_diameter_mm'
            
            if diameter_col and 'sound_speed_ms' in df.columns:
                diameters.extend(df[diameter_col].tolist())
                sound_speeds.extend(df['sound_speed_ms'].tolist())
                source_labels.extend([source] * len(df))
        
        if diameters:
            for i, source in enumerate(set(source_labels)):
                source_mask = [s == source for s in source_labels]
                source_diameters = [d for d, m in zip(diameters, source_mask) if m]
                source_speeds = [s for s, m in zip(sound_speeds, source_mask) if m]
                ax4.scatter(source_diameters, source_speeds, 
                           c=colors[i], marker=markers[i], label=source, 
                           alpha=0.7, s=50)
        
        ax4.set_xlabel('Pipe Diameter (mm)')
        ax4.set_ylabel('Sound Speed (m/s)')
        ax4.set_title('Sound Speed vs Pipe Diameter')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_signal_processing_analysis(self):
        """Analyze and plot signal processing results."""
        signal_files = [k for k in self.datasets.keys() if 'signal_processing' in k]
        
        if len(signal_files) == 0:
            print("No signal processing datasets loaded")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Filter performance comparison
        if 'signal_processing_filtered_signals_fourier.csv' in self.datasets:
            filter_data = self.datasets['signal_processing_filtered_signals_fourier.csv']
            
            filter_types = filter_data['filter_type'].unique()
            snr_improvement = filter_data['snr_filtered_db'] - filter_data['snr_original_db']
            
            for ftype in filter_types:
                mask = filter_data['filter_type'] == ftype
                ax1.scatter(filter_data[mask]['snr_original_db'], 
                           snr_improvement[mask], 
                           label=ftype, alpha=0.7, s=50)
            
            ax1.set_xlabel('Original SNR (dB)')
            ax1.set_ylabel('SNR Improvement (dB)')
            ax1.set_title('Filter Performance: SNR Improvement')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Wavelet energy preservation
        if 'signal_processing_wavelet_filtered_signals.csv' in self.datasets:
            wavelet_data = self.datasets['signal_processing_wavelet_filtered_signals.csv']
            
            wavelet_types = wavelet_data['wavelet_type'].unique()
            for wtype in wavelet_types:
                mask = wavelet_data['wavelet_type'] == wtype
                ax2.scatter(wavelet_data[mask]['original_energy'], 
                           wavelet_data[mask]['energy_ratio'], 
                           label=wtype, alpha=0.7, s=50)
            
            ax2.set_xlabel('Original Energy (J)')
            ax2.set_ylabel('Energy Preservation Ratio')
            ax2.set_title('Wavelet Energy Preservation')
            ax2.set_xscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Cross-correlation analysis
        if 'signal_processing_cross_correlation_functions.csv' in self.datasets:
            cc_data = self.datasets['signal_processing_cross_correlation_functions.csv']
            
            sensor_pairs = cc_data['sensor_pair'].unique()
            for pair in sensor_pairs:
                mask = cc_data['sensor_pair'] == pair
                pair_data = cc_data[mask].sort_values('lag_time_s')
                ax3.plot(pair_data['lag_time_s'], pair_data['correlation_coefficient'], 
                        label=pair, marker='o', markersize=4)
            
            ax3.set_xlabel('Lag Time (s)')
            ax3.set_ylabel('Correlation Coefficient')
            ax3.set_title('Cross-Correlation Functions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Signal characteristics for leak detection
        if 'signal_processing_extracted_signal_characteristics.csv' in self.datasets:
            char_data = self.datasets['signal_processing_extracted_signal_characteristics.csv']
            
            # Create scatter plot colored by leak probability
            scatter = ax4.scatter(char_data['spectral_centroid_hz'], 
                                char_data['crest_factor'], 
                                c=char_data['leak_probability'], 
                                cmap='viridis', alpha=0.7, s=50)
            
            ax4.set_xlabel('Spectral Centroid (Hz)')
            ax4.set_ylabel('Crest Factor')
            ax4.set_title('Signal Characteristics for Leak Detection')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Leak Probability')
        
        plt.tight_layout()
        plt.show()
    
    def generate_correlation_matrix(self, dataset_key):
        """Generate correlation matrix for numerical columns in a dataset."""
        if dataset_key not in self.datasets:
            print(f"Dataset {dataset_key} not found")
            return
        
        df = self.datasets[dataset_key]
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found in dataset")
            return
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(f'Correlation Matrix: {dataset_key}')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def export_summary_report(self, filename="dataset_summary_report.txt"):
        """Export a comprehensive summary report."""
        with open(self.data_path / filename, 'w') as f:
            f.write("STRATIFIED FLOW ATTENUATION DATASETS - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Dataset Overview:\n")
            f.write("-" * 20 + "\n")
            summary = self.get_dataset_summary()
            f.write(summary.to_string())
            f.write("\n\n")
            
            f.write("Metadata Information:\n")
            f.write("-" * 20 + "\n")
            if self.metadata:
                f.write(json.dumps(self.metadata, indent=2))
            f.write("\n\n")
            
            f.write("Dataset Details:\n")
            f.write("-" * 20 + "\n")
            for name, df in self.datasets.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Shape: {df.shape}\n")
                f.write(f"  Columns: {list(df.columns)}\n")
                f.write(f"  Data types:\n")
                for col, dtype in df.dtypes.items():
                    f.write(f"    {col}: {dtype}\n")
        
        print(f"Summary report exported to {filename}")

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = StratifiedFlowAnalyzer()
    
    # Load all datasets
    print("Loading datasets...")
    analyzer.load_all_datasets()
    
    # Display summary
    print("\nDataset Summary:")
    print(analyzer.get_dataset_summary())
    
    # Generate visualizations
    print("\nGenerating baseline comparison plots...")
    analyzer.plot_baseline_comparison()
    
    print("\nGenerating published data comparison plots...")
    analyzer.plot_published_data_comparison()
    
    print("\nGenerating signal processing analysis plots...")
    analyzer.plot_signal_processing_analysis()
    
    # Export summary report
    analyzer.export_summary_report()
    
    print("\nAnalysis complete!")