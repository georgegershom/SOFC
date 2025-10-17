#!/usr/bin/env python3
"""
Data Analysis Script for High-Temperature Experimental Dataset
Provides comprehensive analysis and visualization capabilities
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import ast

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.analysis_results = {}
        
    def load_thermal_data(self):
        """Load all thermal properties data"""
        thermal_data = {}
        
        # Load TGA/DSC data
        tga_path = self.dataset_path / 'thermal_properties' / 'tga_dsc'
        if tga_path.exists():
            json_file = tga_path / 'tga_dsc_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    thermal_data['tga_dsc'] = json.load(f)
        
        # Load thermal conductivity data
        tc_path = self.dataset_path / 'thermal_properties' / 'thermal_conductivity'
        if tc_path.exists():
            json_file = tc_path / 'thermal_conductivity_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    thermal_data['thermal_conductivity'] = json.load(f)
        
        return thermal_data
    
    def load_mechanical_data(self):
        """Load all mechanical testing data"""
        mechanical_data = {}
        
        # Load TTS data
        tts_path = self.dataset_path / 'mechanical_testing' / 'tts_curves'
        if tts_path.exists():
            json_file = tts_path / 'tts_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    mechanical_data['tts'] = json.load(f)
        
        # Load residual properties data
        res_path = self.dataset_path / 'mechanical_testing' / 'residual_properties'
        if res_path.exists():
            json_file = res_path / 'residual_properties_data.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    mechanical_data['residual'] = json.load(f)
        
        return mechanical_data
    
    def analyze_thermal_decomposition(self):
        """Analyze thermal decomposition patterns"""
        print("Analyzing thermal decomposition patterns...")
        
        thermal_data = self.load_thermal_data()
        if 'tga_dsc' not in thermal_data:
            print("TGA/DSC data not found")
            return None
        
        decomposition_analysis = {}
        
        for mix, mix_data in thermal_data['tga_dsc'].items():
            if mix == 'metadata':
                continue
            
            mix_analysis = {
                'decomposition_stages': [],
                'peak_temperatures': [],
                'mass_loss_at_stages': []
            }
            
            # Analyze first replicate
            tga_data = mix_data['tga'][0]['data']
            
            # Handle string representations of numpy arrays
            if isinstance(tga_data['temperature'], str):
                # Convert string representation back to list
                temp_str = tga_data['temperature'].replace('[', '').replace(']', '').replace('\n', ' ')
                temp_list = [float(x.strip()) for x in temp_str.split() if x.strip()]
                temps = np.array(temp_list, dtype=float)
            else:
                temps = np.array(tga_data['temperature'], dtype=float)
                
            if isinstance(tga_data['mass_loss_percent'], str):
                mass_str = tga_data['mass_loss_percent'].replace('[', '').replace(']', '').replace('\n', ' ')
                mass_list = [float(x.strip()) for x in mass_str.split() if x.strip()]
                mass_loss = np.array(mass_list, dtype=float)
            else:
                mass_loss = np.array(tga_data['mass_loss_percent'], dtype=float)
                
            if isinstance(tga_data['mass_loss_rate'], str):
                rate_str = tga_data['mass_loss_rate'].replace('[', '').replace(']', '').replace('\n', ' ')
                rate_list = [float(x.strip()) for x in rate_str.split() if x.strip()]
                mass_loss_rate = np.array(rate_list, dtype=float)
            else:
                mass_loss_rate = np.array(tga_data['mass_loss_rate'], dtype=float)
            
            # Find decomposition stages (peaks in mass loss rate)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(mass_loss_rate, height=0.1, distance=50)
            
            for peak_idx in peaks:
                stage = {
                    'temperature': temps[peak_idx],
                    'mass_loss': mass_loss[peak_idx],
                    'mass_loss_rate': mass_loss_rate[peak_idx]
                }
                mix_analysis['decomposition_stages'].append(stage)
                mix_analysis['peak_temperatures'].append(temps[peak_idx])
                mix_analysis['mass_loss_at_stages'].append(mass_loss[peak_idx])
            
            decomposition_analysis[mix] = mix_analysis
        
        self.analysis_results['thermal_decomposition'] = decomposition_analysis
        return decomposition_analysis
    
    def analyze_mechanical_degradation(self):
        """Analyze mechanical property degradation with temperature"""
        print("Analyzing mechanical property degradation...")
        
        mechanical_data = self.load_mechanical_data()
        if 'tts' not in mechanical_data:
            print("TTS data not found")
            return None
        
        degradation_analysis = {}
        
        for mix, mix_data in mechanical_data['tts'].items():
            if mix == 'metadata':
                continue
            
            mix_analysis = {
                'strength_degradation': [],
                'modulus_degradation': [],
                'ductility_changes': []
            }
            
            temperatures = []
            peak_strengths = []
            elastic_moduli = []
            peak_strains = []
            
            if 'temperatures' in mix_data:
                for temp, temp_data in mix_data['temperatures'].items():
                    # Average across replicates
                    strengths = [r['peak_strength'] for r in temp_data['replicates']]
                    moduli = [r['elastic_modulus'] for r in temp_data['replicates']]
                    strains = [r['peak_strain'] for r in temp_data['replicates']]
                    
                    temperatures.append(int(temp))
                    peak_strengths.append(np.mean(strengths))
                    elastic_moduli.append(np.mean(moduli))
                    peak_strains.append(np.mean(strains))
            
            # Calculate degradation factors
            ambient_strength = peak_strengths[0] if peak_strengths else 1
            ambient_modulus = elastic_moduli[0] if elastic_moduli else 1
            
            for i, temp in enumerate(temperatures):
                degradation = {
                    'temperature': temp,
                    'strength_factor': peak_strengths[i] / ambient_strength,
                    'modulus_factor': elastic_moduli[i] / ambient_modulus,
                    'peak_strength': peak_strengths[i],
                    'elastic_modulus': elastic_moduli[i],
                    'peak_strain': peak_strains[i]
                }
                mix_analysis['strength_degradation'].append(degradation)
            
            degradation_analysis[mix] = mix_analysis
        
        self.analysis_results['mechanical_degradation'] = degradation_analysis
        return degradation_analysis
    
    def analyze_rubber_effects(self):
        """Analyze effects of rubber content on properties"""
        print("Analyzing rubber content effects...")
        
        thermal_data = self.load_thermal_data()
        mechanical_data = self.load_mechanical_data()
        
        rubber_analysis = {
            'thermal_conductivity': {},
            'strength_degradation': {},
            'residual_properties': {}
        }
        
        # Analyze thermal conductivity vs rubber content
        if 'thermal_conductivity' in thermal_data:
            tc_data = thermal_data['thermal_conductivity']
            for temp in [25, 200, 400, 600]:
                if str(temp) in tc_data['mixes']['control']['temperatures']:
                    rubber_contents = []
                    conductivities = []
                    
                    for mix in ['control', 'rubber_10', 'rubber_20', 'rubber_30']:
                        if mix in tc_data['mixes']:
                            rubber_content = 0 if mix == 'control' else int(mix.split('_')[1])
                            temp_data = tc_data['mixes'][mix]['temperatures'][str(temp)]
                            avg_conductivity = np.mean([r['thermal_conductivity'] for r in temp_data['replicates']])
                            
                            rubber_contents.append(rubber_content)
                            conductivities.append(avg_conductivity)
                    
                    rubber_analysis['thermal_conductivity'][temp] = {
                        'rubber_contents': rubber_contents,
                        'conductivities': conductivities
                    }
        
        # Analyze strength degradation vs rubber content
        if 'tts' in mechanical_data:
            tts_data = mechanical_data['tts']
            for temp in [25, 200, 400, 600, 800]:
                if str(temp) in tts_data['mixes']['control']['temperatures']:
                    rubber_contents = []
                    strengths = []
                    
                    for mix in ['control', 'rubber_10', 'rubber_20', 'rubber_30']:
                        if mix in tts_data['mixes']:
                            rubber_content = 0 if mix == 'control' else int(mix.split('_')[1])
                            temp_data = tts_data['mixes'][mix]['temperatures'][str(temp)]
                            avg_strength = np.mean([r['peak_strength'] for r in temp_data['replicates']])
                            
                            rubber_contents.append(rubber_content)
                            strengths.append(avg_strength)
                    
                    rubber_analysis['strength_degradation'][temp] = {
                        'rubber_contents': rubber_contents,
                        'strengths': strengths
                    }
        
        self.analysis_results['rubber_effects'] = rubber_analysis
        return rubber_analysis
    
    def fit_degradation_models(self):
        """Fit mathematical models to degradation data"""
        print("Fitting degradation models...")
        
        mechanical_data = self.load_mechanical_data()
        if 'tts' not in mechanical_data:
            print("TTS data not found")
            return None
        
        models = {}
        
        # Exponential decay model: f(T) = f0 * exp(-α * T)
        def exponential_decay(T, f0, alpha):
            return f0 * np.exp(-alpha * T)
        
        for mix, mix_data in mechanical_data['tts'].items():
            if mix == 'metadata':
                continue
            
            mix_models = {}
            
            # Extract data
            temperatures = []
            strengths = []
            moduli = []
            
            if 'temperatures' in mix_data:
                for temp, temp_data in mix_data['temperatures'].items():
                    temp_val = int(temp)
                    if temp_val > 25:  # Skip ambient temperature
                        avg_strength = np.mean([r['peak_strength'] for r in temp_data['replicates']])
                        avg_modulus = np.mean([r['elastic_modulus'] for r in temp_data['replicates']])
                        
                        temperatures.append(temp_val)
                        strengths.append(avg_strength)
                        moduli.append(avg_modulus)
            
            if len(temperatures) > 2:
                # Fit strength degradation model
                try:
                    popt_strength, _ = curve_fit(exponential_decay, temperatures, strengths, 
                                               p0=[strengths[0], 0.001])
                    mix_models['strength'] = {
                        'model': 'exponential_decay',
                        'parameters': {'f0': popt_strength[0], 'alpha': popt_strength[1]},
                        'r_squared': self._calculate_r_squared(temperatures, strengths, 
                                                             lambda T: exponential_decay(T, *popt_strength))
                    }
                except:
                    mix_models['strength'] = {'model': 'fit_failed', 'parameters': {}, 'r_squared': 0}
                
                # Fit modulus degradation model
                try:
                    popt_modulus, _ = curve_fit(exponential_decay, temperatures, moduli, 
                                              p0=[moduli[0], 0.001])
                    mix_models['modulus'] = {
                        'model': 'exponential_decay',
                        'parameters': {'f0': popt_modulus[0], 'alpha': popt_modulus[1]},
                        'r_squared': self._calculate_r_squared(temperatures, moduli, 
                                                             lambda T: exponential_decay(T, *popt_modulus))
                    }
                except:
                    mix_models['modulus'] = {'model': 'fit_failed', 'parameters': {}, 'r_squared': 0}
            
            models[mix] = mix_models
        
        self.analysis_results['degradation_models'] = models
        return models
    
    def _calculate_r_squared(self, x, y, model_func):
        """Calculate R-squared for model fit"""
        y_pred = [model_func(xi) for xi in x]
        ss_res = sum((y[i] - y_pred[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i] - np.mean(y))**2 for i in range(len(y)))
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def generate_comprehensive_plots(self, output_dir):
        """Generate comprehensive analysis plots"""
        print("Generating comprehensive analysis plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot 1: Thermal decomposition comparison
        self._plot_thermal_decomposition(output_path)
        
        # Plot 2: Mechanical degradation comparison
        self._plot_mechanical_degradation(output_path)
        
        # Plot 3: Rubber content effects
        self._plot_rubber_effects(output_path)
        
        # Plot 4: Model fits
        self._plot_model_fits(output_path)
        
        print(f"Plots saved to: {output_path}")
    
    def _plot_thermal_decomposition(self, output_path):
        """Plot thermal decomposition analysis"""
        if 'thermal_decomposition' not in self.analysis_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Load TGA data for plotting
        thermal_data = self.load_thermal_data()
        if 'tga_dsc' not in thermal_data:
            return
        
        # TGA curves
        ax1 = axes[0, 0]
        for mix in ['control', 'rubber_10', 'rubber_30']:
            if mix in thermal_data['tga_dsc']:
                tga_data = thermal_data['tga_dsc'][mix]['tga'][0]['data']
                
                # Handle string representations
                if isinstance(tga_data['temperature'], str):
                    temp_str = tga_data['temperature'].replace('[', '').replace(']', '').replace('\n', ' ')
                    temp_list = [float(x.strip()) for x in temp_str.split() if x.strip()]
                    temps = np.array(temp_list)
                else:
                    temps = np.array(tga_data['temperature'])
                
                if isinstance(tga_data['remaining_mass_percent'], str):
                    mass_str = tga_data['remaining_mass_percent'].replace('[', '').replace(']', '').replace('\n', ' ')
                    mass_list = [float(x.strip()) for x in mass_str.split() if x.strip()]
                    masses = np.array(mass_list)
                else:
                    masses = np.array(tga_data['remaining_mass_percent'])
                
                ax1.plot(temps, masses, label=mix, linewidth=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Remaining Mass (%)')
        ax1.set_title('TGA Curves Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # DSC curves
        ax2 = axes[0, 1]
        for mix in ['control', 'rubber_10', 'rubber_30']:
            if mix in thermal_data['tga_dsc']:
                dsc_data = thermal_data['tga_dsc'][mix]['dsc'][0]['data']
                
                # Handle string representations
                if isinstance(dsc_data['temperature'], str):
                    temp_str = dsc_data['temperature'].replace('[', '').replace(']', '').replace('\n', ' ')
                    temp_list = [float(x.strip()) for x in temp_str.split() if x.strip()]
                    temps = np.array(temp_list)
                else:
                    temps = np.array(dsc_data['temperature'])
                
                if isinstance(dsc_data['heat_flow_mW_mg'], str):
                    heat_str = dsc_data['heat_flow_mW_mg'].replace('[', '').replace(']', '').replace('\n', ' ')
                    heat_list = [float(x.strip()) for x in heat_str.split() if x.strip()]
                    heats = np.array(heat_list)
                else:
                    heats = np.array(dsc_data['heat_flow_mW_mg'])
                
                ax2.plot(temps, heats, label=mix, linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Heat Flow (mW/mg)')
        ax2.set_title('DSC Curves Comparison')
        ax2.legend()
        ax2.grid(True)
        
        # Decomposition stages
        ax3 = axes[1, 0]
        decomposition_data = self.analysis_results['thermal_decomposition']
        for mix, data in decomposition_data.items():
            if data['peak_temperatures']:
                ax3.scatter([0] * len(data['peak_temperatures']), data['peak_temperatures'], 
                           label=mix, s=100, alpha=0.7)
        ax3.set_xlabel('Mix Type')
        ax3.set_ylabel('Peak Temperature (°C)')
        ax3.set_title('Decomposition Peak Temperatures')
        ax3.legend()
        ax3.grid(True)
        
        # Mass loss at stages
        ax4 = axes[1, 1]
        for mix, data in decomposition_data.items():
            if data['mass_loss_at_stages']:
                ax4.scatter([0] * len(data['mass_loss_at_stages']), data['mass_loss_at_stages'], 
                           label=mix, s=100, alpha=0.7)
        ax4.set_xlabel('Mix Type')
        ax4.set_ylabel('Mass Loss at Peak (%)')
        ax4.set_title('Mass Loss at Decomposition Peaks')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'thermal_decomposition_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mechanical_degradation(self, output_path):
        """Plot mechanical degradation analysis"""
        if 'mechanical_degradation' not in self.analysis_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        degradation_data = self.analysis_results['mechanical_degradation']
        
        # Strength degradation
        ax1 = axes[0, 0]
        for mix, data in degradation_data.items():
            temps = [d['temperature'] for d in data['strength_degradation']]
            factors = [d['strength_factor'] for d in data['strength_degradation']]
            ax1.plot(temps, factors, 'o-', label=mix, linewidth=2, markersize=6)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Strength Factor')
        ax1.set_title('Compressive Strength Degradation')
        ax1.legend()
        ax1.grid(True)
        
        # Modulus degradation
        ax2 = axes[0, 1]
        for mix, data in degradation_data.items():
            temps = [d['temperature'] for d in data['strength_degradation']]
            moduli = [d['elastic_modulus'] for d in data['strength_degradation']]
            ax2.plot(temps, moduli, 'o-', label=mix, linewidth=2, markersize=6)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Elastic Modulus (MPa)')
        ax2.set_title('Elastic Modulus vs Temperature')
        ax2.legend()
        ax2.grid(True)
        
        # Peak strain (ductility)
        ax3 = axes[1, 0]
        for mix, data in degradation_data.items():
            temps = [d['temperature'] for d in data['strength_degradation']]
            strains = [d['peak_strain'] for d in data['strength_degradation']]
            ax3.plot(temps, strains, 'o-', label=mix, linewidth=2, markersize=6)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Peak Strain')
        ax3.set_title('Ductility vs Temperature')
        ax3.legend()
        ax3.grid(True)
        
        # Strength comparison at specific temperatures
        ax4 = axes[1, 1]
        specific_temps = [200, 400, 600, 800]
        for mix, data in degradation_data.items():
            strengths = []
            for temp in specific_temps:
                strength_data = next((d for d in data['strength_degradation'] if d['temperature'] == temp), None)
                if strength_data:
                    strengths.append(strength_data['peak_strength'])
                else:
                    strengths.append(0)
            ax4.plot(specific_temps, strengths, 'o-', label=mix, linewidth=2, markersize=6)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Peak Strength (MPa)')
        ax4.set_title('Strength at Specific Temperatures')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'mechanical_degradation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rubber_effects(self, output_path):
        """Plot rubber content effects"""
        if 'rubber_effects' not in self.analysis_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rubber_data = self.analysis_results['rubber_effects']
        
        # Thermal conductivity vs rubber content
        ax1 = axes[0, 0]
        for temp, data in rubber_data.get('thermal_conductivity', {}).items():
            ax1.plot(data['rubber_contents'], data['conductivities'], 'o-', 
                    label=f'{temp}°C', linewidth=2, markersize=6)
        ax1.set_xlabel('Rubber Content (%)')
        ax1.set_ylabel('Thermal Conductivity (W/m·K)')
        ax1.set_title('Thermal Conductivity vs Rubber Content')
        ax1.legend()
        ax1.grid(True)
        
        # Strength vs rubber content at different temperatures
        ax2 = axes[0, 1]
        for temp, data in rubber_data.get('strength_degradation', {}).items():
            ax2.plot(data['rubber_contents'], data['strengths'], 'o-', 
                    label=f'{temp}°C', linewidth=2, markersize=6)
        ax2.set_xlabel('Rubber Content (%)')
        ax2.set_ylabel('Peak Strength (MPa)')
        ax2.set_title('Strength vs Rubber Content')
        ax2.legend()
        ax2.grid(True)
        
        # Rubber content effect on degradation rate
        ax3 = axes[1, 0]
        if 'degradation_models' in self.analysis_results:
            models = self.analysis_results['degradation_models']
            rubber_contents = []
            degradation_rates = []
            
            for mix, mix_models in models.items():
                if mix != 'metadata' and 'strength' in mix_models:
                    if mix_models['strength']['model'] == 'exponential_decay':
                        rubber_content = 0 if mix == 'control' else int(mix.split('_')[1])
                        alpha = mix_models['strength']['parameters']['alpha']
                        rubber_contents.append(rubber_content)
                        degradation_rates.append(alpha)
            
            if rubber_contents:
                ax3.plot(rubber_contents, degradation_rates, 'o-', linewidth=2, markersize=6)
                ax3.set_xlabel('Rubber Content (%)')
                ax3.set_ylabel('Degradation Rate (α)')
                ax3.set_title('Degradation Rate vs Rubber Content')
                ax3.grid(True)
        
        # Model quality comparison
        ax4 = axes[1, 1]
        if 'degradation_models' in self.analysis_results:
            models = self.analysis_results['degradation_models']
            mixes = []
            r_squared_values = []
            
            for mix, mix_models in models.items():
                if mix != 'metadata' and 'strength' in mix_models:
                    mixes.append(mix)
                    r_squared_values.append(mix_models['strength']['r_squared'])
            
            if mixes:
                ax4.bar(mixes, r_squared_values, alpha=0.7)
                ax4.set_xlabel('Mix Type')
                ax4.set_ylabel('R² Value')
                ax4.set_title('Model Fit Quality (R²)')
                ax4.set_ylim(0, 1)
                ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'rubber_effects_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_fits(self, output_path):
        """Plot model fits for degradation data"""
        if 'degradation_models' not in self.analysis_results:
            return
        
        models = self.analysis_results['degradation_models']
        mechanical_data = self.load_mechanical_data()
        
        if 'tts' not in mechanical_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, mix in enumerate(['control', 'rubber_10', 'rubber_20', 'rubber_30']):
            if mix not in models or mix not in mechanical_data['tts']:
                continue
            
            ax = axes[i//2, i%2]
            
            # Get experimental data
            mix_data = mechanical_data['tts'][mix]
            temperatures = []
            strengths = []
            
            for temp, temp_data in mix_data['temperatures'].items():
                temp_val = int(temp)
                if temp_val > 25:
                    avg_strength = np.mean([r['peak_strength'] for r in temp_data['replicates']])
                    temperatures.append(temp_val)
                    strengths.append(avg_strength)
            
            # Plot experimental data
            ax.scatter(temperatures, strengths, color='blue', s=50, alpha=0.7, label='Experimental')
            
            # Plot model fit
            if mix in models and 'strength' in models[mix]:
                model = models[mix]['strength']
                if model['model'] == 'exponential_decay':
                    f0 = model['parameters']['f0']
                    alpha = model['parameters']['alpha']
                    
                    T_fit = np.linspace(min(temperatures), max(temperatures), 100)
                    S_fit = f0 * np.exp(-alpha * T_fit)
                    
                    ax.plot(T_fit, S_fit, 'r-', linewidth=2, 
                           label=f"Model (R² = {model['r_squared']:.3f})")
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Peak Strength (MPa)')
            ax.set_title(f'{mix.replace("_", " ").title()} - Model Fit')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_fits_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self, output_path):
        """Generate comprehensive analysis report"""
        print("Generating analysis report...")
        
        # Run all analyses
        self.analyze_thermal_decomposition()
        self.analyze_mechanical_degradation()
        self.analyze_rubber_effects()
        self.fit_degradation_models()
        
        # Generate plots
        self.generate_comprehensive_plots(output_path)
        
        # Save analysis results
        report_path = Path(output_path) / 'analysis_results.json'
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"Analysis report saved to: {report_path}")
        return self.analysis_results

if __name__ == "__main__":
    analyzer = DatasetAnalyzer("/workspace/experimental_dataset")
    results = analyzer.generate_analysis_report("/workspace/experimental_dataset/analysis")
    print("Analysis completed!")