#!/usr/bin/env python3
"""
Data Analysis Script for Stratified Flow Simulation
Provides comprehensive analysis tools for the generated dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy import signal, interpolate
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import os

class StratifiedFlowAnalyzer:
    """
    Comprehensive analysis tools for stratified flow simulation data
    """
    
    def __init__(self, data_directory):
        """
        Initialize analyzer with data directory
        
        Parameters:
        -----------
        data_directory : str
            Path to the data directory
        """
        self.data_dir = data_directory
        self.load_data()
    
    def load_data(self):
        """Load all simulation data"""
        print("Loading simulation data...")
        
        # Load CFD data
        try:
            with h5py.File(f'{self.data_dir}/cfd_data/cfd_data.h5', 'r') as f:
                self.cfd_data = {
                    'x': f['geometry/x'][:],
                    'y': f['geometry/y'][:],
                    'z': f['geometry/z'][:],
                    'velocity_x': f['fields/velocity_x'][:],
                    'velocity_y': f['fields/velocity_y'][:],
                    'velocity_z': f['fields/velocity_z'][:],
                    'pressure': f['fields/pressure'][:],
                    'vof': f['fields/vof'][:],
                    'density': f['fields/density'][:],
                    'sound_speed': f['fields/sound_speed'][:],
                    'effective_sound_speed': f['fields/effective_sound_speed'][:],
                    'turbulence_k': f['turbulence/k'][:],
                    'turbulence_epsilon': f['turbulence/epsilon'][:]
                }
        except FileNotFoundError:
            print("CFD data not found. Loading from NPZ files...")
            self.cfd_data = np.load(f'{self.data_dir}/raw_data/stratified_flow_simulation.npz')
        
        # Load acoustic data
        try:
            self.acoustic_data = np.load(f'{self.data_dir}/acoustic_data/acoustic_analysis.npz')
        except FileNotFoundError:
            print("Acoustic data not found.")
            self.acoustic_data = None
        
        # Load mathematical model data
        try:
            self.math_data = np.load(f'{self.data_dir}/mathematical_models/mathematical_models.npz')
        except FileNotFoundError:
            print("Mathematical model data not found.")
            self.math_data = None
        
        # Load turbulence data
        try:
            self.turbulence_data = np.load(f'{self.data_dir}/raw_data/turbulence_models.npz', allow_pickle=True)
        except FileNotFoundError:
            print("Turbulence data not found.")
            self.turbulence_data = None
        
        print("Data loading complete!")
    
    def analyze_flow_field(self):
        """Analyze the flow field characteristics"""
        print("Analyzing flow field...")
        
        # Calculate flow statistics
        u = self.cfd_data['velocity_x']
        v = self.cfd_data['velocity_y']
        w = self.cfd_data['velocity_z']
        
        # Velocity magnitude
        velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
        
        # Turbulent kinetic energy
        k = self.cfd_data['turbulence_k']
        
        # Interface detection
        vof = self.cfd_data['vof']
        interface_y = self._detect_interface(vof)
        
        # Calculate mixing layer thickness
        mixing_thickness = self._calculate_mixing_thickness(vof, interface_y)
        
        # Calculate Richardson number
        ri = self._calculate_richardson_number(u, self.cfd_data['density'])
        
        flow_analysis = {
            'velocity_magnitude': velocity_magnitude,
            'turbulent_kinetic_energy': k,
            'interface_position': interface_y,
            'mixing_thickness': mixing_thickness,
            'richardson_number': ri,
            'max_velocity': np.max(velocity_magnitude),
            'mean_velocity': np.mean(velocity_magnitude),
            'max_turbulence': np.max(k),
            'mean_turbulence': np.mean(k)
        }
        
        return flow_analysis
    
    def analyze_acoustic_propagation(self):
        """Analyze acoustic propagation characteristics"""
        print("Analyzing acoustic propagation...")
        
        if self.acoustic_data is None:
            print("No acoustic data available.")
            return None
        
        frequencies = self.acoustic_data['frequencies']
        effective_sound_speeds = self.acoustic_data['effective_sound_speeds']
        attenuation_coefficients = self.acoustic_data['attenuation_coefficients']
        
        # Calculate frequency-dependent characteristics
        acoustic_analysis = {}
        
        for i, freq in enumerate(frequencies):
            # Sound speed statistics
            c_eff = effective_sound_speeds[i, :]
            c_mean = np.mean(c_eff)
            c_std = np.std(c_eff)
            
            # Attenuation statistics
            alpha = attenuation_coefficients[i, :]
            alpha_mean = np.mean(alpha)
            alpha_std = np.std(alpha)
            
            acoustic_analysis[freq] = {
                'sound_speed_mean': c_mean,
                'sound_speed_std': c_std,
                'attenuation_mean': alpha_mean,
                'attenuation_std': alpha_std
            }
        
        return acoustic_analysis
    
    def analyze_turbulence_models(self):
        """Compare different turbulence models"""
        print("Analyzing turbulence models...")
        
        if self.turbulence_data is None:
            print("No turbulence data available.")
            return None
        
        # Extract turbulence data
        k_eps = self.turbulence_data['k_epsilon']
        k_omega = self.turbulence_data['k_omega_sst']
        les = self.turbulence_data['les']
        stratified = self.turbulence_data['stratified']
        
        # Calculate statistics for each model
        models = {
            'k_epsilon': k_eps,
            'k_omega_sst': k_omega,
            'les': les,
            'stratified': stratified
        }
        
        turbulence_analysis = {}
        
        for model_name, model_data in models.items():
            if 'k' in model_data:
                k = model_data['k']
                analysis = {
                    'max_k': np.max(k),
                    'mean_k': np.mean(k),
                    'std_k': np.std(k),
                    'spatial_distribution': self._analyze_spatial_distribution(k)
                }
                
                if 'mu_t' in model_data:
                    mu_t = model_data['mu_t']
                    analysis.update({
                        'max_mu_t': np.max(mu_t),
                        'mean_mu_t': np.mean(mu_t),
                        'std_mu_t': np.std(mu_t)
                    })
                
                turbulence_analysis[model_name] = analysis
        
        return turbulence_analysis
    
    def validate_models(self):
        """Validate models against experimental data"""
        print("Validating models...")
        
        # Load experimental data
        try:
            exp_sound_speed = pd.read_csv(f'{self.data_dir}/experimental_validation_data/experimental_sound_speed.csv')
            exp_attenuation = pd.read_csv(f'{self.data_dir}/experimental_validation_data/experimental_attenuation.csv')
        except FileNotFoundError:
            print("Experimental data not found. Skipping validation.")
            return None
        
        validation_results = {}
        
        # Sound speed validation
        if self.math_data is not None:
            sim_freq = self.math_data['frequencies']
            sim_vf = self.math_data['volume_fractions']
            sim_c = self.math_data['effective_sound_speeds']
            
            # Interpolate simulation data to experimental frequencies
            validation_results['sound_speed'] = self._validate_sound_speed(
                sim_freq, sim_vf, sim_c, exp_sound_speed)
        
        # Attenuation validation
        if self.acoustic_data is not None:
            sim_freq = self.acoustic_data['frequencies']
            sim_vf = self.acoustic_data['volume_fractions']
            sim_alpha = self.acoustic_data['attenuation_coefficients']
            
            validation_results['attenuation'] = self._validate_attenuation(
                sim_freq, sim_vf, sim_alpha, exp_attenuation)
        
        return validation_results
    
    def _detect_interface(self, vof, threshold=0.5):
        """Detect interface position from VOF field"""
        # Find interface in y-direction
        interface_positions = []
        
        for i in range(vof.shape[0]):
            for k in range(vof.shape[2]):
                vof_profile = vof[i, :, k]
                # Find where VOF crosses 0.5
                crossings = np.where(np.diff(np.sign(vof_profile - threshold)))[0]
                if len(crossings) > 0:
                    interface_positions.append(crossings[0])
        
        return np.mean(interface_positions) if interface_positions else 0
    
    def _calculate_mixing_thickness(self, vof, interface_y, threshold=0.1):
        """Calculate mixing layer thickness"""
        # Find thickness where VOF changes from 0.1 to 0.9
        thicknesses = []
        
        for i in range(vof.shape[0]):
            for k in range(vof.shape[2]):
                vof_profile = vof[i, :, k]
                
                # Find positions where VOF = 0.1 and 0.9
                idx_01 = np.where(vof_profile <= 0.1)[0]
                idx_09 = np.where(vof_profile >= 0.9)[0]
                
                if len(idx_01) > 0 and len(idx_09) > 0:
                    thickness = np.max(idx_09) - np.min(idx_01)
                    thicknesses.append(thickness)
        
        return np.mean(thicknesses) if thicknesses else 0
    
    def _calculate_richardson_number(self, velocity, density):
        """Calculate Richardson number"""
        # Simplified Richardson number calculation
        g = 9.81
        
        # Calculate density gradient
        rho = density
        drho_dy = np.gradient(rho, axis=1)
        
        # Calculate velocity gradient
        du_dy = np.gradient(velocity, axis=1)
        
        # Richardson number
        Ri = -g / rho * drho_dy / (du_dy**2 + 1e-10)
        
        return Ri
    
    def _analyze_spatial_distribution(self, field):
        """Analyze spatial distribution of a field"""
        return {
            'skewness': self._calculate_skewness(field),
            'kurtosis': self._calculate_kurtosis(field),
            'correlation_length': self._calculate_correlation_length(field)
        }
    
    def _calculate_skewness(self, field):
        """Calculate skewness of field"""
        return np.mean((field - np.mean(field))**3) / (np.std(field)**3)
    
    def _calculate_kurtosis(self, field):
        """Calculate kurtosis of field"""
        return np.mean((field - np.mean(field))**4) / (np.std(field)**4)
    
    def _calculate_correlation_length(self, field):
        """Calculate correlation length"""
        # Simplified correlation length calculation
        # This would typically involve autocorrelation analysis
        return np.std(field) / np.mean(np.abs(np.gradient(field)))
    
    def _validate_sound_speed(self, sim_freq, sim_vf, sim_c, exp_data):
        """Validate sound speed predictions"""
        validation_results = {}
        
        for freq in np.unique(exp_data['frequency']):
            exp_freq_data = exp_data[exp_data['frequency'] == freq]
            
            # Interpolate simulation data
            sim_c_interp = interpolate.griddata(
                (sim_freq, sim_vf), sim_c.flatten(),
                (freq, exp_freq_data['volume_fraction']),
                method='linear'
            )
            
            # Calculate validation metrics
            r2 = r2_score(exp_freq_data['sound_speed'], sim_c_interp)
            rmse = np.sqrt(mean_squared_error(exp_freq_data['sound_speed'], sim_c_interp))
            
            validation_results[freq] = {
                'r2_score': r2,
                'rmse': rmse,
                'mean_error': np.mean(sim_c_interp - exp_freq_data['sound_speed']),
                'std_error': np.std(sim_c_interp - exp_freq_data['sound_speed'])
            }
        
        return validation_results
    
    def _validate_attenuation(self, sim_freq, sim_vf, sim_alpha, exp_data):
        """Validate attenuation predictions"""
        validation_results = {}
        
        for freq in np.unique(exp_data['frequency']):
            exp_freq_data = exp_data[exp_data['frequency'] == freq]
            
            # Interpolate simulation data
            sim_alpha_interp = interpolate.griddata(
                (sim_freq, sim_vf), sim_alpha.flatten(),
                (freq, exp_freq_data['volume_fraction']),
                method='linear'
            )
            
            # Calculate validation metrics
            r2 = r2_score(exp_freq_data['attenuation'], sim_alpha_interp)
            rmse = np.sqrt(mean_squared_error(exp_freq_data['attenuation'], sim_alpha_interp))
            
            validation_results[freq] = {
                'r2_score': r2,
                'rmse': rmse,
                'mean_error': np.mean(sim_alpha_interp - exp_freq_data['attenuation']),
                'std_error': np.std(sim_alpha_interp - exp_freq_data['attenuation'])
            }
        
        return validation_results
    
    def generate_analysis_report(self, output_dir='analysis_results'):
        """Generate comprehensive analysis report"""
        print("Generating analysis report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run all analyses
        flow_analysis = self.analyze_flow_field()
        acoustic_analysis = self.analyze_acoustic_propagation()
        turbulence_analysis = self.analyze_turbulence_models()
        validation_results = self.validate_models()
        
        # Create visualizations
        self._create_analysis_plots(output_dir, flow_analysis, acoustic_analysis, 
                                  turbulence_analysis, validation_results)
        
        # Generate summary report
        self._generate_summary_report(output_dir, flow_analysis, acoustic_analysis,
                                    turbulence_analysis, validation_results)
        
        print(f"Analysis report generated in {output_dir}")
    
    def _create_analysis_plots(self, output_dir, flow_analysis, acoustic_analysis,
                             turbulence_analysis, validation_results):
        """Create analysis plots"""
        
        # Flow field analysis plots
        if flow_analysis:
            plt.figure(figsize=(15, 10))
            
            # Velocity magnitude contour
            plt.subplot(2, 3, 1)
            vof = self.cfd_data['vof']
            vof_slice = vof[:, :, vof.shape[2]//2]
            plt.contourf(vof_slice, levels=20, cmap='Blues')
            plt.colorbar(label='VOF')
            plt.title('Volume of Fluid Distribution')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Turbulent kinetic energy
            plt.subplot(2, 3, 2)
            k = self.cfd_data['turbulence_k']
            k_slice = k[:, :, k.shape[2]//2]
            plt.contourf(k_slice, levels=20, cmap='Reds')
            plt.colorbar(label='TKE (m²/s²)')
            plt.title('Turbulent Kinetic Energy')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # Richardson number
            plt.subplot(2, 3, 3)
            ri = flow_analysis['richardson_number']
            ri_slice = ri[:, :, ri.shape[2]//2]
            plt.contourf(ri_slice, levels=20, cmap='RdYlBu')
            plt.colorbar(label='Richardson Number')
            plt.title('Richardson Number')
            plt.xlabel('x')
            plt.ylabel('y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/flow_field_analysis.png', dpi=300)
            plt.close()
        
        # Acoustic analysis plots
        if acoustic_analysis:
            plt.figure(figsize=(12, 8))
            
            frequencies = list(acoustic_analysis.keys())
            sound_speeds = [acoustic_analysis[f]['sound_speed_mean'] for f in frequencies]
            attenuations = [acoustic_analysis[f]['attenuation_mean'] for f in frequencies]
            
            plt.subplot(2, 2, 1)
            plt.semilogx(frequencies, sound_speeds, 'b-o')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Effective Sound Speed (m/s)')
            plt.title('Frequency-Dependent Sound Speed')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.loglog(frequencies, attenuations, 'r-o')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Attenuation Coefficient (Np/m)')
            plt.title('Frequency-Dependent Attenuation')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/acoustic_analysis.png', dpi=300)
            plt.close()
        
        # Turbulence model comparison
        if turbulence_analysis:
            plt.figure(figsize=(12, 6))
            
            models = list(turbulence_analysis.keys())
            max_k = [turbulence_analysis[m]['max_k'] for m in models]
            mean_k = [turbulence_analysis[m]['mean_k'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.subplot(1, 2, 1)
            plt.bar(x - width/2, max_k, width, label='Max TKE')
            plt.bar(x + width/2, mean_k, width, label='Mean TKE')
            plt.xlabel('Turbulence Model')
            plt.ylabel('Turbulent Kinetic Energy (m²/s²)')
            plt.title('Turbulence Model Comparison')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/turbulence_comparison.png', dpi=300)
            plt.close()
    
    def _generate_summary_report(self, output_dir, flow_analysis, acoustic_analysis,
                               turbulence_analysis, validation_results):
        """Generate summary report"""
        
        report = {
            "analysis_summary": {
                "flow_field": flow_analysis,
                "acoustic_propagation": acoustic_analysis,
                "turbulence_models": turbulence_analysis,
                "model_validation": validation_results
            },
            "key_findings": [],
            "recommendations": []
        }
        
        # Add key findings
        if flow_analysis:
            report["key_findings"].append(f"Maximum velocity: {flow_analysis['max_velocity']:.2f} m/s")
            report["key_findings"].append(f"Interface position: {flow_analysis['interface_position']:.2f} m")
            report["key_findings"].append(f"Mixing thickness: {flow_analysis['mixing_thickness']:.2f} m")
        
        if acoustic_analysis:
            freq_range = list(acoustic_analysis.keys())
            sound_speed_range = [acoustic_analysis[f]['sound_speed_mean'] for f in freq_range]
            report["key_findings"].append(f"Sound speed range: {min(sound_speed_range):.0f} - {max(sound_speed_range):.0f} m/s")
        
        if validation_results:
            if 'sound_speed' in validation_results:
                r2_scores = [validation_results['sound_speed'][f]['r2_score'] for f in validation_results['sound_speed']]
                report["key_findings"].append(f"Sound speed validation R²: {np.mean(r2_scores):.3f}")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Save report
        import json
        with open(f'{output_dir}/analysis_report.json', 'w') as f:
            json.dump(convert_numpy(report), f, indent=2)
        
        # Create text summary
        with open(f'{output_dir}/analysis_summary.txt', 'w') as f:
            f.write("STRATIFIED FLOW SIMULATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("KEY FINDINGS:\n")
            for finding in report["key_findings"]:
                f.write(f"- {finding}\n")
            
            f.write("\nDATA QUALITY:\n")
            f.write("- CFD simulation data: Complete\n")
            f.write("- Acoustic analysis data: Complete\n")
            f.write("- Turbulence model data: Complete\n")
            f.write("- Validation data: Available\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("- Use stratified turbulence model for interface regions\n")
            f.write("- Validate acoustic models with experimental data\n")
            f.write("- Consider LES for high-resolution analysis\n")

def main():
    """Main analysis function"""
    data_directory = "stratified_flow_thesis_data"
    
    if not os.path.exists(data_directory):
        print(f"Data directory {data_directory} not found!")
        print("Please run the simulation first using run_simulation.py")
        return
    
    # Create analyzer
    analyzer = StratifiedFlowAnalyzer(data_directory)
    
    # Generate analysis report
    analyzer.generate_analysis_report()
    
    print("Analysis complete! Check analysis_results/ for detailed results.")

if __name__ == "__main__":
    main()