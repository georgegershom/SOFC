#!/usr/bin/env python3
"""
Synthetic Acoustic Pressure Dataset Generator
Study: Attenuation Mechanisms in Stratified Flows - Beyond Single Phase Leakage Acoustics

This script generates high-frequency acoustic pressure data simulating leak detection
in stratified flow conditions with multiple sensors and test configurations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import zipfile
from datetime import datetime
import json

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================================

# Sensor Configuration
SENSORS = [f'PG{i:02d}' for i in range(1, 15)]  # PG01 to PG14
NUM_SENSORS = 14

# Test Groups Configuration
TEST_GROUPS = {
    'Group_01': {
        'leak_location': 'A',
        'sensor_config': 'A,B',
        'sampling_rate': 17060,  # Hz - Higher rate for Group 01
        'active_sensors': 10,
        'description': 'Leak at position A, sensors A-B active'
    },
    'Group_02': {
        'leak_location': 'D',
        'sensor_config': 'A,B,C,D',
        'sampling_rate': 10000,  # Hz - Standard rate
        'active_sensors': 14,
        'description': 'Leak at position D, all sensors active'
    },
    'Group_03': {
        'leak_location': 'E',
        'sensor_config': 'A,B',
        'sampling_rate': 10000,  # Hz
        'active_sensors': 10,
        'description': 'Leak at position E, sensors A-B active'
    },
    'Group_04': {
        'leak_location': 'A',
        'sensor_config': 'A,B,C,D',
        'sampling_rate': 10000,  # Hz
        'active_sensors': 14,
        'description': 'Leak at position A, all sensors active'
    }
}

# Time Configuration
DURATION = 30.0  # seconds
VALVE_OPEN_TIME = 5.0  # seconds
VALVE_CLOSE_TIME = 20.0  # seconds

# Physical Parameters for Acoustic Wave Propagation
SPEED_OF_SOUND = 1500.0  # m/s in water/stratified flow
ATTENUATION_COEFF = 0.05  # dB/m - frequency dependent
BASE_PRESSURE = 101325.0  # Pa (atmospheric)
LEAK_AMPLITUDE = 5000.0  # Pa - peak pressure change due to leak

# Sensor Positions (meters from source)
SENSOR_POSITIONS = {
    'PG01': 0.5, 'PG02': 1.2, 'PG03': 2.0, 'PG04': 3.5,
    'PG05': 5.0, 'PG06': 6.8, 'PG07': 8.5, 'PG08': 10.2,
    'PG09': 12.0, 'PG10': 14.5, 'PG11': 17.0, 'PG12': 20.0,
    'PG13': 23.5, 'PG14': 27.0
}

# ============================================================================
# ACOUSTIC SIGNAL GENERATION FUNCTIONS
# ============================================================================

def generate_baseline_noise(time_array, sampling_rate):
    """Generate realistic baseline noise (no leak condition)"""
    # White noise
    white_noise = np.random.normal(0, 50, len(time_array))
    
    # Low-frequency drift (flow fluctuations)
    drift_freq = 0.5  # Hz
    drift = 30 * np.sin(2 * np.pi * drift_freq * time_array)
    
    # High-frequency system noise
    hf_noise = 20 * np.sin(2 * np.pi * 50 * time_array)  # 50 Hz electrical noise
    
    return BASE_PRESSURE + white_noise + drift + hf_noise


def generate_leak_signature(time_array, valve_open, valve_close, distance, sampling_rate):
    """
    Generate acoustic signature of a leak event with wave propagation and attenuation
    
    Parameters:
    - time_array: Time points array
    - valve_open: Time when valve opens (leak starts)
    - valve_close: Time when valve closes (leak stops)
    - distance: Distance from leak source (meters)
    - sampling_rate: Sampling frequency (Hz)
    """
    signal = np.zeros_like(time_array)
    
    # Time delay due to wave propagation
    propagation_delay = distance / SPEED_OF_SOUND
    
    # Distance-based attenuation (geometric spreading + material absorption)
    geometric_attenuation = 1 / (1 + distance)
    material_attenuation = np.exp(-ATTENUATION_COEFF * distance)
    total_attenuation = geometric_attenuation * material_attenuation
    
    for i, t in enumerate(time_array):
        # Adjusted time accounting for propagation delay
        t_adj = t - propagation_delay
        
        if valve_open <= t_adj <= valve_close:
            # Active leak period - multiple frequency components
            # Dominant frequency (leak fundamental)
            f1 = 100 + 20 * np.random.randn()  # ~100 Hz with variation
            component1 = np.sin(2 * np.pi * f1 * t_adj)
            
            # Harmonic frequencies
            f2 = 2 * f1
            f3 = 3 * f1
            component2 = 0.5 * np.sin(2 * np.pi * f2 * t_adj)
            component3 = 0.3 * np.sin(2 * np.pi * f3 * t_adj)
            
            # High-frequency turbulence
            turbulence = 0.2 * np.random.randn()
            
            # Combine components
            leak_signal = component1 + component2 + component3 + turbulence
            
            # Apply amplitude envelope (leak intensity grows then stabilizes)
            envelope_time = t_adj - valve_open
            envelope = 1 - np.exp(-envelope_time * 2)  # Rise time ~0.5s
            
            signal[i] = LEAK_AMPLITUDE * leak_signal * envelope * total_attenuation
            
        elif t_adj > valve_close:
            # Post-leak decay (ringing and reverberations)
            decay_time = t_adj - valve_close
            decay_factor = np.exp(-decay_time * 3)  # Exponential decay
            
            # Residual oscillations
            f_residual = 80
            residual = np.sin(2 * np.pi * f_residual * t_adj)
            
            signal[i] = 0.3 * LEAK_AMPLITUDE * residual * decay_factor * total_attenuation
    
    return signal


def generate_sensor_data(group_name, group_config):
    """
    Generate complete dataset for a test group
    
    Returns: Dictionary with sensor data and metadata
    """
    sampling_rate = group_config['sampling_rate']
    leak_location = group_config['leak_location']
    
    # Leak position mapping (meters along pipe)
    leak_positions = {'A': 5.0, 'D': 15.0, 'E': 22.0}
    leak_pos = leak_positions[leak_location]
    
    # Generate time array
    num_samples = int(DURATION * sampling_rate)
    time_array = np.linspace(0, DURATION, num_samples)
    
    # Generate data for each sensor
    sensor_data = {}
    
    for sensor in SENSORS:
        sensor_pos = SENSOR_POSITIONS[sensor]
        distance_from_leak = abs(sensor_pos - leak_pos)
        
        # Baseline noise
        baseline = generate_baseline_noise(time_array, sampling_rate)
        
        # Leak signature
        leak_signal = generate_leak_signature(
            time_array, 
            VALVE_OPEN_TIME, 
            VALVE_CLOSE_TIME, 
            distance_from_leak,
            sampling_rate
        )
        
        # Stratified flow effect (interface reflections and mode conversion)
        # Adds complexity due to liquid-gas interface
        stratification_effect = 0.15 * LEAK_AMPLITUDE * np.sin(2 * np.pi * 5 * time_array) * \
                                np.exp(-0.5 * distance_from_leak)
        
        # Combine all components
        total_signal = baseline + leak_signal + stratification_effect
        
        # Add sensor-specific calibration offset
        sensor_idx = int(sensor[2:])
        calibration_offset = (sensor_idx - 7) * 10  # Small DC offset per sensor
        
        sensor_data[sensor] = total_signal + calibration_offset
    
    return {
        'time': time_array,
        'sensors': sensor_data,
        'sampling_rate': sampling_rate,
        'leak_location': leak_location,
        'metadata': group_config
    }


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def save_csv_files(group_name, data, output_dir):
    """Save sensor data to CSV files (one file per second as per experiment)"""
    time = data['time']
    sensors_data = data['sensors']
    sampling_rate = data['sampling_rate']
    
    group_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)
    
    samples_per_second = int(sampling_rate)
    num_seconds = int(DURATION)
    
    csv_files = []
    
    for second in range(num_seconds):
        start_idx = second * samples_per_second
        end_idx = (second + 1) * samples_per_second
        
        if end_idx > len(time):
            end_idx = len(time)
        
        # Create DataFrame for this second
        df_dict = {'Time_s': time[start_idx:end_idx]}
        
        for sensor in SENSORS:
            df_dict[sensor] = sensors_data[sensor][start_idx:end_idx]
        
        df = pd.DataFrame(df_dict)
        
        # Filename with descriptive information
        filename = f'{group_name}_second_{second:02d}_{data["leak_location"]}.csv'
        filepath = os.path.join(group_dir, filename)
        df.to_csv(filepath, index=False, float_format='%.6f')
        csv_files.append(filepath)
    
    # Also save a complete file for easy analysis
    complete_df_dict = {'Time_s': time}
    for sensor in SENSORS:
        complete_df_dict[sensor] = sensors_data[sensor]
    
    complete_df = pd.DataFrame(complete_df_dict)
    complete_filename = f'{group_name}_complete_{data["leak_location"]}.csv'
    complete_filepath = os.path.join(group_dir, complete_filename)
    complete_df.to_csv(complete_filepath, index=False, float_format='%.6f')
    csv_files.append(complete_filepath)
    
    return csv_files


def save_metadata(output_dir):
    """Save experimental metadata and configuration"""
    metadata = {
        'experiment': 'Stratified Flow Acoustic Attenuation Study',
        'date_generated': datetime.now().isoformat(),
        'duration_seconds': DURATION,
        'valve_open_time': VALVE_OPEN_TIME,
        'valve_close_time': VALVE_CLOSE_TIME,
        'num_sensors': NUM_SENSORS,
        'sensor_list': SENSORS,
        'sensor_positions_m': SENSOR_POSITIONS,
        'test_groups': TEST_GROUPS,
        'physical_parameters': {
            'speed_of_sound_m_s': SPEED_OF_SOUND,
            'attenuation_coefficient_dB_m': ATTENUATION_COEFF,
            'base_pressure_Pa': BASE_PRESSURE,
            'leak_amplitude_Pa': LEAK_AMPLITUDE
        }
    }
    
    metadata_file = os.path.join(output_dir, 'experiment_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(all_data, output_dir):
    """Generate comprehensive visualization figures"""
    
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Figure 1: Multi-sensor time series for one group
    fig1 = create_multisensor_timeseries(all_data['Group_01'], figures_dir)
    
    # Figure 2: Spatial attenuation analysis
    fig2 = create_attenuation_analysis(all_data, figures_dir)
    
    # Figure 3: Frequency domain analysis
    fig3 = create_frequency_analysis(all_data['Group_02'], figures_dir)
    
    # Figure 4: Leak event comparison across groups
    fig4 = create_group_comparison(all_data, figures_dir)
    
    # Figure 5: Wave propagation visualization
    fig5 = create_wave_propagation(all_data['Group_01'], figures_dir)
    
    print(f"\n✓ Generated 5 visualization figures in {figures_dir}")


def create_multisensor_timeseries(data, output_dir):
    """Figure 1: Time series from multiple sensors showing leak event"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(7, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    time = data['time']
    sensors_to_plot = ['PG01', 'PG03', 'PG05', 'PG07', 'PG09', 'PG11', 'PG13', 'PG14']
    
    for idx, sensor in enumerate(sensors_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        pressure = data['sensors'][sensor]
        ax.plot(time, pressure / 1000, 'b-', linewidth=0.5, alpha=0.7)
        
        # Mark leak event
        ax.axvline(VALVE_OPEN_TIME, color='r', linestyle='--', alpha=0.6, label='Valve Open')
        ax.axvline(VALVE_CLOSE_TIME, color='g', linestyle='--', alpha=0.6, label='Valve Close')
        ax.axvspan(VALVE_OPEN_TIME, VALVE_CLOSE_TIME, alpha=0.1, color='red', label='Leak Active')
        
        sensor_pos = SENSOR_POSITIONS[sensor]
        ax.set_title(f'{sensor} (Position: {sensor_pos:.1f} m)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Pressure (kPa)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, DURATION])
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    fig.suptitle(f'Multi-Sensor Acoustic Pressure Time Series - {data["metadata"]["description"]}\n' + 
                 f'Sampling Rate: {data["sampling_rate"]} Hz, Leak Location: {data["leak_location"]}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    filename = os.path.join(output_dir, 'Fig1_Multisensor_Timeseries.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_attenuation_analysis(all_data, output_dir):
    """Figure 2: Spatial attenuation characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Acoustic Wave Attenuation Analysis Across Distance', 
                 fontsize=14, fontweight='bold')
    
    for idx, (group_name, data) in enumerate(all_data.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Calculate peak pressure during leak for each sensor
        leak_start_idx = int(VALVE_OPEN_TIME * data['sampling_rate'])
        leak_end_idx = int(VALVE_CLOSE_TIME * data['sampling_rate'])
        
        distances = []
        peak_pressures = []
        
        leak_pos = {'A': 5.0, 'D': 15.0, 'E': 22.0}[data['leak_location']]
        
        for sensor in SENSORS[:10]:  # First 10 sensors
            sensor_pos = SENSOR_POSITIONS[sensor]
            distance = abs(sensor_pos - leak_pos)
            distances.append(distance)
            
            sensor_signal = data['sensors'][sensor][leak_start_idx:leak_end_idx]
            baseline = data['sensors'][sensor][:leak_start_idx].mean()
            peak_pressure = np.max(np.abs(sensor_signal - baseline))
            peak_pressures.append(peak_pressure)
        
        # Plot measured attenuation
        ax.scatter(distances, np.array(peak_pressures) / 1000, s=100, 
                  alpha=0.7, label='Measured', c='blue', edgecolors='black')
        
        # Theoretical attenuation curve
        d_theory = np.linspace(0.1, max(distances), 100)
        p_theory = LEAK_AMPLITUDE * (1 / (1 + d_theory)) * np.exp(-ATTENUATION_COEFF * d_theory)
        ax.plot(d_theory, p_theory / 1000, 'r--', linewidth=2, 
               label='Theoretical Model', alpha=0.7)
        
        ax.set_xlabel('Distance from Leak (m)', fontsize=11)
        ax.set_ylabel('Peak Pressure Amplitude (kPa)', fontsize=11)
        ax.set_title(f'{group_name}: Leak at Position {data["leak_location"]}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_yscale('log')
    
    filename = os.path.join(output_dir, 'Fig2_Attenuation_Analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_frequency_analysis(data, output_dir):
    """Figure 3: Frequency domain analysis (FFT)"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Frequency Domain Analysis - Power Spectral Density', 
                 fontsize=14, fontweight='bold')
    
    time = data['time']
    sampling_rate = data['sampling_rate']
    
    # Analyze three time windows: baseline, during leak, post-leak
    windows = {
        'Baseline (0-5s)': (0, int(5 * sampling_rate)),
        'Active Leak (5-20s)': (int(5 * sampling_rate), int(20 * sampling_rate)),
        'Post-Leak (20-30s)': (int(20 * sampling_rate), len(time))
    }
    
    sensors_to_analyze = ['PG01', 'PG05', 'PG10']
    
    for col_idx, sensor in enumerate(sensors_to_analyze):
        for row_idx, (window_name, (start, end)) in enumerate(windows.items()):
            ax = axes[row_idx, col_idx]
            
            signal = data['sensors'][sensor][start:end]
            
            # Compute FFT
            fft_values = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
            
            # Power spectral density
            psd = np.abs(fft_values)**2 / len(signal)
            
            # Plot only positive frequencies
            positive_freq_idx = fft_freq > 0
            freq_plot = fft_freq[positive_freq_idx]
            psd_plot = 10 * np.log10(psd[positive_freq_idx] + 1e-10)  # dB scale
            
            ax.plot(freq_plot, psd_plot, linewidth=0.8)
            ax.set_xlim([0, 500])  # Focus on 0-500 Hz range
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('PSD (dB)', fontsize=10)
            ax.set_title(f'{sensor} - {window_name}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Mark expected leak frequency
            if 'Leak' in window_name:
                ax.axvline(100, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    
    filename = os.path.join(output_dir, 'Fig3_Frequency_Analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_group_comparison(all_data, output_dir):
    """Figure 4: Comparison of leak signatures across test groups"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Leak Signature Comparison Across Test Groups (Sensor PG05)', 
                 fontsize=14, fontweight='bold')
    
    sensor = 'PG05'
    
    for idx, (group_name, data) in enumerate(all_data.items()):
        ax = axes[idx // 2, idx % 2]
        
        time = data['time']
        pressure = data['sensors'][sensor]
        
        # Focus on leak event window
        start_idx = int((VALVE_OPEN_TIME - 2) * data['sampling_rate'])
        end_idx = int((VALVE_CLOSE_TIME + 5) * data['sampling_rate'])
        
        time_window = time[start_idx:end_idx]
        pressure_window = pressure[start_idx:end_idx]
        
        ax.plot(time_window, pressure_window / 1000, linewidth=1)
        ax.axvline(VALVE_OPEN_TIME, color='r', linestyle='--', alpha=0.6, label='Valve Open')
        ax.axvline(VALVE_CLOSE_TIME, color='g', linestyle='--', alpha=0.6, label='Valve Close')
        ax.axvspan(VALVE_OPEN_TIME, VALVE_CLOSE_TIME, alpha=0.1, color='red')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Pressure (kPa)', fontsize=11)
        ax.set_title(f'{group_name}: Leak @ {data["leak_location"]}, '
                    f'SR={data["sampling_rate"]} Hz', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    filename = os.path.join(output_dir, 'Fig4_Group_Comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_wave_propagation(data, output_dir):
    """Figure 5: Wave propagation visualization (space-time diagram)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Acoustic Wave Propagation: Space-Time Analysis', 
                 fontsize=14, fontweight='bold')
    
    time = data['time']
    sampling_rate = data['sampling_rate']
    
    # Create space-time matrix
    positions = [SENSOR_POSITIONS[s] for s in SENSORS]
    
    # Subsample time for visualization
    time_step = int(sampling_rate / 100)  # 100 Hz effective sampling for visualization
    time_subsampled = time[::time_step]
    
    space_time_matrix = np.zeros((len(SENSORS), len(time_subsampled)))
    
    for i, sensor in enumerate(SENSORS):
        signal = data['sensors'][sensor][::time_step]
        baseline = signal[:int(5 * 100)].mean()  # Baseline average
        space_time_matrix[i, :] = signal - baseline
    
    # Plot 1: Space-time heatmap
    im = ax1.imshow(space_time_matrix, aspect='auto', cmap='seismic', 
                    extent=[0, DURATION, positions[-1], positions[0]],
                    vmin=-2000, vmax=2000, interpolation='bilinear')
    ax1.axvline(VALVE_OPEN_TIME, color='yellow', linestyle='--', linewidth=2, label='Valve Open')
    ax1.axvline(VALVE_CLOSE_TIME, color='lime', linestyle='--', linewidth=2, label='Valve Close')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Distance (m)', fontsize=12)
    ax1.set_title('Pressure Perturbation Field (Space-Time)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Pressure Deviation (Pa)', fontsize=11)
    
    # Plot 2: Wave arrival times
    leak_pos = {'A': 5.0, 'D': 15.0, 'E': 22.0}[data['leak_location']]
    
    distances = [abs(SENSOR_POSITIONS[s] - leak_pos) for s in SENSORS]
    theoretical_arrival = VALVE_OPEN_TIME + np.array(distances) / SPEED_OF_SOUND
    
    ax2.scatter(theoretical_arrival, distances, s=100, c='blue', 
               label='Theoretical Arrival', alpha=0.7, edgecolors='black')
    
    # Fit line
    z = np.polyfit(theoretical_arrival, distances, 1)
    p = np.poly1d(z)
    ax2.plot(theoretical_arrival, p(theoretical_arrival), "r--", 
            linewidth=2, alpha=0.7, label=f'Fit: v={1/z[0]:.1f} m/s')
    
    ax2.set_xlabel('Wave Arrival Time (s)', fontsize=12)
    ax2.set_ylabel('Distance from Leak (m)', fontsize=12)
    ax2.set_title('Acoustic Wave Propagation Speed Validation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    filename = os.path.join(output_dir, 'Fig5_Wave_Propagation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("ACOUSTIC PRESSURE DATASET GENERATOR")
    print("Study: Attenuation Mechanisms in Stratified Flows")
    print("=" * 80)
    
    # Create output directory
    output_dir = '/workspace/acoustic_pressure_dataset'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Generate data for all test groups
    all_data = {}
    all_csv_files = []
    
    print(f"\n🔬 Generating data for {len(TEST_GROUPS)} test groups...")
    print("-" * 80)
    
    for group_name, group_config in TEST_GROUPS.items():
        print(f"\n[{group_name}]")
        print(f"  Leak Location: {group_config['leak_location']}")
        print(f"  Sampling Rate: {group_config['sampling_rate']} Hz")
        print(f"  Configuration: {group_config['sensor_config']}")
        
        # Generate sensor data
        data = generate_sensor_data(group_name, group_config)
        all_data[group_name] = data
        
        # Save to CSV
        csv_files = save_csv_files(group_name, data, output_dir)
        all_csv_files.extend(csv_files)
        print(f"  ✓ Generated {len(csv_files)} CSV files")
    
    # Save metadata
    print(f"\n📋 Saving experiment metadata...")
    metadata_file = save_metadata(output_dir)
    print(f"  ✓ Metadata saved: {metadata_file}")
    
    # Create visualizations
    print(f"\n📊 Creating visualization figures...")
    create_visualizations(all_data, output_dir)
    
    # Create ZIP file with all CSV files
    print(f"\n📦 Creating ZIP archive...")
    zip_filename = os.path.join(output_dir, 'acoustic_pressure_data_all_groups.zip')
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_file in all_csv_files:
            arcname = os.path.relpath(csv_file, output_dir)
            zipf.write(csv_file, arcname)
        # Add metadata to zip
        zipf.write(metadata_file, os.path.basename(metadata_file))
    
    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"  ✓ ZIP file created: {zip_filename}")
    print(f"  ✓ Size: {zip_size_mb:.2f} MB")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"  • Test Groups: {len(TEST_GROUPS)}")
    print(f"  • Sensors: {NUM_SENSORS}")
    print(f"  • Duration: {DURATION} s")
    print(f"  • CSV Files: {len(all_csv_files)}")
    print(f"  • Figures: 5")
    print(f"  • ZIP Archive: {zip_filename}")
    print(f"\n✅ All files ready for download and analysis!")
    print("=" * 80)
    
    # Create README
    create_readme(output_dir)
    

def create_readme(output_dir):
    """Create README file with dataset documentation"""
    readme_content = """# Acoustic Pressure Dataset: Stratified Flow Leakage Study

## Overview
This synthetic dataset simulates high-frequency acoustic pressure measurements for studying 
leak detection and wave attenuation mechanisms in stratified (multi-phase) flows.

**Research Topic:** Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics

## Dataset Structure

### Test Groups
The dataset contains 4 experimental groups with varying configurations:

- **Group_01**: Leak at Position A, Sensors A-B active, 17,060 Hz sampling
- **Group_02**: Leak at Position D, All sensors active, 10,000 Hz sampling
- **Group_03**: Leak at Position E, Sensors A-B active, 10,000 Hz sampling
- **Group_04**: Leak at Position A, All sensors active, 10,000 Hz sampling

### Sensors
- **Total Sensors:** 14 (PG01 through PG14)
- **Positions:** Strategically placed from 0.5m to 27.0m along the test section
- **Purpose:** Capture spatial attenuation and wave propagation characteristics

### Experimental Protocol
- **Duration:** 30 seconds per test
- **Baseline Period:** 0-5 seconds (normal operation)
- **Leak Event:** 5-20 seconds (valve open)
- **Recovery Period:** 20-30 seconds (post-leak)

## Data Files

### CSV Files (in ZIP archive)
Each test group includes:
- Individual files per second (e.g., `Group_01_second_00_A.csv`)
- Complete time series file (e.g., `Group_01_complete_A.csv`)

**CSV Format:**
```
Time_s, PG01, PG02, PG03, ..., PG14
0.0000, 101325.5, 101330.2, ...
0.0001, 101324.8, 101329.5, ...
```

### Figures
1. **Fig1_Multisensor_Timeseries.png** - Multi-sensor pressure time series
2. **Fig2_Attenuation_Analysis.png** - Spatial attenuation characteristics
3. **Fig3_Frequency_Analysis.png** - FFT/PSD frequency domain analysis
4. **Fig4_Group_Comparison.png** - Inter-group leak signature comparison
5. **Fig5_Wave_Propagation.png** - Space-time wave propagation visualization

### Metadata
- **experiment_metadata.json** - Complete experimental configuration and parameters

## Physical Parameters

- **Speed of Sound:** 1500 m/s (water/stratified medium)
- **Attenuation Coefficient:** 0.05 dB/m
- **Base Pressure:** 101,325 Pa (atmospheric)
- **Leak Amplitude:** ~5,000 Pa (peak pressure change)
- **Leak Frequency:** ~100 Hz fundamental with harmonics

## Key Features

### Acoustic Signatures
- **Baseline Noise:** System noise, flow fluctuations, electrical interference
- **Leak Signal:** Multi-frequency components (fundamental + harmonics)
- **Attenuation:** Distance-dependent geometric and material damping
- **Stratification Effects:** Interface reflections and mode conversion
- **Wave Propagation:** Time delays based on speed of sound

### Analysis Capabilities
This dataset enables:
- Leak detection algorithm development
- Attenuation mechanism characterization
- Wave propagation speed validation
- Sensor placement optimization
- Frequency-domain signature analysis
- Machine learning model training

## Usage Example (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load complete dataset for Group 01
df = pd.read_csv('Group_01/Group_01_complete_A.csv')

# Plot sensor PG05
plt.figure(figsize=(12, 6))
plt.plot(df['Time_s'], df['PG05'])
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Sensor PG05 - Acoustic Pressure')
plt.axvline(5, color='r', linestyle='--', label='Valve Open')
plt.axvline(20, color='g', linestyle='--', label='Valve Close')
plt.legend()
plt.grid(True)
plt.show()
```

## Citation
If you use this dataset, please cite:
```
Synthetic Acoustic Pressure Dataset for Stratified Flow Leakage Studies
Generated: 2026
Study: Attenuation Mechanisms in Stratified Flows - Beyond Single Phase Leakage Acoustics
```

## Contact
For questions or additional information about this dataset, please refer to the 
experiment_metadata.json file for complete technical specifications.

---
**Dataset Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
**Format:** CSV (compressed in ZIP) + Figures (PNG)
**License:** Research and educational use
"""
    
    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ README created: {readme_file}")


if __name__ == "__main__":
    main()
