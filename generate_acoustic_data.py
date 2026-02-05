#!/usr/bin/env python3
"""
Synthetic High-Frequency Acoustic Pressure Data Generator
for Leak Detection in Stratified Flows

This script generates realistic synthetic data for acoustic wave propagation
and attenuation analysis in pipeline leak detection experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Experimental Configuration
SAMPLING_RATE_HZ = 10000  # 10 kHz
SAMPLING_RATE_HZ_HIGH = 17060  # Higher rate for Group 01
DURATION_SEC = 30  # Total recording duration
LEAK_START_SEC = 5  # Valve opens at 5s
LEAK_END_SEC = 20  # Valve closes at 20s

# Sensor Configuration (14 sensors)
SENSORS = [f'PG{i:02d}' for i in range(1, 15)]

# Sensor positions along pipeline (meters from reference point)
SENSOR_POSITIONS = {
    'PG01': 0.0,    # Before Valve 1
    'PG02': 2.5,    # After Valve 1
    'PG03': 5.0,    # Before Leak Point A
    'PG04': 7.5,    # After Leak Point A
    'PG05': 10.0,   # Mid-section
    'PG06': 12.5,   # Before Leak Point D
    'PG07': 15.0,   # After Leak Point D
    'PG08': 17.5,   # Before Valve 2
    'PG09': 20.0,   # After Valve 2
    'PG10': 22.5,   # Before Leak Point E
    'PG11': 25.0,   # After Leak Point E
    'PG12': 27.5,   # End section
    'PG13': 30.0,   # Near endpoint
    'PG14': 32.5,   # Endpoint
}

# Leak Locations
LEAK_LOCATIONS = {
    'A': 6.0,   # Between PG03 and PG04
    'D': 13.5,  # Between PG06 and PG07
    'E': 24.0,  # Between PG10 and PG11
}

# Test Groups Configuration
TEST_GROUPS = {
    'Group_01': {
        'leak_location': 'A',
        'active_sensors': ['A', 'B', 'C', 'D'],  # All sensor configurations
        'sampling_rate': SAMPLING_RATE_HZ_HIGH,
        'leak_intensity': 1.0,
    },
    'Group_02': {
        'leak_location': 'D',
        'active_sensors': ['A', 'B'],
        'sampling_rate': SAMPLING_RATE_HZ,
        'leak_intensity': 0.8,
    },
    'Group_03': {
        'leak_location': 'E',
        'active_sensors': ['A', 'B', 'C', 'D'],
        'sampling_rate': SAMPLING_RATE_HZ,
        'leak_intensity': 0.9,
    },
    'Group_04': {
        'leak_location': 'A',
        'active_sensors': ['A', 'B'],
        'sampling_rate': SAMPLING_RATE_HZ,
        'leak_intensity': 0.85,
    },
}

# Physical Parameters
SPEED_OF_SOUND = 1500  # m/s in water
ATTENUATION_COEFFICIENT = 0.15  # dB/m at 1 kHz
AMBIENT_PRESSURE_BAR = 1.0
NOISE_LEVEL_BAR = 0.001

class AcousticDataGenerator:
    """Generate synthetic acoustic pressure data for leak detection experiments."""
    
    def __init__(self, group_name, config):
        self.group_name = group_name
        self.config = config
        self.sampling_rate = config['sampling_rate']
        self.leak_location = LEAK_LOCATIONS[config['leak_location']]
        self.leak_intensity = config['leak_intensity']
        
    def generate_baseline_noise(self, n_samples):
        """Generate baseline pressure noise (normal operation)."""
        # White noise + low frequency drift
        white_noise = np.random.normal(0, NOISE_LEVEL_BAR * 0.5, n_samples)
        drift = 0.0002 * np.sin(2 * np.pi * 0.5 * np.arange(n_samples) / self.sampling_rate)
        return AMBIENT_PRESSURE_BAR + white_noise + drift
    
    def generate_leak_signature(self, time_array, sensor_position):
        """Generate acoustic signature from leak with realistic physics."""
        # Distance from leak to sensor
        distance = abs(sensor_position - self.leak_location)
        
        # Time delay due to propagation
        time_delay = distance / SPEED_OF_SOUND
        
        # Leak signature components
        # 1. Broadband turbulent noise (main component)
        fundamental_freq = 850  # Hz - turbulent jet frequency
        harmonics = [2, 3, 5, 7, 11]  # Harmonic series
        
        signature = np.zeros_like(time_array)
        
        # Fundamental frequency
        amplitude = self.leak_intensity * 0.5 * np.exp(-ATTENUATION_COEFFICIENT * distance)
        signature += amplitude * np.sin(2 * np.pi * fundamental_freq * (time_array - time_delay))
        
        # Add harmonics with decreasing amplitude
        for i, h in enumerate(harmonics):
            harm_amp = amplitude / (h * (i + 2))
            signature += harm_amp * np.sin(2 * np.pi * fundamental_freq * h * (time_array - time_delay))
        
        # 2. Add broadband turbulent noise (500 Hz - 5 kHz range)
        n_samples = len(time_array)
        turbulent_noise = np.random.normal(0, amplitude * 0.3, n_samples)
        
        # Band-pass filter simulation (500-4900 Hz, must be < Nyquist frequency)
        from scipy import signal
        max_freq = min(4900, self.sampling_rate / 2.0 - 100)  # Keep below Nyquist
        b, a = signal.butter(4, [500, max_freq], btype='band', fs=self.sampling_rate)
        turbulent_noise = signal.filtfilt(b, a, turbulent_noise)
        signature += turbulent_noise
        
        # 3. Apply time envelope for leak event
        leak_mask = np.zeros_like(time_array)
        leak_indices = (time_array >= LEAK_START_SEC) & (time_array <= LEAK_END_SEC)
        
        # Smooth onset and offset
        leak_mask[leak_indices] = 1.0
        
        # Apply Hann window for smooth transitions
        transition_samples = int(0.1 * self.sampling_rate)  # 100ms transition
        for i in range(len(leak_mask) - 1):
            if leak_mask[i] == 0 and leak_mask[i + 1] == 1:
                # Onset
                ramp_up = np.linspace(0, 1, transition_samples)
                leak_mask[i:i + transition_samples] = ramp_up
            elif leak_mask[i] == 1 and leak_mask[i + 1] == 0:
                # Offset
                ramp_down = np.linspace(1, 0, transition_samples)
                leak_mask[i:i + transition_samples] = ramp_down
        
        signature *= leak_mask
        
        # 4. Add reflections from pipeline boundaries
        reflection_amplitude = amplitude * 0.15
        reflection_delay = 0.05  # 50ms reflection delay
        reflection = reflection_amplitude * np.sin(2 * np.pi * fundamental_freq * 
                                                   (time_array - time_delay - reflection_delay))
        reflection *= leak_mask
        signature += reflection
        
        return signature
    
    def generate_sensor_data(self, sensor_name, sensor_position):
        """Generate complete pressure time series for one sensor."""
        # Time array
        n_samples = int(DURATION_SEC * self.sampling_rate)
        time_array = np.linspace(0, DURATION_SEC, n_samples)
        
        # Start with baseline noise
        pressure = self.generate_baseline_noise(n_samples)
        
        # Add leak signature
        leak_signal = self.generate_leak_signature(time_array, sensor_position)
        pressure += leak_signal
        
        # Add sensor-specific noise characteristics
        sensor_noise = np.random.normal(0, NOISE_LEVEL_BAR * 0.3, n_samples)
        pressure += sensor_noise
        
        return time_array, pressure
    
    def generate_all_sensors(self):
        """Generate data for all sensors."""
        data_dict = {}
        
        print(f"Generating data for {self.group_name}...")
        print(f"  Leak location: {self.config['leak_location']} ({self.leak_location}m)")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Number of sensors: {len(SENSORS)}")
        
        for sensor_name in SENSORS:
            sensor_position = SENSOR_POSITIONS[sensor_name]
            time_array, pressure = self.generate_sensor_data(sensor_name, sensor_position)
            data_dict[sensor_name] = pressure
        
        # Create DataFrame
        n_samples = int(DURATION_SEC * self.sampling_rate)
        df = pd.DataFrame(data_dict)
        df.insert(0, 'Time_s', np.linspace(0, DURATION_SEC, n_samples))
        
        return df

def save_dataset(df, group_name, output_dir):
    """Save dataset to CSV files (one per second)."""
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    
    sampling_rate = len(df) / DURATION_SEC
    samples_per_second = int(sampling_rate)
    
    print(f"  Saving {DURATION_SEC} CSV files (one per second)...")
    
    for second in range(DURATION_SEC):
        start_idx = second * samples_per_second
        end_idx = (second + 1) * samples_per_second
        
        df_second = df.iloc[start_idx:end_idx].copy()
        df_second['Time_s'] = df_second['Time_s'] - second  # Reset time to 0-1s
        
        filename = f"{group_name}_second_{second:02d}.csv"
        filepath = group_dir / filename
        df_second.to_csv(filepath, index=False, float_format='%.8f')
    
    print(f"  Saved {DURATION_SEC} files to {group_dir}")
    return group_dir

def create_visualizations(df, group_name, config, output_dir):
    """Create comprehensive visualization figures."""
    fig_dir = output_dir / 'figures' / group_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Creating visualizations for {group_name}...")
    
    # Figure 1: Full time series for selected sensors
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f'{group_name}: Acoustic Pressure Time Series\nLeak Location: {config["leak_location"]}', 
                 fontsize=14, fontweight='bold')
    
    selected_sensors = ['PG03', 'PG07', 'PG11', 'PG14']
    for idx, sensor in enumerate(selected_sensors):
        ax = axes[idx]
        time = df['Time_s'].values
        pressure = df[sensor].values
        
        ax.plot(time, pressure, linewidth=0.5, alpha=0.7, color='navy')
        ax.axvline(LEAK_START_SEC, color='red', linestyle='--', linewidth=1.5, label='Leak Start')
        ax.axvline(LEAK_END_SEC, color='green', linestyle='--', linewidth=1.5, label='Leak End')
        ax.set_ylabel(f'{sensor}\nPressure (bar)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, DURATION_SEC)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        if idx == len(selected_sensors) - 1:
            ax.set_xlabel('Time (s)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'{group_name}_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Spectrogram for sensor nearest to leak
    leak_loc = LEAK_LOCATIONS[config['leak_location']]
    nearest_sensor = min(SENSORS, key=lambda s: abs(SENSOR_POSITIONS[s] - leak_loc))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    from scipy import signal as scipy_signal
    
    sampling_rate = len(df) / DURATION_SEC
    f, t, Sxx = scipy_signal.spectrogram(df[nearest_sensor].values, fs=sampling_rate, 
                                         nperseg=1024, noverlap=512)
    
    # Limit frequency range to 0-5000 Hz
    freq_mask = f <= 5000
    
    pcm = ax.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask] + 1e-10), 
                        shading='gouraud', cmap='viridis')
    ax.axvline(LEAK_START_SEC, color='red', linestyle='--', linewidth=2, label='Leak Start')
    ax.axvline(LEAK_END_SEC, color='green', linestyle='--', linewidth=2, label='Leak End')
    
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(f'{group_name}: Spectrogram - {nearest_sensor} (Nearest to Leak)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label('Power Spectral Density (dB/Hz)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'{group_name}_spectrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Spatial attenuation analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    leak_loc = LEAK_LOCATIONS[config['leak_location']]
    
    # Calculate RMS during leak event
    leak_start_idx = int(LEAK_START_SEC * sampling_rate)
    leak_end_idx = int(LEAK_END_SEC * sampling_rate)
    
    positions = []
    rms_values = []
    distances = []
    
    for sensor in SENSORS:
        pos = SENSOR_POSITIONS[sensor]
        leak_data = df[sensor].iloc[leak_start_idx:leak_end_idx].values
        rms = np.sqrt(np.mean((leak_data - AMBIENT_PRESSURE_BAR) ** 2))
        
        positions.append(pos)
        rms_values.append(rms)
        distances.append(abs(pos - leak_loc))
    
    # Plot 1: RMS vs Position
    ax1.plot(positions, rms_values, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax1.axvline(leak_loc, color='red', linestyle='--', linewidth=2, label=f'Leak Location ({config["leak_location"]})')
    ax1.set_xlabel('Sensor Position (m)', fontsize=11)
    ax1.set_ylabel('RMS Pressure (bar)', fontsize=11)
    ax1.set_title(f'{group_name}: Spatial Pressure Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Attenuation vs Distance
    # Sort by distance
    sorted_indices = np.argsort(distances)
    sorted_distances = np.array(distances)[sorted_indices]
    sorted_rms = np.array(rms_values)[sorted_indices]
    
    ax2.semilogy(sorted_distances, sorted_rms, 'o', markersize=8, color='darkgreen', label='Measured')
    
    # Fit exponential decay
    from scipy.optimize import curve_fit
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)
    
    try:
        popt, _ = curve_fit(exponential_decay, sorted_distances, sorted_rms, p0=[1.0, 0.1])
        fit_distances = np.linspace(0, max(sorted_distances), 100)
        fit_values = exponential_decay(fit_distances, *popt)
        ax2.plot(fit_distances, fit_values, '--', linewidth=2, color='red', 
                label=f'Exponential Fit (α={popt[1]:.3f})')
    except:
        pass
    
    ax2.set_xlabel('Distance from Leak (m)', fontsize=11)
    ax2.set_ylabel('RMS Pressure (bar)', fontsize=11)
    ax2.set_title(f'{group_name}: Acoustic Attenuation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'{group_name}_spatial_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Wave propagation visualization (waterfall plot)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Select time window around leak start
    window_start = int((LEAK_START_SEC - 1) * sampling_rate)
    window_end = int((LEAK_START_SEC + 3) * sampling_rate)
    time_window = df['Time_s'].iloc[window_start:window_end].values
    
    offset = 0.015  # Vertical offset between traces
    for idx, sensor in enumerate(SENSORS):
        data = df[sensor].iloc[window_start:window_end].values - AMBIENT_PRESSURE_BAR
        ax.plot(time_window, data + idx * offset, linewidth=0.8, label=sensor)
    
    ax.axvline(LEAK_START_SEC, color='red', linestyle='--', linewidth=2, label='Leak Start')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Sensor (offset for clarity)', fontsize=12)
    ax.set_title(f'{group_name}: Wave Propagation - Waterfall Plot\nLeak Location: {config["leak_location"]}', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f'{group_name}_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved 4 visualization figures to {fig_dir}")

def create_summary_figures(output_dir):
    """Create summary comparison figures across all groups."""
    fig_dir = output_dir / 'figures'
    
    print("\nCreating summary comparison figures...")
    
    # Load summary data from each group
    group_data = {}
    
    for group_name in TEST_GROUPS.keys():
        # Load first second of data
        csv_path = output_dir / group_name / f"{group_name}_second_00.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            group_data[group_name] = df
    
    # Figure: Comparison of leak signatures across groups
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of Leak Signatures Across Test Groups', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (group_name, df) in enumerate(group_data.items()):
        ax = axes[idx]
        config = TEST_GROUPS[group_name]
        
        # Plot first few sensors
        for sensor in ['PG03', 'PG07', 'PG11']:
            if sensor in df.columns:
                ax.plot(df['Time_s'], df[sensor], linewidth=0.8, label=sensor, alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Pressure (bar)', fontsize=10)
        ax.set_title(f'{group_name}\nLeak: {config["leak_location"]}, Intensity: {config["leak_intensity"]}', 
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'all_groups_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary figure to {fig_dir}")

def create_zip_archive(output_dir):
    """Create ZIP archive of all CSV files."""
    print("\nCreating ZIP archive of CSV files...")
    
    zip_path = output_dir / 'acoustic_pressure_data.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for group_name in TEST_GROUPS.keys():
            group_dir = output_dir / group_name
            if group_dir.exists():
                for csv_file in group_dir.glob('*.csv'):
                    arcname = f"{group_name}/{csv_file.name}"
                    zipf.write(csv_file, arcname=arcname)
    
    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Created ZIP archive: {zip_path}")
    print(f"  Archive size: {file_size_mb:.2f} MB")
    
    return zip_path

def create_metadata(output_dir):
    """Create metadata file describing the dataset."""
    metadata = {
        'dataset_name': 'Synthetic High-Frequency Acoustic Pressure Data',
        'purpose': 'Leak Detection in Stratified Flows - Acoustic Wave Propagation and Attenuation Analysis',
        'generated_date': datetime.now().isoformat(),
        'parameters': {
            'sampling_rate_hz': SAMPLING_RATE_HZ,
            'duration_sec': DURATION_SEC,
            'leak_start_sec': LEAK_START_SEC,
            'leak_end_sec': LEAK_END_SEC,
            'num_sensors': len(SENSORS),
            'speed_of_sound_m_s': SPEED_OF_SOUND,
            'attenuation_coefficient_db_m': ATTENUATION_COEFFICIENT,
        },
        'sensors': {
            'names': SENSORS,
            'positions_m': SENSOR_POSITIONS,
        },
        'leak_locations': LEAK_LOCATIONS,
        'test_groups': TEST_GROUPS,
        'file_structure': {
            'csv_files': 'One CSV file per second per group (30 files per group)',
            'total_csv_files': 4 * 30,
            'columns': ['Time_s'] + SENSORS,
            'format': 'CSV with 8 decimal precision',
        },
        'data_characteristics': {
            'baseline_phase': '0-5s: Normal operation with ambient noise',
            'transient_phase': '5-20s: Active leak with acoustic signature',
            'recovery_phase': '20-30s: Post-leak recovery',
            'acoustic_features': [
                'Fundamental frequency: ~850 Hz (turbulent jet)',
                'Harmonics: 2nd, 3rd, 5th, 7th, 11th',
                'Broadband turbulence: 500-5000 Hz',
                'Spatial attenuation: Exponential decay with distance',
                'Propagation delay: Based on speed of sound in water',
                'Reflections: Pipeline boundary reflections included',
            ],
        },
    }
    
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCreated metadata file: {metadata_path}")
    
    return metadata_path

def main():
    """Main execution function."""
    print("=" * 70)
    print("ACOUSTIC PRESSURE DATA GENERATOR")
    print("Stratified Flow Leak Detection - High-Frequency Time Series Data")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path('/workspace/acoustic_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data for each test group
    all_dataframes = {}
    
    for group_name, config in TEST_GROUPS.items():
        print(f"\n{'=' * 70}")
        print(f"Processing {group_name}")
        print(f"{'=' * 70}")
        
        # Generate data
        generator = AcousticDataGenerator(group_name, config)
        df = generator.generate_all_sensors()
        all_dataframes[group_name] = df
        
        # Save CSV files
        save_dataset(df, group_name, output_dir)
        
        # Create visualizations
        create_visualizations(df, group_name, config, output_dir)
    
    # Create summary figures
    create_summary_figures(output_dir)
    
    # Create ZIP archive
    zip_path = create_zip_archive(output_dir)
    
    # Create metadata
    metadata_path = create_metadata(output_dir)
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - CSV data: {4 * 30} files ({4} groups × {30} seconds)")
    print(f"  - ZIP archive: {zip_path}")
    print(f"  - Figures: {4 * 4 + 1} visualization files")
    print(f"  - Metadata: {metadata_path}")
    print()

if __name__ == '__main__':
    main()
