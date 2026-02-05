#!/usr/bin/env python3
"""
Generate Synthetic Core Time-Series Pressure Data for:
"Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"

This script fabricates realistic high-frequency acoustic pressure data simulating
leak detection experiments in pipeline systems with stratified flow conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directories
OUTPUT_DIR = "/workspace/pressure_data_output"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv_data")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Sampling parameters
SAMPLING_RATE_STANDARD = 10000  # 10 kHz
SAMPLING_RATE_HIGH = 17060     # 17.06 kHz for Group 01

# Time parameters (seconds)
BASELINE_DURATION = 30    # Normal operation before leak
LEAK_START = 5            # Valve opens at 5s into transient phase
LEAK_END = 20             # Valve closes at 20s
TRANSIENT_DURATION = 30   # Total transient recording
POST_EVENT_DURATION = 30  # Recovery period

# Sensor configuration
SENSORS = [f"PG{i}" for i in range(1, 15)]  # PG1 to PG14

# Sensor positions (meters from inlet) - strategic placement
SENSOR_POSITIONS = {
    'PG1': 0.5,    # Before Valve 1
    'PG2': 2.0,    # After Valve 1
    'PG3': 5.0,    # Before Leak Point A
    'PG4': 7.5,    # After Leak Point A
    'PG5': 10.0,   # Mid-section 1
    'PG6': 15.0,   # Before Leak Point D
    'PG7': 17.5,   # After Leak Point D
    'PG8': 20.0,   # Mid-section 2
    'PG9': 25.0,   # Before Valve 2
    'PG10': 27.5,  # After Valve 2
    'PG11': 30.0,  # Before Leak Point E
    'PG12': 32.5,  # After Leak Point E
    'PG13': 35.0,  # Near outlet
    'PG14': 37.5,  # At outlet
}

# Leak locations (meters from inlet)
LEAK_LOCATIONS = {
    'A': 6.0,
    'D': 16.0,
    'E': 31.0,
}

# Test groups configuration
TEST_GROUPS = {
    '01': {'leak_location': 'A', 'sensors': ['A', 'B'], 'sampling_rate': 17060, 'description': 'High-rate test at Leak A'},
    '02': {'leak_location': 'A', 'sensors': ['A', 'B', 'C', 'D'], 'sampling_rate': 10000, 'description': 'Full sensor array at Leak A'},
    '03': {'leak_location': 'D', 'sensors': ['A', 'B', 'C', 'D'], 'sampling_rate': 10000, 'description': 'Full sensor array at Leak D'},
    '04': {'leak_location': 'E', 'sensors': ['A', 'B', 'C', 'D'], 'sampling_rate': 10000, 'description': 'Full sensor array at Leak E'},
}

# Physical parameters for stratified flow
SPEED_OF_SOUND_LIQUID = 1480  # m/s (water)
SPEED_OF_SOUND_GAS = 340      # m/s (air)
EFFECTIVE_SOUND_SPEED = 800   # m/s (stratified mixture approximation)
ATTENUATION_COEFF = 0.08      # Neper/m (frequency-dependent attenuation)

# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def generate_baseline_noise(duration, fs, sensor_id):
    """Generate realistic baseline noise for normal pipeline operation."""
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Base pressure (atmospheric + static head) in kPa
    base_pressure = 101.325 + np.random.uniform(5, 15)
    
    # Low-frequency flow noise (turbulence)
    flow_noise = 0.5 * np.sin(2 * np.pi * 2 * t + np.random.uniform(0, 2*np.pi))
    flow_noise += 0.3 * np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
    
    # Pump harmonics (typically 50/60 Hz and harmonics)
    pump_freq = 50 + np.random.uniform(-2, 2)
    pump_noise = 0.2 * np.sin(2 * np.pi * pump_freq * t)
    pump_noise += 0.1 * np.sin(2 * np.pi * 2 * pump_freq * t)
    pump_noise += 0.05 * np.sin(2 * np.pi * 3 * pump_freq * t)
    
    # High-frequency random noise (sensor/electronic noise)
    hf_noise = 0.1 * np.random.randn(n_samples)
    
    # Combine all components
    pressure = base_pressure + flow_noise + pump_noise + hf_noise
    
    return t, pressure

def calculate_attenuation(distance, frequency):
    """
    Calculate acoustic attenuation in stratified flow.
    Attenuation is higher due to:
    - Interface scattering
    - Viscous absorption
    - Heat conduction losses
    """
    # Frequency-dependent attenuation (higher frequencies attenuate faster)
    alpha_freq = ATTENUATION_COEFF * (frequency / 1000) ** 0.5
    
    # Interface scattering contribution (stratified flow specific)
    alpha_interface = 0.02 * np.log10(frequency + 1)
    
    total_alpha = alpha_freq + alpha_interface
    attenuation = np.exp(-total_alpha * distance)
    
    return attenuation

def generate_leak_acoustic_signature(duration, fs, leak_start, leak_end, leak_intensity=1.0):
    """
    Generate the acoustic signature of a leak event.
    Leak acoustics include:
    - Broadband noise from turbulent jet
    - Characteristic frequencies from orifice resonance
    - Transient onset and decay
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Create leak envelope (gradual onset, sustained, gradual decay)
    envelope = np.zeros(n_samples)
    onset_samples = int(0.5 * fs)  # 0.5s onset
    decay_samples = int(1.0 * fs)  # 1s decay
    
    leak_start_idx = int(leak_start * fs)
    leak_end_idx = int(leak_end * fs)
    
    for i in range(n_samples):
        if i < leak_start_idx:
            envelope[i] = 0
        elif i < leak_start_idx + onset_samples:
            # Exponential onset
            progress = (i - leak_start_idx) / onset_samples
            envelope[i] = 1 - np.exp(-5 * progress)
        elif i < leak_end_idx:
            envelope[i] = 1.0
        elif i < leak_end_idx + decay_samples:
            # Exponential decay
            progress = (i - leak_end_idx) / decay_samples
            envelope[i] = np.exp(-3 * progress)
        else:
            envelope[i] = 0
    
    # Generate leak acoustic components
    leak_signal = np.zeros(n_samples)
    
    # Broadband turbulent noise (main leak signature)
    broadband = np.random.randn(n_samples)
    # Apply bandpass characteristic (leak noise is typically 100-5000 Hz)
    from scipy.signal import butter, filtfilt
    try:
        b, a = butter(4, [100/(fs/2), min(4000/(fs/2), 0.99)], btype='band')
        broadband = filtfilt(b, a, broadband)
    except:
        pass
    leak_signal += 5.0 * broadband
    
    # Orifice resonance frequencies (dependent on leak size)
    resonance_freq = 800 + np.random.uniform(-100, 100)  # Hz
    leak_signal += 2.0 * np.sin(2 * np.pi * resonance_freq * t)
    leak_signal += 1.0 * np.sin(2 * np.pi * 2 * resonance_freq * t)
    
    # Harmonics from jet instabilities
    jet_freq = 1500 + np.random.uniform(-200, 200)
    leak_signal += 0.8 * np.sin(2 * np.pi * jet_freq * t + np.random.randn(n_samples) * 0.1)
    
    # Apply envelope and intensity
    leak_signal = leak_signal * envelope * leak_intensity
    
    return leak_signal

def generate_sensor_data(duration, fs, sensor_pos, leak_pos, leak_start, leak_end, 
                         phase='transient', include_leak=True):
    """
    Generate complete sensor data including baseline and leak signatures.
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    
    # Generate baseline
    _, baseline = generate_baseline_noise(duration, fs, None)
    
    if not include_leak or phase == 'baseline':
        return t, baseline
    
    # Calculate distance from leak
    distance = abs(sensor_pos - leak_pos)
    
    # Generate leak signature at source
    leak_source = generate_leak_acoustic_signature(duration, fs, leak_start, leak_end)
    
    # Apply distance-dependent attenuation
    # Calculate dominant frequency for attenuation (assume ~1000 Hz)
    dominant_freq = 1000
    attenuation = calculate_attenuation(distance, dominant_freq)
    
    # Apply time delay based on wave propagation
    delay_samples = int((distance / EFFECTIVE_SOUND_SPEED) * fs)
    
    # Shift and attenuate leak signal
    leak_attenuated = np.zeros(n_samples)
    if delay_samples < n_samples:
        leak_attenuated[delay_samples:] = leak_source[:-delay_samples] * attenuation if delay_samples > 0 else leak_source * attenuation
    
    # Combine baseline and leak
    total_signal = baseline + leak_attenuated
    
    # Add frequency-dependent dispersion (stratified flow effect)
    # Higher frequencies travel slightly faster in stratified media
    dispersion_noise = 0.05 * np.random.randn(n_samples) * (1 - attenuation)
    total_signal += dispersion_noise
    
    return t, total_signal

# ============================================================================
# DATA GENERATION FOR ALL TEST GROUPS
# ============================================================================

def generate_test_group_data(group_id, config):
    """Generate all data for a specific test group."""
    
    print(f"\nGenerating data for Test Group {group_id}: {config['description']}")
    
    fs = config['sampling_rate']
    leak_loc_name = config['leak_location']
    leak_pos = LEAK_LOCATIONS[leak_loc_name]
    
    group_data = {}
    
    # Generate data for each phase
    phases = {
        'baseline': {'duration': BASELINE_DURATION, 'include_leak': False, 'leak_start': 0, 'leak_end': 0},
        'transient': {'duration': TRANSIENT_DURATION, 'include_leak': True, 'leak_start': LEAK_START, 'leak_end': LEAK_END},
        'post_event': {'duration': POST_EVENT_DURATION, 'include_leak': False, 'leak_start': 0, 'leak_end': 0},
    }
    
    for phase_name, phase_config in phases.items():
        print(f"  Generating {phase_name} phase...")
        
        phase_data = {'time': None}
        
        for sensor in SENSORS:
            sensor_pos = SENSOR_POSITIONS[sensor]
            
            t, pressure = generate_sensor_data(
                duration=phase_config['duration'],
                fs=fs,
                sensor_pos=sensor_pos,
                leak_pos=leak_pos,
                leak_start=phase_config['leak_start'],
                leak_end=phase_config['leak_end'],
                phase=phase_name,
                include_leak=phase_config['include_leak']
            )
            
            if phase_data['time'] is None:
                phase_data['time'] = t
            
            phase_data[sensor] = pressure
        
        group_data[phase_name] = phase_data
    
    return group_data

def save_data_to_csv(all_data, output_dir):
    """Save all generated data to CSV files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = []
    
    for group_id, group_data in all_data.items():
        group_dir = os.path.join(output_dir, f"Group_{group_id}")
        os.makedirs(group_dir, exist_ok=True)
        
        for phase_name, phase_data in group_data.items():
            # Create DataFrame
            df = pd.DataFrame(phase_data)
            
            # Reorder columns
            cols = ['time'] + SENSORS
            df = df[cols]
            
            # Save to CSV
            filename = f"Group{group_id}_{phase_name}_pressure_data.csv"
            filepath = os.path.join(group_dir, filename)
            df.to_csv(filepath, index=False, float_format='%.6f')
            csv_files.append(filepath)
            
            print(f"  Saved: {filename} ({len(df)} samples)")
    
    # Create metadata file
    metadata = {
        'Study': 'Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics',
        'Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Sensors': SENSORS,
        'Sensor_Positions_m': SENSOR_POSITIONS,
        'Leak_Locations_m': LEAK_LOCATIONS,
        'Test_Groups': TEST_GROUPS,
        'Sampling_Rates_Hz': {'standard': SAMPLING_RATE_STANDARD, 'high': SAMPLING_RATE_HIGH},
        'Physical_Parameters': {
            'Speed_of_Sound_Liquid_m_s': SPEED_OF_SOUND_LIQUID,
            'Speed_of_Sound_Gas_m_s': SPEED_OF_SOUND_GAS,
            'Effective_Speed_m_s': EFFECTIVE_SOUND_SPEED,
            'Attenuation_Coeff_Np_m': ATTENUATION_COEFF,
        }
    }
    
    import json
    metadata_path = os.path.join(output_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    csv_files.append(metadata_path)
    
    # Create sensor positions reference CSV
    sensor_df = pd.DataFrame([
        {'Sensor_ID': k, 'Position_m': v, 'Description': get_sensor_description(k)}
        for k, v in SENSOR_POSITIONS.items()
    ])
    sensor_path = os.path.join(output_dir, 'sensor_positions.csv')
    sensor_df.to_csv(sensor_path, index=False)
    csv_files.append(sensor_path)
    
    return csv_files

def get_sensor_description(sensor_id):
    """Get description for sensor placement."""
    descriptions = {
        'PG1': 'Before Valve 1 (Inlet)',
        'PG2': 'After Valve 1',
        'PG3': 'Before Leak Point A',
        'PG4': 'After Leak Point A',
        'PG5': 'Mid-section 1',
        'PG6': 'Before Leak Point D',
        'PG7': 'After Leak Point D',
        'PG8': 'Mid-section 2',
        'PG9': 'Before Valve 2',
        'PG10': 'After Valve 2',
        'PG11': 'Before Leak Point E',
        'PG12': 'After Leak Point E',
        'PG13': 'Near Outlet',
        'PG14': 'At Outlet',
    }
    return descriptions.get(sensor_id, 'Unknown')

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(all_data, output_dir):
    """Create comprehensive visualizations of the generated data."""
    
    os.makedirs(output_dir, exist_ok=True)
    figure_files = []
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    
    # 1. Overview figure for each test group
    for group_id, group_data in all_data.items():
        fig = create_group_overview(group_id, group_data)
        filepath = os.path.join(output_dir, f'Group_{group_id}_overview.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        figure_files.append(filepath)
        print(f"  Saved: Group_{group_id}_overview.png")
    
    # 2. Attenuation analysis figure
    fig = create_attenuation_analysis(all_data)
    filepath = os.path.join(output_dir, 'attenuation_analysis.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    figure_files.append(filepath)
    print(f"  Saved: attenuation_analysis.png")
    
    # 3. Sensor array configuration
    fig = create_sensor_layout()
    filepath = os.path.join(output_dir, 'sensor_layout.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    figure_files.append(filepath)
    print(f"  Saved: sensor_layout.png")
    
    # 4. Frequency analysis (spectrogram)
    fig = create_spectrogram_analysis(all_data)
    filepath = os.path.join(output_dir, 'frequency_analysis.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    figure_files.append(filepath)
    print(f"  Saved: frequency_analysis.png")
    
    # 5. Comparative leak signatures
    fig = create_leak_comparison(all_data)
    filepath = os.path.join(output_dir, 'leak_signature_comparison.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    figure_files.append(filepath)
    print(f"  Saved: leak_signature_comparison.png")
    
    # 6. Phase comparison (baseline vs transient vs post-event)
    fig = create_phase_comparison(all_data)
    filepath = os.path.join(output_dir, 'phase_comparison.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    figure_files.append(filepath)
    print(f"  Saved: phase_comparison.png")
    
    return figure_files

def create_group_overview(group_id, group_data):
    """Create overview figure for a test group."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f'Test Group {group_id}: {TEST_GROUPS[group_id]["description"]}\n'
                 f'Leak Location: {TEST_GROUPS[group_id]["leak_location"]} | '
                 f'Sampling Rate: {TEST_GROUPS[group_id]["sampling_rate"]} Hz',
                 fontsize=12, fontweight='bold')
    
    phases = ['baseline', 'transient', 'post_event']
    phase_titles = ['Baseline (Normal Operation)', 'Transient (Leak Event)', 'Post-Event (Recovery)']
    
    # Select representative sensors
    sensors_to_plot = ['PG3', 'PG6', 'PG11']  # Near each leak point
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sensors_to_plot)))
    
    for idx, (phase, title) in enumerate(zip(phases, phase_titles)):
        ax = axes[idx]
        data = group_data[phase]
        t = data['time']
        
        for sensor, color in zip(sensors_to_plot, colors):
            # Downsample for plotting if needed
            step = max(1, len(t) // 5000)
            ax.plot(t[::step], data[sensor][::step], label=sensor, color=color, alpha=0.8, linewidth=0.5)
        
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Pressure (kPa)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if phase == 'transient':
            ax.axvline(x=LEAK_START, color='red', linestyle='--', alpha=0.7, label='Leak Start')
            ax.axvline(x=LEAK_END, color='green', linestyle='--', alpha=0.7, label='Leak End')
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    
    return fig

def create_attenuation_analysis(all_data):
    """Create attenuation analysis visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Acoustic Wave Attenuation Analysis in Stratified Flow', fontsize=12, fontweight='bold')
    
    # For each test group, analyze attenuation
    for idx, (group_id, config) in enumerate(TEST_GROUPS.items()):
        ax = axes[idx // 2, idx % 2]
        
        leak_loc = config['leak_location']
        leak_pos = LEAK_LOCATIONS[leak_loc]
        
        # Get transient data
        transient_data = all_data[group_id]['transient']
        
        # Calculate RMS amplitude during leak for each sensor
        fs = config['sampling_rate']
        leak_start_idx = int(LEAK_START * fs)
        leak_end_idx = int(LEAK_END * fs)
        
        distances = []
        amplitudes = []
        sensor_labels = []
        
        for sensor in SENSORS:
            sensor_pos = SENSOR_POSITIONS[sensor]
            distance = abs(sensor_pos - leak_pos)
            
            # Calculate RMS during leak
            signal = transient_data[sensor][leak_start_idx:leak_end_idx]
            rms = np.sqrt(np.mean(signal**2))
            
            distances.append(distance)
            amplitudes.append(rms)
            sensor_labels.append(sensor)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        distances = np.array(distances)[sorted_indices]
        amplitudes = np.array(amplitudes)[sorted_indices]
        
        # Normalize amplitudes
        amplitudes_norm = amplitudes / amplitudes.max()
        
        # Plot
        ax.scatter(distances, amplitudes_norm, c='blue', s=50, alpha=0.7, label='Measured')
        
        # Fit exponential decay
        from scipy.optimize import curve_fit
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            popt, _ = curve_fit(exp_decay, distances, amplitudes_norm, p0=[1, 0.1, 0.1], maxfev=5000)
            x_fit = np.linspace(0, distances.max(), 100)
            y_fit = exp_decay(x_fit, *popt)
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: α={popt[1]:.3f} Np/m')
        except:
            pass
        
        ax.set_xlabel('Distance from Leak (m)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_title(f'Group {group_id} - Leak at Point {leak_loc}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_sensor_layout():
    """Create sensor array layout visualization."""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Draw pipeline
    pipeline_length = 40
    ax.fill_between([0, pipeline_length], [-0.5, -0.5], [0.5, 0.5], 
                    color='lightgray', alpha=0.5, edgecolor='black', linewidth=2)
    
    # Draw stratified flow layers
    ax.fill_between([0, pipeline_length], [-0.5, -0.5], [0, 0], 
                    color='lightblue', alpha=0.5, label='Liquid Phase')
    ax.fill_between([0, pipeline_length], [0, 0], [0.5, 0.5], 
                    color='lightyellow', alpha=0.5, label='Gas Phase')
    
    # Plot sensors
    for sensor, pos in SENSOR_POSITIONS.items():
        ax.plot(pos, 0.7, 'v', markersize=12, color='red', markeredgecolor='black')
        ax.annotate(sensor, (pos, 0.9), ha='center', fontsize=8, rotation=45)
    
    # Plot leak locations
    for name, pos in LEAK_LOCATIONS.items():
        ax.plot(pos, -0.7, '^', markersize=15, color='orange', markeredgecolor='black')
        ax.annotate(f'Leak {name}', (pos, -1.0), ha='center', fontsize=9, fontweight='bold')
    
    # Add valve positions (assumed)
    valve_positions = [1.0, 26.0]
    for vpos in valve_positions:
        ax.fill_between([vpos-0.3, vpos+0.3], [-0.6, -0.6], [0.6, 0.6], 
                        color='darkgray', alpha=0.8)
        ax.annotate('Valve', (vpos, 1.1), ha='center', fontsize=8)
    
    ax.set_xlim(-1, pipeline_length + 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Position along pipeline (m)', fontsize=11)
    ax.set_title('Sensor Array Configuration and Leak Point Locations\n'
                 'Stratified Flow Test Setup', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Flow Direction →', xy=(20, -1.3), fontsize=10, ha='center')
    
    return fig

def create_spectrogram_analysis(all_data):
    """Create spectrogram analysis for frequency content."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Time-Frequency Analysis (Spectrograms) During Leak Events', fontsize=12, fontweight='bold')
    
    for idx, (group_id, config) in enumerate(TEST_GROUPS.items()):
        ax = axes[idx // 2, idx % 2]
        
        fs = config['sampling_rate']
        leak_loc = config['leak_location']
        
        # Get sensor closest to leak
        leak_pos = LEAK_LOCATIONS[leak_loc]
        closest_sensor = min(SENSORS, key=lambda s: abs(SENSOR_POSITIONS[s] - leak_pos))
        
        # Get transient data
        signal = all_data[group_id]['transient'][closest_sensor]
        
        # Compute spectrogram
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(signal, fs, nperseg=1024, noverlap=512)
        
        # Plot
        pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Group {group_id} - Sensor {closest_sensor} (near Leak {leak_loc})')
        ax.set_ylim(0, min(fs/2, 5000))
        
        # Mark leak event
        ax.axvline(x=LEAK_START, color='red', linestyle='--', alpha=0.8)
        ax.axvline(x=LEAK_END, color='green', linestyle='--', alpha=0.8)
        
        plt.colorbar(pcm, ax=ax, label='Power (dB)')
    
    plt.tight_layout()
    return fig

def create_leak_comparison(all_data):
    """Compare leak signatures across different locations."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Comparison of Leak Acoustic Signatures at Different Locations', 
                 fontsize=12, fontweight='bold')
    
    # Groups with different leak locations
    groups_by_leak = {
        'A': '02',  # Leak A with full sensors
        'D': '03',  # Leak D
        'E': '04',  # Leak E
    }
    
    for idx, (leak_loc, group_id) in enumerate(groups_by_leak.items()):
        ax = axes[idx]
        
        config = TEST_GROUPS[group_id]
        fs = config['sampling_rate']
        leak_pos = LEAK_LOCATIONS[leak_loc]
        
        # Get sensor closest to leak
        closest_sensor = min(SENSORS, key=lambda s: abs(SENSOR_POSITIONS[s] - leak_pos))
        
        # Get transient data
        data = all_data[group_id]['transient']
        t = data['time']
        signal = data[closest_sensor]
        
        # Downsample for plotting
        step = max(1, len(t) // 10000)
        ax.plot(t[::step], signal[::step], color='blue', linewidth=0.5, alpha=0.8)
        
        ax.axvline(x=LEAK_START, color='red', linestyle='--', alpha=0.7, label='Valve Open')
        ax.axvline(x=LEAK_END, color='green', linestyle='--', alpha=0.7, label='Valve Close')
        ax.axvspan(LEAK_START, LEAK_END, alpha=0.1, color='red')
        
        ax.set_ylabel('Pressure (kPa)')
        ax.set_title(f'Leak Location {leak_loc} (Position: {leak_pos}m) - Sensor {closest_sensor}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig

def create_phase_comparison(all_data):
    """Compare baseline, transient, and post-event phases."""
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('Phase Comparison: Baseline vs Transient vs Post-Event\n'
                 'RMS Pressure Amplitude Across Sensor Array', fontsize=12, fontweight='bold')
    
    phases = ['baseline', 'transient', 'post_event']
    phase_colors = ['green', 'red', 'blue']
    
    for col, (group_id, config) in enumerate(TEST_GROUPS.items()):
        fs = config['sampling_rate']
        leak_loc = config['leak_location']
        leak_pos = LEAK_LOCATIONS[leak_loc]
        
        for row, (phase, color) in enumerate(zip(phases, phase_colors)):
            ax = axes[row, col]
            
            data = all_data[group_id][phase]
            
            # Calculate RMS for each sensor
            rms_values = []
            positions = []
            
            for sensor in SENSORS:
                signal = data[sensor]
                
                if phase == 'transient':
                    # Use only leak period
                    start_idx = int(LEAK_START * fs)
                    end_idx = int(LEAK_END * fs)
                    signal = signal[start_idx:end_idx]
                
                rms = np.sqrt(np.mean(signal**2))
                rms_values.append(rms)
                positions.append(SENSOR_POSITIONS[sensor])
            
            ax.bar(range(len(SENSORS)), rms_values, color=color, alpha=0.7)
            ax.axvline(x=list(SENSOR_POSITIONS.values()).index(
                min(SENSOR_POSITIONS.values(), key=lambda x: abs(x - leak_pos))), 
                color='orange', linestyle='--', linewidth=2, label='Nearest to Leak')
            
            ax.set_xticks(range(len(SENSORS)))
            ax.set_xticklabels(SENSORS, rotation=45, fontsize=7)
            
            if col == 0:
                ax.set_ylabel(f'{phase.replace("_", " ").title()}\nRMS (kPa)')
            if row == 0:
                ax.set_title(f'Group {group_id}\n(Leak {leak_loc})')
            if row == 2:
                ax.set_xlabel('Sensor')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_zip_archive(csv_dir, output_path):
    """Create a ZIP archive of all CSV files."""
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(csv_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, csv_dir)
                zipf.write(file_path, arcname)
    
    print(f"\nZIP archive created: {output_path}")
    return output_path

def main():
    """Main function to generate all data and visualizations."""
    
    print("=" * 70)
    print("CORE TIME-SERIES PRESSURE DATA GENERATOR")
    print("Study: Attenuation Mechanisms in Stratified Flows")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Generate data for all test groups
    print("\n[1/4] Generating synthetic pressure data...")
    all_data = {}
    for group_id, config in TEST_GROUPS.items():
        all_data[group_id] = generate_test_group_data(group_id, config)
    
    # Save to CSV files
    print("\n[2/4] Saving data to CSV files...")
    csv_files = save_data_to_csv(all_data, CSV_DIR)
    
    # Create visualizations
    print("\n[3/4] Creating visualizations...")
    figure_files = create_visualizations(all_data, FIGURES_DIR)
    
    # Create ZIP archive
    print("\n[4/4] Creating ZIP archive...")
    zip_path = os.path.join(OUTPUT_DIR, "pressure_data_csv.zip")
    create_zip_archive(CSV_DIR, zip_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"CSV files: {len(csv_files)} files in {CSV_DIR}")
    print(f"Figure files: {len(figure_files)} files in {FIGURES_DIR}")
    print(f"ZIP archive: {zip_path}")
    print("\nGenerated test groups:")
    for gid, cfg in TEST_GROUPS.items():
        print(f"  - Group {gid}: {cfg['description']}")
    print("\nDataset includes:")
    print(f"  - {len(SENSORS)} pressure sensors (PG1-PG14)")
    print(f"  - 3 leak locations (A, D, E)")
    print(f"  - 3 phases per group (baseline, transient, post_event)")
    print(f"  - Sampling rates: 10 kHz (standard), 17.06 kHz (high-rate)")

if __name__ == "__main__":
    main()
