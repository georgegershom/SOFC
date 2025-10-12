#!/usr/bin/env python3
"""
Generate synthetic acoustic time-series data for stratified flow experiments
Based on realistic acoustic propagation through two-phase flows
"""

import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt

def generate_acoustic_timeseries(experiment_id, source_freq, source_amplitude, 
                                received_amplitude, snr_db, phase_shift, 
                                sampling_rate=10000, duration=30):
    """
    Generate realistic acoustic time-series data with attenuation and noise
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate source signal (pure tone with slight amplitude modulation)
    source_signal = source_amplitude * np.sin(2 * np.pi * source_freq * t) * \
                   (1 + 0.05 * np.sin(2 * np.pi * 0.1 * t))  # 0.1 Hz modulation
    
    # Generate received signal with attenuation and phase shift
    received_signal = received_amplitude * np.sin(2 * np.pi * source_freq * t + 
                                                 np.radians(phase_shift)) * \
                     (1 + 0.05 * np.sin(2 * np.pi * 0.1 * t))
    
    # Add realistic noise based on SNR
    noise_power = received_amplitude**2 / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    received_signal_noisy = received_signal + noise
    
    # Add some flow-induced turbulence noise (low frequency)
    turbulence_raw = 0.02 * received_amplitude * np.random.normal(0, 1, len(t))
    sos = signal.butter(2, 10, 'low', fs=sampling_rate, output='sos')
    turbulence_noise = signal.sosfilt(sos, turbulence_raw)
    
    received_signal_noisy += turbulence_noise
    
    return t, source_signal, received_signal_noisy

def main():
    # Create output directory
    os.makedirs('../acoustic_signals/timeseries_data', exist_ok=True)
    
    # Generate time-series for first few experiments
    experiments = [
        {'id': 'EXP001', 'freq': 100, 'src_amp': 1000, 'rec_amp': 635.8, 'snr': 22.7, 'phase': 52.6},
        {'id': 'EXP001', 'freq': 500, 'src_amp': 1000, 'rec_amp': 512.3, 'snr': 17.8, 'phase': 92.1},
        {'id': 'EXP001', 'freq': 1000, 'src_amp': 1000, 'rec_amp': 406.2, 'snr': 13.8, 'phase': 121.8},
        {'id': 'EXP003', 'freq': 100, 'src_amp': 1000, 'rec_amp': 577.4, 'snr': 19.2, 'phase': 63.7},
        {'id': 'EXP003', 'freq': 500, 'src_amp': 1000, 'rec_amp': 460.2, 'snr': 14.7, 'phase': 105.8},
        {'id': 'EXP005', 'freq': 100, 'src_amp': 1000, 'rec_amp': 519.1, 'snr': 15.8, 'phase': 76.2},
        {'id': 'EXP005', 'freq': 1000, 'src_amp': 1000, 'rec_amp': 327.4, 'snr': 9.8, 'phase': 138.7},
    ]
    
    for exp in experiments:
        print(f"Generating timeseries for {exp['id']} at {exp['freq']} Hz...")
        
        t, source, received = generate_acoustic_timeseries(
            exp['id'], exp['freq'], exp['src_amp'], exp['rec_amp'], 
            exp['snr'], exp['phase']
        )
        
        # Save as CSV
        filename = f"../acoustic_signals/timeseries_data/{exp['id']}_f{exp['freq']}Hz_timeseries.csv"
        df = pd.DataFrame({
            'time_s': t,
            'source_signal_pa': source,
            'received_signal_pa': received
        })
        df.to_csv(filename, index=False)
        
        # Generate and save frequency spectrum
        freqs, source_psd = signal.welch(source, fs=10000, nperseg=1024)
        freqs, received_psd = signal.welch(received, fs=10000, nperseg=1024)
        
        spectrum_filename = f"../acoustic_signals/timeseries_data/{exp['id']}_f{exp['freq']}Hz_spectrum.csv"
        spectrum_df = pd.DataFrame({
            'frequency_hz': freqs,
            'source_psd_pa2_hz': source_psd,
            'received_psd_pa2_hz': received_psd
        })
        spectrum_df.to_csv(spectrum_filename, index=False)
    
    print("Acoustic time-series data generation completed!")

if __name__ == "__main__":
    main()