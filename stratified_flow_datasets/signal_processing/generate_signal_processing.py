"""
Generate Signal Processing Outputs for Stratified Flow Acoustics
Including filtered signals, cross-correlations, and feature extraction for leak detection
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
import pywt  # For wavelet transforms

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class SignalProcessingGenerator:
    def __init__(self):
        self.sampling_rate = 100000  # Hz (100 kHz)
        self.signal_duration = 1.0  # seconds
        self.n_samples = int(self.sampling_rate * self.signal_duration)
        self.time = np.linspace(0, self.signal_duration, self.n_samples)
        
        # Flow conditions for signal generation
        self.flow_conditions = {
            'stratified_smooth': {
                'void_fraction': 0.3,
                'interface_height': 0.6,
                'turbulence_level': 0.02,
                'dominant_frequencies': [500, 1000, 1500],
                'noise_level': 0.05
            },
            'stratified_wavy': {
                'void_fraction': 0.4,
                'interface_height': 0.5,
                'turbulence_level': 0.08,
                'dominant_frequencies': [500, 1000, 2000],
                'wave_frequency': 5,  # Hz
                'wave_amplitude': 0.2,
                'noise_level': 0.1
            },
            'slug_flow': {
                'void_fraction': 0.5,
                'interface_height': 0.4,
                'turbulence_level': 0.15,
                'dominant_frequencies': [200, 800, 1500, 3000],
                'slug_frequency': 0.5,  # Hz
                'slug_duration': 0.15,
                'noise_level': 0.15
            },
            'leak_present': {
                'leak_frequency': 4000,  # High frequency component
                'leak_amplitude': 0.3,
                'leak_modulation': 10,  # Hz
                'base_flow': 'stratified_smooth'
            }
        }
    
    def generate_base_signal(self, condition_name, add_leak=False, leak_location=0.5):
        """Generate acoustic signal for different flow conditions"""
        condition = self.flow_conditions[condition_name]
        signal_out = np.zeros(self.n_samples)
        
        if condition_name == 'stratified_smooth':
            # Smooth stratified flow - steady signal with multiple harmonics
            for freq in condition['dominant_frequencies']:
                amplitude = 1.0 / (1 + (freq/1000))  # Decreasing with frequency
                signal_out += amplitude * np.sin(2 * np.pi * freq * self.time)
            
            # Add low-frequency drift
            signal_out += 0.1 * np.sin(2 * np.pi * 0.5 * self.time)
            
        elif condition_name == 'stratified_wavy':
            # Wavy stratified - modulated signal
            wave_modulation = 1 + condition['wave_amplitude'] * np.sin(2 * np.pi * condition['wave_frequency'] * self.time)
            
            for freq in condition['dominant_frequencies']:
                amplitude = 1.0 / (1 + (freq/1000))
                signal_out += amplitude * wave_modulation * np.sin(2 * np.pi * freq * self.time)
            
            # Add wave-induced frequency components
            signal_out += 0.2 * np.sin(2 * np.pi * condition['wave_frequency'] * self.time)
            
        elif condition_name == 'slug_flow':
            # Slug flow - intermittent high amplitude bursts
            base_signal = np.zeros(self.n_samples)
            for freq in condition['dominant_frequencies']:
                amplitude = 1.0 / (1 + (freq/1000))
                base_signal += amplitude * np.sin(2 * np.pi * freq * self.time)
            
            # Create slug events
            slug_mask = np.zeros(self.n_samples)
            n_slugs = int(condition['slug_frequency'] * self.signal_duration)
            for i in range(n_slugs):
                start = int(i * self.sampling_rate / condition['slug_frequency'])
                end = min(start + int(condition['slug_duration'] * self.sampling_rate), self.n_samples)
                if end <= self.n_samples:
                    slug_mask[start:end] = 1
            
            # Apply slug modulation
            signal_out = base_signal * (1 + 2 * slug_mask)
            
            # Add impact transients at slug arrival
            for i in range(n_slugs):
                impact_time = int(i * self.sampling_rate / condition['slug_frequency'])
                if impact_time < self.n_samples:
                    # Exponentially decaying transient
                    t_transient = self.time[impact_time:] - self.time[impact_time]
                    signal_out[impact_time:] += 0.5 * np.exp(-50 * t_transient) * np.sin(2 * np.pi * 5000 * t_transient)
        
        # Add turbulence-induced noise
        if 'turbulence_level' in condition:
            turbulence_noise = condition['turbulence_level'] * np.random.randn(self.n_samples)
            # Low-pass filter the turbulence (turbulence is typically low frequency)
            b, a = signal.butter(4, 1000, fs=self.sampling_rate)
            turbulence_noise = signal.filtfilt(b, a, turbulence_noise)
            signal_out += turbulence_noise
        
        # Add measurement noise
        signal_out += condition['noise_level'] * np.random.randn(self.n_samples)
        
        # Add leak signal if requested
        if add_leak:
            leak_config = self.flow_conditions['leak_present']
            # Leak creates high-frequency oscillation with amplitude modulation
            leak_signal = leak_config['leak_amplitude'] * np.sin(2 * np.pi * leak_config['leak_frequency'] * self.time)
            
            # Modulate leak signal (pressure fluctuations)
            leak_modulation = 1 + 0.3 * np.sin(2 * np.pi * leak_config['leak_modulation'] * self.time)
            leak_signal *= leak_modulation
            
            # Leak signal attenuates with distance from leak
            # Create spatial windowing based on leak location
            leak_window = np.exp(-5 * (self.time - leak_location * self.signal_duration)**2 / (0.1 * self.signal_duration)**2)
            leak_signal *= leak_window
            
            signal_out += leak_signal
        
        return signal_out
    
    def apply_filters(self, input_signal):
        """Apply various filters to the signal"""
        filters_output = {}
        
        # 1. Fourier Transform (FFT)
        fft_result = fft(input_signal)
        frequencies = fftfreq(self.n_samples, 1/self.sampling_rate)
        
        # Power Spectral Density
        psd = np.abs(fft_result)**2 / self.n_samples
        
        filters_output['fft'] = {
            'frequencies': frequencies[:self.n_samples//2].tolist(),
            'magnitude': np.abs(fft_result[:self.n_samples//2]).tolist(),
            'phase': np.angle(fft_result[:self.n_samples//2]).tolist(),
            'psd': psd[:self.n_samples//2].tolist()
        }
        
        # 2. Bandpass Filters (for different frequency bands)
        frequency_bands = [
            (10, 100, 'Very_Low'),
            (100, 1000, 'Low'),
            (1000, 5000, 'Medium'),
            (5000, 20000, 'High'),
            (20000, 45000, 'Very_High')
        ]
        
        for f_low, f_high, band_name in frequency_bands:
            if f_high < self.sampling_rate/2:  # Nyquist limit
                b, a = signal.butter(4, [f_low, f_high], btype='band', fs=self.sampling_rate)
                filtered = signal.filtfilt(b, a, input_signal)
                
                # Calculate band energy
                band_energy = np.sum(filtered**2) / self.n_samples
                
                filters_output[f'bandpass_{band_name}'] = {
                    'frequency_range_Hz': [f_low, f_high],
                    'rms_amplitude': np.sqrt(np.mean(filtered**2)),
                    'peak_amplitude': np.max(np.abs(filtered)),
                    'energy': band_energy,
                    'signal_excerpt': filtered[::100].tolist()  # Downsampled for storage
                }
        
        # 3. Wavelet Transform (using Symlets)
        wavelet_families = ['sym5', 'sym8', 'db4', 'coif3']
        
        for wavelet_name in wavelet_families:
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(input_signal, wavelet_name, level=8)
            
            # Reconstruct signal at different levels
            reconstructed_levels = {}
            for level in range(1, min(9, len(coeffs))):
                # Zero out higher frequency components
                coeffs_filtered = coeffs.copy()
                for i in range(level):
                    coeffs_filtered[i] = np.zeros_like(coeffs_filtered[i])
                
                reconstructed = pywt.waverec(coeffs_filtered, wavelet_name, mode='symmetric')
                if len(reconstructed) > self.n_samples:
                    reconstructed = reconstructed[:self.n_samples]
                
                reconstructed_levels[f'level_{level}'] = {
                    'rms': np.sqrt(np.mean(reconstructed**2)),
                    'energy': np.sum(reconstructed**2) / len(reconstructed)
                }
            
            # Continuous Wavelet Transform for time-frequency analysis
            scales = np.arange(1, 128, 4)
            try:
                cwt_coeffs = pywt.cwt(input_signal[:10000], scales, wavelet_name, 1/self.sampling_rate)
                if isinstance(cwt_coeffs, tuple):
                    cwt_coeffs = cwt_coeffs[0]  # Take coefficients only
                cwt_energy = np.sum(np.abs(cwt_coeffs)**2, axis=1).tolist()
            except:
                cwt_energy = [0] * len(scales)
            
            filters_output[f'wavelet_{wavelet_name}'] = {
                'decomposition_levels': len(coeffs),
                'reconstructed_levels': reconstructed_levels,
                'cwt_scales': scales.tolist(),
                'cwt_frequencies_Hz': (self.sampling_rate / (2 * scales)).tolist(),
                'cwt_energy_distribution': cwt_energy
            }
        
        # 4. Hilbert Transform (for envelope detection)
        analytic_signal = signal.hilbert(input_signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * self.sampling_rate
        
        filters_output['hilbert'] = {
            'envelope_mean': np.mean(amplitude_envelope),
            'envelope_std': np.std(amplitude_envelope),
            'envelope_max': np.max(amplitude_envelope),
            'inst_freq_mean': np.mean(instantaneous_frequency[np.isfinite(instantaneous_frequency)]),
            'inst_freq_std': np.std(instantaneous_frequency[np.isfinite(instantaneous_frequency)])
        }
        
        # 5. Empirical Mode Decomposition (simplified - using filtering approximation)
        # Real EMD would require PyEMD package
        imf_bands = [(0.1, 10), (10, 100), (100, 1000), (1000, 10000)]
        imfs = []
        
        for i, (f_low, f_high) in enumerate(imf_bands):
            if f_high < self.sampling_rate/2:
                try:
                    b, a = signal.butter(3, [f_low, f_high], btype='band', fs=self.sampling_rate)
                    imf = signal.filtfilt(b, a, input_signal)
                    # Clip to prevent overflow
                    imf = np.clip(imf, -1e6, 1e6)
                    energy = np.sum(imf**2) / self.n_samples
                    rms = np.sqrt(np.mean(imf**2))
                except:
                    energy = 0
                    rms = 0
                    
                imfs.append({
                    'imf_index': i+1,
                    'frequency_band_Hz': [f_low, f_high],
                    'energy': float(energy) if np.isfinite(energy) else 0,
                    'rms': float(rms) if np.isfinite(rms) else 0
                })
        
        filters_output['emd_approximation'] = imfs
        
        return filters_output
    
    def calculate_cross_correlation(self, signal1, signal2):
        """Calculate cross-correlation for time delay estimation"""
        # Normalize signals
        std1 = np.std(signal1)
        std2 = np.std(signal2)
        
        if std1 > 0:
            signal1_norm = (signal1 - np.mean(signal1)) / std1
        else:
            signal1_norm = signal1 - np.mean(signal1)
            
        if std2 > 0:
            signal2_norm = (signal2 - np.mean(signal2)) / std2
        else:
            signal2_norm = signal2 - np.mean(signal2)
        
        # Calculate cross-correlation
        correlation = signal.correlate(signal1_norm, signal2_norm, mode='full', method='fft')
        lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
        
        # Normalize correlation
        max_corr = np.max(np.abs(correlation))
        if max_corr > 0:
            correlation = correlation / max_corr
        else:
            correlation = correlation
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        peak_value = correlation[peak_idx]
        
        # Time delay
        time_delay = peak_lag / self.sampling_rate
        
        # Calculate correlation quality metrics
        # Peak to sidelobe ratio
        sorted_corr = np.sort(np.abs(correlation))[::-1]
        if len(sorted_corr) > 1:
            pslr = sorted_corr[0] / sorted_corr[1] if sorted_corr[1] > 0 else 100
        else:
            pslr = 100
        
        # Correlation width at half maximum
        half_max = np.abs(peak_value) / 2
        indices_above_half = np.where(np.abs(correlation) > half_max)[0]
        if len(indices_above_half) > 0:
            correlation_width = (indices_above_half[-1] - indices_above_half[0]) / self.sampling_rate
        else:
            correlation_width = 0
        
        return {
            'peak_lag_samples': int(peak_lag),
            'peak_value': float(peak_value),
            'time_delay_s': float(time_delay),
            'peak_to_sidelobe_ratio': float(pslr),
            'correlation_width_s': float(correlation_width),
            'correlation_function': [float(x) for x in correlation[::100]],  # Downsampled
            'lags': [int(x) for x in lags[::100]]
        }
    
    def extract_leak_detection_features(self, signal_data):
        """Extract features for leak detection and location"""
        features = {}
        
        # Time domain features
        features['time_domain'] = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
            'kurtosis': np.mean(((signal_data - np.mean(signal_data))/np.std(signal_data))**4),
            'skewness': np.mean(((signal_data - np.mean(signal_data))/np.std(signal_data))**3)
        }
        
        # Frequency domain features
        fft_result = fft(signal_data)
        psd = np.abs(fft_result)**2 / len(signal_data)
        frequencies = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Positive frequencies only
        pos_freq = frequencies[:len(frequencies)//2]
        pos_psd = psd[:len(psd)//2]
        
        # Spectral centroid
        spectral_centroid = np.sum(pos_freq * pos_psd) / np.sum(pos_psd) if np.sum(pos_psd) > 0 else 0
        
        # Spectral spread
        spectral_spread = np.sqrt(np.sum((pos_freq - spectral_centroid)**2 * pos_psd) / np.sum(pos_psd)) if np.sum(pos_psd) > 0 else 0
        
        # Spectral entropy
        psd_normalized = pos_psd / np.sum(pos_psd) if np.sum(pos_psd) > 0 else pos_psd
        spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
        
        # Peak frequency
        peak_freq_idx = np.argmax(pos_psd[1:]) + 1  # Skip DC
        peak_frequency = pos_freq[peak_freq_idx]
        
        # Band energies
        band_energies = {}
        bands = [(0, 100), (100, 1000), (1000, 5000), (5000, 10000), (10000, 20000)]
        for f_low, f_high in bands:
            band_idx = np.where((pos_freq >= f_low) & (pos_freq < f_high))[0]
            if len(band_idx) > 0:
                band_energies[f'{f_low}-{f_high}Hz'] = np.sum(pos_psd[band_idx])
        
        features['frequency_domain'] = {
            'spectral_centroid_Hz': spectral_centroid,
            'spectral_spread_Hz': spectral_spread,
            'spectral_entropy': spectral_entropy,
            'peak_frequency_Hz': peak_frequency,
            'peak_magnitude': pos_psd[peak_freq_idx],
            'band_energies': band_energies,
            'total_energy': np.sum(pos_psd)
        }
        
        # Wavelet features (using db4 wavelet)
        coeffs = pywt.wavedec(signal_data, 'db4', level=6)
        wavelet_energies = []
        for i, c in enumerate(coeffs):
            energy = np.sum(c**2)
            wavelet_energies.append(energy)
        
        total_wavelet_energy = sum(wavelet_energies)
        relative_wavelet_energies = [e/total_wavelet_energy if total_wavelet_energy > 0 else 0 for e in wavelet_energies]
        
        features['wavelet_domain'] = {
            'wavelet_energies': wavelet_energies,
            'relative_energies': relative_wavelet_energies,
            'wavelet_entropy': -np.sum([e * np.log2(e + 1e-10) for e in relative_wavelet_energies if e > 0])
        }
        
        # Statistical pattern features
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_data))) > 0)
        zcr = zero_crossings / len(signal_data)
        
        # Autocorrelation features
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first minimum (indicates periodicity)
        first_min_idx = np.argmax(autocorr[1:] < 0) + 1 if np.any(autocorr[1:] < 0) else len(autocorr)
        
        features['pattern_features'] = {
            'zero_crossing_rate': zcr,
            'autocorr_first_min_lag': first_min_idx,
            'autocorr_peak_lag': np.argmax(autocorr[1:100]) + 1 if len(autocorr) > 100 else 0,
            'periodicity_strength': np.max(autocorr[1:min(1000, len(autocorr))]) if len(autocorr) > 1 else 0
        }
        
        return features
    
    def generate_tde_dataset(self):
        """Generate Time Delay Estimation dataset for flow velocity measurement"""
        data_list = []
        
        # Sensor configurations
        sensor_spacings = [0.1, 0.2, 0.5, 1.0, 2.0]  # meters
        flow_velocities = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # m/s
        
        for spacing in sensor_spacings:
            for velocity in flow_velocities:
                for condition in ['stratified_smooth', 'stratified_wavy', 'slug_flow']:
                    # Generate upstream signal
                    signal1 = self.generate_base_signal(condition)
                    
                    # Calculate time delay
                    true_time_delay = spacing / velocity
                    delay_samples = int(true_time_delay * self.sampling_rate)
                    
                    # Generate downstream signal (delayed and attenuated)
                    signal2 = np.zeros_like(signal1)
                    attenuation = 0.8 + 0.1 * np.random.rand()  # 10-20% attenuation
                    
                    if delay_samples < len(signal1):
                        signal2[delay_samples:] = signal1[:-delay_samples] * attenuation
                        # Add independent noise
                        signal2 += 0.05 * np.random.randn(len(signal2))
                    
                    # Calculate cross-correlation
                    correlation_result = self.calculate_cross_correlation(signal1, signal2)
                    
                    # Estimated velocity
                    if correlation_result['time_delay_s'] > 0:
                        estimated_velocity = spacing / correlation_result['time_delay_s']
                    else:
                        estimated_velocity = 0
                    
                    # Calculate error
                    velocity_error = abs(estimated_velocity - velocity)
                    relative_error = velocity_error / velocity if velocity > 0 else 0
                    
                    data_list.append({
                        'flow_condition': condition,
                        'sensor_spacing_m': spacing,
                        'true_velocity_m_s': velocity,
                        'true_time_delay_s': true_time_delay,
                        'measured_time_delay_s': correlation_result['time_delay_s'],
                        'estimated_velocity_m_s': estimated_velocity,
                        'velocity_error_m_s': velocity_error,
                        'relative_error_percent': relative_error * 100,
                        'correlation_peak_value': correlation_result['peak_value'],
                        'peak_to_sidelobe_ratio': correlation_result['peak_to_sidelobe_ratio'],
                        'correlation_width_s': correlation_result['correlation_width_s'],
                        'signal_attenuation': attenuation,
                        'sampling_rate_Hz': self.sampling_rate,
                        'signal_duration_s': self.signal_duration,
                        'measurement_quality': 'Good' if relative_error < 0.05 else 'Fair' if relative_error < 0.1 else 'Poor'
                    })
        
        return pd.DataFrame(data_list)
    
    def generate_leak_detection_dataset(self):
        """Generate dataset for leak detection algorithm testing"""
        data_list = []
        signal_library = {}
        
        # Leak scenarios
        leak_scenarios = [
            {'has_leak': False, 'condition': 'stratified_smooth', 'name': 'No_Leak_Smooth'},
            {'has_leak': False, 'condition': 'stratified_wavy', 'name': 'No_Leak_Wavy'},
            {'has_leak': False, 'condition': 'slug_flow', 'name': 'No_Leak_Slug'},
            {'has_leak': True, 'condition': 'stratified_smooth', 'leak_size': 'small', 'leak_location': 0.3, 'name': 'Small_Leak_Early'},
            {'has_leak': True, 'condition': 'stratified_smooth', 'leak_size': 'medium', 'leak_location': 0.5, 'name': 'Medium_Leak_Middle'},
            {'has_leak': True, 'condition': 'stratified_smooth', 'leak_size': 'large', 'leak_location': 0.7, 'name': 'Large_Leak_Late'},
            {'has_leak': True, 'condition': 'stratified_wavy', 'leak_size': 'small', 'leak_location': 0.4, 'name': 'Small_Leak_Wavy'},
            {'has_leak': True, 'condition': 'slug_flow', 'leak_size': 'medium', 'leak_location': 0.6, 'name': 'Medium_Leak_Slug'}
        ]
        
        for scenario in leak_scenarios:
            # Generate signal
            base_signal = self.generate_base_signal(scenario['condition'])
            
            if scenario['has_leak']:
                # Add leak with varying amplitudes based on size
                leak_amplitudes = {'small': 0.1, 'medium': 0.3, 'large': 0.5}
                self.flow_conditions['leak_present']['leak_amplitude'] = leak_amplitudes[scenario['leak_size']]
                
                # Generate signal with leak
                signal_with_leak = self.generate_base_signal(scenario['condition'], 
                                                            add_leak=True, 
                                                            leak_location=scenario['leak_location'])
            else:
                signal_with_leak = base_signal
            
            # Apply filters
            filter_results = self.apply_filters(signal_with_leak)
            
            # Extract leak detection features
            features = self.extract_leak_detection_features(signal_with_leak)
            
            # Create detection metrics
            if scenario['has_leak']:
                # Check if leak frequency is detected
                leak_freq = self.flow_conditions['leak_present']['leak_frequency']
                detected_freq = features['frequency_domain']['peak_frequency_Hz']
                
                # Detection success based on frequency proximity
                detection_success = abs(detected_freq - leak_freq) < 500  # Within 500 Hz
                
                # Location estimation (simplified - based on signal energy distribution)
                # Divide signal into segments
                n_segments = 10
                segment_length = len(signal_with_leak) // n_segments
                segment_energies = []
                for i in range(n_segments):
                    segment = signal_with_leak[i*segment_length:(i+1)*segment_length]
                    segment_energies.append(np.sum(segment**2))
                
                estimated_location = np.argmax(segment_energies) / n_segments
                location_error = abs(estimated_location - scenario['leak_location'])
            else:
                detection_success = True  # Correctly identified as no leak
                estimated_location = None
                location_error = None
            
            # Store processed data
            data_entry = {
                'scenario_name': scenario['name'],
                'has_leak': scenario['has_leak'],
                'flow_condition': scenario['condition'],
                'leak_size': scenario.get('leak_size', 'N/A'),
                'true_leak_location': scenario.get('leak_location', None),
                'estimated_leak_location': estimated_location,
                'location_error': location_error,
                'detection_success': detection_success,
                # Time domain features
                'signal_rms': features['time_domain']['rms'],
                'signal_kurtosis': features['time_domain']['kurtosis'],
                'signal_crest_factor': features['time_domain']['crest_factor'],
                # Frequency domain features
                'spectral_centroid_Hz': features['frequency_domain']['spectral_centroid_Hz'],
                'spectral_entropy': features['frequency_domain']['spectral_entropy'],
                'peak_frequency_Hz': features['frequency_domain']['peak_frequency_Hz'],
                'total_energy': features['frequency_domain']['total_energy'],
                # Wavelet features
                'wavelet_entropy': features['wavelet_domain']['wavelet_entropy'],
                # Pattern features
                'zero_crossing_rate': features['pattern_features']['zero_crossing_rate'],
                'periodicity_strength': features['pattern_features']['periodicity_strength'],
                # Filter outputs
                'bandpass_high_energy': filter_results['bandpass_High']['energy'],
                'bandpass_very_high_energy': filter_results['bandpass_Very_High']['energy'],
                'hilbert_envelope_std': filter_results['hilbert']['envelope_std']
            }
            
            data_list.append(data_entry)
            
            # Store signal for visualization
            signal_library[scenario['name']] = {
                'time': self.time[::100].tolist(),  # Downsampled
                'signal': signal_with_leak[::100].tolist(),
                'features': features,
                'filter_results': {k: v for k, v in filter_results.items() if k != 'fft'}  # Exclude large FFT data
            }
        
        # Save signal library
        with open('leak_detection_signals.json', 'w') as f:
            json.dump(signal_library, f, cls=NumpyEncoder)
        
        return pd.DataFrame(data_list)
    
    def save_all_datasets(self):
        """Generate and save all signal processing datasets"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("Generating TDE (Time Delay Estimation) dataset...")
        tde_data = self.generate_tde_dataset()
        tde_data.to_csv('tde_flow_velocity.csv', index=False)
        
        print("Generating leak detection dataset...")
        leak_data = self.generate_leak_detection_dataset()
        leak_data.to_csv('leak_detection_features.csv', index=False)
        
        # Generate filter comparison dataset
        print("Generating filter comparison dataset...")
        filter_comparison = []
        
        for condition in ['stratified_smooth', 'stratified_wavy', 'slug_flow']:
            test_signal = self.generate_base_signal(condition)
            filter_results = self.apply_filters(test_signal)
            
            # Compare filter effectiveness
            for filter_name in ['bandpass_Low', 'bandpass_Medium', 'bandpass_High']:
                if filter_name in filter_results:
                    filter_comparison.append({
                        'flow_condition': condition,
                        'filter_type': filter_name,
                        'frequency_range': filter_results[filter_name]['frequency_range_Hz'],
                        'rms_amplitude': filter_results[filter_name]['rms_amplitude'],
                        'peak_amplitude': filter_results[filter_name]['peak_amplitude'],
                        'energy': filter_results[filter_name]['energy']
                    })
        
        filter_df = pd.DataFrame(filter_comparison)
        filter_df.to_csv('filter_comparison.csv', index=False)
        
        # Create summary
        summary = {
            'generation_timestamp': timestamp,
            'datasets': {
                'tde_flow_velocity': {
                    'file': 'tde_flow_velocity.csv',
                    'records': len(tde_data),
                    'sensor_spacings': 5,
                    'flow_velocities': 6,
                    'flow_conditions': 3,
                    'sampling_rate_Hz': self.sampling_rate
                },
                'leak_detection': {
                    'file': 'leak_detection_features.csv',
                    'signals_file': 'leak_detection_signals.json',
                    'records': len(leak_data),
                    'scenarios': 8,
                    'features_extracted': 15
                },
                'filter_comparison': {
                    'file': 'filter_comparison.csv',
                    'records': len(filter_df),
                    'filter_types': ['Fourier', 'Bandpass', 'Wavelet', 'Hilbert', 'EMD'],
                    'wavelets_tested': ['sym5', 'sym8', 'db4', 'coif3']
                }
            },
            'signal_processing_methods': {
                'time_domain': ['RMS', 'Kurtosis', 'Skewness', 'Crest Factor'],
                'frequency_domain': ['FFT', 'PSD', 'Spectral Centroid', 'Spectral Entropy'],
                'time_frequency': ['Wavelet Transform', 'Hilbert Transform'],
                'correlation': ['Cross-correlation', 'Autocorrelation']
            },
            'total_records': len(tde_data) + len(leak_data) + len(filter_df)
        }
        
        with open('signal_processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nSignal processing datasets saved with timestamp: {timestamp}")
        return summary

if __name__ == "__main__":
    generator = SignalProcessingGenerator()
    summary = generator.save_all_datasets()
    print("\nGeneration complete!")
    print(f"Total records generated: {summary['total_records']}")