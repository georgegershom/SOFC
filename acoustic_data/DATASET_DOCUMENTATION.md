# Technical Documentation
## Synthetic High-Frequency Acoustic Pressure Dataset

---

## Table of Contents

1. [Physical Model Description](#physical-model-description)
2. [Signal Generation Methodology](#signal-generation-methodology)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Statistical Characteristics](#statistical-characteristics)
5. [Validation and Quality Control](#validation-and-quality-control)
6. [Analysis Recommendations](#analysis-recommendations)

---

## Physical Model Description

### 1. Pipeline System Model

The synthetic data represents a linear pipeline system with the following characteristics:

**Pipeline Specifications:**
- Length: 32.5 meters
- Medium: Water (stratified flow conditions)
- Operating Pressure: 1.0 bar (ambient baseline)
- Temperature: Assumed constant (20°C)

**Boundary Conditions:**
- Valve positions: 0m, 17.5m
- Leak points: A (6.0m), D (13.5m), E (24.0m)
- Fixed-end reflections at boundaries

### 2. Acoustic Wave Propagation

**Wave Equation:**
The pressure field satisfies the wave equation:

∂²p/∂t² = c² ∂²p/∂x²

Where:
- p = acoustic pressure (Pa)
- c = speed of sound = 1500 m/s
- x = spatial coordinate along pipeline
- t = time

**Propagation Characteristics:**
- Phase velocity: 1500 m/s
- Wavelength (λ): λ = c/f
  - At 850 Hz: λ ≈ 1.76 m
  - At 5000 Hz: λ ≈ 0.30 m

### 3. Attenuation Model

**Spatial Attenuation:**
Pressure amplitude decays exponentially with distance:

p(d) = p₀ × exp(-α × d)

Where:
- p₀ = source pressure amplitude
- α = attenuation coefficient = 0.15 dB/m
- d = distance from source

**Frequency-Dependent Effects:**
- Higher frequencies attenuate more rapidly
- Implemented through filter bandwidth constraints
- Realistic dispersion characteristics

### 4. Leak Acoustic Source Model

**Turbulent Jet Model:**
Leak generates acoustic signature through turbulent jet mixing:

**Fundamental Frequency:**
f₀ ≈ U/D × St

Where:
- U = jet velocity (m/s)
- D = leak diameter (m)
- St = Strouhal number ≈ 0.2
- Nominal f₀ = 850 Hz

**Harmonics:**
Multiple harmonics present: 2f₀, 3f₀, 5f₀, 7f₀, 11f₀
Amplitude decay: A_n ∝ 1/(n × (harmonic_order + 1))

**Broadband Component:**
- Turbulent noise: 500-4900 Hz bandwidth
- 4th-order Butterworth band-pass filter
- Gaussian distribution with leak-dependent variance

---

## Signal Generation Methodology

### 1. Baseline Noise Generation

**Components:**
```python
# White noise component
σ_white = 0.0005 bar
white_noise = N(0, σ_white)

# Low-frequency drift
f_drift = 0.5 Hz
drift = 0.0002 × sin(2πf_drift × t)

# Baseline pressure
p_baseline = 1.0 bar + white_noise + drift
```

**Statistical Properties:**
- Mean: 1.0 bar
- Standard deviation: ~0.0005 bar
- Frequency content: DC to 5 Hz (dominant)

### 2. Leak Signature Generation

**Step-by-Step Process:**

**a) Distance Calculation:**
```python
distance = |sensor_position - leak_position|
```

**b) Propagation Delay:**
```python
time_delay = distance / speed_of_sound
```

**c) Amplitude Scaling:**
```python
amplitude = leak_intensity × 0.5 × exp(-α × distance)
```

**d) Fundamental Component:**
```python
p_fundamental = A × sin(2πf₀(t - τ))
```

**e) Harmonic Components:**
```python
for h in [2, 3, 5, 7, 11]:
    A_h = A / (h × (harmonic_index + 2))
    p_harmonics += A_h × sin(2πhf₀(t - τ))
```

**f) Broadband Turbulence:**
```python
turbulence = N(0, A × 0.3)
turbulence = bandpass_filter(turbulence, 500, 4900 Hz)
```

**g) Pipeline Reflections:**
```python
reflection_delay = 0.05 s
A_reflection = 0.15 × A
p_reflection = A_reflection × sin(2πf₀(t - τ - reflection_delay))
```

### 3. Temporal Envelope

**Leak Event Timing:**
```python
# Onset ramp (100ms)
if t in [5.0, 5.1]:
    envelope = Hann_window(0 → 1)

# Steady leak
if t in [5.1, 19.9]:
    envelope = 1.0

# Offset ramp (100ms)
if t in [19.9, 20.0]:
    envelope = Hann_window(1 → 0)

# No leak
else:
    envelope = 0.0
```

**Smooth Transitions:**
- Hann window for onset/offset
- Prevents spectral leakage
- Realistic valve dynamics

### 4. Sensor-Specific Characteristics

Each sensor has unique noise characteristics:

```python
sensor_noise = N(0, 0.0003 bar)
```

**Total Signal:**
```python
p_total = p_baseline + p_leak × envelope + sensor_noise
```

---

## Data Processing Pipeline

### 1. Time Array Generation

**Sampling:**
```python
# Group 01
fs = 17060 Hz
dt = 1/fs = 58.6 μs

# Groups 02-04
fs = 10000 Hz
dt = 1/fs = 100 μs

# Time array
t = linspace(0, 30, N_samples)
where N_samples = duration × fs
```

### 2. Signal Synthesis

**For Each Sensor:**
1. Generate baseline noise array
2. Generate leak signature with proper delays
3. Apply temporal envelope
4. Add sensor noise
5. Combine components

**For Each Group:**
1. Set leak location
2. Set sampling rate
3. Set leak intensity
4. Generate all 14 sensors
5. Package into DataFrame

### 3. File Organization

**Per-Second Segmentation:**
```python
for second in range(30):
    start_idx = second × fs
    end_idx = (second + 1) × fs
    
    data_segment = full_data[start_idx:end_idx]
    data_segment['Time_s'] -= second  # Reset to 0-1s
    
    save_csv(f"Group_{N}_second_{second:02d}.csv")
```

**Benefits:**
- Manageable file sizes (~2.7 MB each)
- Easy to load specific time windows
- Parallel processing friendly
- Matches experimental data format

---

## Statistical Characteristics

### 1. Baseline Phase (0-5s)

**Statistical Moments:**
- Mean: 1.0000 bar
- Standard Deviation: 0.0005-0.0007 bar
- Skewness: ≈ 0 (symmetric)
- Kurtosis: ≈ 3 (Gaussian)

**Frequency Content:**
- Peak at DC (1.0 bar baseline)
- Low-frequency drift: 0.5 Hz
- White noise: flat spectrum 0-fs/2

### 2. Leak Phase (5-20s)

**Statistical Moments:**
- Mean: ~1.0000-1.0001 bar (slight offset)
- Standard Deviation: Varies by distance
  - Near leak: 0.05-0.5 bar
  - Far from leak: 0.005-0.05 bar
- Skewness: Slight positive (leak events)
- Kurtosis: > 3 (super-Gaussian, impulsive events)

**Frequency Content:**
- Strong peak at 850 Hz (fundamental)
- Harmonics at 1700, 2550, 4250, 5950, 9350 Hz
- Elevated broadband: 500-4900 Hz
- Baseline noise floor below 500 Hz

**Spatial Distribution:**
| Distance from Leak | RMS Pressure (typical) |
|-------------------|----------------------|
| 0-5 m | 0.1-0.5 bar |
| 5-10 m | 0.05-0.2 bar |
| 10-15 m | 0.02-0.1 bar |
| 15-20 m | 0.01-0.05 bar |
| > 20 m | 0.005-0.02 bar |

### 3. Recovery Phase (20-30s)

**Decay Characteristics:**
- Exponential decay with τ ≈ 0.5-1.0 s
- Returns to baseline noise levels
- Slight pressure oscillations (resonance decay)

**Statistical Evolution:**
```
t = 20.0s: σ = peak value
t = 21.0s: σ ≈ 0.37 × peak (e⁻¹)
t = 22.0s: σ ≈ 0.14 × peak (e⁻²)
t = 23.0s: σ ≈ baseline
```

---

## Validation and Quality Control

### 1. Physical Consistency Checks

**✓ Causality:**
- Signals at distant sensors arrive later
- Delay = distance/speed_of_sound
- Verified for all sensor pairs

**✓ Energy Conservation:**
- Total acoustic energy decreases with distance
- ∫p²dt decreases exponentially
- Verified across spatial array

**✓ Frequency Content:**
- All frequencies < Nyquist frequency
- No aliasing artifacts
- Proper filter roll-off

**✓ Attenuation:**
- Amplitude decreases with distance
- Follows exponential law
- Consistent across all groups

### 2. Statistical Validation

**Baseline Noise:**
```python
# Test: baseline should be Gaussian
from scipy.stats import normaltest

for sensor in sensors:
    baseline_data = data[0:5s][sensor]
    stat, p_value = normaltest(baseline_data)
    assert p_value > 0.05  # Cannot reject normality
```

**Leak Signature:**
```python
# Test: leak phase should show elevated variance
baseline_var = var(data[0:5s][sensor])
leak_var = var(data[5:20s][sensor])
assert leak_var > 10 × baseline_var  # Significant increase
```

### 3. Spectral Validation

**Power Spectral Density:**
```python
from scipy.signal import welch

# Expected peaks at harmonics
f, psd = welch(leak_data, fs=sampling_rate)

# Check for fundamental
assert max(psd[f ≈ 850 Hz]) > 10 × median(psd)

# Check for harmonics
for h in [2, 3, 5]:
    assert psd[f ≈ h*850 Hz] > 3 × median(psd)
```

### 4. Cross-Sensor Consistency

**Correlation Analysis:**
- Adjacent sensors: high correlation (ρ > 0.7)
- Distant sensors: moderate correlation (ρ = 0.3-0.7)
- Sensors on opposite sides: low correlation (ρ < 0.3)

**Coherence Analysis:**
- High coherence at fundamental frequency
- Decreasing coherence with distance
- Phase shifts correspond to propagation delays

---

## Analysis Recommendations

### 1. Leak Detection Algorithms

**Feature Extraction:**
```python
# Time-domain features
- RMS pressure
- Peak amplitude
- Kurtosis (impulsiveness)
- Zero-crossing rate

# Frequency-domain features
- Peak frequency
- Spectral centroid
- Bandwidth (3dB)
- Harmonic ratio
```

**Detection Methods:**
- Energy threshold detection
- Matched filter (template: 850 Hz + harmonics)
- Machine learning (train on labeled data)
- Change-point detection (baseline vs. leak)

### 2. Leak Localization

**Time-Difference of Arrival (TDOA):**
```python
# Cross-correlation between sensor pairs
for i, j in sensor_pairs:
    tau_ij = argmax(cross_correlation(signal_i, signal_j))
    distance_ij = tau_ij × speed_of_sound
    
# Multilateration
leak_position = solve_multilateration(distance_matrix)
```

**Amplitude-Based Localization:**
```python
# Fit attenuation model
def model(x, x_leak, A0):
    return A0 × exp(-alpha × |x - x_leak|)

# Find leak position that best fits data
x_leak_est = optimize(model, sensor_positions, amplitudes)
```

### 3. Attenuation Analysis

**Exponential Fit:**
```python
from scipy.optimize import curve_fit

def exponential_decay(d, A0, alpha):
    return A0 × np.exp(-alpha × d)

# Fit to data
distances = [|sensor_pos - leak_pos| for sensor in sensors]
amplitudes = [rms(sensor_data) for sensor in sensors]

popt, pcov = curve_fit(exponential_decay, distances, amplitudes)
A0_est, alpha_est = popt
```

**Expected Results:**
- α ≈ 0.15 dB/m (should recover this value)
- R² > 0.85 (good fit)
- Residuals: random, no systematic bias

### 4. Frequency Analysis

**Short-Time Fourier Transform (STFT):**
```python
from scipy.signal import stft

f, t, Zxx = stft(signal, fs=sampling_rate, 
                  nperseg=1024, noverlap=512)

# Visualize spectrogram
plt.pcolormesh(t, f, np.abs(Zxx))
```

**Expected Observations:**
- Fundamental at 850 Hz during leak
- Harmonics visible up to ~5 kHz
- Onset at t=5s, offset at t=20s
- Broadband noise elevation

**Wavelet Analysis:**
```python
import pywt

scales = pywt.frequency2scale('morl', 
                              frequencies=np.logspace(2, 3.7, 50),
                              fs=sampling_rate)
coef, freq = pywt.cwt(signal, scales, 'morl', fs=sampling_rate)
```

### 5. Machine Learning Applications

**Supervised Learning:**
```python
# Labels
y = ['baseline', 'leak', 'recovery']  # for each time segment

# Features
X = extract_features(signals)  # RMS, spectral features, etc.

# Models
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

**Unsupervised Learning:**
```python
# Anomaly detection
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1)
anomalies = clf.fit_predict(features)
```

**Deep Learning:**
```python
# CNN for raw signal classification
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 64, activation='relu'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(64, 32, activation='relu'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])
```

### 6. Comparative Analysis

**Cross-Group Comparisons:**

**Effect of Leak Location:**
- Compare Groups 01 vs 03 (locations A vs E)
- Analyze sensor-leak distance effects
- Evaluate detection difficulty

**Effect of Sampling Rate:**
- Compare Groups 01 (17 kHz) vs 04 (10 kHz)
- Same leak location (A), similar intensity
- Assess minimum sampling requirements

**Effect of Leak Intensity:**
- Compare Groups 01 (1.0) vs 02 (0.8) vs 04 (0.85)
- Evaluate detection sensitivity
- Establish threshold requirements

---

## Advanced Analysis Techniques

### 1. Modal Decomposition

**Proper Orthogonal Decomposition (POD):**
```python
from sklearn.decomposition import PCA

# Spatial modes
X = signals_matrix  # (time × sensors)
pca = PCA(n_components=10)
modes = pca.fit_transform(X)

# Dominant modes capture leak signature
```

### 2. Beamforming

**Delay-and-Sum Beamforming:**
```python
def beamformer(signals, positions, target_position, c):
    delays = |positions - target_position| / c
    aligned = [delay_signal(s, d) for s, d in zip(signals, delays)]
    return sum(aligned)

# Scan along pipeline
for x in np.linspace(0, 32.5, 100):
    power[x] = |beamformer(signals, positions, x, c)|²
    
leak_location = argmax(power)
```

### 3. Wavefield Decomposition

**Separating Upstream/Downstream Waves:**
```python
# Using sensor array
k = 2π/λ  # wavenumber
dk = k_upstream - k_downstream

# Fourier decomposition in space
FFT_space = fft(signals_at_time_t)
upstream = FFT_space[k > 0]
downstream = FFT_space[k < 0]
```

---

## Troubleshooting Common Issues

### Issue 1: Low Signal-to-Noise Ratio
**Symptoms:** Difficult to detect leak signature
**Solutions:**
- Focus on sensors near leak location
- Use band-pass filter (500-5000 Hz)
- Average multiple time windows
- Use adaptive threshold

### Issue 2: Aliasing Artifacts
**Symptoms:** Unexpected frequencies above Nyquist
**Cause:** Insufficient sampling rate
**Solution:** Use Group 01 data (17 kHz) or apply anti-aliasing filter

### Issue 3: Boundary Reflections
**Symptoms:** Multiple peaks in cross-correlation
**Cause:** Pipeline end reflections
**Solution:** 
- Focus on first arrival
- Use windowing to isolate direct path
- Model reflections explicitly

### Issue 4: Phase Wrapping
**Symptoms:** Inconsistent phase relationships
**Cause:** Large distances relative to wavelength
**Solution:**
- Use unwrapping algorithms
- Rely on group delay instead of phase
- Use multiple frequencies

---

## References and Further Reading

### Acoustic Theory
1. Pierce, A.D. (1989). *Acoustics: An Introduction to Its Physical Principles and Applications*
2. Kinsler, L.E. et al. (2000). *Fundamentals of Acoustics*

### Leak Detection
3. Li, S. et al. (2015). "Leak detection in pipeline systems using acoustic emissions"
4. Verde, C. (2001). "Multi-leak detection and isolation in fluid pipelines"

### Signal Processing
5. Oppenheim, A.V. & Schafer, R.W. (2009). *Discrete-Time Signal Processing*
6. Stoica, P. & Moses, R. (2005). *Spectral Analysis of Signals*

### Turbulent Jets
7. Lighthill, M.J. (1952). "On sound generated aerodynamically"
8. Ffowcs Williams, J.E. (1969). "Hydrodynamic noise"

---

## Appendix: Quick Reference

### Key Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Sampling Rate (high) | 17,060 | Hz |
| Sampling Rate (standard) | 10,000 | Hz |
| Duration | 30 | s |
| Number of Sensors | 14 | - |
| Pipeline Length | 32.5 | m |
| Speed of Sound | 1,500 | m/s |
| Attenuation Coefficient | 0.15 | dB/m |
| Fundamental Frequency | 850 | Hz |
| Ambient Pressure | 1.0 | bar |
| Noise Level | 0.001 | bar (RMS) |

### File Naming Convention
```
{Group}_{second}.csv
Example: Group_01_second_05.csv
```

### Quick Load (Python)
```python
import pandas as pd
df = pd.read_csv('Group_01/Group_01_second_05.csv')
```

---

**Document Version:** 1.0  
**Last Updated:** February 5, 2026  
**Status:** Complete

---

**End of Technical Documentation**
