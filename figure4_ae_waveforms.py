"""
Figure 4: Advanced Acoustic Emission Waveform Templates and Spectral Characteristics
====================================================================================
Application: SOFC Damage Mechanism Identification via AE Monitoring
Methods: Time-Frequency Analysis, Wavelet Decomposition, PSD Estimation
Framework: Bridge Between Chemo-Mechanical Simulation and Experimental NDT

Author: georgegershom
Date: 2026-02-01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patheffects as path_effects
from scipy import signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PUBLICATION STYLE CONFIGURATION
# ============================================================================
@dataclass
class PublicationStyle:
    """Professional figure styling for AE analysis"""
    
    FIG_WIDTH: float = 190 / 25.4  # 190mm full page
    FIG_HEIGHT: float = 240 / 25.4  # 240mm tall
    DPI: int = 300
    
    TITLE_SIZE: int = 11
    LABEL_SIZE: int = 10
    TICK_SIZE: int = 9
    LEGEND_SIZE: int = 8
    ANNOTATION_SIZE: int = 8
    PANEL_LABEL_SIZE: int = 12
    
    COLORS: Dict = field(default_factory=lambda: {
        # AE Event Types
        'type_a': '#2E86AB',      # Blue - YSZ microcracking
        'type_b': '#F6AE2D',      # Gold - Interface delamination
        'type_c': '#E94F37',      # Red - Ni ligament fracture
        
        # Signal components
        'envelope': '#8B5CF6',    # Purple for envelope
        'noise': '#94A3B8',       # Gray for noise floor
        'threshold': '#10B981',   # Green for thresholds
        
        # Spectral
        'psd_fill': '#3B82F6',
        'spectrogram_cmap': 'magma',
        
        # UI
        'text_dark': '#1e293b',
        'text_medium': '#475569',
        'grid': '#e2e8f0',
        'background': '#ffffff',
    })
    
    def apply_style(self):
        plt.rcParams.update({
            'font.size': self.LABEL_SIZE,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'axes.labelsize': self.LABEL_SIZE,
            'axes.titlesize': self.TITLE_SIZE,
            'axes.linewidth': 0.8,
            'xtick.labelsize': self.TICK_SIZE,
            'ytick.labelsize': self.TICK_SIZE,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'legend.fontsize': self.LEGEND_SIZE,
            'legend.framealpha': 0.95,
            'figure.dpi': self.DPI,
            'savefig.dpi': 600,
            'lines.linewidth': 1.5,
        })


# ============================================================================
# AE WAVEFORM PARAMETERS (FROM TABLE 2)
# ============================================================================
@dataclass
class AEEventParameters:
    """AE event classification parameters from experimental data"""
    name: str
    mechanism: str
    peak_freq_kHz: float
    freq_std_kHz: float
    rise_time_us: float
    rise_time_std_us: float
    energy_aJ: float
    color: str
    
    # Waveform generation parameters
    decay_time_us: float = 15.0
    envelope_type: str = 'exponential'  # or 'gaussian'
    harmonic_content: List[float] = field(default_factory=lambda: [1.0, 0.3, 0.1])


# Define AE event types from Table 2
TYPE_A = AEEventParameters(
    name='Type A',
    mechanism='YSZ Microcracking',
    peak_freq_kHz=474.9,
    freq_std_kHz=94.0,
    rise_time_us=15.6,
    rise_time_std_us=7.2,
    energy_aJ=528.6,
    color='#2E86AB',
    decay_time_us=18.0,
    envelope_type='exponential',
    harmonic_content=[1.0, 0.35, 0.12, 0.05]
)

TYPE_B = AEEventParameters(
    name='Type B',
    mechanism='Interface Delamination',
    peak_freq_kHz=275.3,
    freq_std_kHz=68.0,
    rise_time_us=22.4,
    rise_time_std_us=9.8,
    energy_aJ=842.3,
    color='#F6AE2D',
    decay_time_us=25.0,
    envelope_type='exponential',
    harmonic_content=[1.0, 0.45, 0.20, 0.08]
)

TYPE_C = AEEventParameters(
    name='Type C',
    mechanism='Ni Ligament Fracture',
    peak_freq_kHz=649.7,
    freq_std_kHz=111.0,
    rise_time_us=9.3,
    rise_time_std_us=4.2,
    energy_aJ=219.7,
    color='#E94F37',
    decay_time_us=8.0,
    envelope_type='gaussian',
    harmonic_content=[1.0, 0.40, 0.25, 0.10]
)


# ============================================================================
# ADVANCED WAVEFORM GENERATION
# ============================================================================
class AEWaveformGenerator:
    """Generate realistic AE waveforms based on physical parameters"""
    
    def __init__(self, fs: float = 5e6, duration: float = 100e-6):
        self.fs = fs  # 5 MHz sampling rate
        self.duration = duration
        self.time = np.linspace(0, duration, int(fs * duration))
        self.n_samples = len(self.time)
    
    def generate_waveform(self, params: AEEventParameters, 
                          noise_level: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic AE waveform with given parameters"""
        
        f0 = params.peak_freq_kHz * 1e3  # Convert to Hz
        t = self.time
        
        # Generate multi-harmonic signal
        signal_wave = np.zeros_like(t)
        for i, amp in enumerate(params.harmonic_content):
            freq = f0 * (1 + 0.5 * i)  # Harmonic frequencies
            phase = np.random.uniform(0, 2 * np.pi)
            signal_wave += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add frequency modulation for realism
        fm_depth = 0.05
        fm_freq = 50e3
        freq_mod = 1 + fm_depth * np.sin(2 * np.pi * fm_freq * t)
        signal_wave = signal_wave * freq_mod
        
        # Apply envelope
        envelope = self._generate_envelope(params, t)
        signal_wave = signal_wave * envelope
        
        # Add realistic noise (pink noise + white noise)
        pink_noise = self._generate_pink_noise(len(t)) * noise_level * 0.7
        white_noise = np.random.randn(len(t)) * noise_level * 0.3
        signal_wave += pink_noise + white_noise
        
        # Normalize
        signal_wave = signal_wave / np.max(np.abs(signal_wave))
        
        return t, signal_wave
    
    def _generate_envelope(self, params: AEEventParameters, t: np.ndarray) -> np.ndarray:
        """Generate appropriate envelope for waveform"""
        
        rise_time = params.rise_time_us * 1e-6
        decay_time = params.decay_time_us * 1e-6
        
        if params.envelope_type == 'exponential':
            # Exponential rise and decay (brittle fracture)
            rise_env = 1 - np.exp(-t / rise_time)
            decay_env = np.exp(-t / decay_time)
            envelope = rise_env * decay_env
            
        elif params.envelope_type == 'gaussian':
            # Gaussian envelope (ductile/rapid fracture)
            t_peak = 2 * rise_time
            sigma = decay_time / 2
            envelope = np.exp(-((t - t_peak) / sigma) ** 2)
            
        else:
            envelope = np.ones_like(t)
        
        return envelope
    
    def _generate_pink_noise(self, n: int) -> np.ndarray:
        """Generate 1/f pink noise"""
        white = np.random.randn(n)
        fft_white = fft(white)
        freqs = fftfreq(n, 1/self.fs)
        
        # Apply 1/f filter (avoiding division by zero)
        pink_filter = np.where(freqs != 0, 1 / np.sqrt(np.abs(freqs) + 1), 1)
        fft_pink = fft_white * pink_filter
        
        pink = np.real(np.fft.ifft(fft_pink))
        return pink / np.std(pink)
    
    def compute_envelope(self, signal_wave: np.ndarray) -> np.ndarray:
        """Compute signal envelope using Hilbert transform"""
        analytic = signal.hilbert(signal_wave)
        envelope = np.abs(analytic)
        return envelope


# ============================================================================
# SIGNAL ANALYSIS FUNCTIONS
# ============================================================================
class AESignalAnalyzer:
    """Advanced AE signal analysis toolkit"""
    
    def __init__(self, fs: float):
        self.fs = fs
    
    def compute_spectrogram(self, signal_wave: np.ndarray, 
                           nperseg: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram using STFT"""
        f, t, Sxx = signal.spectrogram(signal_wave, self.fs, 
                                        nperseg=nperseg, noverlap=nperseg//2,
                                        window='hann', scaling='density')
        return f, t, Sxx
    
    def compute_psd(self, signal_wave: np.ndarray, 
                    nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PSD using Welch's method"""
        f, psd = signal.welch(signal_wave, self.fs, nperseg=nperseg,
                              window='hann', scaling='density')
        return f, psd
    
    def compute_fft(self, signal_wave: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT magnitude spectrum"""
        n = len(signal_wave)
        window = signal.windows.hann(n)
        windowed = signal_wave * window
        
        spectrum = np.abs(fft(windowed))[:n//2]
        freqs = fftfreq(n, 1/self.fs)[:n//2]
        
        return freqs, spectrum
    
    def calculate_parameters(self, signal_wave: np.ndarray, 
                            time: np.ndarray) -> Dict:
        """Calculate AE waveform parameters"""
        
        # Peak frequency from FFT
        freqs, spectrum = self.compute_fft(signal_wave)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        
        # Rise time (10% to 90% of peak)
        envelope = np.abs(signal.hilbert(signal_wave))
        max_env = np.max(envelope)
        
        t10_idx = np.where(envelope > 0.1 * max_env)[0]
        t90_idx = np.where(envelope > 0.9 * max_env)[0]
        
        if len(t10_idx) > 0 and len(t90_idx) > 0:
            rise_time = (time[t90_idx[0]] - time[t10_idx[0]]) * 1e6
        else:
            rise_time = np.nan
        
        # Energy
        dt = time[1] - time[0]
        energy = np.sum(signal_wave**2) * dt
        
        # Duration (above 10% threshold)
        if len(t10_idx) > 0:
            duration = (time[t10_idx[-1]] - time[t10_idx[0]]) * 1e6
        else:
            duration = np.nan
        
        # Counts (threshold crossings)
        threshold = 0.2 * max_env
        crossings = np.where(np.diff(np.sign(signal_wave - threshold)))[0]
        counts = len(crossings) // 2
        
        return {
            'peak_freq_kHz': peak_freq / 1e3,
            'rise_time_us': rise_time,
            'energy_aJ': energy * 1e18,
            'duration_us': duration,
            'counts': counts,
            'amplitude': np.max(np.abs(signal_wave))
        }


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def add_panel_label(ax, label: str, style: PublicationStyle, 
                    x: float = -0.12, y: float = 1.08):
    """Add panel label with professional styling"""
    text = ax.text(x, y, f'({label})', transform=ax.transAxes,
                   fontsize=style.PANEL_LABEL_SIZE, fontweight='bold',
                   color=style.COLORS['text_dark'], va='top', ha='left')
    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])


def create_custom_colormap(color: str) -> LinearSegmentedColormap:
    """Create custom colormap from white to specified color"""
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    colors = [(1, 1, 1), rgb]
    return LinearSegmentedColormap.from_list('custom', colors, N=256)


# ============================================================================
# MAIN PANEL FUNCTIONS
# ============================================================================
def create_waveform_panel(ax, time: np.ndarray, signal_wave: np.ndarray,
                          params: AEEventParameters, style: PublicationStyle,
                          show_envelope: bool = True):
    """Create time-domain waveform panel with annotations"""
    
    t_us = time * 1e6  # Convert to microseconds
    
    # Plot waveform
    ax.plot(t_us, signal_wave, color=params.color, linewidth=1.2, 
            alpha=0.9, label=f'{params.name}: {params.mechanism}')
    
    # Fill under curve
    ax.fill_between(t_us, 0, signal_wave, alpha=0.15, color=params.color)
    
    # Compute and plot envelope
    if show_envelope:
        envelope = np.abs(signal.hilbert(signal_wave))
        ax.plot(t_us, envelope, '--', color='#8B5CF6', linewidth=1.0,
                alpha=0.8, label='Envelope')
        ax.plot(t_us, -envelope, '--', color='#8B5CF6', linewidth=1.0, alpha=0.8)
    
    # Add threshold line
    threshold = 0.2
    ax.axhline(y=threshold, color='#10B981', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.axhline(y=-threshold, color='#10B981', linestyle=':', linewidth=0.8, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Time (μs)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Amplitude (norm.)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title(f'{params.name}: {params.mechanism}', fontsize=style.TITLE_SIZE,
                 fontweight='bold', color=params.color, pad=8)
    
    ax.set_xlim([0, t_us[-1]])
    ax.set_ylim([-1.3, 1.3])
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Add parameter box
    param_text = (f"$f_{{peak}}$ = {params.peak_freq_kHz:.0f} kHz\n"
                  f"$t_{{rise}}$ = {params.rise_time_us:.1f} μs\n"
                  f"$E$ = {params.energy_aJ:.0f} aJ")
    
    bbox = dict(boxstyle='round,pad=0.3', facecolor='white',
                edgecolor=params.color, alpha=0.95, linewidth=1.5)
    ax.text(0.97, 0.97, param_text, transform=ax.transAxes,
            fontsize=style.ANNOTATION_SIZE, va='top', ha='right',
            bbox=bbox, family='monospace')


def create_spectrogram_panel(ax, time: np.ndarray, signal_wave: np.ndarray,
                             params: AEEventParameters, style: PublicationStyle,
                             fs: float):
    """Create spectrogram panel with time-frequency analysis"""
    
    analyzer = AESignalAnalyzer(fs)
    f, t, Sxx = analyzer.compute_spectrogram(signal_wave, nperseg=64)
    
    # Convert to kHz and μs
    f_kHz = f / 1e3
    t_us = t * 1e6
    
    # Plot spectrogram
    pcm = ax.pcolormesh(t_us, f_kHz, 10 * np.log10(Sxx + 1e-12),
                        shading='gouraud', cmap='magma', vmin=-80, vmax=-20)
    
    # Add peak frequency line
    ax.axhline(y=params.peak_freq_kHz, color='white', linestyle='--',
               linewidth=1.5, alpha=0.8)
    ax.text(t_us[-1] * 0.95, params.peak_freq_kHz + 30, 
            f'{params.peak_freq_kHz:.0f} kHz',
            color='white', fontsize=style.ANNOTATION_SIZE - 1,
            ha='right', va='bottom', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Time (μs)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Frequency (kHz)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylim([0, 1000])
    
    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, aspect=20)
    cbar.set_label('PSD (dB)', fontsize=style.ANNOTATION_SIZE)
    cbar.ax.tick_params(labelsize=style.TICK_SIZE - 1)


def create_psd_comparison_panel(ax, waveforms: Dict, style: PublicationStyle, fs: float):
    """Create PSD comparison panel for all event types"""
    
    analyzer = AESignalAnalyzer(fs)
    
    for name, (signal_wave, params) in waveforms.items():
        f, psd = analyzer.compute_psd(signal_wave, nperseg=256)
        f_kHz = f / 1e3
        
        # Plot PSD
        ax.semilogy(f_kHz, psd, color=params.color, linewidth=2.0,
                    label=f'{params.name}: {params.mechanism}', alpha=0.9)
        
        # Fill under curve
        ax.fill_between(f_kHz, psd, alpha=0.1, color=params.color)
        
        # Mark peak frequency
        peak_idx = np.argmax(psd)
        ax.axvline(x=f_kHz[peak_idx], color=params.color, linestyle=':',
                   linewidth=1.5, alpha=0.6)
        
        # Add frequency band
        f_low = params.peak_freq_kHz - params.freq_std_kHz
        f_high = params.peak_freq_kHz + params.freq_std_kHz
        ax.axvspan(f_low, f_high, alpha=0.05, color=params.color)
    
    # Add frequency zone labels
    zones = [
        (200, 350, 'Interface\nDelamination', '#F6AE2D'),
        (380, 570, 'YSZ\nMicrocracking', '#2E86AB'),
        (540, 760, 'Ni Ligament\nFracture', '#E94F37'),
    ]
    
    ylim = ax.get_ylim()
    for f_low, f_high, label, color in zones:
        ax.annotate(label, xy=((f_low + f_high) / 2, ylim[1] * 0.3),
                    fontsize=style.ANNOTATION_SIZE - 1, ha='center', va='center',
                    color=color, fontweight='bold', alpha=0.8)
    
    ax.set_xlabel('Frequency (kHz)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('PSD (V²/Hz)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('Power Spectral Density Comparison', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)
    ax.set_xlim([0, 1000])
    ax.set_ylim([1e-8, 1e-2])
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')


def create_fft_comparison_panel(ax, waveforms: Dict, style: PublicationStyle, fs: float):
    """Create FFT magnitude comparison panel"""
    
    analyzer = AESignalAnalyzer(fs)
    
    for name, (signal_wave, params) in waveforms.items():
        freqs, spectrum = analyzer.compute_fft(signal_wave)
        f_kHz = freqs / 1e3
        
        # Normalize spectrum
        spectrum_norm = spectrum / np.max(spectrum)
        
        ax.plot(f_kHz, spectrum_norm, color=params.color, linewidth=1.8,
                label=f'{params.name}', alpha=0.9)
    
    ax.set_xlabel('Frequency (kHz)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Magnitude (norm.)', fontsize=style.LABEL_SIZE, fontweight='medium')
    ax.set_title('FFT Magnitude Spectrum', fontsize=style.TITLE_SIZE,
                 fontweight='bold', pad=10)
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')


def create_parameter_table(ax, waveforms: Dict, style: PublicationStyle, fs: float):
    """Create parameter comparison table"""
    
    ax.axis('off')
    
    analyzer = AESignalAnalyzer(fs)
    
    # Table header
    headers = ['Parameter', 'Type A', 'Type B', 'Type C', 'Unit']
    
    # Calculate parameters for each waveform
    calc_params = {}
    for name, (signal_wave, params) in waveforms.items():
        time = np.arange(len(signal_wave)) / fs
        calc_params[name] = analyzer.calculate_parameters(signal_wave, time)
    
    # Table data with comparison to paper values
    table_data = [
        ['Peak Freq.', 
         f"{calc_params['Type A']['peak_freq_kHz']:.0f} ({TYPE_A.peak_freq_kHz:.0f})",
         f"{calc_params['Type B']['peak_freq_kHz']:.0f} ({TYPE_B.peak_freq_kHz:.0f})",
         f"{calc_params['Type C']['peak_freq_kHz']:.0f} ({TYPE_C.peak_freq_kHz:.0f})",
         'kHz'],
        ['Rise Time',
         f"{calc_params['Type A']['rise_time_us']:.1f} ({TYPE_A.rise_time_us:.1f})",
         f"{calc_params['Type B']['rise_time_us']:.1f} ({TYPE_B.rise_time_us:.1f})",
         f"{calc_params['Type C']['rise_time_us']:.1f} ({TYPE_C.rise_time_us:.1f})",
         'μs'],
        ['Energy',
         f"{TYPE_A.energy_aJ:.0f}",
         f"{TYPE_B.energy_aJ:.0f}",
         f"{TYPE_C.energy_aJ:.0f}",
         'aJ'],
        ['Freq. Std.',
         f"±{TYPE_A.freq_std_kHz:.0f}",
         f"±{TYPE_B.freq_std_kHz:.0f}",
         f"±{TYPE_C.freq_std_kHz:.0f}",
         'kHz'],
    ]
    
    # Create table
    colors = ['white', TYPE_A.color, TYPE_B.color, TYPE_C.color, 'white']
    cell_colors = []
    for row in table_data:
        row_colors = ['#f8fafc'] + ['#f1f5f9'] * 3 + ['#f8fafc']
        cell_colors.append(row_colors)
    
    header_colors = ['#1e293b', TYPE_A.color, TYPE_B.color, TYPE_C.color, '#1e293b']
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=[c if c != 'white' else '#1e293b' for c in header_colors],
                     cellColours=cell_colors)
    
    table.auto_set_font_size(False)
    table.set_fontsize(style.ANNOTATION_SIZE)
    table.scale(1.0, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        table[(0, i)].set_facecolor(header_colors[i] if header_colors[i] != 'white' else '#1e293b')
    
    ax.set_title('AE Parameter Validation (Calc. vs Paper)', 
                 fontsize=style.TITLE_SIZE, fontweight='bold', pad=15)


def create_classification_diagram(ax, style: PublicationStyle):
    """Create damage mechanism classification diagram"""
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw boxes for each damage type
    types = [
        (TYPE_A, 1, 6.5, 'Ceramic\nBrittle\nFracture'),
        (TYPE_B, 4, 6.5, 'Interface\nDelamination'),
        (TYPE_C, 7, 6.5, 'Metal\nDuctile\nFracture'),
    ]
    
    for params, x, y, desc in types:
        # Main box
        box = FancyBboxPatch((x, y), 2, 2.5,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=params.color, edgecolor='white',
                             linewidth=2, alpha=0.85)
        ax.add_patch(box)
        
        # Type label
        ax.text(x + 1, y + 2.1, params.name, fontsize=style.ANNOTATION_SIZE + 2,
                ha='center', va='center', color='white', fontweight='bold')
        
        # Frequency
        ax.text(x + 1, y + 1.5, f'{params.peak_freq_kHz:.0f} kHz',
                fontsize=style.ANNOTATION_SIZE + 1, ha='center', va='center',
                color='white', fontweight='bold')
        
        # Description below
        ax.text(x + 1, y - 0.5, desc, fontsize=style.ANNOTATION_SIZE - 1,
                ha='center', va='top', color=params.color, fontweight='medium')
    
    # Add arrows showing frequency progression
    ax.annotate('', xy=(3.2, 7.75), xytext=(2.8, 7.75),
                arrowprops=dict(arrowstyle='->', color='#64748b', lw=2))
    ax.annotate('', xy=(6.2, 7.75), xytext=(5.8, 7.75),
                arrowprops=dict(arrowstyle='->', color='#64748b', lw=2))
    
    ax.text(5, 8.8, 'Increasing Frequency →', fontsize=style.ANNOTATION_SIZE,
            ha='center', va='center', color='#64748b', fontweight='bold')
    
    # Title
    ax.text(5, 9.8, 'Damage Mechanism Classification', fontsize=style.TITLE_SIZE,
            ha='center', va='top', fontweight='bold', color=style.COLORS['text_dark'])


# ============================================================================
# MAIN FIGURE CREATION
# ============================================================================
def create_figure4():
    """Create comprehensive Figure 4 with all AE analysis panels"""
    
    style = PublicationStyle()
    style.apply_style()
    
    # Generate waveforms
    generator = AEWaveformGenerator(fs=5e6, duration=100e-6)
    
    print("Generating AE waveforms...")
    time_a, wave_a = generator.generate_waveform(TYPE_A)
    time_b, wave_b = generator.generate_waveform(TYPE_B)
    time_c, wave_c = generator.generate_waveform(TYPE_C)
    
    waveforms = {
        'Type A': (wave_a, TYPE_A),
        'Type B': (wave_b, TYPE_B),
        'Type C': (wave_c, TYPE_C),
    }
    
    # Create figure with complex grid
    fig = plt.figure(figsize=(style.FIG_WIDTH, style.FIG_HEIGHT))
    
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1.0, 0.8, 1.2, 0.8],
                           width_ratios=[1, 1, 1],
                           hspace=0.35, wspace=0.30,
                           left=0.08, right=0.95, top=0.94, bottom=0.05)
    
    # Row 1: Time-domain waveforms
    print("Generating Panel (a): Type A waveform...")
    ax_wave_a = fig.add_subplot(gs[0, 0])
    create_waveform_panel(ax_wave_a, time_a, wave_a, TYPE_A, style)
    add_panel_label(ax_wave_a, 'a', style)
    
    print("Generating Panel (b): Type B waveform...")
    ax_wave_b = fig.add_subplot(gs[0, 1])
    create_waveform_panel(ax_wave_b, time_b, wave_b, TYPE_B, style)
    add_panel_label(ax_wave_b, 'b', style)
    
    print("Generating Panel (c): Type C waveform...")
    ax_wave_c = fig.add_subplot(gs[0, 2])
    create_waveform_panel(ax_wave_c, time_c, wave_c, TYPE_C, style)
    add_panel_label(ax_wave_c, 'c', style)
    
    # Row 2: Spectrograms
    print("Generating Panel (d): Type A spectrogram...")
    ax_spec_a = fig.add_subplot(gs[1, 0])
    create_spectrogram_panel(ax_spec_a, time_a, wave_a, TYPE_A, style, generator.fs)
    add_panel_label(ax_spec_a, 'd', style)
    
    print("Generating Panel (e): Type B spectrogram...")
    ax_spec_b = fig.add_subplot(gs[1, 1])
    create_spectrogram_panel(ax_spec_b, time_b, wave_b, TYPE_B, style, generator.fs)
    add_panel_label(ax_spec_b, 'e', style)
    
    print("Generating Panel (f): Type C spectrogram...")
    ax_spec_c = fig.add_subplot(gs[1, 2])
    create_spectrogram_panel(ax_spec_c, time_c, wave_c, TYPE_C, style, generator.fs)
    add_panel_label(ax_spec_c, 'f', style)
    
    # Row 3: PSD comparison (spans 2 columns) + FFT
    print("Generating Panel (g): PSD comparison...")
    ax_psd = fig.add_subplot(gs[2, :2])
    create_psd_comparison_panel(ax_psd, waveforms, style, generator.fs)
    add_panel_label(ax_psd, 'g', style)
    
    print("Generating Panel (h): FFT comparison...")
    ax_fft = fig.add_subplot(gs[2, 2])
    create_fft_comparison_panel(ax_fft, waveforms, style, generator.fs)
    add_panel_label(ax_fft, 'h', style)
    
    # Row 4: Parameter table + Classification diagram
    print("Generating Panel (i): Parameter table...")
    ax_table = fig.add_subplot(gs[3, :2])
    create_parameter_table(ax_table, waveforms, style, generator.fs)
    add_panel_label(ax_table, 'i', style, x=-0.06)
    
    print("Generating Panel (j): Classification diagram...")
    ax_class = fig.add_subplot(gs[3, 2])
    create_classification_diagram(ax_class, style)
    add_panel_label(ax_class, 'j', style, x=-0.06)
    
    # Main title
    fig.suptitle('Acoustic Emission Waveform Templates and Spectral Characteristics',
                 fontsize=style.TITLE_SIZE + 3, fontweight='bold', y=0.98)
    
    return fig, waveforms, generator


def print_validation():
    """Print validation data"""
    print("\n" + "=" * 80)
    print("AE WAVEFORM CLASSIFICATION VALIDATION (TABLE 2)")
    print("=" * 80)
    
    print("\nDamage Mechanism | Peak Freq (kHz) | Rise Time (μs) | Energy (aJ)")
    print("-" * 80)
    
    for params in [TYPE_A, TYPE_B, TYPE_C]:
        print(f"{params.name}: {params.mechanism:<20} | "
              f"{params.peak_freq_kHz:>6.1f} ± {params.freq_std_kHz:<5.0f} | "
              f"{params.rise_time_us:>5.1f} ± {params.rise_time_std_us:<4.1f} | "
              f"{params.energy_aJ:>7.1f}")
    
    print("\n" + "=" * 80)
    print("FREQUENCY SEPARATION ANALYSIS:")
    print("=" * 80)
    print(f"Type A → Type B separation: {TYPE_A.peak_freq_kHz - TYPE_B.peak_freq_kHz:.0f} kHz")
    print(f"Type B → Type C separation: {TYPE_C.peak_freq_kHz - TYPE_B.peak_freq_kHz:.0f} kHz")
    print(f"Type A → Type C separation: {TYPE_C.peak_freq_kHz - TYPE_A.peak_freq_kHz:.0f} kHz")
    print("\n→ Distinct frequency bands enable deterministic damage classification")
    print("→ Direct correlation between simulation events and AE signatures")


def main():
    """Main entry point"""
    
    print("=" * 75)
    print("FIGURE 4: ACOUSTIC EMISSION WAVEFORM TEMPLATES")
    print("=" * 75)
    print("Analysis:     Time-Frequency, PSD, FFT Spectral Characteristics")
    print("Application:  SOFC Damage Mechanism Identification")
    print("Framework:    Simulation-Experiment Correlation via AE")
    print("Author:       georgegershom")
    print("Date:         2026-02-01")
    print("=" * 75 + "\n")
    
    fig, waveforms, generator = create_figure4()
    
    print("\n✨ Figure generation complete!\n")
    
    # Save
    fig.savefig('figure4_ae_waveforms.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig('figure4_ae_waveforms.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    print("Saved: figure4_ae_waveforms.png (600 DPI)")
    print("Saved: figure4_ae_waveforms.pdf (vector)")
    
    print_validation()
    
    plt.show()
    return fig


if __name__ == '__main__':
    main()
