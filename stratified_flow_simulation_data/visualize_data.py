"""
Visualization Script for Stratified Flow Simulation Data
Generates publication-quality figures from the simulation dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def create_output_dir():
    """Create output directory for figures"""
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def visualize_velocity_field():
    """Visualize velocity field slices"""
    print("Creating velocity field visualization...")
    
    # Load data
    u = np.load('cfd_outputs/velocity_u.npy')
    v = np.load('cfd_outputs/velocity_v.npy')
    w = np.load('cfd_outputs/velocity_w.npy')
    
    with open('cfd_outputs/coordinates.json', 'r') as f:
        coords = json.load(f)
    x = np.array(coords['x'])
    y = np.array(coords['y'])
    z = np.array(coords['z'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # X-Z slice at y=0.5m (middle)
    y_idx = len(y) // 2
    X, Z = np.meshgrid(x, z)
    
    # u velocity
    im1 = axes[0, 0].contourf(X, Z, u[:, y_idx, :].T, levels=20, cmap='RdBu_r')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('z (m)')
    axes[0, 0].set_title('Velocity u (streamwise)')
    axes[0, 0].axhline(y=0.25, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im1, ax=axes[0, 0], label='u (m/s)')
    
    # v velocity
    im2 = axes[0, 1].contourf(X, Z, v[:, y_idx, :].T, levels=20, cmap='RdBu_r')
    axes[0, 1].set_xlabel('x (m)')
    axes[0, 1].set_ylabel('z (m)')
    axes[0, 1].set_title('Velocity v (cross-stream)')
    axes[0, 1].axhline(y=0.25, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im2, ax=axes[0, 1], label='v (m/s)')
    
    # w velocity
    im3 = axes[1, 0].contourf(X, Z, w[:, y_idx, :].T, levels=20, cmap='RdBu_r')
    axes[1, 0].set_xlabel('x (m)')
    axes[1, 0].set_ylabel('z (m)')
    axes[1, 0].set_title('Velocity w (vertical)')
    axes[1, 0].axhline(y=0.25, color='k', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im3, ax=axes[1, 0], label='w (m/s)')
    
    # Velocity magnitude
    vel_mag = np.sqrt(u[:, y_idx, :]**2 + v[:, y_idx, :]**2 + w[:, y_idx, :]**2)
    im4 = axes[1, 1].contourf(X, Z, vel_mag.T, levels=20, cmap='viridis')
    axes[1, 1].set_xlabel('x (m)')
    axes[1, 1].set_ylabel('z (m)')
    axes[1, 1].set_title('Velocity Magnitude')
    axes[1, 1].axhline(y=0.25, color='w', linestyle='--', linewidth=1, alpha=0.7)
    plt.colorbar(im4, ax=axes[1, 1], label='|V| (m/s)')
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/velocity_fields.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/velocity_fields.png")
    plt.close()

def visualize_vof_and_pressure():
    """Visualize VOF and pressure fields"""
    print("Creating VOF and pressure visualization...")
    
    # Load data
    vof = np.load('cfd_outputs/vof.npy')
    pressure = np.load('cfd_outputs/pressure.npy')
    
    with open('cfd_outputs/coordinates.json', 'r') as f:
        coords = json.load(f)
    x = np.array(coords['x'])
    z = np.array(coords['z'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # X-Z slice at y=0.5m
    y_idx = 50
    X, Z = np.meshgrid(x, z)
    
    # VOF
    im1 = axes[0].contourf(X, Z, vof[:, y_idx, :].T, levels=20, cmap='coolwarm')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('z (m)')
    axes[0].set_title('Volume of Fluid (VOF) - Phase Distribution')
    axes[0].contour(X, Z, vof[:, y_idx, :].T, levels=[0.5], colors='k', linewidths=2)
    plt.colorbar(im1, ax=axes[0], label='VOF (1=liquid, 0=gas)')
    
    # Pressure
    im2 = axes[1].contourf(X, Z, pressure[:, y_idx, :].T / 1000, levels=20, cmap='plasma')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('z (m)')
    axes[1].set_title('Pressure Field')
    axes[1].contour(X, Z, vof[:, y_idx, :].T, levels=[0.5], colors='w', 
                    linewidths=1, alpha=0.5, linestyles='--')
    plt.colorbar(im2, ax=axes[1], label='Pressure (kPa)')
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/vof_pressure.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/vof_pressure.png")
    plt.close()

def visualize_turbulence():
    """Visualize turbulence parameters"""
    print("Creating turbulence visualization...")
    
    # Load data
    k = np.load('cfd_outputs/turbulence_k.npy')
    epsilon = np.load('cfd_outputs/turbulence_epsilon.npy')
    mu_t = np.load('cfd_outputs/eddy_viscosity.npy')
    
    with open('cfd_outputs/coordinates.json', 'r') as f:
        coords = json.load(f)
    x = np.array(coords['x'])
    z = np.array(coords['z'])
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    y_idx = 50
    X, Z = np.meshgrid(x, z)
    
    # Turbulent kinetic energy
    im1 = axes[0].contourf(X, Z, k[:, y_idx, :].T, levels=20, cmap='hot')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('z (m)')
    axes[0].set_title('Turbulent Kinetic Energy (k)')
    axes[0].axhline(y=0.25, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im1, ax=axes[0], label='k (m²/s²)')
    
    # Dissipation rate
    im2 = axes[1].contourf(X, Z, epsilon[:, y_idx, :].T, levels=20, cmap='hot')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('z (m)')
    axes[1].set_title('Turbulent Dissipation Rate (ε)')
    axes[1].axhline(y=0.25, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im2, ax=axes[1], label='ε (m²/s³)')
    
    # Eddy viscosity
    im3 = axes[2].contourf(X, Z, mu_t[:, y_idx, :].T, levels=20, cmap='hot')
    axes[2].set_xlabel('x (m)')
    axes[2].set_ylabel('z (m)')
    axes[2].set_title('Eddy Viscosity (μₜ)')
    axes[2].axhline(y=0.25, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    plt.colorbar(im3, ax=axes[2], label='μₜ (Pa·s)')
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/turbulence_parameters.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/turbulence_parameters.png")
    plt.close()

def visualize_acoustic_propagation():
    """Visualize acoustic pressure propagation"""
    print("Creating acoustic propagation visualization...")
    
    # Load data
    acoustic = np.load('cfd_outputs/acoustic_pressure.npy')
    
    with open('cfd_outputs/coordinates.json', 'r') as f:
        coords = json.load(f)
    x = np.array(coords['x'])
    z = np.array(coords['z'])
    t = np.array(coords['t'])
    
    # Create figure with 4 time snapshots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    X, Z = np.meshgrid(x, z)
    time_indices = [100, 300, 500, 700]
    
    vmax = np.abs(acoustic).max() * 0.5
    
    for i, t_idx in enumerate(time_indices):
        im = axes[i].contourf(X, Z, acoustic[t_idx, :, :].T, 
                             levels=30, cmap='seismic', vmin=-vmax, vmax=vmax)
        axes[i].set_xlabel('x (m)')
        axes[i].set_ylabel('z (m)')
        axes[i].set_title(f'Acoustic Pressure at t = {t[t_idx]*1000:.1f} ms')
        axes[i].axhline(y=0.25, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[i].plot(0.2, 0.3, 'k*', markersize=10, label='Source')
        plt.colorbar(im, ax=axes[i], label='p (Pa)')
        if i == 0:
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/acoustic_propagation.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/acoustic_propagation.png")
    plt.close()

def visualize_sound_speed():
    """Visualize sound speed predictions"""
    print("Creating sound speed visualization...")
    
    # Load data
    c_wood = np.load('mathematical_model_outputs/sound_speed_wood.npy')
    c_disp = np.load('mathematical_model_outputs/sound_speed_dispersive.npy')
    
    with open('mathematical_model_outputs/parameters.json', 'r') as f:
        params = json.load(f)
    alpha = np.array(params['void_fractions'])
    freq = np.array(params['frequencies'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Wood's equation
    axes[0].plot(alpha, c_wood, 'b-', linewidth=2, label="Wood's Equation")
    axes[0].axhline(y=1500, color='b', linestyle='--', alpha=0.3, label='Pure liquid')
    axes[0].axhline(y=343, color='r', linestyle='--', alpha=0.3, label='Pure gas')
    axes[0].set_xlabel('Void Fraction α')
    axes[0].set_ylabel('Sound Speed (m/s)')
    axes[0].set_title("Sound Speed: Wood's Equation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim([0, 1600])
    
    # Dispersive model (selected frequencies)
    freq_indices = [0, 10, 20, 30, 40, 49]
    for idx in freq_indices:
        axes[1].plot(alpha, c_disp[idx, :], label=f'{freq[idx]:.0f} Hz')
    axes[1].set_xlabel('Void Fraction α')
    axes[1].set_ylabel('Sound Speed (m/s)')
    axes[1].set_title('Sound Speed: Dispersive Model (Eq. 27)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 1600])
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/sound_speed_predictions.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/sound_speed_predictions.png")
    plt.close()

def visualize_attenuation():
    """Visualize attenuation coefficients"""
    print("Creating attenuation visualization...")
    
    # Load data
    atten_dB = np.load('mathematical_model_outputs/attenuation_dB.npy')
    
    with open('mathematical_model_outputs/parameters.json', 'r') as f:
        params = json.load(f)
    alpha = np.array(params['void_fractions'])
    freq = np.array(params['frequencies'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Attenuation vs frequency (for selected void fractions)
    alpha_indices = [0, 5, 10, 15, 19]
    for idx in alpha_indices:
        axes[0].loglog(freq, atten_dB[:, idx], linewidth=2, 
                      label=f'α = {alpha[idx]:.2f}')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Attenuation (dB/m)')
    axes[0].set_title('Attenuation vs Frequency')
    axes[0].grid(True, alpha=0.3, which='both')
    axes[0].legend()
    
    # Attenuation vs void fraction (for selected frequencies)
    freq_indices = [0, 15, 30, 45, 49]
    for idx in freq_indices:
        axes[1].semilogy(alpha, atten_dB[idx, :], linewidth=2, 
                         label=f'{freq[idx]:.0f} Hz')
    axes[1].set_xlabel('Void Fraction α')
    axes[1].set_ylabel('Attenuation (dB/m)')
    axes[1].set_title('Attenuation vs Void Fraction')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/attenuation_coefficients.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/attenuation_coefficients.png")
    plt.close()

def visualize_wave_propagation():
    """Visualize wave propagation patterns"""
    print("Creating wave propagation visualization...")
    
    # Load data
    refl_angle = np.load('mathematical_model_outputs/reflection_vs_angle.npy')
    standing = np.load('mathematical_model_outputs/standing_waves.npy')
    
    with open('mathematical_model_outputs/reflection_transmission.json', 'r') as f:
        refl_trans = json.load(f)
    
    with open('mathematical_model_outputs/parameters.json', 'r') as f:
        params = json.load(f)
    pos = np.array(params['positions'])
    freq = np.array(params['frequencies'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reflection vs angle
    angles = np.linspace(0, 89, 90)
    axes[0].plot(angles, refl_angle, 'b-', linewidth=2)
    axes[0].set_xlabel('Incident Angle (degrees)')
    axes[0].set_ylabel('Reflection Coefficient |R|')
    axes[0].set_title('Reflection vs Incident Angle')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=refl_trans['reflection_coefficient'], 
                   color='r', linestyle='--', 
                   label=f"Normal incidence: {refl_trans['reflection_coefficient']:.3f}")
    axes[0].legend()
    axes[0].set_ylim([0, 1.05])
    
    # Standing wave patterns
    for i in range(0, 10, 2):
        axes[1].plot(pos, standing[i, :], linewidth=1.5, 
                    label=f'{freq[i]:.0f} Hz')
    axes[1].set_xlabel('Position (m)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Standing Wave Patterns')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/wave_propagation.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/wave_propagation.png")
    plt.close()

def visualize_validation():
    """Visualize validation comparisons"""
    print("Creating validation visualization...")
    
    # Load waveform data
    wave_sim = np.load('validation_data/waveform_simulated.npy')
    wave_exp = np.load('validation_data/waveform_experimental.npy')
    wave_time = np.load('validation_data/waveform_time.npy')
    
    # Load comparison data
    with open('validation_data/attenuation_comparison.json', 'r') as f:
        atten_comp = json.load(f)
    with open('validation_data/sound_speed_comparison.json', 'r') as f:
        speed_comp = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Waveform comparison
    axes[0, 0].plot(wave_time * 1000, wave_sim, 'b-', linewidth=1.5, 
                   label='Simulated', alpha=0.8)
    axes[0, 0].plot(wave_time * 1000, wave_exp, 'r-', linewidth=1, 
                   label='Experimental', alpha=0.6)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Acoustic Pressure (Pa)')
    axes[0, 0].set_title('Waveform Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Waveform residual
    residual = wave_exp - wave_sim
    axes[0, 1].plot(wave_time * 1000, residual, 'k-', linewidth=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Residual (Pa)')
    axes[0, 1].set_title('Experimental - Simulated')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Attenuation comparison
    alpha_a = np.array(atten_comp['void_fraction'])
    atten_sim = np.array(atten_comp['simulated'])
    atten_exp = np.array(atten_comp['experimental'])
    atten_unc = np.array(atten_comp['uncertainty'])
    
    axes[1, 0].errorbar(alpha_a, atten_exp, yerr=atten_unc, fmt='ro', 
                       label='Experimental', capsize=5)
    axes[1, 0].plot(alpha_a, atten_sim, 'b-', linewidth=2, label='Simulated')
    axes[1, 0].set_xlabel('Void Fraction α')
    axes[1, 0].set_ylabel('Attenuation (Np/m)')
    axes[1, 0].set_title('Attenuation Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Sound speed comparison
    alpha_s = np.array(speed_comp['void_fraction'])
    speed_sim = np.array(speed_comp['simulated'])
    speed_exp = np.array(speed_comp['experimental'])
    speed_unc = np.array(speed_comp['uncertainty'])
    
    axes[1, 1].errorbar(alpha_s, speed_exp, yerr=speed_unc, fmt='ro', 
                       label='Experimental', capsize=5)
    axes[1, 1].plot(alpha_s, speed_sim, 'b-', linewidth=2, label='Simulated')
    axes[1, 1].set_xlabel('Void Fraction α')
    axes[1, 1].set_ylabel('Sound Speed (m/s)')
    axes[1, 1].set_title('Sound Speed Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{create_output_dir()}/validation_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/validation_comparison.png")
    plt.close()

def main():
    """Main visualization function"""
    print("="*70)
    print("STRATIFIED FLOW SIMULATION DATA VISUALIZATION")
    print("="*70)
    print()
    
    # Create all visualizations
    visualize_velocity_field()
    visualize_vof_and_pressure()
    visualize_turbulence()
    visualize_acoustic_propagation()
    visualize_sound_speed()
    visualize_attenuation()
    visualize_wave_propagation()
    visualize_validation()
    
    print()
    print("="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated 8 figure files in './figures/' directory:")
    print("  1. velocity_fields.png")
    print("  2. vof_pressure.png")
    print("  3. turbulence_parameters.png")
    print("  4. acoustic_propagation.png")
    print("  5. sound_speed_predictions.png")
    print("  6. attenuation_coefficients.png")
    print("  7. wave_propagation.png")
    print("  8. validation_comparison.png")
    print()

if __name__ == "__main__":
    main()
