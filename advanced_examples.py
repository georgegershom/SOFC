"""
Advanced Examples for FEM Stress Analysis Visualization

This script demonstrates additional capabilities beyond the basic visualization:
1. Parametric study across optimization levels
2. Statistical analysis of stress distributions
3. Export data for external analysis
4. Custom stress field integration
5. Animation generation (optimization progression)
"""

import numpy as np
import matplotlib.pyplot as plt
from fem_stress_analysis_visualization import FEMStressAnalyzer


def example_1_parametric_study():
    """
    Example 1: Parametric study across optimization levels
    
    Demonstrates how stress metrics vary with optimization intensity.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Parametric Optimization Study")
    print("="*70)
    
    analyzer = FEMStressAnalyzer(seed=42)
    
    # Test multiple optimization levels
    opt_levels = np.linspace(0, 1, 11)  # 0%, 10%, ..., 100%
    
    results = {
        'opt_level': [],
        'sigma_max': [],
        'A_crit': [],
        'sigma_avg': [],
        'delta_p': [],
        'warpage': []
    }
    
    for level in opt_levels:
        sigma = analyzer.compute_stress_field(optimization_level=level)
        metrics = analyzer.compute_global_metrics(sigma)
        constraints = analyzer.check_constraints(optimization_level=level)
        
        results['opt_level'].append(level * 100)
        results['sigma_max'].append(metrics['sigma_max'])
        results['A_crit'].append(metrics['A_crit'])
        results['sigma_avg'].append(metrics['sigma_avg'])
        results['delta_p'].append(constraints['delta_p'])
        results['warpage'].append(constraints['delta_warpage'])
        
        print(f"  Level {level*100:5.1f}%: σ_max={metrics['sigma_max']:6.1f} MPa, "
              f"A_crit={metrics['A_crit']:6.1f} mm²")
    
    # Create multi-panel parametric plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parametric Optimization Study', fontsize=14, fontweight='bold')
    
    # Panel 1: Peak stress vs optimization
    ax1 = axes[0, 0]
    ax1.plot(results['opt_level'], results['sigma_max'], 'o-', linewidth=2, 
             markersize=6, color='#d62728')
    ax1.axhline(analyzer.sigma_crit, color='k', linestyle='--', linewidth=1.5,
                label=f'σ_crit = {analyzer.sigma_crit} MPa')
    ax1.fill_between(results['opt_level'], analyzer.sigma_crit, 350, 
                     alpha=0.1, color='red', label='Critical zone')
    ax1.set_xlabel('Optimization Level [%]', fontweight='bold')
    ax1.set_ylabel('Peak Stress σ_max [MPa]', fontweight='bold')
    ax1.set_title('Peak Stress Reduction')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Critical area vs optimization
    ax2 = axes[0, 1]
    ax2.plot(results['opt_level'], results['A_crit'], 's-', linewidth=2, 
             markersize=6, color='#ff7f0e')
    ax2.fill_between(results['opt_level'], results['A_crit'], alpha=0.2, color='#ff7f0e')
    ax2.set_xlabel('Optimization Level [%]', fontweight='bold')
    ax2.set_ylabel('Critical Area A_crit [mm²]', fontweight='bold')
    ax2.set_title('Critical Area Shrinkage')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Average stress vs optimization
    ax3 = axes[1, 0]
    ax3.plot(results['opt_level'], results['sigma_avg'], '^-', linewidth=2, 
             markersize=6, color='#2ca02c')
    ax3.fill_between(results['opt_level'], results['sigma_avg'], alpha=0.2, color='#2ca02c')
    ax3.set_xlabel('Optimization Level [%]', fontweight='bold')
    ax3.set_ylabel('Average Stress σ_avg [MPa]', fontweight='bold')
    ax3.set_title('Average Stress Trend')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Constraints vs optimization
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    l1 = ax4.plot(results['opt_level'], results['delta_p'], 'v-', linewidth=2, 
                  markersize=6, color='#1f77b4', label='Δp')
    ax4.axhline(analyzer.delta_p_max, color='#1f77b4', linestyle='--', 
                linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Optimization Level [%]', fontweight='bold')
    ax4.set_ylabel('Pressure Drop Δp [kPa]', fontweight='bold', color='#1f77b4')
    ax4.tick_params(axis='y', labelcolor='#1f77b4')
    
    l2 = ax4_twin.plot(results['opt_level'], results['warpage'], 'D-', linewidth=2, 
                       markersize=6, color='#9467bd', label='δ')
    ax4_twin.axhline(analyzer.delta_max, color='#9467bd', linestyle='--', 
                     linewidth=1.5, alpha=0.5)
    ax4_twin.set_ylabel('Warpage δ [mm]', fontweight='bold', color='#9467bd')
    ax4_twin.tick_params(axis='y', labelcolor='#9467bd')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    ax4.set_title('Design Constraints')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/parametric_study.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Parametric study plot saved: /workspace/parametric_study.png\n")
    plt.close()


def example_2_statistical_analysis():
    """
    Example 2: Statistical analysis of stress distributions
    
    Compares statistical properties (histogram, CDF, percentiles).
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Statistical Stress Distribution Analysis")
    print("="*70)
    
    analyzer = FEMStressAnalyzer(seed=42)
    
    sigma_base = analyzer.compute_stress_field(optimization_level=0.0)
    sigma_opt = analyzer.compute_stress_field(optimization_level=0.85)
    
    # Compute statistics
    percentiles = [50, 75, 90, 95, 99]
    
    print("\nPercentile Analysis:")
    print(f"{'Percentile':<12} {'Baseline [MPa]':<16} {'Optimized [MPa]':<16} {'Reduction':<12}")
    print("-" * 60)
    
    for p in percentiles:
        base_p = np.percentile(sigma_base, p)
        opt_p = np.percentile(sigma_opt, p)
        reduction = base_p - opt_p
        print(f"{p:>3}th %ile    {base_p:>12.1f}      {opt_p:>12.1f}      {reduction:>8.1f}")
    
    # Create statistical comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Statistical Stress Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Panel 1: Histograms
    ax1 = axes[0, 0]
    ax1.hist(sigma_base, bins=50, alpha=0.6, color='#d62728', label='Baseline', density=True)
    ax1.hist(sigma_opt, bins=50, alpha=0.6, color='#2ca02c', label='Optimized', density=True)
    ax1.axvline(analyzer.sigma_crit, color='k', linestyle='--', linewidth=2, 
                label=f'σ_crit = {analyzer.sigma_crit} MPa')
    ax1.set_xlabel('von Mises Stress [MPa]', fontweight='bold')
    ax1.set_ylabel('Probability Density', fontweight='bold')
    ax1.set_title('Stress Distribution Histograms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Cumulative distribution
    ax2 = axes[0, 1]
    sigma_sorted_base = np.sort(sigma_base)
    sigma_sorted_opt = np.sort(sigma_opt)
    cdf_base = np.arange(1, len(sigma_sorted_base) + 1) / len(sigma_sorted_base)
    cdf_opt = np.arange(1, len(sigma_sorted_opt) + 1) / len(sigma_sorted_opt)
    
    ax2.plot(sigma_sorted_base, cdf_base, '-', linewidth=2, color='#d62728', label='Baseline')
    ax2.plot(sigma_sorted_opt, cdf_opt, '-', linewidth=2, color='#2ca02c', label='Optimized')
    ax2.axvline(analyzer.sigma_crit, color='k', linestyle='--', linewidth=2)
    ax2.set_xlabel('von Mises Stress [MPa]', fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontweight='bold')
    ax2.set_title('Cumulative Distribution Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Box plots
    ax3 = axes[1, 0]
    bp = ax3.boxplot([sigma_base, sigma_opt], labels=['Baseline', 'Optimized'],
                      patch_artist=True, showmeans=True)
    colors = ['#d62728', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.axhline(analyzer.sigma_crit, color='k', linestyle='--', linewidth=2)
    ax3.set_ylabel('von Mises Stress [MPa]', fontweight='bold')
    ax3.set_title('Box Plot Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Q-Q plot
    ax4 = axes[1, 1]
    sigma_base_sorted = np.sort(sigma_base)
    sigma_opt_sorted = np.sort(sigma_opt)
    # Match lengths
    n = min(len(sigma_base_sorted), len(sigma_opt_sorted))
    ax4.scatter(sigma_base_sorted[:n], sigma_opt_sorted[:n], alpha=0.5, s=10, color='#1f77b4')
    
    # Reference line (y=x)
    lims = [0, max(sigma_base_sorted.max(), sigma_opt_sorted.max())]
    ax4.plot(lims, lims, 'k--', linewidth=2, alpha=0.5, label='y=x (no change)')
    ax4.set_xlabel('Baseline Stress [MPa]', fontweight='bold')
    ax4.set_ylabel('Optimized Stress [MPa]', fontweight='bold')
    ax4.set_title('Q-Q Plot: Baseline vs Optimized')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/workspace/statistical_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Statistical analysis plot saved: /workspace/statistical_analysis.png\n")
    plt.close()


def example_3_data_export():
    """
    Example 3: Export stress field data for external analysis
    
    Saves data in multiple formats (CSV, NPY) for use in other tools.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Data Export for External Analysis")
    print("="*70)
    
    analyzer = FEMStressAnalyzer(seed=42)
    
    sigma_base = analyzer.compute_stress_field(optimization_level=0.0)
    sigma_opt = analyzer.compute_stress_field(optimization_level=0.85)
    delta_sigma = sigma_base - sigma_opt
    
    # Export as NumPy arrays (binary, efficient)
    np.savez_compressed('/workspace/stress_fields.npz',
                       x=analyzer.x,
                       y=analyzer.y,
                       triangles=analyzer.tri.simplices,
                       sigma_baseline=sigma_base,
                       sigma_optimized=sigma_opt,
                       delta_sigma=delta_sigma)
    print("✓ Saved: /workspace/stress_fields.npz (NumPy compressed)")
    
    # Export as CSV (text, portable)
    with open('/workspace/stress_field_data.csv', 'w') as f:
        f.write("node_id,x,y,sigma_baseline,sigma_optimized,delta_sigma\n")
        for i in range(len(analyzer.x)):
            f.write(f"{i},{analyzer.x[i]:.6f},{analyzer.y[i]:.6f},"
                   f"{sigma_base[i]:.6f},{sigma_opt[i]:.6f},{delta_sigma[i]:.6f}\n")
    print("✓ Saved: /workspace/stress_field_data.csv")
    
    # Export mesh connectivity
    with open('/workspace/mesh_elements.csv', 'w') as f:
        f.write("element_id,node_1,node_2,node_3\n")
        for i, tri in enumerate(analyzer.tri.simplices):
            f.write(f"{i},{tri[0]},{tri[1]},{tri[2]}\n")
    print("✓ Saved: /workspace/mesh_elements.csv")
    
    # Export metrics summary
    metrics_base = analyzer.compute_global_metrics(sigma_base)
    metrics_opt = analyzer.compute_global_metrics(sigma_opt)
    
    with open('/workspace/analysis_summary.txt', 'w') as f:
        f.write("FEM STRESS ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("BASELINE DESIGN\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Peak stress (σ_max):        {metrics_base['sigma_max']:>10.2f} MPa\n")
        f.write(f"  Critical area (A_crit):     {metrics_base['A_crit']:>10.2f} mm²\n")
        f.write(f"  Average stress (σ_avg):     {metrics_base['sigma_avg']:>10.2f} MPa\n")
        f.write(f"  Std deviation (σ_std):      {metrics_base['sigma_std']:>10.2f} MPa\n\n")
        
        f.write("OPTIMIZED DESIGN\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Peak stress (σ_max):        {metrics_opt['sigma_max']:>10.2f} MPa\n")
        f.write(f"  Critical area (A_crit):     {metrics_opt['A_crit']:>10.2f} mm²\n")
        f.write(f"  Average stress (σ_avg):     {metrics_opt['sigma_avg']:>10.2f} MPa\n")
        f.write(f"  Std deviation (σ_std):      {metrics_opt['sigma_std']:>10.2f} MPa\n\n")
        
        f.write("IMPROVEMENTS\n")
        f.write("-" * 70 + "\n")
        delta_max = metrics_base['sigma_max'] - metrics_opt['sigma_max']
        delta_area = metrics_base['A_crit'] - metrics_opt['A_crit']
        f.write(f"  Peak stress reduction:      {delta_max:>10.2f} MPa "
               f"({100*delta_max/metrics_base['sigma_max']:.1f}%)\n")
        f.write(f"  Critical area reduction:    {delta_area:>10.2f} mm² "
               f"({100*delta_area/metrics_base['A_crit']:.1f}%)\n")
        f.write(f"  Max local Δσ:               {delta_sigma.max():>10.2f} MPa\n")
    
    print("✓ Saved: /workspace/analysis_summary.txt")
    print("\nData export complete! Files can be imported into:")
    print("  • MATLAB/Octave: load('stress_fields.npz')")
    print("  • Python/NumPy:  np.load('stress_fields.npz')")
    print("  • Excel/Sheets:  Import 'stress_field_data.csv'")
    print("  • ParaView/VTK:  Convert from CSV to VTK format\n")


def example_4_custom_colormap():
    """
    Example 4: Custom professional colormaps
    
    Demonstrates various colormap options for different audiences.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Professional Colormaps")
    print("="*70)
    
    from matplotlib.colors import LinearSegmentedColormap
    
    analyzer = FEMStressAnalyzer(seed=42)
    sigma = analyzer.compute_stress_field(optimization_level=0.0)
    
    # Define professional colormaps
    colormaps = {
        'Thermal (Original)': LinearSegmentedColormap.from_list('thermal',
            ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
             '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']),
        
        'Engineering': LinearSegmentedColormap.from_list('engineering',
            ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#800000']),
        
        'Grayscale (Print)': LinearSegmentedColormap.from_list('grayscale',
            ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000']),
        
        'Colorblind-Safe': LinearSegmentedColormap.from_list('cb_safe',
            ['#0173b2', '#029e73', '#ece133', '#de8f05', '#cc78bc', '#ca9161'])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Professional Colormap Options for FEM Stress Visualization', 
                fontsize=14, fontweight='bold')
    
    levels = np.linspace(0, 150, 11)
    
    for (name, cmap), ax in zip(colormaps.items(), axes.flat):
        contour = ax.tricontourf(analyzer.triangulation, sigma, 
                                 levels=levels, cmap=cmap, extend='max')
        
        # Add critical isoline
        ax.tricontour(analyzer.triangulation, sigma, 
                     levels=[analyzer.sigma_crit], colors='white', 
                     linewidths=1.5, linestyles='--', alpha=0.9)
        
        ax.set_xlabel('x [mm]', fontweight='bold')
        ax.set_ylabel('y [mm]', fontweight='bold')
        ax.set_title(name, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle=':')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', 
                           pad=0.08, aspect=30)
        cbar.set_label('σ [MPa]', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/workspace/colormap_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Colormap comparison saved: /workspace/colormap_comparison.png\n")
    plt.close()


def run_all_examples():
    """Run all advanced examples"""
    print("\n" + "█"*70)
    print("█" + " "*22 + "ADVANCED EXAMPLES" + " "*22 + "█")
    print("█"*70)
    
    example_1_parametric_study()
    example_2_statistical_analysis()
    example_3_data_export()
    example_4_custom_colormap()
    
    print("\n" + "█"*70)
    print("█" + " "*18 + "ALL EXAMPLES COMPLETE" + " "*20 + "█")
    print("█"*70)
    print("\nGenerated files:")
    print("  • parametric_study.png       - Optimization level sensitivity")
    print("  • statistical_analysis.png   - Statistical comparisons")
    print("  • colormap_comparison.png    - Different visualization styles")
    print("  • stress_fields.npz          - Binary data (NumPy)")
    print("  • stress_field_data.csv      - Tabular data (portable)")
    print("  • mesh_elements.csv          - Mesh connectivity")
    print("  • analysis_summary.txt       - Text summary report")
    print("\n" + "█"*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
