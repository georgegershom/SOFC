"""
SENCE Framework - Interactive Demonstration Script
==================================================
This script demonstrates various use cases and customizations
of the SENCE radar chart visualization system.
"""

import numpy as np
import matplotlib.pyplot as plt
from sence_radar_visualization import SENCERadarChart
import pandas as pd


def demo_basic_usage():
    """Demonstrate basic usage."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Usage")
    print("="*80)
    
    radar = SENCERadarChart()
    
    print("\n📊 Available Cities:")
    for city, data in radar.city_data.items():
        print(f"  • {city}: Mean CVI = {data['mean_cvi']:.3f}")
    
    print("\n📈 Generating basic radar chart...")
    fig = radar.create_advanced_radar_chart()
    fig.savefig('/workspace/outputs/demo_basic.png', dpi=300)
    print("✓ Saved: demo_basic.png")
    plt.close(fig)


def demo_custom_city():
    """Demonstrate adding a custom city."""
    print("\n" + "="*80)
    print("DEMO 2: Adding Custom City")
    print("="*80)
    
    radar = SENCERadarChart()
    
    # Add hypothetical "Yenagoa" city
    print("\n➕ Adding custom city: Yenagoa")
    radar.city_data['Yenagoa'] = {
        'values': [0.65, 0.72, 0.68, 0.58, 0.64, 0.75, 0.62, 0.67],
        'mean_cvi': 0.66,
        'color': '#06D6A0',
        'linestyle': '-',
        'marker': 'D',
        'alpha': 0.25
    }
    
    print(f"  • Mean CVI: 0.66")
    print(f"  • Domain values: {radar.city_data['Yenagoa']['values']}")
    
    print("\n📈 Generating chart with custom city...")
    fig = radar.create_advanced_radar_chart()
    fig.savefig('/workspace/outputs/demo_custom_city.png', dpi=300)
    print("✓ Saved: demo_custom_city.png")
    plt.close(fig)


def demo_domain_analysis():
    """Demonstrate domain-specific analysis."""
    print("\n" + "="*80)
    print("DEMO 3: Domain-Specific Analysis")
    print("="*80)
    
    radar = SENCERadarChart()
    
    print("\n🔍 Analyzing Environmental Domain across cities:")
    env_domain_idx = 0  # Environmental Degradation
    
    for city, data in radar.city_data.items():
        env_score = data['values'][env_domain_idx]
        print(f"  • {city:15} - Environmental Score: {env_score:.3f}")
    
    # Find city with highest environmental vulnerability
    env_scores = {city: data['values'][env_domain_idx] 
                 for city, data in radar.city_data.items()}
    max_city = max(env_scores, key=env_scores.get)
    
    print(f"\n⚠️  Highest Environmental Vulnerability: {max_city} ({env_scores[max_city]:.3f})")
    
    # Create focused visualization
    print("\n📈 Creating environmental-focused visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    cities = list(env_scores.keys())
    scores = list(env_scores.values())
    colors = [radar.city_data[city]['color'] for city in cities]
    
    axes[0].bar(cities, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Environmental Degradation Score', fontweight='bold')
    axes[0].set_title('Environmental Vulnerability Comparison', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0.7, color='red', linestyle='--', label='High Risk Threshold')
    axes[0].legend()
    
    # Detailed breakdown
    domain_indices = [0, 7]  # Environmental and Ecological Feedback
    domain_names = ['Environmental\nDegradation', 'Ecological\nFeedback']
    
    x = np.arange(len(cities))
    width = 0.35
    
    for i, idx in enumerate(domain_indices):
        values = [radar.city_data[city]['values'][idx] for city in cities]
        axes[1].bar(x + i*width, values, width, label=domain_names[i], alpha=0.7)
    
    axes[1].set_xlabel('City', fontweight='bold')
    axes[1].set_ylabel('Score', fontweight='bold')
    axes[1].set_title('Environmental Domain Breakdown', fontweight='bold')
    axes[1].set_xticks(x + width/2)
    axes[1].set_xticklabels(cities)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('/workspace/outputs/demo_environmental_analysis.png', dpi=300)
    print("✓ Saved: demo_environmental_analysis.png")
    plt.close(fig)


def demo_temporal_comparison():
    """Demonstrate temporal comparison."""
    print("\n" + "="*80)
    print("DEMO 4: Temporal Evolution Simulation")
    print("="*80)
    
    radar = SENCERadarChart()
    
    print("\n📅 Simulating vulnerability changes over time...")
    print("   Scenario: 10% reduction after intervention")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2*np.pi, radar.num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, city in enumerate(radar.city_data.keys()):
        ax = axes[idx]
        
        # Current state
        values_current = radar.city_data[city]['values']
        values_current_plot = values_current + values_current[:1]
        
        # Projected state (10% improvement)
        values_future = [v * 0.9 for v in values_current]
        values_future_plot = values_future + values_future[:1]
        
        color = radar.city_data[city]['color']
        
        # Plot both states
        ax.plot(angles, values_current_plot, 'o-', linewidth=2, 
               label='Current (2024)', color=color, markersize=6)
        ax.plot(angles, values_future_plot, 's--', linewidth=2,
               label='Projected (2030)', color=color, alpha=0.6, markersize=6)
        ax.fill(angles, values_current_plot, alpha=0.15, color=color)
        
        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar.domains, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'{city}\nCVI: {radar.city_data[city]["mean_cvi"]:.2f} → ' +
                    f'{radar.city_data[city]["mean_cvi"]*0.9:.2f}',
                    fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('SENCE Framework: Temporal Evolution with Intervention (2024-2030)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('/workspace/outputs/demo_temporal_comparison.png', dpi=300)
    print("✓ Saved: demo_temporal_comparison.png")
    plt.close(fig)


def demo_correlation_matrix():
    """Demonstrate correlation analysis between domains."""
    print("\n" + "="*80)
    print("DEMO 5: Cross-Domain Correlation Analysis")
    print("="*80)
    
    radar = SENCERadarChart()
    
    # Create data matrix
    cities = list(radar.city_data.keys())
    domains = radar.domains
    
    data_matrix = np.array([radar.city_data[city]['values'] for city in cities])
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(data_matrix.T)
    
    print("\n🔗 Domain Correlation Matrix:")
    df_corr = pd.DataFrame(correlation_matrix, 
                          index=[d.replace('\n', ' ') for d in domains],
                          columns=[d.replace('\n', ' ') for d in domains])
    print(df_corr.round(3))
    
    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(correlation_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(domains)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels([d.replace('\n', ' ') for d in domains], rotation=45, ha='right')
    ax.set_yticklabels([d.replace('\n', ' ') for d in domains])
    
    # Add correlation values
    for i in range(len(domains)):
        for j in range(len(domains)):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                         ha='center', va='center', color='black', fontsize=9,
                         fontweight='bold' if abs(correlation_matrix[i, j]) > 0.7 else 'normal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=11)
    
    ax.set_title('SENCE Domain Correlation Matrix\nAcross Niger Delta Cities',
                fontweight='bold', fontsize=13, pad=20)
    
    plt.tight_layout()
    fig.savefig('/workspace/outputs/demo_correlation_matrix.png', dpi=300)
    print("\n✓ Saved: demo_correlation_matrix.png")
    plt.close(fig)


def demo_policy_targeting():
    """Demonstrate policy intervention targeting."""
    print("\n" + "="*80)
    print("DEMO 6: Policy Intervention Targeting")
    print("="*80)
    
    radar = SENCERadarChart()
    
    print("\n🎯 Identifying Priority Intervention Domains per City:\n")
    
    intervention_priorities = {}
    
    for city, data in radar.city_data.items():
        values = data['values']
        # Find top 3 highest vulnerability domains
        sorted_indices = np.argsort(values)[::-1][:3]
        
        print(f"{city}:")
        priorities = []
        for i, idx in enumerate(sorted_indices, 1):
            domain = radar.domains[idx].replace('\n', ' ')
            score = values[idx]
            print(f"  {i}. {domain:35} → Score: {score:.3f}")
            priorities.append((domain, score))
        
        intervention_priorities[city] = priorities
        print()
    
    # Visualize intervention priorities
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for idx, (city, priorities) in enumerate(intervention_priorities.items()):
        ax = axes[idx]
        
        domains = [p[0] for p in priorities]
        scores = [p[1] for p in priorities]
        colors = [radar.city_data[city]['color']] * 3
        
        bars = ax.barh(domains, scores, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center',
                   fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Vulnerability Score', fontweight='bold')
        ax.set_title(f'{city}\nTop 3 Priority Domains', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='High Priority Threshold')
        if idx == 0:
            ax.legend(fontsize=9)
    
    plt.suptitle('SENCE Framework: Policy Intervention Targeting Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig('/workspace/outputs/demo_policy_targeting.png', dpi=300)
    print("✓ Saved: demo_policy_targeting.png")
    plt.close(fig)


def demo_export_data():
    """Demonstrate data export capabilities."""
    print("\n" + "="*80)
    print("DEMO 7: Data Export and Reporting")
    print("="*80)
    
    radar = SENCERadarChart()
    
    # Export to various formats
    print("\n💾 Exporting data to multiple formats...")
    
    # CSV export (already done by save_outputs)
    print("✓ CSV: sence_vulnerability_data.csv")
    
    # Excel export
    df = pd.read_csv('/workspace/outputs/sence_vulnerability_data.csv')
    try:
        df.to_excel('/workspace/outputs/sence_vulnerability_data.xlsx', 
                   index=False, sheet_name='Vulnerability Data')
        print("✓ Excel: sence_vulnerability_data.xlsx")
    except ImportError:
        print("⚠ Excel: Skipped (openpyxl not installed - run: pip install openpyxl)")
    
    # JSON export
    import json
    json_data = {
        city: {
            'mean_cvi': data['mean_cvi'],
            'domains': {radar.domains[i].replace('\n', ' '): data['values'][i] 
                       for i in range(len(radar.domains))}
        }
        for city, data in radar.city_data.items()
    }
    
    with open('/workspace/outputs/sence_vulnerability_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print("✓ JSON: sence_vulnerability_data.json")
    
    # LaTeX table export
    try:
        latex_table = df.pivot(index='City', columns='Domain', 
                              values='Normalized_Contribution').to_latex(
            float_format='%.3f',
            caption='Normalized Domain Contributions to Mean CVI',
            label='tab:sence_domains'
        )
        
        with open('/workspace/outputs/sence_table.tex', 'w') as f:
            f.write(latex_table)
        print("✓ LaTeX: sence_table.tex")
    except ImportError:
        # Manual LaTeX generation without jinja2
        pivot_df = df.pivot(index='City', columns='Domain', values='Normalized_Contribution')
        latex_manual = "\\begin{table}[h]\n\\centering\n\\caption{Normalized Domain Contributions to Mean CVI}\n"
        latex_manual += "\\label{tab:sence_domains}\n\\begin{tabular}{l" + "c"*len(pivot_df.columns) + "}\n\\hline\n"
        latex_manual += " & " + " & ".join(pivot_df.columns) + " \\\\\n\\hline\n"
        for idx, row in pivot_df.iterrows():
            latex_manual += idx + " & " + " & ".join([f"{v:.3f}" for v in row]) + " \\\\\n"
        latex_manual += "\\hline\n\\end{tabular}\n\\end{table}"
        
        with open('/workspace/outputs/sence_table.tex', 'w') as f:
            f.write(latex_manual)
        print("✓ LaTeX: sence_table.tex (manual generation)")
    
    print("\n📊 Data Summary:")
    print(df.groupby('City')['Normalized_Contribution'].describe().round(3))


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("SENCE FRAMEWORK - INTERACTIVE DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates advanced features and customization options")
    print("for the SENCE vulnerability visualization system.\n")
    
    # Run all demos
    demo_basic_usage()
    demo_custom_city()
    demo_domain_analysis()
    demo_temporal_comparison()
    demo_correlation_matrix()
    demo_policy_targeting()
    demo_export_data()
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*80)
    print(f"\n✓ All outputs saved to: /workspace/outputs/")
    print("\nGenerated files:")
    print("  • demo_basic.png")
    print("  • demo_custom_city.png")
    print("  • demo_environmental_analysis.png")
    print("  • demo_temporal_comparison.png")
    print("  • demo_correlation_matrix.png")
    print("  • demo_policy_targeting.png")
    print("  • sence_vulnerability_data.xlsx")
    print("  • sence_vulnerability_data.json")
    print("  • sence_table.tex")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
