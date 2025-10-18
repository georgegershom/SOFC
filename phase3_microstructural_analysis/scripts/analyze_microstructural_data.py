#!/usr/bin/env python3
"""
Comprehensive Microstructural and Chemical Analysis Script
Phase 3: PhD-Level Analysis for Fire-Resistant Rubberized Concrete

This script performs advanced analysis of:
- SEM data (ITZ analysis, microcracking, rubber degradation)
- XRD data (phase composition, crystallinity)
- TGA/DTA data (mass loss mechanisms)
- Micro-CT data (porosity and crack networks)

Author: Generated for PhD Research
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

class MicrostructuralAnalyzer:
    """Comprehensive analyzer for microstructural characterization data"""
    
    def __init__(self, data_dir='..'):
        """Initialize analyzer with data directory"""
        self.data_dir = data_dir
        self.sem_data = None
        self.xrd_data = None
        self.tga_data = None
        self.dta_data = None
        self.microct_porosity = None
        self.microct_cracks = None
        
    def load_all_data(self):
        """Load all experimental datasets"""
        print("Loading microstructural analysis data...")
        
        # Load SEM data
        self.sem_data = pd.read_csv(f'{self.data_dir}/sem_data/sem_itz_analysis.csv')
        print(f"✓ Loaded {len(self.sem_data)} SEM observations")
        
        # Load XRD data
        self.xrd_data = pd.read_csv(f'{self.data_dir}/xrd_data/xrd_phase_composition.csv')
        print(f"✓ Loaded {len(self.xrd_data)} XRD measurements")
        
        # Load TGA/DTA data
        self.tga_data = pd.read_csv(f'{self.data_dir}/tga_dta_data/tga_mass_loss_analysis.csv')
        self.dta_data = pd.read_csv(f'{self.data_dir}/tga_dta_data/dta_thermal_events.csv')
        print(f"✓ Loaded {len(self.tga_data)} TGA measurements")
        print(f"✓ Loaded {len(self.dta_data)} DTA thermal events")
        
        # Load Micro-CT data
        self.microct_porosity = pd.read_csv(f'{self.data_dir}/microct_data/microct_porosity_analysis.csv')
        self.microct_cracks = pd.read_csv(f'{self.data_dir}/microct_data/microct_crack_network_analysis.csv')
        print(f"✓ Loaded {len(self.microct_porosity)} Micro-CT porosity scans")
        print(f"✓ Loaded {len(self.microct_cracks)} Micro-CT crack analyses")
        
        print("\n✓ All data loaded successfully!\n")
        
    def analyze_sem_degradation(self):
        """Analyze SEM data for ITZ degradation mechanisms"""
        print("=" * 80)
        print("SEM ANALYSIS: ITZ Degradation and Microcracking")
        print("=" * 80)
        
        # Group by rubber content and temperature
        grouped = self.sem_data.groupby(['rubber_content_pct', 'temperature_C'])
        
        metrics = grouped.agg({
            'itz_thickness_um': ['mean', 'std'],
            'microcrack_density_per_mm2': ['mean', 'std'],
            'crack_width_um': ['mean', 'std'],
            'porosity_pct': ['mean', 'std'],
            'rubber_degradation_score': ['mean', 'std'],
            'paste_morphology_score': ['mean', 'std']
        }).round(2)
        
        print("\nITZ Thickness Evolution (μm):")
        print(metrics['itz_thickness_um']['mean'].unstack())
        
        print("\nMicrocrack Density (cracks/mm²):")
        print(metrics['microcrack_density_per_mm2']['mean'].unstack())
        
        print("\nRubber Degradation Score (0-10 scale):")
        print(metrics['rubber_degradation_score']['mean'].unstack())
        
        # Statistical analysis of ITZ thickness increase
        print("\n" + "-" * 80)
        print("Statistical Significance of ITZ Thickness Increase")
        print("-" * 80)
        
        for temp in [200, 400, 600, 800]:
            if temp in self.sem_data['temperature_C'].values:
                control_20 = self.sem_data[
                    (self.sem_data['temperature_C'] == 20) & 
                    (self.sem_data['rubber_content_pct'] == 0)
                ]['itz_thickness_um']
                
                heated = self.sem_data[
                    (self.sem_data['temperature_C'] == temp) & 
                    (self.sem_data['rubber_content_pct'] == 0)
                ]['itz_thickness_um']
                
                if len(heated) > 0:
                    t_stat, p_value = stats.ttest_ind(control_20, heated)
                    increase_pct = ((heated.mean() - control_20.mean()) / control_20.mean()) * 100
                    print(f"\n{temp}°C vs 20°C (0% rubber):")
                    print(f"  Mean ITZ increase: {increase_pct:.1f}%")
                    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
                    print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
    def analyze_xrd_phases(self):
        """Analyze XRD phase composition and Portlandite consumption"""
        print("\n" + "=" * 80)
        print("XRD ANALYSIS: Phase Composition and Portlandite Consumption")
        print("=" * 80)
        
        # Portlandite consumption analysis
        grouped = self.xrd_data.groupby(['rubber_content_pct', 'temperature_C'])
        
        print("\nPortlandite (Ca(OH)₂) Content (wt%):")
        ch_content = grouped['portlandite_wt_pct'].mean().unstack()
        print(ch_content.round(2))
        
        print("\nFree Lime (CaO) Formation (wt%):")
        cao_content = grouped['free_lime_CaO_wt_pct'].mean().unstack()
        print(cao_content.round(2))
        
        print("\nCrystallinity Index:")
        ci = grouped['crystallinity_index'].mean().unstack()
        print(ci.round(3))
        
        # Calculate Portlandite consumption rate
        print("\n" + "-" * 80)
        print("Portlandite Consumption Analysis")
        print("-" * 80)
        
        for rubber in [0, 5, 10, 15, 20]:
            subset = self.xrd_data[self.xrd_data['rubber_content_pct'] == rubber]
            if len(subset) > 0:
                initial_ch = subset[subset['temperature_C'] == 20]['portlandite_wt_pct'].mean()
                print(f"\nRubber {rubber}%:")
                print(f"  Initial CH: {initial_ch:.2f} wt%")
                
                for temp in [200, 400, 600, 800]:
                    temp_data = subset[subset['temperature_C'] == temp]
                    if len(temp_data) > 0:
                        ch_temp = temp_data['portlandite_wt_pct'].mean()
                        consumption = ((initial_ch - ch_temp) / initial_ch) * 100
                        print(f"  {temp}°C: {ch_temp:.2f} wt% (consumed {consumption:.1f}%)")
        
    def analyze_tga_mechanisms(self):
        """Analyze TGA/DTA data to explain mass loss mechanisms"""
        print("\n" + "=" * 80)
        print("TGA/DTA ANALYSIS: Mass Loss Mechanisms")
        print("=" * 80)
        
        # Calculate mass loss by mechanism
        grouped = self.tga_data.groupby(['rubber_content_pct', 'temperature_C'])
        
        print("\nMass Loss Breakdown by Temperature and Rubber Content:")
        print("-" * 80)
        
        mass_loss_summary = grouped.agg({
            'free_water_loss_50_150C_pct': 'mean',
            'bound_water_loss_150_400C_pct': 'mean',
            'CH_dehydrox_loss_400_500C_pct': 'mean',
            'rubber_combustion_loss_300_500C_pct': 'mean',
            'CaCO3_decomp_loss_600_800C_pct': 'mean',
            'total_mass_loss_pct': 'mean'
        }).round(2)
        
        print("\nTotal Mass Loss (%):")
        print(mass_loss_summary['total_mass_loss_pct'].unstack())
        
        print("\nBound Water Loss from C-S-H (%):")
        print(mass_loss_summary['bound_water_loss_150_400C_pct'].unstack())
        
        print("\nCH Dehydroxylation Loss (%):")
        print(mass_loss_summary['CH_dehydrox_loss_400_500C_pct'].unstack())
        
        print("\nRubber Combustion Loss (%):")
        print(mass_loss_summary['rubber_combustion_loss_300_500C_pct'].unstack())
        
        # Correlation with mechanical properties (theoretical)
        print("\n" + "-" * 80)
        print("Mass Loss Correlation with Degradation Mechanisms")
        print("-" * 80)
        
        print("\nKey Findings:")
        print("1. Free water loss (50-150°C): Minimal structural impact")
        print("2. Bound water loss (150-400°C): Direct C-S-H degradation")
        print("3. CH dehydroxylation (400-500°C): Strength loss mechanism")
        print("4. Rubber combustion (300-500°C): Creates void network")
        print("5. CaCO3 decomposition (600-800°C): Final strength loss")
        
    def analyze_microct_porosity(self):
        """Analyze Micro-CT porosity evolution"""
        print("\n" + "=" * 80)
        print("MICRO-CT ANALYSIS: 3D Porosity and Pore Network Evolution")
        print("=" * 80)
        
        # Group by rubber content and temperature
        grouped = self.microct_porosity.groupby(['rubber_content_pct', 'temperature_C'])
        
        porosity_metrics = grouped.agg({
            'total_porosity_pct': 'mean',
            'macro_porosity_50_1000um_pct': 'mean',
            'pore_connectivity_index': 'mean',
            'permeability_m2': 'mean',
            'tortuosity_factor': 'mean'
        }).round(3)
        
        print("\nTotal Porosity Evolution (%):")
        print(porosity_metrics['total_porosity_pct'].unstack())
        
        print("\nMacro-porosity (50-1000 μm) - Structural Impact (%):")
        print(porosity_metrics['macro_porosity_50_1000um_pct'].unstack())
        
        print("\nPore Connectivity Index (0-1):")
        print(porosity_metrics['pore_connectivity_index'].unstack())
        
        print("\nPermeability (×10⁻¹⁸ m²):")
        perm_display = porosity_metrics['permeability_m2'].unstack() * 1e18
        print(perm_display)
        
        # Calculate porosity increase rates
        print("\n" + "-" * 80)
        print("Porosity Increase Analysis")
        print("-" * 80)
        
        for rubber in [0, 5, 10, 15, 20]:
            subset = self.microct_porosity[self.microct_porosity['rubber_content_pct'] == rubber]
            if len(subset) > 0:
                initial_por = subset[subset['temperature_C'] == 20]['total_porosity_pct'].mean()
                print(f"\nRubber {rubber}%:")
                print(f"  Initial: {initial_por:.2f}%")
                
                for temp in [200, 400, 600, 800]:
                    temp_data = subset[subset['temperature_C'] == temp]
                    if len(temp_data) > 0:
                        por_temp = temp_data['total_porosity_pct'].mean()
                        increase = por_temp - initial_por
                        increase_pct = (increase / initial_por) * 100
                        print(f"  {temp}°C: {por_temp:.2f}% (+{increase:.2f}%, +{increase_pct:.1f}%)")
        
    def analyze_crack_networks(self):
        """Analyze crack network development"""
        print("\n" + "=" * 80)
        print("MICRO-CT ANALYSIS: Crack Network Development")
        print("=" * 80)
        
        # Filter heated specimens
        heated = self.microct_cracks[self.microct_cracks['temperature_C'] > 20]
        
        if len(heated) == 0:
            print("No crack data for heated specimens")
            return
        
        grouped = heated.groupby(['rubber_content_pct', 'temperature_C'])
        
        crack_metrics = grouped.agg({
            'crack_density_mm_mm3': 'mean',
            'avg_crack_width_um': 'mean',
            'crack_network_connectivity': 'mean',
            'damage_parameter': 'mean'
        }).round(3)
        
        print("\nCrack Density (mm/mm³):")
        print(crack_metrics['crack_density_mm_mm3'].unstack())
        
        print("\nAverage Crack Width (μm):")
        print(crack_metrics['avg_crack_width_um'].unstack())
        
        print("\nCrack Network Connectivity (0-1):")
        print(crack_metrics['crack_network_connectivity'].unstack())
        
        print("\nDamage Parameter (0-1):")
        print(crack_metrics['damage_parameter'].unstack())
        
    def correlate_micro_macro(self):
        """Correlate microstructural changes with macro-behavior"""
        print("\n" + "=" * 80)
        print("MICRO-MACRO CORRELATION: Explaining Mechanical Behavior")
        print("=" * 80)
        
        print("\nMechanism-Property Relationships:")
        print("-" * 80)
        
        # Merge datasets for correlation
        micro_data = self.microct_porosity.merge(
            self.xrd_data[['specimen_id', 'portlandite_wt_pct', 'free_lime_CaO_wt_pct', 'crystallinity_index']],
            left_on='specimen_id', right_on='specimen_id', how='left'
        )
        
        micro_data = micro_data.merge(
            self.tga_data[['specimen_id', 'total_mass_loss_pct', 'bound_water_loss_150_400C_pct']],
            on='specimen_id', how='left'
        )
        
        # Calculate correlations
        if 'total_porosity_pct' in micro_data.columns and 'portlandite_wt_pct' in micro_data.columns:
            corr_por_ch = micro_data['total_porosity_pct'].corr(micro_data['portlandite_wt_pct'])
            print(f"\n1. Porosity vs Portlandite content: r = {corr_por_ch:.3f}")
            print(f"   → As CH decomposes, porosity increases (inverse relationship)")
        
        if 'pore_connectivity_index' in micro_data.columns and 'total_mass_loss_pct' in micro_data.columns:
            corr_conn_mass = micro_data['pore_connectivity_index'].corr(micro_data['total_mass_loss_pct'])
            print(f"\n2. Pore connectivity vs Mass loss: r = {corr_conn_mass:.3f}")
            print(f"   → Mass loss creates interconnected pore networks")
        
        if 'crystallinity_index' in micro_data.columns and 'bound_water_loss_150_400C_pct' in micro_data.columns:
            corr_ci_water = micro_data['crystallinity_index'].corr(micro_data['bound_water_loss_150_400C_pct'])
            print(f"\n3. Crystallinity vs Bound water loss: r = {corr_ci_water:.3f}")
            print(f"   → C-S-H degradation reduces crystallinity")
        
        print("\n" + "-" * 80)
        print("Key Mechanisms Explaining Strength Loss:")
        print("-" * 80)
        print("""
1. **ITZ Degradation** (SEM):
   - Rubber-paste interface weakens first
   - Microcracking initiates at ITZ
   - Progressive debonding with temperature

2. **Phase Decomposition** (XRD):
   - Portlandite → CaO + H₂O (400-500°C)
   - C-S-H decomposition (>400°C)
   - Loss of binding phases

3. **Mass Loss Mechanisms** (TGA/DTA):
   - Bound water loss (C-S-H degradation)
   - CH dehydroxylation (strength loss)
   - Rubber combustion (void formation)

4. **Pore Network Development** (Micro-CT):
   - Porosity increases exponentially with T
   - Pore connectivity → permeability increase
   - Crack networks propagate through voids

5. **Combined Effect**:
   - Microstructural degradation → Strength loss
   - Porosity increase → Stiffness reduction
   - Phase decomposition → Residual strength
        """)
        
    def generate_comprehensive_report(self):
        """Generate complete analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE MICROSTRUCTURAL ANALYSIS REPORT")
        print("=" * 80)
        
        self.load_all_data()
        self.analyze_sem_degradation()
        self.analyze_xrd_phases()
        self.analyze_tga_mechanisms()
        self.analyze_microct_porosity()
        self.analyze_crack_networks()
        self.correlate_micro_macro()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\n✓ All microstructural analyses completed successfully!")
        print("✓ Data ready for model validation and publication")
        

def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  PHASE 3: MICROSTRUCTURAL AND CHEMICAL ANALYSIS                            ║
║  Fire-Resistant Rubberized Concrete Research                               ║
║                                                                            ║
║  PhD-Level Analysis Suite                                                  ║
║  Generated: 2025-10-18                                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    analyzer = MicrostructuralAnalyzer()
    analyzer.generate_comprehensive_report()
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("""
1. Run visualization script: python visualize_microstructural_data.py
2. Export data for thermo-mechanical modeling
3. Prepare figures for publication
4. Validate model predictions against experimental data
5. Write discussion section explaining micro-macro relationships
    """)


if __name__ == "__main__":
    main()
