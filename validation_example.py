#!/usr/bin/env python3
"""
Validation Example: Quick demonstration of the sintering simulation framework
"""

import numpy as np
import matplotlib.pyplot as plt
from sintering_simulation import SinteringSimulator, ThermalProfile, ParetoOptimizer

def quick_validation():
    """Run a quick validation of the simulation framework"""
    
    print("🔬 SOFC Sintering Simulation - Quick Validation")
    print("=" * 50)
    
    # Initialize simulator
    simulator = SinteringSimulator()
    
    # Test different process conditions
    test_profiles = [
        ("Conservative", ThermalProfile(0.8, 900, 120)),
        ("Standard", ThermalProfile(1.5, 1000, 90)),
        ("Aggressive", ThermalProfile(2.5, 1100, 60))
    ]
    
    results = []
    
    for name, profile in test_profiles:
        print(f"\n📊 Testing {name} Profile:")
        print(f"   Ramp Rate: {profile.ramp_rate}°C/min")
        print(f"   Soak Temp: {profile.soak_temp}°C")
        print(f"   Soak Time: {profile.soak_time} min")
        
        result = simulator.run_simulation(profile)
        
        print(f"   → Residual Strain: {result['residual_strain']:.1f} µε")
        print(f"   → Warpage: {result['warpage']:.1f} µm")
        
        results.append((name, result))
    
    # Quick visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot thermal profiles
    colors = ['blue', 'orange', 'red']
    for i, (name, result) in enumerate(results):
        ax1.plot(result['time'], result['temperature'], 
                color=colors[i], linewidth=2, label=name)
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Thermal Profiles Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot trade-off
    strains = [r[1]['residual_strain'] for r in results]
    warpages = [r[1]['warpage'] for r in results]
    names = [r[0] for r in results]
    
    scatter = ax2.scatter(strains, warpages, c=colors, s=100, alpha=0.7)
    
    for i, name in enumerate(names):
        ax2.annotate(name, (strains[i], warpages[i]), 
                    xytext=(10, 10), textcoords='offset points')
    
    ax2.set_xlabel('Residual Strain (µε)')
    ax2.set_ylabel('Warpage (µm)')
    ax2.set_title('Stress-Warpage Trade-off')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/validation_results.png', dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Validation complete! Results saved to validation_results.png")
    
    # Performance analysis
    print(f"\n📈 Performance Analysis:")
    print(f"   Best strain: {min(strains):.1f} µε ({names[strains.index(min(strains))]})")
    print(f"   Best warpage: {min(warpages):.1f} µm ({names[warpages.index(min(warpages))]})")
    
    # Process recommendations
    print(f"\n💡 Process Recommendations:")
    if min(strains) < 150:
        print("   ✓ Excellent stress control achieved")
    elif min(strains) < 250:
        print("   ⚠ Moderate stress levels - consider optimization")
    else:
        print("   ❌ High stress levels - process modification needed")
    
    if min(warpages) < 10:
        print("   ✓ Excellent shape fidelity achieved")
    elif min(warpages) < 20:
        print("   ⚠ Moderate warpage - acceptable for most applications")
    else:
        print("   ❌ High warpage - geometric tolerance issues likely")
    
    return results

if __name__ == "__main__":
    results = quick_validation()