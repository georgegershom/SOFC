#!/usr/bin/env python3
"""
SOFC Post-Processing Script
Extracts and analyzes results from Abaqus ODB files
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

def extract_temperature_field(odb_path):
    """Extract temperature field from ODB"""
    try:
        from abaqus import *
        from abaqusConstants import *
        from odbAccess import *
        
        # Open ODB
        odb = openOdb(odb_path)
        
        # Get temperature field from heat transfer step
        heat_step = odb.steps['HEAT_TRANSFER']
        temp_frame = heat_step.frames[-1]  # Last frame
        
        # Extract temperature
        temp_field = temp_frame.fieldOutputs['NT']
        temperatures = []
        coordinates = []
        
        for value in temp_field.values:
            temperatures.append(value.data)
            coordinates.append(value.nodeLabel)
        
        odb.close()
        
        return np.array(temperatures), np.array(coordinates)
        
    except ImportError:
        print("Abaqus Python API not available. Using mock data.")
        # Return mock data for demonstration
        n_nodes = 1000
        return np.random.normal(800, 100, n_nodes), np.arange(n_nodes)
    except Exception as e:
        print(f"Error extracting temperature: {e}")
        return None, None

def extract_stress_field(odb_path):
    """Extract stress field from ODB"""
    try:
        from abaqus import *
        from abaqusConstants import *
        from odbAccess import *
        
        # Open ODB
        odb = openOdb(odb_path)
        
        # Get stress field from thermo-mechanical step
        mech_step = odb.steps['THERMO_MECHANICAL']
        stress_frame = mech_step.frames[-1]  # Last frame
        
        # Extract von Mises stress
        stress_field = stress_frame.fieldOutputs['S']
        von_mises = []
        coordinates = []
        
        for value in stress_field.values:
            s = value.data
            vm = np.sqrt(0.5 * ((s[0] - s[1])**2 + (s[1] - s[2])**2 + (s[2] - s[0])**2 + 6*(s[3]**2 + s[4]**2 + s[5]**2)))
            von_mises.append(vm)
            coordinates.append(value.nodeLabel)
        
        odb.close()
        
        return np.array(von_mises), np.array(coordinates)
        
    except ImportError:
        print("Abaqus Python API not available. Using mock data.")
        # Return mock data for demonstration
        n_nodes = 1000
        return np.random.exponential(50e6, n_nodes), np.arange(n_nodes)
    except Exception as e:
        print(f"Error extracting stress: {e}")
        return None, None

def compute_damage_proxy(stress, coordinates, geometry):
    """Compute damage proxy based on stress and interface proximity"""
    
    # Damage parameters
    sigma_th = 120e6  # Pa - threshold stress
    k_D = 1.5e-5
    p = 2.0
    
    # Interface positions (in meters)
    y_ae = 0.40e-3  # anode-electrolyte
    y_ec = 0.50e-3  # electrolyte-cathode  
    y_ci = 0.90e-3  # cathode-interconnect
    
    damage = np.zeros_like(stress)
    
    for i, (s, coord) in enumerate(zip(stress, coordinates)):
        # Get y-coordinate (assuming 2D, y is second coordinate)
        y = coord[1] if len(coord) > 1 else 0.0
        
        # Interface proximity weight
        w_iface = 0.0
        if abs(y - y_ae) < 0.02e-3:  # Within 0.02 mm of anode-electrolyte
            w_iface = 1.0
        elif abs(y - y_ec) < 0.02e-3:  # Within 0.02 mm of electrolyte-cathode
            w_iface = 1.0
        elif abs(y - y_ci) < 0.02e-3:  # Within 0.02 mm of cathode-interconnect
            w_iface = 1.0
        
        # Damage calculation
        if s > sigma_th:
            damage[i] = k_D * ((s - sigma_th) / sigma_th)**p * (1 + 3 * w_iface)
    
    return np.clip(damage, 0, 1)  # Cap at 1.0

def compute_delamination_proxy(stress, coordinates, geometry):
    """Compute delamination proxy based on interfacial shear stress"""
    
    # Critical shear stress thresholds (Pa)
    tau_crit_ae = 25e6  # anode-electrolyte
    tau_crit_ec = 20e6  # electrolyte-cathode
    tau_crit_ci = 30e6  # cathode-interconnect
    
    delamination = np.zeros_like(stress)
    
    for i, (s, coord) in enumerate(zip(stress, coordinates)):
        y = coord[1] if len(coord) > 1 else 0.0
        
        # Determine which interface and critical stress
        if abs(y - 0.40e-3) < 0.01e-3:  # Near anode-electrolyte
            tau_crit = tau_crit_ae
        elif abs(y - 0.50e-3) < 0.01e-3:  # Near electrolyte-cathode
            tau_crit = tau_crit_ec
        elif abs(y - 0.90e-3) < 0.01e-3:  # Near cathode-interconnect
            tau_crit = tau_crit_ci
        else:
            continue
        
        # Shear stress proxy (using von Mises as approximation)
        tau = s / np.sqrt(3)  # Convert von Mises to shear
        
        if tau > tau_crit:
            delamination[i] = (tau - tau_crit) / tau_crit
    
    return delamination

def create_visualization(temperature, stress, damage, delamination, output_dir):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature field
    axes[0,0].hist(temperature, bins=50, alpha=0.7, color='red')
    axes[0,0].set_xlabel('Temperature (°C)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Temperature Distribution')
    axes[0,0].grid(True)
    
    # Stress field
    axes[0,1].hist(stress/1e6, bins=50, alpha=0.7, color='blue')
    axes[0,1].set_xlabel('Von Mises Stress (MPa)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Stress Distribution')
    axes[0,1].grid(True)
    
    # Damage proxy
    axes[1,0].hist(damage, bins=50, alpha=0.7, color='orange')
    axes[1,0].set_xlabel('Damage Proxy')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Damage Distribution')
    axes[1,0].grid(True)
    
    # Delamination proxy
    axes[1,1].hist(delamination, bins=50, alpha=0.7, color='green')
    axes[1,1].set_xlabel('Delamination Proxy')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Delamination Distribution')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sofc_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_simulation_results(job_name, output_dir):
    """Process results for a single simulation"""
    
    print(f"Processing results for {job_name}...")
    
    # Paths
    odb_path = output_dir / f"{job_name}.odb"
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Extract data
    print("  Extracting temperature field...")
    temperature, temp_coords = extract_temperature_field(str(odb_path))
    
    print("  Extracting stress field...")
    stress, stress_coords = extract_stress_field(str(odb_path))
    
    if temperature is None or stress is None:
        print(f"  Warning: Could not extract data for {job_name}")
        return None
    
    # Compute proxies
    print("  Computing damage proxy...")
    damage = compute_damage_proxy(stress, stress_coords, {})
    
    print("  Computing delamination proxy...")
    delamination = compute_delamination_proxy(stress, stress_coords, {})
    
    # Create visualizations
    print("  Creating visualizations...")
    create_visualization(temperature, stress, damage, delamination, results_dir)
    
    # Save results
    results = {
        'job_name': job_name,
        'temperature': {
            'mean': float(np.mean(temperature)),
            'max': float(np.max(temperature)),
            'min': float(np.min(temperature)),
            'std': float(np.std(temperature))
        },
        'stress': {
            'mean': float(np.mean(stress)),
            'max': float(np.max(stress)),
            'min': float(np.min(stress)),
            'std': float(np.std(stress))
        },
        'damage': {
            'mean': float(np.mean(damage)),
            'max': float(np.max(damage)),
            'damaged_nodes': int(np.sum(damage > 0.1))
        },
        'delamination': {
            'mean': float(np.mean(delamination)),
            'max': float(np.max(delamination)),
            'critical_nodes': int(np.sum(delamination > 0.5))
        }
    }
    
    # Save to JSON
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Results saved to {results_dir}")
    return results

def main():
    """Main post-processing function"""
    
    print("SOFC Post-Processing")
    print("="*50)
    
    base_dir = Path("/workspace/sofc_simulation")
    output_base = base_dir / "outputs"
    
    # Process all simulations
    all_results = {}
    
    for job_name in ['sofc_hr1', 'sofc_hr4', 'sofc_hr10']:
        output_dir = output_base / job_name
        
        if output_dir.exists():
            results = process_simulation_results(job_name, output_dir)
            if results:
                all_results[job_name] = results
        else:
            print(f"Warning: Output directory {output_dir} not found")
    
    # Summary
    print(f"\n{'='*60}")
    print("POST-PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    for job_name, results in all_results.items():
        print(f"\n{job_name}:")
        print(f"  Temperature: {results['temperature']['mean']:.1f}°C ± {results['temperature']['std']:.1f}°C")
        print(f"  Max Stress: {results['stress']['max']/1e6:.1f} MPa")
        print(f"  Damage: {results['damage']['damaged_nodes']} nodes > 0.1")
        print(f"  Delamination: {results['delamination']['critical_nodes']} critical nodes")

if __name__ == "__main__":
    main()