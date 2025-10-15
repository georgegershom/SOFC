#!/usr/bin/env python3
"""
SOFC Dataset Usage Examples
===========================

This script demonstrates how to load and use the multi-fidelity SOFC dataset
for various machine learning and modeling applications.

Author: Generated for PhD Thesis - Multi-Fidelity Digital Twin for SOFCs
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_sofc_dataset(dataset_dir: str = "sofc_dataset") -> Dict[str, Any]:
    """
    Load the complete SOFC dataset.
    
    Returns:
        Dictionary containing all dataset components
    """
    dataset_dir = Path(dataset_dir)
    
    # Load parameter datasets
    datasets = {}
    for fidelity in ['lf', 'mf', 'hf']:
        file_path = dataset_dir / f"sofc_parameters_{fidelity}_fidelity.csv"
        if file_path.exists():
            datasets[fidelity.upper()] = pd.read_csv(file_path)
    
    # Load combined dataset
    combined_path = dataset_dir / "sofc_parameters_combined.csv"
    if combined_path.exists():
        datasets['combined'] = pd.read_csv(combined_path)
    
    # Load microstructural data
    micro_path = dataset_dir / "sofc_microstructural_data.json"
    microstructural_data = {}
    if micro_path.exists():
        with open(micro_path, 'r') as f:
            microstructural_data = json.load(f)
    
    # Load transient profiles
    transient_path = dataset_dir / "sofc_transient_profiles.json"
    transient_profiles = {}
    if transient_path.exists():
        with open(transient_path, 'r') as f:
            transient_profiles = json.load(f)
    
    # Load metadata
    metadata_path = dataset_dir / "dataset_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return {
        'parameters': datasets,
        'microstructural': microstructural_data,
        'transient': transient_profiles,
        'metadata': metadata
    }

def example_1_basic_data_exploration():
    """Example 1: Basic data exploration and statistics."""
    print("="*60)
    print("EXAMPLE 1: Basic Data Exploration")
    print("="*60)
    
    # Load dataset
    data = load_sofc_dataset()
    
    # Explore parameter datasets
    print("\n📊 Parameter Dataset Overview:")
    for fidelity, df in data['parameters'].items():
        if fidelity != 'combined':
            print(f"- {fidelity} Fidelity: {len(df)} samples, {len(df.columns)} parameters")
    
    # Look at key system parameters
    hf_data = data['parameters']['HF']
    key_params = ['fuel_utilization', 'current_density', 'temperature', 'voltage']
    
    print(f"\n📈 Key Parameter Statistics (HF Dataset):")
    for param in key_params:
        if param in hf_data.columns:
            mean_val = hf_data[param].mean()
            std_val = hf_data[param].std()
            print(f"- {param}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    # Microstructural data overview
    print(f"\n🔬 Microstructural Data:")
    print(f"- Total samples: {len(data['microstructural'])}")
    
    # Sample microstructural data
    sample_key = list(data['microstructural'].keys())[0]
    sample_data = data['microstructural'][sample_key]
    print(f"- Sample voxel size: {sample_data['voxel_size_nm']} nm")
    print(f"- Phase fractions: Ni={sample_data['phase_fractions']['ni']:.3f}, "
          f"YSZ={sample_data['phase_fractions']['ysz']:.3f}, "
          f"Pore={sample_data['phase_fractions']['pore']:.3f}")

def example_2_multi_fidelity_modeling():
    """Example 2: Prepare data for multi-fidelity modeling."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Fidelity Modeling Preparation")
    print("="*60)
    
    # Load dataset
    data = load_sofc_dataset()
    
    # Define common input features across all fidelity levels
    common_features = [
        'fuel_utilization', 'oxidant_utilization', 'current_density',
        'temperature', 'pressure', 'h2_fraction'
    ]
    
    # Prepare datasets for each fidelity level
    X_datasets = {}
    
    for fidelity in ['LF', 'MF', 'HF']:
        if fidelity in data['parameters']:
            df = data['parameters'][fidelity]
            
            # Extract common features
            X_common = df[common_features].values
            
            # Add fidelity-specific features
            if fidelity in ['MF', 'HF']:
                # Add geometry features
                geometry_features = [col for col in df.columns 
                                   if any(keyword in col for keyword in 
                                         ['thickness', 'area', 'channel', 'rib'])]
                if geometry_features:
                    X_geometry = df[geometry_features].values
                    X_combined = np.hstack([X_common, X_geometry])
                else:
                    X_combined = X_common
            else:
                X_combined = X_common
            
            X_datasets[fidelity] = X_combined
            
            print(f"- {fidelity} Dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    
    # Example: Create synthetic target variables (e.g., performance metrics)
    np.random.seed(42)
    y_datasets = {}
    
    for fidelity, X in X_datasets.items():
        # Synthetic performance metric based on operating conditions
        # (In real application, this would be simulation/experimental results)
        
        # Extract key features for synthetic target
        fuel_util = data['parameters'][fidelity]['fuel_utilization'].values
        current = data['parameters'][fidelity]['current_density'].values
        temp = data['parameters'][fidelity]['temperature'].values
        
        # Synthetic power density (simplified relationship)
        power_density = (fuel_util * current * (temp - 973) / 300 + 
                        np.random.normal(0, 0.1, len(fuel_util)))
        
        y_datasets[fidelity] = power_density
        
        print(f"- {fidelity} Target: Power density range [{power_density.min():.3f}, {power_density.max():.3f}]")
    
    return X_datasets, y_datasets

def example_3_degradation_analysis():
    """Example 3: Analyze transient profiles for degradation studies."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Degradation Analysis with Transient Profiles")
    print("="*60)
    
    # Load dataset
    data = load_sofc_dataset()
    
    # Analyze transient profiles
    transient_data = data['transient']
    
    # Group profiles by type
    profile_types = {}
    for key, profile in transient_data.items():
        profile_type = profile['type']
        if profile_type not in profile_types:
            profile_types[profile_type] = []
        profile_types[profile_type].append(profile)
    
    print(f"\n⏱️  Transient Profile Analysis:")
    
    for profile_type, profiles in profile_types.items():
        print(f"\n- {profile_type.replace('_', ' ').title()} Profiles: {len(profiles)}")
        
        # Calculate stress indicators for first few profiles
        stress_indicators = []
        
        for i, profile in enumerate(profiles[:5]):  # Analyze first 5 profiles
            temps = np.array(profile['temperature'])
            currents = np.array(profile['current_density'])
            
            # Calculate thermal stress indicator (temperature gradient)
            temp_gradient = np.max(np.diff(temps))
            
            # Calculate electrochemical stress (current ramp rate)
            current_ramp = np.max(np.abs(np.diff(currents)))
            
            # Combined stress indicator
            stress_indicator = temp_gradient / 100 + current_ramp * 10
            stress_indicators.append(stress_indicator)
            
            print(f"  Profile {i+1}: Thermal gradient={temp_gradient:.1f} K, "
                  f"Current ramp={current_ramp:.3f} A/cm², Stress={stress_indicator:.3f}")
        
        avg_stress = np.mean(stress_indicators)
        print(f"  Average stress indicator: {avg_stress:.3f}")

def example_4_microstructural_analysis():
    """Example 4: Microstructural data analysis for high-fidelity modeling."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Microstructural Analysis for High-Fidelity Modeling")
    print("="*60)
    
    # Load dataset
    data = load_sofc_dataset()
    
    microstructural_data = data['microstructural']
    
    # Extract microstructural parameters
    samples = []
    for sample_id, sample_data in microstructural_data.items():
        sample_features = {
            'sample_id': sample_id,
            'voxel_size': sample_data['voxel_size_nm'],
            'ni_fraction': sample_data['phase_fractions']['ni'],
            'ysz_fraction': sample_data['phase_fractions']['ysz'],
            'pore_fraction': sample_data['phase_fractions']['pore'],
            'ni_connectivity': sample_data['connectivity']['ni'],
            'ysz_connectivity': sample_data['connectivity']['ysz'],
            'pore_connectivity': sample_data['connectivity']['pore'],
            'specific_surface_area': sample_data['specific_surface_area'],
            'reconstruction_method': sample_data['reconstruction_method']
        }
        samples.append(sample_features)
    
    # Convert to DataFrame for analysis
    micro_df = pd.DataFrame(samples)
    
    print(f"\n🔬 Microstructural Dataset Overview:")
    print(f"- Total samples: {len(micro_df)}")
    print(f"- FIB-SEM samples: {len(micro_df[micro_df['reconstruction_method'] == 'FIB-SEM'])}")
    print(f"- X-Ray CT samples: {len(micro_df[micro_df['reconstruction_method'] == 'X-Ray_CT'])}")
    
    # Calculate effective properties
    print(f"\n📊 Effective Property Calculations:")
    
    # Effective conductivity estimation (simplified)
    micro_df['effective_ionic_conductivity'] = (
        micro_df['ysz_fraction'] * micro_df['ysz_connectivity'] * 0.1  # Base YSZ conductivity
    )
    
    micro_df['effective_electronic_conductivity'] = (
        micro_df['ni_fraction'] * micro_df['ni_connectivity'] * 1e5  # Base Ni conductivity
    )
    
    # Triple phase boundary density estimation
    micro_df['tpb_density_estimate'] = (
        micro_df['specific_surface_area'] * 
        micro_df['ni_fraction'] * 
        micro_df['ysz_fraction'] / 1e6
    )
    
    print(f"- Effective ionic conductivity range: "
          f"{micro_df['effective_ionic_conductivity'].min():.4f} - "
          f"{micro_df['effective_ionic_conductivity'].max():.4f} S/m")
    
    print(f"- TPB density estimate range: "
          f"{micro_df['tpb_density_estimate'].min():.2e} - "
          f"{micro_df['tpb_density_estimate'].max():.2e} m/m³")
    
    # Identify high-performance microstructures
    performance_score = (
        micro_df['effective_ionic_conductivity'] * 
        micro_df['effective_electronic_conductivity'] * 
        micro_df['tpb_density_estimate']
    )
    
    # Add performance score to dataframe
    micro_df['performance_score'] = performance_score
    top_performers = micro_df.nlargest(5, 'performance_score')
    
    print(f"\n🏆 Top 5 High-Performance Microstructures:")
    for i, (idx, sample) in enumerate(top_performers.iterrows()):
        print(f"{i+1}. {sample['sample_id']}: "
              f"Ni={sample['ni_fraction']:.3f}, "
              f"Porosity={sample['pore_fraction']:.3f}, "
              f"Performance Score={sample['performance_score']:.2e}")

def example_5_integrated_workflow():
    """Example 5: Integrated multi-scale modeling workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Integrated Multi-Scale Modeling Workflow")
    print("="*60)
    
    # Load complete dataset
    data = load_sofc_dataset()
    
    # Step 1: System-level analysis (LF)
    print(f"\n🔧 Step 1: System-Level Analysis (Low Fidelity)")
    lf_data = data['parameters']['LF']
    
    # Select operating point
    operating_point = lf_data.iloc[0]  # First sample as example
    print(f"- Operating temperature: {operating_point['temperature']:.1f} K")
    print(f"- Current density: {operating_point['current_density']:.3f} A/cm²")
    print(f"- Fuel utilization: {operating_point['fuel_utilization']:.3f}")
    
    # Step 2: Cell-level analysis (MF)
    print(f"\n🏗️ Step 2: Cell-Level Analysis (Medium Fidelity)")
    mf_data = data['parameters']['MF']
    
    # Find matching MF sample (simplified matching)
    mf_sample = mf_data.iloc[0]
    print(f"- Cell active area: {mf_sample['cell_active_area']:.1f} cm²")
    print(f"- Anode thickness: {mf_sample['anode_thickness']:.1f} μm")
    print(f"- Electrolyte thickness: {mf_sample['electrolyte_thickness']:.1f} μm")
    
    # Step 3: Material-level analysis (HF)
    print(f"\n🔬 Step 3: Material-Level Analysis (High Fidelity)")
    hf_data = data['parameters']['HF']
    
    hf_sample = hf_data.iloc[0]
    print(f"- Anode porosity: {hf_sample['anode_porosity']:.3f}")
    print(f"- Anode ionic conductivity: {hf_sample['anode_ionic_conductivity']:.4f} S/m")
    print(f"- Electrolyte Young's modulus: {hf_sample['electrolyte_youngs_modulus']:.1f} GPa")
    
    # Step 4: Microstructural analysis
    print(f"\n🔍 Step 4: Microstructural Analysis")
    micro_sample = list(data['microstructural'].values())[0]
    print(f"- Ni connectivity: {micro_sample['connectivity']['ni']:.3f}")
    print(f"- Specific surface area: {micro_sample['specific_surface_area']:.2e} m²/m³")
    
    # Step 5: Degradation assessment
    print(f"\n⚠️ Step 5: Degradation Risk Assessment")
    
    # Simple degradation risk calculation based on operating conditions
    temp_stress = (operating_point['temperature'] - 1073) / 200  # Normalized temp stress
    current_stress = operating_point['current_density'] / 1.0    # Normalized current stress
    
    degradation_risk = (temp_stress + current_stress) / 2
    
    if degradation_risk < 0.3:
        risk_level = "LOW"
    elif degradation_risk < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    print(f"- Temperature stress factor: {temp_stress:.3f}")
    print(f"- Current stress factor: {current_stress:.3f}")
    print(f"- Overall degradation risk: {risk_level} ({degradation_risk:.3f})")
    
    print(f"\n✅ Multi-scale analysis completed!")

def main():
    """Run all usage examples."""
    print("🔬 SOFC Multi-Fidelity Dataset Usage Examples")
    print("=" * 80)
    
    try:
        example_1_basic_data_exploration()
        X_datasets, y_datasets = example_2_multi_fidelity_modeling()
        example_3_degradation_analysis()
        example_4_microstructural_analysis()
        example_5_integrated_workflow()
        
        print("\n" + "="*80)
        print("🎉 All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()