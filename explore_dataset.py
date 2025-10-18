#!/usr/bin/env python3
"""
Quick exploration script for Phase 3 Microstructural Dataset
Demonstrates how to load, analyze, and visualize the data
"""

import json
import pandas as pd
import numpy as np
import os

def explore_dataset():
    """Explore the generated Phase 3 dataset"""
    
    print("=" * 80)
    print("PHASE 3 DATASET EXPLORATION")
    print("=" * 80)
    
    # Load complete dataset
    dataset_path = "phase3_microstructural_data/complete_dataset.json"
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please run phase3_microstructural_analysis.py first.")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"\nDataset generated on: {dataset['metadata']['generation_date']}")
    print(f"Number of samples: {len(dataset['samples'])}")
    
    # Analyze temperature distribution
    temperatures = {}
    for sample_id, sample_data in dataset['samples'].items():
        temp = sample_data['thermal_exposure']['temperature']
        rubber = sample_data['mix_design']['rubber_content']
        
        if temp not in temperatures:
            temperatures[temp] = []
        temperatures[temp].append(rubber)
    
    print("\n" + "-" * 40)
    print("TEMPERATURE DISTRIBUTION")
    print("-" * 40)
    for temp in sorted(temperatures.keys()):
        rubber_contents = set(temperatures[temp])
        print(f"  {temp:4d}°C: {len(temperatures[temp])} samples, "
              f"Rubber contents: {sorted(rubber_contents)}")
    
    # Analyze porosity evolution for one mix design
    print("\n" + "-" * 40)
    print("POROSITY EVOLUTION (C-28-R Mix)")
    print("-" * 40)
    
    porosity_data = []
    for sample_id, sample_data in dataset['samples'].items():
        if sample_data['mix_design']['mix_id'] == 'C-28-R':
            temp = sample_data['thermal_exposure']['temperature']
            if 'MicroCT' in sample_data['analyses']:
                porosity = sample_data['analyses']['MicroCT']['statistics'].get('Porosity_%', 0)
                porosity_data.append((temp, porosity))
    
    porosity_data.sort()
    for temp, porosity in porosity_data:
        bar = '█' * int(porosity)
        print(f"  {temp:4d}°C: {porosity:5.1f}% {bar}")
    
    # Analyze XRD phase changes
    print("\n" + "-" * 40)
    print("PHASE TRANSFORMATIONS (Average across all mixes)")
    print("-" * 40)
    
    phase_avg = {}
    phase_counts = {}
    
    for sample_data in dataset['samples'].values():
        temp = sample_data['thermal_exposure']['temperature']
        if 'XRD' in sample_data['analyses']:
            for phase_data in sample_data['analyses']['XRD']['phases']:
                phase = phase_data['Phase']
                content = phase_data['Content_wt%']
                
                key = (temp, phase)
                if key not in phase_avg:
                    phase_avg[key] = 0
                    phase_counts[key] = 0
                
                phase_avg[key] += content
                phase_counts[key] += 1
    
    # Calculate averages
    for key in phase_avg:
        phase_avg[key] /= phase_counts[key]
    
    # Display key phases
    key_phases = ['CH', 'CSH', 'CaO']
    temps = sorted(set(temp for temp, _ in phase_avg.keys()))
    
    for phase in key_phases:
        print(f"\n  {phase} Content (wt%):")
        for temp in temps:
            if (temp, phase) in phase_avg:
                content = phase_avg[(temp, phase)]
                bar = '▓' * int(content / 2)
                print(f"    {temp:4d}°C: {content:5.1f}% {bar}")
    
    # Analyze rubber degradation
    print("\n" + "-" * 40)
    print("RUBBER DEGRADATION ANALYSIS")
    print("-" * 40)
    
    rubber_integrity = {}
    for sample_data in dataset['samples'].values():
        if sample_data['mix_design']['rubber_content'] > 0:
            temp = sample_data['thermal_exposure']['temperature']
            if 'SEM' in sample_data['analyses']:
                if sample_data['analyses']['SEM']['morphology']:
                    integrity = sample_data['analyses']['SEM']['morphology'][0].get('Rubber_Integrity_%', 100)
                    
                    if temp not in rubber_integrity:
                        rubber_integrity[temp] = []
                    rubber_integrity[temp].append(integrity)
    
    print("\n  Average Rubber Integrity:")
    for temp in sorted(rubber_integrity.keys()):
        avg_integrity = np.mean(rubber_integrity[temp])
        status = "Intact" if avg_integrity > 80 else "Degraded" if avg_integrity > 30 else "Destroyed"
        bar = '●' * int(avg_integrity / 10)
        print(f"    {temp:4d}°C: {avg_integrity:5.1f}% {bar} [{status}]")
    
    # Validation summary
    print("\n" + "-" * 40)
    print("CROSS-TECHNIQUE VALIDATION")
    print("-" * 40)
    
    passed = 0
    failed = 0
    for sample_data in dataset['samples'].values():
        if 'validation' in sample_data:
            if sample_data['validation'].get('passed', False):
                passed += 1
            else:
                failed += 1
    
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"  Passed: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"  Failed: {failed}/{total} ({100-pass_rate:.1f}%)")
    
    # Correlation insights
    if 'correlations' in dataset:
        print("\n" + "-" * 40)
        print("KEY CORRELATIONS")
        print("-" * 40)
        
        if 'correlation_matrix' in dataset['correlations']:
            corr_matrix = dataset['correlations']['correlation_matrix']
            
            # Find strong correlations
            strong_correlations = []
            for param1 in corr_matrix:
                for param2 in corr_matrix[param1]:
                    if param1 < param2:  # Avoid duplicates
                        corr_value = corr_matrix[param1][param2]
                        if abs(corr_value) > 0.7 and param1 != param2:
                            strong_correlations.append((param1, param2, corr_value))
            
            strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            print("\n  Strong Correlations (|r| > 0.7):")
            for param1, param2, corr in strong_correlations[:5]:
                direction = "positive" if corr > 0 else "negative"
                print(f"    • {param1} ↔ {param2}: r = {corr:.3f} ({direction})")
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("DATASET SUMMARY")
    print("-" * 40)
    
    # Count analysis types
    analysis_counts = {}
    for sample_data in dataset['samples'].values():
        if 'analyses' in sample_data:
            for analysis_type in sample_data['analyses']:
                analysis_counts[analysis_type] = analysis_counts.get(analysis_type, 0) + 1
    
    print("\n  Analysis Coverage:")
    for analysis_type, count in sorted(analysis_counts.items()):
        coverage = count / len(dataset['samples']) * 100
        print(f"    {analysis_type}: {count} samples ({coverage:.1f}% coverage)")
    
    # Data completeness
    total_datapoints = 0
    for sample_data in dataset['samples'].values():
        if 'analyses' in sample_data:
            for analysis_type, data in sample_data['analyses'].items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            total_datapoints += len(value)
                        else:
                            total_datapoints += 1
    
    print(f"\n  Total Data Points: {total_datapoints:,}")
    print(f"  Average per Sample: {total_datapoints/len(dataset['samples']):.0f}")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    
    # Provide usage suggestions
    print("\nSUGGESTED NEXT STEPS:")
    print("1. Run phase3_visualization.py for comprehensive plots")
    print("2. Export data using DataExporter for external analysis")
    print("3. Use correlation matrix for feature selection in ML models")
    print("4. Analyze specific temperature thresholds for model calibration")
    print("5. Compare rubber vs. control samples for degradation mechanisms")
    
    return dataset

if __name__ == "__main__":
    dataset = explore_dataset()