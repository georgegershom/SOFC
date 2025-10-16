#!/usr/bin/env python3
"""
SOFC Dataset Quick Start
========================

Quick start script for the SOFC "In-The-Wild" dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_ml_data():
    """Load ML-ready data"""
    data = np.load('sofc_ml_ready.npz')
    return data

def load_summary():
    """Load summary data"""
    manufacturing = pd.read_csv('dataset/manufacturing_parameters.csv')
    quality = pd.read_csv('dataset/quality_analysis.csv')
    measurements = pd.read_csv('dataset/measurement_summary.csv')
    return manufacturing, quality, measurements

def quick_analysis():
    """Run quick analysis"""
    print("SOFC Dataset Quick Analysis")
    print("=" * 30)
    
    # Load data
    ml_data = load_ml_data()
    manufacturing, quality, measurements = load_summary()
    
    print(f"Dataset size: {ml_data['X'].shape[0]} plates")
    print(f"Features: {ml_data['X'].shape[1]}")
    print(f"Date range: {manufacturing['production_date'].min()} to {manufacturing['production_date'].max()}")
    
    # Basic statistics
    print("\nManufacturing Parameters:")
    print(manufacturing[['sintering_temp', 'sintering_time', 'cooling_rate']].describe())
    
    print("\nFailure Analysis:")
    print(quality[['overall_failure_risk', 'edge_crack_risk', 'delamination_risk']].describe())
    
    # Simple visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(manufacturing['sintering_temp'], bins=20, alpha=0.7)
    plt.title('Sintering Temperature')
    plt.xlabel('Temperature (°C)')
    
    plt.subplot(1, 3, 2)
    plt.hist(quality['overall_failure_risk'], bins=20, alpha=0.7)
    plt.title('Failure Risk')
    plt.xlabel('Risk')
    
    plt.subplot(1, 3, 3)
    plt.scatter(measurements['rms_displacement_um'], quality['overall_failure_risk'], alpha=0.6)
    plt.xlabel('RMS Displacement (μm)')
    plt.ylabel('Failure Risk')
    plt.title('Displacement vs Risk')
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Quick analysis complete!")
    print("Check 'quick_analysis.png' for visualizations")

if __name__ == "__main__":
    quick_analysis()
