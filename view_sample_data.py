#!/usr/bin/env python3
"""
Quick Data Viewer - Acoustic Pressure Dataset
View a sample of the generated dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def view_sample():
    """Display sample data from Group 01"""
    
    # Load one second of data
    csv_file = '/workspace/acoustic_pressure_dataset/Group_01/Group_01_second_05_A.csv'
    
    if not os.path.exists(csv_file):
        print("❌ Sample file not found. Please run generate_acoustic_data.py first.")
        return
    
    df = pd.read_csv(csv_file)
    
    print("=" * 70)
    print("ACOUSTIC PRESSURE DATASET - SAMPLE DATA VIEWER")
    print("=" * 70)
    print(f"\nFile: {os.path.basename(csv_file)}")
    print(f"Time period: 5-6 seconds (leak event starts at t=5s)")
    print(f"Data points: {len(df)}")
    print(f"Sensors: {len(df.columns) - 1}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData statistics (Sensor PG05):")
    print(df['PG05'].describe())
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Multiple sensors
    ax1 = axes[0]
    sensors_to_plot = ['PG01', 'PG03', 'PG05', 'PG07', 'PG09']
    for sensor in sensors_to_plot:
        ax1.plot(df['Time_s'], df[sensor]/1000, label=sensor, linewidth=1, alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Pressure (kPa)', fontsize=12)
    ax1.set_title('Multi-Sensor Acoustic Pressure (Second 5: Leak Event Starts)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(5.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Valve Opens')
    
    # Plot 2: Single sensor detail
    ax2 = axes[1]
    ax2.plot(df['Time_s'], df['PG05']/1000, linewidth=1.5, color='blue')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Pressure (kPa)', fontsize=12)
    ax2.set_title('Sensor PG05 Detail (Position: 5.0m)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(5.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    output_file = '/workspace/sample_data_view.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Sample visualization saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("View complete! Check sample_data_view.png for visualization.")
    print("=" * 70)

if __name__ == "__main__":
    view_sample()
