#!/usr/bin/env python3
"""
Validation Plots for Pillar 3: Numerical Modeling Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_temperature_validation():
    """Plot temperature vs time validation"""
    df = pd.read_csv('temperature_vs_time_validation.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Temperature vs Time Validation', fontsize=16)
    
    depths = ['5mm', '10mm', '20mm', '40mm', '80mm']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (depth, color) in enumerate(zip(depths, colors)):
        row = i // 3
        col = i % 3
        
        exp_col = f'Depth_{depth}_Exp'
        model_col = f'Depth_{depth}_Model'
        
        axes[row, col].plot(df['Time_min'], df[exp_col], 'o-', 
                           color=color, label=f'Experimental {depth}', markersize=4)
        axes[row, col].plot(df['Time_min'], df[model_col], '--', 
                           color=color, label=f'Model {depth}', linewidth=2)
        axes[row, col].set_xlabel('Time (min)')
        axes[row, col].set_ylabel('Temperature (°C)')
        axes[row, col].set_title(f'Temperature at {depth} depth')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_validation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pore_pressure_validation():
    """Plot pore pressure vs time validation"""
    df = pd.read_csv('pore_pressure_vs_time_validation.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pore Pressure vs Time Validation', fontsize=16)
    
    depths = ['5mm', '10mm', '20mm', '40mm', '80mm']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (depth, color) in enumerate(zip(depths, colors)):
        row = i // 3
        col = i % 3
        
        exp_col = f'Depth_{depth}_Exp'
        model_col = f'Depth_{depth}_Model'
        
        axes[row, col].plot(df['Time_min'], df[exp_col]/1e6, 'o-', 
                           color=color, label=f'Experimental {depth}', markersize=4)
        axes[row, col].plot(df['Time_min'], df[model_col]/1e6, '--', 
                           color=color, label=f'Model {depth}', linewidth=2)
        axes[row, col].set_xlabel('Time (min)')
        axes[row, col].set_ylabel('Pore Pressure (MPa)')
        axes[row, col].set_title(f'Pore Pressure at {depth} depth')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pore_pressure_validation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_stress_strain_validation():
    """Plot stress-strain validation"""
    df = pd.read_csv('stress_strain_validation.csv')
    
    temperatures = df['Temperature_C'].unique()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Stress-Strain Validation', fontsize=16)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
    
    for i, temp in enumerate(temperatures):
        row = i // 3
        col = i % 3
        
        temp_data = df[df['Temperature_C'] == temp]
        
        axes[row, col].plot(temp_data['Strain_Exp'], temp_data['Stress_Exp_MPa'], 
                           'o-', color=colors[i], label=f'Experimental {temp}°C', markersize=4)
        axes[row, col].plot(temp_data['Strain_Model'], temp_data['Stress_Model_MPa'], 
                           '--', color=colors[i], label=f'Model {temp}°C', linewidth=2)
        axes[row, col].set_xlabel('Strain')
        axes[row, col].set_ylabel('Stress (MPa)')
        axes[row, col].set_title(f'Stress-Strain at {temp}°C')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stress_strain_validation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_failure_validation():
    """Plot time to failure validation"""
    df = pd.read_csv('stt_failure_validation.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time to failure
    ax1.plot(df['Stress_Level_MPa'], df['Time_to_Failure_Exp_min'], 'o-', 
             color='red', label='Experimental', markersize=6)
    ax1.plot(df['Stress_Level_MPa'], df['Time_to_Failure_Model_min'], 's--', 
             color='blue', label='Model', markersize=6)
    ax1.set_xlabel('Stress Level (MPa)')
    ax1.set_ylabel('Time to Failure (min)')
    ax1.set_title('Time to Failure Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature at failure
    ax2.plot(df['Stress_Level_MPa'], df['Temperature_at_Failure_Exp_C'], 'o-', 
             color='red', label='Experimental', markersize=6)
    ax2.plot(df['Stress_Level_MPa'], df['Temperature_at_Failure_Model_C'], 's--', 
             color='blue', label='Model', markersize=6)
    ax2.set_xlabel('Stress Level (MPa)')
    ax2.set_ylabel('Temperature at Failure (°C)')
    ax2.set_title('Temperature at Failure Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('failure_validation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spalling_validation():
    """Plot spalling validation"""
    df = pd.read_csv('spalling_validation.csv')
    
    # Count spalling occurrences
    exp_spalling = df['Spalling_Occurred_Exp'].value_counts()
    model_spalling = df['Spalling_Occurred_Model'].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Spalling occurrence comparison
    categories = ['No Spalling', 'Spalling']
    exp_counts = [exp_spalling.get('No', 0), exp_spalling.get('Yes', 0)]
    model_counts = [model_spalling.get('No', 0), model_spalling.get('Yes', 0)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, exp_counts, width, label='Experimental', color='red', alpha=0.7)
    ax1.bar(x + width/2, model_counts, width, label='Model', color='blue', alpha=0.7)
    ax1.set_xlabel('Spalling Condition')
    ax1.set_ylabel('Number of Tests')
    ax1.set_title('Spalling Occurrence Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time to spalling
    spalling_data = df[df['Spalling_Occurred_Exp'] == 'Yes']
    ax2.plot(spalling_data['Time_to_Spalling_Exp_min'], 
             spalling_data['Time_to_Spalling_Model_min'], 'o', 
             color='green', markersize=8)
    ax2.plot([0, 50], [0, 50], 'k--', alpha=0.5, label='Perfect Agreement')
    ax2.set_xlabel('Experimental Time to Spalling (min)')
    ax2.set_ylabel('Model Time to Spalling (min)')
    ax2.set_title('Time to Spalling Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spalling_validation.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_validation_metrics():
    """Calculate validation metrics"""
    print("Pillar 3: Numerical Modeling Dataset - Validation Metrics")
    print("=" * 60)
    
    # Temperature validation
    temp_df = pd.read_csv('temperature_vs_time_validation.csv')
    temp_rmse = np.sqrt(np.mean((temp_df.iloc[:, 1::2] - temp_df.iloc[:, 2::2])**2))
    print(f"Temperature RMSE: {temp_rmse:.2f}°C")
    
    # Pore pressure validation
    pore_df = pd.read_csv('pore_pressure_vs_time_validation.csv')
    pore_rmse = np.sqrt(np.mean((pore_df.iloc[:, 1::2] - pore_df.iloc[:, 2::2])**2))
    print(f"Pore Pressure RMSE: {pore_rmse/1e6:.2f} MPa")
    
    # Stress-strain validation
    stress_df = pd.read_csv('stress_strain_validation.csv')
    stress_rmse = np.sqrt(np.mean((stress_df['Stress_Exp_MPa'] - stress_df['Stress_Model_MPa'])**2))
    print(f"Stress-Strain RMSE: {stress_rmse:.2f} MPa")
    
    # Failure validation
    failure_df = pd.read_csv('stt_failure_validation.csv')
    failure_rmse = np.sqrt(np.mean((failure_df['Time_to_Failure_Exp_min'] - failure_df['Time_to_Failure_Model_min'])**2))
    print(f"Time to Failure RMSE: {failure_rmse:.2f} min")
    
    # Spalling validation
    spalling_df = pd.read_csv('spalling_validation.csv')
    spalling_accuracy = np.mean(spalling_df['Spalling_Occurred_Exp'] == spalling_df['Spalling_Occurred_Model'])
    print(f"Spalling Prediction Accuracy: {spalling_accuracy*100:.1f}%")

if __name__ == "__main__":
    plot_temperature_validation()
    plot_pore_pressure_validation()
    plot_stress_strain_validation()
    plot_failure_validation()
    plot_spalling_validation()
    calculate_validation_metrics()
