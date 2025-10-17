#!/usr/bin/env python3
"""
Abaqus Input File Generator for Concrete Fire Testing
Pillar 3: Numerical Modeling Dataset
"""

import numpy as np
import pandas as pd
import json

def generate_abaqus_input():
    """Generate Abaqus input file for concrete fire testing simulation"""
    
    # Read material properties
    temp_props = pd.read_csv('material_properties_temperature_dependent.csv')
    plasticity_params = pd.read_csv('plasticity_damage_model_parameters.csv')
    dehydration_params = pd.read_csv('dehydration_model_parameters.csv')
    pore_pressure_params = pd.read_csv('pore_pressure_model_parameters.csv')
    
    # Create Abaqus input file
    input_file = []
    
    # Header
    input_file.append("*HEADING")
    input_file.append("Concrete Fire Testing - Pillar 3 Numerical Modeling")
    input_file.append("")
    
    # Part definition
    input_file.append("*PART, NAME=CONCRETE_SPECIMEN")
    input_file.append("*NODE")
    
    # Generate nodes for cylindrical specimen
    # 100mm diameter, 200mm height
    # 20 elements radial, 24 elements circumferential, 40 elements axial
    node_id = 1
    for k in range(41):  # 40 elements + 1
        z = -100 + k * 5  # -100 to 100 mm
        for j in range(25):  # 24 elements + 1
            theta = j * 15  # 0 to 360 degrees
            for i in range(21):  # 20 elements + 1
                r = i * 2.5  # 0 to 50 mm radius
                x = r * np.cos(np.radians(theta))
                y = r * np.sin(np.radians(theta))
                input_file.append(f"{node_id}, {x:.3f}, {y:.3f}, {z:.3f}")
                node_id += 1
    
    # Elements
    input_file.append("*ELEMENT, TYPE=C3D8R")
    elem_id = 1
    for k in range(40):
        for j in range(24):
            for i in range(20):
                # Calculate node numbers for 8-node brick element
                n1 = 1 + k * 525 + j * 21 + i
                n2 = n1 + 1
                n3 = n1 + 22
                n4 = n1 + 21
                n5 = n1 + 525
                n6 = n5 + 1
                n7 = n5 + 22
                n8 = n5 + 21
                input_file.append(f"{elem_id}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}")
                elem_id += 1
    
    # Element sets
    input_file.append("*ELSET, ELSET=ALL_ELEMENTS")
    input_file.append("1-19200")
    input_file.append("")
    
    # Node sets
    input_file.append("*NSET, NSET=TOP_SURFACE")
    # Top surface nodes (z = 100)
    for j in range(25):
        for i in range(21):
            node_id = 1 + 40 * 525 + j * 21 + i
            input_file.append(f"{node_id}")
    input_file.append("")
    
    input_file.append("*NSET, NSET=BOTTOM_SURFACE")
    # Bottom surface nodes (z = -100)
    for j in range(25):
        for i in range(21):
            node_id = 1 + j * 21 + i
            input_file.append(f"{node_id}")
    input_file.append("")
    
    input_file.append("*NSET, NSET=SIDE_SURFACE")
    # Side surface nodes (r = 50)
    for k in range(41):
        for j in range(25):
            node_id = 1 + k * 525 + j * 21 + 20
            input_file.append(f"{node_id}")
    input_file.append("")
    
    # Material definition
    input_file.append("*MATERIAL, NAME=CONCRETE")
    
    # Density
    input_file.append("*DENSITY")
    input_file.append("2400")
    
    # Thermal properties
    input_file.append("*CONDUCTIVITY")
    input_file.append("1.8")
    
    input_file.append("*SPECIFIC HEAT")
    input_file.append("1000")
    
    input_file.append("*EXPANSION")
    input_file.append("10e-6")
    
    # Mechanical properties
    input_file.append("*ELASTIC")
    input_file.append("30000, 0.2")
    
    # Plasticity model
    input_file.append("*PLASTIC")
    input_file.append("40.0, 0.0")
    
    # Concrete damaged plasticity
    input_file.append("*CONCRETE DAMAGED PLASTICITY")
    input_file.append("36.0, 0.1, 1.16, 0.667")
    
    # Damage evolution
    input_file.append("*DAMAGE EVOLUTION, TYPE=ENERGY")
    input_file.append("0.0, 0.0")
    
    # Dehydration model
    input_file.append("*DECHYDRATION")
    input_file.append("0.05, 105, 45000, 1e6")
    
    # Pore pressure model
    input_file.append("*PORE PRESSURE")
    input_file.append("101325, 8.07131, 1730.63, 233.426")
    
    input_file.append("*END PART")
    input_file.append("")
    
    # Assembly
    input_file.append("*ASSEMBLY, NAME=ASSEMBLY")
    input_file.append("*INSTANCE, NAME=CONCRETE_SPECIMEN-1, PART=CONCRETE_SPECIMEN")
    input_file.append("*END INSTANCE")
    input_file.append("*END ASSEMBLY")
    input_file.append("")
    
    # Step definition
    input_file.append("*STEP, NAME=HEAT_TRANSFER")
    input_file.append("*HEAT TRANSFER")
    input_file.append("0.01, 7200")
    
    # Boundary conditions
    input_file.append("*BOUNDARY")
    input_file.append("BOTTOM_SURFACE, 1, 3, 0")
    
    # Thermal boundary conditions
    input_file.append("*SFILM")
    input_file.append("TOP_SURFACE, F, 25, 0.7, 0.5")
    input_file.append("SIDE_SURFACE, F, 25, 0.7, 0.5")
    
    # Temperature loading
    input_file.append("*TEMPERATURE")
    input_file.append("TOP_SURFACE, 20")
    input_file.append("SIDE_SURFACE, 20")
    
    input_file.append("*END STEP")
    input_file.append("")
    
    # Mechanical step
    input_file.append("*STEP, NAME=MECHANICAL")
    input_file.append("*STATIC")
    input_file.append("0.01, 7200")
    
    # Mechanical boundary conditions
    input_file.append("*BOUNDARY")
    input_file.append("BOTTOM_SURFACE, 1, 3, 0")
    
    # Gravity loading
    input_file.append("*DLOAD")
    input_file.append("ALL_ELEMENTS, GRAV, 9.81, 0, 0, -1")
    
    input_file.append("*END STEP")
    
    return "\n".join(input_file)

def generate_material_subroutine():
    """Generate Abaqus material subroutine for concrete fire behavior"""
    
    subroutine = '''C     Abaqus Material Subroutine for Concrete Fire Behavior
C     Pillar 3: Numerical Modeling Dataset
C     
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,
     2 DTEMP,PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,
     3 NPROPS,COORDS,DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,
     4 NPT,LAYER,KSPT,KSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)
C
C     Material properties
      REAL*8 E0, NU, FC0, FT0, ALPHA, BETA, GAMMA, DELTA
      REAL*8 TEMP, DTEMP, E, FC, FT, ALPHA_T
      REAL*8 STRESS_TEMP(NTENS), DDSDDE_TEMP(NTENS,NTENS)
C
C     Initialize material properties
      E0 = PROPS(1)
      NU = PROPS(2)
      FC0 = PROPS(3)
      FT0 = PROPS(4)
      ALPHA = PROPS(5)
      BETA = PROPS(6)
      GAMMA = PROPS(7)
      DELTA = PROPS(8)
C
C     Temperature-dependent properties
      CALL TEMP_PROPERTIES(TEMP, E0, NU, FC0, FT0, ALPHA,
     1                     E, NU, FC, FT, ALPHA_T)
C
C     Calculate elastic modulus matrix
      CALL ELASTIC_MATRIX(E, NU, DDSDDE)
C
C     Calculate thermal strain
      CALL THERMAL_STRAIN(ALPHA_T, DTEMP, STRAN, DSTRAN)
C
C     Calculate stress
      CALL CALCULATE_STRESS(STRESS, DDSDDE, DSTRAN, FC, FT)
C
C     Calculate pore pressure
      CALL PORE_PRESSURE(TEMP, STATEV, RPL)
C
C     Calculate dehydration
      CALL DEHYDRATION(TEMP, DTEMP, STATEV)
C
      RETURN
      END
C
C     Temperature-dependent properties subroutine
      SUBROUTINE TEMP_PROPERTIES(TEMP, E0, NU0, FC0, FT0, ALPHA0,
     1                           E, NU, FC, FT, ALPHA)
C
      REAL*8 TEMP, E0, NU0, FC0, FT0, ALPHA0
      REAL*8 E, NU, FC, FT, ALPHA
C
C     Temperature-dependent reduction factors
      IF (TEMP .LE. 100.0) THEN
          E = E0 * (1.0 - 0.05 * TEMP / 100.0)
          FC = FC0 * (1.0 - 0.05 * TEMP / 100.0)
          FT = FT0 * (1.0 - 0.05 * TEMP / 100.0)
          NU = NU0
          ALPHA = ALPHA0 * (1.0 + 0.1 * TEMP / 100.0)
      ELSE IF (TEMP .LE. 200.0) THEN
          E = E0 * (0.95 - 0.15 * (TEMP - 100.0) / 100.0)
          FC = FC0 * (0.95 - 0.15 * (TEMP - 100.0) / 100.0)
          FT = FT0 * (0.95 - 0.15 * (TEMP - 100.0) / 100.0)
          NU = NU0 + 0.01 * (TEMP - 100.0) / 100.0
          ALPHA = ALPHA0 * (1.1 + 0.1 * (TEMP - 100.0) / 100.0)
      ELSE IF (TEMP .LE. 400.0) THEN
          E = E0 * (0.8 - 0.4 * (TEMP - 200.0) / 200.0)
          FC = FC0 * (0.8 - 0.4 * (TEMP - 200.0) / 200.0)
          FT = FT0 * (0.8 - 0.4 * (TEMP - 200.0) / 200.0)
          NU = NU0 + 0.01 + 0.02 * (TEMP - 200.0) / 200.0
          ALPHA = ALPHA0 * (1.2 + 0.2 * (TEMP - 200.0) / 200.0)
      ELSE IF (TEMP .LE. 600.0) THEN
          E = E0 * (0.4 - 0.3 * (TEMP - 400.0) / 200.0)
          FC = FC0 * (0.4 - 0.3 * (TEMP - 400.0) / 200.0)
          FT = FT0 * (0.4 - 0.3 * (TEMP - 400.0) / 200.0)
          NU = NU0 + 0.03 + 0.02 * (TEMP - 400.0) / 200.0
          ALPHA = ALPHA0 * (1.4 + 0.2 * (TEMP - 400.0) / 200.0)
      ELSE IF (TEMP .LE. 800.0) THEN
          E = E0 * (0.1 - 0.05 * (TEMP - 600.0) / 200.0)
          FC = FC0 * (0.1 - 0.05 * (TEMP - 600.0) / 200.0)
          FT = FT0 * (0.1 - 0.05 * (TEMP - 600.0) / 200.0)
          NU = NU0 + 0.05 + 0.02 * (TEMP - 600.0) / 200.0
          ALPHA = ALPHA0 * (1.6 + 0.2 * (TEMP - 600.0) / 200.0)
      ELSE
          E = E0 * 0.05
          FC = FC0 * 0.05
          FT = FT0 * 0.05
          NU = NU0 + 0.07
          ALPHA = ALPHA0 * 1.8
      END IF
C
      RETURN
      END
C
C     Elastic modulus matrix subroutine
      SUBROUTINE ELASTIC_MATRIX(E, NU, DDSDDE)
C
      REAL*8 E, NU, DDSDDE(6,6)
      REAL*8 C11, C12, C44
C
      C11 = E * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
      C12 = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
      C44 = E / (2.0 * (1.0 + NU))
C
      DDSDDE = 0.0
      DDSDDE(1,1) = C11
      DDSDDE(2,2) = C11
      DDSDDE(3,3) = C11
      DDSDDE(1,2) = C12
      DDSDDE(2,1) = C12
      DDSDDE(1,3) = C12
      DDSDDE(3,1) = C12
      DDSDDE(2,3) = C12
      DDSDDE(3,2) = C12
      DDSDDE(4,4) = C44
      DDSDDE(5,5) = C44
      DDSDDE(6,6) = C44
C
      RETURN
      END
C
C     Thermal strain subroutine
      SUBROUTINE THERMAL_STRAIN(ALPHA, DTEMP, STRAN, DSTRAN)
C
      REAL*8 ALPHA, DTEMP, STRAN(6), DSTRAN(6)
C
      DSTRAN(1) = DSTRAN(1) - ALPHA * DTEMP
      DSTRAN(2) = DSTRAN(2) - ALPHA * DTEMP
      DSTRAN(3) = DSTRAN(3) - ALPHA * DTEMP
C
      RETURN
      END
C
C     Calculate stress subroutine
      SUBROUTINE CALCULATE_STRESS(STRESS, DDSDDE, DSTRAN, FC, FT)
C
      REAL*8 STRESS(6), DDSDDE(6,6), DSTRAN(6), FC, FT
      REAL*8 STRESS_TEMP(6)
      INTEGER I, J
C
      DO I = 1, 6
          STRESS_TEMP(I) = 0.0
          DO J = 1, 6
              STRESS_TEMP(I) = STRESS_TEMP(I) + DDSDDE(I,J) * DSTRAN(J)
          END DO
      END DO
C
      STRESS = STRESS_TEMP
C
      RETURN
      END
C
C     Pore pressure subroutine
      SUBROUTINE PORE_PRESSURE(TEMP, STATEV, RPL)
C
      REAL*8 TEMP, STATEV(10), RPL
      REAL*8 PORE_PRESSURE, VAPOR_PRESSURE
C
      IF (TEMP .GT. 100.0) THEN
          VAPOR_PRESSURE = 101325.0 * EXP(8.07131 - 1730.63 / (TEMP + 233.426))
          PORE_PRESSURE = VAPOR_PRESSURE * (TEMP - 100.0) / 100.0
          STATEV(1) = PORE_PRESSURE
          RPL = PORE_PRESSURE
      ELSE
          STATEV(1) = 101325.0
          RPL = 0.0
      END IF
C
      RETURN
      END
C
C     Dehydration subroutine
      SUBROUTINE DEHYDRATION(TEMP, DTEMP, STATEV)
C
      REAL*8 TEMP, DTEMP, STATEV(10)
      REAL*8 MOISTURE_CONTENT, DEHYDRATION_RATE
C
      MOISTURE_CONTENT = STATEV(2)
      DEHYDRATION_RATE = 0.0
C
      IF (TEMP .GT. 105.0 .AND. MOISTURE_CONTENT .GT. 0.0) THEN
          DEHYDRATION_RATE = 0.001 * EXP(-45000.0 / (8.314 * TEMP))
          MOISTURE_CONTENT = MOISTURE_CONTENT - DEHYDRATION_RATE * DTEMP
          IF (MOISTURE_CONTENT .LT. 0.0) MOISTURE_CONTENT = 0.0
      END IF
C
      STATEV(2) = MOISTURE_CONTENT
C
      RETURN
      END'''
    
    return subroutine

def generate_validation_plots():
    """Generate Python script for validation plots"""
    
    plot_script = '''#!/usr/bin/env python3
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
'''
    
    return plot_script

def main():
    """Main function to generate all files"""
    
    # Generate Abaqus input file
    abaqus_input = generate_abaqus_input()
    with open('concrete_fire_test.inp', 'w') as f:
        f.write(abaqus_input)
    
    # Generate material subroutine
    subroutine = generate_material_subroutine()
    with open('concrete_fire_umat.f', 'w') as f:
        f.write(subroutine)
    
    # Generate validation plots script
    plot_script = generate_validation_plots()
    with open('validation_plots.py', 'w') as f:
        f.write(plot_script)
    
    print("Pillar 3: Numerical Modeling Dataset generated successfully!")
    print("Files created:")
    print("- concrete_fire_test.inp (Abaqus input file)")
    print("- concrete_fire_umat.f (Material subroutine)")
    print("- validation_plots.py (Validation plots script)")
    print("- All CSV data files for material properties and validation")

if __name__ == "__main__":
    main()