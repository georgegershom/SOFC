#!/usr/bin/env python3
"""
SOFC????????
??????????????????
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # ????????
from matplotlib.gridspec import GridSpec

class SOFCDatasetVisualizer:
    """SOFC???????"""
    
    def __init__(self, dataset_dir="sofc_dataset"):
        self.dataset_dir = dataset_dir
        self.viz_dir = f"{dataset_dir}/visualization"
        
        # ????????
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # ????
        self.colors = {
            'anode': '#E74C3C',
            'electrolyte': '#3498DB',
            'cathode': '#2ECC71',
            'interconnect': '#95A5A6'
        }
    
    def plot_sintering_kinetics(self):
        """?????????"""
        print("??????????...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # ????
        layers = ['anode', 'electrolyte', 'cathode']
        
        # 1. ??????
        ax1 = fig.add_subplot(gs[0, 0])
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_dilatometry.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax1.plot(df['temperature_C'], df['linear_shrinkage_percent'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax1.set_xlabel('Temperature (?C)', fontsize=12)
        ax1.set_ylabel('Linear Shrinkage (%)', fontsize=12)
        ax1.set_title('Sintering Shrinkage Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ??????
        ax2 = fig.add_subplot(gs[0, 1])
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_dilatometry.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax2.plot(df['temperature_C'], df['shrinkage_rate_per_C'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax2.set_xlabel('Temperature (?C)', fontsize=12)
        ax2.set_ylabel('Shrinkage Rate (%/?C)', fontsize=12)
        ax2.set_title('Shrinkage Rate', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ??????
        ax3 = fig.add_subplot(gs[0, 2])
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_dilatometry.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax3.plot(df['temperature_C'], df['relative_density'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax3.set_xlabel('Temperature (?C)', fontsize=12)
        ax3.set_ylabel('Relative Density', fontsize=12)
        ax3.set_title('Densification During Sintering', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ??????
        ax4 = fig.add_subplot(gs[1, 0])
        bilayer_files = ['anode_electrolyte_bilayer.csv', 'electrolyte_cathode_bilayer.csv']
        for bf in bilayer_files:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{bf}"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                label = bf.replace('_bilayer.csv', '').replace('_', '-')
                ax4.plot(df['temperature_C'], df['bilayer_shrinkage_percent'], 
                        label=label.upper(), linewidth=2)
        ax4.set_xlabel('Temperature (?C)', fontsize=12)
        ax4.set_ylabel('Bilayer Shrinkage (%)', fontsize=12)
        ax4.set_title('Bilayer Co-Sintering', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. CTE???
        ax5 = fig.add_subplot(gs[1, 1])
        for bf in bilayer_files:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{bf}"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                label = bf.replace('_bilayer.csv', '').replace('_', '-')
                ax5.plot(df['temperature_C'], df['cte_mismatch_ppm_per_K'], 
                        label=label.upper(), linewidth=2)
        ax5.set_xlabel('Temperature (?C)', fontsize=12)
        ax5.set_ylabel('CTE Mismatch (ppm/K)', fontsize=12)
        ax5.set_title('CTE Mismatch in Bilayers', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # 6. ????
        ax6 = fig.add_subplot(gs[1, 2])
        for bf in bilayer_files:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{bf}"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                label = bf.replace('_bilayer.csv', '').replace('_', '-')
                ax6.plot(df['temperature_C'], df['estimated_stress_MPa'], 
                        label=label.upper(), linewidth=2)
        ax6.set_xlabel('Temperature (?C)', fontsize=12)
        ax6.set_ylabel('Thermal Stress (MPa)', fontsize=12)
        ax6.set_title('Estimated Thermal Stress', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.savefig(f"{self.viz_dir}/sintering_kinetics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ??: {self.viz_dir}/sintering_kinetics.png")
    
    def plot_cte_data(self):
        """??CTE??"""
        print("??CTE???...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Coefficient of Thermal Expansion (CTE) Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        layers = ['anode', 'electrolyte', 'cathode', 'interconnect']
        
        # 1. CTE vs ??
        ax1 = axes[0, 0]
        for layer in layers:
            file_path = f"{self.dataset_dir}/thermal_physical/{layer}_CTE.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax1.plot(df['temperature_C'], df['CTE_ppm_per_K'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax1.set_xlabel('Temperature (?C)', fontsize=12)
        ax1.set_ylabel('CTE (ppm/K)', fontsize=12)
        ax1.set_title('CTE vs Temperature', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ???
        ax2 = axes[0, 1]
        for layer in layers:
            file_path = f"{self.dataset_dir}/thermal_physical/{layer}_CTE.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax2.plot(df['temperature_C'], df['thermal_strain'] * 100, 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax2.set_xlabel('Temperature (?C)', fontsize=12)
        ax2.set_ylabel('Thermal Strain (%)', fontsize=12)
        ax2.set_title('Thermal Strain Accumulation', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ??CTE
        ax3 = axes[1, 0]
        for layer in layers:
            file_path = f"{self.dataset_dir}/thermal_physical/{layer}_CTE.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax3.plot(df['temperature_C'], df['instantaneous_CTE'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax3.set_xlabel('Temperature (?C)', fontsize=12)
        ax3.set_ylabel('Instantaneous CTE (ppm/K)', fontsize=12)
        ax3.set_title('Instantaneous CTE', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. CTE????? (?1000?C)
        ax4 = axes[1, 1]
        cte_at_1000 = []
        layer_names = []
        for layer in layers:
            file_path = f"{self.dataset_dir}/thermal_physical/{layer}_CTE.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # ?????1000?C???
                idx = np.argmin(np.abs(df['temperature_C'] - 1000))
                cte_at_1000.append(df['CTE_ppm_per_K'].iloc[idx])
                layer_names.append(layer.capitalize())
        
        bars = ax4.bar(layer_names, cte_at_1000, color=[self.colors[l.lower()] for l in layer_names])
        ax4.set_ylabel('CTE at 1000?C (ppm/K)', fontsize=12)
        ax4.set_title('CTE Comparison at 1000?C', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # ??????
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/cte_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ??: {self.viz_dir}/cte_analysis.png")
    
    def plot_creep_behavior(self):
        """??????"""
        print("?????????...")
        
        layers = ['anode', 'electrolyte', 'cathode']
        
        for layer in layers:
            file_path = f"{self.dataset_dir}/creep_data/{layer}_creep_tests.json"
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r') as f:
                creep_data = json.load(f)
            
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # ?????????
            temps_to_plot = [1000, 1200, 1400]
            stress_to_plot = [0.5, 2.0, 5.0]
            
            # 1. ?????????? (T=1200?C)
            ax1 = fig.add_subplot(gs[0, 0])
            for stress in stress_to_plot:
                key = f'T1200C_S{stress}MPa'
                if key in creep_data:
                    data = creep_data[key]
                    ax1.plot(data['time_hours'], data['creep_strain'], 
                            label=f'{stress} MPa', linewidth=2)
            ax1.set_xlabel('Time (hours)', fontsize=12)
            ax1.set_ylabel('Creep Strain', fontsize=12)
            ax1.set_title(f'{layer.capitalize()} - Creep at 1200?C', 
                         fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. ?????????? (?=2 MPa)
            ax2 = fig.add_subplot(gs[0, 1])
            for temp in temps_to_plot:
                key = f'T{temp}C_S2.0MPa'
                if key in creep_data:
                    data = creep_data[key]
                    ax2.plot(data['time_hours'], data['creep_strain'], 
                            label=f'{temp}?C', linewidth=2)
            ax2.set_xlabel('Time (hours)', fontsize=12)
            ax2.set_ylabel('Creep Strain', fontsize=12)
            ax2.set_title(f'{layer.capitalize()} - Creep at 2 MPa', 
                         fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. ???? vs ??
            ax3 = fig.add_subplot(gs[0, 2])
            for stress in stress_to_plot:
                key = f'T1200C_S{stress}MPa'
                if key in creep_data:
                    data = creep_data[key]
                    ax3.plot(data['time_hours'], data['creep_rate_per_hour'], 
                            label=f'{stress} MPa', linewidth=2)
            ax3.set_xlabel('Time (hours)', fontsize=12)
            ax3.set_ylabel('Creep Rate (1/hour)', fontsize=12)
            ax3.set_title(f'{layer.capitalize()} - Creep Rate at 1200?C', 
                         fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            
            # 4. ?????? vs ?? (????)
            ax4 = fig.add_subplot(gs[1, 0])
            stresses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            for temp in temps_to_plot:
                rates = []
                stress_vals = []
                for stress in stresses:
                    key = f'T{temp}C_S{stress}MPa'
                    if key in creep_data:
                        rates.append(creep_data[key]['steady_state_rate'])
                        stress_vals.append(stress)
                if rates:
                    ax4.loglog(stress_vals, rates, 'o-', label=f'{temp}?C', linewidth=2)
            ax4.set_xlabel('Stress (MPa)', fontsize=12)
            ax4.set_ylabel('Steady-State Creep Rate (1/h)', fontsize=12)
            ax4.set_title('Stress Dependence (Norton n)', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, which='both')
            
            # 5. ?????? vs 1/T (Arrhenius?)
            ax5 = fig.add_subplot(gs[1, 1])
            test_temps = [800, 900, 1000, 1100, 1200, 1300, 1400]
            for stress in [1.0, 5.0]:
                rates = []
                inv_temps = []
                for temp in test_temps:
                    key = f'T{temp}C_S{stress}MPa'
                    if key in creep_data:
                        rates.append(creep_data[key]['steady_state_rate'])
                        inv_temps.append(1000 / (temp + 273.15))
                if rates:
                    ax5.semilogy(inv_temps, rates, 'o-', label=f'{stress} MPa', linewidth=2)
            ax5.set_xlabel('1000/T (1/K)', fontsize=12)
            ax5.set_ylabel('Steady-State Creep Rate (1/h)', fontsize=12)
            ax5.set_title('Activation Energy (Arrhenius)', fontsize=13, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. ???-????-??????
            ax6 = fig.add_subplot(gs[1, 2])
            key = 'T1200C_S5.0MPa'
            if key in creep_data:
                data = creep_data[key]
                time = np.array(data['time_hours'])
                strain = np.array(data['creep_strain'])
                rate = np.array(data['creep_rate_per_hour'])
                
                ax6_twin = ax6.twinx()
                
                line1 = ax6.plot(time, strain, 'b-', label='Strain', linewidth=2)
                line2 = ax6_twin.plot(time, rate, 'r-', label='Rate', linewidth=2)
                
                ax6.set_xlabel('Time (hours)', fontsize=12)
                ax6.set_ylabel('Creep Strain', fontsize=12, color='b')
                ax6_twin.set_ylabel('Creep Rate (1/h)', fontsize=12, color='r')
                ax6.set_title('Three Stages of Creep', fontsize=13, fontweight='bold')
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax6.legend(lines, labels, loc='upper left')
                ax6.grid(True, alpha=0.3)
            
            plt.savefig(f"{self.viz_dir}/{layer}_creep_behavior.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ??: {self.viz_dir}/{layer}_creep_behavior.png")
    
    def plot_norton_parameters(self):
        """??Norton????"""
        print("??Norton?????...")
        
        file_path = f"{self.dataset_dir}/constitutive_parameters/norton_law_parameters.json"
        if not os.path.exists(file_path):
            return
        
        with open(file_path, 'r') as f:
            norton_data = json.load(f)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Norton Law Parameters Comparison\n(?? = A????exp(-Q/RT))', 
                     fontsize=16, fontweight='bold')
        
        layers = list(norton_data.keys())
        
        # 1. ????? A
        ax1 = axes[0]
        A_values = [norton_data[layer]['A_pre_exponential'] for layer in layers]
        bars1 = ax1.bar(layers, A_values, color=[self.colors[l] for l in layers])
        ax1.set_ylabel('A (1/(MPa^n?s))', fontsize=12)
        ax1.set_title('Pre-exponential Factor', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. ???? n
        ax2 = axes[1]
        n_values = [norton_data[layer]['n_stress_exponent'] for layer in layers]
        bars2 = ax2.bar(layers, n_values, color=[self.colors[l] for l in layers])
        ax2.set_ylabel('n (dimensionless)', fontsize=12)
        ax2.set_title('Stress Exponent', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        # 3. ??? Q
        ax3 = axes[2]
        Q_values = [norton_data[layer]['Q_activation_energy'] for layer in layers]
        bars3 = ax3.bar(layers, Q_values, color=[self.colors[l] for l in layers])
        ax3.set_ylabel('Q (kJ/mol)', fontsize=12)
        ax3.set_title('Activation Energy', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/norton_parameters.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ??: {self.viz_dir}/norton_parameters.png")
    
    def plot_elastic_properties(self):
        """??????"""
        print("?????????...")
        
        layers = ['anode', 'electrolyte', 'cathode', 'interconnect']
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. ???? vs ?? (???)
        ax1 = fig.add_subplot(gs[0, 0])
        for layer in layers:
            file_path = f"{self.dataset_dir}/elastic_properties/{layer}_elastic_properties.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # ???????????
                df_dense = df[df['relative_density'] > 0.95]
                if len(df_dense) > 0:
                    ax1.plot(df_dense['temperature_C'], df_dense['youngs_modulus_GPa'], 
                            label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax1.set_xlabel('Temperature (?C)', fontsize=12)
        ax1.set_ylabel("Young's Modulus (GPa)", fontsize=12)
        ax1.set_title("E vs T (Dense State)", fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ???? vs ???? (??)
        ax2 = fig.add_subplot(gs[0, 1])
        for layer in layers:
            file_path = f"{self.dataset_dir}/elastic_properties/{layer}_elastic_properties.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # ?????????
                df_rt = df[df['temperature_C'] < 100]
                if len(df_rt) > 0:
                    ax2.plot(df_rt['relative_density'], df_rt['youngs_modulus_GPa'], 
                            'o-', label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax2.set_xlabel('Relative Density', fontsize=12)
        ax2.set_ylabel("Young's Modulus (GPa)", fontsize=12)
        ax2.set_title("E vs Density (Room Temp)", fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ??? vs ??
        ax3 = fig.add_subplot(gs[0, 2])
        for layer in layers:
            file_path = f"{self.dataset_dir}/elastic_properties/{layer}_elastic_properties.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_dense = df[df['relative_density'] > 0.95]
                if len(df_dense) > 0:
                    ax3.plot(df_dense['temperature_C'], df_dense['poisson_ratio'], 
                            label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax3.set_xlabel('Temperature (?C)', fontsize=12)
        ax3.set_ylabel("Poisson's Ratio", fontsize=12)
        ax3.set_title("? vs T", fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ???? vs ??
        ax4 = fig.add_subplot(gs[1, 0])
        for layer in layers:
            file_path = f"{self.dataset_dir}/elastic_properties/{layer}_elastic_properties.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_dense = df[df['relative_density'] > 0.95]
                if len(df_dense) > 0:
                    ax4.plot(df_dense['temperature_C'], df_dense['shear_modulus_GPa'], 
                            label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax4.set_xlabel('Temperature (?C)', fontsize=12)
        ax4.set_ylabel('Shear Modulus (GPa)', fontsize=12)
        ax4.set_title("G vs T", fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. ???? vs ????
        ax5 = fig.add_subplot(gs[1, 1])
        for layer in layers:
            file_path = f"{self.dataset_dir}/elastic_properties/{layer}_elastic_properties.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_rt = df[df['temperature_C'] < 100]
                if len(df_rt) > 0:
                    ax5.plot(df_rt['relative_density'], df_rt['bulk_modulus_GPa'], 
                            'o-', label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax5.set_xlabel('Relative Density', fontsize=12)
        ax5.set_ylabel('Bulk Modulus (GPa)', fontsize=12)
        ax5.set_title("K vs Density", fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 3D: E vs T vs ?? (?????)
        ax6 = fig.add_subplot(gs[1, 2], projection='3d')
        file_path = f"{self.dataset_dir}/elastic_properties/anode_elastic_properties.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            scatter = ax6.scatter(df['temperature_C'], df['relative_density'], 
                                 df['youngs_modulus_GPa'], 
                                 c=df['youngs_modulus_GPa'], cmap='viridis', s=20)
            ax6.set_xlabel('Temperature (?C)', fontsize=10)
            ax6.set_ylabel('Relative Density', fontsize=10)
            ax6.set_zlabel("E (GPa)", fontsize=10)
            ax6.set_title("Anode: E(T, ?)", fontsize=13, fontweight='bold')
            plt.colorbar(scatter, ax=ax6, shrink=0.5)
        
        plt.savefig(f"{self.viz_dir}/elastic_properties.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ??: {self.viz_dir}/elastic_properties.png")
    
    def plot_density_evolution(self):
        """??????"""
        print("?????????...")
        
        layers = ['anode', 'electrolyte', 'cathode']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Density Evolution During Sintering', fontsize=16, fontweight='bold')
        
        # 1. ????
        ax1 = axes[0, 0]
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_density_evolution.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax1.plot(df['time_minutes'], df['temperature_C'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax1.set_xlabel('Time (minutes)', fontsize=12)
        ax1.set_ylabel('Temperature (?C)', fontsize=12)
        ax1.set_title('Thermal Profile', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ???? vs ??
        ax2 = axes[0, 1]
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_density_evolution.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax2.plot(df['time_minutes'], df['relative_density'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylabel('Relative Density', fontsize=12)
        ax2.set_title('Densification vs Time', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ???? vs ??
        ax3 = axes[1, 0]
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_density_evolution.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax3.plot(df['temperature_C'], df['relative_density'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax3.set_xlabel('Temperature (?C)', fontsize=12)
        ax3.set_ylabel('Relative Density', fontsize=12)
        ax3.set_title('Densification vs Temperature', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ?????
        ax4 = axes[1, 1]
        for layer in layers:
            file_path = f"{self.dataset_dir}/sintering_kinetics/{layer}_density_evolution.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                ax4.plot(df['temperature_C'], df['densification_rate'], 
                        label=layer.capitalize(), color=self.colors[layer], linewidth=2)
        ax4.set_xlabel('Temperature (?C)', fontsize=12)
        ax4.set_ylabel('Densification Rate (1/min)', fontsize=12)
        ax4.set_title('Densification Rate', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/density_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ??: {self.viz_dir}/density_evolution.png")
    
    def generate_all_visualizations(self):
        """?????????"""
        print("\n" + "="*60)
        print("???????????")
        print("="*60 + "\n")
        
        self.plot_sintering_kinetics()
        self.plot_cte_data()
        self.plot_creep_behavior()
        self.plot_norton_parameters()
        self.plot_elastic_properties()
        self.plot_density_evolution()
        
        print("\n" + "="*60)
        print("???????????!")
        print(f"?????: {self.viz_dir}")
        print("="*60 + "\n")


if __name__ == "__main__":
    visualizer = SOFCDatasetVisualizer()
    visualizer.generate_all_visualizations()
