#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
??????????
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import h5py

class DatasetVisualizer:
    """???????"""
    
    def __init__(self, dataset_path='./material_dataset'):
        self.dataset_path = Path(dataset_path)
        self.load_dataset()
    
    def load_dataset(self):
        """?????"""
        json_path = self.dataset_path / 'material_dataset.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        else:
            raise FileNotFoundError(f"????????: {json_path}")
    
    def plot_green_state_properties(self, save_path=None):
        """????????"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('??????', fontsize=16, fontweight='bold')
        
        data = self.dataset['green_state_properties']
        
        layers = list(data.keys())
        densities = [data[l]['green_density_kg_m3'] for l in layers]
        porosities = [data[l]['initial_porosity'] * 100 for l in layers]
        binders = [data[l]['binder_content_wt_percent'] for l in layers]
        porogens = [data[l]['porogen_content_wt_percent'] for l in layers]
        
        # ??
        axes[0, 0].bar(layers, densities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_ylabel('???? (kg/m?)')
        axes[0, 0].set_title('????')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ???
        axes[0, 1].bar(layers, porosities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_ylabel('????? (%)')
        axes[0, 1].set_title('?????')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ?????
        axes[1, 0].bar(layers, binders, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_ylabel('????? (wt%)')
        axes[1, 0].set_title('?????')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ?????
        axes[1, 1].bar(layers, porogens, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_ylabel('????? (wt%)')
        axes[1, 1].set_title('?????')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_sintering_kinetics(self, save_path=None):
        """?????????"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('???????', fontsize=16, fontweight='bold')
        
        data = self.dataset['sintering_kinetics']
        
        colors = {'anode': '#1f77b4', 'electrolyte': '#ff7f0e', 'cathode': '#2ca02c'}
        
        # ????????
        ax1 = axes[0, 0]
        for layer in ['anode', 'electrolyte', 'cathode']:
            if layer in data:
                T = np.array(data[layer]['temperature_K'])
                shrinkage = np.array(data[layer]['shrinkage_vs_temperature'])
                ax1.plot(T - 273.15, shrinkage * 100, label=layer, linewidth=2, color=colors[layer])
        
        ax1.set_xlabel('?? (?C)')
        ax1.set_ylabel('??? (%)')
        ax1.set_title('??? vs ??')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ????????
        ax2 = axes[0, 1]
        for layer in ['anode', 'electrolyte', 'cathode']:
            if layer in data:
                t = np.array(data[layer]['time_s']) / 3600  # ?????
                shrinkage = np.array(data[layer]['shrinkage_vs_time'])
                ax2.plot(t, shrinkage * 100, label=layer, linewidth=2, color=colors[layer])
        
        ax2.set_xlabel('?? (??)')
        ax2.set_ylabel('??? (%)')
        ax2.set_title('??? vs ?? (????)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ????
        if 'anode_electrolyte_bilayer' in data:
            bilayer_data = data['anode_electrolyte_bilayer']
            T = np.array(bilayer_data['temperature_K'])
            shrinkage = np.array(bilayer_data['shrinkage_vs_temperature'])
            stress = np.array(bilayer_data['stress_mismatch_MPa'])
            
            ax3 = axes[1, 0]
            ax3.plot(T - 273.15, shrinkage * 100, label='????', linewidth=2, color='purple')
            ax3.set_xlabel('?? (?C)')
            ax3.set_ylabel('??? (%)')
            ax3.set_title('??-?????????')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            ax4.plot(T - 273.15, stress, linewidth=2, color='red')
            ax4.set_xlabel('?? (?C)')
            ax4.set_ylabel('???? (MPa)')
            ax4.set_title('????????')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cte_data(self, save_path=None):
        """?????????"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('????? (CTE) ??', fontsize=16, fontweight='bold')
        
        data = self.dataset['cte_data']
        colors = {
            'anode': '#1f77b4',
            'electrolyte': '#ff7f0e',
            'cathode': '#2ca02c',
            'interconnect': '#d62728'
        }
        
        # CTE vs ??
        ax1 = axes[0, 0]
        for layer in data.keys():
            T = np.array(data[layer]['temperature_K'])
            CTE = np.array(data[layer]['CTE_K_inv']) * 1e6  # ??? ppm/K
            ax1.plot(T - 273.15, CTE, label=layer, linewidth=2, color=colors.get(layer, 'black'))
        
        ax1.set_xlabel('?? (?C)')
        ax1.set_ylabel('CTE (?10?? K??)')
        ax1.set_title('????? vs ??')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ??? vs ??
        ax2 = axes[0, 1]
        for layer in data.keys():
            T = np.array(data[layer]['temperature_K'])
            strain = np.array(data[layer]['thermal_strain']) * 1000  # ??? %
            ax2.plot(T - 273.15, strain, label=layer, linewidth=2, color=colors.get(layer, 'black'))
        
        ax2.set_xlabel('?? (?C)')
        ax2.set_ylabel('??? (%)')
        ax2.set_title('??? vs ?? (???25?C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # CTE???
        ax3 = axes[1, 0]
        layers = list(data.keys())
        cte_25C = [data[l]['CTE_25C'] * 1e6 for l in layers]
        cte_800C = [data[l]['CTE_800C'] * 1e6 for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        ax3.bar(x - width/2, cte_25C, width, label='25?C', color='#1f77b4')
        ax3.bar(x + width/2, cte_800C, width, label='800?C', color='#ff7f0e')
        ax3.set_xlabel('???')
        ax3.set_ylabel('CTE (?10?? K??)')
        ax3.set_title('CTE??')
        ax3.set_xticks(x)
        ax3.set_xticklabels(layers)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # CTE??
        ax4 = axes[1, 1]
        electrolyte_cte = np.array(data['electrolyte']['CTE_K_inv']) * 1e6
        T = np.array(data['electrolyte']['temperature_K'])
        
        for layer in ['anode', 'cathode']:
            if layer in data:
                layer_cte = np.array(data[layer]['CTE_K_inv']) * 1e6
                mismatch = (layer_cte - electrolyte_cte) / electrolyte_cte * 100
                ax4.plot(T - 273.15, mismatch, label=f'{layer} vs electrolyte', linewidth=2, color=colors[layer])
        
        ax4.set_xlabel('?? (?C)')
        ax4.set_ylabel('CTE?? (%)')
        ax4.set_title('???????CTE??')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_creep_data(self, save_path=None):
        """??????"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('?????? (Norton??)', fontsize=16, fontweight='bold')
        
        data = self.dataset['creep_constitutive_data']
        
        # ?????????????
        layers_to_plot = ['anode', 'electrolyte', 'cathode']
        states = ['green', 'sintered']
        
        # ??? vs ?? (????)
        ax1 = axes[0, 0]
        colors = {'green': '#1f77b4', 'sintered': '#ff7f0e'}
        for layer in layers_to_plot:
            if layer in data:
                for state in states:
                    if state in data[layer]:
                        creep_data = data[layer][state]['creep_data']
                        df = pd.DataFrame(creep_data)
                        
                        # ?????
                        for T in df['temperature_K'].unique():
                            subset = df[df['temperature_K'] == T]
                            label = f'{layer} {state} {T-273.15:.0f}?C' if len([l for l in layers_to_plot if l == layer]) == 1 else None
                            ax1.loglog(subset['stress_MPa'], subset['strain_rate_s_inv'],
                                      marker='o', markersize=4, label=label, alpha=0.7,
                                      color=colors[state])
        
        ax1.set_xlabel('?? (MPa)')
        ax1.set_ylabel('????? (s??)')
        ax1.set_title('????? vs ??')
        ax1.legend(ncol=2, fontsize=8)
        ax1.grid(True, alpha=0.3, which='both')
        
        # ??????
        ax2 = axes[0, 1]
        params_list = []
        for layer in layers_to_plot:
            if layer in data:
                for state in states:
                    if state in data[layer]:
                        params = data[layer][state]['creep_parameters']
                        params_list.append({
                            'layer': layer,
                            'state': state,
                            'n': params['n_stress_exponent'],
                            'Q': params['Q_activation_energy_kJ_mol']
                        })
        
        if params_list:
            df_params = pd.DataFrame(params_list)
            x = np.arange(len(df_params))
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(x - 0.2, df_params['n'], 0.4, label='???? n', color='#1f77b4')
            bars2 = ax2_twin.bar(x + 0.2, df_params['Q'], 0.4, label='??? Q (kJ/mol)', color='#ff7f0e')
            
            ax2.set_xlabel('??')
            ax2.set_ylabel('???? n', color='#1f77b4')
            ax2_twin.set_ylabel('??? Q (kJ/mol)', color='#ff7f0e')
            ax2.set_title('Norton??????')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{row['layer']}\n{row['state']}" for _, row in df_params.iterrows()], rotation=45, ha='right')
            ax2.tick_params(axis='y', labelcolor='#1f77b4')
            ax2_twin.tick_params(axis='y', labelcolor='#ff7f0e')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # ????????
        ax3 = axes[1, 0]
        if 'electrolyte' in data and 'sintered' in data['electrolyte']:
            creep_data = data['electrolyte']['sintered']['creep_data']
            df = pd.DataFrame(creep_data)
            
            for stress in [10, 50, 100]:
                subset = df[df['stress_MPa'] == stress]
                if len(subset) > 0:
                    ax3.semilogy(subset['temperature_C'], subset['strain_rate_s_inv'],
                                marker='o', label=f'{stress} MPa', linewidth=2)
        
        ax3.set_xlabel('?? (?C)')
        ax3.set_ylabel('????? (s??)')
        ax3.set_title('???????? (???, ????)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        
        # ????
        ax4 = axes[1, 1]
        if 'electrolyte' in data and 'sintered' in data['electrolyte']:
            fit_data = data['electrolyte']['sintered']['fit_validation']
            log_sigma = np.array(fit_data['log_stress'])
            log_eps_dot = np.array(fit_data['log_strain_rate'])
            
            ax4.plot(log_sigma, log_eps_dot, 'o-', label='??', linewidth=2, markersize=6)
            
            # ????
            coeffs = np.polyfit(log_sigma, log_eps_dot, 1)
            ax4.plot(log_sigma, np.polyval(coeffs, log_sigma), '--', label=f'?? (n={coeffs[0]:.2f})', linewidth=2)
            
            ax4.set_xlabel('log(?) [MPa]')
            ax4.set_ylabel('log(??) [s??]')
            ax4.set_title('Norton??????')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_elastic_properties(self, save_path=None):
        """????????"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('??????', fontsize=16, fontweight='bold')
        
        data = self.dataset['elastic_properties']
        colors = {'anode': '#1f77b4', 'electrolyte': '#ff7f0e', 'cathode': '#2ca02c'}
        
        # ???? vs ??
        ax1 = axes[0, 0]
        for layer in data.keys():
            T = np.array(data[layer]['temperature_K'])
            E = np.array(data[layer]['youngs_modulus_vs_temperature_GPa'])
            ax1.plot(T - 273.15, E, label=layer, linewidth=2, color=colors.get(layer, 'black'))
        
        ax1.set_xlabel('?? (?C)')
        ax1.set_ylabel('???? (GPa)')
        ax1.set_title('???? vs ??')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ???? vs ????
        ax2 = axes[0, 1]
        for layer in data.keys():
            rho = np.array(data[layer]['relative_density'])
            E = np.array(data[layer]['youngs_modulus_vs_density_GPa'])
            ax2.plot(rho * 100, E, label=layer, linewidth=2, color=colors.get(layer, 'black'))
        
        ax2.set_xlabel('???? (%)')
        ax2.set_ylabel('???? (GPa)')
        ax2.set_title('???? vs ????')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ??? vs ??
        ax3 = axes[1, 0]
        for layer in data.keys():
            T = np.array(data[layer]['temperature_K'])
            nu = np.array(data[layer]['poissons_ratio_vs_temperature'])
            ax3.plot(T - 273.15, nu, label=layer, linewidth=2, color=colors.get(layer, 'black'))
        
        ax3.set_xlabel('?? (?C)')
        ax3.set_ylabel('???')
        ax3.set_title('??? vs ??')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ??? vs ????
        ax4 = axes[1, 1]
        for layer in data.keys():
            rho = np.array(data[layer]['relative_density'])
            nu = np.array(data[layer]['poissons_ratio_vs_density'])
            ax4.plot(rho * 100, nu, label=layer, linewidth=2, color=colors.get(layer, 'black'))
        
        ax4.set_xlabel('???? (%)')
        ax4.set_ylabel('???')
        ax4.set_title('??? vs ????')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all_plots(self, output_dir=None):
        """?????????"""
        if output_dir is None:
            output_dir = self.dataset_path / 'plots'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"????????: {output_dir}")
        
        self.plot_green_state_properties(output_dir / '01_green_state_properties.png')
        print("  ? ??????")
        
        self.plot_sintering_kinetics(output_dir / '02_sintering_kinetics.png')
        print("  ? ?????")
        
        self.plot_cte_data(output_dir / '03_cte_data.png')
        print("  ? ?????")
        
        self.plot_creep_data(output_dir / '04_creep_data.png')
        print("  ? ????")
        
        self.plot_elastic_properties(output_dir / '05_elastic_properties.png')
        print("  ? ????")
        
        print(f"\n????????: {output_dir.absolute()}")


def main():
    """???"""
    try:
        visualizer = DatasetVisualizer()
        visualizer.create_all_plots()
    except FileNotFoundError as e:
        print(f"??: {e}")
        print("???? generate_material_dataset.py ?????")


if __name__ == '__main__':
    main()
