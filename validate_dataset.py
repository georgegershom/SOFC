#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC???????????
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def validate_dataset(dataset_path='sofc_dataset/sofc_material_dataset.json'):
    """????????"""
    print("=" * 80)
    print("?????")
    print("=" * 80)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    validation_results = {
        'thermophysical': False,
        'sintering': False,
        'cte': False,
        'creep': False,
        'elastic': False
    }
    
    # ???????
    if 'thermophysical_properties' in dataset:
        materials = ['anode', 'electrolyte', 'cathode']
        if all(m in dataset['thermophysical_properties'] for m in materials):
            validation_results['thermophysical'] = True
            print("? ?????????")
    
    # ???????
    if 'sintering_kinetics' in dataset:
        sintering_keys = list(dataset['sintering_kinetics'].keys())
        if len(sintering_keys) >= 5:  # ??3??? + 2???
            validation_results['sintering'] = True
            print(f"? ????????? ({len(sintering_keys)} ???)")
    
    # ??CTE
    if 'coefficient_of_thermal_expansion' in dataset:
        cte_materials = ['anode', 'electrolyte', 'cathode', 'interconnect']
        if all(m in dataset['coefficient_of_thermal_expansion'] for m in cte_materials):
            validation_results['cte'] = True
            print("? ?????????")
    
    # ??????
    if 'creep_constitutive_data' in dataset:
        creep_materials = ['anode', 'electrolyte', 'cathode']
        if all(m in dataset['creep_constitutive_data'] for m in creep_materials):
            validation_results['creep'] = True
            print("? ????????")
    
    # ??????
    if 'elastic_properties' in dataset:
        elastic_materials = ['anode', 'electrolyte', 'cathode']
        if all(m in dataset['elastic_properties'] for m in elastic_materials):
            validation_results['elastic'] = True
            print("? ????????")
    
    # ????
    print("\n?????:")
    print(f"  - ????: {len(dataset.get('thermophysical_properties', {}))}")
    print(f"  - ?????: {len(dataset.get('sintering_kinetics', {}))}")
    print(f"  - CTE???: {len(dataset.get('coefficient_of_thermal_expansion', {}))}")
    print(f"  - ?????: {len(dataset.get('creep_constitutive_data', {}))}")
    print(f"  - ???????: {len(dataset.get('elastic_properties', {}))}")
    
    all_valid = all(validation_results.values())
    print(f"\n{'? ???????' if all_valid else '? ???????'}")
    
    return validation_results, dataset


def visualize_sintering_kinetics(dataset, output_dir='sofc_dataset/figures'):
    """??????????"""
    os.makedirs(output_dir, exist_ok=True)
    
    sintering = dataset['sintering_kinetics']
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SOFC???????', fontsize=16, fontweight='bold')
    
    # ??????
    ax1 = axes[0, 0]
    for key in ['anode_single', 'electrolyte_single', 'cathode_single']:
        if key in sintering:
            data = sintering[key]
            ax1.plot(data['temperature_C'], data['shrinkage_dL_L0'], 
                    label=key.replace('_single', ''), linewidth=2)
    ax1.set_xlabel('?? (?C)', fontsize=12)
    ax1.set_ylabel('??? (dL/L?)', fontsize=12)
    ax1.set_title('????????', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ?????????
    ax2 = axes[0, 1]
    for key in ['anode_electrolyte_bilayer', 'electrolyte_cathode_bilayer']:
        if key in sintering:
            data = sintering[key]
            ax2.plot(data['temperature_C'], data.get('mismatch_strain', [0]*len(data['temperature_C'])), 
                    label=key.replace('_bilayer', ''), linewidth=2)
    ax2.set_xlabel('?? (?C)', fontsize=12)
    ax2.set_ylabel('?????', fontsize=12)
    ax2.set_title('?????????', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # CTE??
    ax3 = axes[1, 0]
    cte_data = dataset['coefficient_of_thermal_expansion']
    for material in ['anode', 'electrolyte', 'cathode']:
        if material in cte_data:
            data = cte_data[material]
            ax3.plot(data['temperature_C'], 
                    [x * 1e6 for x in data['cte_per_K']], 
                    label=material, linewidth=2)
    ax3.set_xlabel('?? (?C)', fontsize=12)
    ax3.set_ylabel('CTE (?10?? K??)', fontsize=12)
    ax3.set_title('???????', fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ?????????
    ax4 = axes[1, 1]
    elastic_data = dataset['elastic_properties']
    for material in ['anode', 'electrolyte', 'cathode']:
        if material in elastic_data:
            data = elastic_data[material]['temperature_dependent']
            ax4.plot(data['temperature_C'], 
                    [x / 1e9 for x in data['youngs_modulus_Pa']], 
                    label=material, linewidth=2)
    ax4.set_xlabel('?? (?C)', fontsize=12)
    ax4.set_ylabel('???? (GPa)', fontsize=12)
    ax4.set_title('?????????', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = os.path.join(output_dir, 'dataset_summary.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"? ????????: {figure_path}")
    plt.close()


def create_dataset_summary(output_dir='sofc_dataset'):
    """?????????"""
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    
    with open(os.path.join(output_dir, 'sofc_material_dataset.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    summary = []
    summary.append("=" * 80)
    summary.append("SOFC??????????????")
    summary.append("=" * 80)
    summary.append("")
    summary.append(f"????: {dataset['metadata']['generation_date']}")
    summary.append("")
    summary.append("?????:")
    summary.append("")
    
    # ?????
    summary.append("1. ????? (????)")
    for material, props in dataset['thermophysical_properties'].items():
        summary.append(f"   {material.upper()} ({props['material']}):")
        gs = props['green_state']
        summary.append(f"     - ????: {gs['density_green']} kg/m?")
        summary.append(f"     - ?????: {gs['density_sintered']} kg/m?")
        summary.append(f"     - ?????: {gs['porosity_initial']:.2%}")
        summary.append(f"     - ?????: {gs['porosity_final']:.2%}")
    summary.append("")
    
    # ?????
    summary.append("2. ???????")
    sintering_keys = list(dataset['sintering_kinetics'].keys())
    summary.append(f"   - ????: {len([k for k in sintering_keys if 'single' in k])} ?")
    summary.append(f"   - ??????: {len([k for k in sintering_keys if 'bilayer' in k])} ?")
    summary.append(f"   - ????: 25-1400?C")
    summary.append("")
    
    # CTE
    summary.append("3. ?????(CTE)")
    for material, data in dataset['coefficient_of_thermal_expansion'].items():
        cte_min = min(data['cte_per_K']) * 1e6
        cte_max = max(data['cte_per_K']) * 1e6
        summary.append(f"   {material.upper()}: {cte_min:.2f} - {cte_max:.2f} ?10?? K??")
    summary.append("")
    
    # ????
    summary.append("4. ?????? (Norton??)")
    for material, data in dataset['creep_constitutive_data'].items():
        params = data['creep_parameters']
        summary.append(f"   {material.upper()}:")
        summary.append(f"     - A = {params['A']:.2e} s?? MPa??")
        summary.append(f"     - n = {params['n']:.2f}")
        summary.append(f"     - Q = {params['Q']/1000:.0f} kJ/mol")
    summary.append("")
    
    # ????
    summary.append("5. ????")
    for material, data in dataset['elastic_properties'].items():
        E_room = data['temperature_dependent']['youngs_modulus_GPa'][0]
        E_high = data['temperature_dependent']['youngs_modulus_GPa'][-1]
        nu = data['temperature_dependent']['poissons_ratio'][0]
        summary.append(f"   {material.upper()}:")
        summary.append(f"     - ??????: {E_room:.1f} GPa")
        summary.append(f"     - ??????: {E_high:.1f} GPa")
        summary.append(f"     - ???: {nu:.2f}")
    summary.append("")
    summary.append("=" * 80)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"? ????????: {summary_path}")


if __name__ == '__main__':
    # ?????
    validation_results, dataset = validate_dataset()
    
    # ????
    create_dataset_summary()
    
    # ????????matplotlib???
    try:
        visualize_sintering_kinetics(dataset)
    except ImportError:
        print("??: matplotlib?????????")
    except Exception as e:
        print(f"??: ???????: {e}")
