"""
High-Fidelity SOFC Model and Dataset Generator
3D FEM models with microstructure-informed properties and detailed physics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import h5py
import json
from datetime import datetime
import sys
import os
from scipy import ndimage, interpolate, sparse
from scipy.sparse import linalg as sparse_linalg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sofc_physics import (
    SOFCOperatingConditions, ElectrochemicalModel,
    ThermalModel, MechanicalModel, DegradationModel
)


class HighFidelitySOFCModel:
    """High-fidelity 3D SOFC model with detailed physics"""
    
    def __init__(self, config: dict):
        self.config = config
        self.electrochem = ElectrochemicalModel(config)
        self.thermal = ThermalModel(config)
        self.mechanical = MechanicalModel(config)
        self.degradation = DegradationModel(config)
        self.hf_config = config['dataset']['high_fidelity']
        self.mesh_res = self.hf_config['mesh_resolution']
        self.geom = config['geometry']
        self.mat = config['material_properties']
        
    def create_microstructure_3D(self) -> Dict[str, np.ndarray]:
        """Generate 3D microstructure with pores and phases"""
        nx, ny, nz = self.mesh_res
        
        microstructure = {
            'phase': np.zeros((nx, ny, nz), dtype=int),  # 0: pore, 1: Ni, 2: YSZ, 3: LSM
            'porosity': np.zeros((nx, ny, nz)),
            'tortuosity': np.zeros((nx, ny, nz)),
            'tpb_density': np.zeros((nx, ny, nz))  # Triple phase boundary density
        }
        
        # Layer assignment
        anode_layers = nz // 3
        electrolyte_layers = nz // 6
        cathode_layers = nz - anode_layers - electrolyte_layers
        
        # Anode microstructure (Ni-YSZ)
        for k in range(anode_layers):
            # Random porous structure
            porosity_target = self.mat['anode']['porosity']
            
            # Generate correlated random field for realistic microstructure
            random_field = np.random.randn(nx, ny)
            random_field = ndimage.gaussian_filter(random_field, sigma=2)
            
            # Threshold to get phases
            threshold_pore = np.percentile(random_field, porosity_target * 100)
            threshold_ni = np.percentile(random_field, (porosity_target + 0.35) * 100)
            
            layer = np.zeros((nx, ny))
            layer[random_field < threshold_pore] = 0  # Pore
            layer[(random_field >= threshold_pore) & (random_field < threshold_ni)] = 1  # Ni
            layer[random_field >= threshold_ni] = 2  # YSZ
            
            microstructure['phase'][:, :, k] = layer
            microstructure['porosity'][:, :, k] = porosity_target * (1 + 0.1 * np.random.randn(nx, ny))
            microstructure['tortuosity'][:, :, k] = self.mat['anode']['tortuosity'] * (1 + 0.05 * np.random.randn(nx, ny))
            
            # TPB density (interfaces between phases)
            tpb = np.zeros((nx, ny))
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    neighbors = layer[i-1:i+2, j-1:j+2].flatten()
                    if 0 in neighbors and 1 in neighbors and 2 in neighbors:
                        tpb[i, j] = 1.0
            microstructure['tpb_density'][:, :, k] = ndimage.gaussian_filter(tpb, sigma=1)
        
        # Electrolyte (dense YSZ)
        for k in range(anode_layers, anode_layers + electrolyte_layers):
            microstructure['phase'][:, :, k] = 2  # YSZ
            microstructure['porosity'][:, :, k] = 0.01  # Nearly dense
            microstructure['tortuosity'][:, :, k] = 1.0
            microstructure['tpb_density'][:, :, k] = 0
        
        # Cathode microstructure (LSM-YSZ)
        for k in range(anode_layers + electrolyte_layers, nz):
            porosity_target = self.mat['cathode']['porosity']
            
            random_field = np.random.randn(nx, ny)
            random_field = ndimage.gaussian_filter(random_field, sigma=2)
            
            threshold_pore = np.percentile(random_field, porosity_target * 100)
            threshold_lsm = np.percentile(random_field, (porosity_target + 0.35) * 100)
            
            layer = np.zeros((nx, ny))
            layer[random_field < threshold_pore] = 0  # Pore
            layer[(random_field >= threshold_pore) & (random_field < threshold_lsm)] = 3  # LSM
            layer[random_field >= threshold_lsm] = 2  # YSZ
            
            microstructure['phase'][:, :, k] = layer
            microstructure['porosity'][:, :, k] = porosity_target * (1 + 0.1 * np.random.randn(nx, ny))
            microstructure['tortuosity'][:, :, k] = self.mat['cathode']['tortuosity'] * (1 + 0.05 * np.random.randn(nx, ny))
            
            # TPB density
            tpb = np.zeros((nx, ny))
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    neighbors = layer[i-1:i+2, j-1:j+2].flatten()
                    if 0 in neighbors and 3 in neighbors and 2 in neighbors:
                        tpb[i, j] = 1.0
            microstructure['tpb_density'][:, :, k] = ndimage.gaussian_filter(tpb, sigma=1)
            
        return microstructure
    
    def simulate_electrochemistry_3D(self, conditions: SOFCOperatingConditions,
                                    microstructure: Dict) -> Dict[str, np.ndarray]:
        """3D electrochemical simulation with microstructure effects"""
        nx, ny, nz = self.mesh_res
        
        # Initialize 3D fields
        current_density = np.zeros((nx, ny, nz))
        voltage_field = np.zeros((nx, ny, nz))
        overpotentials = {
            'ionic': np.zeros((nx, ny, nz)),
            'electronic': np.zeros((nx, ny, nz)),
            'activation': np.zeros((nx, ny, nz)),
            'concentration': np.zeros((nx, ny, nz))
        }
        
        # Channel/rib pattern
        rib_pattern = np.zeros((nx, ny))
        channel_width = nx // 10
        for i in range(0, nx, 2 * channel_width):
            rib_pattern[i:i+channel_width, :] = 1.0
        
        # Temperature field (initial guess)
        T = conditions.temperature
        T_field = T * np.ones((nx, ny, nz))
        
        # Solve for current distribution layer by layer
        anode_layers = nz // 3
        electrolyte_layers = nz // 6
        
        for k in range(nz):
            if k < anode_layers:
                # Anode layer - H2 oxidation
                for i in range(nx):
                    for j in range(ny):
                        if microstructure['phase'][i, j, k] == 1:  # Ni phase
                            # Local current depends on TPB density
                            tpb = microstructure['tpb_density'][i, j, k]
                            if tpb > 0:
                                # Butler-Volmer with microstructure correction
                                local_T = T_field[i, j, k]
                                i0 = 5000 * tpb * np.exp(-100000 / (8.314 * local_T))
                                
                                # Current collection efficiency
                                collection_factor = 1.0 if rib_pattern[i, j] > 0 else 0.7
                                
                                current_density[i, j, k] = conditions.current_density * collection_factor * tpb
                                
                                # Activation overpotential
                                overpotentials['activation'][i, j, k] = \
                                    self.electrochem.activation_overpotential(
                                        current_density[i, j, k], local_T, 'anode'
                                    )
            
            elif k < anode_layers + electrolyte_layers:
                # Electrolyte - ionic conduction
                for i in range(nx):
                    for j in range(ny):
                        # Ohmic losses
                        local_T = T_field[i, j, k]
                        sigma = self.mat['electrolyte']['ionic_conductivity_0'] * \
                                np.exp(-self.mat['electrolyte']['activation_energy'] / (8.314 * local_T))
                        
                        # Current continuity
                        if k > anode_layers:
                            current_density[i, j, k] = current_density[i, j, k-1]
                        
                        # Ionic overpotential
                        dz = self.geom['thickness']['electrolyte'] / electrolyte_layers
                        overpotentials['ionic'][i, j, k] = current_density[i, j, k] * dz / sigma
            
            else:
                # Cathode layer - O2 reduction
                for i in range(nx):
                    for j in range(ny):
                        if microstructure['phase'][i, j, k] == 3:  # LSM phase
                            tpb = microstructure['tpb_density'][i, j, k]
                            if tpb > 0:
                                local_T = T_field[i, j, k]
                                i0 = 2000 * tpb * np.exp(-120000 / (8.314 * local_T))
                                
                                collection_factor = 1.0 if rib_pattern[i, j] > 0 else 0.7
                                current_density[i, j, k] = conditions.current_density * collection_factor * tpb
                                
                                overpotentials['activation'][i, j, k] = \
                                    self.electrochem.activation_overpotential(
                                        current_density[i, j, k], local_T, 'cathode'
                                    )
        
        # Calculate voltage field
        V_cell, overpotentials_avg = self.electrochem.cell_voltage(conditions)
        
        for k in range(nz):
            layer_factor = 1.0
            if k < anode_layers:
                layer_factor = 0.95
            elif k >= anode_layers + electrolyte_layers:
                layer_factor = 0.93
                
            voltage_field[:, :, k] = V_cell * layer_factor - \
                                     overpotentials['activation'][:, :, k] - \
                                     overpotentials['ionic'][:, :, k]
        
        return {
            'current_density': current_density,
            'voltage_field': voltage_field,
            'overpotentials': overpotentials
        }
    
    def simulate_thermal_3D(self, conditions: SOFCOperatingConditions,
                           current_density: np.ndarray,
                           microstructure: Dict) -> np.ndarray:
        """3D thermal simulation with heat generation and conduction"""
        nx, ny, nz = self.mesh_res
        
        # Heat generation from current
        V_cell, overpotentials = self.electrochem.cell_voltage(conditions)
        q_gen = self.thermal.heat_generation_rate(
            conditions.current_density, V_cell, overpotentials['E_nernst']
        )
        
        # 3D temperature field
        T_field = np.zeros((nx, ny, nz))
        T_inlet = conditions.temperature
        
        # Solve heat equation (simplified explicit method)
        # Initialize with inlet temperature
        T_field[:, 0, :] = T_inlet
        
        # Flow direction heat transfer
        for j in range(1, ny):
            for k in range(nz):
                # Layer-specific thermal properties
                if k < nz // 3:  # Anode
                    k_th = self.mat['anode']['thermal_conductivity']
                    cp = self.mat['anode']['specific_heat']
                    rho = self.mat['anode']['density']
                elif k < 2 * nz // 3:  # Electrolyte
                    k_th = self.mat['electrolyte']['thermal_conductivity']
                    cp = self.mat['electrolyte']['specific_heat']
                    rho = self.mat['electrolyte']['density']
                else:  # Cathode
                    k_th = self.mat['cathode']['thermal_conductivity']
                    cp = self.mat['cathode']['specific_heat']
                    rho = self.mat['cathode']['density']
                
                # Account for porosity
                porosity = microstructure['porosity'][:, j, k]
                k_eff = k_th * (1 - porosity)
                
                # Temperature rise due to heat generation
                local_q = q_gen * current_density[:, j, k] / conditions.current_density
                
                # Convection from flow
                flow_cooling = 0.01 * (T_field[:, j-1, k] - T_inlet)
                
                # Update temperature
                T_field[:, j, k] = T_field[:, j-1, k] + local_q * 1e-5 - flow_cooling
        
        # Add hot spots and gradients
        for k in range(nz):
            # Interface heating
            if k == nz // 3 or k == 2 * nz // 3:
                T_field[:, :, k] += 10  # Interface resistance heating
                
            # Current collector heating
            for i in range(0, nx, nx // 5):
                for j in range(ny):
                    hot_spot = 20 * np.exp(-((i - nx//2)**2 + (j - ny//2)**2) / 1000)
                    T_field[max(0, i-5):min(nx, i+5), j, k] += hot_spot
        
        # Smooth the field
        T_field = ndimage.gaussian_filter(T_field, sigma=1)
        
        # Ensure minimum temperature
        T_field = np.maximum(T_field, T_inlet)
        
        return T_field
    
    def simulate_species_3D(self, conditions: SOFCOperatingConditions,
                           current_density: np.ndarray,
                           microstructure: Dict) -> Dict[str, np.ndarray]:
        """3D species transport simulation"""
        nx, ny, nz = self.mesh_res
        
        species = {
            'H2': np.zeros((nx, ny, nz)),
            'H2O': np.zeros((nx, ny, nz)),
            'O2': np.zeros((nx, ny, nz)),
            'N2': np.zeros((nx, ny, nz))
        }
        
        # Inlet compositions
        y_H2_in = conditions.inlet_fuel_composition.get('H2', 0.97)
        y_H2O_in = conditions.inlet_fuel_composition.get('H2O', 0.03)
        y_O2_in = conditions.inlet_air_composition.get('O2', 0.21)
        y_N2_in = conditions.inlet_air_composition.get('N2', 0.79)
        
        anode_layers = nz // 3
        cathode_start = 2 * nz // 3
        
        # Anode species (H2, H2O)
        for k in range(anode_layers):
            for j in range(ny):
                # Consumption along flow
                consumption_factor = j / ny * conditions.fuel_utilization
                
                # Account for porosity and tortuosity
                porosity = microstructure['porosity'][:, j, k]
                tortuosity = microstructure['tortuosity'][:, j, k]
                
                # Effective diffusion
                D_eff_factor = porosity / tortuosity
                
                # H2 depletion and H2O production
                species['H2'][:, j, k] = y_H2_in * (1 - consumption_factor) * D_eff_factor
                species['H2O'][:, j, k] = (y_H2O_in + y_H2_in * consumption_factor) * D_eff_factor
                
                # Add local variations due to current
                local_current = current_density[:, j, k] / np.mean(current_density[:, j, k] + 1e-10)
                species['H2'][:, j, k] *= (2 - local_current)
                species['H2O'][:, j, k] *= local_current
        
        # Cathode species (O2, N2)
        for k in range(cathode_start, nz):
            for j in range(ny):
                consumption_factor = j / ny * conditions.air_utilization
                
                porosity = microstructure['porosity'][:, j, k]
                tortuosity = microstructure['tortuosity'][:, j, k]
                D_eff_factor = porosity / tortuosity
                
                species['O2'][:, j, k] = y_O2_in * (1 - consumption_factor) * D_eff_factor
                species['N2'][:, j, k] = y_N2_in * D_eff_factor
                
                local_current = current_density[:, j, k] / np.mean(current_density[:, j, k] + 1e-10)
                species['O2'][:, j, k] *= (2 - local_current)
        
        # Smooth species fields
        for sp in species:
            species[sp] = ndimage.gaussian_filter(species[sp], sigma=0.5)
            species[sp] = np.maximum(species[sp], 0)  # Ensure non-negative
            
        return species
    
    def simulate_mechanical_3D(self, T_field: np.ndarray,
                              microstructure: Dict) -> Dict[str, np.ndarray]:
        """3D mechanical simulation with thermal stresses"""
        nx, ny, nz = self.mesh_res
        
        # Calculate thermal stresses
        stress_tensor = self.mechanical.thermal_stress(T_field, T_ref=298.15)
        
        # Add interface stresses
        anode_electrolyte = nz // 3
        electrolyte_cathode = 2 * nz // 3
        
        # CTE mismatch stresses at interfaces
        for k in [anode_electrolyte-1, anode_electrolyte, 
                 electrolyte_cathode-1, electrolyte_cathode]:
            if 0 <= k < nz:
                # Enhanced stress at interfaces
                stress_tensor[:, :, k, :] *= 2.0
        
        # Calculate strain tensor (simplified)
        strain_tensor = np.zeros_like(stress_tensor)
        
        for k in range(nz):
            # Layer-specific properties
            if k < anode_electrolyte:
                E = self.mat['anode']['youngs_modulus']
                nu = self.mat['anode']['poissons_ratio']
            elif k < electrolyte_cathode:
                E = self.mat['electrolyte']['youngs_modulus']
                nu = self.mat['electrolyte']['poissons_ratio']
            else:
                E = self.mat['cathode']['youngs_modulus']
                nu = self.mat['cathode']['poissons_ratio']
            
            # Hooke's law (simplified)
            strain_tensor[:, :, k, 0] = (stress_tensor[:, :, k, 0] - 
                                        nu * (stress_tensor[:, :, k, 1] + stress_tensor[:, :, k, 2])) / E
            strain_tensor[:, :, k, 1] = (stress_tensor[:, :, k, 1] - 
                                        nu * (stress_tensor[:, :, k, 0] + stress_tensor[:, :, k, 2])) / E
            strain_tensor[:, :, k, 2] = (stress_tensor[:, :, k, 2] - 
                                        nu * (stress_tensor[:, :, k, 0] + stress_tensor[:, :, k, 1])) / E
            
            # Shear strains
            G = E / (2 * (1 + nu))
            strain_tensor[:, :, k, 3] = stress_tensor[:, :, k, 3] / G
            strain_tensor[:, :, k, 4] = stress_tensor[:, :, k, 4] / G
            strain_tensor[:, :, k, 5] = stress_tensor[:, :, k, 5] / G
        
        # Von Mises stress
        von_mises = self.mechanical.von_mises_stress(stress_tensor)
        
        # Principal stresses
        principal_stress = np.zeros((nx, ny, nz, 3))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Construct stress matrix
                    stress_matrix = np.array([
                        [stress_tensor[i, j, k, 0], stress_tensor[i, j, k, 3], stress_tensor[i, j, k, 4]],
                        [stress_tensor[i, j, k, 3], stress_tensor[i, j, k, 1], stress_tensor[i, j, k, 5]],
                        [stress_tensor[i, j, k, 4], stress_tensor[i, j, k, 5], stress_tensor[i, j, k, 2]]
                    ])
                    
                    # Eigenvalues are principal stresses
                    eigvals = np.linalg.eigvalsh(stress_matrix)
                    principal_stress[i, j, k, :] = np.sort(eigvals)[::-1]
        
        return {
            'stress_tensor': stress_tensor,
            'strain_tensor': strain_tensor,
            'von_mises': von_mises,
            'principal_stress': principal_stress
        }
    
    def simulate_damage_3D(self, stress_data: Dict, T_field: np.ndarray,
                          conditions: SOFCOperatingConditions,
                          cycles: int) -> Dict[str, np.ndarray]:
        """3D damage field simulation"""
        nx, ny, nz = self.mesh_res
        
        von_mises = stress_data['von_mises']
        principal_stress = stress_data['principal_stress']
        
        # Initialize damage fields
        damage_fields = {
            'crack_initiation': np.zeros((nx, ny, nz)),
            'crack_propagation': np.zeros((nx, ny, nz)),
            'creep_damage': np.zeros((nx, ny, nz)),
            'fatigue_damage': np.zeros((nx, ny, nz)),
            'delamination': np.zeros((nx, ny, nz)),
            'total_damage': np.zeros((nx, ny, nz))
        }
        
        # Material failure properties by layer
        anode_layers = nz // 3
        electrolyte_layers = nz // 6
        
        for k in range(nz):
            # Layer-specific properties
            if k < anode_layers:
                sigma_crit = 100e6  # Critical stress for Ni-YSZ
                K_IC = 0.8e6  # Fracture toughness
            elif k < anode_layers + electrolyte_layers:
                sigma_crit = 300e6  # YSZ is stronger
                K_IC = 2e6
            else:
                sigma_crit = 150e6  # LSM-YSZ
                K_IC = 1e6
            
            for i in range(nx):
                for j in range(ny):
                    # Local stress and temperature
                    sigma_vm = von_mises[i, j, k]
                    sigma_1 = principal_stress[i, j, k, 0]  # Maximum principal
                    T_local = T_field[i, j, k]
                    
                    # Crack initiation (Weibull statistics)
                    if sigma_1 > 0:  # Tensile stress
                        P_init = 1 - np.exp(-(sigma_1 / sigma_crit)**10)
                        damage_fields['crack_initiation'][i, j, k] = P_init
                    
                    # Crack propagation (Paris law)
                    if sigma_vm > 0.5 * sigma_crit:
                        a0 = 1e-6  # Initial flaw size
                        K = sigma_vm * np.sqrt(np.pi * a0)
                        if K < K_IC:
                            da_dN = 1e-12 * K**3
                            a_final = a0 + da_dN * cycles
                            damage_fields['crack_propagation'][i, j, k] = min(a_final / 1e-3, 1.0)
                    
                    # Creep damage
                    eps_creep = self.mechanical.creep_strain_rate(sigma_vm, T_local, conditions.time) * conditions.time * 3600
                    damage_fields['creep_damage'][i, j, k] = min(eps_creep / 0.01, 1.0)
                    
                    # Fatigue damage (Coffin-Manson)
                    if cycles > 0:
                        N_f = 1e6 * (sigma_crit / max(sigma_vm, 1e6))**3
                        damage_fields['fatigue_damage'][i, j, k] = min(cycles / N_f, 1.0)
                    
                    # Delamination at interfaces
                    if k in [anode_layers-1, anode_layers, 
                           anode_layers + electrolyte_layers - 1,
                           anode_layers + electrolyte_layers]:
                        # Enhanced delamination risk at interfaces
                        G = 0.5 * (stress_data['stress_tensor'][i, j, k, 2]**2) * 1e-6  # Energy release rate
                        G_c = 50  # Critical value J/m²
                        damage_fields['delamination'][i, j, k] = min(G / G_c, 1.0)
                    
                    # Total damage (interaction rule)
                    damages = [
                        damage_fields['crack_initiation'][i, j, k],
                        damage_fields['crack_propagation'][i, j, k],
                        damage_fields['creep_damage'][i, j, k],
                        damage_fields['fatigue_damage'][i, j, k],
                        damage_fields['delamination'][i, j, k]
                    ]
                    
                    # Linear damage accumulation
                    damage_fields['total_damage'][i, j, k] = min(sum(damages) / 2, 1.0)
        
        return damage_fields
    
    def simulate(self, conditions: SOFCOperatingConditions, cycles: int = 0) -> Dict:
        """Run complete high-fidelity 3D simulation"""
        
        print("  Generating microstructure...")
        microstructure = self.create_microstructure_3D()
        
        print("  Running 3D electrochemistry...")
        electro_3d = self.simulate_electrochemistry_3D(conditions, microstructure)
        
        print("  Running 3D thermal analysis...")
        T_field_3d = self.simulate_thermal_3D(conditions, electro_3d['current_density'], microstructure)
        
        print("  Running 3D species transport...")
        species_3d = self.simulate_species_3D(conditions, electro_3d['current_density'], microstructure)
        
        print("  Running 3D mechanical analysis...")
        mech_3d = self.simulate_mechanical_3D(T_field_3d, microstructure)
        
        print("  Running 3D damage analysis...")
        damage_3d = self.simulate_damage_3D(mech_3d, T_field_3d, conditions, cycles)
        
        # Average voltage
        V_cell = np.mean(electro_3d['voltage_field'])
        
        # Compile results
        results = {
            # Operating conditions
            'temperature_inlet': conditions.temperature,
            'pressure': conditions.pressure,
            'current_density_avg': conditions.current_density,
            'fuel_utilization': conditions.fuel_utilization,
            'air_utilization': conditions.air_utilization,
            'time': conditions.time,
            'cycles': cycles,
            
            # Average outputs
            'voltage': V_cell,
            'power_density': V_cell * conditions.current_density,
            'temperature_max': np.max(T_field_3d),
            'temperature_min': np.min(T_field_3d),
            'stress_max': np.max(mech_3d['von_mises']),
            'damage_max': np.max(damage_3d['total_damage']),
            
            # 3D fields (stored separately due to size)
            'fields_3d': {
                'temperature': T_field_3d,
                'current_density': electro_3d['current_density'],
                'voltage': electro_3d['voltage_field'],
                'stress_tensor': mech_3d['stress_tensor'],
                'strain_tensor': mech_3d['strain_tensor'],
                'von_mises_stress': mech_3d['von_mises'],
                'principal_stress': mech_3d['principal_stress'],
                'damage_total': damage_3d['total_damage'],
                'damage_crack_init': damage_3d['crack_initiation'],
                'damage_creep': damage_3d['creep_damage'],
                'species_H2': species_3d['H2'],
                'species_H2O': species_3d['H2O'],
                'species_O2': species_3d['O2'],
                'microstructure_phase': microstructure['phase'],
                'microstructure_porosity': microstructure['porosity'],
                'microstructure_tpb': microstructure['tpb_density']
            }
        }
        
        return results


class HighFidelityDatasetGenerator:
    """Generate high-fidelity SOFC dataset with 3D fields"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = HighFidelitySOFCModel(config)
        self.hf_config = config['dataset']['high_fidelity']
        
    def select_critical_conditions(self, mf_dataset_path: str, n_samples: int) -> List[Tuple[SOFCOperatingConditions, int]]:
        """Select critical conditions from MF dataset"""
        
        # Load MF dataset
        if os.path.exists(mf_dataset_path):
            df_mf = pd.read_csv(mf_dataset_path.replace('.h5', '_scalars.csv'))
        else:
            print("MF dataset not found, generating random conditions")
            from generators.mid_fidelity import MidFidelityDatasetGenerator
            mf_gen = MidFidelityDatasetGenerator(self.config)
            return mf_gen.generate_random_conditions(n_samples)
        
        # Select most critical conditions
        df_mf['criticality'] = (
            df_mf['stress_max'] / df_mf['stress_max'].max() * 2 +
            df_mf['temperature_max'] / df_mf['temperature_max'].max() +
            df_mf['damage_max'] * 3 +
            (df_mf['voltage'].min() / df_mf['voltage']) * 2
        )
        
        # Select top critical samples
        df_critical = df_mf.nlargest(n_samples, 'criticality')
        
        # Convert to operating conditions
        conditions_list = []
        
        for _, row in df_critical.iterrows():
            fuel_comp = {'H2': 0.97, 'H2O': 0.03}
            air_comp = {'O2': 0.21, 'N2': 0.79}
            
            cond = SOFCOperatingConditions(
                temperature=row['temperature_inlet'],
                pressure=row['pressure'],
                current_density=row['current_density_avg'],
                fuel_utilization=row['fuel_utilization'],
                air_utilization=row['air_utilization'],
                inlet_fuel_composition=fuel_comp,
                inlet_air_composition=air_comp,
                time=row['time']
            )
            
            conditions_list.append((cond, int(row['cycles'])))
            
        return conditions_list
    
    def generate_dataset(self, n_samples: int = None,
                        mf_dataset_path: str = 'data/mf_dataset.h5') -> Tuple[pd.DataFrame, Dict, Dict]:
        """Generate complete high-fidelity dataset"""
        
        if n_samples is None:
            n_samples = self.hf_config['n_samples']
            
        print(f"Generating {n_samples} high-fidelity samples...")
        print("Note: Each HF simulation takes significant time...")
        
        # Select critical conditions
        conditions_with_cycles = self.select_critical_conditions(mf_dataset_path, n_samples)
        
        # Run simulations
        results_list = []
        all_3d_fields = []
        
        for idx, (cond, cycles) in enumerate(conditions_with_cycles[:n_samples]):
            print(f"\nHF Simulation {idx+1}/{n_samples}")
            try:
                result = self.model.simulate(cond, cycles)
                
                # Separate scalar and field data
                scalar_result = {k: v for k, v in result.items() if k != 'fields_3d'}
                results_list.append(scalar_result)
                
                # Store 3D fields
                all_3d_fields.append(result['fields_3d'])
                
            except Exception as e:
                print(f"Simulation failed: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results_list)
        
        # Add metadata
        metadata = {
            'fidelity': 'high',
            'model_type': self.hf_config['model_type'],
            'mesh_resolution': self.hf_config['mesh_resolution'],
            'n_samples': len(df),
            'generation_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        return df, all_3d_fields, metadata
    
    def save_dataset(self, df: pd.DataFrame, fields_3d: List[Dict], metadata: Dict,
                     filename: str = "hf_dataset.h5"):
        """Save high-fidelity dataset with 3D fields"""
        
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save scalar data
            scalar_group = f.create_group('scalar_data')
            for col in df.columns:
                scalar_group.create_dataset(col, data=df[col].values,
                                          compression='gzip')
            
            # Save 3D fields
            fields_group = f.create_group('field_data_3d')
            
            for sample_idx, sample_fields in enumerate(fields_3d):
                sample_group = fields_group.create_group(f'sample_{sample_idx:04d}')
                
                for field_name, field_data in sample_fields.items():
                    sample_group.create_dataset(field_name, data=field_data,
                                              compression='gzip',
                                              compression_opts=9)
            
            # Save metadata
            f.attrs['metadata'] = json.dumps(metadata, default=str)
            f.attrs['scalar_columns'] = list(df.columns)
            f.attrs['n_samples'] = len(fields_3d)
            
        print(f"Dataset saved to {filepath}")
        
        # Save scalar data as CSV
        csv_path = filepath.replace('.h5', '_scalars.csv')
        df.to_csv(csv_path, index=False)
        print(f"Scalar data CSV saved to {csv_path}")
        
        return filepath


def generate_hf_dataset(config_path: str = 'config.yaml',
                       n_samples: int = None,
                       mf_dataset_path: str = 'data/mf_dataset.h5'):
    """Main function to generate high-fidelity dataset"""
    
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = HighFidelityDatasetGenerator(config)
    
    # Generate dataset
    df, fields_3d, metadata = generator.generate_dataset(n_samples, mf_dataset_path)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nScalar feature ranges:")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    print(f"\n3D Fields: {len(fields_3d)} samples")
    if fields_3d:
        print("Field names:", list(fields_3d[0].keys()))
        for field_name, field_data in fields_3d[0].items():
            print(f"  {field_name}: shape {field_data.shape}")
    
    # Save dataset
    filepath = generator.save_dataset(df, fields_3d, metadata)
    
    return df, fields_3d, filepath


if __name__ == "__main__":
    # Generate high-fidelity dataset
    df, fields, filepath = generate_hf_dataset(n_samples=5)  # Very small sample for testing
    print(f"\nDataset generation complete! File saved at: {filepath}")