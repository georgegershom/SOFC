#!/usr/bin/env python3
"""
SOFC Abaqus Model Generator
Creates a complete 2D SOFC simulation model with:
- Geometry with layer partitions
- Temperature-dependent materials
- Mesh with interface refinement
- Sequential heat transfer + thermo-mechanical analysis
- Multiple heating rate scenarios
"""

import numpy as np
import os

def create_sofc_geometry():
    """Create 2D SOFC geometry with proper layer partitions"""
    
    # Geometry dimensions (mm)
    width = 10.0  # mm
    total_thickness = 1.0  # mm
    
    # Layer thicknesses (mm)
    anode_thickness = 0.40
    electrolyte_thickness = 0.10  # 0.40 to 0.50
    cathode_thickness = 0.40     # 0.50 to 0.90
    interconnect_thickness = 0.10 # 0.90 to 1.00
    
    # Interface positions
    y_anode_electrolyte = 0.40
    y_electrolyte_cathode = 0.50
    y_cathode_interconnect = 0.90
    
    return {
        'width': width,
        'total_thickness': total_thickness,
        'anode_thickness': anode_thickness,
        'electrolyte_thickness': electrolyte_thickness,
        'cathode_thickness': cathode_thickness,
        'interconnect_thickness': interconnect_thickness,
        'y_ae': y_anode_electrolyte,
        'y_ec': y_electrolyte_cathode,
        'y_ci': y_cathode_interconnect
    }

def create_mesh_parameters():
    """Define mesh parameters with interface refinement"""
    
    # Base mesh density
    nx_base = 80  # elements along x-direction
    ny_anode = 20
    ny_electrolyte = 12  # refined for thin electrolyte
    ny_cathode = 20
    ny_interconnect = 5
    
    # Interface refinement zones (±0.02 mm around interfaces)
    interface_refinement = 0.02  # mm
    ny_interface = 8  # elements in refinement zone
    
    return {
        'nx_base': nx_base,
        'ny_anode': ny_anode,
        'ny_electrolyte': ny_electrolyte,
        'ny_cathode': ny_cathode,
        'ny_interconnect': ny_interconnect,
        'interface_refinement': interface_refinement,
        'ny_interface': ny_interface
    }

def create_materials():
    """Define temperature-dependent material properties"""
    
    materials = {
        'ni_ysz_anode': {
            'elastic': {
                'E_298': 140.0e9,  # Pa
                'E_1273': 91.0e9,   # Pa
                'nu': 0.30
            },
            'expansion': {
                'alpha_298': 12.5e-6,  # 1/K
                'alpha_1273': 13.5e-6  # 1/K
            },
            'thermal': {
                'k_298': 6.0,  # W/m·K
                'k_1273': 4.0,  # W/m·K
                'cp_298': 450.0,  # J/kg·K
                'cp_1273': 570.0  # J/kg·K
            },
            'plastic': {
                'A': 150.0e6,  # Pa
                'B': 200.0e6,  # Pa
                'n': 0.35,
                'C': 0.02,
                'm': 1.0,
                'T_room': 298.0,  # K
                'T_melt': 1720.0,  # K
                'eps_dot_0': 1.0  # 1/s
            },
            'creep': {
                'B': 1.0e-18,  # Pa^-n s^-1
                'n': 3.5,
                'Q': 2.2e5  # J/mol
            }
        },
        'ysz_electrolyte': {
            'elastic': {
                'E_298': 210.0e9,  # Pa
                'E_1273': 170.0e9,  # Pa
                'nu': 0.28
            },
            'expansion': {
                'alpha_298': 10.5e-6,  # 1/K
                'alpha_1273': 11.2e-6  # 1/K
            },
            'thermal': {
                'k_298': 2.6,  # W/m·K
                'k_1273': 2.0,  # W/m·K
                'cp_298': 400.0,  # J/kg·K
                'cp_1273': 600.0  # J/kg·K
            },
            'creep': {
                'B': 5.0e-22,  # Pa^-n s^-1
                'n': 2.0,
                'Q': 3.8e5  # J/mol
            }
        },
        'lsm_cathode': {
            'elastic': {
                'E_298': 120.0e9,  # Pa
                'E_1273': 84.0e9,   # Pa
                'nu': 0.30
            },
            'expansion': {
                'alpha_298': 11.5e-6,  # 1/K
                'alpha_1273': 12.4e-6  # 1/K
            },
            'thermal': {
                'k_298': 2.0,  # W/m·K
                'k_1273': 1.8,  # W/m·K
                'cp_298': 480.0,  # J/kg·K
                'cp_1273': 610.0  # J/kg·K
            }
        },
        'ferritic_steel': {
            'elastic': {
                'E_298': 205.0e9,  # Pa
                'E_1273': 150.0e9,  # Pa
                'nu': 0.30
            },
            'expansion': {
                'alpha_298': 12.5e-6,  # 1/K
                'alpha_1273': 13.2e-6  # 1/K
            },
            'thermal': {
                'k_298': 20.0,  # W/m·K
                'k_1273': 15.0,  # W/m·K
                'cp_298': 500.0,  # J/kg·K
                'cp_1273': 700.0  # J/kg·K
            }
        }
    }
    
    return materials

def create_heating_rates():
    """Define heating rate amplitude curves"""
    
    # Target temperature and hold time
    T_room = 25.0  # °C
    T_target = 900.0  # °C
    hold_time = 10.0  # minutes
    
    heating_rates = {
        'HR1': {
            'rate': 1.0,  # °C/min
            'ramp_time': (T_target - T_room) / 1.0,  # minutes
            'hold_time': hold_time,
            'cool_time': (T_target - T_room) / 1.0,
            'total_time': 2 * (T_target - T_room) / 1.0 + hold_time
        },
        'HR4': {
            'rate': 4.0,  # °C/min
            'ramp_time': (T_target - T_room) / 4.0,  # minutes
            'hold_time': hold_time,
            'cool_time': (T_target - T_room) / 4.0,
            'total_time': 2 * (T_target - T_room) / 4.0 + hold_time
        },
        'HR10': {
            'rate': 10.0,  # °C/min
            'ramp_time': (T_target - T_room) / 10.0,  # minutes
            'hold_time': hold_time,
            'cool_time': (T_target - T_room) / 10.0,
            'total_time': 2 * (T_target - T_room) / 10.0 + hold_time
        }
    }
    
    return heating_rates

def generate_abaqus_input(geometry, mesh_params, materials, heating_rates, heating_rate='HR1'):
    """Generate complete Abaqus input file"""
    
    # Get current heating rate parameters
    hr_params = heating_rates[heating_rate]
    
    # Convert times to seconds
    ramp_time_s = hr_params['ramp_time'] * 60.0
    hold_time_s = hr_params['hold_time'] * 60.0
    cool_time_s = hr_params['cool_time'] * 60.0
    total_time_s = hr_params['total_time'] * 60.0
    
    # Create amplitude points
    amp_points = [
        (0.0, 25.0),
        (ramp_time_s, 900.0),
        (ramp_time_s + hold_time_s, 900.0),
        (total_time_s, 25.0)
    ]
    
    inp_content = f"""*HEADING
SOFC Multi-Physics Simulation - Heat Transfer + Thermo-Mechanical Analysis
2D Cross-section of Single SOFC Repeat Unit
Sequential Analysis: Heat Transfer -> Thermo-Mechanical
Heating Rate: {heating_rate} ({hr_params['rate']} °C/min)

*PART, NAME=SOFC_PART
*NODE
"""
    
    # Generate nodes
    nx = mesh_params['nx_base']
    ny_total = (mesh_params['ny_anode'] + mesh_params['ny_electrolyte'] + 
                mesh_params['ny_cathode'] + mesh_params['ny_interconnect'])
    
    node_id = 1
    elem_id = 1
    
    # Create node coordinates
    x_coords = np.linspace(0, geometry['width']*1e-3, nx+1)  # Convert to meters
    y_coords = np.linspace(0, geometry['total_thickness']*1e-3, ny_total+1)  # Convert to meters
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            inp_content += f"{node_id:8d},{x:12.6e},{y:12.6e}\n"
            node_id += 1
    
    # Generate elements
    inp_content += f"""
*ELEMENT, TYPE=DC2D4
"""
    
    for i in range(nx):
        for j in range(ny_total):
            n1 = i * (ny_total + 1) + j + 1
            n2 = n1 + 1
            n3 = n1 + (ny_total + 1) + 1
            n4 = n1 + (ny_total + 1)
            inp_content += f"{elem_id:8d},{n1:8d},{n2:8d},{n3:8d},{n4:8d}\n"
            elem_id += 1
    
    # Create node sets
    inp_content += f"""
*NSET, NSET=ANODE_NODES
"""
    # Anode nodes (y = 0 to 0.4 mm)
    anode_y_max = geometry['y_ae'] * 1e-3
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if y <= anode_y_max:
                node_id = i * (ny_total + 1) + j + 1
                inp_content += f"{node_id:8d},\n"
    
    inp_content += f"""
*NSET, NSET=ELYTE_NODES
"""
    # Electrolyte nodes (y = 0.4 to 0.5 mm)
    elyte_y_min = geometry['y_ae'] * 1e-3
    elyte_y_max = geometry['y_ec'] * 1e-3
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if elyte_y_min < y <= elyte_y_max:
                node_id = i * (ny_total + 1) + j + 1
                inp_content += f"{node_id:8d},\n"
    
    inp_content += f"""
*NSET, NSET=CATH_NODES
"""
    # Cathode nodes (y = 0.5 to 0.9 mm)
    cath_y_min = geometry['y_ec'] * 1e-3
    cath_y_max = geometry['y_ci'] * 1e-3
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if cath_y_min < y <= cath_y_max:
                node_id = i * (ny_total + 1) + j + 1
                inp_content += f"{node_id:8d},\n"
    
    inp_content += f"""
*NSET, NSET=INTCONN_NODES
"""
    # Interconnect nodes (y = 0.9 to 1.0 mm)
    intconn_y_min = geometry['y_ci'] * 1e-3
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if intconn_y_min < y:
                node_id = i * (ny_total + 1) + j + 1
                inp_content += f"{node_id:8d},\n"
    
    # Boundary node sets
    inp_content += f"""
*NSET, NSET=X0_NODES
"""
    # Left edge nodes (x = 0)
    for j in range(ny_total + 1):
        node_id = j + 1
        inp_content += f"{node_id:8d},\n"
    
    inp_content += f"""
*NSET, NSET=Y0_NODES
"""
    # Bottom edge nodes (y = 0)
    for i in range(nx + 1):
        node_id = i * (ny_total + 1) + 1
        inp_content += f"{node_id:8d},\n"
    
    inp_content += f"""
*NSET, NSET=YTOP_NODES
"""
    # Top edge nodes (y = 1.0 mm)
    for i in range(nx + 1):
        node_id = i * (ny_total + 1) + ny_total + 1
        inp_content += f"{node_id:8d},\n"
    
    # Element sets
    inp_content += f"""
*ELSET, ELSET=ANODE_ELEMS
"""
    # Anode elements
    for i in range(nx):
        for j in range(mesh_params['ny_anode']):
            elem_id = i * ny_total + j + 1
            inp_content += f"{elem_id:8d},\n"
    
    inp_content += f"""
*ELSET, ELSET=ELYTE_ELEMS
"""
    # Electrolyte elements
    for i in range(nx):
        for j in range(mesh_params['ny_anode'], mesh_params['ny_anode'] + mesh_params['ny_electrolyte']):
            elem_id = i * ny_total + j + 1
            inp_content += f"{elem_id:8d},\n"
    
    inp_content += f"""
*ELSET, ELSET=CATH_ELEMS
"""
    # Cathode elements
    for i in range(nx):
        for j in range(mesh_params['ny_anode'] + mesh_params['ny_electrolyte'], 
                      mesh_params['ny_anode'] + mesh_params['ny_electrolyte'] + mesh_params['ny_cathode']):
            elem_id = i * ny_total + j + 1
            inp_content += f"{elem_id:8d},\n"
    
    inp_content += f"""
*ELSET, ELSET=INTCONN_ELEMS
"""
    # Interconnect elements
    for i in range(nx):
        for j in range(mesh_params['ny_anode'] + mesh_params['ny_electrolyte'] + mesh_params['ny_cathode'], ny_total):
            elem_id = i * ny_total + j + 1
            inp_content += f"{elem_id:8d},\n"
    
    # Materials
    inp_content += f"""
*MATERIAL, NAME=NI_YSZ_ANODE
*ELASTIC, DEPENDENCIES=1
{materials['ni_ysz_anode']['elastic']['E_298']:12.3e},{materials['ni_ysz_anode']['elastic']['nu']:8.3f}, 298.0
{materials['ni_ysz_anode']['elastic']['E_1273']:12.3e},{materials['ni_ysz_anode']['elastic']['nu']:8.3f}, 1273.0
*EXPANSION, DEPENDENCIES=1
{materials['ni_ysz_anode']['expansion']['alpha_298']:12.3e}, 298.0
{materials['ni_ysz_anode']['expansion']['alpha_1273']:12.3e}, 1273.0
*CONDUCTIVITY, DEPENDENCIES=1
{materials['ni_ysz_anode']['thermal']['k_298']:8.3f}, 298.0
{materials['ni_ysz_anode']['thermal']['k_1273']:8.3f}, 1273.0
*SPECIFIC HEAT, DEPENDENCIES=1
{materials['ni_ysz_anode']['thermal']['cp_298']:8.3f}, 298.0
{materials['ni_ysz_anode']['thermal']['cp_1273']:8.3f}, 1273.0
*PLASTIC, HARDENING=JOHNSON COOK
{materials['ni_ysz_anode']['plastic']['A']:12.3e},{materials['ni_ysz_anode']['plastic']['B']:12.3e},{materials['ni_ysz_anode']['plastic']['n']:8.3f},{materials['ni_ysz_anode']['plastic']['C']:8.3f},{materials['ni_ysz_anode']['plastic']['m']:8.3f},{materials['ni_ysz_anode']['plastic']['T_room']:8.3f},{materials['ni_ysz_anode']['plastic']['T_melt']:8.3f},{materials['ni_ysz_anode']['plastic']['eps_dot_0']:8.3f}
*CREEP, LAW=TIME, HARDENING=NONE
{materials['ni_ysz_anode']['creep']['B']:12.3e},{materials['ni_ysz_anode']['creep']['n']:8.3f},{materials['ni_ysz_anode']['creep']['Q']:12.3e}

*MATERIAL, NAME=YSZ_ELECTROLYTE
*ELASTIC, DEPENDENCIES=1
{materials['ysz_electrolyte']['elastic']['E_298']:12.3e},{materials['ysz_electrolyte']['elastic']['nu']:8.3f}, 298.0
{materials['ysz_electrolyte']['elastic']['E_1273']:12.3e},{materials['ysz_electrolyte']['elastic']['nu']:8.3f}, 1273.0
*EXPANSION, DEPENDENCIES=1
{materials['ysz_electrolyte']['expansion']['alpha_298']:12.3e}, 298.0
{materials['ysz_electrolyte']['expansion']['alpha_1273']:12.3e}, 1273.0
*CONDUCTIVITY, DEPENDENCIES=1
{materials['ysz_electrolyte']['thermal']['k_298']:8.3f}, 298.0
{materials['ysz_electrolyte']['thermal']['k_1273']:8.3f}, 1273.0
*SPECIFIC HEAT, DEPENDENCIES=1
{materials['ysz_electrolyte']['thermal']['cp_298']:8.3f}, 298.0
{materials['ysz_electrolyte']['thermal']['cp_1273']:8.3f}, 1273.0
*CREEP, LAW=TIME, HARDENING=NONE
{materials['ysz_electrolyte']['creep']['B']:12.3e},{materials['ysz_electrolyte']['creep']['n']:8.3f},{materials['ysz_electrolyte']['creep']['Q']:12.3e}

*MATERIAL, NAME=LSM_CATHODE
*ELASTIC, DEPENDENCIES=1
{materials['lsm_cathode']['elastic']['E_298']:12.3e},{materials['lsm_cathode']['elastic']['nu']:8.3f}, 298.0
{materials['lsm_cathode']['elastic']['E_1273']:12.3e},{materials['lsm_cathode']['elastic']['nu']:8.3f}, 1273.0
*EXPANSION, DEPENDENCIES=1
{materials['lsm_cathode']['expansion']['alpha_298']:12.3e}, 298.0
{materials['lsm_cathode']['expansion']['alpha_1273']:12.3e}, 1273.0
*CONDUCTIVITY, DEPENDENCIES=1
{materials['lsm_cathode']['thermal']['k_298']:8.3f}, 298.0
{materials['lsm_cathode']['thermal']['k_1273']:8.3f}, 1273.0
*SPECIFIC HEAT, DEPENDENCIES=1
{materials['lsm_cathode']['thermal']['cp_298']:8.3f}, 298.0
{materials['lsm_cathode']['thermal']['cp_1273']:8.3f}, 1273.0

*MATERIAL, NAME=FERRITIC_STEEL
*ELASTIC, DEPENDENCIES=1
{materials['ferritic_steel']['elastic']['E_298']:12.3e},{materials['ferritic_steel']['elastic']['nu']:8.3f}, 298.0
{materials['ferritic_steel']['elastic']['E_1273']:12.3e},{materials['ferritic_steel']['elastic']['nu']:8.3f}, 1273.0
*EXPANSION, DEPENDENCIES=1
{materials['ferritic_steel']['expansion']['alpha_298']:12.3e}, 298.0
{materials['ferritic_steel']['expansion']['alpha_1273']:12.3e}, 1273.0
*CONDUCTIVITY, DEPENDENCIES=1
{materials['ferritic_steel']['thermal']['k_298']:8.3f}, 298.0
{materials['ferritic_steel']['thermal']['k_1273']:8.3f}, 1273.0
*SPECIFIC HEAT, DEPENDENCIES=1
{materials['ferritic_steel']['thermal']['cp_298']:8.3f}, 298.0
{materials['ferritic_steel']['thermal']['cp_1273']:8.3f}, 1273.0

*SOLID SECTION, ELSET=ANODE_ELEMS, MATERIAL=NI_YSZ_ANODE
*SOLID SECTION, ELSET=ELYTE_ELEMS, MATERIAL=YSZ_ELECTROLYTE
*SOLID SECTION, ELSET=CATH_ELEMS, MATERIAL=LSM_CATHODE
*SOLID SECTION, ELSET=INTCONN_ELEMS, MATERIAL=FERRITIC_STEEL

*AMPLITUDE, NAME={heating_rate}_AMPLITUDE, TIME=TOTAL TIME
"""
    
    # Add amplitude points
    for time, temp in amp_points:
        inp_content += f"{time:12.3f},{temp:8.3f}\n"
    
    # Analysis steps
    inp_content += f"""
*STEP, NAME=HEAT_TRANSFER, INC=1000
*HEAT TRANSFER, DELTMX=200.0
*BOUNDARY
Y0_NODES, 11, 11, 25.0
*FILM
YTOP_NODES, F, 25.0, 25.0
*OUTPUT, FIELD
*NODE OUTPUT
NT, HFL
*ELEMENT OUTPUT
HFL
*END STEP

*STEP, NAME=THERMO_MECHANICAL, INC=1000, NLGEOM
*STATIC
*BOUNDARY
X0_NODES, 1, 1, 0.0
Y0_NODES, 2, 2, 0.0
*TEMPERATURE
ALL_NODES, 298.0
*OUTPUT, FIELD
*NODE OUTPUT
TEMP
*ELEMENT OUTPUT
S, Mises, LE, PEEQ, CEEQ
*OUTPUT, HISTORY
*NODE OUTPUT, NSET=INT_AE_NODES
*ELEMENT OUTPUT, ELSET=ANODE_ELEMS
S
*ELEMENT OUTPUT, ELSET=ELYTE_ELEMS
S
*END STEP
"""
    
    return inp_content

def main():
    """Main function to generate SOFC models for all heating rates"""
    
    # Create model parameters
    geometry = create_sofc_geometry()
    mesh_params = create_mesh_parameters()
    materials = create_materials()
    heating_rates = create_heating_rates()
    
    # Generate models for all heating rates
    for hr in ['HR1', 'HR4', 'HR10']:
        print(f"Generating model for {hr}...")
        
        inp_content = generate_abaqus_input(geometry, mesh_params, materials, heating_rates, hr)
        
        # Write to file
        filename = f"/workspace/sofc_simulation/sofc_{hr.lower()}.inp"
        with open(filename, 'w') as f:
            f.write(inp_content)
        
        print(f"Generated {filename}")
    
    print("All SOFC models generated successfully!")

if __name__ == "__main__":
    main()