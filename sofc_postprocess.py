#!/usr/bin/env python
"""
SOFC Post-Processing Script
============================
Extract results from Abaqus ODB and compute:
- Damage proxy field (D)
- Delamination risk at interfaces
- Crack depth metrics in electrolyte
- Export to NPZ format matching synthetic dataset structure

Usage:
  abaqus python sofc_postprocess.py Job_SOFC_HR1.odb
"""

import sys
import os
from odbAccess import *
from abaqusConstants import *
import numpy as np

# ============================================================================
# PARAMETERS (must match simulation setup)
# ============================================================================

# Geometry
WIDTH = 10.0e-3
THICK = 1.0e-3
Y_ANODE_TOP = 0.4e-3
Y_ELYTE_TOP = 0.5e-3
Y_CATH_TOP = 0.9e-3

# Delamination thresholds (Pa)
TAU_CRIT = {
    'AE': 25.0e6,
    'EC': 20.0e6,
    'CI': 30.0e6
}

# Damage model parameters
SIGMA_TH = 120.0e6  # Pa
K_DAMAGE = 1.5e-5
P_DAMAGE = 2.0

# Interface proximity decay length (m)
INTERFACE_DECAY = 0.05e-3

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_layer_from_y(y_coord):
    """Identify layer from y-coordinate."""
    if y_coord < Y_ANODE_TOP:
        return 'ANODE'
    elif y_coord < Y_ELYTE_TOP:
        return 'ELYTE'
    elif y_coord < Y_CATH_TOP:
        return 'CATH'
    else:
        return 'INTCONN'

def distance_to_interfaces(y_coord):
    """Compute minimum distance to any interface."""
    interfaces = [Y_ANODE_TOP, Y_ELYTE_TOP, Y_CATH_TOP]
    return min(abs(y_coord - yi) for yi in interfaces)

def interface_proximity_weight(y_coord):
    """
    Compute interface proximity weight w_iface.
    Decays exponentially from interfaces.
    """
    dist = distance_to_interfaces(y_coord)
    return np.exp(-dist / INTERFACE_DECAY)

def compute_von_mises(stress_tensor):
    """
    Compute von Mises stress from stress tensor.
    stress_tensor: [S11, S22, S33, S12, S13, S23] (Abaqus order)
    For plane stress: S33 = S13 = S23 = 0
    """
    if len(stress_tensor) >= 3:
        s11, s22, s33 = stress_tensor[0], stress_tensor[1], stress_tensor[2]
        s12 = stress_tensor[3] if len(stress_tensor) > 3 else 0.0
        
        # von Mises: sqrt(0.5*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2)))
        vm = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0 * s12**2)
        return vm
    else:
        return 0.0

def compute_damage_rate(sigma_vm, w_iface):
    """
    Compute damage rate dD/dt.
    D_dot = k_D * [max(0, (sigma_vm - sigma_th)/sigma_th)]^p * (1 + 3*w_iface)
    """
    if sigma_vm > SIGMA_TH:
        overstress_ratio = (sigma_vm - SIGMA_TH) / SIGMA_TH
        damage_rate = K_DAMAGE * (overstress_ratio ** P_DAMAGE) * (1.0 + 3.0 * w_iface)
        return damage_rate
    else:
        return 0.0

# ============================================================================
# MAIN POST-PROCESSING
# ============================================================================

def postprocess_odb(odb_path):
    """
    Extract and process results from ODB.
    """
    
    print("\n" + "="*70)
    print(f"POST-PROCESSING: {odb_path}")
    print("="*70 + "\n")
    
    if not os.path.exists(odb_path):
        print(f"ERROR: ODB file not found: {odb_path}")
        return
    
    # Open ODB
    print("Opening ODB...")
    odb = openOdb(path=odb_path, readOnly=True)
    
    # Get instance and steps
    instance_name = odb.rootAssembly.instances.keys()[0]
    instance = odb.rootAssembly.instances[instance_name]
    
    step_names = odb.steps.keys()
    print(f"Steps found: {step_names}")
    
    # We're interested in the mechanical step (Step_Mech)
    if 'Step_Mech' not in step_names:
        print("ERROR: Step_Mech not found in ODB")
        odb.close()
        return
    
    step = odb.steps['Step_Mech']
    num_frames = len(step.frames)
    print(f"Frames in Step_Mech: {num_frames}")
    
    # ========================================================================
    # Extract node coordinates and element connectivity
    # ========================================================================
    
    print("\nExtracting mesh data...")
    
    nodes = instance.nodes
    elements = instance.elements
    
    num_nodes = len(nodes)
    num_elements = len(elements)
    
    print(f"Nodes: {num_nodes}, Elements: {num_elements}")
    
    # Node coordinates
    node_coords = np.zeros((num_nodes, 3))
    node_labels = {}
    for i, node in enumerate(nodes):
        node_coords[i] = node.coordinates
        node_labels[node.label] = i
    
    # Element connectivity and integration points
    elem_connectivity = []
    elem_labels = {}
    elem_centers = np.zeros((num_elements, 3))
    
    for i, elem in enumerate(elements):
        conn = [node_labels[n.label] for n in elem.connectivity]
        elem_connectivity.append(conn)
        elem_labels[elem.label] = i
        
        # Element center (average of node coords)
        elem_center = np.mean([node_coords[n] for n in conn], axis=0)
        elem_centers[i] = elem_center
    
    # ========================================================================
    # Process each frame
    # ========================================================================
    
    print("\nProcessing frames...")
    
    results = {
        'time': [],
        'stress': [],
        'strain': [],
        'temp': [],
        'peeq': [],
        'ceeq': [],
        'damage': [],
        'von_mises': [],
        'interface_shear': {'AE': [], 'EC': [], 'CI': []},
        'delamination_risk': {'AE': [], 'EC': [], 'CI': []}
    }
    
    # Process subset of frames (every 10th frame for efficiency)
    frame_indices = range(0, num_frames, max(1, num_frames // 100))
    
    for frame_idx in frame_indices:
        frame = step.frames[frame_idx]
        time_val = frame.frameValue
        
        print(f"  Frame {frame_idx}/{num_frames-1}, Time = {time_val:.1f} s")
        
        results['time'].append(time_val)
        
        # Initialize arrays for this frame
        stress_frame = np.zeros((num_elements, 6))  # S11, S22, S33, S12, S13, S23
        strain_frame = np.zeros((num_elements, 6))
        temp_frame = np.zeros(num_elements)
        peeq_frame = np.zeros(num_elements)
        ceeq_frame = np.zeros(num_elements)
        vm_frame = np.zeros(num_elements)
        
        # ----------------------------------------------------------------
        # Extract field outputs
        # ----------------------------------------------------------------
        
        # Stress
        if 'S' in frame.fieldOutputs:
            stress_field = frame.fieldOutputs['S']
            for value in stress_field.values:
                if value.elementLabel in elem_labels:
                    elem_idx = elem_labels[value.elementLabel]
                    stress_frame[elem_idx, :len(value.data)] = value.data
        
        # Strain (use LE or E)
        strain_key = 'LE' if 'LE' in frame.fieldOutputs else 'E'
        if strain_key in frame.fieldOutputs:
            strain_field = frame.fieldOutputs[strain_key]
            for value in strain_field.values:
                if value.elementLabel in elem_labels:
                    elem_idx = elem_labels[value.elementLabel]
                    strain_frame[elem_idx, :len(value.data)] = value.data
        
        # Temperature
        if 'TEMP' in frame.fieldOutputs:
            temp_field = frame.fieldOutputs['TEMP']
            for value in temp_field.values:
                # Temperature is at nodes, average to elements
                if value.nodeLabel in node_labels:
                    node_idx = node_labels[value.nodeLabel]
                    # Find elements containing this node
                    for elem_idx, conn in enumerate(elem_connectivity):
                        if node_idx in conn:
                            temp_frame[elem_idx] += value.data / len(conn)
        
        # Plastic strain
        if 'PEEQ' in frame.fieldOutputs:
            peeq_field = frame.fieldOutputs['PEEQ']
            for value in peeq_field.values:
                if value.elementLabel in elem_labels:
                    elem_idx = elem_labels[value.elementLabel]
                    peeq_frame[elem_idx] = value.data
        
        # Creep strain
        if 'CEEQ' in frame.fieldOutputs:
            ceeq_field = frame.fieldOutputs['CEEQ']
            for value in ceeq_field.values:
                if value.elementLabel in elem_labels:
                    elem_idx = elem_labels[value.elementLabel]
                    ceeq_frame[elem_idx] = value.data
        
        # ----------------------------------------------------------------
        # Compute von Mises stress
        # ----------------------------------------------------------------
        
        for i in range(num_elements):
            vm_frame[i] = compute_von_mises(stress_frame[i])
        
        # ----------------------------------------------------------------
        # Compute damage field
        # ----------------------------------------------------------------
        
        damage_frame = np.zeros(num_elements)
        
        # Integrate damage over time (simple forward Euler)
        if len(results['damage']) > 0:
            damage_prev = results['damage'][-1]
            time_prev = results['time'][-2] if len(results['time']) > 1 else 0.0
            dt = time_val - time_prev
            
            for i in range(num_elements):
                y_coord = elem_centers[i, 1]
                w_iface = interface_proximity_weight(y_coord)
                
                damage_rate = compute_damage_rate(vm_frame[i], w_iface)
                damage_frame[i] = min(1.0, damage_prev[i] + damage_rate * dt)
        else:
            # First frame: initialize damage
            damage_frame = np.zeros(num_elements)
        
        # ----------------------------------------------------------------
        # Interface shear stress and delamination risk
        # ----------------------------------------------------------------
        
        interface_shear = {'AE': [], 'EC': [], 'CI': []}
        
        # Find elements near each interface (within ±10 μm)
        tol = 10.0e-6
        
        for i in range(num_elements):
            y_coord = elem_centers[i, 1]
            s12 = stress_frame[i, 3]  # Shear stress
            
            if abs(y_coord - Y_ANODE_TOP) < tol:
                interface_shear['AE'].append(abs(s12))
            elif abs(y_coord - Y_ELYTE_TOP) < tol:
                interface_shear['EC'].append(abs(s12))
            elif abs(y_coord - Y_CATH_TOP) < tol:
                interface_shear['CI'].append(abs(s12))
        
        # Compute max shear at each interface
        max_shear = {}
        delam_risk = {}
        
        for iface in ['AE', 'EC', 'CI']:
            if len(interface_shear[iface]) > 0:
                max_shear[iface] = max(interface_shear[iface])
                delam_risk[iface] = max_shear[iface] / TAU_CRIT[iface]
            else:
                max_shear[iface] = 0.0
                delam_risk[iface] = 0.0
            
            results['interface_shear'][iface].append(max_shear[iface])
            results['delamination_risk'][iface].append(delam_risk[iface])
        
        # ----------------------------------------------------------------
        # Store frame results
        # ----------------------------------------------------------------
        
        results['stress'].append(stress_frame.copy())
        results['strain'].append(strain_frame.copy())
        results['temp'].append(temp_frame.copy())
        results['peeq'].append(peeq_frame.copy())
        results['ceeq'].append(ceeq_frame.copy())
        results['von_mises'].append(vm_frame.copy())
        results['damage'].append(damage_frame.copy())
    
    # ========================================================================
    # Crack depth analysis (electrolyte layer)
    # ========================================================================
    
    print("\nComputing crack depth in electrolyte...")
    
    # Find elements in electrolyte
    elyte_elem_indices = []
    for i in range(num_elements):
        y_coord = elem_centers[i, 1]
        if Y_ANODE_TOP <= y_coord <= Y_ELYTE_TOP:
            elyte_elem_indices.append(i)
    
    print(f"  Electrolyte elements: {len(elyte_elem_indices)}")
    
    # For each frame, find max damage depth
    crack_depth = []
    damage_threshold = 0.2
    
    for damage_frame in results['damage']:
        # Find y-coordinates where damage > threshold
        damaged_y = []
        for i in elyte_elem_indices:
            if damage_frame[i] > damage_threshold:
                damaged_y.append(elem_centers[i, 1])
        
        if len(damaged_y) > 0:
            # Depth = distance from top of electrolyte to lowest damaged point
            depth = Y_ELYTE_TOP - min(damaged_y)
            crack_depth.append(depth * 1e6)  # Convert to μm
        else:
            crack_depth.append(0.0)
    
    results['crack_depth_um'] = crack_depth
    
    print(f"  Max crack depth: {max(crack_depth):.2f} μm")
    
    # ========================================================================
    # Export to NPZ
    # ========================================================================
    
    output_name = os.path.splitext(os.path.basename(odb_path))[0]
    npz_path = output_name + '_results.npz'
    
    print(f"\nExporting results to {npz_path}...")
    
    # Convert lists to arrays
    export_dict = {
        'time': np.array(results['time']),
        'node_coords': node_coords,
        'elem_centers': elem_centers,
        'stress': np.array(results['stress']),
        'strain': np.array(results['strain']),
        'temperature': np.array(results['temp']),
        'peeq': np.array(results['peeq']),
        'ceeq': np.array(results['ceeq']),
        'von_mises': np.array(results['von_mises']),
        'damage_D': np.array(results['damage']),
        'crack_depth_um': np.array(results['crack_depth_um']),
        'interface_shear_AE': np.array(results['interface_shear']['AE']),
        'interface_shear_EC': np.array(results['interface_shear']['EC']),
        'interface_shear_CI': np.array(results['interface_shear']['CI']),
        'delamination_risk_AE': np.array(results['delamination_risk']['AE']),
        'delamination_risk_EC': np.array(results['delamination_risk']['EC']),
        'delamination_risk_CI': np.array(results['delamination_risk']['CI']),
    }
    
    np.savez_compressed(npz_path, **export_dict)
    
    print(f"Saved: {npz_path}")
    print(f"  Arrays: {list(export_dict.keys())}")
    print(f"  Total size: {os.path.getsize(npz_path) / 1e6:.2f} MB")
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Final frame statistics
    final_vm = results['von_mises'][-1]
    final_damage = results['damage'][-1]
    
    print(f"\nFinal frame (t = {results['time'][-1]:.1f} s):")
    print(f"  Max von Mises stress: {np.max(final_vm) / 1e6:.2f} MPa")
    print(f"  Max damage: {np.max(final_damage):.4f}")
    print(f"  Max crack depth: {crack_depth[-1]:.2f} μm")
    
    print(f"\nDelamination risk (final):")
    for iface in ['AE', 'EC', 'CI']:
        risk = results['delamination_risk'][iface][-1]
        status = "CRITICAL" if risk > 1.0 else "OK"
        print(f"  {iface}: {risk:.3f} [{status}]")
    
    print(f"\nElements with damage > 0.2: {np.sum(final_damage > 0.2)} / {num_elements}")
    print(f"Elements with damage > 0.5: {np.sum(final_damage > 0.5)} / {num_elements}")
    
    # ========================================================================
    # Generate summary CSV
    # ========================================================================
    
    csv_path = output_name + '_summary.csv'
    
    print(f"\nGenerating summary CSV: {csv_path}")
    
    with open(csv_path, 'w') as f:
        f.write("Time_s,MaxVonMises_MPa,MaxDamage,CrackDepth_um,DelamRisk_AE,DelamRisk_EC,DelamRisk_CI\n")
        
        for i in range(len(results['time'])):
            f.write(f"{results['time'][i]:.2f},")
            f.write(f"{np.max(results['von_mises'][i]) / 1e6:.4f},")
            f.write(f"{np.max(results['damage'][i]):.6f},")
            f.write(f"{results['crack_depth_um'][i]:.4f},")
            f.write(f"{results['delamination_risk']['AE'][i]:.6f},")
            f.write(f"{results['delamination_risk']['EC'][i]:.6f},")
            f.write(f"{results['delamination_risk']['CI'][i]:.6f}\n")
    
    print(f"Saved: {csv_path}")
    
    # Close ODB
    odb.close()
    
    print("\n" + "="*70)
    print("POST-PROCESSING COMPLETE")
    print("="*70 + "\n")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: abaqus python sofc_postprocess.py <odb_file>")
        print("\nExample:")
        print("  abaqus python sofc_postprocess.py Job_SOFC_HR1.odb")
        sys.exit(1)
    
    odb_path = sys.argv[1]
    
    try:
        postprocess_odb(odb_path)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
