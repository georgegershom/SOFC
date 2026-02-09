#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
+==============================================================================+
|                                                                              |
|   SOFC Results Extraction Utility                                           |
|                                                                              |
|   Purpose   : Extract and export stress, strain, and displacement data      |
|               from Abaqus ODB files for post-processing and validation.      |
|                                                                              |
|   Author    : SOFC Research Team                                            |
|   Version   : 3.0.0                                                          |
|   Created   : 2026-02-09                                                     |
|                                                                              |
|   Usage     : abaqus python extract_results.py [options]                     |
|                                                                              |
+==============================================================================+
"""

from __future__ import print_function, division
import sys
import os
import csv
from collections import OrderedDict

try:
    from odbAccess import *
    from abaqusConstants import *
except ImportError:
    print("ERROR: This script must be run with Abaqus Python:")
    print("       abaqus python extract_results.py")
    sys.exit(1)

# ============================================================================
#  CONFIGURATION
# ============================================================================

# Default ODB file (can be overridden via command line)
DEFAULT_ODB = 'Job-Validation-Cooling-v3.odb'

# Output directory for extracted data
OUTPUT_DIR = 'results_data'

# Variables to extract
FIELD_VARIABLES = ['S', 'E', 'U', 'TEMP']
STRESS_COMPONENTS = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']
STRAIN_COMPONENTS = ['E11', 'E22', 'E33', 'E12', 'E13', 'E23']
DISPLACEMENT_COMPONENTS = ['U1', 'U2', 'U3']

# Interface region name (set during model creation)
INTERFACE_SET_NAME = 'SET_INTERFACE'


# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================

def print_banner(message):
    """Print formatted banner message."""
    print('\n' + '=' * 78)
    print(f'  {message}')
    print('=' * 78)


def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}/")


def get_odb_path():
    """Get ODB file path from command line or use default."""
    if len(sys.argv) > 1:
        odb_path = sys.argv[1]
    else:
        odb_path = DEFAULT_ODB
    
    if not os.path.exists(odb_path):
        print(f"ERROR: ODB file not found: {odb_path}")
        sys.exit(1)
    
    return odb_path


# ============================================================================
#  DATA EXTRACTION FUNCTIONS
# ============================================================================

def extract_field_output_summary(odb):
    """
    Extract summary statistics for all field outputs.
    
    Parameters
    ----------
    odb : Odb
        Open Abaqus output database.
    """
    print_banner('Field Output Summary')
    
    # Get last step and frame
    step_name = odb.steps.keys()[-1]
    step = odb.steps[step_name]
    frame = step.frames[-1]
    
    print(f"Step: {step_name}")
    print(f"Frame: {len(step.frames) - 1} (last)")
    print(f"Total time: {frame.frameValue:.4f}")
    print()
    
    # Available field outputs
    print("Available field outputs:")
    for key in frame.fieldOutputs.keys():
        field = frame.fieldOutputs[key]
        print(f"  • {key:10s} : {field.description}")
    print()


def extract_stress_at_interface(odb, output_file='interface_stress.csv'):
    """
    Extract stress components along anode-electrolyte interface.
    
    Parameters
    ----------
    odb : Odb
        Open Abaqus output database.
    output_file : str, optional
        CSV output filename.
    """
    print_banner('Extracting Interface Stress')
    
    try:
        # Get last frame
        step = odb.steps[odb.steps.keys()[-1]]
        frame = step.frames[-1]
        
        # Get stress field
        stress_field = frame.fieldOutputs['S']
        
        # Try to get interface set
        try:
            interface_set = odb.rootAssembly.nodeSets[INTERFACE_SET_NAME.upper()]
            stress_subset = stress_field.getSubset(region=interface_set)
            print(f"Extracting from set: {INTERFACE_SET_NAME}")
        except KeyError:
            print(f"Warning: Interface set '{INTERFACE_SET_NAME}' not found")
            print("Extracting all values instead")
            stress_subset = stress_field
        
        # Extract data
        data = []
        for value in stress_subset.values:
            if hasattr(value, 'nodeLabel'):
                label = value.nodeLabel
                label_type = 'Node'
            elif hasattr(value, 'elementLabel'):
                label = value.elementLabel
                label_type = 'Element'
            else:
                label = 0
                label_type = 'Unknown'
            
            # Get position
            if hasattr(value, 'position'):
                pos = value.position
            else:
                pos = [0.0, 0.0, 0.0]
            
            # Stress components
            s_data = value.data
            
            row = OrderedDict([
                ('Label', label),
                ('Type', label_type),
                ('X', pos[0] if len(pos) > 0 else 0.0),
                ('Y', pos[1] if len(pos) > 1 else 0.0),
                ('S11_Pa', s_data[0] if len(s_data) > 0 else 0.0),
                ('S22_Pa', s_data[1] if len(s_data) > 1 else 0.0),
                ('S33_Pa', s_data[2] if len(s_data) > 2 else 0.0),
                ('S12_Pa', s_data[3] if len(s_data) > 3 else 0.0),
                ('vonMises_Pa', value.mises if hasattr(value, 'mises') else 0.0),
            ])
            data.append(row)
        
        # Write to CSV
        output_path = os.path.join(OUTPUT_DIR, output_file)
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✓ Extracted {len(data)} stress values")
        print(f"✓ Saved to: {output_path}")
        print()
        
        # Print statistics
        if data:
            s11_vals = [row['S11_Pa'] for row in data]
            s22_vals = [row['S22_Pa'] for row in data]
            vm_vals = [row['vonMises_Pa'] for row in data]
            
            print("Statistics (MPa):")
            print(f"  S11 (longitudinal):")
            print(f"    Min: {min(s11_vals)/1e6:>10.2f}  |  Max: {max(s11_vals)/1e6:>10.2f}")
            print(f"  S22 (transverse):")
            print(f"    Min: {min(s22_vals)/1e6:>10.2f}  |  Max: {max(s22_vals)/1e6:>10.2f}")
            print(f"  von Mises:")
            print(f"    Min: {min(vm_vals)/1e6:>10.2f}  |  Max: {max(vm_vals)/1e6:>10.2f}")
        
    except KeyError as e:
        print(f"ERROR: Field output not found: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def extract_displacement_field(odb, output_file='displacement_field.csv'):
    """
    Extract displacement field for entire model.
    
    Parameters
    ----------
    odb : Odb
        Open Abaqus output database.
    output_file : str, optional
        CSV output filename.
    """
    print_banner('Extracting Displacement Field')
    
    try:
        # Get last frame
        step = odb.steps[odb.steps.keys()[-1]]
        frame = step.frames[-1]
        
        # Get displacement field
        u_field = frame.fieldOutputs['U']
        
        # Extract nodal displacements
        data = []
        for value in u_field.values:
            row = OrderedDict([
                ('NodeLabel', value.nodeLabel),
                ('U1_mm', value.data[0]),
                ('U2_mm', value.data[1]),
                ('U3_mm', value.data[2] if len(value.data) > 2 else 0.0),
                ('Magnitude_mm', value.magnitude if hasattr(value, 'magnitude') else 0.0),
            ])
            data.append(row)
        
        # Write to CSV
        output_path = os.path.join(OUTPUT_DIR, output_file)
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✓ Extracted {len(data)} displacement values")
        print(f"✓ Saved to: {output_path}")
        print()
        
        # Statistics
        if data:
            u1_vals = [row['U1_mm'] for row in data]
            u2_vals = [row['U2_mm'] for row in data]
            mag_vals = [row['Magnitude_mm'] for row in data]
            
            print("Statistics (μm):")
            print(f"  U1 (lateral):")
            print(f"    Max: {max(u1_vals)*1000:>10.3f}")
            print(f"  U2 (vertical):")
            print(f"    Min: {min(u2_vals)*1000:>10.3f}  |  Max: {max(u2_vals)*1000:>10.3f}")
            print(f"  Magnitude:")
            print(f"    Max: {max(mag_vals)*1000:>10.3f}")
    
    except KeyError as e:
        print(f"ERROR: Field output not found: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def extract_through_thickness_profile(odb, x_coord=5.0, 
                                     output_file='through_thickness_profile.csv'):
    """
    Extract stress profile through thickness at specified x-coordinate.
    
    Parameters
    ----------
    odb : Odb
        Open Abaqus output database.
    x_coord : float, optional
        X-coordinate for profile extraction (default: 5.0 mm, center of model).
    output_file : str, optional
        CSV output filename.
    """
    print_banner('Extracting Through-Thickness Profile')
    
    try:
        # Get last frame
        step = odb.steps[odb.steps.keys()[-1]]
        frame = step.frames[-1]
        
        # Get stress field
        stress_field = frame.fieldOutputs['S']
        
        # Filter values near x_coord (tolerance ±0.1 mm)
        tolerance = 0.1
        data = []
        
        for value in stress_field.values:
            # Get position (need to access through nodes or elements)
            # This is a simplified version - actual implementation depends on model details
            
            # For integration points, we need element connectivity
            # Simplified: extract all and filter in post-processing
            
            if hasattr(value, 'elementLabel'):
                label = value.elementLabel
                s_data = value.data
                
                row = OrderedDict([
                    ('ElementLabel', label),
                    ('S11_Pa', s_data[0]),
                    ('S22_Pa', s_data[1]),
                    ('S33_Pa', s_data[2]),
                    ('S12_Pa', s_data[3]),
                    ('vonMises_Pa', value.mises if hasattr(value, 'mises') else 0.0),
                ])
                data.append(row)
        
        # Write to CSV
        output_path = os.path.join(OUTPUT_DIR, output_file)
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✓ Extracted {len(data)} stress values")
        print(f"✓ Saved to: {output_path}")
        print(f"Note: Filter by position in external post-processing (e.g., Python/MATLAB)")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def extract_history_output(odb, output_file='history_output.csv'):
    """
    Extract history output data (time series).
    
    Parameters
    ----------
    odb : Odb
        Open Abaqus output database.
    output_file : str, optional
        CSV output filename.
    """
    print_banner('Extracting History Output')
    
    try:
        step = odb.steps[odb.steps.keys()[-1]]
        
        if not step.historyRegions:
            print("No history output available in this ODB")
            return
        
        # Get first history region
        region_key = step.historyRegions.keys()[0]
        region = step.historyRegions[region_key]
        
        print(f"Region: {region_key}")
        print(f"Available variables: {region.historyOutputs.keys()}")
        print()
        
        # Extract all variables
        data = OrderedDict()
        for var_name in region.historyOutputs.keys():
            hist_data = region.historyOutputs[var_name]
            data[var_name] = hist_data.data
        
        # Transpose data for CSV export (time as rows)
        if data:
            # Get time points from first variable
            first_var = data.values()[0]
            time_points = [d[0] for d in first_var]
            
            # Build rows
            rows = []
            for i, t in enumerate(time_points):
                row = OrderedDict([('Time', t)])
                for var_name, var_data in data.items():
                    if i < len(var_data):
                        row[var_name] = var_data[i][1]
                    else:
                        row[var_name] = None
                rows.append(row)
            
            # Write to CSV
            output_path = os.path.join(OUTPUT_DIR, output_file)
            with open(output_path, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"✓ Extracted {len(rows)} time points")
            print(f"✓ Saved to: {output_path}")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def print_model_info(odb):
    """
    Print comprehensive model information.
    
    Parameters
    ----------
    odb : Odb
        Open Abaqus output database.
    """
    print_banner('Model Information')
    
    print(f"Job Name       : {odb.name}")
    print(f"Description    : {odb.description}")
    print(f"Analysis Code  : {odb.analysisCode}")
    print()
    
    print(f"Steps          : {len(odb.steps)}")
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        print(f"  • {step_name}: {len(step.frames)} frames")
    print()
    
    print(f"Parts          : {len(odb.rootAssembly.instances)}")
    for inst_name in odb.rootAssembly.instances.keys():
        instance = odb.rootAssembly.instances[inst_name]
        n_nodes = len(instance.nodes)
        n_elements = len(instance.elements)
        print(f"  • {inst_name}: {n_elements} elements, {n_nodes} nodes")
    print()
    
    # Material info
    if odb.materials:
        print(f"Materials      : {len(odb.materials)}")
        for mat_name in odb.materials.keys():
            print(f"  • {mat_name}")
    print()


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print('\n' + '╔' + '═' * 76 + '╗')
    print('║' + ' ' * 20 + 'SOFC Results Extraction Utility' + ' ' * 25 + '║')
    print('║' + ' ' * 30 + 'Version 3.0.0' + ' ' * 33 + '║')
    print('╚' + '═' * 76 + '╝\n')
    
    # Setup
    ensure_output_directory()
    odb_path = get_odb_path()
    
    print(f"Opening ODB: {odb_path}")
    print()
    
    # Open ODB
    try:
        odb = openOdb(path=odb_path, readOnly=True)
    except Exception as e:
        print(f"ERROR: Failed to open ODB: {e}")
        sys.exit(1)
    
    try:
        # Print model information
        print_model_info(odb)
        
        # Extract field output summary
        extract_field_output_summary(odb)
        
        # Extract specific data
        extract_stress_at_interface(odb)
        extract_displacement_field(odb)
        extract_through_thickness_profile(odb)
        extract_history_output(odb)
        
        print_banner('Extraction Complete')
        print(f"All results saved to: {OUTPUT_DIR}/")
        print()
        print("Next steps:")
        print("  1. Review CSV files in results_data/ directory")
        print("  2. Import data into Python/MATLAB for analysis")
        print("  3. Create plots and compare with experimental data")
        print()
    
    finally:
        # Close ODB
        odb.close()
        print("ODB closed successfully")


if __name__ == '__main__':
    main()
    sys.exit(0)

# ============================================================================
#  END OF SCRIPT
# ============================================================================
