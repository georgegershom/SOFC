#!/usr/bin/env python
"""
Post-processing script for SOFC damage and delamination analysis
Computes damage metrics and identifies critical regions
"""

import numpy as np
from odbAccess import *
from abaqusConstants import *
import visualization
import sys
import os

class SOFCDamageAnalyzer:
    """Analyzer for SOFC damage and delamination prediction"""
    
    def __init__(self, odb_path):
        """
        Initialize damage analyzer
        
        Args:
            odb_path: Path to Abaqus ODB file
        """
        self.odb = openOdb(path=odb_path, readOnly=False)
        self.assembly = self.odb.rootAssembly
        
        # Critical shear stress thresholds (MPa)
        self.tau_crit = {
            'anode_electrolyte': 25.0e6,     # 25 MPa in Pa
            'electrolyte_cathode': 20.0e6,   # 20 MPa
            'cathode_interconnect': 30.0e6   # 30 MPa
        }
        
        # Damage parameters
        self.sigma_th = 120.0e6  # Threshold stress for damage (Pa)
        self.k_D = 1.5e-5        # Damage rate constant
        self.p_damage = 2.0      # Damage exponent
        
        # Interface y-coordinates (m)
        self.interfaces = {
            'anode_electrolyte': 0.0004,
            'electrolyte_cathode': 0.0005,
            'cathode_interconnect': 0.0009
        }
        
    def compute_interfacial_shear(self, step_name='Thermo_Mechanical'):
        """
        Compute interfacial shear stress at all interfaces
        
        Returns:
            Dictionary with interface shear stress data
        """
        step = self.odb.steps[step_name]
        shear_data = {}
        
        for interface_name, y_coord in self.interfaces.items():
            shear_data[interface_name] = {
                'time': [],
                'max_shear': [],
                'avg_shear': [],
                'delamination_risk': []
            }
            
            # Get critical shear for this interface
            tau_c = self.tau_crit[interface_name]
            
            # Process each frame
            for frame in step.frames:
                time = frame.frameValue
                
                # Get stress field
                stress = frame.fieldOutputs['S']
                
                # Extract shear stress at interface
                shear_values = []
                
                for value in stress.values:
                    # Get element centroid
                    element = self.assembly.instances['SOFC_CELL-1'].elements[value.elementLabel-1]
                    nodes = [self.assembly.instances['SOFC_CELL-1'].nodes[n-1] for n in element.connectivity]
                    centroid_y = np.mean([node.coordinates[1] for node in nodes])
                    
                    # Check if element is near interface
                    if abs(centroid_y - y_coord) < 0.00002:  # Within 20 microns
                        # S12 is the shear component in 2D
                        shear_xy = abs(value.data[2])  # S12 component
                        shear_values.append(shear_xy)
                
                if shear_values:
                    max_shear = max(shear_values)
                    avg_shear = np.mean(shear_values)
                    delamination_risk = max_shear > tau_c
                    
                    shear_data[interface_name]['time'].append(time)
                    shear_data[interface_name]['max_shear'].append(max_shear)
                    shear_data[interface_name]['avg_shear'].append(avg_shear)
                    shear_data[interface_name]['delamination_risk'].append(delamination_risk)
        
        return shear_data
    
    def compute_damage_variable(self, step_name='Thermo_Mechanical'):
        """
        Compute damage variable D based on stress history
        
        Returns:
            Dictionary with damage evolution data
        """
        step = self.odb.steps[step_name]
        
        # Initialize damage field
        num_elements = len(self.assembly.instances['SOFC_CELL-1'].elements)
        damage_field = np.zeros(num_elements)
        
        # Time integration
        prev_time = 0.0
        damage_history = []
        
        for frame_idx, frame in enumerate(step.frames):
            time = frame.frameValue
            dt = time - prev_time
            
            if dt <= 0:
                continue
            
            # Get stress and temperature fields
            stress = frame.fieldOutputs['S']
            temp = frame.fieldOutputs.get('TEMP', None)
            
            # Get plastic and creep strains if available
            peeq = frame.fieldOutputs.get('PEEQ', None)
            ceeq = frame.fieldOutputs.get('CEEQ', None)
            
            frame_damage = np.zeros(num_elements)
            
            for value in stress.values:
                elem_idx = value.elementLabel - 1
                
                # Von Mises stress
                s11, s22, s33, s12 = value.data[0], value.data[1], 0.0, value.data[2]
                sigma_vm = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 3*s12**2)
                
                # Get element centroid for interface proximity
                element = self.assembly.instances['SOFC_CELL-1'].elements[elem_idx]
                nodes = [self.assembly.instances['SOFC_CELL-1'].nodes[n-1] for n in element.connectivity]
                centroid_y = np.mean([node.coordinates[1] for node in nodes])
                
                # Interface proximity weight
                w_iface = 0.0
                for y_interface in self.interfaces.values():
                    dist = abs(centroid_y - y_interface)
                    if dist < 0.0001:  # Within 100 microns
                        w_iface = max(w_iface, np.exp(-dist/0.00002))
                
                # Damage evolution rate
                if sigma_vm > self.sigma_th:
                    damage_rate = self.k_D * ((sigma_vm - self.sigma_th) / self.sigma_th) ** self.p_damage
                    damage_rate *= (1 + 3 * w_iface)  # Enhanced damage near interfaces
                    
                    # Include plastic/creep strain effects if available
                    if peeq:
                        peeq_val = peeq.values[elem_idx].data
                        damage_rate *= (1 + 10 * peeq_val)  # Plastic strain acceleration
                    
                    if ceeq:
                        ceeq_val = ceeq.values[elem_idx].data
                        damage_rate *= (1 + 5 * ceeq_val)  # Creep strain acceleration
                    
                    # Update damage (forward Euler integration)
                    damage_field[elem_idx] += damage_rate * dt * (1 - damage_field[elem_idx])
                    damage_field[elem_idx] = min(damage_field[elem_idx], 1.0)
                
                frame_damage[elem_idx] = damage_field[elem_idx]
            
            damage_history.append({
                'time': time,
                'max_damage': np.max(damage_field),
                'avg_damage': np.mean(damage_field),
                'damaged_elements': np.sum(damage_field > 0.1),
                'critical_elements': np.sum(damage_field > 0.5)
            })
            
            prev_time = time
        
        # Save damage field to ODB
        self._save_damage_to_odb(step_name, damage_field)
        
        return damage_history, damage_field
    
    def _save_damage_to_odb(self, step_name, damage_field):
        """Save damage field as a field output in the ODB"""
        step = self.odb.steps[step_name]
        
        # Create field output for last frame
        last_frame = step.frames[-1]
        
        # Create new field output
        damage_output = last_frame.FieldOutput(
            name='DAMAGE_D',
            description='Damage variable (0=undamaged, 1=failed)',
            type=SCALAR
        )
        
        # Add data
        for elem_idx, damage_val in enumerate(damage_field):
            element = self.assembly.instances['SOFC_CELL-1'].elements[elem_idx]
            damage_output.addData(
                position=CENTROID,
                instance=self.assembly.instances['SOFC_CELL-1'],
                labels=[element.label],
                data=[damage_val]
            )
    
    def analyze_electrolyte_cracking(self, step_name='Thermo_Mechanical'):
        """
        Specific analysis for electrolyte layer cracking
        
        Returns:
            Dictionary with electrolyte damage metrics
        """
        step = self.odb.steps[step_name]
        last_frame = step.frames[-1]
        
        # Get stress field
        stress = last_frame.fieldOutputs['S']
        
        electrolyte_data = {
            'max_tensile_stress': 0.0,
            'max_shear_stress': 0.0,
            'crack_depth': 0.0,
            'crack_locations': []
        }
        
        # Analyze electrolyte elements (y between 0.4-0.5 mm)
        for value in stress.values:
            element = self.assembly.instances['SOFC_CELL-1'].elements[value.elementLabel-1]
            nodes = [self.assembly.instances['SOFC_CELL-1'].nodes[n-1] for n in element.connectivity]
            centroid_y = np.mean([node.coordinates[1] for node in nodes])
            
            if 0.0004 <= centroid_y <= 0.0005:  # Electrolyte layer
                s11, s22, s12 = value.data[0], value.data[1], value.data[2]
                
                # Maximum principal stress (tensile)
                principal_1 = 0.5 * (s11 + s22) + np.sqrt(0.25 * (s11 - s22)**2 + s12**2)
                
                if principal_1 > electrolyte_data['max_tensile_stress']:
                    electrolyte_data['max_tensile_stress'] = principal_1
                
                # Shear stress
                max_shear = np.sqrt(0.25 * (s11 - s22)**2 + s12**2)
                if max_shear > electrolyte_data['max_shear_stress']:
                    electrolyte_data['max_shear_stress'] = max_shear
                
                # Check for cracking (tensile stress > 100 MPa)
                if principal_1 > 100.0e6:
                    centroid_x = np.mean([node.coordinates[0] for node in nodes])
                    electrolyte_data['crack_locations'].append({
                        'x': centroid_x,
                        'y': centroid_y,
                        'stress': principal_1
                    })
        
        # Estimate crack depth based on damage
        if electrolyte_data['crack_locations']:
            y_values = [loc['y'] for loc in electrolyte_data['crack_locations']]
            electrolyte_data['crack_depth'] = (max(y_values) - min(y_values)) * 1e6  # in microns
        
        return electrolyte_data
    
    def generate_report(self, output_file='damage_report.txt'):
        """Generate comprehensive damage analysis report"""
        
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SOFC DAMAGE AND DELAMINATION ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Analyze interfacial shear
            f.write("1. INTERFACIAL SHEAR ANALYSIS\n")
            f.write("-"*40 + "\n")
            shear_data = self.compute_interfacial_shear()
            
            for interface, data in shear_data.items():
                if data['max_shear']:
                    max_shear = max(data['max_shear']) / 1e6  # Convert to MPa
                    avg_shear = np.mean(data['avg_shear']) / 1e6
                    risk_frames = sum(data['delamination_risk'])
                    
                    f.write(f"\n{interface.replace('_', '-').title()}:\n")
                    f.write(f"  Maximum shear stress: {max_shear:.2f} MPa\n")
                    f.write(f"  Average shear stress: {avg_shear:.2f} MPa\n")
                    f.write(f"  Critical threshold: {self.tau_crit[interface]/1e6:.1f} MPa\n")
                    f.write(f"  Delamination risk: {'YES' if risk_frames > 0 else 'NO'}\n")
                    if risk_frames > 0:
                        f.write(f"  Risk duration: {risk_frames} frames\n")
            
            # Analyze damage evolution
            f.write("\n2. DAMAGE EVOLUTION ANALYSIS\n")
            f.write("-"*40 + "\n")
            damage_history, damage_field = self.compute_damage_variable()
            
            if damage_history:
                final_state = damage_history[-1]
                f.write(f"\nFinal damage state:\n")
                f.write(f"  Maximum damage: {final_state['max_damage']:.3f}\n")
                f.write(f"  Average damage: {final_state['avg_damage']:.4f}\n")
                f.write(f"  Damaged elements (D>0.1): {final_state['damaged_elements']}\n")
                f.write(f"  Critical elements (D>0.5): {final_state['critical_elements']}\n")
            
            # Analyze electrolyte cracking
            f.write("\n3. ELECTROLYTE LAYER ANALYSIS\n")
            f.write("-"*40 + "\n")
            elyte_data = self.analyze_electrolyte_cracking()
            
            f.write(f"\nElectrolyte stress state:\n")
            f.write(f"  Max tensile stress: {elyte_data['max_tensile_stress']/1e6:.2f} MPa\n")
            f.write(f"  Max shear stress: {elyte_data['max_shear_stress']/1e6:.2f} MPa\n")
            f.write(f"  Estimated crack depth: {elyte_data['crack_depth']:.1f} μm\n")
            f.write(f"  Number of crack sites: {len(elyte_data['crack_locations'])}\n")
            
            # Summary and recommendations
            f.write("\n4. SUMMARY AND RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            
            # Check critical conditions
            critical_issues = []
            
            for interface, data in shear_data.items():
                if data['delamination_risk'] and any(data['delamination_risk']):
                    critical_issues.append(f"Delamination risk at {interface.replace('_', '-')}")
            
            if final_state['critical_elements'] > 0:
                critical_issues.append(f"{final_state['critical_elements']} elements with critical damage")
            
            if elyte_data['max_tensile_stress'] > 100e6:
                critical_issues.append("Electrolyte tensile stress exceeds 100 MPa")
            
            if critical_issues:
                f.write("\nCRITICAL ISSUES IDENTIFIED:\n")
                for issue in critical_issues:
                    f.write(f"  • {issue}\n")
            else:
                f.write("\nNo critical issues identified.\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Damage report saved to {output_file}")
        
    def close(self):
        """Close ODB file"""
        self.odb.close()


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SOFC Damage Analysis')
    parser.add_argument('odb_file', help='Path to Abaqus ODB file')
    parser.add_argument('--output', default='damage_report.txt',
                       help='Output report filename')
    
    args = parser.parse_args()
    
    # Check if ODB file exists
    if not os.path.exists(args.odb_file):
        print(f"Error: ODB file '{args.odb_file}' not found")
        sys.exit(1)
    
    # Run analysis
    analyzer = SOFCDamageAnalyzer(args.odb_file)
    analyzer.generate_report(args.output)
    analyzer.close()
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()