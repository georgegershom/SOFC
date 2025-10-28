#!/usr/bin/env python3
"""
SOFC (Solid Oxide Fuel Cell) Multi-Physics Simulation - Fast Version
====================================================================

Optimized version with reduced mesh density and time steps for demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import os
import warnings
warnings.filterwarnings('ignore')

class SOFCSimulationFast:
    """Fast SOFC simulation with simplified physics"""
    
    def __init__(self):
        # Geometry parameters (meters)
        self.width = 0.01   # 10 mm
        self.height = 0.001 # 1 mm
        
        # Layer boundaries
        self.layer_bounds = [0.0, 0.0004, 0.0005, 0.0009, 0.001]  # y-coordinates
        self.layer_names = ['anode', 'electrolyte', 'cathode', 'interconnect']
        
        # Material properties (simplified)
        self.materials = {
            'anode': {
                'E': [140e9, 91e9],      # Young's modulus at 298K, 1273K
                'nu': 0.30,              # Poisson's ratio
                'alpha': [12.5e-6, 13.5e-6],  # CTE
                'k': [6.0, 4.0],         # Thermal conductivity
                'cp': [450, 570],        # Specific heat
                'rho': 6000              # Density
            },
            'electrolyte': {
                'E': [210e9, 170e9],
                'nu': 0.28,
                'alpha': [10.5e-6, 11.2e-6],
                'k': [2.6, 2.0],
                'cp': [400, 600],
                'rho': 5900
            },
            'cathode': {
                'E': [120e9, 84e9],
                'nu': 0.30,
                'alpha': [11.5e-6, 12.4e-6],
                'k': [2.0, 1.8],
                'cp': [480, 610],
                'rho': 6500
            },
            'interconnect': {
                'E': [205e9, 150e9],
                'nu': 0.30,
                'alpha': [12.5e-6, 13.2e-6],
                'k': [20, 15],
                'cp': [500, 700],
                'rho': 7800
            }
        }
        
        # Create simplified mesh
        self.create_mesh()
    
    def create_mesh(self):
        """Create simplified 1D mesh through thickness"""
        # 1D mesh through thickness (y-direction)
        ny_total = 50  # Total nodes through thickness
        
        # Distribute nodes with refinement at interfaces
        y_nodes = []
        
        # Layer-wise distribution
        layer_nodes = [15, 8, 15, 12]  # Nodes per layer
        
        for i, (y_start, y_end) in enumerate(zip(self.layer_bounds[:-1], self.layer_bounds[1:])):
            if i == 0:
                y_layer = np.linspace(y_start, y_end, layer_nodes[i])
            else:
                y_layer = np.linspace(y_start, y_end, layer_nodes[i])[1:]  # Skip first to avoid duplication
            y_nodes.extend(y_layer)
        
        self.y_coords = np.array(y_nodes)
        self.n_nodes = len(self.y_coords)
        
        # Element connectivity (1D elements)
        self.elements = []
        self.element_materials = []
        
        for i in range(self.n_nodes - 1):
            self.elements.append([i, i+1])
            
            # Determine material
            y_center = (self.y_coords[i] + self.y_coords[i+1]) / 2
            for j, (y_start, y_end) in enumerate(zip(self.layer_bounds[:-1], self.layer_bounds[1:])):
                if y_start <= y_center < y_end:
                    self.element_materials.append(self.layer_names[j])
                    break
        
        self.n_elements = len(self.elements)
        print(f"1D Mesh: {self.n_nodes} nodes, {self.n_elements} elements")
    
    def get_material_property(self, material, prop, temperature):
        """Get temperature-dependent material property"""
        T_ref = [298, 1273]  # Reference temperatures
        values = self.materials[material][prop]
        
        if isinstance(values, list) and len(values) == 2:
            return np.interp(temperature, T_ref, values)
        else:
            return values
    
    def solve_thermal(self, heating_rate='HR1'):
        """Solve 1D transient heat conduction"""
        # Heating schedules (simplified)
        schedules = {
            'HR1': (875*60, 10*60, 875*60),    # ramp, hold, cool (seconds)
            'HR4': (218.75*60, 10*60, 218.75*60),
            'HR10': (87.5*60, 10*60, 87.5*60)
        }
        
        ramp_time, hold_time, cool_time = schedules[heating_rate]
        total_time = ramp_time + hold_time + cool_time
        
        # Time discretization
        dt = 60.0  # 1 minute time steps
        times = np.arange(0, total_time + dt, dt)
        n_steps = len(times)
        
        # Initialize temperature
        T = np.full(self.n_nodes, 298.15)  # Room temperature
        T_history = np.zeros((n_steps, self.n_nodes))
        T_history[0] = T.copy()
        
        print(f"Solving thermal problem: {heating_rate}, {n_steps} time steps")
        
        for step in range(1, n_steps):
            time = times[step]
            
            # Prescribed temperature at bottom
            T_bottom = self.get_prescribed_temperature(time, ramp_time, hold_time, cool_time)
            
            # Assemble 1D heat conduction matrix
            K = np.zeros((self.n_nodes, self.n_nodes))
            C = np.zeros((self.n_nodes, self.n_nodes))
            
            for elem_id, (n1, n2) in enumerate(self.elements):
                material = self.element_materials[elem_id]
                
                # Element temperature
                T_elem = 0.5 * (T[n1] + T[n2])
                
                # Material properties
                k = self.get_material_property(material, 'k', T_elem)
                rho = self.get_material_property(material, 'rho', T_elem)
                cp = self.get_material_property(material, 'cp', T_elem)
                
                # Element length
                L = self.y_coords[n2] - self.y_coords[n1]
                
                # 1D element matrices
                Ke = k / L * np.array([[1, -1], [-1, 1]])
                Ce = rho * cp * L / 6 * np.array([[2, 1], [1, 2]])
                
                # Assemble
                nodes = [n1, n2]
                for i in range(2):
                    for j in range(2):
                        K[nodes[i], nodes[j]] += Ke[i, j]
                        C[nodes[i], nodes[j]] += Ce[i, j]
            
            # Apply boundary conditions
            # Bottom: prescribed temperature
            penalty = 1e12
            K[0, 0] += penalty
            
            # Top: convection
            h_conv = 25.0  # W/m²K
            T_inf = 298.15
            K[-1, -1] += h_conv
            
            # Right-hand side
            F = np.zeros(self.n_nodes)
            F[0] = penalty * T_bottom
            F[-1] = h_conv * T_inf
            
            # Time integration (backward Euler)
            A = C + dt * K
            b = C @ T + dt * F
            
            # Solve
            T = spsolve(csr_matrix(A), b)
            T_history[step] = T.copy()
        
        return times, T_history
    
    def get_prescribed_temperature(self, time, ramp_time, hold_time, cool_time):
        """Get prescribed temperature based on schedule"""
        T_room = 298.15
        T_target = 1173.15  # 900°C
        
        if time <= ramp_time:
            return T_room + (T_target - T_room) * time / ramp_time
        elif time <= ramp_time + hold_time:
            return T_target
        else:
            cool_progress = (time - ramp_time - hold_time) / cool_time
            return T_target - (T_target - T_room) * min(cool_progress, 1.0)
    
    def solve_mechanical(self, T_history, times):
        """Solve 1D thermo-mechanical problem"""
        n_steps = len(times)
        
        # Storage
        stress_history = np.zeros((n_steps, self.n_elements))
        strain_history = np.zeros((n_steps, self.n_elements))
        damage_history = np.zeros((n_steps, self.n_elements))
        
        # Reference temperature
        T_ref = 298.15
        
        print("Solving mechanical problem...")
        
        for step in range(n_steps):
            T = T_history[step]
            
            # Simple 1D thermo-mechanical analysis
            for elem_id, (n1, n2) in enumerate(self.elements):
                material = self.element_materials[elem_id]
                
                # Element temperature
                T_elem = 0.5 * (T[n1] + T[n2])
                dT = T_elem - T_ref
                
                # Material properties
                E = self.get_material_property(material, 'E', T_elem)
                alpha = self.get_material_property(material, 'alpha', T_elem)
                
                # Thermal strain (free expansion)
                eps_thermal = alpha * dT
                
                # For constrained case, compute thermal stress
                # Assuming constrained expansion
                stress = -E * eps_thermal  # Compressive stress due to constraint
                
                # Store results
                stress_history[step, elem_id] = stress
                strain_history[step, elem_id] = eps_thermal
                
                # Simple damage model
                sigma_vm = abs(stress)
                sigma_th = 120e6  # Threshold stress
                
                if sigma_vm > sigma_th:
                    damage_rate = 1.5e-5 * ((sigma_vm - sigma_th) / sigma_th)**2
                    if step > 0:
                        dt = times[step] - times[step-1]
                        damage_history[step, elem_id] = min(damage_history[step-1, elem_id] + damage_rate * dt, 1.0)
                    else:
                        damage_history[step, elem_id] = 0
                else:
                    if step > 0:
                        damage_history[step, elem_id] = damage_history[step-1, elem_id]
        
        return stress_history, strain_history, damage_history
    
    def check_delamination(self, stress_history):
        """Check for delamination at interfaces"""
        # Interface elements (simplified)
        interface_elements = {
            'anode_electrolyte': [],
            'electrolyte_cathode': [],
            'cathode_interconnect': []
        }
        
        # Find interface elements
        for elem_id, (n1, n2) in enumerate(self.elements):
            y_center = 0.5 * (self.y_coords[n1] + self.y_coords[n2])
            
            if abs(y_center - 0.0004) < 0.00005:  # Anode-electrolyte
                interface_elements['anode_electrolyte'].append(elem_id)
            elif abs(y_center - 0.0005) < 0.00005:  # Electrolyte-cathode
                interface_elements['electrolyte_cathode'].append(elem_id)
            elif abs(y_center - 0.0009) < 0.00005:  # Cathode-interconnect
                interface_elements['cathode_interconnect'].append(elem_id)
        
        # Critical shear stresses
        tau_crit = {
            'anode_electrolyte': 25e6,
            'electrolyte_cathode': 20e6,
            'cathode_interconnect': 30e6
        }
        
        # Check delamination
        delamination_history = []
        for step in range(len(stress_history)):
            delamination = {}
            
            for interface, elements in interface_elements.items():
                max_stress = 0
                for elem_id in elements:
                    max_stress = max(max_stress, abs(stress_history[step, elem_id]))
                
                delamination[interface] = max_stress > tau_crit[interface]
            
            delamination_history.append(delamination)
        
        return delamination_history
    
    def run_simulation(self, heating_rate='HR1'):
        """Run complete simulation"""
        print(f"\n=== SOFC Fast Simulation: {heating_rate} ===")
        
        # Thermal analysis
        times, T_history = self.solve_thermal(heating_rate)
        
        # Mechanical analysis
        stress_history, strain_history, damage_history = self.solve_mechanical(T_history, times)
        
        # Delamination check
        delamination_history = self.check_delamination(stress_history)
        
        # Save results
        results = {
            'times': times,
            'temperature': T_history,
            'stress': stress_history,
            'strain': strain_history,
            'damage': damage_history,
            'delamination': delamination_history,
            'coordinates': self.y_coords
        }
        
        self.save_results(heating_rate, results)
        self.plot_results(heating_rate, results)
        
        return results
    
    def save_results(self, heating_rate, results):
        """Save results to files"""
        output_dir = f"/workspace/sofc_results_{heating_rate.lower()}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as NPZ
        np.savez_compressed(
            f"{output_dir}/sofc_simulation_{heating_rate.lower()}.npz",
            **results
        )
        
        print(f"Results saved to {output_dir}/")
    
    def plot_results(self, heating_rate, results):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'SOFC Simulation Results - {heating_rate}', fontsize=16)
        
        times_hr = results['times'] / 3600  # Convert to hours
        y_mm = results['coordinates'] * 1000  # Convert to mm
        
        # Temperature evolution
        ax = axes[0, 0]
        T_celsius = results['temperature'] - 273.15
        ax.plot(times_hr, T_celsius[:, 0], 'r-', label='Bottom', linewidth=2)
        ax.plot(times_hr, T_celsius[:, -1], 'b-', label='Top', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Evolution')
        ax.legend()
        ax.grid(True)
        
        # Final temperature profile
        ax = axes[0, 1]
        ax.plot(T_celsius[-1], y_mm, 'ro-', markersize=4)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Height (mm)')
        ax.set_title('Final Temperature Profile')
        ax.grid(True)
        
        # Add layer boundaries
        for y_bound in [0.4, 0.5, 0.9]:
            ax.axhline(y=y_bound, color='gray', linestyle='--', alpha=0.5)
        
        # Stress evolution
        ax = axes[0, 2]
        max_stress = np.max(np.abs(results['stress']), axis=1) / 1e6  # MPa
        ax.plot(times_hr, max_stress, 'g-', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Max Stress (MPa)')
        ax.set_title('Maximum Stress Evolution')
        ax.grid(True)
        
        # Damage evolution
        ax = axes[1, 0]
        max_damage = np.max(results['damage'], axis=1)
        ax.plot(times_hr, max_damage, 'm-', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Maximum Damage')
        ax.set_title('Damage Evolution')
        ax.grid(True)
        
        # Final stress profile
        ax = axes[1, 1]
        final_stress = results['stress'][-1] / 1e6  # MPa
        y_elem = []
        for n1, n2 in self.elements:
            y_elem.append(0.5 * (y_mm[n1] + y_mm[n2]))
        
        ax.plot(final_stress, y_elem, 'co-', markersize=4)
        ax.set_xlabel('Stress (MPa)')
        ax.set_ylabel('Height (mm)')
        ax.set_title('Final Stress Profile')
        ax.grid(True)
        
        # Add layer boundaries
        for y_bound in [0.4, 0.5, 0.9]:
            ax.axhline(y=y_bound, color='gray', linestyle='--', alpha=0.5)
        
        # Delamination timeline
        ax = axes[1, 2]
        interfaces = ['anode_electrolyte', 'electrolyte_cathode', 'cathode_interconnect']
        colors = ['red', 'blue', 'green']
        
        for i, interface in enumerate(interfaces):
            delamination_times = []
            for step, delam_dict in enumerate(results['delamination']):
                if delam_dict.get(interface, False):
                    delamination_times.append(times_hr[step])
            
            if delamination_times:
                ax.scatter(delamination_times, [i]*len(delamination_times), 
                          c=colors[i], s=20, label=interface.replace('_', '-'))
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Interface')
        ax.set_yticks(range(len(interfaces)))
        ax.set_yticklabels([iface.replace('_', '-') for iface in interfaces])
        ax.set_title('Delamination Events')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = f"/workspace/sofc_results_{heating_rate.lower()}"
        plt.savefig(f"{output_dir}/sofc_results_{heating_rate.lower()}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plotted for {heating_rate}")

def main():
    """Main simulation runner"""
    print("SOFC Multi-Physics Simulation (Fast Version)")
    print("=" * 60)
    
    # Create simulation
    sofc = SOFCSimulationFast()
    
    # Run simulations for all heating rates
    heating_rates = ['HR1', 'HR4', 'HR10']
    
    for hr in heating_rates:
        try:
            results = sofc.run_simulation(hr)
            
            # Print summary
            print(f"\n{hr} Summary:")
            print(f"  Max temperature: {np.max(results['temperature']) - 273.15:.1f}°C")
            print(f"  Max stress: {np.max(np.abs(results['stress']))/1e6:.1f} MPa")
            print(f"  Max damage: {np.max(results['damage']):.3f}")
            
            # Check final delamination
            final_delamination = results['delamination'][-1]
            for interface, flag in final_delamination.items():
                status = "YES" if flag else "NO"
                print(f"  Delamination at {interface}: {status}")
            
        except Exception as e:
            print(f"Error in {hr} simulation: {e}")
            continue
    
    print("\nSimulation completed!")
    print("Results saved in /workspace/sofc_results_*/")

if __name__ == "__main__":
    main()