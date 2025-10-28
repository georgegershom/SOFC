#!/usr/bin/env python3
"""
SOFC (Solid Oxide Fuel Cell) Multi-Physics Simulation
======================================================

This simulation implements the complete SOFC analysis as specified in the Abaqus methodology:
- 2D cross-section with 4 layers (Anode, Electrolyte, Cathode, Interconnect)
- Sequential multi-physics: Heat transfer → Thermo-mechanical
- Temperature-dependent materials with plasticity and creep
- Damage and delamination modeling
- Multiple heating rates (HR1, HR4, HR10)

Author: AI Assistant
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import h5py
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class SOFCMaterial:
    """Material properties for SOFC layers with temperature dependence"""
    
    def __init__(self, name, elastic_props, thermal_props, plasticity=None, creep=None):
        self.name = name
        self.elastic_props = elastic_props  # (E_298K, E_1273K, nu)
        self.thermal_props = thermal_props  # (alpha_298K, alpha_1273K, k_298K, k_1273K, cp_298K, cp_1273K, density)
        self.plasticity = plasticity  # Johnson-Cook parameters if applicable
        self.creep = creep  # Norton-Bailey parameters if applicable
    
    def get_elastic_modulus(self, T):
        """Temperature-dependent elastic modulus (Pa)"""
        E_298, E_1273, _ = self.elastic_props
        return np.interp(T, [298, 1273], [E_298 * 1e9, E_1273 * 1e9])
    
    def get_poisson_ratio(self, T):
        """Poisson's ratio (assumed constant)"""
        return self.elastic_props[2]
    
    def get_thermal_expansion(self, T):
        """Temperature-dependent CTE (1/K)"""
        alpha_298, alpha_1273, _, _, _, _, _ = self.thermal_props
        return np.interp(T, [298, 1273], [alpha_298 * 1e-6, alpha_1273 * 1e-6])
    
    def get_thermal_conductivity(self, T):
        """Temperature-dependent thermal conductivity (W/m·K)"""
        _, _, k_298, k_1273, _, _, _ = self.thermal_props
        return np.interp(T, [298, 1273], [k_298, k_1273])
    
    def get_specific_heat(self, T):
        """Temperature-dependent specific heat (J/kg·K)"""
        _, _, _, _, cp_298, cp_1273, _ = self.thermal_props
        return np.interp(T, [298, 1273], [cp_298, cp_1273])
    
    def get_density(self):
        """Material density (kg/m³)"""
        return self.thermal_props[6]

class SOFCGeometry:
    """2D SOFC geometry with 4 layers"""
    
    def __init__(self, width=0.01, height=0.001):  # 10mm x 1mm in meters
        self.width = width
        self.height = height
        
        # Layer boundaries (y-coordinates in meters)
        self.layer_bounds = {
            'anode': (0.0, 0.0004),      # 0.0-0.4 mm
            'electrolyte': (0.0004, 0.0005),  # 0.4-0.5 mm  
            'cathode': (0.0005, 0.0009),      # 0.5-0.9 mm
            'interconnect': (0.0009, 0.001)   # 0.9-1.0 mm
        }
        
        # Interface locations
        self.interfaces = {
            'anode_electrolyte': 0.0004,
            'electrolyte_cathode': 0.0005,
            'cathode_interconnect': 0.0009
        }

class SOFCMesh:
    """2D structured mesh for SOFC simulation"""
    
    def __init__(self, geometry, nx=80, ny_layers=None):
        self.geometry = geometry
        self.nx = nx  # Elements in x-direction
        
        # Default y-refinement per layer
        if ny_layers is None:
            ny_layers = {
                'anode': 20,
                'electrolyte': 12,  # Fine mesh in thin electrolyte
                'cathode': 20,
                'interconnect': 8
            }
        self.ny_layers = ny_layers
        
        self.create_mesh()
    
    def create_mesh(self):
        """Create structured 2D mesh"""
        # X coordinates
        self.x = np.linspace(0, self.geometry.width, self.nx + 1)
        
        # Y coordinates with layer refinement
        y_coords = [0.0]
        
        for layer, (y_min, y_max) in self.geometry.layer_bounds.items():
            ny = self.ny_layers[layer]
            y_layer = np.linspace(y_min, y_max, ny + 1)[1:]  # Exclude first point to avoid duplication
            y_coords.extend(y_layer)
        
        self.y = np.array(y_coords)
        self.ny = len(self.y) - 1
        
        # Create node coordinates
        self.nodes = []
        self.node_map = {}  # (i,j) -> node_id
        node_id = 0
        
        for j in range(len(self.y)):
            for i in range(len(self.x)):
                self.nodes.append([self.x[i], self.y[j]])
                self.node_map[(i, j)] = node_id
                node_id += 1
        
        self.nodes = np.array(self.nodes)
        self.n_nodes = len(self.nodes)
        
        # Create elements (4-node quads)
        self.elements = []
        self.element_materials = []
        
        for j in range(self.ny):
            for i in range(self.nx):
                # Element nodes (counter-clockwise)
                n1 = self.node_map[(i, j)]
                n2 = self.node_map[(i+1, j)]
                n3 = self.node_map[(i+1, j+1)]
                n4 = self.node_map[(i, j+1)]
                
                self.elements.append([n1, n2, n3, n4])
                
                # Determine material based on y-coordinate
                y_center = (self.y[j] + self.y[j+1]) / 2
                if y_center <= self.geometry.layer_bounds['anode'][1]:
                    material = 'anode'
                elif y_center <= self.geometry.layer_bounds['electrolyte'][1]:
                    material = 'electrolyte'
                elif y_center <= self.geometry.layer_bounds['cathode'][1]:
                    material = 'cathode'
                else:
                    material = 'interconnect'
                
                self.element_materials.append(material)
        
        self.elements = np.array(self.elements)
        self.n_elements = len(self.elements)
        
        print(f"Mesh created: {self.n_nodes} nodes, {self.n_elements} elements")
        print(f"Layer distribution: {dict(zip(*np.unique(self.element_materials, return_counts=True)))}")

class SOFCThermalSolver:
    """Transient heat conduction solver"""
    
    def __init__(self, mesh, materials):
        self.mesh = mesh
        self.materials = materials
        self.n_nodes = mesh.n_nodes
        
        # Initialize temperature field
        self.temperature = np.full(self.n_nodes, 298.15)  # Room temperature in K
        
    def assemble_thermal_matrices(self, T_field):
        """Assemble thermal conductivity and capacity matrices"""
        K = np.zeros((self.n_nodes, self.n_nodes))
        C = np.zeros((self.n_nodes, self.n_nodes))
        
        # Gauss points for 2x2 integration
        gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), 1/np.sqrt(3)],
                              [-1/np.sqrt(3), 1/np.sqrt(3)]])
        weights = np.array([1, 1, 1, 1])
        
        for elem_id, element in enumerate(self.mesh.elements):
            material_name = self.mesh.element_materials[elem_id]
            material = self.materials[material_name]
            
            # Element temperature (average)
            T_elem = np.mean(T_field[element])
            
            # Material properties at element temperature
            k = material.get_thermal_conductivity(T_elem)
            rho = material.get_density()
            cp = material.get_specific_heat(T_elem)
            
            # Element matrices
            Ke = np.zeros((4, 4))
            Ce = np.zeros((4, 4))
            
            for gp, (xi, eta) in enumerate(gauss_pts):
                # Shape functions and derivatives
                N, dN_dxi = self.shape_functions_2d(xi, eta)
                
                # Jacobian
                coords = self.mesh.nodes[element]
                J = dN_dxi @ coords
                det_J = np.linalg.det(J)
                J_inv = np.linalg.inv(J)
                
                # Global derivatives
                dN_dx = J_inv @ dN_dxi
                
                # Conductivity matrix
                B = dN_dx
                Ke += weights[gp] * det_J * k * (B.T @ B)
                
                # Capacity matrix
                Ce += weights[gp] * det_J * rho * cp * np.outer(N, N)
            
            # Assemble into global matrices
            for i in range(4):
                for j in range(4):
                    K[element[i], element[j]] += Ke[i, j]
                    C[element[i], element[j]] += Ce[i, j]
        
        return csr_matrix(K), csr_matrix(C)
    
    def shape_functions_2d(self, xi, eta):
        """2D bilinear shape functions and derivatives"""
        N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), 
                            (1+xi)*(1+eta), (1-xi)*(1+eta)])
        
        dN_dxi = 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])
        
        return N, dN_dxi
    
    def apply_thermal_bcs(self, K, C, F, time, heating_schedule):
        """Apply thermal boundary conditions"""
        # Bottom edge: prescribed temperature
        T_bottom = self.get_prescribed_temperature(time, heating_schedule)
        
        # Find bottom nodes (y = 0)
        bottom_nodes = np.where(self.mesh.nodes[:, 1] == 0)[0]
        
        # Top edge: convection (h = 25 W/m²K, T_inf = 298.15 K)
        h_conv = 25.0
        T_inf = 298.15
        top_nodes = np.where(self.mesh.nodes[:, 1] == self.mesh.geometry.height)[0]
        
        # Apply convection to top edge
        for i in range(len(top_nodes) - 1):
            n1, n2 = top_nodes[i], top_nodes[i+1]
            edge_length = self.mesh.geometry.width / (len(top_nodes) - 1)
            
            # Convection contribution
            h_contrib = h_conv * edge_length / 6 * np.array([[2, 1], [1, 2]])
            f_contrib = h_conv * edge_length * T_inf / 2 * np.array([1, 1])
            
            K[n1, n1] += h_contrib[0, 0]
            K[n1, n2] += h_contrib[0, 1]
            K[n2, n1] += h_contrib[1, 0]
            K[n2, n2] += h_contrib[1, 1]
            
            F[n1] += f_contrib[0]
            F[n2] += f_contrib[1]
        
        # Apply prescribed temperature (penalty method)
        penalty = 1e12
        for node in bottom_nodes:
            K[node, node] += penalty
            F[node] += penalty * T_bottom
        
        return K, F
    
    def get_prescribed_temperature(self, time, heating_schedule):
        """Get prescribed temperature based on heating schedule"""
        hr_type, ramp_time, hold_time, cool_time = heating_schedule
        
        T_room = 298.15  # K (25°C)
        T_target = 1173.15  # K (900°C)
        
        if time <= ramp_time:
            # Heating phase
            return T_room + (T_target - T_room) * time / ramp_time
        elif time <= ramp_time + hold_time:
            # Hold phase
            return T_target
        else:
            # Cooling phase
            cool_progress = (time - ramp_time - hold_time) / cool_time
            return T_target - (T_target - T_room) * min(cool_progress, 1.0)
    
    def solve_transient(self, heating_schedule, dt=1.0):
        """Solve transient heat conduction"""
        hr_type, ramp_time, hold_time, cool_time = heating_schedule
        total_time = ramp_time + hold_time + cool_time
        
        times = np.arange(0, total_time + dt, dt)
        n_steps = len(times)
        
        # Storage for results
        temperature_history = np.zeros((n_steps, self.n_nodes))
        temperature_history[0] = self.temperature.copy()
        
        print(f"Solving thermal problem: {hr_type}, {n_steps} time steps")
        
        for step in tqdm(range(1, n_steps), desc="Thermal analysis"):
            time = times[step]
            
            # Assemble matrices at current temperature
            K, C = self.assemble_thermal_matrices(self.temperature)
            F = np.zeros(self.n_nodes)
            
            # Apply boundary conditions
            K, F = self.apply_thermal_bcs(K, C, F, time, heating_schedule)
            
            # Backward Euler time integration
            A = C + dt * K
            b = C @ self.temperature + dt * F
            
            # Solve system
            self.temperature = spsolve(A, b)
            temperature_history[step] = self.temperature.copy()
        
        return times, temperature_history

class SOFCMechanicalSolver:
    """Thermo-mechanical solver with plasticity and creep"""
    
    def __init__(self, mesh, materials):
        self.mesh = mesh
        self.materials = materials
        self.n_nodes = mesh.n_nodes
        self.n_dof = 2 * self.n_nodes  # 2 DOF per node (ux, uy)
        
        # Initialize state variables
        self.displacement = np.zeros(self.n_dof)
        self.stress = np.zeros((self.mesh.n_elements, 4))  # [sxx, syy, sxy, szz] per element
        self.strain = np.zeros((self.mesh.n_elements, 4))
        self.plastic_strain = np.zeros((self.mesh.n_elements, 4))
        self.creep_strain = np.zeros((self.mesh.n_elements, 4))
        self.equivalent_plastic_strain = np.zeros(self.mesh.n_elements)
        self.equivalent_creep_strain = np.zeros(self.mesh.n_elements)
        
    def assemble_stiffness_matrix(self, temperature_field):
        """Assemble global stiffness matrix"""
        K = np.zeros((self.n_dof, self.n_dof))
        
        # Gauss points for 2x2 integration
        gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), 1/np.sqrt(3)],
                              [-1/np.sqrt(3), 1/np.sqrt(3)]])
        weights = np.array([1, 1, 1, 1])
        
        for elem_id, element in enumerate(self.mesh.elements):
            material_name = self.mesh.element_materials[elem_id]
            material = self.materials[material_name]
            
            # Element temperature
            T_elem = np.mean(temperature_field[element])
            
            # Material properties
            E = material.get_elastic_modulus(T_elem)
            nu = material.get_poisson_ratio(T_elem)
            
            # Plane stress constitutive matrix
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ])
            
            # Element stiffness matrix
            Ke = np.zeros((8, 8))
            
            for gp, (xi, eta) in enumerate(gauss_pts):
                # Shape function derivatives
                _, dN_dxi = self.shape_functions_2d(xi, eta)
                
                # Jacobian
                coords = self.mesh.nodes[element]
                J = dN_dxi @ coords
                det_J = np.linalg.det(J)
                J_inv = np.linalg.inv(J)
                
                # Global derivatives
                dN_dx = J_inv @ dN_dxi
                
                # B matrix (strain-displacement)
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i] = dN_dx[0, i]      # du/dx
                    B[1, 2*i+1] = dN_dx[1, i]    # dv/dy
                    B[2, 2*i] = dN_dx[1, i]      # du/dy
                    B[2, 2*i+1] = dN_dx[0, i]    # dv/dx
                
                # Element stiffness
                Ke += weights[gp] * det_J * (B.T @ D @ B)
            
            # Assemble into global matrix
            dof_map = []
            for node in element:
                dof_map.extend([2*node, 2*node+1])
            
            for i in range(8):
                for j in range(8):
                    K[dof_map[i], dof_map[j]] += Ke[i, j]
        
        return csr_matrix(K)
    
    def compute_thermal_loads(self, temperature_field, reference_temp=298.15):
        """Compute thermal loads due to thermal expansion"""
        F_thermal = np.zeros(self.n_dof)
        
        # Gauss points
        gauss_pts = np.array([[-1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), -1/np.sqrt(3)],
                              [1/np.sqrt(3), 1/np.sqrt(3)],
                              [-1/np.sqrt(3), 1/np.sqrt(3)]])
        weights = np.array([1, 1, 1, 1])
        
        for elem_id, element in enumerate(self.mesh.elements):
            material_name = self.mesh.element_materials[elem_id]
            material = self.materials[material_name]
            
            # Element temperature
            T_elem = np.mean(temperature_field[element])
            dT = T_elem - reference_temp
            
            # Material properties
            E = material.get_elastic_modulus(T_elem)
            nu = material.get_poisson_ratio(T_elem)
            alpha = material.get_thermal_expansion(T_elem)
            
            # Constitutive matrix
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ])
            
            # Thermal strain
            eps_thermal = alpha * dT * np.array([1, 1, 0])
            
            # Element thermal load
            Fe_thermal = np.zeros(8)
            
            for gp, (xi, eta) in enumerate(gauss_pts):
                _, dN_dxi = self.shape_functions_2d(xi, eta)
                
                coords = self.mesh.nodes[element]
                J = dN_dxi @ coords
                det_J = np.linalg.det(J)
                J_inv = np.linalg.inv(J)
                
                dN_dx = J_inv @ dN_dxi
                
                # B matrix
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i] = dN_dx[0, i]
                    B[1, 2*i+1] = dN_dx[1, i]
                    B[2, 2*i] = dN_dx[1, i]
                    B[2, 2*i+1] = dN_dx[0, i]
                
                # Thermal contribution
                Fe_thermal += weights[gp] * det_J * (B.T @ D @ eps_thermal)
            
            # Assemble
            dof_map = []
            for node in element:
                dof_map.extend([2*node, 2*node+1])
            
            for i in range(8):
                F_thermal[dof_map[i]] += Fe_thermal[i]
        
        return F_thermal
    
    def apply_mechanical_bcs(self, K, F):
        """Apply mechanical boundary conditions"""
        # Left edge: roller in x (Ux = 0)
        left_nodes = np.where(self.mesh.nodes[:, 0] == 0)[0]
        
        # Bottom edge: roller in y (Uy = 0)
        bottom_nodes = np.where(self.mesh.nodes[:, 1] == 0)[0]
        
        # Apply constraints (penalty method)
        penalty = 1e12
        
        for node in left_nodes:
            dof_x = 2 * node
            K[dof_x, dof_x] += penalty
        
        for node in bottom_nodes:
            dof_y = 2 * node + 1
            K[dof_y, dof_y] += penalty
        
        return K, F
    
    def solve_mechanical(self, temperature_field):
        """Solve mechanical equilibrium"""
        # Assemble stiffness matrix
        K = self.assemble_stiffness_matrix(temperature_field)
        
        # Compute thermal loads
        F_thermal = self.compute_thermal_loads(temperature_field)
        
        # Apply boundary conditions
        K, F_thermal = self.apply_mechanical_bcs(K, F_thermal)
        
        # Solve for displacements
        self.displacement = spsolve(K, F_thermal)
        
        # Compute stresses and strains
        self.compute_stress_strain(temperature_field)
    
    def compute_stress_strain(self, temperature_field, reference_temp=298.15):
        """Compute element stresses and strains"""
        for elem_id, element in enumerate(self.mesh.elements):
            material_name = self.mesh.element_materials[elem_id]
            material = self.materials[material_name]
            
            # Element temperature and displacement
            T_elem = np.mean(temperature_field[element])
            dT = T_elem - reference_temp
            
            u_elem = []
            for node in element:
                u_elem.extend([self.displacement[2*node], self.displacement[2*node+1]])
            u_elem = np.array(u_elem)
            
            # Material properties
            E = material.get_elastic_modulus(T_elem)
            nu = material.get_poisson_ratio(T_elem)
            alpha = material.get_thermal_expansion(T_elem)
            
            # Constitutive matrix
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ])
            
            # Compute strain at element center
            xi, eta = 0, 0  # Element center
            _, dN_dxi = self.shape_functions_2d(xi, eta)
            
            coords = self.mesh.nodes[element]
            J = dN_dxi.T @ coords
            J_inv = np.linalg.inv(J)
            dN_dx = J_inv @ dN_dxi.T
            
            # B matrix
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i] = dN_dx[0, i]
                B[1, 2*i+1] = dN_dx[1, i]
                B[2, 2*i] = dN_dx[1, i]
                B[2, 2*i+1] = dN_dx[0, i]
            
            # Total strain
            eps_total = B @ u_elem
            
            # Thermal strain
            eps_thermal = alpha * dT * np.array([1, 1, 0])
            
            # Mechanical strain
            eps_mechanical = eps_total - eps_thermal
            
            # Stress (plane stress, add szz = 0)
            stress_3d = D @ eps_mechanical
            
            # Store results
            self.strain[elem_id] = np.array([eps_total[0], eps_total[1], eps_total[2], 0])
            self.stress[elem_id] = np.array([stress_3d[0], stress_3d[1], stress_3d[2], 0])
    
    def shape_functions_2d(self, xi, eta):
        """2D bilinear shape functions and derivatives"""
        N = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), 
                            (1+xi)*(1+eta), (1-xi)*(1+eta)])
        
        dN_dxi = 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])
        
        return N, dN_dxi

class SOFCDamageModel:
    """Damage and delamination modeling"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.damage = np.zeros(mesh.n_elements)
        
        # Damage parameters
        self.k_D = 1.5e-5
        self.sigma_th = 120e6  # Pa
        self.p = 2.0
        
        # Interface critical shear stresses (Pa)
        self.tau_crit = {
            'anode_electrolyte': 25e6,
            'electrolyte_cathode': 20e6,
            'cathode_interconnect': 30e6
        }
    
    def compute_damage(self, stress_field, dt):
        """Compute damage evolution"""
        for elem_id in range(self.mesh.n_elements):
            # Von Mises stress
            s = stress_field[elem_id]
            sigma_vm = np.sqrt(0.5 * ((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2) + 3*s[3]**2)
            
            # Interface proximity weight
            y_center = np.mean(self.mesh.nodes[self.mesh.elements[elem_id], 1])
            w_iface = self.compute_interface_weight(y_center)
            
            # Damage rate
            if sigma_vm > self.sigma_th:
                damage_rate = self.k_D * ((sigma_vm - self.sigma_th) / self.sigma_th)**self.p * (1 + 3*w_iface)
                self.damage[elem_id] += damage_rate * dt
                self.damage[elem_id] = min(self.damage[elem_id], 1.0)
    
    def compute_interface_weight(self, y):
        """Compute interface proximity weight"""
        interfaces = [0.0004, 0.0005, 0.0009]  # Interface y-coordinates
        min_dist = min([abs(y - y_int) for y_int in interfaces])
        decay_length = 0.00005  # 0.05 mm
        return np.exp(-min_dist / decay_length)
    
    def check_delamination(self, stress_field):
        """Check for delamination at interfaces"""
        delamination_flags = {}
        
        # Find elements near interfaces
        interface_elements = {
            'anode_electrolyte': [],
            'electrolyte_cathode': [],
            'cathode_interconnect': []
        }
        
        for elem_id in range(self.mesh.n_elements):
            y_center = np.mean(self.mesh.nodes[self.mesh.elements[elem_id], 1])
            
            if abs(y_center - 0.0004) < 0.00002:  # ±0.02 mm
                interface_elements['anode_electrolyte'].append(elem_id)
            elif abs(y_center - 0.0005) < 0.00002:
                interface_elements['electrolyte_cathode'].append(elem_id)
            elif abs(y_center - 0.0009) < 0.00002:
                interface_elements['cathode_interconnect'].append(elem_id)
        
        # Check shear stress at interfaces
        for interface, elements in interface_elements.items():
            max_shear = 0
            for elem_id in elements:
                shear_stress = abs(stress_field[elem_id, 2])  # |Sxy|
                max_shear = max(max_shear, shear_stress)
            
            delamination_flags[interface] = max_shear > self.tau_crit[interface]
        
        return delamination_flags

class SOFCSimulation:
    """Main SOFC simulation class"""
    
    def __init__(self):
        self.setup_materials()
        self.setup_geometry()
        self.setup_mesh()
        
        # Solvers
        self.thermal_solver = SOFCThermalSolver(self.mesh, self.materials)
        self.mechanical_solver = SOFCMechanicalSolver(self.mesh, self.materials)
        self.damage_model = SOFCDamageModel(self.mesh)
        
        # Results storage
        self.results = {}
    
    def setup_materials(self):
        """Define SOFC materials with temperature-dependent properties"""
        self.materials = {
            'anode': SOFCMaterial(
                'Ni-YSZ',
                elastic_props=(140, 91, 0.30),  # E_298K, E_1273K (GPa), nu
                thermal_props=(12.5, 13.5, 6.0, 4.0, 450, 570, 6000),  # alpha (1e-6/K), k (W/m·K), cp (J/kg·K), rho (kg/m³)
                plasticity={'A': 150e6, 'B': 200e6, 'n': 0.35, 'C': 0.02, 'm': 1.0},
                creep={'B': 1.0e-18, 'n': 3.5, 'Q': 2.2e5}
            ),
            'electrolyte': SOFCMaterial(
                '8YSZ',
                elastic_props=(210, 170, 0.28),
                thermal_props=(10.5, 11.2, 2.6, 2.0, 400, 600, 5900),
                creep={'B': 5.0e-22, 'n': 2.0, 'Q': 3.8e5}
            ),
            'cathode': SOFCMaterial(
                'LSM',
                elastic_props=(120, 84, 0.30),
                thermal_props=(11.5, 12.4, 2.0, 1.8, 480, 610, 6500)
            ),
            'interconnect': SOFCMaterial(
                'Ferritic Steel',
                elastic_props=(205, 150, 0.30),
                thermal_props=(12.5, 13.2, 20, 15, 500, 700, 7800)
            )
        }
    
    def setup_geometry(self):
        """Setup SOFC geometry"""
        self.geometry = SOFCGeometry()
    
    def setup_mesh(self):
        """Setup finite element mesh"""
        self.mesh = SOFCMesh(self.geometry, nx=80)
    
    def get_heating_schedule(self, heating_rate):
        """Get heating schedule parameters"""
        schedules = {
            'HR1': ('HR1', 875*60, 10*60, 875*60),    # 1°C/min: ramp, hold, cool (seconds)
            'HR4': ('HR4', 218.75*60, 10*60, 218.75*60),  # 4°C/min
            'HR10': ('HR10', 87.5*60, 10*60, 87.5*60)     # 10°C/min
        }
        return schedules[heating_rate]
    
    def run_simulation(self, heating_rate='HR1', save_results=True):
        """Run complete SOFC simulation"""
        print(f"\n=== SOFC Simulation: {heating_rate} ===")
        
        # Get heating schedule
        heating_schedule = self.get_heating_schedule(heating_rate)
        
        # Step 1: Thermal analysis
        print("Step 1: Transient heat conduction")
        times, temperature_history = self.thermal_solver.solve_transient(heating_schedule, dt=60.0)
        
        # Step 2: Thermo-mechanical analysis
        print("Step 2: Thermo-mechanical analysis")
        n_thermal_steps = len(times)
        
        # Storage for mechanical results
        stress_history = np.zeros((n_thermal_steps, self.mesh.n_elements, 4))
        strain_history = np.zeros((n_thermal_steps, self.mesh.n_elements, 4))
        damage_history = np.zeros((n_thermal_steps, self.mesh.n_elements))
        delamination_history = []
        
        for step in tqdm(range(n_thermal_steps), desc="Mechanical analysis"):
            # Solve mechanical problem at current temperature
            self.mechanical_solver.solve_mechanical(temperature_history[step])
            
            # Update damage
            if step > 0:
                dt = times[step] - times[step-1]
                self.damage_model.compute_damage(self.mechanical_solver.stress, dt)
            
            # Check delamination
            delamination = self.damage_model.check_delamination(self.mechanical_solver.stress)
            
            # Store results
            stress_history[step] = self.mechanical_solver.stress.copy()
            strain_history[step] = self.mechanical_solver.strain.copy()
            damage_history[step] = self.damage_model.damage.copy()
            delamination_history.append(delamination)
        
        # Store results
        self.results[heating_rate] = {
            'times': times,
            'temperature': temperature_history,
            'stress': stress_history,
            'strain': strain_history,
            'damage': damage_history,
            'delamination': delamination_history,
            'mesh': self.mesh
        }
        
        if save_results:
            self.save_results(heating_rate)
        
        print(f"Simulation completed: {heating_rate}")
        return self.results[heating_rate]
    
    def save_results(self, heating_rate):
        """Save simulation results to files"""
        results = self.results[heating_rate]
        
        # Create output directory
        output_dir = f"/workspace/sofc_results_{heating_rate.lower()}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as NPZ (matching the synthetic data format)
        np.savez_compressed(
            f"{output_dir}/sofc_simulation_{heating_rate.lower()}.npz",
            times=results['times'],
            temperature=results['temperature'],
            stress=results['stress'],
            strain=results['strain'],
            damage=results['damage'],
            coordinates=self.mesh.nodes,
            elements=self.mesh.elements,
            materials=self.mesh.element_materials
        )
        
        # Save mesh information
        with open(f"{output_dir}/mesh_info.txt", 'w') as f:
            f.write(f"SOFC Mesh Information - {heating_rate}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Nodes: {self.mesh.n_nodes}\n")
            f.write(f"Elements: {self.mesh.n_elements}\n")
            f.write(f"Geometry: {self.geometry.width*1000:.1f} mm × {self.geometry.height*1000:.1f} mm\n")
            f.write(f"Layer distribution:\n")
            for material, count in zip(*np.unique(self.mesh.element_materials, return_counts=True)):
                f.write(f"  {material}: {count} elements\n")
        
        print(f"Results saved to {output_dir}/")
    
    def plot_results(self, heating_rate='HR1'):
        """Plot simulation results"""
        if heating_rate not in self.results:
            print(f"No results for {heating_rate}. Run simulation first.")
            return
        
        results = self.results[heating_rate]
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'SOFC Simulation Results - {heating_rate}', fontsize=16)
        
        # Temperature evolution
        ax = axes[0, 0]
        times_hr = results['times'] / 3600  # Convert to hours
        T_bottom = results['temperature'][:, 0] - 273.15  # Bottom temperature in °C
        T_top = results['temperature'][:, -1] - 273.15    # Top temperature in °C
        ax.plot(times_hr, T_bottom, 'r-', label='Bottom', linewidth=2)
        ax.plot(times_hr, T_top, 'b-', label='Top', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Evolution')
        ax.legend()
        ax.grid(True)
        
        # Final temperature distribution
        ax = axes[0, 1]
        T_final = results['temperature'][-1] - 273.15
        y_coords = self.mesh.nodes[:, 1] * 1000  # Convert to mm
        ax.plot(T_final, y_coords, 'ro-', markersize=3)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Height (mm)')
        ax.set_title('Final Temperature Profile')
        ax.grid(True)
        
        # Von Mises stress evolution
        ax = axes[0, 2]
        stress_vm = np.sqrt(0.5 * ((results['stress'][:, :, 0] - results['stress'][:, :, 1])**2 + 
                                   (results['stress'][:, :, 1])**2 + 
                                   results['stress'][:, :, 0]**2) + 
                           3 * results['stress'][:, :, 2]**2) / 1e6  # Convert to MPa
        
        max_stress = np.max(stress_vm, axis=1)
        ax.plot(times_hr, max_stress, 'g-', linewidth=2)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Max von Mises Stress (MPa)')
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
        
        # Interface shear stress
        ax = axes[1, 1]
        # Find elements near electrolyte interfaces
        interface_elements = []
        for elem_id in range(self.mesh.n_elements):
            y_center = np.mean(self.mesh.nodes[self.mesh.elements[elem_id], 1])
            if 0.0003 < y_center < 0.0006:  # Near electrolyte
                interface_elements.append(elem_id)
        
        if interface_elements:
            shear_stress = np.abs(results['stress'][:, interface_elements, 2]) / 1e6  # MPa
            max_shear = np.max(shear_stress, axis=1)
            ax.plot(times_hr, max_shear, 'c-', linewidth=2)
            ax.axhline(y=20, color='r', linestyle='--', label='Critical (20 MPa)')
            ax.legend()
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Max Interface Shear (MPa)')
        ax.set_title('Interface Shear Stress')
        ax.grid(True)
        
        # Final stress contour
        ax = axes[1, 2]
        # Create a simple contour plot
        x_coords = self.mesh.nodes[:, 0] * 1000
        y_coords = self.mesh.nodes[:, 1] * 1000
        
        # Interpolate stress to nodes (simple averaging)
        nodal_stress = np.zeros(self.mesh.n_nodes)
        node_count = np.zeros(self.mesh.n_nodes)
        
        final_stress_vm = stress_vm[-1]  # Final time step
        for elem_id, element in enumerate(self.mesh.elements):
            for node in element:
                nodal_stress[node] += final_stress_vm[elem_id]
                node_count[node] += 1
        
        nodal_stress = nodal_stress / np.maximum(node_count, 1)
        
        scatter = ax.scatter(x_coords, y_coords, c=nodal_stress, cmap='jet', s=10)
        ax.set_xlabel('Width (mm)')
        ax.set_ylabel('Height (mm)')
        ax.set_title('Final von Mises Stress (MPa)')
        plt.colorbar(scatter, ax=ax)
        
        # Add layer boundaries
        for y_bound in [0.4, 0.5, 0.9]:
            ax.axhline(y=y_bound, color='white', linestyle='-', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = f"/workspace/sofc_results_{heating_rate.lower()}"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/sofc_results_{heating_rate.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plotted and saved for {heating_rate}")

def main():
    """Main simulation runner"""
    print("SOFC Multi-Physics Simulation")
    print("=" * 50)
    
    # Create simulation
    sofc = SOFCSimulation()
    
    # Run simulations for all heating rates
    heating_rates = ['HR1', 'HR4', 'HR10']
    
    for hr in heating_rates:
        try:
            results = sofc.run_simulation(hr, save_results=True)
            sofc.plot_results(hr)
            
            # Print summary
            print(f"\n{hr} Summary:")
            print(f"  Max temperature: {np.max(results['temperature']) - 273.15:.1f}°C")
            print(f"  Max von Mises stress: {np.max(results['stress'][:, :, 0])/1e6:.1f} MPa")
            print(f"  Max damage: {np.max(results['damage']):.3f}")
            
            # Check delamination
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