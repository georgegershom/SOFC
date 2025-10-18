"""
FEA software input file exporters
Generates input files for ABAQUS, ANSYS, and COMSOL
"""

import numpy as np
import os
from typing import Dict, List, Optional
from datetime import datetime

class FEAExporterBase:
    """Base class for FEA exporters"""
    
    def __init__(self, data: Dict):
        """
        Initialize exporter with dataset
        
        Parameters:
        -----------
        data : Dict
            Complete dataset with all properties
        """
        self.data = data
        self.timestamp = datetime.now().isoformat()
        
    def export(self, output_dir: str):
        """Export to specified directory"""
        raise NotImplementedError
        
    def _ensure_dir(self, path: str):
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)


class ABAQUSExporter(FEAExporterBase):
    """
    ABAQUS input file exporter
    Generates .inp files and material property tables
    """
    
    def export(self, output_dir: str):
        """
        Export ABAQUS input files
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        """
        self._ensure_dir(output_dir)
        
        for mix_id in self.data.keys():
            self._export_mix(mix_id, output_dir)
        
        # Generate master input file
        self._generate_master_file(output_dir)
        
    def _export_mix(self, mix_id: str, output_dir: str):
        """Export data for specific mix"""
        mix_data = self.data[mix_id]
        
        # Create mix directory
        mix_dir = os.path.join(output_dir, mix_id)
        self._ensure_dir(mix_dir)
        
        # Generate material properties file
        self._generate_material_file(mix_id, mix_data, mix_dir)
        
        # Generate amplitude curves for temperature loading
        self._generate_amplitude_curves(mix_data, mix_dir)
        
        # Generate field variable dependencies
        self._generate_field_dependencies(mix_data, mix_dir)
        
    def _generate_material_file(self, mix_id: str, data: Dict, output_dir: str):
        """Generate ABAQUS material properties file"""
        
        filepath = os.path.join(output_dir, f'{mix_id}_material.inp')
        
        with open(filepath, 'w') as f:
            # Header
            f.write(f"** ABAQUS Material Properties\n")
            f.write(f"** Mix ID: {mix_id}\n")
            f.write(f"** Generated: {self.timestamp}\n")
            f.write(f"** ===========================================\n\n")
            
            # Material definition
            f.write(f"*Material, name=CONCRETE_{mix_id}\n")
            
            # Density
            if 'thermal' in data:
                thermal = data['thermal']
                if 'density' in thermal:
                    f.write(f"*Density\n")
                    for i, temp in enumerate(thermal['temperature']):
                        f.write(f" {thermal['density'][i]:.1f}, {temp:.1f}\n")
            
            # Elastic properties
            if 'mechanical' in data:
                elastic = data['mechanical']['elastic']
                f.write(f"*Elastic, type=ISOTROPIC\n")
                for i, temp in enumerate(elastic['temperature']):
                    E = elastic['elastic_modulus'][i] * 1e9  # Convert to Pa
                    nu = elastic['poisson_ratio'][i]
                    f.write(f" {E:.3e}, {nu:.4f}, {temp:.1f}\n")
            
            # Thermal properties
            if 'thermal' in data:
                thermal = data['thermal']
                
                # Conductivity
                f.write(f"*Conductivity, type=ISO\n")
                for i, temp in enumerate(thermal['temperature']):
                    k = thermal['thermal_conductivity'][i]
                    f.write(f" {k:.4f}, {temp:.1f}\n")
                
                # Specific heat
                f.write(f"*Specific Heat\n")
                for i, temp in enumerate(thermal['temperature']):
                    c = thermal['specific_heat_capacity'][i]
                    f.write(f" {c:.1f}, {temp:.1f}\n")
                
                # Thermal expansion
                f.write(f"*Expansion, type=ISO\n")
                for i, temp in enumerate(thermal['temperature']):
                    alpha = thermal['thermal_expansion'][i] * 1e-6  # Convert to 1/K
                    f.write(f" {alpha:.3e}, {temp:.1f}\n")
            
            # Concrete Damaged Plasticity
            if 'mechanical' in data:
                plastic = data['mechanical']['plastic']
                constitutive = data['mechanical']['constitutive_model']
                
                f.write(f"*Concrete Damaged Plasticity\n")
                f.write(f" {plastic['dilation_angle'][0]:.1f}, ")  # Dilation angle
                f.write(f"{plastic['eccentricity']:.2f}, ")  # Eccentricity
                f.write(f"{plastic['fb0_fc0_ratio']:.2f}, ")  # fb0/fc0
                f.write(f"{plastic['K']:.3f}, ")  # K
                f.write(f"0.0001\n")  # Viscosity parameter
                
                # Concrete Compression Hardening
                strength = data['mechanical']['strength']
                damage = data['mechanical']['damage']
                
                f.write(f"*Concrete Compression Hardening\n")
                for i, temp in enumerate(strength['temperature']):
                    fc = strength['compressive_strength'][i] * 1e6  # Convert to Pa
                    eps_in = damage['damage_initiation']['compression_strain'][i]
                    f.write(f" {fc:.3e}, {eps_in:.6f}, {temp:.1f}\n")
                
                # Concrete Tension Stiffening
                f.write(f"*Concrete Tension Stiffening, type=STRAIN\n")
                for i, temp in enumerate(strength['temperature']):
                    ft = strength['tensile_strength'][i] * 1e6  # Convert to Pa
                    eps_cr = damage['damage_initiation']['tension_strain'][i]
                    f.write(f" {ft:.3e}, 0.0, {temp:.1f}\n")
                    f.write(f" {ft*0.1:.3e}, {eps_cr*10:.6f}, {temp:.1f}\n")
                
                # Concrete Compression Damage
                f.write(f"*Concrete Compression Damage\n")
                comp_damage = damage['compression_damage']
                for i, strain in enumerate(comp_damage['plastic_strain']):
                    for j, temp in enumerate(strength['temperature'][::10]):  # Sample temperatures
                        d = comp_damage['damage_variable'][j][i]
                        f.write(f" {d:.4f}, {strain:.6f}, {temp:.1f}\n")
                
                # Concrete Tension Damage  
                f.write(f"*Concrete Tension Damage\n")
                tens_damage = damage['tension_damage']
                for i, strain in enumerate(tens_damage['plastic_strain']):
                    for j, temp in enumerate(strength['temperature'][::10]):  # Sample temperatures
                        d = tens_damage['damage_variable'][j][i]
                        f.write(f" {d:.4f}, {strain:.6f}, {temp:.1f}\n")
            
            # Creep properties
            if 'mechanical' in data and 'creep' in data['mechanical']:
                creep = data['mechanical']['creep']
                f.write(f"*Creep, law=TIME\n")
                f.write(f"*Potential, type=HYPERBOLIC\n")
                for i, temp in enumerate(creep['temperature']):
                    A = creep['creep_compliance'][i] * 1e-6  # Convert
                    n = creep['creep_exponent'][i]
                    f.write(f" {A:.3e}, {n:.3f}, 0.0, {temp:.1f}\n")
            
            # Permeability (for coupled analysis)
            if 'transport' in data:
                perm = data['transport']['permeability']
                f.write(f"*Permeability, type=ISOTROPIC\n")
                for i, temp in enumerate(perm['temperature']):
                    k = perm['intrinsic_permeability'][i] * 1e-18  # Convert
                    void_ratio = 0.3  # Approximate
                    f.write(f" {k:.3e}, {void_ratio:.3f}, {temp:.1f}\n")
    
    def _generate_amplitude_curves(self, data: Dict, output_dir: str):
        """Generate temperature loading amplitude curves"""
        
        filepath = os.path.join(output_dir, 'amplitude_curves.inp')
        
        with open(filepath, 'w') as f:
            f.write("** Temperature Loading Curves\n")
            f.write("** ===========================================\n\n")
            
            # ISO 834 Standard Fire Curve
            f.write("*Amplitude, name=ISO834_FIRE\n")
            times = np.array([0, 60, 120, 300, 600, 1200, 1800, 3600, 7200])
            for t in times:
                if t == 0:
                    T = 20
                else:
                    T = 20 + 345 * np.log10(8 * t/60 + 1)
                f.write(f" {t:.1f}, {T:.1f}\n")
            
            # ASTM E119 Fire Curve
            f.write("*Amplitude, name=ASTM_E119\n")
            for t in times:
                if t == 0:
                    T = 20
                else:
                    T = 20 + 750 * (1 - np.exp(-3.79553 * np.sqrt(t/3600)))
                f.write(f" {t:.1f}, {T:.1f}\n")
            
            # Hydrocarbon Fire Curve
            f.write("*Amplitude, name=HYDROCARBON_FIRE\n")
            for t in times:
                T = 20 + 1080 * (1 - 0.325 * np.exp(-0.167 * t/60) - 0.675 * np.exp(-2.5 * t/60))
                f.write(f" {t:.1f}, {T:.1f}\n")
            
            # Slow Heating Curve
            f.write("*Amplitude, name=SLOW_HEATING\n")
            for t in times:
                T = 20 + (800 - 20) * t / 7200  # Linear to 800°C in 2 hours
                f.write(f" {t:.1f}, {T:.1f}\n")
    
    def _generate_field_dependencies(self, data: Dict, output_dir: str):
        """Generate field variable dependencies"""
        
        filepath = os.path.join(output_dir, 'field_dependencies.inp')
        
        with open(filepath, 'w') as f:
            f.write("** Field Variable Dependencies\n")
            f.write("** ===========================================\n\n")
            
            f.write("** Field Variable 1: Temperature (°C)\n")
            f.write("** Field Variable 2: Moisture Content (%)\n")
            f.write("** Field Variable 3: Pore Pressure (MPa)\n")
            f.write("** Field Variable 4: Damage Variable (0-1)\n\n")
            
            f.write("*Initial Conditions, type=TEMPERATURE\n")
            f.write("ALL_NODES, 20.0\n\n")
            
            if 'transport' in data and 'moisture' in data['transport']:
                moisture = data['transport']['moisture']
                f.write("*Initial Conditions, type=FIELD, variable=2\n")
                f.write(f"ALL_NODES, {moisture['moisture_content'][0]:.2f}\n\n")
    
    def _generate_master_file(self, output_dir: str):
        """Generate master input file that includes all components"""
        
        filepath = os.path.join(output_dir, 'master_model.inp')
        
        with open(filepath, 'w') as f:
            f.write("** ABAQUS Master Input File\n")
            f.write(f"** Generated: {self.timestamp}\n")
            f.write("** ===========================================\n\n")
            
            f.write("*Heading\n")
            f.write("Thermo-Mechanical Analysis of Fire-Resistant Rubberized Concrete\n\n")
            
            # Include geometry (user-defined)
            f.write("** Include geometry definition\n")
            f.write("*Include, input=geometry.inp\n\n")
            
            # Include materials for each mix
            f.write("** Include material properties\n")
            for mix_id in self.data.keys():
                f.write(f"*Include, input={mix_id}/{mix_id}_material.inp\n")
            
            # Include amplitude curves
            f.write("\n** Include amplitude curves\n")
            f.write("*Include, input=amplitude_curves.inp\n\n")
            
            # Step definition template
            f.write("** Analysis Steps\n")
            f.write("** ===========================================\n")
            f.write("*Step, name=ThermalStep, nlgeom=YES, inc=1000\n")
            f.write("*Coupled Temperature-Displacement, creep=none, steady state\n")
            f.write("0.1, 7200., 1e-05, 100.\n")
            f.write("** Apply boundary conditions and loads here\n")
            f.write("*End Step\n")


class ANSYSExporter(FEAExporterBase):
    """
    ANSYS APDL input file exporter
    Generates APDL command files and material property tables
    """
    
    def export(self, output_dir: str):
        """Export ANSYS APDL files"""
        self._ensure_dir(output_dir)
        
        for mix_id in self.data.keys():
            self._export_mix(mix_id, output_dir)
        
        # Generate master macro file
        self._generate_master_macro(output_dir)
    
    def _export_mix(self, mix_id: str, output_dir: str):
        """Export data for specific mix"""
        mix_data = self.data[mix_id]
        
        # Create mix directory
        mix_dir = os.path.join(output_dir, mix_id)
        self._ensure_dir(mix_dir)
        
        # Generate material properties macro
        self._generate_material_macro(mix_id, mix_data, mix_dir)
        
        # Generate table arrays for temperature dependencies
        self._generate_table_arrays(mix_id, mix_data, mix_dir)
    
    def _generate_material_macro(self, mix_id: str, data: Dict, output_dir: str):
        """Generate ANSYS APDL material macro"""
        
        filepath = os.path.join(output_dir, f'{mix_id}_material.mac')
        
        with open(filepath, 'w') as f:
            # Header
            f.write(f"! ANSYS APDL Material Properties Macro\n")
            f.write(f"! Mix ID: {mix_id}\n")
            f.write(f"! Generated: {self.timestamp}\n")
            f.write(f"! =========================================\n\n")
            
            f.write(f"/PREP7\n")
            f.write(f"! Define material number\n")
            f.write(f"MAT_{mix_id} = {list(self.data.keys()).index(mix_id) + 1}\n\n")
            
            # Temperature array
            if 'thermal' in data:
                temps = data['thermal']['temperature']
                f.write(f"! Temperature array definition\n")
                f.write(f"*DIM,TEMP_ARRAY,,{len(temps)}\n")
                for i, temp in enumerate(temps):
                    f.write(f"TEMP_ARRAY({i+1}) = {temp:.1f}\n")
                f.write("\n")
            
            # Elastic properties
            if 'mechanical' in data:
                elastic = data['mechanical']['elastic']
                
                # Young's modulus
                f.write(f"! Young's Modulus [Pa]\n")
                f.write(f"*DIM,EX_{mix_id},TABLE,{len(temps)},1,1,TEMP\n")
                for i, temp in enumerate(elastic['temperature']):
                    E = elastic['elastic_modulus'][i] * 1e9
                    f.write(f"EX_{mix_id}({i+1},0,1) = {temp:.1f}\n")
                    f.write(f"EX_{mix_id}({i+1},1,1) = {E:.3e}\n")
                f.write(f"MP,EX,MAT_{mix_id},%EX_{mix_id}%\n\n")
                
                # Poisson's ratio
                f.write(f"! Poisson's Ratio\n")
                f.write(f"*DIM,NUXY_{mix_id},TABLE,{len(temps)},1,1,TEMP\n")
                for i, temp in enumerate(elastic['temperature']):
                    nu = elastic['poisson_ratio'][i]
                    f.write(f"NUXY_{mix_id}({i+1},0,1) = {temp:.1f}\n")
                    f.write(f"NUXY_{mix_id}({i+1},1,1) = {nu:.4f}\n")
                f.write(f"MP,NUXY,MAT_{mix_id},%NUXY_{mix_id}%\n\n")
            
            # Thermal properties
            if 'thermal' in data:
                thermal = data['thermal']
                
                # Density
                f.write(f"! Density [kg/m3]\n")
                f.write(f"*DIM,DENS_{mix_id},TABLE,{len(temps)},1,1,TEMP\n")
                for i, temp in enumerate(thermal['temperature']):
                    rho = thermal['density'][i]
                    f.write(f"DENS_{mix_id}({i+1},0,1) = {temp:.1f}\n")
                    f.write(f"DENS_{mix_id}({i+1},1,1) = {rho:.1f}\n")
                f.write(f"MP,DENS,MAT_{mix_id},%DENS_{mix_id}%\n\n")
                
                # Thermal conductivity
                f.write(f"! Thermal Conductivity [W/m-K]\n")
                f.write(f"*DIM,KXX_{mix_id},TABLE,{len(temps)},1,1,TEMP\n")
                for i, temp in enumerate(thermal['temperature']):
                    k = thermal['thermal_conductivity'][i]
                    f.write(f"KXX_{mix_id}({i+1},0,1) = {temp:.1f}\n")
                    f.write(f"KXX_{mix_id}({i+1},1,1) = {k:.4f}\n")
                f.write(f"MP,KXX,MAT_{mix_id},%KXX_{mix_id}%\n\n")
                
                # Specific heat
                f.write(f"! Specific Heat [J/kg-K]\n")
                f.write(f"*DIM,C_{mix_id},TABLE,{len(temps)},1,1,TEMP\n")
                for i, temp in enumerate(thermal['temperature']):
                    c = thermal['specific_heat_capacity'][i]
                    f.write(f"C_{mix_id}({i+1},0,1) = {temp:.1f}\n")
                    f.write(f"C_{mix_id}({i+1},1,1) = {c:.1f}\n")
                f.write(f"MP,C,MAT_{mix_id},%C_{mix_id}%\n\n")
                
                # Thermal expansion
                f.write(f"! Thermal Expansion Coefficient [1/K]\n")
                f.write(f"*DIM,ALPX_{mix_id},TABLE,{len(temps)},1,1,TEMP\n")
                for i, temp in enumerate(thermal['temperature']):
                    alpha = thermal['thermal_expansion'][i] * 1e-6
                    f.write(f"ALPX_{mix_id}({i+1},0,1) = {temp:.1f}\n")
                    f.write(f"ALPX_{mix_id}({i+1},1,1) = {alpha:.3e}\n")
                f.write(f"MP,ALPX,MAT_{mix_id},%ALPX_{mix_id}%\n\n")
            
            # Concrete nonlinear properties
            if 'mechanical' in data:
                strength = data['mechanical']['strength']
                
                # Use TB commands for concrete material
                f.write(f"! Concrete Material Model\n")
                f.write(f"TB,CONCR,MAT_{mix_id},,,,MISO\n")
                
                # Define stress-strain curves at different temperatures
                for j, temp in enumerate(strength['temperature'][::10]):  # Sample temps
                    f.write(f"TBTEMP,{temp:.1f}\n")
                    
                    # Compression
                    fc = strength['compressive_strength'][j*10] * 1e6
                    f.write(f"TBDATA,1,0.3,0.5,{fc:.3e},2.0\n")  # Shear transfer coefficients and strength
                    
                    # Tension
                    ft = strength['tensile_strength'][j*10] * 1e6
                    f.write(f"TBDATA,5,{ft:.3e}\n")
            
            f.write(f"FINISH\n")
    
    def _generate_table_arrays(self, mix_id: str, data: Dict, output_dir: str):
        """Generate ANSYS table array files"""
        
        filepath = os.path.join(output_dir, f'{mix_id}_tables.txt')
        
        with open(filepath, 'w') as f:
            f.write(f"! Table Arrays for Mix {mix_id}\n")
            f.write(f"! =========================================\n\n")
            
            # Export key properties as tables for easy import
            if 'mechanical' in data:
                elastic = data['mechanical']['elastic']
                strength = data['mechanical']['strength']
                
                # Elastic modulus reduction factor
                f.write("! Elastic Modulus Reduction Factor vs Temperature\n")
                E_0 = elastic['elastic_modulus'][0]
                for i, temp in enumerate(elastic['temperature']):
                    factor = elastic['elastic_modulus'][i] / E_0
                    f.write(f"{temp:.1f}\t{factor:.4f}\n")
                f.write("\n")
                
                # Strength reduction factor
                f.write("! Compressive Strength Reduction Factor vs Temperature\n")
                fc_0 = strength['compressive_strength'][0]
                for i, temp in enumerate(strength['temperature']):
                    factor = strength['compressive_strength'][i] / fc_0
                    f.write(f"{temp:.1f}\t{factor:.4f}\n")
    
    def _generate_master_macro(self, output_dir: str):
        """Generate master ANSYS macro"""
        
        filepath = os.path.join(output_dir, 'master_analysis.mac')
        
        with open(filepath, 'w') as f:
            f.write("! ANSYS APDL Master Analysis Macro\n")
            f.write(f"! Generated: {self.timestamp}\n")
            f.write("! =========================================\n\n")
            
            f.write("FINISH\n")
            f.write("/CLEAR,START\n")
            f.write("/TITLE,Thermo-Mechanical Analysis of Rubberized Concrete\n\n")
            
            f.write("/PREP7\n")
            
            # Element type definition
            f.write("! Element Type Definition\n")
            f.write("ET,1,SOLID227    ! 10-node tetrahedral coupled field solid\n")
            f.write("KEYOPT,1,1,11    ! Structural-thermal\n\n")
            
            # Load material properties
            f.write("! Load Material Properties\n")
            for i, mix_id in enumerate(self.data.keys()):
                f.write(f"*USE,{mix_id}/{mix_id}_material.mac\n")
            
            f.write("\n! Meshing Commands\n")
            f.write("! (Add geometry and meshing commands here)\n\n")
            
            f.write("FINISH\n\n")
            
            # Solution phase
            f.write("/SOLU\n")
            f.write("ANTYPE,TRANS     ! Transient analysis\n")
            f.write("TRNOPT,FULL\n")
            f.write("LUMPM,0\n")
            f.write("NSUBST,20,50,10\n")
            f.write("OUTRES,ALL,ALL\n")
            f.write("AUTOTS,ON\n")
            f.write("TIME,7200        ! 2 hours\n\n")
            
            f.write("! Apply Boundary Conditions\n")
            f.write("! (Add BC commands here)\n\n")
            
            f.write("! Apply Temperature Loading\n")
            f.write("! ISO 834 Fire Curve\n")
            f.write("*DIM,TEMP_LOAD,TABLE,10,1,1,TIME\n")
            times = [0, 60, 300, 600, 1200, 1800, 3600, 5400, 7200]
            for i, t in enumerate(times):
                if t == 0:
                    T = 20
                else:
                    T = 20 + 345 * np.log10(8 * t/60 + 1)
                f.write(f"TEMP_LOAD({i+1},0,1) = {t:.1f}\n")
                f.write(f"TEMP_LOAD({i+1},1,1) = {T:.1f}\n")
            
            f.write("\nSOLVE\n")
            f.write("FINISH\n\n")
            
            f.write("/POST1\n")
            f.write("! Post-processing commands\n")
            f.write("FINISH\n")


class COMSOLExporter(FEAExporterBase):
    """
    COMSOL input file exporter
    Generates Java API files and material function files
    """
    
    def export(self, output_dir: str):
        """Export COMSOL files"""
        self._ensure_dir(output_dir)
        
        for mix_id in self.data.keys():
            self._export_mix(mix_id, output_dir)
        
        # Generate main Java API file
        self._generate_java_api(output_dir)
    
    def _export_mix(self, mix_id: str, output_dir: str):
        """Export data for specific mix"""
        mix_data = self.data[mix_id]
        
        # Create mix directory
        mix_dir = os.path.join(output_dir, mix_id)
        self._ensure_dir(mix_dir)
        
        # Generate material functions file
        self._generate_material_functions(mix_id, mix_data, mix_dir)
        
        # Generate interpolation functions
        self._generate_interpolation_functions(mix_id, mix_data, mix_dir)
    
    def _generate_material_functions(self, mix_id: str, data: Dict, output_dir: str):
        """Generate COMSOL material function definitions"""
        
        filepath = os.path.join(output_dir, f'{mix_id}_materials.txt')
        
        with open(filepath, 'w') as f:
            # Header
            f.write(f"# COMSOL Material Functions\n")
            f.write(f"# Mix ID: {mix_id}\n")
            f.write(f"# Generated: {self.timestamp}\n")
            f.write(f"# =========================================\n\n")
            
            # Function definitions for each property
            if 'thermal' in data:
                thermal = data['thermal']
                
                # Thermal conductivity function
                f.write("# Thermal Conductivity [W/(m*K)] vs Temperature [K]\n")
                f.write("k(T[1/K])\n")
                for i, temp in enumerate(thermal['temperature']):
                    T_K = temp + 273.15
                    k = thermal['thermal_conductivity'][i]
                    f.write(f"{T_K:.2f}\t{k:.4f}\n")
                f.write("\n")
                
                # Specific heat function
                f.write("# Specific Heat [J/(kg*K)] vs Temperature [K]\n")
                f.write("Cp(T[1/K])\n")
                for i, temp in enumerate(thermal['temperature']):
                    T_K = temp + 273.15
                    cp = thermal['specific_heat_capacity'][i]
                    f.write(f"{T_K:.2f}\t{cp:.1f}\n")
                f.write("\n")
                
                # Density function
                f.write("# Density [kg/m^3] vs Temperature [K]\n")
                f.write("rho(T[1/K])\n")
                for i, temp in enumerate(thermal['temperature']):
                    T_K = temp + 273.15
                    rho = thermal['density'][i]
                    f.write(f"{T_K:.2f}\t{rho:.1f}\n")
                f.write("\n")
            
            if 'mechanical' in data:
                elastic = data['mechanical']['elastic']
                
                # Young's modulus function
                f.write("# Young's Modulus [Pa] vs Temperature [K]\n")
                f.write("E(T[1/K])\n")
                for i, temp in enumerate(elastic['temperature']):
                    T_K = temp + 273.15
                    E = elastic['elastic_modulus'][i] * 1e9
                    f.write(f"{T_K:.2f}\t{E:.3e}\n")
                f.write("\n")
                
                # Poisson's ratio function
                f.write("# Poisson's Ratio [-] vs Temperature [K]\n")
                f.write("nu(T[1/K])\n")
                for i, temp in enumerate(elastic['temperature']):
                    T_K = temp + 273.15
                    nu = elastic['poisson_ratio'][i]
                    f.write(f"{T_K:.2f}\t{nu:.4f}\n")
    
    def _generate_interpolation_functions(self, mix_id: str, data: Dict, output_dir: str):
        """Generate COMSOL interpolation function files"""
        
        filepath = os.path.join(output_dir, f'{mix_id}_interpolations.java')
        
        with open(filepath, 'w') as f:
            f.write(f"// COMSOL Interpolation Functions\n")
            f.write(f"// Mix ID: {mix_id}\n")
            f.write(f"// Generated: {self.timestamp}\n\n")
            
            f.write(f"// Create interpolation functions for {mix_id}\n")
            
            # Thermal properties
            if 'thermal' in data:
                thermal = data['thermal']
                temps = [t + 273.15 for t in thermal['temperature']]  # Convert to K
                
                # Conductivity
                f.write(f'model.func().create("k_{mix_id}", "Interpolation");\n')
                f.write(f'model.func("k_{mix_id}").set("funcname", "k_{mix_id}");\n')
                f.write(f'model.func("k_{mix_id}").set("table", new String[][]{{')
                for i, T in enumerate(temps):
                    k = thermal['thermal_conductivity'][i]
                    f.write(f'{{"{T:.2f}", "{k:.4f}"}}')
                    if i < len(temps) - 1:
                        f.write(', ')
                f.write('});\n')
                f.write(f'model.func("k_{mix_id}").set("interp", "piecewisecubic");\n\n')
                
                # Specific heat
                f.write(f'model.func().create("Cp_{mix_id}", "Interpolation");\n')
                f.write(f'model.func("Cp_{mix_id}").set("funcname", "Cp_{mix_id}");\n')
                f.write(f'model.func("Cp_{mix_id}").set("table", new String[][]{{')
                for i, T in enumerate(temps):
                    cp = thermal['specific_heat_capacity'][i]
                    f.write(f'{{"{T:.2f}", "{cp:.1f}"}}')
                    if i < len(temps) - 1:
                        f.write(', ')
                f.write('});\n')
                f.write(f'model.func("Cp_{mix_id}").set("interp", "piecewisecubic");\n\n')
    
    def _generate_java_api(self, output_dir: str):
        """Generate main COMSOL Java API file"""
        
        filepath = os.path.join(output_dir, 'ThermoMechanicalModel.java')
        
        with open(filepath, 'w') as f:
            f.write("import com.comsol.model.*;\n")
            f.write("import com.comsol.model.util.*;\n\n")
            
            f.write("public class ThermoMechanicalModel {\n")
            f.write("    public static Model run() {\n")
            f.write("        Model model = ModelUtil.create(\"ThermoMechanicalModel\");\n\n")
            
            f.write('        model.modelPath("/path/to/model");\n')
            f.write('        model.label("Fire Resistant Rubberized Concrete");\n\n')
            
            # Model setup
            f.write("        // Create component\n")
            f.write('        model.component().create("comp1", true);\n')
            f.write('        model.component("comp1").geom().create("geom1", 3);\n\n')
            
            # Physics setup
            f.write("        // Add physics\n")
            f.write('        model.component("comp1").physics().create("ht", "HeatTransfer", "geom1");\n')
            f.write('        model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");\n\n')
            
            # Multiphysics coupling
            f.write("        // Add multiphysics coupling\n")
            f.write('        model.component("comp1").multiphysics().create("te1", "ThermalExpansion");\n')
            f.write('        model.component("comp1").multiphysics("te1").set("ht", "ht");\n')
            f.write('        model.component("comp1").multiphysics("te1").set("solid", "solid");\n\n')
            
            # Materials
            f.write("        // Create materials\n")
            for i, mix_id in enumerate(self.data.keys()):
                f.write(f'        createMaterial_{mix_id}(model);\n')
            
            f.write("\n        // Create study\n")
            f.write('        model.study().create("std1");\n')
            f.write('        model.study("std1").create("time", "Transient");\n')
            f.write('        model.study("std1").feature("time").set("tlist", "range(0,60,7200)");\n\n')
            
            f.write("        return model;\n")
            f.write("    }\n\n")
            
            # Material creation methods
            for mix_id in self.data.keys():
                f.write(f"    private static void createMaterial_{mix_id}(Model model) {{\n")
                f.write(f'        model.component("comp1").material().create("mat_{mix_id}", "Common");\n')
                f.write(f'        model.component("comp1").material("mat_{mix_id}").label("{mix_id}");\n')
                
                # Link to interpolation functions
                f.write(f'        model.component("comp1").material("mat_{mix_id}")')
                f.write(f'.propertyGroup("def").func("k").set("k_{mix_id}(T)");\n')
                f.write(f'        model.component("comp1").material("mat_{mix_id}")')
                f.write(f'.propertyGroup("def").func("Cp").set("Cp_{mix_id}(T)");\n')
                
                f.write("    }\n\n")
            
            f.write("    public static void main(String[] args) {\n")
            f.write("        run();\n")
            f.write("    }\n")
            f.write("}\n")