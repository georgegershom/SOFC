# -*- coding: utf-8 -*-
"""
+==============================================================================+
|                                                                              |
|   SOFC Model - Custom Configuration Template                                |
|                                                                              |
|   Purpose   : Template for creating custom model configurations.            |
|               Copy this file and modify parameters as needed.                |
|                                                                              |
|   Usage     : 1. Copy to config_custom.py                                   |
|               2. Modify parameters below                                     |
|               3. Import in main script:                                      |
|                  from config_custom import CustomConfig                      |
|                  config = CustomConfig()                                     |
|                                                                              |
+==============================================================================+
"""

from SOFC_Validation_Model_v3 import SOFCModelConfig, GPA_TO_PA, MPA_TO_PA, KG_M3_TO_TONNE_MM3

class CustomConfig(SOFCModelConfig):
    """
    Custom configuration class extending default SOFCModelConfig.
    
    Override any parameters from the base configuration here.
    All other parameters will use default values.
    """
    
    def __init__(self):
        """Initialize with custom parameters."""
        
        # Call parent constructor first
        super(CustomConfig, self).__init__()
        
        # ──────────────────────────────────────────────────────────────────
        # EXAMPLE CUSTOMIZATIONS
        # ──────────────────────────────────────────────────────────────────
        # Uncomment and modify any section below to customize
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Model Identification
        # └──────────────────────────────────────────────────────────────────
        # self.MODEL_NAME = 'Custom_SOFC_Model'
        # self.JOB_NAME = 'Job-Custom-Run'
        # self.JOB_DESC = 'Custom SOFC simulation with modified parameters'
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Geometry - Thicker Electrolyte Example
        # └──────────────────────────────────────────────────────────────────
        # self.GEOM = {
        #     'L_cell': 12.0,          # Wider cell (was 10.0)
        #     'H_anode': 0.600,        # Thicker anode (was 0.500)
        #     'H_electrolyte': 0.015,  # Thicker electrolyte (was 0.010)
        # }
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Thermal Loading - Different Temperature Range
        # └──────────────────────────────────────────────────────────────────
        # self.TEMP = {
        #     'T_sintering': 1400.0,   # Higher sintering temp (was 1300)
        #     'T_room': 100.0,         # Elevated test temp (was 25)
        # }
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Material Properties - Custom Anode
        # └──────────────────────────────────────────────────────────────────
        # # Example: Different Poisson's ratio
        # self.ANODE_NU = 0.27  # (was 0.29)
        # 
        # # Example: Modified density
        # self.ANODE_DENSITY = 7000.0 * KG_M3_TO_TONNE_MM3  # (was 6870)
        # 
        # # Example: Custom elastic modulus table
        # self.ANODE_E_TABLE = [
        #     (60.0 * GPA_TO_PA, self.ANODE_NU, 25.0),
        #     (58.0 * GPA_TO_PA, self.ANODE_NU, 200.0),
        #     (55.0 * GPA_TO_PA, self.ANODE_NU, 400.0),
        #     # ... add more temperature points
        # ]
        # 
        # # Example: Custom CTE table
        # self.ANODE_CTE_TABLE = [
        #     (12.0e-6, 25.0),
        #     (12.5e-6, 500.0),
        #     (13.0e-6, 1000.0),
        #     (13.5e-6, 1400.0),
        # ]
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Material Properties - Custom Electrolyte
        # └──────────────────────────────────────────────────────────────────
        # # Example: 3YSZ instead of 8YSZ (different properties)
        # self.MAT_NAME_ELEC = 'YSZ3_Electrolyte'
        # 
        # self.ELEC_E_TABLE = [
        #     (220.0e3 * MPA_TO_PA, 0.30, 25.0),   # Stiffer at room temp
        #     (215.0e3 * MPA_TO_PA, 0.30, 200.0),
        #     (210.0e3 * MPA_TO_PA, 0.30, 400.0),
        #     # ... more points
        # ]
        # 
        # self.ELEC_CTE_TABLE = [
        #     (10.0e-6, 25.0),    # Different CTE
        #     (10.5e-6, 500.0),
        #     (11.0e-6, 1000.0),
        # ]
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Mesh Control - Finer Mesh
        # └──────────────────────────────────────────────────────────────────
        # self.MESH = {
        #     'elem_code': CPE4R,
        #     'elem_code_tri': CPE3,
        #     'global_size': 0.10,       # Finer mesh (was 0.20)
        #     'anode_bias': 4.0,         # Stronger bias (was 3.0)
        #     'elec_divisions': 8,       # More elements (was 5)
        #     'deviation': 0.03,         # Tighter tolerance (was 0.05)
        #     'min_size_factor': 0.03,
        #     'aspect_ratio_max': 8.0,
        # }
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Solver Options - More Conservative
        # └──────────────────────────────────────────────────────────────────
        # self.SOLVER = {
        #     'nlgeom': ON,
        #     'init_inc': 0.02,          # Smaller initial step (was 0.05)
        #     'min_inc': 1e-12,          # Smaller minimum (was 1e-10)
        #     'max_inc': 0.10,           # Smaller maximum (was 0.20)
        #     'max_num_inc': 2000,       # More increments (was 1000)
        #     'stabilize': True,         # Enable stabilization (was False)
        #     'stabilize_mag': 1e-4,     # Lower magnitude (was 2e-4)
        #     'use_solver_default': True,
        # }
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Output Requests - Extended
        # └──────────────────────────────────────────────────────────────────
        # # Add strain energy and other outputs
        # self.FIELD_OUTPUTS = ('S', 'E', 'LE', 'PE', 'U', 'V', 'A', 
        #                       'RF', 'CF', 'NT', 'TEMP', 'COORD', 
        #                       'SENER', 'ENER', 'PEEQ')
        # 
        # self.HISTORY_OUTPUTS = ('S11', 'S22', 'S12', 'S33', 
        #                         'E11', 'E22', 'E12', 'E33',
        #                         'U1', 'U2', 'TEMP', 'SENER')
        
        # ┌──────────────────────────────────────────────────────────────────
        # │ Job Control - High-Performance Computing
        # └──────────────────────────────────────────────────────────────────
        # self.JOB = {
        #     'memory_pct': 95,          # More memory (was 90)
        #     'num_cpus': 16,            # More cores (was 4)
        #     'num_domains': 16,         # Match CPUs (was 4)
        #     'precision': SINGLE,
        #     'echo_print': OFF,
        #     'model_print': OFF,
        #     'contact_print': OFF,
        #     'history_print': OFF,
        # }


# ============================================================================
#  EXAMPLE USAGE IN MAIN SCRIPT
# ============================================================================

"""
To use this custom configuration:

1. Save as config_custom.py

2. Modify SOFC_Validation_Model_v3.py main() function:

    # Replace:
    config = SOFCModelConfig()
    
    # With:
    from config_custom import CustomConfig
    config = CustomConfig()

3. Run normally:
    abaqus cae noGUI=SOFC_Validation_Model_v3.py

OR create a separate runner script:

```python
# run_custom.py
import sys
from SOFC_Validation_Model_v3 import SOFCModelBuilder, main
from config_custom import CustomConfig

# Override config creation
original_main = main

def custom_main():
    # ... same as original main() but use CustomConfig
    config = CustomConfig()
    config.validate()
    config.print_summary()
    
    builder = SOFCModelBuilder(config)
    model = builder.build_complete_model()
    
    return 0

if __name__ == '__main__':
    exit_code = custom_main()
    sys.exit(exit_code)
```

Then run:
    abaqus cae noGUI=run_custom.py
"""


# ============================================================================
#  PREDEFINED CONFIGURATION VARIANTS
# ============================================================================

class ThinElectrolyteConfig(SOFCModelConfig):
    """Configuration for ultra-thin electrolyte (5 μm) study."""
    
    def __init__(self):
        super(ThinElectrolyteConfig, self).__init__()
        
        self.MODEL_NAME = 'Thin_Electrolyte_Study'
        self.JOB_NAME = 'Job-ThinElec-5um'
        
        self.GEOM['H_electrolyte'] = 0.005  # 5 μm
        self.MESH['elec_divisions'] = 3     # Fewer elements for thin layer
        self.MESH['global_size'] = 0.15     # Adjust mesh accordingly


class HighTemperatureConfig(SOFCModelConfig):
    """Configuration for high-temperature operating conditions."""
    
    def __init__(self):
        super(HighTemperatureConfig, self).__init__()
        
        self.MODEL_NAME = 'High_Temperature_Operation'
        self.JOB_NAME = 'Job-HighTemp-800C'
        
        self.TEMP['T_room'] = 800.0  # Operating temperature
        
        # May need to add more temperature points to material tables
        # if they don't cover this range


class CoarseMeshConfig(SOFCModelConfig):
    """Configuration for quick test runs with coarse mesh."""
    
    def __init__(self):
        super(CoarseMeshConfig, self).__init__()
        
        self.MODEL_NAME = 'Quick_Test_Coarse'
        self.JOB_NAME = 'Job-Test-Coarse'
        
        self.MESH['global_size'] = 0.40     # Very coarse
        self.MESH['elec_divisions'] = 2     # Minimum
        self.SOLVER['max_num_inc'] = 200    # Fewer increments


class FineMeshConfig(SOFCModelConfig):
    """Configuration for publication-quality results with fine mesh."""
    
    def __init__(self):
        super(FineMeshConfig, self).__init__()
        
        self.MODEL_NAME = 'Publication_Fine_Mesh'
        self.JOB_NAME = 'Job-Publication-Fine'
        
        self.MESH['global_size'] = 0.08     # Very fine
        self.MESH['elec_divisions'] = 10    # Many elements
        self.MESH['anode_bias'] = 4.0       # Strong refinement
        
        self.SOLVER['init_inc'] = 0.02      # Conservative
        self.SOLVER['max_num_inc'] = 3000   # Allow many steps
        
        self.JOB['num_cpus'] = 8            # Use more cores
        self.JOB['num_domains'] = 8


# ============================================================================
#  CONFIGURATION VALIDATION UTILITY
# ============================================================================

def compare_configs(config1, config2, name1='Config 1', name2='Config 2'):
    """
    Compare two configuration objects and print differences.
    
    Parameters
    ----------
    config1 : SOFCModelConfig
        First configuration.
    config2 : SOFCModelConfig
        Second configuration.
    name1 : str, optional
        Name label for first config.
    name2 : str, optional
        Name label for second config.
    """
    print('\n' + '=' * 78)
    print(f'Configuration Comparison: {name1} vs {name2}')
    print('=' * 78 + '\n')
    
    # Compare attributes
    attrs_to_compare = [
        'MODEL_NAME', 'JOB_NAME', 'GEOM', 'TEMP', 'MESH', 'SOLVER', 'JOB'
    ]
    
    differences = []
    
    for attr in attrs_to_compare:
        val1 = getattr(config1, attr, None)
        val2 = getattr(config2, attr, None)
        
        if val1 != val2:
            differences.append((attr, val1, val2))
    
    if differences:
        print(f"Found {len(differences)} difference(s):\n")
        for attr, val1, val2 in differences:
            print(f"  {attr}:")
            print(f"    {name1}: {val1}")
            print(f"    {name2}: {val2}")
            print()
    else:
        print("Configurations are identical.")
    
    print('=' * 78 + '\n')


# ============================================================================
#  TESTING
# ============================================================================

if __name__ == '__main__':
    """Test configuration loading."""
    
    print("Testing configuration variants...\n")
    
    # Create instances
    default_config = SOFCModelConfig()
    custom_config = CustomConfig()
    thin_config = ThinElectrolyteConfig()
    fine_config = FineMeshConfig()
    
    # Validate all
    configs = [
        ('Default', default_config),
        ('Custom', custom_config),
        ('Thin Electrolyte', thin_config),
        ('Fine Mesh', fine_config),
    ]
    
    for name, config in configs:
        try:
            config.validate()
            print(f"✓ {name:20s}: Valid")
        except Exception as e:
            print(f"✗ {name:20s}: INVALID - {e}")
    
    print("\nConfiguration comparison:")
    compare_configs(default_config, thin_config, 'Default', 'Thin Electrolyte')

# ============================================================================
#  END OF TEMPLATE
# ============================================================================
