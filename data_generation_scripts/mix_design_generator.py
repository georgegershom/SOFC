#!/usr/bin/env python3
"""
Mix Design Data Generator for Fire-Resistant Rubberized Concrete
Generates comprehensive mix design matrix and fresh properties data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import json
from datetime import datetime
import os

class MixDesignGenerator:
    def __init__(self, seed=42):
        """Initialize the generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.mix_designs = {}
        self.fresh_properties = {}
        
    def generate_mix_design_matrix(self):
        """Generate comprehensive mix design matrix"""
        print("Generating mix design matrix...")
        
        # Define parameter ranges
        rubber_content_fine = [0, 5, 10, 15, 20, 25, 30]  # % by volume
        rubber_content_coarse = [0, 5, 10, 15, 20]  # % by volume
        cement_content = [300, 400, 500]  # kg/m³
        w_c_ratio = [0.35, 0.40, 0.45, 0.50, 0.55]
        fiber_content = [0, 0.5, 1.0, 1.5, 2.0]  # kg/m³
        fire_retardant = [0, 2, 4, 6, 8]  # % by weight of cement
        
        # Rubber size distributions
        rubber_sizes = {
            'fine': [0.075, 1, 2, 4],  # mm
            'coarse': [4, 8, 12]  # mm
        }
        
        mix_id = 1
        
        # Generate control mixes
        for cement in cement_content:
            for wc in w_c_ratio:
                mix_design = self._create_mix_design(
                    mix_id, cement, wc, 0, 0, 0, 0, 'control'
                )
                self.mix_designs[mix_id] = mix_design
                mix_id += 1
        
        # Generate rubber content variation mixes
        for cement in cement_content:
            for wc in wc_ratio:
                for rubber_fine in rubber_content_fine[1:]:  # Skip 0%
                    mix_design = self._create_mix_design(
                        mix_id, cement, wc, rubber_fine, 0, 0, 0, 'rubber_fine'
                    )
                    self.mix_designs[mix_id] = mix_design
                    mix_id += 1
        
        # Generate combined rubber replacement mixes
        for cement in cement_content:
            for wc in wc_ratio:
                for rubber_fine in [10, 15, 20]:
                    for rubber_coarse in [5, 10]:
                        mix_design = self._create_mix_design(
                            mix_id, cement, wc, rubber_fine, rubber_coarse, 0, 0, 'rubber_combined'
                        )
                        self.mix_designs[mix_id] = mix_design
                        mix_id += 1
        
        # Generate fire-resistant mixes
        for cement in cement_content:
            for wc in wc_ratio:
                for rubber_fine in [0, 10, 15, 20]:
                    for fiber in fiber_content[1:]:  # Skip 0%
                        for retardant in fire_retardant[1:]:  # Skip 0%
                            mix_design = self._create_mix_design(
                                mix_id, cement, wc, rubber_fine, 0, fiber, retardant, 'fire_resistant'
                            )
                            self.mix_designs[mix_id] = mix_design
                            mix_id += 1
        
        print(f"Generated {len(self.mix_designs)} mix designs")
        return self.mix_designs
    
    def _create_mix_design(self, mix_id, cement, wc_ratio, rubber_fine, rubber_coarse, 
                          fiber_content, fire_retardant, mix_type):
        """Create individual mix design"""
        
        # Calculate water content
        water = cement * wc_ratio
        
        # Calculate aggregate volumes (assuming 1 m³ total volume)
        # Typical concrete: 60% aggregates, 25% cement paste, 15% air
        total_aggregate_volume = 0.60  # m³
        
        # Calculate rubber volumes
        rubber_fine_volume = total_aggregate_volume * rubber_fine / 100
        rubber_coarse_volume = total_aggregate_volume * rubber_coarse / 100
        
        # Calculate remaining aggregate volumes
        remaining_aggregate_volume = total_aggregate_volume - rubber_fine_volume - rubber_coarse_volume
        
        # Split remaining between fine and coarse aggregates (60% fine, 40% coarse)
        fine_aggregate_volume = remaining_aggregate_volume * 0.6
        coarse_aggregate_volume = remaining_aggregate_volume * 0.4
        
        # Convert volumes to masses (using typical densities)
        cement_density = 3150  # kg/m³
        water_density = 1000  # kg/m³
        fine_aggregate_density = 2650  # kg/m³
        coarse_aggregate_density = 2680  # kg/m³
        rubber_density = 1150  # kg/m³
        
        fine_aggregate_mass = fine_aggregate_volume * fine_aggregate_density
        coarse_aggregate_mass = coarse_aggregate_volume * coarse_aggregate_density
        rubber_fine_mass = rubber_fine_volume * rubber_density
        rubber_coarse_mass = rubber_coarse_volume * rubber_density
        
        # Calculate admixture dosages
        superplasticizer_dosage = self._calculate_superplasticizer_dosage(cement, wc_ratio, rubber_fine)
        air_entraining_dosage = 0.05  # % by weight of cement
        
        # Calculate fire retardant mass
        fire_retardant_mass = cement * fire_retardant / 100
        
        mix_design = {
            'mix_id': mix_id,
            'mix_type': mix_type,
            'cement_content': cement,
            'water_content': water,
            'w_c_ratio': wc_ratio,
            'fine_aggregate_mass': fine_aggregate_mass,
            'coarse_aggregate_mass': coarse_aggregate_mass,
            'rubber_fine_content': rubber_fine,
            'rubber_coarse_content': rubber_coarse,
            'rubber_fine_mass': rubber_fine_mass,
            'rubber_coarse_mass': rubber_coarse_mass,
            'fiber_content': fiber_content,
            'fire_retardant_content': fire_retardant,
            'fire_retardant_mass': fire_retardant_mass,
            'superplasticizer_dosage': superplasticizer_dosage,
            'air_entraining_dosage': air_entraining_dosage,
            'total_volume': 1.0,  # m³
            'total_mass': (cement + water + fine_aggregate_mass + coarse_aggregate_mass + 
                          rubber_fine_mass + rubber_coarse_mass + fire_retardant_mass)
        }
        
        return mix_design
    
    def _calculate_superplasticizer_dosage(self, cement, wc_ratio, rubber_content):
        """Calculate superplasticizer dosage based on mix parameters"""
        # Base dosage for normal concrete
        base_dosage = 0.8  # % by weight of cement
        
        # Increase dosage for lower w/c ratio
        wc_factor = max(0, (0.55 - wc_ratio) * 2)
        
        # Increase dosage for rubber content
        rubber_factor = rubber_content * 0.05
        
        # Add some random variation
        variation = np.random.normal(0, 0.1)
        
        total_dosage = base_dosage + wc_factor + rubber_factor + variation
        
        return max(0.2, min(2.0, total_dosage))  # Clamp between 0.2% and 2.0%
    
    def generate_fresh_properties(self):
        """Generate fresh concrete properties for all mixes"""
        print("Generating fresh properties data...")
        
        for mix_id, mix_design in self.mix_designs.items():
            fresh_props = self._calculate_fresh_properties(mix_design)
            self.fresh_properties[mix_id] = fresh_props
        
        print(f"Generated fresh properties for {len(self.fresh_properties)} mixes")
        return self.fresh_properties
    
    def _calculate_fresh_properties(self, mix_design):
        """Calculate fresh properties for a specific mix"""
        
        # Base properties
        base_slump = 150  # mm
        base_flow = 500  # mm
        base_air_content = 4.5  # %
        base_unit_weight = 2400  # kg/m³
        base_temperature = 20  # °C
        
        # Factors affecting workability
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 200  # Higher w/c = higher slump
        rubber_factor = -mix_design['rubber_fine_content'] * 3  # Rubber reduces workability
        superplasticizer_factor = mix_design['superplasticizer_dosage'] * 50  # SP increases workability
        fiber_factor = -mix_design['fiber_content'] * 10  # Fibers reduce workability
        
        # Calculate slump
        slump = base_slump + wc_factor + rubber_factor + superplasticizer_factor + fiber_factor
        slump += np.random.normal(0, 15)  # Add random variation
        slump = max(50, min(250, slump))  # Clamp between 50-250 mm
        
        # Calculate flow
        flow = base_flow + wc_factor * 1.5 + rubber_factor * 2 + superplasticizer_factor * 1.2 + fiber_factor * 1.5
        flow += np.random.normal(0, 25)
        flow = max(300, min(700, flow))
        
        # Calculate air content
        air_content = base_air_content + mix_design['air_entraining_dosage'] * 20
        air_content += np.random.normal(0, 0.5)
        air_content = max(2.0, min(8.0, air_content))
        
        # Calculate unit weight
        rubber_density_factor = (mix_design['rubber_fine_content'] + mix_design['rubber_coarse_content']) * 0.5
        unit_weight = base_unit_weight - rubber_density_factor * 10
        unit_weight += np.random.normal(0, 25)
        unit_weight = max(2000, min(2600, unit_weight))
        
        # Calculate temperature
        temperature = base_temperature + np.random.normal(0, 2)
        temperature = max(18, min(25, temperature))
        
        # Calculate rheological properties
        yield_stress = self._calculate_yield_stress(mix_design)
        plastic_viscosity = self._calculate_plastic_viscosity(mix_design)
        thixotropy_index = self._calculate_thixotropy_index(mix_design)
        
        fresh_properties = {
            'slump': slump,
            'flow_spread': flow,
            'air_content_pressure': air_content,
            'air_content_gravimetric': air_content + np.random.normal(0, 0.3),
            'unit_weight': unit_weight,
            'temperature': temperature,
            'yield_stress': yield_stress,
            'plastic_viscosity': plastic_viscosity,
            'thixotropy_index': thixotropy_index,
            'v_funnel_time': self._calculate_v_funnel_time(mix_design),
            'l_box_height_ratio': self._calculate_l_box_ratio(mix_design)
        }
        
        return fresh_properties
    
    def _calculate_yield_stress(self, mix_design):
        """Calculate yield stress based on mix parameters"""
        base_yield_stress = 50  # Pa
        
        # Factors affecting yield stress
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 200
        rubber_factor = mix_design['rubber_fine_content'] * 5
        superplasticizer_factor = -mix_design['superplasticizer_dosage'] * 30
        fiber_factor = mix_design['fiber_content'] * 15
        
        yield_stress = base_yield_stress + wc_factor + rubber_factor + superplasticizer_factor + fiber_factor
        yield_stress += np.random.normal(0, 10)
        
        return max(10, min(200, yield_stress))
    
    def _calculate_plastic_viscosity(self, mix_design):
        """Calculate plastic viscosity based on mix parameters"""
        base_viscosity = 2.5  # Pa·s
        
        # Factors affecting viscosity
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 3
        rubber_factor = mix_design['rubber_fine_content'] * 0.3
        superplasticizer_factor = -mix_design['superplasticizer_dosage'] * 1.5
        fiber_factor = mix_design['fiber_content'] * 0.8
        
        viscosity = base_viscosity + wc_factor + rubber_factor + superplasticizer_factor + fiber_factor
        viscosity += np.random.normal(0, 0.2)
        
        return max(0.5, min(8.0, viscosity))
    
    def _calculate_thixotropy_index(self, mix_design):
        """Calculate thixotropy index"""
        base_thixotropy = 0.3
        
        # Factors affecting thixotropy
        rubber_factor = mix_design['rubber_fine_content'] * 0.02
        fiber_factor = mix_design['fiber_content'] * 0.05
        superplasticizer_factor = -mix_design['superplasticizer_dosage'] * 0.1
        
        thixotropy = base_thixotropy + rubber_factor + fiber_factor + superplasticizer_factor
        thixotropy += np.random.normal(0, 0.05)
        
        return max(0.1, min(1.0, thixotropy))
    
    def _calculate_v_funnel_time(self, mix_design):
        """Calculate V-funnel time"""
        base_time = 8  # seconds
        
        # Factors affecting V-funnel time
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 10
        rubber_factor = mix_design['rubber_fine_content'] * 0.5
        superplasticizer_factor = -mix_design['superplasticizer_dosage'] * 3
        fiber_factor = mix_design['fiber_content'] * 2
        
        v_funnel_time = base_time + wc_factor + rubber_factor + superplasticizer_factor + fiber_factor
        v_funnel_time += np.random.normal(0, 1)
        
        return max(3, min(20, v_funnel_time))
    
    def _calculate_l_box_ratio(self, mix_design):
        """Calculate L-box height ratio"""
        base_ratio = 0.85
        
        # Factors affecting L-box ratio
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 0.3
        rubber_factor = -mix_design['rubber_fine_content'] * 0.01
        superplasticizer_factor = mix_design['superplasticizer_dosage'] * 0.1
        fiber_factor = -mix_design['fiber_content'] * 0.05
        
        l_box_ratio = base_ratio + wc_factor + rubber_factor + superplasticizer_factor + fiber_factor
        l_box_ratio += np.random.normal(0, 0.05)
        
        return max(0.3, min(1.0, l_box_ratio))
    
    def generate_mix_design_dataframe(self):
        """Convert mix designs to pandas DataFrame"""
        mix_data = []
        for mix_id, mix_design in self.mix_designs.items():
            row = mix_design.copy()
            if mix_id in self.fresh_properties:
                row.update(self.fresh_properties[mix_id])
            mix_data.append(row)
        
        return pd.DataFrame(mix_data)
    
    def save_data(self, filename='mix_design_data.json'):
        """Save generated data to JSON file"""
        data_to_save = {
            'mix_designs': self.mix_designs,
            'fresh_properties': self.fresh_properties,
            'generation_date': datetime.now().isoformat(),
            'total_mixes': len(self.mix_designs)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_numpy(data_to_save)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Mix design data saved to {filename}")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the dataset"""
        df = self.generate_mix_design_dataframe()
        
        summary = {
            'total_mixes': len(df),
            'mix_types': df['mix_type'].value_counts().to_dict(),
            'rubber_content_range': {
                'fine': [df['rubber_fine_content'].min(), df['rubber_fine_content'].max()],
                'coarse': [df['rubber_coarse_content'].min(), df['rubber_coarse_content'].max()]
            },
            'cement_content_range': [df['cement_content'].min(), df['cement_content'].max()],
            'w_c_ratio_range': [df['w_c_ratio'].min(), df['w_c_ratio'].max()],
            'fresh_properties_stats': {
                'slump': [df['slump'].min(), df['slump'].max(), df['slump'].mean()],
                'flow_spread': [df['flow_spread'].min(), df['flow_spread'].max(), df['flow_spread'].mean()],
                'air_content': [df['air_content_pressure'].min(), df['air_content_pressure'].max(), df['air_content_pressure'].mean()],
                'unit_weight': [df['unit_weight'].min(), df['unit_weight'].max(), df['unit_weight'].mean()]
            }
        }
        
        return summary

def main():
    """Main function to generate mix design data"""
    print("Starting Mix Design Data Generation...")
    print("=" * 50)
    
    # Initialize generator
    generator = MixDesignGenerator(seed=42)
    
    # Generate mix design matrix
    generator.generate_mix_design_matrix()
    
    # Generate fresh properties
    generator.generate_fresh_properties()
    
    # Save data
    generator.save_data('mix_design_data.json')
    
    # Generate summary statistics
    summary = generator.generate_summary_statistics()
    print("\nSummary Statistics:")
    print(json.dumps(summary, indent=2))
    
    print("\nMix design data generation completed successfully!")

if __name__ == "__main__":
    main()