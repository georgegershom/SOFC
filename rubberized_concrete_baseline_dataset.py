#!/usr/bin/env python3
"""
Rubberized Concrete Baseline Dataset Generator
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements
Utilizing High-Performance Rubberized Concrete

This script generates a comprehensive baseline dataset covering:
- Material characterization and mixture design
- Rubber aggregate properties
- Fresh state properties
- Ambient temperature mechanical and physical properties

Author: AI Research Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class RubberizedConcreteDatasetGenerator:
    def __init__(self, seed=42):
        """Initialize the dataset generator with reproducible random seed."""
        np.random.seed(seed)
        random.seed(seed)
        self.dataset_metadata = {
            "project_title": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
            "dataset_type": "Baseline Material Characterization",
            "generation_date": datetime.now().isoformat(),
            "rubber_replacement_levels": [0, 5, 10, 15],  # % by volume of fine aggregate
            "specimen_count_per_mix": 12,  # For statistical significance
            "curing_periods": [7, 28],  # days
            "test_standards": {
                "compressive_strength": "ASTM C39",
                "tensile_splitting": "ASTM C496",
                "modulus_elasticity": "ASTM C469",
                "slump_flow": "ASTM C143",
                "air_content": "ASTM C231",
                "density": "ASTM C642",
                "porosity": "ASTM D4404 (MIP)"
            }
        }
    
    def generate_mixture_proportions(self):
        """Generate concrete mixture proportions for control and rubberized mixes."""
        print("Generating concrete mixture proportions...")
        
        # Base mix design (control mix - 0% rubber)
        base_mix = {
            "cement_type": "Type I/II Portland Cement",
            "cement_grade": "42.5N",
            "cement_content": 400,  # kg/m³
            "water_cement_ratio": 0.45,
            "coarse_aggregate": {
                "type": "Crushed limestone",
                "max_size": 20,  # mm
                "ssd_specific_gravity": 2.65,
                "grading": "20-10mm: 60%, 10-5mm: 40%",
                "content": 1050  # kg/m³
            },
            "fine_aggregate": {
                "type": "Natural sand",
                "ssd_specific_gravity": 2.60,
                "fineness_modulus": 2.8,
                "content": 750  # kg/m³
            },
            "water": {
                "content": 180,  # kg/m³
                "type": "Potable water"
            },
            "superplasticizer": {
                "type": "Polycarboxylate ether (PCE)",
                "dosage": 1.2,  # % by weight of cement
                "content": 4.8  # kg/m³
            },
            "air_entrainer": {
                "type": "Synthetic air-entraining admixture",
                "dosage": 0.05,  # % by weight of cement
                "content": 0.2  # kg/m³
            }
        }
        
        # Generate rubberized mixes
        mixes = {}
        rubber_replacement_levels = [0, 5, 10, 15]  # % by volume of fine aggregate
        
        for replacement_level in rubber_replacement_levels:
            mix_id = f"Mix_{replacement_level}pct_Rubber"
            mix = base_mix.copy()
            
            if replacement_level > 0:
                # Calculate rubber content (by volume of fine aggregate)
                fine_agg_volume = 750 / 2.60  # m³/m³
                rubber_volume = fine_agg_volume * (replacement_level / 100)
                rubber_density = 1.15  # g/cm³ (typical for crumb rubber)
                rubber_content = rubber_volume * rubber_density * 1000  # kg/m³
                
                # Adjust fine aggregate content
                fine_agg_reduction = rubber_volume * 2.60 * 1000  # kg/m³
                new_fine_agg_content = max(0, 750 - fine_agg_reduction)
                
                mix["rubber_aggregate"] = {
                    "type": "Crumb rubber from truck tires",
                    "particle_size": "1-4 mm",
                    "specific_gravity": 1.15,
                    "content": rubber_content,
                    "replacement_level": replacement_level
                }
                mix["fine_aggregate"]["content"] = new_fine_agg_content
                mix["mix_id"] = mix_id
                mix["total_volume"] = 1.0  # m³
            else:
                mix["mix_id"] = mix_id
                mix["total_volume"] = 1.0  # m³
            
            mixes[mix_id] = mix
        
        return mixes
    
    def generate_rubber_characterization(self):
        """Generate comprehensive rubber aggregate characterization data."""
        print("Generating rubber aggregate characterization...")
        
        # Base rubber properties with realistic variation
        base_properties = {
            "source": "Truck tire crumb rubber (recycled)",
            "manufacturer": "EcoRubber Solutions Inc.",
            "particle_size_distribution": {
                "1-2mm": 35,  # % by weight
                "2-3mm": 45,
                "3-4mm": 20
            },
            "physical_properties": {
                "specific_gravity": 1.15,
                "water_absorption": 2.5,  # % after 24h
                "hardness_shore_a": 65,
                "bulk_density": 0.68,  # g/cm³
                "void_content": 41  # %
            },
            "chemical_composition": {
                "natural_rubber": 45,  # %
                "synthetic_rubber": 35,
                "carbon_black": 15,
                "zinc_oxide": 2,
                "sulfur": 1.5,
                "other_additives": 1.5
            },
            "thermal_properties": {
                "glass_transition_temp": -60,  # °C
                "decomposition_start": 280,  # °C
                "peak_decomposition": 420,  # °C
                "residual_mass_600C": 35  # %
            },
            "pre_treatment": {
                "method": "NaOH washing + plasma treatment",
                "naoh_concentration": "2M",
                "treatment_time": 30,  # minutes
                "plasma_power": 100,  # W
                "treatment_duration": 5  # minutes
            }
        }
        
        # Add experimental variation
        rubber_data = []
        for i in range(5):  # 5 batches of rubber
            batch = base_properties.copy()
            batch["batch_id"] = f"RB_{i+1:02d}"
            
            # Add realistic variation to properties
            variation_factor = np.random.normal(1.0, 0.05)  # 5% standard deviation
            batch["physical_properties"]["specific_gravity"] *= variation_factor
            batch["physical_properties"]["hardness_shore_a"] = int(
                base_properties["physical_properties"]["hardness_shore_a"] * variation_factor
            )
            
            rubber_data.append(batch)
        
        return rubber_data
    
    def generate_fresh_state_properties(self, mixes):
        """Generate fresh state properties for all concrete mixes."""
        print("Generating fresh state properties...")
        
        fresh_properties = []
        
        for mix_id, mix in mixes.items():
            replacement_level = int(mix_id.split('_')[1].replace('pct', ''))
            
            # Base properties for control mix
            base_slump = 180  # mm
            base_air_content = 4.5  # %
            base_density = 2350  # kg/m³
            
            # Rubber effect on workability (decreases with higher rubber content)
            workability_factor = 1 - (replacement_level * 0.08)  # 8% reduction per 5% rubber
            
            # Generate multiple test results for statistical analysis
            for test_id in range(3):  # 3 tests per mix
                # Add realistic experimental variation
                slump_variation = np.random.normal(1.0, 0.08)
                air_variation = np.random.normal(1.0, 0.10)
                density_variation = np.random.normal(1.0, 0.02)
                
                test_result = {
                    "mix_id": mix_id,
                    "rubber_replacement": replacement_level,
                    "test_id": test_id + 1,
                    "slump_flow": max(50, base_slump * workability_factor * slump_variation),
                    "air_content": max(2.0, base_air_content * air_variation),
                    "fresh_density": base_density * density_variation,
                    "workability_rating": self._assess_workability(base_slump * workability_factor * slump_variation),
                    "test_temperature": 23 + np.random.normal(0, 1),  # °C
                    "test_humidity": 50 + np.random.normal(0, 5)  # %
                }
                
                fresh_properties.append(test_result)
        
        return fresh_properties
    
    def _assess_workability(self, slump_value):
        """Assess workability based on slump value."""
        if slump_value >= 200:
            return "Excellent"
        elif slump_value >= 150:
            return "Good"
        elif slump_value >= 100:
            return "Fair"
        elif slump_value >= 50:
            return "Poor"
        else:
            return "Very Poor"
    
    def generate_mechanical_properties(self, mixes):
        """Generate ambient temperature mechanical properties."""
        print("Generating mechanical properties...")
        
        mechanical_data = []
        
        for mix_id, mix in mixes.items():
            replacement_level = int(mix_id.split('_')[1].replace('pct', ''))
            
            # Base strength properties (control mix)
            base_fc_7d = 35  # MPa
            base_fc_28d = 45  # MPa
            base_ft_28d = 3.5  # MPa
            base_E_28d = 35  # GPa
            
            # Rubber effect on strength (decreases with higher rubber content)
            strength_factor = 1 - (replacement_level * 0.12)  # 12% reduction per 5% rubber
            modulus_factor = 1 - (replacement_level * 0.15)  # 15% reduction per 5% rubber
            
            # Generate test results for each specimen
            for specimen_id in range(12):  # 12 specimens per mix
                # Add realistic experimental variation
                fc_7d_variation = np.random.normal(1.0, 0.08)
                fc_28d_variation = np.random.normal(1.0, 0.06)
                ft_variation = np.random.normal(1.0, 0.10)
                E_variation = np.random.normal(1.0, 0.07)
                
                specimen_data = {
                    "mix_id": mix_id,
                    "rubber_replacement": replacement_level,
                    "specimen_id": specimen_id + 1,
                    "age_days": 7,
                    "compressive_strength": max(10, base_fc_7d * strength_factor * fc_7d_variation),
                    "test_standard": "ASTM C39",
                    "specimen_type": "100mm cube"
                }
                mechanical_data.append(specimen_data.copy())
                
                specimen_data["age_days"] = 28
                specimen_data["compressive_strength"] = max(15, base_fc_28d * strength_factor * fc_28d_variation)
                specimen_data["tensile_splitting_strength"] = max(1.0, base_ft_28d * strength_factor * ft_variation)
                specimen_data["modulus_elasticity"] = max(10, base_E_28d * modulus_factor * E_variation)
                mechanical_data.append(specimen_data)
        
        return mechanical_data
    
    def generate_physical_properties(self, mixes):
        """Generate physical properties including density, porosity, and UPV."""
        print("Generating physical properties...")
        
        physical_data = []
        
        for mix_id, mix in mixes.items():
            replacement_level = int(mix_id.split('_')[1].replace('pct', ''))
            
            # Base physical properties
            base_oven_dry_density = 2350  # kg/m³
            base_ssd_density = 2400  # kg/m³
            base_porosity = 12  # %
            base_upv = 4500  # m/s
            
            # Rubber effect on physical properties
            density_factor = 1 - (replacement_level * 0.03)  # 3% reduction per 5% rubber
            porosity_factor = 1 + (replacement_level * 0.15)  # 15% increase per 5% rubber
            upv_factor = 1 - (replacement_level * 0.08)  # 8% reduction per 5% rubber
            
            # Generate test results
            for specimen_id in range(6):  # 6 specimens per mix for physical tests
                variation = np.random.normal(1.0, 0.03)
                
                specimen_data = {
                    "mix_id": mix_id,
                    "rubber_replacement": replacement_level,
                    "specimen_id": specimen_id + 1,
                    "oven_dry_density": base_oven_dry_density * density_factor * variation,
                    "ssd_density": base_ssd_density * density_factor * variation,
                    "porosity": base_porosity * porosity_factor * variation,
                    "ultrasonic_pulse_velocity": base_upv * upv_factor * variation,
                    "water_absorption": np.random.normal(3.5, 0.5),  # %
                    "void_content": np.random.normal(8, 1)  # %
                }
                
                physical_data.append(specimen_data)
        
        return physical_data
    
    def generate_porosity_distribution(self, mixes):
        """Generate detailed pore size distribution data using MIP simulation."""
        print("Generating pore size distribution data...")
        
        porosity_data = []
        
        for mix_id, mix in mixes.items():
            replacement_level = int(mix_id.split('_')[1].replace('pct', ''))
            
            # Base pore size distribution (control mix)
            pore_sizes = [0.01, 0.1, 1, 10, 100, 1000]  # μm
            base_volumes = [2, 8, 15, 20, 10, 5]  # % of total porosity
            
            # Rubber effect on pore structure
            if replacement_level > 0:
                # Rubber creates additional large pores
                rubber_pore_contribution = replacement_level * 0.5  # % per 5% rubber
                base_volumes[-1] += rubber_pore_contribution  # Add to largest pore size
                base_volumes[0] += rubber_pore_contribution * 0.3  # Some effect on small pores
            
            # Generate MIP data
            for specimen_id in range(3):  # 3 specimens per mix for MIP
                variation = np.random.normal(1.0, 0.1)
                
                specimen_porosity = {
                    "mix_id": mix_id,
                    "rubber_replacement": replacement_level,
                    "specimen_id": specimen_id + 1,
                    "total_porosity": sum(base_volumes) * variation,
                    "pore_size_distribution": {
                        "pore_sizes_um": pore_sizes,
                        "volume_percentages": [v * variation for v in base_volumes],
                        "cumulative_volume": np.cumsum([v * variation for v in base_volumes]).tolist()
                    },
                    "critical_pore_diameter": 0.1 * (1 + replacement_level * 0.1),  # μm
                    "permeability_coefficient": 1e-15 * (1 + replacement_level * 0.2)  # m²
                }
                
                porosity_data.append(specimen_porosity)
        
        return porosity_data
    
    def generate_chemical_analysis(self):
        """Generate chemical analysis data for rubber and concrete samples."""
        print("Generating chemical analysis data...")
        
        # TGA data for rubber
        tga_data = {
            "sample_type": "Crumb rubber",
            "temperature_range": "25-800°C",
            "heating_rate": "10°C/min",
            "atmosphere": "Nitrogen",
            "weight_loss_stages": [
                {"temperature_range": "25-200°C", "weight_loss": 2.5, "description": "Moisture and volatiles"},
                {"temperature_range": "200-350°C", "weight_loss": 15.0, "description": "Plasticizer decomposition"},
                {"temperature_range": "350-500°C", "weight_loss": 45.0, "description": "Rubber polymer decomposition"},
                {"temperature_range": "500-800°C", "weight_loss": 2.5, "description": "Carbon black oxidation"},
                {"residual_at_800°C": 35.0, "description": "Ash and carbon black residue"}
            ]
        }
        
        # FTIR data for rubber
        ftir_data = {
            "sample_type": "Crumb rubber",
            "wavenumber_range": "4000-400 cm⁻¹",
            "resolution": "4 cm⁻¹",
            "peaks": [
                {"wavenumber": 2950, "intensity": 0.85, "assignment": "C-H stretching (CH₂, CH₃)"},
                {"wavenumber": 1640, "intensity": 0.45, "assignment": "C=C stretching (natural rubber)"},
                {"wavenumber": 1440, "intensity": 0.60, "assignment": "CH₂ bending"},
                {"wavenumber": 1375, "intensity": 0.40, "assignment": "CH₃ bending"},
                {"wavenumber": 1100, "intensity": 0.25, "assignment": "C-O stretching (additives)"},
                {"wavenumber": 800, "intensity": 0.30, "assignment": "C-H out-of-plane bending"}
            ]
        }
        
        return {"tga": tga_data, "ftir": ftir_data}
    
    def generate_curing_regime(self):
        """Generate detailed curing regime data."""
        print("Generating curing regime data...")
        
        curing_data = {
            "curing_method": "Lime-saturated water curing",
            "curing_temperature": "23 ± 2°C",
            "curing_duration": "28 days",
            "specimen_preparation": {
                "demolding_time": "24 hours after casting",
                "initial_curing": "24 hours in molds at 23°C, 95% RH",
                "water_curing": "27 days in lime-saturated water at 23°C"
            },
            "quality_control": {
                "temperature_monitoring": "Continuous with data logger",
                "ph_monitoring": "Weekly pH checks (maintained at 12.5-13.0)",
                "specimen_handling": "Minimal handling, no drying before testing"
            }
        }
        
        return curing_data
    
    def generate_complete_dataset(self):
        """Generate the complete baseline dataset."""
        print("="*60)
        print("GENERATING RUBBERIZED CONCRETE BASELINE DATASET")
        print("="*60)
        
        # Generate all dataset components
        mixes = self.generate_mixture_proportions()
        rubber_data = self.generate_rubber_characterization()
        fresh_properties = self.generate_fresh_state_properties(mixes)
        mechanical_properties = self.generate_mechanical_properties(mixes)
        physical_properties = self.generate_physical_properties(mixes)
        porosity_data = self.generate_porosity_distribution(mixes)
        chemical_analysis = self.generate_chemical_analysis()
        curing_regime = self.generate_curing_regime()
        
        # Compile complete dataset
        complete_dataset = {
            "metadata": self.dataset_metadata,
            "mixture_proportions": mixes,
            "rubber_characterization": rubber_data,
            "fresh_state_properties": fresh_properties,
            "mechanical_properties": mechanical_properties,
            "physical_properties": physical_properties,
            "porosity_distribution": porosity_data,
            "chemical_analysis": chemical_analysis,
            "curing_regime": curing_regime,
            "statistical_summary": self._generate_statistical_summary(
                fresh_properties, mechanical_properties, physical_properties
            )
        }
        
        print(f"\nDataset generation complete!")
        print(f"Total data points: {len(fresh_properties) + len(mechanical_properties) + len(physical_properties)}")
        print(f"Rubber replacement levels: {self.dataset_metadata['rubber_replacement_levels']}")
        
        return complete_dataset
    
    def _generate_statistical_summary(self, fresh_data, mechanical_data, physical_data):
        """Generate statistical summary of the dataset."""
        summary = {}
        
        # Fresh properties summary
        fresh_df = pd.DataFrame(fresh_data)
        summary["fresh_properties"] = fresh_df.groupby('rubber_replacement').agg({
            'slump_flow': ['mean', 'std', 'min', 'max'],
            'air_content': ['mean', 'std', 'min', 'max'],
            'fresh_density': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Mechanical properties summary
        mech_df = pd.DataFrame(mechanical_data)
        summary["mechanical_properties"] = mech_df.groupby(['rubber_replacement', 'age_days']).agg({
            'compressive_strength': ['mean', 'std', 'min', 'max'],
            'tensile_splitting_strength': ['mean', 'std', 'min', 'max'],
            'modulus_elasticity': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Physical properties summary
        phys_df = pd.DataFrame(physical_data)
        summary["physical_properties"] = phys_df.groupby('rubber_replacement').agg({
            'oven_dry_density': ['mean', 'std', 'min', 'max'],
            'ssd_density': ['mean', 'std', 'min', 'max'],
            'porosity': ['mean', 'std', 'min', 'max'],
            'ultrasonic_pulse_velocity': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        return summary
    
    def save_dataset(self, dataset, output_dir="/workspace"):
        """Save the complete dataset to files."""
        print(f"\nSaving dataset to {output_dir}...")
        
        # Save as JSON
        with open(f"{output_dir}/rubberized_concrete_baseline_dataset.json", 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        # Save individual CSV files for easy analysis
        pd.DataFrame(dataset['fresh_state_properties']).to_csv(
            f"{output_dir}/fresh_state_properties.csv", index=False
        )
        pd.DataFrame(dataset['mechanical_properties']).to_csv(
            f"{output_dir}/mechanical_properties.csv", index=False
        )
        pd.DataFrame(dataset['physical_properties']).to_csv(
            f"{output_dir}/physical_properties.csv", index=False
        )
        
        # Save mixture proportions
        with open(f"{output_dir}/mixture_proportions.json", 'w') as f:
            json.dump(dataset['mixture_proportions'], f, indent=2, default=str)
        
        # Save rubber characterization
        with open(f"{output_dir}/rubber_characterization.json", 'w') as f:
            json.dump(dataset['rubber_characterization'], f, indent=2, default=str)
        
        print("Dataset saved successfully!")
        print(f"Files created:")
        print(f"- rubberized_concrete_baseline_dataset.json (complete dataset)")
        print(f"- fresh_state_properties.csv")
        print(f"- mechanical_properties.csv") 
        print(f"- physical_properties.csv")
        print(f"- mixture_proportions.json")
        print(f"- rubber_characterization.json")

def main():
    """Main function to generate and save the dataset."""
    generator = RubberizedConcreteDatasetGenerator(seed=42)
    dataset = generator.generate_complete_dataset()
    generator.save_dataset(dataset)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Project: {dataset['metadata']['project_title']}")
    print(f"Rubber replacement levels: {dataset['metadata']['rubber_replacement_levels']}%")
    print(f"Specimens per mix: {dataset['metadata']['specimen_count_per_mix']}")
    print(f"Total fresh state tests: {len(dataset['fresh_state_properties'])}")
    print(f"Total mechanical tests: {len(dataset['mechanical_properties'])}")
    print(f"Total physical tests: {len(dataset['physical_properties'])}")
    print(f"Rubber characterization batches: {len(dataset['rubber_characterization'])}")

if __name__ == "__main__":
    main()