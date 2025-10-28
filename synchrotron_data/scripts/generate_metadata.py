"""
Generate comprehensive metadata for synchrotron X-ray experiments.
Includes material specifications, operational parameters, and experimental conditions.
"""

import json
import yaml
from datetime import datetime, timedelta
import numpy as np
import os

class MetadataGenerator:
    """Generate realistic metadata for synchrotron experiments."""
    
    def __init__(self):
        self.facility = "Advanced Photon Source (Simulated)"
        self.beamline = "1-ID-E High Energy Diffraction"
        self.start_date = datetime.now()
        
    def generate_material_specifications(self):
        """Generate detailed material specifications."""
        
        materials = {
            "interconnect_alloy": {
                "designation": "Crofer 22 APU",
                "type": "Ferritic Stainless Steel",
                "composition": {
                    "Fe": "balance",
                    "Cr": 22.5,
                    "Mn": 0.45,
                    "Ti": 0.08,
                    "La": 0.08,
                    "Si": 0.3,
                    "Al": 0.05,
                    "C": 0.01,
                    "N": 0.01,
                    "P": 0.02,
                    "S": 0.005
                },
                "heat_treatment": {
                    "process": "Solution annealed",
                    "temperature": 1050,
                    "duration_minutes": 30,
                    "cooling": "Air cooled"
                },
                "initial_microstructure": {
                    "average_grain_size": 45,  # micrometers
                    "grain_size_distribution": "Log-normal",
                    "texture": "Weak rolling texture",
                    "phase_fractions": {
                        "ferrite": 0.92,
                        "laves_phase": 0.05,
                        "carbides": 0.03
                    }
                },
                "mechanical_properties": {
                    "youngs_modulus": 200,  # GPa
                    "poisson_ratio": 0.3,
                    "yield_strength_RT": 280,  # MPa at room temperature
                    "yield_strength_700C": 145,  # MPa at 700°C
                    "CTE": 12.3e-6,  # 1/K (20-800°C)
                    "thermal_conductivity_700C": 25  # W/m·K
                },
                "creep_properties": {
                    "norton_stress_exponent": 5.2,
                    "activation_energy": 280,  # kJ/mol
                    "pre_exponential_factor": 1.2e-10
                }
            },
            "anode_support": {
                "designation": "Ni-YSZ Cermet",
                "type": "Ceramic-Metal Composite",
                "composition": {
                    "NiO": 60,  # wt% (before reduction)
                    "YSZ": 40   # wt% (8 mol% Y2O3-stabilized ZrO2)
                },
                "microstructure": {
                    "porosity": 0.35,  # Volume fraction
                    "pore_size_mean": 2.5,  # micrometers
                    "particle_size_Ni": 1.5,  # micrometers
                    "particle_size_YSZ": 0.8,  # micrometers
                    "percolation": "Triple phase boundary optimized"
                },
                "mechanical_properties": {
                    "youngs_modulus_reduced": 45,  # GPa (after reduction)
                    "fracture_toughness": 1.2,  # MPa·m^0.5
                    "CTE": 12.5e-6,  # 1/K
                    "electrical_conductivity_800C": 1000  # S/cm
                }
            }
        }
        
        return materials
    
    def generate_sample_geometry(self):
        """Generate sample geometry specifications."""
        
        geometries = {
            "dog_bone_tensile": {
                "type": "Miniature dog-bone specimen",
                "gauge_length": 10.0,  # mm
                "gauge_width": 2.0,    # mm
                "gauge_thickness": 0.5,  # mm
                "total_length": 25.0,  # mm
                "grip_width": 5.0,     # mm
                "fillet_radius": 2.0,  # mm
                "surface_finish": "Electropolished",
                "fabrication_method": "Wire EDM"
            },
            "compact_tension": {
                "type": "Compact tension (CT) specimen",
                "width": 12.5,  # mm
                "thickness": 2.0,  # mm
                "notch_length": 5.0,  # mm
                "notch_angle": 60,  # degrees
                "notch_tip_radius": 0.1,  # mm
                "hole_diameter": 2.5,  # mm
                "hole_spacing": 6.25  # mm
            },
            "cylindrical": {
                "type": "Cylindrical compression specimen",
                "diameter": 3.0,  # mm
                "height": 6.0,    # mm
                "aspect_ratio": 2.0,
                "end_preparation": "Parallel ground",
                "surface_finish": "600 grit"
            }
        }
        
        return geometries
    
    def generate_experimental_conditions(self, test_id, temperature, stress, duration_hours):
        """Generate detailed experimental conditions for a specific test."""
        
        conditions = {
            "test_id": test_id,
            "test_type": "In-situ creep test with tomography",
            "start_time": self.start_date.isoformat(),
            "end_time": (self.start_date + timedelta(hours=duration_hours)).isoformat(),
            "duration_hours": duration_hours,
            
            "thermal_conditions": {
                "nominal_temperature": temperature,  # °C
                "temperature_stability": "±2°C",
                "heating_rate": 10,  # °C/min
                "temperature_profile": "Isothermal after stabilization",
                "atmosphere": {
                    "composition": "Ar-5%H2",
                    "flow_rate": 100,  # ml/min
                    "dew_point": -40,  # °C
                    "oxygen_partial_pressure": 1e-20  # atm
                },
                "furnace_type": "Resistance heating with Mo elements",
                "temperature_measurement": "Type S thermocouple"
            },
            
            "mechanical_loading": {
                "load_type": "Constant stress (creep)",
                "nominal_stress": stress,  # MPa
                "stress_uncertainty": "±1 MPa",
                "loading_rate": 50,  # N/s
                "load_cell_capacity": 5000,  # N
                "load_cell_accuracy": "0.1% full scale",
                "alignment": "Class 5 per ASTM E1012"
            },
            
            "synchrotron_parameters": {
                "beam_energy": 80,  # keV
                "wavelength": 0.1549,  # Angstroms
                "beam_size": "0.5 × 0.5 mm²",
                "flux": 1e12,  # photons/s
                "detector": {
                    "tomography": {
                        "type": "Scintillator + sCMOS",
                        "pixel_size": 0.65,  # micrometers
                        "array_size": "2048 × 2048",
                        "bit_depth": 16,
                        "frame_rate": 100  # Hz
                    },
                    "diffraction": {
                        "type": "GE amorphous Si detector",
                        "pixel_size": 200,  # micrometers
                        "array_size": "2048 × 2048",
                        "sample_detector_distance": 1500,  # mm
                        "integration_time": 1  # seconds
                    }
                },
                "scan_parameters": {
                    "tomography": {
                        "projections": 1800,
                        "angular_range": 180,  # degrees
                        "scan_time": 60,  # seconds per scan
                        "scan_interval": duration_hours / 10  # hours between scans
                    },
                    "diffraction": {
                        "2theta_range": [10, 120],  # degrees
                        "step_size": 0.02,  # degrees
                        "count_time": 0.5,  # seconds per step
                        "spatial_resolution": "50 × 50 points"
                    }
                }
            },
            
            "data_acquisition": {
                "sampling_rate": {
                    "load": 10,  # Hz
                    "displacement": 10,  # Hz
                    "temperature": 1,  # Hz
                    "tomography": 1/(duration_hours/10 * 3600),  # Hz
                    "diffraction": 1/(duration_hours/10 * 3600)  # Hz
                },
                "data_storage": {
                    "format": "HDF5",
                    "compression": "gzip",
                    "total_data_size_estimate": f"{10 + duration_hours * 0.5} GB"
                },
                "metadata_standard": "NeXus format compliant"
            },
            
            "environmental_monitoring": {
                "room_temperature": 22,  # °C
                "room_humidity": 45,  # %RH
                "vibration_level": "< 1 μm peak-to-peak",
                "beam_stability": "< 1% intensity variation"
            }
        }
        
        return conditions
    
    def generate_calibration_data(self):
        """Generate calibration information."""
        
        calibration = {
            "spatial_calibration": {
                "method": "Ruby spheres standard",
                "reference_size": 50,  # micrometers
                "calibrated_pixel_size": 0.65,  # micrometers
                "uncertainty": "±0.01 micrometers",
                "date_calibrated": (self.start_date - timedelta(days=7)).isoformat()
            },
            
            "stress_calibration": {
                "method": "Silicon powder standard",
                "reference_lattice_parameter": 5.43102,  # Angstroms at 25°C
                "stress_free_2theta_peaks": {
                    "111": 28.443,
                    "220": 47.303,
                    "311": 56.122,
                    "400": 69.130,
                    "331": 76.375
                },
                "elastic_constants": {
                    "C11": 165.7,  # GPa
                    "C12": 63.9,   # GPa
                    "C44": 79.6    # GPa
                }
            },
            
            "temperature_calibration": {
                "method": "Melting point standards",
                "calibration_points": [
                    {"material": "Zn", "melting_point": 419.53},
                    {"material": "Al", "melting_point": 660.32},
                    {"material": "Ag", "melting_point": 961.78}
                ],
                "uncertainty": "±2°C",
                "thermocouple_correction": "Applied"
            },
            
            "detector_calibration": {
                "dark_field": "Acquired and subtracted",
                "flat_field": "Acquired with open beam",
                "bad_pixels": "Mapped and interpolated",
                "distortion_correction": "Applied using grid phantom",
                "efficiency_correction": "Energy-dependent correction applied"
            }
        }
        
        return calibration
    
    def generate_data_quality_metrics(self):
        """Generate data quality assessment metrics."""
        
        quality_metrics = {
            "tomography_quality": {
                "signal_to_noise_ratio": 45,  # dB
                "spatial_resolution": 1.3,  # micrometers (MTF 10%)
                "contrast_resolution": 0.5,  # % 
                "ring_artifacts": "Corrected",
                "beam_hardening": "Corrected using iterative method",
                "phase_retrieval": "Paganin method applied",
                "reconstruction_algorithm": "Filtered back projection with Shepp-Logan filter"
            },
            
            "diffraction_quality": {
                "peak_to_background_ratio": 100,
                "instrumental_broadening": 0.08,  # degrees 2theta
                "peak_asymmetry": "< 2%",
                "wavelength_stability": "< 0.001 Angstroms",
                "preferred_orientation": "Corrected using March-Dollase",
                "absorption_correction": "Applied using μ·t calculation"
            },
            
            "mechanical_data_quality": {
                "load_noise": "< 0.5 N RMS",
                "displacement_noise": "< 0.1 μm RMS",
                "bending_moment": "< 2% of axial load",
                "temperature_gradient": "< 2°C across gauge length",
                "strain_measurement": "DIC with 0.001 resolution"
            }
        }
        
        return quality_metrics
    
    def save_metadata(self, output_dir='synchrotron_data/metadata'):
        """Save all metadata to files."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all metadata
        all_metadata = {
            "facility_info": {
                "facility": self.facility,
                "beamline": self.beamline,
                "experiment_date": self.start_date.isoformat(),
                "principal_investigator": "Dr. A. Researcher",
                "institution": "Materials Research Institute",
                "project": "SOFC Interconnect Creep Degradation Study",
                "funding": "DOE ARPA-E Grant DE-AR0001234"
            },
            "material_specifications": self.generate_material_specifications(),
            "sample_geometries": self.generate_sample_geometry(),
            "calibration_data": self.generate_calibration_data(),
            "data_quality_metrics": self.generate_data_quality_metrics()
        }
        
        # Save as JSON
        with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
        
        # Save as YAML for better readability
        with open(os.path.join(output_dir, 'experiment_metadata.yaml'), 'w') as f:
            yaml.dump(all_metadata, f, default_flow_style=False, sort_keys=False)
        
        # Generate metadata for each test condition
        test_conditions = [
            {"id": "TEST001", "temperature": 600, "stress": 150, "duration": 500},
            {"id": "TEST002", "temperature": 700, "stress": 100, "duration": 1000},
            {"id": "TEST003", "temperature": 800, "stress": 75, "duration": 2000}
        ]
        
        for test in test_conditions:
            test_metadata = self.generate_experimental_conditions(
                test["id"],
                test["temperature"],
                test["stress"],
                test["duration"]
            )
            
            filename = f"{test['id']}_T{test['temperature']}_S{test['stress']}_metadata.json"
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(test_metadata, f, indent=2, default=str)
        
        print(f"Metadata saved to {output_dir}/")
        
        # Create a summary CSV for quick reference
        import pandas as pd
        
        summary_data = []
        for test in test_conditions:
            summary_data.append({
                'Test_ID': test['id'],
                'Temperature_C': test['temperature'],
                'Stress_MPa': test['stress'],
                'Duration_hours': test['duration'],
                'Material': 'Crofer 22 APU',
                'Specimen_Type': 'Dog-bone tensile',
                'Atmosphere': 'Ar-5%H2',
                'Beam_Energy_keV': 80,
                'Data_Points': 10
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'test_summary.csv'), index=False)
        print(f"Test summary saved to {output_dir}/test_summary.csv")


def main():
    """Generate all metadata files."""
    
    print("\n=== Generating Experimental Metadata ===")
    
    generator = MetadataGenerator()
    generator.save_metadata()
    
    print("\n=== Metadata Generation Complete ===")


if __name__ == '__main__':
    main()