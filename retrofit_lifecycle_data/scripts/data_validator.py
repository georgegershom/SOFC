"""
Data Validation and Quality Assurance Script
Validates the retrofit intervention and lifecycle datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


class RetrofitDataValidator:
    """Validate retrofit datasets for completeness and quality"""
    
    def __init__(self, data_dir: str = '../databases'):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
    def validate_retrofit_measures(self) -> dict:
        """Validate retrofit measures database"""
        print("Validating retrofit_measures.csv...")
        df = pd.read_csv(self.data_dir / 'retrofit_measures.csv')
        
        results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_measures': df['measure_id'].nunique(),
            'categories': df['measure_category'].value_counts().to_dict(),
            'errors': []
        }
        
        # Check for duplicates
        if df['measure_id'].duplicated().any():
            results['errors'].append("Duplicate measure IDs found")
        
        # Validate numeric ranges
        if (df['lifespan_years'] < 0).any() or (df['lifespan_years'] > 100).any():
            results['errors'].append("Invalid lifespan values")
        
        if (df['capex_cost_usd_per_unit'] < 0).any():
            results['errors'].append("Negative CAPEX costs found")
        
        # Check U-value relationships
        u_value_measures = df[df['u_value_before'].notna() & df['u_value_after'].notna()]
        if (u_value_measures['u_value_after'] >= u_value_measures['u_value_before']).any():
            results['errors'].append("U-value not improving for some measures")
        
        results['status'] = 'PASS' if len(results['errors']) == 0 else 'FAIL'
        return results
    
    def validate_lci_database(self) -> dict:
        """Validate lifecycle inventory database"""
        print("Validating lifecycle_inventory_database.csv...")
        df = pd.read_csv(self.data_dir / 'lifecycle_inventory_database.csv')
        
        results = {
            'total_records': len(df),
            'unique_materials': df['material_id'].nunique(),
            'errors': []
        }
        
        # Check LCA stage totals
        for idx, row in df.iterrows():
            a1_a3_calc = row['A1_raw_material_supply_kg_co2e'] + \
                        row['A2_transport_to_factory_kg_co2e'] + \
                        row['A3_manufacturing_kg_co2e']
            
            if not np.isclose(a1_a3_calc, row['A1_A3_total_kg_co2e'], rtol=0.01):
                results['errors'].append(f"A1-A3 total mismatch for {row['material_id']}")
        
        # Check for negative values
        carbon_cols = [col for col in df.columns if 'kg_co2e' in col and 'biogenic' not in col]
        for col in carbon_cols:
            if (df[col] < 0).any() and col != 'D_reuse_recovery_recycling_kg_co2e':
                results['errors'].append(f"Negative values in {col}")
        
        results['status'] = 'PASS' if len(results['errors']) == 0 else 'FAIL'
        return results
    
    def validate_maintenance_data(self) -> dict:
        """Validate maintenance and operational data"""
        print("Validating maintenance_operational_data.csv...")
        df = pd.read_csv(self.data_dir / 'maintenance_operational_data.csv')
        
        results = {
            'total_records': len(df),
            'errors': []
        }
        
        # Check maintenance frequency logic
        if (df['maintenance_frequency_years'] > df['lifespan_years']).any():
            results['errors'].append("Maintenance frequency exceeds lifespan")
        
        # Check degradation rates
        if (df['degradation_rate_percent_per_year'] < 0).any():
            results['errors'].append("Negative degradation rates found")
        
        if (df['degradation_rate_percent_per_year'] > 10).any():
            results['errors'].append("Unrealistically high degradation rates (>10%/year)")
        
        # Check failure rates
        if (df['failure_rate_percent_per_year'] < 0).any() or (df['failure_rate_percent_per_year'] > 100).any():
            results['errors'].append("Invalid failure rates")
        
        results['status'] = 'PASS' if len(results['errors']) == 0 else 'FAIL'
        return results
    
    def validate_carbon_intensity(self) -> dict:
        """Validate carbon intensity factors"""
        print("Validating carbon_intensity_factors.csv...")
        df = pd.read_csv(Path('../reference_data') / 'carbon_intensity_factors.csv')
        
        results = {
            'total_records': len(df),
            'unique_regions': df['region'].nunique(),
            'errors': []
        }
        
        # Check grid intensity ranges
        if (df['electricity_grid_kg_co2e_per_kwh'] < 0).any() or \
           (df['electricity_grid_kg_co2e_per_kwh'] > 1.5).any():
            results['errors'].append("Grid carbon intensity out of realistic range")
        
        # Check renewable fraction
        if (df['renewable_fraction_percent'] < 0).any() or \
           (df['renewable_fraction_percent'] > 100).any():
            results['errors'].append("Invalid renewable fraction")
        
        results['status'] = 'PASS' if len(results['errors']) == 0 else 'FAIL'
        return results
    
    def validate_material_properties(self) -> dict:
        """Validate material properties"""
        print("Validating material_properties.csv...")
        df = pd.read_csv(Path('../reference_data') / 'material_properties.csv')
        
        results = {
            'total_records': len(df),
            'errors': []
        }
        
        # Check thermal conductivity
        if (df['thermal_conductivity_w_per_m_k'] < 0).any():
            results['errors'].append("Negative thermal conductivity")
        
        # Check recycled content
        if (df['recycled_content_percent'] < 0).any() or \
           (df['recycled_content_percent'] > 100).any():
            results['errors'].append("Invalid recycled content percentage")
        
        # Check recyclability
        if (df['recyclability_percent'] < 0).any() or \
           (df['recyclability_percent'] > 100).any():
            results['errors'].append("Invalid recyclability percentage")
        
        results['status'] = 'PASS' if len(results['errors']) == 0 else 'FAIL'
        return results
    
    def run_all_validations(self) -> dict:
        """Run all validation checks"""
        print("=" * 80)
        print("Running Data Validation Suite")
        print("=" * 80)
        print()
        
        all_results = {
            'retrofit_measures': self.validate_retrofit_measures(),
            'lci_database': self.validate_lci_database(),
            'maintenance_data': self.validate_maintenance_data(),
            'carbon_intensity': self.validate_carbon_intensity(),
            'material_properties': self.validate_material_properties()
        }
        
        print()
        print("=" * 80)
        print("Validation Results Summary")
        print("=" * 80)
        print()
        
        for dataset, results in all_results.items():
            status_symbol = "✓" if results['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {dataset}: {results['status']}")
            if results['errors']:
                for error in results['errors']:
                    print(f"    - {error}")
        
        print()
        
        # Overall status
        all_passed = all(r['status'] == 'PASS' for r in all_results.values())
        if all_passed:
            print("🎉 All validations PASSED!")
        else:
            print("⚠️  Some validations FAILED. Please review errors above.")
        
        return all_results


def main():
    validator = RetrofitDataValidator()
    results = validator.run_all_validations()
    
    # Return exit code
    all_passed = all(r['status'] == 'PASS' for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
