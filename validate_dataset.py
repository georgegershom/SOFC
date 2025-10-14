"""
Data Validation Script for Building Retrofit Dataset
Validates data quality, completeness, and consistency across all dataset components
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetValidator:
    """Validates the quality and consistency of the building retrofit dataset"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "categories": {},
            "issues": [],
            "recommendations": []
        }
    
    def load_metadata(self):
        """Load dataset metadata"""
        with open(f"{self.data_dir}/metadata.json", "r") as f:
            self.metadata = json.load(f)
        return self.metadata
    
    def validate_iot_data(self):
        """Validate IoT sensor data quality"""
        print("Validating IoT sensor data...")
        category_results = {
            "name": "IoT Sensor Data",
            "score": 0,
            "checks": [],
            "issues": []
        }
        
        # Load IoT data files
        energy_data = pd.read_csv(f"{self.data_dir}/iot_energy_consumption.csv")
        env_data = pd.read_csv(f"{self.data_dir}/iot_environmental_parameters.csv")
        weather_data = pd.read_csv(f"{self.data_dir}/iot_weather_conditions.csv")
        occupancy_data = pd.read_csv(f"{self.data_dir}/iot_occupancy_patterns.csv")
        
        # Check data completeness
        total_records = len(energy_data) + len(env_data) + len(weather_data) + len(occupancy_data)
        expected_records = self.metadata["num_buildings"] * 1461  # 4 years * 365.25 days
        completeness = min(100, (total_records / expected_records) * 100)
        
        category_results["checks"].append({
            "check": "Data Completeness",
            "status": "PASS" if completeness > 95 else "WARN",
            "value": f"{completeness:.1f}%",
            "expected": ">95%"
        })
        
        # Check for missing values
        energy_missing = energy_data.isnull().sum().sum()
        env_missing = env_data.isnull().sum().sum()
        weather_missing = weather_data.isnull().sum().sum()
        occupancy_missing = occupancy_data.isnull().sum().sum()
        
        total_missing = energy_missing + env_missing + weather_missing + occupancy_missing
        missing_percentage = (total_missing / (total_records * len(energy_data.columns))) * 100
        
        category_results["checks"].append({
            "check": "Missing Values",
            "status": "PASS" if missing_percentage < 5 else "WARN",
            "value": f"{missing_percentage:.2f}%",
            "expected": "<5%"
        })
        
        # Check data ranges
        energy_issues = []
        if (energy_data['total_consumption_kwh'] < 0).any():
            energy_issues.append("Negative energy consumption values found")
        if (energy_data['total_consumption_kwh'] > 10000).any():
            energy_issues.append("Unrealistically high energy consumption values found")
        
        env_issues = []
        if (env_data['co2_ppm'] < 300).any() or (env_data['co2_ppm'] > 5000).any():
            env_issues.append("CO2 levels outside realistic range (300-5000 ppm)")
        if (env_data['temperature_c'] < 10).any() or (env_data['temperature_c'] > 40).any():
            env_issues.append("Temperature values outside realistic range (10-40°C)")
        
        if energy_issues or env_issues:
            category_results["issues"].extend(energy_issues + env_issues)
        
        # Calculate score
        checks_passed = sum(1 for check in category_results["checks"] if check["status"] == "PASS")
        total_checks = len(category_results["checks"])
        category_results["score"] = (checks_passed / total_checks) * 100
        
        self.validation_results["categories"]["iot_data"] = category_results
        print(f"  IoT Data Score: {category_results['score']:.1f}%")
    
    def validate_building_attributes(self):
        """Validate building attributes data quality"""
        print("Validating building attributes data...")
        category_results = {
            "name": "Building Attributes",
            "score": 0,
            "checks": [],
            "issues": []
        }
        
        # Load building data files
        basic_info = pd.read_csv(f"{self.data_dir}/building_basic_info.csv")
        geometric = pd.read_csv(f"{self.data_dir}/building_geometric_data.csv")
        thermal = pd.read_csv(f"{self.data_dir}/building_thermal_properties.csv")
        materials = pd.read_csv(f"{self.data_dir}/building_construction_materials.csv")
        
        # Check data completeness
        expected_buildings = self.metadata["num_buildings"]
        actual_buildings = len(basic_info)
        completeness = (actual_buildings / expected_buildings) * 100
        
        category_results["checks"].append({
            "check": "Building Count",
            "status": "PASS" if completeness == 100 else "FAIL",
            "value": f"{actual_buildings}/{expected_buildings}",
            "expected": f"{expected_buildings}"
        })
        
        # Check for missing values (excluding expected missing values)
        missing_data = []
        for df_name, df in [("basic_info", basic_info), ("geometric", geometric), 
                           ("thermal", thermal), ("materials", materials)]:
            missing_count = df.isnull().sum().sum()
            # For basic_info, last_renovation_year can be missing (not all buildings renovated)
            if df_name == "basic_info":
                expected_missing = df['last_renovation_year'].isnull().sum()
                unexpected_missing = missing_count - expected_missing
                if unexpected_missing > 0:
                    missing_data.append(f"{df_name}: {unexpected_missing} unexpected missing values")
            else:
                if missing_count > 0:
                    missing_data.append(f"{df_name}: {missing_count} missing values")
        
        category_results["checks"].append({
            "check": "Missing Values",
            "status": "PASS" if not missing_data else "WARN",
            "value": f"{len(missing_data)} files with missing data",
            "expected": "0 (excluding expected missing renovation years)"
        })
        
        # Check data consistency
        consistency_issues = []
        
        # Check if all buildings have corresponding records in all tables
        building_ids = set(basic_info['building_id'])
        for df_name, df in [("geometric", geometric), ("thermal", thermal), ("materials", materials)]:
            missing_buildings = building_ids - set(df['building_id'])
            if missing_buildings:
                consistency_issues.append(f"{df_name}: Missing data for {len(missing_buildings)} buildings")
        
        # Check realistic value ranges
        if (geometric['total_floor_area_m2'] <= 0).any():
            consistency_issues.append("Non-positive floor area values found")
        if (geometric['height_m'] <= 0).any():
            consistency_issues.append("Non-positive height values found")
        if (thermal['wall_u_value_wm2k'] <= 0).any():
            consistency_issues.append("Non-positive U-values found")
        
        if consistency_issues:
            category_results["issues"].extend(consistency_issues)
        
        # Calculate score
        checks_passed = sum(1 for check in category_results["checks"] if check["status"] == "PASS")
        total_checks = len(category_results["checks"])
        category_results["score"] = (checks_passed / total_checks) * 100
        
        self.validation_results["categories"]["building_attributes"] = category_results
        print(f"  Building Attributes Score: {category_results['score']:.1f}%")
    
    def validate_energy_performance(self):
        """Validate energy performance data quality"""
        print("Validating energy performance data...")
        category_results = {
            "name": "Energy Performance",
            "score": 0,
            "checks": [],
            "issues": []
        }
        
        # Load energy data files
        historical = pd.read_csv(f"{self.data_dir}/energy_historical_consumption.csv")
        ratings = pd.read_csv(f"{self.data_dir}/energy_efficiency_ratings.csv")
        retrofit = pd.read_csv(f"{self.data_dir}/energy_retrofit_impact.csv")
        
        # Check data completeness
        expected_historical = self.metadata["num_buildings"] * 5 * 12  # 5 years * 12 months
        actual_historical = len(historical)
        historical_completeness = (actual_historical / expected_historical) * 100
        
        category_results["checks"].append({
            "check": "Historical Data Completeness",
            "status": "PASS" if historical_completeness > 90 else "WARN",
            "value": f"{historical_completeness:.1f}%",
            "expected": ">90%"
        })
        
        # Check rating data
        rating_buildings = len(ratings)
        expected_ratings = self.metadata["num_buildings"]
        rating_completeness = (rating_buildings / expected_ratings) * 100
        
        category_results["checks"].append({
            "check": "Rating Data Completeness",
            "status": "PASS" if rating_completeness > 95 else "WARN",
            "value": f"{rating_completeness:.1f}%",
            "expected": ">95%"
        })
        
        # Check data quality
        quality_issues = []
        
        # Check for negative energy values
        if (historical['total_energy_kwh'] < 0).any():
            quality_issues.append("Negative energy consumption values found")
        
        # Check for unrealistic energy values
        if (historical['total_energy_kwh'] > 50000).any():
            quality_issues.append("Unrealistically high energy consumption values found")
        
        # Check rating values
        valid_ratings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        invalid_ratings = set(ratings['eu_energy_rating']) - set(valid_ratings)
        if invalid_ratings:
            quality_issues.append(f"Invalid energy ratings found: {invalid_ratings}")
        
        if quality_issues:
            category_results["issues"].extend(quality_issues)
        
        # Calculate score
        checks_passed = sum(1 for check in category_results["checks"] if check["status"] == "PASS")
        total_checks = len(category_results["checks"])
        category_results["score"] = (checks_passed / total_checks) * 100
        
        self.validation_results["categories"]["energy_performance"] = category_results
        print(f"  Energy Performance Score: {category_results['score']:.1f}%")
    
    def validate_lca_data(self):
        """Validate LCA data quality"""
        print("Validating LCA data...")
        category_results = {
            "name": "Lifecycle Assessment",
            "score": 0,
            "checks": [],
            "issues": []
        }
        
        # Load LCA data files
        epds = pd.read_csv(f"{self.data_dir}/lca_material_epds.csv")
        building_lca = pd.read_csv(f"{self.data_dir}/lca_building_lca.csv")
        
        # Check data completeness
        epd_count = len(epds)
        expected_epds = 8  # Based on material categories
        epd_completeness = (epd_count / expected_epds) * 100
        
        category_results["checks"].append({
            "check": "EPD Data Completeness",
            "status": "PASS" if epd_completeness >= 100 else "WARN",
            "value": f"{epd_count}/{expected_epds}",
            "expected": f"{expected_epds}"
        })
        
        # Check building LCA data
        lca_buildings = len(building_lca['building_id'].unique())
        expected_lca_buildings = self.metadata["num_buildings"]
        lca_completeness = (lca_buildings / expected_lca_buildings) * 100
        
        category_results["checks"].append({
            "check": "Building LCA Completeness",
            "status": "PASS" if lca_completeness >= 95 else "WARN",
            "value": f"{lca_completeness:.1f}%",
            "expected": ">95%"
        })
        
        # Check data quality
        quality_issues = []
        
        # Check for negative environmental impact values
        impact_columns = ['global_warming_potential_kgco2eq', 'acidification_potential_kgso2eq', 
                         'eutrophication_potential_kgpo4eq']
        for col in impact_columns:
            if col in epds.columns and (epds[col] < 0).any():
                quality_issues.append(f"Negative values found in {col}")
        
        if quality_issues:
            category_results["issues"].extend(quality_issues)
        
        # Calculate score
        checks_passed = sum(1 for check in category_results["checks"] if check["status"] == "PASS")
        total_checks = len(category_results["checks"])
        category_results["score"] = (checks_passed / total_checks) * 100
        
        self.validation_results["categories"]["lca_data"] = category_results
        print(f"  LCA Data Score: {category_results['score']:.1f}%")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        
        # Calculate overall score
        category_scores = [cat["score"] for cat in self.validation_results["categories"].values()]
        self.validation_results["overall_score"] = np.mean(category_scores)
        
        print(f"Overall Dataset Score: {self.validation_results['overall_score']:.1f}%")
        print()
        
        # Category scores
        print("Category Scores:")
        for category, results in self.validation_results["categories"].items():
            print(f"  {results['name']}: {results['score']:.1f}%")
        print()
        
        # Detailed results
        for category, results in self.validation_results["categories"].items():
            print(f"{results['name']} Details:")
            for check in results["checks"]:
                status_symbol = "✓" if check["status"] == "PASS" else "⚠" if check["status"] == "WARN" else "✗"
                print(f"  {status_symbol} {check['check']}: {check['value']} (expected: {check['expected']})")
            
            if results["issues"]:
                print("  Issues:")
                for issue in results["issues"]:
                    print(f"    - {issue}")
            print()
        
        # Recommendations
        recommendations = []
        if self.validation_results["overall_score"] < 90:
            recommendations.append("Consider improving data quality in low-scoring categories")
        if any(cat["score"] < 80 for cat in self.validation_results["categories"].values()):
            recommendations.append("Address critical issues in failing categories")
        if not recommendations:
            recommendations.append("Dataset quality is excellent - ready for research use")
        
        self.validation_results["recommendations"] = recommendations
        
        print("Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Save validation report
        with open(f"{self.data_dir}/validation_report.json", "w") as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"\nValidation report saved to: {self.data_dir}/validation_report.json")
    
    def create_data_quality_visualizations(self):
        """Create visualizations for data quality assessment"""
        print("Creating data quality visualizations...")
        
        # Load key datasets
        energy_data = pd.read_csv(f"{self.data_dir}/iot_energy_consumption.csv")
        building_data = pd.read_csv(f"{self.data_dir}/building_basic_info.csv")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Energy consumption distribution
        axes[0, 0].hist(energy_data['total_consumption_kwh'], bins=50, alpha=0.7)
        axes[0, 0].set_title('Energy Consumption Distribution')
        axes[0, 0].set_xlabel('Total Consumption (kWh)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Building types distribution
        building_types = building_data['building_type'].value_counts()
        axes[0, 1].pie(building_types.values, labels=building_types.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Building Types Distribution')
        
        # Construction year distribution
        axes[1, 0].hist(building_data['construction_year'], bins=20, alpha=0.7)
        axes[1, 0].set_title('Construction Year Distribution')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Number of Buildings')
        
        # Energy vs Building Type
        energy_with_type = energy_data.merge(
            building_data[['building_id', 'building_type']], 
            on='building_id'
        )
        energy_by_type = energy_with_type.groupby('building_type')['total_consumption_kwh'].mean()
        axes[1, 1].bar(energy_by_type.index, energy_by_type.values)
        axes[1, 1].set_title('Average Energy Consumption by Building Type')
        axes[1, 1].set_xlabel('Building Type')
        axes[1, 1].set_ylabel('Average Consumption (kWh)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/data_quality_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Data quality visualizations saved to: {self.data_dir}/data_quality_visualizations.png")
    
    def run_full_validation(self):
        """Run complete dataset validation"""
        print("Starting comprehensive dataset validation...")
        print("="*60)
        
        # Load metadata
        self.load_metadata()
        
        # Validate each category
        self.validate_iot_data()
        self.validate_building_attributes()
        self.validate_energy_performance()
        self.validate_lca_data()
        
        # Generate report
        self.generate_validation_report()
        
        # Create visualizations
        self.create_data_quality_visualizations()
        
        print("="*60)
        print("VALIDATION COMPLETE!")
        print("="*60)

if __name__ == "__main__":
    validator = DatasetValidator()
    validator.run_full_validation()