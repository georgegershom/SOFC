#!/usr/bin/env python3
"""
Data analysis tools for geotechnical datasets
Comprehensive analysis and visualization functions for PhD research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GeotechnicalAnalyzer:
    """Comprehensive analysis class for geotechnical data"""
    
    def __init__(self):
        self.sandy_data = None
        self.clay_data = None
        self.case_studies = None
        self.geospatial_data = None
        
    def load_datasets(self, data_dir='geotechnical_datasets'):
        """Load all generated datasets"""
        
        try:
            # Load sandy soil data
            self.sandy_data = pd.read_csv(f'{data_dir}/sandy_soils/sandy_soil_complete_dataset.csv')
            print(f"Loaded {len(self.sandy_data)} sandy soil samples")
            
            # Load clay soil data
            self.clay_data = pd.read_csv(f'{data_dir}/clay_soils/clay_soil_complete_dataset.csv')
            print(f"Loaded {len(self.clay_data)} clay soil samples")
            
            # Load case studies
            sandy_cases = pd.read_csv(f'{data_dir}/case_studies/sandy_soil_failure_cases.csv')
            clay_cases = pd.read_csv(f'{data_dir}/case_studies/clay_soil_failure_cases.csv')
            self.case_studies = {'sandy': sandy_cases, 'clay': clay_cases}
            print(f"Loaded {len(sandy_cases)} sandy and {len(clay_cases)} clay failure cases")
            
            # Load geospatial data
            self.geospatial_data = pd.read_csv(f'{data_dir}/geospatial_data/regional_soil_properties_grid.csv')
            print(f"Loaded {len(self.geospatial_data)} geospatial grid points")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
        
        return True
    
    def analyze_sandy_soil_properties(self, save_plots=True):
        """Comprehensive analysis of sandy soil properties"""
        
        if self.sandy_data is None:
            print("Sandy soil data not loaded")
            return
        
        print("\\n=== SANDY SOIL ANALYSIS ===")
        
        # Basic statistics
        print("\\nBasic Properties Statistics:")
        basic_cols = ['sand_content_pct', 'relative_density_pct', 'friction_angle_deg_y', 
                     'permeability_m_s', 'cyclic_resistance_ratio']
        print(self.sandy_data[basic_cols].describe())
        
        # Correlations
        print("\\nKey Correlations:")
        correlations = self.sandy_data[basic_cols].corr()
        print(correlations)
        
        if save_plots:
            # Create comprehensive plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Sandy Soil Properties Analysis', fontsize=16)
            
            # Grain size distribution
            axes[0,0].hist(self.sandy_data['sand_content_pct'], bins=30, alpha=0.7, color='orange')
            axes[0,0].set_title('Sand Content Distribution')
            axes[0,0].set_xlabel('Sand Content (%)')
            axes[0,0].set_ylabel('Frequency')
            
            # Relative density vs friction angle
            axes[0,1].scatter(self.sandy_data['relative_density_pct'], 
                            self.sandy_data['friction_angle_deg_y'], alpha=0.6)
            axes[0,1].set_title('Relative Density vs Friction Angle')
            axes[0,1].set_xlabel('Relative Density (%)')
            axes[0,1].set_ylabel('Friction Angle (°)')
            
            # Liquefaction susceptibility
            liq_data = self.sandy_data.groupby('factor_safety_m7')['cyclic_resistance_ratio'].mean()
            axes[0,2].plot(liq_data.index, liq_data.values, 'ro-')
            axes[0,2].set_title('Liquefaction Resistance')
            axes[0,2].set_xlabel('Factor of Safety (M7)')
            axes[0,2].set_ylabel('Cyclic Resistance Ratio')
            
            # Permeability distribution (log scale)
            axes[1,0].hist(np.log10(self.sandy_data['permeability_m_s']), bins=30, alpha=0.7, color='blue')
            axes[1,0].set_title('Permeability Distribution')
            axes[1,0].set_xlabel('Log10(Permeability) [m/s]')
            axes[1,0].set_ylabel('Frequency')
            
            # Correlation heatmap
            sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
            axes[1,1].set_title('Property Correlations')
            
            # Depth vs properties
            axes[1,2].scatter(self.sandy_data['depth_m'], self.sandy_data['relative_density_pct'], 
                            alpha=0.6, label='Relative Density')
            axes[1,2].scatter(self.sandy_data['depth_m'], self.sandy_data['friction_angle_deg_y'], 
                            alpha=0.6, label='Friction Angle')
            axes[1,2].set_title('Depth vs Properties')
            axes[1,2].set_xlabel('Depth (m)')
            axes[1,2].set_ylabel('Value')
            axes[1,2].legend()
            
            plt.tight_layout()
            plt.savefig('geotechnical_datasets/documentation/sandy_soil_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Statistical relationships
        self._analyze_sandy_correlations()
        
    def analyze_clay_soil_properties(self, save_plots=True):
        """Comprehensive analysis of clay soil properties"""
        
        if self.clay_data is None:
            print("Clay soil data not loaded")
            return
        
        print("\\n=== CLAY SOIL ANALYSIS ===")
        
        # Basic statistics
        print("\\nBasic Properties Statistics:")
        basic_cols = ['clay_content_pct', 'plasticity_index', 'liquid_limit_pct', 
                     'undrained_shear_strength_kPa', 'sensitivity', 'compression_index']
        print(self.clay_data[basic_cols].describe())
        
        # Mineralogy analysis
        print("\\nMineralogy Distribution:")
        print(self.clay_data['dominant_clay_mineral'].value_counts())
        
        if save_plots:
            # Create comprehensive plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Clay Soil Properties Analysis', fontsize=16)
            
            # Plasticity chart
            axes[0,0].scatter(self.clay_data['liquid_limit_pct'], self.clay_data['plasticity_index'], 
                            alpha=0.6, c=self.clay_data['clay_content_pct'], cmap='viridis')
            # Add A-line
            ll_range = np.linspace(0, 120, 100)
            a_line = 0.73 * (ll_range - 20)
            axes[0,0].plot(ll_range, a_line, 'r--', label='A-line')
            axes[0,0].set_title('Plasticity Chart')
            axes[0,0].set_xlabel('Liquid Limit (%)')
            axes[0,0].set_ylabel('Plasticity Index')
            axes[0,0].legend()
            
            # Strength vs plasticity
            axes[0,1].scatter(self.clay_data['plasticity_index'], 
                            self.clay_data['undrained_shear_strength_kPa'], alpha=0.6)
            axes[0,1].set_title('Strength vs Plasticity')
            axes[0,1].set_xlabel('Plasticity Index')
            axes[0,1].set_ylabel('Undrained Shear Strength (kPa)')
            
            # Mineralogy distribution
            mineral_counts = self.clay_data['dominant_clay_mineral'].value_counts()
            axes[0,2].pie(mineral_counts.values, labels=mineral_counts.index, autopct='%1.1f%%')
            axes[0,2].set_title('Dominant Clay Minerals')
            
            # Sensitivity distribution
            axes[1,0].hist(np.log10(self.clay_data['sensitivity']), bins=30, alpha=0.7, color='red')
            axes[1,0].set_title('Sensitivity Distribution')
            axes[1,0].set_xlabel('Log10(Sensitivity)')
            axes[1,0].set_ylabel('Frequency')
            
            # Compression behavior
            axes[1,1].scatter(self.clay_data['liquid_limit_pct'], self.clay_data['compression_index'], 
                            alpha=0.6, c=self.clay_data['clay_content_pct'], cmap='plasma')
            axes[1,1].set_title('Compression Index vs Liquid Limit')
            axes[1,1].set_xlabel('Liquid Limit (%)')
            axes[1,1].set_ylabel('Compression Index')
            
            # OCR vs strength
            axes[1,2].scatter(self.clay_data['ocr'], self.clay_data['undrained_shear_strength_kPa'], 
                            alpha=0.6)
            axes[1,2].set_title('OCR vs Undrained Strength')
            axes[1,2].set_xlabel('Overconsolidation Ratio')
            axes[1,2].set_ylabel('Undrained Shear Strength (kPa)')
            axes[1,2].set_xscale('log')
            
            plt.tight_layout()
            plt.savefig('geotechnical_datasets/documentation/clay_soil_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Statistical relationships
        self._analyze_clay_correlations()
    
    def analyze_failure_mechanisms(self, save_plots=True):
        """Analysis of failure case studies"""
        
        if self.case_studies is None:
            print("Case study data not loaded")
            return
        
        print("\\n=== FAILURE MECHANISM ANALYSIS ===")
        
        sandy_cases = self.case_studies['sandy']
        clay_cases = self.case_studies['clay']
        
        # Sandy soil failures
        print("\\nSandy Soil Failure Analysis:")
        print(f"Total cases: {len(sandy_cases)}")
        print("Failure types:")
        print(sandy_cases['failure_type'].value_counts())
        
        print(f"\\nAverage damage cost: ${sandy_cases['damage_cost_usd'].mean():,.0f}")
        print(f"Total damage cost: ${sandy_cases['damage_cost_usd'].sum():,.0f}")
        
        # Clay soil failures
        print("\\nClay Soil Failure Analysis:")
        print(f"Total cases: {len(clay_cases)}")
        print("Failure types:")
        print(clay_cases['failure_type'].value_counts())
        
        print(f"\\nAverage damage cost: ${clay_cases['damage_cost_usd'].mean():,.0f}")
        print(f"Total casualties: {clay_cases['casualties'].sum()}")
        
        if save_plots:
            # Create failure analysis plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Failure Mechanism Analysis', fontsize=16)
            
            # Sandy soil failure types
            sandy_failure_counts = sandy_cases['failure_type'].value_counts()
            axes[0,0].bar(range(len(sandy_failure_counts)), sandy_failure_counts.values)
            axes[0,0].set_title('Sandy Soil Failure Types')
            axes[0,0].set_xticks(range(len(sandy_failure_counts)))
            axes[0,0].set_xticklabels(sandy_failure_counts.index, rotation=45, ha='right')
            axes[0,0].set_ylabel('Number of Cases')
            
            # Magnitude vs damage (sandy)
            axes[0,1].scatter(sandy_cases['earthquake_magnitude'], 
                            sandy_cases['damage_cost_usd'], alpha=0.6)
            axes[0,1].set_title('Earthquake Magnitude vs Damage (Sandy)')
            axes[0,1].set_xlabel('Earthquake Magnitude')
            axes[0,1].set_ylabel('Damage Cost (USD)')
            axes[0,1].set_yscale('log')
            
            # Settlement vs relative density
            axes[0,2].scatter(sandy_cases['relative_density_pct'], 
                            sandy_cases['settlement_mm'], alpha=0.6)
            axes[0,2].set_title('Settlement vs Relative Density')
            axes[0,2].set_xlabel('Relative Density (%)')
            axes[0,2].set_ylabel('Settlement (mm)')
            axes[0,2].set_yscale('log')
            
            # Clay soil failure types
            clay_failure_counts = clay_cases['failure_type'].value_counts()
            axes[1,0].bar(range(len(clay_failure_counts)), clay_failure_counts.values)
            axes[1,0].set_title('Clay Soil Failure Types')
            axes[1,0].set_xticks(range(len(clay_failure_counts)))
            axes[1,0].set_xticklabels(clay_failure_counts.index, rotation=45, ha='right')
            axes[1,0].set_ylabel('Number of Cases')
            
            # Factor of safety distribution
            axes[1,1].hist(clay_cases['factor_of_safety'], bins=20, alpha=0.7, color='red')
            axes[1,1].axvline(x=1.0, color='black', linestyle='--', label='FS = 1.0')
            axes[1,1].set_title('Factor of Safety Distribution')
            axes[1,1].set_xlabel('Factor of Safety')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
            
            # Displacement vs slope angle
            axes[1,2].scatter(clay_cases['slope_angle_deg'], 
                            clay_cases['displacement_m'], alpha=0.6)
            axes[1,2].set_title('Displacement vs Slope Angle')
            axes[1,2].set_xlabel('Slope Angle (°)')
            axes[1,2].set_ylabel('Displacement (m)')
            axes[1,2].set_yscale('log')
            
            plt.tight_layout()
            plt.savefig('geotechnical_datasets/documentation/failure_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def predictive_modeling(self, save_plots=True):
        """Develop predictive models for failure mechanisms"""
        
        print("\\n=== PREDICTIVE MODELING ===")
        
        # Sandy soil liquefaction prediction
        if self.sandy_data is not None:
            print("\\nSandy Soil Liquefaction Prediction:")
            
            # Prepare features and target
            X = self.sandy_data[['relative_density_pct', 'silt_content_pct', 'clay_content_pct', 'depth_m']].copy()
            X['fines_content_pct'] = X['silt_content_pct'] + X['clay_content_pct']
            features = ['relative_density_pct', 'fines_content_pct', 'depth_m']
            X = X[features]
            y = self.sandy_data['factor_safety_m7']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"Liquefaction FS Prediction - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            print("Feature importance:")
            for feature, importance in zip(features, rf_model.feature_importances_):
                print(f"  {feature}: {importance:.3f}")
        
        # Clay soil strength prediction
        if self.clay_data is not None:
            print("\\nClay Soil Strength Prediction:")
            
            # Prepare features and target
            features = ['plasticity_index', 'liquid_limit_pct', 'clay_content_pct', 'ocr']
            X = self.clay_data[features]
            y = self.clay_data['undrained_shear_strength_kPa']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"Undrained Strength Prediction - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            print("Feature importance:")
            for feature, importance in zip(features, rf_model.feature_importances_):
                print(f"  {feature}: {importance:.3f}")
    
    def _analyze_sandy_correlations(self):
        """Detailed correlation analysis for sandy soils"""
        
        # Key relationships
        rd_phi_corr = stats.pearsonr(self.sandy_data['relative_density_pct'], 
                                   self.sandy_data['friction_angle_deg_y'])
        print(f"\\nRelative Density vs Friction Angle: r = {rd_phi_corr[0]:.3f}, p = {rd_phi_corr[1]:.3f}")
        
        # Liquefaction relationships
        crr_rd_corr = stats.pearsonr(self.sandy_data['cyclic_resistance_ratio'], 
                                   self.sandy_data['relative_density_pct'])
        print(f"CRR vs Relative Density: r = {crr_rd_corr[0]:.3f}, p = {crr_rd_corr[1]:.3f}")
    
    def _analyze_clay_correlations(self):
        """Detailed correlation analysis for clay soils"""
        
        # Key relationships
        pi_su_corr = stats.pearsonr(self.clay_data['plasticity_index'], 
                                  self.clay_data['undrained_shear_strength_kPa'])
        print(f"\\nPlasticity Index vs Undrained Strength: r = {pi_su_corr[0]:.3f}, p = {pi_su_corr[1]:.3f}")
        
        # Compression relationships
        ll_cc_corr = stats.pearsonr(self.clay_data['liquid_limit_pct'], 
                                  self.clay_data['compression_index'])
        print(f"Liquid Limit vs Compression Index: r = {ll_cc_corr[0]:.3f}, p = {ll_cc_corr[1]:.3f}")

def main():
    """Run comprehensive analysis"""
    
    # Create output directory
    import os
    os.makedirs('geotechnical_datasets/documentation', exist_ok=True)
    
    # Initialize analyzer
    analyzer = GeotechnicalAnalyzer()
    
    # Load data
    if not analyzer.load_datasets():
        print("Failed to load datasets")
        return
    
    # Run analyses
    analyzer.analyze_sandy_soil_properties()
    analyzer.analyze_clay_soil_properties()
    analyzer.analyze_failure_mechanisms()
    analyzer.predictive_modeling()
    
    print("\\n=== ANALYSIS COMPLETE ===")
    print("Generated plots saved in geotechnical_datasets/documentation/")

if __name__ == "__main__":
    main()