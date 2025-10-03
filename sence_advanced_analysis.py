#!/usr/bin/env python3
"""
Advanced SENCE Framework Statistical Analysis and Model Validation
Enhanced vulnerability assessment with comprehensive statistical modeling

This module provides advanced statistical analysis, model validation,
and comprehensive reporting for the SENCE framework vulnerability assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SENCEAdvancedAnalysis:
    """
    Advanced SENCE Framework Statistical Analysis Class
    Provides comprehensive statistical modeling and validation
    """
    
    def __init__(self):
        self.cities = ['Port Harcourt', 'Warri', 'Bonny']
        self.domains = ['Environmental', 'Economic', 'Social', 'Governance', 'Infrastructure']
        
        # Initialize comprehensive dataset
        self._initialize_comprehensive_data()
        
    def _initialize_comprehensive_data(self):
        """Initialize comprehensive dataset with statistical properties"""
        
        # Base vulnerability data
        self.vulnerability_data = {
            'Port Harcourt': {
                'Environmental': 0.45, 'Economic': 0.52, 'Social': 0.48,
                'Governance': 0.41, 'Infrastructure': 0.38, 'CVI': 0.52
            },
            'Warri': {
                'Environmental': 0.68, 'Economic': 0.71, 'Social': 0.65,
                'Governance': 0.58, 'Infrastructure': 0.62, 'CVI': 0.61
            },
            'Bonny': {
                'Environmental': 0.89, 'Economic': 0.76, 'Social': 0.54,
                'Governance': 0.47, 'Infrastructure': 0.51, 'CVI': 0.59
            }
        }
        
        # Generate synthetic detailed data for statistical analysis
        np.random.seed(42)
        self.detailed_data = self._generate_detailed_dataset()
        
    def _generate_detailed_dataset(self):
        """Generate detailed synthetic dataset for comprehensive analysis"""
        
        # Create detailed indicators for each domain
        indicators = {
            'Environmental': ['Oil_Spill_Impact', 'Gas_Flaring_Intensity', 'Vegetation_Health',
                            'Water_Quality', 'Land_Degradation', 'Mangrove_Loss'],
            'Economic': ['Unemployment_Rate', 'Income_Diversity', 'Infrastructure_Access',
                        'Employment_Opportunities', 'Economic_Resilience', 'Poverty_Level'],
            'Social': ['Education_Access', 'Healthcare_Access', 'Community_Cohesion',
                      'Safety_Index', 'Social_Capital', 'Governance_Trust'],
            'Governance': ['Institutional_Trust', 'Policy_Effectiveness', 'Transparency',
                          'Accountability', 'Participation', 'Rule_of_Law'],
            'Infrastructure': ['Water_Supply', 'Electricity_Access', 'Transportation',
                             'Communication', 'Housing_Quality', 'Sanitation']
        }
        
        detailed_data = []
        
        for city in self.cities:
            base_cvi = self.vulnerability_data[city]['CVI']
            
            for domain in self.domains:
                domain_value = self.vulnerability_data[city][domain]
                
                for indicator in indicators[domain]:
                    # Generate realistic values with appropriate correlations
                    base_value = domain_value + np.random.normal(0, 0.1)
                    base_value = np.clip(base_value, 0, 1)
                    
                    detailed_data.append({
                        'City': city,
                        'Domain': domain,
                        'Indicator': indicator,
                        'Value': base_value,
                        'CVI': base_cvi,
                        'Normalized_Value': (base_value - 0) / (1 - 0)  # 0-1 normalization
                    })
        
        return pd.DataFrame(detailed_data)
    
    def perform_pca_analysis(self):
        """Perform comprehensive PCA analysis"""
        
        # Prepare data for PCA
        pivot_data = self.detailed_data.pivot_table(
            index=['City', 'Domain'], 
            columns='Indicator', 
            values='Value', 
            fill_value=0
        ).reset_index()
        
        # Separate features and target
        feature_columns = [col for col in pivot_data.columns if col not in ['City', 'Domain']]
        X = pivot_data[feature_columns]
        
        # Get CVI values from the detailed data
        cvi_data = self.detailed_data.groupby('City')['CVI'].first()
        y = cvi_data.values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Create PCA results
        pca_results = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'components': pca.components_,
            'feature_names': feature_columns,
            'n_components_95': np.argmax(cumulative_variance >= 0.95) + 1
        }
        
        return pca_results, X_pca, y
    
    def perform_correlation_analysis(self):
        """Perform comprehensive correlation analysis"""
        
        # Domain-level correlations
        domain_data = pd.DataFrame(self.vulnerability_data).T
        domain_correlations = domain_data.corr()
        
        # Detailed indicator correlations
        indicator_correlations = self.detailed_data.pivot_table(
            index=['City', 'Domain'], 
            columns='Indicator', 
            values='Value'
        ).corr()
        
        # Cross-domain correlations
        cross_domain_corr = {}
        for i, domain1 in enumerate(self.domains):
            for j, domain2 in enumerate(self.domains):
                if i < j:
                    domain1_data = self.detailed_data[
                        self.detailed_data['Domain'] == domain1
                    ]['Value'].values
                    domain2_data = self.detailed_data[
                        self.detailed_data['Domain'] == domain2
                    ]['Value'].values
                    
                    pearson_corr, pearson_p = pearsonr(domain1_data, domain2_data)
                    spearman_corr, spearman_p = spearmanr(domain1_data, domain2_data)
                    kendall_corr, kendall_p = kendalltau(domain1_data, domain2_data)
                    
                    cross_domain_corr[f"{domain1}-{domain2}"] = {
                        'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                        'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
                        'kendall': {'correlation': kendall_corr, 'p_value': kendall_p}
                    }
        
        return {
            'domain_correlations': domain_correlations,
            'indicator_correlations': indicator_correlations,
            'cross_domain_correlations': cross_domain_corr
        }
    
    def perform_model_validation(self):
        """Perform comprehensive model validation"""
        
        # Prepare data - aggregate by city to get one row per city
        city_data = self.detailed_data.groupby(['City', 'Domain'])['Value'].mean().unstack(fill_value=0)
        
        # Get CVI values from the detailed data
        cvi_data = self.detailed_data.groupby('City')['CVI'].first()
        
        # Ensure we have matching indices
        common_cities = city_data.index.intersection(cvi_data.index)
        X = city_data.loc[common_cities]
        y = cvi_data.loc[common_cities].values
        
        # Initialize models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        }
        
        # Cross-validation setup - use fewer splits for small dataset
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        
        validation_results = {}
        
        for model_name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            # Fit model and get predictions
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            else:
                feature_importance = None
            
            validation_results[model_name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'actual': y
            }
        
        return validation_results
    
    def perform_sensitivity_analysis(self):
        """Perform sensitivity analysis on model parameters"""
        
        # Base parameters
        base_params = {
            'Environmental_weight': 0.25,
            'Economic_weight': 0.25,
            'Social_weight': 0.25,
            'Governance_weight': 0.125,
            'Infrastructure_weight': 0.125
        }
        
        sensitivity_results = {}
        
        # Vary each parameter by ±20%
        for param, base_value in base_params.items():
            variations = [base_value * 0.8, base_value, base_value * 1.2]
            param_results = []
            
            for variation in variations:
                # Calculate CVI with modified weights
                modified_weights = base_params.copy()
                modified_weights[param] = variation
                
                # Normalize weights
                total_weight = sum(modified_weights.values())
                normalized_weights = {k: v/total_weight for k, v in modified_weights.items()}
                
                # Calculate CVI for each city
                city_cvis = {}
                for city in self.cities:
                    cvi = sum(
                        self.vulnerability_data[city][domain.replace('_weight', '')] * weight
                        for domain, weight in normalized_weights.items()
                    )
                    city_cvis[city] = cvi
                
                param_results.append({
                    'variation': variation,
                    'normalized_weight': normalized_weights[param],
                    'city_cvis': city_cvis
                })
            
            sensitivity_results[param] = param_results
        
        return sensitivity_results
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualization suite"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. PCA Analysis Plot
        ax1 = plt.subplot(3, 3, 1)
        pca_results, X_pca, y = self.perform_pca_analysis()
        
        plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1),
                pca_results['explained_variance_ratio'], 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Cumulative Variance Plot
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(range(1, len(pca_results['cumulative_variance']) + 1),
                pca_results['cumulative_variance'], 'go-', linewidth=2, markersize=8)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Variance Explained')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Correlation Heatmap
        ax3 = plt.subplot(3, 3, 3)
        correlation_results = self.perform_correlation_analysis()
        sns.heatmap(correlation_results['domain_correlations'], annot=True, 
                   cmap='RdBu_r', center=0, square=True, ax=ax3)
        plt.title('Domain Correlation Matrix')
        
        # 4. Model Validation Results
        ax4 = plt.subplot(3, 3, 4)
        validation_results = self.perform_model_validation()
        
        model_names = list(validation_results.keys())
        cv_means = [validation_results[model]['cv_mean'] for model in model_names]
        cv_stds = [validation_results[model]['cv_std'] for model in model_names]
        
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        plt.ylabel('Cross-Validation R² Score')
        plt.title('Model Validation Results')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Feature Importance
        ax5 = plt.subplot(3, 3, 5)
        rf_results = validation_results['Random Forest']
        if rf_results['feature_importance']:
            features = list(rf_results['feature_importance'].keys())[:10]  # Top 10
            importances = [rf_results['feature_importance'][f] for f in features]
            
            plt.barh(features, importances, alpha=0.7)
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importances')
            plt.grid(True, alpha=0.3)
        
        # 6. Sensitivity Analysis
        ax6 = plt.subplot(3, 3, 6)
        sensitivity_results = self.perform_sensitivity_analysis()
        
        # Plot sensitivity for Environmental weight
        env_results = sensitivity_results['Environmental_weight']
        variations = [r['variation'] for r in env_results]
        port_harcourt_cvis = [r['city_cvis']['Port Harcourt'] for r in env_results]
        
        plt.plot(variations, port_harcourt_cvis, 'o-', linewidth=2, markersize=8, label='Port Harcourt')
        plt.xlabel('Environmental Weight')
        plt.ylabel('CVI Value')
        plt.title('Sensitivity Analysis: Environmental Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Residual Analysis
        ax7 = plt.subplot(3, 3, 7)
        y_actual = rf_results['actual']
        y_pred = rf_results['predictions']
        residuals = y_actual - y_pred
        
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        plt.grid(True, alpha=0.3)
        
        # 8. Q-Q Plot
        ax8 = plt.subplot(3, 3, 8)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        # 9. Model Performance Metrics
        ax9 = plt.subplot(3, 3, 9)
        metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
        values = [rf_results['r2'], rf_results['rmse'], rf_results['mae'], rf_results['mape']]
        
        bars = plt.bar(metrics, values, alpha=0.7, color=['green', 'red', 'orange', 'purple'])
        plt.ylabel('Metric Value')
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self):
        """Generate comprehensive statistical analysis report"""
        
        print("="*100)
        print("SENCE FRAMEWORK COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        print("Advanced Vulnerability Assessment with Model Validation")
        print("="*100)
        
        # PCA Analysis
        print("\n1. PRINCIPAL COMPONENT ANALYSIS (PCA)")
        print("-" * 60)
        pca_results, X_pca, y = self.perform_pca_analysis()
        
        print(f"Total variance explained by first 3 components: {pca_results['cumulative_variance'][2]:.3f}")
        print(f"Components needed for 95% variance: {pca_results['n_components_95']}")
        print(f"First component explains: {pca_results['explained_variance_ratio'][0]:.3f} of variance")
        
        # Correlation Analysis
        print("\n2. CORRELATION ANALYSIS")
        print("-" * 60)
        correlation_results = self.perform_correlation_analysis()
        
        print("Domain Correlations:")
        domain_corr = correlation_results['domain_correlations']
        for i, domain1 in enumerate(self.domains):
            for j, domain2 in enumerate(self.domains):
                if i < j:
                    corr = domain_corr.loc[domain1, domain2]
                    print(f"  {domain1} - {domain2}: {corr:.3f}")
        
        # Model Validation
        print("\n3. MODEL VALIDATION RESULTS")
        print("-" * 60)
        validation_results = self.perform_model_validation()
        
        for model_name, results in validation_results.items():
            print(f"\n{model_name}:")
            print(f"  Cross-validation R²: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
            print(f"  R² Score: {results['r2']:.3f}")
            print(f"  RMSE: {results['rmse']:.3f}")
            print(f"  MAE: {results['mae']:.3f}")
            print(f"  MAPE: {results['mape']:.1f}%")
        
        # Sensitivity Analysis
        print("\n4. SENSITIVITY ANALYSIS")
        print("-" * 60)
        sensitivity_results = self.perform_sensitivity_analysis()
        
        for param, results in sensitivity_results.items():
            print(f"\n{param}:")
            for result in results:
                variation = result['variation']
                weight = result['normalized_weight']
                print(f"  Weight {variation:.3f} -> Normalized {weight:.3f}")
        
        # Statistical Significance Tests
        print("\n5. STATISTICAL SIGNIFICANCE TESTS")
        print("-" * 60)
        
        # ANOVA test for city differences
        city_data = [self.detailed_data[self.detailed_data['City'] == city]['Value'].values 
                    for city in self.cities]
        f_stat, p_value = stats.f_oneway(*city_data)
        
        print(f"ANOVA Test for City Differences:")
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {p_value:.3f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # T-tests between cities
        print(f"\nPairwise T-tests:")
        for i, city1 in enumerate(self.cities):
            for j, city2 in enumerate(self.cities):
                if i < j:
                    data1 = self.detailed_data[self.detailed_data['City'] == city1]['Value'].values
                    data2 = self.detailed_data[self.detailed_data['City'] == city2]['Value'].values
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    print(f"  {city1} vs {city2}: t={t_stat:.3f}, p={p_val:.3f}")
        
        print("\n" + "="*100)

def main():
    """Main execution function for advanced analysis"""
    
    # Initialize advanced analysis
    analysis = SENCEAdvancedAnalysis()
    
    # Generate comprehensive report
    analysis.generate_comprehensive_report()
    
    # Create comprehensive visualization
    print("\nGenerating comprehensive statistical visualization...")
    fig = analysis.create_comprehensive_visualization()
    fig.savefig('/workspace/sence_advanced_statistical_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("\nAdvanced statistical analysis complete!")
    print("Files saved:")
    print("- sence_advanced_statistical_analysis.png")
    
    # Display the visualization
    plt.show()

if __name__ == "__main__":
    main()