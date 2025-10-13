#!/usr/bin/env python3
"""
Data Analysis Tools for Food Processing FDI Research
Provides statistical analysis and validation tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

class FoodProcessingFDIAnalyzer:
    def __init__(self, dataset_path='food_processing_fdi_dataset.csv'):
        """Initialize analyzer with dataset"""
        self.dataset = pd.read_csv(dataset_path)
        self.results = {}
        
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("=== DESCRIPTIVE STATISTICS ===")
        
        # Basic info
        print(f"Dataset Shape: {self.dataset.shape}")
        print(f"Missing Values: {self.dataset.isnull().sum().sum()}")
        
        # Firm characteristics
        print("\n--- Firm Characteristics ---")
        print("Firm Size Distribution:")
        print(self.dataset['Firm_Size'].value_counts(normalize=True) * 100)
        
        print("\nFDI Presence:")
        print(self.dataset['FDI_Presence'].value_counts(normalize=True) * 100)
        
        print("\nSubsector Distribution:")
        print(self.dataset['Subsector'].value_counts(normalize=True) * 100)
        
        print("\nOwnership Type Distribution:")
        print(self.dataset['Ownership_Type'].value_counts(normalize=True) * 100)
        
        # Performance indicators
        print("\n--- Performance Indicators ---")
        performance_vars = ['ROI_Percent', 'ROA_Percent', 'Export_Intensity', 
                           'Market_Share_Percent', 'Revenue_Growth_Percent']
        
        for var in performance_vars:
            print(f"\n{var}:")
            print(f"  Mean: {self.dataset[var].mean():.2f}")
            print(f"  Std: {self.dataset[var].std():.2f}")
            print(f"  Min: {self.dataset[var].min():.2f}")
            print(f"  Max: {self.dataset[var].max():.2f}")
        
        # FDI constructs
        print("\n--- FDI Constructs (Likert 1-5) ---")
        fdi_vars = ['Knowledge_Absorption', 'Task_Performance', 'Innovation_Score']
        
        for var in fdi_vars:
            print(f"\n{var}:")
            print(f"  Mean: {self.dataset[var].mean():.2f}")
            print(f"  Std: {self.dataset[var].std():.2f}")
        
        # Government policy
        print("\n--- Government Policy (Likert 1-7) ---")
        policy_vars = ['Tax_Incentive_Effectiveness', 'Regulatory_Stability', 
                      'Infrastructure_Support', 'Policy_Effectiveness_Index']
        
        for var in policy_vars:
            print(f"\n{var}:")
            print(f"  Mean: {self.dataset[var].mean():.2f}")
            print(f"  Std: {self.dataset[var].std():.2f}")
    
    def fdi_performance_analysis(self):
        """Analyze relationship between FDI and firm performance"""
        print("\n=== FDI-PERFORMANCE ANALYSIS ===")
        
        # Compare performance by FDI presence
        fdi_firms = self.dataset[self.dataset['FDI_Presence'] == 1]
        non_fdi_firms = self.dataset[self.dataset['FDI_Presence'] == 0]
        
        performance_vars = ['ROI_Percent', 'ROA_Percent', 'Export_Intensity', 
                           'Market_Share_Percent', 'Innovation_Score']
        
        print("\nPerformance Comparison: FDI vs Non-FDI Firms")
        print("-" * 50)
        
        for var in performance_vars:
            fdi_mean = fdi_firms[var].mean()
            non_fdi_mean = non_fdi_firms[var].mean()
            difference = fdi_mean - non_fdi_mean
            
            # T-test
            t_stat, p_value = stats.ttest_ind(fdi_firms[var], non_fdi_firms[var])
            
            print(f"\n{var}:")
            print(f"  FDI Firms Mean: {fdi_mean:.2f}")
            print(f"  Non-FDI Firms Mean: {non_fdi_mean:.2f}")
            print(f"  Difference: {difference:.2f}")
            print(f"  T-statistic: {t_stat:.3f}")
            print(f"  P-value: {p_value:.3f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select key variables for correlation
        key_vars = [
            'FDI_Presence', 'Knowledge_Absorption', 'Task_Performance', 
            'Innovation_Score', 'ROI_Percent', 'ROA_Percent', 'Export_Intensity',
            'Policy_Effectiveness_Index', 'Regulatory_Stability', 'Infrastructure_Support'
        ]
        
        corr_matrix = self.dataset[key_vars].corr()
        
        print("\nKey Correlations:")
        print("-" * 30)
        
        # FDI correlations
        fdi_corrs = corr_matrix['FDI_Presence'].sort_values(ascending=False)
        print("\nFDI Presence Correlations:")
        for var, corr in fdi_corrs.items():
            if var != 'FDI_Presence':
                print(f"  {var}: {corr:.3f}")
        
        # Performance correlations
        roi_corrs = corr_matrix['ROI_Percent'].sort_values(ascending=False)
        print("\nROI Correlations:")
        for var, corr in roi_corrs.items():
            if var != 'ROI_Percent':
                print(f"  {var}: {corr:.3f}")
    
    def regression_analysis(self):
        """Perform basic regression analysis"""
        print("\n=== REGRESSION ANALYSIS ===")
        
        # Prepare data for regression
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Select variables
        X_vars = [
            'FDI_Presence', 'Knowledge_Absorption', 'Task_Performance', 
            'Innovation_Score', 'Policy_Effectiveness_Index', 'Firm_Age',
            'Employees', 'Skilled_Labor_Ratio', 'Technology_Adoption_Score'
        ]
        
        y_var = 'ROI_Percent'
        
        X = self.dataset[X_vars]
        y = self.dataset[y_var]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate R-squared
        r2 = model.score(X_test_scaled, y_test)
        
        print(f"\nROI Regression Model (R² = {r2:.3f})")
        print("-" * 40)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Variable': X_vars,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nFeature Importance (Standardized Coefficients):")
        for _, row in feature_importance.iterrows():
            print(f"  {row['Variable']}: {row['Coefficient']:.3f}")
    
    def generate_visualizations(self):
        """Generate key visualizations"""
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. FDI Performance Comparison
        plt.figure(figsize=fig_size)
        performance_vars = ['ROI_Percent', 'ROA_Percent', 'Export_Intensity', 'Innovation_Score']
        
        fdi_data = []
        non_fdi_data = []
        
        for var in performance_vars:
            fdi_data.append(self.dataset[self.dataset['FDI_Presence'] == 1][var].mean())
            non_fdi_data.append(self.dataset[self.dataset['FDI_Presence'] == 0][var].mean())
        
        x = np.arange(len(performance_vars))
        width = 0.35
        
        plt.bar(x - width/2, fdi_data, width, label='FDI Firms', alpha=0.8)
        plt.bar(x + width/2, non_fdi_data, width, label='Non-FDI Firms', alpha=0.8)
        
        plt.xlabel('Performance Indicators')
        plt.ylabel('Average Score')
        plt.title('Performance Comparison: FDI vs Non-FDI Firms')
        plt.xticks(x, performance_vars, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('fdi_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        key_vars = [
            'FDI_Presence', 'Knowledge_Absorption', 'Innovation_Score', 
            'ROI_Percent', 'Policy_Effectiveness_Index', 'Regulatory_Stability'
        ]
        
        corr_matrix = self.dataset[key_vars].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix: Key Variables')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Government Policy Scores
        plt.figure(figsize=(12, 6))
        policy_vars = [
            'Tax_Incentive_Effectiveness', 'Regulatory_Stability', 
            'Infrastructure_Support', 'Policy_Effectiveness_Index'
        ]
        
        policy_means = [self.dataset[var].mean() for var in policy_vars]
        
        plt.bar(policy_vars, policy_means, alpha=0.7, color='skyblue')
        plt.axhline(y=4, color='red', linestyle='--', alpha=0.7, label='Neutral (4)')
        plt.xlabel('Policy Dimensions')
        plt.ylabel('Average Score (1-7)')
        plt.title('Government Policy Assessment Scores')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('government_policy_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved:")
        print("  - fdi_performance_comparison.png")
        print("  - correlation_heatmap.png") 
        print("  - government_policy_scores.png")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n=== GENERATING ANALYSIS REPORT ===")
        
        report = {
            "Analysis_Summary": {
                "Dataset_Size": f"{self.dataset.shape[0]} firms, {self.dataset.shape[1]} variables",
                "FDI_Firms": int(self.dataset['FDI_Presence'].sum()),
                "Non_FDI_Firms": int((self.dataset['FDI_Presence'] == 0).sum()),
                "Analysis_Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "Key_Findings": {
                "FDI_Performance_Gap": {
                    "ROI_Difference": f"{self.dataset[self.dataset['FDI_Presence']==1]['ROI_Percent'].mean() - self.dataset[self.dataset['FDI_Presence']==0]['ROI_Percent'].mean():.2f}%",
                    "Innovation_Difference": f"{self.dataset[self.dataset['FDI_Presence']==1]['Innovation_Score'].mean() - self.dataset[self.dataset['FDI_Presence']==0]['Innovation_Score'].mean():.2f} points"
                },
                "Government_Policy_Ratings": {
                    "Average_Policy_Effectiveness": f"{self.dataset['Policy_Effectiveness_Index'].mean():.2f}/7",
                    "Infrastructure_Support": f"{self.dataset['Infrastructure_Support'].mean():.2f}/7",
                    "Regulatory_Stability": f"{self.dataset['Regulatory_Stability'].mean():.2f}/7"
                }
            },
            "Recommendations": [
                "FDI firms show higher performance - encourage more foreign investment",
                "Government policy effectiveness is moderate - improve policy implementation",
                "Infrastructure support needs improvement - invest in power and logistics",
                "Focus on innovation and knowledge absorption for better performance"
            ]
        }
        
        # Save report
        with open('analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Analysis report saved to analysis_report.json")
        return report
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Complete Analysis Pipeline...")
        print("=" * 50)
        
        self.descriptive_statistics()
        self.fdi_performance_analysis()
        self.correlation_analysis()
        self.regression_analysis()
        self.generate_visualizations()
        report = self.generate_report()
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        
        return report

def main():
    """Run the complete analysis"""
    try:
        analyzer = FoodProcessingFDIAnalyzer()
        report = analyzer.run_complete_analysis()
        return report
    except FileNotFoundError:
        print("Error: Dataset file not found. Please run the dataset generator first.")
        return None
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    main()