#!/usr/bin/env python3
"""
Financial System & Regulatory Data Analyzer
Creates visualizations and analysis for FinTech Early Warning Model Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

class FinancialDataAnalyzer:
    def __init__(self, data_path='output/financial_system_regulatory_master.csv'):
        """Initialize the analyzer with the master dataset"""
        self.df = pd.read_csv(data_path)
        self.setup_plotting()
        
    def setup_plotting(self):
        """Set up plotting parameters"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def generate_country_profiles(self):
        """Generate country risk profiles"""
        print("Generating country risk profiles...")
        
        # Calculate country averages
        country_stats = self.df.groupby('country_name').agg({
            'bank_npl_ratio': ['mean', 'std'],
            'bank_roa': ['mean', 'std'],
            'domestic_credit_private': ['mean', 'std'],
            'regulatory_quality': ['mean', 'std'],
            'digital_lending_regulation_dummy': 'max',
            'open_banking_initiative_dummy': 'max',
            'fintech_regulatory_sandbox_dummy': 'max'
        }).round(3)
        
        # Flatten column names
        country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns]
        
        # Create risk score (higher NPL, lower regulatory quality = higher risk)
        country_stats['risk_score'] = (
            country_stats['bank_npl_ratio_mean'] * 0.4 +
            (1 - (country_stats['regulatory_quality_mean'] + 2) / 4) * 0.3 +  # Normalize reg quality
            (1 / (country_stats['bank_roa_mean'] + 0.1)) * 0.3  # Inverse ROA
        )
        
        # Create development score
        country_stats['development_score'] = (
            country_stats['domestic_credit_private_mean'] * 0.4 +
            (country_stats['regulatory_quality_mean'] + 2) * 25 * 0.3 +  # Normalize to 0-100
            (country_stats['digital_lending_regulation_dummy_max'] + 
             country_stats['open_banking_initiative_dummy_max'] + 
             country_stats['fintech_regulatory_sandbox_dummy_max']) * 10 * 0.3
        )
        
        country_stats = country_stats.sort_values('risk_score')
        return country_stats
    
    def create_time_series_plots(self):
        """Create time series plots for key indicators"""
        print("Creating time series visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial System Indicators Over Time by Country', fontsize=16, fontweight='bold')
        
        indicators = [
            ('bank_npl_ratio', 'Bank Non-Performing Loans (%)', axes[0,0]),
            ('bank_roa', 'Bank Return on Assets (%)', axes[0,1]),
            ('domestic_credit_private', 'Domestic Credit to Private Sector (% GDP)', axes[1,0]),
            ('regulatory_quality', 'Regulatory Quality Index', axes[1,1])
        ]
        
        for indicator, title, ax in indicators:
            for country in self.df['country_name'].unique():
                country_data = self.df[self.df['country_name'] == country]
                ax.plot(country_data['year'], country_data[indicator], 
                       marker='o', linewidth=2, label=country, alpha=0.7)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Year')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('output/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_correlation_heatmap(self):
        """Create correlation heatmap of financial indicators"""
        print("Creating correlation analysis...")
        
        # Select numeric columns for correlation
        numeric_cols = ['bank_npl_ratio', 'bank_roa', 'domestic_credit_private', 
                       'regulatory_quality', 'digital_lending_regulation_dummy',
                       'open_banking_initiative_dummy', 'fintech_regulatory_sandbox_dummy']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix: Financial System & Regulatory Indicators', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_country_comparison_radar(self):
        """Create radar chart comparing countries"""
        print("Creating country comparison radar charts...")
        
        # Select top 6 countries by economic size/importance
        top_countries = ['Nigeria', 'South Africa', 'Kenya', 'Ghana', 'Ethiopia', 'Rwanda']
        
        # Normalize indicators to 0-1 scale for radar chart
        indicators = ['bank_npl_ratio', 'bank_roa', 'domestic_credit_private', 'regulatory_quality']
        
        country_means = self.df[self.df['country_name'].isin(top_countries)].groupby('country_name')[indicators].mean()
        
        # Normalize (invert NPL ratio as lower is better)
        normalized_data = country_means.copy()
        normalized_data['bank_npl_ratio'] = 1 - (normalized_data['bank_npl_ratio'] - normalized_data['bank_npl_ratio'].min()) / (normalized_data['bank_npl_ratio'].max() - normalized_data['bank_npl_ratio'].min())
        normalized_data['bank_roa'] = (normalized_data['bank_roa'] - normalized_data['bank_roa'].min()) / (normalized_data['bank_roa'].max() - normalized_data['bank_roa'].min())
        normalized_data['domestic_credit_private'] = (normalized_data['domestic_credit_private'] - normalized_data['domestic_credit_private'].min()) / (normalized_data['domestic_credit_private'].max() - normalized_data['domestic_credit_private'].min())
        normalized_data['regulatory_quality'] = (normalized_data['regulatory_quality'] - normalized_data['regulatory_quality'].min()) / (normalized_data['regulatory_quality'].max() - normalized_data['regulatory_quality'].min())
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for country in normalized_data.index:
            values = normalized_data.loc[country].values
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=country)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['NPL Quality\n(inverted)', 'Bank ROA', 'Credit Depth', 'Regulatory Quality'])
        ax.set_ylim(0, 1)
        ax.set_title('Country Financial System Comparison\n(Normalized Indicators)', 
                    fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('output/country_radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_regulatory_timeline(self):
        """Create timeline of regulatory implementations"""
        print("Creating regulatory implementation timeline...")
        
        # Get regulatory implementation data
        reg_data = []
        for country in self.df['country_name'].unique():
            country_data = self.df[self.df['country_name'] == country]
            
            # Find first year each regulation was implemented
            for reg_type in ['digital_lending_regulation_dummy', 'open_banking_initiative_dummy', 'fintech_regulatory_sandbox_dummy']:
                first_impl = country_data[country_data[reg_type] == 1]['year'].min()
                if not pd.isna(first_impl):
                    reg_data.append({
                        'country': country,
                        'regulation': reg_type.replace('_dummy', '').replace('_', ' ').title(),
                        'year': first_impl
                    })
        
        reg_df = pd.DataFrame(reg_data)
        
        if not reg_df.empty:
            plt.figure(figsize=(14, 8))
            
            # Create timeline plot
            for i, reg_type in enumerate(reg_df['regulation'].unique()):
                reg_subset = reg_df[reg_df['regulation'] == reg_type]
                plt.scatter(reg_subset['year'], [i] * len(reg_subset), 
                           s=100, alpha=0.7, label=reg_type)
                
                # Add country labels
                for _, row in reg_subset.iterrows():
                    plt.annotate(row['country'], (row['year'], i), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.8)
            
            plt.yticks(range(len(reg_df['regulation'].unique())), reg_df['regulation'].unique())
            plt.xlabel('Year of Implementation')
            plt.title('Timeline of FinTech Regulatory Implementations in Sub-Saharan Africa', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('output/regulatory_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_risk_assessment_report(self):
        """Generate comprehensive risk assessment report"""
        print("Generating risk assessment report...")
        
        # Calculate latest year data for each country
        latest_data = self.df.loc[self.df.groupby('country_name')['year'].idxmax()]
        
        # Risk scoring
        latest_data['financial_stability_score'] = (
            (20 - latest_data['bank_npl_ratio']) * 0.4 +  # Lower NPL is better
            latest_data['bank_roa'] * 5 * 0.3 +  # Higher ROA is better
            (latest_data['regulatory_quality'] + 2) * 25 * 0.3  # Higher reg quality is better
        )
        
        latest_data['fintech_readiness_score'] = (
            latest_data['domestic_credit_private'] * 0.4 +
            (latest_data['digital_lending_regulation_dummy'] + 
             latest_data['open_banking_initiative_dummy'] + 
             latest_data['fintech_regulatory_sandbox_dummy']) * 20 * 0.6
        )
        
        # Create risk categories
        latest_data['risk_category'] = pd.cut(
            latest_data['financial_stability_score'], 
            bins=[0, 30, 60, 100], 
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        latest_data['fintech_readiness'] = pd.cut(
            latest_data['fintech_readiness_score'], 
            bins=[0, 40, 70, 100], 
            labels=['Low Readiness', 'Medium Readiness', 'High Readiness']
        )
        
        # Save risk assessment
        risk_report = latest_data[['country_name', 'year', 'bank_npl_ratio', 'bank_roa', 
                                  'domestic_credit_private', 'regulatory_quality',
                                  'financial_stability_score', 'fintech_readiness_score',
                                  'risk_category', 'fintech_readiness']].round(2)
        
        risk_report = risk_report.sort_values('financial_stability_score', ascending=False)
        risk_report.to_csv('output/risk_assessment_report.csv', index=False)
        
        return risk_report
    
    def run_complete_analysis(self):
        """Run complete analysis and generate all outputs"""
        print("=== Starting Comprehensive Financial Data Analysis ===")
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Generate all analyses
        country_profiles = self.generate_country_profiles()
        country_profiles.to_csv('output/country_profiles.csv')
        
        self.create_time_series_plots()
        self.create_correlation_heatmap()
        self.create_country_comparison_radar()
        self.create_regulatory_timeline()
        
        risk_report = self.generate_risk_assessment_report()
        
        # Print summary
        print("\n=== Analysis Summary ===")
        print(f"Countries analyzed: {len(self.df['country_name'].unique())}")
        print(f"Time period: {self.df['year'].min()}-{self.df['year'].max()}")
        print(f"Total observations: {len(self.df)}")
        
        print("\nTop 3 Most Stable Countries (Latest Data):")
        top_stable = risk_report.head(3)[['country_name', 'financial_stability_score', 'risk_category']]
        for _, row in top_stable.iterrows():
            print(f"  {row['country_name']}: {row['financial_stability_score']:.1f} ({row['risk_category']})")
        
        print("\nTop 3 FinTech Ready Countries:")
        top_ready = risk_report.nlargest(3, 'fintech_readiness_score')[['country_name', 'fintech_readiness_score', 'fintech_readiness']]
        for _, row in top_ready.iterrows():
            print(f"  {row['country_name']}: {row['fintech_readiness_score']:.1f} ({row['fintech_readiness']})")
        
        print("\n=== Files Generated ===")
        output_files = [
            'financial_system_regulatory_master.csv',
            'country_profiles.csv', 
            'risk_assessment_report.csv',
            'summary_statistics.csv',
            'time_series_analysis.png',
            'correlation_heatmap.png',
            'country_radar_comparison.png',
            'regulatory_timeline.png'
        ]
        
        for file in output_files:
            if os.path.exists(f'output/{file}'):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (missing)")

def main():
    analyzer = FinancialDataAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()