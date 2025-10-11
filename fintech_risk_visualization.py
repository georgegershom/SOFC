#!/usr/bin/env python3
"""
FinTech Risk Nexus Visualization Script
Creates comprehensive visualizations for the FinTech risk dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinTechRiskVisualizer:
    """Creates visualizations for FinTech risk nexus dataset"""
    
    def __init__(self, data_dir='/workspace'):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """Load dataset components"""
        self.cyber_risk = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_cyber_risk.csv")
        self.consumer_sentiment = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_consumer_sentiment.csv")
        self.competitive_dynamics = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_competitive_dynamics.csv")
        self.macro_economic = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_macro_economic.csv")
        self.summary_stats = pd.read_csv(f"{self.data_dir}/fintech_risk_nexus_summary_statistics.csv")
    
    def plot_cyber_risk_heatmap(self):
        """Create heatmap of cyber risk across countries and time"""
        plt.figure(figsize=(15, 10))
        
        # Pivot data for heatmap
        cyber_pivot = self.cyber_risk.pivot_table(
            values='cyber_incidents', 
            index='country', 
            columns='year', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(cyber_pivot, annot=True, fmt='.1f', cmap='Reds', 
                   cbar_kws={'label': 'Average Cyber Incidents per Quarter'})
        plt.title('Cyber Risk Heatmap: Incidents by Country and Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/cyber_risk_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sentiment_trends(self):
        """Plot consumer sentiment trends over time"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sentiment by year
        yearly_sentiment = self.consumer_sentiment.groupby('year')['sentiment_score'].mean()
        axes[0, 0].plot(yearly_sentiment.index, yearly_sentiment.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Average Sentiment Score by Year', fontweight='bold')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trust index by year
        yearly_trust = self.consumer_sentiment.groupby('year')['trust_index'].mean()
        axes[0, 1].plot(yearly_trust.index, yearly_trust.values, marker='s', color='green', linewidth=2)
        axes[0, 1].set_title('Average Trust Index by Year', fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Trust Index')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top countries by sentiment
        top_countries_sentiment = self.consumer_sentiment.groupby('country')['sentiment_score'].mean().nlargest(10)
        axes[1, 0].barh(range(len(top_countries_sentiment)), top_countries_sentiment.values)
        axes[1, 0].set_yticks(range(len(top_countries_sentiment)))
        axes[1, 0].set_yticklabels(top_countries_sentiment.index)
        axes[1, 0].set_title('Top 10 Countries by Sentiment Score', fontweight='bold')
        axes[1, 0].set_xlabel('Average Sentiment Score')
        
        # Sentiment distribution
        axes[1, 1].hist(self.consumer_sentiment['sentiment_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Sentiment Scores', fontweight='bold')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/sentiment_trends.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_competitive_dynamics(self):
        """Plot competitive dynamics and market structure"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # HHI distribution by market concentration
        concentration_hhi = self.competitive_dynamics.groupby('market_concentration')['hhi_index'].apply(list)
        axes[0, 0].boxplot([concentration_hhi['Low'], concentration_hhi['Moderate'], concentration_hhi['High']], 
                          labels=['Low', 'Moderate', 'High'])
        axes[0, 0].set_title('HHI Distribution by Market Concentration', fontweight='bold')
        axes[0, 0].set_xlabel('Market Concentration Level')
        axes[0, 0].set_ylabel('HHI Index')
        axes[0, 0].grid(True, alpha=0.3)
        
        # New licenses by year
        yearly_licenses = self.competitive_dynamics.groupby('year')['new_fintech_licenses'].sum()
        axes[0, 1].bar(yearly_licenses.index, yearly_licenses.values, color='skyblue', edgecolor='navy')
        axes[0, 1].set_title('Total New FinTech Licenses by Year', fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Licenses')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Investment vs HHI scatter
        axes[1, 0].scatter(self.competitive_dynamics['hhi_index'], 
                          self.competitive_dynamics['fintech_investment_millions_usd'],
                          alpha=0.6, s=50)
        axes[1, 0].set_title('FinTech Investment vs Market Concentration', fontweight='bold')
        axes[1, 0].set_xlabel('HHI Index')
        axes[1, 0].set_ylabel('Investment (Millions USD)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Digital adoption by country (top 15)
        top_digital = self.competitive_dynamics.groupby('country')['digital_adoption_rate'].mean().nlargest(15)
        axes[1, 1].barh(range(len(top_digital)), top_digital.values, color='lightcoral')
        axes[1, 1].set_yticks(range(len(top_digital)))
        axes[1, 1].set_yticklabels(top_digital.index)
        axes[1, 1].set_title('Top 15 Countries by Digital Adoption Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Digital Adoption Rate')
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/competitive_dynamics.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_risk_correlations(self):
        """Plot correlation matrix of risk factors"""
        # Prepare data for correlation analysis
        merged_data = self.summary_stats.merge(
            self.cyber_risk.groupby('country').agg({
                'cyber_incidents': 'mean',
                'mobile_money_fraud_cases': 'mean'
            }),
            on='country'
        )
        
        merged_data = merged_data.merge(
            self.consumer_sentiment.groupby('country').agg({
                'sentiment_score': 'mean',
                'trust_index': 'mean'
            }),
            on='country'
        )
        
        # Select variables for correlation
        corr_vars = [
            'avg_cyber_incidents_per_quarter',
            'avg_fraud_cases_per_quarter', 
            'avg_sentiment_score',
            'avg_trust_index',
            'avg_hhi_index',
            'avg_gdp_per_capita_usd',
            'avg_financial_inclusion_pct'
        ]
        
        correlation_matrix = merged_data[corr_vars].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Risk Factors Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/risk_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_early_warning_dashboard(self):
        """Create early warning dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Risk level distribution
        risk_counts = self.summary_stats['risk_level'].value_counts()
        axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Risk Level Distribution', fontweight='bold')
        
        # Top 10 countries by cyber incidents
        top_cyber = self.summary_stats.nlargest(10, 'avg_cyber_incidents_per_quarter')
        axes[0, 1].barh(range(len(top_cyber)), top_cyber['avg_cyber_incidents_per_quarter'])
        axes[0, 1].set_yticks(range(len(top_cyber)))
        axes[0, 1].set_yticklabels(top_cyber['country'])
        axes[0, 1].set_title('Top 10 Countries by Cyber Incidents', fontweight='bold')
        axes[0, 1].set_xlabel('Avg Incidents per Quarter')
        
        # Trust vs Sentiment scatter
        axes[0, 2].scatter(self.summary_stats['avg_sentiment_score'], 
                          self.summary_stats['avg_trust_index'], 
                          alpha=0.7, s=60)
        axes[0, 2].set_title('Trust Index vs Sentiment Score', fontweight='bold')
        axes[0, 2].set_xlabel('Average Sentiment Score')
        axes[0, 2].set_ylabel('Average Trust Index')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Financial inclusion vs GDP
        axes[1, 0].scatter(self.summary_stats['avg_gdp_per_capita_usd'], 
                          self.summary_stats['avg_financial_inclusion_pct'],
                          alpha=0.7, s=60, c=self.summary_stats['avg_cyber_incidents_per_quarter'], 
                          cmap='Reds')
        axes[1, 0].set_title('Financial Inclusion vs GDP (colored by cyber risk)', fontweight='bold')
        axes[1, 0].set_xlabel('GDP per Capita (USD)')
        axes[1, 0].set_ylabel('Financial Inclusion (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Market concentration distribution
        concentration_counts = self.competitive_dynamics['market_concentration'].value_counts()
        axes[1, 1].bar(concentration_counts.index, concentration_counts.values, 
                      color=['lightgreen', 'orange', 'red'])
        axes[1, 1].set_title('Market Concentration Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Concentration Level')
        axes[1, 1].set_ylabel('Number of Countries')
        
        # Time series of key indicators
        yearly_data = self.cyber_risk.groupby('year').agg({
            'cyber_incidents': 'mean',
            'mobile_money_fraud_cases': 'mean'
        })
        
        ax2 = axes[1, 2].twinx()
        line1 = axes[1, 2].plot(yearly_data.index, yearly_data['cyber_incidents'], 
                               'b-o', label='Cyber Incidents', linewidth=2)
        line2 = ax2.plot(yearly_data.index, yearly_data['mobile_money_fraud_cases'], 
                        'r-s', label='Fraud Cases', linewidth=2)
        
        axes[1, 2].set_title('Cyber Risk Trends Over Time', fontweight='bold')
        axes[1, 2].set_xlabel('Year')
        axes[1, 2].set_ylabel('Cyber Incidents', color='b')
        ax2.set_ylabel('Fraud Cases', color='r')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1, 2].legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/early_warning_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("Creating FinTech Risk Nexus Visualizations...")
        
        self.plot_cyber_risk_heatmap()
        self.plot_sentiment_trends()
        self.plot_competitive_dynamics()
        self.plot_risk_correlations()
        self.plot_early_warning_dashboard()
        
        print("All visualizations created and saved!")

def main():
    """Main function to create visualizations"""
    visualizer = FinTechRiskVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()