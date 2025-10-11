"""
Simplified visualization script for Financial System & Regulatory Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_summary_charts():
    """Create summary charts of the data"""
    
    # Load data
    df = pd.read_csv('../data/financial_system_regulatory_data.csv')
    df_annual = pd.read_csv('../data/financial_system_annual_summary.csv')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Banking Health Trends for Key Countries
    ax1 = plt.subplot(3, 3, 1)
    key_countries = ['KEN', 'NGA', 'ZAF', 'GHA', 'RWA']
    for country in key_countries:
        country_data = df_annual[df_annual['Country_Code'] == country]
        ax1.plot(country_data['Year'], country_data['Bank_NPL_to_Total_Loans_%'], 
                label=country, marker='o', markersize=4)
    ax1.set_title('NPL Trends - Key Countries', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('NPL %')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Credit to GDP Evolution
    ax2 = plt.subplot(3, 3, 2)
    for country in key_countries:
        country_data = df_annual[df_annual['Country_Code'] == country]
        ax2.plot(country_data['Year'], country_data['Domestic_Credit_to_Private_Sector_%_GDP'], 
                label=country, marker='s', markersize=4)
    ax2.set_title('Credit to Private Sector (% GDP)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('% of GDP')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Financial Inclusion Growth
    ax3 = plt.subplot(3, 3, 3)
    for country in key_countries:
        country_data = df_annual[df_annual['Country_Code'] == country]
        ax3.plot(country_data['Year'], country_data['Account_Ownership_%_Adults'], 
                label=country, marker='^', markersize=4)
    ax3.set_title('Account Ownership (% Adults)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('% of Adults')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Mobile Money Adoption
    ax4 = plt.subplot(3, 3, 4)
    for country in key_countries:
        country_data = df_annual[df_annual['Country_Code'] == country]
        ax4.plot(country_data['Year'], country_data['Mobile_Money_Account_%_Adults'], 
                label=country, marker='d', markersize=4)
    ax4.set_title('Mobile Money Accounts (% Adults)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('% of Adults')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Regulatory Quality Distribution (2024)
    ax5 = plt.subplot(3, 3, 5)
    df_2024 = df_annual[df_annual['Year'] == 2024]
    ax5.hist(df_2024['WGI_Regulatory_Quality'], bins=15, color='steelblue', edgecolor='black')
    ax5.set_title('Regulatory Quality Distribution (2024)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('WGI Regulatory Quality Score')
    ax5.set_ylabel('Number of Countries')
    ax5.grid(True, alpha=0.3)
    
    # 6. Banking Health Index (2024)
    ax6 = plt.subplot(3, 3, 6)
    top_10 = df_2024.nlargest(10, 'Banking_Health_Index')
    ax6.barh(range(len(top_10)), top_10['Banking_Health_Index'].values, color='green')
    ax6.set_yticks(range(len(top_10)))
    ax6.set_yticklabels(top_10['Country_Code'].values)
    ax6.set_title('Top 10 Banking Health Index (2024)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Banking Health Index')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Financial Development Index (2024)
    ax7 = plt.subplot(3, 3, 7)
    top_10_dev = df_2024.nlargest(10, 'Financial_Development_Index')
    ax7.barh(range(len(top_10_dev)), top_10_dev['Financial_Development_Index'].values, color='blue')
    ax7.set_yticks(range(len(top_10_dev)))
    ax7.set_yticklabels(top_10_dev['Country_Code'].values)
    ax7.set_title('Top 10 Financial Development (2024)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Financial Development Index')
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Systemic Risk Score Distribution
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(df_2024['Systemic_Risk_Score'], bins=15, color='red', alpha=0.7, edgecolor='black')
    ax8.set_title('Systemic Risk Distribution (2024)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Systemic Risk Score')
    ax8.set_ylabel('Number of Countries')
    ax8.axvline(df_2024['Systemic_Risk_Score'].mean(), color='darkred', linestyle='--', 
                label=f'Mean: {df_2024["Systemic_Risk_Score"].mean():.1f}')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Correlation Heatmap
    ax9 = plt.subplot(3, 3, 9)
    correlation_cols = ['Bank_NPL_to_Total_Loans_%', 'Bank_Z_Score', 'Bank_ROA_%',
                       'Account_Ownership_%_Adults', 'Mobile_Money_Account_%_Adults',
                       'Financial_Development_Index', 'Systemic_Risk_Score']
    corr_matrix = df_2024[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax9)
    ax9.set_title('Key Indicators Correlation (2024)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Financial System & Regulatory Data - Sub-Saharan Africa', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('../outputs/financial_system_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Summary charts saved to financial_system_summary.png")
    
    return fig

def create_regulatory_summary():
    """Create regulatory adoption summary"""
    df_annual = pd.read_csv('../data/financial_system_annual_summary.csv')
    df_2024 = df_annual[df_annual['Year'] == 2024]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate total regulations
    df_2024['Total_Regs'] = (df_2024['Digital_Lending_Regulation'] + 
                             df_2024['Mobile_Money_Regulation'] + 
                             df_2024['Data_Protection_Law'] + 
                             df_2024['Regulatory_Sandbox'])
    
    # Left plot: Regulation adoption by type
    reg_types = ['Digital_Lending_Regulation', 'Mobile_Money_Regulation', 
                'Data_Protection_Law', 'Regulatory_Sandbox']
    reg_labels = ['Digital\nLending', 'Mobile\nMoney', 'Data\nProtection', 'Regulatory\nSandbox']
    reg_counts = [df_2024[col].sum() for col in reg_types]
    
    bars1 = axes[0].bar(reg_labels, reg_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0].set_title('FinTech Regulation Adoption (2024)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Countries')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, reg_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(int(val)), ha='center', fontweight='bold')
    
    # Right plot: Distribution by number of regulations
    reg_distribution = df_2024['Total_Regs'].value_counts().sort_index()
    bars2 = axes[1].bar(reg_distribution.index, reg_distribution.values, 
                       color='steelblue', edgecolor='black')
    axes[1].set_title('Countries by Number of Regulations (2024)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of FinTech Regulations')
    axes[1].set_ylabel('Number of Countries')
    axes[1].set_xticks(range(5))
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, reg_distribution.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(val), ha='center', fontweight='bold')
    
    plt.suptitle('Regulatory Landscape in Sub-Saharan Africa', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('../outputs/regulatory_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Regulatory summary saved to regulatory_summary.png")
    
    return fig

def generate_data_report():
    """Generate a summary report of the dataset"""
    df = pd.read_csv('../data/financial_system_regulatory_data.csv')
    df_annual = pd.read_csv('../data/financial_system_annual_summary.csv')
    
    print("\n" + "=" * 60)
    print("FINANCIAL SYSTEM & REGULATORY DATA REPORT")
    print("Sub-Saharan Africa FinTech Risk Analysis")
    print("=" * 60)
    
    print("\n📊 DATASET STATISTICS:")
    print(f"  • Total Records: {len(df):,}")
    print(f"  • Countries: {df['Country_Code'].nunique()}")
    print(f"  • Time Period: {df['Year'].min()} - {df['Year'].max()} (Quarterly)")
    print(f"  • Variables: {len(df.columns)}")
    
    # Get 2024 statistics
    df_2024 = df_annual[df_annual['Year'] == 2024]
    
    print("\n💰 BANKING SECTOR (2024 Average):")
    print(f"  • Non-Performing Loans: {df_2024['Bank_NPL_to_Total_Loans_%'].mean():.2f}%")
    print(f"  • Bank Z-Score: {df_2024['Bank_Z_Score'].mean():.2f}")
    print(f"  • Return on Assets: {df_2024['Bank_ROA_%'].mean():.2f}%")
    print(f"  • Credit to Private Sector: {df_2024['Domestic_Credit_to_Private_Sector_%_GDP'].mean():.1f}% of GDP")
    
    print("\n📱 FINANCIAL INCLUSION (2024 Average):")
    print(f"  • Account Ownership: {df_2024['Account_Ownership_%_Adults'].mean():.1f}% of adults")
    print(f"  • Mobile Money Accounts: {df_2024['Mobile_Money_Account_%_Adults'].mean():.1f}% of adults")
    
    print("\n📋 REGULATORY ADOPTION (2024):")
    print(f"  • Digital Lending Regulations: {df_2024['Digital_Lending_Regulation'].sum():.0f} countries")
    print(f"  • Mobile Money Regulations: {df_2024['Mobile_Money_Regulation'].sum():.0f} countries")
    print(f"  • Data Protection Laws: {df_2024['Data_Protection_Law'].sum():.0f} countries")
    print(f"  • Regulatory Sandboxes: {df_2024['Regulatory_Sandbox'].sum():.0f} countries")
    
    print("\n🏆 TOP 5 COUNTRIES BY FINANCIAL DEVELOPMENT (2024):")
    top_5 = df_2024.nlargest(5, 'Financial_Development_Index')[['Country_Name', 'Financial_Development_Index']]
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {idx}. {row['Country_Name']}: {row['Financial_Development_Index']:.1f}")
    
    print("\n⚠️ HIGH RISK COUNTRIES (Systemic Risk > 70, 2024):")
    high_risk = df_2024[df_2024['Systemic_Risk_Score'] > 70][['Country_Name', 'Systemic_Risk_Score']]
    if len(high_risk) > 0:
        for _, row in high_risk.iterrows():
            print(f"  • {row['Country_Name']}: {row['Systemic_Risk_Score']:.1f}")
    else:
        print("  • No countries with systemic risk > 70")
    
    # Calculate growth rates
    df_2010 = df_annual[df_annual['Year'] == 2010]
    
    print("\n📈 KEY IMPROVEMENTS (2010-2024):")
    npl_change = df_2024['Bank_NPL_to_Total_Loans_%'].mean() - df_2010['Bank_NPL_to_Total_Loans_%'].mean()
    account_change = df_2024['Account_Ownership_%_Adults'].mean() - df_2010['Account_Ownership_%_Adults'].mean()
    mobile_change = df_2024['Mobile_Money_Account_%_Adults'].mean() - df_2010['Mobile_Money_Account_%_Adults'].mean()
    credit_change = df_2024['Domestic_Credit_to_Private_Sector_%_GDP'].mean() - df_2010['Domestic_Credit_to_Private_Sector_%_GDP'].mean()
    
    print(f"  • NPL Ratio: {npl_change:+.1f} percentage points")
    print(f"  • Account Ownership: {account_change:+.1f} percentage points")
    print(f"  • Mobile Money Adoption: {mobile_change:+.1f} percentage points")
    print(f"  • Credit to Private Sector: {credit_change:+.1f} percentage points")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

def main():
    """Main execution function"""
    print("\nGenerating Financial System & Regulatory Data Visualizations...")
    print("-" * 60)
    
    # Create visualizations
    create_summary_charts()
    create_regulatory_summary()
    
    # Generate report
    generate_data_report()
    
    print("\n✅ All visualizations and reports generated successfully!")
    print("\nOutput files created in ../outputs/:")
    print("  • financial_system_summary.png")
    print("  • regulatory_summary.png")

if __name__ == "__main__":
    main()