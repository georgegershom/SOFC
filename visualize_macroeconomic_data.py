"""
Visualization and Analysis Script for SSA Macroeconomic Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load the generated dataset and perform analysis"""
    
    # Load the data
    df = pd.read_csv('ssa_macroeconomic_data_full.csv')
    
    print("=" * 70)
    print("SUB-SAHARAN AFRICA MACROECONOMIC DATA ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    print("\n📊 DATASET OVERVIEW:")
    print(f"Total records: {len(df):,}")
    print(f"Countries: {df['Country_Name'].nunique()}")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Features: {len(df.columns)}")
    
    # Missing data analysis
    print("\n🔍 DATA COMPLETENESS:")
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"  {col}: {pct}% missing")
    
    if missing_pct.sum() == 0:
        print("  ✓ No missing data (all values generated or fetched)")
    
    # Key statistics by indicator
    print("\n📈 KEY MACROECONOMIC INDICATORS (2020-2024 Average):")
    recent_df = df[df['Year'] >= 2020]
    
    indicators = [
        'GDP Growth Rate (%)',
        'Inflation Rate (CPI) (%)',
        'Unemployment Rate (%)',
        'Broad Money (% of GDP)',
        'Central Government Debt (% of GDP)',
        'Mobile Cellular Subscriptions (per 100 people)',
        'Individuals using the Internet (% of population)'
    ]
    
    for indicator in indicators:
        mean_val = recent_df[indicator].mean()
        std_val = recent_df[indicator].std()
        print(f"\n  {indicator}:")
        print(f"    Mean: {mean_val:.2f}")
        print(f"    Std Dev: {std_val:.2f}")
        print(f"    Min: {recent_df[indicator].min():.2f}")
        print(f"    Max: {recent_df[indicator].max():.2f}")
    
    # Risk analysis
    print("\n⚠️ FINTECH RISK ANALYSIS:")
    risk_summary = df.groupby('Risk_Category').size()
    print("\n  Risk Distribution:")
    for category, count in risk_summary.items():
        pct = count / len(df) * 100
        print(f"    {category}: {count} records ({pct:.1f}%)")
    
    # Top risk countries
    print("\n  Top 10 Highest Risk Countries (2020-2024 avg):")
    recent_risk = recent_df.groupby('Country_Name')['FinTech_Risk_Score'].mean().sort_values(ascending=False)
    for i, (country, score) in enumerate(recent_risk.head(10).items(), 1):
        print(f"    {i:2d}. {country:<25} Score: {score:.2f}")
    
    # Digital readiness
    print("\n💻 DIGITAL INFRASTRUCTURE LEADERS (2020-2024 avg):")
    recent_digital = recent_df.groupby('Country_Name')['Digital_Infrastructure_Index'].mean().sort_values(ascending=False)
    for i, (country, score) in enumerate(recent_digital.head(10).items(), 1):
        print(f"    {i:2d}. {country:<25} Score: {score:.2f}")
    
    # Economic volatility
    print("\n📉 MOST VOLATILE ECONOMIES (Exchange Rate Volatility):")
    volatility = recent_df.groupby('Country_Name')['Exchange_Rate_Volatility'].mean().sort_values(ascending=False)
    for i, (country, vol) in enumerate(volatility.head(10).dropna().items(), 1):
        print(f"    {i:2d}. {country:<25} Volatility: {vol:.2f}%")
    
    # Correlation analysis
    print("\n🔗 CORRELATION ANALYSIS (Key Indicators):")
    corr_columns = [
        'GDP Growth Rate (%)',
        'Inflation Rate (CPI) (%)',
        'Digital_Infrastructure_Index',
        'FinTech_Risk_Score',
        'Economic_Stability_Index'
    ]
    
    corr_matrix = df[corr_columns].corr()
    print("\n  Strongest Correlations with FinTech Risk Score:")
    risk_corr = corr_matrix['FinTech_Risk_Score'].sort_values(ascending=False)
    for indicator, corr in risk_corr.items():
        if indicator != 'FinTech_Risk_Score' and abs(corr) > 0.3:
            print(f"    {indicator}: {corr:.3f}")
    
    return df

def create_visualizations(df):
    """Create and save visualization plots"""
    
    print("\n📊 GENERATING VISUALIZATIONS...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Sub-Saharan Africa Macroeconomic Data Analysis (2010-2024)', fontsize=16, fontweight='bold')
    
    # 1. Risk Distribution
    ax1 = axes[0, 0]
    risk_counts = df['Risk_Category'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    ax1.pie(risk_counts.values, labels=risk_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('FinTech Risk Distribution')
    
    # 2. GDP Growth Trends
    ax2 = axes[0, 1]
    gdp_trends = df.groupby('Year')['GDP Growth Rate (%)'].agg(['mean', 'std'])
    ax2.plot(gdp_trends.index, gdp_trends['mean'], marker='o', linewidth=2)
    ax2.fill_between(gdp_trends.index, 
                      gdp_trends['mean'] - gdp_trends['std'], 
                      gdp_trends['mean'] + gdp_trends['std'], 
                      alpha=0.3)
    ax2.set_title('GDP Growth Rate Trends')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('GDP Growth Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Digital Infrastructure Progress
    ax3 = axes[0, 2]
    digital_trends = df.groupby('Year')[['Mobile Cellular Subscriptions (per 100 people)', 
                                          'Individuals using the Internet (% of population)']].mean()
    ax3.plot(digital_trends.index, digital_trends.iloc[:, 0], marker='s', label='Mobile Subscriptions', linewidth=2)
    ax3.plot(digital_trends.index, digital_trends.iloc[:, 1], marker='^', label='Internet Users', linewidth=2)
    ax3.set_title('Digital Infrastructure Evolution')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Per 100 people / % of population')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Top 10 Risk Countries (2024)
    ax4 = axes[1, 0]
    df_2024 = df[df['Year'] == 2024]
    top_risk = df_2024.nlargest(10, 'FinTech_Risk_Score')[['Country_Name', 'FinTech_Risk_Score']]
    ax4.barh(range(len(top_risk)), top_risk['FinTech_Risk_Score'].values, color='#e74c3c')
    ax4.set_yticks(range(len(top_risk)))
    ax4.set_yticklabels(top_risk['Country_Name'].values, fontsize=9)
    ax4.set_xlabel('Risk Score')
    ax4.set_title('Top 10 Highest Risk Countries (2024)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Inflation vs GDP Growth
    ax5 = axes[1, 1]
    recent_df = df[df['Year'] >= 2020]
    ax5.scatter(recent_df['Inflation Rate (CPI) (%)'], 
                recent_df['GDP Growth Rate (%)'], 
                c=recent_df['FinTech_Risk_Score'], 
                cmap='RdYlGn_r', 
                alpha=0.6, 
                s=30)
    ax5.set_xlabel('Inflation Rate (%)')
    ax5.set_ylabel('GDP Growth Rate (%)')
    ax5.set_title('Inflation vs GDP Growth (2020-2024)')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar.set_label('Risk Score', rotation=270, labelpad=15)
    
    # 6. Debt to GDP Trends
    ax6 = axes[1, 2]
    debt_trends = df.groupby('Year')['Central Government Debt (% of GDP)'].agg(['mean', 'std'])
    ax6.plot(debt_trends.index, debt_trends['mean'], marker='o', color='#e67e22', linewidth=2)
    ax6.fill_between(debt_trends.index, 
                      debt_trends['mean'] - debt_trends['std']/2, 
                      debt_trends['mean'] + debt_trends['std']/2, 
                      alpha=0.3, color='#e67e22')
    ax6.set_title('Government Debt Trends')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Debt (% of GDP)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Exchange Rate Volatility Distribution
    ax7 = axes[2, 0]
    volatility_data = df['Exchange_Rate_Volatility'].dropna()
    ax7.hist(volatility_data, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Exchange Rate Volatility (%)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Exchange Rate Volatility Distribution')
    ax7.axvline(volatility_data.median(), color='red', linestyle='--', label=f'Median: {volatility_data.median():.2f}%')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Digital Infrastructure vs Risk Score
    ax8 = axes[2, 1]
    ax8.scatter(df['Digital_Infrastructure_Index'], 
                df['FinTech_Risk_Score'], 
                alpha=0.5, 
                s=20,
                color='#9b59b6')
    # Add trend line
    z = np.polyfit(df['Digital_Infrastructure_Index'].dropna(), 
                   df.loc[df['Digital_Infrastructure_Index'].notna(), 'FinTech_Risk_Score'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Digital_Infrastructure_Index'].min(), 
                          df['Digital_Infrastructure_Index'].max(), 100)
    ax8.plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Trend')
    ax8.set_xlabel('Digital Infrastructure Index')
    ax8.set_ylabel('FinTech Risk Score')
    ax8.set_title('Digital Infrastructure vs Risk')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Economic Stability by Region
    ax9 = axes[2, 2]
    stability_2024 = df_2024.nsmallest(10, 'FinTech_Risk_Score')[['Country_Name', 'Economic_Stability_Index']]
    ax9.barh(range(len(stability_2024)), stability_2024['Economic_Stability_Index'].values, color='#2ecc71')
    ax9.set_yticks(range(len(stability_2024)))
    ax9.set_yticklabels(stability_2024['Country_Name'].values, fontsize=9)
    ax9.set_xlabel('Stability Index')
    ax9.set_title('Top 10 Most Stable Economies (2024)')
    ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('ssa_macroeconomic_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: ssa_macroeconomic_analysis.png")
    
    # Create correlation heatmap
    fig2, ax = plt.subplots(figsize=(12, 10))
    
    corr_columns = [
        'GDP Growth Rate (%)',
        'Inflation Rate (CPI) (%)',
        'Unemployment Rate (%)',
        'Exchange_Rate_Volatility',
        'Real Interest Rate (%)',
        'Broad Money (% of GDP)',
        'Central Government Debt (% of GDP)',
        'Digital_Infrastructure_Index',
        'Economic_Stability_Index',
        'FinTech_Risk_Score'
    ]
    
    corr_matrix = df[corr_columns].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Key Macroeconomic Indicators', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('ssa_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: ssa_correlation_matrix.png")
    
    return

def generate_summary_report(df):
    """Generate a summary report"""
    
    print("\n📝 GENERATING SUMMARY REPORT...")
    
    report = []
    report.append("=" * 80)
    report.append("SUB-SAHARAN AFRICA MACROECONOMIC DATA - EXECUTIVE SUMMARY")
    report.append("For FinTech Early Warning Model Research")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    report.append("\n1. DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"• Total Records: {len(df):,}")
    report.append(f"• Countries Covered: {df['Country_Name'].nunique()}")
    report.append(f"• Time Period: {df['Year'].min()} - {df['Year'].max()}")
    report.append(f"• Total Variables: {len(df.columns)}")
    
    report.append("\n2. KEY FINDINGS (2020-2024)")
    report.append("-" * 40)
    
    recent_df = df[df['Year'] >= 2020]
    
    # Average indicators
    report.append("\nMacroeconomic Averages:")
    report.append(f"• GDP Growth Rate: {recent_df['GDP Growth Rate (%)'].mean():.2f}%")
    report.append(f"• Inflation Rate: {recent_df['Inflation Rate (CPI) (%)'].mean():.2f}%")
    report.append(f"• Unemployment Rate: {recent_df['Unemployment Rate (%)'].mean():.2f}%")
    report.append(f"• Government Debt/GDP: {recent_df['Central Government Debt (% of GDP)'].mean():.2f}%")
    
    report.append("\nDigital Infrastructure:")
    report.append(f"• Mobile Subscriptions: {recent_df['Mobile Cellular Subscriptions (per 100 people)'].mean():.2f} per 100")
    report.append(f"• Internet Users: {recent_df['Individuals using the Internet (% of population)'].mean():.2f}%")
    
    report.append("\n3. FINTECH RISK ASSESSMENT")
    report.append("-" * 40)
    
    risk_dist = df['Risk_Category'].value_counts()
    report.append("\nRisk Distribution:")
    for category in ['Low', 'Medium', 'High', 'Very High']:
        if category in risk_dist.index:
            count = risk_dist[category]
            pct = count / len(df) * 100
            report.append(f"• {category} Risk: {count} observations ({pct:.1f}%)")
    
    report.append("\nHighest Risk Countries (2024):")
    df_2024 = df[df['Year'] == 2024]
    top_risk = df_2024.nlargest(5, 'FinTech_Risk_Score')
    for i, row in enumerate(top_risk.itertuples(), 1):
        report.append(f"  {i}. {row.Country_Name}: {row.FinTech_Risk_Score:.2f}")
    
    report.append("\n4. DIGITAL INFRASTRUCTURE LEADERS (2024)")
    report.append("-" * 40)
    top_digital = df_2024.nlargest(5, 'Digital_Infrastructure_Index')
    for i, row in enumerate(top_digital.itertuples(), 1):
        report.append(f"  {i}. {row.Country_Name}: {row.Digital_Infrastructure_Index:.2f}")
    
    report.append("\n5. ECONOMIC VOLATILITY CONCERNS")
    report.append("-" * 40)
    high_volatility = df_2024.nlargest(5, 'Exchange_Rate_Volatility')
    report.append("\nHighest Exchange Rate Volatility (2024):")
    for i, row in enumerate(high_volatility.itertuples(), 1):
        if not pd.isna(row.Exchange_Rate_Volatility):
            report.append(f"  {i}. {row.Country_Name}: {row.Exchange_Rate_Volatility:.2f}%")
    
    report.append("\n6. DATA SOURCES AND METHODOLOGY")
    report.append("-" * 40)
    report.append("• Primary Source: World Bank Open Data API")
    report.append("• Synthetic Data Generation: For missing values using realistic economic patterns")
    report.append("• Risk Score Calculation: Weighted composite of multiple risk factors")
    report.append("• Volatility Measures: Rolling window calculations (3-year)")
    
    report.append("\n7. RECOMMENDATIONS FOR FINTECH RISK MODELING")
    report.append("-" * 40)
    report.append("• Focus on high-risk countries for early warning system development")
    report.append("• Consider digital infrastructure as a key moderating factor")
    report.append("• Monitor exchange rate volatility for cross-border FinTech operations")
    report.append("• Track government debt levels as systemic risk indicator")
    report.append("• Incorporate both economic and digital readiness metrics")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open('ssa_macroeconomic_summary_report.txt', 'w') as f:
        f.write(report_text)
    
    print("  ✓ Saved: ssa_macroeconomic_summary_report.txt")
    print("\n" + report_text)
    
    return report_text

def main():
    """Main execution"""
    # Load and analyze data
    df = load_and_analyze_data()
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  1. ssa_macroeconomic_data_full.csv - Full dataset")
    print("  2. ssa_macroeconomic_data.xlsx - Excel with multiple sheets")
    print("  3. ssa_macroeconomic_analysis.png - Comprehensive visualizations")
    print("  4. ssa_correlation_matrix.png - Correlation heatmap")
    print("  5. ssa_macroeconomic_summary_report.txt - Executive summary")

if __name__ == "__main__":
    main()