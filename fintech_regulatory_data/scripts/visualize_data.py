"""
Data Visualization Script for Financial System & Regulatory Data
Creates comprehensive visualizations for the Sub-Saharan Africa FinTech dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the generated dataset"""
    df = pd.read_csv('../data/financial_system_regulatory_data.csv')
    df_annual = pd.read_csv('../data/financial_system_annual_summary.csv')
    return df, df_annual

def plot_banking_health_trends():
    """Create visualization of banking sector health trends"""
    df, df_annual = load_data()
    
    # Select key countries for visualization
    key_countries = ['KEN', 'NGA', 'ZAF', 'GHA', 'RWA', 'ETH']
    df_subset = df_annual[df_annual['Country_Code'].isin(key_countries)]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Banking Sector Health Indicators - Key SSA Countries', fontsize=16, fontweight='bold')
    
    # NPL Trends
    for country in key_countries:
        country_data = df_subset[df_subset['Country_Code'] == country]
        axes[0,0].plot(country_data['Year'], country_data['Bank_NPL_to_Total_Loans_%'], 
                      marker='o', label=country, linewidth=2)
    axes[0,0].set_title('Non-Performing Loans Ratio')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('NPL %')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Bank Z-Score
    for country in key_countries:
        country_data = df_subset[df_subset['Country_Code'] == country]
        axes[0,1].plot(country_data['Year'], country_data['Bank_Z_Score'], 
                      marker='s', label=country, linewidth=2)
    axes[0,1].set_title('Bank Z-Score (Stability)')
    axes[0,1].set_xlabel('Year')
    axes[0,1].set_ylabel('Z-Score')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Credit to GDP
    for country in key_countries:
        country_data = df_subset[df_subset['Country_Code'] == country]
        axes[1,0].plot(country_data['Year'], country_data['Domestic_Credit_to_Private_Sector_%_GDP'], 
                      marker='^', label=country, linewidth=2)
    axes[1,0].set_title('Domestic Credit to Private Sector')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('% of GDP')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Banking Health Index
    for country in key_countries:
        country_data = df_subset[df_subset['Country_Code'] == country]
        axes[1,1].plot(country_data['Year'], country_data['Banking_Health_Index'], 
                      marker='d', label=country, linewidth=2)
    axes[1,1].set_title('Composite Banking Health Index')
    axes[1,1].set_xlabel('Year')
    axes[1,1].set_ylabel('Index Value')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/banking_health_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_regulatory_landscape():
    """Visualize regulatory development across countries"""
    df, df_annual = load_data()
    
    # Get 2024 data for regulatory snapshot
    df_2024 = df_annual[df_annual['Year'] == 2024]
    
    # Create regulatory adoption heatmap
    regulatory_cols = ['Digital_Lending_Regulation', 'Mobile_Money_Regulation', 
                      'Data_Protection_Law', 'Regulatory_Sandbox']
    
    # Select top 20 countries by financial development
    top_countries = df_2024.nlargest(20, 'Financial_Development_Index')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Regulatory Landscape in Sub-Saharan Africa (2024)', fontsize=16, fontweight='bold')
    
    # Heatmap of regulations
    reg_matrix = top_countries[['Country_Name'] + regulatory_cols].set_index('Country_Name')
    sns.heatmap(reg_matrix, annot=True, fmt='g', cmap='YlOrRd', 
                cbar_kws={'label': 'Regulation Exists'}, ax=axes[0])
    axes[0].set_title('FinTech Regulatory Adoption')
    axes[0].set_xlabel('Regulation Type')
    axes[0].set_ylabel('Country')
    
    # Bar chart of total regulations
    # Calculate total regulations from individual columns
    df_2024['Total_Regs'] = (df_2024['Digital_Lending_Regulation'] + 
                             df_2024['Mobile_Money_Regulation'] + 
                             df_2024['Data_Protection_Law'] + 
                             df_2024['Regulatory_Sandbox'])
    reg_counts = df_2024.groupby('Total_Regs')['Country_Code'].count()
    axes[1].bar(reg_counts.index, reg_counts.values, color='steelblue', edgecolor='black')
    axes[1].set_title('Distribution of FinTech Regulations Across Countries')
    axes[1].set_xlabel('Number of FinTech Regulations')
    axes[1].set_ylabel('Number of Countries')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(reg_counts.values):
        axes[1].text(reg_counts.index[i], v + 0.5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../outputs/regulatory_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_financial_inclusion_progress():
    """Visualize financial inclusion progress over time"""
    df, df_annual = load_data()
    
    # Calculate regional averages
    regions = {
        'East Africa': ['KEN', 'UGA', 'TZA', 'RWA', 'ETH', 'BDI'],
        'West Africa': ['NGA', 'GHA', 'SEN', 'CIV', 'BFA', 'MLI'],
        'Southern Africa': ['ZAF', 'BWA', 'NAM', 'ZWE', 'MOZ', 'ZMB'],
        'Central Africa': ['CMR', 'GAB', 'COG', 'CAF', 'TCD', 'AGO']
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Account Ownership Trends', 'Mobile Money Adoption',
                       'Banking Infrastructure', 'Financial Development Index'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, (region, countries) in enumerate(regions.items()):
        region_data = df_annual[df_annual['Country_Code'].isin(countries)]
        regional_avg = region_data.groupby('Year').agg({
            'Account_Ownership_%_Adults': 'mean',
            'Mobile_Money_Account_%_Adults': 'mean',
            'Bank_Branches_per_100k_Adults': 'mean',
            'Financial_Development_Index': 'mean'
        }).reset_index()
        
        # Account Ownership
        fig.add_trace(
            go.Scatter(x=regional_avg['Year'], y=regional_avg['Account_Ownership_%_Adults'],
                      mode='lines+markers', name=region, line=dict(color=colors[idx], width=2)),
            row=1, col=1
        )
        
        # Mobile Money
        fig.add_trace(
            go.Scatter(x=regional_avg['Year'], y=regional_avg['Mobile_Money_Account_%_Adults'],
                      mode='lines+markers', name=region, line=dict(color=colors[idx], width=2),
                      showlegend=False),
            row=1, col=2
        )
        
        # Banking Infrastructure
        fig.add_trace(
            go.Scatter(x=regional_avg['Year'], y=regional_avg['Bank_Branches_per_100k_Adults'],
                      mode='lines+markers', name=region, line=dict(color=colors[idx], width=2),
                      showlegend=False),
            row=2, col=1
        )
        
        # Financial Development Index
        fig.add_trace(
            go.Scatter(x=regional_avg['Year'], y=regional_avg['Financial_Development_Index'],
                      mode='lines+markers', name=region, line=dict(color=colors[idx], width=2),
                      showlegend=False),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="% of Adults", row=1, col=1)
    fig.update_yaxes(title_text="% of Adults", row=1, col=2)
    fig.update_yaxes(title_text="per 100k Adults", row=2, col=1)
    fig.update_yaxes(title_text="Index Value", row=2, col=2)
    
    fig.update_layout(
        title_text="Financial Inclusion Progress by Region (2010-2024)",
        title_font_size=16,
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    fig.write_html('../outputs/financial_inclusion_progress.html')
    fig.show()
    
    return fig

def plot_systemic_risk_dashboard():
    """Create a comprehensive systemic risk dashboard"""
    df, df_annual = load_data()
    
    # Get latest year data
    latest_data = df_annual[df_annual['Year'] == 2024]
    
    # Create risk categories
    latest_data['Risk_Category'] = pd.cut(latest_data['Systemic_Risk_Score'], 
                                           bins=[0, 30, 60, 100],
                                           labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Systemic Risk Distribution (2024)', 
                       'Risk vs Financial Development',
                       'Capital Adequacy by Risk Level',
                       'Regional Risk Comparison'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
               [{'type': 'box'}, {'type': 'bar'}]]
    )
    
    # Histogram of systemic risk scores
    fig.add_trace(
        go.Histogram(x=latest_data['Systemic_Risk_Score'], nbinsx=20,
                    marker_color='indianred', name='Countries'),
        row=1, col=1
    )
    
    # Scatter: Risk vs Development
    fig.add_trace(
        go.Scatter(x=latest_data['Financial_Development_Index'], 
                  y=latest_data['Systemic_Risk_Score'],
                  mode='markers+text',
                  text=latest_data['Country_Code'],
                  textposition='top center',
                  marker=dict(size=10, color=latest_data['Banking_Health_Index'],
                            colorscale='RdYlGn_r', showscale=True,
                            colorbar=dict(title="Banking<br>Health", x=1.1)),
                  name='Countries'),
        row=1, col=2
    )
    
    # Box plot of capital adequacy by risk category
    for category in ['Low Risk', 'Medium Risk', 'High Risk']:
        cat_data = latest_data[latest_data['Risk_Category'] == category]
        fig.add_trace(
            go.Box(y=cat_data['Capital_Adequacy_Ratio_%'], name=category),
            row=2, col=1
        )
    
    # Regional risk comparison
    regions = {
        'East Africa': ['KEN', 'UGA', 'TZA', 'RWA', 'ETH'],
        'West Africa': ['NGA', 'GHA', 'SEN', 'CIV', 'BFA'],
        'Southern Africa': ['ZAF', 'BWA', 'NAM', 'ZWE', 'MOZ'],
        'Central Africa': ['CMR', 'GAB', 'COG', 'CAF', 'TCD']
    }
    
    regional_risk = []
    for region, countries in regions.items():
        region_data = latest_data[latest_data['Country_Code'].isin(countries)]
        regional_risk.append({
            'Region': region,
            'Avg_Risk': region_data['Systemic_Risk_Score'].mean()
        })
    
    regional_df = pd.DataFrame(regional_risk)
    fig.add_trace(
        go.Bar(x=regional_df['Region'], y=regional_df['Avg_Risk'],
              marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Systemic Risk Score", row=1, col=1)
    fig.update_xaxes(title_text="Financial Development Index", row=1, col=2)
    fig.update_xaxes(title_text="Risk Category", row=2, col=1)
    fig.update_xaxes(title_text="Region", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Systemic Risk Score", row=1, col=2)
    fig.update_yaxes(title_text="Capital Adequacy Ratio (%)", row=2, col=1)
    fig.update_yaxes(title_text="Average Risk Score", row=2, col=2)
    
    fig.update_layout(
        title_text="Systemic Risk Dashboard - Sub-Saharan Africa",
        title_font_size=16,
        height=800,
        showlegend=False
    )
    
    fig.write_html('../outputs/systemic_risk_dashboard.html')
    fig.show()
    
    return fig

def create_country_profile_map():
    """Create an interactive map showing country profiles"""
    df_annual = pd.read_csv('../data/financial_system_annual_summary.csv')
    latest_data = df_annual[df_annual['Year'] == 2024]
    
    # Create hover text
    hover_text = []
    for idx, row in latest_data.iterrows():
        text = f"""<b>{row['Country_Name']}</b><br>
        Banking Health: {row['Banking_Health_Index']:.1f}<br>
        NPL Ratio: {row['Bank_NPL_to_Total_Loans_%']:.1f}%<br>
        Credit to GDP: {row['Domestic_Credit_to_Private_Sector_%_GDP']:.1f}%<br>
        Account Ownership: {row['Account_Ownership_%_Adults']:.1f}%<br>
        Mobile Money: {row['Mobile_Money_Account_%_Adults']:.1f}%<br>
        Regulatory Score: {row['Regulatory_Strength_Index']:.1f}<br>
        FinTech Regulations: {int(row['Digital_Lending_Regulation'] + row['Mobile_Money_Regulation'] + 
                                   row['Data_Protection_Law'] + row['Regulatory_Sandbox'])}
        """
        hover_text.append(text)
    
    latest_data['hover_text'] = hover_text
    
    fig = px.choropleth(latest_data, 
                        locations="Country_Code",
                        locationmode='ISO-3',
                        color="Financial_Development_Index",
                        hover_name="Country_Name",
                        hover_data={'Financial_Development_Index': ':.1f',
                                   'hover_text': True},
                        color_continuous_scale="Viridis",
                        scope="africa",
                        title="Financial Development Index - Sub-Saharan Africa (2024)")
    
    fig.update_geos(
        showcountries=True,
        countrycolor="White",
        showcoastlines=True,
        coastlinecolor="RebeccaPurple"
    )
    
    fig.update_layout(
        title_font_size=16,
        height=700,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        )
    )
    
    fig.write_html('../outputs/country_profiles_map.html')
    fig.show()
    
    return fig

def generate_summary_report():
    """Generate a comprehensive summary report"""
    df, df_annual = load_data()
    
    print("=" * 60)
    print("FINANCIAL SYSTEM & REGULATORY DATA")
    print("VISUALIZATION SUMMARY REPORT")
    print("=" * 60)
    
    # Latest year statistics
    latest_year = df_annual['Year'].max()
    latest_data = df_annual[df_annual['Year'] == latest_year]
    
    print(f"\n📊 DATASET OVERVIEW (as of {latest_year}):")
    print(f"  • Total Countries: {latest_data['Country_Code'].nunique()}")
    print(f"  • Time Period: {df_annual['Year'].min()} - {df_annual['Year'].max()}")
    print(f"  • Total Data Points: {len(df):,}")
    
    print("\n💰 BANKING SECTOR HEALTH:")
    print(f"  • Average NPL Ratio: {latest_data['Bank_NPL_to_Total_Loans_%'].mean():.2f}%")
    print(f"  • Average Bank Z-Score: {latest_data['Bank_Z_Score'].mean():.2f}")
    print(f"  • Average ROA: {latest_data['Bank_ROA_%'].mean():.2f}%")
    print(f"  • Average Credit to GDP: {latest_data['Domestic_Credit_to_Private_Sector_%_GDP'].mean():.1f}%")
    
    print("\n📋 REGULATORY LANDSCAPE:")
    print(f"  • Countries with Digital Lending Regs: {latest_data['Digital_Lending_Regulation'].sum():.0f}")
    print(f"  • Countries with Mobile Money Regs: {latest_data['Mobile_Money_Regulation'].sum():.0f}")
    print(f"  • Countries with Data Protection Laws: {latest_data['Data_Protection_Law'].sum():.0f}")
    print(f"  • Countries with Regulatory Sandboxes: {latest_data['Regulatory_Sandbox'].sum():.0f}")
    
    print("\n📱 FINANCIAL INCLUSION:")
    print(f"  • Average Account Ownership: {latest_data['Account_Ownership_%_Adults'].mean():.1f}%")
    print(f"  • Average Mobile Money Adoption: {latest_data['Mobile_Money_Account_%_Adults'].mean():.1f}%")
    print(f"  • Bank Branches per 100k: {latest_data['Bank_Branches_per_100k_Adults'].mean():.2f}")
    print(f"  • ATMs per 100k: {latest_data['ATMs_per_100k_Adults'].mean():.2f}")
    
    print("\n⚠️ RISK INDICATORS:")
    print(f"  • Average Systemic Risk Score: {latest_data['Systemic_Risk_Score'].mean():.1f}")
    print(f"  • Average Capital Adequacy Ratio: {latest_data['Capital_Adequacy_Ratio_%'].mean():.1f}%")
    print(f"  • Countries with High Risk (>60): {(latest_data['Systemic_Risk_Score'] > 60).sum()}")
    
    print("\n🏆 TOP PERFORMERS:")
    top_5 = latest_data.nlargest(5, 'Financial_Development_Index')[['Country_Name', 'Financial_Development_Index']]
    for idx, row in top_5.iterrows():
        print(f"  {row['Country_Name']}: {row['Financial_Development_Index']:.1f}")
    
    print("\n📈 KEY TRENDS (2010-2024):")
    trend_start = df_annual[df_annual['Year'] == 2010].mean()
    trend_end = df_annual[df_annual['Year'] == 2024].mean()
    
    print(f"  • NPL Reduction: {trend_start['Bank_NPL_to_Total_Loans_%'] - trend_end['Bank_NPL_to_Total_Loans_%']:.1f}pp")
    print(f"  • Account Ownership Growth: {trend_end['Account_Ownership_%_Adults'] - trend_start['Account_Ownership_%_Adults']:.1f}pp")
    print(f"  • Mobile Money Growth: {trend_end['Mobile_Money_Account_%_Adults'] - trend_start['Mobile_Money_Account_%_Adults']:.1f}pp")
    print(f"  • Credit Expansion: {trend_end['Domestic_Credit_to_Private_Sector_%_GDP'] - trend_start['Domestic_Credit_to_Private_Sector_%_GDP']:.1f}pp")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION FILES GENERATED:")
    print("=" * 60)
    print("  ✓ banking_health_trends.png")
    print("  ✓ regulatory_landscape.png")
    print("  ✓ financial_inclusion_progress.html")
    print("  ✓ systemic_risk_dashboard.html")
    print("  ✓ country_profiles_map.html")
    
def main():
    """Main execution function"""
    print("Generating visualizations...")
    
    # Generate all visualizations
    plot_banking_health_trends()
    print("✓ Banking health trends completed")
    
    plot_regulatory_landscape()
    print("✓ Regulatory landscape completed")
    
    plot_financial_inclusion_progress()
    print("✓ Financial inclusion progress completed")
    
    plot_systemic_risk_dashboard()
    print("✓ Systemic risk dashboard completed")
    
    create_country_profile_map()
    print("✓ Country profiles map completed")
    
    # Generate summary report
    generate_summary_report()
    
    print("\nAll visualizations completed successfully!")

if __name__ == "__main__":
    main()