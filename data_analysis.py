"""
Data Analysis and Visualization for FinTech Early Warning Model
Macroeconomic Data Analysis for Sub-Saharan Africa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MacroeconomicDataAnalyzer:
    def __init__(self, data_file='fintech_macroeconomic_synthetic.csv'):
        """Initialize the analyzer with data file"""
        self.data = pd.read_csv(data_file)
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def basic_statistics(self):
        """Generate basic statistics for the dataset"""
        print("FinTech Early Warning Model - Macroeconomic Data Analysis")
        print("=" * 60)
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Countries: {self.data['Country'].nunique()}")
        print(f"Years: {self.data['Year'].min()} - {self.data['Year'].max()}")
        print(f"Variables: {len(self.data.columns) - 3}")
        
        print("\nBasic Statistics:")
        print("-" * 30)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = self.data[numeric_cols].describe()
        print(stats.round(2))
        
        print("\nMissing Values:")
        print("-" * 30)
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0].round(2))
        
        return stats
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\nCorrelation Analysis:")
        print("-" * 30)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Display high correlations
        high_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        'Variable_1': correlation_matrix.columns[i],
                        'Variable_2': correlation_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr)
            print("High Correlations (|r| > 0.7):")
            print(high_corr_df.round(3))
        else:
            print("No high correlations found.")
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, correlation_matrix, figsize=(12, 10)):
        """Plot correlation heatmap"""
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        plt.title('Macroeconomic Variables Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series(self, variables=None, countries=None, figsize=(15, 10)):
        """Plot time series for selected variables and countries"""
        if variables is None:
            variables = ['GDP_Growth_Rate', 'Inflation_Rate_CPI', 'Unemployment_Rate']
        
        if countries is None:
            countries = ['NGA', 'ZAF', 'KEN', 'GHA', 'ETH']  # Major SSA countries
        
        fig, axes = plt.subplots(len(variables), 1, figsize=figsize, sharex=True)
        if len(variables) == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            for country in countries:
                country_data = self.data[
                    (self.data['Country'] == country) & 
                    (self.data[var].notna())
                ]
                if not country_data.empty:
                    axes[i].plot(country_data['Year'], country_data[var], 
                               label=country, marker='o', markersize=4)
            
            axes[i].set_title(f'{var} Over Time')
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.xlabel('Year')
        plt.tight_layout()
        plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_country_comparison(self, variable, year=2023, top_n=10, figsize=(12, 8)):
        """Plot country comparison for a specific variable and year"""
        year_data = self.data[self.data['Year'] == year].copy()
        
        if year_data.empty:
            print(f"No data available for year {year}")
            return
        
        # Get top N countries by the variable
        year_data_sorted = year_data.nlargest(top_n, variable)
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(year_data_sorted)), year_data_sorted[variable])
        plt.xticks(range(len(year_data_sorted)), 
                  year_data_sorted['Country'], rotation=45)
        plt.title(f'Top {top_n} Countries by {variable} in {year}')
        plt.ylabel(variable)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'country_comparison_{variable}_{year}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distribution_analysis(self, variables=None, figsize=(15, 10)):
        """Plot distribution analysis for selected variables"""
        if variables is None:
            variables = ['GDP_Growth_Rate', 'Inflation_Rate_CPI', 'Unemployment_Rate', 
                        'Exchange_Rate_Volatility', 'Central_Bank_Policy_Rate']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            if i < len(axes):
                data = self.data[var].dropna()
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {var}')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(variables), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create interactive dashboard using Plotly"""
        print("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('GDP Growth Rate', 'Inflation Rate', 
                          'Unemployment Rate', 'Exchange Rate Volatility',
                          'Digital Infrastructure', 'Economic Indicators'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # GDP Growth Rate
        gdp_data = self.data.groupby('Year')['GDP_Growth_Rate'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=gdp_data['Year'], y=gdp_data['GDP_Growth_Rate'],
                      mode='lines+markers', name='GDP Growth Rate'),
            row=1, col=1
        )
        
        # Inflation Rate
        inflation_data = self.data.groupby('Year')['Inflation_Rate_CPI'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=inflation_data['Year'], y=inflation_data['Inflation_Rate_CPI'],
                      mode='lines+markers', name='Inflation Rate'),
            row=1, col=2
        )
        
        # Unemployment Rate
        unemployment_data = self.data.groupby('Year')['Unemployment_Rate'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=unemployment_data['Year'], y=unemployment_data['Unemployment_Rate'],
                      mode='lines+markers', name='Unemployment Rate'),
            row=2, col=1
        )
        
        # Exchange Rate Volatility
        exchange_data = self.data.groupby('Year')['Exchange_Rate_Volatility'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=exchange_data['Year'], y=exchange_data['Exchange_Rate_Volatility'],
                      mode='lines+markers', name='Exchange Rate Volatility'),
            row=2, col=2
        )
        
        # Digital Infrastructure (Internet Users)
        digital_data = self.data.groupby('Year')['Internet_Users_Percent'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=digital_data['Year'], y=digital_data['Internet_Users_Percent'],
                      mode='lines+markers', name='Internet Users %'),
            row=3, col=1
        )
        
        # Economic Indicators Heatmap
        # Create a correlation heatmap for the last year
        last_year = self.data['Year'].max()
        last_year_data = self.data[self.data['Year'] == last_year]
        
        numeric_cols = last_year_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = last_year_data[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(z=correlation_matrix.values,
                      x=correlation_matrix.columns,
                      y=correlation_matrix.columns,
                      colorscale='RdBu',
                      zmid=0),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text="FinTech Early Warning Model - Macroeconomic Dashboard",
            showlegend=True,
            height=1200
        )
        
        fig.write_html('interactive_dashboard.html')
        print("Interactive dashboard saved as 'interactive_dashboard.html'")
        
        return fig
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\nGenerating Summary Report...")
        print("=" * 60)
        
        # Basic statistics
        stats = self.basic_statistics()
        
        # Correlation analysis
        corr_matrix = self.correlation_analysis()
        
        # Country rankings
        print("\nCountry Rankings (Latest Year):")
        print("-" * 40)
        latest_year = self.data['Year'].max()
        latest_data = self.data[self.data['Year'] == latest_year]
        
        key_indicators = ['GDP_Growth_Rate', 'Inflation_Rate_CPI', 'Unemployment_Rate']
        
        for indicator in key_indicators:
            if indicator in latest_data.columns:
                print(f"\n{indicator}:")
                top_5 = latest_data.nlargest(5, indicator)[['Country_Name', indicator]]
                bottom_5 = latest_data.nsmallest(5, indicator)[['Country_Name', indicator]]
                
                print("Top 5:")
                for _, row in top_5.iterrows():
                    print(f"  {row['Country_Name']}: {row[indicator]:.2f}")
                
                print("Bottom 5:")
                for _, row in bottom_5.iterrows():
                    print(f"  {row['Country_Name']}: {row[indicator]:.2f}")
        
        # Risk assessment
        print("\nRisk Assessment:")
        print("-" * 20)
        
        # High inflation countries
        high_inflation = latest_data[latest_data['Inflation_Rate_CPI'] > 10]
        if not high_inflation.empty:
            print(f"High Inflation Risk ({len(high_inflation)} countries):")
            for _, row in high_inflation.iterrows():
                print(f"  {row['Country_Name']}: {row['Inflation_Rate_CPI']:.1f}%")
        
        # High unemployment countries
        high_unemployment = latest_data[latest_data['Unemployment_Rate'] > 15]
        if not high_unemployment.empty:
            print(f"\nHigh Unemployment Risk ({len(high_unemployment)} countries):")
            for _, row in high_unemployment.iterrows():
                print(f"  {row['Country_Name']}: {row['Unemployment_Rate']:.1f}%")
        
        # High debt countries
        high_debt = latest_data[latest_data['Public_Debt_to_GDP_Ratio'] > 80]
        if not high_debt.empty:
            print(f"\nHigh Debt Risk ({len(high_debt)} countries):")
            for _, row in high_debt.iterrows():
                print(f"  {row['Country_Name']}: {row['Public_Debt_to_GDP_Ratio']:.1f}%")
    
    def export_analysis_results(self):
        """Export analysis results to files"""
        print("\nExporting analysis results...")
        
        # Export summary statistics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = self.data[numeric_cols].describe()
        stats.to_csv('summary_statistics.csv')
        
        # Export correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        corr_matrix.to_csv('correlation_matrix.csv')
        
        # Export country rankings
        latest_year = self.data['Year'].max()
        latest_data = self.data[self.data['Year'] == latest_year]
        
        rankings = {}
        key_indicators = ['GDP_Growth_Rate', 'Inflation_Rate_CPI', 'Unemployment_Rate', 
                         'Exchange_Rate_Volatility', 'Central_Bank_Policy_Rate']
        
        for indicator in key_indicators:
            if indicator in latest_data.columns:
                rankings[indicator] = latest_data[['Country_Name', indicator]].sort_values(
                    indicator, ascending=False
                )
        
        with pd.ExcelWriter('country_rankings.xlsx', engine='openpyxl') as writer:
            for indicator, df in rankings.items():
                df.to_excel(writer, sheet_name=indicator, index=False)
        
        print("Analysis results exported to:")
        print("- summary_statistics.csv")
        print("- correlation_matrix.csv")
        print("- country_rankings.xlsx")

def main():
    """Main execution function"""
    print("FinTech Early Warning Model - Data Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MacroeconomicDataAnalyzer()
    
    # Perform analysis
    analyzer.basic_statistics()
    corr_matrix = analyzer.correlation_analysis()
    
    # Create visualizations
    analyzer.plot_correlation_heatmap(corr_matrix)
    analyzer.plot_time_series()
    analyzer.plot_country_comparison('GDP_Growth_Rate', year=2023)
    analyzer.plot_distribution_analysis()
    
    # Create interactive dashboard
    analyzer.create_interactive_dashboard()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Export results
    analyzer.export_analysis_results()
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()