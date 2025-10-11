import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Sub-Saharan African countries with significant FinTech presence
countries = [
    'Nigeria', 'Kenya', 'South Africa', 'Ghana', 'Uganda', 
    'Tanzania', 'Rwanda', 'Senegal', 'Ivory Coast', 'Zambia',
    'Ethiopia', 'Botswana', 'Mozambique', 'Zimbabwe', 'Cameroon'
]

# Major FinTech brands in Sub-Saharan Africa
fintech_brands = {
    'Nigeria': ['Flutterwave', 'Paystack', 'OPay', 'PalmPay', 'Kuda'],
    'Kenya': ['M-Pesa', 'M-Shwari', 'Tala', 'Branch', 'Cellulant'],
    'South Africa': ['Yoco', 'TymeBank', 'Discovery Bank', 'Luno', 'Zapper'],
    'Ghana': ['MTN MoMo', 'Zeepay', 'ExpressPay', 'Slydepay', 'hubtel'],
    'Uganda': ['MTN Mobile Money', 'Airtel Money', 'Chipper Cash', 'Xente', 'Yo! Uganda'],
    'Tanzania': ['M-Pesa', 'Tigo Pesa', 'Airtel Money', 'HaloPesa', 'Vodacom M-Pesa'],
    'Rwanda': ['MTN Mobile Money', 'Airtel Money', 'Chipper Cash', 'Mergims', 'IREMBO'],
    'Senegal': ['Orange Money', 'Wave', 'Free Money', 'E-Money', 'Wizall'],
    'Ivory Coast': ['Orange Money', 'MTN Mobile Money', 'Moov Money', 'Wave', 'Julaya'],
    'Zambia': ['MTN Mobile Money', 'Airtel Money', 'Zoona', 'Kazang', 'Bayport'],
    'Ethiopia': ['M-Birr', 'HelloCash', 'Amole', 'Kacha Digital', 'EthSwitch'],
    'Botswana': ['Orange Money', 'MyZaka', 'Smega', 'BluePay', 'SwitchGlobe'],
    'Mozambique': ['M-Pesa', 'Mkesh', 'e-Mola', 'Standard Bank', 'PagaLu'],
    'Zimbabwe': ['EcoCash', 'OneMoney', 'Telecash', 'ZimSwitch', 'Paynow'],
    'Cameroon': ['Orange Money', 'MTN Mobile Money', 'Express Union Mobile', 'YUP', 'Wafacash']
}

# Generate quarterly data from 2018 to 2024
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 9, 30)
quarters = pd.date_range(start=start_date, end=end_date, freq='QE')

data = []

for country in countries:
    # Country-specific characteristics (baseline values)
    base_cyber_incidents = random.randint(5, 50)
    base_search_trend = random.randint(30, 80)
    base_sentiment = random.uniform(-0.2, 0.6)
    base_hhi = random.uniform(0.15, 0.45)
    base_licenses = random.randint(2, 15)
    
    # Economic development factor (higher for South Africa, Nigeria, Kenya)
    dev_factor = 1.5 if country in ['South Africa', 'Nigeria', 'Kenya'] else 1.0
    
    for i, quarter in enumerate(quarters):
        year = quarter.year
        q = (quarter.month - 1) // 3 + 1
        
        # Time-based trends (growth over years)
        time_factor = 1 + (i * 0.05)  # Gradual increase over time
        
        # Seasonal variation
        seasonal_factor = 1 + np.sin(2 * np.pi * i / 4) * 0.1
        
        # Random shocks (simulate cyber attacks, fraud waves, regulatory changes)
        shock = 1.0
        if random.random() < 0.1:  # 10% chance of significant event
            shock = random.uniform(1.3, 2.5)
        
        # 1. CYBER RISK EXPOSURE
        # Number of cybersecurity incidents (increasing trend with occasional spikes)
        cyber_incidents = int(
            base_cyber_incidents * dev_factor * time_factor * shock * 
            random.uniform(0.7, 1.3)
        )
        
        # Google search trends for "mobile money fraud" (0-100 scale)
        search_trend = min(100, int(
            base_search_trend * time_factor * seasonal_factor * shock * 
            random.uniform(0.8, 1.2)
        ))
        
        # Additional cyber metrics
        phishing_attacks = int(cyber_incidents * random.uniform(0.3, 0.5))
        malware_incidents = int(cyber_incidents * random.uniform(0.2, 0.4))
        data_breaches = int(cyber_incidents * random.uniform(0.1, 0.2))
        
        # 2. CONSUMER SENTIMENT & TRUST
        # Sentiment score (-1 to 1, where 1 is very positive)
        sentiment_drift = (i * 0.002) - 0.05  # Slight negative drift due to fraud concerns
        sentiment_shock = 0.0
        if shock > 1.2:  # Negative event
            sentiment_shock = -random.uniform(0.1, 0.3)
        
        sentiment_score = np.clip(
            base_sentiment + sentiment_drift + sentiment_shock + 
            random.uniform(-0.15, 0.15),
            -1, 1
        )
        
        # Trust index (0-100 scale, derived from sentiment)
        trust_index = int((sentiment_score + 1) * 50)
        
        # Social media mentions (volume)
        mentions_volume = int(
            1000 * dev_factor * time_factor * random.uniform(0.5, 2.0)
        )
        
        # Complaint rate (per 10,000 transactions)
        complaint_rate = max(1, int(
            50 * (1 - sentiment_score) * random.uniform(0.8, 1.2)
        ))
        
        # 3. COMPETITIVE DYNAMICS
        # Herfindahl-Hirschman Index (0 to 1, where higher = more concentrated)
        # Generally decreasing over time as market matures and new entrants arrive
        hhi_trend = -0.01 * i  # Gradual decrease in concentration
        hhi = np.clip(
            base_hhi + hhi_trend + random.uniform(-0.03, 0.03),
            0.10, 0.70
        )
        
        # Number of active FinTech licenses
        # Increasing over time with regulatory development
        new_licenses_year = max(0, int(
            base_licenses * (1 + i * 0.08) + random.uniform(-2, 3)
        ))
        
        # Total active licenses (cumulative with some exits)
        total_licenses = int(base_licenses * (1 + i * 0.15) * random.uniform(0.9, 1.1))
        
        # Market concentration (top 3 firms market share)
        top3_market_share = hhi * 100 * random.uniform(0.9, 1.1)
        
        # Number of new FinTech entrants (per quarter)
        new_entrants = max(0, int(
            np.random.poisson(3) * (1 + i * 0.05)
        ))
        
        # 4. ADDITIONAL NEXUS METRICS
        # Digital financial inclusion rate (%)
        financial_inclusion = min(95, int(
            30 + (i * 1.5) + dev_factor * 10 + random.uniform(-5, 5)
        ))
        
        # Mobile money transaction volume (millions)
        transaction_volume = int(
            50 * dev_factor * time_factor * seasonal_factor * 
            random.uniform(0.8, 1.5)
        )
        
        # Regulatory risk score (0-100, higher = more risk)
        reg_risk = int(
            random.uniform(20, 60) * (1 - sentiment_score * 0.2)
        )
        
        # Technology adoption index (0-100)
        tech_adoption = min(100, int(
            40 + (i * 1.2) + dev_factor * 5 + random.uniform(-8, 8)
        ))
        
        data.append({
            'country': country,
            'year': year,
            'quarter': q,
            'date': f'{year}-Q{q}',
            
            # Cyber Risk Exposure
            'cyber_incidents_total': cyber_incidents,
            'phishing_attacks': phishing_attacks,
            'malware_incidents': malware_incidents,
            'data_breaches': data_breaches,
            'mobile_fraud_search_trend': search_trend,
            
            # Consumer Sentiment & Trust
            'sentiment_score': round(sentiment_score, 3),
            'trust_index': trust_index,
            'social_media_mentions': mentions_volume,
            'complaint_rate_per_10k': complaint_rate,
            
            # Competitive Dynamics
            'hhi_index': round(hhi, 3),
            'new_licenses_issued_annual': new_licenses_year if q == 4 else 0,
            'total_active_licenses': total_licenses,
            'top3_market_share_pct': round(top3_market_share, 2),
            'new_entrants_quarter': new_entrants,
            
            # Additional Nexus Metrics
            'financial_inclusion_rate': financial_inclusion,
            'transaction_volume_millions': transaction_volume,
            'regulatory_risk_score': reg_risk,
            'tech_adoption_index': tech_adoption
        })

# Create DataFrame
df = pd.DataFrame(data)

# Sort by country and date
df = df.sort_values(['country', 'year', 'quarter']).reset_index(drop=True)

# Save to CSV
output_file = '/workspace/fintech_nexus_data_category4.csv'
df.to_csv(output_file, index=False)

print(f"Dataset generated successfully!")
print(f"Total records: {len(df)}")
print(f"Countries covered: {len(countries)}")
print(f"Time period: {df['year'].min()}-{df['year'].max()}")
print(f"Output file: {output_file}")
print("\nFirst few rows:")
print(df.head(10))
print("\nDataset summary statistics:")
print(df.describe())
