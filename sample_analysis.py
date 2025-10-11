#!/usr/bin/env python3
"""
Sample Analysis Script for FinTech Distress Dataset
Demonstrates how to use the dataset for early warning model development
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data(data_dir='fintech_dataset'):
    """Load and merge all dataset components"""
    print("📊 Loading FinTech Distress Dataset...")
    
    # Load all datasets
    companies = pd.read_csv(f'{data_dir}/companies.csv')
    financial = pd.read_csv(f'{data_dir}/financial_metrics.csv')
    operational = pd.read_csv(f'{data_dir}/operational_metrics.csv')
    funding = pd.read_csv(f'{data_dir}/funding_data.csv')
    regulatory = pd.read_csv(f'{data_dir}/regulatory_data.csv')
    distress = pd.read_csv(f'{data_dir}/distress_indicators.csv')
    
    print(f"✅ Loaded {len(companies)} companies with {len(financial)} financial records")
    
    return companies, financial, operational, funding, regulatory, distress

def create_feature_matrix(companies, financial, operational, funding, regulatory, distress):
    """Create a comprehensive feature matrix for modeling"""
    print("🔧 Creating feature matrix...")
    
    # Start with company characteristics
    features = companies[['company_id', 'age_years', 'employees', 'gdp_per_capita', 'mobile_penetration']].copy()
    
    # Add FinTech type as dummy variables
    fintech_dummies = pd.get_dummies(companies['fintech_type'], prefix='fintech_type')
    features = pd.concat([features, fintech_dummies], axis=1)
    
    # Add company size as dummy variables
    size_dummies = pd.get_dummies(companies['company_size'], prefix='size')
    features = pd.concat([features, size_dummies], axis=1)
    
    # Add regulatory status as dummy variables
    reg_dummies = pd.get_dummies(companies['regulatory_status'], prefix='reg_status')
    features = pd.concat([features, reg_dummies], axis=1)
    
    # Add latest financial metrics (Q8)
    latest_financial = financial[financial['quarter'] == 'Q8'][
        ['company_id', 'revenue_usd', 'revenue_growth_rate', 'profit_margin', 'burn_rate_usd']
    ].copy()
    features = features.merge(latest_financial, on='company_id', how='left')
    
    # Add latest operational metrics (Q8)
    latest_operational = operational[operational['quarter'] == 'Q8'][
        ['company_id', 'active_users', 'user_growth_rate', 'churn_rate', 
         'transaction_volume_usd', 'customer_acquisition_cost_usd']
    ].copy()
    features = features.merge(latest_operational, on='company_id', how='left')
    
    # Add funding statistics
    funding_stats = funding.groupby('company_id').agg({
        'amount_raised_usd': ['count', 'sum', 'mean'],
        'valuation_usd': 'mean',
        'months_since_last_round': 'min'
    }).round(2)
    
    funding_stats.columns = ['num_funding_rounds', 'total_funding', 'avg_funding_round', 'avg_valuation', 'months_since_last_funding']
    funding_stats = funding_stats.reset_index()
    features = features.merge(funding_stats, on='company_id', how='left')
    
    # Add regulatory issues count
    reg_issues = regulatory.groupby('company_id').size().reset_index(name='regulatory_issues_count')
    features = features.merge(reg_issues, on='company_id', how='left')
    features['regulatory_issues_count'] = features['regulatory_issues_count'].fillna(0)
    
    # Add distress indicators
    distress_features = distress[['company_id', 'revenue_decline_rate', 'user_decline_rate', 
                                 'has_revenue_decline', 'has_user_decline', 'has_regulatory_issues']]
    features = features.merge(distress_features, on='company_id', how='left')
    
    # Add target variable
    target = distress[['company_id', 'is_distressed']]
    features = features.merge(target, on='company_id', how='left')
    
    # Fill missing values
    features = features.fillna(0)
    
    print(f"✅ Created feature matrix with {features.shape[1]} features")
    
    return features

def build_early_warning_model(features):
    """Build and evaluate an early warning model"""
    print("🤖 Building Early Warning Model...")
    
    # Prepare features and target
    feature_cols = [col for col in features.columns if col not in ['company_id', 'is_distressed']]
    X = features[feature_cols]
    y = features['is_distressed']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    print("\n📈 Model Performance:")
    print("="*50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return rf_model, scaler, feature_importance

def analyze_risk_factors(features):
    """Analyze key risk factors for FinTech distress"""
    print("\n⚠️ Risk Factor Analysis:")
    print("="*50)
    
    # Correlation with distress
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    correlations = features[numeric_cols].corr()['is_distressed'].abs().sort_values(ascending=False)
    
    print("Top Risk Factors (by correlation with distress):")
    for factor, corr in correlations.head(10).items():
        if factor != 'is_distressed':
            print(f"  • {factor}: {corr:.3f}")
    
    # Distress by FinTech type
    print(f"\nDistress Rate by FinTech Type:")
    distress_by_type = features.groupby('fintech_type_Mobile Money')['is_distressed'].mean()
    # This is simplified - in practice, you'd analyze all FinTech types
    
    # Revenue decline analysis
    print(f"\nRevenue Decline Analysis:")
    revenue_decline_stats = features.groupby('is_distressed')['revenue_decline_rate'].agg(['count', 'mean', 'std'])
    print(revenue_decline_stats)
    
    # User decline analysis
    print(f"\nUser Decline Analysis:")
    user_decline_stats = features.groupby('is_distressed')['user_decline_rate'].agg(['count', 'mean', 'std'])
    print(user_decline_stats)

def generate_insights(features, model, feature_importance):
    """Generate actionable insights from the analysis"""
    print("\n💡 Key Insights:")
    print("="*50)
    
    # Overall distress rate
    distress_rate = features['is_distressed'].mean()
    print(f"• Overall distress rate: {distress_rate:.1%}")
    
    # Top risk factors
    top_risk_factors = feature_importance.head(5)['feature'].tolist()
    print(f"• Top risk factors: {', '.join(top_risk_factors)}")
    
    # Revenue vs user decline
    revenue_decline_distressed = features[features['is_distressed'] == 1]['revenue_decline_rate'].mean()
    user_decline_distressed = features[features['is_distressed'] == 1]['user_decline_rate'].mean()
    
    print(f"• Average revenue decline in distressed companies: {revenue_decline_distressed:.1%}")
    print(f"• Average user decline in distressed companies: {user_decline_distressed:.1%}")
    
    # Regulatory impact
    reg_issues_distressed = features[features['is_distressed'] == 1]['regulatory_issues_count'].mean()
    reg_issues_healthy = features[features['is_distressed'] == 0]['regulatory_issues_count'].mean()
    
    print(f"• Average regulatory issues (distressed): {reg_issues_distressed:.1f}")
    print(f"• Average regulatory issues (healthy): {reg_issues_healthy:.1f}")
    
    print(f"\n📋 Recommendations:")
    print(f"  1. Monitor revenue decline rates closely - strongest predictor")
    print(f"  2. Track user growth and churn patterns")
    print(f"  3. Implement regulatory compliance monitoring")
    print(f"  4. Focus on companies with multiple risk factors")
    print(f"  5. Develop early intervention strategies for high-risk companies")

def main():
    """Main analysis pipeline"""
    print("🚀 FinTech Early Warning Model Analysis")
    print("="*60)
    
    # Load data
    companies, financial, operational, funding, regulatory, distress = load_and_merge_data()
    
    # Create feature matrix
    features = create_feature_matrix(companies, financial, operational, funding, regulatory, distress)
    
    # Build model
    model, scaler, feature_importance = build_early_warning_model(features)
    
    # Analyze risk factors
    analyze_risk_factors(features)
    
    # Generate insights
    generate_insights(features, model, feature_importance)
    
    print("\n✅ Analysis complete!")
    print("\n📁 Dataset files available in 'fintech_dataset/' directory")
    print("📊 Use this analysis as a starting point for your research")

if __name__ == "__main__":
    main()