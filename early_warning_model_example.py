#!/usr/bin/env python3
"""
FinTech Early Warning Model - Example Implementation
Demonstrates how to use the generated dataset for early warning system development

Author: Research Assistant
Date: 2025-10-11
Purpose: Example implementation for thesis research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class FinTechEarlyWarningModel:
    """
    Early Warning Model for FinTech Risk Assessment
    Uses the comprehensive nexus dataset for predictive modeling
    """
    
    def __init__(self):
        # Load the datasets
        self.cyber_df = pd.read_csv('cyber_risk_exposure_data.csv')
        self.sentiment_df = pd.read_csv('consumer_sentiment_trust_data.csv')
        self.competitive_df = pd.read_csv('competitive_dynamics_data.csv')
        
        # Convert dates
        for df in [self.cyber_df, self.sentiment_df, self.competitive_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        print("FinTech Early Warning Model initialized")
        print("Loaded comprehensive nexus dataset for risk prediction")
    
    def create_integrated_risk_dataset(self):
        """
        Create integrated dataset combining all risk factors
        """
        print("\nCreating integrated risk assessment dataset...")
        
        # Aggregate sentiment data to monthly country level
        sentiment_monthly = self.sentiment_df.groupby([
            pd.Grouper(key='date', freq='M'), 'country'
        ]).agg({
            'sentiment_score': 'mean',
            'trust_index': 'mean',
            'security_perception_score': 'mean',
            'total_mentions': 'sum',
            'negative_mentions': 'sum'
        }).reset_index()
        
        # Aggregate competitive data to monthly (forward fill quarterly data)
        competitive_monthly = self.competitive_df.set_index('date').groupby('country').resample('M').ffill().reset_index()
        
        # Merge all datasets on date and country
        integrated_df = self.cyber_df.merge(
            sentiment_monthly, on=['date', 'country'], how='left'
        ).merge(
            competitive_monthly, on=['date', 'country'], how='left'
        )
        
        # Fill missing values with country means
        for col in integrated_df.select_dtypes(include=[np.number]).columns:
            if integrated_df[col].isnull().any():
                integrated_df[col] = integrated_df.groupby('country')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # Create composite risk indicators
        integrated_df['cyber_risk_severity'] = (
            integrated_df['total_cyber_incidents'] * 
            integrated_df['avg_incident_severity_score']
        ) / 10
        
        integrated_df['sentiment_risk_score'] = (
            (1 - integrated_df['sentiment_score']) * 50 + 
            (100 - integrated_df['trust_index']) * 0.5
        )
        
        integrated_df['market_risk_score'] = (
            integrated_df['herfindahl_hirschman_index'] / 100 +
            (100 - integrated_df['regulatory_clarity_score']) * 0.1
        )
        
        # Create overall risk categories
        integrated_df['overall_risk_score'] = (
            integrated_df['cyber_risk_severity'] * 0.4 +
            integrated_df['sentiment_risk_score'] * 0.3 +
            integrated_df['market_risk_score'] * 0.3
        )
        
        # Categorize risk levels
        integrated_df['risk_category'] = pd.cut(
            integrated_df['overall_risk_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        print(f"Created integrated dataset with {len(integrated_df)} records")
        print(f"Risk category distribution:")
        print(integrated_df['risk_category'].value_counts())
        
        return integrated_df
    
    def build_cyber_risk_predictor(self, integrated_df):
        """
        Build predictive model for cyber risk incidents
        """
        print("\n" + "="*60)
        print("BUILDING CYBER RISK PREDICTION MODEL")
        print("="*60)
        
        # Prepare features for cyber risk prediction
        feature_cols = [
            'search_mobile_money_fraud', 'search_sim_swap_fraud', 
            'search_fintech_scam', 'sentiment_score', 'trust_index',
            'herfindahl_hirschman_index', 'market_maturity_score',
            'innovation_index', 'regulatory_clarity_score'
        ]
        
        # Create lagged features (use previous month's data to predict current)
        df_lagged = integrated_df.copy()
        for col in feature_cols:
            df_lagged[f'{col}_lag1'] = df_lagged.groupby('country')[col].shift(1)
        
        # Remove rows with missing lagged features
        df_model = df_lagged.dropna()
        
        # Features and target
        X_features = [f'{col}_lag1' for col in feature_cols]
        X = df_model[X_features]
        y = df_model['total_cyber_incidents']
        
        # Encode country as feature
        le = LabelEncoder()
        X['country_encoded'] = le.fit_transform(df_model['country'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=df_model['country']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Cyber Risk Prediction Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            clean_name = row['feature'].replace('_lag1', '').replace('_', ' ').title()
            print(f"{i}. {clean_name}: {row['importance']:.3f}")
        
        return model, scaler, le, feature_importance
    
    def build_risk_classification_model(self, integrated_df):
        """
        Build classification model for overall risk categories
        """
        print("\n" + "="*60)
        print("BUILDING RISK CLASSIFICATION MODEL")
        print("="*60)
        
        # Prepare features
        feature_cols = [
            'total_cyber_incidents', 'mobile_money_fraud_incidents',
            'cyber_risk_index', 'sentiment_score', 'trust_index',
            'security_perception_score', 'herfindahl_hirschman_index',
            'market_maturity_score', 'innovation_index', 'new_fintech_licenses_issued'
        ]
        
        # Remove rows with missing risk categories
        df_model = integrated_df.dropna(subset=['risk_category'])
        
        X = df_model[feature_cols]
        y = df_model['risk_category']
        
        # Encode country as feature
        le_country = LabelEncoder()
        X['country_encoded'] = le_country.fit_transform(df_model['country'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        print("Risk Classification Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Risk Factors:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
            clean_name = row['feature'].replace('_', ' ').title()
            print(f"{i}. {clean_name}: {row['importance']:.3f}")
        
        return model, le_country, feature_importance
    
    def create_early_warning_dashboard(self, integrated_df):
        """
        Create early warning dashboard visualizations
        """
        print("\n" + "="*60)
        print("CREATING EARLY WARNING DASHBOARD")
        print("="*60)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Risk Score Trends Over Time
        monthly_risk = integrated_df.groupby(integrated_df['date'].dt.to_period('M'))['overall_risk_score'].mean()
        axes[0, 0].plot(monthly_risk.index.to_timestamp(), monthly_risk.values, 'r-', linewidth=2)
        axes[0, 0].set_title('Overall Risk Score Trends', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Risk Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=50, color='orange', linestyle='--', label='Medium Risk Threshold')
        axes[0, 0].axhline(y=75, color='red', linestyle='--', label='High Risk Threshold')
        axes[0, 0].legend()
        
        # 2. Risk Distribution by Country
        country_risk = integrated_df.groupby('country')['overall_risk_score'].mean().sort_values()
        axes[0, 1].barh(range(len(country_risk)), country_risk.values, color='coral')
        axes[0, 1].set_yticks(range(len(country_risk)))
        axes[0, 1].set_yticklabels(country_risk.index)
        axes[0, 1].set_title('Average Risk Score by Country', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Risk Score')
        
        # 3. Cyber Incidents vs Sentiment
        axes[1, 0].scatter(integrated_df['sentiment_score'], integrated_df['total_cyber_incidents'], 
                          alpha=0.6, color='blue')
        axes[1, 0].set_title('Cyber Incidents vs Sentiment Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Sentiment Score')
        axes[1, 0].set_ylabel('Cyber Incidents')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Market Concentration vs Risk
        axes[1, 1].scatter(integrated_df['herfindahl_hirschman_index'], integrated_df['overall_risk_score'],
                          alpha=0.6, color='green')
        axes[1, 1].set_title('Market Concentration vs Overall Risk', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('HHI Index')
        axes[1, 1].set_ylabel('Overall Risk Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Risk Category Distribution
        risk_dist = integrated_df['risk_category'].value_counts()
        axes[2, 0].pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%', startangle=90)
        axes[2, 0].set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
        
        # 6. Early Warning Signals Heatmap
        warning_signals = integrated_df.groupby('country')[
            ['cyber_risk_severity', 'sentiment_risk_score', 'market_risk_score']
        ].mean()
        
        im = axes[2, 1].imshow(warning_signals.T, cmap='RdYlGn_r', aspect='auto')
        axes[2, 1].set_xticks(range(len(warning_signals.index)))
        axes[2, 1].set_xticklabels(warning_signals.index, rotation=45)
        axes[2, 1].set_yticks(range(len(warning_signals.columns)))
        axes[2, 1].set_yticklabels(['Cyber Risk', 'Sentiment Risk', 'Market Risk'])
        axes[2, 1].set_title('Risk Heatmap by Country', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[2, 1], label='Risk Score')
        
        plt.tight_layout()
        plt.savefig('early_warning_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Early warning dashboard saved as 'early_warning_dashboard.png'")
    
    def generate_risk_alerts(self, integrated_df):
        """
        Generate automated risk alerts based on thresholds
        """
        print("\n" + "="*60)
        print("RISK ALERT SYSTEM")
        print("="*60)
        
        # Get latest data for each country
        latest_data = integrated_df.loc[integrated_df.groupby('country')['date'].idxmax()]
        
        # Define alert thresholds
        alerts = []
        
        for _, row in latest_data.iterrows():
            country = row['country']
            
            # High cyber risk alert
            if row['total_cyber_incidents'] > 15:
                alerts.append({
                    'country': country,
                    'type': 'CYBER RISK',
                    'level': 'HIGH',
                    'message': f"High cyber incident volume: {row['total_cyber_incidents']} incidents"
                })
            
            # Low sentiment alert
            if row['sentiment_score'] < -0.2:
                alerts.append({
                    'country': country,
                    'type': 'SENTIMENT',
                    'level': 'WARNING',
                    'message': f"Low consumer sentiment: {row['sentiment_score']:.3f}"
                })
            
            # Market concentration alert
            if row['herfindahl_hirschman_index'] > 5000:
                alerts.append({
                    'country': country,
                    'type': 'MARKET',
                    'level': 'WARNING',
                    'message': f"High market concentration: HHI {row['herfindahl_hirschman_index']:.0f}"
                })
            
            # Overall high risk alert
            if row['overall_risk_score'] > 75:
                alerts.append({
                    'country': country,
                    'type': 'OVERALL',
                    'level': 'CRITICAL',
                    'message': f"Critical risk level: {row['overall_risk_score']:.1f}"
                })
        
        # Display alerts
        if alerts:
            print(f"🚨 {len(alerts)} ACTIVE ALERTS DETECTED:")
            print("-" * 80)
            for alert in alerts:
                emoji = "🔴" if alert['level'] == 'CRITICAL' else "🟡" if alert['level'] == 'HIGH' else "🟠"
                print(f"{emoji} {alert['level']} - {alert['country']} ({alert['type']}): {alert['message']}")
        else:
            print("✅ No critical alerts detected")
        
        return alerts
    
    def run_complete_early_warning_analysis(self):
        """
        Run the complete early warning model analysis
        """
        print("="*80)
        print("FINTECH EARLY WARNING MODEL - COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Create integrated dataset
        integrated_df = self.create_integrated_risk_dataset()
        
        # Build predictive models
        cyber_model, cyber_scaler, cyber_le, cyber_importance = self.build_cyber_risk_predictor(integrated_df)
        risk_model, risk_le, risk_importance = self.build_risk_classification_model(integrated_df)
        
        # Create dashboard
        self.create_early_warning_dashboard(integrated_df)
        
        # Generate alerts
        alerts = self.generate_risk_alerts(integrated_df)
        
        print("\n" + "="*80)
        print("EARLY WARNING SYSTEM SUMMARY")
        print("="*80)
        
        print(f"✓ Integrated dataset created: {len(integrated_df)} records")
        print(f"✓ Cyber risk prediction model trained")
        print(f"✓ Risk classification model trained") 
        print(f"✓ Early warning dashboard generated")
        print(f"✓ Alert system activated: {len(alerts)} alerts")
        
        print(f"\nModel Applications:")
        print("• Real-time risk monitoring across 15 countries")
        print("• Predictive analytics for cyber incident forecasting")
        print("• Multi-dimensional risk assessment (cyber, sentiment, market)")
        print("• Automated alert system for policy makers")
        print("• Cross-country risk comparison and benchmarking")
        
        return {
            'integrated_data': integrated_df,
            'cyber_model': cyber_model,
            'risk_model': risk_model,
            'alerts': alerts
        }

def main():
    """Main function to demonstrate the early warning model"""
    model = FinTechEarlyWarningModel()
    results = model.run_complete_early_warning_analysis()
    
    print("\n" + "="*80)
    print("EARLY WARNING MODEL READY FOR DEPLOYMENT")
    print("="*80)
    print("\nThis example demonstrates how the FinTech Risk Nexus dataset")
    print("can be used to build comprehensive early warning systems for")
    print("Sub-Saharan African FinTech markets.")
    print("\nKey capabilities:")
    print("• Multi-dimensional risk assessment")
    print("• Predictive modeling and forecasting") 
    print("• Real-time monitoring and alerting")
    print("• Cross-country comparative analysis")

if __name__ == "__main__":
    main()