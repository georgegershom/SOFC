#!/usr/bin/env python3
"""
Example 2: Machine Learning for Flow Pattern Classification
Stratified Flow Attenuation Dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def flow_pattern_classification():
    """Train ML model for flow pattern classification"""
    
    # Load time series features data
    df = pd.read_csv('../stratified_flow_datasets/time_series_features.csv')
    
    print("Flow Pattern Classification Analysis")
    print("=" * 40)
    print(f"Dataset shape: {df.shape}")
    print(f"Flow patterns: {df['flow_pattern'].unique()}")
    print(f"Pattern distribution:")
    print(df['flow_pattern'].value_counts())
    
    # Prepare features and target
    feature_cols = ['rms_amplitude', 'dominant_frequency_hz', 'spectral_centroid_hz',
                   'zero_crossings_per_sec', 'spectral_bandwidth_hz', 
                   'mfcc_1', 'mfcc_2', 'mfcc_3']
    
    X = df[feature_cols]
    y = df['flow_pattern']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = rf_model.score(X_train_scaled, y_train)
    test_score = rf_model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Predictions and detailed evaluation
    y_pred = rf_model.predict(X_test_scaled)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Feature importance plot
    axes[0].barh(feature_importance['feature'], feature_importance['importance'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Feature Importance for Flow Pattern Classification')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('flow_pattern_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, scaler

if __name__ == "__main__":
    model, scaler = flow_pattern_classification()
    print("\n✅ Machine learning analysis completed!")
