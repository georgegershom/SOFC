#!/usr/bin/env python3
"""
Statistical Analysis Module for Geotechnical Datasets
Provides statistical analysis and machine learning tools
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class GeotechnicalAnalyzer:
    """Statistical analysis tools for geotechnical data"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.scaler = StandardScaler()
        
    def perform_normality_tests(self, df, columns=None):
        """Perform normality tests on specified columns"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        results = {}
        for col in columns:
            data = df[col].dropna()
            if len(data) > 3:
                # Shapiro-Wilk test
                stat, p_value = stats.shapiro(data)
                
                # Anderson-Darling test
                ad_result = stats.anderson(data)
                
                # Skewness and Kurtosis
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                
                results[col] = {
                    'shapiro_statistic': stat,
                    'shapiro_p_value': p_value,
                    'is_normal_shapiro': p_value > 0.05,
                    'anderson_statistic': ad_result.statistic,
                    'anderson_critical_5%': ad_result.critical_values[2],
                    'is_normal_anderson': ad_result.statistic < ad_result.critical_values[2],
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else np.inf
                }
        
        return pd.DataFrame(results).T
    
    def perform_pca(self, df, n_components=None, columns=None):
        """Perform Principal Component Analysis"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        # Prepare data
        X = df[columns].dropna()
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        if n_components is None:
            n_components = min(len(columns), len(X))
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create results
        results = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'components': pd.DataFrame(
                pca.components_,
                columns=columns,
                index=[f'PC{i+1}' for i in range(n_components)]
            ),
            'transformed_data': pd.DataFrame(
                X_pca,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=X.index
            )
        }
        
        return results
    
    def perform_clustering(self, df, n_clusters=3, columns=None):
        """Perform K-means clustering"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        # Prepare data
        X = df[columns].dropna()
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_scaled, clusters)
        
        # Create results
        results = {
            'clusters': clusters,
            'cluster_centers': pd.DataFrame(
                self.scaler.inverse_transform(kmeans.cluster_centers_),
                columns=columns,
                index=[f'Cluster_{i}' for i in range(n_clusters)]
            ),
            'silhouette_score': sil_score,
            'inertia': kmeans.inertia_,
            'cluster_sizes': pd.Series(clusters).value_counts().sort_index()
        }
        
        return results
    
    def regression_analysis(self, df, target_col, feature_cols=None, test_size=0.2):
        """Perform regression analysis using Random Forest"""
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Prepare data
        data = df[[target_col] + feature_cols].dropna()
        X = data[feature_cols]
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Metrics
        results = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'model': rf,
            'predictions': pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred_test,
                'error': y_test - y_pred_test
            })
        }
        
        return results
    
    def classification_analysis(self, df, target_col, feature_cols=None, test_size=0.2):
        """Perform classification analysis using Random Forest"""
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Prepare data
        data = df[[target_col] + feature_cols].dropna()
        X = data[feature_cols]
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X, y, cv=5)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Classification report
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        results = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test),
            'model': rf,
            'predictions': pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred_test
            })
        }
        
        return results
    
    def correlation_analysis(self, df, method='pearson', threshold=0.5):
        """Perform detailed correlation analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        strong_correlations = pd.DataFrame(strong_correlations).sort_values(
            'correlation', key=abs, ascending=False
        )
        
        # Calculate VIF for multicollinearity
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X = df[numeric_cols].dropna()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        results = {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations,
            'vif_scores': vif_data,
            'high_multicollinearity': vif_data[vif_data['VIF'] > 10]
        }
        
        return results
    
    def outlier_detection(self, df, columns=None, method='iqr'):
        """Detect outliers using IQR or Z-score method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        
        for col in columns:
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > 3
                lower_bound = data.mean() - 3 * data.std()
                upper_bound = data.mean() + 3 * data.std()
            
            outliers[col] = {
                'n_outliers': outlier_mask.sum(),
                'pct_outliers': (outlier_mask.sum() / len(data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': data[outlier_mask].index.tolist(),
                'outlier_values': data[outlier_mask].values.tolist()
            }
        
        return pd.DataFrame(outliers).T

# Example usage
if __name__ == "__main__":
    from data_loader import GeotechnicalDataLoader
    
    # Load data
    loader = GeotechnicalDataLoader()
    sandy_data, sandy_merged = loader.load_sandy_soils()
    clay_data, clay_merged = loader.load_clay_soils()
    
    # Initialize analyzer
    analyzer = GeotechnicalAnalyzer()
    
    print("Performing Statistical Analysis...")
    
    # 1. Normality tests for sandy soils
    print("\n1. Normality Tests for Sandy Soil Properties:")
    normality_results = analyzer.perform_normality_tests(
        sandy_merged, 
        ['friction_angle_deg', 'void_ratio', 'relative_density_%']
    )
    print(normality_results[['is_normal_shapiro', 'skewness', 'cv']])
    
    # 2. PCA for clay soils
    print("\n2. PCA for Clay Soil Properties:")
    pca_results = analyzer.perform_pca(
        clay_merged,
        n_components=3,
        columns=['liquid_limit_%', 'plasticity_index', 'water_content_%', 
                'void_ratio', 'undrained_shear_strength_kPa']
    )
    print("Explained variance ratio:", pca_results['explained_variance_ratio'])
    print("Cumulative variance:", pca_results['cumulative_variance_ratio'])
    
    # 3. Clustering analysis
    print("\n3. Clustering Analysis for Sandy Soils:")
    cluster_results = analyzer.perform_clustering(
        sandy_merged,
        n_clusters=3,
        columns=['sand_content_%', 'relative_density_%', 'friction_angle_deg']
    )
    print(f"Silhouette Score: {cluster_results['silhouette_score']:.3f}")
    print("Cluster sizes:", cluster_results['cluster_sizes'].values)
    
    # 4. Regression analysis
    print("\n4. Regression Analysis - Predicting Friction Angle:")
    regression_results = analyzer.regression_analysis(
        sandy_merged,
        target_col='friction_angle_deg',
        feature_cols=['relative_density_%', 'void_ratio', 'd50_mm', 'cu_coefficient']
    )
    print(f"Test R2 Score: {regression_results['test_r2']:.3f}")
    print(f"Cross-validation mean R2: {regression_results['cv_mean']:.3f}")
    print("\nTop 3 Important Features:")
    print(regression_results['feature_importance'].head(3))
    
    # 5. Outlier detection
    print("\n5. Outlier Detection for Clay Soils:")
    outlier_results = analyzer.outlier_detection(
        clay_merged,
        columns=['liquid_limit_%', 'water_content_%', 'void_ratio'],
        method='iqr'
    )
    print(outlier_results[['n_outliers', 'pct_outliers']])