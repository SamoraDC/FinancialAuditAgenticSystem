"""
Anomaly Detection Model for Financial Auditing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import joblib


class AnomalyDetectionModel:
    """
    Comprehensive anomaly detection model for financial data
    """

    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        prepared_data = data.copy()

        # Numerical features
        numerical_cols = prepared_data.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate derived features for financial data
        if 'amount' in prepared_data.columns:
            # Amount-based features
            prepared_data['log_amount'] = np.log1p(prepared_data['amount'])
            prepared_data['amount_squared'] = prepared_data['amount'] ** 2

        # Time-based features
        if 'timestamp' in prepared_data.columns:
            prepared_data['timestamp'] = pd.to_datetime(prepared_data['timestamp'])
            prepared_data['hour'] = prepared_data['timestamp'].dt.hour
            prepared_data['day_of_week'] = prepared_data['timestamp'].dt.dayofweek
            prepared_data['month'] = prepared_data['timestamp'].dt.month
            prepared_data['is_weekend'] = prepared_data['day_of_week'].isin([5, 6]).astype(int)
            prepared_data['is_business_hours'] = prepared_data['hour'].between(9, 17).astype(int)

        # Account-based features (if available)
        if 'account_id' in prepared_data.columns:
            # Transaction frequency per account
            account_counts = prepared_data.groupby('account_id').size()
            prepared_data['account_transaction_count'] = prepared_data['account_id'].map(account_counts)

            # Average transaction amount per account
            account_avg_amount = prepared_data.groupby('account_id')['amount'].mean()
            prepared_data['account_avg_amount'] = prepared_data['account_id'].map(account_avg_amount)

        # Velocity features (if multiple transactions)
        if len(prepared_data) > 1 and 'timestamp' in prepared_data.columns:
            prepared_data = prepared_data.sort_values('timestamp')
            prepared_data['time_since_last'] = prepared_data['timestamp'].diff().dt.total_seconds()

        # Handle categorical variables
        categorical_cols = prepared_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col not in ['timestamp']:  # Skip timestamp
                prepared_data[col] = pd.Categorical(prepared_data[col]).codes

        # Select numerical features for modeling
        feature_cols = prepared_data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target columns if present
        target_cols = ['is_fraud', 'is_anomaly', 'label']
        feature_cols = [col for col in feature_cols if col not in target_cols]

        self.feature_names = feature_cols
        return prepared_data[feature_cols].fillna(0)

    def train(self, data: pd.DataFrame, use_pca: bool = False, n_components: Optional[int] = None):
        """Train the anomaly detection model"""
        # Prepare features
        X = self.prepare_features(data)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA if requested
        if use_pca:
            if n_components is None:
                n_components = min(10, X_scaled.shape[1])

            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)

        # Train model based on method
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0,
                bootstrap=False
            )
        elif self.method == "dbscan":
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit the model
        self.model.fit(X_scaled)

        return self

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Prepare features
        X = self.prepare_features(data)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)

        # Make predictions
        if self.method == "isolation_forest":
            anomaly_labels = self.model.predict(X_scaled)
            anomaly_scores = self.model.decision_function(X_scaled)

            # Convert to binary (1 = normal, -1 = anomaly) -> (0 = normal, 1 = anomaly)
            anomaly_labels = (anomaly_labels == -1).astype(int)

        elif self.method == "dbscan":
            cluster_labels = self.model.fit_predict(X_scaled)
            # In DBSCAN, -1 indicates noise/anomalies
            anomaly_labels = (cluster_labels == -1).astype(int)
            # For DBSCAN, we don't have anomaly scores, so use distance to nearest cluster
            anomaly_scores = np.zeros(len(anomaly_labels))  # Placeholder

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return anomaly_labels, anomaly_scores

    def detect_financial_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect specific financial anomalies"""
        anomalies = {
            'statistical_anomalies': [],
            'business_rule_violations': [],
            'pattern_anomalies': []
        }

        # Statistical anomalies
        if 'amount' in data.columns:
            amount_mean = data['amount'].mean()
            amount_std = data['amount'].std()
            z_threshold = 3

            # Outliers based on Z-score
            z_scores = np.abs((data['amount'] - amount_mean) / amount_std)
            statistical_outliers = data[z_scores > z_threshold]

            for idx, row in statistical_outliers.iterrows():
                anomalies['statistical_anomalies'].append({
                    'type': 'amount_outlier',
                    'index': idx,
                    'amount': row['amount'],
                    'z_score': z_scores.iloc[idx],
                    'description': f"Transaction amount {row['amount']} is {z_scores.iloc[idx]:.2f} standard deviations from mean"
                })

        # Business rule violations
        if 'timestamp' in data.columns:
            data_copy = data.copy()
            data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            data_copy['hour'] = data_copy['timestamp'].dt.hour

            # Transactions outside business hours
            after_hours = data_copy[~data_copy['hour'].between(9, 17)]
            for idx, row in after_hours.iterrows():
                anomalies['business_rule_violations'].append({
                    'type': 'after_hours_transaction',
                    'index': idx,
                    'hour': row['hour'],
                    'description': f"Transaction at {row['hour']}:00 outside business hours"
                })

        # Round number detection
        if 'amount' in data.columns:
            round_amounts = data[data['amount'] % 100 == 0]
            if len(round_amounts) > len(data) * 0.1:  # More than 10% round numbers
                for idx, row in round_amounts.iterrows():
                    anomalies['pattern_anomalies'].append({
                        'type': 'round_number',
                        'index': idx,
                        'amount': row['amount'],
                        'description': f"Suspicious round number: {row['amount']}"
                    })

        return anomalies

    def explain_anomaly(self, data_point: pd.Series, feature_importance: bool = True) -> Dict[str, Any]:
        """Explain why a data point is considered anomalous"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Prepare single data point
        X = self.prepare_features(pd.DataFrame([data_point]))
        X_scaled = self.scaler.transform(X)

        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)

        explanation = {
            'anomaly_score': 0,
            'contributing_factors': [],
            'feature_deviations': {}
        }

        if self.method == "isolation_forest":
            anomaly_score = self.model.decision_function(X_scaled)[0]
            explanation['anomaly_score'] = anomaly_score

            # Feature importance (simplified approach)
            if feature_importance and self.pca is None:
                feature_values = X.iloc[0].values
                feature_means = np.mean(X.values, axis=0)
                feature_stds = np.std(X.values, axis=0)

                for i, (feature_name, value) in enumerate(zip(self.feature_names, feature_values)):
                    if feature_stds[i] > 0:
                        z_score = abs((value - feature_means[i]) / feature_stds[i])
                        if z_score > 2:  # Significant deviation
                            explanation['feature_deviations'][feature_name] = {
                                'value': value,
                                'mean': feature_means[i],
                                'z_score': z_score
                            }

        return explanation

    def save_model(self, path: str):
        """Save the trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'method': self.method,
            'contamination': self.contamination
        }

        joblib.dump(model_data, path)

    def load_model(self, path: str):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.feature_names = model_data['feature_names']
        self.method = model_data['method']
        self.contamination = model_data['contamination']

    def evaluate_performance(self, data: pd.DataFrame, true_labels: np.ndarray = None) -> Dict[str, float]:
        """Evaluate model performance if true labels are available"""
        predictions, scores = self.predict(data)

        metrics = {
            'anomaly_rate': np.mean(predictions),
            'total_anomalies': np.sum(predictions),
            'total_samples': len(predictions)
        }

        if true_labels is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

            metrics.update({
                'precision': precision_score(true_labels, predictions),
                'recall': recall_score(true_labels, predictions),
                'f1_score': f1_score(true_labels, predictions),
                'auc': roc_auc_score(true_labels, scores) if self.method == "isolation_forest" else 0
            })

        return metrics