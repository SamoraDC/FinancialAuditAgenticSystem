"""
Reinforcement Learning-based anomaly detection service
Implements adaptive anomaly detection using RL models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import joblib
import os

logger = logging.getLogger(__name__)


class RLAnomalyDetector:
    """RL-based anomaly detection service"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/rl_anomaly_model.pkl"
        self.model_version = "1.0.0"
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.isolation_forest = None
        self.feature_names = [
            'amount', 'amount_log', 'amount_zscore',
            'day_of_week', 'hour_of_day', 'month',
            'vendor_frequency', 'amount_frequency',
            'running_balance', 'deviation_from_mean',
            'transaction_velocity', 'amount_category'
        ]
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the RL model (placeholder for now)"""
        try:
            if os.path.exists(self.model_path):
                self._load_model()
            else:
                self._create_default_model()
                
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}. Using default model.")
            self._create_default_model()

    def _create_default_model(self):
        """Create default isolation forest model"""
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Assume 10% anomalies
            random_state=42,
            n_estimators=100
        )
        logger.info("Created default Isolation Forest model")

    def _load_model(self):
        """Load pre-trained model using secure joblib"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.isolation_forest = model_data.get('model')
                self.feature_scaler = model_data.get('scaler', StandardScaler())
                self.label_encoders = model_data.get('encoders', {})
                self.model_version = model_data.get('version', '1.0.0')
                logger.info(f"Loaded RL model version {self.model_version}")
            else:
                self._create_default_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._create_default_model()

    def _save_model(self):
        """Save the trained model using secure joblib"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'model': self.isolation_forest,
                'scaler': self.feature_scaler,
                'encoders': self.label_encoders,
                'version': self.model_version,
                'training_date': datetime.utcnow().isoformat()
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    async def prepare_features(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare feature vectors from transaction data"""
        try:
            if not transactions:
                return np.array([])

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(transactions)
            
            # Initialize feature matrix
            feature_matrix = []

            for _, transaction in df.iterrows():
                features = await self._extract_transaction_features(transaction, df)
                feature_matrix.append(features)

            feature_matrix = np.array(feature_matrix)
            
            # Scale features
            if feature_matrix.size > 0:
                feature_matrix = self.feature_scaler.fit_transform(feature_matrix)

            return feature_matrix

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return np.array([])

    async def _extract_transaction_features(self, transaction: pd.Series, df: pd.DataFrame) -> List[float]:
        """Extract features from a single transaction"""
        features = []
        
        try:
            # Amount-based features
            amount = float(transaction.get('amount', 0))
            features.append(amount)  # Raw amount
            features.append(np.log1p(abs(amount)))  # Log amount
            
            # Z-score of amount relative to all transactions
            all_amounts = df['amount'].astype(float)
            amount_mean = all_amounts.mean()
            amount_std = all_amounts.std()
            amount_zscore = (amount - amount_mean) / amount_std if amount_std > 0 else 0
            features.append(amount_zscore)

            # Time-based features
            try:
                if 'date' in transaction and pd.notna(transaction['date']):
                    date = pd.to_datetime(transaction['date'])
                    features.extend([
                        date.dayofweek,  # Day of week
                        date.hour if hasattr(date, 'hour') else 12,  # Hour of day
                        date.month  # Month
                    ])
                else:
                    features.extend([0, 12, 1])  # Default values
            except:
                features.extend([0, 12, 1])  # Default values

            # Vendor frequency (how often this vendor appears)
            vendor = str(transaction.get('vendor', 'unknown'))
            vendor_count = len(df[df.get('vendor', '') == vendor]) if 'vendor' in df.columns else 1
            vendor_frequency = vendor_count / len(df)
            features.append(vendor_frequency)

            # Amount frequency (how common is this amount)
            amount_tolerance = amount * 0.01  # 1% tolerance
            similar_amounts = len(df[abs(df.get('amount', 0).astype(float) - amount) <= amount_tolerance])
            amount_frequency = similar_amounts / len(df)
            features.append(amount_frequency)

            # Running balance approximation
            transaction_index = transaction.name if hasattr(transaction, 'name') else 0
            running_balance = df.iloc[:transaction_index + 1]['amount'].astype(float).sum()
            features.append(running_balance)

            # Deviation from mean
            deviation_from_mean = abs(amount - amount_mean) / amount_mean if amount_mean != 0 else 0
            features.append(deviation_from_mean)

            # Transaction velocity (transactions per time period)
            # Simplified: assume daily transactions
            velocity = len(df) / max(1, (df.index.max() - df.index.min() + 1))
            features.append(velocity)

            # Amount category (discretized)
            amount_category = self._categorize_amount(amount, all_amounts)
            features.append(amount_category)

            # Pad or truncate to match expected feature count
            while len(features) < len(self.feature_names):
                features.append(0.0)
            
            return features[:len(self.feature_names)]

        except Exception as e:
            logger.warning(f"Feature extraction failed for transaction: {e}")
            return [0.0] * len(self.feature_names)

    def _categorize_amount(self, amount: float, all_amounts: pd.Series) -> float:
        """Categorize amount into percentile-based categories"""
        try:
            percentiles = [0, 25, 50, 75, 90, 95, 100]
            thresholds = np.percentile(all_amounts, percentiles)
            
            for i, threshold in enumerate(thresholds[1:]):
                if amount <= threshold:
                    return float(i)
            return float(len(percentiles) - 1)
            
        except:
            return 0.0

    async def detect_anomalies(self, feature_vectors: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalies using the RL model"""
        try:
            if feature_vectors.size == 0:
                return []

            # Ensure model is trained
            if not hasattr(self.isolation_forest, 'decision_function_'):
                # Train on the current data if not already trained
                self.isolation_forest.fit(feature_vectors)

            # Predict anomalies
            predictions = self.isolation_forest.predict(feature_vectors)
            anomaly_scores = self.isolation_forest.decision_function(feature_vectors)

            results = []
            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                is_anomaly = prediction == -1  # Isolation Forest uses -1 for anomalies
                
                # Convert score to confidence (0-1 scale)
                confidence = self._score_to_confidence(score)
                
                # Determine severity based on confidence
                severity = self._determine_severity(confidence)

                result = {
                    'index': i,
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'anomaly_score': float(score),
                    'severity': severity,
                    'features': feature_vectors[i].tolist() if i < len(feature_vectors) else [],
                    'model_version': self.model_version,
                    'detection_timestamp': datetime.utcnow().isoformat()
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

    def _score_to_confidence(self, score: float) -> float:
        """Convert anomaly score to confidence level (0-1)"""
        # Isolation Forest scores are typically between -0.5 and 0.5
        # More negative = more anomalous
        # Normalize to 0-1 confidence scale
        normalized = max(0, min(1, (0.5 + score) / 1.0))
        return 1 - normalized  # Invert so higher confidence = more anomalous

    def _determine_severity(self, confidence: float) -> str:
        """Determine severity level based on confidence"""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        else:
            return 'very_low'

    async def train_model(self, training_data: List[Dict[str, Any]], labels: Optional[List[int]] = None):
        """Train the RL model with new data"""
        try:
            if not training_data:
                logger.warning("No training data provided")
                return

            # Prepare features
            feature_vectors = await self.prepare_features(training_data)
            
            if feature_vectors.size == 0:
                logger.warning("No features extracted from training data")
                return

            # Train the model
            self.isolation_forest.fit(feature_vectors)
            
            # Save the updated model
            self._save_model()
            
            logger.info(f"Model trained on {len(training_data)} samples")

        except Exception as e:
            logger.error(f"Model training failed: {e}")

    async def update_model_with_feedback(self, transaction_id: str, is_anomaly: bool, 
                                       confidence: float, feedback: str):
        """Update model based on human feedback (RL component)"""
        try:
            # In a full RL implementation, this would update model weights
            # For now, we log the feedback for future training
            feedback_data = {
                'transaction_id': transaction_id,
                'predicted_anomaly': is_anomaly,
                'confidence': confidence,
                'human_feedback': feedback,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store feedback for future model updates
            await self._store_feedback(feedback_data)
            
            logger.info(f"Feedback recorded for transaction {transaction_id}: {feedback}")

        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")

    async def _store_feedback(self, feedback_data: Dict[str, Any]):
        """Store feedback data for future model training"""
        try:
            feedback_file = "models/feedback_log.jsonl"
            os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
            
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")

    async def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            return {
                'model_version': self.model_version,
                'model_type': 'IsolationForest',
                'feature_count': len(self.feature_names),
                'features': self.feature_names,
                'contamination_rate': getattr(self.isolation_forest, 'contamination', 0.1),
                'n_estimators': getattr(self.isolation_forest, 'n_estimators', 100),
                'is_trained': hasattr(self.isolation_forest, 'decision_function_'),
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return {'error': str(e)}
