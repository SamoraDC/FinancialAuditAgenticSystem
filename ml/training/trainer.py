"""
ML Model Training Pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from ml.models.fraud_detection_model import FraudDetectionModel
from ml.models.anomaly_detection_model import AnomalyDetectionModel


class ModelTrainer:
    """
    Centralized model training pipeline for financial audit models
    """

    def __init__(self, models_dir: str = "ml/models/saved"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for training pipeline"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def prepare_fraud_training_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for fraud detection training"""
        self.logger.info("Preparing fraud detection training data")

        # Basic data validation
        required_columns = ['amount', 'is_fraud']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove rows with missing target
        clean_data = raw_data.dropna(subset=['is_fraud'])

        # Split into train and test
        train_size = int(0.8 * len(clean_data))
        train_data = clean_data[:train_size]
        test_data = clean_data[train_size:]

        self.logger.info(f"Training data size: {len(train_data)}")
        self.logger.info(f"Test data size: {len(test_data)}")
        self.logger.info(f"Fraud rate in training: {train_data['is_fraud'].mean():.3f}")

        return train_data, test_data

    def train_fraud_detection_model(
        self,
        training_data: pd.DataFrame,
        model_type: str = "PPO",
        total_timesteps: int = 100000,
        save_model: bool = True
    ) -> FraudDetectionModel:
        """Train a fraud detection model"""
        self.logger.info(f"Training fraud detection model: {model_type}")

        # Initialize model
        model = FraudDetectionModel(model_type=model_type)

        # Train the model
        trained_model = model.train(training_data, total_timesteps=total_timesteps)

        # Save model if requested
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"fraud_detection_{model_type}_{timestamp}.zip"
            model.save_model(str(model_path))
            self.logger.info(f"Model saved to: {model_path}")

        return model

    def train_anomaly_detection_model(
        self,
        training_data: pd.DataFrame,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        use_pca: bool = False,
        save_model: bool = True
    ) -> AnomalyDetectionModel:
        """Train an anomaly detection model"""
        self.logger.info(f"Training anomaly detection model: {method}")

        # Initialize model
        model = AnomalyDetectionModel(method=method, contamination=contamination)

        # Train the model
        model.train(training_data, use_pca=use_pca)

        # Save model if requested
        if save_model:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.models_dir / f"anomaly_detection_{method}_{timestamp}.joblib"
            model.save_model(str(model_path))
            self.logger.info(f"Model saved to: {model_path}")

        return model

    def evaluate_fraud_model(
        self,
        model: FraudDetectionModel,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate fraud detection model performance"""
        self.logger.info("Evaluating fraud detection model")

        metrics = model.evaluate(test_data)

        self.logger.info("Fraud Detection Model Performance:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def evaluate_anomaly_model(
        self,
        model: AnomalyDetectionModel,
        test_data: pd.DataFrame,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate anomaly detection model performance"""
        self.logger.info("Evaluating anomaly detection model")

        metrics = model.evaluate_performance(test_data, true_labels)

        self.logger.info("Anomaly Detection Model Performance:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def run_comprehensive_training(
        self,
        data_path: str,
        fraud_model_type: str = "PPO",
        anomaly_method: str = "isolation_forest"
    ) -> Dict[str, Any]:
        """Run comprehensive training pipeline for all models"""
        self.logger.info("Starting comprehensive model training pipeline")

        # Load data
        self.logger.info(f"Loading data from: {data_path}")
        raw_data = pd.read_csv(data_path)

        results = {}

        # Fraud Detection Model Training
        if 'is_fraud' in raw_data.columns:
            try:
                train_data, test_data = self.prepare_fraud_training_data(raw_data)

                fraud_model = self.train_fraud_detection_model(
                    train_data,
                    model_type=fraud_model_type
                )

                fraud_metrics = self.evaluate_fraud_model(fraud_model, test_data)
                results['fraud_detection'] = {
                    'model': fraud_model,
                    'metrics': fraud_metrics
                }
            except Exception as e:
                self.logger.error(f"Fraud detection training failed: {e}")
                results['fraud_detection'] = {'error': str(e)}

        # Anomaly Detection Model Training
        try:
            anomaly_model = self.train_anomaly_detection_model(
                raw_data,
                method=anomaly_method
            )

            # Prepare test data for anomaly detection
            test_size = min(1000, len(raw_data) // 5)  # Use 20% or max 1000 samples
            test_indices = np.random.choice(len(raw_data), test_size, replace=False)
            test_data_anomaly = raw_data.iloc[test_indices]

            true_labels = None
            if 'is_fraud' in raw_data.columns:
                true_labels = test_data_anomaly['is_fraud'].values

            anomaly_metrics = self.evaluate_anomaly_model(
                anomaly_model,
                test_data_anomaly,
                true_labels
            )

            results['anomaly_detection'] = {
                'model': anomaly_model,
                'metrics': anomaly_metrics
            }
        except Exception as e:
            self.logger.error(f"Anomaly detection training failed: {e}")
            results['anomaly_detection'] = {'error': str(e)}

        self.logger.info("Comprehensive training pipeline completed")
        return results

    def generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive training report"""
        report = []
        report.append("="*60)
        report.append("FINANCIAL AUDIT ML MODELS TRAINING REPORT")
        report.append("="*60)
        report.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        for model_type, result in results.items():
            report.append(f"{model_type.replace('_', ' ').title()} Model:")
            report.append("-" * 40)

            if 'error' in result:
                report.append(f"  Status: FAILED")
                report.append(f"  Error: {result['error']}")
            else:
                report.append(f"  Status: SUCCESS")
                if 'metrics' in result:
                    report.append("  Performance Metrics:")
                    for metric, value in result['metrics'].items():
                        if isinstance(value, float):
                            report.append(f"    {metric}: {value:.4f}")
                        else:
                            report.append(f"    {metric}: {value}")

            report.append("")

        return "\n".join(report)