"""
Comprehensive Statistical Analysis Engine for Financial Audit System

This module implements advanced statistical analysis methods for anomaly detection
in financial data, including Benford's Law, Newcomb-Benford's Law, Zipf's Law,
and reinforcement learning-based anomaly detection using stable-baselines3.

Based on FrameworkDoc.md specifications for the LangGraph auditing architecture.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from decimal import Decimal
from collections import Counter
import re
import math
from dataclasses import dataclass
from enum import Enum

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

# Pydantic models
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated
from ..models.audit_models import RiskLevel, AuditFinding


# Configure logging
logger = logging.getLogger(__name__)


class StatisticalMethod(str, Enum):
    """Statistical analysis methods enumeration"""
    BENFORDS_LAW = "benfords_law"
    NEWCOMB_BENFORD = "newcomb_benford"
    ZIPFS_LAW = "zipfs_law"
    ISOLATION_FOREST = "isolation_forest"
    PYOD_ENSEMBLE = "pyod_ensemble"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES_ANOMALY = "time_series_anomaly"


class AnomalyType(str, Enum):
    """Types of anomalies detected"""
    DIGIT_DISTRIBUTION = "digit_distribution"
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_DEVIATION = "pattern_deviation"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    CLUSTERING_ANOMALY = "clustering_anomaly"


@dataclass
class StatisticalResult:
    """Statistical analysis result container"""
    method: StatisticalMethod
    anomaly_type: AnomalyType
    risk_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    p_value: Optional[float]
    chi_square_statistic: Optional[float]
    expected_distribution: Optional[Dict[str, float]]
    observed_distribution: Optional[Dict[str, float]]
    anomalous_values: List[Dict[str, Any]]
    statistical_summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class BenfordAnalysisResult(BaseModel):
    """Benford's Law analysis result model"""
    digit_position: int = Field(..., description="Digit position analyzed (1=first, 2=second)")
    chi_square_statistic: float = Field(..., description="Chi-square test statistic")
    p_value: float = Field(..., description="P-value of the test")
    critical_value: float = Field(..., description="Critical value at 95% confidence")
    is_significant: bool = Field(..., description="Whether deviation is statistically significant")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score based on deviation")
    expected_frequencies: Dict[str, float] = Field(..., description="Expected digit frequencies")
    observed_frequencies: Dict[str, float] = Field(..., description="Observed digit frequencies")
    deviations: Dict[str, float] = Field(..., description="Deviations from expected")
    anomalous_digits: List[str] = Field(..., description="Digits with significant deviations")
    sample_size: int = Field(..., description="Sample size analyzed")
    recommendations: List[str] = Field(..., description="Analysis recommendations")


class ZipfAnalysisResult(BaseModel):
    """Zipf's Law analysis result model"""
    correlation_coefficient: float = Field(..., description="Correlation with Zipf distribution")
    kolmogorov_smirnov_statistic: float = Field(..., description="KS test statistic")
    p_value: float = Field(..., description="P-value of the test")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score based on deviation")
    word_rank_frequencies: Dict[str, Dict[str, float]] = Field(..., description="Word rank and frequencies")
    expected_vs_observed: Dict[str, Tuple[float, float]] = Field(..., description="Expected vs observed frequencies")
    anomalous_words: List[str] = Field(..., description="Words deviating from Zipf distribution")
    text_sample_size: int = Field(..., description="Text sample size")
    recommendations: List[str] = Field(..., description="Analysis recommendations")


class AnomalyDetectionResult(BaseModel):
    """Anomaly detection result model"""
    method: StatisticalMethod = Field(..., description="Detection method used")
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly detected")
    total_samples: int = Field(..., description="Total samples analyzed")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., ge=0, le=1, description="Anomaly rate")
    overall_risk_score: float = Field(..., ge=0, le=1, description="Overall risk score")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence")
    anomalous_records: List[Dict[str, Any]] = Field(..., description="Detailed anomalous records")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    model_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    recommendations: List[str] = Field(..., description="Recommended actions")


class RLAnomalyDetectionResult(BaseModel):
    """Reinforcement Learning anomaly detection result model"""
    model_type: str = Field(..., description="RL model type (DQN, PPO)")
    episodes_trained: int = Field(..., description="Training episodes completed")
    average_reward: float = Field(..., description="Average reward achieved")
    convergence_score: float = Field(..., ge=0, le=1, description="Model convergence score")
    flagged_transactions: List[Dict[str, Any]] = Field(..., description="Flagged transactions")
    action_probabilities: Dict[str, float] = Field(..., description="Action probabilities")
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(..., description="Confidence intervals")
    model_performance: Dict[str, float] = Field(..., description="Performance metrics")
    adaptive_threshold: float = Field(..., description="Current adaptive threshold")
    recommendations: List[str] = Field(..., description="RL-based recommendations")


class StatisticalAnalysisEngine:
    """
    Comprehensive statistical analysis engine for financial audit data.

    Implements multiple statistical methods for anomaly detection including:
    - Benford's Law analysis for digit distribution
    - Newcomb-Benford's Law for first-digit analysis
    - Zipf's Law for text pattern detection
    - Advanced anomaly detection using pyod
    - Reinforcement learning anomaly detection
    """

    def __init__(self,
                 confidence_level: float = 0.95,
                 default_risk_threshold: float = 0.7,
                 enable_rl_training: bool = False):
        """
        Initialize the statistical analysis engine.

        Args:
            confidence_level: Statistical confidence level (default 0.95)
            default_risk_threshold: Default risk threshold for flagging (default 0.7)
            enable_rl_training: Whether to enable RL model training
        """
        self.confidence_level = confidence_level
        self.default_risk_threshold = default_risk_threshold
        self.enable_rl_training = enable_rl_training

        # Statistical constants
        self.alpha = 1 - confidence_level
        self.critical_values = {
            0.95: 15.51,  # Chi-square critical value for 8 degrees of freedom at 95%
            0.99: 20.09   # Chi-square critical value for 8 degrees of freedom at 99%
        }

        # Initialize models
        self.models = {}
        self.scalers = {}
        self.rl_models = {}

        # Benford's Law expected distributions
        self.benford_first_digit = {
            str(d): math.log10(1 + 1/d) for d in range(1, 10)
        }

        self.benford_second_digit = {
            f"{d}": sum(math.log10(1 + 1/(10*i + d)) for i in range(1, 10))
            for d in range(0, 10)
        }

        logger.info("Statistical Analysis Engine initialized")

    def analyze_benfords_law(self,
                           data: List[Union[float, int, Decimal]],
                           digit_position: int = 1) -> BenfordAnalysisResult:
        """
        Perform Benford's Law analysis on numerical data.

        Benford's Law states that in many real-life datasets, the first digit
        follows a specific logarithmic distribution. Deviations can indicate
        data manipulation or fraud.

        Args:
            data: List of numerical values to analyze
            digit_position: Position of digit to analyze (1=first, 2=second)

        Returns:
            BenfordAnalysisResult: Comprehensive analysis results
        """
        logger.info(f"Starting Benford's Law analysis for digit position {digit_position}")

        # Clean and extract digits
        clean_data = [abs(float(x)) for x in data if x != 0]

        if digit_position == 1:
            digits = [str(int(x))[0] for x in clean_data if len(str(int(x))) > 0]
            expected_dist = self.benford_first_digit
        elif digit_position == 2:
            digits = [str(int(x))[1] for x in clean_data if len(str(int(x))) > 1]
            expected_dist = self.benford_second_digit
        else:
            raise ValueError("Only first and second digit analysis supported")

        # Calculate observed frequencies
        digit_counts = Counter(digits)
        total_count = len(digits)

        observed_frequencies = {
            digit: count / total_count
            for digit, count in digit_counts.items()
        }

        # Ensure all digits are represented
        all_digits = list(expected_dist.keys())
        for digit in all_digits:
            if digit not in observed_frequencies:
                observed_frequencies[digit] = 0.0

        # Calculate chi-square statistic
        chi_square = 0.0
        deviations = {}

        for digit in all_digits:
            expected = expected_dist[digit]
            observed = observed_frequencies.get(digit, 0.0)
            expected_count = expected * total_count
            observed_count = observed * total_count

            if expected_count > 0:
                chi_square += ((observed_count - expected_count) ** 2) / expected_count

            deviations[digit] = observed - expected

        # Statistical significance test
        critical_value = self.critical_values[self.confidence_level]
        is_significant = chi_square > critical_value
        p_value = self._calculate_p_value(chi_square, len(all_digits) - 1)

        # Calculate risk score (0-1 scale)
        risk_score = min(chi_square / (critical_value * 2), 1.0)

        # Identify anomalous digits (significant deviations)
        anomalous_digits = [
            digit for digit, dev in deviations.items()
            if abs(dev) > 0.05  # 5% threshold
        ]

        # Generate recommendations
        recommendations = self._generate_benford_recommendations(
            is_significant, risk_score, anomalous_digits, digit_position
        )

        return BenfordAnalysisResult(
            digit_position=digit_position,
            chi_square_statistic=chi_square,
            p_value=p_value,
            critical_value=critical_value,
            is_significant=is_significant,
            risk_score=risk_score,
            expected_frequencies=expected_dist,
            observed_frequencies=observed_frequencies,
            deviations=deviations,
            anomalous_digits=anomalous_digits,
            sample_size=total_count,
            recommendations=recommendations
        )

    def analyze_newcomb_benford_law(self,
                                  data: List[Union[float, int, Decimal]]) -> BenfordAnalysisResult:
        """
        Perform Newcomb-Benford's Law analysis specifically for first digits.

        This is a specialized implementation focusing on the first-digit
        distribution with enhanced fraud detection capabilities.

        Args:
            data: List of numerical values to analyze

        Returns:
            BenfordAnalysisResult: Enhanced first-digit analysis results
        """
        logger.info("Starting Newcomb-Benford's Law analysis")

        # Use enhanced first-digit analysis
        result = self.analyze_benfords_law(data, digit_position=1)

        # Enhanced fraud indicators for first-digit analysis
        enhanced_risk_factors = []

        # Check for round number bias (excessive 1s and 5s)
        if (result.observed_frequencies.get('1', 0) > 0.35 or
            result.observed_frequencies.get('5', 0) > 0.12):
            enhanced_risk_factors.append("Round number bias detected")
            result.risk_score = min(result.risk_score + 0.1, 1.0)

        # Check for psychological number preference (excessive 2s, 3s)
        if (result.observed_frequencies.get('2', 0) > 0.20 or
            result.observed_frequencies.get('3', 0) > 0.15):
            enhanced_risk_factors.append("Psychological number preference detected")
            result.risk_score = min(result.risk_score + 0.05, 1.0)

        # Add enhanced recommendations
        result.recommendations.extend([
            "Review transactions with round numbers",
            "Investigate clustering around specific digits",
            "Cross-reference with vendor payment patterns"
        ])

        if enhanced_risk_factors:
            result.recommendations.extend(enhanced_risk_factors)

        return result

    def analyze_zipfs_law(self,
                         text_data: List[str],
                         min_word_length: int = 3) -> ZipfAnalysisResult:
        """
        Perform Zipf's Law analysis on text data for pattern detection.

        Zipf's Law states that the frequency of a word is inversely proportional
        to its rank. Deviations can indicate unusual text patterns or manipulation.

        Args:
            text_data: List of text strings to analyze
            min_word_length: Minimum word length to consider

        Returns:
            ZipfAnalysisResult: Comprehensive text analysis results
        """
        logger.info("Starting Zipf's Law analysis")

        # Combine and clean text
        combined_text = ' '.join(text_data).lower()
        words = re.findall(r'\b[a-zA-Z]{' + str(min_word_length) + ',}\b', combined_text)

        # Calculate word frequencies
        word_counts = Counter(words)
        total_words = len(words)

        # Sort by frequency (rank)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Calculate rank-frequency relationship
        word_rank_frequencies = {}
        observed_frequencies = []
        expected_frequencies = []

        for rank, (word, count) in enumerate(sorted_words, 1):
            frequency = count / total_words
            expected_freq = 1 / rank  # Zipf's law: f(r) = 1/r

            word_rank_frequencies[word] = {
                'rank': rank,
                'frequency': frequency,
                'count': count
            }

            observed_frequencies.append(frequency)
            expected_frequencies.append(expected_freq)

        # Normalize expected frequencies
        total_expected = sum(expected_frequencies)
        expected_frequencies = [f / total_expected for f in expected_frequencies]

        # Calculate correlation coefficient
        correlation_coeff = np.corrcoef(
            np.log(observed_frequencies[:100]),  # Top 100 words
            np.log(expected_frequencies[:100])
        )[0, 1] if len(observed_frequencies) >= 100 else 0.0

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = self._ks_test(observed_frequencies, expected_frequencies)

        # Calculate risk score
        risk_score = max(0, 1 - abs(correlation_coeff))

        # Identify anomalous words (significant deviations)
        anomalous_words = []
        expected_vs_observed = {}

        for i, (word, _) in enumerate(sorted_words[:50]):  # Top 50 words
            if i < len(expected_frequencies):
                expected = expected_frequencies[i]
                observed = observed_frequencies[i]
                deviation = abs(observed - expected) / expected

                expected_vs_observed[word] = (expected, observed)

                if deviation > 0.5:  # 50% deviation threshold
                    anomalous_words.append(word)

        # Generate recommendations
        recommendations = self._generate_zipf_recommendations(
            correlation_coeff, risk_score, anomalous_words
        )

        return ZipfAnalysisResult(
            correlation_coefficient=correlation_coeff,
            kolmogorov_smirnov_statistic=ks_statistic,
            p_value=p_value,
            risk_score=risk_score,
            word_rank_frequencies=word_rank_frequencies,
            expected_vs_observed=expected_vs_observed,
            anomalous_words=anomalous_words,
            text_sample_size=total_words,
            recommendations=recommendations
        )

    def detect_anomalies_isolation_forest(self,
                                        data: pd.DataFrame,
                                        contamination: float = 0.1) -> AnomalyDetectionResult:
        """
        Detect anomalies using Isolation Forest algorithm.

        Args:
            data: DataFrame with financial data
            contamination: Expected proportion of anomalies

        Returns:
            AnomalyDetectionResult: Anomaly detection results
        """
        logger.info("Starting Isolation Forest anomaly detection")

        # Prepare numerical data
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numerical_cols].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        # Predict anomalies
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)

        # Process results
        anomalies_mask = anomaly_labels == -1
        anomaly_indices = np.where(anomalies_mask)[0]

        # Calculate risk scores for each record
        risk_scores = MinMaxScaler().fit_transform(
            -anomaly_scores.reshape(-1, 1)
        ).flatten()

        # Extract anomalous records
        anomalous_records = []
        for idx in anomaly_indices:
            record = {
                'index': int(idx),
                'risk_score': float(risk_scores[idx]),
                'anomaly_score': float(anomaly_scores[idx]),
                'data': data.iloc[idx].to_dict()
            }
            anomalous_records.append(record)

        # Calculate feature importance (approximate)
        feature_importance = self._calculate_feature_importance_isolation(
            iso_forest, X_scaled, numerical_cols
        )

        # Generate recommendations
        recommendations = self._generate_isolation_forest_recommendations(
            len(anomaly_indices), len(data), feature_importance
        )

        return AnomalyDetectionResult(
            method=StatisticalMethod.ISOLATION_FOREST,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            total_samples=len(data),
            anomalies_detected=len(anomaly_indices),
            anomaly_rate=len(anomaly_indices) / len(data),
            overall_risk_score=np.mean(risk_scores[anomalies_mask]) if anomalies_mask.any() else 0.0,
            confidence_score=0.85,  # Isolation Forest typical confidence
            anomalous_records=anomalous_records,
            feature_importance=feature_importance,
            model_metrics={'contamination': contamination, 'n_estimators': 100},
            recommendations=recommendations
        )

    def detect_anomalies_pyod_ensemble(self,
                                     data: pd.DataFrame,
                                     methods: List[str] = None) -> AnomalyDetectionResult:
        """
        Detect anomalies using PyOD ensemble methods.

        Args:
            data: DataFrame with financial data
            methods: List of PyOD methods to use

        Returns:
            AnomalyDetectionResult: Ensemble anomaly detection results
        """
        logger.info("Starting PyOD ensemble anomaly detection")

        if methods is None:
            methods = ['iforest', 'lof', 'ocsvm', 'knn']

        # Prepare numerical data
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numerical_cols].fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize models
        models = {}
        if 'iforest' in methods:
            models['iforest'] = IForest(contamination=0.1, random_state=42)
        if 'lof' in methods:
            models['lof'] = LOF(contamination=0.1)
        if 'ocsvm' in methods:
            models['ocsvm'] = OCSVM(contamination=0.1)
        if 'knn' in methods:
            models['knn'] = KNN(contamination=0.1)

        # Fit models and collect predictions
        ensemble_scores = []
        model_predictions = {}

        for name, model in models.items():
            try:
                model.fit(X_scaled)
                scores = model.decision_function(X_scaled)
                predictions = model.predict(X_scaled)

                ensemble_scores.append(scores)
                model_predictions[name] = predictions

                logger.info(f"Successfully fitted {name} model")
            except Exception as e:
                logger.warning(f"Failed to fit {name} model: {e}")

        # Ensemble scoring (average of normalized scores)
        if ensemble_scores:
            # Normalize scores to 0-1 range
            normalized_scores = []
            for scores in ensemble_scores:
                norm_scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()
                normalized_scores.append(norm_scores)

            # Average ensemble scores
            final_scores = np.mean(normalized_scores, axis=0)

            # Determine anomalies based on threshold
            threshold = np.percentile(final_scores, 90)  # Top 10% as anomalies
            anomaly_mask = final_scores > threshold
            anomaly_indices = np.where(anomaly_mask)[0]

            # Extract anomalous records
            anomalous_records = []
            for idx in anomaly_indices:
                record = {
                    'index': int(idx),
                    'risk_score': float(final_scores[idx]),
                    'ensemble_agreement': sum(
                        1 for pred in model_predictions.values()
                        if pred[idx] == 1
                    ) / len(model_predictions),
                    'data': data.iloc[idx].to_dict()
                }
                anomalous_records.append(record)

            # Calculate feature importance (weighted average)
            feature_importance = {}
            for col in numerical_cols:
                importance = np.random.random()  # Placeholder - would be calculated properly
                feature_importance[col] = importance

            # Normalize feature importance
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {
                    k: v / total_importance for k, v in feature_importance.items()
                }
        else:
            final_scores = np.zeros(len(data))
            anomaly_indices = []
            anomalous_records = []
            feature_importance = {}

        # Generate recommendations
        recommendations = self._generate_pyod_ensemble_recommendations(
            len(anomaly_indices), len(data), methods
        )

        return AnomalyDetectionResult(
            method=StatisticalMethod.PYOD_ENSEMBLE,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            total_samples=len(data),
            anomalies_detected=len(anomaly_indices),
            anomaly_rate=len(anomaly_indices) / len(data) if len(data) > 0 else 0.0,
            overall_risk_score=np.mean(final_scores[anomaly_mask]) if anomaly_mask.any() else 0.0,
            confidence_score=len(models) / 4.0,  # Confidence based on successful models
            anomalous_records=anomalous_records,
            feature_importance=feature_importance,
            model_metrics={'methods_used': list(models.keys()), 'threshold': threshold if 'threshold' in locals() else 0.0},
            recommendations=recommendations
        )

    def detect_anomalies_reinforcement_learning(self,
                                              data: pd.DataFrame,
                                              model_type: str = "DQN",
                                              training_episodes: int = 1000) -> RLAnomalyDetectionResult:
        """
        Detect anomalies using reinforcement learning models.

        Args:
            data: DataFrame with financial data
            model_type: Type of RL model ("DQN" or "PPO")
            training_episodes: Number of training episodes

        Returns:
            RLAnomalyDetectionResult: RL-based anomaly detection results
        """
        logger.info(f"Starting RL anomaly detection with {model_type}")

        # Create custom environment for anomaly detection
        env = self._create_anomaly_detection_env(data)

        # Initialize RL model
        if model_type == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=100,
                batch_size=32,
                verbose=0
            )
        elif model_type == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                n_steps=2048,
                batch_size=64,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model if training is enabled
        if self.enable_rl_training:
            logger.info(f"Training {model_type} model for {training_episodes} episodes")
            model.learn(total_timesteps=training_episodes)

        # Generate predictions
        obs = env.reset()
        flagged_transactions = []
        action_probabilities = {"FLAG": 0.0, "IGNORE": 0.0}
        rewards = []

        for i in range(len(data)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Record flagged transactions
            if action == 1:  # FLAG action
                flagged_transactions.append({
                    'index': i,
                    'risk_score': float(reward if reward > 0 else 0.5),
                    'action': 'FLAG',
                    'confidence': 0.8,  # Placeholder
                    'data': data.iloc[i].to_dict() if i < len(data) else {}
                })
                action_probabilities["FLAG"] += 1
            else:
                action_probabilities["IGNORE"] += 1

            rewards.append(reward)

            if done:
                obs = env.reset()

        # Normalize action probabilities
        total_actions = sum(action_probabilities.values())
        if total_actions > 0:
            action_probabilities = {
                k: v / total_actions for k, v in action_probabilities.items()
            }

        # Calculate performance metrics
        average_reward = np.mean(rewards) if rewards else 0.0
        convergence_score = min(abs(average_reward) / 1.0, 1.0) if average_reward != 0 else 0.0

        # Generate confidence intervals (placeholder)
        confidence_intervals = {
            "reward": (average_reward - 0.1, average_reward + 0.1),
            "risk_score": (0.4, 0.8)
        }

        # Model performance metrics
        model_performance = {
            "average_reward": average_reward,
            "total_flags": len(flagged_transactions),
            "flag_rate": len(flagged_transactions) / len(data) if len(data) > 0 else 0.0,
            "convergence": convergence_score
        }

        # Adaptive threshold based on performance
        adaptive_threshold = 0.5 + (convergence_score * 0.3)

        # Generate RL-specific recommendations
        recommendations = self._generate_rl_recommendations(
            model_type, average_reward, len(flagged_transactions), len(data)
        )

        return RLAnomalyDetectionResult(
            model_type=model_type,
            episodes_trained=training_episodes,
            average_reward=average_reward,
            convergence_score=convergence_score,
            flagged_transactions=flagged_transactions,
            action_probabilities=action_probabilities,
            confidence_intervals=confidence_intervals,
            model_performance=model_performance,
            adaptive_threshold=adaptive_threshold,
            recommendations=recommendations
        )

    def calculate_comprehensive_risk_score(self,
                                         results: List[Union[BenfordAnalysisResult,
                                                           ZipfAnalysisResult,
                                                           AnomalyDetectionResult,
                                                           RLAnomalyDetectionResult]]) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score from multiple analysis results.

        Args:
            results: List of analysis results from different methods

        Returns:
            Dict containing comprehensive risk assessment
        """
        logger.info("Calculating comprehensive risk score")

        # Collect individual risk scores with weights
        weighted_scores = []
        method_scores = {}
        confidence_scores = []

        for result in results:
            if isinstance(result, BenfordAnalysisResult):
                weight = 0.25 if result.digit_position == 1 else 0.15
                weighted_scores.append(result.risk_score * weight)
                method_scores["benford"] = result.risk_score
                confidence_scores.append(0.9)  # High confidence for statistical tests

            elif isinstance(result, ZipfAnalysisResult):
                weight = 0.15
                weighted_scores.append(result.risk_score * weight)
                method_scores["zipf"] = result.risk_score
                confidence_scores.append(0.8)

            elif isinstance(result, AnomalyDetectionResult):
                if result.method == StatisticalMethod.ISOLATION_FOREST:
                    weight = 0.20
                elif result.method == StatisticalMethod.PYOD_ENSEMBLE:
                    weight = 0.25
                else:
                    weight = 0.15

                weighted_scores.append(result.overall_risk_score * weight)
                method_scores[result.method.value] = result.overall_risk_score
                confidence_scores.append(result.confidence_score)

            elif isinstance(result, RLAnomalyDetectionResult):
                weight = 0.30  # Higher weight for adaptive RL
                rl_risk_score = min(len(result.flagged_transactions) / 100, 1.0)
                weighted_scores.append(rl_risk_score * weight)
                method_scores["reinforcement_learning"] = rl_risk_score
                confidence_scores.append(result.convergence_score)

        # Calculate final risk score
        final_risk_score = sum(weighted_scores)
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        # Determine risk level
        if final_risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif final_risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif final_risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Generate comprehensive recommendations
        comprehensive_recommendations = self._generate_comprehensive_recommendations(
            final_risk_score, risk_level, method_scores
        )

        return {
            "overall_risk_score": final_risk_score,
            "risk_level": risk_level,
            "confidence_score": overall_confidence,
            "method_scores": method_scores,
            "weighted_contributions": {
                f"method_{i}": score for i, score in enumerate(weighted_scores)
            },
            "recommendations": comprehensive_recommendations,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "methods_used": len(results),
            "risk_factors": self._identify_risk_factors(method_scores, final_risk_score)
        }

    # Helper methods

    def _calculate_p_value(self, chi_square: float, degrees_of_freedom: int) -> float:
        """Calculate p-value for chi-square statistic."""
        # Simplified p-value calculation (would use scipy.stats in production)
        if chi_square > 20:
            return 0.001
        elif chi_square > 15:
            return 0.05
        elif chi_square > 10:
            return 0.1
        else:
            return 0.5

    def _ks_test(self, observed: List[float], expected: List[float]) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test."""
        # Simplified KS test (would use scipy.stats in production)
        n = min(len(observed), len(expected))
        max_diff = max(abs(observed[i] - expected[i]) for i in range(n))

        # Calculate approximate p-value
        p_value = 2 * math.exp(-2 * n * max_diff ** 2) if max_diff > 0 else 1.0

        return max_diff, p_value

    def _calculate_feature_importance_isolation(self, model, X, feature_names) -> Dict[str, float]:
        """Calculate approximate feature importance for Isolation Forest."""
        # Simplified feature importance (would use more sophisticated methods)
        importance = {}
        n_features = X.shape[1]

        for i, name in enumerate(feature_names):
            # Random importance for demonstration
            importance[name] = np.random.random()

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def _create_anomaly_detection_env(self, data: pd.DataFrame):
        """Create custom environment for RL anomaly detection."""

        class AnomalyDetectionEnv(gym.Env):
            def __init__(self, data):
                super(AnomalyDetectionEnv, self).__init__()
                self.data = data
                self.current_index = 0

                # Action space: 0 = IGNORE, 1 = FLAG
                self.action_space = spaces.Discrete(2)

                # Observation space: normalized features
                n_features = len(data.select_dtypes(include=[np.number]).columns)
                self.observation_space = spaces.Box(
                    low=-1, high=1, shape=(n_features,), dtype=np.float32
                )

                # Normalize data
                self.scaler = StandardScaler()
                numerical_data = data.select_dtypes(include=[np.number]).fillna(0)
                self.normalized_data = self.scaler.fit_transform(numerical_data)

            def reset(self):
                self.current_index = 0
                return self._get_observation()

            def step(self, action):
                # Calculate reward based on action
                # This is simplified - would include human feedback in production
                reward = 0.1 if action == 1 else 0.05  # Slight preference for flagging

                self.current_index += 1
                done = self.current_index >= len(self.data)

                if done:
                    obs = np.zeros(self.observation_space.shape)
                else:
                    obs = self._get_observation()

                return obs, reward, done, {}

            def _get_observation(self):
                if self.current_index < len(self.normalized_data):
                    return self.normalized_data[self.current_index].astype(np.float32)
                else:
                    return np.zeros(self.observation_space.shape).astype(np.float32)

        return AnomalyDetectionEnv(data)

    def _generate_benford_recommendations(self, is_significant: bool, risk_score: float,
                                        anomalous_digits: List[str], digit_position: int) -> List[str]:
        """Generate recommendations for Benford's Law analysis."""
        recommendations = []

        if is_significant:
            recommendations.append(f"Significant deviation from Benford's Law detected for digit position {digit_position}")
            recommendations.append("Consider detailed review of transactions with unusual digit patterns")

        if risk_score > 0.7:
            recommendations.append("High risk score indicates potential data manipulation")
            recommendations.append("Recommend forensic analysis of affected transactions")

        if anomalous_digits:
            recommendations.append(f"Focus investigation on transactions starting with digits: {', '.join(anomalous_digits)}")

        if digit_position == 1 and '1' in anomalous_digits:
            recommendations.append("Unusual frequency of '1' as first digit may indicate round number preference")

        return recommendations

    def _generate_zipf_recommendations(self, correlation: float, risk_score: float,
                                     anomalous_words: List[str]) -> List[str]:
        """Generate recommendations for Zipf's Law analysis."""
        recommendations = []

        if abs(correlation) < 0.8:
            recommendations.append("Poor correlation with Zipf's Law indicates unusual text patterns")
            recommendations.append("Review document authenticity and text generation methods")

        if risk_score > 0.6:
            recommendations.append("High risk score suggests potential text manipulation")

        if anomalous_words:
            recommendations.append(f"Investigate unusual word usage patterns: {', '.join(anomalous_words[:5])}")

        recommendations.append("Cross-reference text patterns with known document templates")

        return recommendations

    def _generate_isolation_forest_recommendations(self, anomalies: int, total: int,
                                                 feature_importance: Dict[str, float]) -> List[str]:
        """Generate recommendations for Isolation Forest analysis."""
        recommendations = []

        anomaly_rate = anomalies / total if total > 0 else 0

        if anomaly_rate > 0.1:
            recommendations.append(f"High anomaly rate ({anomaly_rate:.1%}) detected")
            recommendations.append("Consider investigating data quality and processing procedures")

        # Top important features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_features:
            feature_names = [f[0] for f in top_features]
            recommendations.append(f"Focus analysis on key features: {', '.join(feature_names)}")

        recommendations.append("Review flagged transactions for common patterns")

        return recommendations

    def _generate_pyod_ensemble_recommendations(self, anomalies: int, total: int,
                                              methods: List[str]) -> List[str]:
        """Generate recommendations for PyOD ensemble analysis."""
        recommendations = []

        anomaly_rate = anomalies / total if total > 0 else 0

        recommendations.append(f"Ensemble of {len(methods)} methods detected {anomalies} anomalies")

        if anomaly_rate > 0.15:
            recommendations.append("High consensus anomaly rate suggests systematic issues")

        recommendations.append("Review transactions flagged by multiple methods with high priority")
        recommendations.append("Consider expanding analysis with additional detection methods")

        return recommendations

    def _generate_rl_recommendations(self, model_type: str, avg_reward: float,
                                   flags: int, total: int) -> List[str]:
        """Generate recommendations for RL-based analysis."""
        recommendations = []

        flag_rate = flags / total if total > 0 else 0

        recommendations.append(f"{model_type} model flagged {flags} transactions ({flag_rate:.1%})")

        if avg_reward < 0:
            recommendations.append("Low average reward suggests model needs additional training")
            recommendations.append("Consider providing more human feedback for model improvement")

        if flag_rate > 0.2:
            recommendations.append("High flag rate indicates aggressive detection threshold")
            recommendations.append("Consider adjusting model sensitivity based on business requirements")

        recommendations.append("Continue collecting human feedback to improve model accuracy")

        return recommendations

    def _generate_comprehensive_recommendations(self, risk_score: float, risk_level: RiskLevel,
                                              method_scores: Dict[str, float]) -> List[str]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []

        # Overall risk assessment
        recommendations.append(f"Overall risk level: {risk_level.value.upper()}")
        recommendations.append(f"Comprehensive risk score: {risk_score:.2f}")

        # Method-specific recommendations
        high_risk_methods = [method for method, score in method_scores.items() if score > 0.7]
        if high_risk_methods:
            recommendations.append(f"High risk detected by: {', '.join(high_risk_methods)}")

        # Risk level specific actions
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED",
                "Halt processing and conduct emergency review",
                "Escalate to senior management and compliance team",
                "Consider external forensic audit"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Urgent review required within 24 hours",
                "Increase sampling and testing procedures",
                "Implement additional controls"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Schedule detailed review within one week",
                "Monitor closely for trend development",
                "Consider additional testing"
            ])
        else:
            recommendations.extend([
                "Continue standard monitoring procedures",
                "Document findings for trend analysis"
            ])

        return recommendations

    def _identify_risk_factors(self, method_scores: Dict[str, float],
                             overall_score: float) -> List[str]:
        """Identify specific risk factors based on analysis results."""
        risk_factors = []

        # Statistical anomalies
        if method_scores.get("benford", 0) > 0.6:
            risk_factors.append("Unusual digit distribution patterns")

        if method_scores.get("zipf", 0) > 0.6:
            risk_factors.append("Atypical text patterns")

        # ML-based anomalies
        if method_scores.get("isolation_forest", 0) > 0.7:
            risk_factors.append("Statistical outliers detected")

        if method_scores.get("pyod_ensemble", 0) > 0.7:
            risk_factors.append("Multiple ML methods agree on anomalies")

        if method_scores.get("reinforcement_learning", 0) > 0.7:
            risk_factors.append("Adaptive AI flagged unusual patterns")

        # Overall assessment
        if overall_score > 0.8:
            risk_factors.append("Multiple detection methods in agreement")

        return risk_factors

    def generate_audit_findings(self,
                              comprehensive_results: Dict[str, Any],
                              session_id: str) -> List[AuditFinding]:
        """
        Generate structured audit findings from statistical analysis results.

        Args:
            comprehensive_results: Results from calculate_comprehensive_risk_score
            session_id: Audit session identifier

        Returns:
            List of AuditFinding objects for LangGraph workflow
        """
        findings = []

        risk_score = comprehensive_results["overall_risk_score"]
        risk_level = comprehensive_results["risk_level"]
        recommendations = comprehensive_results["recommendations"]
        risk_factors = comprehensive_results["risk_factors"]

        # Create main finding
        main_finding = AuditFinding(
            id=f"STAT_ANALYSIS_{session_id}_{int(datetime.utcnow().timestamp())}",
            audit_session_id=session_id,
            category="Statistical Analysis",
            severity=risk_level,
            title="Comprehensive Statistical Anomaly Analysis",
            description=f"Statistical analysis completed with overall risk score of {risk_score:.2f}. "
                       f"Risk factors identified: {', '.join(risk_factors[:3])}",
            recommendation="; ".join(recommendations[:3]),
            confidence_score=comprehensive_results["confidence_score"],
            affected_accounts=[],
            financial_impact=None,
            regulatory_reference="SOX Section 404, PCAOB AS 2201",
            status="open",
            created_at=datetime.utcnow()
        )
        findings.append(main_finding)

        # Create method-specific findings for high-risk methods
        method_scores = comprehensive_results["method_scores"]

        for method, score in method_scores.items():
            if score > 0.7:  # High risk threshold
                method_finding = AuditFinding(
                    id=f"{method.upper()}_{session_id}_{int(datetime.utcnow().timestamp())}",
                    audit_session_id=session_id,
                    category=f"Statistical Analysis - {method.replace('_', ' ').title()}",
                    severity=RiskLevel.HIGH if score > 0.8 else RiskLevel.MEDIUM,
                    title=f"High Risk Detected by {method.replace('_', ' ').title()}",
                    description=f"{method.replace('_', ' ').title()} analysis detected anomalies "
                               f"with risk score {score:.2f}",
                    recommendation=f"Investigate transactions flagged by {method} analysis",
                    confidence_score=score,
                    affected_accounts=[],
                    financial_impact=None,
                    regulatory_reference="PCAOB AS 2301 - Audit Evidence",
                    status="open",
                    created_at=datetime.utcnow()
                )
                findings.append(method_finding)

        logger.info(f"Generated {len(findings)} audit findings from statistical analysis")
        return findings