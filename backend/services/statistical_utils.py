"""
Statistical utilities and helpers for the Financial Audit System.

This module provides common statistical functions, constants, and utilities
used across the statistical analysis engine and other components.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from decimal import Decimal
from collections import Counter
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class StatisticalConstants:
    """Statistical constants and thresholds used throughout the system"""

    # Benford's Law expected first digit probabilities
    BENFORD_FIRST_DIGIT = {
        '1': 0.30103,  # log10(1 + 1/1)
        '2': 0.17609,  # log10(1 + 1/2)
        '3': 0.12494,  # log10(1 + 1/3)
        '4': 0.09691,  # log10(1 + 1/4)
        '5': 0.07918,  # log10(1 + 1/5)
        '6': 0.06695,  # log10(1 + 1/6)
        '7': 0.05799,  # log10(1 + 1/7)
        '8': 0.05115,  # log10(1 + 1/8)
        '9': 0.04576   # log10(1 + 1/9)
    }

    # Benford's Law expected second digit probabilities
    BENFORD_SECOND_DIGIT = {
        '0': 0.11968, '1': 0.11389, '2': 0.10882, '3': 0.10433, '4': 0.10031,
        '5': 0.09668, '6': 0.09337, '7': 0.09035, '8': 0.08757, '9': 0.08500
    }

    # Chi-square critical values at different confidence levels
    CHI_SQUARE_CRITICAL = {
        0.90: {'df_8': 13.362, 'df_9': 14.684},  # 90% confidence
        0.95: {'df_8': 15.507, 'df_9': 16.919},  # 95% confidence
        0.99: {'df_8': 20.090, 'df_9': 21.666}   # 99% confidence
    }

    # Risk score thresholds
    RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8,
        'critical': 0.9
    }

    # Anomaly detection contamination rates
    CONTAMINATION_RATES = {
        'conservative': 0.05,   # 5% expected anomalies
        'moderate': 0.10,       # 10% expected anomalies
        'aggressive': 0.15      # 15% expected anomalies
    }

    # Statistical significance levels
    SIGNIFICANCE_LEVELS = {
        'strict': 0.01,     # 99% confidence
        'standard': 0.05,   # 95% confidence
        'relaxed': 0.10     # 90% confidence
    }


class BenfordUtils:
    """Utilities for Benford's Law analysis"""

    @staticmethod
    def calculate_expected_frequencies(digit_position: int = 1) -> Dict[str, float]:
        """
        Calculate expected frequencies for Benford's Law.

        Args:
            digit_position: Position of digit (1 for first, 2 for second)

        Returns:
            Dictionary mapping digits to expected frequencies
        """
        if digit_position == 1:
            return StatisticalConstants.BENFORD_FIRST_DIGIT.copy()
        elif digit_position == 2:
            return StatisticalConstants.BENFORD_SECOND_DIGIT.copy()
        else:
            raise ValueError(f"Unsupported digit position: {digit_position}")

    @staticmethod
    def extract_digits(data: List[Union[float, int, Decimal]],
                      position: int = 1) -> List[str]:
        """
        Extract specific digit positions from numerical data.

        Args:
            data: List of numerical values
            position: Digit position to extract (1=first, 2=second)

        Returns:
            List of extracted digits as strings
        """
        digits = []
        for value in data:
            if value is None or value == 0:
                continue

            # Convert to positive integer string
            abs_value = abs(float(value))
            int_str = str(int(abs_value))

            if len(int_str) >= position:
                digit = int_str[position - 1]
                digits.append(digit)

        return digits

    @staticmethod
    def calculate_chi_square(observed: Dict[str, float],
                           expected: Dict[str, float],
                           sample_size: int) -> Tuple[float, float, int]:
        """
        Calculate chi-square statistic for goodness of fit test.

        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            sample_size: Total sample size

        Returns:
            Tuple of (chi_square_statistic, p_value, degrees_of_freedom)
        """
        chi_square = 0.0
        degrees_of_freedom = len(expected) - 1

        for digit, exp_freq in expected.items():
            obs_freq = observed.get(digit, 0.0)
            expected_count = exp_freq * sample_size
            observed_count = obs_freq * sample_size

            if expected_count > 0:
                chi_square += ((observed_count - expected_count) ** 2) / expected_count

        # Calculate p-value using chi-square distribution
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)

        return chi_square, p_value, degrees_of_freedom

    @staticmethod
    def detect_manipulation_patterns(observed: Dict[str, float]) -> List[str]:
        """
        Detect common manipulation patterns in digit distributions.

        Args:
            observed: Observed digit frequencies

        Returns:
            List of detected manipulation patterns
        """
        patterns = []

        # Check for round number bias (excessive 1s and 5s)
        if observed.get('1', 0) > 0.35:
            patterns.append("Excessive frequency of digit '1' (round number bias)")

        if observed.get('5', 0) > 0.12:
            patterns.append("Excessive frequency of digit '5' (round number preference)")

        # Check for psychological number preference
        if observed.get('2', 0) > 0.20:
            patterns.append("Unusual frequency of digit '2' (psychological preference)")

        if observed.get('3', 0) > 0.15:
            patterns.append("Unusual frequency of digit '3' (psychological preference)")

        # Check for avoidance of certain digits
        low_threshold = 0.02
        avoided_digits = [d for d, freq in observed.items()
                         if freq < low_threshold and d in ['7', '8', '9']]
        if avoided_digits:
            patterns.append(f"Unusually low frequency of digits: {', '.join(avoided_digits)}")

        # Check for even distribution (non-natural)
        frequencies = list(observed.values())
        if len(frequencies) > 1:
            coefficient_of_variation = np.std(frequencies) / np.mean(frequencies)
            if coefficient_of_variation < 0.3:  # Too uniform
                patterns.append("Suspiciously uniform digit distribution")

        return patterns


class ZipfUtils:
    """Utilities for Zipf's Law analysis"""

    @staticmethod
    def calculate_zipf_expected(rank: int, total_words: int,
                              alpha: float = 1.0) -> float:
        """
        Calculate expected frequency under Zipf's Law.

        Args:
            rank: Word rank (1-based)
            total_words: Total number of words
            alpha: Zipf exponent (default 1.0 for classic Zipf)

        Returns:
            Expected frequency
        """
        # Zipf's law: f(r) = C / r^alpha
        # Where C is a normalization constant
        harmonic_sum = sum(1 / (i ** alpha) for i in range(1, total_words + 1))
        c = 1 / harmonic_sum
        return c / (rank ** alpha)

    @staticmethod
    def preprocess_text(text_list: List[str],
                       min_length: int = 3,
                       remove_stopwords: bool = False) -> List[str]:
        """
        Preprocess text for Zipf analysis.

        Args:
            text_list: List of text strings
            min_length: Minimum word length
            remove_stopwords: Whether to remove common stopwords

        Returns:
            List of preprocessed words
        """
        import re

        # Common stopwords (basic set)
        stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must', 'a', 'an'
        } if remove_stopwords else set()

        words = []
        for text in text_list:
            # Convert to lowercase and extract words
            clean_text = text.lower()
            text_words = re.findall(r'\b[a-zA-Z]+\b', clean_text)

            # Filter by length and stopwords
            filtered_words = [
                word for word in text_words
                if len(word) >= min_length and word not in stopwords
            ]
            words.extend(filtered_words)

        return words

    @staticmethod
    def calculate_correlation_with_zipf(word_frequencies: Dict[str, int],
                                      top_n: int = 100) -> float:
        """
        Calculate correlation between observed and expected Zipf distribution.

        Args:
            word_frequencies: Dictionary of word frequencies
            top_n: Number of top words to analyze

        Returns:
            Correlation coefficient
        """
        # Sort words by frequency
        sorted_words = sorted(word_frequencies.items(),
                            key=lambda x: x[1], reverse=True)[:top_n]

        if len(sorted_words) < 10:  # Need minimum words for meaningful correlation
            return 0.0

        # Calculate observed log frequencies
        observed_log_freq = [math.log(freq) for _, freq in sorted_words]

        # Calculate expected log frequencies (Zipf: log(f) = log(C) - alpha * log(r))
        expected_log_freq = [math.log(1/rank) for rank in range(1, len(sorted_words) + 1)]

        # Calculate correlation
        try:
            correlation = np.corrcoef(observed_log_freq, expected_log_freq)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0


class AnomalyUtils:
    """Utilities for anomaly detection"""

    @staticmethod
    def normalize_features(data: pd.DataFrame,
                          method: str = 'standard') -> Tuple[np.ndarray, object]:
        """
        Normalize features for anomaly detection.

        Args:
            data: DataFrame with features
            method: Normalization method ('standard', 'minmax', 'robust')

        Returns:
            Tuple of (normalized_data, scaler_object)
        """
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numerical_cols].fillna(0)

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        X_normalized = scaler.fit_transform(X)
        return X_normalized, scaler

    @staticmethod
    def calculate_anomaly_scores(predictions: np.ndarray,
                               scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate various anomaly scoring metrics.

        Args:
            predictions: Binary predictions (1 = anomaly, -1 = normal)
            scores: Continuous anomaly scores

        Returns:
            Dictionary of scoring metrics
        """
        anomaly_mask = predictions == 1
        normal_mask = predictions == -1

        metrics = {
            'anomaly_rate': np.mean(anomaly_mask) if len(predictions) > 0 else 0.0,
            'mean_anomaly_score': np.mean(scores[anomaly_mask]) if anomaly_mask.any() else 0.0,
            'mean_normal_score': np.mean(scores[normal_mask]) if normal_mask.any() else 0.0,
            'score_separation': 0.0,
            'max_score': np.max(scores) if len(scores) > 0 else 0.0,
            'min_score': np.min(scores) if len(scores) > 0 else 0.0,
            'score_std': np.std(scores) if len(scores) > 0 else 0.0
        }

        # Calculate separation between anomaly and normal scores
        if anomaly_mask.any() and normal_mask.any():
            metrics['score_separation'] = (
                metrics['mean_anomaly_score'] - metrics['mean_normal_score']
            )

        return metrics

    @staticmethod
    def ensemble_anomaly_scores(score_lists: List[np.ndarray],
                              method: str = 'average') -> np.ndarray:
        """
        Combine multiple anomaly scores using ensemble methods.

        Args:
            score_lists: List of score arrays from different models
            method: Ensemble method ('average', 'max', 'voting')

        Returns:
            Combined anomaly scores
        """
        if not score_lists:
            return np.array([])

        # Normalize all scores to [0, 1] range
        normalized_scores = []
        for scores in score_lists:
            if len(scores) == 0:
                continue
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score - min_score > 0:
                norm_scores = (scores - min_score) / (max_score - min_score)
            else:
                norm_scores = np.zeros_like(scores)
            normalized_scores.append(norm_scores)

        if not normalized_scores:
            return np.array([])

        scores_array = np.array(normalized_scores)

        if method == 'average':
            return np.mean(scores_array, axis=0)
        elif method == 'max':
            return np.max(scores_array, axis=0)
        elif method == 'voting':
            # Binary voting: each model votes anomaly if score > threshold
            threshold = 0.7
            votes = (scores_array > threshold).astype(int)
            return np.mean(votes, axis=0)
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")


class RiskScoringUtils:
    """Utilities for risk scoring and assessment"""

    @staticmethod
    def calculate_composite_risk_score(scores: Dict[str, float],
                                     weights: Dict[str, float] = None) -> float:
        """
        Calculate composite risk score from multiple components.

        Args:
            scores: Dictionary of component scores (0-1 scale)
            weights: Dictionary of component weights (must sum to 1)

        Returns:
            Composite risk score (0-1 scale)
        """
        if not scores:
            return 0.0

        # Default equal weights if not provided
        if weights is None:
            weights = {key: 1.0 / len(scores) for key in scores.keys()}

        # Validate weights
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize weights
            weights = {k: v / weight_sum for k, v in weights.items()}

        # Calculate weighted score
        composite_score = 0.0
        for component, score in scores.items():
            weight = weights.get(component, 0.0)
            composite_score += score * weight

        return min(max(composite_score, 0.0), 1.0)  # Clamp to [0, 1]

    @staticmethod
    def map_score_to_risk_level(score: float,
                              thresholds: Dict[str, float] = None) -> str:
        """
        Map numerical risk score to categorical risk level.

        Args:
            score: Risk score (0-1 scale)
            thresholds: Custom thresholds for risk levels

        Returns:
            Risk level string ('low', 'medium', 'high', 'critical')
        """
        if thresholds is None:
            thresholds = StatisticalConstants.RISK_THRESHOLDS

        if score >= thresholds['critical']:
            return 'critical'
        elif score >= thresholds['high']:
            return 'high'
        elif score >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'

    @staticmethod
    def calculate_confidence_intervals(scores: np.ndarray,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for risk scores.

        Args:
            scores: Array of risk scores
            confidence_level: Confidence level (0-1)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) == 0:
            return 0.0, 0.0

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(scores, lower_percentile)
        upper_bound = np.percentile(scores, upper_percentile)

        return lower_bound, upper_bound

    @staticmethod
    def identify_risk_drivers(feature_scores: Dict[str, float],
                            threshold: float = 0.1) -> List[str]:
        """
        Identify key risk drivers based on feature importance scores.

        Args:
            feature_scores: Dictionary of feature importance scores
            threshold: Minimum importance threshold

        Returns:
            List of significant risk drivers
        """
        risk_drivers = []
        sorted_features = sorted(feature_scores.items(),
                               key=lambda x: x[1], reverse=True)

        for feature, score in sorted_features:
            if score >= threshold:
                risk_drivers.append(feature)

        return risk_drivers


class StatisticalValidation:
    """Statistical validation utilities"""

    @staticmethod
    def validate_sample_size(sample_size: int,
                           minimum_required: int = 30) -> Tuple[bool, str]:
        """
        Validate if sample size is adequate for statistical analysis.

        Args:
            sample_size: Actual sample size
            minimum_required: Minimum required sample size

        Returns:
            Tuple of (is_valid, message)
        """
        if sample_size < minimum_required:
            return False, f"Sample size {sample_size} below minimum required {minimum_required}"
        elif sample_size < 100:
            return True, "Sample size adequate but limited statistical power"
        elif sample_size < 1000:
            return True, "Good sample size for statistical analysis"
        else:
            return True, "Excellent sample size for robust statistical analysis"

    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks for statistical analysis.

        Args:
            data: DataFrame to check

        Returns:
            Dictionary of data quality metrics
        """
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_data': {},
            'data_types': {},
            'outliers': {},
            'duplicates': 0,
            'zero_variance_columns': [],
            'high_correlation_pairs': [],
            'overall_quality_score': 0.0
        }

        if len(data) == 0:
            quality_report['overall_quality_score'] = 0.0
            return quality_report

        # Check missing data
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            quality_report['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }

        # Check data types
        quality_report['data_types'] = {
            col: str(dtype) for col, dtype in data.dtypes.items()
        }

        # Check duplicates
        quality_report['duplicates'] = int(data.duplicated().sum())

        # Check numerical columns for outliers and zero variance
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if data[col].var() == 0:
                quality_report['zero_variance_columns'].append(col)
            else:
                # Simple outlier detection using IQR
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

                quality_report['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': round((len(outliers) / len(data)) * 100, 2)
                }

        # Check correlations (only for numerical data with >1 column)
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr().abs()
            # Find high correlations (> 0.9) excluding diagonal
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr.append((col1, col2, round(corr_matrix.iloc[i, j], 3)))

            quality_report['high_correlation_pairs'] = high_corr

        # Calculate overall quality score
        quality_score = 1.0

        # Penalize for missing data
        avg_missing_pct = np.mean([
            item['percentage'] for item in quality_report['missing_data'].values()
        ])
        quality_score -= (avg_missing_pct / 100) * 0.3

        # Penalize for duplicates
        duplicate_pct = (quality_report['duplicates'] / len(data)) * 100
        quality_score -= (duplicate_pct / 100) * 0.2

        # Penalize for zero variance columns
        zero_var_pct = (len(quality_report['zero_variance_columns']) / len(data.columns)) * 100
        quality_score -= (zero_var_pct / 100) * 0.2

        # Penalize for excessive outliers
        avg_outlier_pct = np.mean([
            item['percentage'] for item in quality_report['outliers'].values()
        ]) if quality_report['outliers'] else 0
        if avg_outlier_pct > 10:  # More than 10% outliers is concerning
            quality_score -= ((avg_outlier_pct - 10) / 100) * 0.3

        quality_report['overall_quality_score'] = max(0.0, min(1.0, quality_score))

        return quality_report


# Export commonly used functions at module level
calculate_benford_expected = BenfordUtils.calculate_expected_frequencies
extract_digits = BenfordUtils.extract_digits
calculate_chi_square = BenfordUtils.calculate_chi_square
preprocess_text = ZipfUtils.preprocess_text
normalize_features = AnomalyUtils.normalize_features
calculate_composite_risk = RiskScoringUtils.calculate_composite_risk_score
validate_data_quality = StatisticalValidation.check_data_quality