"""
Statistical analysis service for financial audit
Implements statistical methods for anomaly detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import math
from collections import Counter

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Service for performing statistical analysis on financial data"""

    def __init__(self):
        self.analysis_methods = [
            'benford_law',
            'zipf_law', 
            'descriptive_stats',
            'outlier_detection',
            'trend_analysis'
        ]

    async def analyze_dataset(self, data: List[float], analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on numerical dataset"""
        try:
            if not data or len(data) < 3:
                return {'error': 'Insufficient data for analysis'}

            data_array = np.array(data)
            
            results = {
                'dataset_size': len(data),
                'analysis_type': analysis_type,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            if analysis_type in ['comprehensive', 'benford']:
                results['benford_analysis'] = await self._benford_law_analysis(data)
            
            if analysis_type in ['comprehensive', 'descriptive']:
                results['descriptive_stats'] = await self._descriptive_statistics(data_array)
            
            if analysis_type in ['comprehensive', 'outliers']:
                results['outlier_analysis'] = await self._outlier_detection(data_array)
            
            if analysis_type in ['comprehensive', 'distribution']:
                results['distribution_analysis'] = await self._distribution_analysis(data_array)

            return results

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}

    async def _benford_law_analysis(self, data: List[float]) -> Dict[str, Any]:
        """Apply Benford's Law analysis to detect anomalies"""
        try:
            # Filter positive numbers and get first digits
            positive_data = [abs(x) for x in data if x != 0]
            
            if len(positive_data) < 10:
                return {'error': 'Insufficient data for Benford analysis (minimum 10 values)'}

            # Extract first digits
            first_digits = []
            for value in positive_data:
                str_value = f"{value:.10f}"  # Convert to string with precision
                for char in str_value:
                    if char.isdigit() and char != '0':
                        first_digits.append(int(char))
                        break

            if not first_digits:
                return {'error': 'No valid first digits found'}

            # Count occurrences of each digit (1-9)
            digit_counts = Counter(first_digits)
            total_count = len(first_digits)

            # Calculate observed frequencies
            observed_freq = np.zeros(9)
            for digit in range(1, 10):
                observed_freq[digit-1] = digit_counts.get(digit, 0) / total_count

            # Benford's Law expected frequencies
            expected_freq = np.array([math.log10(1 + 1/d) for d in range(1, 10)])

            # Chi-squared test
            expected_counts = expected_freq * total_count
            # Add small constant to avoid division by zero
            expected_counts = np.maximum(expected_counts, 0.5)
            
            chi_squared = np.sum((observed_freq * total_count - expected_counts) ** 2 / expected_counts)
            degrees_freedom = 8  # 9 digits - 1
            p_value = 1 - stats.chi2.cdf(chi_squared, degrees_freedom)

            # Calculate deviation score
            deviation_score = np.sum(np.abs(observed_freq - expected_freq))

            return {
                'total_values': total_count,
                'observed_frequencies': observed_freq.tolist(),
                'expected_frequencies': expected_freq.tolist(),
                'chi_squared_statistic': float(chi_squared),
                'p_value': float(p_value),
                'degrees_freedom': degrees_freedom,
                'deviation_score': float(deviation_score),
                'significant_deviation': p_value < 0.05,
                'conformity_level': 'good' if p_value > 0.05 else 'questionable' if p_value > 0.01 else 'poor'
            }

        except Exception as e:
            logger.error(f"Benford's Law analysis failed: {e}")
            return {'error': str(e)}

    async def _descriptive_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics"""
        try:
            return {
                'count': int(len(data)),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std_dev': float(np.std(data)),
                'variance': float(np.var(data)),
                'min_value': float(np.min(data)),
                'max_value': float(np.max(data)),
                'range': float(np.max(data) - np.min(data)),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'q1': float(np.percentile(data, 25)),
                'q3': float(np.percentile(data, 75)),
                'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                'coefficient_variation': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0
            }

        except Exception as e:
            logger.error(f"Descriptive statistics calculation failed: {e}")
            return {'error': str(e)}

    async def _outlier_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        try:
            results = {}

            # IQR method
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            results['iqr_method'] = {
                'outlier_count': len(iqr_outliers),
                'outlier_percentage': len(iqr_outliers) / len(data) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outliers': iqr_outliers.tolist()[:10]  # Limit to first 10
            }

            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            results['zscore_method'] = {
                'outlier_count': len(z_outliers),
                'outlier_percentage': len(z_outliers) / len(data) * 100,
                'threshold': 3.0,
                'outliers': z_outliers.tolist()[:10]
            }

            # Modified Z-score method (more robust)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
            modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
            
            results['modified_zscore_method'] = {
                'outlier_count': len(modified_z_outliers),
                'outlier_percentage': len(modified_z_outliers) / len(data) * 100,
                'threshold': 3.5,
                'outliers': modified_z_outliers.tolist()[:10]
            }

            return results

        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return {'error': str(e)}

    async def _distribution_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze data distribution and test for normality"""
        try:
            results = {}

            # Normality tests
            if len(data) >= 8:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                results['shapiro_wilk_test'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }

            if len(data) >= 20:
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
                results['kolmogorov_smirnov_test'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > 0.05
                }

            # Distribution characteristics
            results['distribution_characteristics'] = {
                'is_symmetric': abs(stats.skew(data)) < 0.5,
                'skewness_interpretation': self._interpret_skewness(stats.skew(data)),
                'kurtosis_interpretation': self._interpret_kurtosis(stats.kurtosis(data)),
                'has_heavy_tails': stats.kurtosis(data) > 3
            }

            return results

        except Exception as e:
            logger.error(f"Distribution analysis failed: {e}")
            return {'error': str(e)}

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value"""
        if abs(skewness) < 0.5:
            return "approximately symmetric"
        elif skewness > 0.5:
            return "positively skewed (right tail)"
        else:
            return "negatively skewed (left tail)"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value"""
        if abs(kurtosis) < 0.5:
            return "mesokurtic (normal)"
        elif kurtosis > 0.5:
            return "leptokurtic (heavy tails)"
        else:
            return "platykurtic (light tails)"

    async def calculate_risk_indicators(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate financial risk indicators"""
        try:
            if len(data) < 2:
                return {'error': 'Insufficient data for risk calculation'}

            # Calculate various risk metrics
            returns = np.diff(data) / data[:-1]  # Simple returns
            
            results = {
                'volatility': float(np.std(returns)),
                'downside_deviation': float(np.std(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0.0,
                'value_at_risk_95': float(np.percentile(returns, 5)),
                'value_at_risk_99': float(np.percentile(returns, 1)),
                'maximum_drawdown': self._calculate_max_drawdown(data),
                'sharpe_ratio': float(np.mean(returns) / np.std(returns)) if np.std(returns) != 0 else 0.0,
                'coefficient_of_variation': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else float('inf')
            }

            return results

        except Exception as e:
            logger.error(f"Risk indicator calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_max_drawdown(self, data: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumprod(1 + np.diff(data) / data[:-1])
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except:
            return 0.0
