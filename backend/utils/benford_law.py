"""
Benford's Law analysis utility
Implements Benford's Law for detecting financial anomalies
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class BenfordAnalyzer:
    """Analyzer for Benford's Law compliance testing"""

    def __init__(self):
        # Expected Benford's Law frequencies for first digits 1-9
        self.expected_frequencies = np.array([
            math.log10(1 + 1/d) for d in range(1, 10)
        ])
        
        # Expected frequencies for second digits 0-9 given first digit
        self.second_digit_expected = self._calculate_second_digit_frequencies()

    def _calculate_second_digit_frequencies(self) -> Dict[int, np.ndarray]:
        """Calculate expected second digit frequencies for each first digit"""
        second_digit_freq = {}
        
        for first_digit in range(1, 10):
            freq = np.zeros(10)
            for second_digit in range(0, 10):
                # Expected frequency for second digit given first digit
                freq[second_digit] = sum(
                    math.log10(1 + 1/(first_digit * 10 + second_digit + k * 100))
                    for k in range(0, 10)
                ) / first_digit
            second_digit_freq[first_digit] = freq
            
        return second_digit_freq

    def analyze(self, data: List[float], analysis_type: str = "first_digit") -> Dict[str, Any]:
        """Perform Benford's Law analysis on dataset"""
        try:
            if not data or len(data) < 10:
                return {
                    'error': 'Insufficient data for Benford analysis (minimum 10 values required)',
                    'data_size': len(data) if data else 0
                }

            # Filter valid positive numbers
            positive_data = [abs(x) for x in data if x != 0 and not math.isnan(x)]
            
            if len(positive_data) < 10:
                return {
                    'error': 'Insufficient positive values for analysis',
                    'valid_values': len(positive_data),
                    'total_values': len(data)
                }

            if analysis_type == "first_digit":
                return self._analyze_first_digits(positive_data)
            elif analysis_type == "second_digit":
                return self._analyze_second_digits(positive_data)
            elif analysis_type == "comprehensive":
                return self._comprehensive_analysis(positive_data)
            else:
                return {'error': f'Unknown analysis type: {analysis_type}'}

        except Exception as e:
            logger.error(f"Benford analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_first_digits(self, data: List[float]) -> Dict[str, Any]:
        """Analyze first digit distribution"""
        try:
            # Extract first digits
            first_digits = self._extract_first_digits(data)
            
            if not first_digits:
                return {'error': 'No valid first digits found'}

            # Count occurrences
            digit_counts = Counter(first_digits)
            total_count = len(first_digits)

            # Calculate observed frequencies
            observed_freq = np.zeros(9)
            for digit in range(1, 10):
                observed_freq[digit-1] = digit_counts.get(digit, 0) / total_count

            # Perform statistical tests
            chi_squared_result = self._chi_squared_test(observed_freq, self.expected_frequencies, total_count)
            ks_result = self._kolmogorov_smirnov_test(observed_freq, self.expected_frequencies)
            
            # Calculate additional metrics
            mad = self._mean_absolute_deviation(observed_freq, self.expected_frequencies)
            deviation_score = np.sum(np.abs(observed_freq - self.expected_frequencies))

            return {
                'analysis_type': 'first_digit',
                'total_values': total_count,
                'observed_frequencies': observed_freq.tolist(),
                'expected_frequencies': self.expected_frequencies.tolist(),
                'digit_counts': dict(digit_counts),
                'chi_squared': chi_squared_result,
                'kolmogorov_smirnov': ks_result,
                'mean_absolute_deviation': mad,
                'deviation_score': deviation_score,
                'conformity_assessment': self._assess_conformity(chi_squared_result['p_value']),
                'anomaly_indicators': self._identify_anomalous_digits(observed_freq, self.expected_frequencies)
            }

        except Exception as e:
            logger.error(f"First digit analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_second_digits(self, data: List[float]) -> Dict[str, Any]:
        """Analyze second digit distribution"""
        try:
            # Extract first and second digits
            digit_pairs = self._extract_digit_pairs(data)
            
            if not digit_pairs:
                return {'error': 'No valid digit pairs found'}

            results = {}
            
            for first_digit in range(1, 10):
                # Get second digits for this first digit
                second_digits = [pair[1] for pair in digit_pairs if pair[0] == first_digit]
                
                if len(second_digits) < 5:  # Need minimum samples
                    continue

                # Count second digit occurrences
                second_counts = Counter(second_digits)
                total_second = len(second_digits)

                # Calculate observed frequencies for second digits (0-9)
                observed_second = np.zeros(10)
                for digit in range(0, 10):
                    observed_second[digit] = second_counts.get(digit, 0) / total_second

                # Expected frequencies for this first digit
                expected_second = self.second_digit_expected.get(first_digit, np.ones(10) / 10)

                # Statistical test
                chi_squared = self._chi_squared_test(observed_second, expected_second, total_second)

                results[f'first_digit_{first_digit}'] = {
                    'sample_size': total_second,
                    'observed_frequencies': observed_second.tolist(),
                    'expected_frequencies': expected_second.tolist(),
                    'chi_squared': chi_squared,
                    'conformity': self._assess_conformity(chi_squared['p_value'])
                }

            return {
                'analysis_type': 'second_digit',
                'total_pairs': len(digit_pairs),
                'first_digit_analysis': results,
                'overall_conformity': self._calculate_overall_second_digit_conformity(results)
            }

        except Exception as e:
            logger.error(f"Second digit analysis failed: {e}")
            return {'error': str(e)}

    def _comprehensive_analysis(self, data: List[float]) -> Dict[str, Any]:
        """Perform comprehensive Benford analysis"""
        try:
            first_digit_analysis = self._analyze_first_digits(data)
            second_digit_analysis = self._analyze_second_digits(data)

            # Calculate overall compliance score
            first_digit_score = 1 - first_digit_analysis.get('deviation_score', 1)
            
            # Average second digit conformity
            second_digit_results = second_digit_analysis.get('first_digit_analysis', {})
            second_digit_scores = []
            for analysis in second_digit_results.values():
                if 'chi_squared' in analysis and 'p_value' in analysis['chi_squared']:
                    second_digit_scores.append(analysis['chi_squared']['p_value'])
            
            avg_second_digit_score = np.mean(second_digit_scores) if second_digit_scores else 0

            overall_score = (first_digit_score * 0.7 + avg_second_digit_score * 0.3)

            return {
                'analysis_type': 'comprehensive',
                'first_digit_analysis': first_digit_analysis,
                'second_digit_analysis': second_digit_analysis,
                'overall_compliance_score': overall_score,
                'overall_assessment': self._assess_overall_compliance(overall_score),
                'recommendations': self._generate_recommendations(first_digit_analysis, second_digit_analysis)
            }

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {'error': str(e)}

    def _extract_first_digits(self, data: List[float]) -> List[int]:
        """Extract first significant digits from data"""
        first_digits = []
        
        for value in data:
            if value <= 0:
                continue
                
            # Convert to string and find first non-zero digit
            str_value = f"{value:.10f}"
            for char in str_value:
                if char.isdigit() and char != '0':
                    first_digits.append(int(char))
                    break
                    
        return first_digits

    def _extract_digit_pairs(self, data: List[float]) -> List[Tuple[int, int]]:
        """Extract first and second digit pairs"""
        pairs = []
        
        for value in data:
            if value <= 0:
                continue
                
            # Find first and second significant digits
            str_value = f"{value:.10f}"
            digits = []
            
            for char in str_value:
                if char.isdigit():
                    digits.append(int(char))
                    if len(digits) == 2:
                        break
            
            if len(digits) >= 2 and digits[0] != 0:
                pairs.append((digits[0], digits[1]))
                
        return pairs

    def _chi_squared_test(self, observed: np.ndarray, expected: np.ndarray, total_count: int) -> Dict[str, Any]:
        """Perform chi-squared goodness of fit test"""
        try:
            expected_counts = expected * total_count
            observed_counts = observed * total_count
            
            # Avoid division by zero
            expected_counts = np.maximum(expected_counts, 0.5)
            
            chi_squared = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
            degrees_freedom = len(observed) - 1
            p_value = 1 - stats.chi2.cdf(chi_squared, degrees_freedom)

            return {
                'statistic': float(chi_squared),
                'p_value': float(p_value),
                'degrees_freedom': degrees_freedom,
                'critical_value_0_05': float(stats.chi2.ppf(0.95, degrees_freedom)),
                'significant': p_value < 0.05
            }

        except Exception as e:
            logger.error(f"Chi-squared test failed: {e}")
            return {'error': str(e)}

    def _kolmogorov_smirnov_test(self, observed: np.ndarray, expected: np.ndarray) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test"""
        try:
            # Calculate cumulative distributions
            observed_cdf = np.cumsum(observed)
            expected_cdf = np.cumsum(expected)
            
            # KS statistic is maximum difference between CDFs
            ks_statistic = np.max(np.abs(observed_cdf - expected_cdf))
            
            # Approximate p-value (for large samples)
            n = len(observed)
            p_value = 2 * math.exp(-2 * n * ks_statistic ** 2) if n > 0 else 1.0

            return {
                'statistic': float(ks_statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

        except Exception as e:
            logger.error(f"KS test failed: {e}")
            return {'error': str(e)}

    def _mean_absolute_deviation(self, observed: np.ndarray, expected: np.ndarray) -> float:
        """Calculate mean absolute deviation"""
        return float(np.mean(np.abs(observed - expected)))

    def _assess_conformity(self, p_value: float) -> str:
        """Assess conformity level based on p-value"""
        if p_value > 0.1:
            return 'good'
        elif p_value > 0.05:
            return 'acceptable'
        elif p_value > 0.01:
            return 'questionable'
        else:
            return 'poor'

    def _identify_anomalous_digits(self, observed: np.ndarray, expected: np.ndarray) -> List[Dict[str, Any]]:
        """Identify digits that deviate significantly from expected frequencies"""
        anomalous = []
        
        for i, (obs, exp) in enumerate(zip(observed, expected)):
            deviation = abs(obs - exp)
            if deviation > 0.02:  # More than 2% deviation
                anomalous.append({
                    'digit': i + 1,
                    'observed_frequency': float(obs),
                    'expected_frequency': float(exp),
                    'deviation': float(deviation),
                    'deviation_percentage': float(deviation / exp * 100) if exp > 0 else 0
                })
        
        return sorted(anomalous, key=lambda x: x['deviation'], reverse=True)

    def _calculate_overall_second_digit_conformity(self, results: Dict[str, Any]) -> str:
        """Calculate overall conformity for second digit analysis"""
        conformity_scores = []
        
        for analysis in results.values():
            conformity = analysis.get('conformity', 'poor')
            score = {'good': 3, 'acceptable': 2, 'questionable': 1, 'poor': 0}.get(conformity, 0)
            conformity_scores.append(score)
        
        if not conformity_scores:
            return 'insufficient_data'
            
        avg_score = np.mean(conformity_scores)
        
        if avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'acceptable'
        elif avg_score >= 0.5:
            return 'questionable'
        else:
            return 'poor'

    def _assess_overall_compliance(self, score: float) -> str:
        """Assess overall Benford compliance"""
        if score >= 0.8:
            return 'high_compliance'
        elif score >= 0.6:
            return 'moderate_compliance'
        elif score >= 0.4:
            return 'low_compliance'
        else:
            return 'poor_compliance'

    def _generate_recommendations(self, first_analysis: Dict[str, Any], 
                                second_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # First digit recommendations
        if 'chi_squared' in first_analysis and first_analysis['chi_squared'].get('significant', False):
            recommendations.append("First digit distribution shows significant deviation from Benford's Law")
            
            anomalous_digits = first_analysis.get('anomaly_indicators', [])
            if anomalous_digits:
                top_anomaly = anomalous_digits[0]
                recommendations.append(f"Digit {top_anomaly['digit']} shows highest deviation ({top_anomaly['deviation_percentage']:.1f}%)")

        # Second digit recommendations
        second_conformity = second_analysis.get('overall_conformity', 'poor')
        if second_conformity in ['questionable', 'poor']:
            recommendations.append("Second digit distribution suggests potential data manipulation")

        # General recommendations
        if not recommendations:
            recommendations.append("Data appears to conform well to Benford's Law")
        else:
            recommendations.append("Consider investigating transactions contributing to digit anomalies")
            recommendations.append("Review data collection and entry processes for potential issues")

        return recommendations
