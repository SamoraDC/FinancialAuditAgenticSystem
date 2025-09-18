"""
Unit tests for Statistical Analysis Service
Tests for Benford's Law, Newcomb-Benford's Law, and Zipf's Law implementations
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from backend.services.statistical_analyzer import (
    BenfordLawAnalyzer,
    ZipfLawAnalyzer,
    StatisticalAnalysisService,
    AnalysisType,
    StatisticalResult
)


class TestBenfordLawAnalyzer:
    """Test cases for Benford's Law analysis"""

    def test_benford_frequencies_calculation(self):
        """Test theoretical Benford's Law frequency calculation"""
        analyzer = BenfordLawAnalyzer()
        frequencies = analyzer.get_benford_frequencies()

        # Test that frequencies sum to 1
        assert abs(np.sum(frequencies) - 1.0) < 1e-10

        # Test that frequency decreases for higher digits
        for i in range(len(frequencies) - 1):
            assert frequencies[i] > frequencies[i + 1]

        # Test specific known values
        expected_freq_1 = np.log10(1 + 1/1)  # ≈ 0.301
        assert abs(frequencies[0] - expected_freq_1) < 1e-10

    def test_benford_analysis_with_compliant_data(self, sample_benford_data):
        """Test Benford's Law analysis with compliant data"""
        analyzer = BenfordLawAnalyzer()

        result = analyzer.analyze_first_digit(sample_benford_data)

        assert isinstance(result, StatisticalResult)
        assert result.analysis_type == AnalysisType.BENFORD
        assert not result.anomaly_detected  # Should not detect anomaly in compliant data
        assert result.p_value > 0.05  # Should not reject null hypothesis
        assert len(result.observed_frequencies) == 9  # Digits 1-9
        assert len(result.expected_frequencies) == 9

    def test_benford_analysis_with_non_compliant_data(self):
        """Test Benford's Law analysis with non-compliant data"""
        analyzer = BenfordLawAnalyzer()

        # Generate uniform distribution (should fail Benford's test)
        uniform_data = [float(x) for x in range(1000, 2000, 10)]  # All start with digit 1

        result = analyzer.analyze_first_digit(uniform_data)

        assert result.analysis_type == AnalysisType.BENFORD
        assert result.anomaly_detected  # Should detect anomaly
        assert result.p_value < 0.05  # Should reject null hypothesis

    def test_benford_analysis_insufficient_data(self):
        """Test Benford's Law analysis with insufficient data"""
        analyzer = BenfordLawAnalyzer()

        # Only 10 data points (minimum is 30)
        small_data = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze_first_digit(small_data)

    def test_benford_analysis_with_zero_amounts(self):
        """Test Benford's Law analysis filtering out zero amounts"""
        analyzer = BenfordLawAnalyzer()

        # Include zeros and negative amounts
        data_with_zeros = [0.0, -100.0, 123.45, 234.56, 345.67, 456.78, 567.89,
                          678.90, 789.01, 890.12] * 10  # Repeat to get enough data

        result = analyzer.analyze_first_digit(data_with_zeros)

        # Should successfully analyze, filtering out invalid amounts
        assert isinstance(result, StatisticalResult)
        assert result.sample_size == 80  # Only positive amounts counted

    def test_newcomb_benford_analysis(self):
        """Test Newcomb-Benford (two-digit) analysis"""
        analyzer = BenfordLawAnalyzer()

        # Generate data with amounts >= 10
        data = [float(f"{i}{j}") for i in range(1, 10) for j in range(0, 10)] * 2

        result = analyzer.analyze_second_digit(data)

        assert result.analysis_type == AnalysisType.NEWCOMB_BENFORD
        assert len(result.observed_frequencies) == 90  # Numbers 10-99
        assert result.sample_size > 0

    def test_newcomb_benford_insufficient_data(self):
        """Test Newcomb-Benford analysis with insufficient data"""
        analyzer = BenfordLawAnalyzer()

        small_data = [10.0, 20.0, 30.0]  # Only 3 data points

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze_second_digit(small_data)


class TestZipfLawAnalyzer:
    """Test cases for Zipf's Law analysis"""

    def test_zipf_analysis_with_vendor_data(self, sample_financial_data):
        """Test Zipf's Law analysis with vendor payment data"""
        analyzer = ZipfLawAnalyzer()

        result = analyzer.analyze_vendor_payments(sample_financial_data)

        assert isinstance(result, StatisticalResult)
        assert result.analysis_type == AnalysisType.ZIPF
        assert result.sample_size > 0
        assert 'vendors' in result.visualization_data
        assert 'ranks' in result.visualization_data

    def test_zipf_analysis_insufficient_vendors(self):
        """Test Zipf's Law analysis with insufficient vendor data"""
        analyzer = ZipfLawAnalyzer()

        # Only 5 vendors (minimum is 20)
        small_data = pd.DataFrame({
            'vendor_name': ['A', 'B', 'C', 'D', 'E'],
            'amount': [100, 200, 300, 400, 500]
        })

        with pytest.raises(ValueError, match="Insufficient vendor data"):
            analyzer.analyze_vendor_payments(small_data)

    def test_zipf_analysis_with_concentrated_vendors(self):
        """Test Zipf's Law analysis with highly concentrated vendor payments"""
        analyzer = ZipfLawAnalyzer()

        # Create concentrated vendor data (one vendor dominates)
        vendors = ['Vendor_A'] * 50 + [f'Vendor_{i}' for i in range(1, 21)]
        amounts = [1000] * 50 + [10] * 20

        data = pd.DataFrame({
            'vendor_name': vendors,
            'amount': amounts
        })

        result = analyzer.analyze_vendor_payments(data)

        assert result.analysis_type == AnalysisType.ZIPF
        # High concentration might trigger anomaly detection
        assert 'vendor' in result.interpretation.lower()


class TestStatisticalAnalysisService:
    """Test cases for comprehensive statistical analysis service"""

    def test_comprehensive_analysis_all_tests(self, sample_financial_data):
        """Test comprehensive analysis running all available tests"""
        service = StatisticalAnalysisService()

        # Add enough data for all tests
        extended_data = pd.concat([sample_financial_data] * 10, ignore_index=True)

        results = service.comprehensive_analysis(extended_data, "test_session")

        assert isinstance(results, dict)
        assert 'benford' in results or 'zipf' in results  # At least one test should run

        for test_name, result in results.items():
            assert isinstance(result, StatisticalResult)
            assert hasattr(result, 'analysis_type')
            assert hasattr(result, 'p_value')

    def test_comprehensive_analysis_empty_data(self):
        """Test comprehensive analysis with empty data"""
        service = StatisticalAnalysisService()

        empty_data = pd.DataFrame()

        results = service.comprehensive_analysis(empty_data, "test_session")

        assert isinstance(results, dict)
        assert len(results) == 0  # No tests should run with empty data

    def test_comprehensive_analysis_insufficient_data(self):
        """Test comprehensive analysis with insufficient data for any test"""
        service = StatisticalAnalysisService()

        # Very small dataset
        small_data = pd.DataFrame({
            'amount': [100.0, 200.0],
            'vendor_name': ['A', 'B']
        })

        results = service.comprehensive_analysis(small_data, "test_session")

        assert isinstance(results, dict)
        assert len(results) == 0  # No tests should pass minimum sample size

    def test_generate_summary_report_no_anomalies(self, sample_financial_data):
        """Test summary report generation with no anomalies"""
        service = StatisticalAnalysisService()

        # Create mock results with no anomalies
        mock_result = StatisticalResult(
            analysis_type=AnalysisType.BENFORD,
            test_statistic=5.0,
            p_value=0.8,  # High p-value = no anomaly
            chi_square_statistic=5.0,
            degrees_of_freedom=8,
            anomaly_detected=False,
            confidence_level=0.05,
            sample_size=100,
            expected_frequencies=[0.1] * 9,
            observed_frequencies=[0.11] * 9,
            deviations=[1.0] * 9,
            visualization_data={},
            interpretation="No anomalies detected",
            recommendations=["Continue standard procedures"]
        )

        results = {'benford': mock_result}

        summary = service.generate_summary_report(results)

        assert summary['total_tests'] == 1
        assert summary['anomalies_detected'] == 0
        assert summary['overall_risk_level'] == 'low'

    def test_generate_summary_report_with_anomalies(self):
        """Test summary report generation with detected anomalies"""
        service = StatisticalAnalysisService()

        # Create mock results with anomalies
        anomaly_result = StatisticalResult(
            analysis_type=AnalysisType.BENFORD,
            test_statistic=25.0,
            p_value=0.001,  # Very low p-value = strong anomaly
            chi_square_statistic=25.0,
            degrees_of_freedom=8,
            anomaly_detected=True,
            confidence_level=0.05,
            sample_size=100,
            expected_frequencies=[0.1] * 9,
            observed_frequencies=[0.2, 0.05] + [0.08] * 7,
            deviations=[100.0, -50.0] + [0.0] * 7,
            visualization_data={},
            interpretation="Significant anomaly detected",
            recommendations=["Investigate further", "Review transactions"]
        )

        results = {'benford': anomaly_result}

        summary = service.generate_summary_report(results)

        assert summary['total_tests'] == 1
        assert summary['anomalies_detected'] == 1
        assert summary['overall_risk_level'] == 'critical'  # Due to very low p-value
        assert len(summary['key_findings']) > 0
        assert len(summary['priority_recommendations']) > 0

    @patch('matplotlib.pyplot.savefig')
    def test_save_visualization_benford(self, mock_savefig):
        """Test visualization saving for Benford's Law results"""
        service = StatisticalAnalysisService()

        benford_result = StatisticalResult(
            analysis_type=AnalysisType.BENFORD,
            test_statistic=5.0,
            p_value=0.8,
            chi_square_statistic=5.0,
            degrees_of_freedom=8,
            anomaly_detected=False,
            confidence_level=0.05,
            sample_size=100,
            expected_frequencies=BenfordLawAnalyzer.get_benford_frequencies().tolist(),
            observed_frequencies=[0.31, 0.18, 0.13, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04],
            deviations=[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            visualization_data={
                'digits': list(range(1, 10)),
                'observed_freq': [0.31, 0.18, 0.13, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04],
                'expected_freq': BenfordLawAnalyzer.get_benford_frequencies().tolist(),
                'deviations': [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'counts': [31, 18, 13, 10, 8, 7, 6, 5, 4]
            },
            interpretation="Test visualization",
            recommendations=[]
        )

        service.save_visualization(benford_result, "/tmp/test_viz.png")

        # Verify matplotlib savefig was called
        mock_savefig.assert_called_once()

    def test_benford_analyzer_interpretation_generation(self):
        """Test interpretation generation for different scenarios"""
        analyzer = BenfordLawAnalyzer()

        # Test normal result interpretation
        normal_interp = analyzer._interpret_benford_results(
            p_value=0.8, chi2_stat=5.0, deviations=np.array([1, -1, 0, 0, 0, 0, 0, 0, 0]),
            confidence_level=0.05
        )
        assert "follows Benford's Law" in normal_interp
        assert "No significant anomalies" in normal_interp

        # Test anomalous result interpretation
        anomaly_interp = analyzer._interpret_benford_results(
            p_value=0.001, chi2_stat=25.0, deviations=np.array([50, -30, 0, 0, 0, 0, 0, 0, 0]),
            confidence_level=0.05
        )
        assert "deviates significantly" in anomaly_interp
        assert "Potential anomalies detected" in anomaly_interp
        assert "digit 1" in anomaly_interp  # Should mention significant deviation

    def test_benford_analyzer_recommendations_generation(self):
        """Test recommendation generation based on analysis results"""
        analyzer = BenfordLawAnalyzer()

        # Test normal recommendations
        normal_recs = analyzer._generate_benford_recommendations(
            anomaly_detected=False,
            deviations=np.array([1, -1, 0, 0, 0, 0, 0, 0, 0]),
            p_value=0.8
        )
        assert "appears natural" in normal_recs[0]

        # Test anomaly recommendations
        anomaly_recs = analyzer._generate_benford_recommendations(
            anomaly_detected=True,
            deviations=np.array([50, -35, 0, 0, 0, 0, 0, 0, 0]),
            p_value=0.001
        )
        assert "Significant deviation" in anomaly_recs[0]
        assert "overrepresented" in anomaly_recs[1]  # For digit 1
        assert "underrepresented" in anomaly_recs[2]  # For digit 2
        assert "potential fraud" in anomaly_recs[-1].lower()  # For very low p-value


class TestStatisticalResultDataClass:
    """Test the StatisticalResult data class"""

    def test_statistical_result_creation(self):
        """Test creating a StatisticalResult instance"""
        result = StatisticalResult(
            analysis_type=AnalysisType.BENFORD,
            test_statistic=10.5,
            p_value=0.02,
            chi_square_statistic=10.5,
            degrees_of_freedom=8,
            anomaly_detected=True,
            confidence_level=0.05,
            sample_size=200,
            expected_frequencies=[0.1] * 9,
            observed_frequencies=[0.12] * 9,
            deviations=[2.0] * 9,
            visualization_data={'test': 'data'},
            interpretation="Test interpretation",
            recommendations=["Test recommendation"]
        )

        assert result.analysis_type == AnalysisType.BENFORD
        assert result.test_statistic == 10.5
        assert result.p_value == 0.02
        assert result.anomaly_detected is True
        assert len(result.expected_frequencies) == 9
        assert len(result.observed_frequencies) == 9


# Performance tests
class TestStatisticalAnalysisPerformance:
    """Performance tests for statistical analysis"""

    def test_benford_analysis_large_dataset_performance(self, performance_test_data):
        """Test Benford analysis performance with large dataset"""
        analyzer = BenfordLawAnalyzer()

        amounts = performance_test_data['amount'].tolist()

        import time
        start_time = time.time()
        result = analyzer.analyze_first_digit(amounts)
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 2.0  # 2 seconds max for 10k records
        assert result.sample_size == len(amounts)

    def test_zipf_analysis_large_dataset_performance(self, performance_test_data):
        """Test Zipf analysis performance with large dataset"""
        analyzer = ZipfLawAnalyzer()

        import time
        start_time = time.time()
        result = analyzer.analyze_vendor_payments(performance_test_data)
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 3.0  # 3 seconds max for 10k records
        assert result.sample_size > 0