"""
Comprehensive tests for the Statistical Analysis Engine.

Tests all statistical methods including Benford's Law, Zipf's Law,
anomaly detection algorithms, and reinforcement learning components.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.services.statistical_analysis import (
    StatisticalAnalysisEngine,
    BenfordAnalysisResult,
    ZipfAnalysisResult,
    AnomalyDetectionResult,
    RLAnomalyDetectionResult,
    StatisticalMethod,
    AnomalyType
)
from backend.models.audit_models import RiskLevel


class TestStatisticalAnalysisEngine:
    """Test suite for Statistical Analysis Engine"""

    @pytest.fixture
    def engine(self):
        """Create statistical analysis engine instance"""
        return StatisticalAnalysisEngine(
            confidence_level=0.95,
            default_risk_threshold=0.7,
            enable_rl_training=False  # Disable for faster tests
        )

    @pytest.fixture
    def sample_financial_data(self):
        """Generate sample financial data for testing"""
        np.random.seed(42)

        # Generate data that roughly follows Benford's Law
        benford_data = []
        for i in range(1000):
            # Create numbers that naturally follow Benford's distribution
            magnitude = np.random.uniform(1, 6)  # 1 to 6 orders of magnitude
            mantissa = np.random.uniform(1, 10)
            value = mantissa * (10 ** magnitude)
            benford_data.append(value)

        return benford_data

    @pytest.fixture
    def sample_text_data(self):
        """Generate sample text data for Zipf analysis"""
        return [
            "The quick brown fox jumps over the lazy dog",
            "Financial statements must be prepared according to accounting standards",
            "Revenue recognition principles require careful consideration of timing",
            "Internal controls are essential for accurate financial reporting",
            "Audit procedures include testing and verification of accounts"
        ] * 20  # Repeat to get better statistical distribution

    @pytest.fixture
    def sample_dataframe(self):
        """Generate sample DataFrame for anomaly detection"""
        np.random.seed(42)

        data = {
            'amount': np.random.lognormal(10, 1, 1000),
            'transaction_count': np.random.poisson(5, 1000),
            'vendor_score': np.random.beta(2, 5, 1000),
            'days_to_payment': np.random.exponential(30, 1000),
            'approval_level': np.random.choice([1, 2, 3], 1000, p=[0.7, 0.25, 0.05])
        }

        # Add some deliberate anomalies
        data['amount'][0] = 1000000  # Outlier
        data['amount'][1] = -500     # Negative value (unusual)
        data['days_to_payment'][2] = 365  # Very long payment period

        return pd.DataFrame(data)

    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.confidence_level == 0.95
        assert engine.default_risk_threshold == 0.7
        assert not engine.enable_rl_training
        assert engine.alpha == 0.05
        assert len(engine.benford_first_digit) == 9
        assert len(engine.benford_second_digit) == 10

    def test_benfords_law_analysis_first_digit(self, engine, sample_financial_data):
        """Test Benford's Law analysis for first digits"""
        result = engine.analyze_benfords_law(sample_financial_data, digit_position=1)

        assert isinstance(result, BenfordAnalysisResult)
        assert result.digit_position == 1
        assert result.chi_square_statistic >= 0
        assert 0 <= result.p_value <= 1
        assert 0 <= result.risk_score <= 1
        assert len(result.expected_frequencies) == 9
        assert len(result.observed_frequencies) >= 1
        assert result.sample_size == len([x for x in sample_financial_data if x != 0])
        assert isinstance(result.recommendations, list)

    def test_benfords_law_analysis_second_digit(self, engine, sample_financial_data):
        """Test Benford's Law analysis for second digits"""
        result = engine.analyze_benfords_law(sample_financial_data, digit_position=2)

        assert isinstance(result, BenfordAnalysisResult)
        assert result.digit_position == 2
        assert result.chi_square_statistic >= 0
        assert len(result.expected_frequencies) == 10

    def test_benfords_law_invalid_digit_position(self, engine, sample_financial_data):
        """Test Benford's Law with invalid digit position"""
        with pytest.raises(ValueError, match="Only first and second digit analysis supported"):
            engine.analyze_benfords_law(sample_financial_data, digit_position=3)

    def test_newcomb_benford_analysis(self, engine, sample_financial_data):
        """Test Newcomb-Benford's Law analysis"""
        result = engine.analyze_newcomb_benford_law(sample_financial_data)

        assert isinstance(result, BenfordAnalysisResult)
        assert result.digit_position == 1
        assert len(result.recommendations) >= 3  # Should have enhanced recommendations

    def test_zipfs_law_analysis(self, engine, sample_text_data):
        """Test Zipf's Law analysis"""
        result = engine.analyze_zipfs_law(sample_text_data, min_word_length=3)

        assert isinstance(result, ZipfAnalysisResult)
        assert -1 <= result.correlation_coefficient <= 1
        assert result.kolmogorov_smirnov_statistic >= 0
        assert 0 <= result.p_value <= 1
        assert 0 <= result.risk_score <= 1
        assert result.text_sample_size > 0
        assert len(result.word_rank_frequencies) > 0
        assert isinstance(result.recommendations, list)

    def test_isolation_forest_anomaly_detection(self, engine, sample_dataframe):
        """Test Isolation Forest anomaly detection"""
        result = engine.detect_anomalies_isolation_forest(
            sample_dataframe, contamination=0.1
        )

        assert isinstance(result, AnomalyDetectionResult)
        assert result.method == StatisticalMethod.ISOLATION_FOREST
        assert result.anomaly_type == AnomalyType.STATISTICAL_OUTLIER
        assert result.total_samples == len(sample_dataframe)
        assert result.anomalies_detected >= 0
        assert 0 <= result.anomaly_rate <= 1
        assert 0 <= result.overall_risk_score <= 1
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.anomalous_records, list)
        assert isinstance(result.feature_importance, dict)

    def test_pyod_ensemble_anomaly_detection(self, engine, sample_dataframe):
        """Test PyOD ensemble anomaly detection"""
        result = engine.detect_anomalies_pyod_ensemble(
            sample_dataframe, methods=['iforest', 'lof']
        )

        assert isinstance(result, AnomalyDetectionResult)
        assert result.method == StatisticalMethod.PYOD_ENSEMBLE
        assert result.total_samples == len(sample_dataframe)
        assert isinstance(result.anomalous_records, list)
        assert isinstance(result.model_metrics, dict)

    @patch('backend.services.statistical_analysis.DQN')
    @patch('backend.services.statistical_analysis.PPO')
    def test_rl_anomaly_detection_dqn(self, mock_ppo, mock_dqn, engine, sample_dataframe):
        """Test RL anomaly detection with DQN"""
        # Mock the RL model
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)  # FLAG action
        mock_dqn.return_value = mock_model

        # Mock environment
        with patch.object(engine, '_create_anomaly_detection_env') as mock_env_creator:
            mock_env = MagicMock()
            mock_env.reset.return_value = np.zeros(5)
            mock_env.step.return_value = (np.zeros(5), 0.5, False, {})
            mock_env_creator.return_value = mock_env

            result = engine.detect_anomalies_reinforcement_learning(
                sample_dataframe, model_type="DQN", training_episodes=10
            )

        assert isinstance(result, RLAnomalyDetectionResult)
        assert result.model_type == "DQN"
        assert result.episodes_trained == 10
        assert isinstance(result.flagged_transactions, list)
        assert isinstance(result.action_probabilities, dict)

    def test_rl_anomaly_detection_invalid_model(self, engine, sample_dataframe):
        """Test RL anomaly detection with invalid model type"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            engine.detect_anomalies_reinforcement_learning(
                sample_dataframe, model_type="INVALID"
            )

    def test_comprehensive_risk_score_calculation(self, engine):
        """Test comprehensive risk score calculation"""
        # Create mock results
        benford_result = BenfordAnalysisResult(
            digit_position=1,
            chi_square_statistic=20.0,
            p_value=0.01,
            critical_value=15.51,
            is_significant=True,
            risk_score=0.8,
            expected_frequencies={'1': 0.301, '2': 0.176},
            observed_frequencies={'1': 0.250, '2': 0.200},
            deviations={'1': -0.051, '2': 0.024},
            anomalous_digits=['1'],
            sample_size=1000,
            recommendations=["Review transactions"]
        )

        zipf_result = ZipfAnalysisResult(
            correlation_coefficient=0.85,
            kolmogorov_smirnov_statistic=0.1,
            p_value=0.05,
            risk_score=0.3,
            word_rank_frequencies={'the': {'rank': 1, 'frequency': 0.1}},
            expected_vs_observed={'the': (0.1, 0.12)},
            anomalous_words=[],
            text_sample_size=1000,
            recommendations=["Continue monitoring"]
        )

        anomaly_result = AnomalyDetectionResult(
            method=StatisticalMethod.ISOLATION_FOREST,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            total_samples=1000,
            anomalies_detected=50,
            anomaly_rate=0.05,
            overall_risk_score=0.6,
            confidence_score=0.85,
            anomalous_records=[],
            feature_importance={'amount': 0.5, 'vendor': 0.3},
            model_metrics={'contamination': 0.1},
            recommendations=["Investigate outliers"]
        )

        results = [benford_result, zipf_result, anomaly_result]
        comprehensive = engine.calculate_comprehensive_risk_score(results)

        assert isinstance(comprehensive, dict)
        assert 'overall_risk_score' in comprehensive
        assert 'risk_level' in comprehensive
        assert 'confidence_score' in comprehensive
        assert 'method_scores' in comprehensive
        assert 'recommendations' in comprehensive
        assert 'risk_factors' in comprehensive

        assert 0 <= comprehensive['overall_risk_score'] <= 1
        assert isinstance(comprehensive['risk_level'], RiskLevel)
        assert isinstance(comprehensive['recommendations'], list)

    def test_generate_audit_findings(self, engine):
        """Test generation of audit findings"""
        comprehensive_results = {
            'overall_risk_score': 0.75,
            'risk_level': RiskLevel.HIGH,
            'confidence_score': 0.85,
            'method_scores': {
                'benford': 0.8,
                'isolation_forest': 0.7,
                'zipf': 0.3
            },
            'recommendations': [
                'Investigate high-risk transactions',
                'Review digit patterns',
                'Implement additional controls'
            ],
            'risk_factors': [
                'Unusual digit distribution patterns',
                'Statistical outliers detected'
            ]
        }

        findings = engine.generate_audit_findings(comprehensive_results, 'TEST_SESSION_123')

        assert isinstance(findings, list)
        assert len(findings) >= 1  # At least main finding

        main_finding = findings[0]
        assert main_finding.audit_session_id == 'TEST_SESSION_123'
        assert main_finding.category == 'Statistical Analysis'
        assert main_finding.severity == RiskLevel.HIGH
        assert main_finding.confidence_score == 0.85

    def test_edge_cases_empty_data(self, engine):
        """Test engine behavior with empty data"""
        # Empty financial data
        with pytest.raises(IndexError):
            engine.analyze_benfords_law([])

        # Empty text data
        result = engine.analyze_zipfs_law([])
        assert result.correlation_coefficient == 0.0

        # Empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):  # Various exceptions possible
            engine.detect_anomalies_isolation_forest(empty_df)

    def test_edge_cases_single_value(self, engine):
        """Test engine behavior with single values"""
        # Single financial value
        result = engine.analyze_benfords_law([100.0])
        assert result.sample_size == 1

        # Single text
        result = engine.analyze_zipfs_law(["hello world hello"])
        assert result.text_sample_size > 0

    def test_decimal_support(self, engine):
        """Test support for Decimal types"""
        decimal_data = [Decimal('123.45'), Decimal('678.90'), Decimal('111.11')]
        result = engine.analyze_benfords_law(decimal_data)

        assert isinstance(result, BenfordAnalysisResult)
        assert result.sample_size == 3

    def test_negative_values_handling(self, engine):
        """Test handling of negative values"""
        mixed_data = [100, -50, 200, -75, 300]
        result = engine.analyze_benfords_law(mixed_data)

        # Should take absolute values
        assert result.sample_size == 5

    def test_zero_values_filtering(self, engine):
        """Test filtering of zero values"""
        data_with_zeros = [0, 100, 0, 200, 0, 300]
        result = engine.analyze_benfords_law(data_with_zeros)

        # Should filter out zeros
        assert result.sample_size == 3

    def test_performance_large_dataset(self, engine):
        """Test performance with large datasets"""
        # Generate large dataset
        large_data = np.random.lognormal(10, 2, 10000).tolist()

        import time
        start_time = time.time()
        result = engine.analyze_benfords_law(large_data)
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        assert result.sample_size == 10000

    def test_statistical_significance_thresholds(self, engine):
        """Test statistical significance thresholds"""
        # Create data that should NOT be significant (follows Benford's closely)
        benford_compliant_data = []
        for digit in range(1, 10):
            expected_freq = engine.benford_first_digit[str(digit)]
            count = int(expected_freq * 1000)
            benford_compliant_data.extend([digit * 100] * count)

        result = engine.analyze_benfords_law(benford_compliant_data)

        # Should have low risk score for compliant data
        assert result.risk_score < 0.3
        assert result.chi_square_statistic < result.critical_value

    def test_text_preprocessing_zipf(self, engine):
        """Test text preprocessing for Zipf analysis"""
        messy_text = [
            "THE Quick! Brown? Fox... JUMPS over the-lazy-dog.",
            "123 Numbers should be filtered out!!! @#$%"
        ]

        result = engine.analyze_zipfs_law(messy_text, min_word_length=3)

        # Should handle preprocessing correctly
        assert result.text_sample_size > 0
        assert 'the' in result.word_rank_frequencies or 'THE' not in result.word_rank_frequencies

    def test_feature_importance_extraction(self, engine, sample_dataframe):
        """Test feature importance extraction from models"""
        result = engine.detect_anomalies_isolation_forest(sample_dataframe)

        # Should have feature importance for all numeric columns
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns
        assert len(result.feature_importance) == len(numeric_cols)

        # Should sum to approximately 1.0 (normalized)
        total_importance = sum(result.feature_importance.values())
        assert abs(total_importance - 1.0) < 0.01

    def test_risk_level_assignment(self, engine):
        """Test correct risk level assignment"""
        # Test different risk score ranges
        test_cases = [
            (0.1, RiskLevel.LOW),
            (0.4, RiskLevel.MEDIUM),
            (0.7, RiskLevel.HIGH),
            (0.9, RiskLevel.CRITICAL)
        ]

        for risk_score, expected_level in test_cases:
            # Create mock comprehensive results
            results = {
                'overall_risk_score': risk_score,
                'confidence_score': 0.8,
                'method_scores': {'test': risk_score},
                'recommendations': ['Test recommendation'],
                'risk_factors': ['Test factor']
            }

            findings = engine.generate_audit_findings(results, 'TEST')
            main_finding = findings[0]

            if risk_score >= 0.8:
                assert main_finding.severity == RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                assert main_finding.severity == RiskLevel.HIGH
            elif risk_score >= 0.3:
                assert main_finding.severity == RiskLevel.MEDIUM
            else:
                assert main_finding.severity == RiskLevel.LOW


@pytest.mark.integration
class TestStatisticalAnalysisIntegration:
    """Integration tests for statistical analysis engine"""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        engine = StatisticalAnalysisEngine()

        # Generate test data
        financial_data = np.random.lognormal(8, 1.5, 1000).tolist()
        text_data = ["Financial audit procedures"] * 100
        df_data = pd.DataFrame({
            'amount': np.random.lognormal(10, 1, 500),
            'vendor_id': np.random.randint(1, 100, 500),
            'payment_days': np.random.exponential(30, 500)
        })

        # Run all analyses
        benford_result = engine.analyze_benfords_law(financial_data)
        zipf_result = engine.analyze_zipfs_law(text_data)
        isolation_result = engine.detect_anomalies_isolation_forest(df_data)

        # Calculate comprehensive score
        all_results = [benford_result, zipf_result, isolation_result]
        comprehensive = engine.calculate_comprehensive_risk_score(all_results)

        # Generate findings
        findings = engine.generate_audit_findings(comprehensive, 'INTEGRATION_TEST')

        # Verify pipeline completion
        assert isinstance(comprehensive, dict)
        assert len(findings) >= 1
        assert all(f.audit_session_id == 'INTEGRATION_TEST' for f in findings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])