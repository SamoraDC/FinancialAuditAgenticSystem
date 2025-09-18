"""
Configuration settings for the Statistical Analysis Engine.

This module defines configuration parameters for statistical analysis,
anomaly detection thresholds, and risk assessment settings.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class AnalysisMode(str, Enum):
    """Statistical analysis modes"""
    CONSERVATIVE = "conservative"    # Strict thresholds, fewer false positives
    STANDARD = "standard"           # Balanced approach
    AGGRESSIVE = "aggressive"       # Sensitive detection, more flags
    CUSTOM = "custom"              # User-defined thresholds


class ModelType(str, Enum):
    """Available ML model types"""
    ISOLATION_FOREST = "isolation_forest"
    PYOD_ENSEMBLE = "pyod_ensemble"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "lof"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class BenfordConfig:
    """Configuration for Benford's Law analysis"""
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    minimum_sample_size: int = 30
    analyze_first_digit: bool = True
    analyze_second_digit: bool = True
    exclude_zeros: bool = True
    exclude_negatives: bool = False  # Convert to absolute values
    manipulation_detection: bool = True
    round_number_threshold: float = 0.35  # Flag if '1' frequency > 35%
    psychological_bias_threshold: float = 0.20  # Flag if '2' frequency > 20%


@dataclass
class ZipfConfig:
    """Configuration for Zipf's Law analysis"""
    minimum_word_length: int = 3
    remove_stopwords: bool = False
    correlation_threshold: float = 0.8
    top_words_analysis: int = 100
    deviation_threshold: float = 0.5  # 50% deviation from expected
    text_preprocessing: bool = True
    case_sensitive: bool = False


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection models"""
    contamination_conservative: float = 0.05
    contamination_standard: float = 0.10
    contamination_aggressive: float = 0.15
    ensemble_methods: List[str] = None
    isolation_forest_estimators: int = 100
    isolation_forest_max_samples: str = "auto"
    lof_neighbors: int = 20
    ocsvm_kernel: str = "rbf"
    ocsvm_gamma: str = "scale"
    feature_scaling: str = "standard"  # "standard", "minmax", "robust"
    outlier_threshold_percentile: float = 90.0

    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ["isolation_forest", "lof", "ocsvm"]


@dataclass
class ReinforcementLearningConfig:
    """Configuration for RL-based anomaly detection"""
    model_type: str = "DQN"  # "DQN" or "PPO"
    training_episodes: int = 1000
    learning_rate: float = 1e-3
    buffer_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 4
    target_update_frequency: int = 100
    reward_positive: float = 1.0  # Reward for correct flag
    reward_negative: float = -1.0  # Penalty for false flag
    reward_miss: float = -10.0  # Heavy penalty for missing fraud
    enable_training: bool = False  # Enable for production
    save_model_path: str = "models/rl_anomaly_detector.pkl"
    load_pretrained: bool = True


@dataclass
class RiskScoringConfig:
    """Configuration for risk scoring and thresholds"""
    # Risk level thresholds (0-1 scale)
    low_threshold: float = 0.3
    medium_threshold: float = 0.6
    high_threshold: float = 0.8
    critical_threshold: float = 0.9

    # Method weights for composite scoring
    benford_weight: float = 0.25
    zipf_weight: float = 0.15
    isolation_forest_weight: float = 0.20
    pyod_ensemble_weight: float = 0.25
    reinforcement_learning_weight: float = 0.15

    # Confidence scoring
    minimum_confidence: float = 0.7
    confidence_decay_factor: float = 0.9  # Reduce confidence for edge cases

    # Risk factor identification
    high_risk_benford_score: float = 0.7
    high_risk_anomaly_rate: float = 0.15
    suspicious_pattern_threshold: int = 3  # Number of patterns to flag as suspicious


@dataclass
class DataQualityConfig:
    """Configuration for data quality assessment"""
    minimum_sample_size: int = 30
    maximum_missing_percentage: float = 20.0  # 20% missing data threshold
    maximum_duplicate_percentage: float = 5.0   # 5% duplicate threshold
    maximum_outlier_percentage: float = 10.0    # 10% outlier threshold
    high_correlation_threshold: float = 0.9     # Flag highly correlated features
    zero_variance_tolerance: bool = False       # Allow zero variance features
    data_type_validation: bool = True           # Validate expected data types


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    analysis_timeout_seconds: int = 300  # 5 minutes per analysis
    cache_results: bool = True
    cache_expiry_hours: int = 24
    memory_limit_mb: int = 1024  # 1GB memory limit
    enable_gpu_acceleration: bool = False  # For large datasets


class StatisticalAnalysisConfig:
    """Main configuration class for statistical analysis"""

    def __init__(self, mode: AnalysisMode = AnalysisMode.STANDARD):
        self.mode = mode
        self.benford = BenfordConfig()
        self.zipf = ZipfConfig()
        self.anomaly_detection = AnomalyDetectionConfig()
        self.reinforcement_learning = ReinforcementLearningConfig()
        self.risk_scoring = RiskScoringConfig()
        self.data_quality = DataQualityConfig()
        self.performance = PerformanceConfig()

        # Apply mode-specific adjustments
        self._apply_mode_settings()

    def _apply_mode_settings(self):
        """Apply mode-specific configuration adjustments"""
        if self.mode == AnalysisMode.CONSERVATIVE:
            # Stricter thresholds, fewer false positives
            self.risk_scoring.low_threshold = 0.2
            self.risk_scoring.medium_threshold = 0.5
            self.risk_scoring.high_threshold = 0.75
            self.risk_scoring.critical_threshold = 0.85

            self.anomaly_detection.contamination_conservative = 0.03
            self.anomaly_detection.contamination_standard = 0.05
            self.anomaly_detection.contamination_aggressive = 0.08

            self.benford.significance_threshold = 0.01  # 99% confidence
            self.benford.minimum_sample_size = 50

            self.data_quality.minimum_sample_size = 50
            self.data_quality.maximum_missing_percentage = 10.0

        elif self.mode == AnalysisMode.AGGRESSIVE:
            # More sensitive, higher detection rate
            self.risk_scoring.low_threshold = 0.4
            self.risk_scoring.medium_threshold = 0.65
            self.risk_scoring.high_threshold = 0.85
            self.risk_scoring.critical_threshold = 0.95

            self.anomaly_detection.contamination_conservative = 0.08
            self.anomaly_detection.contamination_standard = 0.15
            self.anomaly_detection.contamination_aggressive = 0.20

            self.benford.significance_threshold = 0.10  # 90% confidence
            self.benford.minimum_sample_size = 20

            self.benford.round_number_threshold = 0.30
            self.benford.psychological_bias_threshold = 0.15

        # Standard mode uses default values

    def get_contamination_rate(self, sensitivity: str = "standard") -> float:
        """Get contamination rate based on sensitivity level"""
        contamination_map = {
            "conservative": self.anomaly_detection.contamination_conservative,
            "standard": self.anomaly_detection.contamination_standard,
            "aggressive": self.anomaly_detection.contamination_aggressive
        }
        return contamination_map.get(sensitivity, self.anomaly_detection.contamination_standard)

    def get_risk_weights(self) -> Dict[str, float]:
        """Get risk scoring weights as dictionary"""
        return {
            "benford": self.risk_scoring.benford_weight,
            "zipf": self.risk_scoring.zipf_weight,
            "isolation_forest": self.risk_scoring.isolation_forest_weight,
            "pyod_ensemble": self.risk_scoring.pyod_ensemble_weight,
            "reinforcement_learning": self.risk_scoring.reinforcement_learning_weight
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "mode": self.mode.value,
            "benford": self.benford.__dict__,
            "zipf": self.zipf.__dict__,
            "anomaly_detection": self.anomaly_detection.__dict__,
            "reinforcement_learning": self.reinforcement_learning.__dict__,
            "risk_scoring": self.risk_scoring.__dict__,
            "data_quality": self.data_quality.__dict__,
            "performance": self.performance.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StatisticalAnalysisConfig':
        """Create configuration from dictionary"""
        mode = AnalysisMode(config_dict.get("mode", "standard"))
        config = cls(mode)

        # Update configurations from dictionary
        for section_name, section_config in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_config, dict):
                section_obj = getattr(config, section_name)
                for key, value in section_config.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

        return config


# Predefined configurations for different use cases
class PresetConfigurations:
    """Predefined configuration presets for common use cases"""

    @staticmethod
    def financial_audit_standard() -> StatisticalAnalysisConfig:
        """Standard configuration for financial audits"""
        config = StatisticalAnalysisConfig(AnalysisMode.STANDARD)
        config.benford.minimum_sample_size = 50
        config.risk_scoring.benford_weight = 0.30  # Higher weight for financial data
        config.anomaly_detection.ensemble_methods = ["isolation_forest", "lof", "ocsvm"]
        return config

    @staticmethod
    def fraud_investigation() -> StatisticalAnalysisConfig:
        """Aggressive configuration for fraud investigation"""
        config = StatisticalAnalysisConfig(AnalysisMode.AGGRESSIVE)
        config.reinforcement_learning.enable_training = True
        config.risk_scoring.reinforcement_learning_weight = 0.25
        config.benford.manipulation_detection = True
        return config

    @staticmethod
    def compliance_review() -> StatisticalAnalysisConfig:
        """Conservative configuration for compliance reviews"""
        config = StatisticalAnalysisConfig(AnalysisMode.CONSERVATIVE)
        config.data_quality.data_type_validation = True
        config.data_quality.maximum_missing_percentage = 5.0
        config.risk_scoring.minimum_confidence = 0.8
        return config

    @staticmethod
    def high_volume_processing() -> StatisticalAnalysisConfig:
        """Optimized configuration for high-volume data processing"""
        config = StatisticalAnalysisConfig(AnalysisMode.STANDARD)
        config.performance.enable_parallel_processing = True
        config.performance.max_worker_threads = 8
        config.performance.cache_results = True
        config.anomaly_detection.isolation_forest_estimators = 50  # Fewer trees for speed
        return config

    @staticmethod
    def research_analysis() -> StatisticalAnalysisConfig:
        """Comprehensive configuration for research purposes"""
        config = StatisticalAnalysisConfig(AnalysisMode.STANDARD)
        config.benford.analyze_first_digit = True
        config.benford.analyze_second_digit = True
        config.zipf.remove_stopwords = True
        config.zipf.top_words_analysis = 200
        config.anomaly_detection.ensemble_methods = [
            "isolation_forest", "lof", "ocsvm", "knn"
        ]
        return config


# Default configuration instance
default_config = StatisticalAnalysisConfig(AnalysisMode.STANDARD)