"""
Pytest configuration and fixtures for Financial Audit System tests
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Import your application modules
from backend.main import app
from backend.core.config import settings
from agents.definitions.audit_agent import AuditContext
from ml.models.fraud_detection_model import FraudDetectionModel
from ml.models.anomaly_detection_model import AnomalyDetectionModel


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for FastAPI application"""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock application settings for testing"""
    settings.DATABASE_URL = "sqlite:///test.db"
    settings.REDIS_URL = "redis://localhost:6379/1"
    settings.SECRET_KEY = "test-secret-key"
    settings.ENVIRONMENT = "test"
    return settings


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing"""
    return {
        "company_name": "Test Corp",
        "total_assets": 1000000,
        "total_liabilities": 600000,
        "shareholders_equity": 400000,
        "revenue": 500000,
        "net_income": 50000,
        "current_assets": 300000,
        "current_liabilities": 150000,
        "cash": 100000,
        "total_debt": 600000,
        "ebit": 75000,
        "interest_expense": 15000,
        "gross_profit": 200000
    }


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for fraud detection testing"""
    np.random.seed(42)  # For reproducible results

    n_transactions = 1000

    # Generate normal transactions
    normal_transactions = pd.DataFrame({
        'amount': np.random.lognormal(3, 1, int(n_transactions * 0.95)),
        'timestamp': pd.date_range('2024-01-01', periods=int(n_transactions * 0.95), freq='H'),
        'account_id': np.random.choice(['ACC001', 'ACC002', 'ACC003'], int(n_transactions * 0.95)),
        'merchant': np.random.choice(['Merchant A', 'Merchant B', 'Merchant C'], int(n_transactions * 0.95)),
        'is_fraud': 0
    })

    # Generate fraudulent transactions
    fraud_transactions = pd.DataFrame({
        'amount': np.random.uniform(1000, 10000, int(n_transactions * 0.05)),  # Higher amounts
        'timestamp': pd.date_range('2024-01-01', periods=int(n_transactions * 0.05), freq='H'),
        'account_id': np.random.choice(['ACC001', 'ACC002', 'ACC003'], int(n_transactions * 0.05)),
        'merchant': np.random.choice(['Suspicious Merchant', 'Unknown Vendor'], int(n_transactions * 0.05)),
        'is_fraud': 1
    })

    # Combine and shuffle
    all_transactions = pd.concat([normal_transactions, fraud_transactions], ignore_index=True)
    return all_transactions.sample(frac=1).reset_index(drop=True)


@pytest.fixture
def audit_context_sample():
    """Sample audit context for agent testing"""
    return AuditContext(
        audit_id="TEST-AUDIT-001",
        company_name="Test Corporation",
        financial_statements={
            "balance_sheet": {
                "total_assets": 1000000,
                "total_liabilities": 600000,
                "shareholders_equity": 400000
            },
            "income_statement": {
                "revenue": 500000,
                "expenses": 450000,
                "net_income": 50000
            }
        },
        audit_scope=["revenue_recognition", "expense_validation", "asset_verification"],
        risk_threshold=0.7
    )


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.set.return_value = True
    mock_redis.get.return_value = None
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.keys.return_value = []
    return mock_redis


@pytest.fixture
def fraud_detection_model():
    """Initialize fraud detection model for testing"""
    return FraudDetectionModel(model_type="PPO")


@pytest.fixture
def anomaly_detection_model():
    """Initialize anomaly detection model for testing"""
    return AnomalyDetectionModel(method="isolation_forest", contamination=0.1)


@pytest.fixture
def sample_anomaly_data():
    """Sample data for anomaly detection testing"""
    np.random.seed(42)

    # Normal data points
    normal_data = pd.DataFrame({
        'amount': np.random.normal(100, 20, 950),
        'transaction_count': np.random.poisson(5, 950),
        'hour': np.random.choice(range(9, 18), 950),  # Business hours
        'day_of_week': np.random.choice(range(0, 5), 950),  # Weekdays
        'account_age_days': np.random.normal(365, 100, 950)
    })

    # Anomalous data points
    anomaly_data = pd.DataFrame({
        'amount': np.random.uniform(1000, 5000, 50),  # Unusually high amounts
        'transaction_count': np.random.poisson(20, 50),  # High frequency
        'hour': np.random.choice([2, 3, 23], 50),  # Unusual hours
        'day_of_week': np.random.choice([5, 6], 50),  # Weekends
        'account_age_days': np.random.uniform(1, 30, 50)  # New accounts
    })

    # Combine
    all_data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    return all_data.sample(frac=1).reset_index(drop=True)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response from mock OpenAI"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_compliance_data():
    """Sample compliance data for testing"""
    return {
        "internal_controls": {
            "ceo_certification": True,
            "cfo_certification": True,
            "icfr_assessment": True,
            "disclosure_days": 2
        },
        "financial_statements": {
            "accounting_policies": "Comprehensive accounting policies documented",
            "significant_estimates": "Key estimates and assumptions disclosed",
            "contingencies": "Legal contingencies properly disclosed",
            "subsequent_events": "No significant subsequent events",
            "related_party_transactions": "Related party transactions disclosed",
            "segment_information": "Operating segments properly reported",
            "fair_value_measurements": "Fair value hierarchy disclosed"
        }
    }


@pytest.fixture
def test_database_url():
    """Test database URL"""
    return "sqlite:///./test_financial_audit.db"


# Test data cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test"""
    yield
    # Add cleanup logic here if needed
    import os
    test_files = ["test_financial_audit.db", "test_model.joblib", "test_model.zip"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


# Performance testing fixtures
@pytest.fixture
def large_transaction_dataset():
    """Large dataset for performance testing"""
    np.random.seed(42)
    n_transactions = 100000

    return pd.DataFrame({
        'amount': np.random.lognormal(4, 1.5, n_transactions),
        'timestamp': pd.date_range('2023-01-01', periods=n_transactions, freq='1min'),
        'account_id': np.random.choice([f'ACC{i:04d}' for i in range(1000)], n_transactions),
        'merchant_category': np.random.choice(['retail', 'gas', 'restaurant', 'online'], n_transactions),
        'is_weekend': np.random.choice([0, 1], n_transactions, p=[0.7, 0.3]),
        'is_fraud': np.random.choice([0, 1], n_transactions, p=[0.99, 0.01])
    })


# Security testing fixtures
@pytest.fixture
def sensitive_data_samples():
    """Sample data containing sensitive information for security testing"""
    return [
        "SSN: 123-45-6789",
        "Credit Card: 4532-1234-5678-9012",
        "Bank Account: 123456789012",
        "Email: test@example.com",
        "Phone: 555-123-4567",
        "API Key: abcd1234567890abcd1234567890abcd1234567890",
        "Regular text without sensitive data"
    ]


@pytest.fixture
def test_user_context():
    """Test user context for security testing"""
    return {
        "user_id": "test_user_001",
        "roles": ["auditor"],
        "permissions": ["read", "audit"],
        "session_id": "test_session_123",
        "ip_address": "192.168.1.100",
        "user_agent": "pytest-test-client"
    }