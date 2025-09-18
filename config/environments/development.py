"""
Development Environment Configuration
"""

import os
from typing import Dict, Any


class DevelopmentConfig:
    """Development environment specific configuration"""

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/financial_audit_dev"
    )

    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # API Keys (use environment variables in development)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LANGGRAPH_API_KEY = os.getenv("LANGGRAPH_API_KEY", "")

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Longer tokens for development

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4317"
    )
    OTEL_SERVICE_NAME = "financial-audit-backend-dev"

    # Logging
    LOG_LEVEL = "DEBUG"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # CORS
    ALLOWED_HOSTS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    # File uploads
    UPLOAD_FOLDER = "uploads/dev"
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

    # ML Models
    MODEL_CACHE_DIR = "models/cache/dev"
    ENABLE_MODEL_TRAINING = True

    # Feature flags
    ENABLE_FRAUD_DETECTION = True
    ENABLE_ANOMALY_DETECTION = True
    ENABLE_COMPLIANCE_CHECKING = True
    ENABLE_REAL_TIME_MONITORING = True

    # Performance
    WORKER_PROCESSES = 1
    WORKER_CONNECTIONS = 1000

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }


# Environment-specific database configurations
DATABASE_CONFIGS = {
    "default": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "PORT": 5432,
        "NAME": "financial_audit_dev",
        "USER": "postgres",
        "PASSWORD": "password",
        "OPTIONS": {
            "connect_timeout": 10,
            "application_name": "financial_audit_dev"
        }
    }
}

# Redis configurations
REDIS_CONFIGS = {
    "default": {
        "HOST": "localhost",
        "PORT": 6379,
        "DB": 0,
        "PASSWORD": None,
        "TIMEOUT": 5
    },
    "cache": {
        "HOST": "localhost",
        "PORT": 6379,
        "DB": 1,
        "PASSWORD": None,
        "TIMEOUT": 5
    },
    "sessions": {
        "HOST": "localhost",
        "PORT": 6379,
        "DB": 2,
        "PASSWORD": None,
        "TIMEOUT": 5
    }
}

# OpenTelemetry development configuration
OTEL_CONFIG = {
    "TRACES_EXPORTER": "otlp",
    "METRICS_EXPORTER": "otlp",
    "LOGS_EXPORTER": "otlp",
    "RESOURCE_ATTRIBUTES": {
        "service.name": "financial-audit-backend",
        "service.version": "1.0.0-dev",
        "deployment.environment": "development"
    },
    "INSTRUMENTATION": {
        "fastapi": True,
        "redis": True,
        "requests": True,
        "sqlalchemy": True,
        "logging": True
    }
}