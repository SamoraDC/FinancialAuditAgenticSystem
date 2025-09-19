"""
Application configuration using Pydantic settings
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Financial Audit Agentic System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    TEST_MODE: bool = False

    # API
    API_V1_STR: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/financial_audit"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # LangGraph
    LANGGRAPH_API_KEY: str = ""

    # OpenAI
    OPENAI_API_KEY: str = "test-key-for-ci"  # Default test key for CI/CD

    # Groq
    GROQ_API_KEY: str = "test-key-for-ci"  # Default test key for CI/CD
    GROQ_MODEL_MAIN: str = "openai/gpt-oss-120b"  # Main LLM model for Groq
    GROQ_MODEL_GUARDRAILS: str = "meta-llama/llama-guard-4-12b"  # Guardrails model

    # Guardrails
    GUARDRAILS_API_KEY: str = ""

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    OTEL_SERVICE_NAME: str = "financial-audit-backend"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra environment variables


settings = Settings()

def get_settings() -> Settings:
    """Get application settings instance"""
    return settings