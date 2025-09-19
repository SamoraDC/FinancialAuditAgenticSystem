"""
PyTest configuration and fixtures for Financial Audit System tests
"""

import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.duckdb_manager import DuckDBManager
from backend.services.document_processor import DocumentProcessor
from backend.services.statistical_analyzer import StatisticalAnalyzer
from backend.services.groq_llm_service import GroqLLMService
from backend.services.guardrails_service import GuardRailsSecurityService
from backend.workflows.audit_workflow import AuditWorkflow, AuditState


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
async def db_manager(temp_db_path):
    """DuckDB manager with temporary database"""
    manager = DuckDBManager(temp_db_path)
    yield manager
    manager.close()


@pytest.fixture
def document_processor():
    """Document processor instance"""
    return DocumentProcessor()


@pytest.fixture
def statistical_analyzer():
    """Statistical analyzer instance"""
    return StatisticalAnalyzer()


@pytest.fixture
def mock_groq_service():
    """Mock Groq LLM service"""
    service = MagicMock(spec=GroqLLMService)

    # Mock successful LLM response
    mock_response = MagicMock()
    mock_response.content = '{"analysis": "test analysis", "risk_level": "low"}'
    mock_response.model = "openai/gpt-oss-120b"
    mock_response.tokens_used = 150
    mock_response.response_time = 1.2
    mock_response.confidence_score = 0.85
    mock_response.safety_check = {"input_safe": True, "output_safe": True}

    service.safe_completion = AsyncMock(return_value=mock_response)
    service.analyze_financial_document = AsyncMock(return_value=mock_response)
    service.generate_audit_finding = AsyncMock(return_value=mock_response)
    service.assess_compliance_risk = AsyncMock(return_value=mock_response)

    return service


@pytest.fixture
def security_service():
    """GuardRails security service instance"""
    return GuardRailsSecurityService()


@pytest.fixture
def audit_workflow(db_manager, mock_groq_service):
    """Audit workflow with mocked dependencies"""
    workflow = AuditWorkflow()
    workflow.llm_service = mock_groq_service
    workflow.db_manager = db_manager
    return workflow


@pytest.fixture
def sample_financial_data():
    """Sample financial transaction data for testing"""
    return pd.DataFrame({
        'amount': [1234.56, 2345.67, 3456.78, 4567.89, 5678.90,
                  6789.01, 7890.12, 8901.23, 9012.34, 1023.45],
        'vendor_name': ['Vendor A', 'Vendor B', 'Vendor A', 'Vendor C', 'Vendor B',
                       'Vendor D', 'Vendor A', 'Vendor E', 'Vendor B', 'Vendor F'],
        'transaction_date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'description': ['Payment 1', 'Payment 2', 'Payment 3', 'Payment 4', 'Payment 5',
                       'Payment 6', 'Payment 7', 'Payment 8', 'Payment 9', 'Payment 10'],
        'account_code': ['1001', '1002', '1001', '1003', '1002',
                        '1004', '1001', '1005', '1002', '1006']
    })


@pytest.fixture
def sample_benford_data():
    """Sample data that follows Benford's Law for testing"""
    # Generate data that approximately follows Benford's Law
    amounts = []
    for digit in range(1, 10):
        # Benford's Law probability
        prob = np.log10(1 + 1/digit)
        count = int(prob * 1000)  # Scale to get reasonable sample size

        # Generate amounts starting with this digit
        for _ in range(count):
            # Random amount between digit*10^n and (digit+1)*10^n
            power = np.random.randint(1, 5)
            base = digit * (10 ** power)
            amount = base + np.random.uniform(0, 10 ** power)
            amounts.append(amount)

    return amounts


@pytest.fixture
def sample_documents():
    """Sample document data for testing"""
    return [
        {
            'filename': 'invoice_001.pdf',
            'original_filename': 'invoice_001.pdf',
            'file_path': '/tmp/test_docs/invoice_001.pdf',
            'document_type': 'invoice',
            'file_size': 1024,
            'mime_type': 'application/pdf',
            'content': 'Sample invoice content with amount $1,234.56'
        },
        {
            'filename': 'bank_statement.pdf',
            'original_filename': 'bank_statement.pdf',
            'file_path': '/tmp/test_docs/bank_statement.pdf',
            'document_type': 'bank_statement',
            'file_size': 2048,
            'mime_type': 'application/pdf',
            'content': 'Bank statement with multiple transactions'
        }
    ]


@pytest.fixture
def sample_audit_state():
    """Sample audit state for testing workflow"""
    return AuditState(
        session_id="test_session_001",
        client_name="Test Client Corp",
        auditor_id="auditor_001",
        audit_type="financial",
        fiscal_year=2024,
        uploaded_documents=[
            {
                'filename': 'test_doc.pdf',
                'original_filename': 'test_doc.pdf',
                'file_path': '/tmp/test_doc.pdf',
                'document_type': 'invoice',
                'file_size': 1024
            }
        ],
        processed_documents=[],
        extracted_content=[],
        financial_transactions=[],
        statistical_results={},
        anomalies_detected=[],
        llm_analyses=[],
        audit_findings=[],
        risk_assessments=[],
        current_step="initialized",
        progress_percentage=0.0,
        error_messages=[],
        human_review_required=False,
        audit_report=None,
        compliance_status={},
        recommendations=[]
    )


@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing"""
    import fakeredis.aioredis
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def test_file_content():
    """Sample file content for document processing tests"""
    return {
        'pdf_text': """
        INVOICE
        Invoice Number: INV-001
        Date: 2024-01-15
        Amount: $1,234.56
        Vendor: Test Vendor Inc.
        Description: Professional services
        """,
        'financial_data': """
        Account Code | Description | Amount
        1001 | Office Supplies | $234.56
        1002 | Travel Expenses | $1,456.78
        1003 | Professional Fees | $3,456.78
        """,
        'suspicious_data': """
        Transaction: $9,999.99
        Transaction: $10,000.00
        Transaction: $0.01
        """
    }


@pytest.fixture
def mock_langextract(monkeypatch):
    """Mock langextract library for testing"""
    mock_result = {
        'text': 'Sample extracted text content',
        'metadata': {'extraction_method': 'mock'},
        'tables': [
            {
                'data': [
                    ['Description', 'Amount'],
                    ['Test Item 1', '$123.45'],
                    ['Test Item 2', '$678.90']
                ]
            }
        ],
        'images': [],
        'confidence': 0.9
    }

    def mock_extract(file_path, **kwargs):
        return mock_result

    import langextract
    monkeypatch.setattr(langextract, 'extract', mock_extract)
    return mock_result


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing"""
    import numpy as np

    # Generate large dataset for performance testing
    size = 10000
    amounts = np.random.exponential(1000, size)  # Exponential distribution
    vendors = [f"Vendor_{i % 100}" for i in range(size)]  # 100 unique vendors
    dates = pd.date_range('2023-01-01', periods=size, freq='H')

    return pd.DataFrame({
        'amount': amounts,
        'vendor_name': vendors,
        'transaction_date': dates,
        'description': [f"Transaction {i}" for i in range(size)],
        'account_code': [f"100{i % 10}" for i in range(size)]
    })


@pytest.fixture
async def populated_database(db_manager, sample_financial_data):
    """Database with sample data for integration tests"""
    # Create audit session
    session_data = {
        'session_name': 'Test Audit Session',
        'client_name': 'Test Client',
        'auditor_id': 'test_auditor',
        'audit_type': 'financial',
        'fiscal_year': 2024,
        'materiality_threshold': 10000.0,
        'risk_tolerance': 'medium',
        'compliance_frameworks': ['GAAP'],
        'audit_scope': ['financial_statements']
    }

    session_id = db_manager.create_audit_session(session_data)

    # Insert sample transactions
    transactions = []
    for _, row in sample_financial_data.iterrows():
        transactions.append({
            'audit_session_id': session_id,
            'document_id': None,
            'transaction_date': row['transaction_date'],
            'description': row['description'],
            'amount': row['amount'],
            'account_code': row['account_code'],
            'vendor_name': row['vendor_name'],
            'transaction_type': 'expense',
            'confidence_score': 0.9
        })

    db_manager.insert_financial_transactions(transactions)

    return {
        'session_id': session_id,
        'transaction_count': len(transactions)
    }


# Test data generators
def generate_benford_compliant_data(size=1000):
    """Generate data that follows Benford's Law"""
    import numpy as np

    amounts = []
    for digit in range(1, 10):
        prob = np.log10(1 + 1/digit)
        count = int(prob * size)

        for _ in range(count):
            power = np.random.randint(1, 6)
            base = digit * (10 ** power)
            amount = base + np.random.uniform(0, 10 ** power)
            amounts.append(amount)

    return amounts


def generate_benford_non_compliant_data(size=1000):
    """Generate data that does NOT follow Benford's Law"""
    import numpy as np

    # Generate uniform distribution (should fail Benford's test)
    amounts = np.random.uniform(1000, 9999, size)
    return amounts.tolist()


# Utility functions for tests
def create_test_pdf_file(content: str, file_path: str):
    """Create a test PDF file with given content"""
    # This would use a PDF generation library in real implementation
    with open(file_path, 'w') as f:
        f.write(content)


def create_test_document_files(temp_dir: str):
    """Create test document files for processing"""
    test_files = {}

    # Create test PDF
    pdf_path = os.path.join(temp_dir, 'test_invoice.pdf')
    create_test_pdf_file("Test invoice content", pdf_path)
    test_files['pdf'] = pdf_path

    # Create test text file
    txt_path = os.path.join(temp_dir, 'test_statement.txt')
    with open(txt_path, 'w') as f:
        f.write("Test financial statement content")
    test_files['txt'] = txt_path

    return test_files


# Import numpy for data generation
import numpy as np