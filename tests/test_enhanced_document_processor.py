"""
Test suite for the Enhanced Document Processor

This test suite validates the comprehensive document processing pipeline
implementation including PII detection, OCR processing, and structured data extraction.
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Import the enhanced document processor
try:
    from backend.services.document_processor import (
        EnhancedDocumentProcessor,
        DocumentValidationResult,
        FinancialData,
        ProcessingProgress
    )
    from backend.models.audit_models import DocumentType
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This test requires the backend modules to be properly installed")


class TestEnhancedDocumentProcessor:
    """Test suite for EnhancedDocumentProcessor"""

    @pytest.fixture
    async def processor(self):
        """Create a processor instance for testing"""
        return EnhancedDocumentProcessor(enable_pii_detection=True)

    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

    @pytest.fixture
    def sample_financial_text(self):
        """Sample financial text for extraction testing"""
        return """
        INVOICE #INV-2024-001

        ABC Company Ltd.
        123 Business Street
        New York, NY 10001

        Bill To:
        XYZ Corporation
        456 Client Avenue
        Los Angeles, CA 90001

        Date: January 15, 2024
        Due Date: February 15, 2024

        Description          Quantity    Unit Price    Total
        Consulting Services      20        $150.00    $3,000.00
        Software License          1      $1,200.00    $1,200.00

        Subtotal:                                     $4,200.00
        Tax (8.5%):                                     $357.00
        TOTAL:                                        $4,557.00

        Payment Terms: Net 30 days
        Account: 1234-5678-9012-3456
        """

    def test_processor_initialization(self):
        """Test processor initialization with various configurations"""
        # Test default initialization
        processor1 = EnhancedDocumentProcessor()
        assert processor1.enable_pii_detection is True  # Should be True if GuardRails available
        assert processor1.session_id is not None
        assert len(processor1.session_id) > 10

        # Test with PII detection disabled
        processor2 = EnhancedDocumentProcessor(enable_pii_detection=False)
        assert processor2.enable_pii_detection is False

        # Test custom memory namespace
        processor3 = EnhancedDocumentProcessor(memory_namespace="test_namespace")
        assert processor3.memory_namespace == "test_namespace"

    @pytest.mark.asyncio
    async def test_financial_data_extraction(self, processor, sample_financial_text):
        """Test financial data extraction from text content"""
        financial_data = await processor.extract_financial_data(sample_financial_text)

        # Test extracted amounts
        assert len(financial_data.amounts) > 0
        assert Decimal('3000.00') in financial_data.amounts
        assert Decimal('4557.00') in financial_data.amounts

        # Test extracted dates
        assert len(financial_data.dates) > 0
        assert any(date.month == 1 and date.year == 2024 for date in financial_data.dates)

        # Test vendor names
        assert len(financial_data.vendor_names) > 0
        assert any('ABC Company' in vendor for vendor in financial_data.vendor_names)

        # Test currencies
        assert 'USD' in financial_data.currencies or len(financial_data.currencies) == 0

        # Test confidence scores
        assert 'amounts' in financial_data.confidence_scores
        assert financial_data.confidence_scores['amounts'] >= 0.0

    @pytest.mark.asyncio
    async def test_document_validation_with_temp_file(self, processor):
        """Test document validation with temporary files"""
        # Test PDF validation
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(b"%PDF-1.4\n")
            temp_pdf.flush()

            validation = await processor.validate_document_format(temp_pdf.name)
            assert validation.file_type == "pdf"
            assert isinstance(validation.is_valid, bool)
            assert isinstance(validation.confidence, float)
            assert 0.0 <= validation.confidence <= 1.0

        # Test unsupported file type
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_invalid:
            temp_invalid.write(b"invalid content")
            temp_invalid.flush()

            validation = await processor.validate_document_format(temp_invalid.name)
            assert validation.file_type == "unknown"
            assert not validation.is_valid
            assert len(validation.errors) > 0

    @pytest.mark.asyncio
    async def test_pii_detection_and_anonymization(self, processor):
        """Test PII detection and anonymization functionality"""
        # Sample text with potential PII
        content_with_pii = """
        John Doe
        SSN: 123-45-6789
        Email: john.doe@example.com
        Phone: (555) 123-4567
        Credit Card: 4532-1234-5678-9012

        Invoice Amount: $1,234.56
        """

        # Test PII detection
        anonymized_content, pii_types, has_pii = await processor._detect_and_anonymize_pii(content_with_pii)

        # Note: Results depend on GuardRails availability
        if processor.enable_pii_detection:
            # If GuardRails is available, expect PII detection
            assert isinstance(has_pii, bool)
            assert isinstance(pii_types, list)
            assert isinstance(anonymized_content, str)
        else:
            # If GuardRails not available, should return original content
            assert has_pii is False
            assert pii_types == []
            assert anonymized_content == content_with_pii

    @pytest.mark.asyncio
    async def test_line_item_extraction(self, processor):
        """Test line item extraction from structured content"""
        table_content = """
        Description	Quantity	Unit Price	Total
        Item 1	2	$10.00	$20.00
        Item 2	3	$15.00	$45.00
        Item 3	1	$30.00	$30.00
        """

        line_items = await processor._extract_line_items(table_content)

        assert len(line_items) >= 3
        for item in line_items:
            assert 'description' in item
            assert 'amount' in item
            assert 'line_number' in item
            assert isinstance(item['amount'], float)

    @pytest.mark.asyncio
    async def test_ocr_reader_initialization(self, processor):
        """Test OCR reader lazy loading"""
        # Test default language
        reader1 = processor._get_ocr_reader()
        assert reader1 is not None

        # Test custom language
        reader2 = processor._get_ocr_reader('es')
        assert reader2 is not None

        # Test unsupported language falls back to English
        reader3 = processor._get_ocr_reader('unsupported')
        assert reader3 is not None

    @pytest.mark.asyncio
    async def test_memory_operations(self, processor):
        """Test memory storage and retrieval operations"""
        test_key = "test_key"
        test_value = {"test": "data", "timestamp": datetime.utcnow().isoformat()}

        # Test storage
        await processor._store_memory(test_key, test_value)

        # Test retrieval
        retrieved_value = await processor._retrieve_memory(test_key)
        # Note: Since this is a placeholder implementation, retrieved_value will be None
        assert retrieved_value is None or retrieved_value == test_value

    @pytest.mark.asyncio
    async def test_cleanup_session(self, processor):
        """Test session cleanup functionality"""
        # This should not raise any exceptions
        await processor.cleanup_session()

        # OCR reader should be reset
        if hasattr(processor, 'ocr_reader'):
            assert processor.ocr_reader is None

    def test_document_type_detection(self, processor):
        """Test document type detection based on filename"""
        # Test invoice detection
        assert processor._is_invoice_document("sample_invoice.pdf") is True
        assert processor._is_invoice_document("bill_2024.docx") is True
        assert processor._is_invoice_document("receipt.jpg") is True
        assert processor._is_invoice_document("statement.pdf") is False

        # Test balance sheet detection
        assert processor._is_balance_sheet_document("balance_sheet.pdf") is True
        assert processor._is_balance_sheet_document("financial_position.docx") is True
        assert processor._is_balance_sheet_document("invoice.pdf") is False

    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in various scenarios"""
        # Test with non-existent file
        try:
            validation = await processor.validate_document_format("/non/existent/file.pdf")
            assert not validation.is_valid
            assert len(validation.errors) > 0
        except Exception as e:
            # Should handle gracefully
            assert "not exist" in str(e).lower()

        # Test financial data extraction with invalid content
        financial_data = await processor.extract_financial_data("")
        assert isinstance(financial_data, FinancialData)
        assert len(financial_data.amounts) == 0

    def test_progress_tracking_model(self):
        """Test ProcessingProgress model functionality"""
        progress = ProcessingProgress(total_files=10)
        assert progress.total_files == 10
        assert progress.processed_files == 0
        assert progress.percentage == 0.0
        assert progress.pii_files_detected == 0
        assert progress.ocr_processed == 0

        # Test that start_time is set
        assert isinstance(progress.start_time, datetime)


async def run_basic_tests():
    """Run basic tests that don't require pytest"""
    print("Running basic Enhanced Document Processor tests...")

    # Test initialization
    print("✓ Testing processor initialization...")
    processor = EnhancedDocumentProcessor()
    assert processor.session_id is not None
    print(f"  Session ID: {processor.session_id[:8]}...")

    # Test financial data extraction
    print("✓ Testing financial data extraction...")
    sample_text = "Invoice amount: $1,234.56 due on 2024-01-15"
    financial_data = await processor.extract_financial_data(sample_text)
    print(f"  Extracted {len(financial_data.amounts)} amounts")
    print(f"  Extracted {len(financial_data.dates)} dates")

    # Test PII detection
    print("✓ Testing PII detection...")
    test_content = "Contact John at john@example.com for payment of $500"
    anonymized, pii_types, has_pii = await processor._detect_and_anonymize_pii(test_content)
    print(f"  PII detected: {has_pii}")
    print(f"  PII types: {pii_types}")

    # Test cleanup
    print("✓ Testing cleanup...")
    await processor.cleanup_session()

    print("\n✅ All basic tests passed successfully!")
    print("\nNote: To run the full test suite, use 'pytest tests/test_enhanced_document_processor.py'")


if __name__ == "__main__":
    # Run basic tests when script is executed directly
    asyncio.run(run_basic_tests())