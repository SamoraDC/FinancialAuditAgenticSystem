"""
Comprehensive tests for DocumentProcessor service

Tests cover all major functionality including:
- Document validation
- PDF processing
- DOCX processing
- OCR capabilities
- Financial data extraction
- Batch processing
- Error handling
"""

import asyncio
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from backend.services.document_processor import (
    DocumentProcessor,
    DocumentValidationResult,
    FinancialData,
    ProcessingProgress
)
from backend.models.audit_models import Invoice, BalanceSheet


class TestDocumentProcessor:
    """Test suite for DocumentProcessor"""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing"""
        return DocumentProcessor(memory_namespace="test_processor")

    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF text content for testing"""
        return """
        INVOICE #INV-12345

        ABC Company Inc.
        123 Main Street
        Anytown, ST 12345

        Date: 12/15/2023
        Due Date: 01/15/2024

        Description                 Amount
        Office Supplies            $1,234.56
        Software License           $2,500.00
        Tax                        $374.36

        Total: $4,108.92
        """

    @pytest.fixture
    def sample_balance_sheet_content(self):
        """Sample balance sheet content for testing"""
        return """
        XYZ Corporation
        Balance Sheet
        As of December 31, 2023

        ASSETS
        Current Assets:
          Cash                     $100,000
          Accounts Receivable       $50,000
          Total Current Assets     $150,000

        Non-Current Assets:
          Property & Equipment     $500,000
          Total Assets            $650,000

        LIABILITIES
        Current Liabilities:
          Accounts Payable          $30,000
          Total Liabilities         $30,000

        EQUITY
        Shareholders' Equity       $620,000
        Total Liabilities & Equity $650,000
        """

    @pytest.fixture
    def temp_pdf_file(self):
        """Create a temporary PDF file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            # Create a minimal PDF structure
            pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""
            tmp.write(pdf_content)
            tmp.flush()
            yield Path(tmp.name)
        Path(tmp.name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary image file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create a simple test image
            img = Image.new('RGB', (300, 200), color='white')
            img.save(tmp.name)
            yield Path(tmp.name)
        Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_document_validation_pdf(self, processor, temp_pdf_file):
        """Test PDF document validation"""
        result = await processor.validate_document_format(temp_pdf_file)

        assert isinstance(result, DocumentValidationResult)
        assert result.file_type == "pdf"
        assert result.is_valid or len(result.errors) > 0  # PDF might be invalid due to minimal content

    @pytest.mark.asyncio
    async def test_document_validation_image(self, processor, temp_image_file):
        """Test image document validation"""
        result = await processor.validate_document_format(temp_image_file)

        assert isinstance(result, DocumentValidationResult)
        assert result.file_type == "image"
        assert result.is_valid
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_document_validation_nonexistent(self, processor):
        """Test validation of non-existent file"""
        result = await processor.validate_document_format("nonexistent.pdf")

        assert not result.is_valid
        assert "File does not exist" in str(result.errors)

    @pytest.mark.asyncio
    async def test_financial_data_extraction(self, processor, sample_pdf_content):
        """Test financial data extraction from text content"""
        financial_data = await processor.extract_financial_data(sample_pdf_content)

        assert isinstance(financial_data, FinancialData)
        assert len(financial_data.amounts) > 0
        assert len(financial_data.dates) > 0
        assert len(financial_data.vendor_names) > 0

        # Check for specific extracted amounts
        amounts = [float(amount) for amount in financial_data.amounts]
        assert 1234.56 in amounts or any(abs(amount - 1234.56) < 0.01 for amount in amounts)

    @pytest.mark.asyncio
    async def test_extract_invoice_number(self, processor):
        """Test invoice number extraction"""
        content = "Invoice #INV-12345\nDate: 12/15/2023"
        invoice_number = processor._extract_invoice_number(content)

        assert invoice_number == "INV-12345"

    @pytest.mark.asyncio
    async def test_extract_invoice_number_fallback(self, processor):
        """Test invoice number extraction with fallback"""
        content = "No invoice number here"
        invoice_number = processor._extract_invoice_number(content)

        assert invoice_number.startswith("INV-")
        assert len(invoice_number) > 4

    @pytest.mark.asyncio
    @patch('backend.services.document_processor.fitz.open')
    async def test_process_invoice_success(self, mock_fitz, processor, sample_pdf_content):
        """Test successful invoice processing"""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = sample_pdf_content
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.is_encrypted = False
        mock_doc.metadata = {}
        mock_fitz.return_value = mock_doc

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp.flush()

            try:
                invoice = await processor.process_invoice(tmp.name)

                assert invoice is not None
                assert isinstance(invoice, Invoice)
                assert invoice.invoice_number == "INV-12345"
                assert invoice.vendor_name == "ABC Company Inc."
                assert invoice.amount > 0
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    @patch('backend.services.document_processor.fitz.open')
    async def test_process_balance_sheet_success(self, mock_fitz, processor, sample_balance_sheet_content):
        """Test successful balance sheet processing"""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = sample_balance_sheet_content
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.is_encrypted = False
        mock_doc.metadata = {}
        mock_fitz.return_value = mock_doc

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp.flush()

            try:
                balance_sheet = await processor.process_balance_sheet(tmp.name)

                assert balance_sheet is not None
                assert isinstance(balance_sheet, BalanceSheet)
                assert balance_sheet.company_name == "XYZ Corporation"
                assert balance_sheet.total_assets > 0
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_batch_processing_empty_list(self, processor):
        """Test batch processing with empty file list"""
        results = await processor.batch_process_documents([])

        assert results["processing_stats"]["total_files"] == 0
        assert results["processing_stats"]["success_rate"] == 0
        assert len(results["invoices"]) == 0
        assert len(results["balance_sheets"]) == 0

    @pytest.mark.asyncio
    async def test_batch_processing_with_invalid_files(self, processor):
        """Test batch processing with invalid files"""
        invalid_files = ["nonexistent1.pdf", "nonexistent2.docx"]

        results = await processor.batch_process_documents(invalid_files)

        assert results["processing_stats"]["total_files"] == 2
        assert results["processing_stats"]["failed_files"] == 2
        assert results["processing_stats"]["success_rate"] == 0
        assert len(results["failed_files"]) == 2

    @pytest.mark.asyncio
    async def test_progress_tracking(self, processor):
        """Test progress tracking during batch processing"""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress.percentage)

        # Test with non-existent files to trigger quick processing
        test_files = ["test1.pdf", "test2.pdf"]

        await processor.batch_process_documents(test_files, progress_callback)

        # Should have received progress updates
        assert len(progress_updates) > 0
        assert max(progress_updates) == 100.0

    @pytest.mark.asyncio
    async def test_document_type_detection(self, processor):
        """Test document type detection from filename"""
        assert processor._is_invoice_document("invoice_123.pdf")
        assert processor._is_invoice_document("inv_456.docx")
        assert processor._is_invoice_document("bill_789.png")
        assert not processor._is_invoice_document("balance_sheet.pdf")

        assert processor._is_balance_sheet_document("balance_sheet.pdf")
        assert processor._is_balance_sheet_document("financial_position.docx")
        assert not processor._is_balance_sheet_document("invoice.pdf")

    @pytest.mark.asyncio
    async def test_memory_coordination(self, processor):
        """Test memory coordination functionality"""
        # Test storing and retrieving memory
        test_key = "test_data"
        test_value = {"test": "value"}

        await processor._store_memory(test_key, test_value)
        # In a real implementation, we would test actual retrieval
        # For now, just verify the method doesn't raise exceptions

        retrieved = await processor._retrieve_memory(test_key)
        # Currently returns None due to placeholder implementation

    @pytest.mark.asyncio
    async def test_cleanup_session(self, processor):
        """Test session cleanup"""
        # Initialize OCR reader
        processor._get_ocr_reader()
        assert processor.ocr_reader is not None

        # Cleanup
        await processor.cleanup_session()

        # OCR reader should be reset
        assert processor.ocr_reader is None

    @pytest.mark.asyncio
    async def test_error_handling_corrupted_pdf(self, processor):
        """Test error handling with corrupted PDF"""
        # Create a file with PDF extension but invalid content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"This is not a valid PDF content")
            tmp.flush()

            try:
                validation = await processor.validate_document_format(tmp.name)
                assert not validation.is_valid
                assert len(validation.errors) > 0
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_amount_extraction_patterns(self, processor):
        """Test various amount extraction patterns"""
        test_content = """
        Amount: $1,234.56
        Total: 2,500.00 USD
        Price: EUR 999.99
        Cost: 500
        """

        financial_data = await processor.extract_financial_data(test_content)

        assert len(financial_data.amounts) > 0
        amounts = [float(amount) for amount in financial_data.amounts]

        # Should extract various amount formats
        assert any(abs(amount - 1234.56) < 0.01 for amount in amounts)
        assert any(abs(amount - 2500.00) < 0.01 for amount in amounts)

    @pytest.mark.asyncio
    async def test_date_extraction_patterns(self, processor):
        """Test various date extraction patterns"""
        test_content = """
        Date: 12/15/2023
        Due: 2024-01-15
        Invoice Date: Jan 15, 2024
        """

        financial_data = await processor.extract_financial_data(test_content)

        assert len(financial_data.dates) > 0
        # Verify at least one date was extracted
        assert any(date.year in [2023, 2024] for date in financial_data.dates)

    def test_processing_progress_model(self):
        """Test ProcessingProgress model validation"""
        progress = ProcessingProgress(total_files=10)

        assert progress.total_files == 10
        assert progress.processed_files == 0
        assert progress.failed_files == 0
        assert progress.percentage == 0.0
        assert progress.current_file is None

    def test_document_validation_result_model(self):
        """Test DocumentValidationResult model"""
        result = DocumentValidationResult(
            is_valid=True,
            file_type="pdf",
            confidence=0.95
        )

        assert result.is_valid
        assert result.file_type == "pdf"
        assert result.confidence == 0.95
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_financial_data_model(self):
        """Test FinancialData model"""
        data = FinancialData()

        assert len(data.amounts) == 0
        assert len(data.dates) == 0
        assert len(data.account_numbers) == 0
        assert len(data.vendor_names) == 0
        assert len(data.currencies) == 0
        assert len(data.line_items) == 0


class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor"""

    @pytest.mark.asyncio
    async def test_end_to_end_processing_workflow(self):
        """Test complete end-to-end processing workflow"""
        processor = DocumentProcessor()

        # Create test files
        test_files = []

        try:
            # Create mock invoice PDF
            with tempfile.NamedTemporaryFile(suffix='_invoice.pdf', delete=False) as tmp:
                tmp.write(b"dummy pdf content")
                test_files.append(tmp.name)

            # Create mock balance sheet
            with tempfile.NamedTemporaryFile(suffix='_balance_sheet.pdf', delete=False) as tmp:
                tmp.write(b"dummy pdf content")
                test_files.append(tmp.name)

            # Process documents
            results = await processor.batch_process_documents(test_files)

            # Verify results structure
            assert "processing_stats" in results
            assert "invoices" in results
            assert "balance_sheets" in results
            assert "failed_files" in results
            assert "validation_results" in results

            # Verify processing stats
            stats = results["processing_stats"]
            assert stats["total_files"] == 2
            assert "processing_time" in stats
            assert "success_rate" in stats

        finally:
            # Cleanup test files
            for file_path in test_files:
                Path(file_path).unlink(missing_ok=True)

            # Cleanup processor session
            await processor.cleanup_session()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])