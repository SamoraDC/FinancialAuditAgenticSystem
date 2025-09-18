"""
Document processing service for financial audit documents
Supports PDF, CSV, Excel and other financial document formats
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import asyncio
import aiofiles
import pandas as pd
import fitz  # PyMuPDF
from pydantic import BaseModel
from decimal import Decimal
import re
import json

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing financial documents and extracting structured data"""

    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.xlsx', '.xls', '.txt', '.json']
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit

    async def extract_text(self, file_path: str) -> str:
        """Extract text content from document"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")

            if file_path.stat().st_size > self.max_file_size:
                raise ValueError(f"File too large: {file_path.stat().st_size} bytes")

            file_extension = file_path.suffix.lower()

            if file_extension == '.pdf':
                return await self._extract_pdf_text(file_path)
            elif file_extension in ['.csv']:
                return await self._extract_csv_text(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return await self._extract_excel_text(file_path)
            elif file_extension == '.txt':
                return await self._extract_txt_content(file_path)
            elif file_extension == '.json':
                return await self._extract_json_content(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise

    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(file_path)
            text_content = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")

            doc.close()
            return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            raise

    async def _extract_csv_text(self, file_path: Path) -> str:
        """Extract and format CSV data"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any supported encoding")

            # Format as structured text
            result = f"CSV File: {file_path.name}\n"
            result += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            result += "Columns:\n" + ", ".join(df.columns.tolist()) + "\n\n"
            result += "Sample Data (first 10 rows):\n"
            result += df.head(10).to_string(index=False)

            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                result += "\n\nNumeric Column Statistics:\n"
                result += df[numeric_cols].describe().to_string()

            return result

        except Exception as e:
            logger.error(f"CSV extraction failed for {file_path}: {e}")
            raise

    async def _extract_excel_text(self, file_path: Path) -> str:
        """Extract and format Excel data"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            result = f"Excel File: {file_path.name}\n"
            result += f"Sheets: {', '.join(excel_file.sheet_names)}\n\n"

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                result += f"--- Sheet: {sheet_name} ---\n"
                result += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
                result += "Columns: " + ", ".join(df.columns.tolist()) + "\n"
                result += "Sample Data (first 5 rows):\n"
                result += df.head(5).to_string(index=False) + "\n\n"

            return result

        except Exception as e:
            logger.error(f"Excel extraction failed for {file_path}: {e}")
            raise

    async def _extract_txt_content(self, file_path: Path) -> str:
        """Extract plain text content"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return content
        except UnicodeDecodeError:
            # Try with different encoding
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                content = await f.read()
                return content

    async def _extract_json_content(self, file_path: Path) -> str:
        """Extract and format JSON content"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                return json.dumps(data, indent=2, default=str)
        except Exception as e:
            logger.error(f"JSON extraction failed for {file_path}: {e}")
            raise
