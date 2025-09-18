"""
DuckDB Database Manager for Financial Audit System
High-performance analytical database for financial data processing
"""

import duckdb
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Query result wrapper with metadata"""
    data: pd.DataFrame
    execution_time: float
    row_count: int
    columns: List[str]
    query: str


class DuckDBManager:
    """
    High-performance DuckDB manager for financial audit data
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(Path("data/financial_audit.duckdb"))
        self.connection = None
        self._ensure_db_directory()
        self._initialize_schemas()

    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        try:
            if self.connection is None:
                self.connection = duckdb.connect(self.db_path)

            yield self.connection

        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            # Keep connection open for performance
            pass

    def _initialize_schemas(self):
        """Initialize database schemas for financial audit system"""
        with self.get_connection() as conn:
            # Create main schemas
            schemas = [
                "CREATE SCHEMA IF NOT EXISTS audit;",
                "CREATE SCHEMA IF NOT EXISTS documents;",
                "CREATE SCHEMA IF NOT EXISTS analytics;",
                "CREATE SCHEMA IF NOT EXISTS compliance;",
            ]

            for schema in schemas:
                conn.execute(schema)

            # Create core tables
            self._create_core_tables(conn)

    def _create_core_tables(self, conn):
        """Create core tables for the financial audit system"""

        # Documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents.processed_documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename VARCHAR NOT NULL,
                original_filename VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                document_type VARCHAR NOT NULL,
                file_size BIGINT NOT NULL,
                mime_type VARCHAR,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_status VARCHAR DEFAULT 'pending',
                extraction_method VARCHAR,
                confidence_score FLOAT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Extracted content table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents.extracted_content (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID REFERENCES documents.processed_documents(id),
                content_text TEXT,
                content_structure JSON,
                tables_data JSON,
                images_data JSON,
                financial_entities JSON,
                extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Audit sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit.sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_name VARCHAR NOT NULL,
                client_name VARCHAR NOT NULL,
                auditor_id VARCHAR NOT NULL,
                audit_type VARCHAR DEFAULT 'financial',
                fiscal_year INTEGER NOT NULL,
                status VARCHAR DEFAULT 'active',
                materiality_threshold DECIMAL(20,2),
                risk_tolerance VARCHAR DEFAULT 'medium',
                compliance_frameworks JSON,
                audit_scope JSON,
                progress_percentage FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)

        # Financial transactions table for analysis
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.financial_transactions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                audit_session_id UUID REFERENCES audit.sessions(id),
                document_id UUID REFERENCES documents.processed_documents(id),
                transaction_date DATE,
                description TEXT,
                amount DECIMAL(20,2) NOT NULL,
                account_code VARCHAR,
                account_name VARCHAR,
                transaction_type VARCHAR,
                currency VARCHAR DEFAULT 'USD',
                vendor_name VARCHAR,
                reference_number VARCHAR,
                extracted_from_table BOOLEAN DEFAULT FALSE,
                confidence_score FLOAT,
                anomaly_flags JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Statistical analysis results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics.statistical_analysis (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                audit_session_id UUID REFERENCES audit.sessions(id),
                analysis_type VARCHAR NOT NULL, -- 'benford', 'zipf', 'newcomb_benford'
                dataset_description TEXT,
                sample_size INTEGER,
                test_statistic FLOAT,
                p_value FLOAT,
                chi_square_statistic FLOAT,
                degrees_of_freedom INTEGER,
                anomaly_detected BOOLEAN,
                confidence_level FLOAT,
                analysis_results JSON,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Audit findings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit.findings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                audit_session_id UUID REFERENCES audit.sessions(id),
                finding_type VARCHAR NOT NULL,
                severity VARCHAR NOT NULL, -- 'low', 'medium', 'high', 'critical'
                title VARCHAR NOT NULL,
                description TEXT NOT NULL,
                recommendation TEXT,
                financial_impact DECIMAL(20,2),
                affected_accounts JSON,
                source_documents JSON,
                ai_confidence FLOAT,
                human_reviewed BOOLEAN DEFAULT FALSE,
                status VARCHAR DEFAULT 'open', -- 'open', 'in_progress', 'resolved', 'false_positive'
                assigned_to VARCHAR,
                due_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            );
        """)

        # Compliance checks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compliance.checks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                audit_session_id UUID REFERENCES audit.sessions(id),
                framework VARCHAR NOT NULL, -- 'SOX', 'GAAP', 'IFRS', etc.
                rule_name VARCHAR NOT NULL,
                rule_description TEXT,
                check_result VARCHAR NOT NULL, -- 'pass', 'fail', 'warning', 'not_applicable'
                severity VARCHAR,
                details JSON,
                recommendations JSON,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        logger.info("Database schemas and tables initialized successfully")

    def insert_document(self, document_data: Dict[str, Any]) -> str:
        """Insert processed document record"""
        with self.get_connection() as conn:
            result = conn.execute("""
                INSERT INTO documents.processed_documents
                (filename, original_filename, file_path, document_type, file_size,
                 mime_type, processing_status, extraction_method, confidence_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id;
            """, [
                document_data['filename'],
                document_data['original_filename'],
                document_data['file_path'],
                document_data['document_type'],
                document_data['file_size'],
                document_data.get('mime_type'),
                document_data.get('processing_status', 'completed'),
                document_data.get('extraction_method'),
                document_data.get('confidence_score'),
                json.dumps(document_data.get('metadata', {}))
            ]).fetchone()

            return result[0]

    def insert_extracted_content(self, document_id: str, content_data: Dict[str, Any]) -> str:
        """Insert extracted content for a document"""
        with self.get_connection() as conn:
            result = conn.execute("""
                INSERT INTO documents.extracted_content
                (document_id, content_text, content_structure, tables_data,
                 images_data, financial_entities)
                VALUES (?, ?, ?, ?, ?, ?)
                RETURNING id;
            """, [
                document_id,
                content_data['text'],
                json.dumps(content_data.get('structure', {})),
                json.dumps(content_data.get('tables', [])),
                json.dumps(content_data.get('images', [])),
                json.dumps(content_data.get('financial_entities', {}))
            ]).fetchone()

            return result[0]

    def create_audit_session(self, session_data: Dict[str, Any]) -> str:
        """Create new audit session"""
        with self.get_connection() as conn:
            result = conn.execute("""
                INSERT INTO audit.sessions
                (session_name, client_name, auditor_id, audit_type, fiscal_year,
                 materiality_threshold, risk_tolerance, compliance_frameworks, audit_scope)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id;
            """, [
                session_data['session_name'],
                session_data['client_name'],
                session_data['auditor_id'],
                session_data.get('audit_type', 'financial'),
                session_data['fiscal_year'],
                session_data.get('materiality_threshold'),
                session_data.get('risk_tolerance', 'medium'),
                json.dumps(session_data.get('compliance_frameworks', [])),
                json.dumps(session_data.get('audit_scope', []))
            ]).fetchone()

            return result[0]

    def insert_financial_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """Bulk insert financial transactions"""
        if not transactions:
            return 0

        with self.get_connection() as conn:
            # Prepare data for bulk insert
            insert_data = []
            for tx in transactions:
                insert_data.append([
                    tx.get('audit_session_id'),
                    tx.get('document_id'),
                    tx.get('transaction_date'),
                    tx.get('description'),
                    tx.get('amount'),
                    tx.get('account_code'),
                    tx.get('account_name'),
                    tx.get('transaction_type'),
                    tx.get('currency', 'USD'),
                    tx.get('vendor_name'),
                    tx.get('reference_number'),
                    tx.get('extracted_from_table', False),
                    tx.get('confidence_score'),
                    json.dumps(tx.get('anomaly_flags', {}))
                ])

            conn.executemany("""
                INSERT INTO analytics.financial_transactions
                (audit_session_id, document_id, transaction_date, description, amount,
                 account_code, account_name, transaction_type, currency, vendor_name,
                 reference_number, extracted_from_table, confidence_score, anomaly_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, insert_data)

            return len(insert_data)

    def save_statistical_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Save statistical analysis results"""
        with self.get_connection() as conn:
            result = conn.execute("""
                INSERT INTO analytics.statistical_analysis
                (audit_session_id, analysis_type, dataset_description, sample_size,
                 test_statistic, p_value, chi_square_statistic, degrees_of_freedom,
                 anomaly_detected, confidence_level, analysis_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id;
            """, [
                analysis_data['audit_session_id'],
                analysis_data['analysis_type'],
                analysis_data.get('dataset_description'),
                analysis_data.get('sample_size'),
                analysis_data.get('test_statistic'),
                analysis_data.get('p_value'),
                analysis_data.get('chi_square_statistic'),
                analysis_data.get('degrees_of_freedom'),
                analysis_data.get('anomaly_detected', False),
                analysis_data.get('confidence_level'),
                json.dumps(analysis_data.get('analysis_results', {}))
            ]).fetchone()

            return result[0]

    def query_with_performance(self, query: str, params: Optional[List] = None) -> QueryResult:
        """Execute query with performance metrics"""
        start_time = datetime.now()

        with self.get_connection() as conn:
            if params:
                result = conn.execute(query, params).fetchdf()
            else:
                result = conn.execute(query).fetchdf()

            execution_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                data=result,
                execution_time=execution_time,
                row_count=len(result),
                columns=list(result.columns),
                query=query
            )

    def get_transactions_for_analysis(self, audit_session_id: str,
                                    min_amount: Optional[float] = None) -> pd.DataFrame:
        """Get financial transactions for statistical analysis"""
        query = """
            SELECT
                amount,
                transaction_date,
                description,
                account_code,
                vendor_name,
                transaction_type
            FROM analytics.financial_transactions
            WHERE audit_session_id = ?
        """
        params = [audit_session_id]

        if min_amount:
            query += " AND amount >= ?"
            params.append(min_amount)

        query += " ORDER BY amount DESC"

        result = self.query_with_performance(query, params)
        return result.data

    def get_audit_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive audit session summary"""
        with self.get_connection() as conn:
            # Basic session info
            session_info = conn.execute("""
                SELECT * FROM audit.sessions WHERE id = ?
            """, [session_id]).fetchdf()

            # Document counts
            doc_stats = conn.execute("""
                SELECT
                    COUNT(*) as total_documents,
                    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as processed_docs,
                    AVG(confidence_score) as avg_confidence
                FROM documents.processed_documents pd
                JOIN documents.extracted_content ec ON pd.id = ec.document_id
                WHERE pd.id IN (
                    SELECT DISTINCT document_id
                    FROM analytics.financial_transactions
                    WHERE audit_session_id = ?
                )
            """, [session_id]).fetchdf()

            # Transaction stats
            tx_stats = conn.execute("""
                SELECT
                    COUNT(*) as total_transactions,
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount,
                    MIN(amount) as min_amount,
                    MAX(amount) as max_amount
                FROM analytics.financial_transactions
                WHERE audit_session_id = ?
            """, [session_id]).fetchdf()

            # Findings summary
            findings_stats = conn.execute("""
                SELECT
                    COUNT(*) as total_findings,
                    COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_severity,
                    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_findings,
                    COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_findings
                FROM audit.findings
                WHERE audit_session_id = ?
            """, [session_id]).fetchdf()

            return {
                'session': session_info.to_dict('records')[0] if not session_info.empty else {},
                'documents': doc_stats.to_dict('records')[0] if not doc_stats.empty else {},
                'transactions': tx_stats.to_dict('records')[0] if not tx_stats.empty else {},
                'findings': findings_stats.to_dict('records')[0] if not findings_stats.empty else {}
            }

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None


class AnalyticsQueries:
    """
    Optimized analytical queries for financial audit insights
    """

    @staticmethod
    def benford_analysis_query(audit_session_id: str) -> str:
        """Query to prepare data for Benford's Law analysis"""
        return f"""
            SELECT
                CAST(LEFT(CAST(amount AS VARCHAR), 1) AS INTEGER) as first_digit,
                COUNT(*) as frequency,
                amount
            FROM analytics.financial_transactions
            WHERE audit_session_id = '{audit_session_id}'
                AND amount > 0
                AND amount >= 10  -- Minimum threshold for meaningful analysis
            GROUP BY first_digit, amount
            ORDER BY first_digit;
        """

    @staticmethod
    def amount_distribution_query(audit_session_id: str) -> str:
        """Query for amount distribution analysis"""
        return f"""
            SELECT
                amount,
                transaction_date,
                account_code,
                NTILE(10) OVER (ORDER BY amount) as decile,
                ROW_NUMBER() OVER (ORDER BY amount DESC) as rank
            FROM analytics.financial_transactions
            WHERE audit_session_id = '{audit_session_id}'
                AND amount > 0
            ORDER BY amount DESC;
        """

    @staticmethod
    def vendor_analysis_query(audit_session_id: str) -> str:
        """Query for vendor payment pattern analysis"""
        return f"""
            SELECT
                vendor_name,
                COUNT(*) as transaction_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                STDDEV(amount) as amount_std_dev,
                MIN(transaction_date) as first_transaction,
                MAX(transaction_date) as last_transaction
            FROM analytics.financial_transactions
            WHERE audit_session_id = '{audit_session_id}'
                AND vendor_name IS NOT NULL
                AND vendor_name != ''
            GROUP BY vendor_name
            HAVING COUNT(*) >= 3  -- Minimum transactions for pattern analysis
            ORDER BY total_amount DESC;
        """