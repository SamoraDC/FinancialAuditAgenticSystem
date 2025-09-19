#!/usr/bin/env python3
"""
Financial Audit System - Complete Validation Script
Validates all components and integrations end-to-end
"""

import sys
import os
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SystemValidator:
    """Comprehensive system validation"""

    def __init__(self):
        self.results = {
            'imports': {},
            'database': {},
            'services': {},
            'workflow': {},
            'integration': {},
            'performance': {}
        }
        self.errors = []

    def log_result(self, category: str, test: str, success: bool, message: str = "", error: Exception = None):
        """Log validation result"""
        self.results[category][test] = {
            'success': success,
            'message': message,
            'error': str(error) if error else None
        }

        status = "✅" if success else "❌"
        print(f"{status} {category.upper()}: {test} - {message}")

        if error:
            print(f"   Error: {error}")
            self.errors.append(f"{category}.{test}: {error}")

    def validate_imports(self):
        """Validate all critical imports"""
        print("\n🔍 VALIDATING IMPORTS...")

        imports = [
            ('fastapi', 'FastAPI framework'),
            ('pandas', 'Data processing'),
            ('numpy', 'Numerical computing'),
            ('duckdb', 'Database engine'),
            ('langchain', 'LLM framework'),
            ('langgraph', 'State machine workflows'),
            ('redis', 'State persistence'),
            ('opentelemetry', 'Observability'),
            ('groq', 'LLM provider'),
            ('pytest', 'Testing framework')
        ]

        for module, description in imports:
            try:
                __import__(module)
                self.log_result('imports', module, True, description)
            except ImportError as e:
                self.log_result('imports', module, False, description, e)

    def validate_backend_modules(self):
        """Validate backend module imports"""
        print("\n🏗️ VALIDATING BACKEND MODULES...")

        modules = [
            ('backend.database.duckdb_manager', 'DuckDB Manager'),
            ('backend.services.document_processor', 'Document Processor'),
            ('backend.services.statistical_analyzer', 'Statistical Analyzer'),
            ('backend.services.groq_llm_service', 'Groq LLM Service'),
            ('backend.services.guardrails_service', 'GuardRails Security'),
            ('backend.services.observability_service', 'Observability Service'),
            ('backend.workflows.audit_workflow', 'Audit Workflow'),
            ('backend.main', 'FastAPI Application')
        ]

        for module, description in modules:
            try:
                __import__(module)
                self.log_result('services', module.split('.')[-1], True, description)
            except Exception as e:
                self.log_result('services', module.split('.')[-1], False, description, e)

    async def validate_database(self):
        """Validate DuckDB functionality"""
        print("\n🗄️ VALIDATING DATABASE...")

        try:
            from backend.database.duckdb_manager import DuckDBManager

            # Test database creation and basic operations
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
                db_path = tmp.name

            try:
                db = DuckDBManager(db_path)

                # Test audit session creation
                session_data = {
                    'session_name': 'Validation Test',
                    'client_name': 'Test Client',
                    'auditor_id': 'validator',
                    'audit_type': 'financial',
                    'fiscal_year': 2024,
                    'materiality_threshold': 10000.0,
                    'risk_tolerance': 'medium',
                    'compliance_frameworks': ['GAAP'],
                    'audit_scope': ['financial_statements']
                }

                session_id = db.create_audit_session(session_data)
                self.log_result('database', 'session_creation', True, f"Session ID: {session_id}")

                # Test transaction insertion
                transactions = [{
                    'audit_session_id': session_id,
                    'document_id': None,
                    'transaction_date': '2024-01-01',
                    'description': 'Test Transaction',
                    'amount': 1234.56,
                    'account_code': '1001',
                    'vendor_name': 'Test Vendor',
                    'transaction_type': 'expense',
                    'confidence_score': 0.9
                }]

                db.insert_financial_transactions(transactions)
                self.log_result('database', 'transaction_insertion', True, "Test transaction inserted")

                # Test query functionality
                results = db.get_financial_transactions(session_id)
                if len(results) > 0:
                    self.log_result('database', 'query_functionality', True, f"Retrieved {len(results)} transactions")
                else:
                    self.log_result('database', 'query_functionality', False, "No transactions retrieved")

                db.close()

            finally:
                # Cleanup
                if os.path.exists(db_path):
                    os.unlink(db_path)

        except Exception as e:
            self.log_result('database', 'general', False, "Database validation failed", e)

    async def validate_statistical_analysis(self):
        """Validate statistical analysis services"""
        print("\n📊 VALIDATING STATISTICAL ANALYSIS...")

        try:
            from backend.services.statistical_analyzer import (
                BenfordLawAnalyzer, ZipfLawAnalyzer, StatisticalAnalysisService
            )
            import pandas as pd
            import numpy as np

            # Generate test data
            test_amounts = np.random.exponential(1000, 1000)  # Exponential distribution
            test_vendors = [f"Vendor_{i % 50}" for i in range(1000)]  # 50 unique vendors

            test_data = pd.DataFrame({
                'amount': test_amounts,
                'vendor_name': test_vendors
            })

            # Test Benford's Law Analysis
            benford_analyzer = BenfordLawAnalyzer()
            benford_result = benford_analyzer.analyze_first_digit(test_amounts.tolist())
            self.log_result('services', 'benford_analysis', True,
                          f"P-value: {benford_result.p_value:.4f}, Anomaly: {benford_result.anomaly_detected}")

            # Test Zipf's Law Analysis
            zipf_analyzer = ZipfLawAnalyzer()
            zipf_result = zipf_analyzer.analyze_vendor_payments(test_data)
            self.log_result('services', 'zipf_analysis', True,
                          f"Sample size: {zipf_result.sample_size}, Anomaly: {zipf_result.anomaly_detected}")

            # Test Comprehensive Analysis
            analysis_service = StatisticalAnalysisService()
            comprehensive_results = analysis_service.comprehensive_analysis(test_data, "validation_test")
            self.log_result('services', 'comprehensive_analysis', True,
                          f"Tests performed: {len(comprehensive_results)}")

        except Exception as e:
            self.log_result('services', 'statistical_analysis', False, "Statistical analysis validation failed", e)

    async def validate_document_processing(self):
        """Validate document processing capabilities"""
        print("\n📄 VALIDATING DOCUMENT PROCESSING...")

        try:
            from backend.services.document_processor import DocumentProcessor

            # Create test document content
            test_content = """
            INVOICE
            Invoice Number: INV-001
            Date: 2024-01-15
            Amount: $1,234.56
            Vendor: Test Vendor Inc.
            Description: Professional services
            """

            # Test with a temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write(test_content)
                tmp_path = tmp.name

            try:
                processor = DocumentProcessor()

                # Test document processing
                result = await processor.process_document(tmp_path)

                self.log_result('services', 'document_processing', True,
                              f"Extracted {len(result.text)} characters, Confidence: {result.confidence_score}")

                # Test financial analysis
                analysis = await processor.analyze_financial_document(result)
                self.log_result('services', 'financial_analysis', True,
                              f"Found {len(analysis.get('entities', []))} entities")

            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            self.log_result('services', 'document_processing', False, "Document processing validation failed", e)

    async def validate_workflow(self):
        """Validate LangGraph workflow"""
        print("\n🔄 VALIDATING WORKFLOW...")

        try:
            from backend.workflows.audit_workflow import FinancialAuditWorkflow, AuditState

            # Create test audit state
            test_state = AuditState(
                session_id="validation_test",
                client_name="Test Client",
                auditor_id="validator",
                audit_type="financial",
                fiscal_year=2024,
                uploaded_documents=[],
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

            workflow = FinancialAuditWorkflow()

            # Test workflow initialization
            self.log_result('workflow', 'initialization', True, "Workflow created successfully")

            # Test state machine structure
            graph = workflow.create_workflow()
            self.log_result('workflow', 'graph_creation', True, f"Graph nodes: {len(graph.nodes)}")

        except Exception as e:
            self.log_result('workflow', 'validation', False, "Workflow validation failed", e)

    async def validate_integration(self):
        """Validate end-to-end integration"""
        print("\n🔗 VALIDATING INTEGRATION...")

        try:
            # Test FastAPI app creation
            from backend.main import app
            self.log_result('integration', 'fastapi_app', True, "FastAPI app created successfully")

            # Check if routes are properly configured
            routes = [route.path for route in app.routes]
            expected_routes = ['/health', '/api/v1']

            for route in expected_routes:
                if any(r.startswith(route) for r in routes):
                    self.log_result('integration', f'route_{route.replace("/", "_")}', True, f"Route {route} configured")
                else:
                    self.log_result('integration', f'route_{route.replace("/", "_")}', False, f"Route {route} missing")

        except Exception as e:
            self.log_result('integration', 'general', False, "Integration validation failed", e)

    async def validate_security(self):
        """Validate security components"""
        print("\n🛡️ VALIDATING SECURITY...")

        try:
            from backend.services.guardrails_service import GuardRailsSecurityService

            security_service = GuardRailsSecurityService()

            # Test content validation
            test_content = "This is a test document for financial analysis."
            result = await security_service.validate_content(test_content)

            self.log_result('services', 'security_validation', True,
                          f"Content validation completed, Safe: {result.get('is_safe', False)}")

        except Exception as e:
            self.log_result('services', 'security_validation', False, "Security validation failed", e)

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("🎯 VALIDATION SUMMARY")
        print("="*80)

        total_tests = sum(len(category) for category in self.results.values())
        passed_tests = sum(
            sum(1 for test in category.values() if test['success'])
            for category in self.results.values()
        )

        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        print("\n📋 RESULTS BY CATEGORY:")
        for category, tests in self.results.items():
            if tests:
                category_passed = sum(1 for test in tests.values() if test['success'])
                category_total = len(tests)
                status = "✅" if category_passed == category_total else "⚠️" if category_passed > 0 else "❌"
                print(f"   {status} {category.upper()}: {category_passed}/{category_total}")

        if self.errors:
            print("\n🚨 ERRORS ENCOUNTERED:")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"   • {error}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more errors")

        print("\n" + "="*80)

        if passed_tests == total_tests:
            print("🎉 ALL VALIDATIONS PASSED! System is ready for production.")
        elif passed_tests > total_tests * 0.8:
            print("⚠️ Most validations passed. Some minor issues to address.")
        else:
            print("🚨 Multiple validation failures. System needs attention.")

async def main():
    """Main validation routine"""
    print("🚀 FINANCIAL AUDIT SYSTEM - COMPREHENSIVE VALIDATION")
    print("="*80)

    validator = SystemValidator()

    try:
        # Run all validations
        validator.validate_imports()
        validator.validate_backend_modules()
        await validator.validate_database()
        await validator.validate_statistical_analysis()
        await validator.validate_document_processing()
        await validator.validate_workflow()
        await validator.validate_integration()
        await validator.validate_security()

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during validation: {e}")
        traceback.print_exc()

    finally:
        validator.print_summary()

if __name__ == "__main__":
    asyncio.run(main())