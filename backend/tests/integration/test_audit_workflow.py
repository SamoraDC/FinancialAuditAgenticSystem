"""
Integration tests for Financial Audit Workflow
Tests complete end-to-end audit process
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.workflows.audit_workflow import FinancialAuditWorkflow, AuditState
from backend.database.duckdb_manager import DuckDBManager


class TestAuditWorkflowIntegration:
    """Integration tests for complete audit workflow"""

    @pytest.mark.asyncio
    async def test_complete_audit_workflow_success(self, audit_workflow, sample_audit_state,
                                                  mock_langextract, populated_database):
        """Test complete audit workflow execution"""

        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_invoice.pdf"
            test_file.write_text("Sample invoice content with amount $1,234.56")

            # Update state with actual file path
            initial_state = sample_audit_state.copy()
            initial_state['uploaded_documents'][0]['file_path'] = str(test_file)

            # Mock document processor to avoid actual file processing
            with patch('backend.workflows.audit_workflow.DocumentProcessor') as mock_processor:
                mock_extracted = type('ExtractedContent', (), {
                    'text': 'Invoice amount $1,234.56',
                    'metadata': {'extraction_method': 'mock'},
                    'tables': [{'data': [['Description', 'Amount'], ['Test Item', '$1,234.56']]}],
                    'images': [],
                    'structure': {'type': 'pdf'},
                    'confidence_score': 0.9
                })

                mock_processor.return_value.process_document = AsyncMock(return_value=mock_extracted)

                # Execute workflow
                result = await audit_workflow.run_audit(initial_state)

                # Verify workflow completion
                assert result['current_step'] == 'Audit Complete'
                assert result['progress_percentage'] == 100.0
                assert len(result['error_messages']) == 0

                # Verify audit report was generated
                assert result['audit_report'] is not None
                assert 'session_id' in result['audit_report']
                assert 'summary' in result['audit_report']

    @pytest.mark.asyncio
    async def test_audit_workflow_with_anomalies(self, audit_workflow, sample_audit_state):
        """Test audit workflow when statistical anomalies are detected"""

        # Mock statistical analyzer to return anomalies
        with patch('backend.workflows.audit_workflow.StatisticalAnalysisService') as mock_analyzer:
            mock_result = type('StatisticalResult', (), {
                'analysis_type': 'benford',
                'anomaly_detected': True,
                'p_value': 0.001,
                'interpretation': 'Significant deviation from Benford\'s Law',
                'recommendations': ['Investigate transactions starting with digit 1']
            })

            mock_analyzer.return_value.comprehensive_analysis.return_value = {
                'benford': mock_result
            }

            # Execute workflow
            result = await audit_workflow.run_audit(sample_audit_state)

            # Verify anomalies were detected and processed
            assert len(result['anomalies_detected']) > 0
            assert result['human_review_required'] is True

    @pytest.mark.asyncio
    async def test_audit_workflow_error_handling(self, audit_workflow, sample_audit_state):
        """Test audit workflow error handling"""

        # Mock document processor to raise an exception
        with patch('backend.workflows.audit_workflow.DocumentProcessor') as mock_processor:
            mock_processor.return_value.process_document = AsyncMock(
                side_effect=Exception("Document processing failed")
            )

            # Execute workflow
            result = await audit_workflow.run_audit(sample_audit_state)

            # Verify error was captured
            assert len(result['error_messages']) > 0
            assert "Document processing failed" in result['error_messages'][0]

    @pytest.mark.asyncio
    async def test_audit_workflow_resume_functionality(self, audit_workflow, sample_audit_state):
        """Test audit workflow resume after human review"""

        # First, run workflow to a point requiring human review
        initial_state = sample_audit_state.copy()
        initial_state['human_review_required'] = True
        initial_state['current_step'] = 'Awaiting Human Review'

        # Mock workflow to simulate paused state
        thread_id = "test_thread_123"

        # Simulate human input
        human_input = {
            'human_review_completed': True,
            'human_decisions': [
                {
                    'finding_id': 'finding_001',
                    'approved': True,
                    'comments': 'Reviewed and approved'
                }
            ]
        }

        # Resume workflow with human input
        result = await audit_workflow.resume_audit(thread_id, human_input)

        # Verify workflow continued after human review
        assert 'human_review_completed' in str(result)

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, audit_workflow, sample_audit_state, mock_redis):
        """Test that workflow state is properly persisted to Redis"""

        # Mock Redis checkpoint saver
        with patch('backend.workflows.audit_workflow.RedisSaver') as mock_saver:
            mock_saver.return_value = mock_redis

            thread_id = "test_persistence_123"

            # Execute workflow
            result = await audit_workflow.run_audit(sample_audit_state, thread_id)

            # Verify Redis saver was used
            mock_saver.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_document_processing(self, audit_workflow):
        """Test workflow with multiple documents processed in parallel"""

        # Create state with multiple documents
        multi_doc_state = AuditState(
            session_id="multi_doc_test",
            client_name="Test Client",
            auditor_id="test_auditor",
            audit_type="financial",
            fiscal_year=2024,
            uploaded_documents=[
                {
                    'filename': f'doc_{i}.pdf',
                    'original_filename': f'doc_{i}.pdf',
                    'file_path': f'/tmp/doc_{i}.pdf',
                    'document_type': 'invoice',
                    'file_size': 1024
                }
                for i in range(5)  # 5 documents
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

        # Mock document processor for parallel processing
        with patch('backend.workflows.audit_workflow.DocumentProcessor') as mock_processor:
            mock_extracted = type('ExtractedContent', (), {
                'text': 'Sample content',
                'metadata': {'extraction_method': 'mock'},
                'tables': [],
                'images': [],
                'structure': {'type': 'pdf'},
                'confidence_score': 0.8
            })

            mock_processor.return_value.process_document = AsyncMock(return_value=mock_extracted)

            # Execute workflow
            result = await audit_workflow.run_audit(multi_doc_state)

            # Verify all documents were processed
            assert len(result['processed_documents']) == 5

            # Verify parallel processing efficiency (should be faster than sequential)
            assert mock_processor.return_value.process_document.call_count == 5


class TestWorkflowNodeIntegration:
    """Integration tests for individual workflow nodes"""

    @pytest.mark.asyncio
    async def test_initialize_session_node(self, audit_workflow, sample_audit_state, db_manager):
        """Test session initialization node"""

        # Replace workflow's db_manager with test instance
        audit_workflow.db_manager = db_manager

        result = await audit_workflow._initialize_session(sample_audit_state)

        # Verify session was created in database
        assert result['session_id'] is not None
        assert result['current_step'] == 'Session Initialized'
        assert result['progress_percentage'] == 10.0

    @pytest.mark.asyncio
    async def test_document_processing_node(self, audit_workflow, sample_audit_state, mock_langextract):
        """Test document processing node"""

        # Prepare state with session ID
        state = sample_audit_state.copy()
        state['session_id'] = 'test_session_123'

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content")
            state['uploaded_documents'][0]['file_path'] = f.name

        try:
            result = await audit_workflow._process_documents(state)

            # Verify documents were processed
            assert len(result['processed_documents']) > 0
            assert result['current_step'] == 'Documents Processed'
            assert result['progress_percentage'] == 25.0

        finally:
            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_statistical_analysis_node(self, audit_workflow, sample_audit_state, sample_financial_data):
        """Test statistical analysis node"""

        # Prepare state with financial transactions
        state = sample_audit_state.copy()
        state['session_id'] = 'test_session_123'
        state['financial_transactions'] = sample_financial_data.to_dict('records')

        result = await audit_workflow._statistical_analysis(state)

        # Verify statistical analysis was performed
        assert 'statistical_results' in result
        assert result['current_step'] == 'Statistical Analysis Complete'
        assert result['progress_percentage'] == 60.0

    @pytest.mark.asyncio
    async def test_ai_analysis_node(self, audit_workflow, sample_audit_state):
        """Test AI document analysis node"""

        # Prepare state with processed documents
        state = sample_audit_state.copy()
        state['processed_documents'] = [
            {
                'document_id': 'doc_001',
                'extracted_content': type('ExtractedContent', (), {
                    'text': 'Sample document text for AI analysis',
                    'metadata': {},
                    'tables': [],
                    'images': [],
                    'structure': {},
                    'confidence_score': 0.9
                }),
                'original_info': {'document_type': 'invoice'}
            }
        ]

        result = await audit_workflow._ai_document_analysis(state)

        # Verify AI analysis was performed
        assert len(result['llm_analyses']) > 0
        assert result['current_step'] == 'AI Analysis Complete'
        assert result['progress_percentage'] == 75.0

    @pytest.mark.asyncio
    async def test_findings_generation_node(self, audit_workflow, sample_audit_state):
        """Test audit findings generation node"""

        # Prepare state with anomalies
        state = sample_audit_state.copy()
        state['anomalies_detected'] = [
            {
                'type': 'benford',
                'severity': 'high',
                'description': 'Significant deviation from Benford\'s Law'
            }
        ]
        state['llm_analyses'] = [
            type('LLMResponse', (), {
                'content': 'Risk analysis indicates potential issues',
                'confidence_score': 0.8
            })
        ]

        result = await audit_workflow._generate_findings(state)

        # Verify findings were generated
        assert len(result['audit_findings']) > 0
        assert result['current_step'] == 'Findings Generated'
        assert result['progress_percentage'] == 85.0

    @pytest.mark.asyncio
    async def test_compliance_check_node(self, audit_workflow, sample_audit_state):
        """Test compliance checking node"""

        # Prepare state with audit findings
        state = sample_audit_state.copy()
        state['audit_findings'] = [
            {
                'id': 'finding_001',
                'type': 'statistical_anomaly',
                'severity': 'medium',
                'description': 'Test finding'
            }
        ]
        state['compliance_frameworks'] = ['GAAP', 'SOX']

        result = await audit_workflow._compliance_check(state)

        # Verify compliance check was performed
        assert 'compliance_status' in result
        assert result['current_step'] == 'Compliance Check Complete'
        assert result['progress_percentage'] == 90.0

    @pytest.mark.asyncio
    async def test_report_generation_node(self, audit_workflow, sample_audit_state):
        """Test audit report generation node"""

        # Prepare state with complete audit data
        state = sample_audit_state.copy()
        state['processed_documents'] = [{'doc': 'test'}]
        state['financial_transactions'] = [{'tx': 'test'}]
        state['audit_findings'] = [{'finding': 'test'}]
        state['compliance_status'] = {'GAAP': {'assessment': 'compliant'}}

        result = await audit_workflow._generate_report(state)

        # Verify report was generated
        assert result['audit_report'] is not None
        assert 'summary' in result['audit_report']
        assert result['current_step'] == 'Report Generated'
        assert result['progress_percentage'] == 95.0

    def test_should_require_human_review_logic(self, audit_workflow):
        """Test human review requirement logic"""

        # Case 1: No human review required
        state_no_review = {'human_review_required': False}
        result = audit_workflow._should_require_human_review(state_no_review)
        assert result == "generate_report"

        # Case 2: Human review required
        state_with_review = {'human_review_required': True}
        result = audit_workflow._should_require_human_review(state_with_review)
        assert result == "human_review"


class TestWorkflowErrorRecovery:
    """Test workflow error recovery and resilience"""

    @pytest.mark.asyncio
    async def test_workflow_recovery_from_document_processing_error(self, audit_workflow, sample_audit_state):
        """Test workflow continues despite document processing errors"""

        # Mock document processor to fail on first document but succeed on others
        with patch('backend.workflows.audit_workflow.DocumentProcessor') as mock_processor:
            call_count = 0

            async def mock_process_document(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("First document failed")

                return type('ExtractedContent', (), {
                    'text': 'Sample content',
                    'metadata': {'extraction_method': 'mock'},
                    'tables': [],
                    'images': [],
                    'structure': {'type': 'pdf'},
                    'confidence_score': 0.8
                })

            mock_processor.return_value.process_document = mock_process_document

            # Add multiple documents
            state = sample_audit_state.copy()
            state['uploaded_documents'] = [
                {'filename': 'doc1.pdf', 'file_path': '/tmp/doc1.pdf', 'document_type': 'invoice', 'file_size': 1024},
                {'filename': 'doc2.pdf', 'file_path': '/tmp/doc2.pdf', 'document_type': 'invoice', 'file_size': 1024}
            ]

            result = await audit_workflow.run_audit(state)

            # Verify workflow completed despite one document failing
            assert result['current_step'] == 'Audit Complete'
            assert len(result['error_messages']) > 0  # Error was recorded
            assert len(result['processed_documents']) >= 1  # At least one document processed

    @pytest.mark.asyncio
    async def test_workflow_graceful_degradation(self, audit_workflow, sample_audit_state):
        """Test workflow graceful degradation when optional components fail"""

        # Mock statistical analyzer to fail
        with patch('backend.workflows.audit_workflow.StatisticalAnalysisService') as mock_analyzer:
            mock_analyzer.return_value.comprehensive_analysis.side_effect = Exception("Statistical analysis failed")

            result = await audit_workflow.run_audit(sample_audit_state)

            # Verify workflow continued despite statistical analysis failure
            assert result['current_step'] == 'Audit Complete'
            assert len(result['error_messages']) > 0

            # Should still generate some form of report
            assert result['audit_report'] is not None

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, audit_workflow, sample_audit_state):
        """Test workflow handling of timeouts"""

        # Mock LLM service to simulate timeout
        with patch('backend.workflows.audit_workflow.GroqLLMService') as mock_llm:
            mock_llm.return_value.analyze_financial_document = AsyncMock(
                side_effect=asyncio.TimeoutError("LLM request timed out")
            )

            result = await audit_workflow.run_audit(sample_audit_state)

            # Verify workflow handled timeout gracefully
            assert result['current_step'] == 'Audit Complete'
            assert any("timed out" in msg.lower() for msg in result['error_messages'])


class TestWorkflowPerformance:
    """Performance tests for audit workflow"""

    @pytest.mark.asyncio
    async def test_workflow_performance_large_dataset(self, audit_workflow, performance_test_data):
        """Test workflow performance with large dataset"""

        # Create state with large transaction dataset
        large_state = AuditState(
            session_id="perf_test_001",
            client_name="Performance Test Client",
            auditor_id="perf_auditor",
            audit_type="financial",
            fiscal_year=2024,
            uploaded_documents=[],
            processed_documents=[],
            extracted_content=[],
            financial_transactions=performance_test_data.to_dict('records'),
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

        import time
        start_time = time.time()

        # Run statistical analysis node only (most computationally intensive)
        result = await audit_workflow._statistical_analysis(large_state)

        end_time = time.time()

        # Verify performance is acceptable (should complete within 10 seconds for 10k records)
        assert end_time - start_time < 10.0
        assert result['current_step'] == 'Statistical Analysis Complete'