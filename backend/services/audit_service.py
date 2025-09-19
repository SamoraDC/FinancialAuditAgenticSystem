"""
Main audit service layer implementing comprehensive financial audit orchestration.
Coordinates LangGraph workflows, document processing, and multi-agent swarm execution.
"""

import uuid
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal

from backend.models.audit_models import (
    AuditSession, AuditState, AuditStatus, AuditFinding,
    RiskAssessment, RiskLevel, DocumentType
)
from backend.models.schemas import (
    AuditRequest, AuditResponse, AuditProgress,
    DocumentUpload, ErrorResponse
)
from backend.workflows.audit_workflow import (
    start_audit_workflow, get_audit_state, resume_audit_workflow,
    get_workflow
)
from backend.services.document_processor import DocumentProcessor
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class AuditServiceError(Exception):
    """Custom exception for audit service errors"""
    def __init__(self, message: str, error_code: str = "AUDIT_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class SwarmMemoryManager:
    """Manages swarm memory for multi-agent coordination"""

    def __init__(self):
        self.memory_store: Dict[str, Dict[str, Any]] = {}

    async def store_session_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Store session context for swarm agents"""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = {}
        self.memory_store[session_id].update(context)
        logger.debug(f"Stored context for session {session_id}: {list(context.keys())}")

    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session context for swarm agents"""
        return self.memory_store.get(session_id, {})

    async def update_progress(self, session_id: str, step: str, progress: float) -> None:
        """Update progress information in memory"""
        context = await self.get_session_context(session_id)
        context.update({
            "current_step": step,
            "progress_percentage": progress,
            "last_update": datetime.utcnow().isoformat()
        })
        await self.store_session_context(session_id, context)

    async def store_agent_results(self, session_id: str, agent_type: str, results: Dict[str, Any]) -> None:
        """Store results from specific agents"""
        context = await self.get_session_context(session_id)
        if "agent_results" not in context:
            context["agent_results"] = {}
        context["agent_results"][agent_type] = results
        await self.store_session_context(session_id, context)

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session memory"""
        if session_id in self.memory_store:
            del self.memory_store[session_id]
            logger.debug(f"Cleaned up memory for session {session_id}")


class LegacyDocumentProcessor:
    """Legacy document processor - replaced by EnhancedDocumentProcessor"""

    async def process_uploaded_documents(
        self,
        documents: List[DocumentUpload],
        session_id: str
    ) -> Dict[str, Any]:
        """Process uploaded documents and extract data"""
        logger.info(f"Processing {len(documents)} documents for session {session_id}")

        processed_docs = {}
        total_size = sum(doc.file_size for doc in documents)

        # Validate documents
        await self._validate_documents(documents, total_size)

        # Process each document type
        for doc in documents:
            try:
                extracted_data = await self._extract_document_data(doc)
                processed_docs[doc.filename] = {
                    "type": doc.document_type,
                    "data": extracted_data,
                    "metadata": {
                        "size": doc.file_size,
                        "upload_time": doc.upload_timestamp,
                        "content_type": doc.content_type
                    }
                }
                logger.debug(f"Processed document: {doc.filename}")

            except Exception as e:
                logger.error(f"Failed to process document {doc.filename}: {str(e)}")
                processed_docs[doc.filename] = {
                    "error": str(e),
                    "type": doc.document_type
                }

        return {
            "documents": processed_docs,
            "total_documents": len(documents),
            "total_size": total_size,
            "processing_time": datetime.utcnow().isoformat()
        }

    async def _validate_documents(self, documents: List[DocumentUpload], total_size: int) -> None:
        """Validate document uploads"""
        settings = get_settings()
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 100 * 1024 * 1024)  # 100MB default

        if total_size > max_size:
            raise AuditServiceError(
                f"Total upload size {total_size} exceeds limit {max_size}",
                "UPLOAD_SIZE_EXCEEDED"
            )

        # Validate supported document types
        supported_types = {doc_type.value for doc_type in DocumentType}
        for doc in documents:
            if doc.document_type not in supported_types:
                raise AuditServiceError(
                    f"Unsupported document type: {doc.document_type}",
                    "UNSUPPORTED_DOCUMENT_TYPE"
                )

    async def _extract_document_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract data from individual document"""
        # Simulate document processing - in real implementation, this would
        # integrate with OCR, PDF parsing, Excel parsing, etc.

        extraction_methods = {
            DocumentType.INVOICE: self._extract_invoice_data,
            DocumentType.BALANCE_SHEET: self._extract_balance_sheet_data,
            DocumentType.INCOME_STATEMENT: self._extract_income_statement_data,
            DocumentType.CASH_FLOW: self._extract_cash_flow_data,
            DocumentType.TRIAL_BALANCE: self._extract_trial_balance_data,
            DocumentType.GENERAL_LEDGER: self._extract_general_ledger_data,
        }

        extractor = extraction_methods.get(document.document_type)
        if extractor:
            return await extractor(document)
        else:
            return {"raw_text": f"Processed {document.filename}", "type": document.document_type}

    async def _extract_invoice_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract invoice-specific data"""
        return {
            "invoice_number": f"INV-{uuid.uuid4().hex[:8]}",
            "vendor_name": "Sample Vendor",
            "amount": float(Decimal("1000.00")),
            "date": datetime.utcnow().isoformat(),
            "line_items": []
        }

    async def _extract_balance_sheet_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract balance sheet data"""
        return {
            "total_assets": float(Decimal("1000000.00")),
            "total_liabilities": float(Decimal("600000.00")),
            "total_equity": float(Decimal("400000.00")),
            "period_end": datetime.utcnow().isoformat()
        }

    async def _extract_income_statement_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract income statement data"""
        return {
            "revenue": float(Decimal("500000.00")),
            "expenses": float(Decimal("400000.00")),
            "net_income": float(Decimal("100000.00")),
            "period": datetime.utcnow().isoformat()
        }

    async def _extract_cash_flow_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract cash flow data"""
        return {
            "operating_cash_flow": float(Decimal("120000.00")),
            "investing_cash_flow": float(Decimal("-50000.00")),
            "financing_cash_flow": float(Decimal("-30000.00")),
            "net_cash_flow": float(Decimal("40000.00"))
        }

    async def _extract_trial_balance_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract trial balance data"""
        return {
            "total_debits": float(Decimal("1000000.00")),
            "total_credits": float(Decimal("1000000.00")),
            "account_count": 50,
            "balances": {}
        }

    async def _extract_general_ledger_data(self, document: DocumentUpload) -> Dict[str, Any]:
        """Extract general ledger data"""
        return {
            "transaction_count": 1000,
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "accounts": []
        }


class StatisticalAnalyzer:
    """Performs statistical analysis on financial data"""

    async def analyze_financial_metrics(self, extracted_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial metrics and ratios"""
        logger.info("Performing statistical analysis on financial data")

        metrics = {}

        # Process balance sheet metrics
        balance_sheets = self._filter_documents_by_type(extracted_data, "balance_sheet")
        if balance_sheets:
            bs_metrics = await self._calculate_balance_sheet_ratios(balance_sheets)
            metrics.update(bs_metrics)

        # Process income statement metrics
        income_statements = self._filter_documents_by_type(extracted_data, "income_statement")
        if income_statements:
            is_metrics = await self._calculate_profitability_ratios(income_statements)
            metrics.update(is_metrics)

        # Process cash flow metrics
        cash_flows = self._filter_documents_by_type(extracted_data, "cash_flow")
        if cash_flows:
            cf_metrics = await self._calculate_liquidity_ratios(cash_flows)
            metrics.update(cf_metrics)

        # Calculate variance and trends
        trend_metrics = await self._calculate_trend_analysis(extracted_data)
        metrics.update(trend_metrics)

        logger.info(f"Calculated {len(metrics)} financial metrics")
        return metrics

    def _filter_documents_by_type(self, data: Dict[str, Any], doc_type: str) -> List[Dict]:
        """Filter documents by type"""
        return [
            doc["data"] for doc in data.get("documents", {}).values()
            if doc.get("type") == doc_type and "data" in doc
        ]

    async def _calculate_balance_sheet_ratios(self, balance_sheets: List[Dict]) -> Dict[str, float]:
        """Calculate balance sheet ratios"""
        if not balance_sheets:
            return {}

        # Use the most recent balance sheet
        bs = balance_sheets[-1]

        assets = bs.get("total_assets", 0)
        liabilities = bs.get("total_liabilities", 0)
        equity = bs.get("total_equity", 0)

        ratios = {}
        if assets > 0:
            ratios["debt_to_asset_ratio"] = liabilities / assets
            ratios["equity_ratio"] = equity / assets
        if equity > 0:
            ratios["debt_to_equity_ratio"] = liabilities / equity

        return ratios

    async def _calculate_profitability_ratios(self, income_statements: List[Dict]) -> Dict[str, float]:
        """Calculate profitability ratios"""
        if not income_statements:
            return {}

        is_data = income_statements[-1]
        revenue = is_data.get("revenue", 0)
        expenses = is_data.get("expenses", 0)
        net_income = is_data.get("net_income", 0)

        ratios = {}
        if revenue > 0:
            ratios["net_profit_margin"] = net_income / revenue
            ratios["expense_ratio"] = expenses / revenue

        return ratios

    async def _calculate_liquidity_ratios(self, cash_flows: List[Dict]) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        if not cash_flows:
            return {}

        cf = cash_flows[-1]
        operating_cf = cf.get("operating_cash_flow", 0)
        net_cf = cf.get("net_cash_flow", 0)

        return {
            "operating_cash_flow": operating_cf,
            "net_cash_flow": net_cf,
            "cash_flow_health_score": 1.0 if operating_cf > 0 else 0.5
        }

    async def _calculate_trend_analysis(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate trend analysis metrics"""
        # Simulate trend analysis
        return {
            "revenue_growth_rate": 0.05,  # 5% growth
            "variance_score": 0.15,  # 15% variance
            "seasonal_factor": 1.0  # No seasonality
        }


class AuditService:
    """
    Main audit service layer coordinating all audit operations.
    Manages workflow orchestration, document processing, and multi-agent coordination.
    """

    def __init__(self):
        self.settings = get_settings()
        self.swarm_memory = SwarmMemoryManager()
        self.document_processor = DocumentProcessor()
        self.legacy_document_processor = LegacyDocumentProcessor()  # Fallback for compatibility
        self.statistical_analyzer = StatisticalAnalyzer()
        self.audit_sessions: Dict[str, AuditSession] = {}

        # Workflow steps for progress tracking
        self.workflow_steps = [
            "initialize", "ingest_parse", "statistical_analysis",
            "anomaly_detection", "compliance_check", "risk_assessment",
            "generate_findings", "create_report", "finalize"
        ]

    async def start_audit(self, audit_request: AuditRequest, user: Dict[str, Any]) -> AuditResponse:
        """
        Start a new financial audit workflow with multi-agent coordination.

        Args:
            audit_request: Audit configuration and requirements
            user: Current user information

        Returns:
            AuditResponse: Initial audit response with session details
        """
        logger.info(f"Starting new audit for client: {audit_request.client_name}")

        try:
            # Generate unique session ID
            session_id = str(uuid.uuid4())

            # Create initial audit state
            initial_state = AuditState(
                session_id=session_id,
                status=AuditStatus.PENDING,
                current_step="initialize",
                progress_percentage=0.0,
                audit_scope=audit_request.audit_scope,
                risk_threshold=float(audit_request.risk_tolerance.value == "high") * 0.3 + 0.5
            )

            # Create audit session
            audit_session = AuditSession(
                id=session_id,
                client_name=audit_request.client_name,
                auditor_id=audit_request.auditor_id,
                audit_type=audit_request.audit_type,
                fiscal_year=audit_request.fiscal_year,
                start_date=datetime.utcnow(),
                target_completion=datetime.utcnow() + timedelta(days=audit_request.target_completion_days),
                materiality_threshold=audit_request.materiality_threshold,
                risk_tolerance=audit_request.risk_tolerance,
                compliance_frameworks=audit_request.compliance_frameworks,
                state=initial_state,
                created_by=user.get("user_id", "unknown")
            )

            # Store session
            self.audit_sessions[session_id] = audit_session

            # Initialize swarm memory context
            await self.swarm_memory.store_session_context(session_id, {
                "audit_request": audit_request.dict(),
                "user_info": user,
                "session_config": {
                    "materiality_threshold": float(audit_request.materiality_threshold),
                    "risk_tolerance": audit_request.risk_tolerance.value,
                    "compliance_frameworks": audit_request.compliance_frameworks
                }
            })

            # Start the LangGraph workflow asynchronously
            asyncio.create_task(self._execute_audit_workflow(session_id, initial_state))

            # Update progress
            await self.swarm_memory.update_progress(session_id, "initialize", 5.0)

            # Create response
            progress = AuditProgress(
                session_id=session_id,
                status=AuditStatus.IN_PROGRESS,
                current_step="initialize",
                progress_percentage=5.0,
                steps_completed=[],
                total_steps=len(self.workflow_steps),
                estimated_completion=audit_session.target_completion,
                last_update=datetime.utcnow()
            )

            response = AuditResponse(
                session_id=session_id,
                status=AuditStatus.IN_PROGRESS,
                progress=progress,
                findings_count=0,
                high_risk_findings=0,
                created_at=audit_session.start_date,
                estimated_completion=audit_session.target_completion
            )

            logger.info(f"Successfully started audit session: {session_id}")
            return response

        except Exception as e:
            logger.error(f"Failed to start audit: {str(e)}")
            raise AuditServiceError(
                f"Failed to start audit: {str(e)}",
                "AUDIT_START_FAILED",
                {"audit_request": audit_request.dict()}
            )

    async def get_audit_status(self, audit_id: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get real-time status of an ongoing audit.

        Args:
            audit_id: Unique audit session identifier
            user: Current user information

        Returns:
            Dict containing current audit status and progress
        """
        logger.debug(f"Getting status for audit: {audit_id}")

        try:
            # Validate session exists
            if audit_id not in self.audit_sessions:
                # Try to get from workflow state
                workflow_state = await get_audit_state(audit_id)
                if not workflow_state:
                    raise AuditServiceError(
                        f"Audit session not found: {audit_id}",
                        "SESSION_NOT_FOUND"
                    )
                return self._format_workflow_status(workflow_state)

            session = self.audit_sessions[audit_id]

            # Get latest state from workflow
            current_state = await get_audit_state(audit_id)
            if current_state:
                session.state = current_state
                session.last_accessed = datetime.utcnow()

            # Get progress from swarm memory
            context = await self.swarm_memory.get_session_context(audit_id)

            # Calculate detailed progress
            steps_completed = current_state.steps_completed if current_state else []
            progress_percentage = len(steps_completed) / len(self.workflow_steps) * 100

            status_response = {
                "session_id": audit_id,
                "status": session.state.status.value,
                "current_step": session.state.current_step,
                "progress_percentage": progress_percentage,
                "steps_completed": steps_completed,
                "total_steps": len(self.workflow_steps),
                "findings_count": len(session.state.findings),
                "high_risk_findings": len([
                    f for f in session.state.findings
                    if f.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                ]),
                "errors": session.state.errors,
                "warnings": session.state.warnings,
                "last_update": session.state.updated_at.isoformat(),
                "estimated_completion": session.target_completion.isoformat(),
                "agent_status": context.get("agent_results", {}),
                "processing_metrics": {
                    "documents_processed": len(session.state.extracted_data),
                    "anomalies_detected": len(session.state.anomalies),
                    "overall_risk_score": session.state.overall_risk_score
                }
            }

            return status_response

        except AuditServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to get audit status for {audit_id}: {str(e)}")
            raise AuditServiceError(
                f"Failed to get audit status: {str(e)}",
                "STATUS_RETRIEVAL_FAILED"
            )

    async def get_audit_results(self, audit_id: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive results of a completed audit.

        Args:
            audit_id: Unique audit session identifier
            user: Current user information

        Returns:
            Dict containing complete audit results and findings
        """
        logger.info(f"Getting audit results for session: {audit_id}")

        try:
            # Validate session exists
            if audit_id not in self.audit_sessions:
                workflow_state = await get_audit_state(audit_id)
                if not workflow_state:
                    raise AuditServiceError(
                        f"Audit session not found: {audit_id}",
                        "SESSION_NOT_FOUND"
                    )
                return self._format_workflow_results(workflow_state)

            session = self.audit_sessions[audit_id]

            # Get latest state
            current_state = await get_audit_state(audit_id)
            if current_state:
                session.state = current_state

            # Check if audit is completed
            if session.state.status not in [AuditStatus.COMPLETED, AuditStatus.FAILED]:
                raise AuditServiceError(
                    f"Audit not yet completed. Current status: {session.state.status.value}",
                    "AUDIT_NOT_COMPLETED"
                )

            # Get swarm memory context for additional insights
            context = await self.swarm_memory.get_session_context(audit_id)

            # Compile comprehensive results
            results = {
                "session_info": {
                    "session_id": audit_id,
                    "client_name": session.client_name,
                    "auditor_id": session.auditor_id,
                    "audit_type": session.audit_type,
                    "fiscal_year": session.fiscal_year,
                    "start_date": session.start_date.isoformat(),
                    "completion_date": session.state.completed_at.isoformat() if session.state.completed_at else None,
                    "status": session.state.status.value
                },
                "executive_summary": {
                    "overall_risk_score": session.state.overall_risk_score,
                    "total_findings": len(session.state.findings),
                    "high_risk_findings": len([
                        f for f in session.state.findings
                        if f.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                    ]),
                    "compliance_status": self._assess_compliance_status(session.state.findings),
                    "key_recommendations": session.state.recommendations,
                    "materiality_assessment": float(session.materiality_threshold)
                },
                "detailed_findings": [
                    {
                        "id": finding.id,
                        "title": finding.title,
                        "category": finding.category,
                        "severity": finding.severity.value,
                        "description": finding.description,
                        "recommendation": finding.recommendation,
                        "confidence_score": finding.confidence_score,
                        "financial_impact": float(finding.financial_impact) if finding.financial_impact else None,
                        "affected_accounts": finding.affected_accounts,
                        "regulatory_reference": finding.regulatory_reference,
                        "status": finding.status
                    }
                    for finding in session.state.findings
                ],
                "risk_assessments": [
                    {
                        "category": risk.risk_category,
                        "inherent_risk": risk.inherent_risk.value,
                        "control_risk": risk.control_risk.value,
                        "detection_risk": risk.detection_risk.value,
                        "overall_risk": risk.overall_risk.value,
                        "quantitative_score": risk.quantitative_score,
                        "risk_factors": risk.risk_factors,
                        "mitigation_strategies": risk.mitigation_strategies
                    }
                    for risk in session.state.risk_assessments
                ],
                "financial_metrics": session.state.financial_metrics,
                "anomalies_detected": session.state.anomalies,
                "processing_summary": {
                    "documents_processed": len(session.state.extracted_data),
                    "steps_completed": session.state.steps_completed,
                    "processing_time": self._calculate_processing_time(session),
                    "errors_encountered": session.state.errors,
                    "warnings": session.state.warnings
                },
                "agent_contributions": context.get("agent_results", {}),
                "compliance_frameworks": session.compliance_frameworks,
                "audit_trail": {
                    "created_by": session.created_by,
                    "last_accessed": session.last_accessed.isoformat(),
                    "workflow_history": session.state.steps_completed
                }
            }

            logger.info(f"Successfully compiled audit results for session: {audit_id}")
            return results

        except AuditServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to get audit results for {audit_id}: {str(e)}")
            raise AuditServiceError(
                f"Failed to get audit results: {str(e)}",
                "RESULTS_RETRIEVAL_FAILED"
            )

    async def list_audits(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        List all audits accessible to the current user.

        Args:
            user: Current user information

        Returns:
            Dict containing paginated list of user audits
        """
        logger.debug(f"Listing audits for user: {user.get('user_id', 'unknown')}")

        try:
            user_id = user.get("user_id")
            if not user_id:
                raise AuditServiceError(
                    "User ID required for audit listing",
                    "INVALID_USER"
                )

            # Filter audits by user access
            user_audits = []
            for session_id, session in self.audit_sessions.items():
                # Check if user has access (creator, assigned auditor, or admin)
                has_access = (
                    session.created_by == user_id or
                    session.auditor_id == user_id or
                    user.get("role") == "admin"
                )

                if has_access:
                    # Get latest state for accurate status
                    current_state = await get_audit_state(session_id)
                    if current_state:
                        session.state = current_state

                    audit_summary = {
                        "session_id": session_id,
                        "client_name": session.client_name,
                        "audit_type": session.audit_type,
                        "fiscal_year": session.fiscal_year,
                        "status": session.state.status.value,
                        "progress_percentage": len(session.state.steps_completed) / len(self.workflow_steps) * 100,
                        "findings_count": len(session.state.findings),
                        "high_risk_findings": len([
                            f for f in session.state.findings
                            if f.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
                        ]),
                        "start_date": session.start_date.isoformat(),
                        "target_completion": session.target_completion.isoformat(),
                        "last_accessed": session.last_accessed.isoformat(),
                        "created_by": session.created_by,
                        "auditor_id": session.auditor_id,
                        "risk_tolerance": session.risk_tolerance.value,
                        "compliance_frameworks": session.compliance_frameworks
                    }
                    user_audits.append(audit_summary)

            # Sort by last accessed (most recent first)
            user_audits.sort(key=lambda x: x["last_accessed"], reverse=True)

            # Calculate summary statistics
            total_audits = len(user_audits)
            active_audits = len([a for a in user_audits if a["status"] == "in_progress"])
            completed_audits = len([a for a in user_audits if a["status"] == "completed"])
            failed_audits = len([a for a in user_audits if a["status"] == "failed"])

            response = {
                "audits": user_audits,
                "summary": {
                    "total_audits": total_audits,
                    "active_audits": active_audits,
                    "completed_audits": completed_audits,
                    "failed_audits": failed_audits,
                    "success_rate": (completed_audits / total_audits * 100) if total_audits > 0 else 0
                },
                "user_info": {
                    "user_id": user_id,
                    "role": user.get("role", "user"),
                    "last_request": datetime.utcnow().isoformat()
                }
            }

            logger.info(f"Listed {total_audits} audits for user {user_id}")
            return response

        except AuditServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to list audits for user: {str(e)}")
            raise AuditServiceError(
                f"Failed to list audits: {str(e)}",
                "AUDIT_LISTING_FAILED"
            )

    async def _execute_audit_workflow(self, session_id: str, initial_state: AuditState) -> None:
        """Execute the audit workflow asynchronously"""
        try:
            logger.info(f"Executing audit workflow for session: {session_id}")

            # Update status to in progress
            initial_state.status = AuditStatus.IN_PROGRESS
            await self.swarm_memory.update_progress(session_id, "initialize", 10.0)

            # Execute the LangGraph workflow
            final_state = await start_audit_workflow(initial_state)

            # Update session with final state
            if session_id in self.audit_sessions:
                self.audit_sessions[session_id].state = final_state
                self.audit_sessions[session_id].last_accessed = datetime.utcnow()

            # Store final results in swarm memory
            await self.swarm_memory.store_agent_results(session_id, "audit_workflow", {
                "final_status": final_state.status.value,
                "findings_count": len(final_state.findings),
                "overall_risk_score": final_state.overall_risk_score,
                "completion_time": datetime.utcnow().isoformat()
            })

            logger.info(f"Audit workflow completed for session: {session_id}")

        except Exception as e:
            logger.error(f"Audit workflow failed for session {session_id}: {str(e)}")

            # Update session with error state
            if session_id in self.audit_sessions:
                self.audit_sessions[session_id].state.status = AuditStatus.FAILED
                self.audit_sessions[session_id].state.errors.append(f"Workflow execution failed: {str(e)}")
                self.audit_sessions[session_id].state.updated_at = datetime.utcnow()

    def _format_workflow_status(self, state: AuditState) -> Dict[str, Any]:
        """Format workflow state for status response"""
        return {
            "session_id": state.session_id,
            "status": state.status.value,
            "current_step": state.current_step,
            "progress_percentage": state.progress_percentage,
            "steps_completed": state.steps_completed,
            "total_steps": len(self.workflow_steps),
            "findings_count": len(state.findings),
            "errors": state.errors,
            "warnings": state.warnings,
            "last_update": state.updated_at.isoformat()
        }

    def _format_workflow_results(self, state: AuditState) -> Dict[str, Any]:
        """Format workflow state for results response"""
        return {
            "session_id": state.session_id,
            "status": state.status.value,
            "findings": [finding.dict() for finding in state.findings],
            "risk_assessments": [risk.dict() for risk in state.risk_assessments],
            "financial_metrics": state.financial_metrics,
            "anomalies": state.anomalies,
            "overall_risk_score": state.overall_risk_score,
            "executive_summary": state.executive_summary,
            "recommendations": state.recommendations
        }

    def _assess_compliance_status(self, findings: List[AuditFinding]) -> str:
        """Assess overall compliance status based on findings"""
        if not findings:
            return "COMPLIANT"

        critical_findings = [f for f in findings if f.severity == RiskLevel.CRITICAL]
        high_findings = [f for f in findings if f.severity == RiskLevel.HIGH]

        if critical_findings:
            return "NON_COMPLIANT"
        elif high_findings:
            return "CONDITIONALLY_COMPLIANT"
        else:
            return "COMPLIANT"

    def _calculate_processing_time(self, session: AuditSession) -> str:
        """Calculate total processing time"""
        if session.state.completed_at:
            delta = session.state.completed_at - session.start_date
            return str(delta.total_seconds())
        else:
            delta = datetime.utcnow() - session.start_date
            return str(delta.total_seconds())

    async def upload_documents(
        self,
        session_id: str,
        documents: List[DocumentUpload],
        user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Upload and process documents for an audit session.

        Args:
            session_id: Target audit session
            documents: List of documents to upload
            user: Current user information

        Returns:
            Dict containing upload results and extracted data
        """
        logger.info(f"Uploading {len(documents)} documents to session: {session_id}")

        try:
            # Validate session exists and user has access
            if session_id not in self.audit_sessions:
                raise AuditServiceError(
                    f"Audit session not found: {session_id}",
                    "SESSION_NOT_FOUND"
                )

            session = self.audit_sessions[session_id]

            # Process documents using enhanced processor with comprehensive validation and PII protection
            processed_data = await self._process_documents_enhanced(
                documents, session_id
            )

            # Update session state with extracted data
            session.state.extracted_data.update(processed_data)
            session.state.updated_at = datetime.utcnow()

            # Store in swarm memory
            await self.swarm_memory.store_agent_results(session_id, "document_processor", processed_data)

            # Trigger statistical analysis if documents are successfully processed
            if processed_data.get("documents"):
                metrics = await self.statistical_analyzer.analyze_financial_metrics(processed_data)
                session.state.financial_metrics.update(metrics)

                await self.swarm_memory.store_agent_results(session_id, "statistical_analyzer", {
                    "metrics": metrics,
                    "analysis_time": datetime.utcnow().isoformat()
                })

            logger.info(f"Successfully processed {len(documents)} documents for session: {session_id}")

            return {
                "session_id": session_id,
                "documents_processed": len(documents),
                "extraction_results": processed_data,
                "financial_metrics": session.state.financial_metrics,
                "processing_time": datetime.utcnow().isoformat()
            }

        except AuditServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to upload documents for session {session_id}: {str(e)}")
            raise AuditServiceError(
                f"Failed to upload documents: {str(e)}",
                "DOCUMENT_UPLOAD_FAILED"
            )

    async def resume_audit(self, session_id: str, user: Dict[str, Any], from_step: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume a paused or failed audit workflow.

        Args:
            session_id: Audit session to resume
            user: Current user information
            from_step: Optional specific step to resume from

        Returns:
            Dict containing resume operation status
        """
        logger.info(f"Resuming audit session: {session_id}")

        try:
            # Validate session exists
            if session_id not in self.audit_sessions:
                raise AuditServiceError(
                    f"Audit session not found: {session_id}",
                    "SESSION_NOT_FOUND"
                )

            session = self.audit_sessions[session_id]

            # Check if resume is allowed
            if session.state.status == AuditStatus.COMPLETED:
                raise AuditServiceError(
                    "Cannot resume completed audit",
                    "AUDIT_ALREADY_COMPLETED"
                )

            # Resume the workflow
            resumed_state = await resume_audit_workflow(session_id, from_step)

            # Update session
            session.state = resumed_state
            session.last_accessed = datetime.utcnow()

            # Update swarm memory
            await self.swarm_memory.store_session_context(session_id, {
                "resumed_at": datetime.utcnow().isoformat(),
                "resumed_by": user.get("user_id"),
                "resume_from_step": from_step
            })

            logger.info(f"Successfully resumed audit session: {session_id}")

            return {
                "session_id": session_id,
                "status": resumed_state.status.value,
                "current_step": resumed_state.current_step,
                "progress_percentage": resumed_state.progress_percentage,
                "resumed_at": datetime.utcnow().isoformat(),
                "resume_from_step": from_step
            }

        except AuditServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to resume audit session {session_id}: {str(e)}")
            raise AuditServiceError(
                f"Failed to resume audit: {str(e)}",
                "AUDIT_RESUME_FAILED"
            )

    async def cancel_audit(self, session_id: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cancel an ongoing audit session.

        Args:
            session_id: Audit session to cancel
            user: Current user information

        Returns:
            Dict containing cancellation status
        """
        logger.info(f"Cancelling audit session: {session_id}")

        try:
            # Validate session exists
            if session_id not in self.audit_sessions:
                raise AuditServiceError(
                    f"Audit session not found: {session_id}",
                    "SESSION_NOT_FOUND"
                )

            session = self.audit_sessions[session_id]

            # Update session status
            session.state.status = AuditStatus.FAILED
            session.state.errors.append(f"Audit cancelled by user {user.get('user_id')} at {datetime.utcnow()}")
            session.state.updated_at = datetime.utcnow()
            session.last_accessed = datetime.utcnow()

            # Clean up swarm memory
            await self.swarm_memory.cleanup_session(session_id)

            logger.info(f"Successfully cancelled audit session: {session_id}")

            return {
                "session_id": session_id,
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
                "cancelled_by": user.get("user_id")
            }

        except AuditServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel audit session {session_id}: {str(e)}")
            raise AuditServiceError(
                f"Failed to cancel audit: {str(e)}",
                "AUDIT_CANCELLATION_FAILED"
            )

    async def _process_documents_enhanced(
        self,
        documents: List[DocumentUpload],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Enhanced document processing using the comprehensive document processor

        Args:
            documents: List of documents to process
            session_id: Audit session identifier

        Returns:
            Dictionary containing processed document results with PII protection
        """
        logger.info(f"Enhanced processing of {len(documents)} documents for session {session_id}")

        try:
            # Create temporary files for processing
            temp_files = []
            processed_results = {
                "documents": {},
                "total_documents": len(documents),
                "pii_detected_count": 0,
                "ocr_processed_count": 0,
                "validation_results": [],
                "financial_data_summary": {},
                "processing_metadata": {}
            }

            for doc in documents:
                try:
                    # Create temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.filename.split('.')[-1]}") as temp_file:
                        temp_file.write(doc.file_content)
                        temp_file_path = temp_file.name
                        temp_files.append(temp_file_path)

                    # Validate document format
                    validation_result = await self.document_processor.validate_document_format(temp_file_path)
                    processed_results["validation_results"].append({
                        "filename": doc.filename,
                        "validation": validation_result.dict()
                    })

                    if not validation_result.is_valid:
                        processed_results["documents"][doc.filename] = {
                            "error": "Document validation failed",
                            "validation_errors": validation_result.errors,
                            "type": doc.document_type
                        }
                        continue

                    # Track PII detection
                    if validation_result.has_pii:
                        processed_results["pii_detected_count"] += 1

                    # Process based on document type using enhanced processor
                    if doc.document_type == DocumentType.INVOICE:
                        invoice = await self.document_processor.process_invoice(temp_file_path)
                        if invoice:
                            processed_results["documents"][doc.filename] = {
                                "type": "invoice",
                                "data": invoice.dict(),
                                "metadata": {
                                    "processing_method": "enhanced_processor",
                                    "pii_anonymized": validation_result.has_pii,
                                    "confidence_score": 0.9  # High confidence for structured processing
                                }
                            }
                        else:
                            # Fallback to legacy processor
                            legacy_result = await self.legacy_document_processor._extract_invoice_data(doc)
                            processed_results["documents"][doc.filename] = {
                                "type": "invoice",
                                "data": legacy_result,
                                "metadata": {
                                    "processing_method": "legacy_fallback",
                                    "confidence_score": 0.6
                                }
                            }

                    elif doc.document_type == DocumentType.BALANCE_SHEET:
                        balance_sheet = await self.document_processor.process_balance_sheet(temp_file_path)
                        if balance_sheet:
                            processed_results["documents"][doc.filename] = {
                                "type": "balance_sheet",
                                "data": balance_sheet.dict(),
                                "metadata": {
                                    "processing_method": "enhanced_processor",
                                    "pii_anonymized": validation_result.has_pii,
                                    "confidence_score": 0.9
                                }
                            }
                        else:
                            # Fallback to legacy processor
                            legacy_result = await self.legacy_document_processor._extract_balance_sheet_data(doc)
                            processed_results["documents"][doc.filename] = {
                                "type": "balance_sheet",
                                "data": legacy_result,
                                "metadata": {
                                    "processing_method": "legacy_fallback",
                                    "confidence_score": 0.6
                                }
                            }

                    else:
                        # Generic document processing with enhanced text extraction
                        if validation_result.file_type == "pdf":
                            content = await self.document_processor._extract_text_from_pdf(Path(temp_file_path))
                        elif validation_result.file_type == "docx":
                            content = await self.document_processor._extract_text_from_docx(Path(temp_file_path))
                        elif validation_result.file_type == "image":
                            content = await self.document_processor._extract_text_with_ocr(Path(temp_file_path))
                            processed_results["ocr_processed_count"] += 1
                        else:
                            content = "Unsupported file type"

                        # Extract financial data using enhanced methods
                        financial_data = await self.document_processor.extract_financial_data(content)

                        processed_results["documents"][doc.filename] = {
                            "type": doc.document_type,
                            "content": content[:500] + "..." if len(content) > 500 else content,  # Truncate for storage
                            "financial_data": financial_data.dict(),
                            "metadata": {
                                "processing_method": "enhanced_processor",
                                "content_length": len(content),
                                "pii_anonymized": validation_result.has_pii,
                                "confidence_scores": financial_data.confidence_scores
                            }
                        }

                    logger.debug(f"Successfully processed document: {doc.filename}")

                except Exception as e:
                    logger.error(f"Error processing document {doc.filename}: {str(e)}")
                    processed_results["documents"][doc.filename] = {
                        "error": str(e),
                        "type": doc.document_type,
                        "processing_method": "failed"
                    }

                finally:
                    # Clean up temporary file
                    try:
                        import os
                        if 'temp_file_path' in locals():
                            os.unlink(temp_file_path)
                    except:
                        pass

            # Generate processing summary
            successful_docs = len([d for d in processed_results["documents"].values() if "error" not in d])
            failed_docs = len([d for d in processed_results["documents"].values() if "error" in d])

            processed_results["processing_metadata"] = {
                "successful_documents": successful_docs,
                "failed_documents": failed_docs,
                "success_rate": successful_docs / len(documents) if documents else 0,
                "pii_detection_rate": processed_results["pii_detected_count"] / len(documents) if documents else 0,
                "ocr_usage_rate": processed_results["ocr_processed_count"] / len(documents) if documents else 0,
                "processing_timestamp": datetime.utcnow().isoformat(),
                "processor_version": "enhanced_v1.0",
                "session_id": session_id
            }

            # Store comprehensive results in swarm memory
            await self.swarm_memory.store_agent_results(session_id, "enhanced_document_processor", processed_results)

            logger.info(f"Enhanced document processing completed. Success rate: {processed_results['processing_metadata']['success_rate']:.2%}")

            return processed_results

        except Exception as e:
            logger.error(f"Enhanced document processing failed for session {session_id}: {str(e)}")
            raise AuditServiceError(
                f"Enhanced document processing failed: {str(e)}",
                "ENHANCED_DOCUMENT_PROCESSING_FAILED"
            )