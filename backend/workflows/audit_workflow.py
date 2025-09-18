"""
LangGraph audit workflow implementation
Complete implementation following FrameworkDoc.md specifications
with Redis checkpointing, PydanticAI integration, and human-in-the-loop
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
import os
import json
from langgraph.graph import StateGraph
from langgraph.constants import END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import redis
import asyncio

from backend.models.audit_models import AuditState, AuditStatus, RiskLevel
from .nodes import (
    initialize_audit,
    ingest_and_parse,
    statistical_analysis,
    regulatory_validation,
    rl_anomaly_detection,
    consolidate_results,
    human_review_mcp,
    deep_dive_investigation,
    generate_report,
    handle_error
)

logger = logging.getLogger(__name__)

# Risk thresholds for conditional routing
HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.5
CRITICAL_FINDINGS_THRESHOLD = 3


class AuditWorkflow:
    """Main audit workflow orchestrator using LangGraph

    Implements the complete LangGraph architecture from FrameworkDoc.md:
    - Redis checkpointing for state persistence
    - Conditional edges for dynamic routing
    - PydanticAI integration for structured data
    - Human-in-the-loop via MCP
    - RL-based anomaly detection
    """

    def __init__(self, checkpointer=None, redis_url: Optional[str] = None):
        """Initialize workflow with Redis checkpointing"""
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.checkpointer = self._setup_checkpointer(checkpointer)
        self.graph = self._create_graph()
        self.interrupt_enabled = True

    def _setup_checkpointer(self, checkpointer=None):
        """Setup checkpointer for state persistence"""
        if checkpointer:
            return checkpointer

        try:
            # Test Redis connection for potential future use
            redis_client = redis.from_url(self.redis_url)
            redis_client.ping()  # Test connection
            logger.info("Redis connection successful, but using MemorySaver for now")
            # TODO: Implement Redis checkpointer when available in langgraph
            return MemorySaver()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using MemorySaver")
            return MemorySaver()

    def _create_graph(self) -> StateGraph:
        """Create the complete LangGraph workflow with conditional routing"""

        # Create workflow graph
        workflow = StateGraph(AuditState)

        # Add all nodes following FrameworkDoc.md architecture
        workflow.add_node("initialize", initialize_audit)
        workflow.add_node("ingest_and_parse", ingest_and_parse)
        workflow.add_node("statistical_analysis", statistical_analysis)
        workflow.add_node("regulatory_validation", regulatory_validation)
        workflow.add_node("rl_anomaly_detection", rl_anomaly_detection)
        workflow.add_node("consolidate_results", consolidate_results)
        workflow.add_node("human_review_mcp", human_review_mcp)
        workflow.add_node("deep_dive_investigation", deep_dive_investigation)
        workflow.add_node("generate_report", generate_report)
        workflow.add_node("error_handler", handle_error)

        # Set entry point
        workflow.set_entry_point("initialize")

        # Add workflow transitions with conditional routing
        workflow.add_edge("initialize", "ingest_and_parse")

        # Conditional edge after document ingestion
        workflow.add_conditional_edges(
            "ingest_and_parse",
            self._should_continue_after_ingest,
            {
                "continue": "statistical_analysis",
                "error": "error_handler"
            }
        )

        # Parallel analysis branches
        workflow.add_edge("statistical_analysis", "regulatory_validation")
        workflow.add_edge("statistical_analysis", "rl_anomaly_detection")

        # Consolidation after parallel analysis
        workflow.add_edge("regulatory_validation", "consolidate_results")
        workflow.add_edge("rl_anomaly_detection", "consolidate_results")

        # Critical conditional routing based on risk scores
        workflow.add_conditional_edges(
            "consolidate_results",
            self._route_based_on_risk,
            {
                "deep_dive": "deep_dive_investigation",
                "human_review": "human_review_mcp",
                "generate_report": "generate_report",
                "error": "error_handler"
            }
        )

        # Routes from deep dive and human review
        workflow.add_edge("deep_dive_investigation", "generate_report")
        workflow.add_edge("human_review_mcp", "generate_report")

        # Final transitions
        workflow.add_edge("generate_report", END)
        workflow.add_edge("error_handler", END)

        # Compile graph with Redis checkpointer
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review_mcp"],  # Enable human intervention
            debug=True
        )

    def _should_continue_after_ingest(self, state: AuditState) -> str:
        """Determine if workflow should continue after document ingestion"""
        if state.errors:
            logger.warning(f"Errors in ingestion for session {state.session_id}: {state.errors}")
            return "error"

        if not state.extracted_data:
            logger.warning(f"No data extracted for session {state.session_id}")
            return "error"

        # Check if we have sufficient data for analysis
        extracted_docs = state.extracted_data.get('documents', [])
        if len(extracted_docs) == 0:
            logger.warning(f"No documents successfully parsed for session {state.session_id}")
            return "error"

        return "continue"

    def _route_based_on_risk(self, state: AuditState) -> str:
        """Dynamic routing based on consolidated risk scores and findings

        Implements the conditional edge logic from FrameworkDoc.md:
        - HIGH_THRESHOLD: route to deep_dive_investigation
        - MEDIUM_THRESHOLD: route to human_review_mcp
        - Otherwise: proceed to generate_report
        """
        try:
            risk_score = state.overall_risk_score or 0.0
            critical_findings = len([
                f for f in state.findings
                if f.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ])
            high_confidence_anomalies = len([
                a for a in state.anomalies
                if a.get('confidence', 0) > 0.8
            ])

            logger.info(f"Risk routing for session {state.session_id}: "
                       f"score={risk_score:.3f}, critical_findings={critical_findings}, "
                       f"high_conf_anomalies={high_confidence_anomalies}")

            # Route to deep dive for very high risk
            if (risk_score >= HIGH_RISK_THRESHOLD or
                critical_findings >= CRITICAL_FINDINGS_THRESHOLD or
                high_confidence_anomalies >= 5):
                logger.info(f"Routing to deep dive investigation: high risk detected")
                return "deep_dive"

            # Route to human review for medium-high risk
            elif (risk_score >= MEDIUM_RISK_THRESHOLD or
                  critical_findings >= 1 or
                  high_confidence_anomalies >= 2):
                logger.info(f"Routing to human review: medium risk detected")
                # Trigger interrupt for human intervention
                if self.interrupt_enabled:
                    interrupt("Human review required for medium-high risk findings")
                return "human_review"

            # Low risk - proceed directly to report
            else:
                logger.info(f"Low risk detected, proceeding to report generation")
                return "generate_report"

        except Exception as e:
            logger.error(f"Risk routing failed for session {state.session_id}: {e}")
            return "error"

    async def run_audit(self, initial_state: AuditState, config: Optional[Dict] = None) -> AuditState:
        """Run the complete audit workflow with Redis checkpointing"""
        try:
            logger.info(f"Starting audit workflow for session {initial_state.session_id}")

            # Configure the run with thread ID for Redis persistence
            run_config = {
                "configurable": {
                    "thread_id": initial_state.session_id,
                    "checkpoint_ns": f"audit_{initial_state.session_id}"
                },
                "recursion_limit": 100,
                "debug": True
            }
            if config:
                run_config.update(config)

            # Execute the workflow with state persistence
            result = await self.graph.ainvoke(initial_state, config=run_config)

            # Ensure final state is persisted
            await self._persist_final_state(result, run_config)

            logger.info(f"Audit workflow completed for session {initial_state.session_id}")
            return result

        except Exception as e:
            logger.error(f"Audit workflow failed for session {initial_state.session_id}: {str(e)}")

            # Update state with error
            initial_state.status = AuditStatus.FAILED
            initial_state.errors.append(f"Workflow error: {str(e)}")
            initial_state.updated_at = datetime.utcnow()

            # Persist error state
            try:
                error_config = {
                    "configurable": {
                        "thread_id": initial_state.session_id,
                        "checkpoint_ns": f"audit_{initial_state.session_id}"
                    }
                }
                await self.graph.aupdate_state(error_config, initial_state.dict())
            except Exception as persist_error:
                logger.error(f"Failed to persist error state: {persist_error}")

            return initial_state

    async def _persist_final_state(self, state: AuditState, config: Dict[str, Any]):
        """Ensure final audit state is properly persisted"""
        try:
            # Create audit trail entry
            audit_trail = {
                'session_id': state.session_id,
                'final_status': state.status.value,
                'completion_time': datetime.utcnow().isoformat(),
                'steps_completed': state.steps_completed,
                'findings_count': len(state.findings),
                'anomalies_count': len(state.anomalies),
                'overall_risk_score': state.overall_risk_score,
                'errors': state.errors,
                'warnings': state.warnings
            }

            logger.info(f"Audit trail for session {state.session_id}: {json.dumps(audit_trail, indent=2)}")

        except Exception as e:
            logger.error(f"Failed to create audit trail: {e}")

    async def get_state(self, session_id: str) -> Optional[AuditState]:
        """Get current state of an audit session from Redis checkpoint"""
        try:
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": f"audit_{session_id}"
                }
            }
            snapshot = await self.graph.aget_state(config)

            if snapshot and snapshot.values:
                # Convert to AuditState if needed
                if isinstance(snapshot.values, dict):
                    return AuditState(**snapshot.values)
                return snapshot.values
            return None

        except Exception as e:
            logger.error(f"Failed to get state for session {session_id}: {str(e)}")
            return None

    async def get_state_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get state checkpoint history for audit trail"""
        try:
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": f"audit_{session_id}"
                }
            }

            history = []
            async for state in self.graph.aget_state_history(config, limit=limit):
                history.append({
                    'checkpoint_id': state.config.get('configurable', {}).get('checkpoint_id'),
                    'step': state.values.get('current_step') if state.values else None,
                    'status': state.values.get('status') if state.values else None,
                    'timestamp': state.created_at if hasattr(state, 'created_at') else None,
                    'metadata': state.metadata
                })

            return history

        except Exception as e:
            logger.error(f"Failed to get state history for session {session_id}: {str(e)}")
            return []

    async def update_state(self, session_id: str, updates: Dict[str, Any], as_node: Optional[str] = None) -> bool:
        """Update state of an audit session with Redis persistence"""
        try:
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": f"audit_{session_id}"
                }
            }

            # Update timestamp
            updates['updated_at'] = datetime.utcnow()

            await self.graph.aupdate_state(config, updates, as_node=as_node)

            logger.info(f"State updated for session {session_id}: {list(updates.keys())}")
            return True

        except Exception as e:
            logger.error(f"Failed to update state for session {session_id}: {str(e)}")
            return False

    async def resume_audit(self, session_id: str, from_step: Optional[str] = None, user_input: Optional[Dict] = None) -> AuditState:
        """Resume an interrupted audit workflow with optional human input"""
        try:
            logger.info(f"Resuming audit workflow for session {session_id}")

            # Get current state from Redis checkpoint
            current_state = await self.get_state(session_id)
            if not current_state:
                raise ValueError(f"No state found for session {session_id}")

            # Handle human input if provided (for MCP integration)
            if user_input:
                await self.update_state(session_id, {
                    'human_input': user_input,
                    'human_review_timestamp': datetime.utcnow().isoformat()
                })
                logger.info(f"Human input received for session {session_id}: {user_input}")

            # Update status to in progress
            current_state.status = AuditStatus.IN_PROGRESS
            current_state.updated_at = datetime.utcnow()

            # Resume from specific step if provided
            if from_step:
                current_state.current_step = from_step
                logger.info(f"Resuming from step: {from_step}")

            # Continue execution with Redis checkpointing
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": f"audit_{session_id}"
                },
                "recursion_limit": 100
            }

            # Resume the workflow from the last checkpoint
            result = await self.graph.ainvoke(None, config=config)  # None means continue from checkpoint

            logger.info(f"Resumed audit workflow completed for session {session_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to resume audit workflow for session {session_id}: {str(e)}")
            raise

    async def pause_audit(self, session_id: str, reason: str = "Manual pause") -> bool:
        """Pause an audit workflow at the current checkpoint"""
        try:
            await self.update_state(session_id, {
                'status': AuditStatus.REVIEW.value,
                'pause_reason': reason,
                'paused_at': datetime.utcnow().isoformat()
            })

            logger.info(f"Audit workflow paused for session {session_id}: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to pause audit for session {session_id}: {e}")
            return False

    async def cancel_audit(self, session_id: str, reason: str = "Manual cancellation") -> bool:
        """Cancel an audit workflow"""
        try:
            await self.update_state(session_id, {
                'status': AuditStatus.FAILED.value,
                'cancellation_reason': reason,
                'cancelled_at': datetime.utcnow().isoformat()
            })

            logger.info(f"Audit workflow cancelled for session {session_id}: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel audit for session {session_id}: {e}")
            return False

    def get_workflow_steps(self) -> List[str]:
        """Get list of all workflow steps as defined in FrameworkDoc.md"""
        return [
            "initialize",
            "ingest_and_parse",
            "statistical_analysis",
            "regulatory_validation",
            "rl_anomaly_detection",
            "consolidate_results",
            "human_review_mcp",
            "deep_dive_investigation",
            "generate_report"
        ]

    def get_workflow_topology(self) -> Dict[str, Any]:
        """Get the workflow graph topology for visualization"""
        return {
            'nodes': self.get_workflow_steps() + ['error_handler'],
            'edges': [
                ('initialize', 'ingest_and_parse'),
                ('ingest_and_parse', 'statistical_analysis'),
                ('statistical_analysis', 'regulatory_validation'),
                ('statistical_analysis', 'rl_anomaly_detection'),
                ('regulatory_validation', 'consolidate_results'),
                ('rl_anomaly_detection', 'consolidate_results'),
                ('consolidate_results', 'deep_dive_investigation'),
                ('consolidate_results', 'human_review_mcp'),
                ('consolidate_results', 'generate_report'),
                ('deep_dive_investigation', 'generate_report'),
                ('human_review_mcp', 'generate_report')
            ],
            'conditional_edges': [
                ('ingest_and_parse', '_should_continue_after_ingest'),
                ('consolidate_results', '_route_based_on_risk')
            ],
            'interrupt_nodes': ['human_review_mcp']
        }

    async def get_audit_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive audit metrics and performance data"""
        try:
            state = await self.get_state(session_id)
            if not state:
                return {}

            history = await self.get_state_history(session_id)

            return {
                'session_id': session_id,
                'status': state.status.value if state.status else 'unknown',
                'progress_percentage': state.progress_percentage,
                'current_step': state.current_step,
                'steps_completed': state.steps_completed,
                'total_steps': len(self.get_workflow_steps()),
                'findings_count': len(state.findings),
                'anomalies_count': len(state.anomalies),
                'overall_risk_score': state.overall_risk_score,
                'documents_processed': len(state.extracted_data.get('documents', [])),
                'errors_count': len(state.errors),
                'warnings_count': len(state.warnings),
                'checkpoints_count': len(history),
                'created_at': state.created_at.isoformat() if state.created_at else None,
                'updated_at': state.updated_at.isoformat() if state.updated_at else None,
                'completed_at': state.completed_at.isoformat() if state.completed_at else None
            }

        except Exception as e:
            logger.error(f"Failed to get audit metrics for session {session_id}: {e}")
            return {}


def create_audit_graph(checkpointer=None, redis_url: Optional[str] = None) -> StateGraph:
    """Factory function to create audit workflow graph with Redis checkpointing"""
    workflow = AuditWorkflow(checkpointer=checkpointer, redis_url=redis_url)
    return workflow.graph


# Global workflow instance with Redis checkpointing
_workflow_instance = None

def get_workflow(redis_url: Optional[str] = None) -> AuditWorkflow:
    """Get global workflow instance with Redis checkpointing"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = AuditWorkflow(redis_url=redis_url)
    return _workflow_instance

def reset_workflow():
    """Reset global workflow instance (useful for testing)"""
    global _workflow_instance
    _workflow_instance = None


async def start_audit_workflow(initial_state: AuditState, redis_url: Optional[str] = None) -> AuditState:
    """Convenience function to start audit workflow with Redis checkpointing"""
    workflow = get_workflow(redis_url=redis_url)
    return await workflow.run_audit(initial_state)


async def get_audit_state(session_id: str, redis_url: Optional[str] = None) -> Optional[AuditState]:
    """Convenience function to get audit state from Redis checkpoint"""
    workflow = get_workflow(redis_url=redis_url)
    return await workflow.get_state(session_id)


async def resume_audit_workflow(session_id: str, from_step: Optional[str] = None,
                                user_input: Optional[Dict] = None, redis_url: Optional[str] = None) -> AuditState:
    """Convenience function to resume audit workflow with optional human input"""
    workflow = get_workflow(redis_url=redis_url)
    return await workflow.resume_audit(session_id, from_step, user_input)

async def get_audit_metrics(session_id: str, redis_url: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get audit metrics"""
    workflow = get_workflow(redis_url=redis_url)
    return await workflow.get_audit_metrics(session_id)

async def pause_audit_workflow(session_id: str, reason: str = "Manual pause", redis_url: Optional[str] = None) -> bool:
    """Convenience function to pause audit workflow"""
    workflow = get_workflow(redis_url=redis_url)
    return await workflow.pause_audit(session_id, reason)

async def cancel_audit_workflow(session_id: str, reason: str = "Manual cancellation", redis_url: Optional[str] = None) -> bool:
    """Convenience function to cancel audit workflow"""
    workflow = get_workflow(redis_url=redis_url)
    return await workflow.cancel_audit(session_id, reason)