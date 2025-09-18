"""
MCP (Model Context Protocol) integration service
Handles human-in-the-loop interactions for audit workflow
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ReviewRequest(BaseModel):
    """MCP review request model"""
    session_id: str
    request_type: str = "human_review"
    priority: str = "medium"
    findings: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []
    risk_score: float = 0.0
    context: Dict[str, Any] = {}
    timeout_minutes: int = 60


class ReviewResponse(BaseModel):
    """MCP review response model"""
    session_id: str
    approved: bool
    comments: str = ""
    modifications: List[Dict[str, Any]] = []
    reviewer_id: str = ""
    review_timestamp: datetime
    additional_actions: List[str] = []


class MCPClient:
    """Client for MCP server communication"""

    def __init__(self, mcp_server_url: Optional[str] = None):
        self.mcp_server_url = mcp_server_url or "http://localhost:8000/mcp"
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def request_human_review(self, review_request: Dict[str, Any]) -> Dict[str, Any]:
        """Request human review via MCP"""
        try:
            # Prepare MCP request
            mcp_request = {
                "method": "request_review",
                "params": {
                    "session_id": review_request["session_id"],
                    "review_data": {
                        "risk_score": review_request.get("risk_score", 0.0),
                        "findings_summary": self._prepare_findings_summary(review_request),
                        "anomalies_summary": self._prepare_anomalies_summary(review_request),
                        "urgency": self._determine_urgency(review_request),
                        "context": review_request.get("context", {}),
                        "request_timestamp": datetime.utcnow().isoformat()
                    },
                    "timeout_minutes": review_request.get("timeout_minutes", 60)
                },
                "id": f"review_{review_request['session_id']}_{int(datetime.utcnow().timestamp())}"
            }

            # For now, simulate MCP response since we don't have actual MCP server
            # In production, this would make actual HTTP request to MCP server
            response = await self._simulate_mcp_response(mcp_request)

            return {
                "request_id": mcp_request["id"],
                "status": "pending",
                "estimated_response_time": 30,  # minutes
                "mcp_response": response
            }

        except Exception as e:
            logger.error(f"MCP request failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def check_review_status(self, session_id: str, request_id: str) -> Dict[str, Any]:
        """Check status of pending review request"""
        try:
            # In production, this would query the MCP server
            # For now, simulate response
            return await self._simulate_status_check(session_id, request_id)

        except Exception as e:
            logger.error(f"MCP status check failed: {e}")
            return {"error": str(e)}

    async def get_review_response(self, session_id: str, request_id: str) -> Optional[Dict[str, Any]]:
        """Get completed review response"""
        try:
            # In production, this would fetch from MCP server
            return await self._simulate_review_response(session_id, request_id)

        except Exception as e:
            logger.error(f"Failed to get review response: {e}")
            return None

    def _prepare_findings_summary(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare findings summary for human review"""
        findings = request.get("findings", [])
        
        if not findings:
            return {"message": "No critical findings identified"}

        critical_findings = [f for f in findings if f.get("severity") in ["high", "critical"]]
        medium_findings = [f for f in findings if f.get("severity") == "medium"]

        summary = {
            "total_findings": len(findings),
            "critical_count": len(critical_findings),
            "medium_count": len(medium_findings),
            "top_findings": []
        }

        # Add top 5 most critical findings
        sorted_findings = sorted(findings, 
                               key=lambda x: self._severity_score(x.get("severity", "low")), 
                               reverse=True)

        for finding in sorted_findings[:5]:
            summary["top_findings"].append({
                "title": finding.get("title", "Unknown Finding"),
                "severity": finding.get("severity", "unknown"),
                "description": finding.get("description", "")[:200],  # Truncate
                "confidence": finding.get("confidence", 0.0),
                "recommendation": finding.get("recommendation", "")[:150]
            })

        return summary

    def _prepare_anomalies_summary(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare anomalies summary for human review"""
        anomalies = request.get("anomalies", [])
        
        if not anomalies:
            return {"message": "No significant anomalies detected"}

        high_confidence = [a for a in anomalies if a.get("confidence", 0) > 0.8]
        statistical_anomalies = [a for a in anomalies if a.get("type") == "statistical_deviation"]
        rl_anomalies = [a for a in anomalies if a.get("type") == "rl_anomaly"]

        return {
            "total_anomalies": len(anomalies),
            "high_confidence_count": len(high_confidence),
            "statistical_anomalies": len(statistical_anomalies),
            "rl_detected_anomalies": len(rl_anomalies),
            "avg_confidence": sum(a.get("confidence", 0) for a in anomalies) / len(anomalies),
            "top_anomalies": [
                {
                    "type": a.get("type", "unknown"),
                    "confidence": a.get("confidence", 0.0),
                    "description": a.get("description", "")[:150]
                }
                for a in sorted(anomalies, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
            ]
        }

    def _determine_urgency(self, request: Dict[str, Any]) -> str:
        """Determine review urgency based on risk factors"""
        risk_score = request.get("risk_score", 0.0)
        critical_findings = len([f for f in request.get("findings", []) 
                               if f.get("severity") in ["high", "critical"]])
        high_conf_anomalies = len([a for a in request.get("anomalies", []) 
                                 if a.get("confidence", 0) > 0.8])

        if risk_score > 0.8 or critical_findings >= 3:
            return "urgent"
        elif risk_score > 0.5 or critical_findings >= 1 or high_conf_anomalies >= 2:
            return "high"
        elif risk_score > 0.3 or high_conf_anomalies >= 1:
            return "medium"
        else:
            return "low"

    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting"""
        scores = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "unknown": 0
        }
        return scores.get(severity.lower(), 0)

    async def _simulate_mcp_response(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP server response (for development/testing)"""
        # In production, this would be actual HTTP request to MCP server
        await asyncio.sleep(0.1)  # Simulate network delay

        return {
            "status": "acknowledged",
            "request_id": request["id"],
            "estimated_review_time": "30 minutes",
            "assigned_reviewer": "audit_team",
            "priority": request["params"]["review_data"]["urgency"],
            "notification_sent": True
        }

    async def _simulate_status_check(self, session_id: str, request_id: str) -> Dict[str, Any]:
        """Simulate MCP status check response"""
        await asyncio.sleep(0.05)

        # Simulate different states
        import random
        states = ["pending", "in_review", "completed"]
        current_state = random.choice(states)

        return {
            "request_id": request_id,
            "session_id": session_id,
            "status": current_state,
            "reviewer_assigned": current_state != "pending",
            "estimated_completion": datetime.utcnow() + timedelta(minutes=15),
            "last_updated": datetime.utcnow().isoformat()
        }

    async def _simulate_review_response(self, session_id: str, request_id: str) -> Dict[str, Any]:
        """Simulate human reviewer response"""
        await asyncio.sleep(0.05)

        # Simulate reviewer decision
        import random
        approved = random.choice([True, True, False])  # 66% approval rate

        return {
            "request_id": request_id,
            "session_id": session_id,
            "approved": approved,
            "reviewer_id": "senior_auditor_001",
            "review_timestamp": datetime.utcnow().isoformat(),
            "comments": "Findings reviewed and verified" if approved else "Additional investigation required",
            "modifications": [] if approved else [
                {
                    "action": "investigate_further",
                    "target": "high_risk_transactions",
                    "reason": "Pattern requires deeper analysis"
                }
            ],
            "additional_actions": [
                "schedule_followup_review"
            ] if not approved else []
        }


# Global MCP client instance
_mcp_client = None

async def get_mcp_client() -> MCPClient:
    """Get global MCP client instance"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client

async def request_human_intervention(session_id: str, findings: List[Dict], 
                                   anomalies: List[Dict], risk_score: float,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to request human intervention via MCP"""
    client = await get_mcp_client()
    
    request_data = {
        "session_id": session_id,
        "findings": findings,
        "anomalies": anomalies,
        "risk_score": risk_score,
        "context": context or {},
        "timeout_minutes": 60
    }
    
    return await client.request_human_review(request_data)
