"""
API routes for the Financial Audit Agentic System
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

from backend.core.dependencies import get_current_user
from backend.services.audit_service import AuditService
from backend.models.schemas import AuditRequest, AuditResponse

router = APIRouter()


@router.post("/audit/start", response_model=AuditResponse)
async def start_audit(
    audit_request: AuditRequest,
    current_user=Depends(get_current_user)
):
    """Start a new financial audit workflow"""
    audit_service = AuditService()
    return await audit_service.start_audit(audit_request, current_user)


@router.get("/audit/{audit_id}/status")
async def get_audit_status(
    audit_id: str,
    current_user=Depends(get_current_user)
):
    """Get the status of an ongoing audit"""
    audit_service = AuditService()
    return await audit_service.get_audit_status(audit_id, current_user)


@router.get("/audit/{audit_id}/results")
async def get_audit_results(
    audit_id: str,
    current_user=Depends(get_current_user)
):
    """Get the results of a completed audit"""
    audit_service = AuditService()
    return await audit_service.get_audit_results(audit_id, current_user)


@router.get("/audits")
async def list_audits(
    current_user=Depends(get_current_user)
):
    """List all audits for the current user"""
    audit_service = AuditService()
    return await audit_service.list_audits(current_user)