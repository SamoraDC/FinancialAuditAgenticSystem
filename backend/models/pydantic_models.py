"""
Pydantic models for API requests and responses
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from fastapi import UploadFile

from .audit_models import AuditStatus, RiskLevel, DocumentType


class DocumentUpload(BaseModel):
    """Document upload request model"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    document_type: DocumentType = Field(..., description="Type of financial document")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    description: Optional[str] = Field(None, description="Optional document description")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")


class AuditRequest(BaseModel):
    """Audit session creation request"""
    client_name: str = Field(..., min_length=1, description="Client company name")
    auditor_id: str = Field(..., description="Lead auditor identifier")
    audit_type: str = Field(default="financial", description="Type of audit")
    fiscal_year: int = Field(..., ge=2000, le=2030, description="Fiscal year")
    materiality_threshold: Decimal = Field(..., gt=0, description="Materiality threshold")
    risk_tolerance: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk tolerance")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    audit_scope: List[str] = Field(default_factory=list, description="Audit scope areas")
    target_completion_days: int = Field(default=30, ge=1, le=365, description="Target completion in days")

    @validator('client_name')
    def validate_client_name(cls, v):
        return v.strip()


class AuditProgress(BaseModel):
    """Audit progress status"""
    session_id: str = Field(..., description="Audit session ID")
    status: AuditStatus = Field(..., description="Current status")
    current_step: str = Field(..., description="Current workflow step")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    steps_completed: List[str] = Field(..., description="Completed steps")
    total_steps: int = Field(..., description="Total workflow steps")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    last_update: datetime = Field(..., description="Last update timestamp")


class AuditResponse(BaseModel):
    """Audit session response"""
    session_id: str = Field(..., description="Unique session identifier")
    status: AuditStatus = Field(..., description="Current audit status")
    progress: AuditProgress = Field(..., description="Progress information")
    findings_count: int = Field(..., ge=0, description="Number of findings")
    high_risk_findings: int = Field(..., ge=0, description="High risk findings count")
    overall_risk_score: Optional[float] = Field(None, ge=0, le=1, description="Overall risk score")
    created_at: datetime = Field(..., description="Session creation time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")


class FindingResponse(BaseModel):
    """Audit finding response model"""
    id: str = Field(..., description="Finding identifier")
    title: str = Field(..., description="Finding title")
    category: str = Field(..., description="Finding category")
    severity: RiskLevel = Field(..., description="Severity level")
    description: str = Field(..., description="Finding description")
    recommendation: str = Field(..., description="Recommendation")
    confidence_score: float = Field(..., ge=0, le=1, description="AI confidence")
    financial_impact: Optional[Decimal] = Field(None, description="Financial impact")
    status: str = Field(..., description="Finding status")
    created_at: datetime = Field(..., description="Creation time")


class SystemMetrics(BaseModel):
    """System performance and health metrics"""
    active_sessions: int = Field(..., ge=0, description="Active audit sessions")
    total_documents_processed: int = Field(..., ge=0, description="Total documents processed")
    average_processing_time: float = Field(..., ge=0, description="Average processing time in seconds")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate percentage")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    redis_connected: bool = Field(..., description="Redis connection status")
    database_connected: bool = Field(..., description="Database connection status")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last metrics update")


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str = Field(..., description="Message type")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class AuditConfigUpdate(BaseModel):
    """Audit configuration update model"""
    risk_threshold: Optional[float] = Field(None, ge=0, le=1, description="Risk threshold")
    materiality_threshold: Optional[Decimal] = Field(None, gt=0, description="Materiality threshold")
    compliance_frameworks: Optional[List[str]] = Field(None, description="Compliance frameworks")
    audit_scope: Optional[List[str]] = Field(None, description="Audit scope")


class ComplianceRuleRequest(BaseModel):
    """Compliance rule creation request"""
    name: str = Field(..., min_length=1, description="Rule name")
    description: str = Field(..., min_length=1, description="Rule description")
    regulation_framework: str = Field(..., description="Regulatory framework")
    rule_type: str = Field(..., description="Rule type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    severity: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Default severity")


class RiskAssessmentRequest(BaseModel):
    """Risk assessment request"""
    audit_session_id: str = Field(..., description="Associated audit session ID")
    risk_category: str = Field(..., description="Risk category")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors")
    inherent_risk: RiskLevel = Field(..., description="Inherent risk level")
    control_risk: RiskLevel = Field(..., description="Control risk level")


class ErrorResponse(BaseModel):
    """Error response model"""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class SuccessResponse(BaseModel):
    """Success response model"""
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., ge=0, description="Total items count")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, description="Items per page")
    pages: int = Field(..., ge=1, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class ModelValidationResult(BaseModel):
    """AI model validation result"""
    model_id: str = Field(..., description="Model identifier")
    validation_score: float = Field(..., ge=0, le=1, description="Validation score")
    accuracy_metrics: Dict[str, float] = Field(..., description="Accuracy metrics")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    validation_date: datetime = Field(default_factory=datetime.utcnow, description="Validation date")
    is_production_ready: bool = Field(..., description="Production readiness status")