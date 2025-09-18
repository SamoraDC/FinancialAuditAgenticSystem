"""
Core audit data models using PydanticAI
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent


class AuditStatus(str, Enum):
    """Audit status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DocumentType(str, Enum):
    """Document type enumeration"""
    INVOICE = "invoice"
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    TRIAL_BALANCE = "trial_balance"
    GENERAL_LEDGER = "general_ledger"


class Invoice(BaseModel):
    """Invoice data model"""
    id: str = Field(..., description="Unique invoice identifier")
    invoice_number: str = Field(..., description="Invoice number")
    vendor_name: str = Field(..., description="Vendor name")
    invoice_date: datetime = Field(..., description="Invoice date")
    due_date: Optional[datetime] = Field(None, description="Payment due date")
    amount: Decimal = Field(..., description="Invoice amount", ge=0)
    currency: str = Field(default="USD", description="Currency code")
    line_items: List[Dict[str, Any]] = Field(default_factory=list, description="Invoice line items")
    tax_amount: Optional[Decimal] = Field(None, description="Tax amount", ge=0)
    discount_amount: Optional[Decimal] = Field(None, description="Discount amount", ge=0)
    payment_terms: Optional[str] = Field(None, description="Payment terms")
    approval_status: str = Field(default="pending", description="Approval status")
    processed_by: Optional[str] = Field(None, description="Processed by user")

    @validator('amount', 'tax_amount', 'discount_amount')
    def validate_amounts(cls, v):
        if v is not None and v < 0:
            raise ValueError('Amounts must be non-negative')
        return v


class BalanceSheet(BaseModel):
    """Balance sheet data model"""
    id: str = Field(..., description="Unique balance sheet identifier")
    company_name: str = Field(..., description="Company name")
    period_end: datetime = Field(..., description="Period end date")
    reporting_currency: str = Field(default="USD", description="Reporting currency")

    # Assets
    current_assets: Dict[str, Decimal] = Field(default_factory=dict, description="Current assets")
    non_current_assets: Dict[str, Decimal] = Field(default_factory=dict, description="Non-current assets")
    total_assets: Decimal = Field(..., description="Total assets", ge=0)

    # Liabilities
    current_liabilities: Dict[str, Decimal] = Field(default_factory=dict, description="Current liabilities")
    non_current_liabilities: Dict[str, Decimal] = Field(default_factory=dict, description="Non-current liabilities")
    total_liabilities: Decimal = Field(..., description="Total liabilities", ge=0)

    # Equity
    shareholders_equity: Dict[str, Decimal] = Field(default_factory=dict, description="Shareholders equity")
    total_equity: Decimal = Field(..., description="Total equity")

    # Metadata
    prepared_by: Optional[str] = Field(None, description="Prepared by")
    reviewed_by: Optional[str] = Field(None, description="Reviewed by")
    creation_date: datetime = Field(default_factory=datetime.utcnow, description="Creation date")

    @validator('total_assets', 'total_liabilities')
    def validate_totals(cls, v):
        if v < 0:
            raise ValueError('Totals must be non-negative')
        return v


class AuditFinding(BaseModel):
    """Audit finding data model"""
    id: str = Field(..., description="Unique finding identifier")
    audit_session_id: str = Field(..., description="Associated audit session ID")
    category: str = Field(..., description="Finding category")
    severity: RiskLevel = Field(..., description="Severity level")
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Detailed description")
    recommendation: str = Field(..., description="Recommended action")
    confidence_score: float = Field(..., ge=0, le=1, description="AI confidence score")
    affected_accounts: List[str] = Field(default_factory=list, description="Affected account codes")
    financial_impact: Optional[Decimal] = Field(None, description="Estimated financial impact")
    regulatory_reference: Optional[str] = Field(None, description="Relevant regulation reference")
    status: str = Field(default="open", description="Finding status")
    assigned_to: Optional[str] = Field(None, description="Assigned auditor")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class ComplianceRule(BaseModel):
    """Compliance rule data model"""
    id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    regulation_framework: str = Field(..., description="Regulatory framework (SOX, GAAP, etc.)")
    rule_type: str = Field(..., description="Rule type (calculation, threshold, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    severity: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Default severity")
    is_active: bool = Field(default=True, description="Rule active status")
    created_by: str = Field(..., description="Rule creator")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class RiskAssessment(BaseModel):
    """Risk assessment data model"""
    id: str = Field(..., description="Unique assessment identifier")
    audit_session_id: str = Field(..., description="Associated audit session ID")
    risk_category: str = Field(..., description="Risk category")
    inherent_risk: RiskLevel = Field(..., description="Inherent risk level")
    control_risk: RiskLevel = Field(..., description="Control risk level")
    detection_risk: RiskLevel = Field(..., description="Detection risk level")
    overall_risk: RiskLevel = Field(..., description="Overall audit risk")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Risk mitigation strategies")
    quantitative_score: float = Field(..., ge=0, le=1, description="Quantitative risk score")
    assessed_by: str = Field(..., description="Risk assessor")
    assessment_date: datetime = Field(default_factory=datetime.utcnow, description="Assessment date")


class AuditState(BaseModel):
    """LangGraph audit workflow state"""
    session_id: str = Field(..., description="Unique session identifier")
    status: AuditStatus = Field(default=AuditStatus.PENDING, description="Current audit status")
    current_step: str = Field(default="initialize", description="Current workflow step")
    progress_percentage: float = Field(default=0.0, ge=0, le=100, description="Progress percentage")

    # Input data
    uploaded_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Uploaded documents")
    audit_scope: List[str] = Field(default_factory=list, description="Audit scope")
    risk_threshold: float = Field(default=0.7, ge=0, le=1, description="Risk threshold")

    # Processed data
    extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Extracted document data")
    financial_metrics: Dict[str, float] = Field(default_factory=dict, description="Calculated metrics")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    findings: List[AuditFinding] = Field(default_factory=list, description="Audit findings")
    risk_assessments: List[RiskAssessment] = Field(default_factory=list, description="Risk assessments")

    # Workflow metadata
    steps_completed: List[str] = Field(default_factory=list, description="Completed steps")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")

    # Results
    overall_risk_score: Optional[float] = Field(None, ge=0, le=1, description="Overall risk score")
    executive_summary: Optional[str] = Field(None, description="Executive summary")
    recommendations: List[str] = Field(default_factory=list, description="Key recommendations")


class AuditSession(BaseModel):
    """Complete audit session data model"""
    id: str = Field(..., description="Unique session identifier")
    client_name: str = Field(..., description="Client company name")
    auditor_id: str = Field(..., description="Lead auditor identifier")
    audit_type: str = Field(..., description="Type of audit")
    fiscal_year: int = Field(..., description="Fiscal year being audited")
    start_date: datetime = Field(..., description="Audit start date")
    target_completion: datetime = Field(..., description="Target completion date")

    # Configuration
    materiality_threshold: Decimal = Field(..., description="Materiality threshold")
    risk_tolerance: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk tolerance")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable frameworks")

    # Current state
    state: AuditState = Field(..., description="Current workflow state")

    # Metadata
    created_by: str = Field(..., description="Session creator")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access time")
    is_active: bool = Field(default=True, description="Session active status")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }