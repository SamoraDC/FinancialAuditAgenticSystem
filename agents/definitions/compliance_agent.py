"""
Compliance Agent using PydanticAI for regulatory compliance checks
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class ComplianceContext(BaseModel):
    """Context for compliance operations"""
    company_id: str
    industry: str
    jurisdiction: str
    regulatory_framework: List[str]  # e.g., ["SOX", "GAAP", "IFRS"]
    reporting_period: str


class ComplianceViolation(BaseModel):
    """Represents a compliance violation"""
    violation_id: str = Field(description="Unique identifier for the violation")
    regulation: str = Field(description="Specific regulation violated")
    section: str = Field(description="Section or clause of the regulation")
    severity: str = Field(description="Severity: low, medium, high, critical")
    description: str = Field(description="Description of the violation")
    remediation: str = Field(description="Required remediation action")
    deadline: Optional[str] = Field(description="Deadline for remediation")


class ComplianceReport(BaseModel):
    """Complete compliance assessment report"""
    company_id: str
    assessment_date: str
    violations: List[ComplianceViolation]
    compliance_score: float = Field(ge=0, le=100)
    recommendations: List[str]
    next_review_date: str


# Initialize the compliance agent
compliance_agent = Agent(
    model='groq:mixtral-8x7b-32768',  # Use Groq model
    system_prompt="""
    You are a regulatory compliance expert specializing in financial regulations.
    Your expertise covers:

    1. SOX (Sarbanes-Oxley Act) compliance
    2. GAAP (Generally Accepted Accounting Principles)
    3. IFRS (International Financial Reporting Standards)
    4. SEC regulations
    5. Industry-specific regulations

    Your role is to:
    - Identify compliance violations
    - Assess regulatory risks
    - Provide remediation guidance
    - Ensure adherence to reporting standards
    """,
)


@compliance_agent.tool
async def check_sox_compliance(
    ctx: RunContext[ComplianceContext],
    internal_controls: Dict[str, Any]
) -> Dict[str, Any]:
    """Check SOX compliance for internal controls"""

    violations = []

    # Check Section 302 - Corporate responsibility for financial reports
    if not internal_controls.get('ceo_certification', False):
        violations.append({
            'section': 'Section 302',
            'issue': 'Missing CEO certification for financial reports',
            'severity': 'high'
        })

    if not internal_controls.get('cfo_certification', False):
        violations.append({
            'section': 'Section 302',
            'issue': 'Missing CFO certification for financial reports',
            'severity': 'high'
        })

    # Check Section 404 - Management assessment of internal controls
    if not internal_controls.get('icfr_assessment', False):
        violations.append({
            'section': 'Section 404',
            'issue': 'Missing Internal Control over Financial Reporting assessment',
            'severity': 'critical'
        })

    # Check Section 409 - Real-time disclosure
    disclosure_timeliness = internal_controls.get('disclosure_days', 0)
    if disclosure_timeliness > 4:
        violations.append({
            'section': 'Section 409',
            'issue': f'Material changes disclosed after {disclosure_timeliness} days (max 4 days)',
            'severity': 'medium'
        })

    return {
        'sox_violations': violations,
        'compliance_score': max(100 - (len(violations) * 20), 0)
    }


@compliance_agent.tool
async def validate_financial_disclosures(
    ctx: RunContext[ComplianceContext],
    financial_statements: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate financial statement disclosures"""

    missing_disclosures = []

    # Required disclosures checklist
    required_items = [
        'accounting_policies',
        'significant_estimates',
        'contingencies',
        'subsequent_events',
        'related_party_transactions',
        'segment_information',
        'fair_value_measurements'
    ]

    for item in required_items:
        if item not in financial_statements or not financial_statements[item]:
            missing_disclosures.append(f"Missing disclosure: {item.replace('_', ' ').title()}")

    # Check for adequate detail in disclosures
    quality_issues = []

    if 'accounting_policies' in financial_statements:
        policies = financial_statements['accounting_policies']
        if isinstance(policies, str) and len(policies) < 100:
            quality_issues.append("Accounting policies disclosure appears insufficient")

    return {
        'missing_disclosures': missing_disclosures,
        'quality_issues': quality_issues,
        'disclosure_completeness_score': max(100 - (len(missing_disclosures) * 10), 0)
    }