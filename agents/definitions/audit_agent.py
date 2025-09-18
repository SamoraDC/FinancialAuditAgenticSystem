"""
Financial Audit Agent using PydanticAI
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from agents.tools.financial_analysis import FinancialAnalysisTool
from agents.tools.risk_assessment import RiskAssessmentTool


class AuditContext(BaseModel):
    """Context for audit operations"""
    audit_id: str
    company_name: str
    financial_statements: Dict[str, Any]
    audit_scope: List[str]
    risk_threshold: float = 0.7


class AuditFinding(BaseModel):
    """Represents an audit finding"""
    finding_id: str = Field(description="Unique identifier for the finding")
    category: str = Field(description="Category of the finding (e.g., revenue, expenses, assets)")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    description: str = Field(description="Detailed description of the finding")
    recommendation: str = Field(description="Recommended action to address the finding")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the finding")


class AuditReport(BaseModel):
    """Complete audit report"""
    audit_id: str
    company_name: str
    findings: List[AuditFinding]
    overall_risk_score: float
    executive_summary: str
    recommendations: List[str]


# Initialize the audit agent
audit_agent = Agent[AuditContext, AuditReport](
    model=Model(),  # Use default model
    result_type=AuditReport,
    system_prompt="""
    You are an expert financial auditor with deep knowledge of accounting standards,
    risk assessment, and fraud detection. Your role is to:

    1. Analyze financial statements for inconsistencies and anomalies
    2. Assess risks based on industry standards and regulatory requirements
    3. Identify potential fraud indicators
    4. Provide actionable recommendations
    5. Generate comprehensive audit reports

    Always maintain professional skepticism and follow auditing standards.
    Focus on materiality and risk-based approaches.
    """,
    tools=[FinancialAnalysisTool(), RiskAssessmentTool()],
)


@audit_agent.tool
async def analyze_financial_ratios(
    ctx: RunContext[AuditContext],
    financial_data: Dict[str, float]
) -> Dict[str, Any]:
    """Analyze financial ratios for anomalies"""

    # Calculate key financial ratios
    ratios = {}

    if 'total_assets' in financial_data and 'total_liabilities' in financial_data:
        ratios['debt_to_assets'] = financial_data['total_liabilities'] / financial_data['total_assets']

    if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
        ratios['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']

    if 'revenue' in financial_data and 'total_assets' in financial_data:
        ratios['asset_turnover'] = financial_data['revenue'] / financial_data['total_assets']

    # Assess ratio reasonableness
    anomalies = []

    if ratios.get('debt_to_assets', 0) > 0.8:
        anomalies.append("High debt-to-assets ratio indicates potential liquidity risk")

    if ratios.get('current_ratio', 0) < 1.0:
        anomalies.append("Current ratio below 1.0 indicates potential short-term liquidity issues")

    return {
        'ratios': ratios,
        'anomalies': anomalies,
        'risk_score': min(len(anomalies) * 0.3, 1.0)
    }


@audit_agent.tool
async def detect_fraud_indicators(
    ctx: RunContext[AuditContext],
    transaction_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Detect potential fraud indicators in transaction data"""

    indicators = []
    risk_score = 0.0

    # Check for round number transactions (potential red flag)
    round_numbers = sum(1 for tx in transaction_data
                       if isinstance(tx.get('amount'), (int, float)) and
                       tx['amount'] % 1000 == 0)

    if round_numbers / len(transaction_data) > 0.1:
        indicators.append("High frequency of round number transactions")
        risk_score += 0.3

    # Check for unusual transaction timing
    weekend_transactions = sum(1 for tx in transaction_data
                              if tx.get('day_of_week', 0) in [0, 6])  # Assuming 0=Sunday, 6=Saturday

    if weekend_transactions > 0:
        indicators.append("Transactions occurring on weekends")
        risk_score += 0.2

    # Check for duplicate transactions
    amounts = [tx.get('amount', 0) for tx in transaction_data]
    duplicates = len(amounts) - len(set(amounts))

    if duplicates > len(transaction_data) * 0.05:
        indicators.append("High frequency of duplicate transaction amounts")
        risk_score += 0.4

    return {
        'fraud_indicators': indicators,
        'risk_score': min(risk_score, 1.0),
        'total_transactions_analyzed': len(transaction_data)
    }