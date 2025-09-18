"""
Unit tests for PydanticAI agents
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from agents.definitions.audit_agent import audit_agent, AuditContext, AuditFinding, AuditReport
from agents.definitions.compliance_agent import compliance_agent, ComplianceContext, ComplianceViolation
from agents.tools.financial_analysis import FinancialAnalysisTool
from agents.tools.risk_assessment import RiskAssessmentTool


class TestFinancialAnalysisTool:
    """Test FinancialAnalysisTool functionality"""

    def test_calculate_liquidity_ratios(self, sample_financial_data):
        """Test liquidity ratio calculations"""
        tool = FinancialAnalysisTool()
        ratios = tool.calculate_liquidity_ratios(sample_financial_data)

        # Current ratio = current_assets / current_liabilities = 300000 / 150000 = 2.0
        assert ratios['current_ratio'] == 2.0

        # Cash ratio = cash / current_liabilities = 100000 / 150000 = 0.667
        assert abs(ratios['cash_ratio'] - 0.6666666666666666) < 0.001

    def test_calculate_profitability_ratios(self, sample_financial_data):
        """Test profitability ratio calculations"""
        tool = FinancialAnalysisTool()
        ratios = tool.calculate_profitability_ratios(sample_financial_data)

        # Net profit margin = net_income / revenue = 50000 / 500000 = 0.1
        assert ratios['net_profit_margin'] == 0.1

        # ROA = net_income / total_assets = 50000 / 1000000 = 0.05
        assert ratios['return_on_assets'] == 0.05

        # ROE = net_income / shareholders_equity = 50000 / 400000 = 0.125
        assert ratios['return_on_equity'] == 0.125

    def test_calculate_leverage_ratios(self, sample_financial_data):
        """Test leverage ratio calculations"""
        tool = FinancialAnalysisTool()
        ratios = tool.calculate_leverage_ratios(sample_financial_data)

        # Debt-to-equity = total_debt / shareholders_equity = 600000 / 400000 = 1.5
        assert ratios['debt_to_equity'] == 1.5

        # Debt-to-assets = total_debt / total_assets = 600000 / 1000000 = 0.6
        assert ratios['debt_to_assets'] == 0.6

        # Interest coverage = ebit / interest_expense = 75000 / 15000 = 5.0
        assert ratios['interest_coverage'] == 5.0

    def test_detect_anomalies(self):
        """Test anomaly detection in financial ratios"""
        tool = FinancialAnalysisTool()

        # Test ratios
        ratios = {
            'current_ratio': 0.5,  # Below benchmark
            'debt_to_equity': 3.0,  # Above benchmark
            'net_profit_margin': 0.15  # Normal
        }

        # Industry benchmarks
        benchmarks = {
            'current_ratio': {'min': 1.0, 'median': 1.5, 'max': 3.0},
            'debt_to_equity': {'min': 0.0, 'median': 1.0, 'max': 2.0},
            'net_profit_margin': {'min': 0.05, 'median': 0.10, 'max': 0.20}
        }

        anomalies = tool.detect_anomalies(ratios, benchmarks)

        assert len(anomalies) == 2
        assert any('current_ratio' in anomaly and 'below industry minimum' in anomaly for anomaly in anomalies)
        assert any('debt_to_equity' in anomaly and 'exceeds industry maximum' in anomaly for anomaly in anomalies)

    def test_missing_data_handling(self):
        """Test handling of missing financial data"""
        tool = FinancialAnalysisTool()

        # Incomplete data
        incomplete_data = {
            'revenue': 500000,
            'net_income': 50000
            # Missing assets, liabilities, etc.
        }

        ratios = tool.calculate_liquidity_ratios(incomplete_data)
        assert len(ratios) == 0  # Should return empty dict when required fields missing

        profitability_ratios = tool.calculate_profitability_ratios(incomplete_data)
        assert 'net_profit_margin' in profitability_ratios
        assert 'return_on_assets' not in profitability_ratios


class TestRiskAssessmentTool:
    """Test RiskAssessmentTool functionality"""

    def test_assess_credit_risk_high_risk(self, sample_financial_data):
        """Test credit risk assessment - high risk scenario"""
        tool = RiskAssessmentTool()

        # Modify data to create high risk scenario
        high_risk_data = sample_financial_data.copy()
        high_risk_data['total_debt'] = 1200000  # High debt
        high_risk_data['shareholders_equity'] = 200000  # Low equity
        high_risk_data['ebit'] = 20000  # Low earnings
        high_risk_data['current_assets'] = 100000  # Low liquidity

        result = tool.assess_credit_risk(high_risk_data)

        assert result['credit_risk_score'] > 0.5
        assert result['risk_level'] in ['High', 'Critical']
        assert len(result['risk_factors']) >= 2

    def test_assess_credit_risk_low_risk(self, sample_financial_data):
        """Test credit risk assessment - low risk scenario"""
        tool = RiskAssessmentTool()

        # Use sample data which represents a healthy company
        result = tool.assess_credit_risk(sample_financial_data)

        assert result['credit_risk_score'] <= 0.5
        assert result['risk_level'] in ['Low', 'Medium']

    def test_assess_operational_risk(self):
        """Test operational risk assessment"""
        tool = RiskAssessmentTool()

        # High operational risk scenario
        operational_data = {
            'customer_concentration': 0.6,  # 60% revenue from top customer
            'geographic_concentration': 0.8,  # 80% in single geography
            'key_personnel_risk': True,
            'regulatory_changes': True
        }

        result = tool.assess_operational_risk(operational_data)

        assert result['operational_risk_score'] > 0.5
        assert len(result['risk_factors']) >= 3
        assert result['risk_level'] in ['High', 'Critical']

    def test_assess_market_risk(self):
        """Test market risk assessment"""
        tool = RiskAssessmentTool()

        market_data = {
            'variable_rate_debt_pct': 0.7,  # 70% variable rate debt
            'foreign_revenue_pct': 0.4,  # 40% foreign revenue
            'commodity_exposure': True,
            'beta': 1.8  # High beta
        }

        result = tool.assess_market_risk(market_data)

        assert result['market_risk_score'] > 0.5
        assert len(result['risk_factors']) >= 3

    def test_calculate_var(self):
        """Test Value at Risk calculation"""
        tool = RiskAssessmentTool()

        # Sample returns data
        returns = [-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.01, 0.01]

        var_95 = tool.calculate_var(returns, 0.95)
        var_99 = tool.calculate_var(returns, 0.99)

        # VaR should be negative (loss)
        assert var_95 < 0
        assert var_99 < 0
        # 99% VaR should be more extreme than 95% VaR
        assert var_99 <= var_95

    def test_stress_test_scenarios(self, sample_financial_data):
        """Test stress testing scenarios"""
        tool = RiskAssessmentTool()

        scenarios = tool.stress_test_scenarios(sample_financial_data)

        expected_scenarios = ['economic_downturn', 'interest_rate_shock', 'market_crash']
        for scenario in expected_scenarios:
            assert scenario in scenarios

        # Check economic downturn scenario
        econ_scenario = scenarios['economic_downturn']
        assert 'revenue_change' in econ_scenario
        assert econ_scenario['revenue_change'] < 0  # Should be negative


class TestAuditAgent:
    """Test audit agent functionality"""

    @pytest.mark.asyncio
    async def test_analyze_financial_ratios_tool(self, audit_context_sample):
        """Test the analyze_financial_ratios tool function"""
        from agents.definitions.audit_agent import analyze_financial_ratios

        # Create a mock run context
        class MockRunContext:
            def __init__(self, data):
                self.data = data

        ctx = MockRunContext(audit_context_sample)

        financial_data = {
            'total_assets': 1000000,
            'total_liabilities': 800000,
            'current_assets': 200000,
            'current_liabilities': 250000,  # Poor liquidity
            'revenue': 500000
        }

        result = await analyze_financial_ratios(ctx, financial_data)

        assert 'ratios' in result
        assert 'anomalies' in result
        assert 'risk_score' in result

        # Should detect liquidity issue
        assert any('liquidity' in anomaly.lower() for anomaly in result['anomalies'])

    @pytest.mark.asyncio
    async def test_detect_fraud_indicators_tool(self, audit_context_sample):
        """Test the detect_fraud_indicators tool function"""
        from agents.definitions.audit_agent import detect_fraud_indicators

        class MockRunContext:
            def __init__(self, data):
                self.data = data

        ctx = MockRunContext(audit_context_sample)

        # Create suspicious transaction data
        transaction_data = [
            {'amount': 10000, 'day_of_week': 0},  # Sunday transaction
            {'amount': 5000, 'day_of_week': 1},
            {'amount': 10000, 'day_of_week': 2},  # Duplicate amount
            {'amount': 15000, 'day_of_week': 3},
            {'amount': 10000, 'day_of_week': 4},  # Another duplicate
        ]

        result = await detect_fraud_indicators(ctx, transaction_data)

        assert 'fraud_indicators' in result
        assert 'risk_score' in result
        assert result['risk_score'] > 0
        assert len(result['fraud_indicators']) > 0


class TestComplianceAgent:
    """Test compliance agent functionality"""

    @pytest.mark.asyncio
    async def test_check_sox_compliance_tool(self, sample_compliance_data):
        """Test SOX compliance checking"""
        from agents.definitions.compliance_agent import check_sox_compliance

        class MockRunContext:
            def __init__(self, data):
                self.data = data

        ctx = MockRunContext(None)
        internal_controls = sample_compliance_data['internal_controls']

        result = await check_sox_compliance(ctx, internal_controls)

        assert 'sox_violations' in result
        assert 'compliance_score' in result
        assert result['compliance_score'] >= 0
        assert result['compliance_score'] <= 100

        # With good controls, should have high compliance score
        assert result['compliance_score'] >= 80

    @pytest.mark.asyncio
    async def test_sox_compliance_violations(self):
        """Test SOX compliance with violations"""
        from agents.definitions.compliance_agent import check_sox_compliance

        class MockRunContext:
            def __init__(self, data):
                self.data = data

        ctx = MockRunContext(None)

        # Missing certifications
        poor_controls = {
            'ceo_certification': False,
            'cfo_certification': False,
            'icfr_assessment': False,
            'disclosure_days': 10  # Too slow
        }

        result = await check_sox_compliance(ctx, poor_controls)

        assert len(result['sox_violations']) >= 3  # Should have multiple violations
        assert result['compliance_score'] < 50  # Should have low score

    @pytest.mark.asyncio
    async def test_validate_financial_disclosures_tool(self, sample_compliance_data):
        """Test financial disclosure validation"""
        from agents.definitions.compliance_agent import validate_financial_disclosures

        class MockRunContext:
            def __init__(self, data):
                self.data = data

        ctx = MockRunContext(None)
        financial_statements = sample_compliance_data['financial_statements']

        result = await validate_financial_disclosures(ctx, financial_statements)

        assert 'missing_disclosures' in result
        assert 'quality_issues' in result
        assert 'disclosure_completeness_score' in result

        # With complete disclosures, should have good score
        assert result['disclosure_completeness_score'] >= 90

    @pytest.mark.asyncio
    async def test_missing_disclosures(self):
        """Test detection of missing disclosures"""
        from agents.definitions.compliance_agent import validate_financial_disclosures

        class MockRunContext:
            def __init__(self, data):
                self.data = data

        ctx = MockRunContext(None)

        # Incomplete financial statements
        incomplete_statements = {
            'accounting_policies': 'Brief policies',
            # Missing most required disclosures
        }

        result = await validate_financial_disclosures(ctx, incomplete_statements)

        assert len(result['missing_disclosures']) >= 5  # Should find multiple missing items
        assert result['disclosure_completeness_score'] < 50