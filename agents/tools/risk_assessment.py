"""
Risk Assessment Tools for PydanticAI agents
"""

from typing import Dict, Any, List, Tuple
from pydantic import BaseModel
import numpy as np


class RiskAssessmentTool(BaseModel):
    """Tool for financial risk assessment"""

    name: str = "risk_assessment"
    description: str = "Assess various financial risks"

    def assess_credit_risk(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """Assess credit risk based on financial indicators"""
        risk_score = 0.0
        risk_factors = []

        # Debt-to-Equity Ratio Risk
        if 'total_debt' in financial_data and 'shareholders_equity' in financial_data:
            debt_to_equity = financial_data['total_debt'] / financial_data['shareholders_equity']
            if debt_to_equity > 2.0:
                risk_score += 0.3
                risk_factors.append(f"High debt-to-equity ratio: {debt_to_equity:.2f}")

        # Interest Coverage Risk
        if 'ebit' in financial_data and 'interest_expense' in financial_data:
            if financial_data['interest_expense'] > 0:
                interest_coverage = financial_data['ebit'] / financial_data['interest_expense']
                if interest_coverage < 2.5:
                    risk_score += 0.4
                    risk_factors.append(f"Low interest coverage ratio: {interest_coverage:.2f}")

        # Current Ratio Risk
        if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
            current_ratio = financial_data['current_assets'] / financial_data['current_liabilities']
            if current_ratio < 1.0:
                risk_score += 0.3
                risk_factors.append(f"Poor liquidity: Current ratio {current_ratio:.2f}")

        return {
            'credit_risk_score': min(risk_score, 1.0),
            'risk_level': self._categorize_risk(risk_score),
            'risk_factors': risk_factors
        }

    def assess_operational_risk(self, operational_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational risk factors"""
        risk_score = 0.0
        risk_factors = []

        # Revenue Concentration Risk
        if 'customer_concentration' in operational_data:
            top_customer_pct = operational_data['customer_concentration']
            if top_customer_pct > 0.3:  # 30% concentration threshold
                risk_score += 0.4
                risk_factors.append(f"High customer concentration: {top_customer_pct*100:.1f}%")

        # Geographic Concentration Risk
        if 'geographic_concentration' in operational_data:
            concentration = operational_data['geographic_concentration']
            if concentration > 0.5:  # 50% in single geography
                risk_score += 0.2
                risk_factors.append("High geographic concentration")

        # Key Personnel Risk
        if 'key_personnel_risk' in operational_data:
            if operational_data['key_personnel_risk']:
                risk_score += 0.3
                risk_factors.append("Dependency on key personnel")

        # Regulatory Risk
        if 'regulatory_changes' in operational_data:
            if operational_data['regulatory_changes']:
                risk_score += 0.2
                risk_factors.append("Pending regulatory changes")

        return {
            'operational_risk_score': min(risk_score, 1.0),
            'risk_level': self._categorize_risk(risk_score),
            'risk_factors': risk_factors
        }

    def assess_market_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market risk exposure"""
        risk_score = 0.0
        risk_factors = []

        # Interest Rate Risk
        if 'variable_rate_debt_pct' in market_data:
            var_debt_pct = market_data['variable_rate_debt_pct']
            if var_debt_pct > 0.5:  # 50% variable rate debt
                risk_score += 0.3
                risk_factors.append(f"High variable rate debt exposure: {var_debt_pct*100:.1f}%")

        # Foreign Exchange Risk
        if 'foreign_revenue_pct' in market_data:
            foreign_pct = market_data['foreign_revenue_pct']
            if foreign_pct > 0.3:  # 30% foreign revenue
                risk_score += 0.2
                risk_factors.append(f"Foreign exchange exposure: {foreign_pct*100:.1f}%")

        # Commodity Price Risk
        if 'commodity_exposure' in market_data:
            if market_data['commodity_exposure']:
                risk_score += 0.2
                risk_factors.append("Commodity price exposure")

        # Market Volatility
        if 'beta' in market_data:
            beta = market_data['beta']
            if beta > 1.5:
                risk_score += 0.3
                risk_factors.append(f"High market sensitivity (Beta: {beta:.2f})")

        return {
            'market_risk_score': min(risk_score, 1.0),
            'risk_level': self._categorize_risk(risk_score),
            'risk_factors': risk_factors
        }

    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        return np.percentile(returns_array, (1 - confidence_level) * 100)

    def stress_test_scenarios(self, base_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Run stress test scenarios"""
        scenarios = {}

        # Economic Downturn Scenario
        scenarios['economic_downturn'] = {
            'revenue_change': -0.25,
            'cost_increase': 0.15,
            'credit_loss_rate': 0.08
        }

        # Interest Rate Shock Scenario
        scenarios['interest_rate_shock'] = {
            'interest_rate_increase': 0.03,  # 300 basis points
            'debt_refinancing_cost': 0.02
        }

        # Market Crash Scenario
        scenarios['market_crash'] = {
            'asset_value_decline': -0.40,
            'liquidity_constraints': True
        }

        return scenarios

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level based on score"""
        if risk_score < 0.2:
            return "Low"
        elif risk_score < 0.5:
            return "Medium"
        elif risk_score < 0.8:
            return "High"
        else:
            return "Critical"