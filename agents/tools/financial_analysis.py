"""
Financial Analysis Tools for PydanticAI agents
"""

from typing import Dict, Any, List
from pydantic import BaseModel


class FinancialAnalysisTool(BaseModel):
    """Tool for financial statement analysis"""

    name: str = "financial_analysis"
    description: str = "Analyze financial statements and ratios"

    def calculate_liquidity_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        ratios = {}

        # Current Ratio
        if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
            ratios['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']

        # Quick Ratio
        if all(k in financial_data for k in ['current_assets', 'inventory', 'current_liabilities']):
            quick_assets = financial_data['current_assets'] - financial_data['inventory']
            ratios['quick_ratio'] = quick_assets / financial_data['current_liabilities']

        # Cash Ratio
        if 'cash' in financial_data and 'current_liabilities' in financial_data:
            ratios['cash_ratio'] = financial_data['cash'] / financial_data['current_liabilities']

        return ratios

    def calculate_profitability_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate profitability ratios"""
        ratios = {}

        # Gross Profit Margin
        if 'gross_profit' in financial_data and 'revenue' in financial_data:
            ratios['gross_profit_margin'] = financial_data['gross_profit'] / financial_data['revenue']

        # Net Profit Margin
        if 'net_income' in financial_data and 'revenue' in financial_data:
            ratios['net_profit_margin'] = financial_data['net_income'] / financial_data['revenue']

        # Return on Assets (ROA)
        if 'net_income' in financial_data and 'total_assets' in financial_data:
            ratios['return_on_assets'] = financial_data['net_income'] / financial_data['total_assets']

        # Return on Equity (ROE)
        if 'net_income' in financial_data and 'shareholders_equity' in financial_data:
            ratios['return_on_equity'] = financial_data['net_income'] / financial_data['shareholders_equity']

        return ratios

    def calculate_leverage_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate leverage/solvency ratios"""
        ratios = {}

        # Debt-to-Equity Ratio
        if 'total_debt' in financial_data and 'shareholders_equity' in financial_data:
            ratios['debt_to_equity'] = financial_data['total_debt'] / financial_data['shareholders_equity']

        # Debt-to-Assets Ratio
        if 'total_debt' in financial_data and 'total_assets' in financial_data:
            ratios['debt_to_assets'] = financial_data['total_debt'] / financial_data['total_assets']

        # Interest Coverage Ratio
        if 'ebit' in financial_data and 'interest_expense' in financial_data:
            ratios['interest_coverage'] = financial_data['ebit'] / financial_data['interest_expense']

        return ratios

    def detect_anomalies(self, ratios: Dict[str, float], industry_benchmarks: Dict[str, Dict[str, float]]) -> List[str]:
        """Detect anomalies by comparing to industry benchmarks"""
        anomalies = []

        for ratio_name, value in ratios.items():
            if ratio_name in industry_benchmarks:
                benchmark = industry_benchmarks[ratio_name]

                # Check if ratio is outside acceptable range
                if 'min' in benchmark and value < benchmark['min']:
                    anomalies.append(f"{ratio_name} ({value:.2f}) is below industry minimum ({benchmark['min']:.2f})")

                if 'max' in benchmark and value > benchmark['max']:
                    anomalies.append(f"{ratio_name} ({value:.2f}) exceeds industry maximum ({benchmark['max']:.2f})")

                # Check for significant deviation from median
                if 'median' in benchmark:
                    deviation = abs(value - benchmark['median']) / benchmark['median']
                    if deviation > 0.5:  # 50% deviation threshold
                        anomalies.append(f"{ratio_name} deviates significantly from industry median")

        return anomalies