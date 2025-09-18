"""
Risk calculation utility for financial audit
Implements risk scoring and categorization
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCalculator:
    """Calculator for comprehensive risk assessment"""

    def __init__(self):
        # Risk weight factors
        self.weights = {
            'statistical_deviation': 0.25,
            'regulatory_violations': 0.30,
            'anomaly_detection': 0.25,
            'volume_analysis': 0.10,
            'trend_analysis': 0.10
        }
        
        # Risk thresholds
        self.thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.85
        }

    def calculate_overall_risk(self, statistical_score: float = 0.0, 
                             regulatory_violations: int = 0,
                             anomaly_count: int = 0, 
                             total_transactions: int = 1,
                             additional_factors: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall risk score (0-1 scale)"""
        try:
            # Normalize inputs
            statistical_risk = self._normalize_statistical_score(statistical_score)
            regulatory_risk = self._normalize_violation_count(regulatory_violations, total_transactions)
            anomaly_risk = self._normalize_anomaly_count(anomaly_count, total_transactions)
            volume_risk = self._calculate_volume_risk(total_transactions)
            
            # Calculate weighted risk score
            risk_score = (
                statistical_risk * self.weights['statistical_deviation'] +
                regulatory_risk * self.weights['regulatory_violations'] +
                anomaly_risk * self.weights['anomaly_detection'] +
                volume_risk * self.weights['volume_analysis']
            )
            
            # Add additional factors if provided
            if additional_factors:
                for factor, value in additional_factors.items():
                    weight = self.weights.get(factor, 0.05)  # Default small weight
                    risk_score += value * weight
            
            # Ensure score is within bounds
            return max(0.0, min(1.0, risk_score))

        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return 0.5  # Default medium risk

    def _normalize_statistical_score(self, p_value: float) -> float:
        """Convert p-value to risk score (lower p-value = higher risk)"""
        if p_value <= 0:
            return 1.0
        elif p_value >= 1:
            return 0.0
        else:
            # Convert p-value to risk score using exponential decay
            return 1 - math.exp(-5 * (1 - p_value))

    def _normalize_violation_count(self, violations: int, total_transactions: int) -> float:
        """Normalize regulatory violations to risk score"""
        if total_transactions <= 0:
            return 0.0
        
        violation_rate = violations / total_transactions
        
        # Risk increases exponentially with violation rate
        if violation_rate == 0:
            return 0.0
        elif violation_rate >= 0.1:  # 10% or more violations is critical
            return 1.0
        else:
            return min(1.0, violation_rate * 10)  # Linear scaling up to 10%

    def _normalize_anomaly_count(self, anomalies: int, total_transactions: int) -> float:
        """Normalize anomaly count to risk score"""
        if total_transactions <= 0:
            return 0.0
        
        anomaly_rate = anomalies / total_transactions
        
        # Expected anomaly rate is typically 1-5%
        expected_rate = 0.02  # 2%
        
        if anomaly_rate <= expected_rate:
            return anomaly_rate / expected_rate * 0.3  # Low risk if within expected
        else:
            excess_rate = anomaly_rate - expected_rate
            return 0.3 + min(0.7, excess_rate * 20)  # Higher risk for excess anomalies

    def _calculate_volume_risk(self, total_transactions: int) -> float:
        """Calculate risk based on transaction volume"""
        # Very low volume might indicate incomplete data
        # Very high volume might indicate lack of controls
        
        if total_transactions < 10:
            return 0.7  # High risk for very low volume
        elif total_transactions < 100:
            return 0.3  # Medium risk for low volume
        elif total_transactions > 10000:
            return 0.4  # Slightly elevated risk for very high volume
        else:
            return 0.1  # Low risk for normal volume

    def categorize_risk(self, risk_score: float) -> RiskLevel:
        """Categorize numeric risk score into risk level"""
        if risk_score >= self.thresholds['critical']:
            return RiskLevel.CRITICAL
        elif risk_score >= self.thresholds['high']:
            return RiskLevel.HIGH
        elif risk_score >= self.thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def calculate_finding_risk(self, finding: Dict[str, Any]) -> float:
        """Calculate risk score for individual finding"""
        try:
            base_score = 0.0
            
            # Severity factor
            severity = finding.get('severity', 'low').lower()
            severity_scores = {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
            base_score += severity_scores.get(severity, 0.3)
            
            # Confidence factor
            confidence = finding.get('confidence_score', 0.5)
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Scale 0.5-1.0
            
            # Financial impact factor
            financial_impact = finding.get('financial_impact', 0)
            if financial_impact:
                # Log scale for financial impact (assuming impact in currency units)
                impact_factor = min(0.3, math.log10(max(1, financial_impact)) / 10)
                base_score += impact_factor
            
            # Regulatory reference factor
            if finding.get('regulatory_reference'):
                base_score += 0.1  # Regulatory implications increase risk
            
            # Apply confidence multiplier
            final_score = base_score * confidence_multiplier
            
            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logger.error(f"Finding risk calculation failed: {e}")
            return 0.5

    def calculate_portfolio_risk(self, findings: List[Dict[str, Any]], 
                               anomalies: List[Dict[str, Any]],
                               financial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk assessment"""
        try:
            # Individual finding risks
            finding_risks = [self.calculate_finding_risk(f) for f in findings]
            
            # Anomaly risks
            anomaly_risks = []
            for anomaly in anomalies:
                confidence = anomaly.get('confidence', 0.5)
                severity_map = {'high': 0.8, 'medium': 0.5, 'low': 0.3, 'very_low': 0.1}
                severity_score = severity_map.get(anomaly.get('severity', 'low'), 0.3)
                anomaly_risk = confidence * severity_score
                anomaly_risks.append(anomaly_risk)
            
            # Aggregate risks
            max_finding_risk = max(finding_risks) if finding_risks else 0.0
            avg_finding_risk = np.mean(finding_risks) if finding_risks else 0.0
            max_anomaly_risk = max(anomaly_risks) if anomaly_risks else 0.0
            avg_anomaly_risk = np.mean(anomaly_risks) if anomaly_risks else 0.0
            
            # Calculate overall portfolio risk
            portfolio_risk = (
                max_finding_risk * 0.4 +
                avg_finding_risk * 0.2 +
                max_anomaly_risk * 0.3 +
                avg_anomaly_risk * 0.1
            )
            
            # Risk concentration (many findings in same area increases risk)
            concentration_risk = self._calculate_concentration_risk(findings)
            portfolio_risk = min(1.0, portfolio_risk + concentration_risk)
            
            return {
                'portfolio_risk_score': portfolio_risk,
                'risk_level': self.categorize_risk(portfolio_risk).value,
                'finding_count': len(findings),
                'anomaly_count': len(anomalies),
                'max_finding_risk': max_finding_risk,
                'avg_finding_risk': avg_finding_risk,
                'max_anomaly_risk': max_anomaly_risk,
                'avg_anomaly_risk': avg_anomaly_risk,
                'concentration_risk': concentration_risk,
                'risk_distribution': self._calculate_risk_distribution(finding_risks + anomaly_risks)
            }

        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_concentration_risk(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate risk from concentration of findings in specific areas"""
        try:
            if not findings:
                return 0.0
            
            # Group findings by category
            categories = {}
            for finding in findings:
                category = finding.get('category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(finding)
            
            # Calculate concentration risk
            total_findings = len(findings)
            concentration_risk = 0.0
            
            for category, category_findings in categories.items():
                category_proportion = len(category_findings) / total_findings
                
                # Risk increases with concentration
                if category_proportion > 0.5:  # More than 50% in one category
                    concentration_risk += (category_proportion - 0.5) * 0.4
            
            return min(0.2, concentration_risk)  # Cap at 20% additional risk

        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return 0.0

    def _calculate_risk_distribution(self, risk_scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of risks across levels"""
        if not risk_scores:
            return {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for score in risk_scores:
            level = self.categorize_risk(score).value
            distribution[level] += 1
        
        return distribution

    def calculate_trend_risk(self, historical_scores: List[float], 
                           time_periods: List[str]) -> Dict[str, Any]:
        """Calculate risk based on historical trends"""
        try:
            if len(historical_scores) < 2:
                return {'trend_risk': 0.0, 'trend_direction': 'insufficient_data'}
            
            # Calculate trend
            x = np.arange(len(historical_scores))
            y = np.array(historical_scores)
            
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Risk increases with positive trend (increasing risk over time)
            trend_risk = max(0.0, slope * 2)  # Amplify trend effect
            trend_risk = min(0.3, trend_risk)  # Cap at 30% additional risk
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
            
            # Calculate volatility
            volatility = np.std(historical_scores) if len(historical_scores) > 1 else 0.0
            
            return {
                'trend_risk': trend_risk,
                'trend_direction': trend_direction,
                'trend_slope': slope,
                'volatility': volatility,
                'latest_score': historical_scores[-1] if historical_scores else 0.0,
                'score_range': max(historical_scores) - min(historical_scores) if historical_scores else 0.0
            }

        except Exception as e:
            logger.error(f"Trend risk calculation failed: {e}")
            return {'error': str(e)}

    def generate_risk_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        try:
            risk_level = risk_assessment.get('risk_level', 'medium')
            portfolio_risk = risk_assessment.get('portfolio_risk_score', 0.5)
            
            # General recommendations based on risk level
            if risk_level == 'critical':
                recommendations.extend([
                    "Immediate escalation to senior management required",
                    "Consider suspending related transactions pending investigation",
                    "Engage external audit specialists if needed"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Prioritize investigation of high-risk findings",
                    "Implement enhanced monitoring controls",
                    "Schedule follow-up review within 30 days"
                ])
            elif risk_level == 'medium':
                recommendations.extend([
                    "Review and validate medium-risk findings",
                    "Consider implementing additional controls",
                    "Monitor for trend development"
                ])
            else:
                recommendations.append("Continue routine monitoring and controls")
            
            # Specific recommendations based on risk factors
            concentration_risk = risk_assessment.get('concentration_risk', 0.0)
            if concentration_risk > 0.1:
                recommendations.append("Address concentration of issues in specific areas")
            
            finding_count = risk_assessment.get('finding_count', 0)
            if finding_count > 10:
                recommendations.append("Consider process improvements to reduce finding frequency")
            
            max_finding_risk = risk_assessment.get('max_finding_risk', 0.0)
            if max_finding_risk > 0.8:
                recommendations.append("Focus immediate attention on highest-risk finding")
            
            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Review risk assessment and consult with audit team"]
