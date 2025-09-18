"""
Compliance checking service for regulatory validation
Implements SOX, GAAP, IFRS compliance checks
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime, timedelta
from decimal import Decimal
import json

logger = logging.getLogger(__name__)


class ComplianceChecker:
    """Service for checking regulatory compliance"""

    def __init__(self):
        self.regulations = {
            'SOX': self._load_sox_rules(),
            'GAAP': self._load_gaap_rules(), 
            'IFRS': self._load_ifrs_rules()
        }
        
    def _load_sox_rules(self) -> Dict[str, Any]:
        """Load SOX compliance rules"""
        return {
            'section_302': {
                'name': 'CEO/CFO Certification',
                'description': 'Principal executive and financial officers must certify financial statements',
                'checks': ['management_certification', 'financial_statement_accuracy']
            },
            'section_404': {
                'name': 'Internal Controls Assessment',
                'description': 'Management assessment of internal controls over financial reporting',
                'checks': ['internal_controls_effectiveness', 'material_weakness_disclosure']
            },
            'section_409': {
                'name': 'Real-time Disclosure',
                'description': 'Rapid disclosure of material changes in financial condition',
                'checks': ['timely_disclosure', 'material_change_reporting']
            }
        }

    def _load_gaap_rules(self) -> Dict[str, Any]:
        """Load GAAP compliance rules"""
        return {
            'revenue_recognition': {
                'name': 'Revenue Recognition',
                'description': 'Revenue must be recognized when earned',
                'checks': ['revenue_timing', 'performance_obligations', 'contract_modifications']
            },
            'expense_matching': {
                'name': 'Matching Principle',
                'description': 'Expenses must be matched with related revenues',
                'checks': ['expense_period_matching', 'accrual_accuracy']
            },
            'asset_valuation': {
                'name': 'Asset Valuation',
                'description': 'Assets must be valued appropriately',
                'checks': ['fair_value_measurement', 'impairment_testing', 'depreciation_methods']
            }
        }

    def _load_ifrs_rules(self) -> Dict[str, Any]:
        """Load IFRS compliance rules"""
        return {
            'ifrs_15': {
                'name': 'Revenue from Contracts with Customers',
                'description': 'Five-step model for revenue recognition',
                'checks': ['contract_identification', 'performance_obligations', 'transaction_price']
            },
            'ifrs_9': {
                'name': 'Financial Instruments',
                'description': 'Classification and measurement of financial instruments',
                'checks': ['financial_asset_classification', 'impairment_model', 'hedge_accounting']
            },
            'ifrs_16': {
                'name': 'Leases',
                'description': 'Lease accounting standards',
                'checks': ['lease_identification', 'right_of_use_assets', 'lease_liability']
            }
        }

    async def validate_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate transactions against all applicable regulations"""
        try:
            results = {
                'total_transactions': len(transactions),
                'validation_timestamp': datetime.utcnow().isoformat(),
                'violations': [],
                'warnings': [],
                'compliance_summary': {}
            }

            for regulation in ['SOX', 'GAAP', 'IFRS']:
                regulation_results = await self._validate_against_regulation(transactions, regulation)
                results['compliance_summary'][regulation] = regulation_results
                results['violations'].extend(regulation_results.get('violations', []))
                results['warnings'].extend(regulation_results.get('warnings', []))

            # Overall compliance assessment
            results['overall_compliance'] = self._assess_overall_compliance(results)
            
            return results

        except Exception as e:
            logger.error(f"Transaction validation failed: {e}")
            return {'error': str(e)}

    async def _validate_against_regulation(self, transactions: List[Dict[str, Any]], 
                                         regulation: str) -> Dict[str, Any]:
        """Validate transactions against specific regulation"""
        try:
            regulation_rules = self.regulations.get(regulation, {})
            results = {
                'regulation': regulation,
                'violations': [],
                'warnings': [],
                'checks_performed': 0,
                'compliance_rate': 0.0
            }

            total_checks = 0
            passed_checks = 0

            for rule_name, rule_config in regulation_rules.items():
                rule_results = await self._apply_rule(transactions, regulation, rule_name, rule_config)
                
                total_checks += rule_results['checks_performed']
                passed_checks += rule_results['passed_checks']
                
                results['violations'].extend(rule_results['violations'])
                results['warnings'].extend(rule_results['warnings'])

            results['checks_performed'] = total_checks
            results['compliance_rate'] = passed_checks / total_checks if total_checks > 0 else 1.0

            return results

        except Exception as e:
            logger.error(f"Regulation validation failed for {regulation}: {e}")
            return {'error': str(e)}

    async def _apply_rule(self, transactions: List[Dict[str, Any]], regulation: str,
                         rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific compliance rule to transactions"""
        try:
            results = {
                'violations': [],
                'warnings': [],
                'checks_performed': 0,
                'passed_checks': 0
            }

            # Dispatch to specific rule implementation
            if regulation == 'SOX':
                rule_results = await self._apply_sox_rule(transactions, rule_name, rule_config)
            elif regulation == 'GAAP':
                rule_results = await self._apply_gaap_rule(transactions, rule_name, rule_config)
            elif regulation == 'IFRS':
                rule_results = await self._apply_ifrs_rule(transactions, rule_name, rule_config)
            else:
                rule_results = {'violations': [], 'warnings': [], 'checks_performed': 0, 'passed_checks': 0}

            return rule_results

        except Exception as e:
            logger.error(f"Rule application failed for {rule_name}: {e}")
            return {'violations': [], 'warnings': [], 'checks_performed': 0, 'passed_checks': 0}

    async def _apply_sox_rule(self, transactions: List[Dict[str, Any]], 
                             rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SOX-specific compliance rules"""
        violations = []
        warnings = []
        checks_performed = 0
        passed_checks = 0

        if rule_name == 'section_302':
            # Check for management certification requirements
            for transaction in transactions:
                checks_performed += 1
                
                # Check for high-value transactions requiring certification
                amount = float(transaction.get('amount', 0))
                if amount > 1000000:  # $1M threshold
                    if not transaction.get('management_approved'):
                        violations.append({
                            'rule': 'SOX Section 302',
                            'transaction_id': transaction.get('id', 'unknown'),
                            'severity': 'high',
                            'description': 'High-value transaction without management certification',
                            'amount': amount,
                            'regulation': 'SOX'
                        })
                    else:
                        passed_checks += 1
                else:
                    passed_checks += 1

        elif rule_name == 'section_404':
            # Check internal controls
            for transaction in transactions:
                checks_performed += 1
                
                # Check for proper authorization
                if not transaction.get('authorized_by'):
                    violations.append({
                        'rule': 'SOX Section 404',
                        'transaction_id': transaction.get('id', 'unknown'),
                        'severity': 'medium',
                        'description': 'Transaction lacks proper authorization documentation',
                        'regulation': 'SOX'
                    })
                else:
                    passed_checks += 1

        return {
            'violations': violations,
            'warnings': warnings,
            'checks_performed': checks_performed,
            'passed_checks': passed_checks
        }

    async def _apply_gaap_rule(self, transactions: List[Dict[str, Any]], 
                              rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GAAP-specific compliance rules"""
        violations = []
        warnings = []
        checks_performed = 0
        passed_checks = 0

        if rule_name == 'revenue_recognition':
            for transaction in transactions:
                if transaction.get('type') == 'revenue':
                    checks_performed += 1
                    
                    # Check revenue recognition timing
                    transaction_date = transaction.get('date')
                    service_date = transaction.get('service_date')
                    
                    if transaction_date and service_date:
                        # Revenue should be recognized when service is performed
                        if transaction_date < service_date:
                            violations.append({
                                'rule': 'GAAP Revenue Recognition',
                                'transaction_id': transaction.get('id', 'unknown'),
                                'severity': 'medium',
                                'description': 'Revenue recognized before service performance',
                                'regulation': 'GAAP'
                            })
                        else:
                            passed_checks += 1
                    else:
                        warnings.append({
                            'rule': 'GAAP Revenue Recognition',
                            'transaction_id': transaction.get('id', 'unknown'),
                            'description': 'Missing date information for revenue recognition validation'
                        })

        elif rule_name == 'expense_matching':
            for transaction in transactions:
                if transaction.get('type') == 'expense':
                    checks_performed += 1
                    
                    # Check expense matching
                    related_revenue = transaction.get('related_revenue_id')
                    if not related_revenue and float(transaction.get('amount', 0)) > 10000:
                        warnings.append({
                            'rule': 'GAAP Matching Principle',
                            'transaction_id': transaction.get('id', 'unknown'),
                            'description': 'Large expense without clear revenue matching'
                        })
                    
                    passed_checks += 1

        return {
            'violations': violations,
            'warnings': warnings,
            'checks_performed': checks_performed,
            'passed_checks': passed_checks
        }

    async def _apply_ifrs_rule(self, transactions: List[Dict[str, Any]], 
                              rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply IFRS-specific compliance rules"""
        violations = []
        warnings = []
        checks_performed = 0
        passed_checks = 0

        if rule_name == 'ifrs_15':
            for transaction in transactions:
                if transaction.get('type') == 'revenue':
                    checks_performed += 1
                    
                    # Check for contract identification
                    if not transaction.get('contract_id'):
                        violations.append({
                            'rule': 'IFRS 15',
                            'transaction_id': transaction.get('id', 'unknown'),
                            'severity': 'medium',
                            'description': 'Revenue transaction without contract identification',
                            'regulation': 'IFRS'
                        })
                    else:
                        passed_checks += 1

        elif rule_name == 'ifrs_9':
            for transaction in transactions:
                if transaction.get('category') in ['financial_asset', 'financial_liability']:
                    checks_performed += 1
                    
                    # Check financial instrument classification
                    if not transaction.get('classification'):
                        violations.append({
                            'rule': 'IFRS 9',
                            'transaction_id': transaction.get('id', 'unknown'),
                            'severity': 'medium',
                            'description': 'Financial instrument without proper classification',
                            'regulation': 'IFRS'
                        })
                    else:
                        passed_checks += 1

        return {
            'violations': violations,
            'warnings': warnings,
            'checks_performed': checks_performed,
            'passed_checks': passed_checks
        }

    def _assess_overall_compliance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall compliance status"""
        try:
            total_violations = len(results.get('violations', []))
            total_warnings = len(results.get('warnings', []))
            total_transactions = results.get('total_transactions', 1)

            # Calculate compliance rates
            compliance_rates = {}
            for regulation, reg_results in results.get('compliance_summary', {}).items():
                compliance_rates[regulation] = reg_results.get('compliance_rate', 0.0)

            overall_compliance_rate = sum(compliance_rates.values()) / len(compliance_rates) if compliance_rates else 0.0

            # Determine compliance level
            if overall_compliance_rate >= 0.95 and total_violations == 0:
                compliance_level = 'excellent'
            elif overall_compliance_rate >= 0.85 and total_violations <= 2:
                compliance_level = 'good'
            elif overall_compliance_rate >= 0.70:
                compliance_level = 'acceptable'
            else:
                compliance_level = 'poor'

            return {
                'overall_compliance_rate': overall_compliance_rate,
                'compliance_level': compliance_level,
                'total_violations': total_violations,
                'total_warnings': total_warnings,
                'violation_rate': total_violations / total_transactions,
                'regulation_compliance': compliance_rates,
                'recommendations': self._generate_compliance_recommendations(results)
            }

        except Exception as e:
            logger.error(f"Overall compliance assessment failed: {e}")
            return {'error': str(e)}

    def _generate_compliance_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []
        
        try:
            violations = results.get('violations', [])
            warnings = results.get('warnings', [])
            
            # Group violations by regulation
            violation_by_reg = {}
            for violation in violations:
                reg = violation.get('regulation', 'unknown')
                if reg not in violation_by_reg:
                    violation_by_reg[reg] = []
                violation_by_reg[reg].append(violation)

            # Generate specific recommendations
            if violation_by_reg.get('SOX'):
                recommendations.append("Strengthen internal controls and management oversight processes")
                recommendations.append("Implement enhanced authorization requirements for high-value transactions")

            if violation_by_reg.get('GAAP'):
                recommendations.append("Review revenue recognition policies and procedures")
                recommendations.append("Improve expense matching documentation and controls")

            if violation_by_reg.get('IFRS'):
                recommendations.append("Enhance contract management and identification processes")
                recommendations.append("Review financial instrument classification procedures")

            if len(warnings) > len(violations):
                recommendations.append("Address data quality issues to enable complete compliance validation")

            if not recommendations:
                recommendations.append("Maintain current compliance monitoring and control procedures")

            return recommendations

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Review compliance results with regulatory specialists"]
