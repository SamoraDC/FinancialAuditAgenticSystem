"""
Statistical Analysis Integration Service for LangGraph Workflow.

This module provides integration between the Statistical Analysis Engine
and the LangGraph audit workflow, handling data preparation, analysis
orchestration, and result integration.
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from .statistical_analysis import (
    StatisticalAnalysisEngine,
    BenfordAnalysisResult,
    ZipfAnalysisResult,
    AnomalyDetectionResult,
    RLAnomalyDetectionResult,
    StatisticalMethod,
    AnomalyType
)
from .statistical_utils import (
    StatisticalConstants,
    BenfordUtils,
    ZipfUtils,
    AnomalyUtils,
    RiskScoringUtils,
    StatisticalValidation
)
from ..models.audit_models import AuditState, AuditFinding, RiskLevel
from ..core.config import settings

logger = logging.getLogger(__name__)


class StatisticalAnalysisOrchestrator:
    """
    Orchestrates statistical analysis for the LangGraph audit workflow.

    This class acts as the bridge between the audit workflow and the
    statistical analysis engine, managing data preparation, analysis
    execution, and result integration.
    """

    def __init__(self):
        """Initialize the statistical analysis orchestrator."""
        self.engine = StatisticalAnalysisEngine(
            confidence_level=getattr(settings, 'STATISTICAL_CONFIDENCE_LEVEL', 0.95),
            default_risk_threshold=getattr(settings, 'DEFAULT_RISK_THRESHOLD', 0.7),
            enable_rl_training=getattr(settings, 'ENABLE_RL_TRAINING', False)
        )
        self.analysis_cache = {}
        logger.info("Statistical Analysis Orchestrator initialized")

    async def perform_comprehensive_analysis(self,
                                           audit_state: AuditState) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on audit data.

        Args:
            audit_state: Current audit state from LangGraph

        Returns:
            Dictionary containing all statistical analysis results
        """
        logger.info(f"Starting comprehensive statistical analysis for session {audit_state.session_id}")

        try:
            # Prepare data from audit state
            prepared_data = await self._prepare_analysis_data(audit_state)

            # Validate data quality
            data_quality = self._validate_data_quality(prepared_data)

            if data_quality['overall_quality_score'] < 0.3:
                logger.warning(f"Low data quality score: {data_quality['overall_quality_score']}")

            # Execute parallel statistical analyses
            analysis_results = await self._execute_parallel_analyses(prepared_data, audit_state)

            # Calculate comprehensive risk score
            comprehensive_results = self.engine.calculate_comprehensive_risk_score(
                analysis_results['individual_results']
            )

            # Generate audit findings
            findings = self.engine.generate_audit_findings(
                comprehensive_results, audit_state.session_id
            )

            # Prepare final results
            final_results = {
                'session_id': audit_state.session_id,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'data_quality': data_quality,
                'individual_analyses': analysis_results,
                'comprehensive_assessment': comprehensive_results,
                'audit_findings': [finding.dict() for finding in findings],
                'recommendations': self._generate_prioritized_recommendations(
                    comprehensive_results, analysis_results
                ),
                'next_steps': self._determine_next_steps(comprehensive_results),
                'metadata': {
                    'engine_version': '1.0.0',
                    'methods_used': len(analysis_results['individual_results']),
                    'processing_time_seconds': analysis_results.get('total_processing_time', 0)
                }
            }

            # Cache results for future reference
            self.analysis_cache[audit_state.session_id] = final_results

            logger.info(f"Comprehensive analysis completed for session {audit_state.session_id}")
            return final_results

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}", exc_info=True)
            raise

    async def _prepare_analysis_data(self, audit_state: AuditState) -> Dict[str, Any]:
        """
        Prepare and structure data for statistical analysis.

        Args:
            audit_state: Current audit state

        Returns:
            Dictionary containing prepared data for different analysis types
        """
        logger.debug("Preparing analysis data")

        prepared_data = {
            'financial_amounts': [],
            'text_data': [],
            'transaction_dataframe': None,
            'vendor_data': [],
            'temporal_data': [],
            'account_balances': []
        }

        # Extract financial amounts from various sources
        if audit_state.extracted_data:
            # From invoices
            invoices = audit_state.extracted_data.get('invoices', [])
            for invoice in invoices:
                if isinstance(invoice, dict):
                    amount = invoice.get('amount', 0)
                    if amount and amount > 0:
                        prepared_data['financial_amounts'].append(float(amount))

                    # Extract text data
                    vendor_name = invoice.get('vendor_name', '')
                    description = invoice.get('description', '')
                    if vendor_name:
                        prepared_data['text_data'].append(vendor_name)
                    if description:
                        prepared_data['text_data'].append(description)

            # From balance sheets
            balance_sheets = audit_state.extracted_data.get('balance_sheets', [])
            for bs in balance_sheets:
                if isinstance(bs, dict):
                    # Extract asset amounts
                    current_assets = bs.get('current_assets', {})
                    for asset, amount in current_assets.items():
                        if amount and amount > 0:
                            prepared_data['financial_amounts'].append(float(amount))
                            prepared_data['account_balances'].append({
                                'account': asset,
                                'amount': float(amount),
                                'type': 'current_asset'
                            })

                    # Extract liability amounts
                    current_liabilities = bs.get('current_liabilities', {})
                    for liability, amount in current_liabilities.items():
                        if amount and amount > 0:
                            prepared_data['financial_amounts'].append(float(amount))
                            prepared_data['account_balances'].append({
                                'account': liability,
                                'amount': float(amount),
                                'type': 'current_liability'
                            })

        # Create transaction DataFrame for anomaly detection
        if prepared_data['financial_amounts']:
            # Generate synthetic transaction features for demonstration
            n_transactions = len(prepared_data['financial_amounts'])

            transaction_data = {
                'amount': prepared_data['financial_amounts'],
                'log_amount': [np.log(max(amt, 0.01)) for amt in prepared_data['financial_amounts']],
                'amount_zscore': [],
                'vendor_id': np.random.randint(1, min(100, n_transactions), n_transactions),
                'transaction_hour': np.random.randint(0, 24, n_transactions),
                'day_of_week': np.random.randint(0, 7, n_transactions),
                'approval_level': np.random.choice([1, 2, 3], n_transactions, p=[0.7, 0.25, 0.05])
            }

            # Calculate z-scores for amounts
            amounts = np.array(transaction_data['amount'])
            if len(amounts) > 1 and np.std(amounts) > 0:
                transaction_data['amount_zscore'] = ((amounts - np.mean(amounts)) / np.std(amounts)).tolist()
            else:
                transaction_data['amount_zscore'] = [0.0] * len(amounts)

            prepared_data['transaction_dataframe'] = pd.DataFrame(transaction_data)

        logger.debug(f"Prepared data summary: {len(prepared_data['financial_amounts'])} amounts, "
                    f"{len(prepared_data['text_data'])} text entries")

        return prepared_data

    def _validate_data_quality(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality of prepared data for analysis.

        Args:
            prepared_data: Prepared analysis data

        Returns:
            Data quality assessment
        """
        quality_report = {
            'financial_data_quality': {},
            'text_data_quality': {},
            'overall_quality_score': 0.0,
            'warnings': [],
            'recommendations': []
        }

        # Validate financial data
        financial_amounts = prepared_data.get('financial_amounts', [])
        if financial_amounts:
            sample_size_valid, size_message = StatisticalValidation.validate_sample_size(
                len(financial_amounts), minimum_required=30
            )

            quality_report['financial_data_quality'] = {
                'sample_size': len(financial_amounts),
                'sample_size_adequate': sample_size_valid,
                'sample_size_message': size_message,
                'min_amount': min(financial_amounts),
                'max_amount': max(financial_amounts),
                'mean_amount': np.mean(financial_amounts),
                'zero_amounts': sum(1 for x in financial_amounts if x == 0),
                'negative_amounts': sum(1 for x in financial_amounts if x < 0)
            }

            if not sample_size_valid:
                quality_report['warnings'].append(size_message)
        else:
            quality_report['warnings'].append("No financial amounts found for analysis")

        # Validate text data
        text_data = prepared_data.get('text_data', [])
        if text_data:
            total_words = sum(len(text.split()) for text in text_data)
            quality_report['text_data_quality'] = {
                'text_entries': len(text_data),
                'total_words': total_words,
                'average_words_per_entry': total_words / len(text_data) if text_data else 0,
                'empty_entries': sum(1 for text in text_data if not text.strip())
            }
        else:
            quality_report['warnings'].append("No text data found for Zipf analysis")

        # Validate transaction DataFrame
        df = prepared_data.get('transaction_dataframe')
        if df is not None and not df.empty:
            df_quality = StatisticalValidation.check_data_quality(df)
            quality_report['dataframe_quality'] = df_quality
        else:
            quality_report['warnings'].append("No structured transaction data available")

        # Calculate overall quality score
        scores = []

        if quality_report['financial_data_quality']:
            fin_score = 1.0
            if not quality_report['financial_data_quality']['sample_size_adequate']:
                fin_score *= 0.5
            if quality_report['financial_data_quality']['zero_amounts'] > 0:
                fin_score *= 0.9
            scores.append(fin_score)

        if quality_report['text_data_quality']:
            text_score = min(quality_report['text_data_quality']['total_words'] / 100, 1.0)
            scores.append(text_score)

        if quality_report.get('dataframe_quality'):
            scores.append(quality_report['dataframe_quality']['overall_quality_score'])

        quality_report['overall_quality_score'] = np.mean(scores) if scores else 0.0

        return quality_report

    async def _execute_parallel_analyses(self,
                                       prepared_data: Dict[str, Any],
                                       audit_state: AuditState) -> Dict[str, Any]:
        """
        Execute multiple statistical analyses in parallel.

        Args:
            prepared_data: Prepared analysis data
            audit_state: Current audit state

        Returns:
            Dictionary containing all analysis results
        """
        logger.debug("Executing parallel statistical analyses")

        start_time = datetime.utcnow()
        analysis_tasks = []
        individual_results = []

        # Benford's Law analysis (first digit)
        if prepared_data['financial_amounts']:
            analysis_tasks.append(
                self._run_benford_analysis(prepared_data['financial_amounts'], 1)
            )

        # Benford's Law analysis (second digit)
        if len(prepared_data['financial_amounts']) > 50:  # Need more data for second digit
            analysis_tasks.append(
                self._run_benford_analysis(prepared_data['financial_amounts'], 2)
            )

        # Newcomb-Benford analysis
        if prepared_data['financial_amounts']:
            analysis_tasks.append(
                self._run_newcomb_benford_analysis(prepared_data['financial_amounts'])
            )

        # Zipf's Law analysis
        if prepared_data['text_data']:
            analysis_tasks.append(
                self._run_zipf_analysis(prepared_data['text_data'])
            )

        # Isolation Forest anomaly detection
        df = prepared_data.get('transaction_dataframe')
        if df is not None and not df.empty and len(df) > 10:
            analysis_tasks.append(
                self._run_isolation_forest_analysis(df)
            )

        # PyOD ensemble anomaly detection
        if df is not None and not df.empty and len(df) > 20:
            analysis_tasks.append(
                self._run_pyod_ensemble_analysis(df)
            )

        # Reinforcement Learning anomaly detection (if enabled)
        if (getattr(settings, 'ENABLE_RL_ANALYSIS', False) and
            df is not None and not df.empty and len(df) > 50):
            analysis_tasks.append(
                self._run_rl_anomaly_analysis(df)
            )

        # Execute all analyses concurrently
        if analysis_tasks:
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Process results and handle exceptions
            for i, result in enumerate(analysis_results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis task {i} failed: {str(result)}")
                else:
                    individual_results.append(result)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        return {
            'individual_results': individual_results,
            'total_processing_time': processing_time,
            'analyses_completed': len(individual_results),
            'analyses_attempted': len(analysis_tasks),
            'execution_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'success_rate': len(individual_results) / len(analysis_tasks) if analysis_tasks else 0
            }
        }

    async def _run_benford_analysis(self,
                                  amounts: List[float],
                                  digit_position: int) -> BenfordAnalysisResult:
        """Run Benford's Law analysis asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.analyze_benfords_law, amounts, digit_position
        )

    async def _run_newcomb_benford_analysis(self,
                                          amounts: List[float]) -> BenfordAnalysisResult:
        """Run Newcomb-Benford's Law analysis asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.analyze_newcomb_benford_law, amounts
        )

    async def _run_zipf_analysis(self,
                               text_data: List[str]) -> ZipfAnalysisResult:
        """Run Zipf's Law analysis asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.analyze_zipfs_law, text_data
        )

    async def _run_isolation_forest_analysis(self,
                                           df: pd.DataFrame) -> AnomalyDetectionResult:
        """Run Isolation Forest analysis asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.detect_anomalies_isolation_forest, df
        )

    async def _run_pyod_ensemble_analysis(self,
                                        df: pd.DataFrame) -> AnomalyDetectionResult:
        """Run PyOD ensemble analysis asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.detect_anomalies_pyod_ensemble, df
        )

    async def _run_rl_anomaly_analysis(self,
                                     df: pd.DataFrame) -> RLAnomalyDetectionResult:
        """Run RL anomaly detection analysis asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.detect_anomalies_reinforcement_learning, df, "DQN", 100
        )

    def _generate_prioritized_recommendations(self,
                                            comprehensive_results: Dict[str, Any],
                                            analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate prioritized recommendations based on analysis results.

        Args:
            comprehensive_results: Comprehensive risk assessment
            analysis_results: Individual analysis results

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Extract risk level and scores
        risk_level = comprehensive_results.get('risk_level', 'low')
        overall_score = comprehensive_results.get('overall_risk_score', 0.0)
        method_scores = comprehensive_results.get('method_scores', {})

        # Critical and High Risk Recommendations
        if risk_level in ['critical', 'high']:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Immediate Action',
                'title': 'Emergency Review Required',
                'description': f'Overall risk score of {overall_score:.2f} requires immediate attention',
                'action_items': [
                    'Halt current processing pending review',
                    'Escalate to senior management',
                    'Initiate forensic investigation',
                    'Review internal controls'
                ],
                'timeline': '24 hours',
                'risk_mitigation': 'Prevents potential financial losses and compliance violations'
            })

        # Method-specific recommendations
        for method, score in method_scores.items():
            if score > 0.7:
                if method == 'benford':
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Digit Analysis',
                        'title': 'Benford\'s Law Deviation Investigation',
                        'description': f'Significant deviation from expected digit patterns (score: {score:.2f})',
                        'action_items': [
                            'Review transactions with unusual first digits',
                            'Analyze for round number bias',
                            'Investigate potential data manipulation',
                            'Cross-reference with vendor patterns'
                        ],
                        'timeline': '3-5 days',
                        'risk_mitigation': 'Detects potential fraud or data manipulation'
                    })

                elif method == 'isolation_forest' or method == 'pyod_ensemble':
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Anomaly Detection',
                        'title': 'Statistical Outlier Investigation',
                        'description': f'Machine learning models detected significant anomalies (score: {score:.2f})',
                        'action_items': [
                            'Review flagged transactions in detail',
                            'Verify transaction authenticity',
                            'Check approval processes',
                            'Analyze vendor relationships'
                        ],
                        'timeline': '2-4 days',
                        'risk_mitigation': 'Identifies unusual patterns that may indicate errors or fraud'
                    })

                elif method == 'reinforcement_learning':
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'Adaptive Analysis',
                        'title': 'AI-Detected Behavioral Anomalies',
                        'description': f'Reinforcement learning model flagged unusual patterns (score: {score:.2f})',
                        'action_items': [
                            'Review AI-flagged transactions',
                            'Provide feedback to improve model',
                            'Monitor for pattern evolution',
                            'Consider additional training data'
                        ],
                        'timeline': '1 week',
                        'risk_mitigation': 'Adapts to emerging fraud patterns and improves detection'
                    })

        # Data Quality Recommendations
        success_rate = analysis_results.get('execution_summary', {}).get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Data Quality',
                'title': 'Improve Data Quality for Analysis',
                'description': f'Analysis success rate of {success_rate:.1%} indicates data quality issues',
                'action_items': [
                    'Review data extraction processes',
                    'Implement data validation checks',
                    'Standardize data formats',
                    'Enhance data collection procedures'
                ],
                'timeline': '2 weeks',
                'risk_mitigation': 'Improves reliability and accuracy of future analyses'
            })

        # Ongoing Monitoring Recommendations
        recommendations.append({
            'priority': 'LOW',
            'category': 'Continuous Monitoring',
            'title': 'Establish Ongoing Statistical Monitoring',
            'description': 'Implement regular statistical analysis as part of audit procedures',
            'action_items': [
                'Schedule monthly statistical reviews',
                'Set up automated anomaly alerts',
                'Train audit staff on statistical indicators',
                'Establish baseline metrics for comparison'
            ],
            'timeline': '1 month',
            'risk_mitigation': 'Provides early warning system for potential issues'
        })

        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

        return recommendations

    def _determine_next_steps(self, comprehensive_results: Dict[str, Any]) -> List[str]:
        """
        Determine next steps in the audit workflow based on analysis results.

        Args:
            comprehensive_results: Comprehensive risk assessment

        Returns:
            List of next steps for the workflow
        """
        risk_level = comprehensive_results.get('risk_level', 'low')
        overall_score = comprehensive_results.get('overall_risk_score', 0.0)

        next_steps = []

        if risk_level == 'critical':
            next_steps.extend([
                'pause_workflow',
                'trigger_human_review_mcp',
                'escalate_to_management',
                'prepare_emergency_report'
            ])
        elif risk_level == 'high':
            next_steps.extend([
                'trigger_human_review_mcp',
                'deep_dive_investigation',
                'expand_sample_size',
                'additional_testing'
            ])
        elif risk_level == 'medium':
            next_steps.extend([
                'flag_for_review',
                'continue_standard_procedures',
                'monitor_trends',
                'document_findings'
            ])
        else:
            next_steps.extend([
                'continue_standard_procedures',
                'document_findings',
                'proceed_to_next_phase'
            ])

        # Add specific next steps based on method results
        method_scores = comprehensive_results.get('method_scores', {})

        if method_scores.get('benford', 0) > 0.7:
            next_steps.append('detailed_digit_analysis')

        if any(method_scores.get(method, 0) > 0.7
               for method in ['isolation_forest', 'pyod_ensemble']):
            next_steps.append('anomaly_investigation')

        if method_scores.get('reinforcement_learning', 0) > 0.7:
            next_steps.append('adaptive_model_review')

        return list(set(next_steps))  # Remove duplicates

    async def get_analysis_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of statistical analysis results for a session.

        Args:
            session_id: Audit session identifier

        Returns:
            Analysis summary or None if not found
        """
        if session_id in self.analysis_cache:
            full_results = self.analysis_cache[session_id]

            summary = {
                'session_id': session_id,
                'analysis_timestamp': full_results['analysis_timestamp'],
                'overall_risk_score': full_results['comprehensive_assessment']['overall_risk_score'],
                'risk_level': full_results['comprehensive_assessment']['risk_level'],
                'confidence_score': full_results['comprehensive_assessment']['confidence_score'],
                'methods_used': full_results['metadata']['methods_used'],
                'key_findings': len(full_results['audit_findings']),
                'critical_recommendations': len([
                    r for r in full_results['recommendations']
                    if r.get('priority') == 'CRITICAL'
                ]),
                'data_quality_score': full_results['data_quality']['overall_quality_score'],
                'next_steps': full_results['next_steps']
            }

            return summary

        return None

    async def update_analysis_with_feedback(self,
                                          session_id: str,
                                          feedback: Dict[str, Any]) -> bool:
        """
        Update analysis results with human feedback for RL model improvement.

        Args:
            session_id: Audit session identifier
            feedback: Human feedback on analysis results

        Returns:
            True if update successful, False otherwise
        """
        try:
            if session_id in self.analysis_cache:
                # Update cached results with feedback
                self.analysis_cache[session_id]['human_feedback'] = feedback
                self.analysis_cache[session_id]['feedback_timestamp'] = datetime.utcnow().isoformat()

                # If RL analysis was performed, use feedback for model improvement
                # This would typically involve retraining or fine-tuning the RL model
                logger.info(f"Updated analysis results with feedback for session {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating analysis with feedback: {str(e)}")
            return False


# Global orchestrator instance
statistical_orchestrator = StatisticalAnalysisOrchestrator()