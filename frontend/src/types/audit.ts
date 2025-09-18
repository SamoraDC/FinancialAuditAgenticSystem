// Audit Types and Interfaces

export interface AuditSession {
  id: string
  thread_id: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'paused'
  created_at: string
  updated_at: string
  progress: number
  current_step: string
  risk_score?: number
  findings: AuditFinding[]
  documents: UploadedDocument[]
}

export interface AuditFinding {
  id: string
  type: 'anomaly' | 'compliance_violation' | 'statistical_outlier' | 'regulatory_issue'
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  evidence: string[]
  recommended_action: string
  confidence_score: number
  detected_at: string
  reviewed: boolean
  human_validation?: {
    validated_by: string
    validated_at: string
    decision: 'approve' | 'reject' | 'needs_investigation'
    comments: string
  }
}

export interface UploadedDocument {
  id: string
  filename: string
  file_type: string
  file_size: number
  upload_status: 'uploading' | 'processing' | 'completed' | 'failed'
  processing_progress: number
  extracted_data?: any
  error_message?: string
  uploaded_at: string
}

export interface TransactionNode {
  id: string
  entity_name: string
  entity_type: 'company' | 'individual' | 'bank' | 'government'
  total_amount: number
  transaction_count: number
  risk_score: number
  anomaly_flags: string[]
  coordinates?: { x: number; y: number }
}

export interface TransactionEdge {
  id: string
  source: string
  target: string
  transaction_amount: number
  transaction_date: string
  transaction_type: string
  is_anomalous: boolean
  risk_indicators: string[]
}

export interface TransactionGraph {
  nodes: TransactionNode[]
  edges: TransactionEdge[]
}

export interface StatisticalAnalysis {
  benford_law: {
    expected_distribution: number[]
    actual_distribution: number[]
    chi_square_score: number
    p_value: number
    is_suspicious: boolean
  }
  zipf_law: {
    expected_distribution: number[]
    actual_distribution: number[]
    deviation_score: number
    is_suspicious: boolean
  }
  outlier_detection: {
    outliers: Array<{
      transaction_id: string
      amount: number
      z_score: number
      deviation_type: string
    }>
    total_outliers: number
    outlier_rate: number
  }
  fraud_indicators: {
    duplicate_transactions: number
    round_number_bias: number
    time_pattern_anomalies: number
    vendor_concentration_risk: number
  }
}

export interface RealTimeUpdate {
  audit_id: string
  thread_id: string
  timestamp: string
  event_type: 'progress_update' | 'finding_detected' | 'human_intervention_required' | 'analysis_complete'
  data: {
    progress?: number
    current_step?: string
    new_finding?: AuditFinding
    intervention_prompt?: MCPInterventionPrompt
    analysis_result?: any
  }
}

export interface MCPInterventionPrompt {
  id: string
  type: 'anomaly_review' | 'compliance_decision' | 'risk_assessment'
  title: string
  description: string
  context: any
  options: Array<{
    id: string
    label: string
    value: string
  }>
  required_fields: Array<{
    id: string
    label: string
    type: 'text' | 'select' | 'radio' | 'checkbox'
    required: boolean
    options?: string[]
  }>
  deadline?: string
}

export interface WebSocketMessage {
  type: 'audit_update' | 'ping' | 'error'
  payload: any
}

// Risk Assessment Types
export interface RiskMetrics {
  overall_risk_score: number
  credit_risk: number
  operational_risk: number
  market_risk: number
  compliance_risk: number
  fraud_risk: number
}

export interface ComplianceStatus {
  sox_score: number
  gaap_compliance: number
  ifrs_compliance: number
  disclosure_completeness: number
  internal_controls_effectiveness: number
}

// API Response Types
export interface APIResponse<T> {
  success: boolean
  data: T
  message?: string
  errors?: string[]
}

export interface AuditStartResponse {
  audit_id: string
  thread_id: string
  status: string
  message: string
}

// File Upload Types
export interface FileUploadProgress {
  file: File
  progress: number
  status: 'uploading' | 'processing' | 'completed' | 'error'
  error?: string
  document_id?: string
}

// Chart Data Types
export interface ChartDataPoint {
  label: string
  value: number
  color?: string
  metadata?: any
}

export interface TimeSeriesDataPoint {
  timestamp: string
  value: number
  anomaly?: boolean
  confidence?: number
}