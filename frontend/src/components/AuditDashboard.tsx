'use client'

import { useState, useEffect, useCallback } from 'react'
import { useQuery } from 'react-query'
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  DocumentIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  WifiIcon
} from '@heroicons/react/24/outline'
import RiskMetricsChart from './charts/RiskMetricsChart'
import AnomalyDetectionChart from './charts/AnomalyDetectionChart'
import ComplianceStatusCard from './cards/ComplianceStatusCard'
import RecentAuditsTable from './tables/RecentAuditsTable'
import DocumentUpload from './DocumentUpload'
import TransactionGraph from './TransactionGraph'
import StatisticalDashboard from './StatisticalDashboard'
import { useAuditWebSocket } from '../hooks/useWebSocket'
import { AuditSession, RealTimeUpdate, AuditFinding, MCPInterventionPrompt } from '../types/audit'

interface DashboardData {
  riskMetrics: {
    overall_risk_score: number
    credit_risk: number
    operational_risk: number
    market_risk: number
    compliance_risk: number
    fraud_risk: number
  }
  anomalies: {
    detected_anomalies: number
    total_transactions: number
    anomaly_rate: number
  }
  compliance: {
    sox_score: number
    gaap_compliance: number
    ifrs_compliance: number
    disclosure_completeness: number
    internal_controls_effectiveness: number
  }
  recent_audits: Array<{
    id: string
    company_name: string
    status: string
    risk_score: number
    created_at: string
  }>
  current_audit?: AuditSession
}

const mockData: DashboardData = {
  riskMetrics: {
    overall_risk_score: 0.65,
    credit_risk: 0.72,
    operational_risk: 0.58,
    market_risk: 0.43,
    compliance_risk: 0.35,
    fraud_risk: 0.28
  },
  anomalies: {
    detected_anomalies: 23,
    total_transactions: 15420,
    anomaly_rate: 0.0015
  },
  compliance: {
    sox_score: 85,
    gaap_compliance: 92,
    ifrs_compliance: 88,
    disclosure_completeness: 78,
    internal_controls_effectiveness: 91
  },
  recent_audits: [
    {
      id: '1',
      company_name: 'TechCorp Inc.',
      status: 'completed',
      risk_score: 0.72,
      created_at: '2024-01-15T10:30:00Z'
    },
    {
      id: '2',
      company_name: 'FinanceGlobal Ltd.',
      status: 'in_progress',
      risk_score: 0.45,
      created_at: '2024-01-14T14:20:00Z'
    },
    {
      id: '3',
      company_name: 'RetailChain Co.',
      status: 'pending',
      risk_score: 0.83,
      created_at: '2024-01-13T09:15:00Z'
    }
  ],
  current_audit: {
    id: 'audit_001',
    thread_id: 'thread_abc123',
    status: 'in_progress',
    created_at: '2024-01-16T08:00:00Z',
    updated_at: '2024-01-16T09:30:00Z',
    progress: 65,
    current_step: 'Statistical Analysis',
    risk_score: 0.68,
    findings: [],
    documents: []
  }
}

export default function AuditDashboard() {
  const [currentAudit, setCurrentAudit] = useState<AuditSession | null>(mockData.current_audit || null)
  const [recentFindings, setRecentFindings] = useState<AuditFinding[]>([])
  const [pendingInterventions, setPendingInterventions] = useState<MCPInterventionPrompt[]>([])
  const [activeTab, setActiveTab] = useState<'overview' | 'transaction-graph' | 'statistical' | 'documents'>('overview')
  const [showUploadModal, setShowUploadModal] = useState(false)

  // WebSocket connection for real-time updates
  const { isConnected, lastMessage } = useAuditWebSocket(
    currentAudit?.id || 'default',
    {
      onMessage: useCallback((update: RealTimeUpdate) => {
        handleRealTimeUpdate(update)
      }, []),
      onConnect: () => console.log('Connected to audit WebSocket'),
      onDisconnect: () => console.log('Disconnected from audit WebSocket')
    }
  )

  const { data, isLoading, error } = useQuery<DashboardData>(
    'dashboard-data',
    async () => {
      // In a real app, this would fetch from the API
      // const response = await fetch('/api/v1/dashboard')
      // return response.json()

      // For now, return mock data
      return new Promise<DashboardData>((resolve) => {
        setTimeout(() => resolve(mockData), 1000)
      })
    },
    {
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  )

  const handleRealTimeUpdate = useCallback((update: RealTimeUpdate) => {
    switch (update.event_type) {
      case 'progress_update':
        if (currentAudit && update.data.progress !== undefined) {
          setCurrentAudit(prev => prev ? {
            ...prev,
            progress: update.data.progress!,
            current_step: update.data.current_step || prev.current_step,
            updated_at: update.timestamp
          } : null)
        }
        break

      case 'finding_detected':
        if (update.data.new_finding) {
          setRecentFindings(prev => [update.data.new_finding!, ...prev.slice(0, 9)])
        }
        break

      case 'human_intervention_required':
        if (update.data.intervention_prompt) {
          setPendingInterventions(prev => [update.data.intervention_prompt!, ...prev])
        }
        break

      case 'analysis_complete':
        if (currentAudit) {
          setCurrentAudit(prev => prev ? {
            ...prev,
            status: 'completed',
            progress: 100,
            updated_at: update.timestamp
          } : null)
        }
        break
    }
  }, [currentAudit])

  const startNewAudit = useCallback(async () => {
    try {
      // In a real app, this would call the API to start a new audit
      const response = await fetch('/api/v1/audit/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      const result = await response.json()

      setCurrentAudit({
        id: result.audit_id,
        thread_id: result.thread_id,
        status: 'pending',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        progress: 0,
        current_step: 'Initializing',
        findings: [],
        documents: []
      })
      setShowUploadModal(true)
    } catch (error) {
      console.error('Failed to start new audit:', error)
    }
  }, [])

  const handleDocumentUpload = useCallback((auditId: string) => {
    // This will be called when documents are uploaded
    console.log('Documents uploaded for audit:', auditId)
    setShowUploadModal(false)
  }, [])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Error loading dashboard
          </h3>
          <p className="text-gray-500">Please try refreshing the page</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return null
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100'
      case 'in_progress': return 'text-blue-600 bg-blue-100'
      case 'pending': return 'text-yellow-600 bg-yellow-100'
      case 'failed': return 'text-red-600 bg-red-100'
      case 'paused': return 'text-orange-600 bg-orange-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getRiskColor = (score: number) => {
    if (score >= 0.8) return 'text-red-600 bg-red-100'
    if (score >= 0.6) return 'text-orange-600 bg-orange-100'
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100'
    return 'text-green-600 bg-green-100'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header with Connection Status */}
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Audit Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Monitor financial audit activities and risk metrics in real-time
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {/* WebSocket Connection Status */}
          <div className="flex items-center space-x-2">
            <WifiIcon
              className={`h-5 w-5 ${isConnected ? 'text-green-500' : 'text-red-500'}`}
            />
            <span className={`text-sm ${isConnected ? 'text-green-700' : 'text-red-700'}`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Start New Audit Button */}
          <button
            onClick={startNewAudit}
            className="btn-primary flex items-center space-x-2"
          >
            <PlayIcon className="h-4 w-4" />
            <span>Start New Audit</span>
          </button>
        </div>
      </div>

      {/* Current Audit Progress */}
      {currentAudit && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Current Audit</h2>
              <p className="text-sm text-gray-500">ID: {currentAudit.id}</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(currentAudit.status)}`}>
                {currentAudit.status.replace('_', ' ').toUpperCase()}
              </span>
              {currentAudit.risk_score && (
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(currentAudit.risk_score)}`}>
                  Risk: {(currentAudit.risk_score * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Progress: {currentAudit.progress}%</span>
              <span>Current Step: {currentAudit.current_step}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${currentAudit.progress}%` }}
              />
            </div>
          </div>

          {/* Current Step Details */}
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Started:</span>
              <p className="font-medium">{new Date(currentAudit.created_at).toLocaleString()}</p>
            </div>
            <div>
              <span className="text-gray-500">Last Updated:</span>
              <p className="font-medium">{new Date(currentAudit.updated_at).toLocaleString()}</p>
            </div>
            <div>
              <span className="text-gray-500">Documents:</span>
              <p className="font-medium">{currentAudit.documents.length} files</p>
            </div>
          </div>
        </div>
      )}

      {/* Pending Interventions Alert */}
      {pendingInterventions.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600 mr-2" />
            <h3 className="text-lg font-medium text-yellow-800">
              Human Intervention Required
            </h3>
          </div>
          <p className="text-yellow-700 mt-1">
            {pendingInterventions.length} audit decision(s) require your review.
          </p>
          <div className="mt-3 space-y-2">
            {pendingInterventions.map((intervention) => (
              <div key={intervention.id} className="bg-white p-3 rounded border border-yellow-200">
                <h4 className="font-medium text-gray-900">{intervention.title}</h4>
                <p className="text-sm text-gray-600">{intervention.description}</p>
                <button className="mt-2 text-sm bg-yellow-600 text-white px-3 py-1 rounded hover:bg-yellow-700">
                  Review Decision
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Findings */}
      {recentFindings.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Findings</h3>
          <div className="space-y-3">
            {recentFindings.slice(0, 5).map((finding) => (
              <div key={finding.id} className="flex items-start space-x-3 p-3 bg-gray-50 rounded">
                <div className={`p-1 rounded-full ${getRiskColor(finding.confidence_score)}`}>
                  <ExclamationTriangleIcon className="h-4 w-4" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{finding.description}</p>
                  <p className="text-xs text-gray-500">
                    {finding.type} • Confidence: {(finding.confidence_score * 100).toFixed(0)}%
                  </p>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(finding.confidence_score)}`}>
                  {finding.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: ChartBarIcon },
            { id: 'transaction-graph', label: 'Transaction Graph', icon: DocumentIcon },
            { id: 'statistical', label: 'Statistical Analysis', icon: ChartBarIcon },
            { id: 'documents', label: 'Documents', icon: DocumentIcon }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon
                className={`mr-2 h-5 w-5 ${
                  activeTab === tab.id ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-500'
                }`}
              />
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Key Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6">
              <div className="card">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  Overall Risk
                </h3>
                <div className="mt-2 flex items-baseline">
                  <p className="text-2xl font-semibold text-gray-900">
                    {(data.riskMetrics.overall_risk_score * 100).toFixed(0)}%
                  </p>
                  <p className={`ml-2 text-xs font-medium ${getRiskColor(data.riskMetrics.overall_risk_score)}`}>
                    {data.riskMetrics.overall_risk_score > 0.7 ? 'High' :
                     data.riskMetrics.overall_risk_score > 0.5 ? 'Medium' : 'Low'}
                  </p>
                </div>
              </div>

              <div className="card">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  Credit Risk
                </h3>
                <div className="mt-2">
                  <p className="text-2xl font-semibold text-gray-900">
                    {(data.riskMetrics.credit_risk * 100).toFixed(0)}%
                  </p>
                </div>
              </div>

              <div className="card">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  Fraud Risk
                </h3>
                <div className="mt-2">
                  <p className="text-2xl font-semibold text-gray-900">
                    {(data.riskMetrics.fraud_risk * 100).toFixed(0)}%
                  </p>
                </div>
              </div>

              <div className="card">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  Anomalies
                </h3>
                <div className="mt-2 flex items-baseline">
                  <p className="text-2xl font-semibold text-gray-900">
                    {data.anomalies.detected_anomalies}
                  </p>
                  <p className="ml-2 text-xs text-gray-600">
                    / {data.anomalies.total_transactions.toLocaleString()}
                  </p>
                </div>
              </div>

              <div className="card">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  SOX Score
                </h3>
                <div className="mt-2 flex items-baseline">
                  <p className="text-2xl font-semibold text-gray-900">
                    {data.compliance.sox_score}%
                  </p>
                  <p className={`ml-2 text-xs font-medium ${
                    data.compliance.sox_score >= 90 ? 'text-green-600' :
                    data.compliance.sox_score >= 70 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {data.compliance.sox_score >= 90 ? 'Good' :
                     data.compliance.sox_score >= 70 ? 'Fair' : 'Poor'}
                  </p>
                </div>
              </div>

              <div className="card">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wide">
                  Active Audits
                </h3>
                <div className="mt-2 flex items-baseline">
                  <p className="text-2xl font-semibold text-gray-900">
                    {data.recent_audits.filter(audit => audit.status === 'in_progress').length}
                  </p>
                  <p className="ml-2 text-xs text-gray-600">
                    / {data.recent_audits.length}
                  </p>
                </div>
              </div>
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Risk Metrics</h3>
                <RiskMetricsChart data={data.riskMetrics} />
              </div>

              <div className="card">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Anomaly Detection</h3>
                <AnomalyDetectionChart data={data.anomalies} />
              </div>
            </div>

            {/* Compliance and Recent Audits */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <ComplianceStatusCard data={data.compliance} />
              </div>

              <div className="lg:col-span-2">
                <div className="card">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Audits</h3>
                  <RecentAuditsTable audits={data.recent_audits} />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'transaction-graph' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Transaction Network Analysis</h3>
            <TransactionGraph auditId={currentAudit?.id || 'default'} />
          </div>
        )}

        {activeTab === 'statistical' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Statistical Analysis</h3>
            <StatisticalDashboard auditId={currentAudit?.id || 'default'} />
          </div>
        )}

        {activeTab === 'documents' && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-gray-900">Document Management</h3>
              <button
                onClick={() => setShowUploadModal(true)}
                className="btn-primary flex items-center space-x-2"
              >
                <DocumentIcon className="h-4 w-4" />
                <span>Upload Documents</span>
              </button>
            </div>

            {currentAudit && (
              <div className="space-y-4">
                {currentAudit.documents.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {currentAudit.documents.map((doc) => (
                      <div key={doc.id} className="border rounded-lg p-4">
                        <div className="flex items-center space-x-2 mb-2">
                          <DocumentIcon className="h-5 w-5 text-gray-400" />
                          <span className="font-medium text-sm truncate">{doc.filename}</span>
                        </div>
                        <div className="text-xs text-gray-500">
                          <p>Size: {(doc.file_size / 1024 / 1024).toFixed(2)} MB</p>
                          <p>Status: {doc.upload_status}</p>
                          <p>Uploaded: {new Date(doc.uploaded_at).toLocaleDateString()}</p>
                        </div>
                        {doc.upload_status === 'processing' && (
                          <div className="mt-2">
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full"
                                style={{ width: `${doc.processing_progress}%` }}
                              />
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                              Processing: {doc.processing_progress}%
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <DocumentIcon className="mx-auto h-12 w-12 text-gray-400" />
                    <h3 className="mt-2 text-sm font-medium text-gray-900">No documents</h3>
                    <p className="mt-1 text-sm text-gray-500">
                      Get started by uploading financial documents for analysis.
                    </p>
                    <div className="mt-6">
                      <button
                        onClick={() => setShowUploadModal(true)}
                        className="btn-primary"
                      >
                        Upload Documents
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && currentAudit && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900">Upload Documents</h3>
                <button
                  onClick={() => setShowUploadModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <span className="sr-only">Close</span>
                  ✕
                </button>
              </div>

              <DocumentUpload
                auditId={currentAudit.id}
                onUploadComplete={() => handleDocumentUpload(currentAudit.id)}
                acceptedFileTypes={[
                  'application/pdf',
                  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                  'application/vnd.ms-excel',
                  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                  'text/csv',
                  'text/plain',
                  'text/markdown',
                  'application/json'
                ]}
                maxFiles={20}
                maxFileSize={100 * 1024 * 1024} // 100MB
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}