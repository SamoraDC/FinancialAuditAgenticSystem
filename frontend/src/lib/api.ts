// API Integration Library

import axios, { AxiosResponse } from 'axios'
import { AuditSession, UploadedDocument, APIResponse, AuditStartResponse } from '../types/audit'

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth tokens
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Document API
export const documentApi = {
  uploadDocument: async (
    auditId: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<UploadedDocument> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('audit_id', auditId)

    const response = await apiClient.post<APIResponse<UploadedDocument>>(
      `/api/v1/audit/${auditId}/documents/upload`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress(progress)
          }
        },
      }
    )

    return response.data.data
  },

  getDocuments: async (auditId: string): Promise<UploadedDocument[]> => {
    const response = await apiClient.get<APIResponse<UploadedDocument[]>>(
      `/api/v1/audit/${auditId}/documents`
    )
    return response.data.data
  },

  deleteDocument: async (auditId: string, documentId: string): Promise<void> => {
    await apiClient.delete(`/api/v1/audit/${auditId}/documents/${documentId}`)
  },

  downloadDocument: async (auditId: string, documentId: string): Promise<Blob> => {
    const response = await apiClient.get(
      `/api/v1/audit/${auditId}/documents/${documentId}/download`,
      { responseType: 'blob' }
    )
    return response.data
  },
}

// Audit API
export const auditApi = {
  startAudit: async (): Promise<AuditStartResponse> => {
    const response = await apiClient.post<APIResponse<AuditStartResponse>>('/api/v1/audit/start')
    return response.data.data
  },

  getAudit: async (auditId: string): Promise<AuditSession> => {
    const response = await apiClient.get<APIResponse<AuditSession>>(`/api/v1/audit/${auditId}`)
    return response.data.data
  },

  getAuditStatus: async (auditId: string): Promise<AuditSession> => {
    const response = await apiClient.get<APIResponse<AuditSession>>(`/api/v1/audit/${auditId}/status`)
    return response.data.data
  },

  pauseAudit: async (auditId: string): Promise<void> => {
    await apiClient.post(`/api/v1/audit/${auditId}/pause`)
  },

  resumeAudit: async (auditId: string, input?: any): Promise<void> => {
    await apiClient.post(`/api/v1/audit/${auditId}/resume`, { input })
  },

  stopAudit: async (auditId: string): Promise<void> => {
    await apiClient.post(`/api/v1/audit/${auditId}/stop`)
  },

  getAuditResults: async (auditId: string): Promise<any> => {
    const response = await apiClient.get<APIResponse<any>>(`/api/v1/audit/${auditId}/results`)
    return response.data.data
  },

  exportAuditReport: async (auditId: string, format: 'pdf' | 'xlsx' | 'json' = 'pdf'): Promise<Blob> => {
    const response = await apiClient.get(
      `/api/v1/audit/${auditId}/export?format=${format}`,
      { responseType: 'blob' }
    )
    return response.data
  },
}

// Analytics API
export const analyticsApi = {
  getDashboardData: async (): Promise<any> => {
    const response = await apiClient.get<APIResponse<any>>('/api/v1/analytics/dashboard')
    return response.data.data
  },

  getTransactionGraph: async (auditId: string): Promise<any> => {
    const response = await apiClient.get<APIResponse<any>>(`/api/v1/analytics/${auditId}/transaction-graph`)
    return response.data.data
  },

  getStatisticalAnalysis: async (auditId: string): Promise<any> => {
    const response = await apiClient.get<APIResponse<any>>(`/api/v1/analytics/${auditId}/statistical-analysis`)
    return response.data.data
  },

  getRiskMetrics: async (auditId: string): Promise<any> => {
    const response = await apiClient.get<APIResponse<any>>(`/api/v1/analytics/${auditId}/risk-metrics`)
    return response.data.data
  },
}

// MCP API for human intervention
export const mcpApi = {
  submitInterventionResponse: async (interventionId: string, response: any): Promise<void> => {
    await apiClient.post(`/api/v1/mcp/intervention/${interventionId}/response`, response)
  },

  getInterventionHistory: async (auditId: string): Promise<any[]> => {
    const response = await apiClient.get<APIResponse<any[]>>(`/api/v1/mcp/audit/${auditId}/interventions`)
    return response.data.data
  },

  getPendingInterventions: async (): Promise<any[]> => {
    const response = await apiClient.get<APIResponse<any[]>>('/api/v1/mcp/interventions/pending')
    return response.data.data
  },
}

// WebSocket Utilities
export const websocketApi = {
  createAuditWebSocket: (auditId: string): WebSocket => {
    return new WebSocket(`${WS_BASE_URL}/ws/audit/${auditId}`)
  },

  createGeneralWebSocket: (): WebSocket => {
    return new WebSocket(`${WS_BASE_URL}/ws/notifications`)
  },
}

// Auth API (if needed)
export const authApi = {
  login: async (email: string, password: string): Promise<{ token: string; user: any }> => {
    const response = await apiClient.post<APIResponse<{ token: string; user: any }>>(
      '/api/v1/auth/login',
      { email, password }
    )
    return response.data.data
  },

  logout: async (): Promise<void> => {
    await apiClient.post('/api/v1/auth/logout')
    localStorage.removeItem('auth_token')
  },

  getCurrentUser: async (): Promise<any> => {
    const response = await apiClient.get<APIResponse<any>>('/api/v1/auth/me')
    return response.data.data
  },

  refreshToken: async (): Promise<{ token: string }> => {
    const response = await apiClient.post<APIResponse<{ token: string }>>('/api/v1/auth/refresh')
    return response.data.data
  },
}

// Utility functions
export const apiUtils = {
  handleApiError: (error: any): string => {
    if (error.response?.data?.message) {
      return error.response.data.message
    }
    if (error.message) {
      return error.message
    }
    return 'An unexpected error occurred'
  },

  formatFileSize: (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  },

  downloadBlob: (blob: Blob, filename: string): void => {
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  },
}

// Export the API client for custom requests
export { apiClient }

// Export types
export type { UploadedDocument, AuditSession, APIResponse, AuditStartResponse }