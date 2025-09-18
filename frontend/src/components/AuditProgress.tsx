'use client'

import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import {
  CheckCircleIcon,
  ClockIcon,
  ExclamationCircleIcon,
  PlayCircleIcon,
  DocumentTextIcon,
  ChartBarIcon,
  EyeIcon,
  CheckIcon
} from '@heroicons/react/24/outline'
import { auditApi, AuditProgress as AuditProgressType, createWebSocketConnection } from '../lib/api'

interface AuditProgressProps {
  auditId: string
  className?: string
}

interface ProgressStep {
  id: string
  name: string
  description: string
  icon: React.ComponentType<any>
  status: 'pending' | 'in_progress' | 'completed' | 'error'
  progress?: number
  estimatedDuration?: string
}

const defaultSteps: ProgressStep[] = [
  {
    id: 'upload',
    name: 'Document Upload',
    description: 'Uploading and validating financial documents',
    icon: DocumentTextIcon,
    status: 'pending',
    estimatedDuration: '2-5 minutes'
  },
  {
    id: 'processing',
    name: 'Data Processing',
    description: 'Extracting and parsing financial data',
    icon: PlayCircleIcon,
    status: 'pending',
    estimatedDuration: '5-10 minutes'
  },
  {
    id: 'analysis',
    name: 'Financial Analysis',
    description: 'Running audit algorithms and risk assessment',
    icon: ChartBarIcon,
    status: 'pending',
    estimatedDuration: '10-15 minutes'
  },
  {
    id: 'review',
    name: 'Review & Validation',
    description: 'Generating findings and recommendations',
    icon: EyeIcon,
    status: 'pending',
    estimatedDuration: '3-7 minutes'
  },
  {
    id: 'completed',
    name: 'Audit Complete',
    description: 'Final report ready for review',
    icon: CheckIcon,
    status: 'pending',
    estimatedDuration: 'Complete'
  }
]

export default function AuditProgress({ auditId, className = '' }: AuditProgressProps) {
  const [steps, setSteps] = useState<ProgressStep[]>(defaultSteps)
  const [currentPhase, setCurrentPhase] = useState<string>('upload')
  const [overallProgress, setOverallProgress] = useState<number>(0)
  const [currentTask, setCurrentTask] = useState<string>('')
  const [estimatedCompletion, setEstimatedCompletion] = useState<string>('')
  const [errors, setErrors] = useState<string[]>([])

  // Query for initial progress data
  const { data: progressData, isLoading } = useQuery(\n    ['audit-progress', auditId],\n    () => auditApi.getAuditProgress(auditId),\n    {\n      refetchInterval: 5000, // Poll every 5 seconds\n      onSuccess: (data) => {\n        updateProgressFromData(data)\n      }\n    }\n  )

  // WebSocket connection for real-time updates
  useEffect(() => {\n    if (!auditId) return

    const wsConnection = createWebSocketConnection(auditId)

    wsConnection.onProgress((progress: AuditProgressType) => {\n      updateProgressFromData(progress)\n    })

    wsConnection.onError((error) => {\n      console.error('WebSocket error:', error)\n    })

    return () => {\n      wsConnection.close()\n    }\n  }, [auditId])

  const updateProgressFromData = (data: AuditProgressType) => {\n    setCurrentPhase(data.phase)\n    setOverallProgress(data.progress)\n    setCurrentTask(data.current_task)\n    setEstimatedCompletion(data.estimated_completion)\n    setErrors(data.errors || [])\n\n    // Update step statuses based on current phase\n    setSteps(prevSteps => \n      prevSteps.map(step => {\n        if (step.id === data.phase) {\n          return { ...step, status: 'in_progress', progress: data.progress }\n        } else if (getStepIndex(step.id) < getStepIndex(data.phase)) {\n          return { ...step, status: 'completed', progress: 100 }\n        } else {\n          return { ...step, status: 'pending', progress: 0 }\n        }\n      })\n    )\n  }

  const getStepIndex = (stepId: string): number => {\n    return steps.findIndex(step => step.id === stepId)\n  }

  const getStatusColor = (status: ProgressStep['status']) => {\n    switch (status) {\n      case 'completed':\n        return 'text-green-600 bg-green-100'\n      case 'in_progress':\n        return 'text-blue-600 bg-blue-100'\n      case 'error':\n        return 'text-red-600 bg-red-100'\n      default:\n        return 'text-gray-400 bg-gray-100'\n    }\n  }

  const getConnectorColor = (currentIndex: number, nextIndex: number) => {\n    if (getStepIndex(currentPhase) > currentIndex) {\n      return 'bg-green-500'\n    } else if (getStepIndex(currentPhase) === currentIndex) {\n      return 'bg-blue-500'\n    } else {\n      return 'bg-gray-300'\n    }\n  }

  const formatTimeRemaining = (estimatedCompletion: string): string => {\n    if (!estimatedCompletion) return 'Calculating...'\n    \n    try {\n      const completionTime = new Date(estimatedCompletion)\n      const now = new Date()\n      const diffMs = completionTime.getTime() - now.getTime()\n      \n      if (diffMs <= 0) return 'Completing soon...'\n      \n      const diffMinutes = Math.ceil(diffMs / (1000 * 60))\n      \n      if (diffMinutes < 60) {\n        return `${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} remaining`\n      } else {\n        const hours = Math.floor(diffMinutes / 60)\n        const minutes = diffMinutes % 60\n        return `${hours}h ${minutes}m remaining`\n      }\n    } catch {\n      return 'Calculating...'\n    }\n  }

  if (isLoading) {\n    return (\n      <div className={`animate-pulse ${className}`}>\n        <div className=\"space-y-4\">\n          {[...Array(5)].map((_, i) => (\n            <div key={i} className=\"flex items-center space-x-4\">\n              <div className=\"w-8 h-8 bg-gray-300 rounded-full\" />\n              <div className=\"flex-1 space-y-2\">\n                <div className=\"h-4 bg-gray-300 rounded w-1/3\" />\n                <div className=\"h-3 bg-gray-200 rounded w-2/3\" />\n              </div>\n            </div>\n          ))}\n        </div>\n      </div>\n    )\n  }

  return (\n    <div className={`bg-white rounded-lg shadow-sm border ${className}`}>\n      {/* Header */}\n      <div className=\"p-6 border-b\">\n        <div className=\"flex items-center justify-between\">\n          <div>\n            <h3 className=\"text-lg font-semibold text-gray-900\">\n              Audit Progress\n            </h3>\n            <p className=\"text-sm text-gray-600 mt-1\">\n              {currentTask || 'Preparing audit process...'}\n            </p>\n          </div>\n          \n          <div className=\"text-right\">\n            <div className=\"text-2xl font-bold text-blue-600\">\n              {Math.round(overallProgress)}%\n            </div>\n            <div className=\"text-xs text-gray-500\">\n              {formatTimeRemaining(estimatedCompletion)}\n            </div>\n          </div>\n        </div>\n\n        {/* Overall progress bar */}\n        <div className=\"mt-4\">\n          <div className=\"w-full bg-gray-200 rounded-full h-2\">\n            <div \n              className=\"bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out\"\n              style={{ width: `${overallProgress}%` }}\n            />\n          </div>\n        </div>\n      </div>\n\n      {/* Steps */}\n      <div className=\"p-6\">\n        <div className=\"space-y-6\">\n          {steps.map((step, index) => {\n            const Icon = step.icon\n            const isLast = index === steps.length - 1\n            \n            return (\n              <div key={step.id} className=\"relative\">\n                <div className=\"flex items-start\">\n                  {/* Icon */}\n                  <div className={`\n                    flex items-center justify-center w-10 h-10 rounded-full\n                    ${getStatusColor(step.status)}\n                    relative z-10\n                  `}>\n                    {step.status === 'completed' ? (\n                      <CheckCircleIcon className=\"w-6 h-6\" />\n                    ) : step.status === 'error' ? (\n                      <ExclamationCircleIcon className=\"w-6 h-6\" />\n                    ) : step.status === 'in_progress' ? (\n                      <div className=\"w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin\" />\n                    ) : (\n                      <Icon className=\"w-6 h-6\" />\n                    )}\n                  </div>\n\n                  {/* Content */}\n                  <div className=\"ml-4 flex-1\">\n                    <div className=\"flex items-center justify-between\">\n                      <h4 className={`\n                        text-sm font-medium\n                        ${step.status === 'completed' ? 'text-green-900' :\n                          step.status === 'in_progress' ? 'text-blue-900' :\n                          step.status === 'error' ? 'text-red-900' :\n                          'text-gray-500'}\n                      `}>\n                        {step.name}\n                      </h4>\n                      \n                      <div className=\"text-xs text-gray-500\">\n                        {step.status === 'in_progress' && step.progress !== undefined ? (\n                          `${Math.round(step.progress)}%`\n                        ) : step.status === 'completed' ? (\n                          'Completed'\n                        ) : (\n                          step.estimatedDuration\n                        )}\n                      </div>\n                    </div>\n                    \n                    <p className=\"text-sm text-gray-600 mt-1\">\n                      {step.description}\n                    </p>\n\n                    {/* Step progress bar for in-progress steps */}\n                    {step.status === 'in_progress' && step.progress !== undefined && (\n                      <div className=\"mt-2 w-full bg-gray-200 rounded-full h-1.5\">\n                        <div \n                          className=\"bg-blue-600 h-1.5 rounded-full transition-all duration-300\"\n                          style={{ width: `${step.progress}%` }}\n                        />\n                      </div>\n                    )}\n                  </div>\n                </div>\n\n                {/* Connector line */}\n                {!isLast && (\n                  <div className=\"absolute left-5 top-10 w-0.5 h-6 -ml-px\">\n                    <div className={`\n                      w-full h-full transition-colors duration-300\n                      ${getConnectorColor(index, index + 1)}\n                    `} />\n                  </div>\n                )}\n              </div>\n            )\n          })}\n        </div>\n      </div>\n\n      {/* Errors */}\n      {errors.length > 0 && (\n        <div className=\"p-6 border-t bg-red-50\">\n          <h4 className=\"text-sm font-medium text-red-800 mb-2\">\n            Issues Detected:\n          </h4>\n          <ul className=\"space-y-1\">\n            {errors.map((error, index) => (\n              <li key={index} className=\"text-sm text-red-700 flex items-start\">\n                <ExclamationCircleIcon className=\"w-4 h-4 mt-0.5 mr-2 flex-shrink-0\" />\n                {error}\n              </li>\n            ))}\n          </ul>\n        </div>\n      )}\n    </div>\n  )\n}