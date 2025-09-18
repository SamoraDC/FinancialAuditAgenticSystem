'use client'

import React, { useState, useCallback } from 'react'
import {
  ExclamationTriangleIcon,
  XMarkIcon,
  CheckCircleIcon,
  ClockIcon,
  InformationCircleIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'
import { MCPInterventionPrompt } from '../types/audit'

interface MCPInterventionModalProps {
  intervention: MCPInterventionPrompt
  isOpen: boolean
  onClose: () => void
  onSubmit: (response: MCPInterventionResponse) => void
}

interface MCPInterventionResponse {
  intervention_id: string
  decision: string
  responses: Record<string, any>
  comments: string
  confidence_level: number
  timestamp: string
}

export default function MCPInterventionModal({
  intervention,
  isOpen,
  onClose,
  onSubmit
}: MCPInterventionModalProps) {
  const [responses, setResponses] = useState<Record<string, any>>({})
  const [comments, setComments] = useState('')
  const [confidenceLevel, setConfidenceLevel] = useState(80)
  const [selectedOption, setSelectedOption] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = useCallback(async () => {
    setIsSubmitting(true)

    try {
      const response: MCPInterventionResponse = {
        intervention_id: intervention.id,
        decision: selectedOption,
        responses,
        comments,
        confidence_level: confidenceLevel / 100,
        timestamp: new Date().toISOString()
      }

      await onSubmit(response)
      onClose()
    } catch (error) {
      console.error('Failed to submit intervention response:', error)
    } finally {
      setIsSubmitting(false)
    }
  }, [intervention.id, selectedOption, responses, comments, confidenceLevel, onSubmit, onClose])

  const handleFieldChange = useCallback((fieldId: string, value: any) => {
    setResponses(prev => ({
      ...prev,
      [fieldId]: value
    }))
  }, [])

  const getInterventionTypeIcon = () => {
    switch (intervention.type) {
      case 'anomaly_review':
        return <ExclamationTriangleIcon className="h-8 w-8 text-orange-500" />
      case 'compliance_decision':
        return <DocumentTextIcon className="h-8 w-8 text-blue-500" />
      case 'risk_assessment':
        return <InformationCircleIcon className="h-8 w-8 text-red-500" />
      default:
        return <ClockIcon className="h-8 w-8 text-gray-500" />
    }
  }

  const getInterventionTypeColor = () => {
    switch (intervention.type) {
      case 'anomaly_review':
        return 'border-orange-200 bg-orange-50'
      case 'compliance_decision':
        return 'border-blue-200 bg-blue-50'
      case 'risk_assessment':
        return 'border-red-200 bg-red-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  const isFormValid = () => {
    const hasSelectedOption = selectedOption !== ''
    const hasRequiredFields = intervention.required_fields.every(field => {
      if (!field.required) return true
      return responses[field.id] !== undefined && responses[field.id] !== ''
    })
    return hasSelectedOption && hasRequiredFields
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-6 border w-11/12 md:w-3/4 lg:w-2/3 xl:w-1/2 shadow-lg rounded-lg bg-white">
        {/* Header */}
        <div className={`rounded-lg p-4 mb-6 ${getInterventionTypeColor()}`}>
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              {getInterventionTypeIcon()}
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Human Intervention Required
                </h2>
                <p className="text-sm text-gray-600 mt-1">
                  {intervention.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              disabled={isSubmitting}
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
        </div>

        {/* Title and Description */}
        <div className="mb-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {intervention.title}
          </h3>
          <p className="text-gray-700 leading-relaxed">
            {intervention.description}
          </p>
        </div>

        {/* Context Information */}
        {intervention.context && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-900 mb-2">Context Information</h4>
            <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
              {typeof intervention.context === 'string'
                ? intervention.context
                : JSON.stringify(intervention.context, null, 2)
              }
            </pre>
          </div>
        )}

        {/* Decision Options */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Select Your Decision</h4>
          <div className="space-y-2">
            {intervention.options.map((option) => (
              <label key={option.id} className="flex items-center cursor-pointer">
                <input
                  type="radio"
                  name="decision"
                  value={option.value}
                  checked={selectedOption === option.value}
                  onChange={(e) => setSelectedOption(e.target.value)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                  disabled={isSubmitting}
                />
                <span className="ml-3 text-sm text-gray-700">{option.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Required Fields */}
        {intervention.required_fields.length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Additional Information</h4>
            <div className="space-y-4">
              {intervention.required_fields.map((field) => (
                <div key={field.id}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {field.label}
                    {field.required && <span className="text-red-500 ml-1">*</span>}
                  </label>

                  {field.type === 'text' && (
                    <input
                      type="text"
                      value={responses[field.id] || ''}
                      onChange={(e) => handleFieldChange(field.id, e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      disabled={isSubmitting}
                    />
                  )}

                  {field.type === 'select' && (
                    <select
                      value={responses[field.id] || ''}
                      onChange={(e) => handleFieldChange(field.id, e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      disabled={isSubmitting}
                    >
                      <option value="">Select an option...</option>
                      {field.options?.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  )}

                  {field.type === 'radio' && (
                    <div className="space-y-2">
                      {field.options?.map((option) => (
                        <label key={option} className="flex items-center">
                          <input
                            type="radio"
                            name={field.id}
                            value={option}
                            checked={responses[field.id] === option}
                            onChange={(e) => handleFieldChange(field.id, e.target.value)}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                            disabled={isSubmitting}
                          />
                          <span className="ml-2 text-sm text-gray-700">{option}</span>
                        </label>
                      ))}
                    </div>
                  )}

                  {field.type === 'checkbox' && (
                    <div className="space-y-2">
                      {field.options?.map((option) => (
                        <label key={option} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={responses[field.id]?.includes(option) || false}
                            onChange={(e) => {
                              const currentValues = responses[field.id] || []
                              if (e.target.checked) {
                                handleFieldChange(field.id, [...currentValues, option])
                              } else {
                                handleFieldChange(field.id, currentValues.filter((v: string) => v !== option))
                              }
                            }}
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                            disabled={isSubmitting}
                          />
                          <span className="ml-2 text-sm text-gray-700">{option}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Comments Section */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Additional Comments
          </label>
          <textarea
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            rows={4}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            placeholder="Provide any additional context, reasoning, or notes about your decision..."
            disabled={isSubmitting}
          />
        </div>

        {/* Confidence Level */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence Level: {confidenceLevel}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={confidenceLevel}
            onChange={(e) => setConfidenceLevel(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            disabled={isSubmitting}
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Low Confidence</span>
            <span>High Confidence</span>
          </div>
        </div>

        {/* Deadline Warning */}
        {intervention.deadline && (
          <div className="mb-6 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center">
              <ClockIcon className="h-5 w-5 text-yellow-600 mr-2" />
              <span className="text-sm text-yellow-800">
                Deadline: {new Date(intervention.deadline).toLocaleString()}
              </span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            disabled={isSubmitting}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!isFormValid() || isSubmitting}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {isSubmitting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Submitting...
              </>
            ) : (
              <>
                <CheckCircleIcon className="h-4 w-4 mr-2" />
                Submit Decision
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

// Helper component for displaying past intervention decisions
interface InterventionHistoryProps {
  interventions: Array<MCPInterventionPrompt & { response?: MCPInterventionResponse }>
  className?: string
}

export function InterventionHistory({ interventions, className = '' }: InterventionHistoryProps) {
  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-lg font-medium text-gray-900">Intervention History</h3>

      {interventions.length === 0 ? (
        <div className="text-center py-6 text-gray-500">
          <ClockIcon className="mx-auto h-12 w-12 text-gray-400 mb-2" />
          <p>No previous interventions</p>
        </div>
      ) : (
        <div className="space-y-3">
          {interventions.map((intervention) => (
            <div key={intervention.id} className="border rounded-lg p-4">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="font-medium text-gray-900">{intervention.title}</h4>
                  <p className="text-sm text-gray-500">
                    {intervention.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </p>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  intervention.response ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {intervention.response ? 'Resolved' : 'Pending'}
                </span>
              </div>

              <p className="text-sm text-gray-700 mb-3">{intervention.description}</p>

              {intervention.response && (
                <div className="bg-gray-50 rounded p-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium text-gray-700">Decision:</span>
                      <p className="text-gray-900">{intervention.response.decision}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Confidence:</span>
                      <p className="text-gray-900">{(intervention.response.confidence_level * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                  {intervention.response.comments && (
                    <div className="mt-2">
                      <span className="font-medium text-gray-700 text-sm">Comments:</span>
                      <p className="text-gray-900 text-sm mt-1">{intervention.response.comments}</p>
                    </div>
                  )}
                  <p className="text-xs text-gray-500 mt-2">
                    Resolved on {new Date(intervention.response.timestamp).toLocaleString()}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}