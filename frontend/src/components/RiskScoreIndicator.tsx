'use client'

import React from 'react'
import {
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ShieldExclamationIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline'

interface RiskScoreIndicatorProps {
  score: number // 0-1 scale
  label?: string
  size?: 'sm' | 'md' | 'lg'
  showDetails?: boolean
  className?: string
}

interface RiskLevel {
  level: 'low' | 'medium' | 'high' | 'critical'
  color: string
  bgColor: string
  textColor: string
  icon: React.ReactNode
  threshold: number
}

const riskLevels: RiskLevel[] = [
  {
    level: 'low',
    color: 'border-green-500',
    bgColor: 'bg-green-500',
    textColor: 'text-green-700',
    icon: <CheckCircleIcon className="h-5 w-5" />,
    threshold: 0.3
  },
  {
    level: 'medium',
    color: 'border-yellow-500',
    bgColor: 'bg-yellow-500',
    textColor: 'text-yellow-700',
    icon: <ShieldExclamationIcon className="h-5 w-5" />,
    threshold: 0.6
  },
  {
    level: 'high',
    color: 'border-orange-500',
    bgColor: 'bg-orange-500',
    textColor: 'text-orange-700',
    icon: <ExclamationTriangleIcon className="h-5 w-5" />,
    threshold: 0.8
  },
  {
    level: 'critical',
    color: 'border-red-500',
    bgColor: 'bg-red-500',
    textColor: 'text-red-700',
    icon: <ShieldExclamationIcon className="h-5 w-5" />,
    threshold: 1.0
  }
]

export default function RiskScoreIndicator({
  score,
  label = 'Risk Score',
  size = 'md',
  showDetails = false,
  className = ''
}: RiskScoreIndicatorProps) {
  // Determine risk level
  const riskLevel = riskLevels.find(level => score <= level.threshold) || riskLevels[riskLevels.length - 1]

  // Size configurations
  const sizeConfig = {
    sm: {
      container: 'w-16 h-16',
      strokeWidth: '8',
      radius: '24',
      text: 'text-xs',
      iconSize: 'h-3 w-3'
    },
    md: {
      container: 'w-24 h-24',
      strokeWidth: '12',
      radius: '36',
      text: 'text-sm',
      iconSize: 'h-4 w-4'
    },
    lg: {
      container: 'w-32 h-32',
      strokeWidth: '16',
      radius: '48',
      text: 'text-lg',
      iconSize: 'h-6 w-6'
    }
  }

  const config = sizeConfig[size]
  const percentage = Math.round(score * 100)
  const circumference = 2 * Math.PI * parseInt(config.radius)
  const strokeDasharray = circumference
  const strokeDashoffset = circumference - (score * circumference)

  const getRiskDescription = (level: string) => {
    switch (level) {
      case 'low':
        return 'Low risk - Minimal concerns identified'
      case 'medium':
        return 'Medium risk - Some issues require attention'
      case 'high':
        return 'High risk - Significant concerns identified'
      case 'critical':
        return 'Critical risk - Immediate action required'
      default:
        return ''
    }
  }

  const getRiskRecommendation = (level: string) => {
    switch (level) {
      case 'low':
        return 'Continue regular monitoring'
      case 'medium':
        return 'Review findings and implement improvements'
      case 'high':
        return 'Conduct detailed investigation'
      case 'critical':
        return 'Immediate review and corrective action required'
      default:
        return ''
    }
  }

  return (
    <div className={`flex flex-col items-center ${className}`}>
      {/* Circular Progress Indicator */}
      <div className={`relative ${config.container}`}>
        {/* Background Circle */}
        <svg className="transform -rotate-90 w-full h-full">
          <circle
            cx="50%"
            cy="50%"
            r={config.radius}
            stroke="currentColor"
            strokeWidth={config.strokeWidth}
            fill="transparent"
            className="text-gray-200"
          />
          {/* Progress Circle */}
          <circle
            cx="50%"
            cy="50%"
            r={config.radius}
            stroke="currentColor"
            strokeWidth={config.strokeWidth}
            fill="transparent"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className={riskLevel.textColor.replace('text-', 'text-').replace('-700', '-500')}
            style={{
              transition: 'stroke-dashoffset 1s ease-in-out'
            }}
          />
        </svg>

        {/* Center Content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-bold ${config.text} ${riskLevel.textColor}`}>
            {percentage}%
          </span>
          {size !== 'sm' && (
            <div className={`${riskLevel.textColor} ${config.iconSize}`}>
              {React.cloneElement(riskLevel.icon as React.ReactElement, {
                className: `${config.iconSize}`
              })}
            </div>
          )}
        </div>
      </div>

      {/* Label */}
      <div className="text-center mt-2">
        <p className={`font-medium text-gray-900 ${size === 'sm' ? 'text-xs' : 'text-sm'}`}>
          {label}
        </p>
        <p className={`capitalize ${riskLevel.textColor} ${size === 'sm' ? 'text-xs' : 'text-sm'}`}>
          {riskLevel.level} Risk
        </p>
      </div>

      {/* Detailed Information */}
      {showDetails && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg max-w-sm">
          <h4 className={`font-medium ${riskLevel.textColor} mb-2`}>
            Risk Assessment Details
          </h4>
          <p className="text-sm text-gray-700 mb-2">
            {getRiskDescription(riskLevel.level)}
          </p>
          <p className="text-sm text-gray-600">
            <strong>Recommendation:</strong> {getRiskRecommendation(riskLevel.level)}
          </p>

          {/* Risk Breakdown */}
          <div className="mt-3 space-y-1">
            <div className="flex justify-between text-xs text-gray-600">
              <span>Score Range:</span>
              <span>{percentage}% ({riskLevel.level})</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${riskLevel.bgColor}`}
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Alternative Linear Risk Indicator
interface LinearRiskIndicatorProps {
  score: number
  label?: string
  showPercentage?: boolean
  className?: string
}

export function LinearRiskIndicator({
  score,
  label = 'Risk Level',
  showPercentage = true,
  className = ''
}: LinearRiskIndicatorProps) {
  const riskLevel = riskLevels.find(level => score <= level.threshold) || riskLevels[riskLevels.length - 1]
  const percentage = Math.round(score * 100)

  return (
    <div className={`${className}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        {showPercentage && (
          <span className={`text-sm font-medium ${riskLevel.textColor}`}>
            {percentage}%
          </span>
        )}
      </div>

      <div className="w-full bg-gray-200 rounded-full h-3 relative overflow-hidden">
        {/* Background segments */}
        <div className="absolute inset-0 flex">
          <div className="w-1/4 bg-green-200"></div>
          <div className="w-1/4 bg-yellow-200"></div>
          <div className="w-1/4 bg-orange-200"></div>
          <div className="w-1/4 bg-red-200"></div>
        </div>

        {/* Progress bar */}
        <div
          className={`h-3 rounded-full transition-all duration-1000 ease-out ${riskLevel.bgColor}`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
        <span>Critical</span>
      </div>
    </div>
  )
}

// Risk Score Badge Component
interface RiskScoreBadgeProps {
  score: number
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function RiskScoreBadge({ score, size = 'md', className = '' }: RiskScoreBadgeProps) {
  const riskLevel = riskLevels.find(level => score <= level.threshold) || riskLevels[riskLevels.length - 1]
  const percentage = Math.round(score * 100)

  const sizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2'
  }

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full ${sizeClasses[size]} ${riskLevel.bgColor.replace('bg-', 'bg-').replace('-500', '-100')} ${riskLevel.textColor} ${className}`}
    >
      <span className={`mr-1 ${size === 'sm' ? 'h-3 w-3' : size === 'md' ? 'h-4 w-4' : 'h-5 w-5'}`}>
        {React.cloneElement(riskLevel.icon as React.ReactElement, {
          className: size === 'sm' ? 'h-3 w-3' : size === 'md' ? 'h-4 w-4' : 'h-5 w-5'
        })}
      </span>
      {percentage}% {riskLevel.level}
    </span>
  )
}