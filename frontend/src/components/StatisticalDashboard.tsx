'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { useQuery } from 'react-query'
import {
  ChartBarIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { StatisticalAnalysis, ChartDataPoint, TimeSeriesDataPoint } from '../types/audit'

interface StatisticalDashboardProps {
  auditId: string
  className?: string
}

const mockStatisticalData: StatisticalAnalysis = {
  benford_law: {
    expected_distribution: [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046],
    actual_distribution: [0.285, 0.189, 0.132, 0.105, 0.087, 0.072, 0.055, 0.048, 0.027],
    chi_square_score: 12.45,
    p_value: 0.132,
    is_suspicious: false
  },
  zipf_law: {
    expected_distribution: [0.5, 0.25, 0.167, 0.125, 0.1, 0.083, 0.071, 0.063, 0.056],
    actual_distribution: [0.456, 0.278, 0.145, 0.098, 0.123, 0.067, 0.089, 0.044, 0.012],
    deviation_score: 0.234,
    is_suspicious: true
  },
  outlier_detection: {
    outliers: [
      {
        transaction_id: 'txn_001',
        amount: 2500000,
        z_score: 3.45,
        deviation_type: 'amount_outlier'
      },
      {
        transaction_id: 'txn_045',
        amount: 850000,
        z_score: 2.89,
        deviation_type: 'timing_outlier'
      },
      {
        transaction_id: 'txn_089',
        amount: 1200000,
        z_score: 3.12,
        deviation_type: 'frequency_outlier'
      }
    ],
    total_outliers: 23,
    outlier_rate: 0.0149
  },
  fraud_indicators: {
    duplicate_transactions: 5,
    round_number_bias: 0.18,
    time_pattern_anomalies: 12,
    vendor_concentration_risk: 0.67
  }
}

export default function StatisticalDashboard({ auditId, className = '' }: StatisticalDashboardProps) {
  const benfordRef = useRef<SVGSVGElement>(null)
  const zipfRef = useRef<SVGSVGElement>(null)
  const outliersRef = useRef<SVGSVGElement>(null)
  const [selectedTab, setSelectedTab] = useState<'benford' | 'zipf' | 'outliers' | 'fraud'>('benford')

  const { data: statisticalData, isLoading, error } = useQuery<StatisticalAnalysis>(
    ['statistical-analysis', auditId],
    async () => {
      // In a real app, this would fetch from the API
      // const response = await fetch(`/api/v1/audit/${auditId}/statistical-analysis`)
      // return response.json()

      return new Promise<StatisticalAnalysis>((resolve) => {
        setTimeout(() => resolve(mockStatisticalData), 1000)
      })
    },
    {
      refetchInterval: 30000,
    }
  )

  useEffect(() => {
    if (statisticalData && selectedTab === 'benford') {
      renderBenfordChart()
    }
  }, [statisticalData, selectedTab])

  useEffect(() => {
    if (statisticalData && selectedTab === 'zipf') {
      renderZipfChart()
    }
  }, [statisticalData, selectedTab])

  useEffect(() => {
    if (statisticalData && selectedTab === 'outliers') {
      renderOutliersChart()
    }
  }, [statisticalData, selectedTab])

  const renderBenfordChart = () => {
    if (!benfordRef.current || !statisticalData) return

    const svg = d3.select(benfordRef.current)
    svg.selectAll('*').remove()

    const width = 600
    const height = 400
    const margin = { top: 20, right: 20, bottom: 60, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const data = statisticalData.benford_law.expected_distribution.map((expected, i) => ({
      digit: (i + 1).toString(),
      expected: expected * 100,
      actual: statisticalData.benford_law.actual_distribution[i] * 100
    }))

    // Scales
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.digit))
      .range([0, innerWidth])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => Math.max(d.expected, d.actual))!])
      .range([innerHeight, 0])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Expected bars
    g.selectAll('.bar-expected')
      .data(data)
      .enter().append('rect')
      .attr('class', 'bar-expected')
      .attr('x', d => xScale(d.digit)!)
      .attr('y', d => yScale(d.expected))
      .attr('width', xScale.bandwidth() / 2)
      .attr('height', d => innerHeight - yScale(d.expected))
      .attr('fill', '#60a5fa')
      .attr('opacity', 0.7)

    // Actual bars
    g.selectAll('.bar-actual')
      .data(data)
      .enter().append('rect')
      .attr('class', 'bar-actual')
      .attr('x', d => xScale(d.digit)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.actual))
      .attr('width', xScale.bandwidth() / 2)
      .attr('height', d => innerHeight - yScale(d.actual))
      .attr('fill', '#f87171')
      .attr('opacity', 0.7)

    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 40)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('First Digit')

    // Y Axis
    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Frequency (%)')

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 120}, 20)`)

    legend.append('rect')
      .attr('width', 15)
      .attr('height', 15)
      .attr('fill', '#60a5fa')
      .attr('opacity', 0.7)

    legend.append('text')
      .attr('x', 20)
      .attr('y', 12)
      .text('Expected')
      .attr('font-size', '12px')

    legend.append('rect')
      .attr('y', 20)
      .attr('width', 15)
      .attr('height', 15)
      .attr('fill', '#f87171')
      .attr('opacity', 0.7)

    legend.append('text')
      .attr('x', 20)
      .attr('y', 32)
      .text('Actual')
      .attr('font-size', '12px')

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text("Benford's Law Analysis")
  }

  const renderZipfChart = () => {
    if (!zipfRef.current || !statisticalData) return

    const svg = d3.select(zipfRef.current)
    svg.selectAll('*').remove()

    const width = 600
    const height = 400
    const margin = { top: 20, right: 20, bottom: 60, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const data = statisticalData.zipf_law.expected_distribution.map((expected, i) => ({
      rank: i + 1,
      expected: expected,
      actual: statisticalData.zipf_law.actual_distribution[i]
    }))

    // Scales
    const xScale = d3.scaleLinear()
      .domain([1, data.length])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => Math.max(d.expected, d.actual))!])
      .range([innerHeight, 0])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Line generator
    const line = d3.line<{ rank: number; value: number }>()
      .x(d => xScale(d.rank))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX)

    // Expected line
    g.append('path')
      .datum(data.map(d => ({ rank: d.rank, value: d.expected })))
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 3)
      .attr('d', line)

    // Actual line
    g.append('path')
      .datum(data.map(d => ({ rank: d.rank, value: d.actual })))
      .attr('fill', 'none')
      .attr('stroke', '#ef4444')
      .attr('stroke-width', 3)
      .attr('d', line)

    // Points for expected
    g.selectAll('.dot-expected')
      .data(data)
      .enter().append('circle')
      .attr('class', 'dot-expected')
      .attr('cx', d => xScale(d.rank))
      .attr('cy', d => yScale(d.expected))
      .attr('r', 4)
      .attr('fill', '#3b82f6')

    // Points for actual
    g.selectAll('.dot-actual')
      .data(data)
      .enter().append('circle')
      .attr('class', 'dot-actual')
      .attr('cx', d => xScale(d.rank))
      .attr('cy', d => yScale(d.actual))
      .attr('r', 4)
      .attr('fill', '#ef4444')

    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d => d.toString()))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 40)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Rank')

    // Y Axis
    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Frequency')

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 120}, 20)`)

    legend.append('circle')
      .attr('r', 4)
      .attr('fill', '#3b82f6')

    legend.append('text')
      .attr('x', 10)
      .attr('y', 4)
      .text('Expected')
      .attr('font-size', '12px')

    legend.append('circle')
      .attr('cy', 20)
      .attr('r', 4)
      .attr('fill', '#ef4444')

    legend.append('text')
      .attr('x', 10)
      .attr('y', 24)
      .text('Actual')
      .attr('font-size', '12px')

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text("Zipf's Law Analysis")
  }

  const renderOutliersChart = () => {
    if (!outliersRef.current || !statisticalData) return

    const svg = d3.select(outliersRef.current)
    svg.selectAll('*').remove()

    const width = 600
    const height = 400
    const margin = { top: 20, right: 20, bottom: 60, left: 80 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const outliers = statisticalData.outlier_detection.outliers

    // Scales
    const xScale = d3.scaleBand()
      .domain(outliers.map(d => d.transaction_id))
      .range([0, innerWidth])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(outliers, d => d.z_score)!])
      .range([innerHeight, 0])

    const colorScale = d3.scaleOrdinal<string>()
      .domain(['amount_outlier', 'timing_outlier', 'frequency_outlier'])
      .range(['#ef4444', '#f59e0b', '#8b5cf6'])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Bars
    g.selectAll('.bar')
      .data(outliers)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.transaction_id)!)
      .attr('y', d => yScale(d.z_score))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerHeight - yScale(d.z_score))
      .attr('fill', d => colorScale(d.deviation_type))
      .attr('opacity', 0.8)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1)

        // Tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('opacity', 0)

        tooltip.transition().duration(200).style('opacity', 1)
        tooltip.html(`
          <strong>${d.transaction_id}</strong><br/>
          Amount: $${d.amount.toLocaleString()}<br/>
          Z-Score: ${d.z_score.toFixed(2)}<br/>
          Type: ${d.deviation_type}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8)
        d3.selectAll('.tooltip').remove()
      })

    // Threshold line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', yScale(2))
      .attr('y2', yScale(2))
      .attr('stroke', '#dc2626')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')

    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end')

    // Y Axis
    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -50)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Z-Score')

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 150}, 20)`)

    const legendItems = ['amount_outlier', 'timing_outlier', 'frequency_outlier']
    legendItems.forEach((item, i) => {
      legend.append('rect')
        .attr('y', i * 20)
        .attr('width', 12)
        .attr('height', 12)
        .attr('fill', colorScale(item))

      legend.append('text')
        .attr('x', 18)
        .attr('y', i * 20 + 9)
        .text(item.replace('_', ' '))
        .attr('font-size', '10px')
        .style('text-transform', 'capitalize')
    })

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Statistical Outliers (Z-Score > 2)')
  }

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <div className="text-center">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-red-500" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Error loading statistical analysis
          </h3>
          <p className="text-gray-500">Please try refreshing the page</p>
        </div>
      </div>
    )
  }

  if (!statisticalData) return null

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Benford's Law</p>
              <p className={`text-2xl font-bold ${statisticalData.benford_law.is_suspicious ? 'text-red-600' : 'text-green-600'}`}>
                {statisticalData.benford_law.is_suspicious ? 'Suspicious' : 'Normal'}
              </p>
              <p className="text-xs text-gray-500">p-value: {statisticalData.benford_law.p_value.toFixed(3)}</p>
            </div>
            {statisticalData.benford_law.is_suspicious ? (
              <ExclamationTriangleIcon className="h-8 w-8 text-red-500" />
            ) : (
              <CheckCircleIcon className="h-8 w-8 text-green-500" />
            )}
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Zipf's Law</p>
              <p className={`text-2xl font-bold ${statisticalData.zipf_law.is_suspicious ? 'text-red-600' : 'text-green-600'}`}>
                {statisticalData.zipf_law.is_suspicious ? 'Suspicious' : 'Normal'}
              </p>
              <p className="text-xs text-gray-500">Deviation: {statisticalData.zipf_law.deviation_score.toFixed(3)}</p>
            </div>
            {statisticalData.zipf_law.is_suspicious ? (
              <ExclamationTriangleIcon className="h-8 w-8 text-red-500" />
            ) : (
              <CheckCircleIcon className="h-8 w-8 text-green-500" />
            )}
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Outliers Detected</p>
              <p className="text-2xl font-bold text-gray-900">{statisticalData.outlier_detection.total_outliers}</p>
              <p className="text-xs text-gray-500">Rate: {(statisticalData.outlier_detection.outlier_rate * 100).toFixed(2)}%</p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500">Round Number Bias</p>
              <p className={`text-2xl font-bold ${statisticalData.fraud_indicators.round_number_bias > 0.15 ? 'text-red-600' : 'text-green-600'}`}>
                {(statisticalData.fraud_indicators.round_number_bias * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-gray-500">Expected: &lt;15%</p>
            </div>
            {statisticalData.fraud_indicators.round_number_bias > 0.15 ? (
              <ExclamationTriangleIcon className="h-8 w-8 text-red-500" />
            ) : (
              <CheckCircleIcon className="h-8 w-8 text-green-500" />
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'benford', label: "Benford's Law" },
            { id: 'zipf', label: "Zipf's Law" },
            { id: 'outliers', label: 'Outliers' },
            { id: 'fraud', label: 'Fraud Indicators' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as any)}
              className={`group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                selectedTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="bg-white border rounded-lg p-6">
        {selectedTab === 'benford' && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Benford's Law Analysis</h3>
              <div className="flex items-center space-x-2">
                <InformationCircleIcon className="h-5 w-5 text-gray-400" />
                <span className="text-sm text-gray-500">
                  Chi-square: {statisticalData.benford_law.chi_square_score.toFixed(2)} |
                  p-value: {statisticalData.benford_law.p_value.toFixed(3)}
                </span>
              </div>
            </div>
            <svg ref={benfordRef} width="100%" height="400" viewBox="0 0 600 400" />
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-700">
                <strong>Benford's Law</strong> states that in many real-life datasets, the leading digit 1 occurs about 30% of the time,
                and each subsequent digit occurs with decreasing frequency. Significant deviations may indicate data manipulation or fraud.
              </p>
            </div>
          </div>
        )}

        {selectedTab === 'zipf' && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Zipf's Law Analysis</h3>
              <div className="flex items-center space-x-2">
                <InformationCircleIcon className="h-5 w-5 text-gray-400" />
                <span className="text-sm text-gray-500">
                  Deviation Score: {statisticalData.zipf_law.deviation_score.toFixed(3)}
                </span>
              </div>
            </div>
            <svg ref={zipfRef} width="100%" height="400" viewBox="0 0 600 400" />
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-700">
                <strong>Zipf's Law</strong> describes the frequency distribution of items in a dataset. In financial data,
                it can help identify unusual concentrations of activity that may indicate artificial patterns or manipulation.
              </p>
            </div>
          </div>
        )}

        {selectedTab === 'outliers' && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Statistical Outliers</h3>
              <div className="flex items-center space-x-2">
                <InformationCircleIcon className="h-5 w-5 text-gray-400" />
                <span className="text-sm text-gray-500">
                  Z-Score threshold: 2.0 | Total outliers: {statisticalData.outlier_detection.total_outliers}
                </span>
              </div>
            </div>
            <svg ref={outliersRef} width="100%" height="400" viewBox="0 0 600 400" />

            {/* Outliers Table */}
            <div className="mt-6">
              <h4 className="text-md font-medium text-gray-900 mb-3">Top Outliers</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Transaction ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Amount
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Z-Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Deviation Type
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {statisticalData.outlier_detection.outliers.map((outlier) => (
                      <tr key={outlier.transaction_id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {outlier.transaction_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          ${outlier.amount.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <span className={`font-medium ${outlier.z_score > 3 ? 'text-red-600' : 'text-orange-600'}`}>
                            {outlier.z_score.toFixed(2)}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <span className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full">
                            {outlier.deviation_type.replace('_', ' ')}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'fraud' && (
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-6">Fraud Risk Indicators</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">Duplicate Transactions</h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      statisticalData.fraud_indicators.duplicate_transactions > 0 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {statisticalData.fraud_indicators.duplicate_transactions > 0 ? 'Found' : 'None'}
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 mb-1">
                    {statisticalData.fraud_indicators.duplicate_transactions}
                  </p>
                  <p className="text-sm text-gray-500">
                    Identical transactions that may indicate errors or fraud
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">Round Number Bias</h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      statisticalData.fraud_indicators.round_number_bias > 0.15 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {statisticalData.fraud_indicators.round_number_bias > 0.15 ? 'High' : 'Normal'}
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 mb-1">
                    {(statisticalData.fraud_indicators.round_number_bias * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-gray-500">
                    Percentage of transactions with round numbers
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">Time Pattern Anomalies</h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      statisticalData.fraud_indicators.time_pattern_anomalies > 10 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {statisticalData.fraud_indicators.time_pattern_anomalies > 10 ? 'High' : 'Normal'}
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 mb-1">
                    {statisticalData.fraud_indicators.time_pattern_anomalies}
                  </p>
                  <p className="text-sm text-gray-500">
                    Transactions with unusual timing patterns
                  </p>
                </div>

                <div className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">Vendor Concentration</h4>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      statisticalData.fraud_indicators.vendor_concentration_risk > 0.6 ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {statisticalData.fraud_indicators.vendor_concentration_risk > 0.6 ? 'High' : 'Normal'}
                    </span>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 mb-1">
                    {(statisticalData.fraud_indicators.vendor_concentration_risk * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm text-gray-500">
                    Concentration of business with few vendors
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}