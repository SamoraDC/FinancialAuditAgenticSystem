'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'
import { useQuery } from 'react-query'
import {
  MagnifyingGlassIcon,
  AdjustmentsHorizontalIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { TransactionGraph as GraphData, TransactionNode, TransactionEdge } from '../types/audit'

interface TransactionGraphProps {
  auditId: string
  className?: string
}

interface GraphLayout {
  nodes: (TransactionNode & d3.SimulationNodeDatum)[]
  links: (TransactionEdge & d3.SimulationLinkDatum<TransactionNode & d3.SimulationNodeDatum>)[]
}

interface FilterSettings {
  minAmount: number
  maxAmount: number
  showAnomalies: boolean
  entityTypes: string[]
  riskThreshold: number
}

const mockGraphData: GraphData = {
  nodes: [
    {
      id: 'company_a',
      entity_name: 'TechCorp Inc.',
      entity_type: 'company',
      total_amount: 2500000,
      transaction_count: 45,
      risk_score: 0.75,
      anomaly_flags: ['high_amount', 'unusual_timing']
    },
    {
      id: 'bank_central',
      entity_name: 'Central Bank',
      entity_type: 'bank',
      total_amount: 15000000,
      transaction_count: 120,
      risk_score: 0.15,
      anomaly_flags: []
    },
    {
      id: 'vendor_b',
      entity_name: 'Software Solutions Ltd.',
      entity_type: 'company',
      total_amount: 850000,
      transaction_count: 23,
      risk_score: 0.45,
      anomaly_flags: ['duplicate_invoices']
    },
    {
      id: 'individual_c',
      entity_name: 'John Smith',
      entity_type: 'individual',
      total_amount: 75000,
      transaction_count: 8,
      risk_score: 0.85,
      anomaly_flags: ['suspicious_pattern', 'offshore_connection']
    },
    {
      id: 'gov_agency',
      entity_name: 'Tax Authority',
      entity_type: 'government',
      total_amount: 450000,
      transaction_count: 12,
      risk_score: 0.10,
      anomaly_flags: []
    },
    {
      id: 'subsidiary_d',
      entity_name: 'TechCorp Subsidiary',
      entity_type: 'company',
      total_amount: 1200000,
      transaction_count: 35,
      risk_score: 0.60,
      anomaly_flags: ['related_party']
    }
  ],
  edges: [
    {
      id: 'edge_1',
      source: 'company_a',
      target: 'bank_central',
      transaction_amount: 500000,
      transaction_date: '2024-01-15',
      transaction_type: 'payment',
      is_anomalous: false,
      risk_indicators: []
    },
    {
      id: 'edge_2',
      source: 'company_a',
      target: 'vendor_b',
      transaction_amount: 150000,
      transaction_date: '2024-01-10',
      transaction_type: 'invoice_payment',
      is_anomalous: true,
      risk_indicators: ['unusual_amount', 'timing']
    },
    {
      id: 'edge_3',
      source: 'individual_c',
      target: 'company_a',
      transaction_amount: 75000,
      transaction_date: '2024-01-08',
      transaction_type: 'consulting_fee',
      is_anomalous: true,
      risk_indicators: ['high_risk_individual', 'cash_equivalent']
    },
    {
      id: 'edge_4',
      source: 'company_a',
      target: 'gov_agency',
      transaction_amount: 45000,
      transaction_date: '2024-01-12',
      transaction_type: 'tax_payment',
      is_anomalous: false,
      risk_indicators: []
    },
    {
      id: 'edge_5',
      source: 'company_a',
      target: 'subsidiary_d',
      transaction_amount: 300000,
      transaction_date: '2024-01-20',
      transaction_type: 'intercompany_transfer',
      is_anomalous: true,
      risk_indicators: ['related_party', 'large_amount']
    }
  ]
}

export default function TransactionGraph({ auditId, className = '' }: TransactionGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedNode, setSelectedNode] = useState<TransactionNode | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<TransactionEdge | null>(null)
  const [filters, setFilters] = useState<FilterSettings>({
    minAmount: 0,
    maxAmount: 10000000,
    showAnomalies: true,
    entityTypes: ['company', 'individual', 'bank', 'government'],
    riskThreshold: 0
  })
  const [searchTerm, setSearchTerm] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const { data: graphData, isLoading, error } = useQuery<GraphData>(
    ['transaction-graph', auditId],
    async () => {
      // In a real app, this would fetch from the API
      // const response = await fetch(`/api/v1/audit/${auditId}/transaction-graph`)
      // return response.json()

      return new Promise<GraphData>((resolve) => {
        setTimeout(() => resolve(mockGraphData), 1000)
      })
    },
    {
      refetchInterval: 30000,
    }
  )

  const filterData = useCallback((data: GraphData): GraphData => {
    if (!data) return { nodes: [], edges: [] }

    const filteredNodes = data.nodes.filter(node => {
      const matchesSearch = searchTerm === '' ||
        node.entity_name.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesEntityType = filters.entityTypes.includes(node.entity_type)
      const matchesRisk = node.risk_score >= filters.riskThreshold
      const matchesAmount = node.total_amount >= filters.minAmount &&
        node.total_amount <= filters.maxAmount
      const matchesAnomalies = !filters.showAnomalies || node.anomaly_flags.length > 0

      return matchesSearch && matchesEntityType && matchesRisk && matchesAmount
    })

    const nodeIds = new Set(filteredNodes.map(n => n.id))
    const filteredEdges = data.edges.filter(edge =>
      nodeIds.has(edge.source as string) && nodeIds.has(edge.target as string)
    )

    return {
      nodes: filteredNodes,
      edges: filteredEdges
    }
  }, [searchTerm, filters])

  useEffect(() => {
    if (!svgRef.current || !graphData) return

    const filteredData = filterData(graphData)
    renderGraph(filteredData)
  }, [graphData, filterData])

  const renderGraph = (data: GraphData) => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 800
    const height = 600
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    // Create main group
    const container = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform)
      })

    svg.call(zoom as any)

    // Create simulation
    const simulation = d3.forceSimulation<TransactionNode & d3.SimulationNodeDatum>(data.nodes)
      .force('link', d3.forceLink<TransactionNode & d3.SimulationNodeDatum, TransactionEdge & d3.SimulationLinkDatum<TransactionNode & d3.SimulationNodeDatum>>(data.edges)
        .id(d => d.id)
        .distance(100)
        .strength(0.5))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30))

    // Color scales
    const entityTypeColors = d3.scaleOrdinal<string>()
      .domain(['company', 'individual', 'bank', 'government'])
      .range(['#3b82f6', '#ef4444', '#10b981', '#8b5cf6'])

    const riskColorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([1, 0]) // Inverted so red = high risk

    // Create links
    const links = container.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(data.edges)
      .enter().append('line')
      .attr('stroke', (d: TransactionEdge) => d.is_anomalous ? '#ef4444' : '#9ca3af')
      .attr('stroke-width', (d: TransactionEdge) => Math.sqrt(d.transaction_amount / 50000) + 1)
      .attr('stroke-dasharray', (d: TransactionEdge) => d.is_anomalous ? '5,5' : 'none')
      .attr('opacity', 0.7)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedEdge(d)
        setSelectedNode(null)
      })
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1).attr('stroke-width', Math.sqrt(d.transaction_amount / 50000) + 3)

        // Show tooltip
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
          <strong>Transaction</strong><br/>
          Amount: $${d.transaction_amount.toLocaleString()}<br/>
          Type: ${d.transaction_type}<br/>
          Date: ${d.transaction_date}<br/>
          ${d.is_anomalous ? '<span style="color: #ef4444;">⚠ Anomalous</span>' : ''}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function(event, d) {
        d3.select(this).attr('opacity', 0.7).attr('stroke-width', Math.sqrt(d.transaction_amount / 50000) + 1)
        d3.selectAll('.tooltip').remove()
      })

    // Create nodes
    const nodes = container.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(data.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .call(d3.drag<SVGGElement, TransactionNode & d3.SimulationNodeDatum>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        }))

    // Add circles for nodes
    nodes.append('circle')
      .attr('r', (d: TransactionNode) => Math.sqrt(d.total_amount / 100000) + 10)
      .attr('fill', (d: TransactionNode) => entityTypeColors(d.entity_type))
      .attr('stroke', (d: TransactionNode) => riskColorScale(d.risk_score))
      .attr('stroke-width', (d: TransactionNode) => d.anomaly_flags.length > 0 ? 4 : 2)
      .attr('opacity', 0.8)
      .on('click', (event, d) => {
        setSelectedNode(d)
        setSelectedEdge(null)
      })
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 1).attr('r', Math.sqrt(d.total_amount / 100000) + 15)

        // Highlight connected edges
        links.attr('stroke', (link: TransactionEdge) =>
          link.source === d || link.target === d ? '#f59e0b' : (link.is_anomalous ? '#ef4444' : '#9ca3af'))

        // Show tooltip
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
          <strong>${d.entity_name}</strong><br/>
          Type: ${d.entity_type}<br/>
          Total Amount: $${d.total_amount.toLocaleString()}<br/>
          Transactions: ${d.transaction_count}<br/>
          Risk Score: ${(d.risk_score * 100).toFixed(0)}%<br/>
          ${d.anomaly_flags.length > 0 ? `<span style="color: #ef4444;">Flags: ${d.anomaly_flags.join(', ')}</span>` : ''}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function(event, d) {
        d3.select(this).attr('opacity', 0.8).attr('r', Math.sqrt(d.total_amount / 100000) + 10)
        links.attr('stroke', (link: TransactionEdge) => link.is_anomalous ? '#ef4444' : '#9ca3af')
        d3.selectAll('.tooltip').remove()
      })

    // Add labels
    nodes.append('text')
      .text((d: TransactionNode) => d.entity_name.length > 15 ? d.entity_name.substring(0, 15) + '...' : d.entity_name)
      .attr('dy', -15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#374151')
      .attr('font-weight', 'bold')
      .style('pointer-events', 'none')

    // Add anomaly indicators
    nodes.filter((d: TransactionNode) => d.anomaly_flags.length > 0)
      .append('text')
      .text('⚠')
      .attr('dy', 4)
      .attr('dx', -15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('fill', '#ef4444')
      .style('pointer-events', 'none')

    // Update positions on simulation tick
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      nodes
        .attr('transform', (d: any) => `translate(${d.x},${d.y})`)
    })
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
            Error loading transaction graph
          </h3>
          <p className="text-gray-500">Please try refreshing the page</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-between bg-gray-50 p-4 rounded-lg">
        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search entities..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        {/* Filter Toggle */}
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center space-x-2 px-3 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
        >
          <AdjustmentsHorizontalIcon className="h-4 w-4" />
          <span>Filters</span>
        </button>

        {/* Legend */}
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span>Company</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span>Individual</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Bank</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span>Government</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-0.5 bg-red-500" style={{ borderTop: '2px dashed' }}></div>
            <span>Anomalous</span>
          </div>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="bg-white border rounded-lg p-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Amount ($)
              </label>
              <input
                type="number"
                value={filters.minAmount}
                onChange={(e) => setFilters(prev => ({ ...prev, minAmount: Number(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Amount ($)
              </label>
              <input
                type="number"
                value={filters.maxAmount}
                onChange={(e) => setFilters(prev => ({ ...prev, maxAmount: Number(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Risk Threshold
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.riskThreshold}
                onChange={(e) => setFilters(prev => ({ ...prev, riskThreshold: Number(e.target.value) }))}
                className="w-full"
              />
              <span className="text-xs text-gray-500">{(filters.riskThreshold * 100).toFixed(0)}%</span>
            </div>
            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={filters.showAnomalies}
                  onChange={(e) => setFilters(prev => ({ ...prev, showAnomalies: e.target.checked }))}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">Show only anomalies</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Graph Container */}
      <div className="flex gap-4">
        {/* Main Graph */}
        <div className="flex-1 bg-white border rounded-lg p-4">
          <svg
            ref={svgRef}
            width="100%"
            height="600"
            viewBox="0 0 800 600"
            className="border rounded"
          />
        </div>

        {/* Details Panel */}
        {(selectedNode || selectedEdge) && (
          <div className="w-80 bg-white border rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">
                {selectedNode ? 'Entity Details' : 'Transaction Details'}
              </h3>
              <button
                onClick={() => {
                  setSelectedNode(null)
                  setSelectedEdge(null)
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>

            {selectedNode && (
              <div className="space-y-3">
                <div>
                  <span className="text-sm font-medium text-gray-500">Name:</span>
                  <p className="text-gray-900">{selectedNode.entity_name}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Type:</span>
                  <p className="text-gray-900">{selectedNode.entity_type}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Total Amount:</span>
                  <p className="text-gray-900">${selectedNode.total_amount.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Transaction Count:</span>
                  <p className="text-gray-900">{selectedNode.transaction_count}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Risk Score:</span>
                  <p className={`font-medium ${selectedNode.risk_score > 0.7 ? 'text-red-600' : selectedNode.risk_score > 0.4 ? 'text-yellow-600' : 'text-green-600'}`}>
                    {(selectedNode.risk_score * 100).toFixed(0)}%
                  </p>
                </div>
                {selectedNode.anomaly_flags.length > 0 && (
                  <div>
                    <span className="text-sm font-medium text-gray-500">Anomaly Flags:</span>
                    <div className="space-y-1">
                      {selectedNode.anomaly_flags.map((flag, index) => (
                        <span key={index} className="inline-block px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full mr-1">
                          {flag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {selectedEdge && (
              <div className="space-y-3">
                <div>
                  <span className="text-sm font-medium text-gray-500">Amount:</span>
                  <p className="text-gray-900">${selectedEdge.transaction_amount.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Type:</span>
                  <p className="text-gray-900">{selectedEdge.transaction_type}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Date:</span>
                  <p className="text-gray-900">{selectedEdge.transaction_date}</p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Status:</span>
                  <p className={`font-medium ${selectedEdge.is_anomalous ? 'text-red-600' : 'text-green-600'}`}>
                    {selectedEdge.is_anomalous ? 'Anomalous' : 'Normal'}
                  </p>
                </div>
                {selectedEdge.risk_indicators.length > 0 && (
                  <div>
                    <span className="text-sm font-medium text-gray-500">Risk Indicators:</span>
                    <div className="space-y-1">
                      {selectedEdge.risk_indicators.map((indicator, index) => (
                        <span key={index} className="inline-block px-2 py-1 bg-orange-100 text-orange-800 text-xs rounded-full mr-1">
                          {indicator}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Summary Stats */}
      {graphData && (
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-gray-900">{graphData.nodes.length}</p>
              <p className="text-sm text-gray-600">Entities</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{graphData.edges.length}</p>
              <p className="text-sm text-gray-600">Transactions</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-red-600">
                {graphData.nodes.filter(n => n.anomaly_flags.length > 0).length}
              </p>
              <p className="text-sm text-gray-600">Anomalous Entities</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-red-600">
                {graphData.edges.filter(e => e.is_anomalous).length}
              </p>
              <p className="text-sm text-gray-600">Anomalous Transactions</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}