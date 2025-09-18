'use client'

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface AnomalyData {
  detected_anomalies: number
  total_transactions: number
  anomaly_rate: number
}

interface AnomalyDetectionChartProps {
  data: AnomalyData
}

export default function AnomalyDetectionChart({ data }: AnomalyDetectionChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !data) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 400
    const height = 300
    const radius = Math.min(width, height) / 2 - 20

    // Prepare data for donut chart
    const chartData = [
      {
        label: 'Normal Transactions',
        value: data.total_transactions - data.detected_anomalies,
        color: '#22c55e'
      },
      {
        label: 'Anomalous Transactions',
        value: data.detected_anomalies,
        color: '#ef4444'
      }
    ]

    const g = svg.append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`)

    // Arc generator
    const arc = d3.arc<any>()
      .innerRadius(radius * 0.6)
      .outerRadius(radius)

    const labelArc = d3.arc<any>()
      .innerRadius(radius * 0.8)
      .outerRadius(radius * 0.8)

    // Pie generator
    const pie = d3.pie<any>()
      .value(d => d.value)
      .sort(null)

    const arcs = g.selectAll('.arc')
      .data(pie(chartData))
      .enter().append('g')
      .attr('class', 'arc')

    // Draw arcs
    arcs.append('path')
      .attr('d', arc)
      .attr('fill', d => d.data.color)
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

        tooltip.transition()
          .duration(200)
          .style('opacity', 1)

        const percentage = ((d.data.value / data.total_transactions) * 100).toFixed(2)
        tooltip.html(`${d.data.label}<br/>Count: ${d.data.value.toLocaleString()}<br/>Percentage: ${percentage}%`)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8)
        d3.selectAll('.tooltip').remove()
      })

    // Add labels
    arcs.append('text')
      .attr('transform', d => `translate(${labelArc.centroid(d)})`)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .text(d => {
        const percentage = ((d.data.value / data.total_transactions) * 100)
        return percentage > 5 ? `${percentage.toFixed(1)}%` : ''
      })

    // Center text
    const centerGroup = g.append('g')
      .attr('class', 'center-text')

    centerGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '-0.5em')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('fill', '#374151')
      .text(`${(data.anomaly_rate * 100).toFixed(3)}%`)

    centerGroup.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '1em')
      .attr('font-size', '12px')
      .attr('fill', '#6b7280')
      .text('Anomaly Rate')

    // Legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(20, 20)`)

    const legendItems = legend.selectAll('.legend-item')
      .data(chartData)
      .enter().append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 20})`)

    legendItems.append('rect')
      .attr('width', 12)
      .attr('height', 12)
      .attr('fill', d => d.color)
      .attr('opacity', 0.8)

    legendItems.append('text')
      .attr('x', 18)
      .attr('y', 6)
      .attr('dy', '0.35em')
      .attr('font-size', '12px')
      .attr('fill', '#374151')
      .text(d => d.label)

  }, [data])

  return (
    <div className="w-full">
      <svg
        ref={svgRef}
        width="100%"
        height="300"
        viewBox="0 0 400 300"
        className="border rounded"
      />

      {/* Summary Stats */}
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="text-center">
          <p className="text-gray-500">Total Transactions</p>
          <p className="font-semibold text-lg">{data.total_transactions.toLocaleString()}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-500">Anomalies Detected</p>
          <p className="font-semibold text-lg text-danger-600">{data.detected_anomalies}</p>
        </div>
      </div>
    </div>
  )
}