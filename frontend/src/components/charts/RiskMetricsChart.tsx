'use client'

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

interface RiskMetrics {
  overall_risk_score: number
  credit_risk: number
  operational_risk: number
  market_risk: number
}

interface RiskMetricsChartProps {
  data: RiskMetrics
}

export default function RiskMetricsChart({ data }: RiskMetricsChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !data) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove() // Clear previous chart

    const width = 400
    const height = 300
    const margin = { top: 20, right: 20, bottom: 40, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Prepare data
    const chartData = [
      { category: 'Credit Risk', value: data.credit_risk },
      { category: 'Operational Risk', value: data.operational_risk },
      { category: 'Market Risk', value: data.market_risk },
      { category: 'Overall Risk', value: data.overall_risk_score },
    ]

    // Scales
    const xScale = d3.scaleBand()
      .domain(chartData.map(d => d.category))
      .range([0, innerWidth])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0])

    // Color scale
    const colorScale = d3.scaleOrdinal()
      .domain(chartData.map(d => d.category))
      .range(['#ef4444', '#f97316', '#eab308', '#3b82f6'])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Bars
    g.selectAll('.bar')
      .data(chartData)
      .enter().append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.category)!)
      .attr('y', d => yScale(d.value))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerHeight - yScale(d.value))
      .attr('fill', d => colorScale(d.category) as string)
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

        tooltip.html(`${d.category}<br/>Risk Score: ${(d.value * 100).toFixed(1)}%`)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.8)
        d3.selectAll('.tooltip').remove()
      })

    // Value labels on bars
    g.selectAll('.label')
      .data(chartData)
      .enter().append('text')
      .attr('class', 'label')
      .attr('x', d => xScale(d.category)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.value) - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#374151')
      .text(d => `${(d.value * 100).toFixed(0)}%`)

    // X Axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('font-size', '10px')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)')

    // Y Axis
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${(Number(d) * 100).toFixed(0)}%`))
      .selectAll('text')
      .attr('font-size', '10px')

    // Y Axis Label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (innerHeight / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#6b7280')
      .text('Risk Score')

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
    </div>
  )
}