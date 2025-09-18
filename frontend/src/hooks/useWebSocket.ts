'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { RealTimeUpdate, WebSocketMessage } from '../types/audit'

interface UseWebSocketOptions {
  onMessage?: (message: RealTimeUpdate) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  shouldReconnect?: boolean
  reconnectInterval?: number
  maxReconnectAttempts?: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  lastMessage: RealTimeUpdate | null
  sendMessage: (message: any) => void
  connect: () => void
  disconnect: () => void
  reconnectionAttempts: number
}

export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    shouldReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<RealTimeUpdate | null>(null)
  const [reconnectionAttempts, setReconnectionAttempts] = useState(0)

  const websocketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const shouldConnectRef = useRef(true)

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    try {
      websocketRef.current = new WebSocket(url)

      websocketRef.current.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setReconnectionAttempts(0)
        onConnect?.()
      }

      websocketRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)

          if (message.type === 'audit_update') {
            const update: RealTimeUpdate = message.payload
            setLastMessage(update)
            onMessage?.(update)
          } else if (message.type === 'ping') {
            // Send pong response
            websocketRef.current?.send(JSON.stringify({ type: 'pong' }))
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      websocketRef.current.onclose = (event) => {
        console.log('WebSocket disconnected', event.code, event.reason)
        setIsConnected(false)
        onDisconnect?.()

        // Attempt reconnection if enabled and within limits
        if (
          shouldReconnect &&
          shouldConnectRef.current &&
          reconnectionAttempts < maxReconnectAttempts &&
          event.code !== 1000 // Not a normal closure
        ) {
          const nextAttempt = reconnectionAttempts + 1
          setReconnectionAttempts(nextAttempt)

          console.log(`Attempting reconnection ${nextAttempt}/${maxReconnectAttempts} in ${reconnectInterval}ms`)

          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval * Math.pow(1.5, nextAttempt - 1)) // Exponential backoff
        }
      }

      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        onError?.(error)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      onError?.(error as Event)
    }
  }, [url, onMessage, onConnect, onDisconnect, onError, shouldReconnect, reconnectInterval, maxReconnectAttempts, reconnectionAttempts])

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (websocketRef.current) {
      websocketRef.current.close(1000, 'Manual disconnect')
      websocketRef.current = null
    }

    setIsConnected(false)
    setReconnectionAttempts(0)
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message)
    }
  }, [])

  useEffect(() => {
    shouldConnectRef.current = true
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      shouldConnectRef.current = false
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (websocketRef.current) {
        websocketRef.current.close()
      }
    }
  }, [])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnectionAttempts
  }
}

// Hook for audit-specific WebSocket connection
export function useAuditWebSocket(auditId: string, options?: UseWebSocketOptions) {
  const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/audit/${auditId}`

  return useWebSocket(wsUrl, {
    ...options,
    onConnect: () => {
      console.log(`Connected to audit WebSocket for audit: ${auditId}`)
      options?.onConnect?.()
    }
  })
}