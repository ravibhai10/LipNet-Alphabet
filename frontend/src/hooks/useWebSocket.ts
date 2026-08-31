/**
 * useWebSocket.ts — Auto-reconnecting WebSocket hook.
 */
import { useCallback, useEffect, useRef, useState } from "react"

export function useWebSocket(path: string) {
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [lastMessage, setLastMessage] = useState<MessageEvent["data"] | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    setConnecting(true)
    const ws = new WebSocket(`${import.meta.env.VITE_WS_BASE ?? ""}${path}`)
    ws.onopen = () => { setConnected(true); setConnecting(false) }
    ws.onmessage = (e) => setLastMessage(e.data)
    ws.onerror = () => { setConnected(false); setConnecting(false) }
    ws.onclose = () => {
      setConnected(false); setConnecting(false)
      timerRef.current = setTimeout(connect, 2000)
    }
    wsRef.current = ws
  }, [path])

  useEffect(() => {
    connect()
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify(data))
  }, [])

  return { connected, connecting, lastMessage, send }
}
