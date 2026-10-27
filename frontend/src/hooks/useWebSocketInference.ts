import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE, type Prediction } from '../lib/api'
import { InferenceSocket, inferenceSocketUrl, type SocketState } from '../lib/inferenceSocket'
export type InferenceSocketState = SocketState
export interface StreamResult { prediction: Prediction | null; status: 'model_unavailable' | 'ok'; frames: number; latencyMs: number }

/** Legacy opt-in hook; the product's live camera uses HTTP windows. */
export function useWebSocketInference() {
  const client = useRef<InferenceSocket | null>(null)
  const [state, setState] = useState<SocketState>('idle')
  const [lastResult, setLastResult] = useState<StreamResult | null>(null)
  const connect = useCallback(() => {
    if (!client.current) client.current = new InferenceSocket(
      inferenceSocketUrl(API_BASE, window.location.href, import.meta.env.VITE_WS_BASE as string | undefined), setState,
      data => {
        const message = data as { type?: string; status?: string; prediction?: Prediction; frames?: number; latency_ms?: number }
        if (message?.type !== 'prediction') return
        setLastResult({ prediction: message.prediction ?? null, status: message.status === 'ok' ? 'ok' : 'model_unavailable',
          frames: message.frames ?? 0, latencyMs: message.latency_ms ?? 0 })
      })
    client.current.connect()
  }, [])
  const disconnect = useCallback(() => { client.current?.disconnect(); setLastResult(null) }, [])
  const sendFrame = useCallback((frame: string) => client.current?.sendFrame(frame) ?? false, [])
  useEffect(() => { connect(); return disconnect }, [connect, disconnect])
  return { state, lastResult, sendFrame, connect, disconnect }
}
