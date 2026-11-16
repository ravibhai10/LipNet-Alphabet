import { useEffect, useRef, useSyncExternalStore } from 'react'
import { WordAssistant } from './wordAssistant'
import type { WordPrediction } from '../lib/api'
import type { WordCapture } from './wordCapture'
import { useUi } from '../ui/AppChrome'

export function useWordAssistant(request: (frames: string[], signal: AbortSignal, capture?: WordCapture) => Promise<WordPrediction | null>) {
  const requestRef = useRef(request); requestRef.current = request
  const controllerRef = useRef<WordAssistant | null>(null)
  if (!controllerRef.current) controllerRef.current = new WordAssistant({ request: (...args) => requestRef.current(...args) })
  const controller = controllerRef.current
  const state = useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot)
  const { audioEnabled } = useUi()
  const announcedResult = useRef<number | null>(null)
  useEffect(() => {
    if (!audioEnabled || state.result?.status !== 'available' || announcedResult.current === state.result.id) return
    announcedResult.current = state.result.id
    try {
      const context = new AudioContext()
      const oscillator = context.createOscillator()
      const gain = context.createGain()
      oscillator.frequency.value = 660
      gain.gain.setValueAtTime(0.04, context.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.12)
      oscillator.connect(gain); gain.connect(context.destination)
      oscillator.start(); oscillator.stop(context.currentTime + 0.12)
      void context.close()
    } catch { /* Audio may be blocked until a user gesture. */ }
  }, [audioEnabled, state.result])
  useEffect(() => {
    const timer = setInterval(controller.tick, 100)
    return () => { clearInterval(timer); controller.stop() }
  }, [controller])
  return { state, controller }
}
