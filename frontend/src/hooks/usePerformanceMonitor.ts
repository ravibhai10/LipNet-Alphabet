/**
 * usePerformanceMonitor.ts — FPS, memory, and long-task monitoring.
 */
import { useEffect, useRef, useState } from "react"

export interface PerfMetrics { fps: number; memoryMB: number | null; longTasks: number; fpsHistory: number[] }

const FPS_SAMPLES = 30

export function usePerformanceMonitor(): PerfMetrics {
  const [m, setM] = useState<PerfMetrics>({ fps: 0, memoryMB: null, longTasks: 0, fpsHistory: [] })
  const lastRef = useRef(performance.now())
  const countRef = useRef(0)
  const ltRef = useRef(0)

  useEffect(() => {
    let rafId = 0
    const tick = (now: number) => {
      rafId = requestAnimationFrame(tick)
      countRef.current++
      const el = now - lastRef.current
      if (el >= 1000) {
        const fps = Math.round((countRef.current * 1000) / el)
        setM((prev) => ({
          ...prev, fps, fpsHistory: [...prev.fpsHistory, fps].slice(-FPS_SAMPLES),
          memoryMB: (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory
            ? Math.round((performance as unknown as { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize / 1_048_576) : null,
        }))
        countRef.current = 0; lastRef.current = now
      }
    }
    rafId = requestAnimationFrame(tick)
    if ("PerformanceObserver" in window) {
      const obs = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) { if (e.duration > 50) ltRef.current++ }
        setM((prev) => ({ ...prev, longTasks: ltRef.current }))
      })
      obs.observe({ entryTypes: ["longtask"] })
      return () => { cancelAnimationFrame(rafId); obs.disconnect() }
    }
    return () => cancelAnimationFrame(rafId)
  }, [])

  return m
}
