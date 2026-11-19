import { useEffect, useRef } from 'react'

interface VisualSpeechEqualizerProps {
  active?: boolean
  speaking?: boolean
  aperture?: number | null
  height?: number
  className?: string
  showApertureReadout?: boolean
}

export function VisualSpeechEqualizer({
  active = false,
  speaking = false,
  aperture = 0,
  height = 56,
  className = '',
  showApertureReadout = true,
}: VisualSpeechEqualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const historyRef = useRef<{ mar: number; vel: number }[]>([])
  const lastMarRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animId = 0
    const maxPoints = 80
    const thresholdMar = 0.15 // Standard VAD activation threshold

    // Initialize history buffer if empty
    if (historyRef.current.length === 0) {
      for (let i = 0; i < maxPoints; i++) {
        historyRef.current.push({ mar: 0.05, vel: 0 })
      }
    }

    const resize = () => {
      const rect = container.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = rect.width * dpr
      canvas.height = height * dpr
      canvas.style.width = `${rect.width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    let lastTick = performance.now()

    const render = (time: number) => {
      animId = requestAnimationFrame(render)
      const dt = Math.max(1, (time - lastTick) / 1000)
      lastTick = time

      // Ingest live aperture and compute velocity
      const currentMar = active ? (aperture ?? 0.05) : 0.02
      const deltaMar = (currentMar - lastMarRef.current) / dt
      lastMarRef.current = currentMar

      historyRef.current.push({ mar: currentMar, vel: Math.abs(deltaMar) })
      if (historyRef.current.length > maxPoints) {
        historyRef.current.shift()
      }

      const width = container.clientWidth
      ctx.clearRect(0, 0, width, height)

      // Background subtle guide grid
      ctx.strokeStyle = '#1a2233'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, height * 0.5)
      ctx.lineTo(width, height * 0.5)
      ctx.stroke()

      // VAD Activation Threshold Trigger Line
      const thresholdY = height - (thresholdMar / 0.6) * (height - 12) - 6
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)'
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(0, thresholdY)
      ctx.lineTo(width, thresholdY)
      ctx.stroke()
      ctx.setLineDash([])

      // Trace 1: Mouth Aperture Ratio (MAR) Waveform (Cyan)
      ctx.beginPath()
      ctx.strokeStyle = speaking ? '#38bdf8' : active ? 'rgba(56, 189, 248, 0.5)' : '#475569'
      ctx.lineWidth = speaking ? 2 : 1.25

      historyRef.current.forEach((pt, i) => {
        const x = (i / (maxPoints - 1)) * width
        const y = height - (Math.min(pt.mar, 0.6) / 0.6) * (height - 14) - 6
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Trace 2: Articulation Velocity (dMAR/dt) (Emerald)
      ctx.beginPath()
      ctx.strokeStyle = speaking ? 'rgba(16, 185, 129, 0.75)' : 'rgba(16, 185, 129, 0.25)'
      ctx.lineWidth = 1

      historyRef.current.forEach((pt, i) => {
        const x = (i / (maxPoints - 1)) * width
        const normVel = Math.min(pt.vel / 2.5, 1.0)
        const y = height - normVel * (height * 0.45) - 4
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Draw subtle threshold label
      ctx.fillStyle = '#f59e0b'
      ctx.font = '9px "JetBrains Mono", monospace'
      ctx.fillText('VAD THRESHOLD (0.15)', 8, Math.max(12, thresholdY - 4))
    }

    animId = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(animId)
      ro.disconnect()
    }
  }, [active, speaking, aperture, height])

  const curMar = aperture ?? 0
  const curPct = (curMar * 100).toFixed(1)

  return (
    <div
      ref={containerRef}
      className={`ws-visualizer-dock ${className}`}
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        height: `${height}px`,
        background: '#090c14',
        borderTop: '1px solid #1a2233',
        borderBottom: '1px solid #1a2233',
      }}
      aria-label="Real-time Mouth Aperture and Velocity Oscilloscope"
    >
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: `${height}px` }} />
      {showApertureReadout && (
        <div style={{
          position: 'absolute',
          top: '6px',
          right: '12px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          fontFamily: 'var(--font-mono)',
          fontSize: '10px',
          pointerEvents: 'none',
        }}>
          <span style={{ color: '#38bdf8' }}>
            ● MAR: <strong style={{ fontVariantNumeric: 'tabular-nums' }}>{curPct}%</strong>
          </span>
          <span style={{ color: '#10b981' }}>
            ● VELOCITY: <strong style={{ fontVariantNumeric: 'tabular-nums' }}>ΔMAR/Δt</strong>
          </span>
          <span style={{
            color: speaking ? '#10b981' : '#64748b',
            background: speaking ? 'rgba(16, 185, 129, 0.12)' : '#121622',
            padding: '1px 6px',
            borderRadius: '4px',
            border: speaking ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #1f2536',
            fontWeight: 600,
          }}>
            {speaking ? 'VAD ACTIVE' : 'REST / SILENCE'}
          </span>
        </div>
      )}
    </div>
  )
}
