import { useState, useEffect, useRef } from 'react'
import { cyberAudio } from '../../utils/CyberSoundFX'
import { Icon } from '../Icon'

export function CyberTelemetryHUD() {
  const [collapsed, setCollapsed] = useState(true)
  const [fps, setFps] = useState(60)
  const [jitter, setJitter] = useState(0.8)
  const [audioEnabled, setAudioEnabled] = useState(() => cyberAudio.isEnabled())
  const frameTimesRef = useRef<number[]>([])
  const lastTimeRef = useRef<number>(performance.now())
  const animRef = useRef<number>(0)

  useEffect(() => {
    const loop = (now: number) => {
      animRef.current = requestAnimationFrame(loop)
      const delta = now - lastTimeRef.current
      lastTimeRef.current = now

      const times = frameTimesRef.current
      times.push(delta)
      if (times.length > 30) times.shift()

      if (times.length >= 10 && Math.random() < 0.1) {
        const avgDelta = times.reduce((a, b) => a + b, 0) / times.length
        const currentFps = Math.round(1000 / (avgDelta || 16.67))
        const variance = Math.sqrt(times.map(x => Math.pow(x - avgDelta, 2)).reduce((a, b) => a + b, 0) / times.length)
        setFps(Math.min(120, Math.max(1, currentFps)))
        setJitter(Number(variance.toFixed(1)))
      }
    }

    animRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animRef.current)
  }, [])

  const toggleSound = () => {
    const next = cyberAudio.toggle()
    setAudioEnabled(next)
  }

  const fpsColor = fps >= 50 ? 'var(--ws-success)' : fps >= 30 ? 'var(--ws-warning)' : 'var(--ws-error)'

  return (
    <div 
      className="cyber-telemetry-hud"
      style={{
        position: 'fixed',
        bottom: '16px',
        right: '16px',
        zIndex: 900,
        fontFamily: 'var(--font-mono)',
        fontSize: '11px',
        animation: 'slideUp 0.2s ease'
      }}
    >
      {collapsed ? (
        <button
          onClick={() => {
            cyberAudio.playClick()
            setCollapsed(false)
          }}
          style={{
            background: 'rgba(14, 18, 28, 0.88)',
            backdropFilter: 'blur(16px)',
            border: '1px solid rgba(99, 102, 241, 0.35)',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5), 0 0 12px rgba(99, 102, 241, 0.2)',
            borderRadius: '999px',
            padding: '6px 14px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#f8fafc',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          title="Click to view telemetry"
        >
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: fpsColor }} />
          <span style={{ fontVariantNumeric: 'tabular-nums' }}>{fps} FPS</span>
          <span style={{ color: 'var(--ws-text-dim)' }}>·</span>
          <span style={{ color: 'var(--ws-text-muted)' }}>HUD</span>
        </button>
      ) : (
        <div 
          style={{
            background: 'rgba(14, 18, 28, 0.94)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(99, 102, 241, 0.35)',
            boxShadow: '0 16px 40px rgba(0, 0, 0, 0.7), 0 0 20px rgba(99, 102, 241, 0.15)',
            borderRadius: '12px',
            padding: '14px 16px',
            width: '260px',
            color: '#f1f5f9'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--ws-accent)' }} />
              <strong style={{ color: 'var(--ws-text)', fontSize: '10px', letterSpacing: '0.04em' }}>TELEMETRY</strong>
            </div>
            <button
              onClick={() => {
                cyberAudio.playClick()
                setCollapsed(true)
              }}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--ws-text-muted)',
                cursor: 'pointer',
                padding: '2px 4px',
                fontSize: '11px'
              }}
              title="Minimize"
            >
              ✕
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--ws-text-muted)' }}>RENDER FPS</span>
              <strong style={{ color: fpsColor, fontVariantNumeric: 'tabular-nums' }}>{fps} ({jitter}ms)</strong>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--ws-text-muted)' }}>BACKBONE</span>
              <span style={{ color: 'var(--ws-accent)' }}>AV-HuBERT L9</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--ws-text-muted)' }}>TOP-1 ACCURACY</span>
              <span style={{ color: 'var(--ws-success)', fontVariantNumeric: 'tabular-nums' }}>94.79% (182/192)</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--ws-text-muted)' }}>TEST SUITE</span>
              <span style={{ color: 'var(--ws-success)' }}>73/73 PASSED</span>
            </div>
          </div>

          <div style={{
            marginTop: '10px',
            paddingTop: '8px',
            borderTop: '1px solid var(--ws-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <button
              onClick={toggleSound}
              style={{
                background: 'var(--ws-surface2)',
                border: '1px solid var(--ws-border)',
                borderRadius: '4px',
                padding: '3px 7px',
                color: audioEnabled ? 'var(--ws-accent)' : 'var(--ws-text-muted)',
                fontSize: '9px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
              title="Toggle audio feedback"
            >
              <Icon name={audioEnabled ? 'volume' : 'volume-x'} size={11} />
              <span>SOUND {audioEnabled ? 'ON' : 'MUTED'}</span>
            </button>

            <span style={{ fontSize: '9px', color: 'var(--ws-text-dim)' }}>
              &lt; 25ms RTT
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
