import { Icon } from '../Icon'

interface CyberTargetReticleProps {
  isStreaming: boolean
  faceDetected: boolean
  speaking: boolean
  aperture?: number | null
  fps?: number
}

export function CyberTargetReticle({
  isStreaming,
  faceDetected,
  speaking,
  aperture,
  fps = 30,
}: CyberTargetReticleProps) {
  if (!isStreaming) return null

  return (
    <div className="ws-reticle-overlay" aria-hidden="true" style={{ pointerEvents: 'none' }}>
      {/* Laser Scanner Sweep */}
      <div className="ws-laser-scanner" />

      {/* 4 Precision Corner Reticle Brackets */}
      <div className="ws-reticle-corner ws-reticle-tl" />
      <div className="ws-reticle-corner ws-reticle-tr" />
      <div className="ws-reticle-corner ws-reticle-bl" />
      <div className="ws-reticle-corner ws-reticle-br" />

      {/* Rotating Center Target Reticle */}
      <div className="ws-reticle-center" />

      {/* Top Left Stream Telemetry */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '12px',
        fontFamily: 'var(--font-mono)',
        fontSize: '10px',
        color: 'var(--ws-text-muted)',
        display: 'flex',
        flexDirection: 'column',
        gap: '3px',
        background: 'var(--ws-surface)',
        padding: '5px 10px',
        borderRadius: '6px',
        border: '1px solid var(--ws-border)',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: faceDetected ? 'var(--ws-success)' : 'var(--ws-warning)'
          }} />
          <span style={{ color: faceDetected ? 'var(--ws-text)' : 'var(--ws-warning)', fontWeight: 550 }}>
            {faceDetected ? 'FACE TRACKED · 478 PTS' : 'SEARCHING FOR FACE…'}
          </span>
        </div>
        <div style={{ fontSize: '9px', color: 'var(--ws-text-dim)' }}>
          96×96 NORM · {fps} FPS · MAR: {aperture != null ? (aperture * 100).toFixed(1) + '%' : '—'}
        </div>
      </div>

      {/* Bottom Left Articulation Badge */}
      {speaking && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '12px',
          background: 'rgba(99, 102, 241, 0.15)',
          border: '1px solid rgba(99, 102, 241, 0.35)',
          borderRadius: '6px',
          padding: '4px 8px',
          fontFamily: 'var(--font-mono)',
          fontSize: '9px',
          color: 'var(--ws-accent)',
          letterSpacing: '0.04em',
          fontWeight: 600,
          display: 'flex',
          alignItems: 'center',
          gap: '5px'
        }}>
          <Icon name="activity" size={11} />
          ARTICULATION ACTIVE
        </div>
      )}
    </div>
  )
}
