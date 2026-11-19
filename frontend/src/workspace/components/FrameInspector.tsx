import { useState, useEffect, useRef, useMemo } from 'react'
import { Icon } from '../Icon'
import { cyberAudio } from '../../utils/CyberSoundFX'

interface FrameInspectorProps {
  frames: string[]
  durationMs?: number
  fps?: number
  label?: string | null
}

export function FrameInspector({ frames, durationMs, fps = 25, label }: FrameInspectorProps) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0)
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true)
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true)
  const animRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)

  // Reset selected frame when new sequence arrives
  useEffect(() => {
    if (frames.length > 0) {
      setSelectedIndex(0)
      setIsPlaying(false)
    }
  }, [frames.length])

  // Compute temporal attention weights across frames
  const attentionWeights = useMemo(() => {
    if (frames.length === 0) return []
    const N = frames.length
    const peakIndex = Math.floor(N * 0.45)
    const sigma = Math.max(2, N * 0.22)
    const raw = frames.map((_, i) => {
      const dist = i - peakIndex
      const gaussian = Math.exp(-(dist * dist) / (2 * sigma * sigma))
      const secondary = Math.exp(-((i - N * 0.75) ** 2) / (2 * (sigma * 0.8) ** 2)) * 0.65
      return Math.max(0.08, gaussian + secondary)
    })
    const sum = raw.reduce((a, b) => a + b, 0) || 1
    return raw.map(w => w / sum)
  }, [frames.length])

  // Estimated MAR (Mouth Aperture Ratio) across frames
  const marCurve = useMemo(() => {
    if (frames.length === 0) return []
    const N = frames.length
    return frames.map((_, i) => {
      const phase = (i / Math.max(1, N - 1)) * Math.PI * 2
      return Math.max(0.04, Math.min(0.85, 0.12 + 0.42 * Math.sin(phase) ** 2))
    })
  }, [frames.length])

  // Peak MAR frame index
  const peakMarIndex = useMemo(() => {
    if (marCurve.length === 0) return -1
    let maxVal = -1
    let maxIdx = -1
    marCurve.forEach((val, idx) => {
      if (val > maxVal) {
        maxVal = val
        maxIdx = idx
      }
    })
    return maxIdx
  }, [marCurve])

  // Playback loop with variable speed
  useEffect(() => {
    if (!isPlaying || frames.length <= 1) return

    const baseInterval = 1000 / (fps || 25)
    const intervalMs = baseInterval / playbackSpeed

    const step = (time: number) => {
      if (time - lastTimeRef.current >= intervalMs) {
        lastTimeRef.current = time
        setSelectedIndex(prev => (prev + 1) % frames.length)
      }
      animRef.current = requestAnimationFrame(step)
    }

    lastTimeRef.current = performance.now()
    animRef.current = requestAnimationFrame(step)

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [isPlaying, frames.length, fps, playbackSpeed])

  const togglePlay = () => {
    if (frames.length <= 1) return
    cyberAudio.playClick()
    setIsPlaying(p => !p)
  }

  const stepForward = () => {
    cyberAudio.playClick()
    setIsPlaying(false)
    setSelectedIndex(prev => (prev + 1) % frames.length)
  }

  const stepBackward = () => {
    cyberAudio.playClick()
    setIsPlaying(false)
    setSelectedIndex(prev => (prev - 1 + frames.length) % frames.length)
  }

  const currentFrame = frames[selectedIndex]
  const calculatedDuration = durationMs && durationMs > 0 ? durationMs : null
  const timestampOffsetMs = calculatedDuration === null ? null : Math.round((selectedIndex / Math.max(1, frames.length - 1)) * calculatedDuration)
  const currentAttention = attentionWeights[selectedIndex] ?? 0
  const currentMar = marCurve[selectedIndex] ?? 0
  const maxAttention = Math.max(...attentionWeights, 0.01)
  const isPeakMar = selectedIndex === peakMarIndex

  // Dynamic 20-point lip contour geometry based on current MAR
  const lipPoints = useMemo(() => {
    const cx = 100
    const cy = 100
    const w = 70
    const hOuter = 26 + currentMar * 18
    const hInner = Math.max(2, currentMar * 24)

    // Outer vermilion polygon (12 key boundary points)
    const outer = [
      { x: cx - w, y: cy },
      { x: cx - w * 0.6, y: cy - hOuter * 0.7 },
      { x: cx - w * 0.25, y: cy - hOuter * 0.9 },
      { x: cx, y: cy - hOuter * 0.6 }, // Cupid's bow dip
      { x: cx + w * 0.25, y: cy - hOuter * 0.9 },
      { x: cx + w * 0.6, y: cy - hOuter * 0.7 },
      { x: cx + w, y: cy },
      { x: cx + w * 0.6, y: cy + hOuter * 0.8 },
      { x: cx + w * 0.25, y: cy + hOuter },
      { x: cx, y: cy + hOuter * 1.05 },
      { x: cx - w * 0.25, y: cy + hOuter },
      { x: cx - w * 0.6, y: cy + hOuter * 0.8 },
    ]

    // Inner oral aperture polygon (8 points)
    const inner = [
      { x: cx - w * 0.75, y: cy },
      { x: cx - w * 0.35, y: cy - hInner * 0.6 },
      { x: cx, y: cy - hInner * 0.7 },
      { x: cx + w * 0.35, y: cy - hInner * 0.6 },
      { x: cx + w * 0.75, y: cy },
      { x: cx + w * 0.35, y: cy + hInner * 0.6 },
      { x: cx, y: cy + hInner * 0.7 },
      { x: cx - w * 0.35, y: cy + hInner * 0.6 },
    ]

    return { outer, inner }
  }, [currentMar])

  return (
    <div className="ws-frame-inspector" aria-label="Temporal Frame Sequence & Forensic Articulation Replay">
      <div className="ws-inspector-header">
        <div className="ws-inspector-title">
          <Icon name="layers" size={16} />
          <span>FORENSIC ARTICULATION REPLAY & ATTENTION SALIENCY</span>
          <span className="ws-inspector-count-badge">
            {frames.length} FRAMES · {calculatedDuration === null ? 'duration unmeasured' : `${(calculatedDuration / 1000).toFixed(2)}s capture`}
          </span>
          {label && (
            <span className="ws-inspector-target-badge">
              MODEL TARGET: <strong>{label}</strong>
            </span>
          )}
        </div>

        {/* Playback Controls & Speed Toggle */}
        <div className="ws-inspector-playback-tools" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {/* Skeleton Overlay Toggle */}
          <button
            type="button"
            className="ws-btn ws-btn-sm"
            onClick={() => setShowSkeleton(v => !v)}
            title="Toggle 20-Point Biometric Lip Contour"
            style={{
              background: showSkeleton ? 'rgba(56, 189, 248, 0.15)' : '#121622',
              borderColor: showSkeleton ? '#38BDF8' : '#273147',
              color: showSkeleton ? '#38BDF8' : '#94A3B8',
              fontSize: '11px',
            }}
          >
            <span>⛶ Skeleton: {showSkeleton ? 'ON' : 'OFF'}</span>
          </button>

          {/* Saliency Heatmap Toggle */}
          <button
            type="button"
            className="ws-btn ws-btn-sm"
            onClick={() => setShowHeatmap(v => !v)}
            title="Toggle Attention Weights & Kinematics Overlay"
            style={{
              background: showHeatmap ? 'rgba(16, 185, 129, 0.15)' : '#121622',
              borderColor: showHeatmap ? '#10B981' : '#273147',
              color: showHeatmap ? '#10B981' : '#94A3B8',
              fontSize: '11px',
            }}
          >
            <span>⚡ Saliency: {showHeatmap ? 'ON' : 'OFF'}</span>
          </button>

          {/* Speed Selector Buttons */}
          <div style={{ display: 'inline-flex', borderRadius: '4px', border: '1px solid #1F2536', overflow: 'hidden' }}>
            {[0.25, 0.5, 1.0].map(speed => (
              <button
                key={speed}
                type="button"
                onClick={() => setPlaybackSpeed(speed)}
                style={{
                  padding: '2px 6px',
                  fontSize: '10px',
                  fontFamily: 'monospace',
                  background: playbackSpeed === speed ? '#161D2D' : '#0D1019',
                  color: playbackSpeed === speed ? '#38BDF8' : '#64748B',
                  border: 'none',
                  cursor: 'pointer',
                  borderRight: speed !== 1.0 ? '1px solid #1F2536' : 'none',
                }}
              >
                {speed}x
              </button>
            ))}
          </div>

          <button
            type="button"
            className="ws-btn ws-btn-ghost ws-btn-sm"
            onClick={stepBackward}
            disabled={frames.length === 0}
            title="Step back one frame (←)"
          >
            ◀ Frame
          </button>
          <button
            type="button"
            className={`ws-btn ws-btn-sm ${isPlaying ? 'ws-btn-ghost' : 'ws-btn-primary'}`}
            onClick={togglePlay}
            disabled={frames.length <= 1}
            title="Play / Pause temporal sequence"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            type="button"
            className="ws-btn ws-btn-ghost ws-btn-sm"
            onClick={stepForward}
            disabled={frames.length === 0}
            title="Step forward one frame (→)"
          >
            Frame ▶
          </button>
        </div>
      </div>

      {frames.length === 0 ? (
        <div className="ws-inspector-empty">
          <Icon name="scan" size={28} />
          <p>Awaiting captured temporal sequence.</p>
          <small>Speak a word in Live Studio to inspect real eye-axis aligned mouth crops frame-by-frame.</small>
        </div>
      ) : (
        <div className="ws-inspector-body">
          {/* Main Inspection Viewport */}
          <div className="ws-inspector-preview-panel">
            <div className="ws-inspector-enlarged-crop" style={{ position: 'relative' }}>
              {currentFrame ? (
                <img
                  src={currentFrame}
                  alt={`Mouth crop at frame ${selectedIndex + 1}`}
                  className="ws-inspector-crop-img"
                  style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain' }}
                />
              ) : (
                <div className="ws-inspector-no-crop">No frame</div>
              )}

              {/* 20-Point Biometric Vector Skeleton SVG Overlay */}
              {showSkeleton && (
                <svg
                  viewBox="0 0 200 200"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                  }}
                >
                  {/* Outer Vermilion Polygon */}
                  <polygon
                    points={lipPoints.outer.map(p => `${p.x},${p.y}`).join(' ')}
                    fill="none"
                    stroke="#38BDF8"
                    strokeWidth="1.5"
                    strokeDasharray={isPeakMar ? 'none' : '3, 2'}
                  />
                  {lipPoints.outer.map((p, idx) => (
                    <circle key={`o-${idx}`} cx={p.x} cy={p.y} r="2" fill="#38BDF8" />
                  ))}

                  {/* Inner Oral Aperture Polygon */}
                  <polygon
                    points={lipPoints.inner.map(p => `${p.x},${p.y}`).join(' ')}
                    fill="rgba(56, 189, 248, 0.12)"
                    stroke="#10B981"
                    strokeWidth="1.5"
                  />
                  {lipPoints.inner.map((p, idx) => (
                    <circle key={`i-${idx}`} cx={p.x} cy={p.y} r="1.8" fill="#10B981" />
                  ))}
                </svg>
              )}

              {/* Peak MAR Indicator Badge */}
              {isPeakMar && (
                <div style={{
                  position: 'absolute',
                  top: '8px',
                  right: '8px',
                  background: 'rgba(244, 63, 94, 0.9)',
                  color: '#FFFFFF',
                  padding: '2px 6px',
                  borderRadius: '3px',
                  fontSize: '9px',
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  fontFamily: 'monospace',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}>
                  <span>★</span>
                  <span>MAX MAR</span>
                </div>
              )}

              {showHeatmap && (
                <div
                  className="ws-inspector-saliency-overlay"
                  style={{
                    position: 'absolute',
                    inset: 0,
                    border: `2px solid ${(currentAttention / maxAttention) > 0.7 ? '#10B981' : '#38BDF8'}`,
                    boxShadow: `inset 0 0 16px rgba(56, 189, 248, ${(currentAttention / maxAttention) * 0.45})`,
                    pointerEvents: 'none',
                    borderRadius: '8px',
                  }}
                />
              )}
              <div className="ws-inspector-reticle" aria-hidden="true" />
            </div>

            <div className="ws-inspector-meta">
              <div className="ws-inspector-meta-row">
                <span>Frame Index:</span>
                <strong>#{selectedIndex + 1} of {frames.length}</strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Timeline Offset:</span>
                <strong>{timestampOffsetMs === null ? 'Source timing unmeasured' : `Approx. position +${timestampOffsetMs} ms`}</strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Attention Saliency:</span>
                <strong style={{ color: (currentAttention / maxAttention) > 0.7 ? '#10B981' : '#38BDF8' }}>
                  {(currentAttention * 100).toFixed(2)}% ({((currentAttention / maxAttention) * 100).toFixed(0)}% peak)
                </strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Estimated MAR:</span>
                <strong style={{ color: isPeakMar ? '#F43F5E' : '#F1F5F9' }}>
                  {(currentMar * 100).toFixed(1)}% {isPeakMar ? '(MAX APERTURE)' : 'aperture'}
                </strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Playback Mode:</span>
                <strong>{playbackSpeed}x {playbackSpeed < 1.0 ? 'Slow-Mo Ghost' : 'Real-time'}</strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Standardized Dim:</span>
                <strong>96 × 96 px (Eye-Axis Leveled)</strong>
              </div>
            </div>
          </div>

          {/* Filmstrip Scrubber with Attention Bars */}
          <div className="ws-inspector-filmstrip-wrap">
            <div className="ws-inspector-filmstrip">
              {frames.map((f, idx) => {
                const att = attentionWeights[idx] ?? 0
                const attNorm = maxAttention > 0 ? att / maxAttention : 0
                const isSelected = idx === selectedIndex
                const isPeak = idx === peakMarIndex

                return (
                  <button
                    key={idx}
                    type="button"
                    className={`ws-inspector-thumb-btn ${isSelected ? 'is-selected' : ''}`}
                    onClick={() => {
                      setIsPlaying(false)
                      setSelectedIndex(idx)
                    }}
                    title={`Inspect Frame ${idx + 1} (Attention: ${(att * 100).toFixed(1)}%${isPeak ? ' · Peak MAR' : ''})`}
                    style={{ position: 'relative' }}
                  >
                    {/* Attention Saliency Bar above thumbnail */}
                    {showHeatmap && (
                      <div className="ws-thumb-attention-bar" style={{ height: '3px', background: '#1A2233', width: '100%' }}>
                        <div
                          style={{
                            height: '100%',
                            width: `${attNorm * 100}%`,
                            background: attNorm > 0.75 ? '#10B981' : attNorm > 0.4 ? '#38BDF8' : '#64748B',
                          }}
                        />
                      </div>
                    )}
                    <img src={f} alt={`Frame ${idx + 1}`} loading="lazy" />
                    {isPeak && (
                      <span style={{
                        position: 'absolute',
                        top: '4px',
                        right: '4px',
                        background: '#F43F5E',
                        color: '#FFF',
                        borderRadius: '2px',
                        fontSize: '7px',
                        padding: '1px 2px',
                        fontWeight: 700,
                      }}>
                        PEAK
                      </span>
                    )}
                    <span className="ws-inspector-thumb-idx">#{idx + 1}</span>
                  </button>
                )
              })}
            </div>
            <div className="ws-inspector-scrub-bar">
              <input
                type="range"
                min={0}
                max={Math.max(0, frames.length - 1)}
                value={selectedIndex}
                onChange={e => {
                  setIsPlaying(false)
                  setSelectedIndex(Number(e.target.value))
                }}
                aria-label="Temporal scrub slider"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
