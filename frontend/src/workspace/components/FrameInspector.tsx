import { useState, useEffect, useRef } from 'react'
import { Icon } from '../Icon'

interface FrameInspectorProps {
  frames: string[]
  durationMs?: number
  fps?: number
  label?: string | null
}

export function FrameInspector({ frames, durationMs, fps = 30, label }: FrameInspectorProps) {
  const [selectedIndex, setSelectedIndex] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const animRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)

  // Reset selected frame when new sequence arrives
  useEffect(() => {
    if (frames.length > 0) {
      setSelectedIndex(0)
      setIsPlaying(false)
    }
  }, [frames.length])

  // Playback loop through frames
  useEffect(() => {
    if (!isPlaying || frames.length <= 1) return

    const intervalMs = 1000 / (fps || 30)

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
  }, [isPlaying, frames.length, fps])

  const togglePlay = () => {
    if (frames.length <= 1) return
    setIsPlaying(p => !p)
  }

  const stepForward = () => {
    setIsPlaying(false)
    setSelectedIndex(prev => (prev + 1) % frames.length)
  }

  const stepBackward = () => {
    setIsPlaying(false)
    setSelectedIndex(prev => (prev - 1 + frames.length) % frames.length)
  }

  const currentFrame = frames[selectedIndex]
  const calculatedDuration = durationMs ?? (frames.length > 0 ? (frames.length / fps) * 1000 : 0)
  const timestampOffsetMs = Math.round((selectedIndex / Math.max(1, frames.length - 1)) * calculatedDuration)

  return (
    <div className="ws-frame-inspector" aria-label="Temporal Frame Sequence Inspector">
      <div className="ws-inspector-header">
        <div className="ws-inspector-title">
          <Icon name="layers" size={16} />
          <span>TEMPORAL MOUTH SEQUENCE</span>
          <span className="ws-inspector-count-badge">
            {frames.length} FRAMES · {(calculatedDuration / 1000).toFixed(2)}s
          </span>
          {label && (
            <span className="ws-inspector-target-badge">
              TARGET: <strong>{label}</strong>
            </span>
          )}
        </div>

        <div className="ws-inspector-playback-tools">
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
            <div className="ws-inspector-enlarged-crop">
              {currentFrame ? (
                <img
                  src={currentFrame}
                  alt={`Mouth crop at frame ${selectedIndex + 1}`}
                  className="ws-inspector-crop-img"
                />
              ) : (
                <div className="ws-inspector-no-crop">No frame</div>
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
                <strong>+{timestampOffsetMs} ms</strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Standardized Dim:</span>
                <strong>96 × 96 px (Grayscale)</strong>
              </div>
              <div className="ws-inspector-meta-row">
                <span>Normalization:</span>
                <strong>Eye-Axis Aligned · $\mu=0.421, \sigma=0.165$</strong>
              </div>
            </div>
          </div>

          {/* Filmstrip Scrubber */}
          <div className="ws-inspector-filmstrip-wrap">
            <div className="ws-inspector-filmstrip">
              {frames.map((f, idx) => (
                <button
                  key={idx}
                  type="button"
                  className={`ws-inspector-thumb-btn ${idx === selectedIndex ? 'is-selected' : ''}`}
                  onClick={() => {
                    setIsPlaying(false)
                    setSelectedIndex(idx)
                  }}
                  title={`Inspect Frame ${idx + 1}`}
                >
                  <img src={f} alt={`Frame ${idx + 1}`} loading="lazy" />
                  <span className="ws-inspector-thumb-idx">#{idx + 1}</span>
                </button>
              ))}
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
