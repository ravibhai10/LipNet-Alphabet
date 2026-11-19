import { useState, useEffect, useRef } from 'react'

interface NoisePreset {
  label: string
  dB: number
  description: string
}

const PRESETS: NoisePreset[] = [
  { label: 'Quiet Studio', dB: 15, description: 'Near-zero ambient acoustic interference' },
  { label: 'Busy Cafe / ER', dB: 65, description: 'Overlapping conversational chatter' },
  { label: 'Subway Platform', dB: 85, description: 'High-frequency rail rumble & engine whine' },
  { label: 'Helicopter / Jet', dB: 110, description: 'Extreme acoustic saturation (audio clipping)' },
]

export function NoiseStressLab() {
  const [noiseDb, setNoiseDb] = useState<number>(15)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameId = useRef<number | null>(null)
  const phaseRef = useRef(0)

  // Physical calculations
  const speechDb = 65 // standard conversational loudness at 1m
  const snr = speechDb - noiseDb
  // Audio ASR accuracy model: sigmoid transition centered at 0 dB SNR
  const asrAccuracy = Math.round(Math.max(2, Math.min(97, 100 / (1 + Math.exp(-snr * 0.12)))))
  // Visual Speech (LipNet) accuracy is acoustic-invariant
  const visualAccuracy = 94.8

  // Sample sentence degradation simulation
  const sampleWords = ['PEOPLE', 'NEED', 'PUBLIC', 'SECURITY', 'SYSTEM']
  const corruptedSentence = sampleWords.map(w => {
    if (asrAccuracy > 80) return w
    if (asrAccuracy > 50) return Math.random() > 0.4 ? w : '***'
    if (asrAccuracy > 20) return Math.random() > 0.2 ? '???' : '[DROPPED]'
    return '[STATIC / UNINTELLIGIBLE]'
  }).join(' ')

  // Real-time canvas audio waveform animation
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let running = true

    const draw = () => {
      if (!running) return
      phaseRef.current += 0.08
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      // Centerline
      ctx.strokeStyle = '#1F2536'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, h / 2)
      ctx.lineTo(w, h / 2)
      ctx.stroke()

      // Audio waveform with noise corruption
      ctx.strokeStyle = asrAccuracy < 40 ? '#F43F5E' : asrAccuracy < 70 ? '#F59E0B' : '#38BDF8'
      ctx.lineWidth = 1.5
      ctx.beginPath()

      const noiseAmp = (noiseDb / 110) * (h / 2.5)

      for (let x = 0; x < w; x += 2) {
        const speechSignal = Math.sin(x * 0.04 + phaseRef.current) * (h / 5) * Math.sin(x * 0.01)
        const noiseComponent = (Math.random() - 0.5) * noiseAmp
        const y = h / 2 + speechSignal + noiseComponent
        if (x === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()

      animFrameId.current = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      running = false
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current)
    }
  }, [noiseDb, asrAccuracy])

  return (
    <div style={{
      background: '#0D1019',
      border: '1px solid #1F2536',
      borderRadius: '8px',
      padding: '16px',
      fontFamily: 'monospace',
      fontSize: '12px',
      color: '#F1F5F9',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#F43F5E', fontSize: '13px' }}>⚡</span>
          <span style={{ fontWeight: 600, fontSize: '13px' }}>Acoustic Noise Stress Simulator</span>
          <span style={{ fontSize: '10px', color: '#64748B' }}>
            (Audio ASR vs. LipNet Optical Robustness)
          </span>
        </div>
        <div style={{
          fontSize: '11px',
          color: snr >= 0 ? '#10B981' : '#F43F5E',
          fontVariantNumeric: 'tabular-nums',
        }}>
          SNR: {snr > 0 ? `+${snr}` : snr} dB
        </div>
      </div>

      {/* Preset Pills */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', flexWrap: 'wrap' }}>
        {PRESETS.map(p => (
          <button
            key={p.label}
            type="button"
            onClick={() => setNoiseDb(p.dB)}
            style={{
              padding: '4px 10px',
              fontSize: '11px',
              borderRadius: '4px',
              border: '1px solid',
              borderColor: noiseDb === p.dB ? '#38BDF8' : '#1F2536',
              background: noiseDb === p.dB ? '#161D2D' : '#121622',
              color: noiseDb === p.dB ? '#38BDF8' : '#94A3B8',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {p.label} ({p.dB} dB)
          </button>
        ))}
      </div>

      {/* Noise Slider */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748B', marginBottom: '6px' }}>
          <span>Ambient Acoustic Noise Level</span>
          <span style={{ color: '#F1F5F9', fontVariantNumeric: 'tabular-nums' }}>{noiseDb} dB SPL</span>
        </div>
        <input
          type="range"
          min={0}
          max={110}
          step={5}
          value={noiseDb}
          onChange={e => setNoiseDb(parseInt(e.target.value, 10))}
          style={{
            width: '100%',
            accentColor: '#38BDF8',
            cursor: 'pointer',
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#475569', marginTop: '2px' }}>
          <span>0 dB (Anechoic)</span>
          <span>45 dB</span>
          <span>70 dB</span>
          <span>90 dB</span>
          <span>110 dB (Jet Engine)</span>
        </div>
      </div>

      {/* Audio Waveform Canvas */}
      <div style={{
        background: '#07080B',
        border: '1px solid #1F2536',
        borderRadius: '6px',
        padding: '8px',
        marginBottom: '16px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748B', marginBottom: '4px' }}>
          <span>Microphone Input Waveform</span>
          <span>{noiseDb > 75 ? '⚠️ High Distortion' : 'Nominal Signal'}</span>
        </div>
        <canvas
          ref={canvasRef}
          width={600}
          height={60}
          style={{ width: '100%', height: '60px', display: 'block' }}
        />
      </div>

      {/* Side-by-side Comparative Resilience Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '12px' }}>
        {/* Standard Microphone ASR */}
        <div style={{
          background: '#121622',
          border: '1px solid #1F2536',
          borderRadius: '6px',
          padding: '12px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: '#94A3B8', fontWeight: 600 }}>Standard Audio ASR</span>
            <span style={{
              fontSize: '10px',
              padding: '1px 6px',
              borderRadius: '2px',
              background: asrAccuracy < 30 ? 'rgba(244, 63, 94, 0.2)' : 'rgba(56, 189, 248, 0.2)',
              color: asrAccuracy < 30 ? '#F43F5E' : '#38BDF8',
            }}>
              {asrAccuracy < 30 ? 'COLLAPSED' : asrAccuracy < 70 ? 'DEGRADED' : 'NORMAL'}
            </span>
          </div>
          <div style={{ fontSize: '20px', fontWeight: 700, fontVariantNumeric: 'tabular-nums', color: asrAccuracy < 50 ? '#F43F5E' : '#F1F5F9', marginBottom: '4px' }}>
            {asrAccuracy}%
          </div>
          <div style={{ fontSize: '10px', color: '#64748B', marginBottom: '8px' }}>
            Estimated Word Recognition Accuracy
          </div>
          <div style={{
            background: '#07080B',
            border: '1px solid #1F2536',
            borderRadius: '4px',
            padding: '6px 8px',
            fontSize: '10px',
            color: asrAccuracy < 40 ? '#F43F5E' : '#94A3B8',
            minHeight: '36px',
            lineHeight: '1.4',
          }}>
            "{corruptedSentence}"
          </div>
        </div>

        {/* LipNet Visual Speech Engine */}
        <div style={{
          background: '#121622',
          border: '1px solid #10B981',
          borderRadius: '6px',
          padding: '12px',
          position: 'relative',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: '#10B981', fontWeight: 600 }}>LipNet Visual Speech</span>
            <span style={{
              fontSize: '10px',
              padding: '1px 6px',
              borderRadius: '2px',
              background: 'rgba(16, 185, 129, 0.2)',
              color: '#10B981',
            }}>
              IMMUNE TO NOISE
            </span>
          </div>
          <div style={{ fontSize: '20px', fontWeight: 700, fontVariantNumeric: 'tabular-nums', color: '#10B981', marginBottom: '4px' }}>
            {visualAccuracy}%
          </div>
          <div style={{ fontSize: '10px', color: '#64748B', marginBottom: '8px' }}>
            Optical Kinematic Stability (29 Frames)
          </div>
          <div style={{
            background: '#07080B',
            border: '1px solid #1F2536',
            borderRadius: '4px',
            padding: '6px 8px',
            fontSize: '10px',
            color: '#10B981',
            minHeight: '36px',
            lineHeight: '1.4',
          }}>
            "PEOPLE NEED PUBLIC SECURITY SYSTEM"
          </div>
        </div>
      </div>
    </div>
  )
}
