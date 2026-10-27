import { useState } from 'react'
import { MathExpression } from '../../components/MathExpression'
import { EQUATIONS } from '../../lib/equations'

export function TemporalSequenceScene() {
  const [numFrames, setNumFrames] = useState(30)
  const [frameRate, setFrameRate] = useState(25)
  const [scrubIndex, setScrubIndex] = useState(14)

  // Real-time calculated duration
  const durationSec = ((numFrames - 1) / frameRate).toFixed(2)
  const durationMs = Math.round(((numFrames - 1) / frameRate) * 1000)

  // Nyquist limit for speech: Syllable rate in English is ~4–6 Hz.
  // Labial articulatory transitions require at least 20–25 Hz optical sampling to avoid aliasing.
  const isNyquistSafe = frameRate >= 20

  // Synthetic phonetic aperture envelope: Bell curve simulating opening & closing for syllable /ba/
  const calculateAperture = (idx: number) => {
    const center = numFrames / 2
    const sigma = numFrames / 5
    const val = Math.exp(-Math.pow(idx - center, 2) / (2 * Math.pow(sigma, 2)))
    return Math.max(0.05, val)
  }

  const currentAperture = calculateAperture(scrubIndex)
  const prevAperture = calculateAperture(Math.max(0, scrubIndex - 1))
  const velocity = ((currentAperture - prevAperture) * frameRate).toFixed(2)

  return (
    <section id="scene-temporal" className="science-scene" aria-label="Scene 3: Time Becomes Information">
      <div className="scene-header">
        <span className="scene-num">Scene 03 · Temporal Dynamics</span>
        <h2 className="scene-title">Time Becomes Information.</h2>
        <p className="scene-lead">
          In isolation, a single image of open lips is ambiguous: is the mouth expanding into an open vowel, 
          or contracting into a bilabial stop? Lip reading is fundamentally the study of 
          <strong> articulatory velocities</strong> across time.
        </p>
      </div>

      <div className="sci-split">
        {/* Left: Interactive Sampling Duration Calculator */}
        <div className="sci-card">
          <div className="sci-card-header">
            <h3 className="sci-card-title">Articulatory Window Calculator</h3>
            <span className="sci-tag verified">Nyquist Speech Mechanics</span>
          </div>

          <div className="math-box">
            <div className="math-formula">
              <MathExpression tex={EQUATIONS.duration}/>
            </div>
            <p className="math-desc">
              Temporal window length for N discrete video frames captured at frequency f (Hz).
            </p>
          </div>

          {/* Frame count slider */}
          <div className="sci-slider-group">
            <div className="sci-slider-label">
              <span>TEMPORAL WINDOW DEPTH (N FRAMES)</span>
              <span style={{ color: 'var(--accent)' }}>{numFrames} frames</span>
            </div>
            <input
              type="range"
              min="15"
              max="60"
              step="1"
              value={numFrames}
              onChange={(e) => {
                const val = parseInt(e.target.value)
                setNumFrames(val)
                if (scrubIndex >= val) setScrubIndex(val - 1)
              }}
              className="sci-slider"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--ink-faint)', marginTop: '4px' }}>
              <span>15 (Sub-syllabic)</span>
              <span style={{ color: 'var(--accent)' }}>29 (LRW / AV-HuBERT)</span>
              <span style={{ color: 'var(--info)' }}>30 (GRID Alphabet)</span>
              <span>60 (Multi-word)</span>
            </div>
          </div>

          {/* Frame rate slider */}
          <div className="sci-slider-group" style={{ marginTop: '1.5rem' }}>
            <div className="sci-slider-label">
              <span>OPTICAL SAMPLING FREQUENCY (f)</span>
              <span style={{ color: 'var(--accent)' }}>{frameRate} FPS</span>
            </div>
            <input
              type="range"
              min="15"
              max="60"
              step="5"
              value={frameRate}
              onChange={(e) => setFrameRate(parseInt(e.target.value))}
              className="sci-slider"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--ink-faint)', marginTop: '4px' }}>
              <span>15 FPS</span>
              <span style={{ color: 'var(--accent)' }}>25 FPS (GRID Dataset)</span>
              <span style={{ color: 'var(--info)' }}>30 FPS (Webcam Standard)</span>
              <span>60 FPS</span>
            </div>
          </div>

          {/* Results Summary Box */}
          <div style={{
            background: 'rgba(237, 235, 228, 0.03)',
            borderRadius: '8px',
            border: '1px solid var(--line)',
            padding: '1.2rem',
            marginTop: '1.5rem',
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '0.75rem',
            fontFamily: 'var(--font-mono)',
            textAlign: 'center'
          }}>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>TOTAL DURATION</div>
              <div style={{ fontSize: '1.1rem', color: 'var(--ink)', fontWeight: 600 }}>{durationSec}s</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--ink-dim)' }}>({durationMs} ms)</div>
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>STRIDE INTERVAL</div>
              <div style={{ fontSize: '1.1rem', color: 'var(--accent)', fontWeight: 600 }}>
                {(1000 / frameRate).toFixed(1)} ms
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--ink-dim)' }}>per frame Δt</div>
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>NYQUIST STATUS</div>
              <div style={{
                fontSize: '0.85rem',
                fontWeight: 600,
                marginTop: '4px',
                color: isNyquistSafe ? 'var(--success)' : 'var(--warning)'
              }}>
                {isNyquistSafe ? 'Adequate (≥20Hz)' : 'Aliasing Risk (<20Hz)'}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--ink-dim)' }}>Speech Bandwidth</div>
            </div>
          </div>
        </div>

        {/* Right: Interactive Kinematic Aperture Visualizer */}
        <div className="sci-card">
          <div className="sci-card-header">
            <h3 className="sci-card-title">Articulatory Velocity Inspector</h3>
            <span className="sci-tag">Scrub Timeline</span>
          </div>

          <p style={{ fontSize: '0.82rem', color: 'var(--ink-dim)', margin: '0 0 1rem' }}>
            Drag the scrub head across the {numFrames}-frame sequence. Observe how the 
            <strong> velocity derivative</strong> (aperture change over time) reveals whether the mouth is opening or closing.
          </p>

          {/* Interactive Aperture Waveform SVG */}
          <div style={{
            background: '#070709',
            border: '1px solid rgba(237, 235, 228, 0.08)',
            borderRadius: '10px',
            padding: '1rem',
            position: 'relative'
          }}>
            <svg width="100%" height="120" viewBox={`0 0 ${numFrames} 100`} preserveAspectRatio="none">
              {/* Baseline grid */}
              <line x1="0" y1="80" x2={numFrames} y2="80" stroke="rgba(237, 235, 228, 0.1)" strokeWidth="0.5" strokeDasharray="1,1" />
              <line x1="0" y1="20" x2={numFrames} y2="20" stroke="rgba(237, 235, 228, 0.1)" strokeWidth="0.5" strokeDasharray="1,1" />

              {/* Aperture curve */}
              <path
                d={Array.from({ length: numFrames }).map((_, i) => {
                  const x = i
                  const y = 85 - calculateAperture(i) * 70
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                }).join(' ')}
                fill="none"
                stroke="var(--accent)"
                strokeWidth="1.5"
              />

              {/* Shaded area */}
              <path
                d={
                  `M 0 85 ` +
                  Array.from({ length: numFrames }).map((_, i) => `${i} ${85 - calculateAperture(i) * 70}`).join(' L ') +
                  ` L ${numFrames - 1} 85 Z`
                }
                fill="rgba(143, 223, 203, 0.08)"
              />

              {/* Scrubber vertical needle */}
              <line
                x1={scrubIndex}
                y1="0"
                x2={scrubIndex}
                y2="100"
                stroke="var(--ink)"
                strokeWidth="1"
              />
              <circle
                cx={scrubIndex}
                cy={85 - currentAperture * 70}
                r="2.5"
                fill="var(--ink)"
              />
            </svg>

            {/* Scrubber Slider */}
            <input
              type="range"
              min="0"
              max={numFrames - 1}
              value={scrubIndex}
              onChange={(e) => setScrubIndex(parseInt(e.target.value))}
              className="sci-slider"
              style={{ marginTop: '0.8rem' }}
            />
          </div>

          {/* Kinematic Measurements for active frame */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '0.75rem',
            marginTop: '1.2rem',
            fontFamily: 'var(--font-mono)'
          }}>
            <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>FRAME INDEX</div>
              <div style={{ fontSize: '0.95rem', color: 'var(--ink)' }}>Frame {scrubIndex + 1} / {numFrames}</div>
            </div>
            <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>LIP APERTURE</div>
              <div style={{ fontSize: '0.95rem', color: 'var(--accent)' }}>{(currentAperture * 100).toFixed(1)}%</div>
            </div>
            <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>VELOCITY (dy/dt)</div>
              <div style={{ fontSize: '0.95rem', color: parseFloat(velocity) > 0 ? 'var(--info)' : parseFloat(velocity) < 0 ? 'var(--warning)' : 'var(--ink-dim)' }}>
                {parseFloat(velocity) > 0 ? `+${velocity}` : velocity} s⁻¹
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
