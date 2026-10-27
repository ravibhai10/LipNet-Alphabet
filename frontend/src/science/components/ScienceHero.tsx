import { useState } from 'react'
import { MathExpression } from '../../components/MathExpression'

export function ScienceHero() {
  const [activeFrame, setActiveFrame] = useState(2)

  const frames = [
    { idx: 0, label: 't = 0ms', stage: 'Resting Position', desc: 'Neutral closed mouth, landmark baseline initialization.' },
    { idx: 1, label: 't = 66ms', stage: 'Jaw Depression', desc: 'Mandibular lowering begins, vertical aperture expands.' },
    { idx: 2, label: 't = 133ms', stage: 'Max Aperture', desc: 'Peak lip separation (A-viseme), tongue apex partially visible.' },
    { idx: 3, label: 't = 200ms', stage: 'Lateral Retraction', desc: 'Horizontal stretch at oral commissures, labial tension peaks.' },
    { idx: 4, label: 't = 266ms', stage: 'Constriction', desc: 'Labial closure initiated, velocity vector reverses direction.' },
    { idx: 5, label: 't = 333ms', stage: 'Acoustic Equiv.', desc: 'Full bilabial contact corresponding to phoneme terminal release.' },
  ]

  return (
    <section id="scene-hero" className="science-scene" aria-label="Scene 1: Can Lips Reveal Speech?">
      <div className="scene-header">
        <span className="scene-num">Scene 01 · The Foundational Question</span>
        <h1 className="scene-title">Every word leaves a trace in motion.</h1>
        <p className="scene-lead">
          Sound waves dissolve instantly in ambient noise, but the biological mechanics of speech—the coordinated acceleration of 
          jaw, lips, and tongue—leave an indelible optical signature. LipNet-AI translates silent visual kinematics into 
          linguistic tokens using spatio-temporal deep learning.
        </p>
      </div>

      <div className="sci-card" style={{ background: 'linear-gradient(180deg, #121217 0%, #0c0d10 100%)' }}>
        <div className="sci-card-header">
          <div className="sci-card-title">
            <span style={{ color: 'var(--accent)' }}>●</span> Optical Kinematics & Viseme Temporal Ribbon
          </div>
          <span className="sci-tag verified">Verified Silent Pipeline</span>
        </div>

        {/* Frame Ribbon Visualizer */}
        <div style={{ margin: '1.5rem 0' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
            gap: '1rem',
            marginBottom: '1.5rem'
          }}>
            {frames.map((f) => {
              const isActive = activeFrame === f.idx
              return (
                <div
                  key={f.idx}
                  onClick={() => setActiveFrame(f.idx)}
                  style={{
                    background: isActive ? 'rgba(143, 223, 203, 0.08)' : 'rgba(237, 235, 228, 0.02)',
                    border: `1px solid ${isActive ? 'var(--accent, #8fdfcb)' : 'rgba(237, 235, 228, 0.08)'}`,
                    borderRadius: '10px',
                    padding: '0.8rem',
                    cursor: 'pointer',
                    transition: 'all 0.25s ease',
                    boxShadow: isActive ? '0 0 20px rgba(143, 223, 203, 0.15)' : 'none',
                    textAlign: 'center'
                  }}
                >
                  <div style={{
                    width: '100%',
                    height: '70px',
                    borderRadius: '6px',
                    background: '#070709',
                    border: '1px solid rgba(237, 235, 228, 0.06)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '0.6rem',
                    position: 'relative',
                    overflow: 'hidden'
                  }}>
                    {/* Simulated SVG Lip Geometry at different stages */}
                    <svg width="80" height="50" viewBox="0 0 80 50">
                      <ellipse
                        cx="40"
                        cy="25"
                        rx={24 + (f.idx === 2 ? 4 : f.idx === 0 ? -4 : 0)}
                        ry={7 + (f.idx === 2 ? 9 : f.idx === 0 ? 1 : 4)}
                        fill="none"
                        stroke={isActive ? 'var(--accent, #8fdfcb)' : 'rgba(237, 235, 228, 0.3)'}
                        strokeWidth="1.8"
                      />
                      {/* Inner oral opening */}
                      <ellipse
                        cx="40"
                        cy="25"
                        rx={16 + (f.idx === 2 ? 3 : f.idx === 0 ? -6 : -1)}
                        ry={1 + (f.idx === 2 ? 6 : f.idx === 0 ? 0 : 2)}
                        fill={f.idx === 0 ? 'transparent' : 'rgba(143, 223, 203, 0.15)'}
                        stroke={isActive ? 'rgba(143, 223, 203, 0.6)' : 'rgba(237, 235, 228, 0.15)'}
                        strokeWidth="1.2"
                      />
                      {/* Landmarks */}
                      {isActive && (
                        <>
                          <circle cx="16" cy="25" r="2" fill="var(--accent)" />
                          <circle cx="64" cy="25" r="2" fill="var(--accent)" />
                          <circle cx="40" cy={25 - (7 + (f.idx === 2 ? 9 : 3))} r="2" fill="var(--accent)" />
                          <circle cx="40" cy={25 + (7 + (f.idx === 2 ? 9 : 3))} r="2" fill="var(--accent)" />
                        </>
                      )}
                    </svg>
                    <span style={{
                      position: 'absolute',
                      top: '4px',
                      left: '6px',
                      fontFamily: 'var(--font-mono, monospace)',
                      fontSize: '0.65rem',
                      color: isActive ? 'var(--accent)' : 'var(--ink-dim)'
                    }}>
                      <MathExpression tex={`F_{${f.idx}}`} inline/>
                    </span>
                  </div>

                  <div style={{ fontFamily: 'var(--font-mono, monospace)', fontSize: '0.75rem', color: isActive ? 'var(--ink)' : 'var(--ink-dim)' }}>
                    {f.label}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: isActive ? 'var(--accent)' : 'var(--ink-faint)', marginTop: '2px' }}>
                    {f.stage}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Active Frame Insight */}
          <div style={{
            background: 'rgba(10, 10, 12, 0.6)',
            border: '1px solid rgba(143, 223, 203, 0.2)',
            borderRadius: '8px',
            padding: '1rem 1.4rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)', textTransform: 'uppercase' }}>
                Frame Insight: {frames[activeFrame].stage} ({frames[activeFrame].label})
              </span>
              <p style={{ margin: '0.3rem 0 0', fontSize: '0.88rem', color: 'var(--ink-dim)' }}>
                {frames[activeFrame].desc}
              </p>
            </div>
            <div style={{ display: 'flex', gap: '1.2rem', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
              <div><span style={{ color: 'var(--ink-faint)' }}>Audio SNR:</span> <span style={{ color: 'var(--accent)' }}>-∞ dB (Silent)</span></div>
              <div><span style={{ color: 'var(--ink-faint)' }}>Sampling:</span> <span style={{ color: 'var(--ink)' }}>29–30 FPS</span></div>
              <div><span style={{ color: 'var(--ink-faint)' }}>Target:</span> <span style={{ color: 'var(--accent)' }}>Phoneme /æ/</span></div>
            </div>
          </div>
        </div>

        {/* Core Principles Footnote */}
        <div style={{
          borderTop: '1px solid var(--line, rgba(237, 235, 228, 0.08))',
          paddingTop: '1rem',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1.5rem',
          fontSize: '0.82rem',
          color: 'var(--ink-dim)'
        }}>
          <div>
            <strong style={{ color: 'var(--ink)', display: 'block', marginBottom: '4px' }}>Zero Acoustic Leakage</strong>
            Trained and deployed on purely visual video matrices; no microphone permissions or acoustic embeddings are processed.
          </div>
          <div>
            <strong style={{ color: 'var(--ink)', display: 'block', marginBottom: '4px' }}>Temporal Non-Invertibility</strong>
            Mouth movement cannot be classified from a single freeze-frame; phoneme identity is encoded in the rate of aperture expansion.
          </div>
          <div>
            <strong style={{ color: 'var(--ink)', display: 'block', marginBottom: '4px' }}>Biomechanical Invariance</strong>
            Standardized coordinates isolate oral articulators from speaker skull geometry and lighting variances.
          </div>
        </div>
      </div>
    </section>
  )
}
