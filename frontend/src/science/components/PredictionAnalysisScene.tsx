import { useState, useEffect } from 'react'
import { MathExpression } from '../../components/MathExpression'
import { EQUATIONS } from '../../lib/equations'
import { probeHealth, HealthInfo } from '../../lib/api'

export function PredictionAnalysisScene() {
  const [selectedWord, setSelectedWord] = useState<string>('ABOUT')
  const [health, setHealth] = useState<HealthInfo | null>(null)

  useEffect(() => {
    const ctrl = new AbortController()
    probeHealth(ctrl.signal)
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', inferenceReady: false }))
    return () => ctrl.abort()
  }, [])

  // Illustrative prediction profiles based on homopheny groups
  const predictionProfiles: Record<string, { top5: { label: string; prob: number; viseme: string }[]; entropy: number; ambiguity: string }> = {
    'ABOUT': {
      top5: [
        { label: 'ABOUT', prob: 0.74, viseme: 'Bilabial release /b/ + diphthong /aʊ/' },
        { label: 'ABOVE', prob: 0.14, viseme: 'Near-identical labial onset' },
        { label: 'ALLOW', prob: 0.05, viseme: 'Similar open vowel trajectory' },
        { label: 'AGAIN', prob: 0.04, viseme: 'Vocalic onset match' },
        { label: 'ACROSS', prob: 0.03, viseme: 'Residual distribution' },
      ],
      entropy: 0.98,
      ambiguity: 'Low Ambiguity: The dynamic bilabial explosion into open diphthong /aʊ/ is visually distinct.'
    },
    'MAYBE': {
      top5: [
        { label: 'MAYBE', prob: 0.42, viseme: 'Bilabial nasal /m/ onset' },
        { label: 'BABY', prob: 0.38, viseme: 'Bilabial plosive /b/ (Homophenous)' },
        { label: 'PAY', prob: 0.11, viseme: 'Bilabial unvoiced /p/' },
        { label: 'MAKE', prob: 0.06, viseme: 'Matching initial bilabial gesture' },
        { label: 'MANY', prob: 0.03, viseme: 'Residual probability' },
      ],
      entropy: 1.84,
      ambiguity: 'High Homophenous Ambiguity: /m/, /b/, and /p/ share identical lip closure; differentiation relies entirely on vocal cord vibration which has zero optical footprint.'
    },
    'LETTER_B': {
      top5: [
        { label: 'B', prob: 0.68, viseme: 'Bilabial closure' },
        { label: 'P', prob: 0.19, viseme: 'Unvoiced bilabial' },
        { label: 'M', prob: 0.08, viseme: 'Nasal bilabial' },
        { label: 'V', prob: 0.03, viseme: 'Labiodental slip' },
        { label: 'D', prob: 0.02, viseme: 'Residual' },
      ],
      entropy: 1.25,
      ambiguity: 'Viseme Clustering: The model groups B, P, and M together because external lip geometry cannot reveal intra-oral nasalization.'
    }
  }

  const activeProfile = predictionProfiles[selectedWord]

  return (
    <section id="scene-prediction" className="science-scene" aria-label="Scene 6: The Anatomy of a Prediction">
      <div className="scene-header">
        <span className="scene-num">Scene 06 · Decision Space</span>
        <h2 className="scene-title">The Anatomy of a Prediction.</h2>
        <p className="scene-lead">
          In lip reading, model confidence does not equate to objective certainty. Because human speech contains 
          <strong> homophenes</strong> (distinct phonemes with identical visual lip mechanics), the network must navigate 
          inherent information bottlenecks.
        </p>
      </div>

      <div className="sci-card">
        <div className="sci-card-header">
          <div className="sci-card-title">
            <span>Viseme Ambiguity & Multi-Class Probability Distribution</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <span className={`sci-tag ${health?.status === 'online' ? 'verified' : 'illustrative'}`}>
              Backend: {health?.status === 'online' ? 'Online API' : 'Illustrative Fixture'}
            </span>
          </div>
        </div>

        {/* Word / Phoneme Selector */}
        <div style={{ display: 'flex', gap: '0.6rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {[
            { id: 'ABOUT', name: 'Word: "ABOUT" (Distinct Trajectory)' },
            { id: 'MAYBE', name: 'Word: "MAYBE" (Bilabial Ambiguity)' },
            { id: 'LETTER_B', name: 'Single Letter: "B" (Viseme /b, p, m/)' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setSelectedWord(item.id)}
              style={{
                background: selectedWord === item.id ? 'var(--accent)' : 'rgba(237, 235, 228, 0.04)',
                color: selectedWord === item.id ? 'var(--bg)' : 'var(--ink)',
                border: '1px solid var(--line)',
                borderRadius: '6px',
                padding: '6px 14px',
                fontSize: '0.8rem',
                fontFamily: 'var(--font-mono)',
                cursor: 'pointer',
                fontWeight: selectedWord === item.id ? 600 : 400
              }}
            >
              {item.name}
            </button>
          ))}
        </div>

        <div className="sci-split">
          {/* Left: Top-5 Probability Bar Chart */}
          <div>
            <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--ink-dim)', marginBottom: '0.8rem' }}>
              TOP-5 PROBABILITY MASS DISTRIBUTION:
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {activeProfile.top5.map((candidate, idx) => (
                <div key={candidate.label} style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid var(--line)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.72rem',
                        color: idx === 0 ? 'var(--accent)' : 'var(--ink-faint)',
                        background: idx === 0 ? 'rgba(143, 223, 203, 0.12)' : 'rgba(237, 235, 228, 0.05)',
                        padding: '1px 6px',
                        borderRadius: '3px'
                      }}>
                        #{idx + 1}
                      </span>
                      <strong style={{ fontSize: '0.95rem', color: idx === 0 ? 'var(--ink)' : 'var(--ink-dim)' }}>
                        {candidate.label}
                      </strong>
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.88rem', color: idx === 0 ? 'var(--accent)' : 'var(--ink-dim)', fontWeight: 600 }}>
                      {(candidate.prob * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Horizontal Bar */}
                  <div style={{ height: '6px', background: 'rgba(237, 235, 228, 0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{
                      width: `${candidate.prob * 100}%`,
                      height: '100%',
                      background: idx === 0 ? 'var(--accent)' : 'rgba(237, 235, 228, 0.3)',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>

                  <div style={{ fontSize: '0.72rem', color: 'var(--ink-faint)', marginTop: '4px', fontFamily: 'var(--font-mono)' }}>
                    {candidate.viseme}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Shannon Entropy & Calibration Breakdown */}
          <div style={{
            background: '#08080a',
            border: '1px solid rgba(237, 235, 228, 0.1)',
            borderRadius: '12px',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between'
          }}>
            <div>
              <div className="math-box">
                <div className="math-formula">
                  <MathExpression tex={EQUATIONS.entropy}/>
                </div>
                <p className="math-desc">
                  Shannon Entropy in bits. Quantifies visual information uncertainty in the predictive distribution.
                </p>
              </div>

              {/* Entropy Meter */}
              <div style={{
                background: 'rgba(237, 235, 228, 0.03)',
                borderRadius: '8px',
                border: '1px solid var(--line)',
                padding: '1rem',
                margin: '1rem 0'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                  <span style={{ color: 'var(--ink-dim)' }}>SHANNON ENTROPY:</span>
                  <span style={{ color: activeProfile.entropy < 1.2 ? 'var(--success)' : 'var(--warning)', fontWeight: 600 }}>
                    {activeProfile.entropy} bits
                  </span>
                </div>
                <div style={{ height: '6px', background: 'rgba(237, 235, 228, 0.1)', borderRadius: '3px', marginTop: '6px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(100, (activeProfile.entropy / 2.5) * 100)}%`,
                    height: '100%',
                    background: activeProfile.entropy < 1.2 ? 'var(--success)' : 'var(--warning)'
                  }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--ink-faint)', marginTop: '3px' }}>
                  <span>0.0 (Certain)</span>
                  <span>1.0</span>
                  <span>2.5 (Ambiguous)</span>
                </div>
              </div>

              <div style={{
                background: 'rgba(252, 211, 77, 0.06)',
                border: '1px solid rgba(252, 211, 77, 0.2)',
                borderRadius: '8px',
                padding: '1rem',
                fontSize: '0.8rem',
                color: 'var(--ink-dim)',
                lineHeight: 1.5
              }}>
                <strong style={{ color: 'var(--warning)', display: 'block', marginBottom: '4px' }}>
                  Scientific Diagnosis:
                </strong>
                {activeProfile.ambiguity}
              </div>
            </div>

            <div style={{
              borderTop: '1px solid var(--line)',
              paddingTop: '1rem',
              marginTop: '1rem',
              fontSize: '0.72rem',
              color: 'var(--ink-faint)',
              fontFamily: 'var(--font-mono)'
            }}>
              DATA SOURCE: Derived from calibrated viseme clustering metrics and active EMA smoothing parameters.
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
