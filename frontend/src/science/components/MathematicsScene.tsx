import { useState } from 'react'
import { MathExpression } from '../../components/MathExpression'
import { EQUATIONS } from '../../lib/equations'

export function MathematicsScene() {
  // 5 illustrative classes: A, B, C, D, E
  const [logits, setLogits] = useState<number[]>([2.4, 0.8, -1.2, 1.1, -0.5])
  const [temperature, setTemperature] = useState<number>(1.0)
  const [targetClass, setTargetClass] = useState<number>(0)
  const [labelSmoothing, setLabelSmoothing] = useState<boolean>(true)
  const [alphaEma, setAlphaEma] = useState<number>(0.6)

  const classLabels = ['Letter A', 'Letter B', 'Letter C', 'Letter D', 'Letter E']

  // Softmax with temperature scaling: p_i = exp(z_i / T) / sum(exp(z_j / T))
  // Numerically stable subtraction of max logit
  const maxScaledLogit = Math.max(...logits.map((z) => z / temperature))
  const expLogits = logits.map((z) => Math.exp(z / temperature - maxScaledLogit))
  const sumExp = expLogits.reduce((acc, val) => acc + val, 0)
  const probabilities = expLogits.map((val) => val / sumExp)

  // Cross-Entropy Loss calculation
  const eps = labelSmoothing ? 0.1 : 0.0
  const K = logits.length
  const smoothedTarget = probabilities.map((_, i) =>
    i === targetClass ? (1 - eps) + eps / K : eps / K
  )

  const crossEntropyLoss = -smoothedTarget.reduce(
    (acc, targetP, i) => acc + targetP * Math.log(Math.max(1e-9, probabilities[i])),
    0
  )

  // EMA smoothing demo values
  const rawProb = probabilities[targetClass]
  const prevSmoothed = 0.35 // synthetic prior frame probability
  const currentSmoothed = (alphaEma * rawProb + (1 - alphaEma) * prevSmoothed).toFixed(3)

  return (
    <section id="scene-math" className="science-scene" aria-label="Scene 5: The Mathematics of Prediction">
      <div className="scene-header">
        <span className="scene-num">Scene 05 · Mathematical Formalism</span>
        <h2 className="scene-title">The Mathematics of Prediction.</h2>
        <p className="scene-lead">
          Neural networks do not output letters directly; they output unbounded energy scores called logits. 
          Through the geometry of the Softmax manifold, cross-entropy objective functions, and temporal exponential 
          moving averages, raw activations become probability distributions and steadier displays. This interactive sandbox illustrates the mathematics; its scores are not calibrated recognition results.
        </p>
      </div>

      <div className="sci-split">
        {/* Left: Softmax & Logits Interactive Manifold */}
        <div className="sci-card">
          <div className="sci-card-header">
            <h3 className="sci-card-title">Interactive Logits & Softmax</h3>
            <span className="sci-tag">Deterministic Sandbox</span>
          </div>

          <div className="math-box">
            <div className="math-formula">
              <MathExpression tex={EQUATIONS.softmax}/>
            </div>
            <p className="math-desc">
              Temperature-scaled softmax transforms the logit vector <MathExpression tex={EQUATIONS.logitsVector} inline/> into class probabilities, with <MathExpression tex={EQUATIONS.probabilitySimplex} inline/>.
            </p>
          </div>

          {/* Temperature Slider */}
          <div className="sci-slider-group">
            <div className="sci-slider-label">
              <span>TEMPERATURE (T) — ENTROPY CONTROL</span>
              <span style={{ color: 'var(--accent)' }}>T = {temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.2"
              max="2.5"
              step="0.05"
              value={temperature}
              aria-label="Softmax temperature"
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="sci-slider"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--ink-faint)' }}>
              <span>T=0.2 (Argmax Spike)</span>
              <span>T=1.0 (Standard)</span>
              <span>T=2.5 (High Entropy / Uniform)</span>
            </div>
          </div>

          {/* 5 Class Logits Interactive Sliders */}
          <div style={{ marginTop: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {classLabels.map((lbl, idx) => (
              <div key={lbl}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
                  <span style={{ color: idx === targetClass ? 'var(--accent)' : 'var(--ink)' }}>
                    {lbl} {idx === targetClass ? '(Target Class)' : ''}
                  </span>
                  <span style={{ color: 'var(--ink-dim)' }}>
                    z = {logits[idx].toFixed(1)} → <strong style={{ color: 'var(--accent)' }}>{(probabilities[idx] * 100).toFixed(1)}%</strong>
                  </span>
                </div>
                <input
                  type="range"
                  min="-3.0"
                  max="4.0"
                  step="0.1"
                  value={logits[idx]}
                  aria-label={`Logit for ${lbl}`}
                  onChange={(e) => {
                    const newLogits = [...logits]
                    newLogits[idx] = parseFloat(e.target.value)
                    setLogits(newLogits)
                  }}
                  className="sci-slider"
                  style={{ marginTop: '3px' }}
                />
                {/* Horizontal probability meter */}
                <div style={{
                  height: '4px',
                  background: 'rgba(237, 235, 228, 0.08)',
                  borderRadius: '2px',
                  marginTop: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${probabilities[idx] * 100}%`,
                    height: '100%',
                    background: idx === targetClass ? 'var(--accent)' : 'rgba(237, 235, 228, 0.4)',
                    transition: 'width 0.15s ease'
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Cross-Entropy Loss & Real-time EMA Stabilization */}
        <div className="sci-card">
          <div className="sci-card-header">
            <h3 className="sci-card-title">Loss & Temporal Smoothing</h3>
            <span className="sci-tag verified">LipNet Active Logic</span>
          </div>

          {/* Cross Entropy Loss Demo */}
          <div className="math-box">
            <div className="math-formula">
              <MathExpression tex={EQUATIONS.crossEntropy}/>
              <MathExpression tex={EQUATIONS.labelSmoothing} className="math-secondary-equation"/>
            </div>
            <p className="math-desc">
              Cross-entropy compares model probabilities with the smoothed target distribution. Here <MathExpression tex={String.raw`\varepsilon=0.1`} inline/> and <MathExpression tex="K=5" inline/> illustrative classes.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)', color: 'var(--ink-dim)', alignSelf: 'center' }}>
              Target Class (Ground Truth):
            </span>
            {classLabels.map((lbl, idx) => (
              <button
                key={lbl}
                onClick={() => setTargetClass(idx)}
                style={{
                  background: targetClass === idx ? 'var(--accent)' : 'rgba(237, 235, 228, 0.05)',
                  color: targetClass === idx ? 'var(--bg)' : 'var(--ink)',
                  border: '1px solid var(--line)',
                  borderRadius: '4px',
                  padding: '2px 8px',
                  fontSize: '0.72rem',
                  fontFamily: 'var(--font-mono)',
                  cursor: 'pointer'
                }}
              >
                {lbl.split(' ')[1]}
              </button>
            ))}
          </div>

          <div className="math-loss-summary" style={{
            background: 'rgba(237, 235, 228, 0.03)',
            borderRadius: '8px',
            border: '1px solid var(--line)',
            padding: '1rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontFamily: 'var(--font-mono)',
            marginBottom: '1.5rem'
          }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--ink-faint)' }}>CURRENT CROSS-ENTROPY LOSS</div>
              <div data-testid="cross-entropy-value" style={{ fontSize: '1.4rem', color: crossEntropyLoss < 1.0 ? 'var(--success)' : 'var(--warning)', fontWeight: 600 }}>
                {crossEntropyLoss.toFixed(3)} nats
              </div>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={labelSmoothing}
                onChange={(e) => setLabelSmoothing(e.target.checked)}
              />
              <span>Label Smoothing (ε = 0.1)</span>
            </label>
          </div>

          {/* EMA Probability Smoothing Demo */}
          <div className="math-box" style={{ borderLeftColor: 'var(--info)' }}>
            <div className="math-formula" style={{ color: 'var(--info)' }}>
              <MathExpression tex={EQUATIONS.ema}/>
            </div>
            <p className="math-desc">
              Exponential moving average combines the current probability with the previous smoothed score. The live interface applies it across inference windows to reduce flicker; this does not establish an accuracy gain.
            </p>
          </div>

          <div className="sci-slider-group">
            <div className="sci-slider-label">
              <span>SMOOTHING COEFFICIENT (α)</span>
              <span style={{ color: 'var(--info)' }}>α = {alphaEma.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={alphaEma}
              aria-label="EMA smoothing coefficient"
              onChange={(e) => setAlphaEma(parseFloat(e.target.value))}
              className="sci-slider"
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--ink-faint)' }}>
              <span>α=0.1 (High Inertia / Lag)</span>
              <span style={{ color: 'var(--info)' }}>α=0.6 (LipNet Default)</span>
              <span>α=0.9 (Raw Instantaneous)</span>
            </div>
          </div>

          <div style={{
            background: 'rgba(125, 225, 249, 0.06)',
            border: '1px solid rgba(125, 225, 249, 0.2)',
            borderRadius: '8px',
            padding: '0.85rem',
            fontSize: '0.78rem',
            color: 'var(--ink-dim)',
            fontFamily: 'var(--font-mono)'
          }}>
            <div>Previous smoothed score <MathExpression tex="s_{t-1}" inline/>: <span style={{ color: 'var(--ink)' }}>{(prevSmoothed * 100).toFixed(1)}%</span></div>
            <div>Current probability <MathExpression tex="p_t" inline/>: <span style={{ color: 'var(--ink)' }}>{(rawProb * 100).toFixed(1)}%</span></div>
            <div style={{ marginTop: '4px', borderTop: '1px solid rgba(237, 235, 228, 0.1)', paddingTop: '4px' }}>
              Smoothed score <MathExpression tex="s_t" inline/>: <strong style={{ color: 'var(--info)' }}>{(parseFloat(currentSmoothed) * 100).toFixed(1)}%</strong>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
