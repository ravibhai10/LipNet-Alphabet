import { Reveal } from '../animations/Reveal'
import { MathExpression } from '../components/MathExpression'
import { EQUATIONS } from '../lib/equations'
import { SectionHead } from '../components/ui'
import { useBackend } from '../hooks/useBackend'
import { APPEARANCE, GEOMETRY, MOTION, FUSION } from '../data/architecture'

/**
 * SECTION 04 — GATED FUSION
 * Three streams converge through softmax gates α. When a real backend
 * prediction exists its gate values are shown; otherwise the gate bars use a
 * neutral illustrative animation, explicitly labeled as such.
 */

const STREAMS = [
  { key: 'appearance', label: 'α appearance', short: 'αA', dim: `${APPEARANCE.outputDim}-D`, color: 'var(--accent)' },
  { key: 'geometry', label: 'α geometry', short: 'αG', dim: `${GEOMETRY.outputDim}-D`, color: 'var(--ink)' },
  { key: 'motion', label: 'α motion', short: 'αM', dim: `${MOTION.outputDim}-D`, color: 'var(--ink-dim)' },
] as const

export function Fusion() {
  const { lastPrediction } = useBackend()
  const real = lastPrediction?.gates
  const isLive = !!real

  return (
    <section className="section fusion" aria-labelledby="fusion-title">
      <div className="container">
        <SectionHead
          id="fusion-title"
          kicker="04 — Gated Fusion"
          title={
            <>
              Not every frame tells the story <em>the same way.</em>
            </>
          }
          lede="A learned Feature Projection Network produces a softmax weight over the streams at every timestep — the model itself decides when appearance, geometry or motion should dominate."
        />

        <Reveal className="fusion-board">
          <div className="fusion-inputs" aria-hidden="true">
            {STREAMS.map((s) => (
              <div className="fusion-stream" key={s.key}>
                <span className="fusion-stream-name">{s.label.replace('α ', '')}</span>
                <span className="fusion-stream-dim mono-note">{s.dim}</span>
                <span className="fusion-wire" data-stream={s.key} />
              </div>
            ))}
          </div>

          <div className="fusion-gates" aria-hidden="true">
            {STREAMS.map((s, i) => (
              <div className="fusion-gate" key={s.key}>
                <span className="fusion-gate-label mono-note">{s.short}</span>
                <div
                  className={`fusion-gate-bar ${isLive ? '' : 'fusion-gate-bar--anim'}`}
                  style={{ animationDelay: `${i * 0.5}s` }}
                >
                  <span
                    className="fusion-gate-fill"
                    style={
                      isLive && real
                        ? { width: `${Math.max(4, real[s.key] * 100)}%`, background: s.color }
                        : { background: s.color }
                    }
                  />
                </div>
                <span className="fusion-gate-value mono-note">
                  {isLive && real ? real[s.key].toFixed(2) : '~'}
                </span>
              </div>
            ))}
            <p className="fusion-sum mono-note"><MathExpression tex={EQUATIONS.fusion} inline/> · softmax per timestep</p>
          </div>

          <div className="fusion-out">
            <div className="node is-accent">
              <p className="node-title">Fused representation</p>
              <p className="node-sub fusion-out-dim">T × {FUSION.outputDim}</p>
            </div>
          </div>
        </Reveal>

        <p className={`illustrative-note ${isLive ? 'is-live' : ''}`} role="note">
          {isLive
            ? 'Live inference — gate values from the connected backend'
            : 'Illustrative gate visualization — not live inference values'}
        </p>
      </div>
    </section>
  )
}
