import { useEffect, useRef, useState } from 'react'
import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { useInView } from '../hooks/useInView'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { useBackend } from '../hooks/useBackend'
import { UNCERTAINTY } from '../data/architecture'

/**
 * SECTION 08 — UNCERTAINTY
 * Confidence, margin and entropy gate the final answer: the system may
 * abstain instead of forcing a prediction. Thresholds are fit on validation
 * data (configs/alphabet_dataset.yaml). No numbers are invented here.
 */

const STATES = UNCERTAINTY.states

export function Uncertainty() {
  const [active, setActive] = useState(0)
  const [sectionRef, inView] = useInView<HTMLElement>({ threshold: 0.3 })
  const reduced = useReducedMotion()
  const timer = useRef<number | undefined>(undefined)
  const { lastPrediction } = useBackend()

  useEffect(() => {
    if (reduced || !inView) return
    timer.current = window.setInterval(() => {
      setActive((a) => (a + 1) % STATES.length)
    }, 1600)
    return () => window.clearInterval(timer.current)
  }, [reduced, inView])

  const real = lastPrediction?.uncertainty

  return (
    <section className="section uncertainty" id="uncertainty" ref={sectionRef} aria-labelledby="uncertainty-title">
      <div className="container">
        <SectionHead
          id="uncertainty-title"
          kicker="07 — Uncertainty"
          title={
            <>
              Knowing when <em>not to guess.</em>
            </>
          }
          lede="Confidence, margin and entropy are estimated for every prediction. When visual evidence is insufficient the system refuses to force an answer — thresholds are calibrated on validation data only."
        />

        <Reveal className="unc-board">
          <div className="unc-metrics">
            {UNCERTAINTY.metrics.map((m) => (
              <div className="unc-metric" key={m}>
                <span className="unc-metric-name mono-note">{m}</span>
                <span className="unc-metric-value mono-note">
                  {real && m !== 'Prediction' ? 'live' : '—'}
                </span>
              </div>
            ))}
          </div>

          <div className="unc-states" role="list">
            {STATES.map((s, i) => (
              <span
                key={s}
                role="listitem"
                className={`unc-state ${i === active ? 'unc-state--active' : ''} ${
                  real?.state && s.toUpperCase() === real.state.toUpperCase() ? 'unc-state--real' : ''
                }`}
              >
                {s}
              </span>
            ))}
          </div>
        </Reveal>

        <p className={`illustrative-note ${real?.state ? 'is-live' : ''}`} role="note">
          {real?.state
            ? 'Live inference — uncertainty state from the connected backend'
            : 'Illustrative state cycle — no inference values are fabricated'}
        </p>
      </div>
    </section>
  )
}
