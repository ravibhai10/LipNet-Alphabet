import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { useBackend } from '../hooks/useBackend'

/**
 * SECTION 12 — RESEARCH RESULTS
 * Only real results are displayed. Until a real-data evaluation exists
 * (see PHASE6_REPORT.md), every metric shows an elegant placeholder.
 */

const METRICS = ['Accuracy', 'Macro F1', 'Top-3', 'Top-5', 'ECE'] as const

export function Results() {
  const { status } = useBackend()
  const online = status === 'online'

  return (
    <section className="section results" id="results" aria-labelledby="results-title">
      <div className="container">
        <SectionHead
          id="results-title"
          kicker="10 — Research Results"
          title={
            <>
              Numbers, <em>when earned.</em>
            </>
          }
          lede="Results appear here only when produced by a speaker-independent evaluation on real recordings. Nothing is projected, extrapolated or invented."
        />

        <Reveal className="results-grid" role="list">
          {METRICS.map((m) => (
            <div className="results-card" role="listitem" key={m}>
              <p className="results-metric mono-note">{m}</p>
              <p className="results-value">Awaiting real-data evaluation</p>
              <span className="results-placeholder" aria-hidden="true" />
            </div>
          ))}
          <div className="results-card results-card--wide" role="listitem">
            <p className="results-metric mono-note">Confusion matrix</p>
            <div className="results-matrix" aria-hidden="true">
              {Array.from({ length: 36 }).map((_, i) => (
                <span key={i} className="results-cell" />
              ))}
            </div>
            <p className="mono-note">26 × 26 — rendered after the first real-data evaluation</p>
          </div>
        </Reveal>

        <p className={`illustrative-note ${online ? '' : ''}`} role="note">
          {online
            ? 'Backend connected — this research evaluation awaits a verified speaker-independent report. Saved alphabet metrics are available in the workspace Experiments page.'
            : 'Connect the backend to inspect available saved reports. This section awaits a verified speaker-independent research evaluation.'}
        </p>
      </div>
    </section>
  )
}
