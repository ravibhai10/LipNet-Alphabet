import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { useBackend } from '../hooks/useBackend'
import { SPLITS, SEQUENCE } from '../data/architecture'

/**
 * SECTION 11 — DATA
 * Speaker-independent splitting, shown visually. Dataset counts are pulled
 * from the backend when available; nothing is fabricated.
 */

const ROWS = 8 // illustrative speaker rows (visual metaphor, not counts)

export function DataSection() {
  const { datasetStats, status } = useBackend()

  return (
    <section className="section data" id="data" aria-labelledby="data-title">
      <div className="container">
        <SectionHead
          id="data-title"
          kicker="10 — Data"
          title={
            <>
              The model learns from sequences,
              <br />
              <em>not isolated pictures.</em>
            </>
          }
          lede={`One recording is one temporal sample: ${SEQUENCE.minFrames}–${SEQUENCE.maxFrames} ordered frames at ${SEQUENCE.fps} fps, with landmarks, quality flags and metadata.`}
        />

        <Reveal className="data-flow">
          {['Recording', 'Speaker', 'Session', 'Temporal frames', 'Landmarks', 'Representation'].map((s, i) => (
            <div className="data-flow-item" key={s}>
              <span className="data-flow-name">{s}</span>
              {i < 5 && <span className="data-flow-arrow" aria-hidden="true">→</span>}
            </div>
          ))}
        </Reveal>

        <div className="data-split">
          <Reveal className="data-split-visual" aria-label="Speaker-independent split diagram">
            <p className="mono-note">Speaker-independent split · {SPLITS.train}/{SPLITS.validation}/{SPLITS.test} {SPLITS.unit}</p>
            <div className="data-rows">
              {Array.from({ length: ROWS }).map((_, r) => {
                const jitter = ((r * 37) % 7) - 3
                const train = SPLITS.train + jitter
                const val = Math.round((100 - train) / 2)
                const test = 100 - train - val
                return (
                  <div className="data-row" key={r}>
                    <span className="data-row-label mono-note">speaker {String(r + 1).padStart(2, '0')}</span>
                    <div className="data-row-bar" aria-hidden="true">
                      <span style={{ width: `${train}%` }} className="data-seg data-seg--train" />
                      <span style={{ width: `${val}%` }} className="data-seg data-seg--val" />
                      <span style={{ width: `${test}%` }} className="data-seg data-seg--test" />
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="data-legend">
              <span className="data-legend-item"><i className="data-seg data-seg--train" /> Train</span>
              <span className="data-legend-item"><i className="data-seg data-seg--val" /> Validation</span>
              <span className="data-legend-item"><i className="data-seg data-seg--test" /> Test</span>
            </div>
          </Reveal>

          <Reveal className="data-side">
            <div className="data-stamp">
              <p className="data-stamp-text">No speaker<br />leakage</p>
              <p className="mono-note">sessions never span splits</p>
            </div>
            <div className="data-counts">
              <p className="mono-note">Dataset statistics</p>
              {status === 'online' && datasetStats ? (
                <ul>
                  {Object.entries(datasetStats).map(([k, v]) => (
                    <li key={k}>
                      <span className="mono-note">{k}</span>
                      <span>{v}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="data-counts-empty">
                  Live counts unavailable — the inference backend is not connected. No counts
                  are displayed until verified real recordings exist.
                </p>
              )}
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}
