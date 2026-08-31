import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { PIPELINE_STEPS } from '../data/architecture'

/**
 * SECTION 10 — RESEARCH PIPELINE
 * The complete verified pipeline, end to end.
 */
export function ResearchPipeline() {
  return (
    <section className="section research-pipeline" aria-labelledby="rp-title">
      <div className="container">
        <SectionHead
          id="rp-title"
          kicker="09 — Research Pipeline"
          title={
            <>
              Real video, real landmarks,
              <br />
              <em>real architecture.</em>
            </>
          }
        />

        <ol className="rp-list">
          {PIPELINE_STEPS.map((s, i) => (
            <Reveal as="li" key={s.id} className="rp-step">
              <span className="rp-index mono-note">{String(i + 1).padStart(2, '0')}</span>
              <div className="rp-body">
                <p className="rp-label">{s.label}</p>
                <p className="mono-note">{s.note}</p>
              </div>
              {i < PIPELINE_STEPS.length - 1 && <span className="rp-wire" aria-hidden="true" />}
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  )
}
