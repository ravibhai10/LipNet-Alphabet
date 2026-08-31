import { Reveal } from '../animations/Reveal'
import { Arrow } from '../components/ui'

/**
 * SECTION 14 — FINAL STATEMENT
 */
export function FinalStatement() {
  return (
    <section className="section final" id="about" aria-labelledby="final-title">
      <div className="container final-inner">
        <Reveal>
          <h2 className="headline final-title" id="final-title">
            Speech leaves
            <br />
            <em>a visual trace.</em>
          </h2>
        </Reveal>
        <Reveal delay={0.15}>
          <p className="final-sub">We are learning how to read it.</p>
        </Reveal>
        <Reveal className="final-ctas" delay={0.3}>
          <a className="btn btn-solid" href="#research">
            Explore the research <Arrow />
          </a>
          <a className="btn btn-ghost" href="#demo">
            Try the live model
          </a>
        </Reveal>
      </div>
    </section>
  )
}
