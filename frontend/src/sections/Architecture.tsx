import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { APPEARANCE, GEOMETRY, MOTION, FUSION, CONFORMER, DECODER } from '../data/architecture'

/**
 * SECTION 13 — ARCHITECTURE OVERVIEW
 * Presentation-quality full-width diagram of the verified system.
 */
export function Architecture() {
  return (
    <section className="section architecture" id="architecture" aria-labelledby="arch-title">
      <div className="container">
        <SectionHead
          id="arch-title"
          kicker="12 — Architecture Overview"
          title={
            <>
              The whole system, <em>one page.</em>
            </>
          }
        />

        <Reveal className="arch">
          <div className="arch-row">
            <div className="node"><p className="node-title">Video</p><p className="node-sub">input recording</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0s' }} /></div>

          <div className="arch-row">
            <div className="node"><p className="node-title">Face landmarks</p><p className="node-sub">MediaPipe FaceMesh</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.3s' }} /></div>

          <div className="arch-row">
            <div className="node"><p className="node-title">Mouth alignment</p><p className="node-sub">mouth-centered · inner-mouth-width scale</p></div>
          </div>
          <div className="arch-wire arch-wire--fork" aria-hidden="true" />

          <div className="arch-branches">
            <div className="node is-accent">
              <p className="node-title">Appearance</p>
              <p className="node-sub">{APPEARANCE.outputDim}-D</p>
            </div>
            <div className="node is-accent">
              <p className="node-title">Geometry</p>
              <p className="node-sub">{GEOMETRY.outputDim}-D</p>
            </div>
            <div className="node is-accent">
              <p className="node-title">Motion</p>
              <p className="node-sub">{MOTION.outputDim}-D</p>
            </div>
          </div>
          <div className="arch-wire arch-wire--merge" aria-hidden="true" />

          <div className="arch-row">
            <div className="node"><p className="node-title">Gated fusion</p><p className="node-sub">softmax α per timestep</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.6s' }} /></div>

          <div className="arch-row">
            <div className="node"><p className="node-title">{FUSION.outputDim}-D temporal sequence</p><p className="node-sub">T × {FUSION.outputDim}</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.9s' }} /></div>

          <div className="arch-row">
            <div className="node is-accent">
              <p className="node-title">{CONFORMER.blocks}× Conformer</p>
              <p className="node-sub">
                d_model {CONFORMER.dModel} · heads {CONFORMER.heads} · FFN {CONFORMER.ffn} · kernel {CONFORMER.convKernel}
              </p>
            </div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '1.2s' }} /></div>

          <div className="arch-row">
            <div className="node"><p className="node-title">CTC</p><p className="node-sub">{DECODER.classes} classes · blank + letters</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '1.5s' }} /></div>

          <div className="arch-row">
            <div className="node arch-final">
              <p className="node-title arch-final-letters">A–Z</p>
              <p className="node-sub">spoken alphabet characters</p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
