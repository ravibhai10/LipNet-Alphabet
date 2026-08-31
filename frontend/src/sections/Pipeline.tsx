import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { APPEARANCE, GEOMETRY, MOTION, FUSION } from '../data/architecture'

/**
 * SECTION 02 — FROM VIDEO TO REPRESENTATION
 * Scroll-revealed pipeline. Real architecture dimensions only.
 */
export function Pipeline() {
  return (
    <section className="section pipeline" aria-labelledby="pipeline-title">
      <div className="container">
        <SectionHead
          id="pipeline-title"
          kicker="02 — From Video to Representation"
          title={
            <>
              One video.
              <br />
              <em>Three signals.</em>
            </>
          }
          lede="Every recording becomes a temporal sequence of aligned mouth data, decomposed into three parallel representations before fusion."
        />

        <div className="pipe" role="list">
          <Reveal as="div" role="listitem" className="pipe-row">
            <div className="node">
              <p className="node-title">Video</p>
              <p className="node-sub">30 fps · 15–76 frames</p>
            </div>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" />
            <span className="pipe-arrow">↓</span>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-row">
            <div className="node">
              <p className="node-title">Face landmarks</p>
              <p className="node-sub">MediaPipe FaceMesh · 20 canonical mouth points</p>
            </div>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" />
            <span className="pipe-arrow">↓</span>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-row">
            <div className="node">
              <p className="node-title">Mouth alignment</p>
              <p className="node-sub">mouth-centered frame · scale = inner mouth width</p>
            </div>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" />
            <span className="pipe-arrow">↓</span>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-branch">
            <div className="pipe-stream">
              <div className="node is-accent">
                <p className="node-title">Appearance</p>
                <p className="node-sub">{APPEARANCE.crop} → ResNet stem → {APPEARANCE.outputDim}-D</p>
              </div>
            </div>
            <div className="pipe-stream">
              <div className="node is-accent">
                <p className="node-title">Geometry</p>
                <p className="node-sub">{GEOMETRY.scalars} scalars → MLP → {GEOMETRY.outputDim}-D</p>
              </div>
            </div>
            <div className="pipe-stream">
              <div className="node is-accent">
                <p className="node-title">Motion</p>
                <p className="node-sub">Δ + flow → MLP → {MOTION.outputDim}-D</p>
              </div>
            </div>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" />
            <span className="pipe-arrow">↓</span>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-row">
            <div className="node">
              <p className="node-title">Gated fusion</p>
              <p className="node-sub">learned per-timestep α → {FUSION.outputDim}-D</p>
            </div>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" />
            <span className="pipe-arrow">↓</span>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-row">
            <div className="node">
              <p className="node-title">Temporal representation</p>
              <p className="node-sub">T × {FUSION.outputDim}</p>
            </div>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" />
            <span className="pipe-arrow">↓</span>
          </Reveal>

          <Reveal as="div" role="listitem" className="pipe-row">
            <div className="node is-accent">
              <p className="node-title">Letter decoding</p>
              <p className="node-sub">CTC · A–Z</p>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}
