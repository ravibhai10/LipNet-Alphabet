import { useRef } from 'react'
import { SectionHead } from '../components/ui'
import { useGsapScene } from '../animations/useGsapScene'
import { gsap } from '../lib/gsap'

/**
 * SECTION 01 — THE PHENOMENON
 * Pinned stage: appearance → geometry → motion, then all three synchronize.
 */

const STAGES = [
  { id: 'appearance', label: 'Appearance', hint: 'what the mouth looks like' },
  { id: 'geometry', label: 'Geometry', hint: 'where the mouth is' },
  { id: 'motion', label: 'Motion', hint: 'how the mouth moves' },
]

export function Phenomenon() {
  const wrap = useRef<HTMLElement>(null)

  useGsapScene(wrap, () => {
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: wrap.current,
        start: 'top top',
        end: '+=260%',
        scrub: 0.6,
      },
      defaults: { ease: 'power2.inOut' },
    })

    // 1 — appearance
    tl.fromTo('.ph-stage-0', { autoAlpha: 0.12, scale: 0.97 }, { autoAlpha: 1, scale: 1, duration: 1 })
      .to('.ph-label-0', { color: '#8fdfcb', duration: 0.4 }, '<')
    // 2 — geometry over it
      .fromTo('.ph-stage-1', { autoAlpha: 0 }, { autoAlpha: 1, duration: 1 })
      .to('.ph-label-1', { color: '#8fdfcb', duration: 0.4 }, '<')
    // 3 — motion trails
      .fromTo('.ph-stage-2', { autoAlpha: 0 }, { autoAlpha: 1, duration: 1 })
      .to('.ph-label-2', { color: '#8fdfcb', duration: 0.4 }, '<')
    // 4 — synchronize: all pulse together
      .to('.ph-viz', { scale: 1.04, duration: 0.6 })
      .to('.ph-sync', { autoAlpha: 1, duration: 0.4 }, '<0.3')
  })

  return (
    <section className="section phenomenon" id="research" ref={wrap} aria-labelledby="phenomenon-title">
      <div className="container">
        <SectionHead
          id="phenomenon-title"
          kicker="01 — The Phenomenon"
          title={
            <>
              Speech changes <em>the face.</em>
            </>
          }
          lede="Spoken language leaves observable signals in three places: how the mouth looks, where its points sit, and how they move through time."
        />
      </div>

      <div className="ph-pin">
        <div className="container ph-grid">
          <div className="ph-viz" aria-hidden="true">
            <svg viewBox="0 0 800 600" className="ph-svg">
              {/* stage 0 — appearance: shaded mouth crop with pixel grid */}
              <g className="ph-stage ph-stage-0">
                <rect x="220" y="150" width="360" height="300" rx="10" className="ph-crop" />
                {Array.from({ length: 13 }).map((_, i) => (
                  <line key={`gx${i}`} x1={220 + i * 30} y1={150} x2={220 + i * 30} y2={450} className="ph-gridline" />
                ))}
                {Array.from({ length: 11 }).map((_, i) => (
                  <line key={`gy${i}`} x1={220} y1={150 + i * 30} x2={580} y2={150 + i * 30} className="ph-gridline" />
                ))}
                <path
                  d="M 320 290 Q 400 250 480 290 Q 470 330 400 344 Q 330 330 320 290 Z"
                  className="ph-lip"
                />
                <path d="M 336 296 Q 400 268 464 296 Q 452 318 400 326 Q 348 318 336 296 Z" className="ph-lip-inner" />
                <text x="400" y="500" textAnchor="middle" className="ph-svg-label">64 × 64 × 3 crop</text>
              </g>

              {/* stage 1 — geometry: landmark polygon */}
              <g className="ph-stage ph-stage-1">
                <polygon points="320,290 360,278 400,272 440,278 480,290 462,316 420,334 400,338 380,334 338,316" className="ph-geom-poly" />
                {[
                  [320, 290], [360, 278], [400, 272], [440, 278], [480, 290],
                  [462, 316], [420, 334], [400, 338], [380, 334], [338, 316],
                ].map(([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r={4} className="ph-geom-dot" />
                ))}
                <line x1="320" y1="360" x2="480" y2="360" className="ph-measure" />
                <text x="400" y="384" textAnchor="middle" className="ph-svg-label">20 landmarks · inner mouth width scale</text>
              </g>

              {/* stage 2 — motion: velocity trails */}
              <g className="ph-stage ph-stage-2">
                {[0, 1, 2, 3, 4].map((i) => (
                  <path
                    key={i}
                    className="ph-trail"
                    style={{ animationDelay: `${i * 0.22}s` }}
                    d={`M ${300 + i * 50} 250 q 18 40 -6 84 q -20 44 10 90`}
                  />
                ))}
                <text x="400" y="500" textAnchor="middle" className="ph-svg-label">Δ landmarks · Δ scalars · optical flow</text>
              </g>
            </svg>
          </div>

          <div className="ph-labels" role="list">
            {STAGES.map((s, i) => (
              <div className={`ph-label ph-label-${i}`} role="listitem" key={s.id}>
                <span className="ph-label-index mono-note">0{i + 1}</span>
                <div>
                  <p className="ph-label-name">{s.label}</p>
                  <p className="ph-label-hint mono-note">{s.hint}</p>
                </div>
              </div>
            ))}
            <p className="ph-sync mono-note">Three signals — one temporal sample.</p>
          </div>
        </div>
      </div>
    </section>
  )
}
