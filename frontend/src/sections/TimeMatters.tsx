import { useRef } from 'react'
import { useGsapScene } from '../animations/useGsapScene'
import { gsap } from '../lib/gsap'
import { CONFORMER, FUSION, SEQUENCE } from '../data/architecture'

/**
 * SECTION 05 — TIME MATTERS
 * Pin-driven horizontal glide through the frame sequence (01 … T), closing on
 * the T × 192 temporal representation and the Conformer reveal with the real
 * architecture values, revealed progressively.
 */

const FRAME_COUNT = 12

// deterministic pseudo-random mouth openness per frame — purely illustrative
function openness(i: number): number {
  return 0.35 + 0.3 * Math.sin(i * 0.9) + 0.12 * Math.sin(i * 2.3)
}

function MouthFrame({ i, total }: { i: number; total: number }) {
  const o = openness(i)
  const h = 10 + o * 26
  const w = 62 - o * 10
  return (
    <div className="tf-card" aria-hidden="true">
      <svg viewBox="-70 -60 140 120" className="tf-svg">
        <rect x={-w / 2 - 12} y={-h / 2 - 14} width={w + 24} height={h + 28} className="tf-crop" />
        <ellipse cx={0} cy={0} rx={w / 2} ry={h / 2} className="tf-lip" />
        <ellipse cx={0} cy={0} rx={w / 2.5} ry={h / 3.2} className="tf-lip-inner" />
        <circle cx={-w / 2} cy={0} r={1.6} className="tf-dot" />
        <circle cx={w / 2} cy={0} r={1.6} className="tf-dot" />
        <circle cx={0} cy={-h / 2.2} r={1.6} className="tf-dot" />
        <circle cx={0} cy={h / 2.2} r={1.6} className="tf-dot" />
      </svg>
      <p className="tf-idx mono-note">{String(i + 1).padStart(2, '0')} / {total}</p>
    </div>
  )
}

export function TimeMatters() {
  const wrap = useRef<HTMLElement>(null)
  const track = useRef<HTMLDivElement>(null)

  useGsapScene(wrap, () => {
    const trackEl = track.current
    if (!trackEl) return

    // How far the track needs to slide so the last card reaches the viewport
    const slideX = Math.max(0, trackEl.scrollWidth - window.innerWidth)

    // Horizontal glide — GSAP handles the pin internally
    gsap.to(trackEl, {
      x: -slideX,
      ease: 'none',
      scrollTrigger: {
        trigger: wrap.current,
        start: 'top top',
        end: () => `+=${slideX}`,
        scrub: 0.6,
        pin: true,
        invalidateOnRefresh: true,
        anticipatePin: 1,
      },
    })

    // Cards fade in progressively as they scroll into view
    gsap.set('.tf-card', { opacity: 0.3, visibility: 'visible' })
    gsap.to('.tf-card', {
      opacity: 1,
      stagger: { each: 0.15, from: 'start' },
      ease: 'none',
      scrollTrigger: {
        trigger: wrap.current,
        start: 'top top+=8%',
        end: () => `+=${slideX * 0.7}`,
        scrub: true,
      },
    })
  })

  return (
    <section className="timematters" ref={wrap} aria-labelledby="time-title">
      <div className="tf-viewport">
        <div className="container tf-head">
          <p className="kicker">05 — Time Matters</p>
          <h2 className="headline" id="time-title">
            A mouth is not an image.
            <br />
            <em>It is a sequence.</em>
          </h2>
        </div>

        <div className="tf-track no-scrollbar" ref={track}>
          {Array.from({ length: FRAME_COUNT }).map((_, i) => (
            <MouthFrame key={i} i={i} total={FRAME_COUNT} />
          ))}

          <div className="tf-close">
            <div className="node">
              <p className="node-title">Temporal representation</p>
              <p className="node-sub">
                T × {FUSION.outputDim} · up to {SEQUENCE.maxFrames} frames @ {SEQUENCE.fps} fps
              </p>
            </div>
            <span className="tf-arrow" aria-hidden="true">→</span>
            <div className="tf-conformer">
              <p className="tf-conformer-title">4 × Conformer Blocks</p>
              <dl className="tf-specs">
                <div><dt className="mono-note">d_model</dt><dd>{CONFORMER.dModel}</dd></div>
                <div><dt className="mono-note">heads</dt><dd>{CONFORMER.heads}</dd></div>
                <div><dt className="mono-note">FFN</dt><dd>{CONFORMER.ffn}</dd></div>
                <div><dt className="mono-note">kernel</dt><dd>{CONFORMER.convKernel}</dd></div>
              </dl>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
