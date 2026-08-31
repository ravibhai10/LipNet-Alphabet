import { useRef } from 'react'
import { HeroVisual } from '../visualizations/HeroVisual'
import { useGsapScene } from '../animations/useGsapScene'
import { gsap } from '../lib/gsap'
import { Arrow } from '../components/ui'

/**
 * SECTION — HERO
 * Full viewport, pinned for a cinematic scroll transition: typography drifts
 * upward while the face visualization closes in, landmarks ignite and the
 * mouth region becomes the emphasized zone.
 */
export function Hero() {
  const wrap = useRef<HTMLElement>(null)

  useGsapScene(wrap, () => {
    // .hero is 260vh with a position:sticky inner — CSS sticky does the pinning,
    // ScrollTrigger only scrubs the timeline (no pin → no double-pin).
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: wrap.current,
        start: 'top top',
        end: '+=160%',
        scrub: 0.5,
      },
    })

    tl.to('.hero-copy', { y: -110, autoAlpha: 0, ease: 'power1.in', duration: 0.55 }, 0)
      .to('.hero-visual', { scale: 1.28, yPercent: 4, ease: 'power1.inOut', duration: 1 }, 0)
      .to('.hv-dot', { opacity: 1, stagger: { each: 0.004, from: 'random' }, duration: 0.4 }, 0.1)
      .fromTo(
        '.hv-mouth-poly',
        { strokeDashoffset: 620 },
        { strokeDashoffset: 0, duration: 0.5, ease: 'power2.out' },
        0.45,
      )
      .to('.hv-mouth', { opacity: 1, duration: 0.3 }, 0.4)
      .fromTo('.hero-end', { autoAlpha: 0, y: 40 }, { autoAlpha: 1, y: 0, duration: 0.35 }, 0.72)
  })

  return (
    <section className="hero" id="top" ref={wrap} aria-label="Introduction">
      <div className="hero-sticky">
        <HeroVisual />
        <div className="hero-copy container">
          <p className="hero-kicker mono-note">Visual Speech AI — Research System</p>
          <h1 className="hero-title">
            <span>Speech,</span>
            <br />
            <span>without&nbsp;sound.</span>
          </h1>
          <p className="hero-sub">
            Visual Speech AI reconstructs spoken language from facial appearance, mouth
            geometry and temporal motion.
          </p>
          <div className="hero-ctas">
            <a className="btn btn-solid" href="#research">
              Explore the system <Arrow />
            </a>
            <a className="btn btn-ghost" href="#demo">
              Try live demo
            </a>
          </div>
        </div>
        <div className="hero-end" aria-hidden="true">
          <p className="hero-end-quote">Language leaves a trace on the face.</p>
          <p className="hero-end-label mono-note">The Phenomenon ↓</p>
        </div>
      </div>
    </section>
  )
}
