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
          {/* High-Tech System Status Badge */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.6rem',
            background: 'rgba(143, 223, 203, 0.08)',
            border: '1px solid rgba(143, 223, 203, 0.3)',
            borderRadius: '999px',
            padding: '4px 14px',
            marginBottom: '1rem',
            backdropFilter: 'blur(8px)',
            boxShadow: '0 0 20px rgba(143, 223, 203, 0.15)'
          }}>
            <span style={{
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              background: 'var(--accent, #8fdfcb)',
              boxShadow: '0 0 8px var(--accent)'
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.74rem',
              color: 'var(--accent)',
              letterSpacing: '0.06em',
              fontWeight: 600,
              textTransform: 'uppercase'
            }}>
              Visual Speech AI · Research System v0.2.0
            </span>
          </div>

          <h1 className="hero-title">
            <span>Speech,</span>
            <br />
            <span>without&nbsp;sound.</span>
          </h1>

          <p className="hero-sub">
            Reconstructing spoken language directly from facial kinematics, mouth aperture geometry, 
            and deep temporal motion using PyTorch and MediaPipe computer vision.
          </p>

          <div className="hero-ctas" style={{ marginTop: '1.8rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <a className="btn btn-solid" href="#research">
              Explore the system <Arrow />
            </a>
            <a className="btn btn-ghost" href="/demo/live">
              Try Live Demo
            </a>
            <a
              href="/science"
              className="btn btn-ghost"
              style={{
                borderColor: 'rgba(143, 223, 203, 0.4)',
                color: 'var(--accent)',
                background: 'rgba(143, 223, 203, 0.05)'
              }}
            >
              Explore Science Lab ◈
            </a>
          </div>

          {/* Real-Time Research Specifications HUD Bar */}
          <div style={{
            marginTop: '2.5rem',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
            gap: '1rem',
            maxWidth: '680px',
            borderTop: '1px solid rgba(237, 235, 228, 0.1)',
            paddingTop: '1.5rem',
            fontFamily: 'var(--font-mono)'
          }}>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>AUDIO CHANNEL</div>
              <div style={{ fontSize: '0.88rem', color: 'var(--accent)', fontWeight: 600 }}>0 dB (Silent)</div>
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>LANDMARK MESH</div>
              <div style={{ fontSize: '0.88rem', color: 'var(--ink)', fontWeight: 600 }}>468 3D Nodes</div>
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>MOUTH TENSOR</div>
              <div style={{ fontSize: '0.88rem', color: 'var(--info)', fontWeight: 600 }}>140×46 px</div>
            </div>
            <div>
              <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>TEMPORAL SAMPLING</div>
              <div style={{ fontSize: '0.88rem', color: 'var(--ink)', fontWeight: 600 }}>30 FPS</div>
            </div>
          </div>
        </div>

        <div className="hero-end" aria-hidden="true">
          <p className="hero-end-quote">Language leaves an indelible trace on the face.</p>
          <p className="hero-end-label mono-note">The Phenomenon ↓</p>
        </div>
      </div>
    </section>
  )
}
