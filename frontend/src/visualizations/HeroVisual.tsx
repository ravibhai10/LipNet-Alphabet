import { useEffect, useRef } from 'react'
import { useReducedMotion } from '../hooks/useReducedMotion'

/**
 * Cinematic hero visualization: a face suggested by landmark dots, a canonical
 * mouth region, temporal trajectory curves and latent feature particles.
 * Pure SVG + one small canvas (<= 72 particles, DPR capped, paused offscreen).
 */

const CX = 600
const CY = 420

function faceContourPoints(): [number, number][] {
  const pts: [number, number][] = []
  const n = 72
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2
    const x = CX + 250 * Math.sin(a) * (a > 0 ? 0.94 : 1)
    const y = CY + 330 * Math.cos(a) * (Math.sin(a) < 0 ? 0.96 : 1)
    pts.push([x, y])
  }
  return pts
}

function sparseFeaturePoints(): [number, number][] {
  return [
    [500, 330], [545, 318], [655, 318], [700, 330],
    [600, 400], [585, 470], [615, 470],
    [505, 470], [695, 470],
    [540, 585], [660, 585],
  ]
}

function mouthPoints(): [number, number][] {
  const mx = CX
  const my = 560
  const w = 78
  const inner: [number, number][] = [
    [mx - 62, my - 4], [mx - 34, my - 10], [mx, my - 13], [mx + 34, my - 10], [mx + 62, my - 4],
    [mx + 52, my + 10], [mx + 20, my + 20], [mx, my + 23], [mx - 20, my + 20], [mx - 52, my + 10],
  ]
  const outer: [number, number][] = [
    [mx - w - 14, my - 6], [mx - w + 4, my - 20], [mx - w / 2, my - 30], [mx, my - 33],
    [mx + w / 2, my - 30], [mx + w - 4, my - 20], [mx + w + 14, my - 6],
    [mx + w, my + 16], [mx + w / 2, my + 32], [mx, my + 37], [mx - w / 2, my + 32], [mx - w, my + 16],
  ]
  return [...inner, ...outer]
}

const FLOWS = [
  'M 538 560 C 430 545, 350 470, 300 360 S 250 190, 300 120',
  'M 662 560 C 770 545, 850 470, 900 360 S 950 190, 900 120',
  'M 600 597 C 600 660, 560 720, 470 760 S 330 800, 280 780',
  'M 600 597 C 600 660, 640 720, 730 760 S 870 800, 920 780',
]

export function HeroVisual() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || reduced) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf = 0
    let w = 0
    let h = 0
    let ax = 0
    let ay = 0
    const DPR = Math.min(window.devicePixelRatio || 1, 1.5)

    type P = { x: number; y: number; vx: number; vy: number; r: number }
    const parts: P[] = []
    const N = 72

    const resize = () => {
      const rect = wrap.getBoundingClientRect()
      w = rect.width
      h = rect.height
      canvas.width = Math.round(w * DPR)
      canvas.height = Math.round(h * DPR)
      canvas.style.width = w + 'px'
      canvas.style.height = h + 'px'
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0)
      ax = w * 0.5
      ay = h * 0.6
      if (parts.length === 0) {
        for (let i = 0; i < N; i++) {
          parts.push({
            x: Math.random() * w,
            y: Math.random() * h,
            vx: (Math.random() - 0.5) * 0.12,
            vy: (Math.random() - 0.5) * 0.12,
            r: 0.6 + Math.random() * 1.4,
          })
        }
      }
    }
    resize()

    const step = () => {
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = 'rgba(143, 223, 203, 0.5)'
      for (const p of parts) {
        const dx = ax - p.x
        const dy = ay - p.y
        const f = Math.min(0.5, 900 / (dx * dx + dy * dy + 4000))
        p.vx += dx * f * 0.00035
        p.vy += dy * f * 0.00035
        p.vx *= 0.995
        p.vy *= 0.995
        p.x += p.vx
        p.y += p.vy
        if (p.x < -10) p.x = w + 10
        if (p.x > w + 10) p.x = -10
        if (p.y < -10) p.y = h + 10
        if (p.y > h + 10) p.y = -10
        ctx.globalAlpha = 0.25 + (p.r / 2) * 0.3
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1
      raf = requestAnimationFrame(step)
    }

    const ro = new ResizeObserver(resize)
    ro.observe(wrap)
    raf = requestAnimationFrame(step)

    // Pause when offscreen using IntersectionObserver
    let io: IntersectionObserver | null = null
    if (typeof IntersectionObserver !== 'undefined') {
      io = new IntersectionObserver((entries) => {
        if (entries[0]?.isIntersecting) {
          raf = requestAnimationFrame(step)
        } else {
          cancelAnimationFrame(raf)
        }
      }, { threshold: 0.02 })
      io.observe(wrap)
    }

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      io?.disconnect()
    }
  }, [reduced])

  const contour = faceContourPoints()
  const features = sparseFeaturePoints()
  const mouth = mouthPoints()

  return (
    <div
      className="hero-visual"
      ref={wrapRef}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} />
      <svg viewBox="0 0 1200 900" preserveAspectRatio="xMidYMid slice" className="hero-svg">
        <g className="hv-contour">
          {contour.map(([x, y], i) => (
            <circle key={`c${i}`} className="hv-dot" cx={x} cy={y} r={2.1}
              style={{ animationDelay: `${(i % 12) * 0.35}s` }} />
          ))}
        </g>
        <g className="hv-features">
          {features.map(([x, y], i) => (
            <circle key={`f${i}`} className="hv-dot hv-dot--faint" cx={x} cy={y} r={1.7} />
          ))}
        </g>
        <g className="hv-flows">
          {FLOWS.map((d, i) => (
            <path key={`fl${i}`} className="hv-flow" d={d} style={{ animationDelay: `${i * 0.9}s` }} />
          ))}
        </g>
        <g className="hv-mouth">
          <ellipse className="hv-mouth-halo" cx={CX} cy={560} rx={150} ry={95} />
          <line className="hv-measure" x1={CX - 78} y1={618} x2={CX + 78} y2={618} />
          <line className="hv-measure" x1={CX - 78} y1={612} x2={CX - 78} y2={624} />
          <line className="hv-measure" x1={CX + 78} y1={612} x2={CX + 78} y2={624} />
          <polygon className="hv-mouth-poly" points={mouth.map(([x, y]) => `${x},${y}`).join(' ')} />
          {mouth.map(([x, y], i) => (
            <circle key={`m${i}`} className="hv-mouth-dot" cx={x} cy={y} r={2.4} />
          ))}
          <text className="hv-annot" x={CX} y={646} textAnchor="middle">
            inner mouth width · scale reference
          </text>
        </g>
      </svg>
    </div>
  )
}
