import { Reveal } from '../animations/Reveal'
import { SectionHead, SpecChip } from '../components/ui'

/**
 * SECTION 03 — THREE STREAMS
 * Three immersive panels, one per representation stream.
 * Mini-visualizations are illustrative diagrams of the real pipelines.
 */

function AppearanceViz() {
  // shaded 8×8 render of a 64×64 mouth crop grid
  const cells: number[] = []
  const shade = (r: number, c: number) => {
    const cx = (c - 3.5) / 4
    const cy = (r - 3.5) / 4
    const lip = Math.max(0, 1 - (cx * cx) / 0.55 - (cy * cy) / 0.22)
    return 0.06 + lip * 0.55
  }
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) cells.push(shade(r, c))
  return (
    <svg viewBox="0 0 200 200" className="sv" aria-hidden="true">
      {cells.map((a, i) => (
        <rect
          key={i}
          x={(i % 8) * 25}
          y={Math.floor(i / 8) * 25}
          width={25}
          height={25}
          fill={`rgba(237,235,228,${a.toFixed(3)})`}
        />
      ))}
      <rect x="1" y="1" width="198" height="198" className="sv-frame" />
    </svg>
  )
}

function GeometryViz() {
  const pts = '30,100 65,78 100,70 135,78 170,100 152,128 118,148 100,154 82,148 48,128'
  return (
    <svg viewBox="0 0 200 200" className="sv" aria-hidden="true">
      <polygon points={pts} className="sv-geom-poly" />
      {pts.split(' ').map((p, i) => {
        const [x, y] = p.split(',').map(Number)
        return <circle key={i} cx={x} cy={y} r={3.4} className="sv-geom-dot" />
      })}
      <line x1="30" y1="172" x2="170" y2="172" className="sv-measure" />
      <text x="100" y="192" textAnchor="middle" className="sv-note">scale = inner mouth width</text>
    </svg>
  )
}

function MotionViz() {
  return (
    <svg viewBox="0 0 200 200" className="sv" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <path
          key={i}
          className="sv-trail"
          style={{ animationDelay: `${i * 0.3}s` }}
          d={`M ${55 + i * 45} 40 q 14 42 -6 84 q -16 40 8 78`}
        />
      ))}
      <circle cx="100" cy="120" r="5" className="sv-geom-dot" />
      <path d="M 100 120 L 136 96" className="sv-vec" />
      <path d="M 128 94 L 136 96 L 132 104" className="sv-vec" />
      <text x="100" y="190" textAnchor="middle" className="sv-note">Δ landmarks · Δ scalars · 16×16 flow</text>
    </svg>
  )
}

const PANELS = [
  {
    num: 'Panel 01',
    name: 'Appearance',
    quote: 'What the mouth looks like',
    chain: '64 × 64 × 3 input → ResNet-style stem [32 · 64 · 128] → 96-D representation',
    viz: <AppearanceViz />,
    chips: [
      { k: 'Input', v: '64 × 64 × 3' },
      { k: 'Stem', v: '32 → 64 → 128' },
      { k: 'Output', v: '96-D / frame' },
    ],
  },
  {
    num: 'Panel 02',
    name: 'Geometry',
    quote: 'Where the mouth is',
    chain: '20 mouth landmarks → 20 canonical scalars → MLP(96) → 64-D representation',
    viz: <GeometryViz />,
    chips: [
      { k: 'Landmarks', v: '20 points' },
      { k: 'Scalars', v: '20 / frame' },
      { k: 'Output', v: '64-D / frame' },
    ],
  },
  {
    num: 'Panel 03',
    name: 'Motion',
    quote: 'How the mouth moves',
    chain: 'Savitzky–Golay derivatives (window 5) + 16×16 optical flow → MLP → 64-D representation',
    viz: <MotionViz />,
    chips: [
      { k: 'Derivatives', v: '1st + 2nd order' },
      { k: 'Flow', v: '16 × 16 patch' },
      { k: 'Output', v: '64-D / frame' },
    ],
  },
]

export function Streams() {
  return (
    <section className="section streams" aria-labelledby="streams-title">
      <div className="container">
        <SectionHead
          id="streams-title"
          kicker="03 — Three Streams"
          title={
            <>
              Three ways of seeing <em>one mouth.</em>
            </>
          }
          lede="Appearance, geometry and motion are extracted in parallel for every frame — each capturing a different facet of the same visual speech signal."
        />

        <div className="stream-panels">
          {PANELS.map((p) => (
            <Reveal key={p.num} as="article" className="stream-panel">
              <div className="stream-viz">{p.viz}</div>
              <div className="stream-body">
                <p className="mono-note">{p.num}</p>
                <h3 className="stream-name">{p.name}</h3>
                <p className="stream-quote">“{p.quote}”</p>
                <p className="stream-chain">{p.chain}</p>
                <div className="stream-chips">
                  {p.chips.map((c) => (
                    <SpecChip key={c.k} k={c.k} v={c.v} />
                  ))}
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}
