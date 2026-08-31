import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { CONFORMER } from '../data/architecture'

/**
 * SECTION 06 — THE TEMPORAL ENGINE
 * One Conformer block, drawn as an elegant flow diagram with animated data
 * dots, then repeated as blocks 01–04.
 */

const STAGES = [
  { name: 'Input', sub: 'T × 192' },
  { name: 'Positional encoding', sub: 'sinusoidal + learned' },
  { name: 'FFN', sub: '192 → 768 → 192' },
  { name: 'Multi-head attention', sub: '4 heads' },
  { name: 'Convolution', sub: 'depthwise · kernel 7' },
  { name: 'FFN', sub: '192 → 768 → 192' },
  { name: 'Output', sub: 'T × 192' },
]

export function TemporalEngine() {
  return (
    <section className="section engine" aria-labelledby="engine-title">
      <div className="container">
        <SectionHead
          id="engine-title"
          kicker="06 — The Temporal Engine"
          title={
            <>
              Context, computed <em>in time.</em>
            </>
          }
          lede="Each Conformer block combines macaron feed-forward networks, multi-head self-attention and a depthwise convolution module — every stage normalized with pre-norm residuals."
        />

        <div className="engine-grid">
          <Reveal as="div" className="engine-flow" aria-label="Conformer block data flow">
            <ol className="engine-list">
              {STAGES.map((s, i) => (
                <li key={i} className={`engine-step ${i === 0 || i === STAGES.length - 1 ? 'engine-step--term' : ''}`}>
                  <div className="engine-node">
                    <p className="node-title">{s.name}</p>
                    <p className="node-sub">{s.sub}</p>
                  </div>
                  {i < STAGES.length - 1 && (
                    <span className="engine-wire" aria-hidden="true">
                      <span className="flow-dot" style={{ animationDelay: `${i * 0.45}s` }} />
                    </span>
                  )}
                </li>
              ))}
            </ol>
          </Reveal>

          <Reveal as="div" className="engine-blocks">
            <p className="mono-note">The block, repeated</p>
            <div className="engine-chips">
              {['01', '02', '03', '04'].map((n, i) => (
                <div className="engine-chip" key={n} style={{ transitionDelay: `${i * 60}ms` }}>
                  <span className="engine-chip-num">{n}</span>
                  <span className="engine-chip-label">Conformer</span>
                </div>
              ))}
            </div>
            <dl className="engine-specs">
              <div><dt className="mono-note">Blocks</dt><dd>{CONFORMER.blocks}</dd></div>
              <div><dt className="mono-note">d_model</dt><dd>{CONFORMER.dModel}</dd></div>
              <div><dt className="mono-note">Heads</dt><dd>{CONFORMER.heads}</dd></div>
              <div><dt className="mono-note">FFN dim</dt><dd>{CONFORMER.ffn}</dd></div>
              <div><dt className="mono-note">Conv kernel</dt><dd>{CONFORMER.convKernel}</dd></div>
              <div><dt className="mono-note">Pos. encoding</dt><dd>{CONFORMER.positionalEncoding}</dd></div>
            </dl>
            <p className="mono-note">Masked padding — temporal dimension fully preserved.</p>
          </Reveal>
        </div>
      </div>
    </section>
  )
}
