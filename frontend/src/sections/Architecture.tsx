import { useState } from 'react'
import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { APPEARANCE, GEOMETRY, MOTION, FUSION, CONFORMER, DECODER } from '../data/architecture'

/**
 * SECTION 13 — ARCHITECTURE OVERVIEW
 * Presentation-quality full-width interactive diagram with dual architecture modes and animated signal wires.
 */
export function Architecture() {
  const [activeArch, setActiveArch] = useState<'multimodal' | 'bilstm'>('multimodal')
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  const nodeDescriptions: Record<string, string> = {
    'Video': 'Input optical matrix: 640×480 RGB frames sampled at 30 FPS. Standardized duration window.',
    'Face landmarks': 'MediaPipe 468-point 3D mesh. Extracts 20 canonical lip contour vertices with temporal stability.',
    'Mouth alignment': 'Affine rotation normalization eliminating head roll angle θ, followed by 140×46 bounding box cropping.',
    'Appearance': `Pixel texture stream: 140×46 crop processed via 3D/2D CNN stem into a ${APPEARANCE.outputDim}-D feature descriptor.`,
    'Geometry': `Coordinate stream: Euclidean distances and spatial aspect ratios projected into a ${GEOMETRY.outputDim}-D embedding.`,
    'Motion': `Kinematic stream: Optical flow velocity vectors (Δt) projected into a ${MOTION.outputDim}-D dynamic trajectory.`,
    'Gated fusion': 'Softmax-weighted temporal gating assigning learned importance to Appearance, Geometry, and Motion.',
    'Conformer': `${CONFORMER.blocks} multi-head self-attention blocks with depthwise separable convolutions capturing local & global co-articulation.`,
    'BiLSTM': '2-layer Bidirectional LSTM (96 hidden units) capturing past and future phonetic context simultaneously.',
    'CTC': `Connectionist Temporal Classification projecting temporal states across ${DECODER.classes} classes (A–Z + Blank).`,
  }

  return (
    <section className="section architecture" id="architecture" aria-labelledby="arch-title">
      <div className="container">
        <SectionHead
          id="arch-title"
          kicker="11 — Architecture Overview"
          title={
            <>
              The whole system, <em>one interactive canvas.</em>
            </>
          }
          lede="Trace the end-to-end signal propagation from raw photons to reconstructed linguistic characters. Click architecture modes and hover nodes to reveal underlying tensor mechanics."
        />

        {/* Architecture Mode Selector */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '1rem',
          marginBottom: '2.5rem',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => setActiveArch('multimodal')}
            style={{
              background: activeArch === 'multimodal' ? 'rgba(143, 223, 203, 0.15)' : 'rgba(237, 235, 228, 0.03)',
              border: `1px solid ${activeArch === 'multimodal' ? 'var(--accent)' : 'var(--line)'}`,
              color: activeArch === 'multimodal' ? 'var(--accent)' : 'var(--ink-dim)',
              borderRadius: '999px',
              padding: '8px 20px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.82rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: activeArch === 'multimodal' ? '0 0 16px rgba(143, 223, 203, 0.2)' : 'none'
            }}
          >
            ● Tri-Stream Gated Conformer Architecture
          </button>

          <button
            onClick={() => setActiveArch('bilstm')}
            style={{
              background: activeArch === 'bilstm' ? 'rgba(125, 225, 249, 0.15)' : 'rgba(237, 235, 228, 0.03)',
              border: `1px solid ${activeArch === 'bilstm' ? 'var(--info)' : 'var(--line)'}`,
              color: activeArch === 'bilstm' ? 'var(--info)' : 'var(--ink-dim)',
              borderRadius: '999px',
              padding: '8px 20px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.82rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: activeArch === 'bilstm' ? '0 0 16px rgba(125, 225, 249, 0.2)' : 'none'
            }}
          >
            ● Verified Baseline: Alphabet CNN-BiLSTM (312k Params)
          </button>
        </div>

        {/* Hover Inspector Tooltip Card */}
        <div style={{
          maxWidth: '680px',
          margin: '0 auto 2rem',
          background: '#090a0d',
          border: '1px solid rgba(143, 223, 203, 0.3)',
          borderRadius: '10px',
          padding: '1rem 1.4rem',
          textAlign: 'center',
          minHeight: '68px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)'
        }}>
          {hoveredNode ? (
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: 'var(--accent)', textTransform: 'uppercase' }}>
                STAGE DETAIL: {hoveredNode}
              </span>
              <p style={{ margin: '0.3rem 0 0', fontSize: '0.86rem', color: 'var(--ink)' }}>
                {nodeDescriptions[hoveredNode] || 'Active deep neural processing component.'}
              </p>
            </div>
          ) : (
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--ink-dim)', fontFamily: 'var(--font-mono)' }}>
              Hover over any stage in the flow diagram below to inspect tensor specifications.
            </p>
          )}
        </div>

        <Reveal className="arch">
          {/* Stage 1: Video */}
          <div className="arch-row" onMouseEnter={() => setHoveredNode('Video')} onMouseLeave={() => setHoveredNode(null)}>
            <div className="node"><p className="node-title">Video Input</p><p className="node-sub">640×480 RGB · 30 FPS</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0s' }} /></div>

          {/* Stage 2: Landmarks */}
          <div className="arch-row" onMouseEnter={() => setHoveredNode('Face landmarks')} onMouseLeave={() => setHoveredNode(null)}>
            <div className="node"><p className="node-title">Face landmarks</p><p className="node-sub">MediaPipe 3D Mesh</p></div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.3s' }} /></div>

          {/* Stage 3: Mouth alignment */}
          <div className="arch-row" onMouseEnter={() => setHoveredNode('Mouth alignment')} onMouseLeave={() => setHoveredNode(null)}>
            <div className="node"><p className="node-title">Mouth Alignment &amp; Crop</p><p className="node-sub">140×46 px · Affine Tilt Canonicalization</p></div>
          </div>

          {activeArch === 'multimodal' ? (
            <>
              {/* Multimodal Forks */}
              <div className="arch-wire arch-wire--fork" aria-hidden="true" />

              <div className="arch-branches">
                <div className="node is-accent" onMouseEnter={() => setHoveredNode('Appearance')} onMouseLeave={() => setHoveredNode(null)}>
                  <p className="node-title">Appearance</p>
                  <p className="node-sub">{APPEARANCE.outputDim}-D Latent Vector</p>
                </div>
                <div className="node is-accent" onMouseEnter={() => setHoveredNode('Geometry')} onMouseLeave={() => setHoveredNode(null)}>
                  <p className="node-title">Geometry</p>
                  <p className="node-sub">{GEOMETRY.outputDim}-D Coordinate Ratio</p>
                </div>
                <div className="node is-accent" onMouseEnter={() => setHoveredNode('Motion')} onMouseLeave={() => setHoveredNode(null)}>
                  <p className="node-title">Motion</p>
                  <p className="node-sub">{MOTION.outputDim}-D Kinematic Δ</p>
                </div>
              </div>
              <div className="arch-wire arch-wire--merge" aria-hidden="true" />

              <div className="arch-row" onMouseEnter={() => setHoveredNode('Gated fusion')} onMouseLeave={() => setHoveredNode(null)}>
                <div className="node"><p className="node-title">Gated Fusion Layer</p><p className="node-sub">Softmax α per Timestep → {FUSION.outputDim}-D</p></div>
              </div>
              <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.6s' }} /></div>

              <div className="arch-row" onMouseEnter={() => setHoveredNode('Conformer')} onMouseLeave={() => setHoveredNode(null)}>
                <div className="node is-accent">
                  <p className="node-title">{CONFORMER.blocks}× Conformer Blocks</p>
                  <p className="node-sub">
                    d_model {CONFORMER.dModel} · {CONFORMER.heads} Heads · FFN {CONFORMER.ffn} · Kernel {CONFORMER.convKernel}
                  </p>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Direct CNN-BiLSTM Pipeline */}
              <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.5s' }} /></div>

              <div className="arch-row" onMouseEnter={() => setHoveredNode('Appearance')} onMouseLeave={() => setHoveredNode(null)}>
                <div className="node is-accent">
                  <p className="node-title">3-Stage Spatial Conv2D Extractor</p>
                  <p className="node-sub">Conv(1→16→32→64) + AdaptivePool(4,4) → 1,024-D Frame Embedding</p>
                </div>
              </div>
              <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '0.8s' }} /></div>

              <div className="arch-row" onMouseEnter={() => setHoveredNode('BiLSTM')} onMouseLeave={() => setHoveredNode(null)}>
                <div className="node is-accent">
                  <p className="node-title">Bidirectional LSTM (BiLSTM)</p>
                  <p className="node-sub">2 Layers · 96 Hidden Units · Forward + Backward Temporal States (192-D)</p>
                </div>
              </div>
            </>
          )}

          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '1.2s' }} /></div>

          {/* Decoding Stage */}
          <div className="arch-row" onMouseEnter={() => setHoveredNode('CTC')} onMouseLeave={() => setHoveredNode(null)}>
            <div className="node">
              <p className="node-title">CTC / Softmax Linear Projection</p>
              <p className="node-sub">{DECODER.classes} Classes · A–Z Alphabet + Blank Token</p>
            </div>
          </div>
          <div className="arch-wire" aria-hidden="true"><span className="flow-dot" style={{ animationDelay: '1.5s' }} /></div>

          {/* Final Output */}
          <div className="arch-row">
            <div className="node is-accent" style={{
              background: 'linear-gradient(180deg, rgba(143, 223, 203, 0.15) 0%, rgba(10, 10, 12, 0.95) 100%)',
              border: '1px solid var(--accent)'
            }}>
              <p className="node-title" style={{ color: 'var(--accent)' }}>Spoken Text / Character Stream</p>
              <p className="node-sub">Calibrated Probability Distribution &amp; Real-Time EMA Stabilization</p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}
