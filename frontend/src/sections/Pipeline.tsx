import { useState } from 'react'
import { SectionHead } from '../components/ui'
import { APPEARANCE, GEOMETRY, MOTION, FUSION } from '../data/architecture'

/**
 * SECTION 02 — FROM VIDEO TO REPRESENTATION
 * Interactive animated pipeline with simulated packet tracer and tri-stream inspector.
 */
export function Pipeline() {
  const [activeStep, setActiveStep] = useState<number | null>(null)
  const [selectedStream, setSelectedStream] = useState<'appearance' | 'geometry' | 'motion'>('appearance')
  const [isSimulating, setIsSimulating] = useState(false)

  // Trigger sequential data packet animation
  const runSimulation = () => {
    if (isSimulating) return
    setIsSimulating(true)
    setActiveStep(0)

    const steps = [0, 1, 2, 3, 4, 5, 6]
    steps.forEach((step, idx) => {
      setTimeout(() => {
        setActiveStep(step)
        if (idx === steps.length - 1) {
          setTimeout(() => {
            setIsSimulating(false)
            setActiveStep(null)
          }, 1200)
        }
      }, idx * 600)
    })
  }

  const streamDetails = {
    appearance: {
      title: 'Appearance Stream (Pixel Space)',
      input: '140 × 46 px Grayscale Crop',
      features: 'ResNet-18 Stem & 3D Spatial Convolutions',
      output: `${APPEARANCE.outputDim}-D Embedding Vector`,
      role: 'Captures subtle texture variations: teeth visibility, tongue tip occlusion, labial shading, and mucous membrane reflectance.'
    },
    geometry: {
      title: 'Geometry Stream (Coordinate Space)',
      input: '20 Outer & Inner Landmark Vertices',
      features: 'Relative Distances & Spatial Ratios',
      output: `${GEOMETRY.outputDim}-D Structural Representation`,
      role: 'Measures invariant spatial coordinates: vertical aperture, horizontal stretch, vermilion thickness, and jaw depression angle.'
    },
    motion: {
      title: 'Motion Stream (Kinematic Space)',
      input: 'Frame-to-Frame Temporal Differences (Δt)',
      features: 'Optical Flow Derivatives & Velocity Vectors',
      output: `${MOTION.outputDim}-D Dynamic Momentum`,
      role: 'Tracks velocity and acceleration vectors, critical for distinguishing opening vowels from closing bilabial plosives.'
    }
  }

  return (
    <section className="section pipeline" id="pipeline" aria-labelledby="pipeline-title">
      <div className="container">
        <SectionHead
          id="pipeline-title"
          kicker="02 — Dynamic Signal Decomposition"
          title={
            <>
              One video.
              <br />
              <em>Three parallel signals.</em>
            </>
          }
          lede="Every recording becomes a calibrated temporal sequence of aligned mouth frames, decomposed into three parallel streams before gated fusion."
        />

        {/* Pipeline Control Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          maxWidth: '820px',
          margin: '0 auto 2rem',
          flexWrap: 'wrap',
          gap: '1rem',
          background: 'rgba(237, 235, 228, 0.02)',
          border: '1px solid var(--line)',
          borderRadius: '12px',
          padding: '0.85rem 1.4rem'
        }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--ink-dim)' }}>
            STATUS: <strong style={{ color: isSimulating ? 'var(--accent)' : 'var(--ink)' }}>
              {isSimulating ? `PROPAGATING STAGE ${activeStep! + 1} OF 7...` : 'IDLE — READY FOR TRACE'}
            </strong>
          </div>

          <button
            onClick={runSimulation}
            disabled={isSimulating}
            style={{
              background: isSimulating ? 'rgba(143, 223, 203, 0.2)' : 'var(--accent, #8fdfcb)',
              color: 'var(--bg, #0a0a0c)',
              border: 'none',
              borderRadius: '999px',
              padding: '6px 18px',
              fontSize: '0.8rem',
              fontWeight: 600,
              fontFamily: 'var(--font-mono)',
              cursor: isSimulating ? 'wait' : 'pointer',
              boxShadow: isSimulating ? 'none' : '0 0 16px rgba(143, 223, 203, 0.3)',
              transition: 'all 0.2s ease'
            }}
          >
            {isSimulating ? '⚡ Tracing Optical Packet...' : '▶ Simulate Optical Packet'}
          </button>
        </div>

        {/* Visual Pipeline Flow Nodes */}
        <div className="pipe" role="list">
          {/* Node 1: Video */}
          <div className="pipe-row">
            <div
              className={`node ${activeStep === 0 ? 'is-accent' : ''}`}
              style={{
                transition: 'all 0.3s ease',
                transform: activeStep === 0 ? 'scale(1.05)' : 'none',
                boxShadow: activeStep === 0 ? '0 0 25px rgba(143, 223, 203, 0.4)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="node-title">01 · Video Ingestion</p>
                {activeStep === 0 && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent)' }}>● ACTIVE</span>}
              </div>
              <p className="node-sub">Raw Camera Matrix · 30 FPS · 15–76 frames</p>
            </div>
          </div>

          {/* Connector 1 */}
          <div className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" style={{ height: '1.8rem', position: 'relative', overflow: 'hidden' }}>
              {activeStep === 0 && (
                <span style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'linear-gradient(180deg, var(--accent), transparent)',
                  animation: 'pulseDown 0.6s infinite'
                }} />
              )}
            </span>
            <span className="pipe-arrow" style={{ color: activeStep === 1 ? 'var(--accent)' : 'inherit' }}>↓</span>
          </div>

          {/* Node 2: Landmarks */}
          <div className="pipe-row">
            <div
              className={`node ${activeStep === 1 ? 'is-accent' : ''}`}
              style={{
                transition: 'all 0.3s ease',
                transform: activeStep === 1 ? 'scale(1.05)' : 'none',
                boxShadow: activeStep === 1 ? '0 0 25px rgba(143, 223, 203, 0.4)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="node-title">02 · Facial Landmark Tracking</p>
                {activeStep === 1 && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent)' }}>● ACTIVE</span>}
              </div>
              <p className="node-sub">MediaPipe 3D Mesh · 20 Canonical Lip Contour Vertices</p>
            </div>
          </div>

          {/* Connector 2 */}
          <div className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" style={{ height: '1.8rem' }} />
            <span className="pipe-arrow" style={{ color: activeStep === 2 ? 'var(--accent)' : 'inherit' }}>↓</span>
          </div>

          {/* Node 3: Mouth Alignment */}
          <div className="pipe-row">
            <div
              className={`node ${activeStep === 2 ? 'is-accent' : ''}`}
              style={{
                transition: 'all 0.3s ease',
                transform: activeStep === 2 ? 'scale(1.05)' : 'none',
                boxShadow: activeStep === 2 ? '0 0 25px rgba(143, 223, 203, 0.4)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="node-title">03 · Mouth Alignment &amp; Crop</p>
                {activeStep === 2 && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent)' }}>● ACTIVE</span>}
              </div>
              <p className="node-sub">Affine Canonicalization · Scale = Inner Inter-Commissure Width</p>
            </div>
          </div>

          {/* Connector 3 */}
          <div className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" style={{ height: '1.8rem' }} />
            <span className="pipe-arrow" style={{ color: activeStep === 3 ? 'var(--accent)' : 'inherit' }}>↓</span>
          </div>

          {/* Node 4: Tri-Stream Decomposition */}
          <div className="pipe-branch" style={{ position: 'relative' }}>
            <div
              className="pipe-stream"
              onClick={() => setSelectedStream('appearance')}
              style={{ cursor: 'pointer' }}
            >
              <div className={`node is-accent ${selectedStream === 'appearance' ? 'is-selected' : ''}`} style={{
                border: selectedStream === 'appearance' ? '1px solid var(--accent)' : '1px solid var(--line)',
                background: selectedStream === 'appearance' ? 'rgba(143, 223, 203, 0.12)' : 'var(--bg-panel)',
                transform: activeStep === 3 ? 'scale(1.03)' : 'none',
                transition: 'all 0.2s ease'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <p className="node-title">Appearance Stream</p>
                  <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                    {selectedStream === 'appearance' ? '● VIEWING' : 'CLICK'}
                  </span>
                </div>
                <p className="node-sub">{APPEARANCE.crop} → ResNet Stem → {APPEARANCE.outputDim}-D</p>
              </div>
            </div>

            <div
              className="pipe-stream"
              onClick={() => setSelectedStream('geometry')}
              style={{ cursor: 'pointer' }}
            >
              <div className={`node is-accent ${selectedStream === 'geometry' ? 'is-selected' : ''}`} style={{
                border: selectedStream === 'geometry' ? '1px solid var(--info)' : '1px solid var(--line)',
                background: selectedStream === 'geometry' ? 'rgba(125, 225, 249, 0.12)' : 'var(--bg-panel)',
                transform: activeStep === 3 ? 'scale(1.03)' : 'none',
                transition: 'all 0.2s ease'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <p className="node-title">Geometry Stream</p>
                  <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--info)' }}>
                    {selectedStream === 'geometry' ? '● VIEWING' : 'CLICK'}
                  </span>
                </div>
                <p className="node-sub">{GEOMETRY.scalars} Scalars → MLP → {GEOMETRY.outputDim}-D</p>
              </div>
            </div>

            <div
              className="pipe-stream"
              onClick={() => setSelectedStream('motion')}
              style={{ cursor: 'pointer' }}
            >
              <div className={`node is-accent ${selectedStream === 'motion' ? 'is-selected' : ''}`} style={{
                border: selectedStream === 'motion' ? '1px solid var(--warning)' : '1px solid var(--line)',
                background: selectedStream === 'motion' ? 'rgba(252, 211, 77, 0.12)' : 'var(--bg-panel)',
                transform: activeStep === 3 ? 'scale(1.03)' : 'none',
                transition: 'all 0.2s ease'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <p className="node-title">Motion Stream</p>
                  <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: 'var(--warning)' }}>
                    {selectedStream === 'motion' ? '● VIEWING' : 'CLICK'}
                  </span>
                </div>
                <p className="node-sub">Δ + Flow Field → MLP → {MOTION.outputDim}-D</p>
              </div>
            </div>
          </div>

          {/* Interactive Stream Details Sub-Panel */}
          <div style={{
            width: 'min(100%, 60rem)',
            background: '#07080b',
            border: '1px solid rgba(143, 223, 203, 0.25)',
            borderRadius: '12px',
            padding: '1.25rem 1.6rem',
            margin: '1rem 0',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.8rem',
            lineHeight: 1.6
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
              <span style={{ color: 'var(--accent)', fontWeight: 600 }}>
                {streamDetails[selectedStream].title}
              </span>
              <span style={{ color: 'var(--ink-faint)', fontSize: '0.72rem' }}>
                OUTPUT: {streamDetails[selectedStream].output}
              </span>
            </div>
            <p style={{ margin: '0 0 0.5rem', color: 'var(--ink-dim)' }}>
              {streamDetails[selectedStream].role}
            </p>
            <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.74rem', color: 'var(--ink-faint)', flexWrap: 'wrap' }}>
              <div>INPUT: <span style={{ color: 'var(--ink)' }}>{streamDetails[selectedStream].input}</span></div>
              <div>TRANSFORM: <span style={{ color: 'var(--info)' }}>{streamDetails[selectedStream].features}</span></div>
            </div>
          </div>

          {/* Connector 4 */}
          <div className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" style={{ height: '1.8rem' }} />
            <span className="pipe-arrow" style={{ color: activeStep === 4 ? 'var(--accent)' : 'inherit' }}>↓</span>
          </div>

          {/* Node 5: Gated Fusion */}
          <div className="pipe-row">
            <div
              className={`node ${activeStep === 4 ? 'is-accent' : ''}`}
              style={{
                transition: 'all 0.3s ease',
                transform: activeStep === 4 ? 'scale(1.05)' : 'none',
                boxShadow: activeStep === 4 ? '0 0 25px rgba(143, 223, 203, 0.4)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="node-title">04 · Learned Gated Fusion</p>
                {activeStep === 4 && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent)' }}>● ACTIVE</span>}
              </div>
              <p className="node-sub">Dynamic Softmax α per Timestep → Combined {FUSION.outputDim}-D Manifold</p>
            </div>
          </div>

          {/* Connector 5 */}
          <div className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" style={{ height: '1.8rem' }} />
            <span className="pipe-arrow" style={{ color: activeStep === 5 ? 'var(--accent)' : 'inherit' }}>↓</span>
          </div>

          {/* Node 6: Temporal Context */}
          <div className="pipe-row">
            <div
              className={`node ${activeStep === 5 ? 'is-accent' : ''}`}
              style={{
                transition: 'all 0.3s ease',
                transform: activeStep === 5 ? 'scale(1.05)' : 'none',
                boxShadow: activeStep === 5 ? '0 0 25px rgba(143, 223, 203, 0.4)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="node-title">05 · Bidirectional Recurrent Context</p>
                {activeStep === 5 && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--accent)' }}>● ACTIVE</span>}
              </div>
              <p className="node-sub">BiLSTM Hidden States · T × 192-D Temporal Memory</p>
            </div>
          </div>

          {/* Connector 6 */}
          <div className="pipe-connector" aria-hidden="true">
            <span className="pipe-line" style={{ height: '1.8rem' }} />
            <span className="pipe-arrow" style={{ color: activeStep === 6 ? 'var(--accent)' : 'inherit' }}>↓</span>
          </div>

          {/* Node 7: CTC Letter Decoding */}
          <div className="pipe-row">
            <div
              className={`node is-accent ${activeStep === 6 ? 'is-accent' : ''}`}
              style={{
                transition: 'all 0.3s ease',
                transform: activeStep === 6 ? 'scale(1.08)' : 'none',
                boxShadow: activeStep === 6 ? '0 0 30px rgba(143, 223, 203, 0.6)' : 'none',
                background: 'linear-gradient(180deg, rgba(143, 223, 203, 0.15) 0%, rgba(10, 10, 12, 0.95) 100%)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="node-title" style={{ color: 'var(--accent)' }}>06 · CTC Linguistic Decoding</p>
                {activeStep === 6 && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--success)' }}>✔ OUTPUT RECONSTRUCTED</span>}
              </div>
              <p className="node-sub">Connectionist Temporal Classification · A–Z Letter Stream</p>
            </div>
          </div>
        </div>

        {/* Quick Follow-up Action Bar */}
        <div style={{
          marginTop: '3.5rem',
          display: 'flex',
          justifyContent: 'center',
          gap: '1.2rem',
          flexWrap: 'wrap'
        }}>
          <a
            href="/demo/live"
            className="btn btn-solid"
            style={{ padding: '10px 24px', fontSize: '0.88rem' }}
          >
            Launch Live Webcam Studio →
          </a>
          <a
            href="/science#scene-architecture"
            className="btn btn-ghost"
            style={{ padding: '10px 24px', fontSize: '0.88rem' }}
          >
            Inspect Mathematical Equations →
          </a>
        </div>
      </div>
    </section>
  )
}
