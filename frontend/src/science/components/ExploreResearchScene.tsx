export function ExploreResearchScene() {
  return (
    <section id="scene-explore" className="science-scene" aria-label="Scene 10: Explore the Research">
      <div className="sci-card" style={{
        background: 'radial-gradient(circle at 50% 0%, rgba(143, 223, 203, 0.12) 0%, rgba(10, 10, 12, 0.95) 70%)',
        border: '1px solid rgba(143, 223, 203, 0.3)',
        textAlign: 'center',
        padding: 'clamp(2.5rem, 5vw, 5rem) 2rem'
      }}>
        <span className="scene-num" style={{ margin: '0 auto 1rem' }}>
          Scene 10 · The Continuing Inquiry
        </span>

        <h2 style={{
          fontFamily: 'var(--font-display, Georgia, serif)',
          fontSize: 'clamp(2.2rem, 5vw, 4rem)',
          fontWeight: 400,
          lineHeight: 1.15,
          color: 'var(--ink, #edebe4)',
          maxWidth: '850px',
          margin: '0 auto 1.5rem'
        }}>
          From motion to meaning — one experiment at a time.
        </h2>

        <p style={{
          fontSize: 'clamp(1rem, 1.4vw, 1.2rem)',
          color: 'var(--ink-dim, #a3a2a9)',
          maxWidth: '700px',
          margin: '0 auto 2.5rem',
          lineHeight: 1.6
        }}>
          The journey from silent biological kinematics to deep semantic understanding is an open frontier. 
          Step into our live testing environments, explore model weights, or inspect the real-time webcam inference engine.
        </p>

        {/* Action Navigation Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '1.2rem',
          maxWidth: '960px',
          margin: '0 auto',
          textAlign: 'left'
        }}>
          <a
            href="/demo/live"
            style={{
              background: 'rgba(143, 223, 203, 0.08)',
              border: '1px solid var(--accent, #8fdfcb)',
              borderRadius: '12px',
              padding: '1.5rem',
              textDecoration: 'none',
              transition: 'all 0.25s ease',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between'
            }}
          >
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--accent)', textTransform: 'uppercase' }}>
                Interactive Application
              </span>
              <h3 style={{ fontSize: '1.15rem', color: 'var(--ink)', margin: '0.4rem 0' }}>
                Launch Live Demo →
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--ink-dim)', margin: 0 }}>
                Test real-time webcam inference with live landmark tracking and EMA probability smoothing.
              </p>
            </div>
          </a>

          <a
            href="/demo/models"
            style={{
              background: 'rgba(237, 235, 228, 0.02)',
              border: '1px solid var(--line)',
              borderRadius: '12px',
              padding: '1.5rem',
              textDecoration: 'none',
              transition: 'all 0.25s ease',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between'
            }}
          >
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--info)', textTransform: 'uppercase' }}>
                Model Registry
              </span>
              <h3 style={{ fontSize: '1.15rem', color: 'var(--ink)', margin: '0.4rem 0' }}>
                Model Checkpoints →
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--ink-dim)', margin: 0 }}>
                Inspect verified model checkpoints, parameter footprints, and cryptographic hashes.
              </p>
            </div>
          </a>

          <a
            href="/demo/pipeline"
            style={{
              background: 'rgba(237, 235, 228, 0.02)',
              border: '1px solid var(--line)',
              borderRadius: '12px',
              padding: '1.5rem',
              textDecoration: 'none',
              transition: 'all 0.25s ease',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between'
            }}
          >
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--ink-dim)', textTransform: 'uppercase' }}>
                Engineering Architecture
              </span>
              <h3 style={{ fontSize: '1.15rem', color: 'var(--ink)', margin: '0.4rem 0' }}>
                Pipeline Graph →
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--ink-dim)', margin: 0 }}>
                Explore the modular pipeline architecture connecting MediaPipe, PyTorch, and FastAPI.
              </p>
            </div>
          </a>

          <a
            href="/"
            style={{
              background: 'rgba(237, 235, 228, 0.02)',
              border: '1px solid var(--line)',
              borderRadius: '12px',
              padding: '1.5rem',
              textDecoration: 'none',
              transition: 'all 0.25s ease',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between'
            }}
          >
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--ink-faint)', textTransform: 'uppercase' }}>
                Executive Overview
              </span>
              <h3 style={{ fontSize: '1.15rem', color: 'var(--ink)', margin: '0.4rem 0' }}>
                Main Landing Page →
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--ink-dim)', margin: 0 }}>
                Return to the main visual presentation, interactive playgrounds, and project summary.
              </p>
            </div>
          </a>
        </div>

        {/* Footer Citations & Standards */}
        <div style={{
          marginTop: '4rem',
          borderTop: '1px solid rgba(237, 235, 228, 0.08)',
          paddingTop: '2rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.74rem',
          color: 'var(--ink-faint)'
        }}>
          <div>LIPNET-AI · SCIENTIFIC RESEARCH & BENCHMARK SUITE</div>
          <div>STANDARDS: REC. 601 LUMA · SHANNON ENTROPY · PYTORCH 2.X</div>
          <div>CPU-OPTIMIZED INFERENCE PIPELINE</div>
        </div>
      </div>
    </section>
  )
}
