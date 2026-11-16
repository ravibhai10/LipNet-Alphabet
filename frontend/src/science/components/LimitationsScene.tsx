export function LimitationsScene() {
  const challenges = [
    {
      title: 'Viseme Homopheny (Phonetic Ambiguity)',
      impact: 'Intrinsic Information Limit',
      metric: 'Confusion groups have not been measured for this classifier',
      desc: 'Phonemes like /p/ (pat), /b/ (bat), and /m/ (mat) are bilabial articulatory gestures. They differ only in vocal cord vibration and velum elevation—internal biomechanics that produce zero optical deformation on the external face.',
      solution: 'No language-model beam search is connected to the active classifier. Its uncalibrated scores and errors require held-out, labeled recordings.'
    },
    {
      title: 'Non-Frontal Head Pose & Yaw Angular Drift',
      impact: 'Geometric Occlusion',
      metric: 'No validated yaw or pitch operating range',
      desc: '2D bounding-box affine normalization assumes near-frontal orientation. As the speaker turns their head, one oral commissure is occluded and depth cues are lost.',
      solution: 'Measured eye-axis alignment is implemented. Pose robustness and recovery need physical-camera tests; no multi-view recognition system is deployed.'
    },
    {
      title: 'Sensor Motion Blur & Rolling Shutter',
      impact: 'Temporal Smear',
      metric: 'Physical-webcam blur and speed tolerance are untested',
      desc: 'Low-cost laptop webcams automatically lengthen exposure times in dim room lighting. A fast lip explosion (occurring in 20–30ms) turns into a blurry gray streak across adjacent video frames.',
      solution: 'The browser has mouth-motion gating and bounded requests. These mechanisms do not establish recognition accuracy under blur or CPU load.'
    },
    {
      title: 'Domain Shift: Studio Benchmarks vs Real Life',
      impact: 'Generalization Gap',
      metric: 'Distributional drift from high-end cameras to consumer webcams',
      desc: 'The current real corpus has one GRID speaker. Local LRW videos are generated fixtures. Neither provides a speaker-independent webcam benchmark.',
      solution: 'Training-only brightness/contrast jitter, spatial jitter, and temporal masking are implemented. Their effect on held-out recognition has not been measured.'
    }
  ]

  return (
    <section id="scene-limitations" className="science-scene" aria-label="Scene 9: Limitations Are Part of the Science">
      <div className="scene-header">
        <span className="scene-num">Scene 09 · Scientific Boundaries</span>
        <h2 className="scene-title">Limitations Are Part of the Science.</h2>
        <p className="scene-lead">
          Real scientific progress requires defining the operational boundaries of our models. 
          Optical speech recognition operates under hard physical constraints imposed by optics, anatomy, and information theory.
        </p>
      </div>

      <div className="sci-card">
        <div className="sci-card-header">
          <h3 className="sci-card-title">Physical & Biomechanical Frontiers</h3>
          <span className="sci-tag">Open Research Horizons</span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
          gap: '1.25rem',
          margin: '1.5rem 0'
        }}>
          {challenges.map((c) => (
            <div
              key={c.title}
              style={{
                background: 'rgba(237, 235, 228, 0.02)',
                border: '1px solid var(--line)',
                borderRadius: '10px',
                padding: '1.4rem',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between'
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.72rem',
                    color: 'var(--accent)',
                    background: 'rgba(143, 223, 203, 0.1)',
                    padding: '2px 8px',
                    borderRadius: '4px'
                  }}>
                    {c.impact}
                  </span>
                </div>
                <h4 style={{ fontSize: '1rem', color: 'var(--ink)', margin: '0 0 0.5rem' }}>
                  {c.title}
                </h4>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--warning)', marginBottom: '0.75rem' }}>
                  ● Boundary: {c.metric}
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--ink-dim)', lineHeight: 1.6, margin: 0 }}>
                  {c.desc}
                </p>
              </div>

              <div style={{
                marginTop: '1.2rem',
                borderTop: '1px solid rgba(237, 235, 228, 0.08)',
                paddingTop: '0.8rem',
                fontSize: '0.76rem',
                color: 'var(--accent)'
              }}>
                <strong>Engineering Solution: </strong> {c.solution}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
