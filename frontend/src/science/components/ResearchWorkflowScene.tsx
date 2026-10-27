export function ResearchWorkflowScene() {
  const steps = [
    {
      num: '01',
      title: 'Acoustic-Visual Corpus Acquisition',
      data: 'GRID Corpus (s1: 3,005 recordings) + LRW 500-Word Spec',
      detail: 'Curated studio-quality videos with precise word and phoneme boundary alignment files (.align).'
    },
    {
      num: '02',
      title: 'Facial Mesh & Landmark Extraction',
      data: 'MediaPipe 468-point 3D Face Mesh',
      detail: '22 critical lip contour indices sampled at 30 FPS. Automatic discard of frames with facial occlusion.'
    },
    {
      num: '03',
      title: 'Spatial Canonicalization & Augmentation',
      data: 'Affine Rotation + 140×46 Crop + Z-Score Norm',
      detail: 'Horizontal flip (p=0.5) and random temporal frame jitter (±2 frames) to simulate speech rate variability.'
    },
    {
      num: '04',
      title: 'Speaker-Disjoint Dataset Partitioning',
      data: 'Strict Split Protocol: Train (80%) / Val (10%) / Test (10%)',
      detail: 'Enforces zero speaker leakage: unseen subjects in evaluation to measure true generalized visual reading.'
    },
    {
      num: '05',
      title: 'Optimization & Regularized Training',
      data: 'AdamW (lr=3e-4) + CosineAnnealingLR + Label Smoothing (0.1)',
      detail: 'Multi-epoch backpropagation on PyTorch CPU-only / CUDA graph with gradient clipping at norm 1.0.'
    },
    {
      num: '06',
      title: 'Cryptographic Checkpoint Staging',
      data: 'SHA-256 Verified: 7C00D5117B7FAE9A... (alphabet_classifier.pt)',
      detail: 'Implements immutable weight caching ensuring 100% reproducible baseline evaluations.'
    }
  ]

  return (
    <section id="scene-workflow" className="science-scene" aria-label="Scene 8: Experiments and Reproducibility">
      <div className="scene-header">
        <span className="scene-num">Scene 08 · Scientific Methodology</span>
        <h2 className="scene-title">Experiments and Reproducibility.</h2>
        <p className="scene-lead">
          Reproducibility is the bedrock of scientific computing. In machine learning for computer vision, 
          a model checkpoint is only as valid as the data partitioning protocols and verification pipelines behind it.
        </p>
      </div>

      <div className="sci-card">
        <div className="sci-card-header">
          <h3 className="sci-card-title">End-to-End Experimental Lineage</h3>
          <span className="sci-tag verified">6-Stage Protocol</span>
        </div>

        {/* 6-Stage Timeline Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1rem',
          margin: '1.5rem 0'
        }}>
          {steps.map((s) => (
            <div
              key={s.num}
              style={{
                background: 'rgba(237, 235, 228, 0.02)',
                border: '1px solid var(--line)',
                borderRadius: '8px',
                padding: '1.2rem',
                position: 'relative'
              }}
            >
              <div style={{
                display: 'inline-block',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.72rem',
                color: 'var(--accent)',
                background: 'rgba(143, 223, 203, 0.1)',
                padding: '2px 8px',
                borderRadius: '4px',
                marginBottom: '0.6rem'
              }}>
                STAGE {s.num}
              </div>
              <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--ink)', marginBottom: '0.3rem' }}>
                {s.title}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--info)', marginBottom: '0.5rem' }}>
                {s.data}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--ink-dim)', lineHeight: 1.5 }}>
                {s.detail}
              </div>
            </div>
          ))}
        </div>

        {/* Scientific Alert Box: The Danger of Data Leakage */}
        <div style={{
          background: 'rgba(252, 211, 77, 0.05)',
          border: '1px solid rgba(252, 211, 77, 0.25)',
          borderLeft: '4px solid var(--warning, #fcd34d)',
          borderRadius: '8px',
          padding: '1.25rem 1.5rem',
          marginTop: '1.5rem'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.82rem',
            color: 'var(--warning)',
            fontWeight: 600,
            textTransform: 'uppercase',
            marginBottom: '0.4rem'
          }}>
            <span>⚠</span> Scientific Guardrail: Speaker Identity Leakage
          </div>
          <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-dim)', lineHeight: 1.6 }}>
            In speech recognition, naive random splitting of video clips causes <strong>speaker leakage</strong>: 
            the neural network memorizes individual facial contours, lip pigmentation, and jaw morphology rather than 
            learning speech kinematics. In LipNet-AI, rigorous evaluation requires holding out entire speaker corpora 
            to prove genuine speaker-independent generalization.
          </p>
        </div>
      </div>
    </section>
  )
}
