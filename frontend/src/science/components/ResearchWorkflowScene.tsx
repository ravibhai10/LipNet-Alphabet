export function ResearchWorkflowScene() {
  const steps = [
    { num: '01', title: 'Inventory the Available Data', data: 'GRID: 32,891 usable sentences / 33 speakers; MIRACL: real RGB sequences', detail: 'GRID letter slots stay separate from word recognition. MIRACL supplied eight unambiguous word classes. The 45 legacy LRW fixture videos are not a genuine LRW training corpus.' },
    { num: '02', title: 'Facial Mesh & Landmark Extraction', data: 'Measured landmarks; multiple-face and lighting checks', detail: 'The eight-word model uses its original eye-axis crop. The separate 500-word research path uses publisher mean-face geometry with explicitly approximate MediaPipe anatomical correspondences.' },
    { num: '03', title: 'Spatial and Temporal Preprocessing', data: 'Word models: 96-pixel ROIs, center 88, mean 0.421 / std 0.165', detail: 'The eight-word model resamples to its trained 29-frame setting. The webcam 500-word path samples actual timestamps at 25 Hz; interpolation does not recover missing visual information. Alphabet preprocessing remains separate.' },
    { num: '04', title: 'Require Independent Evaluation Data', data: 'Eight-word train / validation / test speakers are separate', detail: 'The 192 test utterances exclude 48 previously used pilot utterances. Test people had appeared in that historical pilot. Native MIRACL FPS is unknown. Complete authorized LRW evaluation data and publisher-training speaker separation remain unavailable.' },
    { num: '05', title: 'Select Using Validation Only', data: 'Frozen AV-HuBERT layer 9 + mean/std pooling + linear word head', detail: 'The eight-word head was selected on 240 validation utterances before the frozen 192-clip test. The separate publisher-trained 500-word model was not trained or tuned on local diagnostic examples.' },
    { num: '06', title: 'Preserve and Verify Artifacts', data: 'Checkpoint, vocabulary, preprocessing and source SHA-256 fingerprints', detail: 'Original model weights and eight-word evaluation artifacts remain protected. The 500-word diagnostic records real predictions, failures, limited class coverage and CPU measurements; it does not establish broad-English or webcam accuracy.' },
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
          <h3 className="sci-card-title">Available Code and Remaining Evidence</h3>
          <span className="sci-tag">Audited inventory · not a completed benchmark</span>
        </div>

        {/* 6-Stage Timeline Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))',
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
