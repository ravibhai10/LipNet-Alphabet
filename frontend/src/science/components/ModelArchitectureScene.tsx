import { useState } from 'react'
import { MathExpression } from '../../components/MathExpression'
import { EQUATIONS } from '../../lib/equations'

export function ModelArchitectureScene() {
  const [modelType, setModelType] = useState<'alphabet' | 'lrw'>('alphabet')
  const [selectedLayer, setSelectedLayer] = useState<number>(0)

  const alphabetLayers = [
    {
      name: 'Input Video Tensor',
      shape: '(B, 30, 1, 140, 46)',
      op: 'Batch formatting · CPU browser inference',
      math: EQUATIONS.inputAlphabet,
      desc: '30 consecutive frames of 140×46 single-channel grayscale mouth ROIs normalized to [0, 1].',
      why: 'Preserves fine vertical labial aperture dynamics while discarding irrelevant facial skin and chromatic variations.'
    },
    {
      name: 'Spatial Conv Block 1',
      shape: '(B·30, 16, 4, 4)',
      op: 'Conv2D(1→16, k=3, p=1) + BatchNorm2D + ReLU + AdaptiveAvgPool2d(4,4)',
      math: EQUATIONS.convFirst,
      desc: 'Collapses batch and temporal dimensions into (B·T) 2D images. Filters learn horizontal and vertical lip edge gradients.',
      why: 'Extracts shared per-frame spatial features. The learned filters have not been audited for specific anatomical meanings.'
    },
    {
      name: 'Spatial Conv Block 2 & 3',
      shape: '(B·30, 64, 4, 4)',
      op: 'Conv2D(16→32) + MaxPool(2) → Conv2D(32→64) + MaxPool(2) + AdaptiveAvgPool(4,4)',
      math: EQUATIONS.convThird,
      desc: 'Hierarchical receptive field expansion. Feature dimension flattens to 64 × 4 × 4 = 1,024 scalar descriptors per frame.',
      why: 'Produces the 1,024-dimensional input expected by the saved recurrent layer.'
    },
    {
      name: 'Temporal Sequence Reshape',
      shape: '(B, 30, 1024)',
      op: 'Tensor unflattening & temporal alignment',
      math: EQUATIONS.flatten,
      desc: 'Restores chronological time axis: B batches of 30 distinct 1,024-dimensional spatial embedding vectors.',
      why: 'Prepares frame-level visual embeddings for sequential temporal recurrent modeling.'
    },
    {
      name: 'Bidirectional LSTM (BiLSTM)',
      shape: '(B, 30, 192)',
      op: 'nn.LSTM(1024, hidden=96, layers=1, bidirectional=True)',
      math: EQUATIONS.bidirectional,
      desc: 'Processes the 30-frame sequence in both forward (t=0→29) and reverse (t=29→0) directions simultaneously.',
      why: 'Provides sequence context in both directions. It still returns a single clip-level class distribution after the dense head.'
    },
    {
      name: 'Linear Dense Bottleneck',
      shape: '(B, 128)',
      op: 'Linear(192 → 128) + ReLU + Dropout(0.5)',
      math: EQUATIONS.dropout,
      desc: 'Extracts the final temporal hidden state and compresses it into a regularized 128-dimensional representation.',
      why: 'Dropout masks activations during training and is disabled in evaluation mode. Its generalization benefit is not measured here.'
    },
    {
      name: '26-Class Logits & Softmax',
      shape: '(B, 26)',
      op: 'Linear(128 → 26) → Unnormalized Logits → Softmax Top-3',
      math: EQUATIONS.alphabetLogits,
      desc: 'Projects the clip representation into 26 raw logits for A–Z. The strictly loaded model has 913,274 parameters.',
      why: 'Deployed in app/services/browser_inference.py for isolated alphabet classification at POST /api/predict.'
    }
  ]

  const lrwLayers = [
    {
      name: 'Input Video Tensor (LRW Spec)',
      shape: '(B, 29, 1, 140, 46)',
      op: 'LRW 500-Word Specification (29 frames @ 25fps = 1.16s)',
      math: EQUATIONS.inputWords,
      desc: '29 frames corresponding to standard Oxford LRW word snippets centered on the target keyword utterance.',
      why: 'Uniform word-level duration standard across 500 English vocabulary classes.'
    },
    {
      name: 'Per-Frame 2D CNN Features',
      shape: '(B, 29, 1024)',
      op: 'Conv2D 16→32→64 + BatchNorm + pooling + flatten',
      math: EQUATIONS.visualFeatures,
      desc: 'The offline LRW files use the same per-frame 2D CNN family as the alphabet classifier, producing 1,024 features per frame.',
      why: 'Strict state-dictionary loading confirms this architecture; these files do not contain a ResNet, 3D CNN, or AV-HuBERT backbone.'
    },
    {
      name: 'Bidirectional LSTM',
      shape: '(B, 29, 192)',
      op: 'nn.LSTM(1024, hidden=96, bidirectional=True)',
      math: EQUATIONS.bidirectional,
      desc: 'The saved 500-output CNN–BiLSTM uses a one-layer recurrent encoder, with 96 hidden units in each direction.',
      why: 'Its temporal architecture is verified by strict loading, independently of whether its recognition quality is established.'
    },
    {
      name: '500-Word Classification Projection',
      shape: '(B, 500)',
      op: 'Linear(192→128) + ReLU + Dropout + Linear(128→500)',
      math: EQUATIONS.wordLogits,
      desc: 'The checkpoint produces 500 logits. The code maps them using data/vocab.txt order; the checkpoint does not embed a vocabulary fingerprint.',
      why: 'One-epoch training metadata and generated development clips do not prove genuine LRW recognition. No word prediction endpoint is connected.'
    }
  ]

  const layers = modelType === 'alphabet' ? alphabetLayers : lrwLayers
  const active = layers[Math.min(selectedLayer, layers.length - 1)]

  return (
    <section id="scene-architecture" className="science-scene" aria-label="Scene 4: Inside the Neural Network">
      <div className="scene-header">
        <span className="scene-num">Scene 04 · Neural Architecture</span>
        <h2 className="scene-title">Inside the Deep Neural Pipeline.</h2>
        <p className="scene-lead">
          The inspected alphabet and experimental word checkpoints use per-frame 2D convolutions and a bidirectional LSTM. AV-HuBERT is a separate encoder, not a stage inside either saved CNN–BiLSTM.
        </p>
      </div>

      <div className="sci-card">
        {/* Model Selector Tabs */}
        <div className="sci-tabs">
          <button
            className={`sci-tab-btn ${modelType === 'alphabet' ? 'active' : ''}`}
            onClick={() => { setModelType('alphabet'); setSelectedLayer(0); }}
          >
            AlphabetCNNBiLSTM (26 Classes · CPU Tested)
          </button>
          <button
            className={`sci-tab-btn ${modelType === 'lrw' ? 'active' : ''}`}
            onClick={() => { setModelType('lrw'); setSelectedLayer(0); }}
          >
            LRW 500-Word Spec (Offline Experimental Checkpoint)
          </button>
        </div>

        {/* Status Notification Banner */}
        <div style={{
          marginBottom: '1.2rem',
          padding: '0.65rem 1rem',
          borderRadius: '8px',
          fontSize: '0.78rem',
          fontFamily: 'var(--font-mono)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: modelType === 'alphabet' ? 'rgba(143, 223, 203, 0.08)' : 'rgba(235, 178, 86, 0.08)',
          border: `1px solid ${modelType === 'alphabet' ? 'rgba(143, 223, 203, 0.3)' : 'rgba(235, 178, 86, 0.3)'}`,
          color: modelType === 'alphabet' ? 'var(--accent)' : '#ebb256'
        }}>
          <span>
            {modelType === 'alphabet'
              ? 'CPU / ASGI verified: models/alphabet_classifier.pt (913,274 parameters) returns a 26-class clip distribution at POST /api/predict. No CTC or word decoding.'
              : 'Offline only: three 500-output CNN–BiLSTM files load strictly (974,420 parameters each). Genuine LRW training and recognition are unverified; these legacy files are not used by the separate publisher DC-TCN word API.'}
          </span>
          <span style={{
            background: modelType === 'alphabet' ? 'rgba(143, 223, 203, 0.2)' : 'rgba(235, 178, 86, 0.2)',
            padding: '2px 8px',
            borderRadius: '4px',
            textTransform: 'uppercase',
            fontWeight: 600,
            fontSize: '0.7rem'
          }}>
            {modelType === 'alphabet' ? 'Inference tested' : 'Unverified recognition'}
          </span>
        </div>

        <div className="sci-split">
          {/* Left: Interactive Layer Flow Column */}
          <div>
            <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--ink-dim)', marginBottom: '0.8rem' }}>
              PIPELINE STAGES (CLICK TO INSPECT TRANSFORMATION):
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {layers.map((layer, idx) => {
                const isSelected = selectedLayer === idx
                return (
                  <div
                    key={layer.name}
                    onClick={() => setSelectedLayer(idx)}
                    style={{
                      background: isSelected ? 'rgba(143, 223, 203, 0.1)' : 'rgba(237, 235, 228, 0.02)',
                      border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--line)'}`,
                      borderRadius: '8px',
                      padding: '0.75rem 1rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <div>
                      <div style={{ fontSize: '0.84rem', fontWeight: 600, color: isSelected ? 'var(--ink)' : 'var(--ink-dim)' }}>
                        {layer.name}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--ink-faint)', fontFamily: 'var(--font-mono)' }}>
                        {layer.op.slice(0, 45)}...
                      </div>
                    </div>
                    <div style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.72rem',
                      color: isSelected ? 'var(--accent)' : 'var(--ink-dim)',
                      background: isSelected ? 'rgba(143, 223, 203, 0.15)' : 'rgba(237, 235, 228, 0.05)',
                      padding: '2px 8px',
                      borderRadius: '4px'
                    }}>
                      {layer.shape}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Right: Selected Stage Deep Mathematical Inspector */}
          <div style={{
            background: '#08080a',
            border: '1px solid rgba(143, 223, 203, 0.25)',
            borderRadius: '12px',
            padding: '1.5rem',
            position: 'sticky',
            top: '80px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <span className="sci-tag verified">Stage {selectedLayer + 1} of {layers.length}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)' }}>
                TENSOR: {active.shape}
              </span>
            </div>

            <h3 style={{ fontSize: '1.2rem', margin: '0 0 0.5rem', color: 'var(--ink)' }}>
              {active.name}
            </h3>

            <div className="math-box" style={{ margin: '1rem 0' }}>
              <div className="math-formula" style={{ fontSize: '1rem' }}>
                <MathExpression tex={active.math}/>
              </div>
              <p className="math-desc">
                {active.op}
              </p>
            </div>

            <div style={{ margin: '1rem 0', fontSize: '0.85rem', color: 'var(--ink-dim)', lineHeight: 1.6 }}>
              <strong style={{ color: 'var(--ink)', display: 'block', marginBottom: '4px' }}>Functional Role:</strong>
              {active.desc}
            </div>

            <div style={{
              background: 'rgba(143, 223, 203, 0.06)',
              border: '1px solid rgba(143, 223, 203, 0.2)',
              borderRadius: '8px',
              padding: '0.85rem 1rem',
              fontSize: '0.8rem',
              color: 'var(--accent)',
              lineHeight: 1.5
            }}>
              <strong>Biomechanical Rationale: </strong>
              {active.why}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
