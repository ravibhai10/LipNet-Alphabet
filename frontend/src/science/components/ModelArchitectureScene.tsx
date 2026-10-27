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
      op: 'Batch formatting & device staging',
      math: EQUATIONS.inputAlphabet,
      desc: '30 consecutive frames of 140×46 single-channel grayscale mouth ROIs with pixel values scaled to [0, 1].',
      why: 'Preserves fine vertical labial aperture dynamics while discarding irrelevant facial skin and RGB chromatic variations.'
    },
    {
      name: 'Spatial Conv Block 1',
      shape: '(B·30, 16, 4, 4)',
      op: 'Conv2D(1→16, k=3, p=1) + BatchNorm2D + ReLU + AdaptiveAvgPool2d(4,4)',
      math: EQUATIONS.convFirst,
      desc: 'Collapses batch and temporal dimensions into (B·T) 2D images. Filters learn horizontal and vertical lip edge gradients.',
      why: 'Extracts low-level edge contours (vermilion border) with translation invariance.'
    },
    {
      name: 'Spatial Conv Block 2 & 3',
      shape: '(B·30, 64, 4, 4)',
      op: 'Conv2D(16→32) + MaxPool(2) → Conv2D(32→64) + MaxPool(2) + AdaptiveAvgPool(4,4)',
      math: EQUATIONS.convThird,
      desc: 'Hierarchical receptive field expansion. Feature dimension flattens to 64 × 4 × 4 = 1,024 scalar descriptors per frame.',
      why: 'Captures composite mouth geometries (oral cavity darkness, teeth occlusion, lip rounding).'
    },
    {
      name: 'Temporal Sequence Reshape',
      shape: '(B, 30, 1024)',
      op: 'Tensor unflattening & temporal alignment',
      math: EQUATIONS.flatten,
      desc: 'Restores the chronological time axis: B batches of 30 distinct 1,024-dimensional spatial embedding vectors.',
      why: 'Prepares frame-level visual embeddings for sequential temporal recurrent modeling.'
    },
    {
      name: 'Bidirectional LSTM (BiLSTM)',
      shape: '(B, 30, 192)',
      op: 'nn.LSTM(1024, hidden=96, layers=1, bidirectional=True)',
      math: EQUATIONS.bidirectional,
      desc: 'Processes the 30-frame sequence in both forward (t=0→29) and reverse (t=29→0) directions simultaneously.',
      why: 'Resolves phonetic co-articulation: the mouth anticipates future sounds (forward) while lingering from prior sounds (backward).'
    },
    {
      name: 'Linear Dense Bottleneck',
      shape: '(B, 128)',
      op: 'Linear(192 → 128) + ReLU + Dropout(0.5)',
      math: EQUATIONS.dropout,
      desc: 'Extracts the final (30th) temporal hidden state and compresses it into a regularized 128-dimensional representation.',
      why: 'Prevents overfitting on small speaker training subsets by randomly masking 50% of latent activations.'
    },
    {
      name: 'Classification Head & Logits',
      shape: '(B, 26)',
      op: 'Linear(128 → 26) → Unnormalized Logits z',
      math: EQUATIONS.alphabetLogits,
      desc: 'Projects the speech representation into unnormalized log-probability scores corresponding to the 26 English letters (A–Z).',
      why: 'Provides raw logits ready for numerically stable Softmax probability transformation and Cross-Entropy loss.'
    }
  ]

  const lrwLayers = [
    {
      name: 'Input Video Tensor',
      shape: '(B, 29, 1, 140, 46)',
      op: 'LRW 500-Word Specification',
      math: EQUATIONS.inputWords,
      desc: '29 frames corresponding to standard Oxford LRW word snippets centered on the target keyword utterance.',
      why: 'Uniform word-level duration standard across 500 English vocabulary classes.'
    },
    {
      name: 'Spatio-Temporal Visual Frontend',
      shape: '(B, 29, 1024)',
      op: 'ResNet / 3D-CNN Feature Extractor',
      math: EQUATIONS.visualFeatures,
      desc: 'Extracts deep spatio-temporal representations from moving visual lips using 3D convolutions across adjacent frames.',
      why: 'Integrates local motion across 3-frame temporal receptive fields before global sequence modeling.'
    },
    {
      name: 'AV-HuBERT Transformer / BiLSTM Encoder',
      shape: '(B, 29, 768 or 192)',
      op: 'Multi-Head Self-Attention or BiLSTM',
      math: EQUATIONS.attention,
      desc: 'Self-supervised acoustic-visual masked prediction encoder capturing long-range contextual dependencies.',
      why: 'Enables word discrimination across phonetically complex multi-syllable vocabularies.'
    },
    {
      name: '500-Word Classification Projection',
      shape: '(B, 500)',
      op: 'Linear(d → 500) + Label Smoothing (0.1)',
      math: EQUATIONS.wordLogits,
      desc: 'High-dimensional projection across the complete 500-word English vocabulary defined in data/vocab.txt.',
      why: 'Provides vocabulary-scale classification with regularized loss to prevent overconfidence.'
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
          LipNet-AI does not use generic black-box networks. The system couples a spatial convolutional frontend 
          with a temporal bidirectional recurrent backbone designed specifically for the physics of co-articulation.
        </p>
      </div>

      <div className="sci-card">
        {/* Model Selector Tabs */}
        <div className="sci-tabs">
          <button
            className={`sci-tab-btn ${modelType === 'alphabet' ? 'active' : ''}`}
            onClick={() => { setModelType('alphabet'); setSelectedLayer(0); }}
          >
            AlphabetCNNBiLSTM (Verified 26-Class Baseline)
          </button>
          <button
            className={`sci-tab-btn ${modelType === 'lrw' ? 'active' : ''}`}
            onClick={() => { setModelType('lrw'); setSelectedLayer(0); }}
          >
            LRW 500-Word Extension & AV-HuBERT
          </button>
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
