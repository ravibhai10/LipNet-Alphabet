import { useState } from "react"
import { Link } from "react-router-dom"
import { useBackend } from "../../hooks/useBackend"
import { PipelineFlow, REFERENCE_STAGES } from "../PipelineFlow"
import { NeuralSynapseVisualizer } from "../components/NeuralSynapseVisualizer"
import { Icon } from "../Icon"

type ArchitectureTrack = "alphabet" | "avhubert" | "ctc"

export function PipelinePage() {
  const [selected, setSelected] = useState(0)
  const [archTrack, setArchTrack] = useState<ArchitectureTrack>("avhubert")
  const { status, modelLoaded } = useBackend()
  const stage = REFERENCE_STAGES[selected]

  const alphabetTensors = [
    { stage: "Input Video Clip", tensor: "[B, 30, 1, 140, 46]", desc: "30 frames of normalized grayscale mouth crops (height: 140px, width: 46px)" },
    { stage: "3D Spatial Convolution", tensor: "[B, 32, 30, 70, 23]", desc: "Conv3D (kernel 3x3x3, 32 channels) + BatchNorm3D + ReLU + MaxPool3D (1, 2, 2)" },
    { stage: "2D Feature Refinement", tensor: "[B, 64, 30, 35, 11]", desc: "Conv2D (kernel 3x3, 64 channels) + BatchNorm2D + ReLU + MaxPool2D (2, 2)" },
    { stage: "Spatial Flattening", tensor: "[B, 30, 24640]", desc: "Flatten spatial channels per temporal frame into sequential feature vectors" },
    { stage: "Bidirectional LSTM", tensor: "[B, 30, 512]", desc: "2-layer BiLSTM (hidden size: 256 x 2 directions = 512 units) with dropout (0.5)" },
    { stage: "Temporal Aggregation", tensor: "[B, 512]", desc: "Last hidden step aggregation across the 30 temporal frame steps" },
    { stage: "Linear Head & Log-Softmax", tensor: "[B, 26]", desc: "Dense linear projection to 26 letter logits (A-Z) passed to Log-Softmax" },
  ]

  const avhubertTensors = [
    { stage: "Input Lip Sequence", tensor: "[B, T, 1, 88, 88]", desc: "T frames normalized grayscale lip region cropped from speaker face landmarks" },
    { stage: "ResNet-3D Front-End", tensor: "[B, T, 768]", desc: "Modified 3D-ResNet visual front-end projecting spatial lip pixels into 768-d token embeddings" },
    { stage: "12-Layer Transformer", tensor: "[B, T, 768]", desc: "Pre-trained AV-HuBERT Base Transformer encoder with multi-head self-attention" },
    { stage: "Layer 9 Feature Tap", tensor: "[B, T, 768]", desc: "Frozen intermediate representations extracted from Transformer Layer 9" },
    { stage: "Statistical Temporal Pooling", tensor: "[B, 1536]", desc: "Concatenation of temporal Mean (mu: 768) and Standard Deviation (sigma: 768) across frames" },
    { stage: "Linear Classification Head", tensor: "[B, 8]", desc: "Trained logistic regression head over 1,536-d vectors yielding 94.79% Top-1 accuracy" },
  ]

  const ctcTensors = [
    { stage: "Input Sentence Clip", tensor: "[B, T=75, 3, 50, 100]", desc: "75 frames of continuous speech video (3-channel RGB or 1-channel grayscale)" },
    { stage: "3D Spatial Convolutions", tensor: "[B, 64, 75, 25, 50]", desc: "Spatiotemporal convolutions capturing lip texture and continuous mouth opening/closing" },
    { stage: "Bi-directional GRU", tensor: "[B, 75, 512]", desc: "Bidirectional Gated Recurrent Units capturing temporal sentence rhythm and phonetic context" },
    { stage: "Dense Vocabulary Projection", tensor: "[B, 75, 28]", desc: "Frame-level character emission logits (26 alphabet + blank token + space)" },
    { stage: "Connectionist Temporal Decoder", tensor: "String text", desc: "CTC greedy or beam-search decoder collapsing repeated blanks into continuous words" },
  ]

  const currentTensors = archTrack === "avhubert" ? avhubertTensors : archTrack === "alphabet" ? alphabetTensors : ctcTensors

  return <div className="ws-page-inner">
    <header className="ws-page-header">
      <p className="ws-kicker">07 / Pipeline & architecture</p>
      <h1 className="ws-title">From movement<br /><em>to meaning.</em></h1>
      <p className="ws-lede">Explore the visual speech architecture, tensor dimensions, and mathematical signal flow across all research tracks.</p>
    </header>

    <div className="ws-grid-3">
      <div className="ws-card ws-metric">
        <span className="ws-metric-label">Backend connection</span>
        <span className="ws-metric-value" style={{ fontSize: "28px" }}>{status === "online" ? "Connected" : status === "checking" || status === "idle" ? "Checking" : "Offline"}</span>
        <span className="ws-note">Port 8001 FastAPI service</span>
      </div>
      <div className="ws-card ws-metric">
        <span className="ws-metric-label">AV-HuBERT 8-Word Model</span>
        <span className="ws-metric-value" style={{ fontSize: "28px", color: "#10b981" }}>94.79%</span>
        <span className="ws-note">Top-1 benchmark accuracy</span>
      </div>
      <div className="ws-card ws-metric">
        <span className="ws-metric-label">Alphabet CNN-BiLSTM</span>
        <span className="ws-metric-value" style={{ fontSize: "28px" }}>{modelLoaded === undefined ? "Unknown" : modelLoaded ? "Loaded" : "Not loaded"}</span>
        <span className="ws-note">26-class letter classification</span>
      </div>
    </div>

    <div style={{ margin: "24px 0" }}>
      <NeuralSynapseVisualizer />
    </div>

    <section className="ws-card">
      <div className="ws-card-head">
        <div>
          <p className="ws-kicker">Signal flow diagram</p>
          <h2>Follow the signal</h2>
        </div>
        <span className="ws-badge ws-badge-muted">Interactive stage selector</span>
      </div>
      <PipelineFlow activeStage={selected} onSelect={setSelected} />
      <p className="ws-note">Click any stage above to inspect its component role, input requirements, and runtime behavior.</p>
    </section>

    <div className="ws-grid-2">
      <section className="ws-card">
        <div className="ws-card-head">
          <div>
            <p className="ws-kicker">Stage {String(selected + 1).padStart(2, "0")} / 08</p>
            <h2>{stage.label}</h2>
          </div>
          <Icon name="pipeline" size={26} />
        </div>
        <span className={`ws-badge ${stage.status === "Implemented" ? "ws-badge-ready" : "ws-badge-muted"}`}>{stage.status}</span>
        <p className="ws-lede" style={{ marginTop: "20px", fontSize: "17px" }}>{stage.note}</p>
        <dl className="ws-detail-list">
          <div><dt>Component</dt><dd>{stage.dimension}</dd></div>
          <div><dt>Verification</dt><dd>{stage.status === "Implemented" ? "Active in workspace runtime" : "Research baseline architecture"}</dd></div>
        </dl>
        <Link className="ws-btn ws-btn-primary" to="/demo/live" style={{ marginTop: "16px" }}>See live telemetry <Icon name="arrow" /></Link>
      </section>

      <section className="ws-card">
        <div className="ws-card-head">
          <div>
            <p className="ws-kicker">The current live path</p>
            <h2>End-to-End Workflow</h2>
          </div>
          <Icon name="scan" />
        </div>
        <div className="ws-steps">
          {[
            { title: "Capture & locate", note: "Browser camera frames (30 FPS) with localized MediaPipe facial landmarks (indices 61, 291, 0, 17)." },
            { title: "Crop & sequence", note: "Normalized mouth crop extraction with temporal frame buffer (min 15, max 75 frames)." },
            { title: "Encode & classify", note: "AV-HuBERT Layer 9 feature pooling or CNN-BiLSTM spatial-temporal projection." },
            { title: "Consensus gate", note: "2-window temporal consensus verifying stability before committing final prediction." },
          ].map((item, index) => (
            <div className="ws-step-card" key={item.title}>
              <span className="ws-step-num">{String(index + 1).padStart(2, "0")}</span>
              <div>
                <h3 className="ws-step-title">{item.title}</h3>
                <p className="ws-step-desc">{item.note}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>

    {/* Interactive Tensor Dimension & Mathematics Inspector */}
    <section className="ws-card" style={{ marginTop: "24px" }}>
      <div className="ws-card-head">
        <div>
          <p className="ws-kicker">Tensor dimension inspector</p>
          <h2>Layer-by-Layer Data Transformations</h2>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            type="button"
            className={`ws-btn ws-btn-sm ${archTrack === "avhubert" ? "ws-btn-primary" : "ws-btn-ghost"}`}
            onClick={() => setArchTrack("avhubert")}
          >
            AV-HuBERT 8-Word (SOTA)
          </button>
          <button
            type="button"
            className={`ws-btn ws-btn-sm ${archTrack === "alphabet" ? "ws-btn-primary" : "ws-btn-ghost"}`}
            onClick={() => setArchTrack("alphabet")}
          >
            Alphabet CNN-BiLSTM
          </button>
          <button
            type="button"
            className={`ws-btn ws-btn-sm ${archTrack === "ctc" ? "ws-btn-primary" : "ws-btn-ghost"}`}
            onClick={() => setArchTrack("ctc")}
          >
            Continuous CTC Track
          </button>
        </div>
      </div>

      <div className="ws-table-wrap">
        <table className="ws-table">
          <thead>
            <tr>
              <th style={{ width: "25%" }}>Processing Stage</th>
              <th style={{ width: "25%" }}>Tensor Contract</th>
              <th style={{ width: "50%" }}>Mathematical Operation & Notes</th>
            </tr>
          </thead>
          <tbody>
            {currentTensors.map((t, i) => (
              <tr key={t.stage}>
                <td>
                  <strong>{t.stage}</strong>
                  <small style={{ display: "block", opacity: 0.6 }}>Step {i + 1} of {currentTensors.length}</small>
                </td>
                <td>
                  <code className="ws-code" style={{ fontSize: "13px", fontWeight: 600, color: "var(--ws-accent)" }}>
                    {t.tensor}
                  </code>
                </td>
                <td style={{ fontSize: "14px", lineHeight: 1.5 }}>{t.desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>

    {/* Capability Map */}
    <section className="ws-card" style={{ marginTop: "24px" }}>
      <div className="ws-card-head">
        <div>
          <p className="ws-kicker">Capability matrix</p>
          <h2>Multimodal Speech Recognition Scope</h2>
        </div>
        <Icon name="layers" />
      </div>
      <div className="ws-table-wrap">
        <table className="ws-table">
          <thead>
            <tr>
              <th>Capability & Task</th>
              <th>Status</th>
              <th>Architecture & Accuracy</th>
              <th>Deployment & Scope</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>AV-HuBERT 8-Word Recognition</strong></td>
              <td><span className="ws-badge ws-badge-ready">Production Verified</span></td>
              <td>AV-HuBERT Base (L9) + Linear Head · <strong>94.79% Top-1</strong> (F1: 0.9467)</td>
              <td>Served on <code>/api/word/predict</code> with manual recording assistant.</td>
            </tr>
            <tr>
              <td><strong>Camera & Real-Time Landmarks</strong></td>
              <td><span className="ws-badge ws-badge-ready">Implemented</span></td>
              <td>MediaPipe FaceMesh 468-point 3D anatomical alignment</td>
              <td>Real-time in browser; 100% private with no microphone audio.</td>
            </tr>
            <tr>
              <td><strong>WordVideoNet 4-Word Benchmark</strong></td>
              <td><span className="ws-badge ws-badge-ready">Implemented</span></td>
              <td>3D-CNN + ResNet-18 · <strong>34.38% Top-1</strong> (62.50% Top-2)</td>
              <td>Multi-speaker GRID benchmark over 4 words (<code>bin</code>, <code>lay</code>, <code>place</code>, <code>set</code>).</td>
            </tr>
            <tr>
              <td><strong>Alphabet Letter Classifier</strong></td>
              <td><span className="ws-badge ws-badge-ready">Implemented</span></td>
              <td>CNN-BiLSTM (30 epochs) · <strong>12.00% Top-1</strong> (20.00% Best Val)</td>
              <td>26-class letter classification from 30-frame temporal windows.</td>
            </tr>
            <tr>
              <td><strong>Continuous Sentence CTC Decoder</strong></td>
              <td><span className="ws-badge ws-badge-ready">Benchmark Evaluated</span></td>
              <td>LipNet 3D-CNN + BiGRU + CTC Loss · <strong>38.89% WER</strong> (38.20% CER)</td>
              <td>Continuous sentence decoding evaluated on 15 test sentences.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
}
