import { useState } from 'react'
import { Link } from 'react-router-dom'
import { API_BASE } from '../../lib/api'
import { Icon } from '../Icon'

export function DocumentationPage() {
  const [copied, setCopied] = useState(false)

  const deliverables = [
    { name: '12_WhiteBook.pdf', label: 'Approved Project Report', type: 'PDF Document', status: 'Guide Approved', size: '2.4 MB' },
    { name: '12_ESE_Presentation.pptx', label: 'End Semester Viva Slide Deck', type: 'PowerPoint PPTX', status: 'Verified', size: '4.8 MB' },
    { name: '12_ResearchPaper.pdf', label: 'Research Paper Manuscript', type: 'IEEE Format PDF', status: 'Guide Approved', size: '1.2 MB' },
    { name: '12_PlagiarismReport.pdf', label: 'Plagiarism & Originality Clearance', type: 'PDF Report', status: 'Verified < 10%', size: '850 KB' },
    { name: '12_TechnicalManual.pdf', label: 'Technical & Architecture Manual', type: 'PDF Document', status: 'Verified', size: '1.9 MB' },
    { name: '12_UserManual.pdf', label: 'User & Live Demo Manual', type: 'PDF Document', status: 'Verified', size: '1.5 MB' },
    { name: '12_ProjectPoster.jpg', label: 'Mandatory A4 Project Poster', type: 'High-Res JPG', status: 'A4 Mandated', size: '3.1 MB' },
    { name: '12_Technical_Results_and_Parameter_Analysis_Report.pdf', label: 'Empirical Parameter Analysis', type: '6-Page PDF Report', status: 'Verified', size: '1.8 MB' },
  ]

  const citationText = `@article{lipnet_ai_2026,
  title={LipNet-AI: Multi-Dataset Visual Speech Recognition and Viseme-Independent Word Decoding from Silent Video},
  author={Group 12 Engineering Project Team},
  institution={Department of Computer Engineering, Somaiya Trust},
  year={2026},
  note={Capstone Final Year Engineering Submission}
}`

  const copyCitation = () => {
    try {
      navigator.clipboard.writeText(citationText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      /* Fallback */
    }
  }

  return (
    <div className="ws-page-inner">
      <header className="ws-page-header">
        <p className="ws-kicker">07 / DOCUMENTATION & CAPSTONE REPOSITORY</p>
        <h1 className="ws-title">Architecture, manuals,<br /><em>and academic deliverables.</em></h1>
        <p className="ws-lede">Approved capstone submission package, system operating guidelines, and technical verification reports for Group 12.</p>
      </header>

      {/* College Capstone Submission Package Section */}
      <section className="ws-card" style={{ marginBottom: '24px' }}>
        <div className="ws-card-head">
          <div>
            <p className="ws-kicker">Official College Submission Package</p>
            <h2>Group 12 Final Capstone Deliverables</h2>
          </div>
          <span className="ws-badge ws-badge-ready">Deadline: 31 October 2026 · Ready for Viva</span>
        </div>

        <p className="ws-note" style={{ marginBottom: '16px' }}>
          All deliverables follow official college templates and naming conventions. Stored in <code>submission_package/</code> at project root.
        </p>

        <div className="ws-table-wrap">
          <table className="ws-table">
            <thead>
              <tr>
                <th>Deliverable File</th>
                <th>Description</th>
                <th>Format</th>
                <th>File Size</th>
                <th>Review Status</th>
              </tr>
            </thead>
            <tbody>
              {deliverables.map(item => (
                <tr key={item.name}>
                  <td><strong className="ws-code">{item.name}</strong></td>
                  <td>{item.label}</td>
                  <td><span className="ws-badge ws-badge-muted">{item.type}</span></td>
                  <td>{item.size}</td>
                  <td><span className="ws-badge ws-badge-ready">{item.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* IEEE Citation & Attribution */}
      <section className="ws-card" style={{ marginBottom: '24px' }}>
        <div className="ws-card-head">
          <div>
            <p className="ws-kicker">Academic Citation & Attribution</p>
            <h2>Cite This Work</h2>
          </div>
          <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={copyCitation}>
            <Icon name={copied ? 'check' : 'copy'} size={14} />
            {copied ? 'Citation Copied!' : 'Copy BibTeX'}
          </button>
        </div>
        <pre className="ws-code" style={{ padding: '16px', borderRadius: '8px', overflowX: 'auto', fontSize: '12px', background: 'var(--ws-bg)' }}>
          {citationText}
        </pre>
      </section>

      <div className="ws-grid-2">
        <section className="ws-card">
          <div className="ws-card-head">
            <h2>Start a live session</h2>
            <Icon name="camera"/>
          </div>
          <ol className="ws-guide-list">
            <li>Open Live lip reading and select Start camera. Only this action requests camera permission.</li>
            <li>Keep your face centered and well lit. The pipeline extracts real mouth landmarks and buffers a short sequence.</li>
            <li>When the backend model is ready, temporal crops are sent to your configured API for inference.</li>
            <li>Stop camera to release the device and clear the captured frame buffer.</li>
          </ol>
          <Link className="ws-btn ws-btn-primary" to="/demo/live">
            Open live studio<Icon name="arrow" size={16}/>
          </Link>
        </section>

        <section className="ws-card">
          <div className="ws-card-head">
            <h2>Readiness has layers</h2>
            <Icon name="layers"/>
          </div>
          <dl className="ws-detail-list">
            <div>
              <dt>Backend connected</dt>
              <dd>The health endpoint responded. Port 8001 is active with CPU inference services.</dd>
            </div>
            <div>
              <dt>AV-HuBERT 8-word model</dt>
              <dd>Supervised transfer head loaded and active on <code>/api/word/predict</code> (94.79% Top-1 accuracy).</dd>
            </div>
            <div>
              <dt>Alphabet classifier</dt>
              <dd>Experimental CNN-BiLSTM letter classification model loaded (12.00% test accuracy).</dd>
            </div>
            <div>
              <dt>Continuous Sentence Decoder</dt>
              <dd>Track B closed-vocabulary CTC model (WER 38.89%, CER 38.20% on 15 test sentences).</dd>
            </div>
          </dl>
        </section>

        <section className="ws-card">
          <h2>Camera troubleshooting</h2>
          <p className="ws-note">If permission is denied, enable camera access in your browser's site settings and retry. If the device is busy, close other camera applications. Camera APIs require localhost or HTTPS.</p>
          <p className="ws-note">MediaPipe initialization also needs its vision runtime. If loading fails, the page shows the error and keeps camera controls available.</p>
        </section>

        <section className="ws-card">
          <h2>Connection & privacy</h2>
          <p className="ws-note">Configured backend API:</p>
          <code className="ws-code">{API_BASE}</code>
          <p className="ws-note">Live mouth crops go to this API only when inference is available. The frontend does not save camera recordings. Session activity is held in memory and can be cleared.</p>
          <p className="ws-note">The pipeline is CPU compatible. Training and dataset preparation are separate actions; opening a page does not start them.</p>
        </section>

        <section className="ws-card">
          <h2>Speaking & stable results</h2>
          <p className="ws-note">Mouth movement starts decoding. A resting mouth pauses new requests. The live buffer holds up to 29 aligned mouth frames, and new windows are considered every four captured frames.</p>
          <p className="ws-note">Results appear after two consistent windows. Full model scores are smoothed to reduce flicker. These movement thresholds and webcam scores still need validation with labeled recordings.</p>
        </section>

        <section className="ws-card">
          <h2>Pretrained model heads</h2>
          <p className="ws-note">AV-HuBERT uses a supervised 8-word linear head evaluated on 192 held-out clips with disjoint speakers.</p>
          <p className="ws-note">The training form trains the alphabet CNN-BiLSTM workflow. Checkpoints and evaluation sidecars are permanently stored in <code>runs/</code>.</p>
        </section>

        {/* Examiner Viva Defense & Theoretical Foundations Section */}
        <section className="ws-card ws-card-glow ws-hud-bracket" style={{ gridColumn: '1 / -1', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: '14px', padding: '26px' }}>
          <div className="ws-card-head" style={{ marginBottom: '16px' }}>
            <div>
              <p className="ws-kicker" style={{ color: '#10b981' }}>CAPSTONE VIVA CHEAT SHEET</p>
              <h2 style={{ fontSize: '22px', margin: '4px 0' }}>🎓 External Examiner Defense & Theoretical Foundations</h2>
            </div>
            <span className="ws-badge ws-badge-ready">Group 12 Verified</span>
          </div>

          <p className="ws-note" style={{ fontSize: '14px', lineHeight: 1.5, marginBottom: '20px' }}>
            Key theoretical concepts, mathematical formulations, and definitive answers prepared for the external project evaluation committee.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '24px' }}>
            <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <h3 style={{ fontSize: '15px', color: '#10b981', marginBottom: '8px' }}>1. The Viseme Ambiguity Phenomenon</h3>
              <p style={{ fontSize: '13px', opacity: 0.85, lineHeight: 1.5 }}>
                In visual speech, many distinct acoustic phonemes share identical visible articulator positions (homophenes):
              </p>
              <ul style={{ fontSize: '12px', opacity: 0.8, paddingLeft: '18px', marginTop: '6px', lineHeight: 1.6 }}>
                <li><strong>Bilabials {`/p, b, m/`}:</strong> Complete lip closure (identical contour).</li>
                <li><strong>Labiodentals {`/f, v/`}:</strong> Lower lip touches upper incisors.</li>
                <li><strong>Alveolars {`/t, d, n/`}:</strong> Occluded by teeth; identical visual exterior.</li>
              </ul>
              <p style={{ fontSize: '12px', color: '#f59e0b', marginTop: '8px' }}>
                👉 <em>Examiner Takeaway:</em> Explains why single-letter alphabet accuracy is ~12%, whereas whole words provide multi-syllable temporal trajectories reaching 94.8%.
              </p>
            </div>

            <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <h3 style={{ fontSize: '15px', color: '#38bdf8', marginBottom: '8px' }}>2. AV-HuBERT Conformer & Layer 9 Selection</h3>
              <p style={{ fontSize: '13px', opacity: 0.85, lineHeight: 1.5 }}>
                Facebook AV-HuBERT Base features 12 Transformer blocks with depthwise convolution.
              </p>
              <ul style={{ fontSize: '12px', opacity: 0.8, paddingLeft: '18px', marginTop: '6px', lineHeight: 1.6 }}>
                <li><strong>Early Layers (1–4):</strong> Low-level edges, pixel luminosity, lighting.</li>
                <li><strong>Layer 9 (Selected):</strong> Optimal balance of phonological visual abstractions without task overfitting.</li>
                <li><strong>Pooling:</strong> Mean + Standard Deviation concatenation (1536 dimensions).</li>
              </ul>
              <p style={{ fontSize: '12px', color: '#10b981', marginTop: '8px' }}>
                👉 <em>Examiner Takeaway:</em> Freezing 90M parameters preserves generalizability on modest academic datasets.
              </p>
            </div>

            <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <h3 style={{ fontSize: '15px', color: '#a855f7', marginBottom: '8px' }}>3. Entropy-Calibrated Softmax Rejection</h3>
              <p style={{ fontSize: '13px', opacity: 0.85, lineHeight: 1.5 }}>
                Instead of naive argmax, we reject non-target speech using:
              </p>
              <ul style={{ fontSize: '12px', opacity: 0.8, paddingLeft: '18px', marginTop: '6px', lineHeight: 1.6 }}>
                <li><strong>Confidence Threshold:</strong> {`P(top_1) >= 0.65`}</li>
                <li><strong>Softmax Margin:</strong> {`Δ = P(top_1) - P(top_2) >= 0.25`}</li>
                <li><strong>Shannon Entropy:</strong> {`H(P) = -Σ P_i ln(P_i) < 1.20 nats`}</li>
              </ul>
              <p style={{ fontSize: '12px', color: '#a855f7', marginTop: '8px' }}>
                👉 <em>Examiner Takeaway:</em> Achieved 0% false acceptance across 60 phrase negatives.
              </p>
            </div>
          </div>

          <details style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '14px 18px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#10b981', fontSize: '15px' }}>
              📚 Click to expand 5 Tough Viva Questions & Recommended Answers
            </summary>
            <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '13px', lineHeight: 1.5 }}>
              <div>
                <strong>Q1: What does your 94.79% 8-word accuracy represent?</strong>
                <p style={{ margin: '4px 0 0 0', opacity: 0.85 }}>
                  It measures top-1 classification on 192 test clips from 3 strictly held-out speakers (F10, F11, M08) who were never seen during training or validation, establishing speaker-independent performance.
                </p>
              </div>
              <div>
                <strong>Q2: Why is the letter 'W' missing in the GRID dataset?</strong>
                <p style={{ margin: '4px 0 0 0', opacity: 0.85 }}>
                  The GRID corpus grammar requires monosyllabic or disyllabic words for uniform rhythm. 'W' is trisyllabic ("double-u") and was intentionally omitted by Cooke et al. during speech corpus synthesis.
                </p>
              </div>
              <div>
                <strong>Q3: How does personal adaptation fit words like 'help' without retraining 90M parameters?</strong>
                <p style={{ margin: '4px 0 0 0', opacity: 0.85 }}>
                  We keep the 90M AV-HuBERT encoder frozen, compute the normalized prototype embedding for user-recorded clips in the 1536-dimensional feature space, and project it as a new class row into the linear classification matrix while retaining the base class weights.
                </p>
              </div>
              <div>
                <strong>Q4: Why does the system operate on 96×96 crops instead of full faces?</strong>
                <p style={{ margin: '4px 0 0 0', opacity: 0.85 }}>
                  Face tracking extracts eye-axis orientation to eliminate head rotation (roll tilt), then normalizes the mouth region to 96×96. This decouples mouth articulation from facial identity and background noise.
                </p>
              </div>
              <div>
                <strong>Q5: Can this system run real-time continuous sentence transcription?</strong>
                <p style={{ margin: '4px 0 0 0', opacity: 0.85 }}>
                  Not on commodity laptop CPUs. Continuous visual CTC decoding requires 25 FPS deep feature extraction, incurring ~2.5s latency. Isolated word recognition executes bounded 29-frame inferences with ~350ms turnaround time.
                </p>
              </div>
            </div>
          </details>
        </section>
      </div>
    </div>
  )
}
