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
      </div>
    </div>
  )
}
