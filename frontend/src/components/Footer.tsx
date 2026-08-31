import { API_BASE } from '../lib/api'

export function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <div className="footer-brand">
          <span className="footer-mark" aria-hidden="true" />
          <span>VISUAL SPEECH AI</span>
          <span className="footer-tag">Seeing speech beyond sound.</span>
        </div>
        <div className="footer-meta">
          <p>Research system — spoken alphabet decoding from visual speech.</p>
          <p className="mono-note">
            Conformer d_model 192 · gated fusion · CTC A–Z · speaker-independent splits
          </p>
          <p className="mono-note">Inference backend: {API_BASE} (optional)</p>
        </div>
      </div>
    </footer>
  )
}
