import { useEffect, useState } from 'react'
import { API_BASE } from '../../lib/api'
import { Icon } from '../Icon'

interface Evidence {
  schema_version: number
  audio_inference: boolean
  isolated_word: { status: string; live_ready: boolean; vocabulary: string[]; reason: string; evaluation: { test_samples: number; top1_accuracy: number } | null }
  video_to_text: { status: string; live_ready: boolean; reason: string }
}

export function RecognitionEvidence() {
  const [evidence, setEvidence] = useState<Evidence | null>(null)
  const [failed, setFailed] = useState(false)
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    let disposed = false
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 8000)
    fetch(`${API_BASE}/api/recognition/capabilities`, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Unavailable'); return response.json() as Promise<Evidence> })
      .then(value => {
        if (value.schema_version !== 1 || value.audio_inference !== false || !Array.isArray(value.isolated_word?.vocabulary)) throw new Error('Unsupported response')
        if (!controller.signal.aborted) setEvidence(value)
      })
      .catch(() => { if (!disposed) { setEvidence(null); setFailed(true) } })
      .finally(() => clearTimeout(timer))
    return () => { disposed = true; clearTimeout(timer); controller.abort() }
  }, [])

  return (
    <section className="ws-card ws-recognition-evidence" aria-label="Word and transcription research availability">
      <div className="ws-card-head" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setIsOpen(v => !v)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Icon name="activity" size={15} />
          <h2>Beyond the Alphabet · Capability Matrix</h2>
          <span style={{
            fontSize: '10px',
            fontFamily: 'var(--font-mono)',
            padding: '2px 8px',
            borderRadius: '6px',
            background: evidence?.isolated_word.live_ready ? 'rgba(16, 185, 129, 0.15)' : 'rgba(99, 102, 241, 0.15)',
            border: `1px solid ${evidence?.isolated_word.live_ready ? 'rgba(16, 185, 129, 0.4)' : 'rgba(99, 102, 241, 0.3)'}`,
            color: evidence?.isolated_word.live_ready ? '#34d399' : '#a5b4fc'
          }}>
            {evidence ? (evidence.isolated_word.live_ready ? 'WORD ENGINE ACTIVE' : 'RESEARCH EVALUATION') : 'SYNCING'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="ws-mono">{isOpen ? 'COLLAPSE' : 'EXPAND SPEC'}</span>
          <span style={{ transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease', color: 'var(--ws-text-muted)' }}>▾</span>
        </div>
      </div>

      {!evidence ? (
        <p style={{ margin: '8px 0 0', fontSize: '12px', color: 'var(--ws-text-muted)' }}>
          {failed ? 'Word and transcription evidence is unavailable. Offline fallback active.' : 'Verifying visual speech model checkpoint readiness…'}
        </p>
      ) : (
        <div style={{ display: isOpen ? 'grid' : 'none', gap: '10px', marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(99, 102, 241, 0.15)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', fontSize: '12px' }}>
            <div>
              <strong style={{ color: '#ffffff' }}>Isolated Word Classification:</strong>
              <p style={{ margin: '3px 0 0', color: '#94a3b8' }}>{evidence.isolated_word.reason}</p>
            </div>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '4px',
              background: evidence.isolated_word.live_ready ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
              color: evidence.isolated_word.live_ready ? '#34d399' : '#fbbf24',
              whiteSpace: 'nowrap'
            }}>
              {evidence.isolated_word.live_ready ? '● Online Ready' : '○ Evaluation Track'}
            </span>
          </div>

          {evidence.isolated_word.vocabulary.length > 0 && (
            <div style={{ fontSize: '12px' }}>
              <span style={{ color: '#94a3b8' }}>Eight-word vocabulary: </span>
              <strong style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>{evidence.isolated_word.vocabulary.join(', ')}</strong>
            </div>
          )}

          {evidence.isolated_word.evaluation && (
            <div style={{ fontSize: '11.5px', color: '#94a3b8', background: 'rgba(255, 255, 255, 0.02)', padding: '8px 12px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
              ⚡ <strong>AV-HuBERT Benchmark:</strong> {(evidence.isolated_word.evaluation.top1_accuracy * 100).toFixed(1)}% top-1 on {evidence.isolated_word.evaluation.test_samples} MIRACL clips. (Unseen webcam variation strictly unmeasured).
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', fontSize: '12px' }}>
            <div>
              <strong style={{ color: '#ffffff' }}>Continuous Free Speech (LipNet V2T):</strong>
              <p style={{ margin: '3px 0 0', color: '#94a3b8' }}>{evidence.video_to_text.reason}</p>
            </div>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '4px',
              background: 'rgba(148, 163, 184, 0.1)',
              color: '#94a3b8',
              whiteSpace: 'nowrap'
            }}>
              {evidence.video_to_text.live_ready ? '● Online' : '○ Unsupported'}
            </span>
          </div>
        </div>
      )}
    </section>
  )
}
