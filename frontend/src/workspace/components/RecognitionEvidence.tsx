import { useEffect, useState } from 'react'
import { API_BASE } from '../../lib/api'

interface Evidence {
  schema_version: number
  audio_inference: boolean
  isolated_word: { status: string; live_ready: boolean; vocabulary: string[]; reason: string; evaluation: { test_samples: number; top1_accuracy: number } | null }
  video_to_text: { status: string; live_ready: boolean; reason: string }
}

export function RecognitionEvidence() {
  const [evidence, setEvidence] = useState<Evidence | null>(null)
  const [failed, setFailed] = useState(false)
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
  return <section className="ws-card" aria-label="Word and transcription research availability">
    <div className="ws-card-head"><h2>Beyond the alphabet</h2><span className="ws-mono">RESEARCH STATUS</span></div>
    {!evidence ? <p>{failed ? 'Word and transcription evidence is unavailable. No readiness claim can be made.' : 'Checking word and transcription evidence…'}</p> : <>
      <p><strong>Whole words · {evidence.isolated_word.live_ready ? 'Available' : 'Research evaluation'}</strong><br />{evidence.isolated_word.reason}</p>
      {evidence.isolated_word.vocabulary.length > 0 && <p>Eight-word vocabulary: {evidence.isolated_word.vocabulary.join(', ')}.</p>}
      {evidence.isolated_word.evaluation && <p>Eight-word MIRACL pilot: {(evidence.isolated_word.evaluation.top1_accuracy * 100).toFixed(1)}% top-1 on {evidence.isolated_word.evaluation.test_samples} clips. This is not a webcam accuracy measurement.</p>}
      <p><strong>Free speech · {evidence.video_to_text.live_ready ? 'Available' : 'Unavailable'}</strong><br />{evidence.video_to_text.reason}</p>
    </>}
  </section>
}
