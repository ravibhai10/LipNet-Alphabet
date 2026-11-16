import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useWebcam } from '../../live-demo/useWebcam'
import { useLandmarker } from '../../live-demo/useLandmarker'
import { mouthAlignment } from '../../live-demo/mouthUtils'
import {
  COLLECTION_QUOTAS,
  EIGHT_WORDS,
  EXTRA_WORDS,
  collectionCounts,
  deleteRecording,
  digest,
  readRecordings,
  saveRecording,
  type CollectionSplit,
  type WordRecording
} from '../../live-demo/wordCollection'
import '../../styles/word-collection.css'

type Draft = Omit<WordRecording, 'frames_sha256' | 'label_confirmed'>
const sessionName = (split: CollectionSplit) => `${split}-${Date.now().toString(36)}`

export function WordCollectionPage() {
  const cam = useWebcam(), lm = useLandmarker()
  const [rows, setRows] = useState<WordRecording[]>([])
  const [customWords, setCustomWords] = useState<string[]>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('lipnet-custom-word-vocabulary-v1') || 'null')
      return Array.isArray(saved) ? saved.filter(word => typeof word === 'string') : ['help', 'yes', 'no', 'water', 'doctor', 'please', 'thanks']
    } catch { return ['help', 'yes', 'no', 'water', 'doctor', 'please', 'thanks'] }
  })
  const [label, setLabel] = useState<string>('help')
  const [newWordInput, setNewWordInput] = useState('')
  const [showAddWord, setShowAddWord] = useState(false)
  
  const [speaker, setSpeaker] = useState('participant01')
  const [split, setSplit] = useState<CollectionSplit>('train')
  const [session, setSession] = useState(() => sessionName('train'))
  const [consent, setConsent] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [phase, setPhase] = useState<'ready' | 'capturing' | 'review' | 'saving'>('ready')
  const [draft, setDraft] = useState<Draft | null>(null)
  const [frames, setFrames] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [quality, setQuality] = useState('Start camera to check face and lighting.')
  const [usable, setUsable] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  // Training state
  const [isTraining, setIsTraining] = useState(false)
  const [trainProgress, setTrainProgress] = useState<string | null>(null)
  const [trainSuccess, setTrainSuccess] = useState<string | null>(null)
  const [autoTrain, setAutoTrain] = useState(false)

  const current = useRef<Draft | null>(null)
  const recording = useRef(false)
  const mounted = useRef(true)
  const started = useRef(0)
  const assetHash = useRef('')
  const saving = useRef(false)
  const preview = useRef<HTMLCanvasElement>(null)
  const locked = phase !== 'ready'
  const live = cam.status === 'live'

  const allWords = Array.from(new Set([...EIGHT_WORDS, ...customWords]))

  const refresh = useCallback(async () => {
    const saved = await readRecordings()
    if (mounted.current) setRows(saved)
  }, [])

  useEffect(() => {
    mounted.current = true
    void refresh().catch(e => { if (mounted.current) setError(String(e)) })
    return () => {
      mounted.current = false
      recording.current = false
      current.current = null
    }
  }, [refresh])

  const discard = useCallback(() => {
    recording.current = false
    current.current = null
    setDraft(null)
    setFrames(0)
    setElapsed(0)
    setConfirmed(false)
    setPhase('ready')
  }, [])

  const stop = () => {
    cam.stop()
    discard()
    setUsable(false)
  }

  const start = async () => {
    if (cam.status === 'requesting' || live) return
    setError(null)
    try {
      if (!assetHash.current) {
        const response = await fetch('/models/face_landmarker.task')
        if (!response.ok) throw new Error('Landmark model is unavailable.')
        assetHash.current = await digest(await response.arrayBuffer())
      }
      if (!mounted.current) return
      const status = await cam.start()
      if (status === 'live' && mounted.current) await lm.load()
    } catch (e) {
      if (mounted.current) setError(e instanceof Error ? e.message : 'Camera setup failed.')
    }
  }

  useEffect(() => {
    if (!live || lm.status !== 'ready') return
    const crop = document.createElement('canvas')
    crop.width = crop.height = 96
    const ctx = crop.getContext('2d', { willReadFrequently: true })
    let raf = 0, last = 0, lastVideo = -1, sync = 0, disposed = false
    const fail = (message: string) => {
      recording.current = false
      current.current = null
      setPhase('ready')
      setFrames(0)
      setError(message)
    }
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      const video = cam.videoRef.current, tracker = lm.landmarkerRef.current
      if (!video || !tracker || !ctx || video.readyState < 2 || video.currentTime === lastVideo || now - last < 66) return
      last = now
      lastVideo = video.currentTime
      try {
        const faces = tracker.detectForVideo(video, now).faceLandmarks
        const a = faces.length === 1 ? mouthAlignment(faces[0], video.videoWidth, video.videoHeight) : null
        let okay = !!a
        let message = faces.length > 1
          ? 'Keep only one face in view.'
          : !a
            ? 'Center your face and keep your mouth visible.'
            : 'Ready. Look toward the camera and say only the prompted word.'
        if (a) {
          ctx.save()
          ctx.clearRect(0, 0, 96, 96)
          ctx.translate(48, 48)
          ctx.scale(96 / a.size, 96 / a.size)
          ctx.rotate(-a.angle)
          ctx.drawImage(video, -a.cx, -a.cy)
          ctx.restore()
          const pixels = ctx.getImageData(0, 0, 96, 96).data
          let sum = 0, squared = 0
          for (let i = 0; i < pixels.length; i += 4) {
            const v = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3
            sum += v
            squared += v * v
          }
          const mean = sum / (96 * 96), contrast = Math.sqrt(Math.max(0, squared / (96 * 96) - mean * mean))
          if (mean < 15 || mean > 240 || contrast < 8) {
            okay = false
            message = 'Improve front lighting; avoid glare or a dark mouth crop.'
          }
          if (preview.current) preview.current.getContext('2d')?.drawImage(crop, 0, 0, 96, 96)
        }
        if (recording.current) {
          const value = current.current!
          if (!okay) { fail('Face or lighting quality was lost. This incomplete clip was discarded; retry.'); return }
          if (now - started.current > 5000 || value.frames.length >= 76) { fail('Capture exceeded five seconds or 76 frames. Retry one word.'); return }
          value.frames.push(crop.toDataURL('image/jpeg', .75))
          value.timestamps_ms.push(now - started.current)
          value.duration_ms = now - started.current
        }
        if (now - sync > 100) {
          sync = now
          setUsable(okay)
          setQuality(message)
          if (recording.current) {
            setFrames(current.current!.frames.length)
            setElapsed(now - started.current)
          }
        }
      } catch (e) {
        if (!disposed) {
          setUsable(false)
          if (recording.current) fail('Tracking failed. Stop the camera and retry.')
          else setError(e instanceof Error ? e.message : 'Tracking failed.')
        }
      }
    }
    raf = requestAnimationFrame(loop)
    return () => { disposed = true; cancelAnimationFrame(raf) }
  }, [live, lm.status, cam.videoRef, lm.landmarkerRef])

  useEffect(() => {
    if (cam.status === 'error' || cam.status === 'denied') { discard(); setUsable(false) }
  }, [cam.status, discard])

  useEffect(() => {
    const timer = setInterval(() => {
      if (recording.current && performance.now() - started.current > 5000) {
        discard()
        setError('Video stalled or capture timed out. Retry a single word.')
      }
    }, 100)
    return () => clearInterval(timer)
  }, [discard])

  const handleAddCustomWord = () => {
    const cleaned = newWordInput.trim().toLowerCase()
    if (!/^[a-z]{2,20}$/.test(cleaned)) {
      setError('Please enter a single word using 2–20 letters without spaces or numbers.')
      return
    }
    if (!customWords.includes(cleaned)) {
      setCustomWords(prev => {
        const next = [...prev, cleaned]
        localStorage.setItem('lipnet-custom-word-vocabulary-v1', JSON.stringify(next))
        return next
      })
    }
    setLabel(cleaned)
    setNewWordInput('')
    setShowAddWord(false)
    setNotice(`Word "${cleaned}" added! You can now record clips for it.`)
  }

  const begin = () => {
    if (locked || recording.current || !usable || !consent) return
    if (!/^[a-zA-Z0-9_-]{1,80}$/.test(speaker) || !/^[a-zA-Z0-9_-]{1,80}$/.test(session)) {
      setError('Participant and session IDs must use letters, numbers, underscores or hyphens.')
      return
    }
    if (rows.some(r => r.speaker_id === speaker && r.session_id === session && r.split !== split)) {
      setError('Use a new session ID for each dataset partition.')
      return
    }
    current.current = {
      id: crypto.randomUUID(),
      speaker_id: speaker,
      session_id: session,
      split,
      label,
      recorded_at: new Date().toISOString(),
      frames: [],
      timestamps_ms: [],
      duration_ms: 0,
      preprocessing: 'browser-eyeaxis96-jpeg-v1',
      landmarker_sha256: assetHash.current,
      source: 'user_camera',
      consent: true
    }
    started.current = performance.now()
    recording.current = true
    setFrames(0)
    setElapsed(0)
    setNotice(null)
    setError(null)
    setPhase('capturing')
  }

  const finish = () => {
    if (!recording.current) return
    recording.current = false
    const value = current.current
    current.current = null
    if (!value || value.frames.length < 8) {
      setPhase('ready')
      setError('At least eight mouth frames are required. Retry a complete word.')
      return
    }
    setFrames(value.frames.length)
    setElapsed(value.duration_ms)
    setDraft(value)
    setConfirmed(false)
    setPhase('review')
  }

  const save = async () => {
    if (!draft || !confirmed || phase !== 'review' || saving.current) return
    saving.current = true
    setPhase('saving')
    setError(null)
    const savedLabel = draft.label
    try {
      await saveRecording({
        ...draft,
        label_confirmed: true,
        frames_sha256: await digest(new TextEncoder().encode(JSON.stringify(draft.frames)))
      })
      if (!mounted.current) return
      await refresh()
      if (!mounted.current) return
      discard()

      if (autoTrain) {
        setNotice(`Saved clip for "${savedLabel}". ⚡ Saving recordings for validation...`)
        try {
          await trainPersonalModel()
        } catch (trainErr) {
          console.warn("Auto-train warning:", trainErr)
        }
      } else {
        setNotice(`Saved clip for "${savedLabel}" locally. Record more samples or click "Train Model" below!`)
      }
    } catch (e) {
      if (mounted.current) {
        setPhase('review')
        setError(e instanceof Error ? e.message : 'Could not save recording.')
      }
    } finally {
      saving.current = false
    }
  }

  const exportData = async () => {
    try {
      const clips = await readRecordings()
      if (!clips.length) throw new Error('Record and save at least one clip first.')
      if (!mounted.current) return
      const blob = new Blob([JSON.stringify({
        format: 'lipnet-word-collection-v1',
        exported_at: new Date().toISOString(),
        protocol: 'personal_cross_session',
        vocabulary: allWords,
        clips
      })], { type: 'application/json' })
      const url = URL.createObjectURL(blob), a = document.createElement('a')
      a.href = url
      a.download = `lipnet-words-${new Date().toISOString().slice(0, 10)}-${Date.now()}.json`
      a.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
      setNotice('Export downloaded. Keep the JSON file for your records.')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Export failed.')
    }
  }

  const trainPersonalModel = async () => {
    try {
      setIsTraining(true)
      setTrainProgress("Gathering all recorded clips from local storage...")
      setTrainSuccess(null)
      setError(null)
      const clips = await readRecordings()
      if (!clips.length) {
        throw new Error("Please record and save at least one clip (e.g. for 'help') before training.")
      }
      setTrainProgress("Validating and saving recordings for the separate training workflow...")
      const res = await fetch("/api/word/train-personal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speaker_id: speaker,
          clips,
          vocabulary: allWords
        })
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || "Training failed on backend.")
      }
      const data = await res.json()
      setTrainProgress(null)
      setTrainSuccess(data.message || 'Recordings saved. Training and held-out evaluation are still required.')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setTrainProgress(null)
    } finally {
      setIsTraining(false)
    }
  }

  const counts = collectionCounts(rows.filter(r => r.speaker_id === speaker), split, allWords)

  return (
    <div className="word-collection">
      <header className="ws-page-header">
        <div>
          <p className="ws-kicker">YOUR CAMERA / CUSTOM VOCABULARY</p>
          <h1 className="ws-title">Teach it <em>your words.</em></h1>
          <p className="ws-lede">Collect genuine examples for whole-word recognition. Record extra words like <strong>help</strong>, save them, and train the model in one click.</p>
        </div>
        <Link className="ws-inline-link" to="/demo/live">Open Live Studio →</Link>
      </header>

      {trainSuccess && (
        <div style={{ padding: '16px 20px', background: 'rgba(16, 185, 129, 0.12)', border: '1px solid #10b981', borderRadius: '10px', color: '#10b981', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <strong style={{ fontSize: '15px' }}>Model Updated Successfully!</strong>
            <p style={{ margin: '4px 0 0 0', opacity: 0.9, fontSize: '13px' }}>{trainSuccess}</p>
          </div>
          <Link to="/demo/live" className="ws-btn ws-btn-primary ws-btn-sm" style={{ fontWeight: 600 }}>
            Test New Words in Live Lip Reading →
          </Link>
        </div>
      )}

      {trainProgress && (
        <div style={{ padding: '14px 18px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', borderRadius: '8px', color: '#60a5fa', marginBottom: '20px', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="ws-spinner" /> {trainProgress}
        </div>
      )}

      <div className="wc-layout">
        <section className="ws-card">
          <div className="wc-fields">
            <label>Participant ID
              <input disabled={locked || live} value={speaker} onChange={e => setSpeaker(e.target.value)} maxLength={80} />
            </label>
            <label>Dataset partition
              <select disabled={locked || live} value={split} onChange={e => { const s = e.target.value as CollectionSplit; setSplit(s); setSession(sessionName(s)) }}>
                <option value="train">Training</option>
                <option value="valid">Validation</option>
                <option value="test">Test — keep held out</option>
              </select>
            </label>
            <label>Recording session ID
              <input disabled={locked || live} value={session} onChange={e => setSession(e.target.value)} maxLength={80} />
            </label>
          </div>
          <p className="wc-hint">Stop the camera to change partitions. Keep lighting and distance natural.</p>
          <label className="wc-consent">
            <input type="checkbox" checked={consent} disabled={locked} onChange={e => setConsent(e.target.checked)} />
            I consent to recording my mouth video locally for this project and will speak each prompted word.
          </label>
          <div className="wc-video">
            <video ref={cam.videoRef} muted playsInline aria-label="Collection camera preview" />
            {!live && <p>{cam.status === 'requesting' ? 'Allow camera access…' : 'Camera off · no microphone'}</p>}
          </div>
          <div className="wc-actions">
            {!live ? (
              <button className="ws-btn ws-btn-primary" disabled={cam.status === 'requesting' || !consent} onClick={() => void start()}>
                Start camera
              </button>
            ) : (
              <button className="ws-btn ws-btn-ghost" onClick={stop}>Stop camera</button>
            )}
            {cam.status === 'requesting' && <button className="ws-btn ws-btn-ghost" onClick={stop}>Cancel</button>}
            <span role="status">{live ? quality : 'Camera permission is requested only on Start.'}</span>
          </div>
          {cam.error && <p className="wc-error" role="alert">{cam.error}</p>}
          {lm.error && <p className="wc-error" role="alert">{lm.error}</p>}
        </section>

        <section className="ws-card wc-prompt">
          <span className="ws-kicker">SAY THIS COMPLETE WORD</span>
          <h2 style={{ fontSize: '32px', color: '#10b981', margin: '8px 0 16px 0' }}>{label}</h2>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px' }}>Select Word to Record</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <select
                aria-label="Word to record"
                disabled={locked}
                value={label}
                onChange={e => setLabel(e.target.value)}
                style={{ flex: 1, padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--ws-border)', background: 'var(--ws-card-bg, #111)', color: 'inherit' }}
              >
                <optgroup label="⭐ High Priority Words">
                  <option value="help">help (User Requested ⭐)</option>
                  <option value="yes">yes</option>
                  <option value="no">no</option>
                  <option value="water">water</option>
                  <option value="doctor">doctor</option>
                  <option value="please">please</option>
                  <option value="thanks">thanks</option>
                </optgroup>
                <optgroup label="Standard 8-Word Baseline">
                  {EIGHT_WORDS.map(w => <option key={w} value={w}>{w}</option>)}
                </optgroup>
                {customWords.filter(w => !EXTRA_WORDS.includes(w as any) && !EIGHT_WORDS.includes(w as any)).length > 0 && (
                  <optgroup label="Custom Words">
                    {customWords.filter(w => !EXTRA_WORDS.includes(w as any) && !EIGHT_WORDS.includes(w as any)).map(w => (
                      <option key={w} value={w}>{w}</option>
                    ))}
                  </optgroup>
                )}
              </select>
              <button
                type="button"
                className="ws-btn ws-btn-ghost ws-btn-sm"
                onClick={() => setShowAddWord(prev => !prev)}
                title="Add any custom word to record"
              >
                + New Word
              </button>
            </div>

            {showAddWord && (
              <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                <input
                  type="text"
                  placeholder="Type word (e.g. food, alert)..."
                  value={newWordInput}
                  onChange={e => setNewWordInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleAddCustomWord() }}
                  style={{ flex: 1, padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--ws-border)', background: 'var(--ws-card-bg, #111)', color: 'inherit', fontSize: '13px' }}
                />
                <button type="button" className="ws-btn ws-btn-primary ws-btn-sm" onClick={handleAddCustomWord}>
                  Add
                </button>
              </div>
            )}
          </div>

          <canvas ref={preview} width={96} height={96} aria-label="Measured aligned mouth crop" style={{ borderRadius: '8px', border: '1px solid var(--ws-border)' }} />
          <div className="wc-capture-status" aria-live="polite">
            {phase === 'capturing' ? 'Recording your word. Click Finish Word when done.' : phase === 'review' ? 'Review the clip and confirm its label.' : phase === 'saving' ? 'Saving to this browser…' : 'Ready for a new labelled example.'}
          </div>
          <p style={{ margin: '8px 0 4px 0', fontSize: '13px', opacity: 0.8 }}>{frames} / 76 frames · {(elapsed / 1000).toFixed(1)} seconds</p>
          <progress value={frames} max={76} aria-label="Recording buffer usage" style={{ width: '100%', height: '8px' }} />

          <div className="wc-actions" style={{ marginTop: '16px' }}>
            {phase === 'ready' && (
              <button className="ws-btn ws-btn-primary" disabled={!live || !usable || !consent || lm.status !== 'ready'} onClick={begin}>
                Record Word
              </button>
            )}
            {phase === 'capturing' && (
              <button className="ws-btn ws-btn-primary" disabled={frames < 8} onClick={finish}>
                Finish Word
              </button>
            )}
            {draft && (
              <>
                <div className="wc-review">
                  {draft.frames.filter((_, i) => i % Math.max(1, Math.floor(draft.frames.length / 6)) === 0).slice(0, 6).map((f, i) => (
                    <img key={i} src={f} alt={`Recorded mouth frame ${i + 1}`} />
                  ))}
                </div>
                <label>
                  <input type="checkbox" checked={confirmed} disabled={phase === 'saving'} onChange={e => setConfirmed(e.target.checked)} />
                  I said “{draft.label}” correctly in this clip.
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', margin: '4px 0 10px 0', color: '#10b981', cursor: 'pointer' }}>
                  <input type="checkbox" checked={autoTrain} disabled={phase === 'saving'} onChange={e => setAutoTrain(e.target.checked)} />
                  ⚡ Save a validated backend bundle after each recording
                </label>
                <button className="ws-btn ws-btn-primary" disabled={!confirmed || phase === 'saving'} onClick={() => void save()}>
                  Save labelled clip
                </button>
                <button className="ws-btn ws-btn-ghost" disabled={phase === 'saving'} onClick={discard}>
                  Discard and retry
                </button>
              </>
            )}
          </div>
        </section>
      </div>

      {error && <p className="wc-error" role="alert" style={{ marginTop: '16px' }}>{error}</p>}
      {notice && <p className="wc-notice" role="status" style={{ marginTop: '16px' }}>{notice}</p>}

      {/* Progress & 1-Click Retraining Section */}
      <section className="ws-card" style={{ marginTop: '24px' }}>
        <div className="ws-card-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ margin: 0 }}>{split} Progress · {rows.length} Total Clips Saved</h2>
            <p style={{ margin: '4px 0 0 0', opacity: 0.75, fontSize: '13px' }}>Record clips for each word, then save a validated bundle for training and evaluation.</p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="ws-btn ws-btn-primary"
              disabled={!rows.length || isTraining}
              onClick={() => void trainPersonalModel()}
              style={{ background: '#10b981', borderColor: '#10b981', color: '#fff', fontWeight: 600 }}
            >
              ⚡ {isTraining ? 'Saving Recordings...' : 'Save Recordings for Training'}
            </button>
            <button className="ws-btn ws-btn-ghost" disabled={!rows.length || locked} onClick={() => void exportData()}>
              Export JSON Bundle
            </button>
          </div>
        </div>

        <div className="wc-counts" style={{ marginTop: '16px' }}>
          {allWords.map(w => (
            <button key={w} disabled={locked} onClick={() => setLabel(w)} aria-pressed={label === w}>
              <strong>{w}</strong>
              <span>{counts[w] || 0} / {COLLECTION_QUOTAS[split] || 15}</span>
            </button>
          ))}
        </div>
        <p style={{ fontSize: '13px', opacity: 0.7, marginTop: '12px' }}>
          Click any word card above to select it for recording. Once you have saved recordings, click <strong>"Save Recordings for Training"</strong> to save a bundle. Train and evaluate it before selecting a new model.
        </p>

        <details style={{ marginTop: '16px' }}>
          <summary style={{ cursor: 'pointer', opacity: 0.85 }}>Saved recordings list ({rows.length})</summary>
          <ul className="wc-saved" style={{ marginTop: '10px' }}>
            {[...rows].reverse().map(r => (
              <li key={r.id}>
                <span><strong>{r.label}</strong> · {r.split} · {r.speaker_id}/{r.session_id} · {r.frames.length} frames</span>
                <button disabled={locked} onClick={() => {
                  if (window.confirm(`Delete recording for "${r.label}"?`)) {
                    void deleteRecording(r.id).then(refresh).catch(e => setError(String(e)))
                  }
                }}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </details>
      </section>
    </div>
  )
}
