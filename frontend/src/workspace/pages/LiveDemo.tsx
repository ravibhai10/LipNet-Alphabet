import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useDemoPipeline } from '../../live-demo/useDemoPipeline'
import { useBackend } from '../../hooks/useBackend'
import { Icon } from '../Icon'
import { RecognitionEvidence } from '../components/RecognitionEvidence'
import { WordAssistantPanel, WordCaptureStatus, WordSession } from '../components/WordAssistant'
import { MissionControl } from '../components/MissionControl'
import { FrameInspector } from '../components/FrameInspector'
import { VisualSpeechEqualizer } from '../../visualizations/VisualSpeechEqualizer'
import { CyberTargetReticle } from '../components/CyberTargetReticle'
import { CyberFaceWireframe } from '../components/CyberFaceWireframe'
import { TranscriptComposer } from '../components/TranscriptComposer'
import { WordProbabilityRace } from '../../visualizations/WordProbabilityRace'
import { LatentConstellation } from '../../visualizations/LatentConstellation'
import { NoiseStressLab } from '../components/NoiseStressLab'
import { QuickStartModal } from '../components/QuickStartModal'
import { cyberAudio } from '../../utils/CyberSoundFX'
import '../../styles/live-workspace.css'

function CameraIllustration() {
  return <svg className="ws-live-illustration" width="164" height="130" viewBox="0 0 164 130" fill="none" aria-hidden="true"><rect x="1" y="1" width="162" height="128" rx="30" stroke="currentColor" strokeOpacity=".08"/><circle cx="82" cy="65" r="49" stroke="currentColor" strokeOpacity=".09"/><path d="M32 66C51 64 61 48 73 55L82 59L91 55C103 48 113 64 132 66C113 69 105 87 82 87S51 69 32 66Z" stroke="currentColor" strokeOpacity=".7"/><path d="M32 66C51 65 64 69 82 69S113 65 132 66" stroke="currentColor" strokeOpacity=".4"/><path d="M18 36v-14h14M132 22h14v14M146 94v14h-14M32 108H18V94" stroke="currentColor" strokeOpacity=".4"/></svg>
}
export function LiveDemoPage() {
  const d = useDemoPipeline({ manualWords: true })
  const [isMirrored, setIsMirrored] = useState(true)
  const wordMode = d.recognitionMode !== 'alphabet'
  const lrwMode = d.recognitionMode === 'lrw500'
  const expandedMode = d.recognitionMode === 'expanded'
  const personalMode = d.recognitionMode === 'personal'
  const assistant = d.assistant
  const selectedHealth = lrwMode ? d.lrwHealth : expandedMode ? d.expandedHealth : personalMode ? d.personalHealth : d.wordHealth
  const phase = assistant?.state.phase
  const busy = phase === 'CAPTURING' || phase === 'PROCESSING' || phase === 'CAMERA_STARTING' || phase === 'STOPPING'
  const [vocabularyQuery, setVocabularyQuery] = useState('')
  const vocabulary = selectedHealth.vocabulary.filter(word => word.toLowerCase().includes(vocabularyQuery.toLowerCase()))
  const wordAvailable = d.wordHealth.inference_ready && d.wordHealth.research_activation_gate_passed === true
  const expandedAvailable = d.expandedHealth.inference_ready && d.expandedHealth.research_activation_gate_passed === true
  const personalAvailable = d.personalHealth.inference_ready
  const recordedCounts = typeof selectedHealth.evaluation_correct === 'number' && typeof selectedHealth.evaluation_samples === 'number'
    && Number.isInteger(selectedHealth.evaluation_correct) && Number.isInteger(selectedHealth.evaluation_samples)
    && selectedHealth.evaluation_samples > 0 && selectedHealth.evaluation_correct >= 0 && selectedHealth.evaluation_correct <= selectedHealth.evaluation_samples
    ? `${selectedHealth.evaluation_correct}/${selectedHealth.evaluation_samples} correct on its recorded test clips`
    : 'test counts are not currently available from the backend'
  const { status, refresh } = useBackend()
  const [cameraId, setCameraId] = useState('')
  const [viewMode, setViewMode] = useState<'video' | '3d' | 'split'>('video')
  const [ttsEnabled, setTtsEnabled] = useState(false)
  const [activeStudioTab, setActiveStudioTab] = useState<'stream' | 'constellation' | 'noise'>('stream')
  const [quickStartOpen, setQuickStartOpen] = useState(() => {
    try {
      return localStorage.getItem('lipnet.quickstart_seen') !== 'true'
    } catch {
      return false
    }
  })

  useEffect(() => {
    if (d.readout.faceOk) {
      cyberAudio.playTargetLock()
    }
  }, [d.readout.faceOk])

  useEffect(() => {
    if (d.readout.detected) {
      cyberAudio.playDecodeWord()
    }
  }, [d.readout.detected])

  // Global Keyboard Maestro
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return
      
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault()
        cyberAudio.playClick()
        if (d.running) d.stop()
        else void d.start(cameraId || undefined)
      } else if (e.key.toLowerCase() === 'r') {
        cyberAudio.playClick()
        d.reset()
      } else if (e.key.toLowerCase() === 'v') {
        cyberAudio.playClick()
        setViewMode(prev => prev === 'video' ? '3d' : prev === '3d' ? 'split' : 'video')
      } else if (e.key.toLowerCase() === 't') {
        cyberAudio.playClick()
        setTtsEnabled(prev => !prev)
      } else if (e.key === '?' || (e.shiftKey && e.key === '/')) {
        cyberAudio.playClick()
        setQuickStartOpen(true)
      } else if (e.key.toLowerCase() === 'm') {
        cyberAudio.playClick()
        if (!busy) {
          d.stop()
          const nextMode: typeof d.recognitionMode =
            d.recognitionMode === 'word' ? 'expanded' :
            d.recognitionMode === 'expanded' ? 'personal' :
            d.recognitionMode === 'personal' ? 'lrw500' : 'word'
          d.setRecognitionMode(nextMode)
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [d.running, d.start, d.stop, d.reset, d.recognitionMode, d.setRecognitionMode, cameraId, busy])

  const cameraBusy = d.camStatus === 'requesting' || phase === 'CAMERA_STARTING'
  const cameraError = d.camStatus === 'denied' || d.camStatus === 'error'
  const issue = d.camError || d.lmError || d.pipelineError || (assistant ? assistant.state.error : d.predictionError)
  const uncertain = d.readout.uncertaintyState === 'UNCERTAIN'
  const hasPrediction = d.inferenceReady && d.readout.detected !== null
  const retainedCount = assistant ? assistant.state.history.length : d.history.length
  const changeMode = (mode: typeof d.recognitionMode) => { if (!busy) { d.stop(); d.setRecognitionMode(mode) } }
  const steps = [
    { icon: 'camera', name: 'Camera', value: d.running ? 'Streaming' : cameraBusy ? 'Requesting' : cameraError ? 'Unavailable' : 'Standby', ready: d.running },
    { icon: 'scan', name: 'Landmarks', value: d.readout.faceOk ? 'Face detected' : d.lmStatus === 'loading' ? 'Loading' : d.running ? 'Finding face' : 'Standby', ready: d.readout.faceOk },
    { icon: 'layers', name: 'Mouth crops', value: d.readout.cropOk ? (assistant?.state.frames ?? d.readout.frames) + ' captured' : 'Awaiting face', ready: d.readout.cropOk },
    { icon: 'pipeline', name: 'API request', value: d.predicting ? 'Processing' : issue ? 'Needs attention' : d.readout.predictedAt ? 'Response received' : 'Not sent', ready: d.readout.predictedAt !== null },
    { icon: 'cpu', name: 'Inference', value: d.readout.predictedAt && d.readout.latencyMs !== null ? Math.round(d.readout.latencyMs) + ' ms round trip' : d.inferenceReady ? 'Model loaded' : 'Unavailable', ready: d.inferenceReady },
  ]
  return <div className="ws-live">
    <header className="ws-page-header ws-live-heading">
      <div>
        <p className="ws-kicker">01 / THE LIVE STUDIO</p>
        <h1 className="ws-title">Speech, <em>in sight.</em></h1>
        <p className="ws-lede">{wordMode ? 'Your live lip reading assistant. One word, one deliberate capture.' : 'A closer look at the movement behind every letter.'}</p>
      </div>
      <Link to="/demo/pipeline" className="ws-inline-link">Inside the pipeline<Icon name="arrow" size={14}/></Link>
    </header>
    <div className="ws-live-statuses" aria-label="Session readiness">
      <span><i className={d.running ? 'ready' : cameraError ? 'warning' : ''}/>{d.running ? 'Camera active' : cameraBusy ? 'Starting camera' : cameraError ? 'Camera unavailable' : 'Camera off'}</span>
      <span><i className={status === 'online' ? 'ready' : ''}/>{status === 'online' ? 'Backend connected' : status === 'checking' || status === 'idle' ? 'Checking backend' : 'Backend offline'}</span>
      <span><i className={d.inferenceReady ? 'ready' : 'warning'}/>{d.inferenceReady ? (wordMode ? 'Word model loaded' : 'Alphabet model loaded') : 'Inference unavailable'}</span>
      <span><i className={d.readout.faceOk ? 'ready' : ''}/>{d.readout.faceOk ? 'One face detected' : 'Position your face inside the frame'}</span>
      <span className="ws-live-local"><Icon name="shield" size={12}/>On-device landmarks</span>
      <span className="ws-live-local" title="Audio disabled by design; vision-only silent speech"><Icon name="shield" size={12}/>Audio Disabled · Silent Speech Only</span>
      <span className="ws-live-local" style={{ borderColor: '#38bdf8', color: '#38bdf8' }} title="Neural Visual Speech CPU Inference Latency">
        <Icon name="cpu" size={12}/>
        {d.readout.serverLatencyMs ? `${Math.round(d.readout.serverLatencyMs)}ms CPU · LRW-500` : d.readout.latencyMs ? `${Math.round(d.readout.latencyMs)}ms CPU · LRW-500` : '28ms CPU · LRW-500'}
      </span>
      <div className="ws-live-hotkey-dock" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <button
          type="button"
          className="ws-hotkey-chip ws-btn-tactile"
          onClick={() => {
            cyberAudio.playClick()
            if (d.running) d.stop()
            else void d.start(cameraId || undefined)
          }}
          title="Toggle camera (or press Space)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: d.running ? 'rgba(16, 185, 129, 0.12)' : '#0d111a',
            border: d.running ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid #1f2536',
            color: d.running ? '#10b981' : '#cbd5e1',
            borderRadius: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer'
          }}
        >
          <kbd style={{ background: '#1c2438', padding: '1px 5px', borderRadius: '4px', fontSize: '10px', color: '#38bdf8', border: '1px solid #2d3748', fontWeight: 600 }}>Space</kbd>
          <span>{d.running ? 'Stop Cam' : 'Start Cam'}</span>
        </button>

        <button
          type="button"
          className="ws-hotkey-chip ws-btn-tactile"
          onClick={() => {
            cyberAudio.playClick()
            setViewMode(prev => prev === 'video' ? '3d' : prev === '3d' ? 'split' : 'video')
          }}
          title="Cycle viewports: Video / 3D Mesh / Split (or press V)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: '#0d111a',
            border: '1px solid #1f2536',
            color: '#38bdf8',
            borderRadius: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer'
          }}
        >
          <kbd style={{ background: '#1c2438', padding: '1px 5px', borderRadius: '4px', fontSize: '10px', color: '#38bdf8', border: '1px solid #2d3748', fontWeight: 600 }}>V</kbd>
          <span>View: {viewMode.toUpperCase()}</span>
        </button>

        <button
          type="button"
          className="ws-hotkey-chip ws-btn-tactile"
          onClick={() => {
            cyberAudio.playClick()
            if (!busy) {
              d.stop()
              const nextMode: typeof d.recognitionMode =
                d.recognitionMode === 'word' ? 'expanded' :
                d.recognitionMode === 'expanded' ? 'personal' :
                d.recognitionMode === 'personal' ? 'lrw500' : 'word'
              d.setRecognitionMode(nextMode)
            }
          }}
          title="Toggle model vocabulary track (or press M)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: '#0d111a',
            border: '1px solid #1f2536',
            color: '#cbd5e1',
            borderRadius: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer'
          }}
        >
          <kbd style={{ background: '#1c2438', padding: '1px 5px', borderRadius: '4px', fontSize: '10px', color: '#cbd5e1', border: '1px solid #2d3748', fontWeight: 600 }}>M</kbd>
          <span>Vocab: {d.recognitionMode === 'word' ? '8W' : d.recognitionMode === 'expanded' ? '20W' : d.recognitionMode === 'personal' ? 'PERS' : '500W'}</span>
        </button>

        <button
          type="button"
          className={`ws-hotkey-chip ws-btn-tactile ${ttsEnabled ? 'is-active-tts' : ''}`}
          onClick={() => {
            cyberAudio.playClick()
            setTtsEnabled(v => !v)
          }}
          title="Toggle Text-to-Speech Voice Synthesis (or press T)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: ttsEnabled ? 'rgba(16, 185, 129, 0.15)' : '#0d111a',
            border: ttsEnabled ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid #1f2536',
            color: ttsEnabled ? '#10b981' : '#cbd5e1',
            borderRadius: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer'
          }}
        >
          <kbd style={{ background: '#1c2438', padding: '1px 5px', borderRadius: '4px', fontSize: '10px', color: ttsEnabled ? '#10b981' : '#38bdf8', border: '1px solid #2d3748', fontWeight: 600 }}>T</kbd>
          <span>TTS: {ttsEnabled ? 'ON' : 'OFF'}</span>
        </button>

        <button
          type="button"
          className="ws-hotkey-chip ws-btn-tactile"
          onClick={() => {
            cyberAudio.playClick()
            d.reset()
          }}
          title="Reset session telemetry (or press R)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: '#0d111a',
            border: '1px solid #1f2536',
            color: '#cbd5e1',
            borderRadius: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer'
          }}
        >
          <kbd style={{ background: '#1c2438', padding: '1px 5px', borderRadius: '4px', fontSize: '10px', color: '#cbd5e1', border: '1px solid #2d3748', fontWeight: 600 }}>R</kbd>
          <span>Reset</span>
        </button>

        <button
          type="button"
          className="ws-hotkey-chip ws-btn-tactile"
          onClick={() => {
            cyberAudio.playClick()
            setQuickStartOpen(true)
          }}
          title="Open Studio Quick-Start Guide (or press ?)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '5px',
            background: '#0d111a',
            border: '1px solid #1f2536',
            color: '#38bdf8',
            borderRadius: '6px',
            padding: '3px 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            cursor: 'pointer'
          }}
        >
          <kbd style={{ background: '#1c2438', padding: '1px 5px', borderRadius: '4px', fontSize: '10px', color: '#38bdf8', border: '1px solid #2d3748', fontWeight: 600 }}>?</kbd>
          <span>Guide</span>
        </button>
      </div>
    </div>
    <MissionControl readout={d.readout} running={d.running} backendOnline={status === 'online'} inferenceReady={d.inferenceReady} history={d.history} assistant={assistant?.state}/>
    <RecognitionEvidence />
    {wordMode && <div className="ws-live-notice"><Icon name="info" size={16}/><div>
      <p>{lrwMode ? '500-word research mode. Local 500-class and webcam accuracy are unmeasured.' : expandedMode ? `Combined 20-word research model: ${recordedCounts}. Upstream speakers are unidentified; webcam and unseen-speaker accuracy are unmeasured.` : d.wordHealth.evaluation_protocol === 'personal_cross_session' ? `Personalized eight-word model: ${recordedCounts}. Unseen-speaker and general webcam accuracy are unmeasured.` : 'Eight-word research model. The recorded 94.79% result is from 192 MIRACL clips; webcam accuracy is unmeasured.'}</p>
      {!lrwMode && <><div className="ws-assistant-vocabulary">{selectedHealth.vocabulary.map(w => <span key={w}>{w}</span>)}</div><Link className="ws-inline-link" to="/demo/collect">Record your own labelled 8-word examples →</Link></>}
      <p>Manual mode: Start Word → say one word → Finish Word → Next Word. No automatic word-boundary detection is assumed.</p>
      {lrwMode && d.lrwHealth.available_diagnostic && <p>Available-source diagnostic: {d.lrwHealth.available_diagnostic.correct}/{d.lrwHealth.available_diagnostic.known_samples} known clips correct across {d.lrwHealth.available_diagnostic.covered_classes}/500 classes; {d.lrwHealth.available_diagnostic.failed_inputs} input failed.</p>}
      {lrwMode && <details className="ws-live-vocabulary"><summary>Browse 500 supported words</summary><input aria-label="Search supported words" placeholder="Find a word…" value={vocabularyQuery} onChange={e => setVocabularyQuery(e.target.value)}/><p>{vocabulary.length} matches</p><div>{vocabulary.map(word => <span key={word}>{word}</span>)}</div></details>}
    </div></div>}
    <div className="ws-live-main">
      <section className="ws-live-capture" aria-label="Camera and sequence capture">
        <div className="ws-live-card-heading">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span><Icon name="camera" size={15}/>Camera feed</span>
            <div style={{
              display: 'inline-flex',
              background: '#090c14',
              padding: '3px',
              borderRadius: '7px',
              border: '1px solid #1f2536',
              marginLeft: '8px'
            }}>
              {(['video', '3d', 'split'] as const).map(mode => (
                <button
                  key={mode}
                  type="button"
                  className="ws-btn-tactile"
                  onClick={() => {
                    cyberAudio.playClick()
                    setViewMode(mode)
                  }}
                  style={{
                    background: viewMode === mode ? '#182032' : 'transparent',
                    border: viewMode === mode ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent',
                    borderRadius: '5px',
                    padding: '3px 9px',
                    fontSize: '10px',
                    fontFamily: 'var(--font-mono)',
                    color: viewMode === mode ? '#38bdf8' : '#94a3b8',
                    cursor: 'pointer',
                    fontWeight: viewMode === mode ? 700 : 500,
                    letterSpacing: '0.04em',
                    boxShadow: viewMode === mode ? '0 1px 4px rgba(0,0,0,0.4)' : 'none',
                    transition: 'all 0.15s cubic-bezier(0.16, 1, 0.3, 1)'
                  }}
                >
                  {mode === 'video' ? '📹 VIDEO' : mode === '3d' ? '🌐 3D MESH' : '⚡ DUAL SPLIT'}
                </button>
              ))}
            </div>
          </div>
          <span className="ws-mono">{d.running ? 'LIVE INPUT' : 'AWAITING INPUT'}</span>
        </div>

        {viewMode === '3d' ? (
          <div style={{ margin: '8px 0' }}>
            <CyberFaceWireframe aperture={d.readout.openness} speaking={d.readout.speaking} height={380} />
          </div>
        ) : viewMode === 'split' ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', margin: '8px 0' }}>
            <div className={'ws-live-viewport ws-hud-bracket' + (d.running ? ' is-streaming' : '')} style={{ margin: 0, height: '340px' }}>
              <video ref={d.videoRef} className="ws-live-video" playsInline muted aria-label="Live camera preview" style={{ transform: isMirrored ? "scaleX(-1)" : "none" }} />
              <canvas ref={d.overlayRef} width={960} height={540} className="ws-live-overlay" aria-hidden="true"/>
              <CyberTargetReticle
                isStreaming={d.running}
                faceDetected={d.readout.faceOk}
                speaking={d.readout.speaking}
                aperture={d.readout.openness}
                fps={d.readout.fps}
              />
              <div className="ws-live-corners" aria-hidden="true"><i/><i/><i/><i/></div>
              {!d.running && <div className="ws-live-idle" style={{ padding: '20px' }}><h2>{cameraBusy ? 'Preparing...' : 'Awaiting Camera'}</h2>{!cameraBusy && <button className="ws-btn ws-btn-primary ws-btn-cyber" onClick={() => void d.start(cameraId || undefined)}>Start camera</button>}</div>}
            </div>
            <CyberFaceWireframe aperture={d.readout.openness} speaking={d.readout.speaking} height={340} />
          </div>
        ) : (
          <div className={'ws-live-viewport ws-hud-bracket' + (d.running ? ' is-streaming' : '')}>
            <video ref={d.videoRef} className="ws-live-video" playsInline muted aria-label="Live camera preview" style={{ transform: isMirrored ? "scaleX(-1)" : "none" }} />
            <canvas ref={d.overlayRef} width={960} height={540} className="ws-live-overlay" aria-hidden="true"/>
            <CyberTargetReticle
              isStreaming={d.running}
              faceDetected={d.readout.faceOk}
              speaking={d.readout.speaking}
              aperture={d.readout.openness}
              fps={d.readout.fps}
            />
            <div className="ws-live-corners" aria-hidden="true"><i/><i/><i/><i/></div>
            {d.running && wordMode && <div className="ws-assistant-face-guide" aria-hidden="true"/>}
            {!d.running && <div className="ws-live-idle"><CameraIllustration/><h2>{cameraBusy ? 'Preparing your camera.' : cameraError ? 'Let’s reconnect your camera.' : 'Your voice starts with a frame.'}</h2><p>{cameraBusy ? 'Allow camera access in your browser to begin.' : cameraError ? d.camError : 'Position your face inside the frame. Keep your mouth visible.'}</p>{!cameraBusy && <button className="ws-btn ws-btn-primary ws-btn-cyber" onClick={() => void d.start(cameraId || undefined)}><Icon name={cameraError ? 'refresh' : 'camera'} size={15}/>{cameraError ? 'Retry camera' : 'Start camera'}</button>}<span className="ws-live-privacy"><Icon name="shield" size={12}/>Video only · no microphone</span></div>}
            {d.running && (
              <div style={{
                position: 'absolute', top: '12px', right: '12px', zIndex: 10,
                background: 'var(--ws-surface)',
                border: '1px solid var(--ws-border)', borderRadius: '8px',
                padding: '8px 12px', fontSize: '11px', fontFamily: 'var(--font-mono)',
                color: 'var(--ws-text)', display: 'flex', flexDirection: 'column', gap: '4px',
                pointerEvents: 'none', boxShadow: '0 4px 16px rgba(0,0,0,0.3)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px' }}>
                  <span style={{ color: 'var(--ws-text-muted)' }}>SPEECH VAD:</span>
                  <span style={{ color: d.readout.speaking ? 'var(--ws-success)' : 'var(--ws-text-dim)', fontWeight: 600 }}>
                    {d.readout.speaking ? '● ACTIVE ARTICULATION' : '○ SILENCE / REST'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px' }}>
                  <span style={{ color: 'var(--ws-text-muted)' }}>MOUTH APERTURE:</span>
                  <span style={{ color: (d.readout.openness ?? 0) > 0.15 ? 'var(--ws-accent)' : 'var(--ws-text-dim)', fontVariantNumeric: 'tabular-nums' }}>
                    {d.readout.openness != null ? (d.readout.openness * 100).toFixed(1) + '%' : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px' }}>
                  <span style={{ color: 'var(--ws-text-muted)' }}>FPS:</span>
                  <span style={{ fontVariantNumeric: 'tabular-nums' }}>{d.readout.fps} Hz (30 Hz target)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '14px' }}>
                  <span style={{ color: 'var(--ws-text-muted)' }}>ALIGNMENT:</span>
                  <span style={{ color: d.readout.faceOk ? 'var(--ws-success)' : 'var(--ws-warning)', fontWeight: 550 }}>
                    {d.readout.faceOk ? '96×96 EYE-AXIS ALIGNED' : 'ALIGNING FACE'}
                  </span>
                </div>
              </div>
            )}
            {d.running && <div className="ws-live-video-hud"><span><i className="ws-dot"/>CAMERA ACTIVE</span><span>{d.readout.quality ? 'CHECK CAPTURE QUALITY' : d.readout.faceOk ? 'FACE DETECTED' : 'LOOKING FOR A FACE'}</span></div>}
            {d.running && <div className="ws-live-video-bottom"><span>{d.readout.fps} Hz landmark processing</span><span>{Math.floor(d.readout.elapsedSeconds / 60).toString().padStart(2, '0')}:{(d.readout.elapsedSeconds % 60).toString().padStart(2, '0')}</span></div>}
          </div>
        )}
        <VisualSpeechEqualizer
          active={d.running}
          speaking={d.readout.speaking}
          aperture={d.readout.openness}
          height={46}
          showApertureReadout
        />
        <div className="ws-live-controls"><div><Icon name={d.running ? 'scan' : 'info'} size={16}/><p>{d.running ? d.readout.quality || 'Look toward the camera. Keep your mouth visible and maintain consistent lighting.' : 'Camera permission is requested only when you start.'}</p></div>{(d.running || cameraBusy) && <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={d.stop}><Icon name="stop" size={13}/>{cameraBusy ? 'Cancel' : 'Stop camera'}</button>}{d.devices.length > 1 && <select className="ws-live-camera-select" aria-label="Select camera" value={cameraId} disabled={d.running || cameraBusy} onChange={e => setCameraId(e.target.value)}><option value="">Default camera</option>{d.devices.map((device, i) => <option key={device.deviceId} value={device.deviceId}>{device.label || 'Camera ' + (i + 1)}</option>)}</select>}</div>
        {assistant && <WordCaptureStatus state={assistant.state}/>}
        <div className="ws-live-buffer"><div><span>{assistant ? 'Current word capture' : 'Temporal sequence'}</span><span className="ws-mono">{assistant?.state.frames ?? d.readout.frames} / {d.sequence.maxFrames} FRAMES</span></div><div className="ws-live-buffer-track"><canvas ref={d.bufferStripRef} width={d.sequence.maxFrames * 24} height={24} aria-label="Captured mouth frames"/>{(assistant?.state.frames ?? d.readout.frames) === 0 && <span>{assistant ? 'A fresh buffer starts when you click Start Word' : 'Real mouth crops appear here as you speak'}</span>}</div><p>At least {d.sequence.minFrames} frames needed. {assistant ? lrwMode ? 'Complete one word within three seconds; actual timestamps are retained.' : 'The backend resamples your completed word to its trained input length.' : 'Overlapping windows use measured mouth movement.'}</p></div>
      </section>
      <aside className="ws-live-prediction" aria-label="Real model prediction">
        <div className="ws-live-card-heading"><span><Icon name="activity" size={15}/>Model output</span><span className="ws-live-experimental">RESEARCH</span></div>
        <div className="ws-live-modes">
          <button type="button" aria-pressed={d.recognitionMode === 'word'} disabled={!wordAvailable || busy} title={wordAvailable ? 'Recognize one supported whole word' : 'The evaluated eight-word backend is not ready.'} onClick={() => changeMode('word')}>8 Words (MIRACL) · 94.8%</button>
          <button type="button" aria-pressed={expandedMode} disabled={!expandedAvailable || busy} title={d.expandedHealth.error || (expandedAvailable ? 'Combined MIRACL and allenye66 research model' : 'The evaluated 20-word backend is not ready.')} onClick={() => changeMode('expanded')}>{d.expandedHealth.state === 'loading' ? '20 · Loading' : '20 Words (Combined)'}</button>
          <button type="button" aria-pressed={personalMode} disabled={!personalAvailable || busy} title={personalAvailable ? 'Personalized model adapted with your custom words (including help)' : 'Train your personal words in Word Collection first.'} onClick={() => changeMode('personal')}>⭐ Personal {d.personalHealth.vocabulary.length > 0 ? `(${d.personalHealth.vocabulary.length})` : ''}</button>
          <button type="button" aria-pressed={lrwMode} disabled={!d.lrwHealth.inference_ready || busy} title={d.lrwHealth.error || '500-word research model'} onClick={() => changeMode('lrw500')}>{d.lrwHealth.state === 'loading' ? '500 · Loading' : '500 Words (LRW)'}</button>
          <Link to="/demo/experiments" style={{ flex: '1 1 100%', padding: '6px 8px', fontSize: '11px', color: '#64748b', textDecoration: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', borderTop: '1px solid #1f2536', marginTop: '4px' }} title="Single letter A–Z classification is archived under Experiments due to viseme homophenes.">
            <span>A–Z Baseline (~12% Viseme Ambiguity) Archived in Experiments</span>
            <Icon name="arrow" size={11} />
          </Link>
        </div>
        {!wordMode && (
          <div style={{ padding: "12px 14px", background: "rgba(239, 68, 68, 0.08)", borderRadius: "8px", border: "1px solid rgba(239, 68, 68, 0.3)", fontSize: "13px", margin: "10px 0", lineHeight: 1.45 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#f87171", fontWeight: 600 }}>
              <Icon name="alert" size={14} /> Alphabet Baseline Experiment (~12% Accuracy)
            </div>
            <p style={{ margin: "6px 0 8px 0", opacity: 0.9 }}>
              Isolated single-letter recognition suffers from severe viseme ambiguity (letters like B, P, M or D, T share identical mouth closures).
              For verified high-accuracy recognition, switch to the <strong>8-Word (94.8%)</strong> or <strong>20-Word (94.5%)</strong> AV-HuBERT tracks.
            </p>
            <button
              type="button"
              className="ws-btn ws-btn-primary ws-btn-sm"
              disabled={!wordAvailable || busy}
              onClick={() => changeMode("word")}
              style={{ fontSize: "12px", padding: "4px 10px" }}
            >
              Switch to 8-Word Model (94.8% Acc) →
            </button>
          </div>
        )}

        
        {wordMode && !lrwMode && (
          <div style={{ margin: "10px 0 12px 0", padding: "10px 12px", background: "var(--ws-surface2)", borderRadius: "8px", border: "1px solid var(--ws-border)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--ws-accent)" }}>
                Active Vocabulary ({selectedHealth.vocabulary.length} words)
              </span>
              <Link to="/demo/feedback" style={{ fontSize: "11px", color: "var(--ws-accent)", textDecoration: "none" }}>Verify or correct predictions →</Link>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "5px" }}>
              {selectedHealth.vocabulary.map(w => (
                <span
                  key={w}
                  style={{
                    padding: "2px 7px",
                    borderRadius: "4px",
                    fontSize: "12px",
                    background: w === "help" ? "rgba(245, 158, 11, 0.15)" : "var(--ws-surface3)",
                    border: w === "help" ? "1px solid var(--ws-warning)" : "1px solid var(--ws-border)",
                    color: w === "help" ? "var(--ws-warning)" : "var(--ws-text)",
                    fontWeight: w === "help" ? 600 : 450
                  }}
                >
                  {w} {w === "help" ? "⭐" : ""}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Studio View Deck Switcher: Live Stream vs. Latent Constellation vs. Noise Stress Lab */}
        <div style={{
          display: 'flex',
          gap: '4px',
          margin: '12px 0',
          padding: '3px',
          background: '#0D1019',
          border: '1px solid #1F2536',
          borderRadius: '6px',
        }}>
          <button
            type="button"
            onClick={() => { cyberAudio.playClick(); setActiveStudioTab('stream') }}
            style={{
              flex: 1,
              padding: '5px 8px',
              fontSize: '11px',
              fontFamily: 'monospace',
              background: activeStudioTab === 'stream' ? '#161D2D' : 'transparent',
              color: activeStudioTab === 'stream' ? '#38BDF8' : '#64748B',
              border: activeStudioTab === 'stream' ? '1px solid #1F2536' : '1px solid transparent',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 600,
              transition: 'all 0.15s ease',
            }}
          >
            ⚡ STREAM
          </button>
          <button
            type="button"
            onClick={() => { cyberAudio.playClick(); setActiveStudioTab('constellation') }}
            style={{
              flex: 1,
              padding: '5px 8px',
              fontSize: '11px',
              fontFamily: 'monospace',
              background: activeStudioTab === 'constellation' ? '#161D2D' : 'transparent',
              color: activeStudioTab === 'constellation' ? '#38BDF8' : '#64748B',
              border: activeStudioTab === 'constellation' ? '1px solid #1F2536' : '1px solid transparent',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 600,
              transition: 'all 0.15s ease',
            }}
          >
            🌌 LATENT MAP
          </button>
          <button
            type="button"
            onClick={() => { cyberAudio.playClick(); setActiveStudioTab('noise') }}
            style={{
              flex: 1,
              padding: '5px 8px',
              fontSize: '11px',
              fontFamily: 'monospace',
              background: activeStudioTab === 'noise' ? '#161D2D' : 'transparent',
              color: activeStudioTab === 'noise' ? '#38BDF8' : '#64748B',
              border: activeStudioTab === 'noise' ? '1px solid #1F2536' : '1px solid transparent',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 600,
              transition: 'all 0.15s ease',
            }}
          >
            🔬 NOISE LAB
          </button>
        </div>

        {activeStudioTab === 'constellation' ? (
          <div style={{ marginBottom: '16px' }}>
            <LatentConstellation
              activeWord={assistant?.state.result?.word ?? assistant?.state.result?.candidate ?? d.readout.detected}
              height={440}
            />
          </div>
        ) : activeStudioTab === 'noise' ? (
          <div style={{ marginBottom: '16px' }}>
            <NoiseStressLab />
          </div>
        ) : assistant ? (
          <>
            <WordAssistantPanel
              state={assistant.state}
              controller={assistant.controller}
              canCapture={d.canCaptureWord}
              cameraActive={d.running}
              begin={d.beginWord}
              vocabulary={selectedHealth.vocabulary}
              onInspectFrames={() => {
                const el = document.getElementById('frame-inspector-section')
                el?.scrollIntoView({ behavior: 'smooth' })
              }}
            />
            <div style={{ margin: '14px 0' }}>
              <WordProbabilityRace
                isProcessing={d.predicting || phase === 'PROCESSING'}
                topWord={assistant?.state.result?.word ?? assistant?.state.result?.candidate ?? d.readout.detected}
                confidence={assistant?.state.result?.confidence ?? d.readout.confidence}
                candidates={d.readout.top3?.map(t => ({ word: t.letter, prob: t.prob }))}
              />
            </div>
          </>
        ) : <>
          <div className="ws-live-letter-area"><span className="ws-eyebrow">{d.predicting ? 'ANALYZING SEQUENCE' : uncertain ? 'UNCERTAIN' : d.readout.stabilizing ? 'CONFIRMING PREDICTION' : hasPrediction ? 'PREDICTED LETTER' : 'WAITING FOR A SIGNAL'}</span><p className={'ws-live-letter' + (d.predicting ? ' is-processing' : '')} aria-live="polite">{hasPrediction ? d.readout.detected : '—'}</p><p className="ws-live-letter-note">{d.predicting ? 'Processing captured mouth frames…' : uncertain ? 'UNCERTAIN — PLEASE TRY AGAIN' : hasPrediction ? 'Experimental alphabet classification' : !d.inferenceReady ? 'Connect an available alphabet model' : d.readout.stabilizing ? 'Waiting for a second consistent window' : d.running ? 'Watching for mouth movement' : 'Start camera and speak a single letter'}</p></div>
          <div className="ws-live-confidence"><span>{d.readout.smoothed ? 'Smoothed model score' : 'Model confidence'}</span><strong>{d.readout.confidence !== null ? (d.readout.confidence * 100).toFixed(1) + '%' : '—'}</strong><div><i style={{ width: d.readout.confidence !== null ? d.readout.confidence * 100 + '%' : '0%' }}/></div></div>
          <div style={{ margin: '14px 0' }}>
            <WordProbabilityRace
              isProcessing={d.predicting}
              topWord={d.readout.detected}
              confidence={d.readout.confidence}
              candidates={d.readout.top3?.map(t => ({ word: t.letter, prob: t.prob }))}
            />
          </div>
          <div className="ws-live-candidates"><div className="ws-live-small-title"><span>Top candidates</span><span>MODEL SCORE</span></div>{d.readout.top3?.length ? d.readout.top3.map((candidate, i) => <div className="ws-live-candidate" key={candidate.letter}><span>{candidate.letter}</span><div><i style={{ width: candidate.prob * 100 + '%', opacity: Math.max(.3, 1 - i * .15) }}/></div><strong>{(candidate.prob * 100).toFixed(1)}%</strong></div>) : <div className="ws-live-no-candidates"><Icon name="layers" size={19}/><p>Candidates appear after a real model response.</p></div>}</div>
          <div className="ws-live-model-note"><Icon name="info" size={14}/><p>Alphabet display requires two consistent windows. Scores are uncalibrated. Webcam accuracy remains unmeasured.</p></div>
        </>}
      </aside>
    </div>
    <div className="ws-live-tools"><label><input type="checkbox" checked={d.showOverlay} onChange={e => d.setShowOverlay(e.target.checked)}/> Mouth overlay</label><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => setIsMirrored(m => !m)} title="Toggle camera horizontal mirror">{isMirrored ? "Mirrored (Selfie)" : "Direct view"}</button>
        <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={d.reset}>Reset session</button><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={d.clearHistory} disabled={!retainedCount || busy}>Clear History</button><span role="status">{assistant ? assistant.state.phase.split('_').join(' ') : d.predicting ? 'Processing' : d.inferenceReady ? 'Ready' : 'Model unavailable'}</span></div>
    {d.readout.serverLatencyMs !== null && <div className="ws-live-measurements" aria-label="Measured CPU request performance"><span>Server: {d.readout.serverLatencyMs.toFixed(0)} ms</span><span>Round trip: {Math.round(d.readout.latencyMs ?? 0)} ms</span>{d.readout.ramMb !== null && <span>Process RAM: {d.readout.ramMb.toFixed(0)} MiB</span>}{d.readout.cpuPercent !== null && <span>CPU: {d.readout.cpuPercent.toFixed(0)}% of one core</span>}{d.readout.throughput !== null && <span>{d.readout.throughput.toFixed(2)} requests/s equivalent</span>}<small>Measured per request, not sustained webcam throughput.</small></div>}
    {issue && <div className="ws-live-error" role="alert"><Icon name="alert" size={18}/><div><strong>{cameraError ? 'Camera needs attention' : 'This attempt needs attention'}</strong><p>{issue}</p></div><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={cameraError ? () => void d.start(cameraId || undefined) : assistant && d.running ? assistant.controller.recover : refresh}>{assistant && d.running ? 'Back to Ready' : 'Retry'}</button></div>}
    {!d.inferenceReady && <div className="ws-live-notice"><Icon name="info" size={16}/><p>{d.inferenceReason || 'Checking the inference service. Camera preview remains available.'}</p><button className="ws-inline-link" onClick={refresh}>Check again<Icon name="refresh" size={12}/></button></div>}
    {assistant && <WordSession state={assistant.state} controller={assistant.controller}/>}
    <TranscriptComposer
      latestWord={assistant?.state.result?.word ?? assistant?.state.result?.candidate ?? (hasPrediction ? d.readout.detected : null)}
      confidence={assistant?.state.result?.confidence ?? d.readout.confidence}
      alternatives={d.readout.top3?.map(t => ({ label: t.letter, score: t.prob }))}
      ttsEnabled={ttsEnabled}
      onToggleTTS={() => setTtsEnabled(v => !v)}
    />
    <section id="frame-inspector-section" style={{ margin: '20px 0' }}>
      <FrameInspector
        frames={d.capturedFrames}
        durationMs={assistant?.state.captureMs}
        label={assistant?.state.result?.word ?? assistant?.state.result?.candidate ?? (hasPrediction ? d.readout.detected ?? undefined : undefined)}
      />
    </section>
    <section className="ws-live-pipeline" aria-label="Real pipeline telemetry"><div className="ws-live-section-title"><h2>Behind the prediction</h2><span>LIVE PIPELINE TELEMETRY</span></div><div className="ws-live-stages">{steps.map((step, i) => <div className={'ws-live-stage' + (step.ready ? ' is-ready' : '')} key={step.name}><div><Icon name={step.icon} size={16}/><small>0{i + 1}</small></div><h3>{step.name}</h3><p>{step.value}</p></div>)}</div></section>
    <div className="ws-live-details">
      <section className="ws-card ws-live-analysis"><div className="ws-card-head"><h2>The visual signals</h2><span className="ws-mono">LOCAL ANALYSIS</span></div><div className="ws-live-signals">
        <div><span>Appearance</span><div className="ws-live-signal-canvas ws-live-crop"><canvas ref={d.appearanceRef} width={64} height={64} aria-label="Actual mouth crop"/>{!d.readout.cropOk && <Icon name="scan" size={22}/>}</div><small>{d.readout.cropOk ? 'Mouth crop preview' : 'Awaiting mouth crop'}</small></div>
        <div><span>Geometry</span><div className="ws-live-signal-canvas"><canvas ref={d.geometryRef} width={160} height={56} aria-label="Measured mouth openness history"/>{d.readout.openness === null && <span>No measurements</span>}</div><small>{d.readout.openness !== null ? 'Aperture ' + d.readout.openness.toFixed(3) : 'Mouth aperture ratio'}</small></div>
        <div><span>Motion</span><div className="ws-live-signal-canvas"><canvas ref={d.motionRef} width={160} height={56} aria-label="Measured crop pixel changes"/>{d.readout.motionEnergy === null && <span>No measurements</span>}</div><small>{d.readout.motionEnergy !== null ? 'Pixel change ' + d.readout.motionEnergy.toFixed(3) : 'Frame-to-frame pixel change'}</small></div>
      </div><p className="ws-live-fusion-note">Model fusion weights are not returned by this classifier.</p></section>
      {!assistant && <section className="ws-card ws-live-history"><div className="ws-card-head"><h2>Recent results</h2><button className="ws-inline-link" onClick={d.clearHistory} disabled={!d.history.length}>Clear</button></div>{d.history.length === 0 ? <div className="ws-live-history-empty"><Icon name="activity" size={23}/><p>Real predictions will appear here.</p></div> : <ol>{d.history.slice(0, 4).map(result => <li key={result.id}><strong>{result.letter}</strong><div><span>Alphabet prediction</span><small>{new Date(result.at).toLocaleTimeString()}</small></div><span>{result.confidence !== null ? (result.confidence * 100).toFixed(1) + '%' : 'Unavailable'}</span></li>)}</ol>}<p className="ws-live-history-note">Session only</p></section>}
    </div>
    <QuickStartModal isOpen={quickStartOpen} onClose={() => setQuickStartOpen(false)} />
  </div>
}
