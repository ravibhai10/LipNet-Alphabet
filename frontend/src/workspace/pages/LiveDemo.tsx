import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useDemoPipeline } from '../../live-demo/useDemoPipeline'
import { useBackend } from '../../hooks/useBackend'
import { Icon } from '../Icon'
import { RecognitionEvidence } from '../components/RecognitionEvidence'
import { WordAssistantPanel, WordCaptureStatus, WordSession } from '../components/WordAssistant'
import { MissionControl } from '../components/MissionControl'
import '../../styles/live-workspace.css'

function CameraIllustration() {
  return <svg className="ws-live-illustration" width="164" height="130" viewBox="0 0 164 130" fill="none" aria-hidden="true"><rect x="1" y="1" width="162" height="128" rx="30" stroke="currentColor" strokeOpacity=".08"/><circle cx="82" cy="65" r="49" stroke="currentColor" strokeOpacity=".09"/><path d="M32 66C51 64 61 48 73 55L82 59L91 55C103 48 113 64 132 66C113 69 105 87 82 87S51 69 32 66Z" stroke="currentColor" strokeOpacity=".7"/><path d="M32 66C51 65 64 69 82 69S113 65 132 66" stroke="currentColor" strokeOpacity=".4"/><path d="M18 36v-14h14M132 22h14v14M146 94v14h-14M32 108H18V94" stroke="currentColor" strokeOpacity=".4"/></svg>
}
export function LiveDemoPage() {
  const d = useDemoPipeline({ manualWords: true })
  const [isMirrored, setIsMirrored] = useState(true)
  const wordMode = d.recognitionMode !== 'alphabet', lrwMode = d.recognitionMode === 'lrw500', expandedMode = d.recognitionMode === 'expanded'
  const assistant = d.assistant, selectedHealth = lrwMode ? d.lrwHealth : expandedMode ? d.expandedHealth : d.wordHealth
  const phase = assistant?.state.phase
  const busy = phase === 'CAPTURING' || phase === 'PROCESSING' || phase === 'CAMERA_STARTING' || phase === 'STOPPING'
  const [vocabularyQuery, setVocabularyQuery] = useState('')
  const vocabulary = selectedHealth.vocabulary.filter(word => word.toLowerCase().includes(vocabularyQuery.toLowerCase()))
  const wordAvailable = d.wordHealth.inference_ready && d.wordHealth.research_activation_gate_passed === true
  const expandedAvailable = d.expandedHealth.inference_ready && d.expandedHealth.research_activation_gate_passed === true
  const recordedCounts = typeof selectedHealth.evaluation_correct === 'number' && typeof selectedHealth.evaluation_samples === 'number'
    && Number.isInteger(selectedHealth.evaluation_correct) && Number.isInteger(selectedHealth.evaluation_samples)
    && selectedHealth.evaluation_samples > 0 && selectedHealth.evaluation_correct >= 0 && selectedHealth.evaluation_correct <= selectedHealth.evaluation_samples
    ? `${selectedHealth.evaluation_correct}/${selectedHealth.evaluation_samples} correct on its recorded test clips`
    : 'test counts are not currently available from the backend'
  const { status, refresh } = useBackend()
  const [cameraId, setCameraId] = useState('')
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
    <RecognitionEvidence />
    <header className="ws-page-header ws-live-heading"><div><p className="ws-kicker">01 / THE LIVE STUDIO</p><h1 className="ws-title">Speech, <em>in sight.</em></h1><p className="ws-lede">{wordMode ? 'Your live lip reading assistant. One word, one deliberate capture.' : 'A closer look at the movement behind every letter.'}</p></div><Link to="/demo/pipeline" className="ws-inline-link">Inside the pipeline<Icon name="arrow" size={14}/></Link></header>
    <div className="ws-live-statuses" aria-label="Session readiness">
      <span><i className={d.running ? 'ready' : cameraError ? 'warning' : ''}/>{d.running ? 'Camera active' : cameraBusy ? 'Starting camera' : cameraError ? 'Camera unavailable' : 'Camera off'}</span>
      <span><i className={status === 'online' ? 'ready' : ''}/>{status === 'online' ? 'Backend connected' : status === 'checking' || status === 'idle' ? 'Checking backend' : 'Backend offline'}</span>
      <span><i className={d.inferenceReady ? 'ready' : 'warning'}/>{d.inferenceReady ? (wordMode ? 'Word model loaded' : 'Alphabet model loaded') : 'Inference unavailable'}</span>
      <span><i className={d.readout.faceOk ? 'ready' : ''}/>{d.readout.faceOk ? 'One face detected' : 'Position your face inside the frame'}</span>
      <span className="ws-live-local"><Icon name="shield" size={12}/>On-device landmarks</span>
      <span className="ws-live-local" title="Audio disabled by design; vision-only silent speech"><Icon name="shield" size={12}/>Audio Disabled · Silent Speech Only</span>
    </div>
    <MissionControl readout={d.readout} running={d.running} backendOnline={status === 'online'} inferenceReady={d.inferenceReady} history={d.history} assistant={assistant?.state}/>
    {wordMode && <div className="ws-live-notice"><Icon name="info" size={16}/><div>
      <p>{lrwMode ? '500-word research mode. Local 500-class and webcam accuracy are unmeasured.' : expandedMode ? `Combined 20-word research model: ${recordedCounts}. Upstream speakers are unidentified; webcam and unseen-speaker accuracy are unmeasured.` : d.wordHealth.evaluation_protocol === 'personal_cross_session' ? `Personalized eight-word model: ${recordedCounts}. Unseen-speaker and general webcam accuracy are unmeasured.` : 'Eight-word research model. The recorded 94.79% result is from 192 MIRACL clips; webcam accuracy is unmeasured.'}</p>
      {!lrwMode && <><div className="ws-assistant-vocabulary">{selectedHealth.vocabulary.map(w => <span key={w}>{w}</span>)}</div><Link className="ws-inline-link" to="/demo/collect">Record your own labelled 8-word examples →</Link></>}
      <p>Manual mode: Start Word → say one word → Finish Word → Next Word. No automatic word-boundary detection is assumed.</p>
      {lrwMode && d.lrwHealth.available_diagnostic && <p>Available-source diagnostic: {d.lrwHealth.available_diagnostic.correct}/{d.lrwHealth.available_diagnostic.known_samples} known clips correct across {d.lrwHealth.available_diagnostic.covered_classes}/500 classes; {d.lrwHealth.available_diagnostic.failed_inputs} input failed.</p>}
      {lrwMode && <details className="ws-live-vocabulary"><summary>Browse 500 supported words</summary><input aria-label="Search supported words" placeholder="Find a word…" value={vocabularyQuery} onChange={e => setVocabularyQuery(e.target.value)}/><p>{vocabulary.length} matches</p><div>{vocabulary.map(word => <span key={word}>{word}</span>)}</div></details>}
    </div></div>}
    <div className="ws-live-main">
      <section className="ws-live-capture" aria-label="Camera and sequence capture">
        <div className="ws-live-card-heading"><span><Icon name="camera" size={15}/>Camera feed</span><span className="ws-mono">{d.running ? 'LIVE INPUT' : 'AWAITING INPUT'}</span></div>
        <div className={'ws-live-viewport' + (d.running ? ' is-streaming' : '')}>
          <video ref={d.videoRef} className="ws-live-video" playsInline muted aria-label="Live camera preview" style={{ transform: isMirrored ? "scaleX(-1)" : "none" }} />
          <canvas ref={d.overlayRef} width={960} height={540} className="ws-live-overlay" aria-hidden="true"/>
          <div className="ws-live-corners" aria-hidden="true"><i/><i/><i/><i/></div>
          {d.running && wordMode && <div className="ws-assistant-face-guide" aria-hidden="true"/>}
          {!d.running && <div className="ws-live-idle"><CameraIllustration/><h2>{cameraBusy ? 'Preparing your camera.' : cameraError ? 'Let’s reconnect your camera.' : 'Your voice starts with a frame.'}</h2><p>{cameraBusy ? 'Allow camera access in your browser to begin.' : cameraError ? d.camError : 'Position your face inside the frame. Keep your mouth visible.'}</p>{!cameraBusy && <button className="ws-btn ws-btn-primary" onClick={() => void d.start(cameraId || undefined)}><Icon name={cameraError ? 'refresh' : 'camera'} size={15}/>{cameraError ? 'Retry camera' : 'Start camera'}</button>}<span className="ws-live-privacy"><Icon name="shield" size={12}/>Video only · no microphone</span></div>}
          {d.running && <div className="ws-live-video-hud"><span><i className="ws-dot"/>CAMERA ACTIVE</span><span>{d.readout.quality ? 'CHECK CAPTURE QUALITY' : d.readout.faceOk ? 'FACE DETECTED' : 'LOOKING FOR A FACE'}</span></div>}
          {d.running && <div className="ws-live-video-bottom"><span>{d.readout.fps} Hz landmark processing</span><span>{Math.floor(d.readout.elapsedSeconds / 60).toString().padStart(2, '0')}:{(d.readout.elapsedSeconds % 60).toString().padStart(2, '0')}</span></div>}
        </div>
        <div className="ws-live-controls"><div><Icon name={d.running ? 'scan' : 'info'} size={16}/><p>{d.running ? d.readout.quality || 'Look toward the camera. Keep your mouth visible and maintain consistent lighting.' : 'Camera permission is requested only when you start.'}</p></div>{(d.running || cameraBusy) && <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={d.stop}><Icon name="stop" size={13}/>{cameraBusy ? 'Cancel' : 'Stop camera'}</button>}{d.devices.length > 1 && <select className="ws-live-camera-select" aria-label="Select camera" value={cameraId} disabled={d.running || cameraBusy} onChange={e => setCameraId(e.target.value)}><option value="">Default camera</option>{d.devices.map((device, i) => <option key={device.deviceId} value={device.deviceId}>{device.label || 'Camera ' + (i + 1)}</option>)}</select>}</div>
        {assistant && <WordCaptureStatus state={assistant.state}/>}
        <div className="ws-live-buffer"><div><span>{assistant ? 'Current word capture' : 'Temporal sequence'}</span><span className="ws-mono">{assistant?.state.frames ?? d.readout.frames} / {d.sequence.maxFrames} FRAMES</span></div><div className="ws-live-buffer-track"><canvas ref={d.bufferStripRef} width={d.sequence.maxFrames * 24} height={24} aria-label="Captured mouth frames"/>{(assistant?.state.frames ?? d.readout.frames) === 0 && <span>{assistant ? 'A fresh buffer starts when you click Start Word' : 'Real mouth crops appear here as you speak'}</span>}</div><p>At least {d.sequence.minFrames} frames needed. {assistant ? lrwMode ? 'Complete one word within three seconds; actual timestamps are retained.' : 'The backend resamples your completed word to its trained input length.' : 'Overlapping windows use measured mouth movement.'}</p></div>
      </section>
      <aside className="ws-live-prediction" aria-label="Real model prediction">
        <div className="ws-live-card-heading"><span><Icon name="activity" size={15}/>Model output</span><span className="ws-live-experimental">RESEARCH</span></div>
        <div className="ws-live-modes"><button type="button" aria-pressed={!wordMode} disabled={busy} onClick={() => changeMode('alphabet')}>Alphabet <span>A-Z</span></button><button type="button" aria-pressed={d.recognitionMode === 'word'} disabled={!wordAvailable || busy} title={wordAvailable ? 'Recognize one supported whole word' : 'The evaluated eight-word backend is not ready.'} onClick={() => changeMode('word')}>{d.wordHealth.vocabulary.length > 8 ? `${d.wordHealth.vocabulary.length} words` : '8 words'}</button><button type="button" aria-pressed={expandedMode} disabled={!expandedAvailable || busy} title={d.expandedHealth.error || (expandedAvailable ? 'Combined MIRACL and allenye66 research model' : 'The evaluated 20-word backend is not ready.')} onClick={() => changeMode('expanded')}>{d.expandedHealth.state === 'loading' ? '20 · Loading' : '20 words'}</button><button type="button" aria-pressed={lrwMode} disabled={!d.lrwHealth.inference_ready || busy} title={d.lrwHealth.error || '500-word research model'} onClick={() => changeMode('lrw500')}>{d.lrwHealth.state === 'loading' ? '500 · Loading' : '500 words'}</button><span aria-disabled="true" title="No trained visual-only sentence decoder is connected.">Free speech <Icon name="info" size={11}/></span></div>
        {!wordMode && (
          <div style={{ padding: "10px 14px", background: "rgba(16, 185, 129, 0.08)", borderRadius: "8px", border: "1px solid rgba(16, 185, 129, 0.3)", fontSize: "13px", margin: "10px 0", lineHeight: 1.4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#10b981", fontWeight: 600 }}>
              <Icon name="check" size={14} /> Separate whole-word research mode
            </div>
            <p style={{ margin: "4px 0 8px 0", opacity: 0.85 }}>
              Choose word mode for its supported isolated-word vocabulary. Offline corpus scores do not establish webcam accuracy.
            </p>
            <button
              type="button"
              className="ws-btn ws-btn-primary ws-btn-sm"
              disabled={!wordAvailable || busy}
              onClick={() => changeMode("word")}
              style={{ fontSize: "12px", padding: "4px 10px" }}
            >
              Switch to 8 Words →
            </button>
          </div>
        )}

        
        {wordMode && !lrwMode && (
          <div style={{ margin: "10px 0 12px 0", padding: "10px 12px", background: "rgba(16, 185, 129, 0.05)", borderRadius: "8px", border: "1px solid rgba(16, 185, 129, 0.2)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "#10b981" }}>
                Active Vocabulary ({selectedHealth.vocabulary.length} words)
              </span>
              <Link to="/demo/collect" style={{ fontSize: "11px", color: "#34d399", textDecoration: "none" }}>+ Record / Add Words</Link>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "5px" }}>
              {selectedHealth.vocabulary.map(w => (
                <span
                  key={w}
                  style={{
                    padding: "2px 7px",
                    borderRadius: "4px",
                    fontSize: "12px",
                    background: w === "help" ? "rgba(245, 158, 11, 0.2)" : "rgba(255,255,255,0.06)",
                    border: w === "help" ? "1px solid #f59e0b" : "1px solid rgba(255,255,255,0.12)",
                    color: w === "help" ? "#fbbf24" : "inherit",
                    fontWeight: w === "help" ? 700 : 500
                  }}
                >
                  {w} {w === "help" ? "⭐" : ""}
                </span>
              ))}
            </div>
          </div>
        )}

        {assistant ? <WordAssistantPanel state={assistant.state} controller={assistant.controller} canCapture={d.canCaptureWord} cameraActive={d.running} begin={d.beginWord}/> : <>
          <div className="ws-live-letter-area"><span className="ws-eyebrow">{d.predicting ? 'ANALYZING SEQUENCE' : uncertain ? 'UNCERTAIN' : d.readout.stabilizing ? 'CONFIRMING PREDICTION' : hasPrediction ? 'PREDICTED LETTER' : 'WAITING FOR A SIGNAL'}</span><p className={'ws-live-letter' + (d.predicting ? ' is-processing' : '')} aria-live="polite">{hasPrediction ? d.readout.detected : '—'}</p><p className="ws-live-letter-note">{d.predicting ? 'Processing captured mouth frames…' : uncertain ? 'UNCERTAIN — PLEASE TRY AGAIN' : hasPrediction ? 'Experimental alphabet classification' : !d.inferenceReady ? 'Connect an available alphabet model' : d.readout.stabilizing ? 'Waiting for a second consistent window' : d.running ? 'Watching for mouth movement' : 'Start camera and speak a single letter'}</p></div>
          <div className="ws-live-confidence"><span>{d.readout.smoothed ? 'Smoothed model score' : 'Model confidence'}</span><strong>{d.readout.confidence !== null ? (d.readout.confidence * 100).toFixed(1) + '%' : '—'}</strong><div><i style={{ width: d.readout.confidence !== null ? d.readout.confidence * 100 + '%' : '0%' }}/></div></div>
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
    <section className="ws-live-pipeline" aria-label="Real pipeline telemetry"><div className="ws-live-section-title"><h2>Behind the prediction</h2><span>LIVE PIPELINE TELEMETRY</span></div><div className="ws-live-stages">{steps.map((step, i) => <div className={'ws-live-stage' + (step.ready ? ' is-ready' : '')} key={step.name}><div><Icon name={step.icon} size={16}/><small>0{i + 1}</small></div><h3>{step.name}</h3><p>{step.value}</p></div>)}</div></section>
    <div className="ws-live-details">
      <section className="ws-card ws-live-analysis"><div className="ws-card-head"><h2>The visual signals</h2><span className="ws-mono">LOCAL ANALYSIS</span></div><div className="ws-live-signals">
        <div><span>Appearance</span><div className="ws-live-signal-canvas ws-live-crop"><canvas ref={d.appearanceRef} width={64} height={64} aria-label="Actual mouth crop"/>{!d.readout.cropOk && <Icon name="scan" size={22}/>}</div><small>{d.readout.cropOk ? 'Mouth crop preview' : 'Awaiting mouth crop'}</small></div>
        <div><span>Geometry</span><div className="ws-live-signal-canvas"><canvas ref={d.geometryRef} width={160} height={56} aria-label="Measured mouth openness history"/>{d.readout.openness === null && <span>No measurements</span>}</div><small>{d.readout.openness !== null ? 'Aperture ' + d.readout.openness.toFixed(3) : 'Mouth aperture ratio'}</small></div>
        <div><span>Motion</span><div className="ws-live-signal-canvas"><canvas ref={d.motionRef} width={160} height={56} aria-label="Measured crop pixel changes"/>{d.readout.motionEnergy === null && <span>No measurements</span>}</div><small>{d.readout.motionEnergy !== null ? 'Pixel change ' + d.readout.motionEnergy.toFixed(3) : 'Frame-to-frame pixel change'}</small></div>
      </div><p className="ws-live-fusion-note">Model fusion weights are not returned by this classifier.</p></section>
      {!assistant && <section className="ws-card ws-live-history"><div className="ws-card-head"><h2>Recent results</h2><button className="ws-inline-link" onClick={d.clearHistory} disabled={!d.history.length}>Clear</button></div>{d.history.length === 0 ? <div className="ws-live-history-empty"><Icon name="activity" size={23}/><p>Real predictions will appear here.</p></div> : <ol>{d.history.slice(0, 4).map(result => <li key={result.id}><strong>{result.letter}</strong><div><span>Alphabet prediction</span><small>{new Date(result.at).toLocaleTimeString()}</small></div><span>{result.confidence !== null ? (result.confidence * 100).toFixed(1) + '%' : 'Unavailable'}</span></li>)}</ol>}<p className="ws-live-history-note">Session only</p></section>}
    </div>
  </div>
}
