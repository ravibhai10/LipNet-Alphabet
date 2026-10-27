import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useDemoPipeline } from '../../live-demo/useDemoPipeline'
import { useBackend } from '../../hooks/useBackend'
import { Icon } from '../Icon'
import '../../styles/live-workspace.css'

function CameraIllustration() {
  return <svg className="ws-live-illustration" width="164" height="130" viewBox="0 0 164 130" fill="none" aria-hidden="true">
    <rect x="1" y="1" width="162" height="128" rx="30" stroke="currentColor" strokeOpacity=".08"/>
    <circle cx="82" cy="65" r="49" stroke="currentColor" strokeOpacity=".09"/><circle cx="82" cy="65" r="36" stroke="currentColor" strokeOpacity=".07"/>
    <path d="M32 66C51 64 61 48 73 55L82 59L91 55C103 48 113 64 132 66C113 69 105 87 82 87S51 69 32 66Z" stroke="currentColor" strokeOpacity=".7"/>
    <path d="M32 66C51 65 64 69 82 69S113 65 132 66" stroke="currentColor" strokeOpacity=".4"/>
    <path d="M18 36v-14h14M132 22h14v14M146 94v14h-14M32 108H18V94" stroke="currentColor" strokeOpacity=".4"/>
    <circle cx="32" cy="66" r="2.5" fill="currentColor"/><circle cx="132" cy="66" r="2.5" fill="currentColor"/>
  </svg>
}
export function LiveDemoPage() {
  const d = useDemoPipeline()
  const { status, refresh } = useBackend()
  const [cameraId, setCameraId] = useState('')
  const cameraBusy = d.camStatus === 'requesting'
  const cameraError = d.camStatus === 'denied' || d.camStatus === 'error'
  const issue = d.camError || d.lmError || d.pipelineError || d.predictionError
  const hasPrediction = d.inferenceReady && d.readout.detected !== null
  const steps = [
    { icon: 'camera', name: 'Camera', value: d.running ? 'Streaming' : cameraBusy ? 'Requesting' : cameraError ? 'Unavailable' : 'Standby', ready: d.running },
    { icon: 'scan', name: 'Landmarks', value: d.readout.faceOk ? 'Face detected' : d.lmStatus === 'loading' ? 'Loading' : d.lmStatus === 'failed' ? 'Unavailable' : d.running ? 'Finding face' : 'Standby', ready: d.readout.faceOk },
    { icon: 'layers', name: 'Mouth crops', value: d.readout.cropOk ? d.readout.frames + ' buffered' : 'Awaiting face', ready: d.readout.cropOk },
    { icon: 'pipeline', name: 'API request', value: d.predicting ? 'Processing' : d.predictionError ? 'Failed' : d.readout.predictedAt ? 'Response received' : 'Not sent', ready: d.readout.predictedAt !== null },
    { icon: 'cpu', name: 'Inference', value: d.readout.predictedAt ? d.readout.latencyMs + ' ms round trip' : d.inferenceReady ? 'Model loaded' : 'Unavailable', ready: d.inferenceReady },
  ]
  return <div className="ws-live">
    <header className="ws-page-header ws-live-heading"><div><p className="ws-kicker">01 / THE LIVE STUDIO</p><h1 className="ws-title">Speech, <em>in sight.</em></h1><p className="ws-lede">A closer look at the movement behind every letter.</p></div><Link to="/demo/pipeline" className="ws-inline-link">Inside the pipeline<Icon name="arrow" size={14}/></Link></header>
    <div className="ws-live-statuses" aria-label="Session readiness">
      <span><i className={d.running ? 'ready' : cameraError ? 'warning' : ''}/>{d.running ? 'Camera streaming' : cameraBusy ? 'Requesting camera' : cameraError ? 'Camera unavailable' : 'Camera off'}</span>
      <span><i className={status === 'online' ? 'ready' : ''}/>{status === 'online' ? 'Backend connected' : status === 'checking' || status === 'idle' ? 'Checking backend' : 'Backend offline'}</span>
      <span><i className={d.inferenceReady ? 'ready' : 'warning'}/>{d.inferenceReady ? 'Alphabet model loaded' : 'Inference unavailable'}</span>
      <span><i className={d.readout.speaking ? 'ready' : ''}/>{!d.readout.faceOk ? 'Movement not measured' : d.readout.speaking ? 'Mouth movement detected' : 'Listening for movement'}</span>
      <span className="ws-live-local"><Icon name="shield" size={12}/>On-device landmarks</span>
    </div>
    <div className="ws-live-main">
      <section className="ws-live-capture" aria-label="Camera and sequence capture">
        <div className="ws-live-card-heading"><span><Icon name="camera" size={15}/>Camera feed</span><span className="ws-mono">{d.running ? 'LIVE INPUT' : 'AWAITING INPUT'}</span></div>
        <div className={'ws-live-viewport' + (d.running ? ' is-streaming' : '')}>
          <video ref={d.videoRef} className="ws-live-video" playsInline muted aria-label="Live camera preview" />
          <canvas ref={d.overlayRef} width={960} height={540} className="ws-live-overlay" aria-hidden="true"/>
          <div className="ws-live-corners" aria-hidden="true"><i/><i/><i/><i/></div>
          {!d.running && <div className="ws-live-idle"><CameraIllustration/><h2>{cameraBusy ? 'One moment, one frame.' : cameraError ? 'Let’s reconnect your camera.' : 'Your voice starts with a frame.'}</h2><p>{cameraBusy ? 'Allow camera access in your browser to begin.' : cameraError ? d.camError : 'Bring your face into view. We’ll take care of the details.'}</p>{!cameraBusy && <button className="ws-btn ws-btn-primary" onClick={() => void d.start(cameraId || undefined)}><Icon name={cameraError ? 'refresh' : 'camera'} size={15}/>{cameraError ? 'Retry camera' : 'Start camera'}</button>}<span className="ws-live-privacy"><Icon name="shield" size={12}/>No audio. No saved recordings.</span></div>}
          {d.running && <div className="ws-live-video-hud"><span><i className="ws-dot"/>LIVE</span><span>{d.readout.faceOk ? 'FACE DETECTED' : d.lmStatus === 'loading' ? 'LOADING LANDMARKS' : 'LOOKING FOR A FACE'}</span></div>}
          {d.running && <div className="ws-live-video-bottom"><span>{d.readout.fps} Hz landmark processing</span><span>{Math.floor(d.readout.elapsedSeconds / 60).toString().padStart(2,'0')}:{(d.readout.elapsedSeconds % 60).toString().padStart(2,'0')}</span></div>}
        </div>
        <div className="ws-live-controls"><div><Icon name={d.running ? 'scan' : 'info'} size={16}/><p>{d.running ? 'Keep your face centered and naturally lit.' : 'Camera permission is requested only when you start.'}</p></div>{(d.running || cameraBusy) && <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={d.stop}><Icon name="stop" size={13}/>{cameraBusy ? 'Cancel' : 'Stop camera'}</button>}{d.devices.length > 1 && <select className="ws-live-camera-select" aria-label="Select camera" value={cameraId} disabled={d.running || cameraBusy} onChange={e => setCameraId(e.target.value)}><option value="">Default camera</option>{d.devices.map((device,i)=><option key={device.deviceId} value={device.deviceId}>{device.label || 'Camera ' + (i+1)}</option>)}</select>}</div>
        <div className="ws-live-buffer"><div><span>Temporal sequence</span><span className="ws-mono">{d.readout.frames.toString().padStart(2,'0')} / {d.sequence.maxFrames} FRAMES</span></div><div className="ws-live-buffer-track"><canvas ref={d.bufferStripRef} width={d.sequence.maxFrames * 24} height={24} aria-label={d.readout.frames + ' captured mouth frames'}/>{d.readout.frames === 0 && <span>Real mouth crops appear here as you speak</span>}</div><p>At least {d.sequence.minFrames} frames needed for alphabet inference.</p></div>
      </section>
      <aside className="ws-live-prediction" aria-label="Real model prediction">
        <div className="ws-live-card-heading"><span><Icon name="activity" size={15}/>Model output</span><span className="ws-live-experimental">EXPERIMENTAL</span></div>
        <div className="ws-live-modes"><span>Alphabet <span>A–Z</span></span><span title="A compatible sentence checkpoint and decoder are not connected.">Sentence <Icon name="info" size={11}/></span></div>
        <div className="ws-live-letter-area"><span className="ws-eyebrow">{d.predicting ? 'ANALYZING SEQUENCE' : d.readout.stabilizing ? 'CONFIRMING PREDICTION' : hasPrediction ? 'PREDICTED LETTER' : 'WAITING FOR A SIGNAL'}</span><p className={'ws-live-letter' + (d.predicting ? ' is-processing' : '')} aria-live="polite">{hasPrediction ? d.readout.detected : '—'}</p><p className="ws-live-letter-note">{d.predicting ? 'Processing captured mouth frames…' : hasPrediction ? 'Experimental alphabet classification' : !d.inferenceReady ? 'Connect an available alphabet model' : d.readout.stabilizing ? 'Waiting for a second consistent window' : d.running ? 'Listening for mouth movement' : 'Start camera and speak a single letter'}</p></div>
        <div className="ws-live-confidence"><span>{d.readout.smoothed ? 'Smoothed confidence' : 'Model confidence'}</span><strong>{hasPrediction && d.readout.confidence !== null ? (d.readout.confidence * 100).toFixed(1) + '%' : '—'}</strong><div><i style={{width: hasPrediction && d.readout.confidence !== null ? d.readout.confidence * 100 + '%' : '0%'}}/></div></div>
        <div className="ws-live-candidates"><div className="ws-live-small-title"><span>Top candidates</span><span>PROBABILITY</span></div>{hasPrediction && d.readout.top3?.length ? d.readout.top3.map((candidate,i)=><div className="ws-live-candidate" key={candidate.letter}><span>{candidate.letter}</span><div><i style={{width:candidate.prob * 100 + '%',opacity:1-i*.25}}/></div><strong>{(candidate.prob * 100).toFixed(1)}%</strong></div>) : <div className="ws-live-no-candidates"><Icon name="layers" size={19}/><p>Candidates appear after a real<br/>model response.</p></div>}</div>
        <div className="ws-live-model-note"><Icon name="info" size={14}/><p>Display requires two consistent windows. EMA uses real model scores. Webcam accuracy remains unvalidated. Sentence inference is unavailable.</p></div>
      </aside>
    </div>
    {issue && <div className="ws-live-error" role="alert"><Icon name="alert" size={18}/><div><strong>{cameraError ? 'Camera needs attention' : d.lmError || d.pipelineError ? 'Landmark pipeline needs attention' : 'Inference needs attention'}</strong><p>{issue}</p></div><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={cameraError ? () => void d.start(cameraId || undefined) : refresh}>Retry</button></div>}
    {!d.inferenceReady && <div className="ws-live-notice"><Icon name="info" size={16}/><p>{d.inferenceReason || 'Checking the local alphabet inference service. You can still inspect camera capture and mouth landmarks.'}</p><button className="ws-inline-link" onClick={refresh}>Check again<Icon name="refresh" size={12}/></button></div>}
    <section className="ws-live-pipeline" aria-label="Real pipeline telemetry"><div className="ws-live-section-title"><h2>Behind the prediction</h2><span>LIVE PIPELINE TELEMETRY</span></div><div className="ws-live-stages">{steps.map((step,i)=><div className={'ws-live-stage' + (step.ready ? ' is-ready' : '')} key={step.name}><div><Icon name={step.icon} size={16}/><small>0{i+1}</small></div><h3>{step.name}</h3><p>{step.value}</p></div>)}</div></section>
    <div className="ws-live-details">
      <section className="ws-card ws-live-analysis"><div className="ws-card-head"><h2>The visual signals</h2><span className="ws-mono">LOCAL ANALYSIS</span></div><div className="ws-live-signals">
        <div><span>Appearance</span><div className="ws-live-signal-canvas ws-live-crop"><canvas ref={d.appearanceRef} width={64} height={64} aria-label="Actual mouth crop"/>{!d.readout.cropOk && <Icon name="scan" size={22}/>}</div><small>{d.readout.cropOk ? '64 × 64 mouth crop' : 'Awaiting mouth crop'}</small></div>
        <div><span>Geometry</span><div className="ws-live-signal-canvas"><canvas ref={d.geometryRef} width={160} height={56} aria-label="Measured mouth openness history"/>{d.readout.openness === null && <span>No measurements</span>}</div><small>{d.readout.openness !== null ? 'Aperture ' + d.readout.openness.toFixed(3) : 'Mouth aperture ratio'}</small></div>
        <div><span>Motion</span><div className="ws-live-signal-canvas"><canvas ref={d.motionRef} width={160} height={56} aria-label="Measured crop pixel changes"/>{d.readout.motionEnergy === null && <span>No measurements</span>}</div><small>{d.readout.motionEnergy !== null ? 'Pixel change ' + d.readout.motionEnergy.toFixed(3) : 'Frame-to-frame pixel change'}</small></div>
      </div><p className="ws-live-fusion-note">Fusion weights are not returned by the alphabet classifier.</p></section>
      <section className="ws-card ws-live-history"><div className="ws-card-head"><h2>Recent results</h2><button className="ws-inline-link" onClick={d.clearHistory} disabled={!d.history.length}>Clear</button></div>{d.history.length === 0 ? <div className="ws-live-history-empty"><Icon name="activity" size={23}/><p>A fresh session.<br/><span>Real predictions will appear here.</span></p></div> : <ol>{d.history.slice(0,4).map(result=><li key={result.id}><strong>{result.letter}</strong><div><span>Alphabet prediction</span><small>{new Date(result.at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</small></div><span>{(result.confidence * 100).toFixed(1)}%</span></li>)}</ol>}<p className="ws-live-history-note">Session only · Cleared when camera stops</p></section>
    </div>
  </div>
}
