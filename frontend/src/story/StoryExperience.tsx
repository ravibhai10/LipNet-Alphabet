import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { BackendProvider, useBackend } from '../hooks/useBackend'
import { API_BASE, fetchWithTimeout } from '../lib/api'
import { useDemoPipeline } from '../live-demo/useDemoPipeline'
import { BrandMark, Icon } from '../workspace/Icon'
import { StoryNavigation, ProjectDestinations } from './StoryNavigation'
import { StoryHeroVisual } from './StoryHeroVisual'
import { useGsapScene } from '../animations/useGsapScene'
import { gsap } from '../lib/gsap'
import '../styles/story.css'

const steps = [
  ['Capture', 'A short video holds a sequence of visual clues.'],
  ['Align', 'Measured landmarks keep the mouth crop centered and level.'],
  ['Extract', 'A visual encoder turns each frame into useful features.'],
  ['Connect', 'Temporal patterns describe how the mouth moves.'],
  ['Classify', 'A trained head scores the possible spoken letters.'],
  ['Display', 'Consistent predictions appear with their model scores.'],
]
function Chapter({ number, children }: { number: string; children: React.ReactNode }) {
  return <p className="story-chapter"><span>{number}</span>{children}</p>
}
function LipDrawing({ open = 1 }: { open?: number }) {
  return <svg viewBox="0 0 200 110" fill="none" aria-hidden="true"><path d="M24 55C50 48 66 25 85 34L100 40L115 34C134 25 150 48 176 55C145 65 131 86 100 84C69 86 55 65 24 55Z" stroke="currentColor" strokeWidth="1.3"/><path d={`M24 55Q66 ${55 - open * 19} 100 ${55 - open * 12}Q134 ${55 - open * 19} 176 55Q134 ${55 + open * 27} 100 ${55 + open * 26}Q66 ${55 + open * 27} 24 55Z`} stroke="currentColor" strokeWidth="1.3"/><path d="M85 17L100 23L115 17M80 96Q100 101 120 96" stroke="currentColor" opacity=".3"/></svg>
}
function StoryDemo() {
  const d = useDemoPipeline()
  const backend = useBackend()
  const connection = d.backendStatus === 'online' ? d.inferenceReady ? 'Alphabet model ready' : 'Backend online · model unavailable' : d.backendStatus === 'offline' ? 'Backend offline' : 'Checking connection…'
  const hint = d.camStatus === 'requesting' ? 'Waiting for camera permission…' : d.running ? d.lmStatus === 'loading' ? 'Preparing face tracking…' : d.lmStatus === 'failed' ? 'Face tracking could not start.' : !d.readout.faceOk ? 'Bring your face into the frame.' : d.readout.stabilizing ? 'Checking a second window…' : d.readout.speaking ? 'Reading mouth movement…' : 'Speak one letter naturally.' : 'Your camera stays off until you start it.'
  return <div className="story-demo" data-testid="story-demo">
    <div className="story-demo-stage">
      <div className="story-camera">
        <video ref={d.videoRef} playsInline muted aria-label="Live camera preview"/>
        <canvas ref={d.overlayRef} className="story-camera-overlay" aria-hidden="true"/>
        {!d.running && <div className="story-camera-idle"><Icon name="camera" size={36}/><p>{d.camStatus === 'requesting' ? 'A moment for permission.' : 'A quiet frame. Your voice, visible.'}</p><small>No microphone is used.</small></div>}
        <span className="story-camera-caption">{d.running ? 'Camera on' : 'Camera off'}<span>VIDEO ONLY</span></span>
        <div className="story-camera-control">
        {d.running || d.camStatus === 'requesting' ? <button className="story-button story-button-outline" onClick={d.stop}>{d.camStatus === 'requesting' ? 'Cancel' : 'Stop camera'}</button> : <button className="story-button" onClick={() => void d.start()}><Icon name="camera" size={17}/>{d.camStatus === 'denied' || d.camStatus === 'error' ? 'Retry camera' : 'Start camera'}</button>}
        </div>
      </div>
      <div className="story-camera-actions">
        <p role="status">{hint}</p>
      </div>
      {(d.camError || d.lmError || d.pipelineError || d.predictionError) && <p className="story-error" role="alert">{d.camError || d.lmError || d.pipelineError || d.predictionError}</p>}
    </div>
    <div className="story-result">
      <p className="story-eyebrow">THE MODEL'S RESPONSE</p>
      <p className="story-prediction" aria-live="polite" aria-label="Predicted letter">{d.readout.detected ?? '—'}</p>
      <p className="story-response-note">{d.readout.detected ? 'A consistent alphabet prediction.' : d.inferenceReady ? 'Your next movement starts the story.' : 'A connection is needed for a prediction.'}</p>
      {d.readout.confidence !== null && <p className="story-score">{(d.readout.confidence * 100).toFixed(1)}% <span>model score · uncalibrated</span></p>}
      <div className="story-connection" role="status"><i className={d.inferenceReady ? 'is-ready' : ''}/>{connection}<button onClick={backend.refresh} aria-label="Refresh story backend connection"><Icon name="refresh" size={15}/></button></div>
      {d.backendStatus === 'offline' && <p className="story-small">The camera preview still works. Start the local API and retry the connection to receive predictions.</p>}
      {d.backendStatus === 'online' && !d.inferenceReady && <p className="story-small">{d.inferenceReason}</p>}
      <p className="story-small">This is an experimental A–Z letter model. It does not transcribe sentences. Webcam accuracy has not been independently measured.</p>
    </div>
    <details className="story-technical"><summary>Behind this prediction <span>Technical details <Icon name="plus" size={15}/></span></summary><div className="story-technical-content">
      <dl><div><dt>Selected model</dt><dd>{d.modelKind ?? 'Checking…'}</dd></div><div><dt>Backend</dt><dd>{API_BASE || 'Same origin'}</dd></div><div><dt>Aligned frames</dt><dd>{d.readout.frames} / {d.sequence.maxFrames}</dd></div><div><dt>Response time</dt><dd>{d.readout.latencyMs === null ? 'Not measured' : `${d.readout.latencyMs} ms`}</dd></div><div><dt>AV-HuBERT encoder</dt><dd>{backend.capabilities ? backend.capabilities.avhubert_encoder.present ? 'Weights present' : 'Not present' : 'Not reported'}</dd></div><div><dt>AV-HuBERT head</dt><dd>{backend.capabilities ? backend.capabilities.avhubert_head.ready ? 'Verified and ready' : backend.capabilities.avhubert_head.present ? 'Present; compatibility not verified' : 'Not trained / not present' : 'Not reported'}</dd></div></dl>
      <p>Movement gates new requests. Full probability vectors use EMA smoothing; two consistent windows confirm the display. Smoothing has not been shown to improve held-out accuracy.</p>
      {d.readout.top3 && <p>Returned candidates: {d.readout.top3.map(item => `${item.letter} ${(item.prob * 100).toFixed(1)}%`).join(' · ')}</p>}
      <div className="story-diagnostic-crops"><canvas ref={d.appearanceRef} width={64} height={64} aria-label="Aligned mouth crop"/><canvas ref={d.geometryRef} width={180} height={64} aria-label="Mouth aperture history"/><canvas ref={d.motionRef} width={180} height={64} aria-label="Crop motion history"/><canvas ref={d.bufferStripRef} width={d.sequence.maxFrames * 24} height={24} aria-label="Captured frame buffer"/></div>
      <Link to="/demo/live" className="story-text-link">Open the full live workspace <Icon name="arrow" size={16}/></Link>
    </div></details>
    <div className="story-demo-page-link"><span>Want the complete camera studio?</span><Link className="story-text-link" to="/demo/live">Open Live Demo <Icon name="arrow" size={16}/></Link></div>
  </div>
}
interface Evidence { alphabet: { recordings: number; speakers: number }; grid: { utterances: number; speakers: number }; lrw: { clips: number }; benchmark: { status: string; accuracy: number | null } }
function EvidenceScene() {
  const [data, setData] = useState<Evidence | null>(null)
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setFailed(false)
    void fetchWithTimeout(`${API_BASE}/api/evidence`, { signal: controller.signal }, 8000).then(async response => {
      if (!response.ok) throw new Error('Evidence unavailable')
      const body = await response.json() as Evidence
      if (!body.alphabet || !body.grid || !body.lrw || !body.benchmark) throw new Error('Invalid evidence')
      if (!controller.signal.aborted) setData(body)
    }).catch(() => { if (!controller.signal.aborted) setFailed(true) })
    return () => controller.abort()
  }, [attempt])
  return <section className="story-scene story-evidence" id="evidence" data-story-scene="5"><div className="story-container story-reveal">
    <Chapter number="05">THE EVIDENCE</Chapter><div className="story-evidence-heading"><h2>Measure what the model<br/>can really <em>recognize.</em></h2><p>A working prediction is a starting point. A trustworthy result needs authentic labels and people the model has never seen during training.</p></div>
    <div className="story-evidence-body"><div className="story-benchmark"><span className="story-status-label">INDEPENDENT A–Z BENCHMARK</span><h3>Still to be measured.</h3><p>The current data cannot establish a speaker-independent alphabet benchmark. We do not report an accuracy gain or a per-class result from fixtures.</p><p>Next: collect labeled A–Z clips from separate training, validation, and test speakers. Select the model on validation data, then evaluate the held-out test set.</p><Link to="/demo/testing" className="story-text-link">Explore evaluation tools <Icon name="arrow" size={16}/></Link></div>
    <div className="story-corpus"><h3>What is actually available</h3><p className="story-small">Counts read from the local repository.</p>{failed ? <div className="story-error" role="status">Corpus counts are unavailable while the API is offline. <button className="story-text-link" onClick={() => setAttempt(value => value + 1)}>Retry counts</button></div> : <dl><div><dt>Authentic alphabet recordings<small>Registered local collection</small></dt><dd>{data?.alphabet.recordings ?? '—'}<small>{data ? `${data.alphabet.speakers} speakers` : 'Reading data…'}</small></dd></div><div><dt>GRID sentence videos<small>Not isolated A–Z annotations</small></dt><dd>{data?.grid.utterances.toLocaleString() ?? '—'}<small>{data ? `${data.grid.speakers} speaker${data.grid.speakers === 1 ? '' : 's'}` : 'Reading data…'}</small></dd></div><div><dt>LRW development clips<small>Generated fixtures; excluded from evidence</small></dt><dd>{data?.lrw.clips ?? '—'}<small>Software checks only</small></dd></div></dl>}</div></div>
    <details className="story-limitations"><summary>Where visual speech can fall short <Icon name="plus" size={16}/></summary><p>Different sounds can share a visible mouth shape. Head movement, low light, occlusion, speech pace, and unfamiliar speakers can change predictions. The current confidence scores and activity thresholds are uncalibrated. Real-person failure cases and per-class precision, recall, F1, and confusion matrices require a valid labeled test set.</p></details>
  </div></section>
}
export function StoryExperience() {
  const scope = useRef<HTMLDivElement>(null)
  useGsapScene(scope, () => {
    gsap.fromTo('.story-hero-copy > *', { autoAlpha: 0, y: 28 }, { autoAlpha: 1, y: 0, duration: 1, stagger: .12, ease: 'power3.out' })
    gsap.fromTo('.story-hero .hero-visual', { autoAlpha: 0, scale: .88 }, { autoAlpha: 1, scale: 1, duration: 1.7, ease: 'power3.out' })
    gsap.fromTo('.story-hero .hv-contour', { opacity: .12 }, { opacity: 1, duration: 1.8, delay: .3 })
    gsap.fromTo('.story-hero .hv-mouth-poly', { strokeDashoffset: 620 }, { strokeDashoffset: 0, duration: 2, delay: .5, ease: 'power2.out' })
    gsap.timeline({ scrollTrigger: { trigger: '.story-hero', start: 'top top', end: 'bottom top', scrub: .6 } })
      .to('.story-hero .hero-visual', { scale: 1.18, yPercent: 8, ease: 'none' }, 0)
      .to('.story-hero-copy', { y: -45, ease: 'none' }, 0)
    gsap.utils.toArray<HTMLElement>('.story-reveal').forEach(section => {
      gsap.fromTo(section, { autoAlpha: 0, y: 45 }, { autoAlpha: 1, y: 0, duration: 1.1, ease: 'power3.out', scrollTrigger: { trigger: section, start: 'top 94%', once: true } })
    })
    gsap.fromTo('.story-frame', { autoAlpha: .25, y: 26 }, { autoAlpha: 1, y: 0, duration: .85, stagger: .14, ease: 'power3.out', scrollTrigger: { trigger: '.story-frames', start: 'top 88%', once: true } })
    gsap.fromTo('.story-method-steps', { '--story-progress': '0%' }, { '--story-progress': '100%', ease: 'none', scrollTrigger: { trigger: '.story-method-steps', start: 'top 85%', end: 'bottom 50%', scrub: .5 } })
    gsap.fromTo('.story-journey-list > li', { autoAlpha: .2, x: 20 }, { autoAlpha: 1, x: 0, duration: .7, stagger: .09, scrollTrigger: { trigger: '.story-journey-list', start: 'top 85%', once: true } })
  })
  useEffect(() => {
    document.title = 'LipNet AI — From a silent face to spoken meaning'
  }, [])
  return <BackendProvider><div className="story-experience" ref={scope}>
    <a className="skip-link" href="#main">Skip to content</a>
    <StoryNavigation/>
    <main id="main">
      <section className="story-hero" id="top" data-story-scene="1"><div className="story-container"><div className="story-hero-topline"><span>A STUDY IN VISUAL SPEECH</span><span>VIDEO IN. POSSIBILITY OUT.</span></div><div className="story-hero-grid"><div className="story-hero-copy"><Chapter number="01">THE QUESTION</Chapter><h1 className="hero-title">What if a camera could understand speech <em>without hearing<br className="story-desktop-break"/> a sound?</em></h1><p>Speech leaves a trace before we hear it. LipNet explores how a camera can learn from the movement of a face—and how far that understanding can go.</p><div className="story-hero-actions"><a href="#idea" className="story-button">Follow the story <Icon name="arrow" size={17}/></a><Link to="/demo/live" className="story-button story-button-outline">Open Live Demo <Icon name="camera" size={17}/></Link></div></div><StoryHeroVisual/></div><div className="story-hero-footer"><span>From a silent face to spoken meaning.</span><a href="#idea">SCROLL TO EXPLORE <span>↓</span></a></div></div></section>
      <section className="story-scene story-idea" id="idea" data-story-scene="2"><div className="story-container story-reveal"><Chapter number="02">THE IDEA</Chapter><div className="story-split-heading"><h2>Every movement<br/>carries a <em>clue.</em></h2><div><p>A single frame shows a mouth shape. A sequence reveals a movement: opening, closing, pausing, changing.</p><p>These patterns contain clues about speech. The challenge is learning which clues travel across different faces, lighting, and speaking styles.</p></div></div><figure className="story-frame-sequence"><div className="story-frames">{[.12,.45,.95,.7,.3].map((open,index) => <div className="story-frame" key={index} style={{animationDelay:`${index * .12}s`}}><span>0{index+1}</span><LipDrawing open={open}/><i/></div>)}</div><figcaption><span>One shape is a clue. Movement adds context.</span><span>ILLUSTRATED TEMPORAL SEQUENCE →</span></figcaption></figure></div></section>
      <section className="story-scene story-method" id="method" data-story-scene="3"><div className="story-container story-reveal"><Chapter number="03">THE METHOD</Chapter><div className="story-split-heading"><h2>From pixels<br/>to a <em>prediction.</em></h2><p>Six connected steps turn visible movement into an experimental alphabet result. Each step has a job. Each can introduce an error.</p></div><ol className="story-method-steps">{steps.map(([title,note], index) => <li key={title}><span className="story-step-number">0{index+1}</span><h3>{title}</h3><p>{note}</p></li>)}</ol><div className="story-model-paths"><div><span className="story-eyebrow">CURRENT LIVE PATH</span><h3>CNN → BiLSTM → A–Z</h3><p>A trained image encoder and bidirectional temporal model score 26 letter classes.</p></div><div><span className="story-eyebrow">THE PRETRAINED PATH</span><h3>AV-HuBERT → supervised head</h3><p>The pretrained encoder has run on Windows CPU. A separately trained classifier head is still required before it can predict letters.</p></div></div></div></section>
      <section className="story-scene story-demo-scene" id="demo" data-story-scene="4"><div className="story-container story-reveal"><Chapter number="04">YOUR TURN</Chapter><div className="story-split-heading"><h2>Try the model<br/><em>yourself.</em></h2><p>Find good light, face the camera, and say a single letter. The model reads video only. No microphone, no invented results.</p></div><StoryDemo/></div></section>
      <EvidenceScene/>
      <section className="story-scene story-journey" id="journey" data-story-scene="6"><div className="story-container story-reveal"><Chapter number="06">THE RESEARCH JOURNEY</Chapter><div className="story-journey-grid"><div><h2>Building toward<br/>more reliable<br/><em>visual speech.</em></h2><p>This project connects a working experiment with a clear next question: how well will it recognize people it has never seen?</p><Link to="/demo/overview" className="story-button story-button-outline">Enter technical workspace <Icon name="arrow" size={17}/></Link></div><ol className="story-journey-list">{[
        ['Data validation','Implemented','Provenance, duplicate-content, and speaker split checks.'],
        ['Face & mouth alignment','Implemented','Measured landmarks and a consistent mouth crop.'],
        ['Pretrained features','CPU test recorded','Official AV-HuBERT produced finite 29 × 768 features from a real clip.'],
        ['Supervised training','Awaiting labeled data','Feature caching and head training are ready. A real A–Z head is not trained.'],
        ['Independent evaluation','Awaiting test speakers','Per-class metrics and model comparison need an eligible test set.'],
        ['Live inference','Experimental','The existing alphabet model runs on CPU; webcam accuracy remains unmeasured.'],
      ].map(([title,status,note],index) => <li key={title}><span className="story-journey-number">0{index+1}</span><div><h3>{title}<small>{status}</small></h3><p>{note}</p></div></li>)}</ol></div><ProjectDestinations/><div className="story-closing"><p>Make the invisible part of communication<br/><em>a little more visible.</em></p><div><a href="#demo" className="story-text-link">Return to the live demo <Icon name="arrow" size={16}/></a><Link to="/demo/docs" className="story-text-link">Read the technical guide <Icon name="arrow" size={16}/></Link><Link to="/research" className="story-text-link">Explore the research exhibits <Icon name="arrow" size={16}/></Link></div></div></div></section>
    </main><footer className="story-footer"><a className="story-brand" href="#top"><BrandMark/><strong>LIPNET<span> / AI</span></strong></a><p>A visual speech research project.<br/>Built to demonstrate. Designed to be questioned.</p><Link to="/demo/overview">Technical workspace ↗</Link></footer>
  </div></BackendProvider>
}
