import { Link } from 'react-router-dom'
import { useBackend } from '../../hooks/useBackend'
import { Icon } from '../Icon'

function PipelineVisual() {
  return <div className="ws-overview-visual" aria-label="Illustration of visual speech processing, not live data">
    <div className="ws-orbit ws-orbit-one"/><div className="ws-orbit ws-orbit-two"/>
    <svg viewBox="0 0 340 230" fill="none" aria-hidden="true">
      <defs><linearGradient id="mouth-line" x1="40" y1="100" x2="300" y2="160" gradientUnits="userSpaceOnUse"><stop stopColor="#8fdfcb" stopOpacity=".2"/><stop offset=".5" stopColor="#8fdfcb"/><stop offset="1" stopColor="#8fdfcb" stopOpacity=".2"/></linearGradient></defs>
      <path d="M42 118C81 112 109 64 145 91Q170 108 195 91C231 64 259 112 298 118C254 119 239 173 170 173C101 173 86 119 42 118Z" stroke="url(#mouth-line)" strokeWidth="1.5"/>
      <path d="M42 118C100 109 115 124 170 124S240 109 298 118 M72 119C104 137 133 148 170 148S236 137 268 119" stroke="#8fdfcb" strokeOpacity=".4"/>
      <path d="M42 118 110 95 126 133 145 91 170 124 195 91 214 133 230 95 298 118M72 119 110 95 145 91M126 133 170 173 214 133M110 95 170 148 230 95" stroke="#8fdfcb" strokeOpacity=".13"/>
      {[ [42,118],[110,95],[145,91],[170,124],[195,91],[230,95],[298,118],[126,133],[214,133],[170,173] ].map(([cx,cy],i)=><circle key={i} cx={cx} cy={cy} r="2.6" fill="#8fdfcb"/>)}
      <path d="M28 55V37h18M294 37h18v18M28 184v18h18M294 202h18v-18" stroke="#8fdfcb" strokeOpacity=".35"/>
    </svg>
    <span className="ws-visual-caption">VISUAL SPEECH / PIPELINE SCHEMATIC</span>
  </div>
}
export function OverviewPage() {
  const { status, modelLoaded, datasetStats, refresh } = useBackend()
  const online = status === 'online'
  const metrics = [
    { label: 'Recordings', value: datasetStats?.recordings, icon: 'camera', note: 'Available dataset metadata' },
    { label: 'Speakers', value: datasetStats?.speakers, icon: 'scan', note: 'Reported by the dataset service' },
    { label: 'Letter classes', value: datasetStats?.letters, icon: 'layers', note: 'Labels found in your dataset' },
  ]
  return <div className="ws-overview">
    <header className="ws-page-header ws-overview-heading"><div><p className="ws-kicker">THE RESEARCH WORKSPACE</p><h1 className="ws-title">Every movement. <em>A meaning.</em></h1><p className="ws-lede">Your visual speech system, from the first frame to the final prediction.</p></div><span className="ws-edition">LIPNET AI<br/><b>RESEARCH EDITION</b></span></header>
    <section className="ws-overview-hero">
      <div className="ws-overview-copy"><span className="ws-eyebrow"><Icon name="scan" size={14}/> VISUAL SPEECH INTELLIGENCE</span><h2>Give silence<br/><em>a voice.</em></h2><p>Explore the signals behind speech. Capture mouth movement, inspect the visual pipeline, and connect to your alphabet model.</p><div className="ws-action-row"><Link className="ws-btn ws-btn-primary" to="/demo/live">Open live studio<Icon name="arrow" size={16}/></Link><Link className="ws-btn ws-btn-ghost" to="/demo/pipeline">Explore pipeline</Link></div><div className="ws-hero-footnote"><Icon name="shield" size={13}/> Camera off by default · CPU compatible</div></div>
      <PipelineVisual />
    </section>
    <div className="ws-section-heading"><h2>Your system, at a glance</h2><button onClick={refresh} className="ws-btn ws-btn-ghost ws-btn-sm" disabled={status === 'checking'}><Icon name="refresh" size={14}/>Refresh status</button></div>
    <section className="ws-overview-metrics" aria-label="Dataset and model overview">
      {metrics.map(metric=><article className="ws-summary-card" key={metric.label}><div className="ws-summary-label"><span>{metric.label}</span><Icon name={metric.icon} size={17}/></div><div className="ws-summary-value">{metric.value?.toLocaleString() ?? '—'}</div><p>{metric.value == null ? online ? 'Metadata not available' : 'Connect backend to inspect' : metric.note}</p></article>)}
      <article className="ws-summary-card ws-model-summary"><div className="ws-summary-label"><span>Alphabet checkpoint</span><Icon name="cpu" size={17}/></div><div className="ws-summary-value ws-summary-status">{!online ? 'Unverified' : modelLoaded ? 'Loaded' : 'Not loaded'}</div><p>Inference readiness is checked in live studio.</p></article>
    </section>
    <div className="ws-overview-bottom">
      <section className="ws-card ws-workflow-card"><div className="ws-card-head"><h2>From pixels to prediction</h2><Link to="/demo/pipeline" className="ws-inline-link">View pipeline<Icon name="arrow" size={14}/></Link></div><p className="ws-note">The architecture, step by step. Availability depends on your camera and backend.</p><div className="ws-overview-flow">{[{name:'Capture',icon:'camera',note:'Browser camera'},{name:'Extract',icon:'scan',note:'Mouth landmarks'},{name:'Sequence',icon:'layers',note:'Temporal frames'},{name:'Decode',icon:'activity',note:'Alphabet model'}].map((step,i)=><div key={step.name}><span className="ws-flow-icon"><Icon name={step.icon} size={21}/></span><small>0{i+1}</small><h3>{step.name}</h3><p>{step.note}</p></div>)}</div></section>
      <section className="ws-card ws-next-card"><span className="ws-eyebrow">MODEL SCOPE</span><h2>Letters first.<br/><em>Language next.</em></h2><p>Alphabet classification and sentence lip reading are separate tasks. The sentence decoder is not connected to this workspace.</p><span className="ws-badge ws-badge-warning">Sentence inference unavailable</span><Link to="/demo/models" className="ws-inline-link">Inspect model registry<Icon name="arrow" size={14}/></Link></section>
    </div>
  </div>
}

