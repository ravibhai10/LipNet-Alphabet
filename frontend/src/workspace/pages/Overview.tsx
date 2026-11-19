import { Link } from 'react-router-dom'
import { useBackend } from '../../hooks/useBackend'
import { Icon } from '../Icon'
import { VisualSpeechEqualizer } from '../../visualizations/VisualSpeechEqualizer'
import { NeuralSynapseVisualizer } from '../components/NeuralSynapseVisualizer'

function PipelineVisual() {
  return <div className="ws-overview-visual" aria-label="Illustration of visual speech processing, not live data">
    <div className="ws-orbit ws-orbit-one"/><div className="ws-orbit ws-orbit-two"/>
    <svg viewBox="0 0 340 230" fill="none" aria-hidden="true">
      <defs><linearGradient id="mouth-line" x1="40" y1="100" x2="300" y2="160" gradientUnits="userSpaceOnUse"><stop stopColor="var(--ws-accent)" stopOpacity=".2"/><stop offset=".5" stopColor="var(--ws-accent)"/><stop offset="1" stopColor="var(--ws-accent)" stopOpacity=".2"/></linearGradient></defs>
      <path d="M42 118C81 112 109 64 145 91Q170 108 195 91C231 64 259 112 298 118C254 119 239 173 170 173C101 173 86 119 42 118Z" stroke="url(#mouth-line)" strokeWidth="1.5"/>
      <path d="M42 118C100 109 115 124 170 124S240 109 298 118 M72 119C104 137 133 148 170 148S236 137 268 119" stroke="var(--ws-accent)" strokeOpacity=".4"/>
      <path d="M42 118 110 95 126 133 145 91 170 124 195 91 214 133 230 95 298 118M72 119 110 95 145 91M126 133 170 173 214 133M110 95 170 148 230 95" stroke="var(--ws-accent)" strokeOpacity=".13"/>
      {[ [42,118],[110,95],[145,91],[170,124],[195,91],[230,95],[298,118],[126,133],[214,133],[170,173] ].map(([cx,cy],i)=><circle key={i} cx={cx} cy={cy} r="2.6" fill="var(--ws-accent)"/>)}
      <path d="M28 55V37h18M294 37h18v18M28 184v18h18M294 202h18v-18" stroke="var(--ws-accent)" strokeOpacity=".35"/>
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
    <header className="ws-page-header ws-overview-heading">
      <div>
        <p className="ws-kicker">THE RESEARCH WORKSPACE</p>
        <h1 className="ws-title">Every movement. <em>A meaning.</em></h1>
        <p className="ws-lede">Your visual speech system, from the first frame to the final prediction.</p>
      </div>
      <span className="ws-edition">LIPNET AI<br/><b>RESEARCH EDITION</b></span>
    </header>

    <section className="ws-overview-hero ws-hud-bracket">
      <div className="ws-overview-copy">
        <span className="ws-eyebrow"><Icon name="scan" size={14}/> VISUAL SPEECH INTELLIGENCE</span>
        <h2>Give silence<br/><em>a voice.</em></h2>
        <p>Explore the signals behind speech. Capture mouth movement, inspect the visual pipeline, and connect to your alphabet model.</p>
        <div className="ws-action-row">
          <Link className="ws-btn ws-btn-primary ws-btn-tactile ws-btn-respond" to="/demo/live">Open live studio<Icon name="arrow" size={16}/></Link>
          <Link className="ws-btn ws-btn-ghost ws-btn-tactile ws-btn-respond" to="/demo/pipeline">Explore pipeline</Link>
        </div>
        <div className="ws-hero-footnote"><Icon name="shield" size={13}/> Camera off by default · CPU compatible</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', width: '100%', maxWidth: '380px' }}>
        <PipelineVisual />
        <div style={{ marginTop: '10px', borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--ws-border)' }}>
          <VisualSpeechEqualizer active={true} speaking={false} height={38} />
        </div>
      </div>
    </section>

    <div className="ws-section-heading">
      <h2>Your system, at a glance</h2>
      <button onClick={refresh} className="ws-btn ws-btn-ghost ws-btn-sm ws-btn-tactile ws-btn-respond" disabled={status === 'checking'}>
        <Icon name="refresh" size={14}/>Refresh status
      </button>
    </div>

    <section className="ws-overview-metrics" aria-label="Dataset and model overview">
      {metrics.map(metric=>(
        <article className="ws-summary-card" key={metric.label}>
          <div className="ws-summary-label"><span>{metric.label}</span><Icon name={metric.icon} size={17}/></div>
          <div className="ws-summary-value" style={{ fontVariantNumeric: 'tabular-nums' }}>{metric.value?.toLocaleString() ?? '—'}</div>
          <p>{metric.value == null ? online ? 'Metadata not available' : 'Connect backend to inspect' : metric.note}</p>
        </article>
      ))}
      <article className="ws-summary-card ws-model-summary">
        <div className="ws-summary-label"><span>Alphabet checkpoint</span><Icon name="cpu" size={17}/></div>
        <div className="ws-summary-value ws-summary-status">{!online ? 'Unverified' : modelLoaded ? 'Loaded' : 'Not loaded'}</div>
        <p>Inference readiness is checked in live studio.</p>
      </article>
    </section>

    <section className="ws-card ws-capstone-banner ws-hud-bracket" style={{ marginTop: '24px', padding: '24px', border: '1px solid var(--ws-border)', borderRadius: '12px', background: 'var(--ws-surface)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <span className="ws-eyebrow" style={{ color: 'var(--ws-accent)', fontWeight: 600 }}>ACADEMIC CAPSTONE · GROUP 12 SUBMISSION</span>
          <h2 style={{ margin: '4px 0 8px 0', fontSize: '18px', letterSpacing: '-0.02em' }}>LipNet-AI: Multimodal Silent Speech Recognition System</h2>
          <p style={{ margin: 0, opacity: 0.8, maxWidth: '750px', fontSize: '13px', lineHeight: '1.6' }}>
            Production-grade visual speech recognition engine combining localized landmark alignment, temporal frame sequencing, and deep neural lip reading across isolated letters, multi-speaker words, and continuous sentence benchmarks.
          </p>
        </div>
        <Link to="/demo/docs" className="ws-btn ws-btn-primary ws-btn-sm" style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
          <Icon name="download" size={14} /> View Submission Package (8/8)
        </Link>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px', marginTop: '20px' }}>
        <div style={{ padding: '14px 16px', background: 'var(--ws-surface2)', borderRadius: '8px', borderLeft: '3px solid var(--ws-success)' }}>
          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ws-text-muted)' }}>MIRACL-VC1 Word SOTA</div>
          <div style={{ fontSize: '20px', fontWeight: 650, color: 'var(--ws-success)', marginTop: '3px', fontVariantNumeric: 'tabular-nums' }}>94.79% Top-1</div>
          <div style={{ fontSize: '11px', color: 'var(--ws-text-dim)', marginTop: '2px' }}>182/192 correct (AV-HuBERT L9)</div>
        </div>

        <div style={{ padding: '14px 16px', background: 'var(--ws-surface2)', borderRadius: '8px', borderLeft: '3px solid var(--ws-accent)' }}>
          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ws-text-muted)' }}>GRID Corpus Benchmark</div>
          <div style={{ fontSize: '20px', fontWeight: 650, color: 'var(--ws-accent)', marginTop: '3px', fontVariantNumeric: 'tabular-nums' }}>32,891 Clips</div>
          <div style={{ fontSize: '11px', color: 'var(--ws-text-dim)', marginTop: '2px' }}>33 speakers · 25 letter classes</div>
        </div>

        <div style={{ padding: '14px 16px', background: 'var(--ws-surface2)', borderRadius: '8px', borderLeft: '3px solid var(--ws-warning)' }}>
          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ws-text-muted)' }}>Alphabet Model</div>
          <div style={{ fontSize: '20px', fontWeight: 650, color: 'var(--ws-warning)', marginTop: '3px' }}>CNN-BiLSTM</div>
          <div style={{ fontSize: '11px', color: 'var(--ws-text-dim)', marginTop: '2px' }}>30 epochs · 26 A-Z classes</div>
        </div>

        <div style={{ padding: '14px 16px', background: 'var(--ws-surface2)', borderRadius: '8px', borderLeft: '3px solid #8b5cf6' }}>
          <div style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ws-text-muted)' }}>Continuous CTC</div>
          <div style={{ fontSize: '20px', fontWeight: 650, color: '#8b5cf6', marginTop: '3px', fontVariantNumeric: 'tabular-nums' }}>38.89% WER</div>
          <div style={{ fontSize: '11px', color: 'var(--ws-text-dim)', marginTop: '2px' }}>38.20% CER sentence decoder</div>
        </div>
      </div>
    </section>

    <div style={{ margin: '20px 0' }}>
      <NeuralSynapseVisualizer />
    </div>

    <div className="ws-overview-bottom">
      <section className="ws-card ws-workflow-card">
        <div className="ws-card-head">
          <h2>From pixels to prediction</h2>
          <Link to="/demo/pipeline" className="ws-inline-link">View pipeline<Icon name="arrow" size={14}/></Link>
        </div>
        <p className="ws-note">The architecture, step by step. Availability depends on your camera and backend.</p>
        <div className="ws-overview-flow">
          {[{name:'Capture',icon:'camera',note:'Browser camera'},{name:'Extract',icon:'scan',note:'Mouth landmarks'},{name:'Sequence',icon:'layers',note:'Temporal frames'},{name:'Decode',icon:'activity',note:'Alphabet model'}].map((step,i)=>(
            <div key={step.name}>
              <span className="ws-flow-icon"><Icon name={step.icon} size={21}/></span>
              <small>0{i+1}</small>
              <h3>{step.name}</h3>
              <p>{step.note}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="ws-card ws-next-card">
        <span className="ws-eyebrow">MODEL SCOPE</span>
        <h2>Letters first.<br/><em>Language next.</em></h2>
        <p>Alphabet classification and sentence lip reading are separate tasks. The sentence decoder is not connected to this workspace.</p>
        <span className="ws-badge ws-badge-warning">Sentence inference unavailable</span>
        <Link to="/demo/models" className="ws-inline-link">Inspect model registry<Icon name="arrow" size={14}/></Link>
      </section>
    </div>
  </div>
}
