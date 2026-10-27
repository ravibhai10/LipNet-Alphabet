import { useEffect, useRef, useState } from 'react'
import { useBackend } from '../hooks/useBackend'
import { SectionHead } from '../components/ui'

/** Inspect real responses from the landing page camera; capture is owned by LiveDemoSection. */
export function PredictionPlayground() {
  const { lastPrediction } = useBackend()
  const [history, setHistory] = useState<{id: number; letter: string; confidence: number}[]>([])
  const [query, setQuery] = useState('')
  const [copyStatus, setCopyStatus] = useState('')
  const timeout = useRef<ReturnType<typeof setTimeout>>()
  const sequence = useRef(0)
  useEffect(() => {
    if (!lastPrediction) { setHistory([]); return }
    setHistory(items => [{id: ++sequence.current, letter: lastPrediction.letter, confidence: lastPrediction.confidence}, ...items].slice(0,30))
  }, [lastPrediction])
  useEffect(() => () => clearTimeout(timeout.current), [])
  async function copy() {
    try {
      await navigator.clipboard.writeText(history.map(item => item.letter + ' ' + (item.confidence * 100).toFixed(1) + '%').join('\n'))
      setCopyStatus('Copied')
    } catch { setCopyStatus('Copy unavailable') }
    clearTimeout(timeout.current); timeout.current = setTimeout(() => setCopyStatus(''), 2000)
  }
  const filtered = history.filter(item=>item.letter.toLowerCase().includes(query.toLowerCase()))
  return <section id="prediction-playground" className="section prediction-inspector"><div className="container">
    <SectionHead kicker="RESPONSE INSPECTOR" title={<>A closer look at <em>the output.</em></>} lede="Inspect the real alphabet responses from the live camera above. Confidence scores are uncalibrated and do not measure webcam accuracy."/>
    <div className="prediction-inspector-grid">
      <div className="prediction-inspector-card"><div className="prediction-inspector-head"><h3>Top candidates</h3><a href="/demo/live">Open live studio ↗</a></div>{lastPrediction?.top3.length ? lastPrediction.top3.map(candidate=><div className="prediction-inspector-result" key={candidate.letter}><strong>{candidate.letter}</strong><div><i style={{width:candidate.prob * 100 + '%'}}/></div><span>{(candidate.prob * 100).toFixed(1)}%</span></div>) : <p className="prediction-inspector-empty">Start the camera above to inspect actual model responses.</p>}<a className="btn btn-ghost" href="#demo">Go to camera</a></div>
      <div className="prediction-inspector-card"><div className="prediction-inspector-head"><h3>Session history</h3><button onClick={() => void copy()} disabled={!history.length}>{copyStatus || 'Copy'}</button></div><label className="prediction-inspector-search"><span>Filter returned letters</span><input type="search" value={query} onChange={event=>setQuery(event.target.value)} placeholder="Find a letter…"/></label>{filtered.length ? <ol>{filtered.map(item=><li key={item.id}><strong>{item.letter}</strong><span>{(item.confidence * 100).toFixed(1)}%</span></li>)}</ol> : <p className="prediction-inspector-empty">{history.length ? 'No matching results.' : 'No predictions in this session yet.'}</p>}<p className="mono-note" role="status">{copyStatus || 'Memory only · Cleared when the camera stops'}</p></div>
    </div>
  </div></section>
}
